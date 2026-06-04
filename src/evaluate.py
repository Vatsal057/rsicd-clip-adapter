"""
Cross-modal retrieval evaluation on RSICD.

Public API:
    encode_images(model, image_loader, device)        -> (embeddings, imgids)
    encode_captions(model, caption_loader, device)     -> (embeddings, imgids)
    compute_recall_at_k(query, query_ids, gallery, gallery_ids, k_values)
                                                         -> {"R@1": float, "R@5": ..., "R@10": ...}
    evaluate_model(model, retrieval_dataset, device, split_name)
                                                         -> full results dict (saved as JSON)

The `model` argument is duck-typed: it just needs `.encode_image(x)` and
`.encode_text(x)` returning L2-normalized embeddings. This lets the same
eval function work for zero-shot CLIP, the adapter model, and the full
fine-tuned model.

Note on FAISS:
    faiss-cpu is installed but, on macOS Apple Silicon, FAISS linked
    against one OpenMP runtime and sklearn/numpy against another will
    sometimes hard-crash Python at module import. We therefore default to
    a pure-numpy top-k (which is instant for our 1k-10k gallery sizes).
    Set RSICD_USE_FAISS=1 to opt back in for large galleries.
"""

import json
import os
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
from tqdm import tqdm

_USE_FAISS = os.environ.get("RSICD_USE_FAISS", "0") == "1"
if _USE_FAISS:
    try:
        import faiss  # type: ignore
    except Exception as e:
        print(f"[evaluate] FAISS requested but not importable: {e}. Falling back to numpy.")
        _USE_FAISS = False


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _iter_ids(ids_or_tensor):
    if isinstance(ids_or_tensor, torch.Tensor):
        return ids_or_tensor.tolist()
    return list(ids_or_tensor)


@torch.no_grad()
def encode_images(model, image_loader, device: str) -> tuple[np.ndarray, list]:
    """
    Encode all images in the loader into L2-normalized embedding vectors.

    Returns:
        embeddings: (N_images, D) float32
        imgids:     list of N_images image IDs in the same order
    """
    model.eval()
    all_embs:  list[np.ndarray] = []
    all_ids:   list            = []

    for images, imgids in tqdm(image_loader, desc="Encoding images"):
        if not isinstance(images, torch.Tensor):
            images = torch.as_tensor(images)
        images = images.to(device, non_blocking=True)

        feats = model.encode_image(images)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        all_embs.append(_to_numpy(feats).astype(np.float32))
        all_ids.extend(_iter_ids(imgids))

    return np.vstack(all_embs), all_ids


@torch.no_grad()
def encode_captions(model, caption_loader, device: str) -> tuple[np.ndarray, list]:
    """
    Encode all captions in the loader into L2-normalized embedding vectors.

    Returns:
        embeddings: (N_captions, D) float32
        imgids:     list of N_captions image IDs (the image each caption belongs to)
    """
    model.eval()
    all_embs:  list[np.ndarray] = []
    all_ids:   list            = []

    for tokens, imgids in tqdm(caption_loader, desc="Encoding captions"):
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.as_tensor(tokens)
        tokens = tokens.to(device, non_blocking=True)

        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        all_embs.append(_to_numpy(feats).astype(np.float32))
        all_ids.extend(_iter_ids(imgids))

    return np.vstack(all_embs), all_ids


def _faiss_topk(query: np.ndarray, gallery: np.ndarray, k: int) -> np.ndarray:
    """Return top-k indices into `gallery` for each row of `query` (cosine via IP on L2-normed)."""
    if _USE_FAISS:
        D = gallery.shape[1]
        index = faiss.IndexFlatIP(D)
        index.add(gallery.astype(np.float32))
        _, idx = index.search(query.astype(np.float32), k)
        return idx
    # Pure-numpy top-k via partition. O(N_gallery) memory, but for our
    # 1k-10k gallery this is instant (1k x 1k float32 = 4 MB).
    sims = query.astype(np.float32) @ gallery.astype(np.float32).T
    k_use = min(k, sims.shape[1] - 1)
    # argpartition is not stable for ties at the boundary, so we sort the
    # top-k for determinism in the reported numbers.
    part = np.argpartition(-sims, kth=k_use, axis=1)[:, :k_use]
    # reorder within the partition
    rows = np.arange(part.shape[0])[:, None]
    part_sorted = part[rows, np.argsort(-sims[rows, part], axis=1)]
    return part_sorted


def compute_recall_at_k(query_embeddings:   np.ndarray,
                        query_imgids:       list,
                        gallery_embeddings: np.ndarray,
                        gallery_imgids:     list,
                        k_values:           list = (1, 5, 10)) -> dict:
    """
    Compute Recall@K for cross-modal retrieval.

    A query is a "hit" at K if at least one of the top-K retrieved gallery items
    shares the query's imgid. (Multiple captions per image all share the same
    imgid, so a single matching image counts as a hit for all of its captions.)

    Returns dict like {"R@1": 12.34, "R@5": 30.0, "R@10": 45.0} (percentages).
    """
    query_embeddings   = np.ascontiguousarray(query_embeddings,   dtype=np.float32)
    gallery_embeddings = np.ascontiguousarray(gallery_embeddings, dtype=np.float32)

    max_k = max(k_values)
    top_k_indices = _faiss_topk(query_embeddings, gallery_embeddings, max_k)

    gallery_ids_arr = np.asarray(gallery_imgids)
    n               = len(query_imgids)
    results         = {}

    for k in k_values:
        hits = 0
        retrieved = top_k_indices[:, :k]                          # (n, k)
        retrieved_ids = gallery_ids_arr[retrieved]                 # (n, k)
        # hit if any of the top-k retrieved ids equals the query id
        hit_mask = (retrieved_ids == np.asarray(query_imgids)[:, None]).any(axis=1)
        hits = int(hit_mask.sum())
        results[f"R@{k}"] = round(100.0 * hits / n, 2)

    return results


@torch.no_grad()
def evaluate_model(model, retrieval_dataset, device: str,
                   split_name: str = "test",
                   image_batch_size: int = 64,
                   caption_batch_size: int = 256,
                   num_workers: int = 4) -> dict:
    """
    Full bidirectional evaluation: text->image and image->text Recall@K.

    Args:
        model:              anything with .encode_image() / .encode_text() (L2-normed).
        retrieval_dataset:  an RSICDRetrievalDataset (val or test).
        device:             torch device string.
        split_name:         label for the saved JSON.
        *_batch_size:       loader batch sizes.
        num_workers:        loader workers (0 is safest for quick smoke tests).

    Returns:
        dict ready to be json.dump'd:
            {
              "split": "test",
              "text_to_image":  {"R@1": ..., "R@5": ..., "R@10": ...},
              "image_to_text":  {"R@1": ..., "R@5": ..., "R@10": ...}
            }
    """
    print(f"\nEvaluating on {split_name} split...")

    image_loader   = retrieval_dataset.get_image_loader(
        batch_size=image_batch_size, num_workers=num_workers
    )
    caption_loader = retrieval_dataset.get_caption_loader(
        batch_size=caption_batch_size, num_workers=num_workers
    )

    img_embs, img_ids     = encode_images(model, image_loader, device)
    cap_embs, cap_imgids  = encode_captions(model, caption_loader, device)

    print(f"  {len(img_embs)} unique images, {len(cap_embs)} captions encoded.")

    # Text -> Image
    t2i = compute_recall_at_k(
        query_embeddings   = cap_embs,
        query_imgids       = cap_imgids,
        gallery_embeddings = img_embs,
        gallery_imgids     = img_ids,
    )
    # Image -> Text
    i2t = compute_recall_at_k(
        query_embeddings   = img_embs,
        query_imgids       = img_ids,
        gallery_embeddings = cap_embs,
        gallery_imgids     = cap_imgids,
    )

    print(f"  Text -> Image: R@1={t2i['R@1']:>5.2f}  R@5={t2i['R@5']:>5.2f}  R@10={t2i['R@10']:>5.2f}")
    print(f"  Image -> Text: R@1={i2t['R@1']:>5.2f}  R@5={i2t['R@5']:>5.2f}  R@10={i2t['R@10']:>5.2f}")

    return {
        "split":         split_name,
        "text_to_image": t2i,
        "image_to_text": i2t,
    }


def save_results(results: dict, path: str | Path) -> None:
    """Save a results dict to JSON, creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results -> {path}")
