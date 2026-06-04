"""
Generate all paper-quality figures (300 DPI PDFs in paper/figures/).

  fig_training_curve.pdf  - training loss + val R@1 over epochs
  fig_ablation_dim.pdf    - R@1 + trainable params vs hidden dim
  fig_qualitative.pdf     - 4 text queries x top-3 retrieved images (green/red)
  fig_failures.pdf        - 2 failure cases

Reads:
  results/metrics/training_history_<tag>.json      (default: training_history_adapter.json)
  results/metrics/ablations/ablation_hidden_dim.json
  results/checkpoints/<tag>_best.pt                (default: adapter_best.pt)

Run:  python scripts/06_qualitative.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")  # no display
import matplotlib.pyplot as plt

import torch

sys.path.insert(0, ".")

from src.dataset  import RSICDRetrievalDataset
from src.evaluate import encode_images, encode_captions
from src.model    import load_adapter_model
from src.utils    import get_device

# Where figures go
RESULTS_DIR  = Path("results/figures")
PAPER_FIGS   = Path("paper/figures")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PAPER_FIGS.mkdir(parents=True, exist_ok=True)

DEVICE = get_device()

# Visual style
plt.rcParams.update({
    "font.size":       11,
    "axes.titlesize":  12,
    "axes.labelsize":  11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":      120,
    "savefig.dpi":     300,
    "savefig.bbox":    "tight",
})

GREEN = "#0F6E56"
RED   = "#D85A30"
BLUE  = "#185FA5"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Training curve
# ─────────────────────────────────────────────────────────────────────────────
def fig_training_curve(history_path: str = "results/metrics/training_history_adapter.json",
                       out_name:      str = "fig_training_curve.pdf"):
    if not Path(history_path).exists():
        print(f"[skip] {history_path} not found")
        return
    with open(history_path) as f:
        history = json.load(f)
    if not history:
        print(f"[skip] {history_path} is empty")
        return

    epochs = [h["epoch"]            for h in history]
    losses = [h["train_loss"]       for h in history]
    val_r1 = [h["val_results"]["text_to_image"]["R@1"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))

    ax1.plot(epochs, losses, color=BLUE, linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training loss (InfoNCE)")
    ax1.set_title("Training Loss")
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, val_r1, color=GREEN, linewidth=2, marker="o", markersize=4)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Recall@1 (%)")
    ax2.set_title("Validation T\u2192I Recall@1")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out = PAPER_FIGS / out_name
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Ablation hidden-dim curve
# ─────────────────────────────────────────────────────────────────────────────
def fig_ablation_hidden_dim(abl_path: str = "results/metrics/ablations/ablation_hidden_dim.json",
                            out_name: str = "fig_ablation_dim.pdf"):
    if not Path(abl_path).exists():
        print(f"[skip] {abl_path} not found")
        return
    with open(abl_path) as f:
        data = json.load(f)
    if not data:
        print(f"[skip] {abl_path} is empty")
        return

    dims   = sorted(int(k) for k in data.keys())
    r1s    = [data[str(d)]["T2I_R@1"]                       for d in dims]
    params = [data[str(d)]["trainable_params"] / 1e6        for d in dims]

    fig, ax1 = plt.subplots(figsize=(7, 3.8))

    ax1.plot(dims, r1s, color=BLUE, linewidth=2, marker="o", markersize=7, label="R@1 (%)")
    ax1.set_xlabel("Adapter hidden dimension")
    ax1.set_ylabel("T\u2192I Recall@1 (%)", color=BLUE)
    ax1.tick_params(axis="y", labelcolor=BLUE)
    ax1.set_xticks(dims)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.bar(dims, params, width=25, alpha=0.25, color=RED, label="Params (M)")
    ax2.set_ylabel("Trainable parameters (M)", color=RED)
    ax2.tick_params(axis="y", labelcolor=RED)
    ax2.spines[["top"]].set_visible(False)

    plt.title("Adapter Size vs Retrieval Performance")
    fig.tight_layout()
    out = PAPER_FIGS / out_name
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Qualitative retrieval
# ─────────────────────────────────────────────────────────────────────────────
def fig_qualitative_retrieval(ckpt_path: str = "results/checkpoints/adapter_best.pt",
                              splits_dir: str = "data/splits",
                              images_dir: str = "data/raw/RSICD_images",
                              out_name:   str = "fig_qualitative.pdf",
                              n_queries:  int = 4,
                              top_k:      int = 3):
    if not Path(ckpt_path).exists():
        print(f"[skip] {ckpt_path} not found")
        return

    from PIL import Image
    import open_clip

    # Load adapter
    model = load_adapter_model(ckpt_path, DEVICE)
    model.eval()
    _, preprocess = open_clip.create_model_from_pretrained(
        "ViT-B-32", pretrained="openai", force_quick_gelu=True
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    test_retrieval = RSICDRetrievalDataset(
        Path(splits_dir) / "test.json", images_dir, preprocess, tokenizer
    )
    image_loader = test_retrieval.get_image_loader(batch_size=64, num_workers=0)
    img_embs, img_ids = encode_images(model, image_loader, DEVICE)

    imgid_to_filename = {img["imgid"]: img["image_filename"] for img in test_retrieval.images}

    with open(Path(splits_dir) / "test.json") as f:
        test_data = json.load(f)
    sample_pairs = [(img["captions"][0], img["imgid"]) for img in test_data["images"][:20]]

    queries      = sample_pairs[:n_queries]
    failure_qs   = sample_pairs[n_queries:n_queries + 2]

    def retrieve(caption: str, k: int = top_k) -> list:
        tokens = tokenizer([caption]).to(DEVICE)
        with torch.no_grad():
            txt = model.encode_text(tokens)
        sims = txt.cpu().numpy() @ img_embs.T
        top  = np.argsort(-sims, axis=1)[:, :k]
        return [img_ids[i] for i in top[0]]

    def make_figure(query_list, title, out_path, n_cols=top_k):
        n_rows = len(query_list)
        fig, axes = plt.subplots(n_rows, n_cols + 1, figsize=(3 * (n_cols + 1), 3 * n_rows))
        if n_rows == 1:
            axes = axes[np.newaxis, :]

        for row, (caption, gt_id) in enumerate(query_list):
            retrieved = retrieve(caption)
            axes[row, 0].text(0.5, 0.5, f'"{caption}"', wrap=True, ha="center", va="center",
                              fontsize=9, style="italic", transform=axes[row, 0].transAxes)
            axes[row, 0].axis("off")
            if row == 0:
                axes[row, 0].set_title("Query", fontweight="bold")
            for col, ret_id in enumerate(retrieved, 1):
                try:
                    img = Image.open(Path(images_dir) / imgid_to_filename[ret_id]).convert("RGB")
                    axes[row, col].imshow(img)
                    color = GREEN if ret_id == gt_id else RED
                    for spine in axes[row, col].spines.values():
                        spine.set_edgecolor(color)
                        spine.set_linewidth(3)
                except FileNotFoundError:
                    axes[row, col].text(0.5, 0.5, "Image\nnot found", ha="center", va="center")
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
                if row == 0:
                    axes[row, col].set_title(f"Top-{col}", fontweight="bold")

        plt.suptitle(title, fontsize=12, fontweight="bold", y=1.01)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved {out_path}")

    make_figure(queries, "Qualitative Retrieval Results (green = correct, red = incorrect)",
                PAPER_FIGS / out_name)
    make_figure(failure_qs, "Failure Cases", PAPER_FIGS / "fig_failures.pdf")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    fig_training_curve()
    fig_ablation_hidden_dim()
    fig_qualitative_retrieval()
    print(f"\nAll figures saved to {PAPER_FIGS}/")


if __name__ == "__main__":
    main()
