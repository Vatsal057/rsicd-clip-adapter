"""
PyTorch Dataset classes for RSICD image-caption retrieval.

Provides:
    RSICDDataset           - training dataset (one image, one caption per __getitem__)
    RSICDRetrievalDataset  - evaluation dataset (separate image and caption loaders)
    get_dataloaders        - convenience entry point used by train.py and the baselines

Reads:
    data/splits/{train,val,test}.json
    data/raw/RSICD_images/<file>.jpg
"""

import json
import os
from pathlib import Path
from typing import Callable, List, Tuple

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

import open_clip


class RSICDDataset(Dataset):
    """
    Training dataset.

    Each call to __getitem__ returns ONE image and ONE caption.
    The same image appears 5 times (once per caption) across the dataset,
    so each epoch the model sees every (image, caption) pair.

    Returns:
        image_tensor  - preprocessed image, float32, shape (3, 224, 224)
        caption_tokens- tokenized caption, long, shape (77,)
        imgid         - integer image id (for diagnostics only)
    """

    def __init__(self, split_json_path: str | os.PathLike,
                 images_dir:        str | os.PathLike,
                 preprocess:        Callable,
                 tokenizer:         Callable):
        with open(split_json_path) as f:
            data = json.load(f)

        self.pairs      = data["pairs"]
        self.images     = data["images"]
        self.images_dir = Path(images_dir)
        self.preprocess = preprocess
        self.tokenizer  = tokenizer

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        pair = self.pairs[idx]
        img_path = self.images_dir / pair["image_filename"]

        image = Image.open(img_path).convert("RGB")
        image_tensor = self.preprocess(image)

        # open_clip tokenizer expects a list of strings
        caption_tokens = self.tokenizer([pair["caption"]])[0]

        return image_tensor, caption_tokens, pair["imgid"]


class RSICDRetrievalDataset:
    """
    Evaluation-side dataset.

    Wraps the same split JSON but exposes two separate loaders:
      - one over all unique images
      - one over all (caption, imgid) pairs

    These two streams are encoded independently and then cross-matched
    with FAISS to compute T->I and I->T Recall@K.
    """

    def __init__(self, split_json_path: str | os.PathLike,
                 images_dir:        str | os.PathLike,
                 preprocess:        Callable,
                 tokenizer:         Callable):
        with open(split_json_path) as f:
            data = json.load(f)

        self.images     = data["images"]
        self.images_dir = Path(images_dir)
        self.preprocess = preprocess
        self.tokenizer  = tokenizer

        # Flat list of (caption, imgid) for text->image evaluation
        self.all_captions: List[Tuple[str, int]] = []
        for img in self.images:
            for cap in img["captions"]:
                self.all_captions.append((cap, img["imgid"]))

    def get_image_loader(self, batch_size: int = 64,
                         num_workers: int = 4) -> DataLoader:
        images = self.images
        images_dir = self.images_dir
        preprocess = self.preprocess

        class _ImageDS(Dataset):
            def __init__(self):
                pass

            def __len__(self):
                return len(images)

            def __getitem__(self, idx):
                info = images[idx]
                img = Image.open(images_dir / info["image_filename"]).convert("RGB")
                return preprocess(img), info["imgid"]

        return DataLoader(
            _ImageDS(), batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )

    def get_caption_loader(self, batch_size: int = 256,
                           num_workers: int = 4) -> DataLoader:
        captions = self.all_captions
        tokenizer = self.tokenizer

        class _CapDS(Dataset):
            def __init__(self):
                pass

            def __len__(self):
                return len(captions)

            def __getitem__(self, idx):
                text, imgid = captions[idx]
                tokens = tokenizer([text])[0]
                return tokens, imgid

        return DataLoader(
            _CapDS(), batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )


def _get_clip_preprocess_and_tokenizer(model_name: str = "ViT-B-32",
                                       pretrained: str = "openai"):
    """Load CLIP preprocessing transform and tokenizer.

    The OpenAI-pretrained ViT-B/32 uses QuickGELU activation. We must request
    the matching model config (`force_quick_gelu=True`) or open_clip emits a
    warning and the activations are slightly off, which degrades retrieval
    quality.
    """
    _, preprocess = open_clip.create_model_from_pretrained(
        model_name, pretrained=pretrained, force_quick_gelu=True
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    return preprocess, tokenizer


def get_dataloaders(splits_dir: str | os.PathLike = "data/splits",
                    images_dir: str | os.PathLike = "data/raw/RSICD_images",
                    batch_size: int = 64,
                    num_workers: int = 4,
                    model_name: str = "ViT-B-32",
                    pretrained: str = "openai"):
    """
    Convenience entry point used by training and evaluation scripts.

    Returns:
        train_loader    - DataLoader over training (image, caption) pairs
        val_retrieval   - RSICDRetrievalDataset for the validation split
        test_retrieval  - RSICDRetrievalDataset for the test split
        preprocess      - CLIP image preprocessing transform
        tokenizer       - CLIP tokenizer
    """
    splits_dir = Path(splits_dir)
    preprocess, tokenizer = _get_clip_preprocess_and_tokenizer(model_name, pretrained)

    train_dataset = RSICDDataset(
        splits_dir / "train.json", images_dir, preprocess, tokenizer
    )
    val_retrieval  = RSICDRetrievalDataset(
        splits_dir / "val.json",   images_dir, preprocess, tokenizer
    )
    test_retrieval = RSICDRetrievalDataset(
        splits_dir / "test.json",  images_dir, preprocess, tokenizer
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )

    return train_loader, val_retrieval, test_retrieval, preprocess, tokenizer
