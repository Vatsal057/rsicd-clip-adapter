"""
Baseline 1: Zero-shot CLIP on the RSICD test set.

Loads CLIP ViT-B/32 with no fine-tuning, runs text->image and image->text
retrieval on the test split, and writes the numbers to
results/metrics/baseline_zeroshot.json.

Run:  python scripts/02_run_baseline.py
"""

import json
import os
import sys
from pathlib import Path

import open_clip
import torch

sys.path.insert(0, ".")

from src.dataset   import RSICDRetrievalDataset
from src.evaluate  import evaluate_model, save_results
from src.utils     import get_device, set_seed, print_model_summary


def main():
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    print("Loading CLIP ViT-B/32 (zero-shot, no fine-tuning)...")
    model, preprocess = open_clip.create_model_from_pretrained(
        "ViT-B-32", pretrained="openai", force_quick_gelu=True
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(device)
    model.eval()
    print_model_summary(model, "CLIP ViT-B/32 (zero-shot)")

    splits_dir = Path(os.environ.get("RSICD_SPLITS",  "data/splits"))
    images_dir = Path(os.environ.get("RSICD_IMAGES_DIR", "data/raw/RSICD_images"))

    test_retrieval = RSICDRetrievalDataset(
        splits_dir / "test.json", images_dir, preprocess, tokenizer
    )

    results = evaluate_model(
        model, test_retrieval, device,
        split_name="test", image_batch_size=64, caption_batch_size=256, num_workers=0,
    )
    results["model"]            = "zero_shot_clip"
    results["clip_model"]       = "ViT-B-32"
    results["trainable_params"] = 0
    results["device"]           = device

    save_results(results, "results/metrics/baseline_zeroshot.json")
    print("\nZero-shot CLIP baseline DONE. These are the 'before' numbers.")
    print(f"  T->I R@1: {results['text_to_image']['R@1']}%  (expected ~5-15% on RSICD)")
    print(f"  I->T R@1: {results['image_to_text']['R@1']}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
