"""
Create fixed 80/10/10 train/val/test splits for RSICD.

CRITICAL: Run this ONCE. Never re-run or change the seed.
The split JSONs are the ground truth for every experiment in the paper.

Reads:  data/raw/dataset_rsicd.json
Writes: data/splits/{train,val,test,metadata}.json

Environment overrides:
    RSICD_JSON    - source captions JSON
    RSICD_SPLITS  - output directory (default data/splits)
"""

import json
import os
import random
import sys
from pathlib import Path


SEED        = 42
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10


def main() -> int:
    src_json  = Path(os.environ.get("RSICD_JSON", "data/raw/dataset_rsicd.json"))
    splits_dir = Path(os.environ.get("RSICD_SPLITS", "data/splits"))
    splits_dir.mkdir(parents=True, exist_ok=True)

    if not src_json.exists():
        print(f"[FAIL] Source JSON not found at {src_json}")
        print("Run scripts/00_download_data.py first (or place files manually).")
        return 1

    with open(src_json) as f:
        data = json.load(f)

    images = data["images"]
    print(f"Total images in source: {len(images)}")

    has_predefined = all("split" in img for img in images)

    if has_predefined:
        print("Using pre-assigned splits from the source JSON.")
        # Some RSICD distributions name the val split "val", others "valid".
        # We canonicalize to "val" for the rest of the codebase.
        for img in images:
            if img.get("split") == "valid":
                img["split"] = "val"
        train = [img for img in images if img["split"] == "train"]
        val   = [img for img in images if img["split"] == "val"]
        test  = [img for img in images if img["split"] == "test"]
    else:
        print(f"No pre-assigned splits; creating random splits with seed={SEED}.")
        random.seed(SEED)
        shuffled = images.copy()
        random.shuffle(shuffled)
        n        = len(shuffled)
        n_train  = int(n * TRAIN_RATIO)
        n_val    = int(n * VAL_RATIO)
        train    = shuffled[:n_train]
        val      = shuffled[n_train:n_train + n_val]
        test     = shuffled[n_train + n_val:]

    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    def build_pairs(image_list):
        pairs = []
        for img in image_list:
            for sent in img["sentences"]:
                pairs.append({
                    "image_filename": img["filename"],
                    "imgid":          img["imgid"],
                    "caption":        sent["raw"],
                    "sentid":         sent["sentid"],
                })
        return pairs

    def build_image_list(image_list):
        return [
            {
                "image_filename": img["filename"],
                "imgid":          img["imgid"],
                "captions":       [s["raw"] for s in img["sentences"]],
            }
            for img in image_list
        ]

    splits = {
        "train": {"pairs": build_pairs(train),      "images": build_image_list(train)},
        "val":   {"pairs": build_pairs(val),        "images": build_image_list(val)},
        "test":  {"pairs": build_pairs(test),       "images": build_image_list(test)},
    }

    for split_name, split_data in splits.items():
        out = splits_dir / f"{split_name}.json"
        with open(out, "w") as f:
            json.dump(split_data, f, indent=2)
        print(f"Saved {split_name}: "
              f"{len(split_data['pairs'])} pairs, "
              f"{len(split_data['images'])} images -> {out}")

    meta = {
        "seed":                   SEED,
        "train_ratio":            TRAIN_RATIO,
        "val_ratio":              VAL_RATIO,
        "test_ratio":             TEST_RATIO,
        "n_train_images":         len(train),
        "n_val_images":           len(val),
        "n_test_images":          len(test),
        "used_predefined_splits": has_predefined,
        "source_json":            str(src_json),
    }
    with open(splits_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata -> {splits_dir / 'metadata.json'}")
    print("Splits created. DO NOT re-run this script.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
