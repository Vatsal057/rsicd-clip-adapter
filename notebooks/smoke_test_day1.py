"""
Day 1 smoke test: verify the dataset pipeline end-to-end.
Run:  .venv/bin/python notebooks/smoke_test_day1.py
"""

import sys
from pathlib import Path

sys.path.insert(0, ".")

from src.dataset import get_dataloaders
from src.utils import get_device


def main():
    print(f"Device: {get_device()}")
    train_loader, val_ret, test_ret, _, _ = get_dataloaders(
        splits_dir="data/splits",
        images_dir="data/raw/RSICD_images",
        batch_size=4,
        num_workers=0,
    )

    print(f"Train batches:        {len(train_loader)}")
    print(f"Train pairs:          {len(train_loader.dataset.pairs)}")
    print(f"Val images:           {len(val_ret.images)}")
    print(f"Val captions (flat):  {len(val_ret.all_captions)}")
    print(f"Test images:          {len(test_ret.images)}")
    print(f"Test captions (flat): {len(test_ret.all_captions)}")
    print()

    images, captions, imgids = next(iter(train_loader))
    print(f"Image tensor shape:   {images.shape}")
    print(f"Caption token shape:  {captions.shape}")
    print(f"Image IDs:            {imgids.tolist()}")
    print(f"Caption tokens dtype: {captions.dtype}")
    print(f"Image tensor dtype:   {images.dtype}")
    print(f"Image value range:    [{images.min().item():.3f}, {images.max().item():.3f}]")
    print()

    sample = train_loader.dataset.pairs[0]
    print(f"First training pair:")
    print(f"  filename:  {sample['image_filename']}")
    print(f"  imgid:     {sample['imgid']}")
    print(f"  caption:   {sample['caption']!r}")
    print(f"  sentid:    {sample['sentid']}")
    print()

    # Show a sample caption from val and test too
    for name, ds in [("val", val_ret), ("test", test_ret)]:
        print(f"{name} first image: {ds.images[0]['image_filename']}")
        print(f"  caption:  {ds.images[0]['captions'][0]!r}")
    print()
    print("=== Dataset smoke test PASSED ===")


if __name__ == "__main__":
    main()
