"""
One-time converter: HuggingFace-style RSICD CSV (images inlined as bytes)
                  -> folder of jpg files + dataset_rsicd.json

This produces the exact layout the rest of the project expects:

    data/raw/RSICD_images/<NNNNN>.jpg
    data/raw/dataset_rsicd.json

The source CSVs are the HuggingFace "arampacha/rsicd" layout:
    archive/{train,valid,test}.csv
with columns (filename, captions, image). The "captions" field is a
single Python-source-style string of 5 concatenated quoted captions,
e.g. "'cap a.' 'cap b.' 'cap c.' 'cap d.' 'cap e.'". The "image"
field is a HuggingFace image dict: {'bytes': b'\\xff\\xd8\\xff...'}.

We use the *predefined* train/valid/test split from the source files
(8,734 / 1,094 / 1,093 = 80/10/10) and stamp it onto each image as
img["split"], so scripts/01_prepare_splits.py will pick it up.

Idempotent: re-running this script overwrites outputs but is safe.

Run:  python scripts/00b_prepare_rsicd.py
"""

import ast
import csv
import json
import os
import re
import sys
from pathlib import Path


ARCHIVE_DIR  = Path(os.environ.get("RSICD_ARCHIVE",  "archive"))
RAW_DIR      = Path(os.environ.get("RSICD_RAW_DIR",  "data/raw"))
IMAGES_DIR   = Path(os.environ.get("RSICD_IMAGES_DIR", RAW_DIR / "RSICD_images"))
OUT_JSON     = Path(os.environ.get("RSICD_JSON",     RAW_DIR / "dataset_rsicd.json"))


def split_concatenated_captions(blob: str) -> list[str]:
    """
    The HF CSV stores 5 captions as a single Python list literal of length 1,
    where the single element is the 5 captions concatenated. The boundaries
    between captions are unreliable in this preprocessing (no separator,
    inconsistent capitalization of the next caption), so we deterministically
    take the FIRST sentence as the canonical caption and discard the rest.

    This means we end up with 1 (image, caption) pair per image, totaling
    10,921 pairs instead of the canonical 54,605 (5 per image). The paper
    notes this and notes that 5x caption augmentation is not used.
    """
    try:
        parsed = ast.literal_eval(blob)
    except Exception:
        return [blob.strip().lstrip("'")]
    if isinstance(parsed, list):
        joined = " ".join(str(x) for x in parsed)
    else:
        joined = str(parsed)
    joined = joined.strip().strip("'").strip()

    first = re.split(r"(?<=[.!?])\s*", joined, maxsplit=1)[0].strip()
    if not first:
        first = joined
    if not first.endswith((".", "!", "?")):
        first = first + "."
    return [first]


def load_image_bytes(image_field: str) -> bytes:
    """
    The HF CSV "image" column is a Python dict literal:
        {'bytes': b'\\xff\\xd8\\xff...'}
    ast.literal_eval gives us the dict, then we pull out ['bytes'].
    """
    try:
        d = ast.literal_eval(image_field)
    except Exception as e:
        raise ValueError(f"Could not parse image dict: {e}") from e
    if isinstance(d, dict) and "bytes" in d:
        b = d["bytes"]
        if isinstance(b, (bytes, bytearray, memoryview)):
            return bytes(b)
    if isinstance(d, (bytes, bytearray, memoryview)):
        return bytes(d)
    raise ValueError("image field is not in expected HF {'bytes': ...} format")


def main() -> int:
    if not ARCHIVE_DIR.exists():
        print(f"[FAIL] Archive directory not found: {ARCHIVE_DIR.absolute()}")
        print("Expected layout: archive/{train,valid,test}.csv")
        return 1

    csv_paths = {split: ARCHIVE_DIR / f"{split}.csv" for split in ("train", "valid", "test")}
    for split, p in csv_paths.items():
        if not p.exists():
            print(f"[FAIL] Missing {p}")
            return 1

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    csv.field_size_limit(sys.maxsize)

    images_out: list[dict] = []
    total_imgs = 0
    total_caps = 0
    skipped    = 0

    for split in ("train", "valid", "test"):
        n_split_imgs = 0
        n_split_caps = 0
        print(f"\n=== {split} ===")
        with open(csv_paths[split]) as f:
            for row_idx, row in enumerate(csv.DictReader(f)):
                filename_raw = row["filename"]
                # The CSVs use "rsicd_images/00001.jpg". Strip the prefix
                # so the JSON matches the IMAGES_DIR layout.
                fname = Path(filename_raw).name
                img_path = IMAGES_DIR / fname

                # Write image bytes (only if missing, to make re-runs cheap)
                if not img_path.exists():
                    try:
                        img_bytes = load_image_bytes(row["image"])
                        with open(img_path, "wb") as g:
                            g.write(img_bytes)
                    except Exception as e:
                        print(f"  [skip] {fname}: cannot decode image: {e}")
                        skipped += 1
                        continue
                # else: assume the file is already on disk; cheap trust.

                captions = split_concatenated_captions(row["captions"])

                images_out.append({
                    "filename": fname,
                    "imgid":    total_imgs,
                    "split":    split,
                    "sentences": [
                        {"raw": cap, "sentid": i} for i, cap in enumerate(captions)
                    ],
                })
                total_imgs += 1
                n_split_imgs += 1
                n_split_caps += len(captions)

                if total_imgs % 1000 == 0:
                    print(f"  processed {total_imgs} images, {n_split_imgs} in {split}")
        total_caps += n_split_caps
        print(f"  {split}: {n_split_imgs} images, {n_split_caps} captions")

    out_doc = {"images": images_out}
    with open(OUT_JSON, "w") as f:
        json.dump(out_doc, f, indent=2)

    print(f"\nWrote {len(images_out)} image entries -> {OUT_JSON}")
    print(f"Wrote jpg files -> {IMAGES_DIR}")
    print(f"Total images: {total_imgs}")
    print(f"Total captions: {total_caps}")
    print(f"Skipped (decode errors): {skipped}")
    print("Done. You can now run: python scripts/01_prepare_splits.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
