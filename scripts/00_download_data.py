"""
Download and verify the RSICD dataset.

Run:  python scripts/00_download_data.py

Tries the Kaggle CLI first. If that fails (no kaggle.json, no network,
or no Kaggle account), prints manual download instructions and verifies
whatever already exists at data/raw/.

Environment variables:
    RSICD_IMAGES_DIR  - override default data/raw/RSICD_images
    RSICD_JSON        - override default data/raw/dataset_rsicd.json
"""

import os
import json
import sys
import shutil
from pathlib import Path


def default_data_dir() -> Path:
    return Path(os.environ.get("RSICD_DATA_DIR", "data/raw"))


def images_dir() -> Path:
    return Path(os.environ.get("RSICD_IMAGES_DIR", default_data_dir() / "RSICD_images"))


def captions_file() -> Path:
    return Path(os.environ.get("RSICD_JSON", default_data_dir() / "dataset_rsicd.json"))


def download_via_kaggle() -> bool:
    """Try to download via the Kaggle CLI. Returns True on success."""
    if shutil.which("kaggle") is None:
        print("Kaggle CLI not found. Install with: pip install kaggle")
        return False
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print(f"No Kaggle credentials at {kaggle_json}.")
        print("Either place kaggle.json there, or follow the manual instructions below.")
        return False

    target = default_data_dir()
    target.mkdir(parents=True, exist_ok=True)
    print(f"Downloading RSICD via Kaggle into {target} ...")
    rc = os.system(
        f"kaggle datasets download -d thedevastator/rsicd-image-caption-dataset "
        f"--path {target} --unzip"
    )
    return rc == 0


def print_manual_instructions() -> None:
    target = default_data_dir().absolute()
    print("=" * 60)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    print("1. Go to: https://www.kaggle.com/datasets/thedevastator/rsicd-image-caption-dataset")
    print("   (free Kaggle account required)")
    print(f"2. Extract the zip into: {target}/")
    print("3. Expected final layout:")
    print(f"   {target}/RSICD_images/   (folder of .jpg files)")
    print(f"   {target}/dataset_rsicd.json  (captions file)")
    print("")
    print("Mirror via GitHub (captions only, no images):")
    print("   git clone https://github.com/201528014227051/RSICD_optimal.git data/raw/")
    print("   then download images from the link in the README of that repo.")
    print("=" * 60)


def verify_data(verbose: bool = True) -> bool:
    """Verify the dataset is in place. Returns True if it looks good."""
    img_dir = images_dir()
    cap     = captions_file()

    ok = True
    if not img_dir.exists():
        print(f"[FAIL] Images folder not found at {img_dir}")
        ok = False
    if not cap.exists():
        print(f"[FAIL] Captions file not found at {cap}")
        ok = False
    if not ok:
        return False

    image_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    n_images    = len(image_files)

    with open(cap) as f:
        data = json.load(f)
    images_in_json = len(data["images"])
    total_captions = sum(len(img["sentences"]) for img in data["images"])

    if verbose:
        print(f"Found {n_images} images in {img_dir}")
        print(f"Found {images_in_json} image entries in JSON")
        print(f"Found {total_captions} total captions")
        if images_in_json:
            print(f"Average captions per image: {total_captions / images_in_json:.1f}")

    if n_images < 5000 or images_in_json < 5000:
        print(f"[WARN] Fewer images than expected (RSICD has ~10,921). Got {n_images}.")
        ok = False
    else:
        print("[OK] Data verification PASSED")
    return ok


def main() -> int:
    default_data_dir().mkdir(parents=True, exist_ok=True)
    downloaded = False
    try:
        downloaded = download_via_kaggle()
    except Exception as e:
        print(f"Kaggle download raised: {e}")
        downloaded = False

    if not downloaded:
        print("Kaggle download did not succeed.")
        print_manual_instructions()

    if verify_data():
        return 0
    print("\nDataset not ready. Re-run this script after placing the files manually.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
