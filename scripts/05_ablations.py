"""
4 ablation experiments. Each isolates one design choice from the base
adapter config. Results feed directly into Tables 2-5 of the paper.

  1. Adapter placement (image-only / text-only / both)
  2. Adapter hidden dimension (64 / 128 / 256 / 512)
  3. Training data size (25% / 50% / 75% / 100%)
  4. Residual connection (with / without)

Each experiment saves its own JSON to results/metrics/ablations/.

Usage:
    python scripts/05_ablations.py                    # all 4 ablations
    python scripts/05_ablations.py placement          # one ablation
    python scripts/05_ablations.py hidden_dim data_size   # subset
"""

import json
import os
import random
import shutil
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

import yaml

sys.path.insert(0, ".")

from src.model import CLIPAdapterModel
from src.train import train
from src.utils import get_device


METRICS_DIR    = Path("results/metrics/ablations")
METRICS_DIR.mkdir(parents=True, exist_ok=True)
BASE_CONFIG    = "configs/adapter_base.yaml"
SPLITS_DIR     = Path("data/splits")


def _load_base() -> dict:
    with open(BASE_CONFIG) as f:
        return yaml.safe_load(f)


def _dump_tmp(cfg: dict, tag: str) -> str:
    """Write a modified config to a temp file and return the path."""
    path = Path(tempfile.gettempdir()) / f"abl_{tag}.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f)
    return str(path)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Adapter placement
# ─────────────────────────────────────────────────────────────────────────────
def ablation_placement() -> dict:
    print("\n" + "=" * 60)
    print("ABLATION 1: Adapter placement")
    print("=" * 60)
    out = {}
    runs = [
        (True,  False, "image_only"),
        (False, True,  "text_only"),
        (True,  True,  "both"),
    ]
    for img_on, txt_on, label in runs:
        print(f"\n--- placement = {label} ---")
        cfg = _load_base()
        cfg["model"]["adapter_on_image"] = img_on
        cfg["model"]["adapter_on_text"]  = txt_on
        tmp = _dump_tmp(cfg, f"placement_{label}")
        res = train(tmp, run_tag=f"placement_{label}")
        out[label] = {
            "T2I_R@1":  res["text_to_image"]["R@1"],
            "T2I_R@5":  res["text_to_image"]["R@5"],
            "T2I_R@10": res["text_to_image"]["R@10"],
            "I2T_R@1":  res["image_to_text"]["R@1"],
            "I2T_R@5":  res["image_to_text"]["R@5"],
            "I2T_R@10": res["image_to_text"]["R@10"],
        }
        print(f"  {label}: T->I R@1 = {out[label]['T2I_R@1']}%")
    with open(METRICS_DIR / "ablation_placement.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Ablation 1 saved.")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2. Hidden dimension
# ─────────────────────────────────────────────────────────────────────────────
def ablation_hidden_dim() -> dict:
    print("\n" + "=" * 60)
    print("ABLATION 2: Adapter hidden dimension")
    print("=" * 60)
    out = {}
    for hidden_dim in [64, 128, 256, 512]:
        print(f"\n--- hidden_dim = {hidden_dim} ---")
        cfg = _load_base()
        cfg["model"]["hidden_dim"] = hidden_dim
        # Compute trainable-param count for this config
        dummy = CLIPAdapterModel(
            clip_model_name=cfg["model"]["clip_model_name"],
            clip_pretrained=cfg["model"]["clip_pretrained"],
            hidden_dim=hidden_dim,
            dropout=cfg["model"]["dropout"],
            adapter_on_image=cfg["model"]["adapter_on_image"],
            adapter_on_text=cfg["model"]["adapter_on_text"],
            use_residual=cfg["model"].get("use_residual", True),
        )
        n_params = dummy.count_trainable_params()
        del dummy
        tmp = _dump_tmp(cfg, f"dim_{hidden_dim}")
        res = train(tmp, run_tag=f"dim_{hidden_dim}")
        out[str(hidden_dim)] = {
            "hidden_dim":       hidden_dim,
            "trainable_params": n_params,
            "T2I_R@1":  res["text_to_image"]["R@1"],
            "T2I_R@5":  res["text_to_image"]["R@5"],
            "T2I_R@10": res["text_to_image"]["R@10"],
            "I2T_R@1":  res["image_to_text"]["R@1"],
        }
        print(f"  dim={hidden_dim} ({n_params:,} params): T->I R@1 = {out[str(hidden_dim)]['T2I_R@1']}%")
    with open(METRICS_DIR / "ablation_hidden_dim.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Ablation 2 saved.")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. Training data size
# ─────────────────────────────────────────────────────────────────────────────
def ablation_data_size() -> dict:
    print("\n" + "=" * 60)
    print("ABLATION 3: Training data fraction")
    print("=" * 60)
    out = {}

    with open(SPLITS_DIR / "train.json") as f:
        full_train = json.load(f)
    full_pairs = full_train["pairs"]

    # We need a writable splits dir that the config can point to.
    work_dir = Path(tempfile.mkdtemp(prefix="rsicd_abl_size_"))

    # Copy val/test unchanged; train.json will be replaced per fraction.
    for split in ("val", "test"):
        shutil.copy(SPLITS_DIR / f"{split}.json", work_dir / f"{split}.json")
    # Copy the full train.json as the starting point too
    shutil.copy(SPLITS_DIR / "train.json", work_dir / "train.json")

    for fraction in [0.25, 0.50, 0.75, 1.00]:
        print(f"\n--- data_fraction = {fraction} ---")
        n_pairs = int(len(full_pairs) * fraction)
        random.seed(42)
        subset_pairs = random.sample(full_pairs, n_pairs)

        with open(work_dir / "train.json", "w") as f:
            json.dump({"pairs": subset_pairs, "images": full_train["images"]}, f)

        cfg = _load_base()
        cfg["data"]["splits_dir"] = str(work_dir)
        tmp = _dump_tmp(cfg, f"size_{fraction}")
        res = train(tmp, run_tag=f"size_{fraction}")
        out[str(fraction)] = {
            "fraction":      fraction,
            "n_train_pairs": n_pairs,
            "T2I_R@1":  res["text_to_image"]["R@1"],
            "T2I_R@5":  res["text_to_image"]["R@5"],
            "T2I_R@10": res["text_to_image"]["R@10"],
            "I2T_R@1":  res["image_to_text"]["R@1"],
        }
        print(f"  fraction={fraction} ({n_pairs} pairs): T->I R@1 = {out[str(fraction)]['T2I_R@1']}%")

    with open(METRICS_DIR / "ablation_data_size.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Ablation 3 saved.")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 4. Residual connection (with / without)
# ─────────────────────────────────────────────────────────────────────────────
def ablation_residual() -> dict:
    print("\n" + "=" * 60)
    print("ABLATION 4: Residual connection")
    print("=" * 60)
    out = {}

    # With residual (default base config)
    print("\n--- residual = with ---")
    res_with = train(BASE_CONFIG, run_tag="residual_with")
    out["with_residual"] = {
        "T2I_R@1":  res_with["text_to_image"]["R@1"],
        "T2I_R@5":  res_with["text_to_image"]["R@5"],
        "T2I_R@10": res_with["text_to_image"]["R@10"],
        "I2T_R@1":  res_with["image_to_text"]["R@1"],
    }

    # Without residual
    print("\n--- residual = without ---")
    cfg = _load_base()
    cfg["model"]["use_residual"] = False
    tmp = _dump_tmp(cfg, "residual_without")
    res_without = train(tmp, run_tag="residual_without")
    out["without_residual"] = {
        "T2I_R@1":  res_without["text_to_image"]["R@1"],
        "T2I_R@5":  res_without["text_to_image"]["R@5"],
        "T2I_R@10": res_without["text_to_image"]["R@10"],
        "I2T_R@1":  res_without["image_to_text"]["R@1"],
    }

    print(f"  with residual:    T->I R@1 = {out['with_residual']['T2I_R@1']}%")
    print(f"  without residual: T->I R@1 = {out['without_residual']['T2I_R@1']}%")

    with open(METRICS_DIR / "ablation_residual.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Ablation 4 saved.")
    return out


_ABLATIONS = {
    "placement":   ablation_placement,
    "hidden_dim":  ablation_hidden_dim,
    "data_size":   ablation_data_size,
    "residual":    ablation_residual,
}


def main():
    get_device()  # warm up
    if len(sys.argv) > 1:
        names = sys.argv[1:]
    else:
        names = list(_ABLATIONS.keys())
    print(f"Running ablations: {names}")
    for name in names:
        if name not in _ABLATIONS:
            print(f"  [skip] Unknown ablation: {name}")
            continue
        _ABLATIONS[name]()
    print(f"\nAll requested ablations complete. JSONs in {METRICS_DIR}/")


if __name__ == "__main__":
    main()
