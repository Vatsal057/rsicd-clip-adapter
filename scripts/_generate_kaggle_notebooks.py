"""
Generate 5 Kaggle notebooks for the RSICD Adapter-CLIP project.

Each notebook is a self-contained .ipynb file with all cells pre-filled.
Run: python scripts/_generate_kaggle_notebooks.py

Outputs:
    notebooks/kaggle_s1_adapter_start.ipynb
    notebooks/kaggle_s2_adapter_mid.ipynb
    notebooks/kaggle_s3_adapter_end.ipynb
    notebooks/kaggle_s4_fullfinetune.ipynb
    notebooks/kaggle_s5_ablations_figures.ipynb
"""

import json
import os
from pathlib import Path

KAGGLE_DATASET_SLUG = "rsicd-image-caption-dataset"
KAGGLE_DATASET_PATH = f"/kaggle/input/{KAGGLE_DATASET_SLUG}"
GITHUB_REPO         = "https://github.com/Vatsal057/rsicd-clip-adapter.git"
KAGGLE_RESULTS_DIR  = "/kaggle/working/rsicd_results"

OUT_DIR = Path(__file__).resolve().parent.parent / "notebooks"


def cell_md(source):
    if isinstance(source, str):
        source = [source]
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def cell_code(source):
    if isinstance(source, str):
        source = [source]
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def shared_setup_cells(session_label):
    """Cells that every session starts with."""
    return [
        cell_md(f"# RSICD Adapter-CLIP — Session {session_label}\n"
                f"\n"
                f"Auto-generated from the rsicd-clip-adapter repo. Source of truth: `KAGGLE_RUNBOOK.md`.\n"
                f"\n"
                f"**Before running this notebook:**\n"
                f"1. Click **+ Add data** in the right panel\n"
                f"2. Search for and add:\n"
                f"   - `thedevastator/rsicd-image-caption-dataset` (the original CSV — only needed for session 1)\n"
                f"   - Any `rsicd-*` Datasets you saved in previous sessions (e.g. `rsicd-adapter-s1`)\n"
                f"3. Settings: **Accelerator = GPU P100 or T4**, **Internet = ON**\n"
                f"4. Click **Save Version → Save Output** at the end of this session"),
        cell_code(
            "# === Install + clone + env ===\n"
            "!pip install open_clip_torch faiss-cpu ftfy accelerate pyyaml -q\n"
            "!git clone https://github.com/Vatsal057/rsicd-clip-adapter.git\n"
            "%cd rsicd-clip-adapter\n"
            "import os\n"
            "os.environ[\"KMP_DUPLICATE_LIB_OK\"] = \"TRUE\"\n"
            "print(\"Repo cloned, deps installed.\")"
        ),
        cell_code(
            "# === Sanity: GPU + dataset location ===\n"
            "import torch\n"
            "from pathlib import Path\n"
            "print(f\"PyTorch:  {torch.__version__}\")\n"
            "print(f\"CUDA:     {torch.cuda.is_available()}\")\n"
            "if torch.cuda.is_available():\n"
            "    print(f\"GPU:      {torch.cuda.get_device_name(0)}\")\n"
            "print()\n"
            f"data_path = Path(\"{KAGGLE_DATASET_PATH}\")\n"
            "if data_path.exists():\n"
            "    print(f\"Dataset:  {data_path}\")\n"
            "    csvs = sorted(p.name for p in data_path.glob('*.csv'))\n"
            "    print(f\"  CSVs:   {csvs}\")\n"
            "else:\n"
            "    print(f\"WARN: {data_path} does not exist.\")\n"
            "    print(\"   -> Click '+ Add data' in the right panel.\")\n"
            "    print(\"   -> Search for 'rsicd-image-caption-dataset' (the thedevastator version).\")\n"
            "    print(\"   -> Click 'Add' to attach it to this notebook.\")\n"
            "    print()\n"
            "    # Show what IS attached, so the user can spot a typo in the dataset name\n"
            "    print(\"Currently attached under /kaggle/input/:\")\n"
            "    try:\n"
            "        attached = sorted(p.name for p in Path('/kaggle/input').iterdir())\n"
            "    except FileNotFoundError:\n"
            "        print(\"   (no /kaggle/input/ directory — running outside Kaggle?)\")\n"
            "        attached = []\n"
            "    for name in attached:\n"
            "        print(f\"   - {name}\")\n"
            "    if not any('rsicd' in n.lower() for n in attached):\n"
            "        print()\n"
            "        print(\"   -> No RSICD-looking dataset found. The cell above is the fix:\")\n"
            "        print(\"      click '+ Add data', search 'rsicd-image-caption-dataset', and Add.\")\n"
            "    raise SystemExit(0)  # Stop here so the user can fix and re-run"
        ),
    ]


def cell_convert_data():
    """Convert the Kaggle CSV to our standard layout. Idempotent (skips existing images)."""
    return cell_code(
        "# === Convert Kaggle CSV -> our standard layout (one-time per session) ===\n"
        "import os\n"
        "from pathlib import Path\n"
        "\n"
        "# Auto-detect: try the standard slug first, then fall back to any\n"
        "# attached directory that looks like an RSICD dataset.\n"
        "candidate_paths = [\n"
        f"    Path(\"{KAGGLE_DATASET_PATH}\"),\n"
        "    Path(\"/kaggle/input/rsicd-image-caption-dataset\"),\n"
        "    Path(\"/kaggle/input/rsicd-dataset\"),\n"
        "    Path(\"/kaggle/input/rsicd\"),\n"
        "]\n"
        "# Also: any /kaggle/input/ subdir that contains train.csv\n"
        "for p in Path(\"/kaggle/input\").iterdir():\n"
        "    if p.is_dir() and (p / \"train.csv\").exists():\n"
        "        candidate_paths.append(p)\n"
        "\n"
        "data_path = None\n"
        "for p in candidate_paths:\n"
        "    if p.exists() and (p / \"train.csv\").exists():\n"
        "        data_path = p\n"
        "        break\n"
        "\n"
        "if data_path is None:\n"
        "    print(\"ERROR: Could not find an attached RSICD dataset.\")\n"
        "    print(\"  Looked in:\")\n"
        "    for p in candidate_paths:\n"
        "        print(f\"    - {p}\")\n"
        "    print(\"  Attached under /kaggle/input/:\")\n"
        "    for p in sorted(Path('/kaggle/input').iterdir()):\n"
        "        print(f\"    - {p.name}\")\n"
        "    print()\n"
        "    print(\"  -> Click '+ Add data' and search 'rsicd-image-caption-dataset'.\")\n"
        "    raise SystemExit(1)\n"
        "\n"
        f"os.environ[\"RSICD_ARCHIVE\"] = str(data_path)\n"
        "print(f\"Using dataset: {{data_path}}\")\n"
        "\n"
        "!python scripts/00b_prepare_rsicd.py\n"
        "!ls data/raw/RSICD_images/ | wc -l"
    )


def cell_prepare_splits_and_baseline():
    return [
        cell_code(
            "# === Prepare splits + verify baseline ===\n"
            "!python scripts/01_prepare_splits.py\n"
            "!python scripts/02_run_baseline.py"
        ),
    ]


def cell_restore_checkpoint(prev_dataset_name, ckpt_basename="adapter_best.pt"):
    """Restore a checkpoint from a previous session's saved output."""
    return cell_code(
        f"# === Restore checkpoint from previous session ===\n"
        f"import shutil, os\n"
        f"from pathlib import Path\n"
        f"src = Path(\"/kaggle/input/{prev_dataset_name}/{ckpt_basename}\")\n"
        f"if not src.exists():\n"
        f"    print(f\"WARN: {{src}} not found. Did you forget to '+ Add data' {prev_dataset_name}?\")\n"
        f"else:\n"
        f"    Path(\"results/checkpoints\").mkdir(parents=True, exist_ok=True)\n"
        f"    shutil.copy(src, \"results/checkpoints/adapter_best.pt\")\n"
        f"    print(f\"Restored: {{src}}\")\n"
        f"    # Also restore training history if it was saved\n"
        f"    src_h = Path(\"/kaggle/input/{prev_dataset_name}/training_history_adapter.json\")\n"
        f"    if src_h.exists():\n"
        f"        Path(\"results/metrics\").mkdir(parents=True, exist_ok=True)\n"
        f"        shutil.copy(src_h, \"results/metrics/training_history_adapter.json\")\n"
        f"        print(\"Restored training history.\")"
    )


def cell_train(epochs_target, config_path="configs/adapter_base.yaml",
               resume=True, run_tag="adapter", extra_cfg=None):
    """Generate a train cell with the right num_epochs and optional resume_from."""
    extra_cfg = extra_cfg or {}
    extra_yaml = "\n".join(f'cfg["{k}"] = {repr(v)}' for k, v in extra_cfg.items())
    resume_block = (
        f'cfg["training"]["resume_from"] = "results/checkpoints/adapter_best.pt"'
        if resume else ""
    )
    return cell_code(
        f"# === Train (target epoch {epochs_target}) ===\n"
        f"import yaml, os\n"
        f"with open(\"{config_path}\") as f:\n"
        f"    cfg = yaml.safe_load(f)\n"
        f"cfg[\"training\"][\"num_epochs\"] = {epochs_target}\n"
        f"{resume_block}\n"
        f"{extra_yaml}\n"
        f"with open(\"configs/{run_tag}_session.yaml\", \"w\") as f:\n"
        f"    yaml.dump(cfg, f)\n"
        f"!python scripts/03_train_adapter.py configs/{run_tag}_session.yaml {run_tag}"
    )


def cell_full_finetune():
    return cell_code(
        "# === Full fine-tune baseline ===\n"
        "!python scripts/04_run_fullfinetune.py"
    )


def cell_ablations():
    return cell_code(
        "# === Ablations (placement + hidden_dim + data_size + residual) ===\n"
        "# This runs ~20 hours on P100. If you hit the 12 h Kaggle limit, save what you have\n"
        "# and resume in another notebook. Each ablation writes its own JSON when it finishes.\n"
        "!python scripts/05_ablations.py"
    )


def cell_figures():
    return cell_code(
        "# === Regenerate 4 paper figures with real data ===\n"
        "!python scripts/06_qualitative.py"
    )


def cell_package(dataset_name, files, dest_subdir=None):
    """Cell that copies output files to a /kaggle/working/ dir, ready to be saved as a Dataset."""
    copy_lines = []
    for f in files:
        copy_lines.append(
            f"src = \"{f}\"\n"
            f"if os.path.exists(src):\n"
            f"    dst = os.path.join(out_dir, os.path.basename(src)) if not os.path.isdir(src) else out_dir\n"
            f"    if os.path.isdir(src):\n"
            f"        shutil.copytree(src, os.path.join(out_dir, os.path.basename(src)), dirs_exist_ok=True)\n"
            f"    else:\n"
            f"        shutil.copy(src, dst)\n"
            f"    print(f\"  + {{src}} -> {{out_dir}}\")"
        )
    copy_block = "\n".join(copy_lines)
    return cell_code(
        f"# === Package output for next session ===\n"
        f"# After this cell, click 'Save Version' (top right) with 'Save Output' enabled.\n"
        f"# Then go to the Output tab -> 'New Dataset' -> name it '{dataset_name}'.\n"
        f"# The next session will attach this Dataset via '+ Add data'.\n"
        f"import shutil, os\n"
        f"out_dir = \"/kaggle/working/{dataset_name}\"\n"
        f"os.makedirs(out_dir, exist_ok=True)\n"
        f"{copy_block}\n"
        f"print(f\"\\nReady. Save this notebook version with 'Save Output' ON, then convert output to Dataset '{dataset_name}'.\")"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Build the 5 notebooks
# ─────────────────────────────────────────────────────────────────────────────

def build_s1():
    """Session 1: Adapter epochs 1-7. The first session does data conversion + baseline."""
    return notebook(shared_setup_cells("1 of 5 — Adapter training (epochs 1-7)") + [
        cell_convert_data(),
        *cell_prepare_splits_and_baseline(),
        cell_train(epochs_target=7, resume=False, run_tag="adapter"),
        cell_package(
            "rsicd-adapter-s1",
            files=[
                "results/checkpoints/adapter_best.pt",
                "results/metrics/training_history_adapter.json",
                "results/metrics/adapter_results.json",
            ],
        ),
        cell_md("## Done!\n"
                "**Action required:** Click **Save Version** (top right) → make sure **Save Output** is **ON** → **Save**.\n"
                "\n"
                "After the version is saved, go to the **Output** tab at the bottom → click **New Dataset** → name it `rsicd-adapter-s1` → **Create**.\n"
                "\n"
                "Then open `kaggle_s2_adapter_mid.ipynb` for session 2."),
    ])


def build_s2(prev_dataset="rsicd-adapter-s1"):
    """Session 2: Adapter epochs 8-14. Restores from s1."""
    return notebook(shared_setup_cells("2 of 5 — Adapter training (epochs 8-14)") + [
        cell_md("## Required: attach the previous session's Dataset\n"
                "Click **+ Add data** → search `rsicd-adapter-s1` → **Add**."),
        cell_restore_checkpoint(prev_dataset),
        cell_train(epochs_target=14, resume=True, run_tag="adapter"),
        cell_package(
            "rsicd-adapter-s2",
            files=[
                "results/checkpoints/adapter_best.pt",
                "results/metrics/training_history_adapter.json",
                "results/metrics/adapter_results.json",
            ],
        ),
        cell_md("## Done!\n"
                "Save the version, then convert output to Dataset `rsicd-adapter-s2`."),
    ])


def build_s3(prev_dataset="rsicd-adapter-s2"):
    """Session 3: Adapter epochs 15-20. Restores from s2."""
    return notebook(shared_setup_cells("3 of 5 — Adapter training (epochs 15-20)") + [
        cell_md("## Required: attach the previous session's Dataset\n"
                "Click **+ Add data** → search `rsicd-adapter-s2` → **Add**."),
        cell_restore_checkpoint(prev_dataset),
        cell_train(epochs_target=20, resume=True, run_tag="adapter"),
        cell_package(
            "rsicd-adapter-s3",
            files=[
                "results/checkpoints/adapter_best.pt",
                "results/metrics/training_history_adapter.json",
                "results/metrics/adapter_results.json",
                "results/metrics/adapter_results.json",
            ],
        ),
        cell_md("## Done!\n"
                "Save the version, then convert output to Dataset `rsicd-adapter-s3`.\n"
                "\n"
                "**This is the final adapter run.** The numbers in `adapter_results.json` are the paper's headline result."),
    ])


def build_s4(prev_dataset="rsicd-adapter-s3"):
    """Session 4: Full fine-tune baseline. Restores the adapter from s3 only to inherit the
    training-history/structure. Trains a separate, full-fine-tuned CLIP."""
    return notebook(shared_setup_cells("4 of 5 — Full fine-tune baseline") + [
        cell_md("## Required: attach the previous session's Dataset\n"
                "Click **+ Add data** → search `rsicd-adapter-s3` → **Add** (we use its environment "
                "but the full FT script is independent)."),
        cell_restore_checkpoint(prev_dataset),
        cell_full_finetune(),
        cell_package(
            "rsicd-fullfinetune",
            files=[
                "results/checkpoints/fullfinetune_best.pt",
                "results/metrics/fullfinetune_results.json",
            ],
        ),
        cell_md("## Done!\n"
                "Save the version, then convert output to Dataset `rsicd-fullfinetune`.\n"
                "\n"
                "**Heads up:** the full-fine-tune checkpoint is ~600 MB. Kaggle Datasets cap at "
                "~20 GB so it's fine, but downloading it from the notebook output to your Mac will "
                "take a while. Don't commit it to GitHub."),
    ])


def build_s5(prev_adapter="rsicd-adapter-s3", prev_fullft="rsicd-fullfinetune"):
    """Session 5: Ablations + figures. Longest session (~22h), but ablations are independent so
    you can split into 2-3 sessions if needed."""
    return notebook(shared_setup_cells("5 of 5 — Ablations + Figures") + [
        cell_md("## Required: attach the previous sessions' Datasets\n"
                "Click **+ Add data** → search and add:\n"
                f"- `rsicd-adapter-s3` (for the adapter checkpoint used by figures)\n"
                f"- `rsicd-fullfinetune` (for the full-fine-tune metrics used by figures)"),
        cell_restore_checkpoint(prev_adapter),
        cell_md("If you want to split this session, run cells one at a time. Each ablation writes "
                "its own JSON when it finishes, so a mid-run disconnect loses at most one ablation."),
        cell_ablations(),
        cell_figures(),
        cell_package(
            "rsicd-final",
            files=[
                "results",
                "paper/figures",
                "paper/main.tex",
                "paper/refs.bib",
            ],
        ),
        cell_md("## Done!\n"
                "Save the version, then convert output to Dataset `rsicd-final`.\n"
                "\n"
                "**This is the last session.** Download `rsicd-final` from Kaggle → "
                "merge into your local repo → fill in `[TBD]` in `main.tex` → push to GitHub."),
    ])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    notebooks = {
        "kaggle_s1_adapter_start.ipynb":     build_s1(),
        "kaggle_s2_adapter_mid.ipynb":       build_s2(),
        "kaggle_s3_adapter_end.ipynb":       build_s3(),
        "kaggle_s4_fullfinetune.ipynb":      build_s4(),
        "kaggle_s5_ablations_figures.ipynb": build_s5(),
    }
    for name, nb in notebooks.items():
        path = OUT_DIR / name
        with open(path, "w") as f:
            json.dump(nb, f, indent=1)
        size_kb = path.stat().st_size / 1024
        print(f"  wrote {path.name}  ({size_kb:.1f} KB, {len(nb['cells'])} cells)")
    print(f"\nAll 5 notebooks in {OUT_DIR}/")
    print("Upload them to Kaggle one at a time, in order (s1, s2, s3, s4, s5).")


if __name__ == "__main__":
    main()
