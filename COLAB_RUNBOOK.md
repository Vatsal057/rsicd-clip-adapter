# Colab T4 Runbook — RSICD Adapter-CLIP

> Read this top to bottom before starting. Total T4 time: ~30 hours.
> Free Colab tier: 12-hour session limit + cooldown. **Plan for 3 sessions** (or get Colab Pro for one 24-hour block).

## Pre-flight: push code to GitHub

```bash
# On your Mac, after committing all the local files:
git init
git add .
git commit -m "Adapter-CLIP for RSICD cross-modal retrieval"
gh repo create rsicd-clip-adapter --public --source=. --remote=origin
git push -u origin main
```

> **One-time edit before push:** replace `YOUR_USERNAME` with your real GitHub handle in `README.md`, `paper/main.tex`, and `paper/cover_letter.tex` (3 files, search-and-replace).
>
> Also update `main.tex` author block (`Vatsal Vaghasiya`, `Supervisor Name`, `M.S. Ramaiah University`), and the cover letter sender details.

## Phase 0 — Session 1, first 10 minutes: setup

Open https://colab.research.google.com → New notebook → **Runtime → Change runtime type → T4 GPU**.

```python
# Cell 0: Mount Drive + install
from google.colab import drive
drive.mount('/content/drive')

!pip install open_clip_torch faiss-cpu ftfy accelerate pyyaml -q

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
```

```python
# Cell 1: Clone repo
!git clone https://github.com/Vatsal057/rsicd-clip-adapter.git
%cd rsicd-clip-adapter

# Make your data directory
!mkdir -p data/raw
```

**Upload the dataset** to Colab. Two options:

| Option | Steps |
|---|---|
| **A. ZIP upload (smallest hassle)** | Zip `data/raw/RSICD_images/` + `data/raw/dataset_rsicd.json` on your Mac, drag-drop the zip into the Colab Files panel, then: `!unzip -q /content/data_raw.zip -d data/` |
| **B. Drive upload** | Upload zip to Drive, then: `!cp /content/drive/MyDrive/data_raw.zip . && !unzip -q data_raw.zip -d data/` |

You should now have:
```
data/raw/RSICD_images/00000.jpg ... 10920.jpg
data/raw/dataset_rsicd.json
```

```python
# Cell 2: Smoke test (30 sec)
import sys; sys.path.insert(0, '.')
import torch
from src.utils import get_device
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU'}")
print(f"Device: {get_device()}")
```

Expected: `GPU: Tesla T4` and `Device: cuda`.

```python
# Cell 3: Create splits (one-time, 5 sec)
!python scripts/01_prepare_splits.py
```

Expected: `train: 8734, val: 1094, test: 1093`.

```python
# Cell 4: Confirm baseline
!python scripts/02_run_baseline.py
```

Expected: T→I R@1 ≈ 5.95 (same as your Mac, ±0.3 noise). If it differs wildly, something's wrong with the data — stop and debug.

## Phase 1 — Session 1, next 2 hours: main adapter run

```python
# Cell 5: Main adapter training
!python scripts/03_train_adapter.py configs/adapter_base.yaml adapter 2>&1 | tee /content/drive/MyDrive/adapter_train.log
```

Expected runtime: **~2 hours on T4**. The script prints loss every 50 steps and runs validation every epoch. After epoch 1 you should already see val R@1 climb above 5.95. If you see NaN loss, stop and check for a bug.

**Backup to Drive every epoch** (the script saves `adapter_best.pt` on each val improvement):

```python
# Cell 6: Backup checkpoint + metrics to Drive
import shutil, os
DRIVE = "/content/drive/MyDrive/rsicd_runs/main_adapter"
os.makedirs(DRIVE, exist_ok=True)
for p in ["results/checkpoints/adapter_best.pt",
          "results/metrics/adapter_results.json",
          "results/metrics/training_history_adapter.json"]:
    if os.path.exists(p):
        shutil.copy(p, DRIVE)
        print(f"  -> {DRIVE}/{os.path.basename(p)}")
```

## Phase 2 — Session 1, last 5 hours: full fine-tune

```python
# Cell 7: Full fine-tune baseline
!python scripts/04_run_fullfinetune.py 2>&1 | tee /content/drive/MyDrive/fullfinetune_train.log
```

Expected runtime: **~5 hours on T4**. Larger memory footprint than the adapter run. If you OOM at batch=32, edit `configs/fullfinetune.yaml` and drop to `batch_size: 16` (rerun).

```python
# Cell 8: Backup
import shutil, os
DRIVE = "/content/drive/MyDrive/rsicd_runs/fullfinetune"
os.makedirs(DRIVE, exist_ok=True)
for p in ["results/checkpoints/fullfinetune_best.pt",
          "results/metrics/fullfinetune_results.json"]:
    if os.path.exists(p):
        shutil.copy(p, DRIVE)
```

## ⚠️ End of free-tier session

If you hit the 12-hour limit, **all variables in RAM are lost but anything in Drive is safe**. Just open a new notebook and:

```python
from google.colab import drive
drive.mount('/content/drive')
!git clone https://github.com/Vatsal057/rsicd-clip-adapter.git
%cd rsicd-clip-adapter
!pip install open_clip_torch faiss-cpu ftfy accelerate pyyaml -q
import os; os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Restore checkpoints
!cp -r /content/drive/MyDrive/rsicd_runs results/
```

Then continue from wherever you were.

## Phase 3 — Session 2 (~20 hours): ablations

The ablation script takes flags: `python scripts/05_ablations.py [placement] [hidden_dim] [data_size] [residual]`.

```python
# Cell 9: All 4 ablations sequentially (~20 hours)
!python scripts/05_ablations.py 2>&1 | tee /content/drive/MyDrive/ablations.log
```

| Ablation | Runs | Est. time | Output JSON |
|---|---|---|---|
| **placement** | image-only, text-only, both | 3 × 2h = 6h | `ablations/ablation_placement.json` |
| **hidden_dim** | 64, 128, 256, 512 | 4 × 2h = 8h | `ablations/ablation_hidden_dim.json` |
| **data_size** | 25%, 50%, 75% (100% is main) | 3 × 1h = 3h | `ablations/ablation_data_size.json` |
| **residual** | on (main), off | 1 × 2h = 2h | `ablations/ablation_residual.json` |

The script writes each ablation's JSON as it finishes, so a mid-run disconnect loses at most one ablation.

## Phase 4 — Session 3, last 30 minutes: figures + paper

```python
# Cell 10: Regenerate all 4 paper figures
!python scripts/06_qualitative.py
```

This produces 4 PDFs in `paper/figures/`. The script reads the adapter checkpoint + the ablation JSONs, so all Phase 1-3 outputs must be in `results/`.

```python
# Cell 11: Final backup of everything
import shutil
shutil.copytree("results", "/content/drive/MyDrive/rsicd_final_results", dirs_exist_ok=True)
shutil.copytree("paper/figures", "/content/drive/MyDrive/rsicd_final_figures", dirs_exist_ok=True)
shutil.copy("paper/main.tex", "/content/drive/MyDrive/main.tex")
```

## Filling in the paper

After all runs are done, on your Mac (or in a Colab cell), run this to substitute numbers into `main.tex`:

```python
import json, re, pathlib

def fmt(x): return f"{x:.2f}"

results = {}
for name, path in [
    ("zs",  "results/metrics/baseline_zeroshot.json"),
    ("ad",  "results/metrics/adapter_results.json"),
    ("ff",  "results/metrics/fullfinetune_results.json"),
]:
    with open(path) as f: results[name] = json.load(f)

z_r1  = results["zs"]["text_to_image"]["R@1"]
a_r1  = results["ad"]["text_to_image"]["R@1"]
f_r1  = results["ff"]["text_to_image"]["R@1"]
delta = round(a_r1 - z_r1, 1)
pct   = round(100 * a_r1 / f_r1, 1)

# Build a substitution map
subs = {
    "ZERO_R1": fmt(z_r1),
    "ADAPTER_R1": fmt(a_r1),
    "FULL_R1": fmt(f_r1),
    "DELTA": str(delta),
    "PCT_FULL": str(pct),
    # ...etc
}
```

The cleanest approach: turn every `[TBD]` in `main.tex` into a unique token like `MAIN_ADAPTER_T2I_R1`, and replace them in one pass. (Day 14's grep checklist covers this.)

## Compile the PDF

You have two options:
1. **Overleaf** — recommended. New project → paste `main.tex` + `refs.bib` → upload `paper/figures/*.pdf` → compile.
2. **Install TeX Live locally** — `brew install --cask mactex-no-gui` (4 GB, takes 30 min). Then `cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main`.

## What you'll have at the end

| File | Size | Source |
|---|---|---|
| `results/metrics/baseline_zeroshot.json` | 0.4 KB | rerun (Phase 0) |
| `results/metrics/adapter_results.json` | 0.4 KB | Phase 1 |
| `results/metrics/training_history_adapter.json` | ~10 KB | Phase 1 (per-epoch loss + val R@1) |
| `results/metrics/fullfinetune_results.json` | 0.4 KB | Phase 2 |
| `results/metrics/ablations/*.json` (4 files) | ~3 KB each | Phase 3 |
| `results/checkpoints/adapter_best.pt` | ~2 MB | Phase 1 |
| `results/checkpoints/fullfinetune_best.pt` | ~600 MB | Phase 2 — too big for GitHub, upload to Drive |
| `paper/figures/fig_training_curve.pdf` | ~30 KB | Phase 4 |
| `paper/figures/fig_ablation_dim.pdf` | ~30 KB | Phase 4 |
| `paper/figures/fig_qualitative.pdf` | ~200 KB | Phase 4 |
| `paper/figures/fig_failures.pdf` | ~100 KB | Phase 4 |
| `paper/main.pdf` (compiled) | ~1 MB | Overleaf or local TeX |

## Total time budget

| Phase | What | Time |
|---|---|---|
| 0 | Setup, smoke test, baseline rerun | 10 min |
| 1 | Adapter training | 2 h |
| 2 | Full fine-tune | 5 h |
| 3 | 4 ablations | 20 h |
| 4 | Figures + paper | 30 min |
| **Total** | | **~28 hours** |

Free Colab = 3 sessions of 8-10 hours each. Colab Pro ($10/mo) = 1 session of 24+ hours.

## What to add AFTER the runs

1. **Run `fill_paper.py`** (or do the search-and-replace by hand) to replace `[TBD]` with real numbers
2. **Push the new metric JSONs and updated `main.tex` to GitHub**
3. **Compile to PDF on Overleaf** — fix any warnings
4. **Run the final greps in `days/14-...md`** to catch leftover placeholders
5. **Update `results/checkpoints/adapter_best.pt`** in the repo (2 MB, fits without LFS)
6. **Add a "Reproduction notes" section to README** with anything you learned (OOM at batch=32, T4 ~2h for 20 epochs, etc.)
7. **Submit via the journal portal**

## Common pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| NaN loss | lr too high, or adapter init broken | Check `learn_logit_scale=True` is set; rerun with `lr=5e-5` |
| OOM at batch=64 (full FT) | T4 has 16 GB | Drop to `batch_size: 16` in `configs/fullfinetune.yaml` |
| Baseline R@1 = 0.0 | Data path wrong or images broken | Check `data/raw/RSICD_images/` has 10,921 jpgs; rerun `01_prepare_splits.py` |
| `ModuleNotFoundError: src` | Colab session lost; need to `%cd` and re-clone | Re-run Cell 1 |
| Adapter R@1 ≈ 5 (no improvement) | Training collapsed; check gradient flow | Add `print(model.get_trainable_params())` and confirm it's > 0 |
