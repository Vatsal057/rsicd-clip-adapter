# Kaggle Notebooks Runbook — RSICD Adapter-CLIP

> **Why Kaggle:** 30 free GPU-hours/week (P100), 12 h session limit, **no idle disconnect**.
> Total compute: ~28 GPU-hours ≈ fits in 1 week of free quota, no babysitting.
>
> **Plan: 3 sessions, 3 separate Kaggle Notebooks.** Each session trains a slice of epochs,
> saves its checkpoint as a "Kaggle Dataset", and the next session resumes from it.

---

## Pre-flight (one-time, on your Mac)

### 1. Package the dataset

```bash
cd /Users/vatsal/Desktop/Satellite
zip -r rsicd_data.zip data/raw/RSICD_images data/raw/dataset_rsicd.json
# This produces a ~430 MB zip
```

### 2. Upload as a Kaggle Dataset

1. Go to https://www.kaggle.com/datasets/new
2. **Title:** `rsicd-images`
3. **File:** drag-drop `rsicd_data.zip`
4. **Visibility:** Private (you can make it public later)
5. Click **Create** — Kaggle takes ~5 min to process the 430 MB

### 3. Create the 3 notebooks

For each, go to https://www.kaggle.com/code → **+ New Notebook**:

| Notebook | Title | Visibility | Used for |
|---|---|---|---|
| #1 | `rsicd-adapter-session1` | Private | Main adapter run, epochs 1-7 |
| #2 | `rsicd-adapter-session2` | Private | Continue adapter, epochs 8-14 |
| #3 | `rsicd-finalize` | Private | Continue adapter, epochs 15-20 + ablations |

(You can rename later. Each one will need the previous one's output as a Dataset — see below.)

---

## Notebook #1 — Main adapter training (epochs 1-7)

### Settings

- **Accelerator:** GPU T4 × 2 OR GPU P100 (both work; P100 has more RAM)
- **Internet:** ON (need `pip install` and `git clone`)
- **Persistence:** Files only (default)

### Cells

**Cell 1 — install + clone + attach data**
```python
!pip install open_clip_torch faiss-cpu ftfy accelerate pyyaml -q
!git clone https://github.com/Vatsal057/rsicd-clip-adapter.git
%cd rsicd-clip-adapter

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Attach the dataset (click "Add data" -> search "rsicd-images" -> Add)
# It mounts at /kaggle/input/rsicd-images/rsicd_data.zip
!unzip -q /kaggle/input/rsicd-images/rsicd_data.zip
# After unzip: /kaggle/working/rsicd-clip-adapter/data/raw/...
!ls data/raw/RSICD_images/ | wc -l   # should print 10921
```

**Cell 2 — prepare splits + verify baseline**
```python
!python scripts/01_prepare_splits.py
!python scripts/02_run_baseline.py
```
Expected: T→I R@1 ≈ 5.95. If not, stop and debug.

**Cell 3 — train (epochs 1-7 only)**
```python
import yaml
with open("configs/adapter_base.yaml") as f:
    cfg = yaml.safe_load(f)
cfg["training"]["num_epochs"] = 7   # cap at 7 for session 1
with open("configs/adapter_session1.yaml", "w") as f:
    yaml.dump(cfg, f)

!python scripts/03_train_adapter.py configs/adapter_session1.yaml adapter
```
Expected: ~6-7 hours wall clock on P100. But Kaggle's 12 h limit is fine. Actually the adapter at batch=64 is ~6 minutes/epoch on P100, so 7 epochs = 42 min. Done well under 12h.

**Cell 4 — save output as Kaggle Dataset (CRITICAL)**
```python
# This is the ONLY way to get the checkpoint to persist into session 2.
# After the cell runs, the output appears in the right panel.
# Click "Save Version" (top right) -> "Save Output" must be ON -> "Save".
# After save, go to the Output tab -> "New Dataset" -> name it "rsicd-adapter-s1"
# Now you have a persistent Kaggle Dataset with the checkpoint.
import shutil, os
os.makedirs("/kaggle/working/rsicd-adapter-s1", exist_ok=True)
for p in [
    "results/checkpoints/adapter_best.pt",
    "results/metrics/training_history_adapter.json",
    "results/metrics/adapter_results.json",
    "configs/adapter_session1.yaml",
]:
    if os.path.exists(p):
        shutil.copy(p, f"/kaggle/working/rsicd-adapter-s1/{os.path.basename(p)}")
print("Ready to save. Click 'Save Version' with 'Save Output' enabled.")
```

> **Why this is needed:** Kaggle's notebook output is wiped when the session ends, UNLESS you "Save Version". A saved version with output is downloadable, and you can also turn it into a permanent Dataset that future notebooks can `+ Add`.

---

## Notebook #2 — Continue adapter (epochs 8-14)

### Settings
- Same as #1, plus: click **+ Add data** → search `rsicd-adapter-s1` → Add (the dataset you created in step 4 above)

### Cells

**Cell 1 — same as #1**
```python
!pip install open_clip_torch faiss-cpu ftfy accelerate pyyaml -q
!git clone https://github.com/Vatsal057/rsicd-clip-adapter.git
%cd rsicd-clip-adapter
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
!unzip -q /kaggle/input/rsicd-images/rsicd_data.zip
!ls data/raw/RSICD_images/ | wc -l
```

**Cell 2 — restore checkpoint from previous session**
```python
import shutil, os
# Copy the previous session's checkpoint into the new results dir
!mkdir -p results/checkpoints
shutil.copy("/kaggle/input/rsicd-adapter-s1/adapter_best.pt",
            "results/checkpoints/adapter_best.pt")
print("Checkpoint restored.")
```

**Cell 3 — resume training from epoch 8 to 14**
```python
import yaml
with open("configs/adapter_base.yaml") as f:
    cfg = yaml.safe_load(f)
cfg["training"]["num_epochs"]      = 14   # run epochs 8-14
cfg["training"]["resume_from"]     = "results/checkpoints/adapter_best.pt"
with open("configs/adapter_session2.yaml", "w") as f:
    yaml.dump(cfg, f)

!python scripts/03_train_adapter.py configs/adapter_session2.yaml adapter
```

The training function will:
- Load `adapter_best.pt` (which is from epoch 7)
- Print `resumed at epoch 7, best val R@1 so far: XX.XX%`
- Start training epochs 8-14
- Save the new best checkpoint at the **same path**, overwriting

**Cell 4 — save again as Dataset `rsicd-adapter-s2`**
```python
import shutil, os
os.makedirs("/kaggle/working/rsicd-adapter-s2", exist_ok=True)
for p in [
    "results/checkpoints/adapter_best.pt",
    "results/metrics/training_history_adapter.json",
    "results/metrics/adapter_results.json",
    "configs/adapter_session2.yaml",
]:
    if os.path.exists(p):
        shutil.copy(p, f"/kaggle/working/rsicd-adapter-s2/{os.path.basename(p)}")
print("Ready to save as rsicd-adapter-s2 dataset.")
```

---

## Notebook #3 — Finish adapter (epochs 15-20) + full fine-tune + ablations + figures

### Settings
- Same as #1, plus: **+ Add data** → `rsicd-images` AND `rsicd-adapter-s2`

### Cells

**Cell 1 — same as session 1**
**Cell 2 — restore checkpoint**
**Cell 3 — finish adapter training (epochs 15-20)**
```python
import yaml
with open("configs/adapter_base.yaml") as f:
    cfg = yaml.safe_load(f)
cfg["training"]["num_epochs"]  = 20
cfg["training"]["resume_from"] = "results/checkpoints/adapter_best.pt"
!python scripts/03_train_adapter.py configs/adapter_base.yaml adapter
```

**Cell 4 — full fine-tune baseline (~5 h on P100)**
```python
!python scripts/04_run_fullfinetune.py
```

**Cell 5 — ablations (~20 h on P100 — but you can split this into 2 more sessions if needed)**
```python
!python scripts/05_ablations.py
```

> **Time check:** adapter (5h) + full FT (5h) + ablations (20h) = 30 h. This is over the 12 h Kaggle limit. So you'll need to split this further into 3-4 more sessions, each ending with a `Save Version` and a new Dataset.
>
> Recommended split:
> - 3a: adapter finish + full fine-tune (~10 h)
> - 3b: ablations placement + hidden_dim (~12 h)
> - 3c: ablations data + residual + figures (~10 h)

**Cell 6 — regenerate 4 paper figures**
```python
!python scripts/06_qualitative.py
```

**Cell 7 — package all results**
```python
import shutil, os
os.makedirs("/kaggle/working/rsicd-final", exist_ok=True)
# Copy everything
for src in ["results", "paper/figures", "paper/main.tex", "paper/refs.bib"]:
    if os.path.exists(src):
        dst = f"/kaggle/working/rsicd-final/{os.path.basename(src)}"
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy(src, dst)
print("All results packaged. Save this version.")
```

---

## After the runs — on your Mac

```bash
cd /Users/vatsal/Desktop/Satellite
git pull   # get the new metric JSONs you'll be pushing

# Download the Kaggle output
# In Kaggle UI: Notebook output -> "Download All"
# This gives you a zip with results/ and paper/figures/

# Or use the Kaggle CLI (after `pip install kaggle` + API token):
kaggle datasets download -d YOUR_USERNAME/rsicd-final
unzip rsicd-final.zip -d /tmp/rsicd-final

# Then merge into the repo
cp -r /tmp/rsicd-final/results/* results/
cp -r /tmp/rsicd-final/paper/figures/* paper/figures/
git add results/ paper/figures/
git commit -m "Fill in [TBD] placeholders with real numbers from Kaggle runs"
git push
```

## Filling in `[TBD]` in main.tex

```python
import json, re, pathlib

def load(p): return json.load(open(p)) if pathlib.Path(p).exists() else None

# Load all results
zs  = load("results/metrics/baseline_zeroshot.json")
ad  = load("results/metrics/adapter_results.json")
ff  = load("results/metrics/fullfinetune_results.json")
his = load("results/metrics/training_history_adapter.json")

if ad and zs:
    z_r1  = zs["text_to_image"]["R@1"]
    a_r1  = ad["text_to_image"]["R@1"]
    f_r1  = ff["text_to_image"]["R@1"] if ff else 0
    delta = round(a_r1 - z_r1, 1)
    pct   = round(100 * a_r1 / f_r1, 1) if f_r1 else 0

    # Read main.tex
    tex = pathlib.Path("paper/main.tex").read_text()

    # Specific substitutions
    subs = {
        # Abstract
        r"\\textbf\{\\textit\{[TBD]\}-percentage-point\}": f"\\textbf{{{delta:.1f}-percentage-point}}",
        r"\\textbf\{\\textit\{[TBD]\%\}\} of full":         f"\\textbf{{{pct:.1f}\\%}} of full",
        # Main results table — last row
        # ... etc
    }
    for pat, rep in subs.items():
        tex = re.sub(pat, rep, tex)
    pathlib.Path("paper/main.tex").write_text(tex)
    print("Updated main.tex")
```

The cleanest approach: open `main.tex` and use Find-Replace to swap each `[TBD]` with the actual number from `results/metrics/*.json`. The `\textbf{\textit{[TBD]}}` formatting makes them visually obvious in the compiled PDF.

## What you'll have at the end

| File | Source |
|---|---|
| `results/metrics/baseline_zeroshot.json` | Already had this |
| `results/metrics/adapter_results.json` | Session 3 (last adapter epoch) |
| `results/metrics/training_history_adapter.json` | Sessions 1-3 merged (continuous) |
| `results/metrics/fullfinetune_results.json` | Session 3 |
| `results/metrics/ablations/*.json` (4 files) | Session 3 |
| `results/checkpoints/adapter_best.pt` (2 MB) | Session 3 final |
| `results/checkpoints/fullfinetune_best.pt` (600 MB) | Session 3 — **do not commit**, link via Drive |
| `paper/figures/fig_*.pdf` (4 files) | Session 3, regenerated with real data |

Then `git push` everything to GitHub and compile `main.tex` to PDF (Overleaf recommended).

## Total time budget

| Session | What | Time on P100 |
|---|---|---|
| 1 | Adapter epochs 1-7 | ~45 min |
| 2 | Adapter epochs 8-14 | ~45 min |
| 3a | Adapter epochs 15-20 + full FT | ~10 h |
| 3b | Ablations placement + hidden_dim | ~12 h |
| 3c | Ablations data + residual + figures | ~10 h |
| **Total** | | **~33 h** |

Kaggle free: 30 h/week. So this is **exactly at the limit** — and you have 1 buffer hour per week for retries.

## Common pitfalls (Kaggle-specific)

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: src` | Add `%cd rsicd-clip-adapter` at the top |
| `No such file: data/raw/...` | Forgot to `!unzip` in Cell 1 |
| `CUDA OOM` on P100 | P100 has 16 GB; drop `batch_size: 32` for full FT (edit config first) |
| `Internet access disabled` | Settings → Internet → ON (right panel of notebook editor) |
| Session disconnects mid-training | Should be impossible on Kaggle. If it happens, the `!python` cell will be re-runnable from the saved checkpoint. |
| Output disappears after save | You forgot to enable "Save Output" when saving the version. Re-save. |
| `Dataset not found` in session 2 | Forgot to make `rsicd-adapter-s1` a Dataset (it's a saved version by default — convert it via the Output tab → "New Dataset") |
