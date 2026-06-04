# Kaggle Notebooks Runbook — RSICD Adapter-CLIP

> **Why Kaggle:** 30 free GPU-hours/week (P100), 12 h session limit, **no idle disconnect**.
> Total compute: ~28 GPU-hours ≈ fits in 1 week of free quota, no babysitting.

The 5 notebooks in `notebooks/` are pre-generated. You upload them to Kaggle, attach the right
Datasets, and click "Run All".

---

## 1. The 5 sessions

| # | Notebook | What it does | Run time (P100) | Attach these Datasets | Save output as |
|---|---|---|---|---|---|
| **1** | `kaggle_s1_adapter_start.ipynb` | Convert CSV → jpg, prepare splits, baseline, train adapter epochs 1-7 | ~50 min | `rsicd-image-caption-dataset` | `rsicd-adapter-s1` |
| **2** | `kaggle_s2_adapter_mid.ipynb` | Resume adapter, epochs 8-14 | ~50 min | `rsicd-adapter-s1` | `rsicd-adapter-s2` |
| **3** | `kaggle_s3_adapter_end.ipynb` | Resume adapter, epochs 15-20 | ~50 min | `rsicd-adapter-s2` | `rsicd-adapter-s3` |
| **4** | `kaggle_s4_fullfinetune.ipynb` | Full fine-tune baseline | ~5 h | `rsicd-adapter-s3` | `rsicd-fullfinetune` |
| **5** | `kaggle_s5_ablations_figures.ipynb` | 4 ablations + regenerate 4 figures | ~20 h | `rsicd-adapter-s3`, `rsicd-fullfinetune` | `rsicd-final` |
| | | **Total** | **~28 h** | | |

The adapter training is split into 3 sessions because each epoch is ~7 minutes on P100 and Kaggle
sessions are 12 h, so we have plenty of headroom. If you want to fit more into one session, just
change `num_epochs` in the train cell.

---

## 2. Walkthrough: session 1

1. Go to https://www.kaggle.com/code → **+ New Notebook** → **File → Import Notebook**
2. Upload `notebooks/kaggle_s1_adapter_start.ipynb`
3. In the right panel → **+ Add data** → search `thedevastator/rsicd-image-caption-dataset` → **Add**
4. Settings (top right): **Accelerator = GPU P100 or T4**, **Internet = ON**
5. Click **Run All** (or step through)
6. When the last cell finishes:
   - **Save Version** (top right) → make sure **Save Output** is **ON** → **Save**
   - Wait ~30 sec for the version to commit
   - Output tab at the bottom → **+ New Dataset** → name it `rsicd-adapter-s1` → **Create**
7. Open `kaggle_s2_adapter_mid.ipynb` for session 2.

The conversion in cell 3 takes ~5 minutes the first time, ~30 sec on re-runs (it's idempotent).
Training takes ~45 min for 7 epochs. Total session 1: ~50 min.

---

## 3. Walkthrough: sessions 2-5

Same as session 1, but:

- **Different notebook** (`kaggle_s2_*.ipynb`, `kaggle_s3_*.ipynb`, etc.)
- **Different attached Datasets** (see the table above)
- **Each session's last cell packages the output** for the next session to consume

### Important: do NOT clear the data between sessions

Each Kaggle session starts with empty `/kaggle/working/`. The notebook re-runs the data
conversion in cell 3 every time, but the conversion script is idempotent (it skips images
that are already on disk), so re-conversion takes ~30 sec instead of 5 min.

---

## 4. After all 5 sessions

Download the final `rsicd-final` Dataset from Kaggle and merge into your repo:

```bash
# In the Kaggle UI: Notebook -> "rsicd-final" output -> "Download All"
# This gives you a zip with results/, paper/figures/, paper/main.tex, etc.

# Or use the Kaggle CLI:
pip install kaggle  # if not already
kaggle datasets download -d YOUR_USERNAME/rsicd-final
unzip rsicd-final.zip -d /tmp/rsicd-final

# Merge into the repo
cd /Users/vatsal/Desktop/Satellite
cp -r /tmp/rsicd-final/results/*     results/
cp -r /tmp/rsicd-final/paper/figures/* paper/figures/
git add results/ paper/figures/
git commit -m "Fill in [TBD] placeholders with real numbers from Kaggle runs"
git push
```

## 5. Filling in `[TBD]` in main.tex

The placeholders are uniformly formatted as `\textit{[TBD]}` or `\textbf{\textit{[TBD]}}`. After all
runs are done, the cleanest path is a small Python script that reads each `results/metrics/*.json`
and does a single-pass regex substitution. The `days/14-...md` file has the grep checklist for the
final sanity pass.

Alternatively, open `paper/main.tex` in any editor and use Find-Replace to swap each `[TBD]` for
the value in the corresponding `results/metrics/*.json` field.

## 6. Compile the PDF

Easiest: Overleaf. New project → paste `main.tex` + `refs.bib` → upload `paper/figures/*.pdf` →
compile.

Local: `brew install --cask mactex-no-gui` (4 GB, 30 min) → `cd paper && pdflatex main && bibtex
main && pdflatex main && pdflatex main`.

---

## Total time budget

| Phase | What | Time on P100 |
|---|---|---|
| 1 | Adapter epochs 1-7 | ~50 min |
| 2 | Adapter epochs 8-14 | ~50 min |
| 3 | Adapter epochs 15-20 | ~50 min |
| 4 | Full fine-tune | ~5 h |
| 5 | 4 ablations + figures | ~20 h |
| **Total** | | **~28 h** |

Kaggle free: 30 GPU-hours/week. **Fits in 1 week**, 2-hour buffer for retries.

---

## Regenerating the notebooks

If you change the source code (e.g. add a new script, change a config default), regenerate them
with:

```bash
python scripts/_generate_kaggle_notebooks.py
```

This is idempotent — it always writes the 5 notebooks fresh from the latest code. Commit the
regenerated notebooks to GitHub after any code change.

---

## Common pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: src` | Forgot `%cd rsicd-clip-adapter` | The setup cell does this; if you run cells out of order, re-run cell 1 first |
| `FileNotFoundError: data/raw/...` | Skipped the conversion cell | Run cell 3 of session 1 |
| `WARN: /kaggle/input/... not found` | Forgot to "+ Add data" the previous session's Dataset | Add it; re-run the restore cell |
| `CUDA OOM` on P100 | P100 has 16 GB. Full FT needs careful batch | Edit `configs/fullfinetune.yaml`: `batch_size: 16` instead of 32 |
| `Internet access disabled` | Settings → Internet → OFF | Settings → Internet → ON (right panel) |
| `resume_from checkpoint does not exist` | Wrong Dataset attached, or saved with a different name | Check the table above for the right Dataset name |
| Output disappears after Save Version | Forgot to enable "Save Output" | Re-save with the toggle on |
| `WARN: dataset path /kaggle/input/... not found` (session 1) | The original RSICD CSV not attached | Click "+ Add data" → search `rsicd-image-caption-dataset` |
| Adapter R@1 ≈ 5 (no improvement) | Training collapsed | Add `print(model.count_trainable_params())` after model construction; should be ~530K |
| Time-budget exceeded | Did ablations in one session | Split: run ablations 1-2 in one session, save, run 3-4 in next |
