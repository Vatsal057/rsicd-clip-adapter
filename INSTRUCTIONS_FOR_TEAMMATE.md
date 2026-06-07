# Instructions for Running Remaining Kaggle Sessions

## What is this project?

We are training a CLIP adapter for a research paper on cross-modal retrieval in remote sensing. The first session (adapter training) is already complete. Two more sessions remain:

1. **Full fine-tune** (~15 min on GPU) — trains the full CLIP model (not just the adapter) as a comparison baseline
2. **Ablations + Figures** (~80 min on GPU) — runs 4 ablation experiments and generates all paper figures

After both sessions, download the final output, merge it into the local repo, and push to GitHub.

---

## Before you start

1. **Clone the repo** to get the notebook files:
   ```bash
   git clone https://github.com/Vatsal057/rsicd-clip-adapter.git
   cd rsicd-clip-adapter
   ```
   The notebook files are in the `notebooks/` folder.

2. **Kaggle account**: You need a Kaggle account (free). Go to https://www.kaggle.com and sign in.

3. **GPU access**: In Kaggle, go to Account → Settings → under "Accelerators", make sure "GPU" is enabled. Free accounts get 30 GPU-hours/week.

4. **GitHub access**: You need push access to the repo https://github.com/Vatsal057/rsicd-clip-adapter. Ask Vatsal if you don't have it.

5. **Local machine**: You need Python 3.10+ and Git installed. You also need `pdflatex` to compile the paper (or use Overleaf).

---

## Session 2: Full Fine-tune (~15 min)

### Step 1: Upload the notebook to Kaggle

1. Go to https://www.kaggle.com/code → click **+ New Notebook** → **File → Import Notebook**
2. Upload the file `notebooks/kaggle_s4_fullfinetune.ipynb` (from the repo)

### Step 2: Attach the datasets

You need **two** datasets:

**Dataset 1 — RSICD images (needed for data conversion):**
1. In the right panel, click **+ Add data**
2. Search for `thedevastator/rsicd-image-caption-dataset`
3. Click **Add**

**Dataset 2 — Adapter checkpoint from Session 1:**
1. Click **+ Add data** again
2. Search for `vatsalvaghasiya/rsicd-adapter-final`
3. Click **Add**

### Step 3: Set GPU

1. In the top-right of the notebook editor, look for "Settings" or the gear icon
2. Set **Accelerator** to **GPU T4** (or P100 if available)
3. Make sure **Internet** is ON

### Step 4: Run the notebook

1. Click **Run All** (or step through cells manually)
2. Wait ~15 minutes for it to finish
3. The notebook will:
   - Install dependencies and clone the repo
   - Convert the RSICD dataset (images from CSVs)
   - Run the full fine-tune training (10 epochs)
   - Package the output for the next session

### Step 5: Save the output

1. Click **Save Version** (top right)
2. In the dialog, select **"Save & Run All (Commit)"**
3. Click **Save**
4. Wait for the run to complete (~15 min)
5. When done, go to the **Output** tab at the bottom
6. Click **"+ New Dataset"** → name it `rsicd-fullfinetune` → **Create**

---

## Session 3: Ablations + Figures (~80 min)

This is the longest session. It runs 4 ablation experiments (different adapter configurations) and generates all paper figures.

### Step 1: Upload the notebook to Kaggle

1. Go to https://www.kaggle.com/code → click **+ New Notebook** → **File → Import Notebook**
2. Upload the file `notebooks/kaggle_s5_ablations_figures.ipynb`

### Step 2: Attach the datasets

You need **three** datasets:

**Dataset 1 — RSICD images (needed for data conversion):**
1. In the right panel, click **+ Add data**
2. Search for `thedevastator/rsicd-image-caption-dataset`
3. Click **Add**

**Dataset 2 — Adapter checkpoint from Session 1:**
1. Click **+ Add data** again
2. Search for `vatsalvaghasiya/rsicd-adapter-final`
3. Click **Add**

**Dataset 3 — Full fine-tune results from Session 2:**
1. Click **+ Add data** again
2. Search for `vatsalvaghasiya/rsicd-fullfinetune`
3. Click **Add**

### Step 3: Set GPU

Same as Session 2: **Accelerator = GPU T4** (or P100), **Internet = ON**.

### Step 4: Run the notebook

1. Click **Run All** or step through cells
2. Wait ~80 minutes (may take longer depending on GPU)
3. The notebook will:
   - Run 4 ablation experiments (different adapter hidden dims, placement, data sizes)
   - Generate all paper figures (training curves, ablation plots, qualitative examples)
   - Package everything for the final output

### Step 5: Save the output

Same as Session 2, but name the dataset **`rsicd-final`** (this is the final output).

---

## After both sessions: Merge and push to GitHub

### Step 1: Download the final output

1. Go to https://www.kaggle.com/datasets/vatsalvaghasiya/rsicd-final
2. Click **Download** (or use the Kaggle CLI: `kaggle datasets download -d vatsalvaghasiya/rsicd-final`)
3. Unzip the downloaded file

### Step 2: Merge into the local repo

On your local machine:

```bash
cd /Users/vatsal/Desktop/Satellite

# Copy results from the downloaded dataset
cp -r /path/to/downloaded/rsicd-final/results/* results/
cp -r /path/to/downloaded/rsicd-final/paper/figures/* paper/figures/

# Stage and commit
git add results/ paper/figures/
git commit -m "Add Kaggle results and figures from sessions 2-3"
```

### Step 3: Fill in the paper placeholders

The paper (`paper/main.tex`) has `[TBD]` placeholders. Fill them in using the results from the JSON files:

```bash
# Check what needs to be filled
grep -n "TBD" paper/main.tex
```

Then manually replace each `[TBD]` with the corresponding value from the JSON files in `results/metrics/`.

### Step 4: Push to GitHub

```bash
git push origin main
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| "GPU not available" error | Go to Kaggle Account → Settings → Enable GPU accelerator |
| "Dataset not found" | Make sure you attached the dataset via **+ Add data** |
| Training crashes with "FileNotFoundError" | The dataset conversion didn't run. Re-run cell 3 (data conversion) |
| Training crashes with "RuntimeError: missing key(s)" | This is the known bug we fixed. Make sure `git pull` ran in cell 1. If not, add `!cd /kaggle/working/rsicd-clip-adapter && git pull` before cell 5 |
| Kaggle session disconnects | Kaggle free tier sessions last 12 hours. If it disconnects, re-run from where it stopped (training resumes from checkpoint) |
| "Save Output" option not visible | In the Save Version dialog, select "Save & Run All (Commit)" — output is saved automatically |

---

## Summary of datasets

| Dataset name | Contents | Created by |
|---|---|---|
| `rsicd-adapter-final` | Adapter checkpoint (1.6 MB) + training history + test results | Session 1 (already done) |
| `rsicd-fullfinetune` | Full fine-tune checkpoint (~600 MB) + test results | Session 2 |
| `rsicd-final` | All results + all paper figures + paper tex/bib | Session 3 |

---

## Contact

If you get stuck, message Vatsal on WhatsApp or ask in the project group chat.
