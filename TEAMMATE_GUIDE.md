# Guide for My Teammate

Hey! Here's exactly what you need to do. You don't need to understand the
project — just follow these steps.

## What this project is (one sentence)

We trained a small add-on module on top of CLIP (an AI model) so it can
better search satellite images using text descriptions. That's it.

## What's already done

- I ran **Session 1** (adapter training) on Kaggle. It finished successfully.
- The output is saved as a Kaggle Dataset called **`rsicd-adapter-final`**
  at: https://www.kaggle.com/datasets/vatsalvaghasiya/rsicd-adapter-final

## What you need to do

You need to run **2 more notebooks** on Kaggle. Each takes about 15-80 minutes.
Both are already pre-built `.ipynb` files — you just upload and click Run.

---

### Notebook 1: Full Fine-Tune (~15 min)

**File:** `notebooks/kaggle_s4_fullfinetune.ipynb`

1. Go to https://www.kaggle.com/code → click **+ New Notebook**
2. Click **File → Import Notebook** → upload `kaggle_s4_fullfinetune.ipynb`
3. In the right panel, click **+ Add data** → search `thedevastator/rsicd-image-caption-dataset` → click **Add**
4. Click **+ Add data** again → search `rsicd-adapter-final` → click **Add**
5. Set **Accelerator** to GPU (T4 or P100)
6. Click **Run All** (or run cells one by one)
7. Wait for it to finish (~15 min)
8. Click **Save Version** (top right) → select **Save & Run All (Commit)** → click **Save**
9. After it finishes, go to the **Output** tab at the bottom → click **+ New Dataset** → name it `rsicd-fullfinetune` → **Create**

---

### Notebook 2: Ablations + Figures (~80 min)

**File:** `notebooks/kaggle_s5_ablations_figures.ipynb`

1. Go to https://www.kaggle.com/code → click **+ New Notebook**
2. Click **File → Import Notebook** → upload `kaggle_s5_ablations_figures.ipynb`
3. In the right panel, click **+ Add data** → search `thedevastator/rsicd-image-caption-dataset` → click **Add**
4. Click **+ Add data** again → search `rsicd-adapter-final` → click **Add**
5. Click **+ Add data** again → search `rsicd-fullfinetune` → click **Add** (this is the output from Notebook 1)
6. Set **Accelerator** to GPU (T4 or P100)
7. Click **Run All**
8. Wait for it to finish (~80 min — this one runs 4 ablation experiments)
9. Click **Save Version** → **Save & Run All (Commit)** → **Save**
10. After it finishes, go to **Output** tab → **+ New Dataset** → name it `rsicd-final` → **Create**

---

## That's it!

After you create `rsicd-final`, tell me. I'll download it and finish the paper.

## Important tips

- **Don't close the browser tab** while a notebook is running. Kaggle runs
  server-side, so you CAN close it, but it's safer to keep it open.
- **You'll get an email** when a notebook finishes or fails.
- **If a notebook fails**, the most common reason is: you forgot to attach a
  Dataset. Click **+ Add data** and make sure all required Datasets are there.
- **Each notebook needs these Datasets attached:**
  - Notebook 1: `thedevastator/rsicd-image-caption-dataset` + `rsicd-adapter-final`
  - Notebook 2: `thedevastator/rsicd-image-caption-dataset` + `rsicd-adapter-final` + `rsicd-fullfinetune`

## Where the files are

| File | Location |
|------|----------|
| Notebook 1 | `notebooks/kaggle_s4_fullfinetune.ipynb` |
| Notebook 2 | `notebooks/kaggle_s5_ablations_figures.ipynb` |

Both are in the `notebooks/` folder of the repo. Download them from GitHub
or I can send them to you directly.
