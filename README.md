# Adapter-CLIP for Remote Sensing Cross-Modal Retrieval

> **Paper:** *Bridging the Domain Gap: Lightweight Adapter-Based CLIP Fine-Tuning for Cross-Modal Retrieval in Remote Sensing*
> **Target journal:** *Neurocomputing* — Special Issue on Multimodal Representation Learning Based on Vision Foundation Models

## Overview

Lightweight bottleneck adapters trained on frozen CLIP achieve competitive
cross-modal retrieval on the **RSICD** dataset with only **527,873** trainable
parameters (0.35% of CLIP total).

| Method               | Trainable Params | T→I R@1 | T→I R@5 | T→I R@10 | I→T R@1 | I→T R@5 | I→T R@10 |
|----------------------|------------------|---------|---------|----------|---------|---------|----------|
| Zero-shot CLIP       | 0                | _5.95_  | _18.21_ | _28.64_  | _4.39_  | _16.19_ | _25.07_  |
| Full fine-tune CLIP  | 150,000,000      | _TBD_   | _TBD_   | _TBD_    | _TBD_   | _TBD_   | _TBD_    |
| **Adapter-CLIP (ours)** | **527,873**   | _TBD_   | _TBD_   | _TBD_    | _TBD_   | _TBD_   | _TBD_    |

Numbers in _italics_ are the actual values from `results/metrics/*.json` (zero-shot row is filled in; the rest are filled in after the full Colab run, see `paper/main.tex` for the latest values).

## Setup

```bash
git clone https://github.com/Vatsal057/rsicd-clip-adapter
cd rsicd-clip-adapter
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you don't have a CUDA GPU, the same code runs on Apple Silicon (MPS) or CPU. `get_device()` picks the best available automatically.

## Dataset

The expected layout is:

```
data/raw/
├── RSICD_images/<NNNNN>.jpg     # 10,921 images
└── dataset_rsicd.json            # one entry per image, with "split" field
```

If you have the **HuggingFace redistribution** (CSV with inlined bytes, like `archive/{train,valid,test}.csv`):
```bash
python scripts/00b_prepare_rsicd.py    # converts to the layout above
```

Otherwise (Kaggle download or manual):
```bash
python scripts/00_download_data.py    # Kaggle CLI + manual fallback
```

Then create the fixed splits (run **once**):
```bash
python scripts/01_prepare_splits.py
```

## Reproduce all experiments

```bash
bash reproduce.sh
```

This runs, in order:
1. `01_prepare_splits.py` — fixed train/val/test splits
2. `02_run_baseline.py` — zero-shot CLIP evaluation
3. `03_train_adapter.py configs/adapter_base.yaml adapter` — adapter training
4. `04_run_fullfinetune.py` — full fine-tune baseline
5. `05_ablations.py` — 4 ablations
6. `06_qualitative.py` — paper figures (PDFs to `paper/figures/`)

End-to-end runtime:
- On **Google Colab T4**: ~30-40 GPU-hours (training is the bottleneck)
- On **Apple Silicon (MPS)**: ~6-8 hours for the adapter training; full fine-tune is impractical (>50 hours)

For a quick smoke test (1 epoch, hidden_dim=64, batch=8, ~2 minutes on MPS):
```bash
python scripts/03_train_adapter.py configs/smoke_adapter.yaml smoke
```

## Project structure

```
src/           — model, dataset, loss, training, evaluation, utilities
scripts/       — 00..06 numbered runnable scripts
configs/       — YAML experiment configurations
results/       — metrics JSONs (committed), checkpoints, figures
paper/         — LaTeX source + paper figures
notebooks/     — exploration + Colab quickstart + smoke tests
days/          — per-day progress logs
data/          — raw images (gitignored) + splits JSONs (committed)
```

## Citation

```bibtex
@article{vaghasiya2025adapterclip,
  title   = {Bridging the Domain Gap: Lightweight Adapter-Based CLIP
             Fine-Tuning for Cross-Modal Retrieval in Remote Sensing},
  author  = {Vaghasiya, Vatsal and Supervisor Name},
  journal = {Neurocomputing},
  year    = {2025}
}
```

## License

MIT (see `LICENSE`).
