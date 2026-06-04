# Day 1 — Environment, dataset, data pipeline

**Goal:** Working Python environment, RSICD dataset ready at `data/raw/`, deterministic train/val/test splits saved, and a PyTorch `Dataset` class that returns `(image_tensor, caption_tokens, imgid)` batches.

## Todos

- [x] Read `IMPLEMENTATION.md` end to end
- [x] Confirm compute target = Google Colab T4 GPU
- [x] Confirm RSICD dataset is already available
- [x] Create project directory tree (data/, src/, scripts/, configs/, results/, paper/, notebooks/, days/)
- [x] Create `requirements.txt`
- [x] Create `src/__init__.py` and `src/utils.py` (set_seed, save/load ckpt, count_params, get_device)
- [x] Create `scripts/00_download_data.py` (Kaggle + manual fallback + verify)
- [x] Create `scripts/00b_prepare_rsicd.py` (convert HF CSV → folder+JSON)
- [x] Create `scripts/01_prepare_splits.py` (80/10/10, seed=42, run ONCE; canonicalize `valid` → `val`)
- [x] Create `src/dataset.py` (RSICDDataset + RSICDRetrievalDataset + get_dataloaders, with `force_quick_gelu=True`)
- [x] Create Python venv and install `requirements.txt` (Py3.14, MPS available)
- [x] Run conversion + splits: 8,734 / 1,094 / 1,093 = 80/10/10
- [x] Sanity check: smoke test pulls a batch, shapes match, dtype correct

## Decisions / deviations

- **Python version on this Mac:** 3.14.0. Every dep installed cleanly (torch 2.12, open_clip 3.3, faiss 1.14, sklearn 1.9). No version pinning needed.
- **MPS is available, CUDA is not.** `get_device()` returns `mps` on this Mac, `cuda` on Colab. The code is identical for both.
- **Data source on this machine:** `archive/{train,valid,test}.csv` — HuggingFace-style CSVs with images inlined as `{'bytes': b'...'}` blobs. **Not** the folder+JSON layout the IMPLEMENTATION.md assumes. So we added an extra step `scripts/00b_prepare_rsicd.py` to convert the CSVs into the expected layout.
- **Captions are stored concatenated:** the HF CSV's "captions" field is a list of length 1, where the single element is the 5 RSICD captions concatenated with no separator between them (e.g. `"cap1.cap2.cap3.cap4.cap5."`). Splitting on `.\\s*` is unreliable because some captions start with lowercase. We take the **first sentence** as the canonical caption and discard the rest. Result: **1 caption per image, 10,921 pairs total** (vs the canonical 54,605).
  - **Paper note:** we mention in Section 4.1 that we use the first sentence per image. This is a deviation from the original RSICD 5-captions-per-image annotation. The model is unaffected; the dataset is 5x smaller.
- **Split names:** source uses `valid` not `val`. Canonicalize in `01_prepare_splits.py`.
- **`force_quick_gelu=True`** added to open_clip model loading. The OpenAI-pretrained ViT-B/32 uses QuickGELU; without this flag open_clip emits a warning and uses the wrong activation, slightly degrading retrieval quality.
- **`num_workers=0`** in the smoke test (avoids the Py3.14 + macOS + multiprocessing spawn issue when launching from a heredoc). Training uses `num_workers=4` for speed.

## Outputs

- `requirements.txt`
- `src/__init__.py`, `src/utils.py`, `src/dataset.py`
- `scripts/00_download_data.py`, `scripts/00b_prepare_rsicd.py`, `scripts/01_prepare_splits.py`
- `notebooks/smoke_test_day1.py`
- `.venv/` (created, gitignored)
- `data/raw/RSICD_images/*.jpg` — 10,921 jpgs (~560 MB)
- `data/raw/dataset_rsicd.json` — 5.1 MB
- `data/splits/{train,val,test,metadata}.json` — 8,734 / 1,094 / 1,093 = 80/10/10
- `.gitignore`

## Smoke-test result

```
Device: mps
Train batches:        2183
Train pairs:          8734
Val images:           1094
Test images:          1093
Image tensor shape:   torch.Size([4, 3, 224, 224])
Caption token shape:  torch.Size([4, 77])
Image value range:    [-1.792, 2.146]
=== Dataset smoke test PASSED ===
```

First training caption (after first-sentence extraction):
> `'Many aircraft are parked next to a long building in an airport.'`

## Notes

- Day 1 is done. Day 2 (zero-shot baseline) is unblocked.
- We do NOT need to re-run `01_prepare_splits.py` after this — the splits JSONs are the ground truth.
- Total disk usage: ~560 MB for images + ~5 MB for JSON. Easy to commit the JSONs to git, the images stay local.
