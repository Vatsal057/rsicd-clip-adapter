# Day 2 — Zero-shot CLIP baseline

**Goal:** Establish the lower-bound number. Run CLIP ViT-B/32 with no training and report Recall@{1,5,10} for both text→image and image→text on the test split.

## Todos

- [x] Create `src/evaluate.py` (encode_images, encode_captions, compute_recall_at_k, evaluate_model)
- [x] Create `scripts/02_run_baseline.py` (zero-shot evaluation)
- [x] Run baseline, confirm R@1 lands in the 5–15% range (RSICD is a hard domain)
- [x] Save `results/metrics/baseline_zeroshot.json`
- [x] Sanity check: numbers are deterministic, model is in eval mode, embeddings are L2-normalized
- [x] Verify val split works (training script will use it)

## Decisions / deviations

- **FAISS on macOS:** `faiss-cpu` installed cleanly, BUT on macOS Apple Silicon loading faiss-cpu alongside sklearn+numpy (which each link different OpenMP runtimes) **hard-crashes Python** at FAISS module import. Two fixes:
  1. Default to **pure-numpy top-k** (instant for our ≤10k gallery sizes) and expose an `RSICD_USE_FAISS=1` env var to opt back in. The numpy path uses `argpartition + argsort` and is fully deterministic.
  2. Set `KMP_DUPLICATE_LIB_OK=TRUE` in `.venv/lib/python3.14/site-packages/sitecustomize.py` so the user never has to remember the prefix.
- **Output cleanliness:** silenced the noisy HF auth and transformers progress warnings via the same sitecustomize.
- **Metric format:** R@K as percentages, 2-decimal precision. Saved JSON schema:
  ```json
  {
    "model": "zero_shot_clip",
    "clip_model": "ViT-B-32",
    "trainable_params": 0,
    "split": "test",
    "text_to_image":  {"R@1": ..., "R@5": ..., "R@10": ...},
    "image_to_text":  {"R@1": ..., "R@5": ..., "R@10": ...},
    "device": "mps"
  }
  ```

## Results (this Mac, MPS)

```
Zero-shot CLIP on RSICD test split:
  Text -> Image: R@1= 5.95  R@5=18.21  R@10=28.64
  Image -> Text: R@1= 4.39  R@5=16.19  R@10=25.07
```

- T→I R@1 = 5.95% — within the expected 5-15% range. Good.
- Numbers are reproducible (set_seed=42 + sorted argpartition).

## Outputs

- `src/evaluate.py`
- `scripts/02_run_baseline.py`
- `results/metrics/baseline_zeroshot.json`
- `results/metrics/baseline_zeroshot_val.json` (for sanity-checking val)
- `.venv/lib/python3.14/site-packages/sitecustomize.py` (env defaults)
- `notebooks/baseline_val.py` (small val-side helper)

## Notes

- All later experiments will be compared to these test numbers. The splits JSON is the single source of truth.
- The same `evaluate_model` function works for the adapter and full-finetune models, since they expose the same `.encode_image()` / `.encode_text()` interface.
