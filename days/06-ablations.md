# Day 6 — Ablation experiments

**Goal:** Four ablations, each isolating one design choice. Feeds Tables 2–5 of the paper.

## Todos

- [x] Create `scripts/05_ablations.py`
- [x] Add `use_residual` flag to `BottleneckAdapter` (done in Day 3, not refactored here)
- [x] Verify script imports cleanly
- [ ] Ablation 1: placement (image-only / text-only / both) → `ablation_placement.json`  (Colab)
- [ ] Ablation 2: hidden_dim (64 / 128 / 256 / 512) → `ablation_hidden_dim.json`     (Colab)
- [ ] Ablation 3: training data fraction (25 / 50 / 75 / 100%) → `ablation_data_size.json`  (Colab)
- [ ] Ablation 4: residual (with / without) → `ablation_residual.json`              (Colab)

## Decisions / deviations

- **Why `use_residual` flag instead of monkey-patching:** monkey-patching in the original IMPLEMENTATION.md is fragile (it depends on `train()` importing `BottleneckAdapter` by name). A flag on the adapter is cleaner and survives any refactor. We added the flag in Day 3 and just toggle it here via the config.
- **Subset data ablation:** write temp splits to a fresh `tempfile.mkdtemp(prefix="rsicd_abl_size_")` dir. Keep the canonical `data/splits/{train,val,test}.json` untouched. The temp dir is auto-cleaned on process exit.
- **Each ablation trains 20 epochs** (same as base config). Total compute: 3 + 4 + 4 + 2 = 13 training runs, each ~2 hours on T4 = ~26 GPU-hours.
- **Selective run:** `python scripts/05_ablations.py placement hidden_dim` runs just those two. Useful for parallelism across multiple Colab notebooks.
- **Metric reported in tables:** T→I R@1, R@5, R@10 (and I→T R@1 for placement and hidden_dim — the most "interesting" ablations).
- **`run_tag`** in the training script is set per ablation (e.g. `placement_image_only`, `dim_256`, `size_0.5`, `residual_with`) so each run saves its own checkpoint + history + results without overwriting the base.

## Outputs

- `scripts/05_ablations.py`
- (after Colab run) `results/metrics/ablations/{ablation_placement,ablation_hidden_dim,ablation_data_size,ablation_residual}.json`
- (after Colab run) one checkpoint + one results file per ablation run, e.g.:
  - `results/checkpoints/placement_image_only_best.pt`
  - `results/metrics/placement_image_only_results.json`
  - etc.

## Expected findings (hypotheses)

1. **Placement:** "both" > "image-only" ≈ "text-only". Both modalities carry domain shift.
2. **Hidden dim:** 256 ≈ 512 > 128 > 64. Diminishing returns past 256; we picked 256 as the sweet spot.
3. **Data size:** monotonic improvement with more data, but with diminishing returns — adapter is data-efficient.
4. **Residual:** with > without, by a small-to-moderate margin. Justifies the design.

## Things to watch for

- **Total runtime:** ~26 GPU-hours sequential. On Colab free tier (T4, ~4 hrs/day cap), this is 6+ days. **Plan: run on Colab Pro, or run subsets in parallel across multiple notebooks.**
- **Each ablation run is fully independent** — they don't share checkpoints, so they can be parallelized across notebooks.
