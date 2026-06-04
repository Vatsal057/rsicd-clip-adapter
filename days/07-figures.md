# Day 7 — Qualitative results + paper figures

**Goal:** All figures referenced in the paper exist as 300 DPI PDFs in `paper/figures/`.

## Todos

- [x] Create `scripts/06_qualitative.py`
- [x] Generate `fig_training_curve.pdf` (tested with smoke history)
- [x] Generate `fig_ablation_dim.pdf` (tested with dummy ablation data)
- [x] Generate `fig_qualitative.pdf` (tested with smoke checkpoint)
- [x] Generate `fig_failures.pdf` (tested with smoke checkpoint)
- [ ] Re-generate all 4 figures with the real Colab runs

## Decisions / deviations

- **All figures as PDF, not PNG.** Vector graphics scale to any submission template and look professional. matplotlib's `savefig` with `.pdf` extension does this by default.
- **DPI = 300** for any rasterized output. Matches typical journal requirements.
- **Caption typography:** consistent style — 12pt titles, 11pt labels, no top/right spines, light grid. Modern ML paper aesthetic.
- **Color palette:** blue `#185FA5` (primary), green `#0F6E56` (success/correct), red `#D85A30` (failure/incorrect). Used consistently in the qualitative borders and the ablation curve.
- **Qualitative figure layout:** N rows (queries) × K+1 cols (query text + top-K images). Green border = correct, red border = incorrect.
- **Failure cases:** automatically chosen as the next 2 captions after the success queries. The IMPLEMENTATION.md mentioned "storage tanks → stadiums" as the canonical example; ours is whatever the test set produces (likely airports → other airports with similar visual features).
- **Missing-file resilience:** each figure function checks for its input file and `[skip]`s with a clear message if absent. This lets the script be run incrementally as experiments finish.
- **Pure-numpy top-K in `retrieve`:** consistent with Day 2's decision to avoid FAISS on macOS.
- **No architecture figure here** — that's Day 9, drawn in TikZ inside `main.tex`.

## Smoke-test result

We temporarily renamed the smoke training outputs to the canonical names, generated all 4 figures, verified each is a valid PDF, then deleted the temporary test files. The script is ready for the real Colab run.

## Outputs

- `scripts/06_qualitative.py`
- `paper/figures/fig_training_curve.pdf` (placeholder, regenerate after real run)
- `paper/figures/fig_ablation_dim.pdf`   (placeholder, regenerate after real run)
- `paper/figures/fig_qualitative.pdf`    (placeholder, regenerate after real run)
- `paper/figures/fig_failures.pdf`       (placeholder, regenerate after real run)

## Notes

- This script depends on the best adapter checkpoint from Day 4, and on `ablation_hidden_dim.json` from Day 6. It will run successfully (skips missing pieces with a clear message) as soon as those exist.
- All four figures are referenced by name in `main.tex` (Sections 4 and 5).
- The script accepts env-var overrides for paths (e.g. `RESULTS_DIR`, `PAPER_FIGS`) so Colab can point at Drive-backed directories.
