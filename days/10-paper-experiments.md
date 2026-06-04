# Day 10 — Section 4 (Experiments) + all result tables

**Goal:** Experiments section complete with all 5 tables filled with real numbers.

## Todos

- [x] Draft Section 4.1 (Dataset)
- [x] Draft Section 4.2 (Evaluation Protocol)
- [x] Draft Section 4.3 (Baselines)
- [x] Draft Section 4.4 (Implementation Details) + **NEW hyperparameter Table (Table 1)**
- [x] Add Figure 3 (training curve) — `figures/training_curve.pdf` referenced
- [x] Draft Section 4.5 (Main Results) + Table 2 (was Table 1)
- [x] Draft Section 4.6 (Ablation Studies) + Tables 3-6 (was 2-5)
- [x] Draft Section 4.7 (Qualitative Analysis) + Figures 4-6 references
- [ ] Fill every `[TBD]` placeholder in tables with real numbers from `results/metrics/*.json`
- [ ] Compute `(adapter_R@1 / full_finetune_R@1) * 100` for the discussion section
- [ ] Verify table numbers match the JSON files (no copy-paste errors)

## Decisions / deviations

- **Renumbered tables** to add a hyperparameter table at the start of Section 4:
  - Table 1 (Hparams) → 4.4 Implementation Details
  - Table 2 (Main Results) → 4.5
  - Tables 3-6 (Ablations) → 4.6
- **4 figures in Section 4:** training_curve (3), qualitative (4), failures (5), ablation_dim (6)
- **Placeholder format:** all `XX.X` → `\textit{[TBD]}` for visual consistency in compiled PDF
- **Table format:** `\multirow` + `\cmidrule` for the main results table (Method × Trainable Params × T→I R@{1,5,10} × I→T R@{1,5,10}). Matches the IMPLEMENTATION.md template.
- **Footnotes:** `$\dagger$` = full model trained, `$\ddagger$` = only adapter trained. Defined in the caption.
- **Three numbered takeaways** added after main results table: (i) absorbs domain shift, (ii) parameter reduction, (iii) symmetric T↔I. These are the anchors that Section 5 (Discussion) elaborates on.
- **Don't re-run the smoke test** — it polluted the val metric; val R@1 = 2.65 should be ignored.

## Outputs

- Updated `paper/main.tex` Section 4 (4.1-4.7, 6 tables, 4 figures, ~330 lines)
- All XX.X placeholders uniformly formatted as `[TBD]`
- Hyperparameter Table~\ref{tab:hparams} with all real values (bottleneck=256, batch=64, lr=1e-4, 20 epochs, etc.)

## Auto-compute script (run before filling tables)

```python
import json

zs = json.load(open("results/metrics/baseline_zeroshot.json"))
ad = json.load(open("results/metrics/adapter_results.json"))
ff = json.load(open("results/metrics/fullfinetune_results.json"))

z_r1  = zs["text_to_image"]["R@1"]
a_r1  = ad["text_to_image"]["R@1"]
f_r1  = ff["text_to_image"]["R@1"]
delta = round(a_r1 - z_r1, 1)
pct   = round(100 * a_r1 / f_r1, 1)
ratio = round(150_000_000 / ad["trainable_params"])

print(f"Zero-shot R@1:          {z_r1}%")
print(f"Adapter R@1:            {a_r1}%")
print(f"Full fine-tune R@1:     {f_r1}%")
print(f"Delta (improvement):    +{delta} pp")
print(f"% of full fine-tune:    {pct}%")
print(f"Parameter ratio:        {ratio}×")
```

## Notes

- **Number integrity is paramount.** Every number in the paper must be traceable to a JSON file. Keep the JSONs committed.
- The reviewer can ask "what is the exact value of X" — your answer should be "line 42 of `adapter_results.json`".
- The 4 figures are already generated as placeholders (`paper/figures/{training_curve,qualitative,failures,ablation_dim}.pdf`). After Colab runs, regenerate them with real data.
