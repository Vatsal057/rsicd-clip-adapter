# Day 4 — Training loop + first adapter run

**Goal:** Adapter trains end-to-end on RSICD train split, evaluates on val and test every epoch, saves the best-val checkpoint, and writes final test metrics to `adapter_results.json`.

## Todos

- [x] Create `configs/adapter_base.yaml`
- [x] Create `configs/smoke_adapter.yaml` (tiny config for end-to-end loop verification)
- [x] Create `src/train.py` (train(), set_seed, get_warmup_cosine_scheduler)
- [x] Create `scripts/03_train_adapter.py` (entrypoint)
- [x] Run smoke test (1 epoch, hidden_dim=64, batch=8) on this Mac MPS
- [ ] Train on Colab T4 (~2 hours for 20 epochs, batch=64) — for the paper

## Decisions / deviations

- **Scheduler:** linear warmup for `warmup_epochs` epochs, then cosine decay to 0. Matches IMPLEMENTATION.md.
- **Optimizer:** AdamW on trainable params only (`[p for p in model.parameters() if p.requires_grad]`). The logit_scale is included (it has `requires_grad=True`).
- **Gradient clipping:** `clip_grad_norm_(model.parameters(), 1.0)`. Protects against the rare spike in early training when up_proj is still ~0 but down_proj receives a strong gradient.
- **Best-checkpoint metric:** val T→I R@1. We track this and save the model whenever it improves.
- **Final evaluation:** load best checkpoint, evaluate on test, save `{run_tag}_results.json`.
- **Logging cadence:** print loss every `log_every` steps (default 50), evaluate on val every `eval_every` epochs (default 1).
- **Device:** use `get_device()` from `src/utils.py` (cuda → mps → cpu). Identical script works on Colab and Mac.
- **Mixed precision:** NOT enabled. On MPS, autocast is unreliable; on T4, BF16 is fine but the IMPLEMENTATION.md doesn't specify it, so we stay in FP32 for reproducibility. The adapter is tiny so memory is not a concern.
- **Save artifact naming:** `{run_tag}_best.pt` (not hardcoded `adapter_best.pt`). The ablation runner can pass distinct tags to avoid overwriting.
- **Smoke test config** lives at `configs/smoke_adapter.yaml` and is intentionally small (hidden_dim=64, batch=8, 1 epoch, num_workers=0). Used only to verify the loop runs.

## Smoke-test result (MPS, 1 epoch, hidden_dim=64)

```
Training on: mps
Train batches/epoch: 1091 | Val images: 1094 | Test images: 1093

  Ep 01 | Step 0050/1091 | Loss 1.20 | LR 1.0e-04
  ...
  Ep 01 | Step 1090/1091 | Loss 0.63 | LR 2e-10
Epoch 01 done | avg loss 0.7552 | 105.4s

Evaluating on val split...
  Text -> Image: R@1= 4.30  R@5=15.17  R@10=26.87
  Image -> Text: R@1= 4.11  R@5=15.45  R@10=25.32
  *** New best val R@1 = 4.30% -> saved smoke_adapter_best.pt ***

Evaluating on test split...
  Text -> Image: R@1= 5.49  R@5=20.13  R@10=31.29
  Image -> Text: R@1= 5.22  R@5=18.21  R@10=30.19
  Trainable : 134,273
```

Loss decreases; test R@1 = 5.49% (vs zero-shot 5.95%) for a tiny 134K-param adapter after 1 epoch. The full 20-epoch run with hidden_dim=256 will be much better.

## Outputs

- `configs/adapter_base.yaml`
- `configs/smoke_adapter.yaml`
- `src/train.py`
- `scripts/03_train_adapter.py`
- `results/checkpoints/smoke_adapter_best.pt` (~2 MB)
- `results/metrics/training_history_smoke.json`
- `results/metrics/smoke_results.json`

## Notes

- The smoke test took 105s on this Mac's MPS. A real 20-epoch run (batch=64, hidden_dim=256) would be ~30-40 min on MPS. We will defer the real run to Colab T4 (~1-2 hours).
- The smoke test confirms the entire pipeline works: train loader, model, loss, backward, scheduler, gradient clip, periodic eval, best-checkpoint save, final test eval.
