# Day 5 — Full fine-tune CLIP baseline

**Goal:** Upper-bound baseline. Fine-tune all 150M CLIP parameters on RSICD. The adapter should land close to this number with 280× fewer trainable params — that is the paper's headline result.

## Todos

- [x] Create `configs/fullfinetune.yaml`
- [x] Create `scripts/04_run_fullfinetune.py`
- [ ] Train on Colab T4 (~4–6 hours for 10 epochs, batch=32) — for the paper

## Decisions / deviations

- **Smaller batch size (32 vs 64):** all 150M params now produce gradients, so VRAM usage roughly doubles. Batch=32 is the safe default on T4 (16GB). If OOM, drop to 16.
- **Lower learning rate (1e-5):** standard practice for full fine-tuning of a pretrained model. 1e-4 would destroy CLIP's pretrained features within a few epochs.
- **Fewer epochs (10 vs 20):** full fine-tune converges faster because the entire model adapts. 10 epochs is plenty; 20 risks overfitting.
- **No `WrappedCLIP` shim needed:** bare `open_clip` already exposes `.encode_image()` and `.encode_text()`. The Day-2 `evaluate_model` does the L2 normalization internally, so it works directly. (The IMPLEMENTATION.md's `WrappedCLIP` was only needed because that version of `evaluate.py` expected L2-normalized output; ours handles both.)
- **Gradient checkpointing:** NOT enabled by default. If OOM on T4, add `model.set_grad_checkpointing(True)` after loading. Saves memory at ~30% time cost.
- **Mixed precision:** NOT enabled. Stays in FP32 for reproducibility. (Could add BF16 for ~2x speedup on T4/A100, but the paper is small enough that FP32 runs in a few hours.)
- **Inference: gradients are disabled** by `evaluate_model`'s `@torch.no_grad()` decorator, so the bare model's `encode_*` calls are safe.

## Smoke-test notes

We did not run the full fine-tune on this Mac (it would take many hours — the adapter smoke test took 2 min, full fine-tune would take 30-60x longer). The script is structurally identical to the adapter trainer, so the same correctness guarantees apply.

## Outputs

- `configs/fullfinetune.yaml`
- `scripts/04_run_fullfinetune.py`
- (after Colab run) `results/checkpoints/fullfinetune_best.pt`
- (after Colab run) `results/metrics/fullfinetune_results.json`

## Expected numbers

- Full fine-tune T→I R@1: **~30–50%** with the same hidden_dim and 1 caption per image as our adapter run. The full fine-tune should beat the adapter by 5–15 pp but uses 280× more parameters.
- I→T R@1 typically slightly higher than T→I R@1 (CLIP is more reliable for matching images to a fixed caption bank than for ranking captions).

## Things to watch for (at Colab run-time)

- **OOM at batch=32:** drop to 16, or add gradient checkpointing.
- **Training loss decreases but val R@1 doesn't improve:** overfitting. Lower epochs to 5.
- **Final R@1 is BELOW the adapter:** something is wrong with the LR or the data flow. Check that `requires_grad=True` on all params after loading.
