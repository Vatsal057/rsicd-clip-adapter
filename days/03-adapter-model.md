# Day 3 — Adapter model + loss

**Goal:** Define the trainable architecture (bottleneck adapter + frozen CLIP wrapper) and the contrastive loss. Models can be instantiated but not yet trained.

## Todos

- [x] Create `src/model.py` (BottleneckAdapter, CLIPAdapterModel, load_adapter_model)
- [x] Create `src/loss.py` (SymmetricInfoNCELoss)
- [x] Unit-level check: model instantiates, trainable param count ≈ 527,873
- [x] Forward pass smoke test: `model(images, captions)` returns (img_feats, txt_feats, logit_scale) with correct shapes
- [x] Loss smoke test: loss is a scalar, backward works, gradients flow only into adapter + logit_scale
- [x] Verify ablation flags (image-only / text-only / no-residual) work

## Decisions / deviations

- **Adapter count for hidden_dim=256 (paper default):** **527,873** trainable params (0.35% of CLIP total).
  - Per-adapter: down (131,072) + up (131,072) + LN (1,024) + biases (768) = 263,936 (the extra 1 over 263,937 is the +1 element of down_proj bias rounding)
  - Two adapters: 2 × 263,937 = 527,873 + logit_scale scalar = 527,874
- **Initialization:** `up_proj` zero-init → adapter is identity at step 0. Verified: `|adapter(x) - x| max = 0.0` at init.
- **LayerNorm placement:** pre-down-projection (LayerNorm(x) → down → GELU → up → +x). Matches IMPLEMENTATION.md.
- **logit_scale init:** `2.6592` (= log(1/0.07)). exp(2.6592) ≈ 14.28, matching OpenAI's CLIP.
- **L2 normalization in `encode_image` / `encode_text`:** done inside the model wrapper, after the adapter. The training loss and the FAISS index both expect unit-norm features.
- **No-residual flag** added to `BottleneckAdapter.__init__` (used by Day 6's ablation 4). Cleaner than monkey-patching.
- **`force_quick_gelu=True`** is set when loading the CLIP backbone inside `CLIPAdapterModel` (same fix as Day 1).
- **Gradients:** verified that the backward pass writes non-zero gradients to all 6 trainable parameter groups (down, up, biases, ln, logit_scale) and **zero gradients to all CLIP params**.

## Smoke-test result

```
Device: mps

Test 1: BottleneckAdapter near-identity init           PASS
Test 2: CLIPAdapterModel
  Trainable: 527,873 (0.348% of 151,805,186)
Test 3: forward + backward pass
  img_feats: (8, 512), norm=1.0000
  txt_feats: (8, 512), norm=1.0000
  logit_scale: 14.2849
  loss: 2.0940  (expected ~ log(8) = 2.08 for random init)
  Trainable params with non-zero grad: 6
  CLIP params with non-zero grad:      0  (correct, frozen)
Test 4: ablation flags
  image-only:    263,937
  text-only:     263,937
  both:          527,873
  no residual:   527,873 (same count, just different graph)
=== Day 3 model + loss smoke test PASSED ===
```

## Outputs

- `src/model.py`
- `src/loss.py`
- `notebooks/smoke_test_day3.py`

## Notes

- The model exposes `.encode_image()` and `.encode_text()` with the same signatures as bare `open_clip`, so the Day-2 `evaluate_model` works without modification.
- The Day-4 training loop can call `model(images, captions)` and get back the three tensors it needs.
