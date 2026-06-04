"""Smoke test for src/model.py and src/loss.py."""
import sys
sys.path.insert(0, '.')

import torch
from src.utils    import get_device, count_parameters, print_model_summary
from src.model    import CLIPAdapterModel, BottleneckAdapter, load_adapter_model
from src.loss     import SymmetricInfoNCELoss


def main():
    device = get_device()
    print(f"Device: {device}\n")

    # ── Test 1: BottleneckAdapter near-identity init ────────────────────────
    print("Test 1: BottleneckAdapter near-identity init")
    ad = BottleneckAdapter(input_dim=512, hidden_dim=256).to(device)
    x = torch.randn(4, 512, device=device)
    y = ad(x)
    print(f"  x norm: {x.norm(dim=-1).mean().item():.4f}")
    print(f"  y norm: {y.norm(dim=-1).mean().item():.4f}")
    print(f"  |y - x| max: {(y - x).abs().max().item():.6f}  (should be ~0, up_proj is zero)")
    assert torch.allclose(y, x, atol=1e-5), "Adapter should be identity at init"
    print("  PASS\n")

    # ── Test 2: CLIPAdapterModel instantiation ──────────────────────────────
    print("Test 2: CLIPAdapterModel")
    model = CLIPAdapterModel(
        clip_model_name="ViT-B-32",
        clip_pretrained="openai",
        hidden_dim=256,
        dropout=0.1,
        adapter_on_image=True,
        adapter_on_text=True,
    ).to(device)
    print_model_summary(model, "CLIPAdapterModel")
    trainable = model.count_trainable_params()
    total     = model.count_total_params()
    pct       = 100.0 * trainable / total
    assert trainable < 1_000_000, f"Expected ~530K trainable, got {trainable:,}"
    assert pct < 1.0, f"Expected <1% trainable, got {pct:.2f}%"
    print(f"  Trainable {trainable:,} ({pct:.3f}% of {total:,}) — within target")

    # ── Test 3: Forward + backward pass ─────────────────────────────────────
    print("\nTest 3: forward + backward pass")
    model.train()
    images   = torch.randn(8, 3, 224, 224, device=device)
    captions = torch.randint(0, 49000, (8, 77), device=device)

    img_feats, txt_feats, logit_scale = model(images, captions)
    print(f"  img_feats: {tuple(img_feats.shape)}, norm={img_feats.norm(dim=-1).mean().item():.4f}")
    print(f"  txt_feats: {tuple(txt_feats.shape)}, norm={txt_feats.norm(dim=-1).mean().item():.4f}")
    print(f"  logit_scale: {logit_scale.item():.4f}")
    assert img_feats.shape == (8, 512)
    assert txt_feats.shape == (8, 512)
    assert torch.allclose(img_feats.norm(dim=-1), torch.ones(8, device=device), atol=1e-4)
    assert torch.allclose(txt_feats.norm(dim=-1), torch.ones(8, device=device), atol=1e-4)

    loss_fn = SymmetricInfoNCELoss()
    loss = loss_fn(img_feats, txt_feats, logit_scale)
    print(f"  loss: {loss.item():.4f}  (expected ~ log(N)=2.08 for random init)")
    assert 1.5 < loss.item() < 3.0, f"Unexpected loss: {loss.item()}"

    # Backward: gradient should flow into adapter + logit_scale, NOT into CLIP
    loss.backward()
    n_grad_adapters = sum(
        1 for n, p in model.named_parameters() if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0
    )
    n_clip_with_grad = sum(
        1 for n, p in model.clip.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0
    )
    n_trainable_with_grad = sum(
        1 for n, p in model.named_parameters() if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0
    )
    print(f"  Trainable params with non-zero grad: {n_trainable_with_grad}")
    print(f"  CLIP params with non-zero grad:      {n_clip_with_grad}  (should be 0)")
    assert n_clip_with_grad == 0, "CLIP weights should be frozen"

    # ── Test 4: Abalation flags (image-only, text-only, no-residual) ────────
    print("\nTest 4: ablation flags")
    for img_on, txt_on in [(True, False), (False, True), (True, True)]:
        m = CLIPAdapterModel(adapter_on_image=img_on, adapter_on_text=txt_on).to(device)
        n = m.count_trainable_params()
        print(f"  image={img_on} text={txt_on}: {n:,} trainable")

    m_no_res = CLIPAdapterModel(use_residual=False).to(device)
    n = m_no_res.count_trainable_params()
    print(f"  no residual: {n:,} trainable  (should equal with-residual)")

    print("\n=== Day 3 model + loss smoke test PASSED ===")


if __name__ == "__main__":
    main()
