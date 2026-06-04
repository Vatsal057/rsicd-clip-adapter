"""
Baseline 2: Full fine-tune CLIP on RSICD.

This is the upper-bound baseline (expensive, impractical). The adapter
should land close to this number with 280x fewer trainable parameters --
that is the paper's headline result.

Loads bare open_clip CLIP, unfreezes all parameters, trains with the
same InfoNCE loss and cosine schedule as the adapter.
"""

import json
import sys
import time
from pathlib import Path

import open_clip
import torch
import yaml
from torch.optim import AdamW

sys.path.insert(0, ".")

from src.dataset  import get_dataloaders
from src.evaluate import evaluate_model, save_results
from src.loss     import SymmetricInfoNCELoss
from src.train    import get_warmup_cosine_scheduler
from src.utils    import get_device, set_seed, print_model_summary


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/fullfinetune.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"]["seed"])
    device = get_device()
    print(f"Full fine-tune on: {device}")

    ckpt_dir    = Path(cfg["paths"]["checkpoint_dir"])
    metrics_dir = Path(cfg["paths"]["metrics_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / cfg["paths"]["save_best_name"]

    # ── Data ────────────────────────────────────────────────────────────────
    train_loader, val_retrieval, test_retrieval, _, _ = get_dataloaders(
        splits_dir = cfg["data"]["splits_dir"],
        images_dir = cfg["data"]["images_dir"],
        batch_size = cfg["training"]["batch_size"],
        num_workers= cfg["training"].get("num_workers", 4),
    )
    print(f"Train batches/epoch: {len(train_loader)}")

    # ── Model: bare CLIP, everything trainable ──────────────────────────────
    model, _ = open_clip.create_model_from_pretrained(
        cfg["model"]["clip_model_name"],
        pretrained=cfg["model"]["clip_pretrained"],
        force_quick_gelu=True,
    )
    model = model.to(device)
    if not cfg["model"].get("freeze_clip", False):
        for p in model.parameters():
            p.requires_grad = True

    print_model_summary(model, "CLIP (full fine-tune)")

    # ── Loss + Optim + Scheduler ───────────────────────────────────────────
    loss_fn   = SymmetricInfoNCELoss()
    optimizer = AdamW(
        model.parameters(),
        lr           = cfg["training"]["learning_rate"],
        weight_decay = cfg["training"]["weight_decay"],
    )
    total_steps  = max(1, len(train_loader) * cfg["training"]["num_epochs"])
    warmup_steps = len(train_loader) * cfg["training"]["warmup_epochs"]
    scheduler    = get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)

    best_val_r1   = 0.0
    history: list = []
    log_every     = cfg["paths"].get("log_every", 50)
    eval_every    = cfg["paths"].get("eval_every", 1)
    grad_clip     = cfg["training"]["grad_clip_norm"]

    print(f"\nStarting full fine-tune for {cfg['training']['num_epochs']} epochs...")
    print(f"  total steps: {total_steps}, warmup: {warmup_steps}, peak LR: {cfg['training']['learning_rate']:.2e}\n")

    for epoch in range(1, cfg["training"]["num_epochs"] + 1):
        model.train()
        ep_loss = 0.0
        t0 = time.time()

        for step, (images, captions, _) in enumerate(train_loader, 1):
            images   = images.to(device, non_blocking=True)
            captions = captions.to(device, non_blocking=True)

            img_feats = model.encode_image(images)
            txt_feats = model.encode_text(captions)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            logit_scale = model.logit_scale.exp()

            loss = loss_fn(img_feats, txt_feats, logit_scale)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            ep_loss += loss.item()
            if step % log_every == 0 or step == 1:
                lr_now = scheduler.get_last_lr()[0]
                print(f"  Ep {epoch:02d} | Step {step:04d}/{len(train_loader)} "
                      f"| Loss {loss.item():.4f} | LR {lr_now:.2e} | "
                      f"τ {logit_scale.item():.2f}")

        avg_loss = ep_loss / max(1, len(train_loader))
        elapsed  = time.time() - t0
        print(f"Epoch {epoch:02d} done | avg loss {avg_loss:.4f} | {elapsed:.1f}s")

        if epoch % eval_every == 0:
            # Bare open_clip model satisfies evaluate_model's interface:
            # .encode_image() / .encode_text() (L2 normalization happens inside).
            val_results = evaluate_model(
                model, val_retrieval, device, split_name="val",
                image_batch_size  = cfg["eval"]["image_batch_size"],
                caption_batch_size= cfg["eval"]["caption_batch_size"],
                num_workers       = cfg["training"].get("num_workers", 4),
            )
            val_r1 = val_results["text_to_image"]["R@1"]
            history.append({"epoch": epoch, "train_loss": avg_loss, "val_results": val_results})

            if val_r1 > best_val_r1:
                best_val_r1 = val_r1
                torch.save({
                    "epoch":            epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_r1":           val_r1,
                    "config":           cfg,
                    "timestamp":        time.strftime("%Y-%m-%d %H:%M:%S"),
                }, best_ckpt)
                print(f"  *** New best val R@1 = {val_r1:.2f}% -> saved {best_ckpt.name} ***")

    with open(metrics_dir / "training_history_fullfinetune.json", "w") as f:
        json.dump(history, f, indent=2)

    # ── Final test eval on best checkpoint ──────────────────────────────────
    print("\n" + "=" * 60)
    print("Loading best checkpoint for final test evaluation...")
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    test_results = evaluate_model(
        model, test_retrieval, device, split_name="test",
        image_batch_size  = cfg["eval"]["image_batch_size"],
        caption_batch_size= cfg["eval"]["caption_batch_size"],
        num_workers       = cfg["training"].get("num_workers", 4),
    )
    test_results["model"]            = "full_finetune_clip"
    test_results["trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    test_results["best_val_epoch"]   = ckpt["epoch"]
    test_results["config"]           = cfg
    test_results["device"]           = device

    save_results(test_results, metrics_dir / "fullfinetune_results.json")
    print("\n=== FULL FINE-TUNE COMPLETE ===")
    print(f"  T->I  R@1 : {test_results['text_to_image']['R@1']:>6.2f}%")
    print(f"  I->T  R@1 : {test_results['image_to_text']['R@1']:>6.2f}%")
    print(f"  Trainable : {test_results['trainable_params']:,}")


if __name__ == "__main__":
    main()
