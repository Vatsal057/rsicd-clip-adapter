"""
Training loop for the CLIP adapter model.

Reads a YAML config, instantiates model + data + optimizer, and runs the
training loop. Saves the best-val-checkpoint by val T->I R@1, then loads
that checkpoint for a final test evaluation.

Usage:
    from src.train import train
    train("configs/adapter_base.yaml")
"""

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

sys.path.insert(0, ".")

from src.dataset  import get_dataloaders
from src.evaluate import evaluate_model, save_results
from src.loss     import SymmetricInfoNCELoss
from src.model    import CLIPAdapterModel
from src.utils    import get_device, set_seed, print_model_summary


def get_warmup_cosine_scheduler(optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    """Linear warmup for `warmup_steps`, cosine decay to 0 over the rest."""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def build_model_from_config(model_cfg: dict, device: str) -> CLIPAdapterModel:
    """Instantiate CLIPAdapterModel from a model config dict."""
    return CLIPAdapterModel(
        clip_model_name    = model_cfg["clip_model_name"],
        clip_pretrained     = model_cfg["clip_pretrained"],
        hidden_dim          = model_cfg["hidden_dim"],
        dropout             = model_cfg["dropout"],
        adapter_on_image    = model_cfg["adapter_on_image"],
        adapter_on_text     = model_cfg["adapter_on_text"],
        use_residual        = model_cfg.get("use_residual", True),
        learn_logit_scale   = model_cfg.get("learn_logit_scale", True),
    ).to(device)


def train(config_path: str = "configs/adapter_base.yaml",
          run_tag:     str = "adapter") -> dict:
    """Run the training loop. Returns the final test results dict."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"]["seed"])
    device = get_device()
    print(f"Training on: {device}")
    print(f"Run tag:     {run_tag}")

    # ── Directories ─────────────────────────────────────────────────────────
    ckpt_dir    = Path(cfg["paths"]["checkpoint_dir"])
    metrics_dir = Path(cfg["paths"]["metrics_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / cfg["paths"].get("save_best_name", f"{run_tag}_best.pt")

    # ── Data ────────────────────────────────────────────────────────────────
    train_loader, val_retrieval, test_retrieval, _, _ = get_dataloaders(
        splits_dir = cfg["data"]["splits_dir"],
        images_dir = cfg["data"]["images_dir"],
        batch_size = cfg["training"]["batch_size"],
        num_workers= cfg["training"].get("num_workers", 4),
    )
    print(f"Train batches/epoch: {len(train_loader)} | "
          f"Val images: {len(val_retrieval.images)} | "
          f"Test images: {len(test_retrieval.images)}")

    # ── Model ───────────────────────────────────────────────────────────────
    model = build_model_from_config(cfg["model"], device)
    trainable = model.count_trainable_params()
    total     = model.count_total_params()
    print_model_summary(model, f"CLIPAdapterModel ({run_tag})")

    # ── Loss + Optimizer + Scheduler ────────────────────────────────────────
    loss_fn   = SymmetricInfoNCELoss()
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr           = cfg["training"]["learning_rate"],
        weight_decay = cfg["training"]["weight_decay"],
    )
    total_steps   = max(1, len(train_loader) * cfg["training"]["num_epochs"])
    warmup_steps  = len(train_loader) * cfg["training"]["warmup_epochs"]
    scheduler     = get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)

    # ── Training state ──────────────────────────────────────────────────────
    best_val_r1     = 0.0
    history: list   = []
    log_every       = cfg["paths"].get("log_every", 50)
    eval_every      = cfg["paths"].get("eval_every", 1)
    grad_clip_norm  = cfg["training"]["grad_clip_norm"]
    num_epochs      = cfg["training"]["num_epochs"]

    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"  total steps: {total_steps}, warmup: {warmup_steps}, "
          f"peak LR: {cfg['training']['learning_rate']:.2e}\n")

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, (images, captions, _) in enumerate(train_loader, 1):
            images   = images.to(device, non_blocking=True)
            captions = captions.to(device, non_blocking=True)

            img_feats, txt_feats, logit_scale = model(images, captions)
            loss = loss_fn(img_feats, txt_feats, logit_scale)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            if step % log_every == 0 or step == 1:
                lr_now = scheduler.get_last_lr()[0]
                print(f"  Ep {epoch:02d} | Step {step:04d}/{len(train_loader)} "
                      f"| Loss {loss.item():.4f} | LR {lr_now:.2e} | "
                      f"τ {logit_scale.item():.2f}")

        avg_loss = epoch_loss / max(1, len(train_loader))
        elapsed  = time.time() - t0
        print(f"Epoch {epoch:02d} done | avg loss {avg_loss:.4f} | {elapsed:.1f}s")

        # ── Periodic val evaluation + best-checkpoint save ──────────────────
        if epoch % eval_every == 0:
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
                    "config":           cfg["model"],
                    "timestamp":        time.strftime("%Y-%m-%d %H:%M:%S"),
                }, best_ckpt_path)
                print(f"  *** New best val R@1 = {val_r1:.2f}% -> saved {best_ckpt_path.name} ***")

    # ── Save training history ───────────────────────────────────────────────
    with open(metrics_dir / f"training_history_{run_tag}.json", "w") as f:
        json.dump(history, f, indent=2)

    # ── Final test eval on best checkpoint ──────────────────────────────────
    print("\n" + "=" * 60)
    print("Loading best checkpoint for final test evaluation...")
    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_results = evaluate_model(
        model, test_retrieval, device, split_name="test",
        image_batch_size  = cfg["eval"]["image_batch_size"],
        caption_batch_size= cfg["eval"]["caption_batch_size"],
        num_workers       = cfg["training"].get("num_workers", 4),
    )
    test_results["model"]            = run_tag
    test_results["trainable_params"] = trainable
    test_results["total_params"]     = total
    test_results["best_val_epoch"]   = ckpt["epoch"]
    test_results["config"]           = cfg
    test_results["device"]           = device

    out_path = metrics_dir / f"{run_tag}_results.json"
    save_results(test_results, out_path)
    print(f"\nFinal test results -> {out_path}")
    return test_results


if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "configs/adapter_base.yaml"
    train(cfg)
