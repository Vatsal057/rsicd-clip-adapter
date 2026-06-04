"""
Shared utilities: device selection, seeding, checkpointing, metrics IO, model summary.
"""

import os
import json
import random
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def get_device() -> str:
    """
    Pick the best available device, in order: CUDA > MPS > CPU.
    Used everywhere instead of `cuda or cpu` so the same code runs on
    Colab (cuda), Apple Silicon Macs (mps), and CI / dev machines (cpu).
    """
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(name: str, log_file: str | None = None,
                 level: int = logging.INFO) -> logging.Logger:
    """Create a logger that writes to console and optionally to a file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def save_checkpoint(model, optimizer, epoch: int, val_r1: float,
                    config: dict, path: str) -> None:
    """Save a full training checkpoint (model + optimizer + meta)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_r1": val_r1,
        "config": config,
        "timestamp": datetime.now().isoformat(),
    }, path)


def load_checkpoint(path: str, model, optimizer=None, device: str = "cpu"):
    """Load a checkpoint saved with `save_checkpoint`. Returns (epoch, val_r1)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt.get("epoch", 0), ckpt.get("val_r1", 0.0)


def save_metrics(metrics: dict, path: str) -> None:
    """Save a metrics dict to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved -> {path}")


def load_metrics(path: str) -> dict:
    """Load a metrics JSON file."""
    with open(path) as f:
        return json.load(f)


def count_parameters(model) -> dict:
    """Return trainable and total parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    return {
        "trainable":     trainable,
        "frozen":        total - trainable,
        "total":         total,
        "trainable_pct": round(100.0 * trainable / total, 3) if total else 0.0,
    }


def print_model_summary(model, model_name: str = "Model") -> None:
    """Print a concise parameter count summary."""
    c = count_parameters(model)
    print(f"\n{'=' * 50}")
    print(f"  {model_name} Parameter Summary")
    print(f"{'=' * 50}")
    print(f"  Trainable : {c['trainable']:>12,}  ({c['trainable_pct']:.2f}%)")
    print(f"  Frozen    : {c['frozen']:>12,}")
    print(f"  Total     : {c['total']:>12,}")
    print(f"{'=' * 50}\n")
