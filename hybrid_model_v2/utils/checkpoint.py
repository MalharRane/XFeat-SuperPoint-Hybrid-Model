from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    *,
    output_dir: str,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau],
    cfg: Dict[str, Any],
    val_loss: float,
    is_best: bool,
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "cfg": cfg,
        "val_loss": float(val_loss),
    }
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()

    ckpt = out_dir / f"epoch_{epoch:04d}.pth"
    torch.save(state, ckpt)

    if is_best:
        torch.save(state, out_dir / "best.pth")
    return ckpt


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau],
    device: torch.device,
) -> int:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")

    state = torch.load(str(p), map_location=device, weights_only=False)
    for key in ("epoch", "model", "optimizer", "scaler"):
        if key not in state:
            raise RuntimeError(f"Invalid checkpoint: missing '{key}'")

    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    return int(state["epoch"]) + 1
