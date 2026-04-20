from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format='[%(asctime)s] %(levelname)s %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    )
    return logging.getLogger("hybrid_model_v2")


def setup_tracking(cfg: Dict[str, Any]) -> Tuple[Optional[Any], Optional[Any]]:
    backend = str(cfg.get("logging_backend", "tensorboard")).lower()
    log_dir = Path(str(cfg.get("log_dir", "hybrid_model_v2/runs")))
    log_dir.mkdir(parents=True, exist_ok=True)

    tb_writer = None
    wandb_run = None

    if backend == "wandb":
        import wandb  # optional

        wandb_run = wandb.init(
            project=str(cfg.get("wandb_project", "hybrid_model_v2")),
            config=cfg,
            dir=str(log_dir),
        )
    else:
        from torch.utils.tensorboard import SummaryWriter

        tb_writer = SummaryWriter(log_dir=str(log_dir))

    return tb_writer, wandb_run


def log_metrics(
    *,
    metrics: Dict[str, float],
    step: int,
    prefix: str,
    tb_writer: Optional[Any],
    wandb_run: Optional[Any],
) -> None:
    if tb_writer is not None:
        for k, v in metrics.items():
            tb_writer.add_scalar(f"{prefix}/{k}", float(v), step)
    if wandb_run is not None:
        wandb_run.log({f"{prefix}/{k}": float(v) for k, v in metrics.items()}, step=step)
