from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from hybrid_model_v2.datasets import build_dataloaders_v2
from hybrid_model_v2.losses import ScoreWeightedHingeRepeatabilityLoss
from hybrid_model_v2.models import (
    HybridModelV2,
    build_superpoint,
    build_xfeat,
    ensure_file,
    load_weights_strictish,
)
from hybrid_model_v2.utils.amp import grads_are_finite, is_amp_related_error, tensor_is_finite
from hybrid_model_v2.utils.checkpoint import load_checkpoint, save_checkpoint
from hybrid_model_v2.utils.config import build_arg_parser, load_yaml_config, merge_config_with_args
from hybrid_model_v2.utils.logging_utils import log_metrics, setup_logging, setup_tracking
from hybrid_model_v2.utils.metrics import mean_stats, pick_model_score
from hybrid_model_v2.utils.preflight import assert_superpoint_frozen, check_plateau_break, validate_lightglue_contract


log = setup_logging()
_GRID_DIVISIBILITY = 8


def _add_new_trainable_params(model: nn.Module, optimizer: optim.Optimizer, lr: float, weight_decay: float) -> int:
    existing = {id(p) for g in optimizer.param_groups for p in g["params"]}
    new_params = [p for p in model.parameters() if p.requires_grad and id(p) not in existing]
    if not new_params:
        return 0
    optimizer.add_param_group({"params": new_params, "lr": lr, "weight_decay": weight_decay})
    return sum(p.numel() for p in new_params)


def build_model_v2(cfg: Dict[str, Any], device: torch.device) -> HybridModelV2:
    xfeat = build_xfeat()
    superpoint = build_superpoint()

    sp_path = ensure_file(
        Path(str(cfg.get("superpoint_weights_path", "weights/superpoint_v1.pth"))),
        str(cfg["superpoint_weights_url"]),
    )
    sp_overlap, sp_total, sp_miss, sp_unexp = load_weights_strictish(
        superpoint,
        sp_path,
        module_name="SuperPoint",
        strict=False,
        min_overlap_ratio=0.10,
    )
    log.info(
        "SuperPoint weights loaded: overlap=%d/%d missing=%d unexpected=%d",
        sp_overlap,
        sp_total,
        sp_miss,
        sp_unexp,
    )

    xf_path = Path(str(cfg.get("xfeat_weights_path", "weights/xfeat.pt")))
    if not xf_path.exists() and xf_path.suffix == ".pt":
        alt = xf_path.with_suffix(".pth")
        if alt.exists():
            xf_path = alt
    if not xf_path.exists():
        raise FileNotFoundError(f"XFeat weights not found: {xf_path}")

    xf_overlap, xf_total, xf_miss, xf_unexp = load_weights_strictish(
        xfeat,
        xf_path,
        module_name="XFeat",
        strict=False,
        min_overlap_ratio=0.10,
    )
    log.info(
        "XFeat weights loaded: overlap=%d/%d missing=%d unexpected=%d",
        xf_overlap,
        xf_total,
        xf_miss,
        xf_unexp,
    )

    model = HybridModelV2(
        xfeat_core=xfeat,
        superpoint_core=superpoint,
        num_keypoints=int(cfg.get("num_keypoints", 1024)),
        nms_radius=int(cfg.get("nms_radius", 4)),
        min_keypoint_score=float(cfg.get("min_keypoint_score", 0.01)),
        descriptor_dim=int(cfg.get("descriptor_dim", 256)),
    ).to(device)

    assert_superpoint_frozen(model)
    return model


def run_preflight(
    cfg: Dict[str, Any],
    model: HybridModelV2,
    train_loader: torch.utils.data.DataLoader,
    loss_fn: ScoreWeightedHingeRepeatabilityLoss,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
) -> None:
    if int(cfg["image_height"]) % _GRID_DIVISIBILITY != 0 or int(cfg["image_width"]) % _GRID_DIVISIBILITY != 0:
        raise RuntimeError(f"image_height/image_width must be divisible by {_GRID_DIVISIBILITY}")

    ds = train_loader.dataset
    if hasattr(ds, "items_by_scene"):
        scene_stats = {k: len(v) for k, v in ds.items_by_scene.items()}
        log.info("Preflight scene discovery: %s", scene_stats)

    batch = next(iter(train_loader))
    image1 = batch["image1"].to(device)
    image2 = batch["image2"].to(device)
    homography = batch["homography"].to(device)
    warp_field = batch.get("warp_field")
    warp_valid = batch.get("warp_valid")
    depth_valid1 = batch.get("depth_valid1")
    if isinstance(warp_field, torch.Tensor):
        warp_field = warp_field.to(device)
    if isinstance(warp_valid, torch.Tensor):
        warp_valid = warp_valid.to(device)
    if isinstance(depth_valid1, torch.Tensor):
        depth_valid1 = depth_valid1.to(device)

    model.train()
    amp_enabled = bool(cfg.get("mixed_precision", True) and device.type == "cuda")
    optimizer.zero_grad(set_to_none=True)

    try:
        with autocast("cuda", enabled=amp_enabled):
            out1 = model.forward_train(image1)
            out2 = model.forward_train(image2)
            validate_lightglue_contract(out1, descriptor_dim=int(cfg["descriptor_dim"]))
            validate_lightglue_contract(out2, descriptor_dim=int(cfg["descriptor_dim"]))
            loss, stats = loss_fn.forward_batch(
                out1,
                out2,
                homographies=homography,
                image2_hw=(image2.shape[-2], image2.shape[-1]),
                warp_fields=warp_field,
                warp_valids=warp_valid,
                depth_valid_1=depth_valid1,
            )
        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
    except RuntimeError as exc:
        if amp_enabled and is_amp_related_error(exc):
            cfg["mixed_precision"] = False
            log.warning("AMP preflight failed; forcing fp32. reason=%s", exc)
            optimizer.zero_grad(set_to_none=True)
            out1 = model.forward_train(image1)
            out2 = model.forward_train(image2)
            loss, stats = loss_fn.forward_batch(
                out1,
                out2,
                homographies=homography,
                image2_hw=(image2.shape[-2], image2.shape[-1]),
                warp_fields=warp_field,
                warp_valids=warp_valid,
                depth_valid_1=depth_valid1,
            )
            loss.backward()
        else:
            raise

    if not tensor_is_finite(loss) or not grads_are_finite(model.parameters()):
        raise RuntimeError("Preflight failed: found NaN/Inf in loss or gradients")

    log.info("Preflight forward/backward success | loss=%.4f sim_gap=%.4f", float(stats.get("loss", 0.0)), float(stats.get("sim_gap", 0.0)))


def train_step(
    cfg: Dict[str, Any],
    model: HybridModelV2,
    loss_fn: ScoreWeightedHingeRepeatabilityLoss,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    batch: Dict[str, Any],
    device: torch.device,
) -> Dict[str, float]:
    image1 = batch["image1"].to(device)
    image2 = batch["image2"].to(device)
    homography = batch["homography"].to(device)
    warp_field = batch.get("warp_field")
    warp_valid = batch.get("warp_valid")
    depth_valid1 = batch.get("depth_valid1")
    if isinstance(warp_field, torch.Tensor):
        warp_field = warp_field.to(device)
    if isinstance(warp_valid, torch.Tensor):
        warp_valid = warp_valid.to(device)
    if isinstance(depth_valid1, torch.Tensor):
        depth_valid1 = depth_valid1.to(device)

    optimizer.zero_grad(set_to_none=True)

    amp_enabled = bool(cfg.get("mixed_precision", True) and device.type == "cuda")

    def _forward(use_amp: bool):
        with autocast("cuda", enabled=use_amp):
            out1 = model.forward_train(image1)
            out2 = model.forward_train(image2)
            validate_lightglue_contract(out1, descriptor_dim=int(cfg["descriptor_dim"]))
            validate_lightglue_contract(out2, descriptor_dim=int(cfg["descriptor_dim"]))
            loss, stats = loss_fn.forward_batch(
                out1,
                out2,
                homographies=homography,
                image2_hw=(image2.shape[-2], image2.shape[-1]),
                warp_fields=warp_field,
                warp_valids=warp_valid,
                depth_valid_1=depth_valid1,
            )
        return loss, stats

    try:
        loss, stats = _forward(amp_enabled)
        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
    except RuntimeError as exc:
        if amp_enabled and is_amp_related_error(exc):
            cfg["mixed_precision"] = False
            log.warning("AMP runtime error; switching to fp32 for rest of run. reason=%s", exc)
            optimizer.zero_grad(set_to_none=True)
            loss, stats = _forward(False)
            loss.backward()
        else:
            raise

    if not tensor_is_finite(loss) or not grads_are_finite(model.parameters()):
        cfg["mixed_precision"] = False
        raise RuntimeError("Detected NaN/Inf in training step")

    nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("grad_clip_norm", 10.0)))

    if bool(cfg.get("mixed_precision", True)) and device.type == "cuda":
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    return stats


@torch.no_grad()
def validate(
    cfg: Dict[str, Any],
    model: HybridModelV2,
    loss_fn: ScoreWeightedHingeRepeatabilityLoss,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int = 20,
) -> Dict[str, float]:
    model.eval()
    all_stats: List[Dict[str, float]] = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        image1 = batch["image1"].to(device)
        image2 = batch["image2"].to(device)
        homography = batch["homography"].to(device)
        warp_field = batch.get("warp_field")
        warp_valid = batch.get("warp_valid")
        depth_valid1 = batch.get("depth_valid1")
        if isinstance(warp_field, torch.Tensor):
            warp_field = warp_field.to(device)
        if isinstance(warp_valid, torch.Tensor):
            warp_valid = warp_valid.to(device)
        if isinstance(depth_valid1, torch.Tensor):
            depth_valid1 = depth_valid1.to(device)

        out1 = model.forward_train(image1)
        out2 = model.forward_train(image2)
        _, stats = loss_fn.forward_batch(
            out1,
            out2,
            homographies=homography,
            image2_hw=(image2.shape[-2], image2.shape[-1]),
            warp_fields=warp_field,
            warp_valids=warp_valid,
            depth_valid_1=depth_valid1,
        )
        all_stats.append(stats)
    model.train()
    return mean_stats(all_stats)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = merge_config_with_args(load_yaml_config(args.config), args)

    is_colab = bool(os.environ.get("COLAB_GPU"))
    if is_colab and int(cfg.get("num_workers", 0)) > 0:
        log.warning("Colab detected; forcing num_workers=0 for stability")
        cfg["num_workers"] = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    model = build_model_v2(cfg, device)

    train_loader, val_loader = build_dataloaders_v2(cfg)

    loss_fn = ScoreWeightedHingeRepeatabilityLoss(
        positive_margin=float(cfg.get("positive_margin", 1.0)),
        negative_margin=float(cfg.get("negative_margin", 0.2)),
        lambda_d=float(cfg.get("lambda_d", 250.0)),
        lambda_rep=float(cfg.get("lambda_rep", 0.5)),
        correspondence_threshold=float(cfg.get("correspondence_threshold", 6.0)),
        safe_radius=float(cfg.get("safe_radius", 8.0)),
        balance_pos_neg=bool(cfg.get("balance_pos_neg", True)),
    )

    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg.get("lr", 1e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(cfg.get("lr_factor", 0.5)),
        patience=int(cfg.get("lr_patience", 3)),
        min_lr=float(cfg.get("min_lr", 1e-6)),
    )
    scaler = GradScaler("cuda", enabled=bool(cfg.get("mixed_precision", True) and device.type == "cuda"))

    tb_writer, wandb_run = setup_tracking(cfg)

    run_preflight(cfg, model, train_loader, loss_fn, optimizer, scaler, device)

    start_epoch = 0
    resume = cfg.get("resume")
    if resume:
        start_epoch = load_checkpoint(str(resume), model, optimizer, scaler, scheduler, device)

    best_score = float("-inf")
    bad_epochs = 0
    early_stop_pat = int(cfg.get("early_stop_patience", 12))
    global_step = 0

    for epoch in range(start_epoch, int(cfg.get("max_epochs", 40))):
        model.train()
        assert_superpoint_frozen(model)

        if cfg.get("unfreeze_at_epoch") is not None and epoch >= int(cfg["unfreeze_at_epoch"]):
            newly = model.unfreeze_xfeat_modules(cfg.get("unfreeze_keywords", []))
            added = _add_new_trainable_params(model, optimizer, float(cfg.get("lr", 1e-4)), float(cfg.get("weight_decay", 1e-4)))
            if newly > 0:
                log.info("Staged unfreeze at epoch %d: newly_unfrozen=%d optimizer_added=%d", epoch, newly, added)
                cfg["unfreeze_at_epoch"] = None

        train_stats_list: List[Dict[str, float]] = []
        for batch in train_loader:
            stats = train_step(cfg, model, loss_fn, optimizer, scaler, batch, device)
            train_stats_list.append(stats)
            global_step += 1

        train_stats = mean_stats(train_stats_list)
        val_stats = validate(cfg, model, loss_fn, val_loader, device, max_batches=int(cfg.get("val_max_batches", 20)))
        val_loss = float(val_stats.get("loss", float("inf")))
        scheduler.step(val_loss)

        # milestone check (can be disabled by setting thresholds outside [0,1])
        lo = float(cfg.get("preflight_plateau_threshold_min", 0.45))
        hi = float(cfg.get("preflight_plateau_threshold_max", 0.55))
        if 0.0 <= lo <= hi <= 2.0:
            try:
                check_plateau_break(val_stats, lo, hi)
            except RuntimeError as e:
                log.warning("Milestone check: %s", e)

        score = pick_model_score(str(cfg.get("model_selection_metric", "sim_gap")), val_stats, val_loss)
        is_best = score > best_score
        if is_best:
            best_score = score
            bad_epochs = 0
        else:
            bad_epochs += 1

        log.info(
            "Epoch %03d | train_loss=%.4f val_loss=%.4f sim_gap=%.4f repeatability=%.4f lr=%.2e %s",
            epoch,
            float(train_stats.get("loss", 0.0)),
            val_loss,
            float(val_stats.get("sim_gap", 0.0)),
            float(val_stats.get("repeatability_mean", 0.0)),
            optimizer.param_groups[0]["lr"],
            "[BEST]" if is_best else "",
        )

        log_metrics(metrics=train_stats, step=epoch, prefix="train", tb_writer=tb_writer, wandb_run=wandb_run)
        log_metrics(metrics=val_stats, step=epoch, prefix="val", tb_writer=tb_writer, wandb_run=wandb_run)

        save_checkpoint(
            output_dir=str(cfg.get("checkpoint_dir", "hybrid_model_v2/checkpoints")),
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            cfg=cfg,
            val_loss=val_loss,
            is_best=is_best,
        )

        if bad_epochs >= early_stop_pat:
            log.info("Early stopping triggered after %d bad epochs", bad_epochs)
            break

    if tb_writer is not None:
        tb_writer.close()
    if wandb_run is not None:
        wandb_run.finish()

    log.info("Training complete.")


if __name__ == "__main__":
    main()
