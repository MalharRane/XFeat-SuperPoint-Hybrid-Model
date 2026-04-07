"""
train.py
========
Training entry point for the XFeat-SuperPoint Hybrid Model.

Usage
-----
  # Synthetic pre-training (recommended first step)
  python train.py --mode synthetic --data_root /path/to/coco/images

  # MegaDepth fine-tuning
  python train.py --mode megadepth --data_root /path/to/megadepth

  # Resume from checkpoint
  python train.py --mode megadepth --data_root /path/to/megadepth \
                  --resume checkpoints/best.pth

  # Full config override
  python train.py --config config.yaml

See config.yaml for all available hyperparameters.
"""

import os
import sys
import time
import argparse
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

# ---------------------------------------------------------------------------
# Local imports — these assume XFeatCore and SuperPointCore are importable.
# See README.md for instructions on linking the upstream repositories.
# ---------------------------------------------------------------------------
try:
    from models.hybrid_model import HybridModel
    from losses.hinge_loss import HomographyHingeLoss
    from data.megadepth_dataset import build_dataloader
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    print("Make sure you have installed all requirements and linked the "
          "upstream XFeat / SuperPoint repositories.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('train')


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    # Training mode
    'mode':           'synthetic',     # 'synthetic' | 'megadepth'
    'data_root':      './data/images', # path to images or megadepth root
    'scene_info_dir': None,            # megadepth scene_info directory

    # Image
    'image_height':   480,
    'image_width':    640,

    # Model
    'num_keypoints':  1024,
    'nms_radius':     4,
    'min_keypoint_score': 0.01,
    'descriptor_dim': 256,

    # Loss
    'positive_margin':          1.0,
    'negative_margin':          0.2,
    'lambda_d':                 250.0,
    'lambda_rep':               0.5,
    'correspondence_threshold': 6.0,
    'safe_radius':              8.0,
    'balance_pos_neg':          True,

    # Training
    'batch_size':       4,
    'num_workers':      4,
    'max_epochs':       50,
    'lr':               1e-4,
    'lr_patience':      3,       # ReduceLROnPlateau patience
    'lr_factor':        0.5,     # ReduceLROnPlateau factor
    'min_lr':           1e-6,    # minimum learning rate
    'lr_threshold':     1e-4,    # ReduceLROnPlateau threshold
    'lr_cooldown':      0,       # ReduceLROnPlateau cooldown
    'weight_decay':     1e-4,
    'grad_clip_norm':   10.0,
    'mixed_precision':  True,
    'early_stop_patience': 10,
    'model_selection_metric': 'loss',  # 'loss' | 'sim_gap' | 'repeatability'

    # Checkpointing
    'checkpoint_dir':  'checkpoints',
    'log_dir':         'runs',
    'save_every':      5,
    'val_every':       1,

    # Data
    'min_overlap':   0.15,
    'max_overlap':   0.70,
    'max_pairs_per_scene': 1000,
    'augment':       True,
    'val_max_batches': 50,

    # Optional staged capacity expansion
    'unfreeze_at_epoch': None,
    'unfreeze_keywords': [],

    # Optional 2-stage schedule (synthetic -> megadepth)
    'two_stage': False,
    'synthetic_data_root': None,
    'megadepth_data_root': None,
    'stage1_epochs': 30,
    'stage2_epochs': 50,
    'stage1_checkpoint_subdir': 'stage1_synthetic',
    'stage2_checkpoint_subdir': 'stage2_megadepth',
    'stage1_log_subdir': 'stage1_synthetic',
    'stage2_log_subdir': 'stage2_megadepth',
}


def _to_optional_tensor_batch(
    value: object,
    device: torch.device,
) -> Optional[Union[torch.Tensor, List[Optional[torch.Tensor]]]]:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        moved: List[Optional[torch.Tensor]] = []
        for v in value:
            if isinstance(v, torch.Tensor):
                moved.append(v.to(device))
            else:
                moved.append(None)
        return moved
    return None


def _normalize_keywords(value: Any) -> Tuple[str, ...]:
    if value is None:
        return tuple()
    if isinstance(value, str):
        return tuple(v.strip() for v in value.split(',') if v.strip())
    if isinstance(value, Sequence):
        normalized_keywords: List[str] = []
        for v in value:
            if v is None:
                continue
            s = str(v).strip()
            if s:
                normalized_keywords.append(s)
        return tuple(normalized_keywords)
    return tuple()


def _add_new_trainable_params_to_optimizer(
    model: nn.Module,
    optimizer: optim.Optimizer,
    lr: float,
    weight_decay: float,
) -> int:
    existing = {id(p) for group in optimizer.param_groups for p in group['params']}
    new_params = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in existing
    ]
    if not new_params:
        return 0

    optimizer.add_param_group({
        'params': new_params,
        'lr': lr,
        'weight_decay': weight_decay,
    })
    return sum(p.numel() for p in new_params)


def _pick_model_score(
    metric: str,
    val_stats: Dict[str, float],
    val_loss: float,
) -> float:
    metric = str(metric or 'loss').lower()
    if metric == 'sim_gap':
        return val_stats.get('sim_gap', float('-inf'))
    if metric == 'repeatability':
        return val_stats.get('repeatability_mean', float('-inf'))
    return -val_loss


def load_config(config_path: Optional[str], cli_args: argparse.Namespace) -> Dict:
    """Merge default config with YAML file and CLI overrides."""
    cfg = DEFAULT_CONFIG.copy()

    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            file_cfg = yaml.safe_load(f)
        cfg.update(file_cfg)
        log.info(f"Loaded config from {config_path}")

    # CLI overrides (non-None values)
    for key, val in vars(cli_args).items():
        if val is not None and key in cfg:
            cfg[key] = val

    cfg['unfreeze_keywords'] = list(_normalize_keywords(cfg.get('unfreeze_keywords')))

    return cfg


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(cfg: Dict, device: torch.device) -> HybridModel:
    """
    Instantiate XFeatCore and SuperPointCore, then wrap in HybridModel.

    NOTE: You must adapt this function to import your specific implementations
    of XFeatCore and SuperPointCore.  Refer to:
        XFeat:      https://github.com/verlab/accelerated_features
        SuperPoint: https://github.com/rpautrat/SuperPoint
    """
    # ── XFeat ───────────────────────────────────────────────────────────
    try:
        # Option A: official XFeat release
        from modules.xfeat import XFeat as XFeatCore
        xfeat = XFeatCore()
    except ImportError:
        try:
            # Option B: local wrapper
            from models.xfeat_core import XFeatCore
            xfeat = XFeatCore()
        except ImportError:
            log.error(
                "Could not import XFeatCore. "
                "Clone https://github.com/verlab/accelerated_features "
                "and add it to PYTHONPATH."
            )
            raise

    # ── SuperPoint ──────────────────────────────────────────────────────
    try:
        from models.superpoint_core import SuperPointCore
        superpoint = SuperPointCore()
    except ImportError:
        try:
            # Try rpautrat implementation
            from superpoint import SuperPoint as SuperPointCore
            superpoint = SuperPointCore({})
        except ImportError:
            log.error(
                "Could not import SuperPointCore. "
                "Clone https://github.com/rpautrat/SuperPoint "
                "and add it to PYTHONPATH."
            )
            raise

    # Load pretrained SuperPoint weights if available
    sp_weights = Path('weights/superpoint_v1.pth')
    if sp_weights.exists():
        state = torch.load(str(sp_weights), map_location='cpu')
        if 'model' in state:
            state = state['model']
        superpoint.load_state_dict(state, strict=False)
        log.info(f"Loaded SuperPoint weights from {sp_weights}")

    # Load pretrained XFeat weights if available
    xf_weights = Path('weights/xfeat.pth')
    if xf_weights.exists():
        state = torch.load(str(xf_weights), map_location='cpu')
        if 'model' in state:
            state = state['model']
        xfeat.load_state_dict(state, strict=False)
        log.info(f"Loaded XFeat weights from {xf_weights}")

    # ── Wrap in HybridModel ──────────────────────────────────────────────
    model = HybridModel(
        xfeat_core=xfeat,
        superpoint_core=superpoint,
        num_keypoints=cfg['num_keypoints'],
        nms_radius=cfg['nms_radius'],
        min_keypoint_score=cfg['min_keypoint_score'],
        descriptor_dim=cfg['descriptor_dim'],
    ).to(device)

    return model


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_step(
    model:      HybridModel,
    batch:      Dict[str, torch.Tensor],
    loss_fn:    HomographyHingeLoss,
    optimizer:  optim.Optimizer,
    scaler:     GradScaler,
    cfg:        Dict,
    device:     torch.device,
) -> Dict[str, float]:
    """
    One gradient update step on a batch of image pairs.

    Pipeline
    --------
    1. Forward image1 and image2 through HybridModel (training mode)
    2. Compute score-weighted HomographyHingeLoss (+ repeatability reward)
    3. Backward + gradient clip + optimizer step

    Gradient path
    -------------
    loss → score weights W[i,j] = score1[i]*score2[j]
         → score values (differentiable via boolean-mask indexing on sm)
         → sm = f(heatmap logits)
         → XFeat kp_head parameters  ← updated by Adam
    """
    image1 = batch['image1'].to(device)
    image2 = batch['image2'].to(device)
    homographies = batch['homography'].to(device)
    B = image1.shape[0]

    # Optional depth-based warp fields (more accurate than planar H)
    warp_fields = _to_optional_tensor_batch(batch.get('warp_field'), device)
    warp_valids = _to_optional_tensor_batch(batch.get('warp_valid'), device)

    optimizer.zero_grad(set_to_none=True)

    with autocast('cuda', enabled=cfg.get('mixed_precision', True)):
        out1 = model.forward_train(image1)
        out2 = model.forward_train(image2)

        loss, stats = loss_fn.forward_batch(
            desc1_list=out1['descriptors'],      # (N, 256) — correct shape
            desc2_list=out2['descriptors'],
            kp1_list=out1['keypoints_px'],
            kp2_list=out2['keypoints_px'],
            homographies=homographies,
            image2_hws=[(image2.shape[2], image2.shape[3])] * B,
            scores1_list=out1.get('scores'),     # enables gradient to kp_head
            scores2_list=out2.get('scores'),
            warp_fields=warp_fields,
            warp_valids=warp_valids,
        )

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=cfg.get('grad_clip_norm', 10.0)
    )
    scaler.step(optimizer)
    scaler.update()

    return stats


# ---------------------------------------------------------------------------
# Validation step
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model:   HybridModel,
    loader:  torch.utils.data.DataLoader,
    loss_fn: HomographyHingeLoss,
    device:  torch.device,
    max_batches: int = 50,
) -> Dict[str, float]:
    """Run validation on a subset of the validation loader."""
    model.eval()
    agg: Dict[str, float] = {}
    count = 0

    for batch in loader:
        if count >= max_batches:
            break

        image1 = batch['image1'].to(device)
        image2 = batch['image2'].to(device)
        homographies = batch['homography'].to(device)
        B = image1.shape[0]

        warp_fields = _to_optional_tensor_batch(batch.get('warp_field'), device)
        warp_valids = _to_optional_tensor_batch(batch.get('warp_valid'), device)

        out1 = model.forward_train(image1)
        out2 = model.forward_train(image2)

        _, stats = loss_fn.forward_batch(
            desc1_list=out1['descriptors'],
            desc2_list=out2['descriptors'],
            kp1_list=out1['keypoints_px'],
            kp2_list=out2['keypoints_px'],
            homographies=homographies,
            image2_hws=[(image2.shape[2], image2.shape[3])] * B,
            scores1_list=out1.get('scores'),
            scores2_list=out2.get('scores'),
            warp_fields=warp_fields,
            warp_valids=warp_valids,
        )

        for k, v in stats.items():
            agg[k] = agg.get(k, 0.0) + v
        count += 1

    model.train()
    return {k: v / max(count, 1) for k, v in agg.items()}


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    model:     HybridModel,
    optimizer: optim.Optimizer,
    scaler:    GradScaler,
    epoch:     int,
    loss:      float,
    cfg:       Dict,
    is_best:   bool = False,
) -> None:
    ckpt_dir = Path(cfg['checkpoint_dir'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    state = {
        'epoch':       epoch,
        'loss':        loss,
        'config':      cfg,
        'model':       model.state_dict(),
        'optimizer':   optimizer.state_dict(),
        'scaler':      scaler.state_dict(),
    }

    path = ckpt_dir / f'epoch_{epoch:04d}.pth'
    torch.save(state, str(path))
    log.info(f"  Saved checkpoint → {path}")

    if is_best:
        best_path = ckpt_dir / 'best.pth'
        torch.save(state, str(best_path))
        log.info(f"  ★ New best → {best_path}")


def load_checkpoint(
    path:      str,
    model:     HybridModel,
    optimizer: optim.Optimizer,
    scaler:    GradScaler,
    device:    torch.device,
) -> int:
    """Load checkpoint, return start epoch."""
    state = torch.load(path, map_location=device)
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    scaler.load_state_dict(state['scaler'])
    epoch = state['epoch'] + 1
    log.info(f"Resumed from {path}  (epoch {epoch})")
    return epoch


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: Dict, resume: Optional[str] = None) -> None:
    # ── Device ──────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Device: {device}")

    # ── Model ───────────────────────────────────────────────────────────
    model = build_model(cfg, device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Trainable parameters: {trainable:,}")

    # ── Loss ────────────────────────────────────────────────────────────
    loss_fn = HomographyHingeLoss(
        positive_margin=cfg['positive_margin'],
        negative_margin=cfg['negative_margin'],
        lambda_d=cfg['lambda_d'],
        lambda_rep=cfg.get('lambda_rep', 0.5),
        correspondence_threshold=cfg['correspondence_threshold'],
        safe_radius=cfg.get('safe_radius', 8.0),
        balance_pos_neg=cfg.get('balance_pos_neg', True),
    )

    # ── Optimizer + Scheduler ────────────────────────────────────────────
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg['lr'],
        weight_decay=cfg['weight_decay'],
    )
    # ReduceLROnPlateau adjusts lr based on val loss — more adaptive than
    # a fixed milestone schedule and works well when early stopping is used.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=cfg.get('lr_factor', 0.5),
        patience=cfg.get('lr_patience', 5),
        threshold=cfg.get('lr_threshold', 1e-4),
        cooldown=cfg.get('lr_cooldown', 0),
        min_lr=cfg.get('min_lr', 1e-6),
    )
    scaler = GradScaler('cuda', enabled=cfg.get('mixed_precision', True))

    # ── Data ────────────────────────────────────────────────────────────
    image_size = (cfg['image_height'], cfg['image_width'])
    train_loader = build_dataloader(
        mode=cfg['mode'], root=cfg['data_root'],
        image_size=image_size, batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'], shuffle=True,
        scene_info_dir=cfg.get('scene_info_dir'),
        min_overlap=cfg['min_overlap'], max_overlap=cfg['max_overlap'],
        max_pairs_per_scene=cfg.get('max_pairs_per_scene', 1000),
        augment=cfg['augment'],
    )
    val_loader = build_dataloader(
        mode=cfg['mode'], root=cfg['data_root'],
        image_size=image_size, batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'], shuffle=False,
        scene_info_dir=cfg.get('scene_info_dir'),
        min_overlap=cfg['min_overlap'], max_overlap=cfg['max_overlap'],
        max_pairs_per_scene=cfg.get('max_pairs_per_scene', 1000),
        augment=False,
    )

    # ── Tensorboard ─────────────────────────────────────────────────────
    writer = SummaryWriter(log_dir=cfg['log_dir'])

    # ── Resume ──────────────────────────────────────────────────────────
    start_epoch = 0
    if resume:
        start_epoch = load_checkpoint(resume, model, optimizer, scaler, device)

    # ── Training Loop ───────────────────────────────────────────────────
    model.train()
    best_val_loss = float('inf')
    best_checkpoint_val_loss = float('inf')
    best_score = float('-inf')
    global_step = 0
    bad_epochs = 0
    early_stop_patience = cfg.get('early_stop_patience', 10)
    selection_metric = str(cfg.get('model_selection_metric', 'loss')).lower()
    unfreeze_epoch = cfg.get('unfreeze_at_epoch')
    unfreeze_done = False
    unfreeze_keywords = tuple(_normalize_keywords(cfg.get('unfreeze_keywords')))

    for epoch in range(start_epoch, cfg['max_epochs']):
        if (
            not unfreeze_done
            and unfreeze_epoch is not None
            and epoch >= int(unfreeze_epoch)
            and unfreeze_keywords
        ):
            newly_unfrozen = model.unfreeze_xfeat_modules(unfreeze_keywords)
            newly_added = _add_new_trainable_params_to_optimizer(
                model=model,
                optimizer=optimizer,
                lr=cfg['lr'],
                weight_decay=cfg['weight_decay'],
            )
            if newly_unfrozen > 0:
                log.info(
                    f"Scheduled unfreeze activated at epoch {epoch}: "
                    f"{newly_unfrozen:,} params unfrozen, "
                    f"{newly_added:,} params added to optimizer."
                )
            unfreeze_done = True

        epoch_stats: Dict[str, float] = {}
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            stats = train_step(
                model, batch, loss_fn, optimizer, scaler, cfg, device
            )
            global_step += 1

            # Accumulate epoch stats
            for k, v in stats.items():
                epoch_stats[k] = epoch_stats.get(k, 0.0) + v

            # Logging (every 50 steps)
            if batch_idx % 50 == 0:
                lr = optimizer.param_groups[0]['lr']
                log.info(
                    f"E{epoch:03d} [{batch_idx:04d}/{len(train_loader):04d}]  "
                    f"loss={stats.get('loss', 0):.4f}  "
                    f"hinge={stats.get('hinge', 0):.4f}  "
                    f"pos_sim={stats.get('pos_sim_mean', 0):.3f}  "
                    f"neg_sim={stats.get('neg_sim_mean', 0):.3f}  "
                    f"lr={lr:.2e}"
                )
                for k, v in stats.items():
                    writer.add_scalar(f'train_step/{k}', v, global_step)

        # Epoch averages
        n_batches = max(len(train_loader), 1)
        avg_train_loss = epoch_stats.get('loss', 0.0) / n_batches
        elapsed = time.time() - t0

        for k, v in epoch_stats.items():
            writer.add_scalar(f'train_epoch/{k}', v / n_batches, epoch)

        # Validation
        val_stats: Dict[str, float] = {}
        if (epoch + 1) % cfg.get('val_every', 1) == 0:
            val_stats = validate(
                model,
                val_loader,
                loss_fn,
                device,
                max_batches=cfg.get('val_max_batches', 50),
            )
            val_loss = val_stats.get('loss', float('inf'))
            for k, v in val_stats.items():
                writer.add_scalar(f'val/{k}', v, epoch)
        else:
            val_loss = avg_train_loss  # fallback when not validating

        # ReduceLROnPlateau — step on val loss
        scheduler.step(val_loss)
        best_val_loss = min(best_val_loss, val_loss)

        score = _pick_model_score(selection_metric, val_stats, val_loss)
        is_best = score > best_score
        if is_best:
            best_score = score
            best_checkpoint_val_loss = val_loss
            bad_epochs = 0
        else:
            bad_epochs += 1

        lr = optimizer.param_groups[0]['lr']
        log.info(
            f"\n── Epoch {epoch:03d}  "
            f"train_loss={avg_train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"pos_sim={val_stats.get('pos_sim_mean', 0):.3f}  "
            f"neg_sim={val_stats.get('neg_sim_mean', 0):.3f}  "
            f"sim_gap={val_stats.get('sim_gap', 0):.3f}  "
            f"repeatability={val_stats.get('repeatability_mean', 0):.3f}  "
            f"bad={bad_epochs}/{early_stop_patience}  "
            f"lr={lr:.2e}  {elapsed:.1f}s "
            f"{'★ BEST' if is_best else ''} ──\n"
        )

        if (epoch + 1) % cfg['save_every'] == 0 or is_best:
            save_checkpoint(
                model, optimizer, scaler, epoch, val_loss, cfg, is_best
            )

        # Early stopping
        if bad_epochs >= early_stop_patience:
            log.info(
                f"Early stopping at epoch {epoch} "
                f"(no val improvement for {early_stop_patience} epochs)."
            )
            break

    writer.close()
    score_label = (
        "best_sim_gap(max)"
        if selection_metric == 'sim_gap'
        else "best_repeatability(max)"
        if selection_metric == 'repeatability'
        else "best_loss(min)"
    )
    log.info(
        "Training complete. "
        f"Best val loss overall: {best_val_loss:.4f}  "
        f"Best-checkpoint val loss: {best_checkpoint_val_loss:.4f}  "
        f"{score_label}={best_score:.4f}"
    )


def train_two_stage(cfg: Dict) -> None:
    """
    Run synthetic pre-training then MegaDepth fine-tuning.
    """
    root_ckpt = Path(cfg['checkpoint_dir'])
    root_log = Path(cfg['log_dir'])

    stage1_cfg = cfg.copy()
    stage1_cfg.update({
        'mode': 'synthetic',
        'data_root': cfg.get('synthetic_data_root') or cfg['data_root'],
        'max_epochs': int(cfg.get('stage1_epochs', 30)),
        'checkpoint_dir': str(root_ckpt / cfg.get('stage1_checkpoint_subdir', 'stage1_synthetic')),
        'log_dir': str(root_log / cfg.get('stage1_log_subdir', 'stage1_synthetic')),
    })

    stage2_cfg = cfg.copy()
    stage2_cfg.update({
        'mode': 'megadepth',
        'data_root': cfg.get('megadepth_data_root') or cfg['data_root'],
        'max_epochs': int(cfg.get('stage2_epochs', 50)),
        'checkpoint_dir': str(root_ckpt / cfg.get('stage2_checkpoint_subdir', 'stage2_megadepth')),
        'log_dir': str(root_log / cfg.get('stage2_log_subdir', 'stage2_megadepth')),
    })

    Path(stage1_cfg['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    Path(stage1_cfg['log_dir']).mkdir(parents=True, exist_ok=True)
    Path(stage2_cfg['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    Path(stage2_cfg['log_dir']).mkdir(parents=True, exist_ok=True)

    log.info("Starting stage 1/2: synthetic pre-training")
    train(stage1_cfg, resume=None)

    stage1_best = Path(stage1_cfg['checkpoint_dir']) / 'best.pth'
    if not stage1_best.exists():
        raise FileNotFoundError(
            f"Stage-1 checkpoint not found: {stage1_best}. "
            "Ensure stage-1 completed and produced a best checkpoint."
        )

    log.info("Starting stage 2/2: MegaDepth fine-tuning from stage-1 best")
    train(stage2_cfg, resume=str(stage1_best))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Train XFeat-SuperPoint Hybrid')
    p.add_argument('--config',       type=str, default=None,
                   help='Path to YAML config file')
    p.add_argument('--mode',         type=str, choices=['synthetic', 'megadepth'],
                   help='Training data mode')
    p.add_argument('--data_root',    type=str, help='Path to training data')
    p.add_argument('--batch_size',   type=int, help='Batch size')
    p.add_argument('--max_epochs',   type=int, help='Number of epochs')
    p.add_argument('--lr',           type=float, help='Learning rate')
    p.add_argument('--lr_patience',  type=int, help='ReduceLROnPlateau patience')
    p.add_argument('--lr_factor',    type=float, help='ReduceLROnPlateau factor')
    p.add_argument('--min_lr',       type=float, help='Minimum learning rate')
    p.add_argument('--early_stop_patience', type=int, help='Early stopping patience')
    p.add_argument('--num_keypoints',type=int, help='Max keypoints per image')
    p.add_argument('--min_keypoint_score', type=float,
                   help='Drop keypoints below this score before top-K')
    p.add_argument('--checkpoint_dir', type=str, help='Checkpoint directory')
    p.add_argument('--log_dir',      type=str, help='TensorBoard log directory')
    p.add_argument('--lambda_d',     type=float, help='Positive hinge weight')
    p.add_argument('--lambda_rep',   type=float, help='Repeatability reward weight')
    p.add_argument('--correspondence_threshold', type=float,
                   help='Positive correspondence threshold in px')
    p.add_argument('--max_pairs_per_scene', type=int,
                   help='Max MegaDepth pairs sampled per scene')
    p.add_argument('--model_selection_metric', type=str,
                   choices=['loss', 'sim_gap', 'repeatability'],
                   help='Checkpoint selection metric')
    p.add_argument('--val_max_batches', type=int,
                   help='Validation batches per epoch')
    p.add_argument('--unfreeze_at_epoch', type=int,
                   help='Epoch to unfreeze extra XFeat params')
    p.add_argument('--unfreeze_keywords', type=str,
                   help='Comma-separated XFeat parameter-name keywords to unfreeze')
    p.add_argument('--two_stage', action='store_true', default=None,
                   help='Run synthetic pre-training then MegaDepth fine-tuning')
    p.add_argument('--synthetic_data_root', type=str,
                   help='Stage-1 synthetic data root')
    p.add_argument('--megadepth_data_root', type=str,
                   help='Stage-2 MegaDepth data root')
    p.add_argument('--stage1_epochs', type=int, help='Stage-1 epochs')
    p.add_argument('--stage2_epochs', type=int, help='Stage-2 epochs')
    p.add_argument('--resume',       type=str, default=None,
                   help='Checkpoint path to resume from')
    p.add_argument('--no_amp',       action='store_true',
                   help='Disable automatic mixed precision')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg  = load_config(args.config, args)

    if args.no_amp:
        cfg['mixed_precision'] = False

    log.info("Config:\n" + yaml.dump(cfg, default_flow_style=False))

    Path(cfg['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    Path(cfg['log_dir']).mkdir(parents=True, exist_ok=True)

    if cfg.get('two_stage', False):
        train_two_stage(cfg)
    else:
        train(cfg, resume=args.resume)
