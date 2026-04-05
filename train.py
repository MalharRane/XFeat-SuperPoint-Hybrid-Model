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
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
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
    'num_keypoints':  2048,
    'nms_radius':     4,
    'descriptor_dim': 256,

    # Loss
    'positive_margin':          1.0,
    'negative_margin':          0.2,
    'lambda_d':                 250.0,
    'correspondence_threshold': 8.0,

    # Training
    'batch_size':       4,
    'num_workers':      4,
    'max_epochs':       50,
    'lr':               3e-4,
    'lr_decay_factor':  0.5,
    'lr_decay_epochs':  [20, 35, 45],
    'weight_decay':     1e-4,
    'grad_clip_norm':   10.0,
    'mixed_precision':  True,

    # Checkpointing
    'checkpoint_dir':  'checkpoints',
    'log_dir':         'runs',
    'save_every':      5,        # save checkpoint every N epochs
    'val_every':       1,        # validate every N epochs

    # Data
    'min_overlap':   0.15,
    'max_overlap':   0.70,
    'augment':       True,
}


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
    2. Compute HomographyHingeLoss across the pairs
    3. Backward + gradient clip + optimizer step
    """
    image1 = batch['image1'].to(device)        # (B, 1, H, W)
    image2 = batch['image2'].to(device)        # (B, 1, H, W)
    homographies = batch['homography'].to(device)  # (B, 3, 3)

    B = image1.shape[0]

    optimizer.zero_grad(set_to_none=True)

    with autocast(enabled=cfg.get('mixed_precision', True)):
        # Forward pass (returns intermediates for loss computation)
        out1 = model.forward_train(image1)
        out2 = model.forward_train(image2)

        # Compute hinge loss over the batch
        loss, stats = loss_fn.forward_batch(
            desc1_list=out1['descriptors'],     # list of (256, N) → transpose
            desc2_list=out2['descriptors'],
            kp1_list=out1['keypoints_px'],
            kp2_list=out2['keypoints_px'],
            homographies=homographies,
            image2_hws=[(image2.shape[2], image2.shape[3])] * B,
        )

    # Backward
    scaler.scale(loss).backward()

    # Gradient clipping (prevents exploding gradients with large λd)
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

        out1 = model.forward_train(image1)
        out2 = model.forward_train(image2)

        _, stats = loss_fn.forward_batch(
            desc1_list=out1['descriptors'],
            desc2_list=out2['descriptors'],
            kp1_list=out1['keypoints_px'],
            kp2_list=out2['keypoints_px'],
            homographies=homographies,
            image2_hws=[(image2.shape[2], image2.shape[3])] * B,
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
        correspondence_threshold=cfg['correspondence_threshold'],
    )

    # ── Optimizer + Scheduler ────────────────────────────────────────────
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg['lr'],
        weight_decay=cfg['weight_decay'],
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg['lr_decay_epochs'],
        gamma=cfg['lr_decay_factor'],
    )
    scaler = GradScaler(enabled=cfg.get('mixed_precision', True))

    # ── Data ────────────────────────────────────────────────────────────
    image_size = (cfg['image_height'], cfg['image_width'])
    train_loader = build_dataloader(
        mode=cfg['mode'], root=cfg['data_root'],
        image_size=image_size, batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'], shuffle=True,
        scene_info_dir=cfg.get('scene_info_dir'),
        min_overlap=cfg['min_overlap'], max_overlap=cfg['max_overlap'],
        augment=cfg['augment'],
    )

    # ── Tensorboard ─────────────────────────────────────────────────────
    writer = SummaryWriter(log_dir=cfg['log_dir'])

    # ── Resume ──────────────────────────────────────────────────────────
    start_epoch = 0
    if resume:
        start_epoch = load_checkpoint(resume, model, optimizer, scaler, device)

    # ── Training Loop ───────────────────────────────────────────────────
    model.train()
    best_loss = float('inf')
    global_step = 0

    for epoch in range(start_epoch, cfg['max_epochs']):
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
                    f"pos_sim={stats.get('pos_sim_mean', 0):.3f}  "
                    f"neg_sim={stats.get('neg_sim_mean', 0):.3f}  "
                    f"lr={lr:.2e}"
                )
                for k, v in stats.items():
                    writer.add_scalar(f'train_step/{k}', v, global_step)

        # Epoch averages
        n_batches = len(train_loader)
        avg_loss = epoch_stats.get('loss', 0.0) / n_batches
        elapsed  = time.time() - t0
        log.info(
            f"\n── Epoch {epoch:03d}  avg_loss={avg_loss:.4f}  "
            f"time={elapsed:.1f}s ──\n"
        )
        for k, v in epoch_stats.items():
            writer.add_scalar(f'train_epoch/{k}', v / n_batches, epoch)

        # Scheduler step
        scheduler.step()

        # Checkpoint
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss

        if (epoch + 1) % cfg['save_every'] == 0 or is_best:
            save_checkpoint(
                model, optimizer, scaler, epoch, avg_loss, cfg, is_best
            )

    writer.close()
    log.info("Training complete.")


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
    p.add_argument('--num_keypoints',type=int, help='Max keypoints per image')
    p.add_argument('--checkpoint_dir', type=str, help='Checkpoint directory')
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

    train(cfg, resume=args.resume)
