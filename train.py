"""
train.py
========
Training entry point for the XFeat-SuperPoint Hybrid Model.

Usage
-----
  # Synthetic pre-training (recommended first step)
  python train.py --mode synthetic --data_root /path/to/coco/images

  # MegaDepth fine-tuning (.npz scene_info)
  python train.py --mode megadepth --data_root /path/to/megadepth

  # MegaDepth raw-scene fine-tuning (no .npz required)
  python train.py --mode megadepth_raw --data_root /path/to/megadepth

  # Resume from checkpoint
  python train.py --mode megadepth_raw --data_root /path/to/megadepth \
                  --resume checkpoints/best.pth

  # Full config override
  python train.py --config config.yaml

See config.yaml for all available hyperparameters.
"""

import os
import sys
import time
import argparse
import tempfile
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast

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
    'mode':           'synthetic',     # 'synthetic' | 'megadepth' | 'megadepth_raw'
    'data_root':      './data/images', # path to images or megadepth root
    'scene_info_dir': None,            # megadepth-only scene_info directory
    'train_split': 'train',            # megadepth* split for training loader
    'val_split': 'val',                # megadepth* split for validation/eval loader
    'megadepth_val_split_ratio': 0.2,  # used only when no train/val scene_info subdirs exist
    'verify_dataset_pairs': True,      # preflight-check image/depth path existence
    'seed': 42,

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
    'num_workers':      0,
    'dataloader_timeout_s': 0,
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

    # Optional 2-stage schedule (synthetic -> megadepth_raw by default)
    'two_stage': False,
    'synthetic_data_root': None,
    'megadepth_data_root': None,
    'stage2_mode': 'megadepth_raw',  # 'megadepth_raw' (default) or 'megadepth'
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
# Safety / compatibility helpers
# ---------------------------------------------------------------------------

def _runtime_error_with_stage(stage: str, exc: Exception) -> RuntimeError:
    return RuntimeError(
        f"\n========== Failure stage: {stage} ==========\n"
        f"{type(exc).__name__}: {exc}\n"
    )


def _is_amp_runtime_error(exc: RuntimeError) -> bool:
    msg = str(exc).lower()
    has_amp_context = any(s in msg for s in ('amp', 'autocast', 'gradscaler', 'fp16', 'bf16'))
    has_dtype_context = any(s in msg for s in ('float16', 'bfloat16', 'half', 'dtype'))
    has_cuda_context = 'cuda' in msg
    return (has_amp_context and (has_dtype_context or has_cuda_context)) or (
        has_dtype_context and 'autocast' in msg
    )


def _validate_image_size(cfg: Dict) -> None:
    h = int(cfg['image_height'])
    w = int(cfg['image_width'])
    if h % 8 != 0 or w % 8 != 0:
        raise ValueError(
            f"Invalid image size ({h}, {w}): both image_height and image_width must be divisible by 8."
        )


def _set_reproducible_seed(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _summarize_state_dict_compat(module: nn.Module, state: Dict[str, Any]) -> Tuple[int, int, List[str], List[str]]:
    current_keys = set(module.state_dict().keys())
    incoming_keys = set(state.keys())
    common = sorted(current_keys.intersection(incoming_keys))
    missing = sorted(current_keys - incoming_keys)
    unexpected = sorted(incoming_keys - current_keys)
    return len(common), len(current_keys), missing, unexpected


def _load_checkpoint_payload(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    if path.stat().st_size <= 0:
        raise RuntimeError(f"Checkpoint file is empty: {path}")
    payload = torch.load(str(path), map_location='cpu')
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Invalid checkpoint format for {path}: expected dict payload, got {type(payload).__name__}."
        )
    return payload


def _atomic_torch_save(state: Dict[str, Any], output_path: Path, prefix: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=prefix, suffix='.pth', dir=str(output_path.parent), delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        torch.save(state, str(tmp_path))
        os.replace(tmp_path, output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _validate_and_load_pretrained_weights(
    module: nn.Module,
    weight_path: Path,
    module_name: str,
    strict: bool = False,
) -> None:
    if not weight_path.exists():
        raise FileNotFoundError(
            f"{module_name} weights not found: {weight_path}. "
            "Download the expected checkpoint before training."
        )
    if weight_path.stat().st_size <= 0:
        raise RuntimeError(f"{module_name} weights file is empty: {weight_path}")

    payload = _load_checkpoint_payload(weight_path)
    state = payload.get('model', payload)
    if not isinstance(state, dict):
        raise RuntimeError(
            f"{module_name} checkpoint payload is invalid: expected state_dict dict, got {type(state).__name__}."
        )

    common, total, missing, unexpected = _summarize_state_dict_compat(module, state)
    if common == 0:
        raise RuntimeError(
            f"{module_name} checkpoint format mismatch for {weight_path}. "
            "No overlapping parameter keys with model architecture."
        )

    missing_keys: List[str] = []
    unexpected_keys: List[str] = []
    if strict:
        module.load_state_dict(state, strict=True)
    else:
        result = module.load_state_dict(state, strict=False)
        missing_keys = list(result.missing_keys)
        unexpected_keys = list(result.unexpected_keys)

    policy = 'strict' if strict else 'non-strict'
    log.info(
        f"{module_name} weights loaded ({policy}) from {weight_path} | "
        f"matching_keys={common}/{total} missing={len(missing_keys or missing)} unexpected={len(unexpected_keys or unexpected)}"
    )
    if missing_keys:
        log.warning(f"{module_name} missing keys (sample): {missing_keys[:8]}")
    if unexpected_keys:
        log.warning(f"{module_name} unexpected keys (sample): {unexpected_keys[:8]}")


def _validate_forward_output_keys_shapes(
    out: Dict[str, Any],
    batch_size: int,
) -> None:
    required = ('keypoints', 'descriptors', 'keypoints_px', 'scores')
    missing = [k for k in required if k not in out]
    if missing:
        raise RuntimeError(f"forward_train output missing required keys: {missing}")

    for key in required:
        value = out[key]
        if not isinstance(value, list):
            raise RuntimeError(f"forward_train['{key}'] must be list, got {type(value).__name__}")
        if len(value) != batch_size:
            raise RuntimeError(
                f"forward_train['{key}'] batch length mismatch: expected {batch_size}, got {len(value)}"
            )

    for b, (kp, desc, kp_px, sc) in enumerate(zip(out['keypoints'], out['descriptors'], out['keypoints_px'], out['scores'])):
        if not isinstance(kp, torch.Tensor) or kp.dim() != 2 or kp.shape[-1] != 2:
            raise RuntimeError(f"Sample {b}: keypoints must be (N,2) tensor.")
        if not isinstance(desc, torch.Tensor) or desc.dim() != 2:
            raise RuntimeError(f"Sample {b}: descriptors must be (N,D) tensor.")
        if not isinstance(kp_px, torch.Tensor) or kp_px.dim() != 2 or kp_px.shape[-1] != 2:
            raise RuntimeError(f"Sample {b}: keypoints_px must be (N,2) tensor.")
        if not isinstance(sc, torch.Tensor) or sc.dim() != 1:
            raise RuntimeError(f"Sample {b}: scores must be (N,) tensor.")
        if kp.shape[0] != desc.shape[0] or kp.shape[0] != kp_px.shape[0] or kp.shape[0] != sc.shape[0]:
            raise RuntimeError(f"Sample {b}: inconsistent N across keypoints/descriptors/keypoints_px/scores.")
        if desc.dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
            raise RuntimeError(f"Sample {b}: descriptors dtype must be floating, got {desc.dtype}")


def _run_dummy_forward_preflight(model: HybridModel, cfg: Dict, device: torch.device) -> None:
    bsz = max(1, min(2, int(cfg.get('batch_size', 1))))
    h = int(cfg['image_height'])
    w = int(cfg['image_width'])
    dummy = torch.rand((bsz, 1, h, w), device=device, dtype=torch.float32)
    with torch.no_grad():
        out = model.forward_train(dummy)
    _validate_forward_output_keys_shapes(out, bsz)
    adapter = out.get('xfeat_adapter_path', 'unknown')
    log.info(f"Forward preflight OK | xfeat_adapter={adapter} | batch={bsz} shape=({h},{w})")


def _validate_resume_checkpoint_payload(
    payload: Dict[str, Any],
    path: str,
) -> None:
    if 'model' not in payload or not isinstance(payload['model'], dict):
        raise RuntimeError(
            f"Invalid resume checkpoint {path}: missing 'model' state dict. "
            "Suggested recovery: start fresh or provide a valid training checkpoint."
        )
    if 'optimizer' not in payload or not isinstance(payload['optimizer'], dict):
        raise RuntimeError(
            f"Invalid resume checkpoint {path}: missing 'optimizer' state. "
            "Suggested recovery: start fresh or use a checkpoint saved by this trainer."
        )
    if 'epoch' not in payload:
        raise RuntimeError(
            f"Invalid resume checkpoint {path}: missing 'epoch' metadata. "
            "Suggested recovery: start fresh or use a complete checkpoint."
        )


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
    def _instantiate_superpoint(cls):
        # Handle constructor variants across SuperPoint forks/wrappers.
        try:
            return cls()
        except TypeError as e:
            msg = str(e)
            if "required positional argument" not in msg and "unexpected keyword argument" not in msg:
                raise
            return cls({})

    try:
        from models.superpoint_core import SuperPointCore
        superpoint = _instantiate_superpoint(SuperPointCore)
    except ImportError:
        try:
            # rpautrat wrapper layout
            from superpoint.superpoint import SuperPoint as SuperPointCore
            superpoint = _instantiate_superpoint(SuperPointCore)
        except ImportError:
            try:
                # rpautrat PyTorch file layout (e.g. superpoint_pytorch.py)
                from superpoint_pytorch import SuperPoint as SuperPointCore
                superpoint = _instantiate_superpoint(SuperPointCore)
            except ImportError:
                try:
                    # Alternate import path (some setups expose SuperPoint at top-level)
                    from superpoint import SuperPoint as SuperPointCore
                    superpoint = _instantiate_superpoint(SuperPointCore)
                except ImportError:
                    log.error(
                        "Could not import SuperPointCore. "
                        "Clone https://github.com/rpautrat/SuperPoint "
                        "and add it to PYTHONPATH."
                    )
                    raise

    # Validate/load pretrained weights (fail fast on mismatch)
    sp_weights = Path('weights/superpoint_v1.pth')
    _validate_and_load_pretrained_weights(
        module=superpoint,
        weight_path=sp_weights,
        module_name='SuperPoint',
        strict=False,
    )

    # Accept xfeat.pth (preferred in notebook) and xfeat.pt as fallback.
    xf_weights = Path('weights/xfeat.pth')
    if not xf_weights.exists():
        alt = Path('weights/xfeat.pt')
        if alt.exists():
            xf_weights = alt
    _validate_and_load_pretrained_weights(
        module=xfeat,
        weight_path=xf_weights,
        module_name='XFeat',
        strict=False,
    )

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

    amp_enabled = bool(
        cfg.get('mixed_precision', True)
        and device.type == 'cuda'
        and not cfg.get('_amp_disabled_due_to_error', False)
    )

    def _forward_loss(use_amp: bool):
        with autocast('cuda', enabled=use_amp):
            out1 = model.forward_train(image1)
            out2 = model.forward_train(image2)

            loss_value, stats_value = loss_fn.forward_batch(
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
        return loss_value, stats_value

    try:
        loss, stats = _forward_loss(amp_enabled)
        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
    except RuntimeError as exc:
        if amp_enabled and _is_amp_runtime_error(exc):
            cfg['_amp_disabled_due_to_error'] = True
            cfg['mixed_precision'] = False
            log.warning(
                "AMP runtime error detected; auto-falling back to FP32 for remaining training. "
                f"reason={type(exc).__name__}: {exc}"
            )
            optimizer.zero_grad(set_to_none=True)
            loss, stats = _forward_loss(False)
            loss.backward()
        else:
            raise

    nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=cfg.get('grad_clip_norm', 10.0)
    )
    if amp_enabled and not cfg.get('_amp_disabled_due_to_error', False):
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

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
    scheduler: Optional[optim.lr_scheduler.ReduceLROnPlateau],
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
    if scheduler is not None:
        state['scheduler'] = scheduler.state_dict()

    path = ckpt_dir / f'epoch_{epoch:04d}.pth'
    _atomic_torch_save(state, path, prefix='ckpt_')
    log.info(f"  Saved checkpoint → {path}")

    if is_best:
        best_path = ckpt_dir / 'best.pth'
        _atomic_torch_save(state, best_path, prefix='ckpt_best_')
        log.info(f"  ★ New best → {best_path}")


def load_checkpoint(
    path:      str,
    model:     HybridModel,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.ReduceLROnPlateau,
    scaler:    GradScaler,
    device:    torch.device,
    require_scheduler_state: bool = True,
) -> int:
    """Load checkpoint, return start epoch."""
    state = torch.load(path, map_location=device)
    _validate_resume_checkpoint_payload(state, path)
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    if 'scheduler' in state and isinstance(state['scheduler'], dict):
        scheduler.load_state_dict(state['scheduler'])
    elif require_scheduler_state:
        raise RuntimeError(
            f"Invalid resume checkpoint {path}: missing 'scheduler' state. "
            "Suggested recovery: start fresh or resume from a full training checkpoint."
        )
    else:
        log.warning("Resume checkpoint has no scheduler state; continuing with fresh scheduler state.")
    if 'scaler' in state and isinstance(state['scaler'], dict):
        scaler.load_state_dict(state['scaler'])
    epoch = state['epoch'] + 1
    log.info(f"Resumed from {path}  (epoch {epoch})")
    return epoch


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: Dict, resume: Optional[str] = None) -> None:
    # TensorBoard's transitive deps can trigger TensorFlow plugin logs in some runtimes.
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    if cfg.get('mode') in {'megadepth', 'megadepth_raw'}:
        train_split = str(cfg.get('train_split', 'train')).lower()
        val_split = str(cfg.get('val_split', 'val')).lower()
        if train_split == val_split:
            raise ValueError(
                f"Invalid split configuration: train_split and val_split are both '{train_split}'. "
                "Use disjoint splits (train/val)."
            )
        if train_split == 'val' and val_split == 'train':
            raise ValueError(
                "Invalid split configuration: train_split=val and val_split=train. "
                "Use train_split=train and val_split=val."
            )
    _validate_image_size(cfg)
    _set_reproducible_seed(int(cfg.get('seed', 42)))
    is_colab = bool(os.environ.get('COLAB_GPU')) or ('google.colab' in sys.modules)
    if is_colab and int(cfg.get('num_workers', 0)) > 0:
        log.warning(
            "Colab runtime detected: using num_workers=0 safe default. "
            "You can test 2/4 workers after confirming stability."
        )
        cfg['num_workers'] = 0

    # ── Device ──────────────────────────────────────────────────────────
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        log.info(f"Device: {device}")
    except Exception as exc:
        raise _runtime_error_with_stage('setup/device', exc) from exc

    # ── Model ───────────────────────────────────────────────────────────
    try:
        model = build_model(cfg, device)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info(f"Trainable parameters: {trainable:,}")
    except Exception as exc:
        raise _runtime_error_with_stage('model init', exc) from exc

    # ── Loss ────────────────────────────────────────────────────────────
    try:
        loss_fn = HomographyHingeLoss(
            positive_margin=cfg['positive_margin'],
            negative_margin=cfg['negative_margin'],
            lambda_d=cfg['lambda_d'],
            lambda_rep=cfg.get('lambda_rep', 0.5),
            correspondence_threshold=cfg['correspondence_threshold'],
            safe_radius=cfg.get('safe_radius', 8.0),
            balance_pos_neg=cfg.get('balance_pos_neg', True),
        )
    except Exception as exc:
        raise _runtime_error_with_stage('loss init', exc) from exc

    # ── Optimizer + Scheduler ────────────────────────────────────────────
    try:
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg['lr'],
            weight_decay=cfg['weight_decay'],
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=cfg.get('lr_factor', 0.5),
            patience=cfg.get('lr_patience', 5),
            threshold=cfg.get('lr_threshold', 1e-4),
            cooldown=cfg.get('lr_cooldown', 0),
            min_lr=cfg.get('min_lr', 1e-6),
        )
        amp_enabled = bool(cfg.get('mixed_precision', True) and device.type == 'cuda')
        scaler = GradScaler('cuda', enabled=amp_enabled)
    except Exception as exc:
        raise _runtime_error_with_stage('optimizer/scheduler init', exc) from exc

    # ── Data ────────────────────────────────────────────────────────────
    try:
        image_size = (cfg['image_height'], cfg['image_width'])
        train_loader = build_dataloader(
            mode=cfg['mode'], root=cfg['data_root'], split=cfg.get('train_split', 'train'),
            image_size=image_size, batch_size=cfg['batch_size'],
            num_workers=cfg['num_workers'], shuffle=True,
            scene_info_dir=cfg.get('scene_info_dir'),
            val_split_ratio=cfg.get('megadepth_val_split_ratio', 0.2),
            min_overlap=cfg['min_overlap'], max_overlap=cfg['max_overlap'],
            max_pairs_per_scene=cfg.get('max_pairs_per_scene', 1000),
            augment=cfg['augment'],
            verify_pairs=cfg.get('verify_dataset_pairs', True),
            timeout_s=int(cfg.get('dataloader_timeout_s', 0)),
        )
        val_loader = build_dataloader(
            mode=cfg['mode'], root=cfg['data_root'], split=cfg.get('val_split', 'val'),
            image_size=image_size, batch_size=cfg['batch_size'],
            num_workers=cfg['num_workers'], shuffle=False,
            scene_info_dir=cfg.get('scene_info_dir'),
            val_split_ratio=cfg.get('megadepth_val_split_ratio', 0.2),
            min_overlap=cfg['min_overlap'], max_overlap=cfg['max_overlap'],
            max_pairs_per_scene=cfg.get('max_pairs_per_scene', 1000),
            augment=False,
            verify_pairs=cfg.get('verify_dataset_pairs', True),
            timeout_s=int(cfg.get('dataloader_timeout_s', 0)),
        )
    except Exception as exc:
        raise _runtime_error_with_stage('dataloaders', exc) from exc

    # ── Forward preflight ────────────────────────────────────────────────
    try:
        _run_dummy_forward_preflight(model, cfg, device)
        first_batch = next(iter(train_loader))
        with torch.no_grad():
            out1 = model.forward_train(first_batch['image1'].to(device))
            out2 = model.forward_train(first_batch['image2'].to(device))
        _validate_forward_output_keys_shapes(out1, first_batch['image1'].shape[0])
        _validate_forward_output_keys_shapes(out2, first_batch['image2'].shape[0])
        log.info("First-batch preflight step OK.")
    except Exception as exc:
        raise _runtime_error_with_stage('first batch forward/preflight', exc) from exc

    # ── Tensorboard ─────────────────────────────────────────────────────
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=cfg['log_dir'])
    except Exception as exc:
        raise _runtime_error_with_stage('logging/tensorboard init', exc) from exc

    # ── Resume ──────────────────────────────────────────────────────────
    start_epoch = 0
    if resume:
        try:
            start_epoch = load_checkpoint(resume, model, optimizer, scheduler, scaler, device)
        except Exception as exc:
            raise _runtime_error_with_stage('resume checkpoint load', exc) from exc

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

        try:
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
        except RuntimeError as exc:
            msg = str(exc).lower()
            if 'dataloader worker' in msg or 'worker exited unexpectedly' in msg or 'timed out' in msg:
                raise _runtime_error_with_stage(
                    'training loop/dataloader',
                    RuntimeError(
                        f"{exc}\nHint: set num_workers=0 on Colab, then test 2 or 4 only if stable. "
                        "If using workers, increase dataloader_timeout_s."
                    ),
                ) from exc
            raise _runtime_error_with_stage('training loop', exc) from exc

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
                model, optimizer, scheduler, scaler, epoch, val_loss, cfg, is_best
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
    Run synthetic pre-training then stage-2 fine-tuning.
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
        'mode': str(cfg.get('stage2_mode', 'megadepth_raw')),
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
    p.add_argument('--mode',         type=str, choices=['synthetic', 'megadepth', 'megadepth_raw'],
                   help='Training data mode')
    p.add_argument('--data_root',    type=str, help='Path to training data')
    p.add_argument('--scene_info_dir', type=str,
                   help='MegaDepth scene_info directory containing .npz metadata')
    p.add_argument('--batch_size',   type=int, help='Batch size')
    p.add_argument('--num_workers',  type=int, help='DataLoader workers (Colab-safe default: 0)')
    p.add_argument('--dataloader_timeout_s', type=int,
                   help='DataLoader timeout in seconds (used only when num_workers > 0)')
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
    p.add_argument('--train_split', type=str,
                   choices=['train', 'val'],
                   help='MegaDepth split used for training loader')
    p.add_argument('--val_split', type=str,
                   choices=['train', 'val'],
                   help='MegaDepth split used for validation loader')
    p.add_argument('--megadepth_val_split_ratio', type=float,
                   help='Fallback val ratio when scene_info has no train/val subfolders')
    p.add_argument('--no_verify_dataset_pairs', action='store_true',
                   help='Skip MegaDepth file-existence preflight checks')
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
    p.add_argument('--stage2_mode', type=str, choices=['megadepth', 'megadepth_raw'],
                   help='Stage-2 mode for --two_stage')
    p.add_argument('--stage1_epochs', type=int, help='Stage-1 epochs')
    p.add_argument('--stage2_epochs', type=int, help='Stage-2 epochs')
    p.add_argument('--seed', type=int, help='Random seed for reproducibility')
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
    if args.no_verify_dataset_pairs:
        cfg['verify_dataset_pairs'] = False

    log.info("Config:\n" + yaml.dump(cfg, default_flow_style=False))

    Path(cfg['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    Path(cfg['log_dir']).mkdir(parents=True, exist_ok=True)

    if cfg.get('two_stage', False):
        train_two_stage(cfg)
    else:
        train(cfg, resume=args.resume)
