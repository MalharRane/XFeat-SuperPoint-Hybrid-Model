"""
data/megadepth_dataset.py
=========================
MegaDepth dataset for training the XFeat-SuperPoint hybrid model.

Dataset Layout Expected
-----------------------
  megadepth_root/
    ├── Undistorted_SfM/
    │   └── <scene_id>/
    │       ├── sparse/
    │       │   └── cameras.txt, images.txt, points3D.txt
    │       └── images/
    │           └── <image_name>.jpg
    ├── phoenix/S6/zl548/MegaDepth_v1/
    │   └── <scene_id>/
    │       └── dense0/
    │           ├── depths/    (*.h5)
    │           └── imgs/      (*.jpg)
    └── scene_info/            (pre-processed .npz overlap files)

The scene_info directory contains pre-processed .npz files with
overlap scores and image-pair indices, following the LoFTR/LightGlue
train/val splits (Sun et al., CVPR 2021).

Two Modes
---------
  'megadepth'  – real image pairs with depth-based reprojection
  'synthetic'  – single images with random homographic warps (faster)

The synthetic mode is recommended for initial pre-training before
fine-tuning on real MegaDepth pairs.
"""

import os
import math
import random
import hashlib
import h5py
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

# Minimum positive depth for a reprojected point to be considered in front
# of the camera.  Must match the constant in losses/hinge_loss.py.
_MIN_DEPTH_Z: float = 0.01
_RAW_PAIR_MAX_DELTA: int = 20       # Nearby frame window for same-scene pairing.
_RAW_ORB_NFEATURES: int = 2000      # Dense ORB keypoints for robust homography fitting.
_RAW_LOAD_RETRIES: int = 5          # Retry budget for occasional corrupt/unreadable files.


def _hash_bucket_is_val(stem: str, val_split_ratio: float) -> bool:
    """
    Deterministic scene-level split helper using SHA-256 hash bucketing.

    The scene stem is hashed to a stable [0, 1) bucket value; buckets below
    val_split_ratio are assigned to val, the rest to train.
    """
    digest = hashlib.sha256(stem.encode('utf-8')).hexdigest()
    bucket = int(digest[:8], 16) / float(16**8)
    return bucket < val_split_ratio


def _scale_intrinsics_to_size(
    K: np.ndarray,
    orig_hw: Tuple[int, int],
    target_hw: Tuple[int, int],
) -> np.ndarray:
    """Scale camera intrinsics from original image size to target resize."""
    H0, W0 = orig_hw
    Ht, Wt = target_hw
    if H0 <= 0 or W0 <= 0:
        raise ValueError(
            f"Invalid original image size for intrinsics scaling: {(H0, W0)}"
        )
    sx = float(Wt) / float(W0)
    sy = float(Ht) / float(H0)

    K_scaled = K.astype(np.float32).copy()
    K_scaled[0, 0] *= sx
    K_scaled[0, 2] *= sx
    K_scaled[1, 1] *= sy
    K_scaled[1, 2] *= sy
    return K_scaled


# ---------------------------------------------------------------------------
# Homography generation utilities
# ---------------------------------------------------------------------------

def sample_random_homography(
    height: int,
    width:  int,
    max_angle:     float = 30.0,
    max_shear:     float = 0.2,
    max_scale:     float = 0.3,
    max_translate: float = 0.1,
    perspective_strength: float = 0.0008,
    device: torch.device = torch.device('cpu'),
) -> torch.Tensor:
    """
    Sample a random 3×3 homography composed of:
        translation + rotation + scale + shear + perspective distortion.

    Returns
    -------
    H : (3, 3) float32 tensor  mapping image1 → warped image2
    """
    # Random rotation
    angle = random.uniform(-max_angle, max_angle)
    angle_rad = math.radians(angle)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    R = torch.tensor([
        [cos_a, -sin_a, 0],
        [sin_a,  cos_a, 0],
        [0,      0,     1],
    ], dtype=torch.float32)

    # Random scale
    sx = 1.0 + random.uniform(-max_scale, max_scale)
    sy = 1.0 + random.uniform(-max_scale, max_scale)
    S = torch.diag(torch.tensor([sx, sy, 1.0]))

    # Random shear
    sh = random.uniform(-max_shear, max_shear)
    Sh = torch.eye(3)
    Sh[0, 1] = sh

    # Random translation (fraction of image size)
    tx = random.uniform(-max_translate, max_translate) * width
    ty = random.uniform(-max_translate, max_translate) * height
    T_mat = torch.eye(3)
    T_mat[0, 2] = tx
    T_mat[1, 2] = ty

    # Random perspective (small)
    P = torch.eye(3)
    P[2, 0] = random.uniform(-perspective_strength, perspective_strength)
    P[2, 1] = random.uniform(-perspective_strength, perspective_strength)

    # Center-based composition: T_center * components * T_center_inv
    cx, cy = width / 2.0, height / 2.0
    T_c     = torch.eye(3); T_c[0,2] = cx;  T_c[1,2] = cy
    T_c_inv = torch.eye(3); T_c_inv[0,2] = -cx; T_c_inv[1,2] = -cy

    H = T_c @ R @ S @ Sh @ P @ T_c_inv @ T_mat

    return H.to(device)


def apply_homography_to_image(
    image:   torch.Tensor,
    H:       torch.Tensor,
) -> torch.Tensor:
    """
    Warp image by homography H using differentiable grid_sample.

    Args
    ----
    image : (1, C, H, W) or (C, H, W)  float tensor in [0, 1]
    H     : (3, 3) homography  input → warped

    Returns
    -------
    warped : same shape as image
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    B, C, H_im, W_im = image.shape
    device = image.device

    # Build grid of target pixel coordinates (homogeneous)
    yy, xx = torch.meshgrid(
        torch.arange(H_im, dtype=torch.float32, device=device),
        torch.arange(W_im, dtype=torch.float32, device=device),
        indexing='ij'
    )
    ones = torch.ones_like(xx)
    grid_h = torch.stack([xx, yy, ones], dim=0).reshape(3, -1)   # (3, H*W)

    # Map target pixels back through H_inv to find source pixels
    H_inv = torch.inverse(H.to(torch.float32))
    src_h = (H_inv.to(device) @ grid_h)                          # (3, H*W)
    src_xy = src_h[:2] / src_h[2:].clamp(min=1e-8)               # (2, H*W)

    # Normalise to [-1, 1] for grid_sample
    src_x_norm = 2.0 * src_xy[0] / (W_im - 1) - 1.0
    src_y_norm = 2.0 * src_xy[1] / (H_im - 1) - 1.0
    grid = torch.stack([src_x_norm, src_y_norm], dim=-1)
    grid = grid.reshape(1, H_im, W_im, 2).expand(B, -1, -1, -1)  # (B,H,W,2)

    warped = F.grid_sample(
        image, grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )

    return warped.squeeze(0) if squeeze else warped


# ---------------------------------------------------------------------------
# Synthetic Homography Dataset (pre-training)
# ---------------------------------------------------------------------------

class SyntheticHomographyDataset(Dataset):
    """
    Generates image pairs on-the-fly using random homographic warps.

    Each sample consists of:
        'image1'      : (1, H, W) float32 in [0, 1]  — original
        'image2'      : (1, H, W) float32 in [0, 1]  — warped
        'homography'  : (3, 3) float32               — I₁ → I₂

    Compatible with any image directory (COCO, Oxford-Paris, etc.)
    """

    def __init__(
        self,
        image_dir:  str,
        image_size: Tuple[int, int] = (480, 640),   # (H, W)
        max_angle:  float = 25.0,
        max_scale:  float = 0.25,
        augment:    bool  = True,
        extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.webp'),
    ):
        self.image_size = image_size
        self.max_angle  = max_angle
        self.max_scale  = max_scale
        self.augment    = augment

        # Collect all image paths
        root = Path(image_dir)
        self.image_paths = [
            p for p in root.rglob('*')
            if p.suffix.lower() in extensions
        ]
        assert len(self.image_paths) > 0, \
            f"No images found in {image_dir} with extensions {extensions}"

        # Photometric augmentation for image2
        if augment:
            self.photo_aug = T.Compose([
                T.ColorJitter(
                    brightness=0.3, contrast=0.3,
                    saturation=0.0, hue=0.0
                ),
                T.RandomApply([T.GaussianBlur(kernel_size=5)], p=0.3),
            ])
        else:
            self.photo_aug = None

        print(f"[SyntheticHomographyDataset] {len(self.image_paths):,} images")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.image_paths[idx]

        # Load and convert to grayscale float [0,1]
        try:
            img = Image.open(path).convert('L')             # grayscale PIL
        except Exception:
            # Fallback to random index on corrupted file
            return self.__getitem__(random.randint(0, len(self) - 1))

        H, W = self.image_size
        img = TF.resize(img, [H, W], interpolation=T.InterpolationMode.BILINEAR)
        img_t = TF.to_tensor(img)   # (1, H, W) float32 in [0, 1]

        # Sample random homography
        homography = sample_random_homography(H, W, max_angle=self.max_angle,
                                              max_scale=self.max_scale)

        # Warp image2
        img2_t = apply_homography_to_image(img_t, homography)

        # Photometric augmentation on image2
        if self.photo_aug is not None:
            # Convert to [0,255] uint8 for ColorJitter, back to float
            img2_pil = TF.to_pil_image(img2_t.clamp(0, 1))
            img2_pil = self.photo_aug(img2_pil)
            img2_t   = TF.to_tensor(img2_pil)

        return {
            'image1':     img_t.float(),           # (1, H, W)
            'image2':     img2_t.float().clamp(0, 1),  # (1, H, W)
            'homography': homography,              # (3, 3)
        }


# ---------------------------------------------------------------------------
# MegaDepth Dataset (fine-tuning)
# ---------------------------------------------------------------------------

class MegaDepthDataset(Dataset):
    """
    MegaDepth image-pair dataset for fine-tuning.

    Uses pre-processed scene-info .npz files (LoFTR/LightGlue split format).
    Each scene_info file contains:
        'image_paths'   : array of relative paths to images
        'depth_paths'   : array of relative paths to depth .h5 files
        'intrinsics'    : (N, 3, 3) camera intrinsic matrices
        'poses'         : (N, 4, 4) world-to-camera poses
        'overlap_matrix': (N, N) covisibility scores
        'image_pairs'   : (P, 2) precomputed valid pair indices

    For the hinge loss we need a homography.  Since MegaDepth scenes are
    not planar, we use an *approximate* homography estimated from:
        H ≈ K₂ · R₂₁ · K₁⁻¹   (valid when scene is at distance >> baseline)

    Args
    ----
    root          : path to megadepth_root/
    scene_info_dir: path to scene_info/ (default: root/scene_info)
    split         : 'train' | 'val'
    image_size    : (H, W) resize target
    min_overlap   : minimum covisibility score for a valid pair
    max_overlap   : maximum covisibility (too-easy pairs excluded)
    max_pairs_per_scene : limit pairs sampled per scene per epoch
    """

    # Scene-level .npz files are split into train / val following LoFTR
    TRAIN_SCENES_URL = (
        "https://raw.githubusercontent.com/zju3dv/LoFTR/master/"
        "assets/megadepth_test_1500_scene_info.txt"
    )

    def __init__(
        self,
        root:           str,
        scene_info_dir: Optional[str] = None,
        split:          str  = 'train',
        val_split_ratio: float = 0.2,
        image_size:     Tuple[int, int] = (480, 640),
        min_overlap:    float = 0.15,
        max_overlap:    float = 0.70,
        max_pairs_per_scene: int = 200,
        augment:        bool  = True,
        verify_pairs:   bool  = True,
    ):
        self.root       = Path(root)
        self.split      = str(split).lower()
        if self.split not in {'train', 'val'}:
            raise ValueError(f"split must be 'train' or 'val', got '{split}'")
        self.val_split_ratio = float(val_split_ratio)
        if not (0.0 <= self.val_split_ratio <= 1.0):
            raise ValueError(
                f"val_split_ratio must be in [0, 1], got {self.val_split_ratio}"
            )
        self.image_size = image_size
        self.min_ov     = min_overlap
        self.max_ov     = max_overlap
        self.max_pairs  = max_pairs_per_scene
        self.verify_pairs = bool(verify_pairs)

        scene_info_dir = scene_info_dir or str(self.root / 'scene_info')
        self.scene_info_dir = Path(scene_info_dir)

        if augment:
            self.photo_aug = T.Compose([
                T.RandomApply(
                    [T.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5
                ),
                T.RandomApply([T.GaussianBlur(kernel_size=5)], p=0.2),
            ])
        else:
            self.photo_aug = None

        # Build pair index
        self.pairs: List[Dict] = []
        self.preflight_stats: Dict[str, float] = {
            'num_scene_npz_total': 0,
            'num_scene_npz_selected': 0,
            'num_scene_npz_loaded': 0,
            'num_scene_npz_failed': 0,
            'pairs_candidate': 0,
            'pairs_kept': 0,
            'pairs_missing_image': 0,
            'pairs_missing_depth_files': 0,
            'pairs_no_depth_metadata': 0,
            'pairs_depth_unavailable': 0,
            'overlap_min': float('inf'),
            'overlap_max': float('-inf'),
        }
        self._build_pair_index()
        self._print_preflight_summary()
        if len(self.pairs) == 0:
            raise RuntimeError(
                f"[MegaDepthDataset] split={self.split} has 0 valid pairs. "
                "Check scene_info split coverage, path alignment, and overlap thresholds."
            )
        print(
            f"[MegaDepthDataset] split={self.split}  "
            f"{len(self.pairs):,} pairs loaded"
        )

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _build_pair_index(self) -> None:
        """Scan scene_info directory and collect valid image pairs."""
        npz_files = sorted(self.scene_info_dir.glob('*.npz'))
        if not npz_files:
            raise FileNotFoundError(
                f"No .npz scene-info files found in {self.scene_info_dir}. "
                "Download them from the LoFTR/LightGlue release."
            )

        self.preflight_stats['num_scene_npz_total'] = len(npz_files)
        selected_npz = self._select_scene_files(npz_files)
        self.preflight_stats['num_scene_npz_selected'] = len(selected_npz)

        for npz_path in selected_npz:
            try:
                info = np.load(str(npz_path), allow_pickle=True)
                self._ingest_scene(info, npz_path.stem)
                self.preflight_stats['num_scene_npz_loaded'] += 1
            except Exception as e:
                self.preflight_stats['num_scene_npz_failed'] += 1
                print(f"  [WARN] Could not load {npz_path.name}: {e}")

    def _select_scene_files(self, npz_files: List[Path]) -> List[Path]:
        """
        Resolve scene files for split.
        Priority:
          1) scene_info_dir/<split>/*.npz
          2) deterministic hash split over scene-id stems
        """
        split_dir = self.scene_info_dir / self.split
        if split_dir.exists() and split_dir.is_dir():
            split_npz = sorted(split_dir.glob('*.npz'))
            if split_npz:
                return split_npz

        selected: List[Path] = []
        for p in npz_files:
            scene_is_val = _hash_bucket_is_val(p.stem, self.val_split_ratio)
            if (self.split == 'val' and scene_is_val) or (self.split == 'train' and not scene_is_val):
                selected.append(p)
        return selected

    def _ingest_scene(self, info: np.lib.npyio.NpzFile, scene_id: str) -> None:
        """Extract valid pairs from one scene .npz."""
        image_paths  = info['image_paths']
        depth_paths  = info.get('depth_paths', None)
        intrinsics   = info['intrinsics']
        poses        = info['poses']
        overlap      = info.get('overlap_matrix', None)

        if overlap is None:
            return

        N = len(image_paths)
        if intrinsics.shape[0] != N or poses.shape[0] != N:
            raise ValueError(
                f"Scene {scene_id} has inconsistent lengths: "
                f"N={N}, intrinsics={intrinsics.shape[0]}, poses={poses.shape[0]}"
            )
        if depth_paths is not None:
            depth_paths_arr = np.asarray(depth_paths)
            if depth_paths_arr.ndim != 1 or depth_paths_arr.shape[0] != N:
                raise ValueError(
                    f"Scene {scene_id} has invalid depth_paths shape: "
                    f"{depth_paths_arr.shape}, expected ({N},)"
                )
        if overlap.shape[0] != N or overlap.shape[1] != N:
            raise ValueError(
                f"Scene {scene_id} has invalid overlap_matrix shape: {overlap.shape}, expected ({N}, {N})"
            )

        count = 0
        indices = list(range(N))
        random.shuffle(indices)

        for i in indices:
            if count >= self.max_pairs:
                break
            for j in indices:
                if i >= j:
                    continue
                ov = float(overlap[i, j])
                if not (self.min_ov <= ov <= self.max_ov):
                    continue

                self.preflight_stats['pairs_candidate'] += 1
                img1 = self.root / str(image_paths[i])
                img2 = self.root / str(image_paths[j])
                if self.verify_pairs and (not img1.exists() or not img2.exists()):
                    self.preflight_stats['pairs_missing_image'] += 1
                    continue

                d1 = None
                d2 = None
                if depth_paths is not None:
                    d1_abs = self.root / str(depth_paths[i])
                    d2_abs = self.root / str(depth_paths[j])
                    d1_ok = d1_abs.exists()
                    d2_ok = d2_abs.exists()
                    if d1_ok:
                        d1 = str(d1_abs)
                    if d2_ok:
                        d2 = str(d2_abs)
                    if not d1_ok or not d2_ok:
                        self.preflight_stats['pairs_missing_depth_files'] += 1
                        self.preflight_stats['pairs_depth_unavailable'] += 1
                else:
                    self.preflight_stats['pairs_no_depth_metadata'] += 1
                    self.preflight_stats['pairs_depth_unavailable'] += 1

                self.pairs.append({
                    'scene_id':    scene_id,
                    'image_path1': str(img1),
                    'image_path2': str(img2),
                    'depth_path1': d1,
                    'depth_path2': d2,
                    'K1': intrinsics[i].astype(np.float32),
                    'K2': intrinsics[j].astype(np.float32),
                    'T1': poses[i].astype(np.float32),  # world-to-cam
                    'T2': poses[j].astype(np.float32),
                    'overlap': ov,
                })
                self.preflight_stats['pairs_kept'] += 1
                self.preflight_stats['overlap_min'] = min(self.preflight_stats['overlap_min'], ov)
                self.preflight_stats['overlap_max'] = max(self.preflight_stats['overlap_max'], ov)
                count += 1
                if count >= self.max_pairs:
                    break

    def _print_preflight_summary(self) -> None:
        stats = self.preflight_stats
        ov_min = stats['overlap_min'] if stats['pairs_kept'] > 0 else float('nan')
        ov_max = stats['overlap_max'] if stats['pairs_kept'] > 0 else float('nan')
        print(
            "[MegaDepthDataset preflight] "
            f"split={self.split} "
            f"scenes_total={int(stats['num_scene_npz_total'])} "
            f"scenes_selected={int(stats['num_scene_npz_selected'])} "
            f"scenes_loaded={int(stats['num_scene_npz_loaded'])} "
            f"scenes_failed={int(stats['num_scene_npz_failed'])} "
            f"pairs_candidate={int(stats['pairs_candidate'])} "
            f"pairs_kept={int(stats['pairs_kept'])} "
            f"missing_image={int(stats['pairs_missing_image'])} "
            f"pairs_depth_unavailable={int(stats['pairs_depth_unavailable'])} "
            f"(no_metadata={int(stats['pairs_no_depth_metadata'])}, "
            f"missing_depth_files={int(stats['pairs_missing_depth_files'])}) "
            f"overlap_range=[{ov_min:.3f}, {ov_max:.3f}]"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_image(path: str, size: Tuple[int, int]) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Load image as (1, H, W) float32 in [0, 1], plus original (H, W)."""
        img = Image.open(path).convert('L')
        orig_hw = (img.height, img.width)
        img = TF.resize(img, [size[0], size[1]],
                        interpolation=T.InterpolationMode.BILINEAR)
        return TF.to_tensor(img).float(), orig_hw

    @staticmethod
    def _load_depth(path: str, size: Tuple[int, int]) -> torch.Tensor:
        """Load depth map from .h5 file and resize."""
        with h5py.File(path, 'r') as f:
            depth = torch.from_numpy(f['depth'][:].astype(np.float32))

        if depth.dim() == 2:
            depth = depth.unsqueeze(0)   # (1, H_orig, W_orig)

        depth = F.interpolate(
            depth.unsqueeze(0),
            size=size, mode='nearest'
        ).squeeze(0)                     # (1, H, W)

        return depth

    @staticmethod
    def _approx_homography(
        K1: np.ndarray, K2: np.ndarray,
        T1: np.ndarray, T2: np.ndarray,
    ) -> torch.Tensor:
        """
        Compute approximate planar homography from camera matrices.

        H ≈ K₂ · R₂₁ · K₁⁻¹

        This is exact only for a pure rotation (or scene at infinity), but
        provides a reasonable training signal for nearby image pairs.
        Use depth-based correspondences (_compute_warp_field) for better accuracy.
        """
        R1 = T1[:3, :3]
        R2 = T2[:3, :3]
        R21 = R2 @ R1.T                              # rotation: cam1 → cam2

        K2_t = torch.from_numpy(K2).float()
        R21_t = torch.from_numpy(R21).float()
        K1_inv = torch.from_numpy(np.linalg.inv(K1)).float()

        H = K2_t @ R21_t @ K1_inv                   # (3, 3)
        return H

    @staticmethod
    def _compute_warp_field(
        K1:     np.ndarray,
        K2:     np.ndarray,
        T1:     np.ndarray,
        T2:     np.ndarray,
        depth1: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute a dense depth-based warp field for accurate correspondences.

        For each pixel (x, y) in image-1, the warp field stores the
        corresponding pixel (x2, y2) in image-2, computed via:
          1. Unproject (x, y) to 3-D using depth and K₁
          2. Transform to camera-2 frame using T₁⁻¹ and T₂
          3. Project onto image-2 using K₂

        This is exact for rigid scenes (unlike H ≈ K₂·R₂₁·K₁⁻¹ which
        ignores translation and only holds for scenes at infinity).

        Args
        ----
        K1, K2       : (3, 3) camera intrinsics
        T1, T2       : (4, 4) world-to-camera extrinsics
        depth1       : (1, H, W) depth map for image-1 (metres)
        image_size   : (H, W) output size

        Returns
        -------
        warp_field : (H, W, 2) float32 — (x2, y2) for each pixel of image-1
        warp_valid : (H, W) bool       — True where depth was > 0 and
                                         the reprojection is in front of cam-2
        """
        H, W = image_size
        d = depth1[0].numpy().astype(np.float32)  # (H, W)

        # Pixel grid
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')  # (H, W)
        pixels_flat = np.stack(
            [xx.flatten(), yy.flatten(), np.ones(H * W)], axis=0
        )  # (3, H*W)

        # Unproject to camera-1 3-D
        K1_inv = np.linalg.inv(K1.astype(np.float64))
        rays   = K1_inv @ pixels_flat                          # (3, H*W)
        d_flat = d.flatten()                                   # (H*W,)
        P_cam1 = rays * d_flat[np.newaxis, :]                 # (3, H*W)

        # Camera-1 → world → camera-2
        # T is world-to-camera: P_world = T1_inv @ P_cam1_h
        R1, t1 = T1[:3, :3].astype(np.float64), T1[:3, 3].astype(np.float64)
        R2, t2 = T2[:3, :3].astype(np.float64), T2[:3, 3].astype(np.float64)

        # P_world = R1.T @ (P_cam1 - t1)
        P_world = R1.T @ (P_cam1 - t1[:, np.newaxis])         # (3, H*W)
        # P_cam2 = R2 @ P_world + t2
        P_cam2  = R2 @ P_world + t2[:, np.newaxis]            # (3, H*W)

        # Project onto image-2
        z2 = P_cam2[2]                                         # (H*W,)
        valid_z = (z2 > _MIN_DEPTH_Z) & (d_flat > 0.0)  # in front of cam-2 and has depth

        proj = K2.astype(np.float64) @ (P_cam2 / np.where(valid_z, z2, 1.0)[np.newaxis, :])
        x2 = proj[0].astype(np.float32)
        y2 = proj[1].astype(np.float32)

        warp_field = np.stack([x2, y2], axis=-1).reshape(H, W, 2)
        warp_valid = valid_z.reshape(H, W)

        return (
            torch.from_numpy(warp_field),
            torch.from_numpy(warp_valid),
        )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pair = self.pairs[idx]

        try:
            img1, img1_orig_hw = self._load_image(pair['image_path1'], self.image_size)
            img2, img2_orig_hw = self._load_image(pair['image_path2'], self.image_size)
        except Exception:
            return self.__getitem__(random.randint(0, len(self) - 1))

        K1 = _scale_intrinsics_to_size(pair['K1'], img1_orig_hw, self.image_size)
        K2 = _scale_intrinsics_to_size(pair['K2'], img2_orig_hw, self.image_size)

        # Compute approximate homography (always available as fallback)
        H = self._approx_homography(K1, K2,
                                     pair['T1'], pair['T2'])

        # Attempt depth-based warp field for accurate correspondences.
        # Falls back to None (use homography) if depth is unavailable.
        warp_field: Optional[torch.Tensor] = None
        warp_valid: Optional[torch.Tensor] = None
        dp1 = pair.get('depth_path1')
        if dp1 is not None and Path(dp1).exists():
            try:
                depth1 = self._load_depth(dp1, self.image_size)
                warp_field, warp_valid = self._compute_warp_field(
                    K1, K2,
                    pair['T1'], pair['T2'],
                    depth1, self.image_size,
                )
            except Exception:
                warp_field = None
                warp_valid = None

        # Photometric augmentation on image2
        if self.photo_aug is not None:
            img2_pil = TF.to_pil_image(img2.clamp(0, 1))
            img2_pil = self.photo_aug(img2_pil)
            img2     = TF.to_tensor(img2_pil)

        out: Dict[str, object] = {
            'image1':     img1,
            'image2':     img2,
            'homography': H,
            'overlap':    torch.tensor(pair['overlap'], dtype=torch.float32),
        }
        if warp_field is not None:
            out['warp_field'] = warp_field  # (H, W, 2)
            out['warp_valid'] = warp_valid  # (H, W) bool
        return out


# ---------------------------------------------------------------------------
# Raw MegaDepth Dataset (no scene_info .npz required)
# ---------------------------------------------------------------------------

class MegaDepthRawDataset(Dataset):
    """
    MegaDepth scene-folder dataset that does not depend on scene_info .npz files.

    Expected subset layout (examples):
      root/0001/dense0/imgs/*.jpg
      root/0001/dense0/depths/*.h5
      root/0002/dense0/imgs/*.jpg
      ...

    Strategy:
      1) Build same-scene image pairs with a lightweight overlap proxy.
      2) Try to estimate a homography directly from the two images (ORB+RANSAC).
      3) If estimation fails, fall back to synthetic homographic warping of image1.
    """

    def __init__(
        self,
        root: str,
        split: str = 'train',
        val_split_ratio: float = 0.2,
        image_size: Tuple[int, int] = (480, 640),
        min_overlap: float = 0.15,
        max_overlap: float = 0.95,
        max_pairs_per_scene: int = 200,
        augment: bool = True,
        verify_pairs: bool = True,
    ):
        self.root = Path(root)
        self.split = str(split).lower()
        if self.split not in {'train', 'val'}:
            raise ValueError(f"split must be 'train' or 'val', got '{split}'")
        self.val_split_ratio = float(val_split_ratio)
        if not (0.0 <= self.val_split_ratio <= 1.0):
            raise ValueError(
                f"val_split_ratio must be in [0, 1], got {self.val_split_ratio}"
            )
        self.image_size = image_size
        self.min_ov = float(min_overlap)
        self.max_ov = float(max_overlap)
        self.max_pairs = int(max_pairs_per_scene)
        self.verify_pairs = bool(verify_pairs)

        if augment:
            self.photo_aug = T.Compose([
                T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5),
                T.RandomApply([T.GaussianBlur(kernel_size=5)], p=0.2),
            ])
        else:
            self.photo_aug = None

        self.pairs: List[Dict[str, Union[str, float]]] = []
        self.preflight_stats: Dict[str, float] = {
            'num_scene_dirs_total': 0,
            'num_scene_dirs_selected': 0,
            'pairs_candidate': 0,
            'pairs_kept': 0,
            'pairs_missing_image': 0,
            'overlap_min': float('inf'),
            'overlap_max': float('-inf'),
        }
        self._build_pair_index()
        self._print_preflight_summary()
        if len(self.pairs) == 0:
            raise RuntimeError(
                f"[MegaDepthRawDataset] split={self.split} has 0 valid pairs. "
                "Check root layout and overlap thresholds."
            )
        print(
            f"[MegaDepthRawDataset] split={self.split}  "
            f"{len(self.pairs):,} pairs loaded"
        )

    @staticmethod
    def _load_thumbnail(path: Path, size: int = 64) -> np.ndarray:
        """Load grayscale thumbnail normalized to [0, 1] for overlap proxy."""
        img = Image.open(path).convert('L').resize((size, size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr

    @staticmethod
    def _overlap_proxy(a: np.ndarray, b: np.ndarray) -> float:
        """Compute normalized cross-correlation and map it from [-1,1] to [0,1]."""
        a0 = a - a.mean()
        b0 = b - b.mean()
        denom = float(np.sqrt((a0 * a0).sum() * (b0 * b0).sum()) + 1e-8)
        corr = float((a0 * b0).sum() / denom)
        return max(0.0, min(1.0, 0.5 * (corr + 1.0)))

    def _discover_scene_img_dirs(self) -> List[Tuple[str, Path]]:
        candidates = sorted(self.root.glob('**/dense0/imgs'))
        out: List[Tuple[str, Path]] = []
        for img_dir in candidates:
            try:
                scene_id = img_dir.parent.parent.name  # .../<scene>/dense0/imgs
            except Exception:
                continue
            out.append((scene_id, img_dir))
        return out

    def _build_pair_index(self) -> None:
        scene_dirs = self._discover_scene_img_dirs()
        self.preflight_stats['num_scene_dirs_total'] = len(scene_dirs)

        selected_scene_dirs: List[Tuple[str, Path]] = []
        for scene_id, img_dir in scene_dirs:
            scene_is_val = _hash_bucket_is_val(scene_id, self.val_split_ratio)
            if (self.split == 'val' and scene_is_val) or (self.split == 'train' and not scene_is_val):
                selected_scene_dirs.append((scene_id, img_dir))
        self.preflight_stats['num_scene_dirs_selected'] = len(selected_scene_dirs)

        exts = {'.jpg', '.jpeg', '.png', '.webp'}
        for scene_id, img_dir in selected_scene_dirs:
            img_paths = sorted([p for p in img_dir.glob('*') if p.suffix.lower() in exts])
            if len(img_paths) < 2:
                continue

            thumbs: List[np.ndarray] = []
            valid_img_paths: List[Path] = []
            for p in img_paths:
                if self.verify_pairs and not p.exists():
                    continue
                try:
                    thumbs.append(self._load_thumbnail(p, size=64))
                    valid_img_paths.append(p)
                except Exception:
                    continue
            if len(valid_img_paths) < 2:
                continue

            n = len(valid_img_paths)
            local_pairs: List[Tuple[int, int, float]] = []
            # Same-scene, near-index candidates are usually more overlapping.
            max_delta = min(_RAW_PAIR_MAX_DELTA, n - 1)
            for i in range(n):
                for delta in range(1, max_delta + 1):
                    j = i + delta
                    if j >= n:
                        break
                    ov = self._overlap_proxy(thumbs[i], thumbs[j])
                    self.preflight_stats['pairs_candidate'] += 1
                    if self.min_ov <= ov <= self.max_ov:
                        local_pairs.append((i, j, ov))

            # Fallback: if strict overlap bounds kept nothing, keep nearby pairs.
            if not local_pairs:
                for i in range(0, n - 1):
                    j = min(i + 1, n - 1)
                    if i < j:
                        ov = self._overlap_proxy(thumbs[i], thumbs[j])
                        local_pairs.append((i, j, ov))

            random.shuffle(local_pairs)
            local_pairs = local_pairs[: self.max_pairs]

            for i, j, ov in local_pairs:
                p1 = valid_img_paths[i]
                p2 = valid_img_paths[j]
                if self.verify_pairs and (not p1.exists() or not p2.exists()):
                    self.preflight_stats['pairs_missing_image'] += 1
                    continue
                self.pairs.append({
                    'scene_id': scene_id,
                    'image_path1': str(p1),
                    'image_path2': str(p2),
                    'overlap': float(ov),
                })
                self.preflight_stats['pairs_kept'] += 1
                self.preflight_stats['overlap_min'] = min(self.preflight_stats['overlap_min'], float(ov))
                self.preflight_stats['overlap_max'] = max(self.preflight_stats['overlap_max'], float(ov))

    def _print_preflight_summary(self) -> None:
        stats = self.preflight_stats
        ov_min = stats['overlap_min'] if stats['pairs_kept'] > 0 else float('nan')
        ov_max = stats['overlap_max'] if stats['pairs_kept'] > 0 else float('nan')
        print(
            "[MegaDepthRawDataset preflight] "
            f"split={self.split} "
            f"scenes_total={int(stats['num_scene_dirs_total'])} "
            f"scenes_selected={int(stats['num_scene_dirs_selected'])} "
            f"pairs_candidate={int(stats['pairs_candidate'])} "
            f"pairs_kept={int(stats['pairs_kept'])} "
            f"missing_image={int(stats['pairs_missing_image'])} "
            f"overlap_range=[{ov_min:.3f}, {ov_max:.3f}]"
        )

    @staticmethod
    def _load_image(path: str, size: Tuple[int, int]) -> torch.Tensor:
        img = Image.open(path).convert('L')
        img = TF.resize(img, [size[0], size[1]], interpolation=T.InterpolationMode.BILINEAR)
        return TF.to_tensor(img).float()

    @staticmethod
    def _estimate_homography_or_none(
        img1: torch.Tensor,
        img2: torch.Tensor,
        min_matches: int = 40,
        min_inliers: int = 20,
    ) -> Optional[torch.Tensor]:
        """
        Estimate pair homography via ORB features + BF matching + RANSAC.

        Returns None when matches/inliers are insufficient or when the
        estimated matrix is numerically invalid/degenerate.
        """
        arr1 = (img1.squeeze(0).cpu().numpy() * 255.0).astype(np.uint8)
        arr2 = (img2.squeeze(0).cpu().numpy() * 255.0).astype(np.uint8)

        orb = cv2.ORB_create(nfeatures=_RAW_ORB_NFEATURES)
        k1, d1 = orb.detectAndCompute(arr1, None)
        k2, d2 = orb.detectAndCompute(arr2, None)
        if d1 is None or d2 is None:
            return None

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(d1, d2)
        if len(matches) < min_matches:
            return None
        matches = sorted(matches, key=lambda m: m.distance)[: min(len(matches), 500)]

        pts1 = np.float32([k1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([k2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
        if H is None or mask is None:
            return None

        inliers = int(mask.astype(np.uint8).sum())
        if inliers < min_inliers:
            return None
        if not np.isfinite(H).all():
            return None
        if abs(float(np.linalg.det(H[:2, :2]))) < 1e-4:
            return None
        return torch.from_numpy(H.astype(np.float32))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pair = self.pairs[idx]
        img1: Optional[torch.Tensor] = None
        img2: Optional[torch.Tensor] = None
        for _ in range(_RAW_LOAD_RETRIES):
            try:
                img1 = self._load_image(str(pair['image_path1']), self.image_size)
                img2 = self._load_image(str(pair['image_path2']), self.image_size)
                break
            except (OSError, ValueError):
                pair = self.pairs[random.randint(0, len(self) - 1)]
        if img1 is None or img2 is None:
            raise RuntimeError("[MegaDepthRawDataset] Failed to load image pair after retries.")

        H = self._estimate_homography_or_none(img1, img2)

        # Fallback to synthetic pair if geometric estimation fails.
        if H is None:
            Hh, Ww = self.image_size
            H = sample_random_homography(Hh, Ww, max_angle=20.0, max_scale=0.2)
            img2 = apply_homography_to_image(img1, H).clamp(0, 1)

        if self.photo_aug is not None:
            img2_pil = TF.to_pil_image(img2.clamp(0, 1))
            img2_pil = self.photo_aug(img2_pil)
            img2 = TF.to_tensor(img2_pil)

        return {
            'image1': img1,
            'image2': img2,
            'homography': H.float(),
            'overlap': torch.tensor(float(pair['overlap']), dtype=torch.float32),
        }


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloader(
    mode:              str,
    root:              str,
    split:             str = 'train',
    image_size:        Tuple[int, int] = (480, 640),
    batch_size:        int  = 4,
    num_workers:       int  = 4,
    shuffle:           bool = True,
    scene_info_dir:    Optional[str] = None,
    val_split_ratio:   float = 0.2,
    min_overlap:       float = 0.15,
    max_overlap:       float = 0.70,
    max_pairs_per_scene: int = 200,
    augment:           bool  = True,
    verify_pairs:      bool  = True,
) -> DataLoader:
    """
    Build a DataLoader for either synthetic or MegaDepth training.

    Parameters
    ----------
    mode : 'synthetic' | 'megadepth' | 'megadepth_raw'
    root : path to images (synthetic) or megadepth_root (megadepth*)
    """
    if mode == 'synthetic':
        dataset = SyntheticHomographyDataset(
            image_dir=root,
            image_size=image_size,
            augment=augment,
        )
    elif mode == 'megadepth':
        dataset = MegaDepthDataset(
            root=root,
            scene_info_dir=scene_info_dir,
            split=split,
            val_split_ratio=val_split_ratio,
            image_size=image_size,
            min_overlap=min_overlap,
            max_overlap=max_overlap,
            max_pairs_per_scene=max_pairs_per_scene,
            augment=augment,
            verify_pairs=verify_pairs,
        )
    elif mode == 'megadepth_raw':
        dataset = MegaDepthRawDataset(
            root=root,
            split=split,
            val_split_ratio=val_split_ratio,
            image_size=image_size,
            min_overlap=min_overlap,
            max_overlap=max_overlap,
            max_pairs_per_scene=max_pairs_per_scene,
            augment=augment,
            verify_pairs=verify_pairs,
        )
    else:
        raise ValueError(
            f"mode must be 'synthetic', 'megadepth', or 'megadepth_raw', got '{mode}'"
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=_collate_fn,
    )


def _collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Stack tensors; handle variable-size items and optional keys."""
    # Collect all keys present in *any* sample
    all_keys = set()
    for b in batch:
        all_keys.update(b.keys())

    out: Dict[str, object] = {}
    for key in all_keys:
        vals = [b.get(key) for b in batch]

        if all(v is None for v in vals):
            out[key] = None
        elif any(v is None for v in vals):
            # Mixed optional values are preserved per-sample so downstream
            # can use depth warp where available and homography otherwise.
            out[key] = vals
        elif all(isinstance(v, torch.Tensor) for v in vals):
            out[key] = torch.stack(vals, dim=0)
        else:
            out[key] = vals
    return out
