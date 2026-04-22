from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF

_ORB_MAX_MATCHES = 500   # cap candidate matches for stable/faster homography fit
_PAIR_MAX_DELTA = 20
_MIN_MATCHES_FOR_H = 30
_MIN_INLIERS_FOR_H = 15
_RANSAC_THRESHOLD = 2.0  # FIX: tightened from 3.0 → 2.0 px for better inlier quality

# Co-visibility filter bounds (FIX: pairs outside this range are rejected).
#
# Pairs with overlap < _OVERLAP_MIN have little or no shared scene content —
# no positive correspondences can be found and sim_gap becomes numerically
# negative by construction.  Pairs with overlap > _OVERLAP_MAX are nearly
# identical — they produce trivially easy positives and no useful negatives,
# which skews metric-learning objectives and inflates repeatability falsely.
# Restricting to [0.30, 0.70] keeps the pair difficulty in the "hard but
# solvable" regime that metric learning requires.
_OVERLAP_MIN = 0.30
_OVERLAP_MAX = 0.70


def _list_images(img_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted([p for p in img_dir.glob("*") if p.suffix.lower() in exts])


def _find_scene_dirs(root: Path) -> Dict[str, Tuple[Path, Path]]:
    out: Dict[str, Tuple[Path, Path]] = {}
    for img_dir in sorted(root.glob("*/dense0/imgs")):
        scene = img_dir.parent.parent.name
        depth_dir = img_dir.parent / "depths"
        if depth_dir.exists():
            out[scene] = (img_dir, depth_dir)
    return out


def _basename_no_ext(p: Path) -> str:
    return p.stem


def _build_aligned_items(img_dir: Path, depth_dir: Path) -> List[Tuple[Path, Path]]:
    imgs = _list_images(img_dir)
    depth_map = {d.stem: d for d in depth_dir.glob("*.h5")}
    items: List[Tuple[Path, Path]] = []
    for img in imgs:
        key = _basename_no_ext(img)
        if key in depth_map:
            items.append((img, depth_map[key]))
    return items


def _estimate_homography(img1: torch.Tensor, img2: torch.Tensor) -> Optional[torch.Tensor]:
    """Estimate a homography between two (1, H, W) grayscale tensors using ORB + RANSAC.

    Returns None if fewer than _MIN_INLIERS_FOR_H inliers are found or if the
    homography is degenerate (non-finite entries).
    """
    a = (img1.squeeze(0).numpy() * 255.0).astype(np.uint8)
    b = (img2.squeeze(0).numpy() * 255.0).astype(np.uint8)
    orb = cv2.ORB_create(nfeatures=2000)
    k1, d1 = orb.detectAndCompute(a, None)
    k2, d2 = orb.detectAndCompute(b, None)
    if d1 is None or d2 is None:
        return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    m = sorted(bf.match(d1, d2), key=lambda x: x.distance)
    if len(m) < _MIN_MATCHES_FOR_H:
        return None
    pts1 = np.float32([k1[x.queryIdx].pt for x in m[:_ORB_MAX_MATCHES]]).reshape(-1, 1, 2)
    pts2 = np.float32([k2[x.trainIdx].pt for x in m[:_ORB_MAX_MATCHES]]).reshape(-1, 1, 2)
    # FIX: RANSAC threshold tightened from 3.0 → 2.0 px for better inlier quality.
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, _RANSAC_THRESHOLD)
    if H is None or mask is None:
        return None
    if int(mask.sum()) < _MIN_INLIERS_FOR_H:
        return None
    if not np.isfinite(H).all():
        return None
    return torch.from_numpy(H.astype(np.float32))


def _estimate_overlap_ratio(H: torch.Tensor, image_hw: Tuple[int, int]) -> float:
    """Estimate the fraction of img1 pixels visible in img2 after applying H.

    Projects a 10×10 grid of sample points from image-1 space into image-2
    space and returns the fraction that land within the image-2 bounds.

    Args:
        H: (3, 3) homography tensor mapping img1 coords → img2 coords.
        image_hw: (height, width) of both images (assumed equal after resize).

    Returns:
        Overlap ratio in [0, 1].
    """
    h, w = image_hw
    ys = np.linspace(0, h - 1, 10)
    xs = np.linspace(0, w - 1, 10)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.stack([xx.ravel(), yy.ravel(), np.ones(100)], axis=0)  # (3, 100)
    H_np = H.numpy().astype(np.float64)
    warped = H_np @ pts  # (3, 100)
    z = np.abs(warped[2]).clip(1e-8)
    xy = warped[:2] / z  # (2, 100)
    in_bounds = (xy[0] >= 0) & (xy[0] < w) & (xy[1] >= 0) & (xy[1] < h)
    return float(in_bounds.mean())


def _sample_random_homography(h: int, w: int) -> torch.Tensor:
    """Sample a random near-planar homography (kept for reference / ablations)."""
    angle = np.deg2rad(np.random.uniform(-20.0, 20.0))
    scale = 1.0 + np.random.uniform(-0.2, 0.2)
    tx = np.random.uniform(-0.05, 0.05) * w
    ty = np.random.uniform(-0.05, 0.05) * h
    ca, sa = np.cos(angle), np.sin(angle)
    H = np.array(
        [
            [scale * ca, -scale * sa, tx],
            [scale * sa, scale * ca, ty],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return torch.from_numpy(H)


def _build_dense_warp_from_homography(H: torch.Tensor, image_hw: Tuple[int, int]) -> torch.Tensor:
    h, w = image_hw
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    ones = torch.ones_like(xx)
    grid = torch.stack([xx, yy, ones], dim=0).reshape(3, -1).float()
    warped = H.float() @ grid
    z = warped[2:].clamp(min=1e-8)
    xy = (warped[:2] / z).T.reshape(h, w, 2)
    return xy


class MegaDepthRawDatasetV2(Dataset):
    """MegaDepth pair dataset with co-visibility filtering and pre-computed homographies.

    Key changes vs. earlier version
    --------------------------------
    1. Co-visibility filter (FIX for negative sim_gap):
       Pairs are pre-filtered during ``__init__`` so that only pairs whose
       overlap ratio lies strictly in [_OVERLAP_MIN, _OVERLAP_MAX] enter the
       training pool.  Zero-overlap pairs produce no positive correspondences,
       making sim_gap numerically negative by construction.  Near-identical
       pairs are too easy and skew metric-learning objectives.

    2. Pre-computed homographies:
       The ORB homography is now computed once per pair during ``__init__``
       (with a per-scene image cache to avoid redundant disk reads) and stored
       in the pair tuple.  ``__getitem__`` uses the stored H directly — no ORB
       at training time.

    3. Geometric augmentation composition removed (FIX for low repeatability):
       The 0.2-probability ``H = H_random @ H_orb`` composition is gone.
       Composing the ORB-estimated H (already an approximation for a 3D scene)
       with a synthetic random H further degrades correspondence quality and
       keeps repeatability near zero.  Photometric augmentation on img2 is
       retained — it does not break geometric consistency.
    """

    def __init__(
        self,
        root: str,
        split: str,
        train_scenes: Sequence[str],
        val_scenes: Sequence[str],
        image_size: Tuple[int, int] = (480, 640),
        depth_h5_key: str = "depth",
        max_pairs_per_scene: int = 1000,
        augment_photometric: bool = True,
        augment_geometric: bool = True,
    ):
        self.root = Path(root)
        self.split = str(split).lower()
        self.train_scenes = {str(s) for s in train_scenes}
        self.val_scenes = {str(s) for s in val_scenes}
        self.image_size = tuple(image_size)
        self.depth_h5_key = str(depth_h5_key)
        self.max_pairs_per_scene = int(max_pairs_per_scene)
        self.augment_photometric = bool(augment_photometric)
        self.augment_geometric = bool(augment_geometric)

        all_scenes = _find_scene_dirs(self.root)
        scene_pool = self.train_scenes if self.split == "train" else self.val_scenes

        self.items_by_scene: Dict[str, List[Tuple[Path, Path]]] = {}
        for scene in sorted(scene_pool):
            if scene not in all_scenes:
                continue
            img_dir, depth_dir = all_scenes[scene]
            aligned = _build_aligned_items(img_dir, depth_dir)
            if len(aligned) >= 2:
                self.items_by_scene[scene] = aligned

        # FIX: Build pairs with co-visibility filtering.
        # Pair tuples now store the pre-computed H: (scene, i, j, H).
        # Only pairs with _OVERLAP_MIN ≤ overlap ≤ _OVERLAP_MAX are accepted.
        self.pairs: List[Tuple[str, int, int, torch.Tensor]] = []
        total_candidates = 0
        total_accepted = 0

        for scene, items in self.items_by_scene.items():
            n = len(items)
            candidates: List[Tuple[int, int]] = []
            max_delta = min(_PAIR_MAX_DELTA, n - 1)
            for i in range(n):
                for d in range(1, max_delta + 1):
                    j = i + d
                    if j < n:
                        candidates.append((i, j))
            random.shuffle(candidates)
            total_candidates += len(candidates)

            # Per-scene image cache — avoids re-loading the same image for
            # multiple candidate pairs.  Cleared after each scene to bound
            # peak memory usage.
            img_cache: Dict[int, torch.Tensor] = {}

            accepted: List[Tuple[str, int, int, torch.Tensor]] = []
            for i, j in candidates:
                if len(accepted) >= self.max_pairs_per_scene:
                    break

                # Load images with per-scene cache
                if i not in img_cache:
                    img_cache[i] = self._load_gray(items[i][0])
                if j not in img_cache:
                    img_cache[j] = self._load_gray(items[j][0])

                H = _estimate_homography(img_cache[i], img_cache[j])
                if H is None:
                    continue

                overlap = _estimate_overlap_ratio(H, self.image_size)
                if _OVERLAP_MIN <= overlap <= _OVERLAP_MAX:
                    accepted.append((scene, i, j, H))

            total_accepted += len(accepted)
            self.pairs.extend(accepted)
            img_cache.clear()  # free per-scene cache

        if not self.pairs:
            raise RuntimeError(
                f"No valid scene pairs found for split={self.split!r} after co-visibility "
                f"filtering (overlap range [{_OVERLAP_MIN:.2f}, {_OVERLAP_MAX:.2f}]). "
                f"Checked {total_candidates} candidates, accepted {total_accepted}. "
                f"Scenes available: {list(self.items_by_scene.keys())}"
            )

        print(
            f"[MegaDepthRaw] split={self.split!r}: accepted {len(self.pairs)}/{total_candidates} "
            f"pairs after co-visibility filter "
            f"(overlap ∈ [{_OVERLAP_MIN:.2f}, {_OVERLAP_MAX:.2f}])"
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_gray(self, path: Path) -> torch.Tensor:
        im = Image.open(path).convert("L")
        im = TF.resize(im, list(self.image_size), interpolation=TF.InterpolationMode.BILINEAR)
        return TF.to_tensor(im).float().clamp(0.0, 1.0)

    def _load_depth_and_valid(self, path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        with h5py.File(path, "r") as f:
            if self.depth_h5_key not in f:
                raise KeyError(f"Depth key '{self.depth_h5_key}' not found in {path}")
            depth = torch.from_numpy(f[self.depth_h5_key][:].astype(np.float32))
        if depth.dim() == 2:
            depth = depth.unsqueeze(0)
        depth = F.interpolate(depth.unsqueeze(0), size=self.image_size, mode="nearest").squeeze(0)
        valid = depth > 0.0
        return depth, valid

    def _photo_aug(self, img: torch.Tensor) -> torch.Tensor:
        if not self.augment_photometric:
            return img
        pil = TF.to_pil_image(img)
        if random.random() < 0.7:
            factor = random.uniform(0.7, 1.3)
            pil = ImageEnhance.Brightness(pil).enhance(factor)
        if random.random() < 0.35:
            pil = pil.filter(ImageFilter.GaussianBlur(radius=float(random.randint(0, 2))))
        if random.random() < 0.25:
            pil = pil.filter(ImageFilter.BoxBlur(radius=int(random.randint(1, 2))))
        out = TF.to_tensor(pil).float()
        if random.random() < 0.5:
            noise = torch.randn_like(out) * random.uniform(0.005, 0.03)
            out = (out + noise).clamp(0.0, 1.0)
        return out

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # FIX: Pair tuples now include the pre-computed H (computed and
        # overlap-filtered at __init__ time).  No ORB is run here.
        scene, i, j, H = self.pairs[idx]
        a_img_path, a_depth_path = self.items_by_scene[scene][i]
        b_img_path, b_depth_path = self.items_by_scene[scene][j]

        img1 = self._load_gray(a_img_path)
        img2 = self._load_gray(b_img_path)
        _, valid1 = self._load_depth_and_valid(a_depth_path)
        _, valid2 = self._load_depth_and_valid(b_depth_path)

        # Use the pre-stored ORB homography directly.
        # Photometric augmentation on img2 is retained — it does not break
        # geometric consistency.
        img2 = self._photo_aug(img2)

        warp_field = _build_dense_warp_from_homography(H, self.image_size)

        # FIX (critical): Do NOT use depth validity as warp_valid.
        #
        # The ORB-estimated homography is a 2-D geometric transform — it does
        # not use depth.  Depth validity (depth > 0) tells you whether MVS
        # reconstructed a 3-D point at that pixel; it says nothing about
        # whether the ORB homography maps that pixel correctly.
        #
        # Applying the depth mask here eliminates ~97–99 % of image-1
        # keypoints for typical MegaDepth outdoor scenes (where < 3 % of
        # pixels have valid depth reconstructions), leaving n_pos ≈ 1.  With
        # n_pos = 1 the single surviving "positive" is often spurious
        # (pos_sim ≈ 0.003 < neg_sim ≈ 0.105), so sim_gap is permanently
        # negative and no descriptor signal reaches the network.
        #
        # The safe_radius border check inside _build_correspondence_from_warp
        # is the correct out-of-bounds guard for homography-based
        # correspondences.  No additional depth mask is needed.
        warp_valid = torch.ones(
            self.image_size[0], self.image_size[1], dtype=torch.bool
        )

        return {
            "scene": scene,
            "image1": img1,
            "image2": img2,
            "depth_valid1": valid1.float(),
            "depth_valid2": valid2.float(),
            "homography": H.float(),
            "warp_field": warp_field.float(),
            "warp_valid": warp_valid,
        }


def collate_v2(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    keys = set().union(*[b.keys() for b in batch])
    for k in keys:
        vals = [b.get(k) for b in batch]
        if all(isinstance(v, torch.Tensor) for v in vals):
            out[k] = torch.stack(vals, dim=0)
        else:
            out[k] = vals
    return out


def build_dataloaders_v2(cfg: Dict[str, object]) -> Tuple[DataLoader, DataLoader]:
    image_size = (int(cfg["image_height"]), int(cfg["image_width"]))
    common = dict(
        root=str(cfg["data_root"]),
        train_scenes=cfg.get("train_scenes", ["0001", "0002"]),
        val_scenes=cfg.get("val_scenes", ["0003"]),
        image_size=image_size,
        depth_h5_key=str(cfg.get("depth_h5_key", "depth")),
        max_pairs_per_scene=int(cfg.get("max_pairs_per_scene", 1000)),
        augment_photometric=bool(cfg.get("augment_photometric", True)),
        augment_geometric=bool(cfg.get("augment_geometric", True)),
    )

    train_ds = MegaDepthRawDatasetV2(split="train", **common)
    val_ds = MegaDepthRawDatasetV2(
        split="val",
        root=common["root"],
        train_scenes=common["train_scenes"],
        val_scenes=common["val_scenes"],
        image_size=common["image_size"],
        depth_h5_key=common["depth_h5_key"],
        max_pairs_per_scene=max(1, int(cfg.get("max_pairs_per_scene", 1000) // 2)),
        augment_photometric=False,
        augment_geometric=False,
    )

    nw = int(cfg.get("num_workers", 0))
    bs = int(cfg.get("batch_size", 4))
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=nw,
        drop_last=True, collate_fn=collate_v2,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, num_workers=nw,
        drop_last=True, collate_fn=collate_v2,
    )
    return train_loader, val_loader