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

_ORB_MAX_MATCHES = 500  # cap candidate matches for stable/faster homography fit
_PAIR_MAX_DELTA = 20
_MIN_MATCHES_FOR_H = 30
_MIN_INLIERS_FOR_H = 15


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
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
    if H is None or mask is None:
        return None
    if int(mask.sum()) < _MIN_INLIERS_FOR_H:
        return None
    if not np.isfinite(H).all():
        return None
    return torch.from_numpy(H.astype(np.float32))


def _sample_random_homography(h: int, w: int) -> torch.Tensor:
    angle = np.deg2rad(np.random.uniform(-20.0, 20.0))
    scale = 1.0 + np.random.uniform(-0.2, 0.2)
    tx = np.random.uniform(-0.05, 0.05) * w
    ty = np.random.uniform(-0.05, 0.05) * h
    ca, sa = np.cos(angle), np.sin(angle)
    H = np.array([
        [scale * ca, -scale * sa, tx],
        [scale * sa, scale * ca, ty],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
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

        self.pairs: List[Tuple[str, int, int]] = []
        for scene, items in self.items_by_scene.items():
            n = len(items)
            local = []
            max_delta = min(_PAIR_MAX_DELTA, n - 1)
            for i in range(n):
                for d in range(1, max_delta + 1):
                    j = i + d
                    if j < n:
                        local.append((scene, i, j))
            random.shuffle(local)
            self.pairs.extend(local[: self.max_pairs_per_scene])

        if not self.pairs:
            raise RuntimeError(f"No valid scene pairs found for split={self.split}")

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
        scene, i, j = self.pairs[idx]
        a_img_path, a_depth_path = self.items_by_scene[scene][i]
        b_img_path, b_depth_path = self.items_by_scene[scene][j]

        img1 = self._load_gray(a_img_path)
        img2 = self._load_gray(b_img_path)
        _, valid1 = self._load_depth_and_valid(a_depth_path)
        _, valid2 = self._load_depth_and_valid(b_depth_path)

        H = _estimate_homography(img1, img2)
        if H is None:
            H = _sample_random_homography(self.image_size[0], self.image_size[1])

        if self.augment_geometric and random.random() < 0.2:
            H = _sample_random_homography(self.image_size[0], self.image_size[1]) @ H

        img2 = self._photo_aug(img2)

        warp_field = _build_dense_warp_from_homography(H, self.image_size)
        warp_valid = valid1.squeeze(0).bool()

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
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, drop_last=True, collate_fn=collate_v2)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, drop_last=True, collate_fn=collate_v2)
    return train_loader, val_loader
