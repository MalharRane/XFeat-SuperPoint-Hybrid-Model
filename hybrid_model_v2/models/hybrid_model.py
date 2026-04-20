from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adapters import extract_superpoint_desc_map
from .sampler import DifferentiableDescriptorSampler


class HybridModelV2(nn.Module):
    # Grayscale normalization constants used by the XFeat preprocessing path.
    # These are the single-channel equivalents expected by the upstream model.
    _XFEAT_MEAN = 0.485
    _XFEAT_STD = 0.229

    def __init__(
        self,
        xfeat_core: nn.Module,
        superpoint_core: nn.Module,
        num_keypoints: int = 1024,
        nms_radius: int = 4,
        min_keypoint_score: float = 0.01,
        descriptor_dim: int = 256,
        border_margin: int = 16,
        xfeat_kp_head_attrs: Sequence[str] = ("kp_head", "keypoint_head", "det_head"),
    ):
        super().__init__()
        self.xfeat = xfeat_core
        self.superpoint = superpoint_core

        self.num_keypoints = int(num_keypoints)
        self.nms_radius = int(nms_radius)
        self.min_keypoint_score = float(min_keypoint_score)
        self.descriptor_dim = int(descriptor_dim)
        self.border_margin = int(border_margin)

        self.register_buffer("xfeat_mean", torch.tensor(self._XFEAT_MEAN).view(1, 1, 1, 1))
        self.register_buffer("xfeat_std", torch.tensor(self._XFEAT_STD).view(1, 1, 1, 1))

        self.sampler = DifferentiableDescriptorSampler(mode="bicubic")

        self._freeze_superpoint()
        self._configure_xfeat_trainable_kp_head(tuple(xfeat_kp_head_attrs))

    def train(self, mode: bool = True):
        super().train(mode)
        self.superpoint.eval()  # strict guard: keep frozen SP in eval always
        return self

    def _freeze_superpoint(self) -> None:
        for p in self.superpoint.parameters():
            p.requires_grad_(False)
        self.superpoint.eval()

    def _configure_xfeat_trainable_kp_head(self, attrs: Tuple[str, ...]) -> None:
        for p in self.xfeat.parameters():
            p.requires_grad_(False)

        for attr in attrs:
            mod = self.xfeat
            ok = True
            for part in attr.split("."):
                if not hasattr(mod, part):
                    ok = False
                    break
                mod = getattr(mod, part)
            if ok and isinstance(mod, nn.Module):
                for p in mod.parameters():
                    p.requires_grad_(True)
                return

        # fallback: search by name keywords
        activated = False
        for n, m in self.xfeat.named_modules():
            if any(k in n.lower() for k in ("kp", "key", "det")):
                for p in m.parameters():
                    p.requires_grad_(True)
                activated = True
        if not activated:
            raise RuntimeError("Could not isolate XFeat keypoint head; aborting by design in V2")

    def unfreeze_xfeat_modules(self, keywords: Sequence[str]) -> int:
        kws = [k.strip().lower() for k in keywords if str(k).strip()]
        newly = 0
        if not kws:
            return newly
        for name, p in self.xfeat.named_parameters():
            lname = name.lower()
            if any(k in lname for k in kws) and not p.requires_grad:
                p.requires_grad_(True)
                newly += p.numel()
        return newly

    def _normalize_for_xfeat(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.xfeat_mean) / self.xfeat_std

    @staticmethod
    def _normalize_for_superpoint(x: torch.Tensor) -> torch.Tensor:
        return x.clamp(0.0, 1.0)

    @staticmethod
    def _nms(scores: torch.Tensor, radius: int) -> torch.Tensor:
        if radius <= 0:
            return scores
        max_pool = F.max_pool2d(scores, kernel_size=2 * radius + 1, stride=1, padding=radius)
        keep = (scores == max_pool).float()
        return scores * keep

    def _decode_xfeat_heatmap(self, logits: torch.Tensor, image_hw: Tuple[int, int]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        b, c, h8, w8 = logits.shape
        h, w = image_hw

        if c == 65:
            prob = F.softmax(logits, dim=1)[:, :-1]
        elif c == 64:
            prob = torch.sigmoid(logits)
        elif c == 1:
            prob = torch.sigmoid(logits).repeat(1, 64, 1, 1)
        else:
            raise RuntimeError(f"Unsupported XFeat keypoint channels: {c}")

        heat = F.pixel_shuffle(prob, upscale_factor=8)
        if heat.shape[-2:] != (h, w):
            heat = F.interpolate(heat, size=(h, w), mode="bilinear", align_corners=False)

        if self.border_margin > 0:
            m = self.border_margin
            mask = torch.ones_like(heat)
            mask[..., :m, :] = 0
            mask[..., -m:, :] = 0
            mask[..., :, :m] = 0
            mask[..., :, -m:] = 0
            heat = heat * mask

        heat = self._nms(heat, self.nms_radius)

        keypoints_px: List[torch.Tensor] = []
        scores: List[torch.Tensor] = []
        for bi in range(b):
            sm = heat[bi, 0]
            ys, xs = torch.where(sm > self.min_keypoint_score)
            if xs.numel() == 0:
                keypoints_px.append(torch.empty((0, 2), device=logits.device, dtype=logits.dtype))
                scores.append(torch.empty((0,), device=logits.device, dtype=logits.dtype))
                continue
            sc = sm[ys, xs]
            if sc.numel() > self.num_keypoints:
                top = torch.topk(sc, k=self.num_keypoints, largest=True).indices
                ys, xs, sc = ys[top], xs[top], sc[top]
            kp = torch.stack([xs.float(), ys.float()], dim=-1)
            keypoints_px.append(kp)
            scores.append(sc)
        return keypoints_px, scores

    def _call_xfeat_forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.xfeat(x)
        if isinstance(out, dict):
            for key in ("keypoints", "heatmap", "logits"):
                if key in out and isinstance(out[key], torch.Tensor):
                    return out[key]
        if isinstance(out, torch.Tensor):
            return out

        # wrapper APIs -> convert sparse points to heatmap
        for method in ("detectAndCompute", "detect_and_compute", "extract", "detect"):
            if hasattr(self.xfeat, method):
                data = getattr(self.xfeat, method)(x)
                if isinstance(data, dict) and "keypoints" in data:
                    return self._sparse_to_dense_heatmap(data, x.shape[0], x.shape[-2:])
        raise RuntimeError("Unsupported XFeat API for V2")

    @staticmethod
    def _sparse_to_dense_heatmap(data: Dict[str, Any], b: int, image_hw: Tuple[int, int]) -> torch.Tensor:
        h, w = image_hw
        heat = torch.zeros((b, 1, h, w), dtype=torch.float32)
        kps = data.get("keypoints", [torch.empty(0, 2)] * b)
        scores = data.get("scores", [torch.empty(0)] * b)
        for i in range(min(b, len(kps))):
            kp = kps[i]
            sc = scores[i] if i < len(scores) else None
            if not isinstance(kp, torch.Tensor) or kp.numel() == 0:
                continue
            xy = kp.long().clamp(min=0)
            xy[:, 0] = xy[:, 0].clamp(max=w - 1)
            xy[:, 1] = xy[:, 1].clamp(max=h - 1)
            if isinstance(sc, torch.Tensor) and sc.numel() == xy.shape[0]:
                heat[i, 0, xy[:, 1], xy[:, 0]] = sc.float().clamp(0, 1)
            else:
                heat[i, 0, xy[:, 1], xy[:, 0]] = 1.0
        return F.avg_pool2d(heat, kernel_size=8, stride=8)

    def forward_train(self, image: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        b, c, h, w = image.shape
        if c != 1:
            raise RuntimeError(f"Input must be grayscale (B,1,H,W), got shape {tuple(image.shape)}")
        if h % 8 != 0 or w % 8 != 0:
            raise RuntimeError("Input H and W must be divisible by 8")

        self.superpoint.eval()
        xfeat_logits = self._call_xfeat_forward(self._normalize_for_xfeat(image))
        keypoints_px, scores = self._decode_xfeat_heatmap(xfeat_logits, (h, w))

        desc_map = extract_superpoint_desc_map(self.superpoint, self._normalize_for_superpoint(image)).float()

        keypoints: List[torch.Tensor] = []
        descriptors: List[torch.Tensor] = []
        for i in range(b):
            kp_px = keypoints_px[i]
            desc = self.sampler(kp_px, desc_map[i:i + 1], (h, w))
            if desc.shape[1] != self.descriptor_dim:
                raise RuntimeError(f"Descriptor dimension mismatch: got {desc.shape[1]}, expected {self.descriptor_dim}")
            kp = kp_px / kp_px.new_tensor([max(w - 1, 1), max(h - 1, 1)]) if kp_px.numel() > 0 else kp_px
            keypoints.append(kp)
            descriptors.append(desc)

        return {
            "keypoints": keypoints,
            "keypoints_px": keypoints_px,
            "descriptors": descriptors,
            "scores": scores,
            "heatmap": xfeat_logits,
        }

    def export_descriptors(self, descriptors: List[torch.Tensor], layout: str = "BN256") -> torch.Tensor:
        if not descriptors:
            return torch.empty(0, 0, self.descriptor_dim)
        max_n = max(d.shape[0] for d in descriptors)
        b = len(descriptors)
        out = descriptors[0].new_zeros((b, max_n, self.descriptor_dim))
        for i, d in enumerate(descriptors):
            out[i, : d.shape[0]] = d
        if layout.upper() == "B256N":
            return out.transpose(1, 2).contiguous()
        return out
