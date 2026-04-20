from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_FEATURE_STRIDE = 8


class DifferentiableDescriptorSampler(nn.Module):
    def __init__(self, mode: str = "bicubic", padding_mode: str = "border"):
        super().__init__()
        self.mode = mode
        self.padding_mode = padding_mode

    @staticmethod
    def _pixels_to_grid(keypoints_px: torch.Tensor, image_hw: Tuple[int, int]) -> torch.Tensor:
        h, w = image_hw
        hc, wc = h // _FEATURE_STRIDE, w // _FEATURE_STRIDE
        x = keypoints_px[:, 0] / float(_FEATURE_STRIDE)
        y = keypoints_px[:, 1] / float(_FEATURE_STRIDE)
        x = 2.0 * x / max(wc - 1, 1) - 1.0
        y = 2.0 * y / max(hc - 1, 1) - 1.0
        return torch.stack([x, y], dim=-1).unsqueeze(0).unsqueeze(0)

    def forward(self, keypoints_px: torch.Tensor, descriptor_map: torch.Tensor, image_hw: Tuple[int, int]) -> torch.Tensor:
        if keypoints_px.numel() == 0:
            c = int(descriptor_map.shape[1])
            return torch.zeros((0, c), device=descriptor_map.device, dtype=descriptor_map.dtype)

        grid = self._pixels_to_grid(keypoints_px.float(), image_hw)
        sampled = F.grid_sample(
            descriptor_map.float(),
            grid,
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=True,
        )
        sampled = sampled.squeeze(0).squeeze(1).T
        return F.normalize(sampled, p=2, dim=-1)
