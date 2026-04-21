from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_FEATURE_STRIDE = 8


class DifferentiableDescriptorSampler(nn.Module):
    """Bicubic grid-sampler that extracts per-keypoint descriptors from a dense map.

    Gradient path
    -------------
    SuperPoint is frozen, so ``descriptor_map`` has ``requires_grad=False``.
    The only training signal that reaches the XFeat keypoint head flows through
    the score-weight term ``w = scores1 * scores2`` inside the loss, *not*
    through the descriptor values.  This is correct by design: we are training
    the detector to predict repeatable keypoint positions, not to change the
    descriptor space (which is fixed to SuperPoint's learned space).

    AMP safety
    ----------
    ``F.grid_sample`` with ``mode='bicubic'`` requires matching float32 dtypes
    for both the input feature map and the sampling grid.  Under PyTorch AMP
    autocast, tensors can silently be in float16.  Bicubic interpolation in
    float16 is notoriously unstable and can produce NaN values that silently
    corrupt the entire backward pass.

    Primary guard: ``extract_superpoint_desc_map`` in adapters.py casts the
    descriptor map to float32 before returning it.

    Secondary guard (here): an explicit ``descriptor_map.float()`` cast with a
    dtype assertion fires *after* any autocast promotion so that even if the
    primary guard is ever relaxed, this sampler remains safe.

    L2 normalisation
    ----------------
    The 256-D output descriptors are L2-normalised before returning, which is
    required for cosine-similarity-based matching (e.g. LightGlue).  Feeding
    unnormalized descriptors to LightGlue causes gradient explosion/vanishing
    and prevents sim_gap from separating correctly.
    """

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

    def forward(
        self,
        keypoints_px: torch.Tensor,
        descriptor_map: torch.Tensor,
        image_hw: Tuple[int, int],
    ) -> torch.Tensor:
        if keypoints_px.numel() == 0:
            c = int(descriptor_map.shape[1])
            return torch.zeros((0, c), device=descriptor_map.device, dtype=torch.float32)

        # FIX: Secondary AMP guard — cast to float32 before F.grid_sample.
        # bicubic grid_sample requires float32; float16 inputs produce NaN.
        desc_map_fp32 = descriptor_map.float()
        if desc_map_fp32.dtype != torch.float32:
            raise RuntimeError(
                f"descriptor_map must be float32 before grid_sample, "
                f"got {desc_map_fp32.dtype}"
            )

        grid = self._pixels_to_grid(keypoints_px.float(), image_hw)
        sampled = F.grid_sample(
            desc_map_fp32,
            grid,
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=True,
        )
        sampled = sampled.squeeze(0).squeeze(1).T
        # L2-normalise: required for cosine similarity in LightGlue attention layers.
        return F.normalize(sampled, p=2, dim=-1)