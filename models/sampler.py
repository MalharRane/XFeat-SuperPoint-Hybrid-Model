"""
models/sampler.py
=================
DifferentiableDescriptorSampler
--------------------------------
Bridges XFeat's pixel-space keypoints to SuperPoint's H/8×W/8 descriptor
grid through a fully-differentiable bicubic spatial sampling operation.

Mathematical Flow
-----------------

  Pixel coord  (x_px, y_px)  ∈ [0, W-1] × [0, H-1]

       Step 1 — Scale by 1/8 (match descriptor grid resolution)
  ─────────────────────────────────────────────────────────────────
  x_desc = x_px / 8  ∈ [0, (W/8) - 1]
  y_desc = y_px / 8  ∈ [0, (H/8) - 1]

       Step 2 — Normalize to PyTorch grid_sample [-1, 1] range
  ─────────────────────────────────────────────────────────────────
  x_norm = 2 · x_desc / ((W/8) - 1)  -  1
  y_norm = 2 · y_desc / ((H/8) - 1)  -  1

  With align_corners=True:  -1 maps to pixel-centre of index 0,
                            +1 maps to pixel-centre of index Wc-1.

       Step 3 — Differentiable bicubic interpolation
  ─────────────────────────────────────────────────────────────────
  desc = grid_sample(desc_map, grid, mode='bicubic', align_corners=True)

       Step 4 — L2 normalisation  (unit hypersphere)
  ─────────────────────────────────────────────────────────────────
  desc = desc / ||desc||_2

Gradient Path
-------------
  Loss  →  L2-norm  →  grid_sample  →  XFeat heatmap logits
                             ↑
                       (frozen SP map, detached)

The gradient with respect to XFeat's logits flows through the
*coordinate* computation (via the differentiable heatmap weighting),
not through the descriptor values themselves.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DifferentiableDescriptorSampler(nn.Module):
    """
    Differentiable bicubic descriptor sampler.

    Parameters
    ----------
    mode : str
        Interpolation mode for ``F.grid_sample``.
        'bicubic' gives sub-pixel accuracy matching SuperPoint's own
        descriptor decoder (which also uses bicubic interpolation).
    padding_mode : str
        How to handle out-of-bounds coordinates.
        'border' replicates the border value, preventing gradient spikes.
    """

    def __init__(
        self,
        mode: str = 'bicubic',
        padding_mode: str = 'border',
    ):
        super().__init__()
        assert mode in ('bicubic', 'bilinear', 'nearest'), \
            f"mode must be 'bicubic', 'bilinear', or 'nearest', got '{mode}'"
        self.mode         = mode
        self.padding_mode = padding_mode

    # ------------------------------------------------------------------
    # Coordinate transformation
    # ------------------------------------------------------------------

    @staticmethod
    def pixels_to_norm_grid(
        keypoints_px: torch.Tensor,
        image_hw: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Convert pixel-space keypoints to the normalised [-1, 1] grid
        coordinates expected by ``F.grid_sample`` on a H/8×W/8 feature map.

        Two-step transformation
        -----------------------
        1. Divide pixel coords by 8  →  descriptor-grid space
        2. Affine map to [-1, 1] using align_corners=True convention

        Args
        ----
        keypoints_px : (N, 2) tensor  [x, y]  in pixel space
        image_hw     : (H, W) original image dimensions

        Returns
        -------
        grid : (1, 1, N, 2) tensor of normalised (x, y) coords
               Format matches F.grid_sample input shape (B, H_out, W_out, 2)
        """
        H, W = image_hw
        Hc   = H // 8          # descriptor map height
        Wc   = W // 8          # descriptor map width

        # ── Step 1: scale to descriptor-grid space ──────────────────────
        # Each descriptor "cell" covers an 8×8 pixel block
        x_desc = keypoints_px[:, 0] / 8.0          # (N,)  ∈ [0, Wc-1]
        y_desc = keypoints_px[:, 1] / 8.0          # (N,)  ∈ [0, Hc-1]

        # ── Step 2: normalise to [-1, 1] ────────────────────────────────
        # PyTorch grid_sample with align_corners=True:
        #   grid value = -1.0 → sample from index 0   (left/top edge)
        #   grid value = +1.0 → sample from index Wc-1 (right/bottom edge)
        # Formula: norm = 2 * idx / (dim - 1) - 1
        x_norm = 2.0 * x_desc / (Wc - 1) - 1.0    # (N,)  ∈ [-1, 1]
        y_norm = 2.0 * y_desc / (Hc - 1) - 1.0    # (N,)  ∈ [-1, 1]

        # ── Assemble into grid_sample format ────────────────────────────
        # grid_sample expects (B, H_out, W_out, 2) where last dim = (x, y)
        # We model N keypoints as a (1, 1, N) spatial "grid"
        grid = torch.stack([x_norm, y_norm], dim=-1)    # (N, 2)
        grid = grid.unsqueeze(0).unsqueeze(0)            # (1, 1, N, 2)

        return grid

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        keypoints_px:   torch.Tensor,
        descriptor_map: torch.Tensor,
        image_hw:       Tuple[int, int],
    ) -> torch.Tensor:
        """
        Sample and L2-normalise descriptors from SuperPoint's feature map
        at the continuous sub-pixel locations specified by XFeat's keypoints.

        Args
        ----
        keypoints_px   : (N, 2)       XFeat keypoints  [x, y]  in pixel space
        descriptor_map : (1, 256, Hc, Wc)  SuperPoint pre-upsampled desc map
        image_hw       : (H, W)        original image dimensions

        Returns
        -------
        descriptors : (N, 256)  L2-normalised descriptor vectors
        """
        N = keypoints_px.shape[0]
        desc_dim = descriptor_map.shape[1]      # 256

        # ── Edge case: no keypoints ──────────────────────────────────────
        if N == 0:
            return torch.zeros(
                0, desc_dim,
                device=descriptor_map.device,
                dtype=descriptor_map.dtype
            )

        # ── Step 1-2: pixel coords → normalised grid ────────────────────
        grid = self.pixels_to_norm_grid(keypoints_px, image_hw)
        # grid: (1, 1, N, 2)

        # ── Step 3: differentiable bicubic sampling ──────────────────────
        #
        # descriptor_map : (1, 256, Hc, Wc)   frozen SuperPoint features
        # grid           : (1, 1,  N,  2)     query locations in [-1, 1]
        # output         : (1, 256, 1,  N)
        #
        # Gradient note:
        #   • grid_sample differentiates w.r.t. *both* input and grid.
        #   • descriptor_map has no gradient (SuperPoint is frozen).
        #   • grid is computed from keypoints_px, which are derived from
        #     XFeat's heatmap → gradient flows back to XFeat kp head. ✓
        #
        # AMP note: grid is float32 (keypoints are always cast via .float()).
        # descriptor_map must also be float32: F.grid_sample requires matching
        # dtypes and bicubic mode is numerically unstable in float16.
        sampled = F.grid_sample(
            descriptor_map.float(),   # (1, 256, Hc, Wc) — ensure float32
            grid,                     # (1, 1,  N,  2)   — already float32
            mode=self.mode,           # 'bicubic'
            padding_mode=self.padding_mode,  # 'border'
            align_corners=True,       # consistent with SuperPoint's decoder
        )
        # sampled: (1, 256, 1, N)

        # ── Reshape to (N, 256) ─────────────────────────────────────────
        sampled = sampled.squeeze(0).squeeze(1)   # (256, N)
        sampled = sampled.T                        # (N, 256)

        # ── Step 4: strict L2-normalisation ─────────────────────────────
        # Projects each descriptor onto the unit 256-sphere.
        # eps prevents division-by-zero for near-zero vectors.
        descriptors = F.normalize(sampled, p=2, dim=-1)   # (N, 256)

        return descriptors

    # ------------------------------------------------------------------
    # Batch convenience wrapper
    # ------------------------------------------------------------------

    def forward_batch(
        self,
        keypoints_px_list:  list,
        descriptor_map:     torch.Tensor,
        image_hw:           Tuple[int, int],
    ) -> list:
        """
        Process a whole batch, returning one descriptor tensor per image.

        Args
        ----
        keypoints_px_list : list[Tensor(N_b, 2)]  one set of kps per image
        descriptor_map    : (B, 256, Hc, Wc)
        image_hw          : (H, W)

        Returns
        -------
        list[Tensor(N_b, 256)]  L2-normalised descriptors
        """
        B = descriptor_map.shape[0]
        assert len(keypoints_px_list) == B, \
            "len(keypoints_px_list) must equal batch size"

        results = []
        for b in range(B):
            desc = self.forward(
                keypoints_px_list[b],
                descriptor_map[b:b+1],
                image_hw,
            )
            results.append(desc)

        return results
