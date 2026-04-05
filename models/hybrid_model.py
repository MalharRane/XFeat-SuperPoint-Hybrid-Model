"""
models/hybrid_model.py
======================
HybridModel: XFeat keypoint detection + SuperPoint descriptor manifold.

Architecture Overview
---------------------
                         ┌─────────────────────────────┐
  Grayscale Image ───────┤  XFeat Stream (active grad) ├──► Keypoint Heatmap
  (B, 1, H, W)          │  Normalize → XFeatCore      │    (B, 65, H/8, W/8)
                         └─────────────────────────────┘
         │                                                         │
         │               ┌─────────────────────────────┐           │  decode
         └───────────────┤  SP Stream  (frozen grad)   ├──► Desc Map (B,256,H/8,W/8)
                         │  Normalize → SuperPointCore │           │
                         └─────────────────────────────┘           │
                                                                    │
                         ┌──────────────────────────────────────────┘
                         │  Sampling Interconnect
                         │   kp_px × (1/8) → desc-grid coords
                         │   → normalize [-1,1] → grid_sample (bicubic)
                         │   → L2-normalize
                         └──► {keypoints, descriptors}  (LightGlue payload)

Gradient Flow
-------------
  HomographyHingeLoss
        │
        ▼  (gradients)
  XFeat Keypoint Head  ← only these weights update
  SuperPoint Backbone  ← completely frozen

References
----------
  XFeat   : Potje et al., CVPR 2024
  SuperPt  : DeTone et al., CVPRW 2018
  LightGlue: Lindenberger et al., ICCV 2023
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union

from .sampler import DifferentiableDescriptorSampler


# ---------------------------------------------------------------------------
# Helper: gradient-isolation utilities
# ---------------------------------------------------------------------------

def _count_params(module: nn.Module, trainable_only: bool = False) -> int:
    """Count total or trainable parameters in a module."""
    return sum(
        p.numel() for p in module.parameters()
        if (not trainable_only) or p.requires_grad
    )


# ---------------------------------------------------------------------------
# Main HybridModel
# ---------------------------------------------------------------------------

class HybridModel(nn.Module):
    """
    Hybrid neural network: XFeat detection speed × SuperPoint descriptor quality.

    Parameters
    ----------
    xfeat_core : nn.Module
        Instantiated XFeat backbone.  Expected to expose
        ``forward(x)`` returning either:
          • a dict  {'keypoints': Tensor(B,65,Hc,Wc), ...}
          • a tuple  (Tensor(B,65,Hc,Wc), ...)
          • raw Tensor(B,65,Hc,Wc)
        where the first / 'keypoints' element is the 65-channel heatmap.

    superpoint_core : nn.Module
        Instantiated SuperPoint backbone.  Expected to expose
        ``get_descriptor_map(x)`` → Tensor(B,256,Hc,Wc)  **before**
        any spatial upsampling.  If this method is absent the module
        falls back to a registered forward hook (see ``_install_sp_hook``).

    num_keypoints : int
        Maximum keypoints to extract per image.

    nms_radius : int
        Half-width of the NMS window (in pixels on the full-res heatmap).

    descriptor_dim : int
        Dimensionality of SuperPoint descriptors (256 by default).

    xfeat_kp_head_attr : str
        Attribute name(s) for the XFeat keypoint head sub-module.
        The model tries these names in order and unfreezes the first match.
    """

    # ------------------------------------------------------------------
    # Normalization constants
    # ------------------------------------------------------------------
    #  XFeat: trained with ImageNet-style grayscale stats
    _XFEAT_MEAN = 0.485
    _XFEAT_STD  = 0.229

    #  SuperPoint: trained with images in [0, 1] – no shift/scale required

    def __init__(
        self,
        xfeat_core: nn.Module,
        superpoint_core: nn.Module,
        num_keypoints: int = 4096,
        nms_radius: int = 4,
        descriptor_dim: int = 256,
        xfeat_kp_head_attrs: Tuple[str, ...] = (
            'kp_head', 'keypoint_head', 'kpoint_head', 'det_head'
        ),
    ):
        super().__init__()

        # ── 1 · Store sub-networks ──────────────────────────────────────
        self.xfeat      = xfeat_core
        self.superpoint = superpoint_core

        # ── 2 · Gradient Isolation (Spec §1) ───────────────────────────
        # Freeze SuperPoint entirely – preserves its geometric prior
        self._freeze_superpoint()

        # Freeze all XFeat params first, then selectively unfreeze kp head
        self._configure_xfeat_gradients(xfeat_kp_head_attrs)

        # ── 3 · Sampling Interconnect (Spec §3) ─────────────────────────
        self.sampler = DifferentiableDescriptorSampler(mode='bicubic')

        # ── 4 · Hyperparameters ─────────────────────────────────────────
        self.num_keypoints   = num_keypoints
        self.nms_radius      = nms_radius
        self.descriptor_dim  = descriptor_dim

        # Register normalization buffers (moves with .to(device))
        self.register_buffer(
            'xfeat_mean',
            torch.tensor(self._XFEAT_MEAN).view(1, 1, 1, 1)
        )
        self.register_buffer(
            'xfeat_std',
            torch.tensor(self._XFEAT_STD).view(1, 1, 1, 1)
        )

        # Hook storage for SuperPoint descriptor interception
        self._sp_desc_map: Optional[torch.Tensor] = None
        self._sp_hook_handle = None

    # ------------------------------------------------------------------
    # Gradient isolation
    # ------------------------------------------------------------------

    def _freeze_superpoint(self) -> None:
        """
        Completely freeze the SuperPoint backbone (requires_grad=False).

        SuperPoint's descriptor manifold is used as a fixed, rich feature
        space.  Keeping it frozen prevents catastrophic forgetting of its
        learned geometric priors and ensures training stability, since only
        the XFeat keypoint head adapts.
        """
        for param in self.superpoint.parameters():
            param.requires_grad_(False)

        total = _count_params(self.superpoint)
        print(
            f"[HybridModel] SuperPoint → {total:,} params FROZEN (all)"
        )

    def _configure_xfeat_gradients(
        self,
        kp_head_attrs: Tuple[str, ...]
    ) -> None:
        """
        Freeze the entire XFeat model, then selectively unfreeze only
        the 1×1 convolution layers of the keypoint detection head.

        The keypoint head (65-channel heatmap output) is the only part
        that must adapt: it learns to detect locations whose corresponding
        SuperPoint descriptors are highly discriminative.

        Strategy
        --------
        1. Freeze everything first.
        2. Walk ``kp_head_attrs`` and unfreeze the first matching sub-module.
        3. If no named match is found, fall back to scanning all modules
           whose name contains 'kp' or 'keypoint'.
        """
        # Step 1: freeze all XFeat
        for param in self.xfeat.parameters():
            param.requires_grad_(False)

        # Step 2: unfreeze keypoint head
        unfrozen = False
        for attr in kp_head_attrs:
            head = None
            # Support nested attributes via getattr chain
            obj = self.xfeat
            for part in attr.split('.'):
                head = getattr(obj, part, None)
                if head is None:
                    break
                obj = head

            if head is not None and isinstance(head, nn.Module):
                for param in head.parameters():
                    param.requires_grad_(True)
                print(
                    f"[HybridModel] XFeat keypoint head '{attr}' UNFROZEN"
                )
                unfrozen = True
                break

        # Step 3: name-scan fallback
        if not unfrozen:
            for name, module in self.xfeat.named_modules():
                if any(kw in name.lower() for kw in ('kp', 'keypoint', 'det')):
                    for param in module.parameters():
                        param.requires_grad_(True)
                    unfrozen = True
            if unfrozen:
                print(
                    "[HybridModel] XFeat keypoint head found via name-scan "
                    "and UNFROZEN"
                )
            else:
                # Last resort: unfreeze the entire XFeat (suboptimal)
                for param in self.xfeat.parameters():
                    param.requires_grad_(True)
                print(
                    "[HybridModel] WARNING: Could not isolate XFeat keypoint "
                    "head — entire XFeat UNFROZEN (check xfeat_kp_head_attrs)"
                )

        trainable = _count_params(self.xfeat, trainable_only=True)
        total     = _count_params(self.xfeat)
        print(
            f"[HybridModel] XFeat → {trainable:,} / {total:,} params "
            f"trainable ({100*trainable/max(total,1):.1f}% – keypoint head)"
        )

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

    def _normalize_for_xfeat(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply XFeat's training normalization.

        Input : (B, 1, H, W) grayscale in [0, 1]
        Output: (B, 1, H, W) standardized with ImageNet-grayscale stats
        """
        return (x - self.xfeat_mean) / self.xfeat_std

    @staticmethod
    def _normalize_for_superpoint(x: torch.Tensor) -> torch.Tensor:
        """
        Apply SuperPoint's expected normalization.

        SuperPoint was trained on raw [0, 1] grayscale.
        No mean subtraction or std division is required.

        Input / Output: (B, 1, H, W) grayscale in [0, 1]
        """
        # Clamp to expected range to guard against float precision drift
        return x.clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # XFeat heatmap decoding
    # ------------------------------------------------------------------

    def _decode_xfeat_heatmap(
        self,
        heatmap: torch.Tensor,
        image_hw: Tuple[int, int],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Convert XFeat's raw 65-channel logit map into pixel-space keypoints.

        XFeat Keypoint Head Output
        --------------------------
        K ∈ ℝ^{B × 65 × H/8 × W/8}

        Each spatial location (i, j) holds a 65-dim vector whose entries
        are logits for:
          • indices 0–63  → probability that the keypoint lies at sub-pixel
                            position (idx % 8, idx // 8) within the 8×8 cell
          • index 64      → "dustbin" (no keypoint in this cell)

        Decoding Steps
        --------------
        1. Softmax over channel dim → probability distribution
        2. Discard dustbin channel (index 64)
        3. Pixel-shuffle: (B, 64, H/8, W/8) → (B, H, W)  full-res score map
        4. Non-Maximum Suppression + top-K selection

        Args
        ----
        heatmap   : (B, 65, Hc, Wc) raw logits from XFeat keypoint head
        image_hw  : (H, W) original image size

        Returns
        -------
        keypoints_list : list of B tensors, each (N_b, 2) [x, y] in pixels
        scores_list    : list of B tensors, each (N_b,)  confidence scores
        """
        B, _, Hc, Wc = heatmap.shape
        H, W = image_hw

        # --- Softmax + remove dustbin ---
        prob = torch.softmax(heatmap, dim=1)          # (B, 65, Hc, Wc)
        prob_no_dust = prob[:, :-1, :, :]             # (B, 64, Hc, Wc)

        # --- Pixel shuffle: reconstruct full-resolution heatmap ---
        # Reshape 64 channels → 8×8 spatial block per cell
        # (B, 64, Hc, Wc) → (B, Hc, Wc, 8, 8)
        score = prob_no_dust.permute(0, 2, 3, 1)      # (B, Hc, Wc, 64)
        score = score.reshape(B, Hc, Wc, 8, 8)
        score = score.permute(0, 1, 3, 2, 4)           # (B, Hc, 8, Wc, 8)
        score = score.reshape(B, H, W)                  # (B, H, W)

        keypoints_list: List[torch.Tensor] = []
        scores_list:    List[torch.Tensor] = []

        for b in range(B):
            sm = score[b]  # (H, W)

            # -- NMS via max-pool --
            # A pixel survives if it equals the local maximum in its window
            sm_4d = sm[None, None]                     # (1, 1, H, W)
            kernel = 2 * self.nms_radius + 1
            local_max = F.max_pool2d(
                sm_4d, kernel_size=kernel,
                stride=1, padding=self.nms_radius
            ).squeeze()                                 # (H, W)
            nms_mask = (sm == local_max)                # (H, W) bool

            # -- Border suppression --
            r = self.nms_radius
            nms_mask[:r,  :] = False
            nms_mask[-r:, :] = False
            nms_mask[:,  :r] = False
            nms_mask[:, -r:] = False

            # -- Gather surviving locations --
            yx = nms_mask.nonzero(as_tuple=False)       # (N_surv, 2)  [y, x]
            s  = sm[nms_mask]                           # (N_surv,)

            if yx.shape[0] == 0:
                # Fallback: just take global top-K without NMS
                flat_idx = sm.flatten().topk(
                    min(self.num_keypoints, sm.numel())
                ).indices
                yx = torch.stack(
                    [flat_idx // W, flat_idx % W], dim=1
                )
                s  = sm.flatten()[flat_idx]

            # -- Top-K selection --
            K = min(self.num_keypoints, s.shape[0])
            top_idx = torch.topk(s, K).indices

            kp = yx[top_idx].flip(1).float()           # (K, 2) → [x, y]
            sc = s[top_idx]                             # (K,)

            keypoints_list.append(kp)
            scores_list.append(sc)

        return keypoints_list, scores_list

    # ------------------------------------------------------------------
    # SuperPoint descriptor extraction (with hook fallback)
    # ------------------------------------------------------------------

    def _install_sp_hook(self) -> None:
        """
        Register a forward hook on the SuperPoint descriptor head to capture
        the pre-upsampling descriptor map D ∈ ℝ^{B×256×Hc×Wc}.

        This hook is only installed when SuperPoint does not expose a clean
        ``get_descriptor_map`` interface.  The captured tensor is stored in
        ``self._sp_desc_map`` after each forward call.

        The hook targets the module at attribute path ``superpoint.desc_head``
        (or the equivalent for the particular SuperPoint implementation).
        """
        target_attrs = ('desc_head', 'descriptor_head', 'desc_decoder')
        target = None
        for attr in target_attrs:
            candidate = getattr(self.superpoint, attr, None)
            if candidate is not None and isinstance(candidate, nn.Module):
                target = candidate
                break

        if target is None:
            # Scan for the last Conv2d whose output has 256 channels
            # (a heuristic to find the descriptor head conv)
            for name, mod in self.superpoint.named_modules():
                if isinstance(mod, nn.Conv2d) and mod.out_channels == 256:
                    target = mod  # keep overwriting → last match wins

        if target is not None:
            def _hook(module, input, output):
                # output: (B, 256, Hc, Wc) – capture before L2/upsample
                self._sp_desc_map = output.detach()

            self._sp_hook_handle = target.register_forward_hook(_hook)
            print(f"[HybridModel] SuperPoint desc hook installed on {target}")
        else:
            print(
                "[HybridModel] WARNING: Could not install SuperPoint hook. "
                "Implement 'get_descriptor_map(x)' in your SuperPointCore."
            )

    def _get_superpoint_desc_map(
        self,
        sp_input: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract D ∈ ℝ^{B×256×H/8×W/8} from SuperPoint BEFORE upsampling.

        Preference order
        ----------------
        1. ``self.superpoint.get_descriptor_map(x)``  – clean API
        2. Forward hook on the descriptor head conv     – fallback

        Args
        ----
        sp_input : (B, 1, H, W) normalized for SuperPoint

        Returns
        -------
        desc_map : (B, 256, Hc, Wc)  raw semi-dense descriptor map
        """
        with torch.no_grad():
            if hasattr(self.superpoint, 'get_descriptor_map'):
                # ── Clean interface ──────────────────────────────────
                desc_map = self.superpoint.get_descriptor_map(sp_input)

            elif hasattr(self.superpoint, 'encode'):
                # ── Alternative: encoder then desc head ─────────────
                features  = self.superpoint.encode(sp_input)
                desc_map  = self.superpoint.desc_head(features)

            else:
                # ── Hook-based interception fallback ─────────────────
                if self._sp_hook_handle is None:
                    self._install_sp_hook()
                _ = self.superpoint(sp_input)       # triggers hook
                desc_map = self._sp_desc_map
                if desc_map is None:
                    raise RuntimeError(
                        "SuperPoint hook returned None. "
                        "Implement 'get_descriptor_map(x)' in SuperPointCore."
                    )

        return desc_map     # (B, 256, Hc, Wc)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        image: torch.Tensor,
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Inference forward pass – returns a LightGlue-compatible payload.

        Args
        ----
        image : (B, 1, H, W) grayscale in [0, 1]

        Returns
        -------
        dict:
            'keypoints'   : list[Tensor(N_b, 2)]  coordinates ∈ [0,1]²
            'descriptors' : list[Tensor(256, N_b)] L2-normalised
        """
        return self._forward_impl(image, return_intermediates=False)

    def forward_train(
        self,
        image: torch.Tensor,
    ) -> Dict[str, object]:
        """
        Training forward pass – also returns raw heatmap & pixel-space kps
        needed by the loss function.

        Returns
        -------
        dict:
            'keypoints'       : list[Tensor(N_b, 2)]   normalised ∈ [0,1]²
            'descriptors'     : list[Tensor(256, N_b)]  L2-normalised
            'keypoints_px'    : list[Tensor(N_b, 2)]   pixel coords
            'scores'          : list[Tensor(N_b,)]      keypoint confidences
            'heatmap'         : Tensor(B,65,Hc,Wc)     raw XFeat logits
        """
        return self._forward_impl(image, return_intermediates=True)

    def _forward_impl(
        self,
        image: torch.Tensor,
        return_intermediates: bool,
    ) -> Dict[str, object]:
        """Shared forward implementation."""
        B, C, H, W = image.shape
        assert C == 1, (
            f"HybridModel expects single-channel grayscale (C=1), got C={C}. "
            "Convert with: image = image.mean(dim=1, keepdim=True)"
        )

        # ────────────────────────────────────────────────────────────────
        # STREAM 1 — XFeat: keypoint detection  (gradients ACTIVE)
        # ────────────────────────────────────────────────────────────────
        xfeat_input = self._normalize_for_xfeat(image)
        raw_output  = self.xfeat(xfeat_input)

        # Unpack heatmap K ∈ ℝ^{B×65×Hc×Wc}
        if isinstance(raw_output, dict):
            heatmap = raw_output['keypoints']
        elif isinstance(raw_output, (list, tuple)):
            heatmap = raw_output[0]
        else:
            heatmap = raw_output          # (B, 65, Hc, Wc)

        # Decode heatmap → pixel-space keypoints (with soft-NMS)
        keypoints_px_list, scores_list = self._decode_xfeat_heatmap(
            heatmap, (H, W)
        )

        # ────────────────────────────────────────────────────────────────
        # STREAM 2 — SuperPoint: descriptor extraction  (FROZEN)
        # ────────────────────────────────────────────────────────────────
        sp_input = self._normalize_for_superpoint(image)
        desc_map = self._get_superpoint_desc_map(sp_input)
        # desc_map: (B, 256, Hc, Wc)

        # ────────────────────────────────────────────────────────────────
        # SAMPLING INTERCONNECT (Spec §3)
        # XFeat pixel-space coords → SuperPoint descriptor grid → L2-norm
        # ────────────────────────────────────────────────────────────────
        descriptors_list:   List[torch.Tensor] = []
        keypoints_norm_list: List[torch.Tensor] = []

        for b in range(B):
            kp_px = keypoints_px_list[b]       # (N, 2)  [x, y] pixel space
            dm    = desc_map[b:b+1]            # (1, 256, Hc, Wc)

            # Differentiable bicubic sampling at XFeat's keypoint locations
            # Returns (N, 256) L2-normalised
            sampled_desc = self.sampler(kp_px, dm, (H, W))

            # Normalise keypoints to [0, 1]² for LightGlue (Spec §4)
            scale = kp_px.new_tensor([W - 1, H - 1])
            kp_norm = kp_px / scale            # (N, 2)  ∈ [0, 1]²

            descriptors_list.append(sampled_desc.T)    # (256, N)
            keypoints_norm_list.append(kp_norm)        # (N,  2)

        # ────────────────────────────────────────────────────────────────
        # LightGlue-compatible output payload (Spec §4)
        # ────────────────────────────────────────────────────────────────
        output: Dict[str, object] = {
            'keypoints':   keypoints_norm_list,    # list[(N,2)]  ∈ [0,1]²
            'descriptors': descriptors_list,        # list[(256,N)]
        }

        if return_intermediates:
            output.update({
                'keypoints_px': keypoints_px_list, # list[(N,2)] pixel space
                'scores':       scores_list,        # list[(N,)]
                'heatmap':      heatmap,            # (B,65,Hc,Wc) raw logits
            })

        return output
