import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from .sampler import DifferentiableDescriptorSampler


def _count_params(module: nn.Module, trainable_only: bool = False) -> int:
    return sum(
        p.numel() for p in module.parameters()
        if (not trainable_only) or p.requires_grad
    )


class HybridModel(nn.Module):
    _XFEAT_MEAN = 0.485
    _XFEAT_STD = 0.229

    def __init__(
        self,
        xfeat_core: nn.Module,
        superpoint_core: nn.Module,
        num_keypoints: int = 512,      # reduced from 1024
        nms_radius: int = 8,           # increased from 4
        descriptor_dim: int = 256,
        border_margin: int = 16,       # suppress border-heavy artifacts
        xfeat_kp_head_attrs: Tuple[str, ...] = (
            'kp_head', 'keypoint_head', 'kpoint_head', 'det_head'
        ),
    ):
        super().__init__()

        self.xfeat = xfeat_core
        self.superpoint = superpoint_core

        self._freeze_superpoint()
        self._configure_xfeat_gradients(xfeat_kp_head_attrs)

        self.sampler = DifferentiableDescriptorSampler(mode='bicubic')

        self.num_keypoints = num_keypoints
        self.nms_radius = nms_radius
        self.descriptor_dim = descriptor_dim
        self.border_margin = border_margin

        self.register_buffer('xfeat_mean', torch.tensor(self._XFEAT_MEAN).view(1, 1, 1, 1))
        self.register_buffer('xfeat_std', torch.tensor(self._XFEAT_STD).view(1, 1, 1, 1))

        self._sp_desc_map: Optional[torch.Tensor] = None
        self._sp_hook_handle = None

    # ------------------------------------------------------------------
    # Gradient configuration
    # ------------------------------------------------------------------

    def _freeze_superpoint(self) -> None:
        for p in self.superpoint.parameters():
            p.requires_grad_(False)
        total = _count_params(self.superpoint)
        print(f"[HybridModel] SuperPoint → {total:,} params FROZEN (all)")

    def _configure_xfeat_gradients(self, kp_head_attrs: Tuple[str, ...]) -> None:
        for p in self.xfeat.parameters():
            p.requires_grad_(False)

        unfrozen = False
        for attr in kp_head_attrs:
            head = None
            obj = self.xfeat
            for part in attr.split('.'):
                head = getattr(obj, part, None)
                if head is None:
                    break
                obj = head

            if head is not None and isinstance(head, nn.Module):
                for p in head.parameters():
                    p.requires_grad_(True)
                print(f"[HybridModel] XFeat keypoint head '{attr}' UNFROZEN")
                unfrozen = True
                break

        if not unfrozen:
            for name, mod in self.xfeat.named_modules():
                if any(kw in name.lower() for kw in ('kp', 'keypoint', 'det')):
                    for p in mod.parameters():
                        p.requires_grad_(True)
                    unfrozen = True
            if unfrozen:
                print("[HybridModel] XFeat keypoint head found via name-scan and UNFROZEN")
            else:
                for p in self.xfeat.parameters():
                    p.requires_grad_(True)
                print("[HybridModel] WARNING: Could not isolate kp head; entire XFeat UNFROZEN")

        trainable = _count_params(self.xfeat, trainable_only=True)
        total = _count_params(self.xfeat)
        print(f"[HybridModel] XFeat → {trainable:,}/{total:,} trainable ({100 * trainable / max(total,1):.1f}%)")

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def _normalize_for_xfeat(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.xfeat_mean) / self.xfeat_std

    @staticmethod
    def _normalize_for_superpoint(x: torch.Tensor) -> torch.Tensor:
        return x.clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Heatmap decoding
    # ------------------------------------------------------------------

    def _xfeat_logits_to_scoremap(self, heatmap: torch.Tensor, image_hw: Tuple[int, int]) -> torch.Tensor:
        B, C, Hc, Wc = heatmap.shape
        H, W = image_hw

        if C == 1:
            # XFeat dense heatmap: upsample directly to full resolution
            score = F.interpolate(
                heatmap, size=(H, W), mode='bilinear', align_corners=False
            ).squeeze(1)
        elif C == 65:
            prob = torch.softmax(heatmap, dim=1)[:, :-1]   # drop dustbin → (B,64,Hc,Wc)
            score = self._pixel_shuffle_to_scoremap(prob, B, Hc, Wc, H, W)
        elif C == 64:
            prob = torch.sigmoid(heatmap)
            score = self._pixel_shuffle_to_scoremap(prob, B, Hc, Wc, H, W)
        else:
            raise ValueError(f"Expected C in {{1, 64, 65}}, got {C}")

        return score

    @staticmethod
    def _pixel_shuffle_to_scoremap(
        prob: torch.Tensor,
        B: int, Hc: int, Wc: int, H: int, W: int,
    ) -> torch.Tensor:
        """Reconstruct a full-resolution (B, H, W) score map from the
        (B, 64, Hc, Wc) cell-probability tensor via pixel-shuffle.

        Each cell of size 8×8 holds 64 per-pixel scores. The two
        permute+reshape steps interleave the cell and intra-cell axes
        to produce the spatial layout of the original image:

          (B, 64, Hc, Wc)
          → permute(0,2,3,1) → (B, Hc, Wc, 64)
          → reshape(B,Hc,Wc,8,8) — split 64 into 8×8 intra-cell grid
          → permute(0,1,3,2,4)  → (B, Hc, 8, Wc, 8) — interleave axes
          → reshape(B, H, W)    — merge to full resolution (H=Hc*8, W=Wc*8)
        """
        return (
            prob
            .permute(0, 2, 3, 1)            # (B, Hc, Wc, 64)
            .reshape(B, Hc, Wc, 8, 8)       # (B, Hc, Wc, 8, 8)
            .permute(0, 1, 3, 2, 4)          # (B, Hc, 8, Wc, 8)
            .reshape(B, H, W)                # (B, H, W)
        )

    def _decode_xfeat_heatmap(
        self,
        heatmap: torch.Tensor,
        image_hw: Tuple[int, int],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        B, _, _, _ = heatmap.shape
        H, W = image_hw
        score = self._xfeat_logits_to_scoremap(heatmap, image_hw)
        # score: (B, H, W) — differentiable from heatmap logits ✓

        keypoints_list: List[torch.Tensor] = []
        scores_list: List[torch.Tensor] = []

        for b in range(B):
            # Keep sm in the computation graph — do NOT clone/detach here.
            sm = score[b]  # (H, W)

            # ── Border suppression (OUT-OF-PLACE to preserve gradient) ──────
            # In-place zeroing (sm[:m,:] = 0) would break the autograd graph.
            m = self.border_margin
            border_mask = sm.new_zeros(H, W)
            border_mask[m:H - m, m:W - m] = 1.0
            sm = sm * border_mask  # new tensor, gradient preserved ✓

            # ── Mild smoothing (creates a new tensor, gradient preserved) ───
            sm = F.avg_pool2d(
                sm[None, None], kernel_size=3, stride=1, padding=1
            ).squeeze()

            # ── NMS: use DETACHED sm for position selection only ─────────────
            # We want integer keypoint coordinates (non-differentiable), but
            # the SCORE VALUES at those positions must remain differentiable.
            sm_d = sm.detach()
            kernel = 2 * self.nms_radius + 1
            local_max = F.max_pool2d(
                sm_d[None, None], kernel_size=kernel,
                stride=1, padding=self.nms_radius,
            ).squeeze()
            nms_mask = (sm_d == local_max)  # bool from detached — no grad

            r = max(self.nms_radius, self.border_margin)
            nms_mask[:r, :]  = False   # in-place on bool tensor — fine ✓
            nms_mask[-r:, :] = False
            nms_mask[:,  :r] = False
            nms_mask[:, -r:] = False

            yx = nms_mask.nonzero(as_tuple=False)   # (K, 2) integer [y, x]
            # Boolean-mask indexing on sm (not sm_d) — values remain
            # differentiable w.r.t. heatmap ✓
            s = sm[nms_mask]                        # (K,)

            if yx.shape[0] == 0:
                flat_idx = sm_d.flatten().topk(
                    min(self.num_keypoints, sm_d.numel())
                ).indices
                yx = torch.stack([flat_idx // W, flat_idx % W], dim=1)
                s  = sm.flatten()[flat_idx]          # differentiable ✓

            K = min(self.num_keypoints, s.shape[0])
            # Detach for index selection; index sm (not sm_d) for values
            top_idx = s.detach().topk(K).indices     # LongTensor, no grad
            kp = yx[top_idx].flip(1).float()         # (K, 2) [x, y]
            sc = s[top_idx]                          # (K,) — differentiable ✓

            keypoints_list.append(kp)
            scores_list.append(sc)

        return keypoints_list, scores_list

    # ------------------------------------------------------------------
    # SuperPoint descriptor extraction
    # ------------------------------------------------------------------

    def _install_sp_hook(self) -> None:
        target_attrs = ('desc_head', 'descriptor_head', 'desc_decoder')
        target = None
        for attr in target_attrs:
            cand = getattr(self.superpoint, attr, None)
            if cand is not None and isinstance(cand, nn.Module):
                target = cand
                break

        if target is None:
            for _, mod in self.superpoint.named_modules():
                if isinstance(mod, nn.Conv2d) and mod.out_channels == 256:
                    target = mod

        if target is not None:
            def _hook(module, inp, out):
                self._sp_desc_map = out
            self._sp_hook_handle = target.register_forward_hook(_hook)
            print(f"[HybridModel] SuperPoint desc hook installed on {target}")
        else:
            print("[HybridModel] WARNING: Could not install SuperPoint hook.")

    def _get_superpoint_desc_map(self, sp_input: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if hasattr(self.superpoint, 'get_descriptor_map'):
                desc_map = self.superpoint.get_descriptor_map(sp_input)
            elif hasattr(self.superpoint, 'encode'):
                feat = self.superpoint.encode(sp_input)
                desc_map = self.superpoint.desc_head(feat)
            else:
                if self._sp_hook_handle is None:
                    self._install_sp_hook()
                _ = self.superpoint(sp_input)
                desc_map = self._sp_desc_map
                if desc_map is None:
                    raise RuntimeError("SuperPoint hook returned None.")
        return desc_map

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, image: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        return self._forward_impl(image, return_intermediates=False)

    def forward_train(self, image: torch.Tensor) -> Dict[str, object]:
        return self._forward_impl(image, return_intermediates=True)

    def _forward_impl(self, image: torch.Tensor, return_intermediates: bool) -> Dict[str, object]:
        B, C, H, W = image.shape
        assert C == 1, f"Expected grayscale C=1, got {C}"

        # XFeat stream
        xfeat_input = self._normalize_for_xfeat(image)
        raw_output = self.xfeat(xfeat_input)
        if isinstance(raw_output, dict):
            heatmap = raw_output['keypoints']
        elif isinstance(raw_output, (list, tuple)):
            heatmap = raw_output[0]
        else:
            heatmap = raw_output

        keypoints_px_list, scores_list = self._decode_xfeat_heatmap(heatmap, (H, W))

        # SuperPoint stream (frozen)
        sp_input = self._normalize_for_superpoint(image)
        desc_map = self._get_superpoint_desc_map(sp_input)

        descriptors_list: List[torch.Tensor] = []
        keypoints_norm_list: List[torch.Tensor] = []

        for b in range(B):
            kp_px = keypoints_px_list[b]
            dm = desc_map[b:b+1]
            sampled_desc = self.sampler(kp_px, dm, (H, W))
            sampled_desc = F.normalize(sampled_desc, p=2, dim=1)

            scale = kp_px.new_tensor([W - 1, H - 1])
            kp_norm = kp_px / scale

            descriptors_list.append(sampled_desc)   # (N, 256)
            keypoints_norm_list.append(kp_norm)       # (N, 2)

        output: Dict[str, object] = {
            'keypoints': keypoints_norm_list,
            'descriptors': descriptors_list,
        }

        if return_intermediates:
            output.update({
                'keypoints_px': keypoints_px_list,
                'scores': scores_list,
                'heatmap': heatmap,
            })

        return output