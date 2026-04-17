import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import inspect
from typing import Dict, List, Tuple, Optional, Any

from .sampler import DifferentiableDescriptorSampler

log = logging.getLogger(__name__)


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
        min_keypoint_score: float = 0.01,
        descriptor_dim: int = 256,
        border_margin: int = 16,       # suppress border-heavy artifacts
        xfeat_kp_head_attrs: Tuple[str, ...] = (
            'kp_head', 'keypoint_head', 'kpoint_head', 'det_head'
        ),
    ):
        super().__init__()

        self.xfeat = xfeat_core
        self.superpoint = superpoint_core

        self._sp_desc_map: Optional[torch.Tensor] = None
        self._sp_hook_handle = None

        # Initialize XFeat hook state before gradient configuration, because
        # _configure_xfeat_gradients may call _install_xfeat_kp_hook.
        self._xfeat_kp_output: Optional[torch.Tensor] = None
        self._xfeat_kp_hook_handle = None

        self._freeze_superpoint()
        self._configure_xfeat_gradients(xfeat_kp_head_attrs)

        self.sampler = DifferentiableDescriptorSampler(mode='bicubic')

        self.num_keypoints = num_keypoints
        self.nms_radius = nms_radius
        self.min_keypoint_score = float(min_keypoint_score)
        self.descriptor_dim = descriptor_dim
        self.border_margin = border_margin

        self.register_buffer('xfeat_mean', torch.tensor(self._XFEAT_MEAN).view(1, 1, 1, 1))
        self.register_buffer('xfeat_std', torch.tensor(self._XFEAT_STD).view(1, 1, 1, 1))

        # Captures the trainable kp-head output during XFeat's forward so that
        # _forward_impl can use it directly.  This guarantees gradient flows to
        # the unfrozen head even when raw_output['keypoints'] is produced by a
        # *different* (frozen) head inside the XFeat model (e.g. heatmap_head
        # vs keypoint_head).  Without this, loss.requires_grad would be False
        # and scaler.scale(loss).backward() raises RuntimeError.

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
                self._install_xfeat_kp_hook(head)
                unfrozen = True
                break

        if not unfrozen:
            hooked = False
            for name, mod in self.xfeat.named_modules():
                if any(kw in name.lower() for kw in ('kp', 'keypoint', 'det')):
                    for p in mod.parameters():
                        p.requires_grad_(True)
                    if not hooked:
                        self._install_xfeat_kp_hook(mod)
                        hooked = True
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

    def _install_xfeat_kp_hook(self, head: nn.Module) -> None:
        """Register a forward hook on the trainable XFeat head.

        The hook stores the head's output tensor in ``self._xfeat_kp_output``
        so that ``_forward_impl`` can use it as the heatmap.  This is
        necessary because ``raw_output['keypoints']`` might point to a
        *different* (frozen) head inside XFeat (e.g. ``heatmap_head`` vs
        ``keypoint_head``), which would give a tensor with
        ``requires_grad=False`` and break the gradient path.
        """
        def _hook(module: nn.Module, inp: tuple, out: object) -> None:
            # Guard against heads that return tuples/dicts rather than a
            # plain tensor (e.g. multi-output detection heads).
            if isinstance(out, torch.Tensor):
                self._xfeat_kp_output = out
            elif isinstance(out, (list, tuple)) and out and isinstance(out[0], torch.Tensor):
                self._xfeat_kp_output = out[0]
            # Other output types are silently ignored; _forward_impl falls
            # back to raw_output['keypoints'] in that case.

        if self._xfeat_kp_hook_handle is not None:
            self._xfeat_kp_hook_handle.remove()
        self._xfeat_kp_hook_handle = head.register_forward_hook(_hook)

    def unfreeze_xfeat_modules(self, name_keywords: Tuple[str, ...]) -> int:
        """
        Unfreeze additional XFeat parameters by name keyword match.

        Args
        ----
        name_keywords : tuple[str, ...]
            Lower/upper-case-insensitive substrings matched against
            ``self.xfeat.named_parameters()`` names.

        Returns
        -------
        int
            Number of newly-unfrozen parameters (counted by elements).
        """
        keywords = tuple(k.strip().lower() for k in name_keywords if k.strip())
        if not keywords:
            return 0

        newly_unfrozen = 0
        matched_names = 0
        for name, p in self.xfeat.named_parameters():
            lname = name.lower()
            if any(k in lname for k in keywords):
                matched_names += 1
                if not p.requires_grad:
                    p.requires_grad_(True)
                    newly_unfrozen += p.numel()

        trainable = _count_params(self.xfeat, trainable_only=True)
        total = _count_params(self.xfeat)
        log.info(
            "[HybridModel] Scheduled unfreeze "
            f"keywords={keywords} matched={matched_names} "
            f"new={newly_unfrozen:,} params; "
            f"trainable now {trainable:,}/{total:,} ({100 * trainable / max(total,1):.1f}%)"
        )
        return newly_unfrozen

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def _normalize_for_xfeat(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.xfeat_mean) / self.xfeat_std

    @staticmethod
    def _normalize_for_superpoint(x: torch.Tensor) -> torch.Tensor:
        return x.clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # XFeat compatibility adapter
    # ------------------------------------------------------------------

    @staticmethod
    def _build_heatmap_from_keypoints(
        keypoints: Any,
        scores: Any,
        batch_size: int,
        image_hw: Tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Best-effort conversion of keypoints/scores to a dense heatmap tensor."""
        H, W = image_hw
        heatmap = torch.zeros((batch_size, 1, H, W), device=device, dtype=dtype)

        def _to_kp_list(value: Any, bsz: int) -> List[torch.Tensor]:
            if isinstance(value, list):
                return [v if isinstance(v, torch.Tensor) else torch.empty(0, 2, device=device, dtype=dtype) for v in value]
            if isinstance(value, torch.Tensor):
                if value.dim() == 3 and value.shape[0] == bsz:
                    return [value[b] for b in range(bsz)]
                if value.dim() == 2 and bsz == 1:
                    return [value]
            return [torch.empty(0, 2, device=device, dtype=dtype) for _ in range(bsz)]

        def _to_score_list(value: Any, bsz: int) -> List[torch.Tensor]:
            if value is None:
                return [torch.ones(0, device=device, dtype=dtype) for _ in range(bsz)]
            if isinstance(value, list):
                out = []
                for v in value:
                    if isinstance(v, torch.Tensor):
                        out.append(v)
                    else:
                        out.append(torch.ones(0, device=device, dtype=dtype))
                return out
            if isinstance(value, torch.Tensor):
                if value.dim() == 2 and value.shape[0] == bsz:
                    return [value[b] for b in range(bsz)]
                if value.dim() == 1 and bsz == 1:
                    return [value]
            return [torch.ones(0, device=device, dtype=dtype) for _ in range(bsz)]

        kp_list = _to_kp_list(keypoints, batch_size)
        sc_list = _to_score_list(scores, batch_size)

        for b in range(batch_size):
            kp = kp_list[b]
            if not isinstance(kp, torch.Tensor) or kp.numel() == 0:
                continue
            kp = kp.to(device=device, dtype=torch.float32)
            if kp.dim() != 2 or kp.shape[1] < 2:
                continue
            sc = sc_list[b]
            if not isinstance(sc, torch.Tensor) or sc.numel() != kp.shape[0]:
                sc = torch.ones(kp.shape[0], device=device, dtype=dtype)
            else:
                sc = sc.to(device=device, dtype=dtype).view(-1)

            x = kp[:, 0].round().long().clamp(0, W - 1)
            y = kp[:, 1].round().long().clamp(0, H - 1)
            heatmap[b, 0, y, x] = torch.maximum(heatmap[b, 0, y, x], sc)

        return heatmap

    def _normalize_xfeat_result_to_heatmap(
        self,
        result: Any,
        xfeat_input: torch.Tensor,
        image_hw: Tuple[int, int],
    ) -> torch.Tensor:
        """Normalize multiple XFeat API outputs to a dense heatmap tensor."""
        B = xfeat_input.shape[0]
        H, W = image_hw

        def _as_heatmap_tensor(value: Any) -> Optional[torch.Tensor]:
            if not isinstance(value, torch.Tensor):
                return None
            t = value
            if t.dim() == 4:
                if t.shape[0] != B:
                    return None
                return t
            if t.dim() == 3:
                if t.shape[-1] == 2:
                    return None
                if t.shape[0] == B:
                    return t.unsqueeze(1)
                if B == 1:
                    return t.unsqueeze(0)
            if t.dim() == 2 and B == 1:
                if t.shape[-1] == 2:
                    return None
                return t.unsqueeze(0).unsqueeze(0)
            return None

        if isinstance(result, dict):
            for key in ('heatmap', 'heatmaps', 'logits', 'score_map', 'scores', 'keypoints'):
                t = _as_heatmap_tensor(result.get(key))
                if t is not None:
                    return t
            if 'keypoints' in result:
                return self._build_heatmap_from_keypoints(
                    keypoints=result.get('keypoints'),
                    scores=result.get('scores'),
                    batch_size=B,
                    image_hw=(H, W),
                    device=xfeat_input.device,
                    dtype=xfeat_input.dtype,
                )

        if isinstance(result, (tuple, list)):
            for v in result:
                t = _as_heatmap_tensor(v)
                if t is not None:
                    return t

        t = _as_heatmap_tensor(result)
        if t is not None:
            return t

        raise RuntimeError(
            "XFeat adapter could not normalize model output to a heatmap tensor. "
            f"output_type={type(result).__name__}"
        )

    def _call_xfeat_forward(self, xfeat_input: torch.Tensor) -> Tuple[torch.Tensor, str]:
        """Call XFeat across forward and wrapper-style inference APIs."""
        xfeat_cls = type(self.xfeat)
        xfeat_info = f"{xfeat_cls.__module__}.{xfeat_cls.__name__}"
        methods = (
            ('forward', lambda: self.xfeat(xfeat_input)),
            ('detectAndCompute', lambda: self._call_xfeat_detect_compute('detectAndCompute', xfeat_input)),
            ('detect_and_compute', lambda: self._call_xfeat_detect_compute('detect_and_compute', xfeat_input)),
            ('extract', lambda: self._call_xfeat_detect_compute('extract', xfeat_input)),
            ('detect', lambda: self._call_xfeat_detect_compute('detect', xfeat_input)),
        )

        tried: List[str] = []
        first_err: Optional[Exception] = None
        H, W = int(xfeat_input.shape[2]), int(xfeat_input.shape[3])

        for name, fn in methods:
            if name != 'forward' and not hasattr(self.xfeat, name):
                continue
            try:
                raw = fn()
                heatmap = self._normalize_xfeat_result_to_heatmap(raw, xfeat_input, (H, W))
                return heatmap, name
            except Exception as exc:
                tried.append(name)
                if first_err is None:
                    first_err = exc

        supported = ', '.join(tried) if tried else '(none)'
        raise RuntimeError(
            "No supported XFeat invocation API found for HybridModel. "
            f"Detected module={xfeat_info}. Tried APIs={supported}. "
            "Expected one of: nn.Module.forward(tensor) returning tensor/dict/tuple; "
            "or wrapper methods detectAndCompute/detect_and_compute/extract/detect "
            "returning heatmap-like outputs or keypoints/scores."
        ) from first_err

    def _call_xfeat_detect_compute(self, method_name: str, xfeat_input: torch.Tensor) -> Any:
        method = getattr(self.xfeat, method_name)
        try:
            return method(xfeat_input)
        except Exception:
            per_image_out = []
            for b in range(xfeat_input.shape[0]):
                xb = xfeat_input[b:b + 1]
                try:
                    out = method(xb)
                except Exception:
                    out = method(xb.squeeze(0))
                per_image_out.append(out)
            if len(per_image_out) == 1:
                return per_image_out[0]
            keypoints_list: List[Any] = []
            scores_list: List[Any] = []
            for o in per_image_out:
                if isinstance(o, dict):
                    keypoints_list.append(o.get('keypoints'))
                    scores_list.append(o.get('scores'))
                elif isinstance(o, (tuple, list)) and len(o) >= 1:
                    keypoints_list.append(o[0])
                    scores_list.append(o[1] if len(o) > 1 else None)
                else:
                    keypoints_list.append(o)
                    scores_list.append(None)
            return {'keypoints': keypoints_list, 'scores': scores_list}

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

            if self.min_keypoint_score > 0.0 and s.numel() > 0:
                keep = s >= self.min_keypoint_score
                yx = yx[keep]
                s = s[keep]

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
            sc = s[top_idx].float()                  # (K,) — float32, differentiable ✓

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
                _ = self._call_superpoint_forward(sp_input)
                desc_map = self._sp_desc_map
                if desc_map is None:
                    raise RuntimeError("SuperPoint hook returned None.")
        # Under torch.amp.autocast the SuperPoint encoder runs in float16, but
        # F.grid_sample (bicubic) requires input and grid to share dtype.  The
        # grid is always float32 (derived from keypoints cast with .float()), so
        # we cast the descriptor map to float32 here to prevent a RuntimeError.
        return desc_map.float()

    def _call_superpoint_forward(self, sp_input: torch.Tensor):
        """Call SuperPoint forward across API variants.

        Some SuperPoint forks expect ``forward({'image': tensor})`` while
        others expect ``forward(tensor)``. This helper tries the likely format
        first and falls back to the other one.
        """
        payload = {'image': sp_input}

        likely_dict = False
        try:
            sig = inspect.signature(self.superpoint.forward)
            params = [n.lower() for n in sig.parameters.keys() if n != 'self']
            likely_dict = bool(params) and params[0] in {'data', 'batch', 'inputs'}
        except (TypeError, ValueError):
            likely_dict = False

        call_order = ((payload, sp_input) if likely_dict else (sp_input, payload))
        first_err = None

        # rpautrat-style forward(data) can raise IndexError when accidentally
        # indexed with a Tensor instead of dict; keep IndexError in fallback.
        expected_input_errors = (TypeError, KeyError, IndexError)

        for arg in call_order:
            try:
                return self.superpoint(arg)
            except expected_input_errors as err:
                if first_err is None:
                    first_err = err
                continue

        raise RuntimeError(
            "SuperPoint forward failed for both tensor input and {'image': tensor} input."
        ) from first_err

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

        # Reset the hook capture before each forward so we always get the
        # output from *this* call (not a stale tensor from a previous step).
        self._xfeat_kp_output = None
        adapter_heatmap, adapter_name = self._call_xfeat_forward(xfeat_input)

        # Prefer the hook-captured trainable head output when available.
        if self._xfeat_kp_output is not None:
            heatmap = self._xfeat_kp_output
        else:
            heatmap = adapter_heatmap

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
                'xfeat_adapter_path': adapter_name,
            })

        return output
