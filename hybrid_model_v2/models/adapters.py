from __future__ import annotations

import inspect
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_SP_DESC_DIM = 256


class ModelImportError(RuntimeError):
    """Raised when a required upstream XFeat/SuperPoint dependency cannot be imported."""
    pass


class _SuperPointLegacy(nn.Module):
    """Minimal SuperPoint re-implementation whose state-dict keys exactly match
    the official ``superpoint_v1.pth`` checkpoint distributed by LightGlue::

        conv1a.weight  conv1a.bias  …  convDb.weight  convDb.bias  (24 tensors)

    The post-2023 rpautrat/SuperPoint uses VGGBlock + BatchNorm with nested keys
    (``backbone.0.0.conv.weight``) that are completely incompatible with that
    checkpoint.  This flat-key implementation avoids the dependency on any
    external clone and guarantees weight loading succeeds without remapping.
    """

    def __init__(self) -> None:
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, 3, 1, 1)
        self.conv1b = nn.Conv2d(c1, c1, 3, 1, 1)
        self.conv2a = nn.Conv2d(c1, c2, 3, 1, 1)
        self.conv2b = nn.Conv2d(c2, c2, 3, 1, 1)
        self.conv3a = nn.Conv2d(c2, c3, 3, 1, 1)
        self.conv3b = nn.Conv2d(c3, c3, 3, 1, 1)
        self.conv4a = nn.Conv2d(c3, c4, 3, 1, 1)
        self.conv4b = nn.Conv2d(c4, c4, 3, 1, 1)

        self.convPa = nn.Conv2d(c4, c5, 3, 1, 1)
        self.convPb = nn.Conv2d(c5, 65, 1, 1, 0)

        self.convDa = nn.Conv2d(c4, c5, 3, 1, 1)
        self.convDb = nn.Conv2d(c5, 256, 1, 1, 0)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        return x

    def get_descriptor_map(self, x: torch.Tensor) -> torch.Tensor:
        """Return the dense descriptor map ``(B, 256, H/8, W/8)``, unnormalized.
        This is the direct output of ``convDb`` and is consumed by
        ``extract_superpoint_desc_map`` without going through the hook path.
        """
        if isinstance(x, dict):
            x = x["image"]
        if x.shape[1] == 3:
            scale = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            x = (x * scale).sum(1, keepdim=True)
        feat = self._encode(x)
        return self.convDb(self.relu(self.convDa(feat)))

    def forward(self, data: Any) -> Dict[str, torch.Tensor]:
        if isinstance(data, dict):
            x = data["image"]
        else:
            x = data
        if x.shape[1] == 3:
            scale = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            x = (x * scale).sum(1, keepdim=True)
        feat = self._encode(x)
        desc_map = self.convDb(self.relu(self.convDa(feat)))
        scores = self.convPb(self.relu(self.convPa(feat)))
        return {"descriptors_dense": desc_map, "scores": scores}


def build_xfeat() -> nn.Module:
    try:
        from modules.xfeat import XFeat
    except ImportError as e:
        raise ModelImportError(
            "Could not import XFeat from modules.xfeat. Clone accelerated_features and set PYTHONPATH."
        ) from e
    # Pass weights=None to skip the constructor's built-in auto-loading, which
    # uses a default path that does not exist after a plain `git clone`.
    # Weights are loaded separately by load_weights_strictish in build_model_v2.
    return XFeat(weights=None)


def _instantiate_sp(cls: Any) -> nn.Module:
    try:
        return cls()
    except TypeError:
        # Some SuperPoint forks expect a config-dict constructor signature.
        return cls({})


def build_superpoint() -> nn.Module:
    # Use the built-in flat-key implementation that matches superpoint_v1.pth.
    #
    # The post-2023 rpautrat/SuperPoint (commit 1411bbd and later) switched to a
    # VGGBlock + BatchNorm architecture whose state-dict keys look like
    # ``backbone.0.0.conv.weight``.  The LightGlue-distributed superpoint_v1.pth
    # uses the original flat keys (``conv1a.weight``, …).  Loading the new
    # rpautrat architecture from that checkpoint yields 0 % key overlap and a
    # RuntimeError.  _SuperPointLegacy replicates the original flat-key layout,
    # making weight loading work without any key remapping or external dependency.
    return _SuperPointLegacy()


def call_superpoint_forward(sp: nn.Module, x: torch.Tensor) -> Any:
    payload = {"image": x}
    likely_dict = False
    try:
        sig = inspect.signature(sp.forward)
        params = [k.lower() for k in sig.parameters.keys() if k != "self"]
        likely_dict = bool(params) and params[0] in {"data", "batch", "inputs"}
    except (TypeError, ValueError):
        likely_dict = False

    order = (payload, x) if likely_dict else (x, payload)
    first_err = None
    for arg in order:
        try:
            return sp(arg)
        except Exception as e:
            if first_err is None:
                first_err = e
    raise RuntimeError("SuperPoint forward failed for both tensor and dict payload") from first_err


def extract_superpoint_desc_map(sp: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Extract the dense (B, 256, H/8, W/8) descriptor map from a frozen SuperPoint.

    AMP safety:
        This function is always called from within a ``torch.autocast`` scope
        (``forward_train``).  Under autocast, tensors can be silently promoted
        to float16.  Bicubic ``F.grid_sample`` in PyTorch is notoriously
        unstable in float16 and can produce NaN or grossly incorrect values.
        The authoritative guard is to cast ``x`` to float32 **inside** the
        ``no_grad`` block, before any SuperPoint op runs.  The secondary guard
        in ``DifferentiableDescriptorSampler.forward`` provides defense-in-depth.

    Gradient path:
        ``torch.no_grad()`` is correct and intentional here.  SuperPoint is
        frozen; its parameters have ``requires_grad=False``.  Removing
        ``no_grad`` would waste backward compute and memory without producing
        any useful gradient signal.  The only training signal reaching the
        XFeat keypoint head flows through ``w = scores1 * scores2`` in the
        loss, not through the descriptor values themselves.
    """
    with torch.no_grad():
        # FIX: cast to float32 inside the no_grad block so that any AMP
        # autocast promotion that happened in the enclosing scope is undone
        # before the SuperPoint convolutions run.
        x_fp32 = x.float()

        if hasattr(sp, "get_descriptor_map"):
            dm = sp.get_descriptor_map(x_fp32)
        elif hasattr(sp, "encode") and hasattr(sp, "desc_head"):
            dm = sp.desc_head(sp.encode(x_fp32))
        else:
            hook_out: Dict[str, torch.Tensor] = {}
            handle = None
            target = None
            for _, m in sp.named_modules():
                if isinstance(m, nn.Conv2d) and m.out_channels == _SP_DESC_DIM:
                    target = m
            if target is None:
                raise RuntimeError(
                    f"Could not locate {_SP_DESC_DIM}-channel descriptor head in SuperPoint"
                )

            def _hook(_m, _i, out):
                if isinstance(out, torch.Tensor):
                    hook_out["desc"] = out

            handle = target.register_forward_hook(_hook)
            try:
                _ = call_superpoint_forward(sp, x_fp32)
            finally:
                handle.remove()
            if "desc" not in hook_out:
                raise RuntimeError("SuperPoint descriptor hook produced no output")
            dm = hook_out["desc"]

    dm_fp32 = dm.float()

    # Shape and dtype assertions — surface any future AMP regression immediately
    # as a clear error rather than a silent NaN.
    if dm_fp32.dtype != torch.float32:
        raise RuntimeError(
            f"SuperPoint desc_map dtype should be float32 after cast, got {dm_fp32.dtype}"
        )
    if dm_fp32.ndim != 4:
        raise RuntimeError(
            f"SuperPoint desc_map should be (B, 256, H/8, W/8), got ndim={dm_fp32.ndim} "
            f"shape={tuple(dm_fp32.shape)}"
        )
    if dm_fp32.shape[1] != _SP_DESC_DIM:
        raise RuntimeError(
            f"SuperPoint desc_map channel dim should be {_SP_DESC_DIM}, "
            f"got {dm_fp32.shape[1]} (shape={tuple(dm_fp32.shape)})"
        )

    return dm_fp32