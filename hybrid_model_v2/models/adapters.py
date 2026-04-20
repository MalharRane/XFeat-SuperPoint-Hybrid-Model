from __future__ import annotations

import inspect
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

_SP_DESC_DIM = 256


class ModelImportError(RuntimeError):
    """Raised when a required upstream XFeat/SuperPoint dependency cannot be imported."""
    pass


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
    tried = []
    for path in (
        "superpoint.superpoint:SuperPoint",
        "superpoint_pytorch:SuperPoint",
        "superpoint:SuperPoint",
    ):
        mod_name, cls_name = path.split(":")
        tried.append(path)
        try:
            mod = __import__(mod_name, fromlist=[cls_name])
            cls = getattr(mod, cls_name)
            return _instantiate_sp(cls)
        except Exception:
            continue
    raise ModelImportError(
        f"Could not import SuperPoint from supported paths: {tried}. "
        "Install/clone a compatible SuperPoint implementation."
    )


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
    with torch.no_grad():
        if hasattr(sp, "get_descriptor_map"):
            dm = sp.get_descriptor_map(x)
        elif hasattr(sp, "encode") and hasattr(sp, "desc_head"):
            dm = sp.desc_head(sp.encode(x))
        else:
            hook_out: Dict[str, torch.Tensor] = {}
            handle = None
            target = None
            for _, m in sp.named_modules():
                if isinstance(m, nn.Conv2d) and m.out_channels == _SP_DESC_DIM:
                    target = m
            if target is None:
                raise RuntimeError(f"Could not locate {_SP_DESC_DIM}-channel descriptor head in SuperPoint")

            def _hook(_m, _i, out):
                if isinstance(out, torch.Tensor):
                    hook_out["desc"] = out

            handle = target.register_forward_hook(_hook)
            try:
                _ = call_superpoint_forward(sp, x)
            finally:
                handle.remove()
            if "desc" not in hook_out:
                raise RuntimeError("SuperPoint descriptor hook produced no output")
            dm = hook_out["desc"]
    return dm.float()
