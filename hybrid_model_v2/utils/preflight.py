from __future__ import annotations

from typing import Any, Dict

import torch


def validate_lightglue_contract(out: Dict[str, Any], descriptor_dim: int = 256) -> None:
    keys = ("keypoints", "keypoints_px", "descriptors", "scores")
    for k in keys:
        if k not in out or not isinstance(out[k], list):
            raise RuntimeError(f"forward output missing list key: {k}")

    b = len(out["descriptors"])
    for k in keys:
        if len(out[k]) != b:
            raise RuntimeError(f"batch mismatch for {k}")

    for i in range(b):
        kp = out["keypoints"][i]
        kp_px = out["keypoints_px"][i]
        d = out["descriptors"][i]
        s = out["scores"][i]
        if kp.dim() != 2 or kp.shape[-1] != 2:
            raise RuntimeError(f"sample {i}: keypoints must be (N,2)")
        if kp_px.dim() != 2 or kp_px.shape[-1] != 2:
            raise RuntimeError(f"sample {i}: keypoints_px must be (N,2)")
        if d.dim() != 2 or d.shape[-1] != descriptor_dim:
            raise RuntimeError(f"sample {i}: descriptors must be (N,{descriptor_dim})")
        if s.dim() != 1:
            raise RuntimeError(f"sample {i}: scores must be (N,)")
        if not (kp.shape[0] == kp_px.shape[0] == d.shape[0] == s.shape[0]):
            raise RuntimeError(f"sample {i}: inconsistent N across outputs")
        norms = torch.linalg.norm(d, dim=1)
        if norms.numel() > 0 and not torch.allclose(norms.mean(), torch.tensor(1.0, device=norms.device), atol=5e-2):
            raise RuntimeError(f"sample {i}: descriptor norms drift from 1")


def assert_superpoint_frozen(model: torch.nn.Module) -> None:
    if hasattr(model, "superpoint"):
        for p in model.superpoint.parameters():
            if p.requires_grad:
                raise RuntimeError("SuperPoint parameter found trainable; freeze policy violated")
        if model.superpoint.training:
            raise RuntimeError("SuperPoint must stay in eval() mode")


def check_plateau_break(stats: Dict[str, float], lo: float, hi: float) -> None:
    loss = float(stats.get("loss", 0.0))
    if lo <= loss <= hi:
        raise RuntimeError(
            f"Loss is inside legacy plateau [{lo:.2f},{hi:.2f}] -> {loss:.4f}; milestone not met"
        )
