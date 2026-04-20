from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Dict, Tuple

import torch


def ensure_file(path: Path, url: str) -> Path:
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, str(path))
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"Failed to download weights from {url}")
    return path


def validate_state_compat(module: torch.nn.Module, state_dict: Dict[str, torch.Tensor], min_overlap_ratio: float = 0.10) -> Tuple[int, int]:
    current = set(module.state_dict().keys())
    incoming = set(state_dict.keys())
    overlap = len(current.intersection(incoming))
    total = len(current)
    ratio = overlap / max(total, 1)
    if ratio < float(min_overlap_ratio):
        raise RuntimeError(
            f"Checkpoint/model overlap too low: {overlap}/{total} ({ratio:.3f}) < {min_overlap_ratio:.3f}"
        )
    return overlap, total


def load_weights_strictish(
    module: torch.nn.Module,
    weight_path: Path,
    *,
    module_name: str,
    strict: bool = False,
    min_overlap_ratio: float = 0.10,
) -> Tuple[int, int, int, int]:
    payload = torch.load(str(weight_path), map_location="cpu")
    state = payload.get("model", payload)
    if not isinstance(state, dict):
        raise RuntimeError(f"{module_name} checkpoint payload invalid")

    overlap, total = validate_state_compat(module, state, min_overlap_ratio=min_overlap_ratio)
    res = module.load_state_dict(state, strict=strict)
    missing = len(list(res.missing_keys))
    unexpected = len(list(res.unexpected_keys))
    return overlap, total, missing, unexpected
