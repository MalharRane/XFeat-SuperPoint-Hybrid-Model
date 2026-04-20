from __future__ import annotations

from typing import Iterable

import torch


def tensor_is_finite(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x).all().item())


def grads_are_finite(params: Iterable[torch.nn.Parameter]) -> bool:
    for p in params:
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            return False
    return True


def is_amp_related_error(exc: RuntimeError) -> bool:
    msg = str(exc).lower()
    return (
        "autocast" in msg
        or "gradscaler" in msg
        or "float16" in msg
        or "half" in msg
        or "bfloat16" in msg
    )
