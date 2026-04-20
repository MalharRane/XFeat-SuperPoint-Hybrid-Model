from __future__ import annotations

from typing import Dict, List


def mean_stats(stats_list: List[Dict[str, float]]) -> Dict[str, float]:
    if not stats_list:
        return {}
    out: Dict[str, float] = {}
    for stats in stats_list:
        for k, v in stats.items():
            out[k] = out.get(k, 0.0) + float(v)
    n = float(len(stats_list))
    return {k: v / n for k, v in out.items()}


def pick_model_score(metric: str, stats: Dict[str, float], val_loss: float) -> float:
    metric = str(metric).lower()
    if metric == "sim_gap":
        return float(stats.get("sim_gap", float("-inf")))
    if metric == "repeatability_mean":
        return float(stats.get("repeatability_mean", float("-inf")))
    return -float(val_loss)
