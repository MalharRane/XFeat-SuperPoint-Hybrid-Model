from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml_config(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("YAML config must map keys to values")
    return cfg


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("hybrid_model_v2 trainer")
    parser.add_argument("--config", type=str, default="hybrid_model_v2/config.yaml")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--mode", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--logging_backend", type=str, default=None)
    parser.add_argument("--model_selection_metric", type=str, default=None)
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--no_mixed_precision", action="store_true")
    return parser


def merge_config_with_args(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    out = dict(cfg)
    for k, v in vars(args).items():
        if k == "config" or k == "resume":
            continue
        if v is not None:
            out[k] = v

    if args.mixed_precision:
        out["mixed_precision"] = True
    if args.no_mixed_precision:
        out["mixed_precision"] = False

    out["resume"] = args.resume
    return out
