"""
Fixed A/B benchmark for checkpoint comparison + LightGlue match evaluation.

Evaluates two checkpoints on the same held-out pairs and reports:
  - RANSAC inlier ratio
  - MMA@{1,3,5}px (configurable)
  - match precision at a pixel threshold
  - number of matches
  - sim_gap and repeatability_mean sanity diagnostics

Optional: saves LightGlue match visualizations.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from data.megadepth_dataset import build_dataloader
from losses.hinge_loss import HomographyHingeLoss
from train import DEFAULT_CONFIG, build_model

try:
    from lightglue import LightGlue  # type: ignore
except Exception:
    LightGlue = None


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_thresholds(raw: str) -> List[float]:
    vals = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            vals.append(float(token))
    if not vals:
        raise ValueError("mma_thresholds must contain at least one value.")
    return vals


def _load_cfg(config_path: Optional[str]) -> Dict:
    cfg = DEFAULT_CONFIG.copy()
    if config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg.update(yaml.safe_load(f) or {})
    return cfg


def _load_checkpoint_weights(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> None:
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=False)


def _to_optional_tensor_batch(
    value: object,
    device: torch.device,
) -> Optional[Union[torch.Tensor, List[Optional[torch.Tensor]]]]:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        out: List[Optional[torch.Tensor]] = []
        for v in value:
            out.append(v.to(device) if isinstance(v, torch.Tensor) else None)
        return out
    return None


def _extract_pair_items(batch: Dict, device: torch.device) -> Dict[str, object]:
    image1 = batch["image1"].to(device)
    image2 = batch["image2"].to(device)
    homography = batch["homography"].to(device)
    warp_field = _to_optional_tensor_batch(batch.get("warp_field"), device)
    warp_valid = _to_optional_tensor_batch(batch.get("warp_valid"), device)
    return {
        "image1": image1,
        "image2": image2,
        "homography": homography,
        "warp_field": warp_field,
        "warp_valid": warp_valid,
    }


def _prepare_lightglue_features(
    out: Dict[str, List[torch.Tensor]],
    image_hw: Tuple[int, int],
) -> Dict[str, torch.Tensor]:
    h, w = image_hw
    keypoints = out["keypoints_px"][0].unsqueeze(0)
    descriptors = out["descriptors"][0].unsqueeze(0)
    features = {
        "keypoints": keypoints,
        "descriptors": descriptors,
        "image_size": torch.tensor([[w, h]], device=keypoints.device, dtype=torch.float32),
    }
    scores = out.get("scores")
    if scores:
        features["keypoint_scores"] = scores[0].unsqueeze(0).float()
    return features


def _get_matches_from_lightglue(pred: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    if "matches" in pred:
        matches = pred["matches"][0]
        if matches.numel() == 0:
            return matches.new_zeros((0,), dtype=torch.long), matches.new_zeros((0,), dtype=torch.long)
        return matches[:, 0].long(), matches[:, 1].long()

    if "matches0" in pred:
        m0 = pred["matches0"][0].long()
        idx0 = torch.nonzero(m0 >= 0, as_tuple=False).squeeze(1)
        idx1 = m0[idx0]
        return idx0, idx1

    raise RuntimeError("Unknown LightGlue output format. Expected 'matches' or 'matches0'.")


def _warp_with_optional_depth(
    kp1: torch.Tensor,
    homography: torch.Tensor,
    warp_field: Optional[torch.Tensor],
    warp_valid: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if warp_field is not None:
        h, w = warp_field.shape[:2]
        xi = kp1[:, 0].long().clamp(0, w - 1)
        yi = kp1[:, 1].long().clamp(0, h - 1)
        warped = warp_field[yi, xi]
        if warp_valid is None:
            valid = torch.ones(kp1.shape[0], device=kp1.device, dtype=torch.bool)
        else:
            valid = warp_valid[yi, xi].bool()
        return warped, valid

    warped = HomographyHingeLoss.warp_keypoints(kp1, homography)
    valid = torch.ones(kp1.shape[0], device=kp1.device, dtype=torch.bool)
    return warped, valid


def _unwrap_single_sample_optional_tensor(
    value: Optional[Union[torch.Tensor, List[Optional[torch.Tensor]]]],
) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if isinstance(value, list):
        return value[0]
    if isinstance(value, torch.Tensor) and value.ndim >= 1:
        return value[0]
    return value


def _compute_pair_metrics(
    out1: Dict[str, List[torch.Tensor]],
    out2: Dict[str, List[torch.Tensor]],
    pair: Dict[str, object],
    matcher,
    loss_fn: HomographyHingeLoss,
    mma_thresholds: Sequence[float],
    precision_threshold: float,
    ransac_threshold: float,
) -> Dict[str, float]:
    image1 = pair["image1"]
    image2 = pair["image2"]
    h, w = int(image1.shape[2]), int(image1.shape[3])

    feats0 = _prepare_lightglue_features(out1, (h, w))
    feats1 = _prepare_lightglue_features(out2, (h, w))
    pred = matcher({"image0": feats0, "image1": feats1})
    idx0, idx1 = _get_matches_from_lightglue(pred)

    kp1_all = out1["keypoints_px"][0]
    kp2_all = out2["keypoints_px"][0]
    n_matches = int(idx0.numel())
    if n_matches == 0:
        metrics = {
            "n_matches": 0.0,
            "inlier_ratio": 0.0,
            "precision": 0.0,
        }
        for t in mma_thresholds:
            metrics[f"mma@{t:g}px"] = 0.0
        _, stats = loss_fn.forward_batch(
            desc1_list=out1["descriptors"],
            desc2_list=out2["descriptors"],
            kp1_list=out1["keypoints_px"],
            kp2_list=out2["keypoints_px"],
            homographies=pair["homography"],
            image2_hws=[(h, w)],
            scores1_list=out1.get("scores"),
            scores2_list=out2.get("scores"),
            warp_fields=pair.get("warp_field"),
            warp_valids=pair.get("warp_valid"),
        )
        metrics["sim_gap"] = float(stats.get("sim_gap", 0.0))
        metrics["repeatability_mean"] = float(stats.get("repeatability_mean", 0.0))
        return metrics

    kp1 = kp1_all[idx0]
    kp2 = kp2_all[idx1]

    warp_field = _unwrap_single_sample_optional_tensor(pair.get("warp_field"))
    warp_valid = _unwrap_single_sample_optional_tensor(pair.get("warp_valid"))

    gt_kp2, valid_gt = _warp_with_optional_depth(
        kp1=kp1,
        homography=pair["homography"][0],
        warp_field=warp_field,
        warp_valid=warp_valid,
    )

    errs = torch.norm(gt_kp2 - kp2, dim=1)
    valid_errs = errs[valid_gt]

    precision = float((valid_errs <= precision_threshold).float().mean().item()) if valid_errs.numel() > 0 else 0.0
    mma = {
        f"mma@{t:g}px": (
            float((valid_errs <= t).float().mean().item()) if valid_errs.numel() > 0 else 0.0
        )
        for t in mma_thresholds
    }

    inlier_mask = _compute_ransac_inlier_mask(kp1, kp2, ransac_threshold)
    inlier_ratio = float(inlier_mask.mean()) if inlier_mask is not None else 0.0

    _, stats = loss_fn.forward_batch(
        desc1_list=out1["descriptors"],
        desc2_list=out2["descriptors"],
        kp1_list=out1["keypoints_px"],
        kp2_list=out2["keypoints_px"],
        homographies=pair["homography"],
        image2_hws=[(h, w)],
        scores1_list=out1.get("scores"),
        scores2_list=out2.get("scores"),
        warp_fields=pair.get("warp_field"),
        warp_valids=pair.get("warp_valid"),
    )

    metrics = {
        "n_matches": float(n_matches),
        "inlier_ratio": inlier_ratio,
        "precision": precision,
        "sim_gap": float(stats.get("sim_gap", 0.0)),
        "repeatability_mean": float(stats.get("repeatability_mean", 0.0)),
    }
    metrics.update(mma)
    return metrics


def _compute_ransac_inlier_mask(
    kp1: torch.Tensor,
    kp2: torch.Tensor,
    ransac_threshold: float,
) -> Optional[np.ndarray]:
    if kp1.shape[0] < 4 or kp2.shape[0] < 4:
        return None
    _, inlier_mask = cv2.findHomography(
        kp1.detach().cpu().numpy().astype(np.float32),
        kp2.detach().cpu().numpy().astype(np.float32),
        cv2.RANSAC,
        ransac_threshold,
    )
    if inlier_mask is None:
        return None
    return inlier_mask.reshape(-1).astype(bool)


def _draw_matches(
    image1: torch.Tensor,
    image2: torch.Tensor,
    kp1: torch.Tensor,
    kp2: torch.Tensor,
    inlier_mask: Optional[np.ndarray],
    save_path: Path,
) -> None:
    im1 = image1[0, 0].detach().cpu().numpy()
    im2 = image2[0, 0].detach().cpu().numpy()
    h, w = im1.shape

    canvas = np.concatenate([im1, im2], axis=1)
    plt.figure(figsize=(12, 5))
    plt.imshow(canvas, cmap="gray")
    plt.axis("off")

    k1 = kp1.detach().cpu().numpy()
    k2 = kp2.detach().cpu().numpy()
    k2_shift = k2.copy()
    k2_shift[:, 0] += w

    for i in range(k1.shape[0]):
        is_inlier = bool(inlier_mask[i]) if inlier_mask is not None else False
        color = "lime" if is_inlier else "red"
        plt.plot([k1[i, 0], k2_shift[i, 0]], [k1[i, 1], k2_shift[i, 1]], color=color, linewidth=0.7, alpha=0.8)
    plt.scatter(k1[:, 0], k1[:, 1], s=8, c="cyan")
    plt.scatter(k2_shift[:, 0], k2_shift[:, 1], s=8, c="yellow")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close()


def _evaluate_checkpoint(
    model: torch.nn.Module,
    loader: Iterable[Dict],
    device: torch.device,
    num_pairs: int,
    matcher,
    loss_fn: HomographyHingeLoss,
    mma_thresholds: Sequence[float],
    precision_threshold: float,
    ransac_threshold: float,
    vis_dir: Optional[Path] = None,
    save_vis_count: int = 0,
) -> Dict[str, float]:
    model.eval()
    totals: Dict[str, float] = {}
    seen = 0

    with torch.no_grad():
        for batch in loader:
            if seen >= num_pairs:
                break

            pair = _extract_pair_items(batch, device)
            out1 = model.forward_train(pair["image1"])
            out2 = model.forward_train(pair["image2"])

            metrics = _compute_pair_metrics(
                out1=out1,
                out2=out2,
                pair=pair,
                matcher=matcher,
                loss_fn=loss_fn,
                mma_thresholds=mma_thresholds,
                precision_threshold=precision_threshold,
                ransac_threshold=ransac_threshold,
            )
            for k, v in metrics.items():
                totals[k] = totals.get(k, 0.0) + float(v)

            if vis_dir is not None and seen < save_vis_count:
                pred = matcher(
                    {
                        "image0": _prepare_lightglue_features(out1, (pair["image1"].shape[2], pair["image1"].shape[3])),
                        "image1": _prepare_lightglue_features(out2, (pair["image2"].shape[2], pair["image2"].shape[3])),
                    }
                )
                idx0, idx1 = _get_matches_from_lightglue(pred)
                if idx0.numel() > 0:
                    kp1 = out1["keypoints_px"][0][idx0]
                    kp2 = out2["keypoints_px"][0][idx1]
                    # We only need the inlier mask for visualization coloring.
                    inlier_mask = _compute_ransac_inlier_mask(kp1, kp2, ransac_threshold)
                    _draw_matches(
                        image1=pair["image1"],
                        image2=pair["image2"],
                        kp1=kp1,
                        kp2=kp2,
                        inlier_mask=inlier_mask,
                        save_path=vis_dir / f"pair_{seen:04d}.png",
                    )

            seen += 1

    return {k: v / max(seen, 1) for k, v in totals.items()}


def _build_eval_loader(cfg: Dict, args: argparse.Namespace):
    return build_dataloader(
        mode=cfg["mode"],
        root=cfg["data_root"],
        image_size=(cfg["image_height"], cfg["image_width"]),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        scene_info_dir=cfg.get("scene_info_dir"),
        min_overlap=cfg["min_overlap"],
        max_overlap=cfg["max_overlap"],
        max_pairs_per_scene=cfg.get("max_pairs_per_scene", 1000),
        augment=False,
    )


def _print_summary(title: str, metrics: Dict[str, float]) -> None:
    keys = [
        "n_matches",
        "precision",
        "inlier_ratio",
        "mma@1px",
        "mma@3px",
        "mma@5px",
        "sim_gap",
        "repeatability_mean",
    ]
    print(f"\n{title}")
    print("-" * len(title))
    for k in keys:
        if k in metrics:
            print(f"{k:>20s}: {metrics[k]:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed A/B benchmark with LightGlue matching.")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["synthetic", "megadepth"],
        help="Dataset mode.",
    )
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--scene_info_dir", type=str, default=None)
    parser.add_argument("--old_ckpt", type=str, required=True, help="Baseline checkpoint")
    parser.add_argument("--new_ckpt", type=str, required=True, help="Candidate checkpoint")
    parser.add_argument("--num_pairs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--mma_thresholds", type=str, default="1,3,5")
    parser.add_argument("--precision_threshold", type=float, default=3.0)
    parser.add_argument("--ransac_threshold", type=float, default=3.0)
    parser.add_argument("--lightglue_features", type=str, default="superpoint")
    parser.add_argument("--output_dir", type=str, default="benchmarks/ab_lightglue")
    parser.add_argument("--save_vis_count", type=int, default=10)
    args = parser.parse_args()

    _set_seed(args.seed)
    cfg = _load_cfg(args.config)
    if args.mode is not None:
        cfg["mode"] = args.mode
    if args.data_root is not None:
        cfg["data_root"] = args.data_root
    if args.scene_info_dir is not None:
        cfg["scene_info_dir"] = args.scene_info_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mma_thresholds = _parse_thresholds(args.mma_thresholds)

    if LightGlue is None:
        raise RuntimeError(
            "LightGlue is required for this benchmark. Install it in your environment "
            "(e.g. pip install lightglue)."
        )

    matcher = LightGlue(features=args.lightglue_features).eval().to(device)
    loss_fn = HomographyHingeLoss(
        positive_margin=cfg["positive_margin"],
        negative_margin=cfg["negative_margin"],
        lambda_d=cfg["lambda_d"],
        lambda_rep=cfg.get("lambda_rep", 0.5),
        correspondence_threshold=cfg["correspondence_threshold"],
        safe_radius=cfg.get("safe_radius", 8.0),
        balance_pos_neg=cfg.get("balance_pos_neg", True),
    )

    val_loader = _build_eval_loader(cfg, args)

    old_model = build_model(cfg, device)
    _load_checkpoint_weights(old_model, args.old_ckpt, device)
    old_vis = Path(args.output_dir) / "old_vis"
    old_metrics = _evaluate_checkpoint(
        model=old_model,
        loader=val_loader,
        device=device,
        num_pairs=args.num_pairs,
        matcher=matcher,
        loss_fn=loss_fn,
        mma_thresholds=mma_thresholds,
        precision_threshold=args.precision_threshold,
        ransac_threshold=args.ransac_threshold,
        vis_dir=old_vis,
        save_vis_count=args.save_vis_count,
    )

    # Rebuild loader with same seed to keep pair ordering fixed for new checkpoint.
    _set_seed(args.seed)
    val_loader = _build_eval_loader(cfg, args)

    new_model = build_model(cfg, device)
    _load_checkpoint_weights(new_model, args.new_ckpt, device)
    new_vis = Path(args.output_dir) / "new_vis"
    new_metrics = _evaluate_checkpoint(
        model=new_model,
        loader=val_loader,
        device=device,
        num_pairs=args.num_pairs,
        matcher=matcher,
        loss_fn=loss_fn,
        mma_thresholds=mma_thresholds,
        precision_threshold=args.precision_threshold,
        ransac_threshold=args.ransac_threshold,
        vis_dir=new_vis,
        save_vis_count=args.save_vis_count,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "old": old_metrics,
        "new": new_metrics,
        "delta_new_minus_old": {
            k: float(new_metrics.get(k, 0.0) - old_metrics.get(k, 0.0))
            for k in set(old_metrics) | set(new_metrics)
        },
    }
    with open(out_dir / "summary.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=True)

    _print_summary("OLD CHECKPOINT", old_metrics)
    _print_summary("NEW CHECKPOINT", new_metrics)
    _print_summary("DELTA (new - old)", summary["delta_new_minus_old"])
    print(f"\nSaved summary: {out_dir / 'summary.yaml'}")
    print(f"Saved match visualizations to: {old_vis} and {new_vis}")


if __name__ == "__main__":
    main()
