from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-8


class ScoreWeightedHingeRepeatabilityLoss(nn.Module):
    def __init__(
        self,
        positive_margin: float = 0.5,
        negative_margin: float = 0.2,
        lambda_d: float = 25.0,
        lambda_rep: float = 2.0,
        correspondence_threshold: float = 4.0,
        safe_radius: float = 8.0,
        balance_pos_neg: bool = True,
    ):
        super().__init__()
        self.mp = float(positive_margin)
        self.mn = float(negative_margin)
        self.lambda_d = float(lambda_d)
        self.lambda_rep = float(lambda_rep)
        self.threshold = float(correspondence_threshold)
        self.safe_radius = float(safe_radius)
        self.balance_pos_neg = bool(balance_pos_neg)

    @staticmethod
    def _warp_kp_h(kp: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        n = kp.shape[0]
        ones = torch.ones((n, 1), device=kp.device, dtype=kp.dtype)
        p = torch.cat([kp, ones], dim=1)
        wh = (H.to(kp.dtype) @ p.T)
        return (wh[:2] / wh[2:].clamp(min=_EPS)).T

    def _build_correspondence_from_homography(
        self,
        kp1: torch.Tensor,
        kp2: torch.Tensor,
        H: torch.Tensor,
        image2_hw: Tuple[int, int],
        depth_valid_1: Optional[torch.Tensor],
    ) -> torch.Tensor:
        warped = self._warp_kp_h(kp1, H)
        h2, w2 = image2_hw
        r = self.safe_radius
        valid = (
            (warped[:, 0] >= r)
            & (warped[:, 0] <= w2 - r)
            & (warped[:, 1] >= r)
            & (warped[:, 1] <= h2 - r)
        )
        if depth_valid_1 is not None and depth_valid_1.numel() > 0:
            yi = kp1[:, 1].long().clamp(0, depth_valid_1.shape[-2] - 1)
            xi = kp1[:, 0].long().clamp(0, depth_valid_1.shape[-1] - 1)
            dv = depth_valid_1[yi, xi] > 0.0
            valid = valid & dv

        dist = torch.norm(warped.unsqueeze(1) - kp2.unsqueeze(0), dim=-1)
        S = (dist <= self.threshold).float()
        return S * valid.float().unsqueeze(1)

    def _build_correspondence_from_warp(
        self,
        kp1: torch.Tensor,
        kp2: torch.Tensor,
        warp_field: torch.Tensor,
        warp_valid: Optional[torch.Tensor],
        image2_hw: Tuple[int, int],
    ) -> torch.Tensor:
        h, w = warp_field.shape[:2]
        xi = kp1[:, 0].long().clamp(0, w - 1)
        yi = kp1[:, 1].long().clamp(0, h - 1)
        kp1_in_2 = warp_field[yi, xi]

        valid = torch.ones(kp1.shape[0], dtype=torch.bool, device=kp1.device)
        if warp_valid is not None:
            valid = valid & warp_valid[yi, xi].bool()

        h2, w2 = image2_hw
        r = self.safe_radius
        border_ok = (
            (kp1_in_2[:, 0] >= r)
            & (kp1_in_2[:, 0] <= w2 - r)
            & (kp1_in_2[:, 1] >= r)
            & (kp1_in_2[:, 1] <= h2 - r)
        )
        valid = valid & border_ok

        dist = torch.norm(kp1_in_2.unsqueeze(1) - kp2.unsqueeze(0), dim=-1)
        S = (dist <= self.threshold).float()
        return S * valid.float().unsqueeze(1)

    def forward_pair(
        self,
        d1: torch.Tensor,
        d2: torch.Tensor,
        kp1: torch.Tensor,
        kp2: torch.Tensor,
        scores1: torch.Tensor,
        scores2: torch.Tensor,
        homography: torch.Tensor,
        image2_hw: Tuple[int, int],
        warp_field: Optional[torch.Tensor] = None,
        warp_valid: Optional[torch.Tensor] = None,
        depth_valid_1: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        n, m = d1.shape[0], d2.shape[0]
        if n == 0 or m == 0:
            z = torch.tensor(0.0, device=d1.device, requires_grad=True)
            return z, {"loss": 0.0, "n_pos": 0.0, "n_neg": 0.0}

        if warp_field is not None:
            S = self._build_correspondence_from_warp(
                kp1, kp2, warp_field, warp_valid, image2_hw
            )
        else:
            S = self._build_correspondence_from_homography(
                kp1, kp2, homography, image2_hw, depth_valid_1
            )
        S = S.detach()

        sim = d1 @ d2.T
        w = scores1.float().unsqueeze(1) * scores2.float().unsqueeze(0)
        w = w / w.mean().clamp(min=1e-6)

        pos = self.lambda_d * S * w * F.relu(self.mp - sim)
        neg = (1.0 - S) * w * F.relu(sim - self.mn)

        if self.balance_pos_neg:
            n_pos = S.sum().clamp(min=1.0)
            n_neg = (1.0 - S).sum().clamp(min=1.0)
            hinge = 0.5 * (pos.sum() / n_pos + neg.sum() / n_neg)
        else:
            hinge = (pos + neg).mean()

        # ── Contrastive repeatability loss ─────────────────────────────────
        # FIX: Replace one-sided rep_loss with a contrastive formulation.
        #
        # PROBLEM WITH PREVIOUS FORMULATION:
        #   rep_loss = -mean(score at matched kps)
        #
        # This saturates in ~3 epochs: all matched kp scores → 1.0, gradient → 0.
        # The optimizer reaches the minimum (rep=-1.0) by setting EVERY score
        # to 1.0 indiscriminately — no discrimination between matched and
        # unmatched positions.  After saturation, this term provides zero
        # gradient and the detector stops learning.
        #
        # FIX — CONTRASTIVE FORMULATION:
        #   rep_loss = -(mean_score_matched - mean_score_unmatched)
        #
        # This cannot be short-circuited by setting all scores to 1.0:
        #   If all scores = 1.0: rep_loss = -(1.0 - 1.0) = 0.0  (BAD — no credit)
        #   If matched=1.0, unmatched=0.0: rep_loss = -1.0        (GOOD — global min)
        #
        # The gradient pushes matched kp scores UP and unmatched kp scores DOWN
        # simultaneously.  The detector must DISCRIMINATE between positions that
        # have visual correspondences and positions that don't.  This is exactly
        # what a keypoint detector is supposed to do.
        #
        # With n_pos≈90 (4px threshold) out of 1024 kps, ~934 kps are
        # "unmatched" and now receive a gradient signal for the first time.
        has1 = (S.sum(dim=1) > 0).float()   # 1 if kp1[i] has any match in kp2
        has2 = (S.sum(dim=0) > 0).float()   # 1 if kp2[j] has any match in kp1
        no_match1 = 1.0 - has1
        no_match2 = 1.0 - has2

        s1 = scores1.float()
        s2 = scores2.float()

        matched_score1   = (has1 * s1).sum() / has1.sum().clamp(min=1.0)
        unmatched_score1 = (no_match1 * s1).sum() / no_match1.sum().clamp(min=1.0)
        matched_score2   = (has2 * s2).sum() / has2.sum().clamp(min=1.0)
        unmatched_score2 = (no_match2 * s2).sum() / no_match2.sum().clamp(min=1.0)

        # Minimize loss ↔ maximise (matched - unmatched) score gap.
        rep = -((matched_score1 - unmatched_score1) + (matched_score2 - unmatched_score2)) * 0.5

        loss = hinge + self.lambda_rep * rep

        with torch.no_grad():
            n_pos_f = float(S.sum().item())
            n_tot = float(n * m)
            n_neg_f = n_tot - n_pos_f
            pos_sim = float((sim * S).sum().item() / (n_pos_f + _EPS))
            neg_sim = float((sim * (1.0 - S)).sum().item() / (n_neg_f + _EPS))
            stats = {
                "loss": float(loss.item()),
                "hinge": float(hinge.item()),
                "rep_loss": float(rep.item()),
                "rep_matched_score": float((0.5 * (matched_score1 + matched_score2)).item()),
                "rep_unmatched_score": float((0.5 * (unmatched_score1 + unmatched_score2)).item()),
                "n_pos": n_pos_f,
                "n_neg": n_neg_f,
                "pos_sim_mean": pos_sim,
                "neg_sim_mean": neg_sim,
                "sim_gap": pos_sim - neg_sim,
                "repeatability_1": float(has1.mean().item()),
                "repeatability_2": float(has2.mean().item()),
                "repeatability_mean": float(0.5 * (has1.mean().item() + has2.mean().item())),
            }
        return loss, stats

    def forward_batch(
        self,
        out1: Dict[str, List[torch.Tensor]],
        out2: Dict[str, List[torch.Tensor]],
        homographies: torch.Tensor,
        image2_hw: Tuple[int, int],
        warp_fields: Optional[Union[torch.Tensor, List[Optional[torch.Tensor]]]] = None,
        warp_valids: Optional[Union[torch.Tensor, List[Optional[torch.Tensor]]]] = None,
        depth_valid_1: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        b = len(out1["descriptors"])
        total = torch.tensor(0.0, device=out1["descriptors"][0].device)
        agg: Dict[str, float] = {}

        for i in range(b):
            wf = (
                warp_fields[i]
                if isinstance(warp_fields, list)
                else (warp_fields[i] if warp_fields is not None else None)
            )
            wv = (
                warp_valids[i]
                if isinstance(warp_valids, list)
                else (warp_valids[i] if warp_valids is not None else None)
            )
            dv = depth_valid_1[i, 0] if isinstance(depth_valid_1, torch.Tensor) else None

            li, st = self.forward_pair(
                d1=out1["descriptors"][i],
                d2=out2["descriptors"][i],
                kp1=out1["keypoints_px"][i],
                kp2=out2["keypoints_px"][i],
                scores1=out1["scores"][i],
                scores2=out2["scores"][i],
                homography=homographies[i],
                image2_hw=image2_hw,
                warp_field=wf,
                warp_valid=wv,
                depth_valid_1=dv,
            )
            total = total + li
            for k, v in st.items():
                agg[k] = agg.get(k, 0.0) + float(v)

        mean_loss = total / max(b, 1)
        mean_stats_out = {k: v / max(b, 1) for k, v in agg.items()}
        mean_stats_out["loss"] = float(mean_loss.item())
        return mean_loss, mean_stats_out