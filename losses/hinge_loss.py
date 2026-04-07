"""
losses/hinge_loss.py
====================
HomographyHingeLoss
--------------------
Descriptor hinge loss for training the XFeat keypoint detector to select
locations that produce highly discriminative SuperPoint descriptors.

Mathematical Formulation
------------------------
Given a pair of images (I₁, I₂) related by homography H, and two sets of
descriptors sampled at XFeat's detected keypoints:

  D₁ = {d₁ᵢ}  (N × 256, L2-normalised)  from I₁
  D₂ = {d₂ⱼ}  (M × 256, L2-normalised)  from I₂

Correspondence matrix (SuperPoint Eq. 4):
  s_{ij} = 1   if  ||H·k₁ᵢ - k₂ⱼ|| ≤ τ   (τ = correspondence threshold)
  s_{ij} = 0   otherwise

Score-Weighted Hinge Loss
--------------------------
Per-pair hinge loss (extended with score weighting):
  lᵈ(dᵢ, dⱼ; sᵢⱼ) =
      W_{ij} · [ λd · sᵢⱼ  · max(0, mp - dᵢᵀdⱼ)     ← positive term
               + (1 - sᵢⱼ) · max(0, dᵢᵀdⱼ - mn) ]   ← negative term

  W_{ij} = score₁ᵢ · score₂ⱼ  / mean(W)   (score weight matrix)

  mp = 1.0   positive margin (pull matching descs to cosine-sim = 1)
  mn = 0.2   negative margin (push non-matches below cosine-sim = 0.2)
  λd = 250   weighting to compensate for the pos/neg imbalance

Gradient w.r.t. score₁ᵢ
-------------------------
  ∂L/∂score₁ᵢ = (1/NM) Σⱼ [λd·s_{ij}·relu(mp-sim_{ij})
                             + (1-s_{ij})·relu(sim_{ij}-mn)] · score₂ⱼ

Both relu() terms are non-negative (clamped to 0 when already satisfied)
and score₂ⱼ ≥ 0 (sigmoid output), so the sum is always ≥ 0.  Gradient
descent therefore reduces scores wherever descriptor matches are poor,
teaching XFeat to avoid non-repeatable or ambiguous spots.

Repeatability Reward
---------------------
To also *increase* scores at geometrically repeatable positions:
  L_rep = -(1/n₁) Σᵢ [has_match₁ᵢ · score₁ᵢ]
         -(1/n₂) Σⱼ [has_match₂ⱼ · score₂ⱼ]

  where has_match₁ᵢ = max_j(S_{ij}) > 0

This rewards high confidence at positions that have a ground-truth
geometric correspondence in the paired image.

Combined:
  L_total = L_hinge + λ_rep · L_rep

Depth-Based Correspondences
-----------------------------
When a dense warp field (from depth reprojection) is provided instead of
an approximate planar homography, correspondence labels are more accurate
for real 3D scenes with parallax.

Reference: DeTone et al., "SuperPoint", CVPRW 2018 §3.4
"""

# Minimum positive depth (metres) for a 3-D point to be considered valid.
# Used both here (border check) and in MegaDepthDataset._compute_warp_field.
_MIN_DEPTH_Z: float = 0.01

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union


class HomographyHingeLoss(nn.Module):
    """
    Score-weighted hinge loss on descriptor cosine similarities with
    homography-induced (or depth-based) correspondence labels.

    Parameters
    ----------
    positive_margin : float
        mp — minimum desired cosine similarity for positive pairs.
        Default 1.0.

    negative_margin : float
        mn — maximum allowed cosine similarity for negative pairs.
        Default 0.2.

    lambda_d : float
        Weighting factor for positive pairs. Default 250.

    correspondence_threshold : float
        τ — pixel distance below which two keypoints are corresponding.
        Default 8.0 px.

    safe_radius : float
        Keypoints warped within this distance of the image border in
        image-2 are excluded from the loss.  Default 8.0 px.

    lambda_rep : float
        Weight for the repeatability reward term.  Rewards high scores
        at positions that have a geometric match in the paired image.
        Default 0.5.  Set 0.0 to disable.
    """

    def __init__(
        self,
        positive_margin:           float = 1.0,
        negative_margin:           float = 0.2,
        lambda_d:                  float = 250.0,
        correspondence_threshold:  float = 8.0,
        safe_radius:               float = 8.0,
        lambda_rep:                float = 0.5,
    ):
        super().__init__()
        assert positive_margin > negative_margin, (
            f"positive_margin ({positive_margin}) must exceed "
            f"negative_margin ({negative_margin})"
        )
        self.mp        = positive_margin
        self.mn        = negative_margin
        self.lambda_d  = lambda_d
        self.threshold = correspondence_threshold
        self.safe_r    = safe_radius
        self.lambda_rep = lambda_rep

    # ------------------------------------------------------------------
    # Geometric helpers
    # ------------------------------------------------------------------

    @staticmethod
    def warp_keypoints(
        keypoints: torch.Tensor,
        H_mat:     torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply a 3×3 homography to a set of 2D keypoints.

        Args
        ----
        keypoints : (N, 2)  [x, y] in pixel space
        H_mat     : (3, 3)  homography  I₁ → I₂

        Returns
        -------
        warped : (N, 2)  transformed keypoints in I₂ pixel space
        """
        N = keypoints.shape[0]
        dtype, device = keypoints.dtype, keypoints.device

        ones   = torch.ones(N, 1, dtype=dtype, device=device)
        kp_h   = torch.cat([keypoints, ones], dim=1)       # (N, 3)
        warped_h = (H_mat.to(dtype) @ kp_h.T)              # (3, N)
        w        = warped_h[2:3].clamp(min=1e-8)
        warped   = (warped_h[:2] / w).T                    # (N, 2)
        return warped

    def _build_correspondence_matrix(
        self,
        kp1:    torch.Tensor,
        kp2:    torch.Tensor,
        H_mat:  torch.Tensor,
        img2_hw: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Build S ∈ {0,1}^{N×M} from a planar homography."""
        N = kp1.shape[0]

        kp1_in_2 = self.warp_keypoints(kp1, H_mat)  # (N, 2)

        if img2_hw is not None:
            H2, W2 = img2_hw
            r = self.safe_r
            valid_mask = (
                (kp1_in_2[:, 0] >= r) & (kp1_in_2[:, 0] <= W2 - r) &
                (kp1_in_2[:, 1] >= r) & (kp1_in_2[:, 1] <= H2 - r)
            )
        else:
            valid_mask = torch.ones(N, dtype=torch.bool, device=kp1.device)

        dist = torch.norm(
            kp1_in_2.unsqueeze(1) - kp2.unsqueeze(0), dim=-1
        )  # (N, M)
        S = (dist <= self.threshold).float()
        S = S * valid_mask.float().unsqueeze(1)
        return S

    def _build_correspondence_from_warp(
        self,
        kp1:        torch.Tensor,
        kp2:        torch.Tensor,
        warp_field: torch.Tensor,
        warp_valid: Optional[torch.Tensor] = None,
        img2_hw:    Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Build S ∈ {0,1}^{N×M} from a dense depth-based warp field.

        warp_field[y, x] = (x2, y2) — where pixel (x, y) in image-1 maps
        to in image-2, computed via depth reprojection.  More accurate than
        the planar approximation H ≈ K₂·R₂₁·K₁⁻¹ for real 3D scenes.

        Args
        ----
        kp1        : (N, 2)  [x, y] keypoints in image-1
        kp2        : (M, 2)  [x, y] keypoints in image-2
        warp_field : (H, W, 2) dense warp (x2, y2) for each image-1 pixel
        warp_valid : (H, W) bool, True where depth was valid
        img2_hw    : (H2, W2) for border exclusion
        """
        N      = kp1.shape[0]
        device = kp1.device
        H_f, W_f = warp_field.shape[:2]

        xi = kp1[:, 0].long().clamp(0, W_f - 1)
        yi = kp1[:, 1].long().clamp(0, H_f - 1)

        kp1_in_2 = warp_field[yi, xi]          # (N, 2)

        if warp_valid is not None:
            valid = warp_valid[yi, xi].bool()   # (N,)
        else:
            valid = torch.ones(N, dtype=torch.bool, device=device)

        if img2_hw is not None:
            H2, W2 = img2_hw
            r = self.safe_r
            border_ok = (
                (kp1_in_2[:, 0] >= r) & (kp1_in_2[:, 0] <= W2 - r) &
                (kp1_in_2[:, 1] >= r) & (kp1_in_2[:, 1] <= H2 - r)
            )
            valid = valid & border_ok

        dist = torch.norm(
            kp1_in_2.unsqueeze(1) - kp2.unsqueeze(0), dim=-1
        )  # (N, M)
        S = (dist <= self.threshold).float()
        S = S * valid.float().unsqueeze(1)
        return S

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def forward(
        self,
        descriptors1: torch.Tensor,
        descriptors2: torch.Tensor,
        keypoints1:   torch.Tensor,
        keypoints2:   torch.Tensor,
        homography:   torch.Tensor,
        image2_hw:    Optional[Tuple[int, int]] = None,
        scores1:      Optional[torch.Tensor]    = None,
        scores2:      Optional[torch.Tensor]    = None,
        warp_field:   Optional[torch.Tensor]    = None,
        warp_valid:   Optional[torch.Tensor]    = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the score-weighted hinge loss for one image pair.

        Args
        ----
        descriptors1 : (N, 256)  L2-normalised descs from image 1
        descriptors2 : (M, 256)  L2-normalised descs from image 2
        keypoints1   : (N, 2)    pixel-space keypoints in image 1
        keypoints2   : (M, 2)    pixel-space keypoints in image 2
        homography   : (3, 3)    H: image1 → image2
        image2_hw    : (H, W)    image-2 size for border masking
        scores1      : (N,)      XFeat detection scores — enables gradient
                                 flow to the keypoint head via score weighting
        scores2      : (M,)      XFeat detection scores for image 2
        warp_field   : (H, W, 2) dense depth-based warp (overrides homography)
        warp_valid   : (H, W)    bool validity mask for warp_field

        Returns
        -------
        loss  : scalar tensor   (backprop-able)
        stats : dict of floats  (for logging, detached)
        """
        N = descriptors1.shape[0]
        M = descriptors2.shape[0]
        device = descriptors1.device

        # ── Edge cases ───────────────────────────────────────────────────
        if N == 0 or M == 0:
            dummy = torch.tensor(0.0, device=device, requires_grad=True)
            return dummy, {'loss': 0.0, 'n_pos': 0, 'n_neg': 0}

        # ── Step 1: Correspondence matrix ─────────────────────────────────
        if warp_field is not None:
            S = self._build_correspondence_from_warp(
                keypoints1, keypoints2, warp_field, warp_valid, image2_hw
            )
        else:
            S = self._build_correspondence_matrix(
                keypoints1, keypoints2, homography, image2_hw
            )
        S = S.detach()

        # ── Step 2: Pairwise cosine similarities ──────────────────────────
        sim = descriptors1 @ descriptors2.T   # (N, M)

        # ── Step 3: Score weight matrix (differentiable → kp_head) ────────
        # W[i,j] = score1[i] * score2[j], normalised so mean(W) = 1.
        # Gradient: ∂L/∂score1[i] = Σⱼ hinge_ij * score2[j] / NM
        # Both relu() terms and score2[j] are always ≥ 0, so this gradient
        # is non-negative: gradient descent reduces scores at positions with
        # poor descriptor matches.  The repeatability reward below provides
        # the complementary positive signal (raises scores at matched spots).
        if scores1 is not None and scores2 is not None:
            W = scores1.unsqueeze(1) * scores2.unsqueeze(0)  # (N, M)
            W = W / W.mean().clamp(min=1e-8)                 # mean-normalise
        else:
            W = torch.ones(N, M, device=device, dtype=sim.dtype)

        # ── Step 4: Hinge terms ───────────────────────────────────────────
        pos_loss = self.lambda_d * S       * W * F.relu(self.mp - sim)
        neg_loss = (1.0 - S)               * W * F.relu(sim - self.mn)

        total_pairs = float(N * M)
        hinge = (pos_loss + neg_loss).sum() / total_pairs

        # ── Step 5: Repeatability reward ──────────────────────────────────
        # For positions that have a geometric match, reward high scores.
        # This provides the *positive* gradient component that encourages
        # the kp_head to score good (repeatable) locations highly.
        rep = hinge.new_zeros(1).squeeze()
        if scores1 is not None and scores2 is not None and self.lambda_rep > 0:
            has_match_1 = (S.sum(dim=1) > 0).float()   # (N,) no grad needed
            has_match_2 = (S.sum(dim=0) > 0).float()   # (M,)
            n1 = has_match_1.sum().clamp(min=1.0)
            n2 = has_match_2.sum().clamp(min=1.0)
            rep = (
                -(has_match_1 * scores1).sum() / n1
                - (has_match_2 * scores2).sum() / n2
            ) / 2.0

        loss = hinge + self.lambda_rep * rep

        # ── Diagnostics (detached) ────────────────────────────────────────
        with torch.no_grad():
            n_pos = S.sum().item()
            n_neg = float(N * M) - n_pos
            eps   = 1e-8
            pos_sim_mean = (sim * S).sum().item()         / (n_pos + eps)
            neg_sim_mean = (sim * (1.0 - S)).sum().item() / (n_neg + eps)
            stats: Dict[str, float] = {
                'loss':          loss.item(),
                'hinge':         hinge.item(),
                'rep_loss':      rep.item() if self.lambda_rep > 0 else 0.0,
                'pos_loss_mean': pos_loss.sum().item() / total_pairs,
                'neg_loss_mean': neg_loss.sum().item() / total_pairs,
                'n_pos':         n_pos,
                'n_neg':         n_neg,
                'pos_sim_mean':  pos_sim_mean,
                'neg_sim_mean':  neg_sim_mean,
                'pos_ratio':     n_pos / total_pairs,
            }

        return loss, stats

    # ------------------------------------------------------------------
    # Batch wrapper
    # ------------------------------------------------------------------

    def forward_batch(
        self,
        desc1_list:   List[torch.Tensor],
        desc2_list:   List[torch.Tensor],
        kp1_list:     List[torch.Tensor],
        kp2_list:     List[torch.Tensor],
        homographies: torch.Tensor,
        image2_hws:   Optional[List[Tuple[int, int]]]  = None,
        scores1_list: Optional[List[torch.Tensor]]     = None,
        scores2_list: Optional[List[torch.Tensor]]     = None,
        warp_fields:  Optional[torch.Tensor]           = None,
        warp_valids:  Optional[torch.Tensor]           = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute mean loss across a batch of image pairs.

        Args
        ----
        desc1_list    : list[Tensor(N_b, 256)]
        desc2_list    : list[Tensor(M_b, 256)]
        kp1_list      : list[Tensor(N_b, 2)]
        kp2_list      : list[Tensor(M_b, 2)]
        homographies  : (B, 3, 3)
        image2_hws    : optional list[(H, W)] per image
        scores1_list  : optional list[Tensor(N_b,)]  XFeat scores for img1
        scores2_list  : optional list[Tensor(M_b,)]  XFeat scores for img2
        warp_fields   : optional (B, H, W, 2) depth-based warp fields
        warp_valids   : optional (B, H, W) bool validity masks

        Returns
        -------
        mean_loss : scalar tensor
        mean_stats: dict of averaged stats
        """
        B = len(desc1_list)
        assert len(desc2_list) == B and len(kp1_list) == B and len(kp2_list) == B

        total_loss   = torch.tensor(0.0, device=desc1_list[0].device)
        total_stats: Dict[str, float] = {}

        for b in range(B):
            img2_hw = image2_hws[b]   if image2_hws   is not None else None
            sc1     = scores1_list[b] if scores1_list  is not None else None
            sc2     = scores2_list[b] if scores2_list  is not None else None
            wf      = warp_fields[b]  if warp_fields   is not None else None
            wv      = warp_valids[b]  if warp_valids   is not None else None

            loss_b, stats_b = self.forward(
                desc1_list[b], desc2_list[b],
                kp1_list[b],   kp2_list[b],
                homographies[b],
                img2_hw,
                sc1, sc2, wf, wv,
            )
            total_loss = total_loss + loss_b
            for k, v in stats_b.items():
                total_stats[k] = total_stats.get(k, 0.0) + v

        mean_loss  = total_loss / B
        mean_stats = {k: v / B for k, v in total_stats.items()}
        mean_stats['loss'] = mean_loss.item()

        return mean_loss, mean_stats

