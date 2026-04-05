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

Per-pair hinge loss (SuperPoint Eq. 6):
  lᵈ(dᵢ, dⱼ; sᵢⱼ) =
      λd ·  sᵢⱼ  · max(0, mp - dᵢᵀdⱼ)    ← positive term
    + (1 - sᵢⱼ) · max(0, dᵢᵀdⱼ - mn)     ← negative term

  mp = 1.0   positive margin (pull matching descs to cosine-sim = 1)
  mn = 0.2   negative margin (push non-matches below cosine-sim = 0.2)
  λd = 250   weighting to compensate for the pos/neg imbalance

Total loss (SuperPoint Eq. 5):
  Lᵈ = (1 / (N·M)) · Σᵢ Σⱼ lᵈ(dᵢ, dⱼ; sᵢⱼ)

Training Signal
---------------
  • Matching pairs:     loss is non-zero when cosine-sim < 1.0
                        → gradient pulls XFeat to detect REPEATABLE spots
  • Non-matching pairs: loss is non-zero when cosine-sim > 0.2
                        → gradient pushes XFeat away from AMBIGUOUS spots

Reference: DeTone et al., "SuperPoint", CVPRW 2018 §3.4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union


class HomographyHingeLoss(nn.Module):
    """
    Hinge loss on descriptor cosine similarities with homography-induced
    correspondence labels.

    Parameters
    ----------
    positive_margin : float
        mp — minimum desired cosine similarity for positive pairs.
        Default 1.0 (matching descriptors should be identical directions).

    negative_margin : float
        mn — maximum allowed cosine similarity for negative pairs.
        Default 0.2 (non-matching descs must be < 0.2 similar).

    lambda_d : float
        Weighting factor for positive pairs. Necessary because the number
        of positive pairs N_pos is much smaller than N_neg = N·M - N_pos.
        Default 250 (from SuperPoint's published training setup).

    correspondence_threshold : float
        τ — pixel distance (in image-1 space) below which two keypoints
        are declared corresponding after warping by H.
        Default 8.0 px (one descriptor-cell width).

    safe_radius : float
        Keypoints warped within this radius of the image border in image-2
        are excluded from loss computation (unreliable boundary estimates).
        Default 8.0 px.
    """

    def __init__(
        self,
        positive_margin:           float = 1.0,
        negative_margin:           float = 0.2,
        lambda_d:                  float = 250.0,
        correspondence_threshold:  float = 8.0,
        safe_radius:               float = 8.0,
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

        Formula: p' = π(H · [x, y, 1]ᵀ),   π([u,v,w]) = [u/w, v/w]

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

        # Homogeneous coordinates (N, 3)
        ones = torch.ones(N, 1, dtype=dtype, device=device)
        kp_h = torch.cat([keypoints, ones], dim=1)      # (N, 3)

        # Apply homography  (3×3) @ (3×N) → (3×N)
        warped_h = (H_mat.to(dtype) @ kp_h.T)           # (3, N)

        # Perspective division
        w = warped_h[2:3].clamp(min=1e-8)               # (1, N)
        warped = (warped_h[:2] / w).T                   # (N, 2)

        return warped

    def _build_correspondence_matrix(
        self,
        kp1:    torch.Tensor,
        kp2:    torch.Tensor,
        H_mat:  torch.Tensor,
        img2_hw: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Build binary matrix S ∈ {0,1}^{N×M} where S[i,j] = 1 iff
        k₁ᵢ and k₂ⱼ are mutual correspondences under H.

        Mutual correspondence criterion
        --------------------------------
        Forward:  ||H·k₁ᵢ  - k₂ⱼ|| ≤ threshold
        Backward: ||H⁻¹·k₂ⱼ - k₁ᵢ|| ≤ threshold   (optional, improves precision)

        Only the forward check is used by default for efficiency. Enable
        mutual check via ``mutual=True`` in __init__ if needed.

        Args
        ----
        kp1      : (N, 2)  keypoints in image 1  [x, y]
        kp2      : (M, 2)  keypoints in image 2  [x, y]
        H_mat    : (3, 3)  H: image1 → image2
        img2_hw  : (H, W)  image-2 dimensions for border masking (optional)

        Returns
        -------
        S : (N, M) float tensor of binary correspondence labels
        """
        N = kp1.shape[0]
        M = kp2.shape[0]

        # Warp all k₁ᵢ into image-2 space
        kp1_in_2 = self.warp_keypoints(kp1, H_mat)     # (N, 2)

        # Optional: exclude warped points near image-2 border
        if img2_hw is not None:
            H2, W2 = img2_hw
            r = self.safe_r
            valid_mask = (
                (kp1_in_2[:, 0] >= r) & (kp1_in_2[:, 0] <= W2 - r) &
                (kp1_in_2[:, 1] >= r) & (kp1_in_2[:, 1] <= H2 - r)
            )  # (N,) bool
        else:
            valid_mask = torch.ones(N, dtype=torch.bool, device=kp1.device)

        # Pairwise L2 distances  ||H·k₁ᵢ - k₂ⱼ||
        # (N, 1, 2) - (1, M, 2)  →  (N, M)
        kp1_in_2_exp = kp1_in_2.unsqueeze(1)           # (N, 1, 2)
        kp2_exp      = kp2.unsqueeze(0)                # (1, M, 2)
        dist = torch.norm(kp1_in_2_exp - kp2_exp, dim=-1)  # (N, M)

        # Binary correspondence labels
        S = (dist <= self.threshold).float()            # (N, M)

        # Zero-out rows where the warped point is outside image-2
        S = S * valid_mask.float().unsqueeze(1)        # (N, M)

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
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the hinge loss for one image pair.

        Args
        ----
        descriptors1 : (N, 256)  L2-normalised descs from image 1
        descriptors2 : (M, 256)  L2-normalised descs from image 2
        keypoints1   : (N, 2)    pixel-space keypoints in image 1
        keypoints2   : (M, 2)    pixel-space keypoints in image 2
        homography   : (3, 3)    H: image1 → image2
        image2_hw    : (H, W)    image-2 size for border masking (optional)

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

        # ── Step 1: Correspondence matrix ────────────────────────────────
        # S[i,j] = 1  iff k₁ᵢ ↔ k₂ⱼ under homography H
        S = self._build_correspondence_matrix(
            keypoints1, keypoints2, homography, image2_hw
        )  # (N, M)  — no gradient needed (labels are fixed)
        S = S.detach()

        # ── Step 2: Pairwise cosine similarities ─────────────────────────
        # Since both descriptor sets are L2-normalised:
        #   d₁ᵢᵀd₂ⱼ  =  cosine_similarity(d₁ᵢ, d₂ⱼ)  ∈ [-1, 1]
        sim = descriptors1 @ descriptors2.T             # (N, M)

        # ── Step 3: Positive hinge (pull matching pairs toward sim = mp) ─
        # l_pos[i,j] = λd · s_{ij} · max(0, mp - d₁ᵢ·d₂ⱼ)
        pos_loss = self.lambda_d * S * F.relu(self.mp - sim)   # (N, M)

        # ── Step 4: Negative hinge (push non-matching below mn) ──────────
        # l_neg[i,j] = (1 - s_{ij}) · max(0, d₁ᵢ·d₂ⱼ - mn)
        neg_loss = (1.0 - S) * F.relu(sim - self.mn)          # (N, M)

        # ── Step 5: Total loss (normalised over all N·M pairs) ───────────
        total_pairs = float(N * M)
        loss = (pos_loss + neg_loss).sum() / total_pairs

        # ── Diagnostics (detached from graph) ────────────────────────────
        with torch.no_grad():
            n_pos  = S.sum().item()
            n_neg  = (N * M) - n_pos

            eps = 1e-8
            pos_sim_mean = (sim * S).sum().item()         / (n_pos + eps)
            neg_sim_mean = (sim * (1.0 - S)).sum().item() / (n_neg + eps)

            stats: Dict[str, float] = {
                'loss':           loss.item(),
                'pos_loss_mean':  pos_loss.sum().item() / total_pairs,
                'neg_loss_mean':  neg_loss.sum().item() / total_pairs,
                'n_pos':          n_pos,
                'n_neg':          n_neg,
                'pos_sim_mean':   pos_sim_mean,
                'neg_sim_mean':   neg_sim_mean,
                'pos_ratio':      n_pos / total_pairs,
            }

        return loss, stats

    # ------------------------------------------------------------------
    # Batch wrapper
    # ------------------------------------------------------------------

    def forward_batch(
        self,
        desc1_list:  List[torch.Tensor],
        desc2_list:  List[torch.Tensor],
        kp1_list:    List[torch.Tensor],
        kp2_list:    List[torch.Tensor],
        homographies: torch.Tensor,
        image2_hws:  Optional[List[Tuple[int, int]]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute mean hinge loss across a batch of image pairs.

        Args
        ----
        desc1_list   : list[Tensor(N_b, 256)]   batch of desc sets from img1
        desc2_list   : list[Tensor(M_b, 256)]   batch of desc sets from img2
        kp1_list     : list[Tensor(N_b, 2)]
        kp2_list     : list[Tensor(M_b, 2)]
        homographies : (B, 3, 3)
        image2_hws   : optional list[(H, W)] per-image

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
            img2_hw = image2_hws[b] if image2_hws is not None else None
            loss_b, stats_b = self.forward(
                desc1_list[b], desc2_list[b],
                kp1_list[b],   kp2_list[b],
                homographies[b],
                img2_hw,
            )
            total_loss = total_loss + loss_b
            for k, v in stats_b.items():
                total_stats[k] = total_stats.get(k, 0.0) + v

        mean_loss  = total_loss / B
        mean_stats = {k: v / B for k, v in total_stats.items()}
        mean_stats['loss'] = mean_loss.item()

        return mean_loss, mean_stats
