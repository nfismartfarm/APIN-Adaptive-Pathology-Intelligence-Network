"""
Loss functions for LADI-Net.

- supcon_loss: Khosla et al. 2020 supervised contrastive loss on L2-normed projections.
- weighted_ce_loss: cross-entropy with per-sample field-image weighting (Decision 17 §17.4).

CORAL loss is implemented in the Phase 2 training script since it depends on
global EMA state (Decision 17 §17.4, Decisions 26, 29, 31 §31.5).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ladinet_config import SUPCON_TAU, FIELD_LOSS_WEIGHT


def supcon_loss(proj: torch.Tensor, labels: torch.Tensor,
                tau: float = SUPCON_TAU) -> torch.Tensor:
    """Supervised contrastive loss on L2-normalised projections.

    proj:   [B, D] already L2-normed per-sample
    labels: [B]    int64 class labels
    tau:    temperature scalar
    """
    B = proj.size(0)
    if B < 2:
        return proj.new_zeros(())

    sim = (proj @ proj.t()) / tau                     # [B, B]
    # Numerical stability: subtract per-row max (detached)
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()

    self_mask = torch.eye(B, dtype=torch.bool, device=proj.device)
    labels = labels.view(-1, 1)
    pos_mask = (labels == labels.t()) & (~self_mask)  # [B, B]

    exp_sim = torch.exp(sim) * (~self_mask)           # exclude self
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    pos_count = pos_mask.sum(dim=1).clamp(min=1)
    per_sample = -(log_prob * pos_mask).sum(dim=1) / pos_count

    # Only count anchors that have ≥1 positive
    valid = pos_mask.any(dim=1)
    if not valid.any():
        return proj.new_zeros(())
    return per_sample[valid].mean()


def weighted_ce_loss(logits: torch.Tensor, labels: torch.Tensor,
                     is_field_photo: torch.Tensor,
                     field_weight: float = FIELD_LOSS_WEIGHT) -> torch.Tensor:
    """CE with per-sample 8× weight for field images (Decision 17 §17.4).

    logits:         [B, C]
    labels:         [B]  int64
    is_field_photo: [B]  bool or 0/1 float
    """
    per_sample_ce = F.cross_entropy(logits, labels, reduction="none")     # [B]
    weights = torch.where(
        is_field_photo.bool(),
        torch.full_like(per_sample_ce, field_weight),
        torch.ones_like(per_sample_ce),
    )
    # Weighted mean — divide by the weight sum to keep loss scale comparable across batches
    return (per_sample_ce * weights).sum() / weights.sum().clamp(min=1.0)
