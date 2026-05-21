"""Supervised contrastive projection MLP + weighted loss.

Spec reference: Part 4.7.
- ProjectionMLP: 768 -> 128 (GELU + LayerNorm) -> 64, L2-normalized output.
- Receives features from SE block BEFORE MixStyle (so class prototypes are stable).
- Temperature = 0.1 (NOT 0.07 per Reb-SupCon 2025).
- Per-class weights: septoria=0.50 (primary target), anthracnose=0.20 (downgraded).
- CutMix samples excluded via `use_supcon_mask` (SupCon can't handle soft labels).
- If batch has <2 pure same-class positives, loss is zero (safe skip).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..model3_config import (
    FEAT_DIM, SUPCON_PROJ_HIDDEN, SUPCON_PROJ_OUT,
    SUPCON_TEMPERATURE, SUPCON_CLASS_WEIGHTS, CLASS_NAMES,
)


class SupConProjectionMLP(nn.Module):
    """Maps 768-d SE features to 64-d unit hypersphere."""
    def __init__(self,
                 input_dim: int = FEAT_DIM,
                 hidden_dim: int = SUPCON_PROJ_HIDDEN,
                 output_dim: int = SUPCON_PROJ_OUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, dim=-1)


class WeightedSupConLoss(nn.Module):
    """
    Per-class-weighted supervised contrastive loss.

    features: [B, D] L2-normalized projections
    labels:   [B] integer class ids (in CLASS_NAMES order)
    use_supcon_mask: [B] bool — True for pure samples, False for CutMix
                     (CutMix carries soft labels, incompatible with SupCon)

    Returns a scalar. Zero tensor (still differentiable) when fewer than 2
    pure samples remain in the batch.
    """
    def __init__(self,
                 temperature: float = SUPCON_TEMPERATURE,
                 class_weights: Optional[dict] = None):
        super().__init__()
        self.temperature = temperature
        cw = class_weights if class_weights is not None else SUPCON_CLASS_WEIGHTS
        self.class_weight_tensor = torch.tensor(
            [cw.get(c, 0.25) for c in CLASS_NAMES], dtype=torch.float32
        )

    def forward(self,
                features: torch.Tensor,
                labels: torch.Tensor,
                use_supcon_mask: torch.Tensor) -> torch.Tensor:
        device = features.device
        mask = use_supcon_mask.to(device) if use_supcon_mask.device != device else use_supcon_mask
        if mask.sum().item() < 2:
            return features.new_zeros(())   # differentiable zero

        feat = features[mask]
        lbl  = labels[mask]
        B = feat.size(0)
        if B < 2:
            return features.new_zeros(())

        # Cosine similarity matrix over unit sphere features.
        sim = torch.matmul(feat, feat.T) / self.temperature   # [B, B]

        # Positive mask = same class, excluding self.
        lbl_eq = (lbl.unsqueeze(0) == lbl.unsqueeze(1))       # [B, B]
        self_mask = torch.eye(B, dtype=torch.bool, device=device)
        pos_mask = lbl_eq & ~self_mask                        # [B, B]

        # log_softmax denominator over all non-self entries.
        sim_no_self = sim.masked_fill(self_mask, float('-inf'))
        log_prob = sim - torch.logsumexp(sim_no_self, dim=1, keepdim=True)

        # Mean over positives per anchor; safe when no positive present.
        n_pos = pos_mask.sum(dim=1).clamp_min(1)
        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / n_pos
        per_sample_loss = -mean_log_prob_pos                  # [B]

        # Zero out rows with zero positives (undefined loss — don't backprop).
        valid = (pos_mask.sum(dim=1) > 0).float()
        per_sample_loss = per_sample_loss * valid

        # Apply per-class weights.
        cw = self.class_weight_tensor.to(device)
        w_per_sample = cw[lbl]                                # [B]
        numer = (w_per_sample * per_sample_loss).sum()
        denom = (w_per_sample * valid).sum().clamp_min(1e-8)
        return numer / denom
