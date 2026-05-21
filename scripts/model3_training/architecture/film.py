"""Hard FiLM conditioning with 3 discrete modes.

Spec reference: Part 4.6.
- Mode 0 = tomato, Mode 1 = chilli, Mode 2 = uncertain.
- Replaces soft-router FiLM (which interpolates continuously between
  conditioning vectors at inference — the model has never been trained on
  the 0.72/0.28 interpolated points).
- Mode 2 is exact identity at init, kept near identity during training via
  an auxiliary MSE loss (identity_regularization_loss, weight=0.01).

Training-time mode assignment schedule (implemented in Dataset):
  75% correct crop / 15% mode 2 / 10% wrong crop (only after epoch 5).

Inference-time (from APIN server, future): router conf > 0.80 -> correct mode,
0.65-0.80 -> mode 2, <0.65 -> mode 2.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HardFiLMConditioner(nn.Module):
    TOMATO = 0
    CHILLI = 1
    UNCERTAIN = 2

    def __init__(self, feature_dim: int = 768, n_modes: int = 3):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_modes = n_modes
        self.gamma = nn.Embedding(n_modes, feature_dim)
        self.beta  = nn.Embedding(n_modes, feature_dim)
        # Near-identity init for all modes: gamma ~ 1, beta = 0.
        nn.init.normal_(self.gamma.weight, mean=1.0, std=0.01)
        nn.init.zeros_(self.beta.weight)
        # Mode 2 (uncertain): exact identity — don't let init noise move it.
        with torch.no_grad():
            self.gamma.weight[self.UNCERTAIN].fill_(1.0)
            self.beta.weight[self.UNCERTAIN].zero_()

    def forward(self, x: torch.Tensor, crop_mode: torch.Tensor) -> torch.Tensor:
        """
        x: [B, feature_dim]
        crop_mode: [B] int64 in {0, 1, 2}
        """
        gamma = self.gamma(crop_mode)  # [B, feature_dim]
        beta  = self.beta(crop_mode)   # [B, feature_dim]
        return gamma * x + beta

    def identity_regularization_loss(self) -> torch.Tensor:
        """Pull mode 2's gamma toward 1 and beta toward 0.
        Total-loss weight for this term is set in config (0.01).

        Monitoring: self.gamma[UNCERTAIN].norm() should stay near sqrt(D)
        (i.e. all-ones vector), and self.beta[UNCERTAIN].norm() near 0.
        If either drifts beyond 0.5 away from identity, raise the weight."""
        g2 = self.gamma.weight[self.UNCERTAIN]
        b2 = self.beta.weight[self.UNCERTAIN]
        return F.mse_loss(g2, torch.ones_like(g2)) + F.mse_loss(b2, torch.zeros_like(b2))

    @torch.no_grad()
    def identity_drift(self) -> dict:
        """Return ||gamma[2] - 1||_2 and ||beta[2]||_2 for monitoring."""
        g2 = self.gamma.weight[self.UNCERTAIN]
        b2 = self.beta.weight[self.UNCERTAIN]
        return {
            'gamma_drift': float((g2 - 1.0).norm()),
            'beta_drift':  float(b2.norm()),
        }
