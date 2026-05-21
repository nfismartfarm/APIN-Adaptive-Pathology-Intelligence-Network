"""Squeeze-and-Excitation block for 1D pooled features.

Spec reference: Part 4.4.
- 768 -> 48 -> 768 (reduction=16), two Linear layers (no bias), sigmoid gating.
- fc2 initialized with weight=0.01 so the block starts near identity; this
  speeds convergence on top of already-strong DINOv2 features.
- get_weight_stats() is used by the trainer every 5 epochs to verify that
  SE is learning non-trivial recalibration (std>=0.05).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, channels: int = 768, reduction: int = 16):
        super().__init__()
        assert channels % reduction == 0, (
            f"channels {channels} must be divisible by reduction {reduction}")
        bottleneck = channels // reduction
        self.fc1 = nn.Linear(channels, bottleneck, bias=False)
        self.fc2 = nn.Linear(bottleneck, channels, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        # Kaiming on fc1 (ReLU nonlinearity), near-identity on fc2.
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.fc2.weight, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C=channels]
        scale = self.sigmoid(self.fc2(self.relu(self.fc1(x))))
        return x * scale

    @torch.no_grad()
    def get_weight_stats(self, x: torch.Tensor) -> dict:
        """Return per-feature gate statistics for a sample batch.
        std<0.05 after 5 epochs = SE not learning useful recalibration."""
        w = self.sigmoid(self.fc2(self.relu(self.fc1(x))))
        return {
            'mean': float(w.mean()),
            'std':  float(w.std()),
            'min':  float(w.min()),
            'max':  float(w.max()),
        }
