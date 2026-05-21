"""MixStyle — post-pool 1D variant.

Spec reference: Part 4.5. Design decision rationale:
- Canonical MixStyle (Zhou 2021) is for CNN spatial feature maps.
- TFS-ViT (Noori 2023) adapts it at the token level for frozen ViTs.
- OUR backbone is NOT fully frozen (LoRA adapts qkv). Inserting MixStyle at
  the token level would require hooks inside LoRA's forward, which is brittle.
- Therefore we operate on the POST-POOL 1D vector (768-d), mixing per-instance
  mean/std across the feature dimension. This is closer to "feature-level
  Mixup" than true MixStyle but is the right call for this architecture.

Hard disabled at inference via `if not self.training: return x`.
Cross-domain permutation prefers field <-> lab partner pairs when domain_labels
are provided (so the synthesized styles span the actual deployment distribution).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MixStyle(nn.Module):
    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps
        self._beta = torch.distributions.Beta(alpha, alpha)

    def forward(self, x: torch.Tensor, domain_labels: torch.Tensor = None) -> torch.Tensor:
        """
        x: [B, C]  — post-pool 1D features (C=768 in our architecture).
        domain_labels: optional [B] tensor; 1=field, 0=lab/recomposed.

        Returns x unchanged when:
          - model.eval() is active (self.training == False)
          - a Bernoulli(p) sample comes up as "skip"
        """
        if not self.training:
            return x
        if torch.rand(1, device=x.device).item() > self.p:
            return x

        B = x.size(0)
        lam = self._beta.sample((B, 1)).to(x.device)  # [B, 1]

        if domain_labels is not None:
            perm = self._cross_domain_perm(domain_labels, B, x.device)
        else:
            perm = torch.randperm(B, device=x.device)

        # Statistics over the feature dimension (C).
        mu    = x.mean(dim=1, keepdim=True)                    # [B, 1]
        sigma = (x.var(dim=1, keepdim=True) + self.eps).sqrt() # [B, 1]
        mu2    = mu[perm]
        sigma2 = sigma[perm]

        mu_mix    = lam * mu + (1.0 - lam) * mu2
        sigma_mix = lam * sigma + (1.0 - lam) * sigma2

        # Re-normalize x to (0,1) per-instance, then apply mixed stats.
        x_normed = (x - mu) / sigma
        return x_normed * sigma_mix + mu_mix

    @staticmethod
    def _cross_domain_perm(domain_labels: torch.Tensor,
                           B: int,
                           device: torch.device) -> torch.Tensor:
        """Prefer field <-> lab partner pairs."""
        perm = torch.randperm(B, device=device)
        field_idx = (domain_labels == 1).nonzero(as_tuple=True)[0]
        lab_idx = (domain_labels == 0).nonzero(as_tuple=True)[0]
        if len(field_idx) > 0 and len(lab_idx) > 0:
            # Pair up field[i] <-> lab[i%L]. Overwrites part of the random perm.
            for i, f_idx in enumerate(field_idx):
                l_idx = lab_idx[i % len(lab_idx)]
                perm[f_idx] = l_idx
                perm[l_idx] = f_idx
        return perm
