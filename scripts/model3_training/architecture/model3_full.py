"""Model 3 full architecture assembly.

Spec reference: Part 4.1.
Forward diagram:
    x [B, 3, 224, 224]
      -> DINOv2-Reg (frozen + LoRA rank=4 on qkv)
      -> forward_features -> [B, 261, 384]
      -> CLS[B,384] ++ mean(non-CLS)[B,384] -> [B, 768]
    branch A (SupCon):
      -> SE(768) -> SupConProjectionMLP(768->128->64, L2-norm)
    branch B (CE):
      -> same SE output (shared) -> MixStyle(p=0.5, post-pool, eval-off)
      -> HardFiLM(3 modes) -> LayerNorm(768) -> Linear(768, 10) -> logits

Return: dict(logits, supcon_features, film_identity_reg).

Assertions at construct time:
 - forward_features shape check (first call)
 - LoRA trainable param count in [60k, 100k]
 - All non-LoRA backbone params are float32 AND requires_grad==False
"""
from __future__ import annotations

from typing import Optional

import timm
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from ..model3_config import (
    BACKBONE, PROBE_IMG_SIZE, DINOV2_EMBED_DIM, FEAT_DIM, NUM_CLASSES,
    LORA_RANK, LORA_ALPHA, LORA_TARGET_MODULES, LORA_DROPOUT,
    LORA_EXPECTED_PARAMS_MIN, LORA_EXPECTED_PARAMS_MAX,
    SE_REDUCTION, MIXSTYLE_P, MIXSTYLE_ALPHA, MIXSTYLE_EPS, N_FILM_MODES,
)
from .se_block import SEBlock
from .mixstyle import MixStyle
from .film import HardFiLMConditioner
from .supcon_head import SupConProjectionMLP


class Model3(nn.Module):
    """
    DINOv2-Small-Registers (frozen + LoRA) + SE + MixStyle + HardFiLM + Linear.

    Args:
        n_classes: number of output classes (10 for tomato+chilli).
        pretrained: whether to load DINOv2 pretrained weights.
        use_lora: True = attach LoRA rank-4 adapters; False = purely frozen.
        lora_rank: override default LoRA rank (default=config.LORA_RANK=4).
    """
    def __init__(self,
                 n_classes: int = NUM_CLASSES,
                 pretrained: bool = True,
                 use_lora: bool = True,
                 lora_rank: int = LORA_RANK):
        super().__init__()

        # 1) Backbone — img_size=224 is CRITICAL (default=518 crashes on 224 input).
        backbone = timm.create_model(
            BACKBONE,
            pretrained=pretrained,
            num_classes=0,            # strip head; we do pooling ourselves
            img_size=PROBE_IMG_SIZE,  # MUST be explicit
        )

        # Freeze everything in the backbone first.
        for p in backbone.parameters():
            p.requires_grad = False

        # 2) LoRA (optional — decided by probe).
        self.use_lora = use_lora
        if use_lora:
            cfg = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_rank * 2,            # alpha = 2 * r convention
                target_modules=LORA_TARGET_MODULES,  # ['qkv'] for timm DINOv2 fused projection
                lora_dropout=LORA_DROPOUT,
                bias='none',
            )
            backbone = get_peft_model(backbone, cfg)

            # Assert LoRA actually attached — ['query','value'] silently finds ZERO.
            n_lora = sum(p.numel() for n, p in backbone.named_parameters()
                         if p.requires_grad and 'lora' in n.lower())
            if n_lora == 0:
                qkv_candidates = [
                    n for n, _ in backbone.named_modules()
                    if ('qkv' in n.lower() or 'attn' in n.lower())
                ]
                raise RuntimeError(
                    f"LoRA attached ZERO params. target_modules={LORA_TARGET_MODULES} "
                    f"did not match any module. Candidates containing 'qkv'/'attn':\n"
                    + "\n".join(qkv_candidates[:30])
                )
            if not (LORA_EXPECTED_PARAMS_MIN <= n_lora <= LORA_EXPECTED_PARAMS_MAX):
                raise RuntimeError(
                    f"LoRA params={n_lora:,} outside expected range "
                    f"[{LORA_EXPECTED_PARAMS_MIN:,}, {LORA_EXPECTED_PARAMS_MAX:,}] "
                    f"for rank={lora_rank}."
                )

        self.backbone = backbone

        # 3) Heads (all trainable).
        self.se = SEBlock(channels=FEAT_DIM, reduction=SE_REDUCTION)
        self.mixstyle = MixStyle(p=MIXSTYLE_P, alpha=MIXSTYLE_ALPHA, eps=MIXSTYLE_EPS)
        self.supcon_proj = SupConProjectionMLP()
        self.film = HardFiLMConditioner(feature_dim=FEAT_DIM, n_modes=N_FILM_MODES)
        self.classifier = nn.Sequential(
            nn.LayerNorm(FEAT_DIM),
            nn.Linear(FEAT_DIM, n_classes),
        )
        nn.init.xavier_uniform_(self.classifier[1].weight)
        nn.init.zeros_(self.classifier[1].bias)

        # PDA Round 2 finding C5: remove _shape_verified latch.
        # Shape check is ~zero-cost; check on every forward to catch any
        # downstream issue (DataParallel / torch.compile / checkpoint-load
        # device-mismatch) loudly and immediately.

    # ────────────────────────────────────────────────────────────────────
    # Feature extraction: CLS + mean(non-CLS) -> 768-d
    # (Includes 4 register tokens in the mean — matches Signal 4 convention,
    #  Decision 13 in architecture_claude_decisions.md.)
    # ────────────────────────────────────────────────────────────────────
    def extract_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(x)  # [B, 261, 384]

        # Always check shape — PDA Round 2 finding C5.
        B = x.shape[0]
        expected = (B, 261, DINOV2_EMBED_DIM)
        actual = tuple(features.shape)
        if actual != expected:
            raise RuntimeError(
                f"Backbone shape mismatch. Expected {expected}, got {actual}. "
                f"Check img_size=224 was passed to timm.create_model."
            )

        cls_token = features[:, 0, :]          # [B, 384]
        non_cls = features[:, 1:, :]           # [B, 260, 384] — registers + patches
        mean_token = non_cls.mean(dim=1)       # [B, 384]
        return torch.cat([cls_token, mean_token], dim=-1)  # [B, 768]

    # ────────────────────────────────────────────────────────────────────
    # Forward
    # ────────────────────────────────────────────────────────────────────
    def forward(self,
                x: torch.Tensor,
                crop_mode: torch.Tensor,
                domain_labels: Optional[torch.Tensor] = None) -> dict:
        """
        x: [B, 3, 224, 224]
        crop_mode: [B] int64 in {0, 1, 2}
        domain_labels: optional [B] int64 in {0, 1} for cross-domain MixStyle mixing

        Returns dict with:
            logits:           [B, n_classes]
            supcon_features:  [B, 64] L2-normalized
        """
        # Shared feature extraction
        raw = self.extract_backbone_features(x)    # [B, 768]

        # SE first, then SPLIT:
        #   SupCon branch:  SE output (stable, no MixStyle randomization)
        #   CE branch:      SE output -> MixStyle -> FiLM -> classifier
        se_out = self.se(raw)                      # [B, 768]

        supcon_z = self.supcon_proj(se_out)        # [B, 64]

        mixed = self.mixstyle(se_out, domain_labels)  # [B, 768] (identity if eval or skipped)
        conditioned = self.film(mixed, crop_mode)  # [B, 768]
        logits = self.classifier(conditioned)      # [B, n_classes]

        return {'logits': logits, 'supcon_features': supcon_z}

    # ────────────────────────────────────────────────────────────────────
    # Diagnostics
    # ────────────────────────────────────────────────────────────────────
    def count_trainable(self) -> dict:
        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        lora = sum(p.numel() for n, p in self.named_parameters()
                   if p.requires_grad and 'lora' in n.lower())
        heads = total_trainable - lora
        total = sum(p.numel() for p in self.parameters())
        return {
            'total_params': total,
            'total_trainable': total_trainable,
            'lora_trainable': lora,
            'head_trainable': heads,
            'frozen': total - total_trainable,
        }

    def assert_backbone_dtype_and_freeze(self) -> None:
        """Verify at startup that non-LoRA backbone params are float32 AND frozen."""
        for n, p in self.named_parameters():
            if 'backbone' not in n:
                continue
            if 'lora' in n.lower():
                continue
            if p.dtype != torch.float32:
                raise RuntimeError(
                    f"Non-LoRA backbone param {n} is {p.dtype}, expected float32. "
                    f"BF16 permanent cast bug — would destroy gradients."
                )
            if p.requires_grad:
                raise RuntimeError(
                    f"Non-LoRA backbone param {n} has requires_grad=True — "
                    f"should be frozen."
                )
