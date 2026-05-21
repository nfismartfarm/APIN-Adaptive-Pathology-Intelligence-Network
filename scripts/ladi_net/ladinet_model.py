"""
LADI-Net model components: backbone wrapper, ABMIL, fusion, SupCon projector,
manual fused-qkv LoRA (Decision 35), and fallback_flag computation (Decisions 23, 30).

Phase 1: only heads are trained (ABMIL + gated MLP + SupCon projector). The
backbone runs in torch.no_grad().
Phase 2: LoRA adapters on blocks 4-11 of DINOv2-Base-Registers are added.

The fusion MLP input is 1537-dim per Decision 18:
  768 (spatial ABMIL) + 768 (global CLS) + 1 (fallback_flag) = 1537
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ladinet_config import (
    BACKBONE, EMBED_DIM, PATCH_SIZE, RESOLUTION, NUM_PATCHES, PREFIX_TOKENS,
    ABMIL_HIDDEN, FUSION_HIDDEN, FUSION_MID, FUSION_INPUT_DIM, NUM_CLASSES,
    SUPCON_PROJ_HIDDEN, SUPCON_PROJ_DIM,
    LORA_RANK, LORA_ALPHA, LORA_SCALE, LORA_DROPOUT, LORA_TARGET_BLOCKS,
    FALLBACK_MAX_ATTN_THRESHOLD, FALLBACK_ENTROPY_THRESHOLD,
)


# ===========================================================================
# Manual fused-qkv LoRA (Decision 35)
# ===========================================================================
class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a trainable low-rank delta.

    y = base(x) + scale * (B @ A @ x) where A: (rank, in), B: (out, rank).

    Initial lora_B = 0 so the LoRA-attached layer returns exactly base(x) at init.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: int, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.lora_A = nn.Parameter(torch.zeros(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B stays zero
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self._merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        if self._merged:
            return base_out
        # [Decision 49] Pass-1 LoRA scale injection: read module-level _current_lora_scale.
        # Default 1.0 = normal operation (Pass 2 and inference).
        # 0.0 = bypass (Pass 1 epochs 0-6; fast path skips LoRA compute entirely).
        # 0.0-1.0 = gradual injection (Pass 1 ramp epochs 7-9).
        lora_scale = getattr(self, "_current_lora_scale", 1.0)
        if lora_scale == 0.0:
            return base_out
        dx = self.dropout(x)
        delta = F.linear(F.linear(dx, self.lora_A), self.lora_B) * self.scale
        return base_out + lora_scale * delta

    @torch.no_grad()
    def merge(self):
        """Bake the LoRA delta into base.weight; subsequent forward() is pure base."""
        if self._merged:
            return
        delta = (self.lora_B @ self.lora_A) * self.scale     # (out, in)
        self.base.weight.data += delta
        self.lora_A.data.zero_()
        self.lora_B.data.zero_()
        self._merged = True


def attach_lora_to_backbone(backbone: nn.Module, target_blocks=None,
                            rank=LORA_RANK, alpha=LORA_ALPHA,
                            dropout=LORA_DROPOUT) -> int:
    """Wraps the `qkv` Linear in each target block with a LoRALinear.

    Returns number of attached adapters (== len(target_blocks)).
    """
    if target_blocks is None:
        target_blocks = LORA_TARGET_BLOCKS
    blocks = list(backbone.blocks)
    n_attached = 0
    for i in target_blocks:
        blk = blocks[i]
        attn = getattr(blk, "attn", None)
        if attn is None or not isinstance(getattr(attn, "qkv", None), nn.Linear):
            raise RuntimeError(f"Block {i} has no attn.qkv Linear; cannot attach LoRA")
        attn.qkv = LoRALinear(attn.qkv, rank=rank, alpha=alpha, dropout=dropout)
        n_attached += 1
    return n_attached


def count_lora_params(backbone: nn.Module) -> int:
    total = 0
    for m in backbone.modules():
        if isinstance(m, LoRALinear):
            total += m.lora_A.numel() + m.lora_B.numel()
    return total


def merge_lora_weights(backbone: nn.Module):
    """Bake LoRA into base weights. Call after training, before inference-time
    latency benchmark and before saving the deployable checkpoint."""
    for m in backbone.modules():
        if isinstance(m, LoRALinear):
            m.merge()


# ===========================================================================
# ABMIL gated-attention pool (Ilse et al. 2018 — Decision 17 §17.2)
# ===========================================================================
class ABMIL(nn.Module):
    """Attention-based MIL pool over patch tokens.

    Input:  patch_tokens [B, N, D], N=784, D=768
    Output: bag_feat [B, D] = softmax-weighted sum of patches,
            attn_weights [B, N] for downstream fallback_flag computation.
    """

    def __init__(self, in_dim: int = EMBED_DIM, hidden: int = ABMIL_HIDDEN):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, patch_tokens: torch.Tensor):
        # [B, N, D] -> [B, N, hidden]
        h = torch.tanh(self.fc1(patch_tokens))
        logits = self.fc2(h).squeeze(-1)       # [B, N]
        attn = F.softmax(logits, dim=1)        # [B, N]
        bag = (patch_tokens * attn.unsqueeze(-1)).sum(dim=1)  # [B, D]
        return bag, attn


# ===========================================================================
# Gated MLP fusion (Decision 17 §17.2)
# ===========================================================================
class GatedMLPFusion(nn.Module):
    """Fuses (spatial_bag, global_cls, fallback_flag) via a gated MLP.

    Input: concat of [B, 768] + [B, 768] + [B, 1] = [B, 1537]
    Output: class logits [B, 6]

    Architecture: Linear(1537, 512) → (SiLU × Sigmoid gate) → Linear(512, 256)
                  → GELU → Linear(256, 6).
    """

    def __init__(self, in_dim: int = FUSION_INPUT_DIM,
                 hidden: int = FUSION_HIDDEN, mid: int = FUSION_MID,
                 num_classes: int = NUM_CLASSES):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        self.gate = nn.Linear(in_dim, hidden)
        self.l2 = nn.Linear(hidden, mid)
        self.out = nn.Linear(mid, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.proj(x)) * torch.sigmoid(self.gate(x))
        h = F.gelu(self.l2(h))
        return self.out(h)


# ===========================================================================
# SupCon projection head (Decision 17 §17.2, Issue 6-C)
# ===========================================================================
class SupConProjector(nn.Module):
    """768 → 256 → 128 → L2-norm. Used only at training; discarded at inference."""

    def __init__(self, in_dim: int = EMBED_DIM,
                 hidden: int = SUPCON_PROJ_HIDDEN, out: int = SUPCON_PROJ_DIM):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out)

    def forward(self, bag: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.fc1(bag))
        z = self.fc2(h)
        return F.normalize(z, dim=-1)


# ===========================================================================
# Fallback flag (Decision 23)
# ===========================================================================
def compute_fallback_flag(attn_weights: torch.Tensor) -> torch.Tensor:
    """Computes the 0/1 fallback flag per sample.

    Fires if EITHER max_attn < 0.15 OR attention_entropy > 0.90 × log(N).

    attn_weights: [B, N] softmax-normalised ABMIL attention weights.
    Returns: [B, 1] float tensor with values in {0.0, 1.0}.
    """
    max_attn = attn_weights.max(dim=1).values          # [B]
    # Shannon entropy
    eps = 1e-12
    entropy = -(attn_weights * (attn_weights + eps).log()).sum(dim=1)   # [B]
    fires = (max_attn < FALLBACK_MAX_ATTN_THRESHOLD) | (entropy > FALLBACK_ENTROPY_THRESHOLD)
    return fires.float().unsqueeze(-1)                 # [B, 1]


# ===========================================================================
# Backbone loading (DINOv2-Base-Registers, 392px, frozen)
# ===========================================================================
def load_backbone(device: torch.device, phase: str = "phase1") -> nn.Module:
    """Loads DINOv2-Base-Registers with the right config for the given phase.

    Phase 1: backbone FROZEN completely. No LoRA attached.
    Phase 2: backbone frozen except LoRA on top 8 blocks.
    """
    import timm
    model = timm.create_model(
        BACKBONE,
        pretrained=True, num_classes=0,
        img_size=RESOLUTION, dynamic_img_size=True,
    ).to(device)

    # Verify register count
    if getattr(model, "num_reg_tokens", 0) != 4:
        raise RuntimeError(
            f"Expected num_reg_tokens=4, got {getattr(model, 'num_reg_tokens', 'missing')}"
        )

    # Freeze everything to start
    for p in model.parameters():
        p.requires_grad = False

    if phase == "phase2":
        n_adapters = attach_lora_to_backbone(model)
        # LoRA params already require_grad=True by default
        print(f"  Attached {n_adapters} LoRA adapters (total params: {count_lora_params(model)})")
    elif phase == "phase1":
        # No LoRA in Phase 1
        pass
    else:
        raise ValueError(f"Unknown phase: {phase!r}")

    return model


# ===========================================================================
# Full LADI-Net model (Phase 1 = backbone frozen + heads; Phase 2 = adds LoRA)
# ===========================================================================
class LADINet(nn.Module):
    """Assembles backbone + ABMIL + gated MLP fusion + SupCon projector.

    Forward returns: logits [B, C], bag_feat [B, D], supcon_proj [B, 128],
                     fallback_flag [B, 1], global_cls [B, D].
    """

    def __init__(self, device: torch.device, phase: str = "phase1"):
        super().__init__()
        self.phase = phase
        self.backbone = load_backbone(device, phase=phase)
        self.abmil = ABMIL().to(device)
        self.fusion = GatedMLPFusion().to(device)
        self.supcon = SupConProjector().to(device)

    def trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def head_params(self):
        params = []
        params.extend(self.abmil.parameters())
        params.extend(self.fusion.parameters())
        params.extend(self.supcon.parameters())
        return params

    def lora_params(self):
        return [p for p in self.backbone.parameters() if p.requires_grad]

    def forward(self, x: torch.Tensor):
        # Backbone forward: frozen in Phase 1 (no grad); Phase 2 gradient flows through LoRA only.
        if self.phase == "phase1":
            with torch.no_grad():
                feat = self.backbone.forward_features(x)     # [B, 789, 768]
        else:
            feat = self.backbone.forward_features(x)

        cls = feat[:, 0]                                     # [B, 768]
        patches = feat[:, PREFIX_TOKENS:]                    # [B, 784, 768]

        bag, attn = self.abmil(patches)                      # [B, 768], [B, 784]
        flag = compute_fallback_flag(attn)                   # [B, 1]

        fusion_in = torch.cat([bag, cls, flag], dim=-1)      # [B, 1537]
        logits = self.fusion(fusion_in)                      # [B, 6]

        proj = self.supcon(bag)                              # [B, 128]

        return {
            "logits": logits,
            "bag_feat": bag,
            "supcon_proj": proj,
            "fallback_flag": flag,
            "global_cls": cls,
            "attn_weights": attn,
        }


if __name__ == "__main__":
    # Smoke test: build Phase 1 model and run one forward pass.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = LADINet(device, phase="phase1").to(device)
    n_train = sum(p.numel() for p in model.trainable_params())
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_train:,} / total {n_total:,}")

    with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                            dtype=torch.bfloat16):
        x = torch.randn(2, 3, RESOLUTION, RESOLUTION, device=device)
        out = model(x)

    for k, v in out.items():
        if torch.is_tensor(v):
            print(f"  {k:14s}: shape {tuple(v.shape)}  dtype {v.dtype}")
    print(f"  fallback_flag fires: {out['fallback_flag'].sum().item():.0f}/2")
