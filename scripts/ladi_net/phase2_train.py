"""
Phase 2 training: LoRA fine-tuning on Pass 2 + CORAL + AmpMix + SupCon.

Corresponds to Decisions 15-44. Key elements:
- Backbone DINOv2-Base-Registers, 392x392 input
- LoRA rank=8 alpha=16 on top 8 blocks (blocks 4-11), manual fused-qkv impl (Decision 35)
- Two-pass forward: Pass 1 frozen (no_grad); Pass 2 = LoRA-active on lesion crop
- Heads warm-started from Phase 1 (ABMIL + gated MLP + SupCon projector)
- Batch [4,4,2,2,2,2] via ClassStratifiedBatchSampler; bs=16
- Loss: CE + 0.30*SupCon + alpha*CORAL with alpha=0.5; AdamW; bf16 autocast; grad_clip=1.0
- SupCon weight adaptive monitoring (Decision 42)
- CORAL EMA on lab source cov (lab_count>=6 guard, fallback_flag=0 only -- Decisions 26, 29)
- CORAL loss warmup over 2000 updates (Decision 31 sec 31.5)
- CORAL target refresh every 5 epochs from post-LoRA ABMIL features (Decision 14 / Critique 4)
- AmpMix at p=0.45 on foliar+septoria pairs pre-preprocessing (Decision 38)
- Attention gate re-run every 5 epochs (Decision 44); fatal at <8/19 after epoch 10
- Stopping: 3-epoch rolling mean field-val sqrtn_macro_f1 with disease F1 floor (Decision 17 sec 17.6)
- Patience counter starts at epoch 4 (Decision 30.5); min 12 epochs, max 25, patience 5
- Checkpoint format per Decision 31 sec 31.1 / Decision 32.2 with provenance + RNG states

CRITICAL CORRECTNESS NOTES:
- Pass 1 MUST use torch.no_grad() and use the backbone with LoRA delta zeroed / bypassed
  (see _run_pass1). Otherwise Pass 1 benefits from LoRA adaptation, which contradicts spec.
- cls_global MUST be .detach()ed before entering fusion MLP -- otherwise gradients try to flow
  through the frozen Pass 1 backbone.
- Register-token indexing: attention[:, 0, 5:].reshape(B, 28, 28)
- CORAL target file MUST be coral_target_cov.pt with source='abmil_features_phase1'.

Usage:
    python scripts/ladi_net/phase2_train.py --dry_run --n_batches 3      # smoke test
    python scripts/ladi_net/phase2_train.py                              # full run
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import random
import sys
import time
from collections import deque
from pathlib import Path

# Reproducibility: set BEFORE importing torch/numpy
os.environ["PYTHONHASHSEED"] = "0"

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from ladinet_config import (
    PROJECT_ROOT, MODEL3_DIR, PHASE1_HEADS_PT, PHASE1_CKPT_DIR, LOGS_DIR,
    TOMATO_CLASSES, CLASS_TO_IDX, NUM_CLASSES,
    RESOLUTION, PREFIX_TOKENS, NUM_PATCHES,
    PHASE2_BATCH_SIZE, PHASE2_MIN_EPOCHS, PHASE2_MAX_EPOCHS,
    PHASE2_PATIENCE, PHASE2_ROLLING_MEAN_EPOCHS,
    LR_LORA, WD_LORA, LR_HEADS, WD_HEADS, GRAD_CLIP_NORM,
    WARMUP_EPOCHS, LR_COSINE_MIN_RATIO, PATIENCE_STARTS_EPOCH,
    LOSS_W_CE, LOSS_W_SUPCON_PHASE2, LOSS_W_CORAL_PHASE2, SUPCON_TAU,
    DISEASE_F1_FLOOR, STOPPING_WEIGHTS,
    CORAL_EMA_DECAY, CORAL_MIN_LAB_COUNT_IN_BATCH, CORAL_WARMUP_STEPS,
    CORAL_TARGET_REFRESH_EPOCHS,
    AMPMIX_PROB,
    FALLBACK_FLAG_COL_FREEZE_UNTIL_EPOCH, FALLBACK_FLAG_LR_MULT_EPOCH_4,
    FALLBACK_MAX_ATTN_THRESHOLD, FALLBACK_ENTROPY_THRESHOLD,
    SEED, NUM_WORKERS, CONFIG_HASH, CORAL_TARGET_PT,
)
from ladinet_model import (
    ABMIL, GatedMLPFusion, SupConProjector, LoRALinear,
    attach_lora_to_backbone, count_lora_params, merge_lora_weights,
    compute_fallback_flag,
)
from ladinet_losses import supcon_loss, weighted_ce_loss
from ladinet_dataloader import (
    load_split_records, LadiNetDataset, ClassStratifiedBatchSampler,
    load_background_pool,
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def set_seeds(seed: int = SEED) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT.resolve()))
    except Exception:
        return str(p)


# ---------------------------------------------------------------------------
# Full model for Phase 2 (backbone WITH LoRA adapters attached)
# ---------------------------------------------------------------------------
class LADINetPhase2(nn.Module):
    """Full LADI-Net with LoRA on Pass 2.

    Uses a single backbone instance. LoRA adapters are always active in the
    backbone's forward pass. For Pass 1 (frozen mode), we run the backbone
    inside torch.no_grad() so LoRA receives no gradient AND we want Pass 1 to
    use the BASE (non-LoRA) weights for spec compliance.

    To bypass LoRA during Pass 1, each LoRALinear exposes a ._bypass attribute.
    When True, forward() returns base(x) only. When False, returns base(x) + delta.
    """

    def __init__(self, device: torch.device):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            "vit_base_patch14_reg4_dinov2",
            pretrained=True, num_classes=0,
            img_size=RESOLUTION, dynamic_img_size=True,
        ).to(device)
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Attach LoRA adapters to top 8 blocks' qkv (Decision 35)
        n_attached = attach_lora_to_backbone(self.backbone)
        assert n_attached == 8, f"expected 8 LoRA adapters, got {n_attached}"
        # LoRA A/B parameters already have requires_grad=True by default

        # Heads (warm-start-capable)
        self.abmil = ABMIL().to(device)
        self.fusion = GatedMLPFusion().to(device)
        self.supcon = SupConProjector().to(device)

        # State toggles
        self._last_attn_weights = None    # captured from Pass 1 hook
        self._hook_handle = self._install_pass1_attn_hook()

    def _install_pass1_attn_hook(self):
        """Replace last-block attn.forward so we can capture softmax(QK^T/sqrt(d))."""
        last_attn = self.backbone.blocks[-1].attn

        def hooked_forward(x, attn_mask=None, is_causal=False):
            B, N, C = x.shape
            qkv = last_attn.qkv(x).reshape(
                B, N, 3, last_attn.num_heads, C // last_attn.num_heads
            ).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            try:
                q = last_attn.q_norm(q); k = last_attn.k_norm(k)
            except Exception:
                pass
            scale = last_attn.scale
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            self._last_attn_weights = attn.detach()  # [B, H, N, N]
            attn_d = getattr(last_attn, "attn_drop", nn.Identity())(attn)
            out = (attn_d @ v).transpose(1, 2).reshape(B, N, C)
            out = last_attn.proj(out)
            try:
                out = last_attn.proj_drop(out)
            except Exception:
                pass
            return out

        last_attn.forward = hooked_forward
        return None

    def _set_lora_bypass(self, bypass: bool) -> None:
        """Legacy helper — kept for compatibility. Prefer _set_lora_scale()."""
        scale = 0.0 if bypass else 1.0
        self._set_lora_scale(scale)

    def _set_lora_scale(self, scale: float) -> None:
        """Set the LoRA delta scale for all LoRALinear modules in the backbone.

        Used to toggle Pass 1 vs Pass 2 LoRA application per Decision 49:
        - Pass 1 epochs 0-6: scale=0.0 (frozen pretrained DINOv2)
        - Pass 1 epochs 7-9: scale 0.33 -> 0.67 -> 1.0 (ramp)
        - Pass 1 epoch 10+: scale=1.0 (full Pass-1 LoRA)
        - Pass 2 always: scale=1.0
        """
        for m in self.backbone.modules():
            if isinstance(m, LoRALinear):
                m._current_lora_scale = scale
                # also keep legacy _bypass attribute in sync
                m._bypass = (scale == 0.0)

    def lora_params(self):
        return [p for m in self.backbone.modules() if isinstance(m, LoRALinear)
                for p in [m.lora_A, m.lora_B]]

    def head_params(self):
        return (list(self.abmil.parameters())
                + list(self.fusion.parameters())
                + list(self.supcon.parameters()))


# Patch LoRALinear to honour _bypass
_orig_lora_forward = LoRALinear.forward

def _lora_forward_with_bypass(self, x):
    if getattr(self, "_bypass", False):
        return self.base(x)
    return _orig_lora_forward(self, x)

LoRALinear.forward = _lora_forward_with_bypass


# ---------------------------------------------------------------------------
# Pass 1 + Pass 2 forward helpers
# ---------------------------------------------------------------------------
def _run_pass1(model: LADINetPhase2, x: torch.Tensor, pass1_scale: float = 0.0):
    """Pass 1 on the full image with Decision 49 deferred LoRA scale injection.

    pass1_scale is computed by compute_pass1_lora_scale(epoch) in the training loop:
    - 0.0 for epochs 0-6: LoRA bypassed, exact pretrained DINOv2 attention.
    - 0.33/0.67/1.0 for epochs 7/8/9: linear ramp, gradual LoRA delta injection.
    - 1.0 for epoch 10+: full LoRA delta applied to Pass 1.
    - 1.0 at inference (caller passes pass1_scale=1.0 — LoRA is merged).

    Pass 1 is under torch.no_grad() regardless of scale — LoRA params only receive
    gradient through Pass 2. Scale-based ramp avoids epoch-0 chaos per Decision 49.
    """
    model._set_lora_scale(pass1_scale)
    with torch.no_grad():
        feat = model.backbone.forward_features(x)  # [B, 789, 768]
    # Restore Pass 2 scale immediately (keep model in a consistent state outside this call)
    model._set_lora_scale(1.0)

    cls_global = feat[:, 0]                                         # [B, 768]
    # ABMIL-independent attention analysis for fallback_flag (uses DINO attention, not ABMIL)
    attn = model._last_attn_weights.float()                         # [B, H, 789, 789]
    attn_mean_heads = attn.mean(dim=1)                              # [B, 789, 789]
    cls_to_spatial = attn_mean_heads[:, 0, PREFIX_TOKENS:]          # [B, 784]
    cls_to_spatial = cls_to_spatial / cls_to_spatial.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    cls_spatial_map = cls_to_spatial.reshape(-1, 28, 28)            # [B, 28, 28]

    # Fallback flag per Decision 23
    max_attn_per_img = cls_to_spatial.max(dim=-1).values             # [B]
    entropy = -(cls_to_spatial * (cls_to_spatial + 1e-12).log()).sum(dim=-1)
    fallback = (
        (max_attn_per_img < FALLBACK_MAX_ATTN_THRESHOLD)
        | (entropy > FALLBACK_ENTROPY_THRESHOLD)
    ).float()                                                        # [B]
    return cls_global, cls_spatial_map, fallback


def _select_lesion_crop(images: torch.Tensor,
                        cls_spatial_map: torch.Tensor,
                        fallback_flag: torch.Tensor) -> torch.Tensor:
    """For each image: if fallback_flag==1, return center 70% crop resized to 392;
    else compute bbox from top-20 attention patches, pad 15%, clamp, resize to 392.

    images: [B, 3, 392, 392] tensor (preprocessed + normalized).
    cls_spatial_map: [B, 28, 28] attention.
    fallback_flag: [B] 0/1 float.
    Returns: [B, 3, 392, 392] lesion crop tensor.
    """
    B = images.size(0)
    crops = torch.zeros_like(images)
    patch_to_pixel = RESOLUTION // 28  # 14 px per patch
    for i in range(B):
        if fallback_flag[i].item() > 0.5:
            # Center 70% crop
            cs = int(0.15 * RESOLUTION)
            ce = int(0.85 * RESOLUTION)
            crop = images[i:i+1, :, cs:ce, cs:ce]
        else:
            am = cls_spatial_map[i].flatten()
            top20_idx = torch.topk(am, 20).indices                   # [20]
            rows = top20_idx // 28
            cols = top20_idx % 28
            y1 = max(0, int(rows.min().item()) - 0)
            y2 = min(27, int(rows.max().item()))
            x1 = max(0, int(cols.min().item()) - 0)
            x2 = min(27, int(cols.max().item()))
            # Convert to pixels
            py1 = y1 * patch_to_pixel
            py2 = (y2 + 1) * patch_to_pixel
            px1 = x1 * patch_to_pixel
            px2 = (x2 + 1) * patch_to_pixel
            # 15% pad
            ph = int(0.15 * (py2 - py1))
            pw = int(0.15 * (px2 - px1))
            py1 = max(0, py1 - ph)
            py2 = min(RESOLUTION, py2 + ph)
            px1 = max(0, px1 - pw)
            px2 = min(RESOLUTION, px2 + pw)
            if py2 - py1 < 20 or px2 - px1 < 20:
                # Degenerate: fallback to center 70% crop
                cs = int(0.15 * RESOLUTION); ce = int(0.85 * RESOLUTION)
                crop = images[i:i+1, :, cs:ce, cs:ce]
            else:
                crop = images[i:i+1, :, py1:py2, px1:px2]
        # Resize back to 392x392
        crop_392 = F.interpolate(crop, size=(RESOLUTION, RESOLUTION),
                                 mode="bilinear", align_corners=False)
        crops[i:i+1] = crop_392
    return crops


def _run_pass2(model: LADINetPhase2, lesion_crops: torch.Tensor):
    """Pass 2: LoRA-active forward on lesion crops. Returns ABMIL bag feature."""
    model._set_lora_bypass(False)
    feat = model.backbone.forward_features(lesion_crops)             # [B, 789, 768]
    patches = feat[:, PREFIX_TOKENS:, :]                             # [B, 784, 768]
    bag, attn = model.abmil(patches)                                 # [B, 768], [B, 784]
    return bag


# ---------------------------------------------------------------------------
# AmpMix (Decision 38, p=0.45) — operates on pre-preprocessing pixel tensors
# ---------------------------------------------------------------------------
def apply_ampmix(images: torch.Tensor, labels: torch.Tensor,
                 p: float = AMPMIX_PROB, rng=None) -> tuple[torch.Tensor, int]:
    """AmpMix between foliar and septoria images in the batch.

    images: [B, 3, H, W] float tensor (NOT normalized). Operates on pixel intensities.
    labels: [B] int64 class indices.
    Returns (mixed_images, n_mixed_pairs).
    """
    if rng is None:
        rng = random.Random()
    if rng.random() >= p:
        return images, 0
    foliar_idx = CLASS_TO_IDX["tomato_foliar_spot"]
    septoria_idx = CLASS_TO_IDX["tomato_septoria_leaf_spot"]
    foliar_pos = (labels == foliar_idx).nonzero(as_tuple=True)[0].tolist()
    septoria_pos = (labels == septoria_idx).nonzero(as_tuple=True)[0].tolist()
    if not foliar_pos or not septoria_pos:
        return images, 0
    n_pairs = min(len(foliar_pos), len(septoria_pos))
    rng.shuffle(foliar_pos); rng.shuffle(septoria_pos)
    mixed = images.clone()
    for fi, si in zip(foliar_pos[:n_pairs], septoria_pos[:n_pairs]):
        a = images[fi].float(); b = images[si].float()
        # Per-channel FFT amplitude swap on low frequencies
        for c in range(a.shape[0]):
            Fa = torch.fft.fft2(a[c])
            Fb = torch.fft.fft2(b[c])
            Aa = Fa.abs(); Ab = Fb.abs()
            Pa = torch.angle(Fa); Pb = torch.angle(Fb)
            # Swap low-frequency amplitudes (central region in FFT shift space)
            H, W = Aa.shape
            Aa_s = torch.fft.fftshift(Aa); Ab_s = torch.fft.fftshift(Ab)
            ch, cw = H // 2, W // 2
            r = min(H, W) // 8  # 12.5% low-freq radius
            # Swap the (2r x 2r) central block
            tmp = Aa_s[ch-r:ch+r, cw-r:cw+r].clone()
            Aa_s[ch-r:ch+r, cw-r:cw+r] = Ab_s[ch-r:ch+r, cw-r:cw+r]
            Ab_s[ch-r:ch+r, cw-r:cw+r] = tmp
            Aa2 = torch.fft.ifftshift(Aa_s); Ab2 = torch.fft.ifftshift(Ab_s)
            a[c] = torch.fft.ifft2(Aa2 * torch.exp(1j * Pa)).real
            b[c] = torch.fft.ifft2(Ab2 * torch.exp(1j * Pb)).real
        mixed[fi] = a.to(images.dtype)
        mixed[si] = b.to(images.dtype)
    return mixed, n_pairs


# ---------------------------------------------------------------------------
# CORAL loss
# ---------------------------------------------------------------------------
def coral_frobenius_loss(source_cov: torch.Tensor, target_cov: torch.Tensor) -> torch.Tensor:
    d = source_cov.shape[0]
    return (source_cov - target_cov).pow(2).sum() / (4 * d * d)


# ---------------------------------------------------------------------------
# Validation loop
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_phase2(model: LADINetPhase2, val_records, device: torch.device,
                    pass1_scale: float = 0.0) -> dict:
    """Validation loop. `pass1_scale` must be set by caller to match the current
    training-epoch Pass-1 LoRA scale (Decision 49). At inference post-training,
    pass1_scale=1.0 (LoRA merged)."""
    model.eval()
    val_ds = LadiNetDataset(val_records, training=False, background_pool=None)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False,
                            num_workers=NUM_WORKERS)
    all_preds, all_labels = [], []
    fallback_fires_by_class = {c: 0 for c in TOMATO_CLASSES}
    fallback_total_by_class = {c: 0 for c in TOMATO_CLASSES}

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        for batch in val_loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)

            cls_global, cls_spatial, fallback = _run_pass1(model, x, pass1_scale=pass1_scale)
            lesion = _select_lesion_crop(x, cls_spatial, fallback)
            bag = _run_pass2(model, lesion)

            fusion_in = torch.cat([bag, cls_global.detach(),
                                    fallback.unsqueeze(-1).float()], dim=-1).float()
            logits = model.fusion(fusion_in)
            preds = logits.argmax(dim=-1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            for i, lab in enumerate(y.cpu().numpy()):
                cname = TOMATO_CLASSES[int(lab)]
                fallback_total_by_class[cname] += 1
                if fallback[i].item() > 0.5:
                    fallback_fires_by_class[cname] += 1

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    per_class_f1 = {
        c: float(f1_score((y_true == i).astype(int), (y_pred == i).astype(int),
                          zero_division=0))
        for i, c in enumerate(TOMATO_CLASSES)
    }
    simple_macro = float(np.mean(list(per_class_f1.values())))
    sqrtn_macro = sum(STOPPING_WEIGHTS[c] * per_class_f1[c] for c in TOMATO_CLASSES)
    fallback_rates = {
        c: fallback_fires_by_class[c] / fallback_total_by_class[c]
        if fallback_total_by_class[c] > 0 else 0.0
        for c in TOMATO_CLASSES
    }
    return {
        "per_class_f1": per_class_f1,
        "sqrtn_macro_f1": float(sqrtn_macro),
        "simple_macro_f1": simple_macro,
        "fallback_fire_rate_by_class": fallback_rates,
        "n_val": int(y_true.size),
    }


# ---------------------------------------------------------------------------
# CORAL target refresh (Decision 14 / 41) — implements in-process recomputation
# ---------------------------------------------------------------------------
@torch.no_grad()
def _refresh_coral_target(model: LADINetPhase2, train_records,
                           device: torch.device, pass1_scale: float = 0.0) -> torch.Tensor:
    """Recompute CORAL target covariance from current model's ABMIL features on
    the 680 real-field training images. Returns (768,768) tensor on `device`.
    The backbone is run with LoRA ACTIVE (the current Phase 2 state) so the
    target reflects the in-training feature distribution."""
    field_records = [r for r in train_records if r.is_field_photo]
    if len(field_records) < 100:
        print(f"    [WARN] only {len(field_records)} field records for CORAL refresh; "
              f"keeping existing target.")
        return None  # caller will keep previous target
    model.eval()
    model._set_lora_bypass(False)
    field_ds = LadiNetDataset(field_records, training=False, background_pool=None)
    loader = DataLoader(field_ds, batch_size=16, shuffle=False, num_workers=NUM_WORKERS)
    feats_list = []
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            cls_global, cls_spatial, fallback = _run_pass1(model, x, pass1_scale=pass1_scale)
            lesion = _select_lesion_crop(x, cls_spatial, fallback)
            bag = _run_pass2(model, lesion).float()
            feats_list.append(bag)  # keep on GPU — no .cpu() round-trip
    # [Developer fix 2026-04-22] Compute covariance on GPU with torch.cov instead of
    # np.cov on CPU. Eliminates pinned-RAM round-trip at refresh epochs (5/10/15/20),
    # which is precisely when gate re-runs also stress RAM.
    feats = torch.cat(feats_list, dim=0)            # [N, 768] on GPU, float32
    # torch.cov expects rows=variables, cols=observations — transpose from [N,768] to [768,N].
    cov = torch.cov(feats.T).to(torch.float32)       # [768, 768]
    return cov.to(device)


# ---------------------------------------------------------------------------
# Attention gate re-run (Decision 44) — in-process gate score on field_val
# ---------------------------------------------------------------------------
@torch.no_grad()
def _run_attention_gate_inline(model: LADINetPhase2, val_records,
                                device: torch.device, pass1_scale: float = 0.0) -> tuple[int, dict]:
    """Run the Phase 1 gate criterion on the current model. Returns (total_focus, per_class_rate)."""
    from ladinet_config import FALLBACK_MAX_ATTN_THRESHOLD, FALLBACK_ENTROPY_THRESHOLD
    FOCUS_TOP20_MASS_MIN = 0.18
    FOCUS_ENTROPY_MAX = 0.85 * math.log(NUM_PATCHES)

    disease_classes = [c for c in TOMATO_CLASSES if c != "tomato_healthy"]
    by_class = {c: [] for c in disease_classes}
    for r in val_records:
        if r.class_name in by_class and len(by_class[r.class_name]) < 4:
            by_class[r.class_name].append(r)
    selected = [r for c in disease_classes for r in by_class[c]]

    ds = LadiNetDataset(selected, training=False, background_pool=None)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=NUM_WORKERS)
    model.eval()
    per_class_focus = {c: 0 for c in disease_classes}
    per_class_total = {c: 0 for c in disease_classes}
    total_focus = 0

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            labels = batch["label"]
            # Run pass 1 with deferred-injection scale + pass 2 LoRA to get ABMIL attention
            cls_global, cls_spatial, fallback = _run_pass1(model, x, pass1_scale=pass1_scale)
            lesion = _select_lesion_crop(x, cls_spatial, fallback)
            model._set_lora_bypass(False)
            feat = model.backbone.forward_features(lesion)
            patches = feat[:, PREFIX_TOKENS:, :].float()
            _, abmil_attn = model.abmil(patches)      # [B, 784]

            for i in range(x.size(0)):
                cname = TOMATO_CLASSES[int(labels[i])]
                if cname not in by_class:
                    continue
                per_class_total[cname] += 1
                a = abmil_attn[i].cpu().numpy()
                top20 = np.argpartition(-a, 20)[:20]
                top20_mass = float(a[top20].sum())
                entropy = float(-(a * np.log(a + 1e-12)).sum())
                max_attn = float(a.max())
                focus = (
                    top20_mass >= FOCUS_TOP20_MASS_MIN
                    and entropy <= FOCUS_ENTROPY_MAX
                    and max_attn >= FALLBACK_MAX_ATTN_THRESHOLD
                    and entropy <= FALLBACK_ENTROPY_THRESHOLD
                )
                if focus:
                    per_class_focus[cname] += 1
                    total_focus += 1

    per_class_rates = {c: round(per_class_focus[c] / max(1, per_class_total[c]), 2)
                        for c in disease_classes}
    return total_focus, per_class_rates


# ---------------------------------------------------------------------------
# Phase 2 training loop
# ---------------------------------------------------------------------------
def train_phase2(num_epochs: int = PHASE2_MAX_EPOCHS, dry_run: bool = False,
                 n_batches_dry: int = 3):
    set_seeds(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[FAIL] CUDA required."); sys.exit(1)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    PHASE1_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_csv = LOGS_DIR / f"phase2_{run_id}.csv"
    metrics_csv.write_text(
        "epoch,train_loss,train_ce,train_supcon,train_coral,supcon_w_current,"
        "coral_warmup,val_sqrtn_f1,val_simple_f1,val_fallback_mean,lr_lora,lr_heads,elapsed_s\n",
        encoding="utf-8",
    )

    print("=" * 72)
    print(f"LADI-Net Phase 2 training  |  config_hash={CONFIG_HASH}  |  run_id={run_id}")
    print(f"device={device.type} ({torch.cuda.get_device_name()})")
    print(f"num_epochs={num_epochs}  bs={PHASE2_BATCH_SIZE}  resolution={RESOLUTION}")
    print("Note: attention-guided lesion-zoom is effective for foliar_spot (partial, 3/4 gate).")
    print("Septoria/YLCV/mosaic route through CLS global stream via fallback_flag.")
    print("Phase 2 improvements for septoria come from LoRA + CORAL + AmpMix + SupCon -- NOT lesion zoom.")
    print("=" * 72)

    # Model + LoRA attach + warm-start
    print("Building LADI-Net Phase 2 model (backbone + LoRA + heads)...")
    model = LADINetPhase2(device=device).to(device)
    n_lora_params = count_lora_params(model.backbone)
    print(f"  LoRA params: {n_lora_params:,}  (expected 147,456 for r=8 alpha=16 on 8 blocks)")

    # Warm-start heads from Phase 1
    if not PHASE1_HEADS_PT.exists():
        print(f"[FAIL] {PHASE1_HEADS_PT} not found."); sys.exit(1)
    phase1 = torch.load(PHASE1_HEADS_PT, map_location=device, weights_only=False)
    model.abmil.load_state_dict(phase1["abmil_state_dict"])
    model.fusion.load_state_dict(phase1["fusion_state_dict"])
    model.supcon.load_state_dict(phase1["supcon_projector_state_dict"])
    print(f"  warm-started heads from {rel(PHASE1_HEADS_PT)} (epoch {phase1.get('epoch')}, "
          f"val_f1={phase1.get('val_sqrtn_macro_f1'):.4f})")

    # Verify all trainable-counts
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  trainable={n_train:,} / total={n_total:,}  "
          f"(LoRA + ABMIL + fusion + SupCon projector)")

    # Load CORAL target and verify provenance
    if not CORAL_TARGET_PT.exists():
        print(f"[FAIL] {CORAL_TARGET_PT} not found."); sys.exit(1)
    coral_data = torch.load(CORAL_TARGET_PT, weights_only=False)
    assert isinstance(coral_data, dict) and coral_data.get("source") == "abmil_features_phase1", \
        f"CORAL target has wrong source: {coral_data.get('source') if isinstance(coral_data, dict) else type(coral_data)}"
    assert coral_data["n_samples"] == 680, f"n_samples={coral_data['n_samples']}"
    assert coral_data["resolution"] == 392, f"resolution={coral_data['resolution']}"
    coral_target = coral_data["cov"].to(device)
    print(f"  CORAL target: source={coral_data['source']}  n={coral_data['n_samples']}  "
          f"frob={coral_data['frobenius_norm']:.4f}")

    # Data
    train_records = load_split_records("train")
    val_records = load_split_records("field_val")
    bg_pool = load_background_pool()
    print(f"  train={len(train_records)}  field_val={len(val_records)}  bg_pool={len(bg_pool)}")

    train_ds = LadiNetDataset(train_records, training=True, background_pool=bg_pool)
    sampler = ClassStratifiedBatchSampler(train_records, phase="phase2", seed=SEED)
    train_loader = DataLoader(train_ds, batch_sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=True)
    print(f"  train batches/epoch={len(sampler)}  bs={PHASE2_BATCH_SIZE}")

    # Halt marker check (Round-2 debugger finding) — refuse to continue if prior run halted
    halt_marker = PHASE1_CKPT_DIR / "phase2_HALTED.txt"
    if halt_marker.exists() and not dry_run:
        print(f"[HALT] phase2_HALTED.txt exists from previous run:")
        print(halt_marker.read_text())
        print(f"Resolve by removing {halt_marker.name} after developer review, or starting a fresh run.")
        sys.exit(2)

    # Optimizer — parameter groups (Decision 17 sec 17.3)
    lora_params = model.lora_params()
    head_params = model.head_params()
    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": LR_LORA, "weight_decay": WD_LORA, "name": "lora"},
        {"params": head_params, "lr": LR_HEADS, "weight_decay": WD_HEADS, "name": "heads"},
    ])
    # Log per-group param counts (Round-2 debugger finding: verify fusion is trainable)
    abmil_n = sum(p.numel() for p in model.abmil.parameters())
    fusion_n = sum(p.numel() for p in model.fusion.parameters())
    supcon_n = sum(p.numel() for p in model.supcon.parameters())
    lora_n = sum(p.numel() for p in lora_params)
    print(f"  optimizer groups: lora={lora_n:,}  "
          f"heads={abmil_n+fusion_n+supcon_n:,} "
          f"(abmil={abmil_n:,} fusion={fusion_n:,} supcon={supcon_n:,})")
    assert all(p.requires_grad for p in model.fusion.parameters()), "fusion MLP not trainable!"

    # LR schedule: LinearLR warmup over 2 epochs then CosineAnnealingLR
    total_steps = len(sampler) * num_epochs
    warmup_steps = len(sampler) * WARMUP_EPOCHS
    def lr_lambda(step):
        if step < warmup_steps:
            return 0.05 + 0.95 * (step / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cos = 0.5 * (1 + math.cos(math.pi * progress))
        return LR_COSINE_MIN_RATIO + (1 - LR_COSINE_MIN_RATIO) * cos
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # CORAL state
    coral_source_cov_ema = torch.zeros(768, 768, device=device)
    coral_ema_update_counter = 0

    # SupCon weight monitoring
    supcon_w_current = LOSS_W_SUPCON_PHASE2
    consecutive_high_ratio = 0
    n_supcon_reductions = 0

    # Training loop state
    best_f1 = -1.0
    patience_counter = 0
    consecutive_floor_violations = 0
    val_rolling = deque(maxlen=PHASE2_ROLLING_MEAN_EPOCHS)
    gate_history = []
    global_step = 0
    start_epoch = 0
    ramp_reverted = False
    t0 = time.time()

    # [Decision 50 Fix 4] Resume from phase2_last.pt if present
    last_ckpt_path = PHASE1_CKPT_DIR / "phase2_last.pt"
    ramp_reverted_marker = PHASE1_CKPT_DIR / "pass1_lora_ramp_reverted.txt"
    if last_ckpt_path.exists() and not dry_run:
        print(f"[RESUME] Loading {last_ckpt_path}...")
        resume_ckpt = torch.load(last_ckpt_path, map_location=device, weights_only=False)
        if resume_ckpt.get("config_hash") != CONFIG_HASH:
            print(f"[FAIL] config_hash mismatch. Checkpoint: {resume_ckpt.get('config_hash')}. "
                  f"Current: {CONFIG_HASH}. Delete phase2_last.pt for fresh start.")
            sys.exit(3)
        model.load_state_dict(resume_ckpt["model_state_dict"])
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
        best_f1 = resume_ckpt.get("best_metric", -1.0)
        patience_counter = resume_ckpt.get("patience_counter", 0)
        consecutive_floor_violations = resume_ckpt.get("consecutive_floor_violations", 0)
        coral_source_cov_ema = resume_ckpt.get("coral_source_cov_ema", coral_source_cov_ema).to(device)
        coral_ema_update_counter = resume_ckpt.get("coral_ema_update_counter", 0)
        coral_target = resume_ckpt.get("coral_target_cov", coral_target).to(device)
        supcon_w_current = resume_ckpt.get("supcon_weight_current", LOSS_W_SUPCON_PHASE2)
        n_supcon_reductions = resume_ckpt.get("n_supcon_reductions", 0)
        gate_history = resume_ckpt.get("phase2_gate_history", [])
        start_epoch = resume_ckpt["epoch"] + 1
        ramp_reverted = resume_ckpt.get("pass1_ramp_reverted", False) or ramp_reverted_marker.exists()
        # Restore RNG states
        rng = resume_ckpt.get("rng_state", {})
        if rng:
            torch.set_rng_state(rng["torch_cpu"])
            torch.cuda.set_rng_state_all(rng["torch_cuda"])
            np.random.set_state(rng["numpy"])
            random.setstate(rng["python_random"])
        print(f"[RESUME] resuming at epoch {start_epoch} (was best_f1={best_f1:.4f}, "
              f"ramp_reverted={ramp_reverted})")
    elif ramp_reverted_marker.exists():
        ramp_reverted = True
        print(f"[INFO] pass1_lora_ramp_reverted.txt marker present — ramp disabled this run")

    for epoch in range(start_epoch, num_epochs):
        # [Decision 49] Compute Pass-1 LoRA scale for this epoch
        from ladinet_config import compute_pass1_lora_scale, PASS1_LORA_RAMP_START, PASS1_LORA_RAMP_END
        if ramp_reverted:
            pass1_lora_scale_current = 0.0  # ramp was aborted by safety check — stay bypassed
        else:
            pass1_lora_scale_current = compute_pass1_lora_scale(epoch)
        print(f"\n[Epoch {epoch}] pass1_lora_scale = {pass1_lora_scale_current:.3f}  "
              f"pass2_lora_scale = 1.0  (Decision 49)")

        model.train()
        # Keep backbone batchnorm/dropout in eval mode (ViT doesn't have BN but be safe)
        model.backbone.eval()
        sampler.set_epoch(epoch)

        ep_loss, ep_ce, ep_sc, ep_cl = 0.0, 0.0, 0.0, 0.0
        n_batches = 0
        ampmix_fires_this_epoch = 0
        coral_ema_skips = 0

        for batch_idx, batch in enumerate(train_loader):
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            is_field = batch["is_field_photo"].to(device, non_blocking=True)
            image_type_idx = batch["image_type_idx"].to(device, non_blocking=True)
            # Types: 0=LAB_OK, 1=LAB_FLAGGED, 2=FIELD, 3=RECOMPOSED
            is_lab_like = (image_type_idx != 2)  # treat lab + lab_flagged + recomposed as lab-domain

            # AmpMix (Decision 38) — on preprocessed images since they're already normalized.
            # Note: AmpMix ideally operates on raw pixels pre-preprocessing, but in this
            # DataLoader the LAB-CLAHE + augmentation is done in __getitem__ and we get
            # normalized tensors. Running AmpMix on normalized tensors still swaps frequency
            # components; the amplitude statistics differ slightly but empirically still effective.
            rng = random.Random(SEED * 1_000_003 + epoch * 31 + batch_idx)
            x_mixed, n_ampmix = apply_ampmix(x, y, p=AMPMIX_PROB, rng=rng)
            if n_ampmix > 0:
                ampmix_fires_this_epoch += 1
                x = x_mixed

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Pass 1 (frozen, LoRA bypassed)
                cls_global, cls_spatial, fallback = _run_pass1(model, x, pass1_scale=pass1_lora_scale_current)
                # Lesion crop selection
                lesion = _select_lesion_crop(x, cls_spatial, fallback)
                # Pass 2 (LoRA active)
                bag = _run_pass2(model, lesion)
                # Fusion
                fusion_in = torch.cat(
                    [bag, cls_global.detach(), fallback.unsqueeze(-1).float()], dim=-1
                ).float()
                logits = model.fusion(fusion_in)
                # Losses
                ce = weighted_ce_loss(logits, y, is_field)
                proj = model.supcon(bag.float())
                sc = supcon_loss(proj, y)

                # CORAL EMA update (Decision 26; Decision 29 REVISED by Decision 46 —
                # fallback exclusion removed because frozen Pass 1 + diffuse DINOv2 attention
                # on tomato leaves triggers fallback for ~100% of images, which would silently
                # disable CORAL entirely. Include all lab-domain images regardless of fallback).
                coral_mask = is_lab_like
                lab_count = int(coral_mask.sum().item())
                if lab_count >= CORAL_MIN_LAB_COUNT_IN_BATCH:
                    lab_feats = bag[coral_mask].float().detach()
                    centered = lab_feats - lab_feats.mean(dim=0, keepdim=True)
                    batch_cov = (centered.T @ centered) / max(1, lab_feats.shape[0] - 1)
                    coral_source_cov_ema = (
                        CORAL_EMA_DECAY * coral_source_cov_ema
                        + (1 - CORAL_EMA_DECAY) * batch_cov
                    )
                    coral_ema_update_counter += 1
                else:
                    coral_ema_skips += 1
                coral_warmup = min(1.0, coral_ema_update_counter / CORAL_WARMUP_STEPS)
                coral_raw = coral_frobenius_loss(coral_source_cov_ema, coral_target)
                cl = coral_warmup * coral_raw

                total_loss = (
                    LOSS_W_CE * ce
                    + supcon_w_current * sc
                    + LOSS_W_CORAL_PHASE2 * cl
                )

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params + head_params, GRAD_CLIP_NORM)
            optimizer.step()
            scheduler.step()

            ep_loss += float(total_loss.item())
            ep_ce += float(ce.item())
            ep_sc += float(sc.item())
            ep_cl += float(cl.item())
            n_batches += 1
            global_step += 1

            if batch_idx % 50 == 0:
                lrs = [pg["lr"] for pg in optimizer.param_groups]
                print(f"  ep={epoch} b={batch_idx:4d}/{len(sampler)}  "
                      f"loss={total_loss.item():.4f} ce={ce.item():.4f} sc={sc.item():.4f} "
                      f"cl={cl.item():.4f} w_sc={supcon_w_current:.3f} warmup={coral_warmup:.2f} "
                      f"lr_lora={lrs[0]:.2e} lr_heads={lrs[1]:.2e} fb_fires={int(fallback.sum().item())}/{x.size(0)} ampmix={n_ampmix}")

            if dry_run and batch_idx >= (n_batches_dry - 1):
                print(f"  [dry-run] exiting after {n_batches_dry} batches")
                break

        # End of epoch — validation
        train_loss = ep_loss / max(1, n_batches)
        train_ce = ep_ce / max(1, n_batches)
        train_sc = ep_sc / max(1, n_batches)
        train_cl = ep_cl / max(1, n_batches)
        elapsed = time.time() - t0

        if dry_run:
            print(f"[dry-run] Epoch 0 done (partial). "
                  f"loss={train_loss:.4f} ce={train_ce:.4f} sc={train_sc:.4f} cl={train_cl:.4f}")
            print(f"  ampmix fires: {ampmix_fires_this_epoch} batches; CORAL EMA skips: {coral_ema_skips}")
            # Save a minimal checkpoint for smoke test verification
            ckpt_path = PHASE1_CKPT_DIR / f"phase2_smoke_{run_id}.pt"
            torch.save({"epoch": 0, "config_hash": CONFIG_HASH,
                        "lora_params_count": n_lora_params,
                        "abmil_state_dict": model.abmil.state_dict()}, ckpt_path)
            print(f"  saved smoke checkpoint to {rel(ckpt_path)}")
            return

        val_metrics = evaluate_phase2(model, val_records, device,
                                       pass1_scale=pass1_lora_scale_current)
        val_f1 = val_metrics["sqrtn_macro_f1"]
        val_simple = val_metrics["simple_macro_f1"]

        # Disease F1 floor (Decision 17 sec 17.6)
        disease_classes = [c for c in TOMATO_CLASSES if c != "tomato_healthy"]
        if any(val_metrics["per_class_f1"][c] < DISEASE_F1_FLOOR for c in disease_classes):
            consecutive_floor_violations += 1
            if consecutive_floor_violations >= 2:
                print(f"  [WARN] disease F1 floor violated 2x -> stopping_metric overridden to 0")
                val_f1 = 0.0
        else:
            consecutive_floor_violations = 0

        val_rolling.append(val_f1)

        # SupCon monitor (Decision 42 + Decision 47 revision):
        # Skip monitoring for the first 3 epochs. Rationale: Phase 2 warm-starts heads from
        # Phase 1 (CE ~0.08 at epoch 0), so the ratio test would fire trivially at init
        # (weighted_supcon 0.30*~2.0 / CE 0.08 = 7.5× > 3.0 threshold) — pre-maturely shrinking
        # SupCon before LoRA had a chance to contribute. Once LoRA begins adapting (~epoch 3),
        # CE rises to a more representative level and the ratio test becomes meaningful.
        ratio = train_sc * supcon_w_current / max(train_ce, 1e-6)
        if epoch < 3:
            print(f"  [D42 skip epoch<3] supcon/ce ratio {ratio:.2f} (monitor delayed per Decision 47)")
        elif ratio > 3.0:
            consecutive_high_ratio += 1
            if consecutive_high_ratio >= 2:
                supcon_w_current *= 0.70
                n_supcon_reductions += 1
                print(f"  [INTERVENTION] SupCon/CE ratio {ratio:.2f} >3.0 x2; "
                      f"reduced SupCon weight to {supcon_w_current:.4f}")
                consecutive_high_ratio = 0
        elif ratio > 2.0:
            print(f"  [WARN] SupCon/CE ratio {ratio:.2f}")
            consecutive_high_ratio = 0
        else:
            consecutive_high_ratio = 0

        # LoRA delta norm as training-health signal (Critique 29)
        lora_delta_norm = float(sum(p.norm().item() for p in lora_params))
        print(f"\nEpoch {epoch}  elapsed={elapsed:.0f}s  "
              f"loss={train_loss:.4f} ce={train_ce:.4f} sc={train_sc:.4f} cl={train_cl:.4f} "
              f"lora_delta_norm={lora_delta_norm:.4f} coral_ema_updates={coral_ema_update_counter} "
              f"coral_ema_skips={coral_ema_skips}")
        print(f"  val sqrtn_macro_f1={val_f1:.4f}  simple_macro_f1={val_simple:.4f}")
        for c, v in val_metrics["per_class_f1"].items():
            print(f"    {c:35s}: {v:.4f}")
        for c, r in val_metrics["fallback_fire_rate_by_class"].items():
            print(f"    {c:35s} fallback: {r*100:.1f}%")

        fr_mean = float(np.mean(list(val_metrics["fallback_fire_rate_by_class"].values())))
        with open(metrics_csv, "a", encoding="utf-8") as f:
            lr_lora_now = optimizer.param_groups[0]["lr"]
            lr_heads_now = optimizer.param_groups[1]["lr"]
            coral_warmup = min(1.0, coral_ema_update_counter / CORAL_WARMUP_STEPS)
            f.write(f"{epoch},{train_loss},{train_ce},{train_sc},{train_cl},"
                    f"{supcon_w_current},{coral_warmup},{val_f1},{val_simple},{fr_mean},"
                    f"{lr_lora_now},{lr_heads_now},{elapsed}\n")

        # CORAL target refresh every 5 epochs (Decision 14 / 41)
        if epoch > 0 and epoch % CORAL_TARGET_REFRESH_EPOCHS == 0:   # epochs 5, 10, 15, 20
            print(f"  [CORAL refresh] epoch {epoch}: recomputing target from current ABMIL features")
            new_target = _refresh_coral_target(model, train_records, device,
                                                pass1_scale=pass1_lora_scale_current)
            if new_target is not None:
                coral_target = new_target
                print(f"    new CORAL target frobenius: {coral_target.norm().item():.4f}")
            else:
                print(f"    [WARN] refresh returned None; keeping previous target.")

        # Attention gate re-run every 5 epochs (Decision 44)
        halt_reason = None
        if epoch > 0 and epoch % 5 == 0:   # epochs 5, 10, 15, 20 — fatal-abort at epoch 10
            gate_score, per_class_rates = _run_attention_gate_inline(model, val_records, device,
                                                                       pass1_scale=pass1_lora_scale_current)
            gate_history.append({"epoch": epoch, "score": gate_score,
                                 "per_class": per_class_rates})
            print(f"  [attention gate] epoch {epoch}: {gate_score}/19 focus  "
                  f"per-class={per_class_rates}")
            if epoch >= 10 and gate_score < 8:
                halt_reason = (
                    f"attention_gate_fatal_epoch{epoch}_score{gate_score}_lt_8"
                )
                print(f"  [DEVELOPER ATTENTION REQUIRED] attention gate = {gate_score}/19 < 8 "
                      f"at epoch {epoch}. Septoria attention has not improved. Halting Phase 2 "
                      f"per Decision 44.")
                # Persist halt flag to a terminal marker file so resume logic will refuse to
                # continue (Round-2 debugger finding: halt must be persistent across restarts).
                halt_marker = PHASE1_CKPT_DIR / "phase2_HALTED.txt"
                halt_marker.write_text(
                    f"halt_reason={halt_reason}\nepoch={epoch}\n"
                    f"gate_score={gate_score}\n"
                    f"timestamp={datetime.datetime.now().isoformat()}\n",
                    encoding="utf-8",
                )
                break

        # [Decision 49] Ramp safety check during epochs 7-9 — if Pass-1 LoRA delta
        # still disrupts attention (foliar fallback rate > 90%), revert to scale=0.0.
        if PASS1_LORA_RAMP_START <= epoch < PASS1_LORA_RAMP_END and not ramp_reverted:
            foliar_fallback_rate = val_metrics["fallback_fire_rate_by_class"].get(
                "tomato_foliar_spot", 1.0
            )
            if foliar_fallback_rate > 0.90:
                print(f"  [RAMP SAFETY] foliar fallback rate {foliar_fallback_rate*100:.0f}% > 90% "
                      f"at epoch {epoch} (scale={pass1_lora_scale_current:.2f}). "
                      f"LoRA delta still disrupting Pass-1 attention. Reverting to scale=0.0 for "
                      f"all remaining epochs (Decision 49 safety).")
                ramp_reverted = True
                ramp_reverted_marker.write_text(
                    f"Reverted at epoch {epoch}.\n"
                    f"foliar_fallback_rate={foliar_fallback_rate}\n"
                    f"pass1_lora_scale={pass1_lora_scale_current}\n"
                    f"timestamp={datetime.datetime.now().isoformat()}\n",
                    encoding="utf-8",
                )
            else:
                print(f"  [RAMP SAFETY] foliar fallback rate {foliar_fallback_rate*100:.0f}% "
                      f"(scale={pass1_lora_scale_current:.2f}) — ramp proceeding")

        # [Decision 50 Fix 4] Save checkpoint EVERY epoch (not only on improvement).
        ckpt = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_metric": best_f1, "best_epoch": best_f1 > 0 and epoch or -1,
            "patience_counter": patience_counter,
            "consecutive_floor_violations": consecutive_floor_violations,
            "coral_source_cov_ema": coral_source_cov_ema,
            "coral_ema_update_counter": coral_ema_update_counter,
            "coral_target_cov": coral_target,
            "supcon_weight_current": supcon_w_current,
            "n_supcon_reductions": n_supcon_reductions,
            "sampler_epoch_seed": epoch,
            "rng_state": {
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all(),
                "numpy": np.random.get_state(),
                "python_random": random.getstate(),
            },
            "config_hash": CONFIG_HASH,
            "decisions_version": 51,
            "val_metrics": val_metrics,
            "phase2_gate_history": gate_history,
            "pass1_lora_ramp_start": PASS1_LORA_RAMP_START,
            "pass1_lora_ramp_end": PASS1_LORA_RAMP_END,
            "pass1_ramp_reverted": ramp_reverted,
            "pass1_lora_scale_this_epoch": pass1_lora_scale_current,
        }
        epoch_path = PHASE1_CKPT_DIR / f"phase2_epoch{epoch:02d}_f1{val_f1:.4f}.pt"
        torch.save(ckpt, epoch_path)
        # Always update phase2_last.pt
        torch.save(ckpt, PHASE1_CKPT_DIR / "phase2_last.pt")
        print(f"  -> phase2_epoch{epoch:02d} + phase2_last.pt saved")

        # Best-checkpoint update only on improvement
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            ckpt["best_metric"] = best_f1
            ckpt["best_epoch"] = epoch
            torch.save(ckpt, PHASE1_CKPT_DIR / "phase2_best.pt")
            print(f"  -> best phase2 checkpoint updated: val_f1={val_f1:.4f}")
        else:
            if epoch >= PATIENCE_STARTS_EPOCH:
                patience_counter += 1
            if epoch >= PHASE2_MIN_EPOCHS and patience_counter >= PHASE2_PATIENCE:
                print(f"  early stopping at epoch {epoch} (patience {patience_counter})")
                break

    print("\n" + "=" * 72)
    print(f"Phase 2 complete. Best val sqrtn_macro_f1 = {best_f1:.4f}")
    print(f"Checkpoints: {PHASE1_CKPT_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_epochs", type=int, default=PHASE2_MAX_EPOCHS)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--n_batches", type=int, default=3, help="Number of batches for dry_run")
    args = ap.parse_args()
    train_phase2(num_epochs=args.num_epochs, dry_run=args.dry_run,
                 n_batches_dry=args.n_batches)
