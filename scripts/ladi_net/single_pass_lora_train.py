"""
Single-Pass LoRA training for LADI-Net tomato specialist.

MOTIVATION:
- Phase 1 (frozen DINOv2 + ABMIL + gated MLP) achieved field_val sqrtn_f1 = 0.9112
  and final held-out (one-shot) sqrtn_f1 = 0.7958 (per ladi_final_evaluation_report.md).
- Phase 2 (ABMIL + LoRA two-pass) regressed to field_val 0.8662 due to ABMIL
  maladaptation (trained on frozen Pass-1 features, then asked to work with
  LoRA-adapted Pass-2 features -> gate degraded 6/19 -> 5/19).
- This script removes ALL two-pass complexity: no ABMIL, no gated MLP, no lesion
  crop, no fallback flag. Just DINOv2 + LoRA on blocks 4-11 + Linear(768,6) +
  SupCon projector. CE + SupCon + CORAL (on CLS tokens) + AmpMix.

PRE-COMMITTED EVALUATION GATE (LOCK-1 from session log):
  Evaluate this model on the locked 104-image held-out set ONLY IF its best
  field_val sqrtn_macro_f1 >= 0.9212 (Phase 1's 0.9112 + 0.01).
  If field_val_best < 0.9212: Phase 1 remains production; this model is
  documented as "tested, did not improve" and the held-out set is NOT
  evaluated a second time. This is locked BEFORE training starts to prevent
  post-hoc selection bias.

Differences from Phase 2 (explicit, for PVA):
  [SP-1] No warm-start from phase1_heads.pt. Classifier is random-init (nn.Linear
         Kaiming). CE at epoch 0 batch 0 should therefore ~ln(6) = 1.79.
  [SP-2] LoRA target blocks = 4..11 (top 8 of 12). Same as Phase 2.
  [SP-3] CORAL target = covariance of CLS tokens (NOT ABMIL features).
         Computed at startup from FROZEN backbone over 680 real-field training
         images. Refresh every 5 epochs with LoRA-adapted features.
  [SP-4] No ABMIL, no fallback_flag, no lesion_crop anywhere in this script.
         Grep-safe: those strings appear only in comments/docstrings.
  [SP-5] SupCon/CE ratio: starts low (ratio = 0.30*1.2/1.79 ~ 0.20) because CE
         starts high at random-init. No Decision 47 epoch-3 delay needed.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from ladinet_config import (
    PROJECT_ROOT, TOMATO_CLASSES, CLASS_TO_IDX, RESOLUTION,
    BACKBONE, SEED, SUPCON_TAU, AMPMIX_PROB, FIELD_LOSS_WEIGHT,
    PHASE2_BATCH_SIZE as BATCH_SIZE,        # reuse batch constant
)
from ladinet_dataloader import (
    LadiRecord, LadiNetDataset, load_split_records, load_background_pool,
    ClassStratifiedBatchSampler,
)
from ladinet_model import (
    attach_lora_to_backbone, count_lora_params, SupConProjector,
)
from ladinet_losses import supcon_loss, weighted_ce_loss


# ----------------------------------------------------------------------------
# Constants — single-pass-specific
# ----------------------------------------------------------------------------
LR_LORA = 5e-5         # Phase 2 used 1e-4; LoRA delta grew to ~28. This is halved.
LR_HEADS = 5e-4        # classifier + projector
WEIGHT_DECAY_LORA = 0.01
WEIGHT_DECAY_HEADS = 0.0
GRAD_CLIP_MAX_NORM = 1.0

MAX_EPOCHS = 25
MIN_EPOCHS = 12
PATIENCE = 5
SUPCON_WEIGHT = 0.30
CORAL_WEIGHT = 0.50

# Stopping floor: any non-healthy class F1 < 0.30 for 2 consec epochs -> metric = 0
DISEASE_F1_FLOOR = 0.30

# PDA-1.3-inspired watch: ratio monitor (no epoch-3 delay needed since CE starts high)
SUPCON_CE_RATIO_ALARM = 2.0
SUPCON_CE_RATIO_INTERVENTION = 3.0     # reduce SUPCON_WEIGHT by 30% for 2 consec

# CORAL EMA settings (reused from Phase 2 discipline)
CORAL_EMA_MOMENTUM = 0.99
CORAL_MIN_LAB_IN_BATCH = 6             # Decision 26
CORAL_REFRESH_EVERY = 5

# LoRA delta-norm watch
LORA_DELTA_RED_FLAG = 30.0    # hard stop, reduce LR_LORA to 2e-5 and restart

# Pre-committed final-val gate (LOCK-1)
PRE_COMMITTED_FINAL_GATE = 0.9212      # = 0.9112 + 0.01


# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
CKPT_DIR = PROJECT_ROOT / "models" / "specialist" / "sp_lora_checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
SP_BEST = CKPT_DIR / "sp_lora_best.pt"
SP_LAST = CKPT_DIR / "sp_lora_last.pt"
SP_CORAL_TARGET = CKPT_DIR / "sp_coral_target_cls.pt"
SP_TRAIN_CSV = CKPT_DIR / "sp_train_log.csv"


def log(msg: str):
    print(f"[sp-lora] {msg}", flush=True)


def set_seeds(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------
# SinglePassLoRA model
# ----------------------------------------------------------------------------
class SinglePassLoRA(nn.Module):
    """DINOv2-Base-Registers + LoRA on blocks 4-11 + Linear classifier + SupCon projector.

    Forward returns dict: {"logits", "cls", "proj"}. CLS is the raw 768-dim token;
    proj is the L2-normalised SupCon embedding (discard at inference).
    """

    def __init__(self, device: torch.device, n_classes: int = 6):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            BACKBONE, pretrained=True, num_classes=0,
            img_size=RESOLUTION, dynamic_img_size=True,
        ).to(device)
        # Register-token sanity
        if getattr(self.backbone, "num_reg_tokens", 0) != 4:
            raise RuntimeError(
                f"Expected num_reg_tokens=4, got {self.backbone.num_reg_tokens}"
            )
        # Freeze backbone entirely
        for p in self.backbone.parameters():
            p.requires_grad = False
        # Attach LoRA to blocks 4..11
        n_adapters = attach_lora_to_backbone(self.backbone)
        log(f"  Attached {n_adapters} LoRA adapters "
            f"(LoRA params: {count_lora_params(self.backbone):,})")

        # Head: simple linear classifier on CLS token
        self.classifier = nn.Linear(768, n_classes).to(device)
        # Head: SupCon projector (reuse existing 768->256->128)
        self.projector = SupConProjector().to(device)

    def forward(self, x: torch.Tensor) -> dict:
        feat = self.backbone.forward_features(x)        # [B, 789, 768]
        cls = feat[:, 0, :]                              # [B, 768]
        logits = self.classifier(cls)                    # [B, 6]
        proj = F.normalize(self.projector(cls), dim=-1)  # [B, 128]
        return {"logits": logits, "cls": cls, "proj": proj}

    def head_params(self):
        return list(self.classifier.parameters()) + list(self.projector.parameters())

    def lora_params(self):
        return [p for p in self.backbone.parameters() if p.requires_grad]

    def lora_delta_frobenius_norm(self) -> float:
        """Sum of squared LoRA deltas (B @ A scaled) -> sqrt. Diagnostic only."""
        import math
        total = 0.0
        for m in self.backbone.modules():
            if hasattr(m, "lora_A") and hasattr(m, "lora_B") and hasattr(m, "scale"):
                # Per LoRALinear: delta_W = lora_B @ lora_A * scale
                with torch.no_grad():
                    dW = (m.lora_B @ m.lora_A) * m.scale
                    total += dW.pow(2).sum().item()
        return math.sqrt(total)


# ----------------------------------------------------------------------------
# AmpMix — reused from phase2_train.py (FFT amplitude swap, low-freq central block)
# ----------------------------------------------------------------------------
def apply_ampmix(images: torch.Tensor, labels: torch.Tensor,
                 p: float = AMPMIX_PROB, rng=None) -> tuple[torch.Tensor, int]:
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
        for c in range(a.shape[0]):
            Fa = torch.fft.fft2(a[c]); Fb = torch.fft.fft2(b[c])
            Aa = Fa.abs(); Ab = Fb.abs()
            Pa = torch.angle(Fa); Pb = torch.angle(Fb)
            H, W = Aa.shape
            Aa_s = torch.fft.fftshift(Aa); Ab_s = torch.fft.fftshift(Ab)
            ch, cw = H // 2, W // 2
            r = min(H, W) // 8
            tmp = Aa_s[ch-r:ch+r, cw-r:cw+r].clone()
            Aa_s[ch-r:ch+r, cw-r:cw+r] = Ab_s[ch-r:ch+r, cw-r:cw+r]
            Ab_s[ch-r:ch+r, cw-r:cw+r] = tmp
            Aa2 = torch.fft.ifftshift(Aa_s); Ab2 = torch.fft.ifftshift(Ab_s)
            a[c] = torch.fft.ifft2(Aa2 * torch.exp(1j * Pa)).real
            b[c] = torch.fft.ifft2(Ab2 * torch.exp(1j * Pb)).real
        mixed[fi] = a.to(images.dtype)
        mixed[si] = b.to(images.dtype)
    return mixed, n_pairs


# ----------------------------------------------------------------------------
# CORAL loss
# ----------------------------------------------------------------------------
def coral_frobenius_loss(source_cov: torch.Tensor, target_cov: torch.Tensor) -> torch.Tensor:
    d = source_cov.shape[0]
    return (source_cov - target_cov).pow(2).sum() / (4 * d * d)


def compute_cls_coral_target(model: SinglePassLoRA, field_records: list[LadiRecord],
                             device: torch.device) -> torch.Tensor:
    """Covariance of CLS tokens across all real-field training images (frozen or current LoRA).

    [SP-3] Uses CLS tokens (NOT ABMIL features). Called at startup (frozen) and
    at every CORAL_REFRESH_EVERY epochs (LoRA-adapted).
    """
    model.eval()
    ds = LadiNetDataset(field_records, training=False, background_pool=None)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0,
                        collate_fn=lambda batch: {
                            "image": torch.stack([b["image"] for b in batch]),
                            "label": torch.stack([b["label"] for b in batch]),
                        })
    cls_list = []
    with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            out = model(x)
            cls_list.append(out["cls"].float())       # keep on GPU
    feats = torch.cat(cls_list, dim=0)                 # [N, 768] on GPU
    cov = torch.cov(feats.T).to(torch.float32)        # [768, 768]
    return cov.to(device)


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def per_class_f1_np(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    C = len(TOMATO_CLASSES)
    out = np.zeros(C, dtype=np.float64)
    for c in range(C):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        if (2 * tp + fp + fn) == 0:
            out[c] = 0.0
        else:
            out[c] = 2.0 * tp / (2 * tp + fp + fn)
    return out


def sqrtn_macro_f1_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    f1s = per_class_f1_np(y_true, y_pred)
    present = []; weights = []
    for c in range(len(TOMATO_CLASSES)):
        n_c = int((y_true == c).sum())
        if n_c > 0:
            present.append(c); weights.append(float(np.sqrt(n_c)))
    if not present:
        return 0.0
    w = np.array(weights) / sum(weights)
    return float(sum(w[i] * f1s[c] for i, c in enumerate(present)))


@torch.no_grad()
def evaluate(model: SinglePassLoRA, val_records: list[LadiRecord],
             device: torch.device) -> dict:
    model.eval()
    ds = LadiNetDataset(val_records, training=False, background_pool=None)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0,
                        collate_fn=lambda batch: {
                            "image": torch.stack([b["image"] for b in batch]),
                            "label": torch.stack([b["label"] for b in batch]),
                        })
    all_preds, all_labels = [], []
    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].numpy()
            out = model(x)
            preds = out["logits"].argmax(dim=-1).cpu().numpy()
            all_preds.append(preds); all_labels.append(y)
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    per_class = per_class_f1_np(y_true, y_pred)
    sqrtn = sqrtn_macro_f1_np(y_true, y_pred)
    return {
        "sqrtn_macro_f1": sqrtn,
        "per_class_f1": {c: float(per_class[i]) for i, c in enumerate(TOMATO_CLASSES)},
        "n": len(y_true),
    }


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def train(num_epochs: int = MAX_EPOCHS, dry_run: bool = False,
          n_batches_dry: int = 3, resume: bool = True,
          no_early_stop: bool = False):
    set_seeds(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    log(f"Pre-committed final-val gate: field_val sqrtn_macro_f1 >= {PRE_COMMITTED_FINAL_GATE}")
    log(f"  (If best field_val < {PRE_COMMITTED_FINAL_GATE}, the locked held-out set is NOT evaluated.)")

    # === Model =====================
    log("\n=== Building SinglePassLoRA model ===")
    model = SinglePassLoRA(device).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"  trainable {trainable:,} / total {total_params:,}")

    # === Data ======================
    log("\n=== Loading training data ===")
    train_records = load_split_records("train")
    field_val_records = load_split_records("field_val")
    log(f"  train={len(train_records)}, field_val={len(field_val_records)}")

    # Filter train records that are real-field for CORAL target
    field_train_records = [r for r in train_records if r.is_field_photo]
    log(f"  field-only training subset (for CORAL target): {len(field_train_records)}")

    log("\n=== Loading background pool ===")
    bg_pool = load_background_pool()
    log(f"  background pool: {len(bg_pool)} images")

    ds_train = LadiNetDataset(train_records, training=True, background_pool=bg_pool,
                              rng_seed=SEED)
    sampler = ClassStratifiedBatchSampler(train_records, phase="phase2", seed=SEED)
    loader_train = DataLoader(ds_train, batch_sampler=sampler, num_workers=0,
                              collate_fn=lambda batch: {
                                  "image": torch.stack([b["image"] for b in batch]),
                                  "label": torch.stack([b["label"] for b in batch]),
                                  "is_field_photo": torch.stack([b["is_field_photo"] for b in batch]),
                              })
    log(f"  batch_size={BATCH_SIZE}  batches/epoch={len(sampler)}")

    # === CORAL target (startup) ===
    log("\n=== Computing CLS CORAL target on FROZEN backbone ===")
    if SP_CORAL_TARGET.exists() and resume:
        td = torch.load(SP_CORAL_TARGET, map_location=device, weights_only=False)
        coral_target = td["target_cov"].to(device)
        log(f"  Loaded cached CORAL target (frob={coral_target.norm().item():.4f})")
    else:
        coral_target = compute_cls_coral_target(model, field_train_records, device)
        torch.save({
            "target_cov": coral_target.cpu(),
            "source": "CLS_tokens_from_frozen_backbone",
            "n_images": len(field_train_records),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, SP_CORAL_TARGET)
        log(f"  Computed CORAL target (frob={coral_target.norm().item():.4f}, "
            f"shape={tuple(coral_target.shape)}), saved: {SP_CORAL_TARGET.name}")

    # === Optimizer =================
    lora_params = model.lora_params()
    head_params = model.head_params()
    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": LR_LORA, "weight_decay": WEIGHT_DECAY_LORA, "name": "lora"},
        {"params": head_params, "lr": LR_HEADS, "weight_decay": WEIGHT_DECAY_HEADS, "name": "heads"},
    ])
    total_steps = len(sampler) * num_epochs
    warmup_steps = len(sampler) * 2
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        # Cosine anneal to 10% peak
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        import math
        cos = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return 0.1 + 0.9 * cos
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # BF16 autocast + GradScaler (never permanent cast)
    amp_dtype = torch.bfloat16
    # BF16 generally doesn't need GradScaler (no gradient underflow like FP16)
    # but we use one anyway for consistency; enabled=False for bfloat16 autocast.

    # === Resume logic ==============
    start_epoch = 0
    best_f1 = 0.0
    consec_below_floor = 0
    supcon_weight_current = SUPCON_WEIGHT
    ratio_intervention_counter = 0

    if resume and SP_LAST.exists():
        log(f"\n=== Resuming from {SP_LAST.name} ===")
        ckpt = torch.load(SP_LAST, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_f1 = ckpt.get("best_field_val_f1", 0.0)
        consec_below_floor = ckpt.get("consec_below_floor", 0)
        supcon_weight_current = ckpt.get("supcon_weight_current", SUPCON_WEIGHT)
        log(f"  resumed at epoch {start_epoch}, best_f1={best_f1:.4f}")

    # CSV log
    csv_header = ["epoch", "train_loss", "ce", "sc", "cl", "lora_delta_norm",
                  "val_sqrtn_f1", "lr_lora", "lr_heads", "supcon_weight", "ampmix_fires"]
    if not SP_TRAIN_CSV.exists():
        with open(SP_TRAIN_CSV, "w", newline="") as f:
            csv.writer(f).writerow(csv_header)

    # Rolling F1 for stopping
    recent_f1s = []

    log(f"\n=== Training for epochs {start_epoch}..{num_epochs - 1} ===")
    rng = random.Random(SEED)
    for epoch in range(start_epoch, num_epochs):
        sampler.set_epoch(epoch)
        model.train()
        ep_loss = ep_ce = ep_sc = ep_cl = 0.0
        n_batches = 0
        ampmix_fires_epoch = 0

        # Refresh CORAL target every 5 epochs (except epoch 0)
        if epoch > 0 and epoch % CORAL_REFRESH_EVERY == 0:
            log(f"\n  [epoch {epoch}] Refreshing CORAL target on LoRA-adapted CLS...")
            coral_target = compute_cls_coral_target(model, field_train_records, device)
            log(f"    new target frob={coral_target.norm().item():.4f}")
            model.train()

        t0 = time.time()
        for batch_idx, batch in enumerate(loader_train):
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            is_field = batch["is_field_photo"].to(device, non_blocking=True)

            # AmpMix (p=0.45, foliar-septoria pairs only)
            x, n_mix = apply_ampmix(x, y, p=AMPMIX_PROB, rng=rng)
            if n_mix > 0:
                ampmix_fires_epoch += 1

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
                out = model(x)
                logits = out["logits"]
                proj = out["proj"]
                cls = out["cls"]

                # CE with 8x field weighting
                ce = weighted_ce_loss(logits, y, is_field)
                # SupCon
                sc = supcon_loss(proj, y)
                # CORAL on CLS tokens (source cov = current batch CLS covariance)
                # Use only lab images to avoid CORAL from field-on-field (SP-3)
                lab_mask = (is_field == 0)
                if int(lab_mask.sum().item()) >= CORAL_MIN_LAB_IN_BATCH:
                    cls_lab = cls[lab_mask].float()
                    source_cov = torch.cov(cls_lab.T)
                    cl = coral_frobenius_loss(source_cov, coral_target)
                else:
                    cl = torch.zeros((), device=device)

                total = ce + supcon_weight_current * sc + CORAL_WEIGHT * cl

            if torch.isnan(total).any():
                log(f"  [epoch {epoch} b{batch_idx}] NaN in loss — HALT")
                halt_marker = CKPT_DIR / "sp_lora_HALTED.txt"
                halt_marker.write_text(f"NaN at epoch {epoch} batch {batch_idx}", encoding="utf-8")
                return
            total.backward()
            # Grad clip
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], GRAD_CLIP_MAX_NORM
            )
            optimizer.step()
            scheduler.step()

            ep_loss += float(total.item())
            ep_ce += float(ce.item())
            ep_sc += float(sc.item())
            ep_cl += float(cl.item())
            n_batches += 1

            if batch_idx % 50 == 0:
                lrs = [pg["lr"] for pg in optimizer.param_groups]
                log(f"  ep={epoch} b={batch_idx:4d}/{len(sampler)}  "
                    f"loss={total.item():.4f} ce={ce.item():.4f} sc={sc.item():.4f} "
                    f"cl={cl.item():.4f}  lr_lora={lrs[0]:.2e} lr_heads={lrs[1]:.2e}  "
                    f"ampmix={ampmix_fires_epoch}")

            if dry_run and batch_idx >= (n_batches_dry - 1):
                log(f"  [dry-run] exiting after {n_batches_dry} batches")
                break

        avg_loss = ep_loss / max(n_batches, 1)
        avg_ce = ep_ce / max(n_batches, 1)
        avg_sc = ep_sc / max(n_batches, 1)
        avg_cl = ep_cl / max(n_batches, 1)
        lora_delta = model.lora_delta_frobenius_norm()

        log(f"\nEpoch {epoch} done  elapsed={time.time() - t0:.0f}s  "
            f"loss={avg_loss:.4f} ce={avg_ce:.4f} sc={avg_sc:.4f} cl={avg_cl:.4f}  "
            f"lora_delta={lora_delta:.4f}  ampmix_fires={ampmix_fires_epoch}")

        # RED FLAG check: LoRA delta
        if epoch >= 2 and lora_delta > LORA_DELTA_RED_FLAG:
            log(f"  RED FLAG: lora_delta_norm = {lora_delta:.2f} > {LORA_DELTA_RED_FLAG}")
            log(f"  Recommendation: halt and restart with LR_LORA halved (2.5e-5).")
            halt_marker = CKPT_DIR / "sp_lora_HALTED.txt"
            halt_marker.write_text(
                f"LoRA delta {lora_delta:.2f} exceeded threshold {LORA_DELTA_RED_FLAG} "
                f"at epoch {epoch}. Halt for developer review.",
                encoding="utf-8",
            )
            return

        if dry_run:
            log("[dry-run] exiting after partial epoch")
            # Save smoke checkpoint
            smoke_path = CKPT_DIR / f"sp_lora_smoke_{time.strftime('%Y%m%d_%H%M%S')}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "partial": True,
                "avg_loss_partial": avg_loss, "avg_ce": avg_ce, "avg_sc": avg_sc,
                "lora_delta_norm": lora_delta,
            }, smoke_path)
            log(f"  saved smoke ckpt: {smoke_path}")
            return

        # SupCon/CE ratio monitoring (no delay needed — [SP-5])
        ratio = (supcon_weight_current * avg_sc) / max(avg_ce, 1e-6)
        if ratio > SUPCON_CE_RATIO_INTERVENTION:
            ratio_intervention_counter += 1
            if ratio_intervention_counter >= 2:
                old = supcon_weight_current
                supcon_weight_current *= 0.7
                log(f"  [D42] SupCon/CE ratio {ratio:.2f} > {SUPCON_CE_RATIO_INTERVENTION} "
                    f"for 2 epochs -> reduce SupCon weight {old:.3f} -> {supcon_weight_current:.3f}")
                ratio_intervention_counter = 0
        elif ratio > SUPCON_CE_RATIO_ALARM:
            log(f"  [D42] ALARM: SupCon/CE ratio = {ratio:.2f} > {SUPCON_CE_RATIO_ALARM}")
        else:
            ratio_intervention_counter = 0

        # === Validation ==============
        log(f"  validating on field_val (n={len(field_val_records)})...")
        val = evaluate(model, field_val_records, device)
        val_f1 = val["sqrtn_macro_f1"]
        log(f"  field_val sqrtn_macro_f1 = {val_f1:.4f}")
        for cls, f1 in val["per_class_f1"].items():
            warn = "  <-- below floor" if (cls != "tomato_healthy" and f1 < DISEASE_F1_FLOOR) else ""
            log(f"    {cls:35s}: {f1:.4f}{warn}")

        # Disease F1 floor check
        below_floor = any(
            f1 < DISEASE_F1_FLOOR
            for cls, f1 in val["per_class_f1"].items()
            if cls != "tomato_healthy"
        )
        if below_floor:
            consec_below_floor += 1
            if consec_below_floor >= 2:
                log(f"  [D17 §17.6] 2 consec epochs below floor -> overriding sqrtn_f1 = 0.0")
                val_f1 = 0.0
        else:
            consec_below_floor = 0

        # 3-epoch rolling mean
        recent_f1s.append(val_f1)
        if len(recent_f1s) > 3:
            recent_f1s.pop(0)
        rolling_f1 = sum(recent_f1s) / len(recent_f1s)
        log(f"  3-epoch rolling mean field_val_f1 = {rolling_f1:.4f}")

        # CSV log
        with open(SP_TRAIN_CSV, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, f"{avg_loss:.6f}", f"{avg_ce:.6f}", f"{avg_sc:.6f}", f"{avg_cl:.6f}",
                f"{lora_delta:.4f}", f"{val_f1:.4f}",
                f"{optimizer.param_groups[0]['lr']:.6e}",
                f"{optimizer.param_groups[1]['lr']:.6e}",
                f"{supcon_weight_current:.4f}", ampmix_fires_epoch,
            ])

        # Checkpoint
        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_sqrtn_macro_f1": val_f1,
            "rolling_f1": rolling_f1,
            "best_field_val_f1": max(best_f1, rolling_f1),
            "per_class_f1": val["per_class_f1"],
            "lora_delta_norm": lora_delta,
            "supcon_weight_current": supcon_weight_current,
            "consec_below_floor": consec_below_floor,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        ep_ckpt = CKPT_DIR / f"sp_lora_epoch{epoch:02d}_f1{val_f1:.4f}.pt"
        torch.save(payload, ep_ckpt)
        torch.save(payload, SP_LAST)
        if rolling_f1 > best_f1:
            best_f1 = rolling_f1
            torch.save(payload, SP_BEST)
            log(f"  -> new best rolling f1={rolling_f1:.4f} saved: {SP_BEST.name}")

        # Early stopping: after min_epochs, patience-based (disabled by --no_early_stop)
        if no_early_stop:
            # Still log what WOULD have happened for post-run analysis
            import pandas as pd
            df = pd.read_csv(SP_TRAIN_CSV)
            recent = df[["epoch", "val_sqrtn_f1"]].tail(PATIENCE + 1).reset_index(drop=True)
            max_recent = recent["val_sqrtn_f1"].astype(float).max()
            best_epoch_idx = int(recent["val_sqrtn_f1"].astype(float).idxmax())
            epochs_since_best = len(recent) - 1 - best_epoch_idx
            would_trigger = epoch + 1 >= MIN_EPOCHS and epochs_since_best >= PATIENCE
            if would_trigger:
                log(f"  [early-stop bypassed] Would have stopped here: "
                    f"{epochs_since_best} epochs since best in last "
                    f"{PATIENCE + 1} window (best={max_recent:.4f}). Continuing due to --no_early_stop.")
        elif epoch + 1 >= MIN_EPOCHS:
            hist_path = SP_TRAIN_CSV
            import pandas as pd
            df = pd.read_csv(hist_path)
            recent = df[["epoch", "val_sqrtn_f1"]].tail(PATIENCE + 1).reset_index(drop=True)
            max_recent = recent["val_sqrtn_f1"].astype(float).max()
            best_epoch_idx = int(recent["val_sqrtn_f1"].astype(float).idxmax())
            epochs_since_best = len(recent) - 1 - best_epoch_idx
            if epochs_since_best >= PATIENCE:
                log(f"  EARLY STOP: {epochs_since_best} epochs since best in last "
                    f"{PATIENCE + 1} window (best={max_recent:.4f})")
                break

    # === Done =====================
    log(f"\n{'='*64}")
    log("SINGLE-PASS LoRA TRAINING COMPLETE")
    log(f"{'='*64}")
    log(f"  best field_val sqrtn_macro_f1 (rolling): {best_f1:.4f}")
    log(f"  pre-committed gate: {PRE_COMMITTED_FINAL_GATE}")
    if best_f1 >= PRE_COMMITTED_FINAL_GATE:
        log(f"  -> GATE PASSED. Single-pass is eligible for held-out evaluation.")
        log(f"     (Developer decision: run a separate Phase-4-equivalent script with")
        log(f"      the marker guard temporarily bypassed. Document justification.)")
    else:
        log(f"  -> GATE FAILED. Single-pass did not exceed Phase 1 by 0.01.")
        log(f"     Phase 1 remains production. Held-out set will NOT be evaluated again.")
        log(f"     This script's field_val best ({best_f1:.4f}) is the final word.")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--n_batches", type=int, default=3)
    parser.add_argument("--num_epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--no_early_stop", action="store_true",
                        help="Bypass patience-based early stop. Used for continuation runs.")
    args = parser.parse_args()
    train(num_epochs=args.num_epochs, dry_run=args.dry_run,
          n_batches_dry=args.n_batches, resume=(not args.no_resume),
          no_early_stop=args.no_early_stop)
