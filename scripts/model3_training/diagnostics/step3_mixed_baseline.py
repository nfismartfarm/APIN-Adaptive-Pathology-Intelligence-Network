"""
Step 3 / Diagnostic — 5-epoch MIXED-domain baseline (Stage 0.5 sanity check).

Purpose: literature gives no support for "field-first then lab" curricula.
Before committing 32 epochs to that curriculum, run a mixed-domain baseline
with field oversampling and see what field_val_f1 hits in 5 epochs. If it's
already strong, the field-first curriculum is suspect.

Architecture (matches the eventual Stage 1 spec MINUS MixStyle, MINUS SupCon,
MINUS AmpMix, MINUS LoRA — we add those after the curriculum question is settled):
  - timm vit_small_patch14_reg4_dinov2 (FROZEN, img_size=224)
  - 768-dim feature = CLS + mean(non-CLS tokens)
  - SEBlock(768 → 48 → 768)
  - HardFiLMConditioner(3 modes: tomato=0, chilli=1, uncertain=2)
  - LayerNorm + Linear(768 → 10)

Training:
  - WeightedRandomSampler with field photos sampled 2× relative to lab (gives
    ~55% field per batch on this train pool, in line with the curriculum's
    Stage 2 batch composition)
  - CE loss with ENS class weights (beta=0.999, capped 5:1, anthracnose ×1.5)
  - AdamW(lr=1e-3, weight_decay=1e-4) on heads only (backbone frozen)
  - 75/15/10 crop_mode dropout from epoch 5 onward (we hit epoch 5 at end so
    only the 75% case fires — adversarial wrong-crop deferred to Stage 2)
  - num_workers=4 (winner from Step 2)
  - 5 epochs

Per-epoch: dual-stream evaluation (overall, field, lab F1; per-class).

Writes:
  scripts/model3_training/logs/step3_mixed_baseline.json
  scripts/model3_training/checkpoints/step3_baseline_epoch5.pt
"""
from __future__ import annotations

import sys
sys.stdout.reconfigure(line_buffering=True)

import json
import time
import math
import random
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, accuracy_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from app.config_model3 import (
    CLASS_NAMES, CLASS_TO_IDX, NUM_CLASSES, CROP_FROM_IDX,
    IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD,
)

# ── Constants ──────────────────────────────────────────────────────────────
BACKBONE      = 'vit_small_patch14_reg4_dinov2'
PROBE_IMG_SIZE = 224
FEAT_DIM      = 768
SE_REDUCTION  = 16
N_FILM_MODES  = 3
N_EPOCHS      = 5
BATCH_SIZE    = 32
NUM_WORKERS   = 4
LR_HEADS      = 1e-3
WEIGHT_DECAY  = 1e-4
SEED          = 42
ENS_BETA      = 0.999
MAX_WEIGHT_RATIO = 5.0
ANTHRACNOSE_BOOST = 1.5
FIELD_OVERSAMPLE = 2.0    # field examples sampled 2x relative to lab

CSV_PATH    = ROOT / 'data' / 'specialist' / 'model3' / 'model3_unified_source_map.csv'
SPLIT_PATH  = ROOT / 'data' / 'specialist' / 'model3' / 'split_indices.json'
LOG_PATH    = ROOT / 'scripts' / 'model3_training' / 'logs' / 'step3_mixed_baseline.json'
CKPT_PATH   = ROOT / 'scripts' / 'model3_training' / 'checkpoints' / 'step3_baseline_epoch5.pt'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Reproducibility ────────────────────────────────────────────────────────
def set_seeds(seed: int = SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# ── Preprocessing ──────────────────────────────────────────────────────────
def apply_lab_clahe(image_bgr: np.ndarray, clip_limit: float = 2.0,
                    tile_grid_size=(8, 8)) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ── Dataset ────────────────────────────────────────────────────────────────
class M3Dataset(Dataset):
    """Returns (img tensor, label, crop_mode, is_field).

    crop_mode is the GROUND-TRUTH crop (0=tomato, 1=chilli) for training.
    Validation uses the same — we measure ceiling, not deployment uncertainty.
    """
    def __init__(self, df: pd.DataFrame, augment: bool = False):
        self.df = df.reset_index(drop=True)
        self.augment = augment
        self.mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
        self.std  = np.array(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)

    def __len__(self):
        return len(self.df)

    def _load(self, row):
        img = None
        clahe = row.get('clahe_path')
        if isinstance(clahe, str) and clahe:
            full = ROOT / clahe
            if full.exists():
                img = cv2.imread(str(full), cv2.IMREAD_COLOR)
        if img is None:
            full = ROOT / row['image_path']
            img = cv2.imread(str(full), cv2.IMREAD_COLOR)
            if img is None:
                return np.zeros((PROBE_IMG_SIZE, PROBE_IMG_SIZE, 3), dtype=np.uint8)
            img = apply_lab_clahe(img)
        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = self._load(row)
        # Light augmentation in training only (matches Stage 1 spec for non-anthracnose)
        if self.augment:
            if random.random() < 0.5:
                img = cv2.flip(img, 1)
            if random.random() < 0.3:
                img = cv2.flip(img, 0)
            # Random rotation 0/90/180/270
            k = random.randint(0, 3)
            if k:
                img = np.rot90(img, k=k).copy()
        img = cv2.resize(img, (PROBE_IMG_SIZE, PROBE_IMG_SIZE))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = (rgb.transpose(2, 0, 1) - self.mean) / self.std
        x = torch.from_numpy(x).float()
        cls = row['class_name']
        label = CLASS_TO_IDX.get(cls, 0)
        crop_mode = CROP_FROM_IDX.get(label, 0)
        is_field = bool(row['is_field_photo'])
        return x, label, int(crop_mode), is_field


# ── Architecture ───────────────────────────────────────────────────────────
class SEBlock(nn.Module):
    def __init__(self, channels: int = FEAT_DIM, reduction: int = SE_REDUCTION):
        super().__init__()
        bottleneck = channels // reduction
        self.fc1 = nn.Linear(channels, bottleneck, bias=False)
        self.fc2 = nn.Linear(bottleneck, channels, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.fc2.weight, 0.01)  # near-identity init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.sigmoid(self.fc2(self.relu(self.fc1(x))))
        return x * scale


class HardFiLM(nn.Module):
    TOMATO = 0
    CHILLI = 1
    UNCERTAIN = 2

    def __init__(self, feature_dim: int = FEAT_DIM, n_modes: int = N_FILM_MODES):
        super().__init__()
        self.gamma = nn.Embedding(n_modes, feature_dim)
        self.beta  = nn.Embedding(n_modes, feature_dim)
        nn.init.normal_(self.gamma.weight, mean=1.0, std=0.01)
        nn.init.zeros_(self.beta.weight)
        with torch.no_grad():
            self.gamma.weight[2] = torch.ones(feature_dim)
            self.beta.weight[2]  = torch.zeros(feature_dim)

    def forward(self, x: torch.Tensor, crop_mode: torch.Tensor) -> torch.Tensor:
        return self.gamma(crop_mode) * x + self.beta(crop_mode)


class Model3Baseline(nn.Module):
    """Frozen DINOv2 + SE + HardFiLM + LayerNorm + Linear. No LoRA, no MixStyle,
    no SupCon. The simplest version that supports the mixed-baseline test."""
    def __init__(self, n_classes: int = NUM_CLASSES):
        super().__init__()
        self.backbone = timm.create_model(
            BACKBONE, pretrained=True, num_classes=0, img_size=PROBE_IMG_SIZE
        )
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()  # keep BN/dropout in eval forever
        self.se = SEBlock(FEAT_DIM, SE_REDUCTION)
        self.film = HardFiLM(FEAT_DIM, N_FILM_MODES)
        self.head = nn.Sequential(
            nn.LayerNorm(FEAT_DIM),
            nn.Linear(FEAT_DIM, n_classes),
        )
        nn.init.xavier_uniform_(self.head[1].weight)
        nn.init.zeros_(self.head[1].bias)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            f = self.backbone.forward_features(x)  # [B, 261, 384]
        cls = f[:, 0, :]
        non_cls = f[:, 1:, :]
        return torch.cat([cls, non_cls.mean(dim=1)], dim=-1)  # [B, 768]

    def forward(self, x: torch.Tensor, crop_mode: torch.Tensor) -> torch.Tensor:
        feats = self.extract_features(x)
        feats = self.se(feats)
        feats = self.film(feats, crop_mode)
        return self.head(feats)


# ── ENS class weights ──────────────────────────────────────────────────────
def compute_ens_weights(class_counts: dict, beta: float = ENS_BETA) -> dict:
    eff_n = {c: (1 - beta**n) / (1 - beta) for c, n in class_counts.items()}
    max_eff = max(eff_n.values())
    w = {c: max_eff / e for c, e in eff_n.items()}
    max_w = max(w.values())
    floor = max_w / MAX_WEIGHT_RATIO
    w = {c: max(v, floor) for c, v in w.items()}  # cap ratio at 5:1
    if 'chilli_anthracnose' in w:
        w['chilli_anthracnose'] *= ANTHRACNOSE_BOOST
    return w


# ── Mixed sampler (field oversampled) ──────────────────────────────────────
def build_mixed_sampler(df: pd.DataFrame, field_oversample: float = FIELD_OVERSAMPLE
                        ) -> WeightedRandomSampler:
    """Per-sample weights = field_oversample if is_field else 1.0.
    Stratifies the batch toward more field examples without removing any lab data."""
    weights = np.where(df['is_field_photo'].values, field_oversample, 1.0).astype(np.float64)
    return WeightedRandomSampler(weights=weights, num_samples=len(df), replacement=True)


# ── Dual-stream eval ───────────────────────────────────────────────────────
@torch.no_grad()
def dual_stream_eval(model: nn.Module, dl: DataLoader) -> dict:
    model.eval()
    all_pred, all_lab, all_isf = [], [], []
    for x, y, c, isf in dl:
        x = x.to(DEVICE, non_blocking=True)
        c = c.to(DEVICE, non_blocking=True)
        logits = model(x, c)
        all_pred.append(logits.argmax(dim=1).cpu())
        all_lab.append(y)
        all_isf.append(isf if torch.is_tensor(isf) else torch.tensor(isf))
    preds = torch.cat(all_pred).numpy()
    labs  = torch.cat(all_lab).numpy()
    isf   = torch.cat(all_isf).numpy().astype(bool)

    overall_acc = float(accuracy_score(labs, preds))
    overall_f1  = float(f1_score(labs, preds, average='macro', zero_division=0,
                                 labels=list(range(NUM_CLASSES))))
    field_f1 = float(f1_score(labs[isf], preds[isf], average='macro', zero_division=0,
                              labels=list(range(NUM_CLASSES)))) if isf.any() else 0.0
    lab_f1   = float(f1_score(labs[~isf], preds[~isf], average='macro', zero_division=0,
                              labels=list(range(NUM_CLASSES)))) if (~isf).any() else 0.0
    pcf1 = f1_score(labs, preds, average=None, zero_division=0,
                    labels=list(range(NUM_CLASSES)))
    fpcf1 = f1_score(labs[isf], preds[isf], average=None, zero_division=0,
                     labels=list(range(NUM_CLASSES))) if isf.any() else np.zeros(NUM_CLASSES)
    lpcf1 = f1_score(labs[~isf], preds[~isf], average=None, zero_division=0,
                     labels=list(range(NUM_CLASSES))) if (~isf).any() else np.zeros(NUM_CLASSES)
    return {
        'overall_acc': overall_acc,
        'overall_f1': overall_f1,
        'field_f1': field_f1,
        'lab_f1': lab_f1,
        'per_class_overall': {CLASS_NAMES[i]: float(pcf1[i]) for i in range(NUM_CLASSES)},
        'per_class_field':   {CLASS_NAMES[i]: float(fpcf1[i]) for i in range(NUM_CLASSES)},
        'per_class_lab':     {CLASS_NAMES[i]: float(lpcf1[i]) for i in range(NUM_CLASSES)},
    }


def main():
    set_seeds(SEED)
    print("=" * 72)
    print("STEP 3 — Mixed-domain baseline (Stage 0.5 sanity check)")
    print("=" * 72)
    print(f"DEVICE: {DEVICE}")
    print(f"BACKBONE: {BACKBONE}, img_size={PROBE_IMG_SIZE}, FROZEN")
    print(f"BATCH_SIZE: {BATCH_SIZE}, NUM_WORKERS: {NUM_WORKERS}, EPOCHS: {N_EPOCHS}")
    print(f"FIELD_OVERSAMPLE: {FIELD_OVERSAMPLE}x")

    # ── Data ───────────────────────────────────────────────────────────────
    df = pd.read_csv(CSV_PATH)
    with open(SPLIT_PATH) as f:
        splits = json.load(f)
    train_df = df.iloc[splits['train']].reset_index(drop=True)
    val_df   = df.iloc[splits['val']].reset_index(drop=True)
    print(f"\nTrain: {len(train_df)}  Val: {len(val_df)}")

    train_field_pct = train_df['is_field_photo'].mean()
    val_field_pct   = val_df['is_field_photo'].mean()
    print(f"  train field share: {train_field_pct:.1%}")
    print(f"  val field share:   {val_field_pct:.1%}")

    # Effective field share with 2x oversampling = 2*F / (2*F + L)
    F_share = train_field_pct
    eff_field = (FIELD_OVERSAMPLE * F_share) / (FIELD_OVERSAMPLE * F_share + (1 - F_share))
    print(f"  expected per-batch field share with {FIELD_OVERSAMPLE}x oversample: {eff_field:.1%}")

    cls_counts = Counter(train_df['class_name'])
    ens_w = compute_ens_weights(dict(cls_counts))
    print("\nENS class weights:")
    for c in CLASS_NAMES:
        print(f"  {c:<35} count={cls_counts[c]:5d}  weight={ens_w[c]:.3f}")
    weight_tensor = torch.tensor([ens_w[c] for c in CLASS_NAMES], dtype=torch.float32, device=DEVICE)

    # ── Loaders ────────────────────────────────────────────────────────────
    train_ds = M3Dataset(train_df, augment=True)
    val_ds   = M3Dataset(val_df, augment=False)
    sampler  = build_mixed_sampler(train_df, FIELD_OVERSAMPLE)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=NUM_WORKERS, pin_memory=True,
                          persistent_workers=True, prefetch_factor=2)
    val_dl   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True,
                          persistent_workers=True, prefetch_factor=2)

    # ── Model & optimizer ──────────────────────────────────────────────────
    print("\nBuilding Model3Baseline (frozen backbone + SE + 3-mode FiLM + LN + Linear)...")
    model = Model3Baseline(n_classes=NUM_CLASSES).to(DEVICE)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  trainable: {n_train:,} / total: {n_total:,}")

    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and not n.startswith('backbone.')]
    optim = torch.optim.AdamW(head_params, lr=LR_HEADS, weight_decay=WEIGHT_DECAY)

    criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=0.1)

    # ── Training loop ──────────────────────────────────────────────────────
    epoch_logs = []
    best_field_f1 = 0.0
    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        # Backbone stays in eval forever (frozen)
        model.backbone.eval()
        t_epoch = time.time()
        total_loss = 0.0
        n_batches = 0
        n_imgs = 0
        for batch_idx, (x, y, c, isf) in enumerate(train_dl):
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            c = c.to(DEVICE, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            logits = model(x, c)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head_params, 1.0)
            optim.step()
            total_loss += loss.item()
            n_batches += 1
            n_imgs += x.size(0)
            if batch_idx % 50 == 0:
                rate = n_imgs / (time.time() - t_epoch + 1e-6)
                print(f"  epoch {epoch} batch {batch_idx}/{len(train_dl)}  "
                      f"loss={loss.item():.4f}  ({rate:.0f} img/s)", flush=True)

        train_loss = total_loss / max(n_batches, 1)
        train_secs = time.time() - t_epoch

        # Validation
        t_val = time.time()
        eval_out = dual_stream_eval(model, val_dl)
        val_secs = time.time() - t_val

        epoch_logs.append({
            'epoch': epoch,
            'train_loss': round(train_loss, 4),
            'train_seconds': round(train_secs, 1),
            'val_seconds': round(val_secs, 1),
            **{k: v for k, v in eval_out.items()},
        })
        print(f"\n[epoch {epoch}/{N_EPOCHS}]  train_loss={train_loss:.4f}  "
              f"({train_secs:.0f}s train, {val_secs:.0f}s val)")
        print(f"  overall_f1={eval_out['overall_f1']:.4f}  "
              f"field_f1={eval_out['field_f1']:.4f}  lab_f1={eval_out['lab_f1']:.4f}")
        print(f"  per-class field F1:")
        for c in CLASS_NAMES:
            f = eval_out['per_class_field'][c]
            print(f"    {c:<35} {f:.4f}")

        if eval_out['field_f1'] > best_field_f1:
            best_field_f1 = eval_out['field_f1']
            CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_field_f1': best_field_f1,
                'overall_f1': eval_out['overall_f1'],
                'lab_f1': eval_out['lab_f1'],
                'config': {
                    'backbone': BACKBONE,
                    'batch_size': BATCH_SIZE,
                    'num_workers': NUM_WORKERS,
                    'lr_heads': LR_HEADS,
                    'field_oversample': FIELD_OVERSAMPLE,
                    'ens_weights': ens_w,
                },
            }, CKPT_PATH)

    # ── Save log ───────────────────────────────────────────────────────────
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, 'w') as f:
        json.dump({
            'config': {
                'backbone': BACKBONE,
                'img_size': PROBE_IMG_SIZE,
                'feature_dim': FEAT_DIM,
                'batch_size': BATCH_SIZE,
                'num_workers': NUM_WORKERS,
                'epochs': N_EPOCHS,
                'lr_heads': LR_HEADS,
                'weight_decay': WEIGHT_DECAY,
                'field_oversample': FIELD_OVERSAMPLE,
                'ens_beta': ENS_BETA,
                'anthracnose_boost': ANTHRACNOSE_BOOST,
                'expected_per_batch_field_share': round(eff_field, 4),
                'seed': SEED,
            },
            'class_counts_train': dict(cls_counts),
            'ens_weights': ens_w,
            'epoch_logs': epoch_logs,
            'best_field_f1': best_field_f1,
            'final_field_f1': epoch_logs[-1]['field_f1'] if epoch_logs else None,
            'final_overall_f1': epoch_logs[-1]['overall_f1'] if epoch_logs else None,
            'final_lab_f1': epoch_logs[-1]['lab_f1'] if epoch_logs else None,
            'checkpoint': str(CKPT_PATH.relative_to(ROOT)),
        }, f, indent=2)
    print(f"\nSaved log: {LOG_PATH.relative_to(ROOT)}")
    print(f"Best field_val_f1 over 5 epochs: {best_field_f1:.4f}")


if __name__ == '__main__':
    main()
