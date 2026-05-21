"""
Step 1 / Diagnostic — evaluate the existing models/model3_specialist/model3_simple_best.pt
on the locked Model 3 val split.

Per spec Fix 3: if overall_val_f1 > 0.85, STOP and report — surprising result given
the BF16 bug that was active during training. Otherwise note the result and proceed.

Reads:
  models/model3_specialist/model3_simple_best.pt
  data/specialist/model3/model3_unified_source_map.csv
  data/specialist/model3/split_indices.json

Writes:
  scripts/model3_training/logs/step1_eval_existing.json
"""
from __future__ import annotations

import sys
sys.stdout.reconfigure(line_buffering=True)

import os
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import cv2
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from app.config_model3 import (
    CLASS_NAMES, CLASS_TO_IDX, NUM_CLASSES, CROP_FROM_IDX,
    DINOV2_BACKBONE, DINOV2_IMG_SIZE, IMG_SIZE,
    IMAGENET_MEAN, IMAGENET_STD,
)
from scripts.models import Model3DINOLoRA


CKPT_PATH = ROOT / 'models' / 'model3_specialist' / 'model3_simple_best.pt'
CSV_PATH  = ROOT / 'data' / 'specialist' / 'model3' / 'model3_unified_source_map.csv'
SPLIT_PATH = ROOT / 'data' / 'specialist' / 'model3' / 'split_indices.json'
OUT_PATH  = ROOT / 'scripts' / 'model3_training' / 'logs' / 'step1_eval_existing.json'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def apply_lab_clahe(image_bgr: np.ndarray, clip_limit: float = 2.0,
                    tile_grid_size=(8, 8)) -> np.ndarray:
    """Canonical LAB-CLAHE preprocessing (matches Model 2 / Signal 4)."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


class ValDataset(Dataset):
    """Reads images via clahe_path if present and on disk, else applies LAB-CLAHE
    on the fly. Returns ImageNet-normalized 224x224 RGB tensor + label + is_field
    + crop_id."""
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
        self.std  = np.array(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)

    def __len__(self):
        return len(self.df)

    def _load(self, row):
        clahe_path = row.get('clahe_path')
        if isinstance(clahe_path, str) and clahe_path:
            full = ROOT / clahe_path
            if full.exists():
                img = cv2.imread(str(full), cv2.IMREAD_COLOR)
                if img is not None:
                    return img  # already CLAHE'd at preprocessing time
        # Fallback — load original and apply CLAHE here
        full = ROOT / row['image_path']
        img = cv2.imread(str(full), cv2.IMREAD_COLOR)
        if img is None:
            return None
        return apply_lab_clahe(img)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = self._load(row)
        if img is None:
            # Black square fallback so the loop doesn't crash; counted as wrong prediction
            img = np.zeros((IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.uint8)
        img = cv2.resize(img, IMG_SIZE)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = (img_rgb.transpose(2, 0, 1) - self.mean) / self.std
        x = torch.from_numpy(x).float()

        cls = row['class_name']
        if cls not in CLASS_TO_IDX:
            # Unknown class -> mark with -1; we'll filter later
            label = -1
            crop = 0
        else:
            label = CLASS_TO_IDX[cls]
            crop = CROP_FROM_IDX[label]
        is_field = bool(row['is_field_photo'])
        return x, label, int(crop), is_field


def main():
    print("=" * 72)
    print("STEP 1 — Evaluate existing model3_simple_best.pt on val split")
    print("=" * 72)
    print(f"DEVICE: {DEVICE}")
    print(f"Checkpoint: {CKPT_PATH}")
    print(f"  size: {CKPT_PATH.stat().st_size / 1e6:.1f} MB")

    # ── Load split indices and val rows ────────────────────────────────────
    df = pd.read_csv(CSV_PATH)
    with open(SPLIT_PATH) as f:
        splits = json.load(f)
    val_idx = splits['val']
    val_df = df.iloc[val_idx].reset_index(drop=True)
    print(f"Val rows: {len(val_df)} (expected 3225)")
    field_count = int(val_df['is_field_photo'].sum())
    print(f"  field: {field_count}  lab: {len(val_df) - field_count}")
    cls_counts = val_df['class_name'].value_counts().to_dict()
    print(f"  class counts: {cls_counts}")

    # Filter unknown classes
    valid = val_df[val_df['class_name'].isin(CLASS_TO_IDX)].reset_index(drop=True)
    n_skipped = len(val_df) - len(valid)
    if n_skipped:
        print(f"  WARNING skipped {n_skipped} rows with classes not in CLASS_NAMES")

    # ── Build model and load checkpoint ────────────────────────────────────
    print("\nBuilding Model3DINOLoRA (existing architecture)...")
    print(f"  backbone={DINOV2_BACKBONE}, img_size={DINOV2_IMG_SIZE}")
    model = Model3DINOLoRA(num_classes=NUM_CLASSES, num_crops=2,
                           pretrained=False,
                           enable_gradient_checkpointing=False)
    model = model.to(DEVICE)

    print(f"\nLoading checkpoint state_dict ({CKPT_PATH.name})...")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    sd = ckpt.get('model_state_dict', ckpt)
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"  missing keys: {len(miss)}  unexpected keys: {len(unexp)}")
    if miss[:5]:
        print(f"  missing[:5]: {miss[:5]}")
    if unexp[:5]:
        print(f"  unexpected[:5]: {unexp[:5]}")
    # Try EMA if main state_dict didn't load cleanly enough
    ema_sd = ckpt.get('ema_state_dict')
    if ema_sd and len(unexp) > 50:
        print("  attempting EMA state_dict instead...")
        model.load_state_dict(ema_sd, strict=False)

    model.eval()
    print(f"  trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"  total params:     {sum(p.numel() for p in model.parameters()):,}")

    # Print available checkpoint metadata
    for k in ('epoch', 'best_val_f1', 'val_f1', 'field_val_f1', 'overall_val_f1',
              'macro_f1', 'config', 'training_args', 'metrics'):
        if k in ckpt:
            v = ckpt[k]
            if isinstance(v, dict):
                print(f"  ckpt[{k}]: <dict {len(v)} keys>")
            else:
                print(f"  ckpt[{k}]: {v}")

    # ── DataLoader and inference ───────────────────────────────────────────
    ds = ValDataset(valid)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    all_pred = []
    all_lab  = []
    all_crop = []
    all_isfield = []

    print(f"\nRunning inference on {len(ds)} val images...")
    t0 = time.time()
    with torch.no_grad():
        for batch_idx, (x, y, crop, isf) in enumerate(dl):
            x = x.to(DEVICE, non_blocking=True)
            crop_ids = crop.to(DEVICE, non_blocking=True)
            try:
                logits = model(x, crop_ids=crop_ids)
            except TypeError:
                logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            preds = logits.argmax(dim=1).cpu().numpy()
            all_pred.extend(preds.tolist())
            all_lab.extend(y.numpy().tolist())
            all_crop.extend(crop.numpy().tolist())
            all_isfield.extend(isf.numpy().tolist() if torch.is_tensor(isf) else list(isf))
            if batch_idx % 20 == 0:
                print(f"  batch {batch_idx}/{len(dl)}  ({time.time()-t0:.1f}s)")
    elapsed = time.time() - t0
    print(f"  done. {elapsed:.1f}s ({len(ds) / elapsed:.0f} img/s)")

    all_pred = np.array(all_pred)
    all_lab  = np.array(all_lab)
    all_isfield = np.array(all_isfield, dtype=bool)

    # ── Dual-stream metrics ────────────────────────────────────────────────
    overall_acc = float(accuracy_score(all_lab, all_pred))
    overall_f1  = float(f1_score(all_lab, all_pred, average='macro', zero_division=0,
                                 labels=list(range(NUM_CLASSES))))
    pcf1 = f1_score(all_lab, all_pred, average=None, zero_division=0,
                    labels=list(range(NUM_CLASSES)))
    per_class_overall = {CLASS_NAMES[i]: float(pcf1[i]) for i in range(NUM_CLASSES)}

    field_pred = all_pred[all_isfield]
    field_lab  = all_lab[all_isfield]
    field_f1 = float(f1_score(field_lab, field_pred, average='macro', zero_division=0,
                              labels=list(range(NUM_CLASSES)))) if len(field_pred) else 0.0
    field_pcf1 = f1_score(field_lab, field_pred, average=None, zero_division=0,
                          labels=list(range(NUM_CLASSES))) if len(field_pred) else np.zeros(NUM_CLASSES)
    per_class_field = {CLASS_NAMES[i]: float(field_pcf1[i]) for i in range(NUM_CLASSES)}

    lab_pred = all_pred[~all_isfield]
    lab_lab  = all_lab[~all_isfield]
    lab_f1 = float(f1_score(lab_lab, lab_pred, average='macro', zero_division=0,
                            labels=list(range(NUM_CLASSES)))) if len(lab_pred) else 0.0
    lab_pcf1 = f1_score(lab_lab, lab_pred, average=None, zero_division=0,
                        labels=list(range(NUM_CLASSES))) if len(lab_pred) else np.zeros(NUM_CLASSES)
    per_class_lab = {CLASS_NAMES[i]: float(lab_pcf1[i]) for i in range(NUM_CLASSES)}

    gap = {c: per_class_lab[c] - per_class_field[c] for c in CLASS_NAMES}

    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)
    print(f"Overall val accuracy: {overall_acc:.4f}")
    print(f"Overall val macro F1: {overall_f1:.4f}")
    print(f"Field val macro F1:   {field_f1:.4f}  (n={int(all_isfield.sum())})")
    print(f"Lab   val macro F1:   {lab_f1:.4f}  (n={int((~all_isfield).sum())})")
    print()
    print(f"{'Class':<35} {'overall':>8} {'field':>8} {'lab':>8} {'gap':>8}")
    print("-" * 72)
    for c in CLASS_NAMES:
        flag = ' GAP>0.20' if gap[c] > 0.20 else ('  THIN' if per_class_field[c] < 0.50 and per_class_field[c] > 0 else '')
        print(f"{c:<35} {per_class_overall[c]:>8.4f} {per_class_field[c]:>8.4f} "
              f"{per_class_lab[c]:>8.4f} {gap[c]:>+8.4f}{flag}")

    # ── Save ──────────────────────────────────────────────────────────────
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = {
        'checkpoint': str(CKPT_PATH.relative_to(ROOT)),
        'checkpoint_size_mb': round(CKPT_PATH.stat().st_size / 1e6, 2),
        'val_n': len(ds),
        'val_field_n': int(all_isfield.sum()),
        'val_lab_n': int((~all_isfield).sum()),
        'overall_acc': overall_acc,
        'overall_macro_f1': overall_f1,
        'field_macro_f1': field_f1,
        'lab_macro_f1': lab_f1,
        'per_class_overall_f1': per_class_overall,
        'per_class_field_f1': per_class_field,
        'per_class_lab_f1': per_class_lab,
        'per_class_lab_minus_field_gap': gap,
        'inference_seconds': round(elapsed, 1),
        'images_per_second': round(len(ds) / elapsed, 1),
        'verdict': ('SURPRISE >0.85 — investigate before discarding' if overall_f1 > 0.85
                    else ('moderate — usable as warm-start' if overall_f1 > 0.50
                          else 'BF16-corrupted — discard, retrain fresh')),
    }
    with open(OUT_PATH, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH.relative_to(ROOT)}")
    print(f"Verdict: {out['verdict']}")


if __name__ == '__main__':
    main()
