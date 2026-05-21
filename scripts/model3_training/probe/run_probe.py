"""
Step 4 / Probe — DINOv2-Small-Registers feature extraction + logistic-regression
probe for the 10-class tomato+chilli problem.

Backbone: vit_small_patch14_reg4_dinov2 (4 register tokens) — the spec's variant.
NOT vit_small_patch14_dinov2.lvd142m (the without-registers variant used by the
existing Model3DINOLoRA in scripts/models.py — that's a different backbone).

img_size=224 explicitly passed (timm default is 518 — would crash without this).

Features per image: 768-dim = concat(CLS_384, mean(non-CLS tokens)_384).
Per spec Fix 8: mean is taken over ALL non-CLS tokens (4 register + 256 patch tokens),
matching the Signal 4 nonlinear head convention.

LoRA decision (spec Part 3):
  field_val_f1 ≥ 0.78 → no LoRA
  0.65 ≤ field_val_f1 < 0.78 → LoRA rank=4
  field_val_f1 < 0.65 → LoRA rank=8

Confusable pairs: (a,b) where misclassification rate > 10% in either direction.
Used as the CutMix candidate set in Stage 2.

Writes:
  scripts/dinov2_probe/results/tomato_chilli_features_cache.pkl  (NEW — does NOT
    overwrite the existing okra+brassica cache, per spec Fix 6)
  scripts/model3_training/probe/probe_results.json
"""
from __future__ import annotations

import sys
sys.stdout.reconfigure(line_buffering=True)

import json
import pickle
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import timm
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from app.config_model3 import (
    CLASS_NAMES, CLASS_TO_IDX, NUM_CLASSES, CROP_FROM_IDX,
    IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD,
)

# ── Spec-mandated backbone (NOT the existing-model3 backbone) ─────────────
PROBE_BACKBONE = 'vit_small_patch14_reg4_dinov2'
PROBE_IMG_SIZE = 224
PROBE_FEAT_DIM = 768  # CLS(384) + mean-non-CLS(384)

CSV_PATH    = ROOT / 'data' / 'specialist' / 'model3' / 'model3_unified_source_map.csv'
SPLIT_PATH  = ROOT / 'data' / 'specialist' / 'model3' / 'split_indices.json'
CACHE_PATH  = ROOT / 'scripts' / 'dinov2_probe' / 'results' / 'tomato_chilli_features_cache.pkl'
RESULTS_PATH = ROOT / 'scripts' / 'model3_training' / 'probe' / 'probe_results.json'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64
NUM_WORKERS = 4   # winner from Step 2


def apply_lab_clahe(image_bgr: np.ndarray, clip_limit: float = 2.0,
                    tile_grid_size=(8, 8)) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


class ProbeDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
        self.std  = np.array(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Prefer pre-CLAHE'd image when available (Phase 0 work)
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
                img = np.zeros((PROBE_IMG_SIZE, PROBE_IMG_SIZE, 3), dtype=np.uint8)
            else:
                img = apply_lab_clahe(img)
        img = cv2.resize(img, (PROBE_IMG_SIZE, PROBE_IMG_SIZE))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = (rgb.transpose(2, 0, 1) - self.mean) / self.std
        x = torch.from_numpy(x).float()
        cls = row['class_name']
        label = CLASS_TO_IDX.get(cls, -1)
        is_field = bool(row['is_field_photo'])
        return x, label, is_field


def extract_features(backbone: torch.nn.Module, dl: DataLoader,
                     n_total: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward pass; returns (X[N,768], y[N], is_field[N]). Skips label==-1."""
    backbone.eval()
    X = np.zeros((n_total, PROBE_FEAT_DIM), dtype=np.float32)
    y = np.zeros(n_total, dtype=np.int64)
    isf = np.zeros(n_total, dtype=bool)

    cur = 0
    t0 = time.time()
    with torch.no_grad():
        for batch_idx, (x, lab, fld) in enumerate(dl):
            x = x.to(DEVICE, non_blocking=True)
            feats = backbone.forward_features(x)  # [B, 261, 384]
            cls = feats[:, 0, :]                  # [B, 384]
            non_cls = feats[:, 1:, :]              # [B, 260, 384]
            mean_non_cls = non_cls.mean(dim=1)     # [B, 384]
            combined = torch.cat([cls, mean_non_cls], dim=-1)  # [B, 768]
            n = combined.size(0)
            X[cur:cur+n] = combined.cpu().numpy()
            y[cur:cur+n] = lab.numpy()
            isf[cur:cur+n] = fld.numpy() if torch.is_tensor(fld) else np.array(fld)
            cur += n
            if batch_idx % 20 == 0:
                rate = cur / (time.time() - t0 + 1e-6)
                print(f"  batch {batch_idx}/{len(dl)}  ({cur}/{n_total})  {rate:.1f} img/s", flush=True)
    elapsed = time.time() - t0
    print(f"  done. {cur} features in {elapsed:.1f}s ({cur/elapsed:.1f} img/s)")

    # Filter out unknown labels
    mask = y >= 0
    if (~mask).any():
        print(f"  filtering {(~mask).sum()} rows with unknown labels")
    return X[mask], y[mask], isf[mask]


def lora_decision(field_val_f1: float) -> str:
    if field_val_f1 >= 0.78:
        return 'no_lora'
    elif field_val_f1 >= 0.65:
        return 'rank4'
    else:
        return 'rank8'


def find_confusable_pairs(cm: np.ndarray, threshold: float = 0.10) -> list:
    """Pairs (a,b) where row a -> col b misclassification rate > threshold,
    OR row b -> col a > threshold. Returns sorted list of [class_a, class_b, max_rate]."""
    pairs = {}
    for i in range(cm.shape[0]):
        row_total = cm[i].sum()
        if row_total == 0:
            continue
        for j in range(cm.shape[1]):
            if i == j:
                continue
            rate = cm[i, j] / row_total
            if rate > threshold:
                key = tuple(sorted([i, j]))
                pairs[key] = max(pairs.get(key, 0.0), rate)
    out = []
    for (i, j), rate in sorted(pairs.items(), key=lambda kv: -kv[1]):
        out.append([CLASS_NAMES[i], CLASS_NAMES[j], round(float(rate), 4)])
    return out


def main():
    print("=" * 72)
    print("STEP 4 — DINOv2-Small-Registers probe (tomato+chilli)")
    print("=" * 72)
    print(f"DEVICE: {DEVICE}")
    print(f"BACKBONE: {PROBE_BACKBONE}")
    print(f"img_size: {PROBE_IMG_SIZE}, feature_dim: {PROBE_FEAT_DIM}, num_workers: {NUM_WORKERS}")

    # ── Load data ──────────────────────────────────────────────────────────
    df = pd.read_csv(CSV_PATH)
    with open(SPLIT_PATH) as f:
        splits = json.load(f)
    train_df = df.iloc[splits['train']].reset_index(drop=True)
    val_df   = df.iloc[splits['val']].reset_index(drop=True)
    print(f"\nTrain: {len(train_df)}  Val: {len(val_df)}")
    print(f"  train field: {int(train_df['is_field_photo'].sum())}  "
          f"lab: {int((~train_df['is_field_photo']).sum())}")
    print(f"  val field:   {int(val_df['is_field_photo'].sum())}  "
          f"lab: {int((~val_df['is_field_photo']).sum())}")

    # ── Cache check ────────────────────────────────────────────────────────
    if CACHE_PATH.exists():
        print(f"\nFound existing cache: {CACHE_PATH.relative_to(ROOT)}")
        with open(CACHE_PATH, 'rb') as f:
            cache = pickle.load(f)
        if (cache.get('backbone') == PROBE_BACKBONE and
            cache.get('feature_dim') == PROBE_FEAT_DIM and
            cache.get('img_size') == PROBE_IMG_SIZE and
            len(cache['X_train']) == len(train_df) and
            len(cache['X_val']) == len(val_df)):
            print("  cache valid — skipping extraction")
            X_train, y_train, isf_train = cache['X_train'], cache['y_train'], cache['isf_train']
            X_val,   y_val,   isf_val   = cache['X_val'],   cache['y_val'],   cache['isf_val']
        else:
            print("  cache mismatch — re-extracting")
            cache = None
    else:
        cache = None

    if cache is None:
        # ── Build backbone ─────────────────────────────────────────────────
        print(f"\nBuilding backbone with pretrained weights + img_size={PROBE_IMG_SIZE}...")
        backbone = timm.create_model(
            PROBE_BACKBONE, pretrained=True, num_classes=0, img_size=PROBE_IMG_SIZE
        )
        backbone = backbone.to(DEVICE)
        for p in backbone.parameters():
            p.requires_grad = False
        n_params = sum(p.numel() for p in backbone.parameters())
        print(f"  backbone params: {n_params:,} (frozen)")

        # Sanity: forward shape
        x = torch.zeros(2, 3, PROBE_IMG_SIZE, PROBE_IMG_SIZE, device=DEVICE)
        with torch.no_grad():
            f = backbone.forward_features(x)
        assert f.shape[-1] == 384, f"Expected 384-dim per token, got {f.shape[-1]}"
        print(f"  forward_features shape: {tuple(f.shape)}  (expected (2, 261, 384))")

        # ── Extract train features ─────────────────────────────────────────
        print(f"\nExtracting TRAIN features ({len(train_df)} images)...")
        train_ds = ProbeDataset(train_df)
        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=True, prefetch_factor=2)
        X_train, y_train, isf_train = extract_features(backbone, train_dl, len(train_df))
        del train_dl, train_ds
        torch.cuda.empty_cache()

        # ── Extract val features ───────────────────────────────────────────
        print(f"\nExtracting VAL features ({len(val_df)} images)...")
        val_ds = ProbeDataset(val_df)
        val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True,
                            persistent_workers=True, prefetch_factor=2)
        X_val, y_val, isf_val = extract_features(backbone, val_dl, len(val_df))
        del val_dl, val_ds, backbone
        torch.cuda.empty_cache()

        # ── Save cache (NEW file, doesn't touch okra+brassica cache) ───────
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_PATH, 'wb') as f:
            pickle.dump({
                'backbone': PROBE_BACKBONE,
                'feature_dim': PROBE_FEAT_DIM,
                'img_size': PROBE_IMG_SIZE,
                'class_names': CLASS_NAMES,
                'X_train': X_train, 'y_train': y_train, 'isf_train': isf_train,
                'X_val': X_val, 'y_val': y_val, 'isf_val': isf_val,
            }, f)
        print(f"\nSaved cache: {CACHE_PATH.relative_to(ROOT)}  "
              f"({CACHE_PATH.stat().st_size/1e6:.1f} MB)")

    print(f"\nFeature shapes: X_train {X_train.shape}, X_val {X_val.shape}")

    # ── Standardise ────────────────────────────────────────────────────────
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(X_train)
    Xv_s  = scaler.transform(X_val)

    # ── Logistic regression probe ──────────────────────────────────────────
    print("\nFitting LogisticRegression(C=0.1, balanced, multinomial, lbfgs)...")
    t0 = time.time()
    probe = LogisticRegression(
        C=0.1, max_iter=2000, class_weight='balanced',
        solver='lbfgs', n_jobs=-1,
    )
    probe.fit(Xtr_s, y_train)
    print(f"  fit done in {time.time()-t0:.1f}s")

    yv_pred = probe.predict(Xv_s)

    overall_acc = float(accuracy_score(y_val, yv_pred))
    overall_f1  = float(f1_score(y_val, yv_pred, average='macro', zero_division=0,
                                 labels=list(range(NUM_CLASSES))))
    field_pred = yv_pred[isf_val]
    field_true = y_val[isf_val]
    lab_pred = yv_pred[~isf_val]
    lab_true = y_val[~isf_val]
    field_f1 = float(f1_score(field_true, field_pred, average='macro', zero_division=0,
                              labels=list(range(NUM_CLASSES)))) if len(field_pred) else 0.0
    lab_f1   = float(f1_score(lab_true, lab_pred, average='macro', zero_division=0,
                              labels=list(range(NUM_CLASSES)))) if len(lab_pred) else 0.0

    pcf1 = f1_score(y_val, yv_pred, average=None, zero_division=0,
                    labels=list(range(NUM_CLASSES)))
    field_pcf1 = f1_score(field_true, field_pred, average=None, zero_division=0,
                          labels=list(range(NUM_CLASSES))) if len(field_pred) else np.zeros(NUM_CLASSES)
    lab_pcf1 = f1_score(lab_true, lab_pred, average=None, zero_division=0,
                        labels=list(range(NUM_CLASSES))) if len(lab_pred) else np.zeros(NUM_CLASSES)

    per_class_overall = {CLASS_NAMES[i]: float(pcf1[i]) for i in range(NUM_CLASSES)}
    per_class_field = {CLASS_NAMES[i]: float(field_pcf1[i]) for i in range(NUM_CLASSES)}
    per_class_lab   = {CLASS_NAMES[i]: float(lab_pcf1[i]) for i in range(NUM_CLASSES)}
    gap = {c: per_class_lab[c] - per_class_field[c] for c in CLASS_NAMES}

    cm = confusion_matrix(y_val, yv_pred, labels=list(range(NUM_CLASSES)))
    confusable = find_confusable_pairs(cm, threshold=0.10)

    decision = lora_decision(field_f1)

    # ── Print ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("PROBE RESULTS")
    print("=" * 72)
    print(f"Overall val accuracy: {overall_acc:.4f}")
    print(f"Overall val macro F1: {overall_f1:.4f}")
    print(f"Field   val macro F1: {field_f1:.4f}  (n={int(isf_val.sum())})  <- LoRA decision driver")
    print(f"Lab     val macro F1: {lab_f1:.4f}  (n={int((~isf_val).sum())})")
    print()
    print(f"{'Class':<35} {'overall':>8} {'field':>8} {'lab':>8} {'gap':>8}")
    print("-" * 75)
    for c in CLASS_NAMES:
        flag = ' GAP>0.20' if gap[c] > 0.20 else ''
        print(f"{c:<35} {per_class_overall[c]:>8.4f} {per_class_field[c]:>8.4f} "
              f"{per_class_lab[c]:>8.4f} {gap[c]:>+8.4f}{flag}")

    print("\nConfusable pairs (>10% misclassification rate):")
    if confusable:
        for a, b, r in confusable:
            print(f"  {a} <-> {b}  rate={r:.4f}")
    else:
        print("  (none)")

    print(f"\nLoRA decision: {decision}")
    if decision == 'no_lora':
        print("  Frozen DINOv2 features already separate the classes well (field F1 ≥ 0.78).")
    elif decision == 'rank4':
        print("  Moderate field gap — LoRA rank=4 on qkv recommended.")
    else:
        print("  Significant field gap — LoRA rank=8 on qkv recommended.")

    # ── Save ───────────────────────────────────────────────────────────────
    out = {
        'backbone': PROBE_BACKBONE,
        'img_size': PROBE_IMG_SIZE,
        'feature_dim': PROBE_FEAT_DIM,
        'cache_path': str(CACHE_PATH.relative_to(ROOT)),
        'cache_size_mb': round(CACHE_PATH.stat().st_size / 1e6, 2),
        'n_train': int(len(X_train)),
        'n_val': int(len(X_val)),
        'n_val_field': int(isf_val.sum()),
        'n_val_lab': int((~isf_val).sum()),
        'overall_acc': overall_acc,
        'overall_macro_f1': overall_f1,
        'field_macro_f1': field_f1,
        'lab_macro_f1': lab_f1,
        'lora_decision': decision,
        'per_class_overall_f1': per_class_overall,
        'per_class_field_f1': per_class_field,
        'per_class_lab_f1': per_class_lab,
        'per_class_lab_minus_field_gap': gap,
        'confusable_pairs': confusable,
        'confusion_matrix_rows_truth_cols_pred': cm.tolist(),
        'class_names': CLASS_NAMES,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {RESULTS_PATH.relative_to(ROOT)}")


if __name__ == '__main__':
    main()
