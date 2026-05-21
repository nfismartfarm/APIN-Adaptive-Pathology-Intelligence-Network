"""
DINOv2 Linear Probe Training — Train and evaluate linear head on frozen features.

Uses sklearn LogisticRegression (not PyTorch) because:
  1. Exact solution via LBFGS, no training instability
  2. Built-in cross-validation, no need for custom training loop
  3. CPU-only, no GPU needed (features already extracted)
  4. Reproducible: same seed = same result every time

Research-informed choices:
  - L2 normalization (unit sphere) before regression per DINOv2 paper protocol
  - Grid search over C values with stratified 5-fold CV
  - Primary metric: macro F1 (handles class imbalance correctly)
  - Secondary: min(per-class F1) — optimizes the weakest class
"""

import os
import sys
import json
import time
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix,
    precision_recall_curve, accuracy_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.dinov2_probe.config import (
    CLASS_NAMES, CLASS_TO_IDX, IDX_TO_CLASS, NUM_CLASSES,
    FAILURE_CLASSES, THIN_CLASSES, MODEL2_VAL_F1,
    C_VALUES, SOLVER, MAX_ITER, SCALER_TYPES, CV_FOLDS,
    FIELD_PHOTO_MIN_SAMPLES, SOURCE_MIN_SAMPLES,
    SPLIT_INDICES, FEATURES_CACHE_PATH, RESULTS_DIR,
    FEATURE_AGGREGATION, FEATURE_DIM, RANDOM_SEED,
)
from scripts.dinov2_probe.feature_cache import load_cache


# ═══════════════════════════════════════════════════════════════════════
# DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════

def prepare_data(cache: Dict) -> Dict[str, Dict]:
    """
    Split cached features into train/val using split_indices.json.

    Returns dict with 'train' and 'val' keys, each containing:
      X: np.array (n_samples, feature_dim)
      y: np.array (n_samples,) int class labels
      is_field: np.array (n_samples,) bool
      sources: list of source_dataset strings
      paths: list of image_path strings
    """
    with open(SPLIT_INDICES) as f:
        splits = json.load(f)

    # Map CSV row indices to image paths
    import pandas as pd
    from scripts.dinov2_probe.config import MODEL2_CSV
    df = pd.read_csv(MODEL2_CSV)

    result = {}
    for split_name, split_key in [('train', 'train'), ('val', 'val_and_soup'),
                                    ('final_val', 'final_val')]:
        indices = splits.get(split_key, [])
        split_df = df.iloc[indices]

        X_list, y_list, field_list, source_list, path_list = [], [], [], [], []

        for _, row in split_df.iterrows():
            img_path = str(row['image_path'])
            if img_path not in cache:
                continue
            entry = cache[img_path]
            X_list.append(entry['feature'])
            y_list.append(entry['label'])
            field_list.append(entry['is_field_photo'])
            source_list.append(entry['source_dataset'])
            path_list.append(img_path)

        if X_list:
            result[split_name] = {
                'X': np.array(X_list, dtype=np.float32),
                'y': np.array(y_list, dtype=np.int64),
                'is_field': np.array(field_list, dtype=bool),
                'sources': source_list,
                'paths': path_list,
            }
            logger.info(f"Split '{split_name}': {len(X_list)} samples, "
                       f"{sum(field_list)} field photos")

    return result


def normalize_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    scaler_type: str,
) -> Tuple[np.ndarray, np.ndarray, object]:
    """
    Normalize features. Fit ONLY on training data.

    Returns (X_train_norm, X_val_norm, fitted_scaler_or_None)
    """
    if scaler_type == 'standard':
        scaler = StandardScaler()
        X_train_n = scaler.fit_transform(X_train)
        X_val_n = scaler.transform(X_val)
        return X_train_n, X_val_n, scaler
    elif scaler_type == 'l2_norm':
        # L2 normalize each sample to unit sphere (DINOv2 paper protocol)
        X_train_n = normalize(X_train, norm='l2', axis=1)
        X_val_n = normalize(X_val, norm='l2', axis=1)
        return X_train_n, X_val_n, None
    elif scaler_type == 'none':
        return X_train.copy(), X_val.copy(), None
    else:
        raise ValueError(f"Unknown scaler_type: {scaler_type}")


# ═══════════════════════════════════════════════════════════════════════
# GRID SEARCH
# ═══════════════════════════════════════════════════════════════════════

def grid_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> List[Dict]:
    """
    Grid search over C values and normalization strategies.
    Uses stratified k-fold CV on training data ONLY.

    Returns list of result dicts sorted by best score.
    """
    results = []

    for scaler_type in SCALER_TYPES:
        for C in C_VALUES:
            # Stratified k-fold CV
            # [FIX Issue 2] Fit scaler INSIDE each fold to prevent leakage
            skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                                   random_state=RANDOM_SEED)
            fold_f1s = []
            fold_min_f1s = []

            for train_idx, val_idx in skf.split(X_train, y_train):
                # Normalize within fold (no leakage)
                if scaler_type == 'standard':
                    fold_scaler = StandardScaler()
                    X_fold_train = fold_scaler.fit_transform(X_train[train_idx])
                    X_fold_val = fold_scaler.transform(X_train[val_idx])
                elif scaler_type == 'l2_norm':
                    X_fold_train = normalize(X_train[train_idx], norm='l2', axis=1)
                    X_fold_val = normalize(X_train[val_idx], norm='l2', axis=1)
                else:
                    X_fold_train = X_train[train_idx]
                    X_fold_val = X_train[val_idx]

                clf_fold = LogisticRegression(
                    C=C, solver=SOLVER, max_iter=MAX_ITER,
                    random_state=RANDOM_SEED,
                )
                clf_fold.fit(X_fold_train, y_train[train_idx])
                y_pred = clf_fold.predict(X_fold_val)

                macro_f1 = f1_score(y_train[val_idx], y_pred, average='macro',
                                   labels=list(range(NUM_CLASSES)), zero_division=0)
                per_class = f1_score(y_train[val_idx], y_pred, average=None,
                                    labels=list(range(NUM_CLASSES)), zero_division=0)
                min_f1 = float(per_class.min())

                fold_f1s.append(macro_f1)
                fold_min_f1s.append(min_f1)

            result = {
                'C': C,
                'scaler_type': scaler_type,
                'cv_macro_f1': float(np.mean(fold_f1s)),
                'cv_macro_f1_std': float(np.std(fold_f1s)),
                'cv_min_class_f1': float(np.mean(fold_min_f1s)),
                'cv_min_class_f1_std': float(np.std(fold_min_f1s)),
            }
            results.append(result)
            logger.info(f"  C={C:.4f} scaler={scaler_type}: "
                       f"macro_f1={result['cv_macro_f1']:.4f} "
                       f"min_f1={result['cv_min_class_f1']:.4f}")

    # Sort by cv_macro_f1 descending
    results.sort(key=lambda x: x['cv_macro_f1'], reverse=True)
    return results


# ═══════════════════════════════════════════════════════════════════════
# TRAIN AND EVALUATE
# ═══════════════════════════════════════════════════════════════════════

def train_and_evaluate(data: Dict) -> Dict:
    """
    Full training pipeline: grid search -> train best -> evaluate on val.

    Returns comprehensive results dict.
    """
    train_data = data['train']
    val_data = data['val']

    X_train, y_train = train_data['X'], train_data['y']
    X_val, y_val = val_data['X'], val_data['y']

    print(f"\nTraining linear probe...", flush=True)
    print(f"  Train: {len(X_train)} samples, {X_train.shape[1]}d features", flush=True)
    print(f"  Val: {len(X_val)} samples", flush=True)

    # ── Grid search ────────────────────────────────────────────────
    print(f"\nGrid search ({len(C_VALUES)} C values x {len(SCALER_TYPES)} scalers):",
          flush=True)
    t0 = time.time()
    gs_results = grid_search(X_train, y_train)
    print(f"Grid search done in {time.time()-t0:.0f}s", flush=True)

    best = gs_results[0]
    print(f"\nBest: C={best['C']}, scaler={best['scaler_type']}, "
          f"CV macro F1={best['cv_macro_f1']:.4f}", flush=True)

    # ── Train final model on full training set ──────────────────────
    X_train_n, X_val_n, scaler = normalize_features(
        X_train, X_val, best['scaler_type'])

    clf = LogisticRegression(
        C=best['C'], solver=SOLVER, max_iter=MAX_ITER,
        random_state=RANDOM_SEED,
    )
    clf.fit(X_train_n, y_train)

    # ── Evaluate on val set ─────────────────────────────────────────
    y_pred = clf.predict(X_val_n)
    y_proba = clf.predict_proba(X_val_n)

    # Overall metrics
    val_macro_f1 = float(f1_score(y_val, y_pred, average='macro',
                                   labels=list(range(NUM_CLASSES)), zero_division=0))
    val_weighted_f1 = float(f1_score(y_val, y_pred, average='weighted',
                                      labels=list(range(NUM_CLASSES)), zero_division=0))
    val_accuracy = float(accuracy_score(y_val, y_pred))

    per_class_f1 = f1_score(y_val, y_pred, average=None,
                            labels=list(range(NUM_CLASSES)), zero_division=0)

    # Per-class comparison with Model 2
    print(f"\n{'='*70}", flush=True)
    print(f"RESULTS — DINOv2 Linear Probe vs Model 2", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"Val macro F1:    {val_macro_f1:.4f} (Model 2: {MODEL2_VAL_F1['macro']:.4f}, "
          f"delta: {val_macro_f1 - MODEL2_VAL_F1['macro']:+.4f})", flush=True)
    print(f"Val weighted F1: {val_weighted_f1:.4f}", flush=True)
    print(f"Val accuracy:    {val_accuracy:.4f}", flush=True)

    print(f"\n{'Class':<30s} {'Probe F1':>10s} {'Model2 F1':>10s} {'Delta':>8s} {'N':>5s}",
          flush=True)
    print("-" * 70, flush=True)
    class_results = {}
    for i, cls in enumerate(CLASS_NAMES):
        probe_f1 = float(per_class_f1[i])
        model2_f1 = MODEL2_VAL_F1.get(cls, 0)
        delta = probe_f1 - model2_f1
        n_val = int((y_val == i).sum())
        winner = "PROBE" if delta > 0.01 else "MODEL2" if delta < -0.01 else "TIE"
        print(f"{cls:<30s} {probe_f1:>10.4f} {model2_f1:>10.4f} {delta:>+8.4f} {n_val:>5d}  [{winner}]",
              flush=True)
        class_results[cls] = {
            'probe_f1': probe_f1, 'model2_f1': model2_f1,
            'delta': delta, 'n_val': n_val, 'winner': winner,
        }

    # ── Train vs val gap ────────────────────────────────────────────
    y_train_pred = clf.predict(X_train_n)
    train_f1 = float(f1_score(y_train, y_train_pred, average='macro',
                               labels=list(range(NUM_CLASSES)), zero_division=0))
    gap = train_f1 - val_macro_f1
    print(f"\nTrain macro F1: {train_f1:.4f}, Val macro F1: {val_macro_f1:.4f}, "
          f"Gap: {gap:.4f}", flush=True)
    if gap > 0.15:
        print("WARNING: Large train-val gap suggests overfitting!", flush=True)

    # ── Field-photo-only metrics ────────────────────────────────────
    print(f"\n{'='*70}", flush=True)
    print(f"FIELD-PHOTO-ONLY ANALYSIS (failure classes)", flush=True)
    print(f"{'='*70}", flush=True)

    field_results = {}
    for cls in FAILURE_CLASSES:
        cls_idx = CLASS_TO_IDX[cls]
        cls_mask = y_val == cls_idx
        field_mask = val_data['is_field'] & cls_mask
        n_field = int(field_mask.sum())

        if n_field < FIELD_PHOTO_MIN_SAMPLES:
            print(f"\n{cls}: Only {n_field} field val photos. "
                  f"Field-only metric UNRELIABLE (min: {FIELD_PHOTO_MIN_SAMPLES}).",
                  flush=True)
            field_results[cls] = {'n_field': n_field, 'field_acc': None,
                                   'note': 'insufficient samples'}
        else:
            field_correct = (y_pred[field_mask] == cls_idx).sum()
            field_acc = float(field_correct / n_field)
            # Confidence on field photos
            field_conf = y_proba[field_mask, cls_idx]
            print(f"\n{cls} (n_field={n_field}):", flush=True)
            print(f"  Field accuracy: {field_acc:.4f} ({field_correct}/{n_field})", flush=True)
            print(f"  Field confidence: mean={field_conf.mean():.4f}, "
                  f"std={field_conf.std():.4f}", flush=True)
            field_results[cls] = {
                'n_field': n_field, 'field_acc': field_acc,
                'field_conf_mean': float(field_conf.mean()),
                'field_conf_std': float(field_conf.std()),
            }

    # ── Per-source breakdown ────────────────────────────────────────
    print(f"\n{'='*70}", flush=True)
    print(f"PER-SOURCE BREAKDOWN (failure classes)", flush=True)
    print(f"{'='*70}", flush=True)

    source_results = {}
    for cls in FAILURE_CLASSES:
        cls_idx = CLASS_TO_IDX[cls]
        cls_mask = y_val == cls_idx
        cls_sources = np.array(val_data['sources'])[cls_mask]
        cls_preds = y_pred[cls_mask]

        unique_sources = set(cls_sources)
        print(f"\n{cls}:", flush=True)
        print(f"  {'Source':<35s} {'N':>5s} {'Acc':>8s} {'Dominant?':>10s}", flush=True)
        print(f"  {'-'*60}", flush=True)

        source_breakdown = []
        for src in sorted(unique_sources):
            src_mask = cls_sources == src
            n_src = int(src_mask.sum())
            if n_src < SOURCE_MIN_SAMPLES:
                continue
            src_correct = (cls_preds[src_mask] == cls_idx).sum()
            src_acc = float(src_correct / n_src)
            # Check if this is the dominant training source
            is_dominant = src == 'original_pool'
            source_breakdown.append({
                'source': src, 'n': n_src, 'acc': src_acc,
                'is_dominant': is_dominant,
            })

        # Sort by accuracy ascending (worst first)
        source_breakdown.sort(key=lambda x: x['acc'])
        for sb in source_breakdown:
            dom_str = "YES" if sb['is_dominant'] else "no"
            print(f"  {sb['source']:<35s} {sb['n']:>5d} {sb['acc']:>8.4f} {dom_str:>10s}",
                  flush=True)

        source_results[cls] = source_breakdown

    # ── Confusion matrix ────────────────────────────────────────────
    cm = confusion_matrix(y_val, y_pred, labels=list(range(NUM_CLASSES)))

    # Save confusion matrix plot
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=range(NUM_CLASSES), yticks=range(NUM_CLASSES),
           xticklabels=[c[:15] for c in CLASS_NAMES],
           yticklabels=[c[:15] for c in CLASS_NAMES],
           ylabel='True', xlabel='Predicted',
           title=f'DINOv2 Probe Confusion Matrix (macro F1={val_macro_f1:.4f})')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    # Add text annotations
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=7)
    plt.tight_layout()
    cm_path = RESULTS_DIR / f'probe_confusion_matrix_{ts}.png'
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nConfusion matrix saved: {cm_path}", flush=True)

    # ── Save all results ────────────────────────────────────────────
    all_results = {
        'timestamp': ts,
        'aggregation': FEATURE_AGGREGATION,
        'best_C': best['C'],
        'best_scaler': best['scaler_type'],
        'cv_macro_f1': best['cv_macro_f1'],
        'val_macro_f1': val_macro_f1,
        'val_weighted_f1': val_weighted_f1,
        'val_accuracy': val_accuracy,
        'train_macro_f1': train_f1,
        'train_val_gap': gap,
        'per_class': class_results,
        'field_analysis': field_results,
        'source_analysis': source_results,
        'grid_search': gs_results,
        'confusion_matrix': cm.tolist(),
    }

    results_path = RESULTS_DIR / f'probe_results_{ts}.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved: {results_path}", flush=True)

    # Save fitted model
    model_path = RESULTS_DIR / f'fitted_probe_{ts}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({'clf': clf, 'scaler': scaler, 'scaler_type': best['scaler_type'],
                     'C': best['C'], 'aggregation': FEATURE_AGGREGATION}, f)
    print(f"Fitted model saved: {model_path}", flush=True)

    return all_results


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def run_probe() -> Dict:
    """Run the complete probe training pipeline."""
    np.random.seed(RANDOM_SEED)

    # Load cache
    cache = load_cache(FEATURES_CACHE_PATH)
    if cache is None:
        raise FileNotFoundError(f"Feature cache not found at {FEATURES_CACHE_PATH}. "
                               f"Run feature_cache.py first.")

    # Prepare data splits
    data = prepare_data(cache)

    if 'train' not in data or 'val' not in data:
        raise ValueError("Missing train or val split in prepared data")

    # Train and evaluate
    results = train_and_evaluate(data)
    return results


if __name__ == '__main__':
    run_probe()
