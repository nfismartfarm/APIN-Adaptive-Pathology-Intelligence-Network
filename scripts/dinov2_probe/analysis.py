"""
DINOv2 Linear Probe Analysis — Deep analysis beyond basic metrics.

Answers the "why" not just the "what":
  - Source dominance impact on generalization
  - Feature space geometry (centroids, distances, variance)
  - Decision threshold analysis for failure classes
  - Comparison with EfficientNet baseline
"""

import os
import sys
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List
from datetime import datetime

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import precision_recall_curve, f1_score
from scipy.spatial.distance import cosine, cdist

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.dinov2_probe.config import (
    CLASS_NAMES, CLASS_TO_IDX, NUM_CLASSES,
    FAILURE_CLASSES, THIN_CLASSES, MODEL2_VAL_F1,
    RESULTS_DIR, RANDOM_SEED,
)


def run_analysis(data: Dict, probe_results: Dict) -> Dict:
    """
    Run all deep analyses on the extracted features and probe results.

    Args:
        data: dict with 'train' and 'val' splits (X, y, is_field, sources)
        probe_results: output from probe_train.run_probe()
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    analysis = {}

    val_X = data['val']['X']
    val_y = data['val']['y']
    train_X = data['train']['X']
    train_y = data['train']['y']
    train_sources = data['train']['sources']

    # ═══════════════════════════════════════════════════════════════
    # ANALYSIS 1: Source Dominance Impact
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}", flush=True)
    print("ANALYSIS 1: Source Dominance Impact", flush=True)
    print(f"{'='*60}", flush=True)

    source_analysis = {}
    for cls in FAILURE_CLASSES + ['brassica_alternaria', 'okra_healthy']:
        cls_idx = CLASS_TO_IDX.get(cls)
        if cls_idx is None:
            continue
        cls_mask = np.array(train_y) == cls_idx
        cls_sources = np.array(train_sources)[cls_mask]

        # Source distribution
        from collections import Counter
        src_counts = Counter(cls_sources)
        total = sum(src_counts.values())
        entropy = 0
        for count in src_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)

        print(f"\n{cls} (n={total}, entropy={entropy:.3f}):", flush=True)
        for src, cnt in src_counts.most_common():
            print(f"  {src:35s}: {cnt:5d} ({cnt/total*100:.1f}%)", flush=True)

        source_analysis[cls] = {
            'total': total, 'entropy': entropy,
            'sources': {src: cnt for src, cnt in src_counts.items()},
        }

    analysis['source_dominance'] = source_analysis

    # ═══════════════════════════════════════════════════════════════
    # ANALYSIS 2: Feature Space Geometry
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}", flush=True)
    print("ANALYSIS 2: Feature Space Geometry", flush=True)
    print(f"{'='*60}", flush=True)

    # Per-class centroids
    centroids = {}
    for cls_idx in range(NUM_CLASSES):
        mask = val_y == cls_idx
        if mask.any():
            centroids[cls_idx] = val_X[mask].mean(axis=0)

    # Pairwise cosine distances between centroids
    print(f"\nCosine distance matrix (val centroids):", flush=True)
    dist_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))
    print(f"{'':>20s}", end='', flush=True)
    for j in range(NUM_CLASSES):
        print(f" {CLASS_NAMES[j][:8]:>8s}", end='', flush=True)
    print(flush=True)

    for i in range(NUM_CLASSES):
        print(f"{CLASS_NAMES[i][:20]:>20s}", end='', flush=True)
        for j in range(NUM_CLASSES):
            if i in centroids and j in centroids:
                d = cosine(centroids[i], centroids[j])
                dist_matrix[i, j] = d
                print(f" {d:>8.4f}", end='', flush=True)
            else:
                print(f" {'N/A':>8s}", end='', flush=True)
        print(flush=True)

    # Key question: is black_rot closer to alternaria than to other brassica?
    br_idx = CLASS_TO_IDX['brassica_black_rot']
    alt_idx = CLASS_TO_IDX['brassica_alternaria']
    dm_idx = CLASS_TO_IDX['brassica_downy_mildew']
    bh_idx = CLASS_TO_IDX['brassica_healthy']

    if all(idx in centroids for idx in [br_idx, alt_idx, dm_idx, bh_idx]):
        br_alt = dist_matrix[br_idx, alt_idx]
        br_dm = dist_matrix[br_idx, dm_idx]
        br_bh = dist_matrix[br_idx, bh_idx]
        print(f"\nBlack rot distances:", flush=True)
        print(f"  to alternaria:    {br_alt:.4f}", flush=True)
        print(f"  to downy_mildew:  {br_dm:.4f}", flush=True)
        print(f"  to brassica_healthy: {br_bh:.4f}", flush=True)
        if br_alt < br_dm:
            print("  NOTE: Black rot IS closer to alternaria than downy_mildew "
                  "in DINOv2 feature space.", flush=True)

    analysis['centroid_distances'] = dist_matrix.tolist()

    # Within-class variance (field vs non-field)
    print(f"\nWithin-class feature variance:", flush=True)
    print(f"{'Class':<25s} {'All':>8s} {'Field':>8s} {'Lab':>8s}", flush=True)
    variance_results = {}
    for cls_idx in range(NUM_CLASSES):
        mask = val_y == cls_idx
        all_var = val_X[mask].var() if mask.any() else 0
        field_mask = mask & data['val']['is_field']
        lab_mask = mask & ~data['val']['is_field']
        field_var = val_X[field_mask].var() if field_mask.any() else 0
        lab_var = val_X[lab_mask].var() if lab_mask.any() else 0
        print(f"{CLASS_NAMES[cls_idx]:<25s} {all_var:>8.4f} {field_var:>8.4f} {lab_var:>8.4f}",
              flush=True)
        variance_results[CLASS_NAMES[cls_idx]] = {
            'all': float(all_var), 'field': float(field_var), 'lab': float(lab_var)}

    analysis['within_class_variance'] = variance_results

    # ═══════════════════════════════════════════════════════════════
    # ANALYSIS 5: Decision Threshold Analysis
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}", flush=True)
    print("ANALYSIS 5: Decision Threshold for Black Rot", flush=True)
    print(f"{'='*60}", flush=True)

    # Reload probe model for probability analysis
    import pickle, glob
    probe_files = sorted(glob.glob(str(RESULTS_DIR / 'fitted_probe_*.pkl')))
    if probe_files:
        with open(probe_files[-1], 'rb') as f:
            probe_data = pickle.load(f)
        clf = probe_data['clf']
        scaler_type = probe_data['scaler_type']

        # [FIX Issue 3] Use saved scaler, not a fresh one
        from sklearn.preprocessing import StandardScaler, normalize
        if scaler_type == 'standard':
            scaler = probe_data.get('scaler')
            if scaler is None:
                scaler = StandardScaler()
                scaler.fit(train_X)
            X_val_n = scaler.transform(val_X)
        elif scaler_type == 'l2_norm':
            X_val_n = normalize(val_X, norm='l2', axis=1)
        else:
            X_val_n = val_X

        y_proba = clf.predict_proba(X_val_n)

        for cls in FAILURE_CLASSES:
            cls_idx = CLASS_TO_IDX[cls]
            cls_mask = val_y == cls_idx
            if not cls_mask.any():
                continue

            cls_proba = y_proba[:, cls_idx]
            y_binary = (val_y == cls_idx).astype(int)

            # Precision-recall curve
            precision, recall, thresholds = precision_recall_curve(y_binary, cls_proba)

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(recall, precision, 'b-', linewidth=2)
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title(f'Precision-Recall: {cls} (DINOv2 probe)')
            ax.set_xlim(0, 1.05)
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)
            pr_path = RESULTS_DIR / f'pr_curve_{cls}_{ts}.png'
            plt.savefig(pr_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"PR curve saved: {pr_path}", flush=True)

            # Threshold analysis
            print(f"\n{cls} threshold analysis:", flush=True)
            for thresh in [0.2, 0.3, 0.4, 0.5]:
                preds_at_thresh = (cls_proba >= thresh).astype(int)
                tp = ((preds_at_thresh == 1) & (y_binary == 1)).sum()
                fp = ((preds_at_thresh == 1) & (y_binary == 0)).sum()
                fn = ((preds_at_thresh == 0) & (y_binary == 1)).sum()
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0
                print(f"  threshold={thresh:.1f}: precision={prec:.3f} recall={rec:.3f} "
                      f"(TP={tp}, FP={fp}, FN={fn})", flush=True)

    analysis['timestamp'] = ts

    # Save analysis
    analysis_path = RESULTS_DIR / f'analysis_{ts}.json'
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"\nAnalysis saved: {analysis_path}", flush=True)

    return analysis


if __name__ == '__main__':
    # Standalone usage requires loading data from cache
    from scripts.dinov2_probe.feature_cache import load_cache
    from scripts.dinov2_probe.probe_train import prepare_data
    cache = load_cache(FEATURES_CACHE_PATH)
    data = prepare_data(cache)
    run_analysis(data, {})
