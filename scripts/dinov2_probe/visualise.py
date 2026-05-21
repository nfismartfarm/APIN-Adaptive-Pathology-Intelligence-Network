"""
DINOv2 Feature Space Visualization — t-SNE/UMAP plots and analysis charts.

Generates 6 publication-quality plots for the experiment report.
"""

import sys
import logging
import numpy as np
from pathlib import Path
from typing import Dict
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.dinov2_probe.config import (
    CLASS_NAMES, CLASS_TO_IDX, NUM_CLASSES,
    FAILURE_CLASSES, MODEL2_VAL_F1,
    RESULTS_DIR, RANDOM_SEED,
)

# 9 distinct colors for classes (colorblind-friendly palette)
CLASS_COLORS = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
    '#911eb4', '#42d4f4', '#f032e6', '#469990',
]


def run_visualisations(data: Dict, probe_results: Dict) -> None:
    """Generate all visualisation plots."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    val_X = data['val']['X']
    val_y = data['val']['y']
    val_field = data['val']['is_field']
    val_sources = data['val']['sources']

    # ═══════════════════════════════════════════════════════════════
    # PLOT 1: t-SNE by class
    # ═══════════════════════════════════════════════════════════════
    print("Generating t-SNE plots (this may take a few minutes)...", flush=True)

    tsne = TSNE(n_components=2, random_state=RANDOM_SEED, perplexity=30,
                n_iter=1000, learning_rate='auto', init='pca')
    embedded = tsne.fit_transform(val_X)

    fig, ax = plt.subplots(figsize=(12, 10))
    for cls_idx in range(NUM_CLASSES):
        mask = val_y == cls_idx
        field_mask = mask & val_field
        lab_mask = mask & ~val_field

        # Lab photos: circles
        if lab_mask.any():
            ax.scatter(embedded[lab_mask, 0], embedded[lab_mask, 1],
                      c=CLASS_COLORS[cls_idx], marker='o', s=20, alpha=0.5,
                      label=f'{CLASS_NAMES[cls_idx]} (lab)')
        # Field photos: stars
        if field_mask.any():
            ax.scatter(embedded[field_mask, 0], embedded[field_mask, 1],
                      c=CLASS_COLORS[cls_idx], marker='*', s=60, alpha=0.8,
                      label=f'{CLASS_NAMES[cls_idx]} (field)')

    ax.set_title(f'DINOv2-Small val features — {len(val_X)} images\n'
                 f'(circles=lab, stars=field)', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, markerscale=1.5)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    plt.tight_layout()
    path = RESULTS_DIR / f'tsne_by_class_{ts}.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}", flush=True)

    # ═══════════════════════════════════════════════════════════════
    # PLOT 2: t-SNE by source
    # ═══════════════════════════════════════════════════════════════
    unique_sources = sorted(set(val_sources))
    source_colors = plt.cm.tab20(np.linspace(0, 1, min(len(unique_sources), 20)))

    fig, ax = plt.subplots(figsize=(12, 10))
    for i, src in enumerate(unique_sources):
        mask = np.array(val_sources) == src
        color = source_colors[i % len(source_colors)]
        ax.scatter(embedded[mask, 0], embedded[mask, 1],
                  c=[color], marker='o', s=20, alpha=0.6, label=src)

    ax.set_title(f'DINOv2 features colored by source_dataset\n'
                 f'(mixing = disease-discriminative, clustering = source-specific)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    plt.tight_layout()
    path = RESULTS_DIR / f'tsne_by_source_{ts}.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}", flush=True)

    # ═══════════════════════════════════════════════════════════════
    # PLOT 3: LDA separation for failing classes
    # ═══════════════════════════════════════════════════════════════
    for cls_a, cls_b in [('brassica_black_rot', 'brassica_alternaria'),
                          ('okra_cercospora', 'okra_healthy')]:
        idx_a = CLASS_TO_IDX[cls_a]
        idx_b = CLASS_TO_IDX[cls_b]
        mask = (val_y == idx_a) | (val_y == idx_b)
        if mask.sum() < 10:
            continue

        X_pair = val_X[mask]
        y_pair = val_y[mask]

        lda = LinearDiscriminantAnalysis(n_components=1)
        X_1d = lda.fit_transform(X_pair, y_pair).flatten()

        # [FIX Issue 1] Use full_field array aligned with y_pair
        full_field = val_field[mask]  # same length as y_pair and X_1d
        fig, ax = plt.subplots(figsize=(10, 5))
        for idx, cls, color in [(idx_a, cls_a, 'red'), (idx_b, cls_b, 'blue')]:
            cls_mask = y_pair == idx
            lab_mask = cls_mask & ~full_field
            field_mask_cls = cls_mask & full_field
            if lab_mask.any():
                ax.hist(X_1d[lab_mask], bins=30, alpha=0.4, color=color,
                        label=f'{cls} (lab)', density=True)
            if field_mask_cls.any():
                ax.hist(X_1d[field_mask_cls], bins=15, alpha=0.7, color=color,
                        label=f'{cls} (field)', density=True, histtype='step', linewidth=2)

        ax.set_title(f'LDA projection: {cls_a} vs {cls_b}')
        ax.set_xlabel('LDA discriminant')
        ax.set_ylabel('Density')
        ax.legend()
        plt.tight_layout()
        path = RESULTS_DIR / f'lda_{cls_a}_vs_{cls_b}_{ts}.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}", flush=True)

    # ═══════════════════════════════════════════════════════════════
    # PLOT 4: Per-class F1 comparison bar chart
    # ═══════════════════════════════════════════════════════════════
    probe_f1s = []
    model2_f1s = []
    for cls in CLASS_NAMES:
        pf1 = probe_results.get('per_class', {}).get(cls, {}).get('probe_f1', 0)
        mf1 = MODEL2_VAL_F1.get(cls, 0)
        probe_f1s.append(pf1)
        model2_f1s.append(mf1)

    x = np.arange(NUM_CLASSES)
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width / 2, probe_f1s, width, label='DINOv2 Probe', color='#3498db')
    bars2 = ax.bar(x + width / 2, model2_f1s, width, label='Model 2', color='#e67e22')
    ax.axhline(y=0.70, color='red', linestyle='--', alpha=0.5, label='0.70 threshold')
    ax.set_xlabel('Class')
    ax.set_ylabel('F1 Score')
    ax.set_title('DINOv2 Linear Probe vs Model 2 Specialist — Val Set F1')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', '\n') for c in CLASS_NAMES], fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = RESULTS_DIR / f'f1_comparison_{ts}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}", flush=True)

    print("Visualizations complete.", flush=True)


if __name__ == '__main__':
    from scripts.dinov2_probe.feature_cache import load_cache
    from scripts.dinov2_probe.probe_train import prepare_data
    from scripts.dinov2_probe.config import FEATURES_CACHE_PATH
    cache = load_cache(FEATURES_CACHE_PATH)
    data = prepare_data(cache)
    run_visualisations(data, {})
