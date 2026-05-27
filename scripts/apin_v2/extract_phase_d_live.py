"""extract_phase_d_live.py · Phase 4D live extraction from cached signal predictions.

Reads the 4 per-signal prediction caches in scripts/apin/caches/ (9,006 per-image
post-softmax outputs across 4 backbones) and computes:

  · C.3  multi-alpha conformal sweep    · per-class tau, set-size distribution,
                                       empirical coverage for alpha in [0.05 .. 0.50]
  · C.5  classic 10-bin reliability · per-signal binned confidence-vs-accuracy
                                       + the simple-mean ensemble curve
  · C.6  per-class miscalibration   · 5-confidence-bin × 9-class gap heatmap
  · C.4  softmax-PCA scatter        · 2D PCA of per-image 9-D softmax vectors
                                       as a data-grounded proxy for the spec's
                                       CLS-embedding PCA (which would need a
                                       backbone forward pass to produce 384-D
                                       embeddings — clearly labelled as proxy)

Output: _qa_tmp/_pipeline_atlas_phase_d_live.json
  Same top-level shape as the stitcher's output but with the previously-
  pending fields filled in with measured numbers from the 736 field photos
  in the final_val split (plus 307 field photos in the conformal split for
  the calibration anchor).

Honest scope:
  · Router 4×4 confusion + Tomato 6×6 confusion still pending — no per-image
    prediction caches for those pipelines exist. They need a separate run.
  · The "before vs after temperature scaling" comparison uses the per-signal
    raw softmax outputs (which are pre-T at the stacking level). True per-image
    pre-T logits are not in the caches.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = os.path.join(ROOT, 'scripts', 'apin', 'caches')
CAL_JSON = os.path.join(CACHE_DIR, 'apin_calibration.json')
OUT = os.path.join(ROOT, '_qa_tmp', '_pipeline_atlas_phase_d_live.json')

CLASS_ORDER = [
    'okra_yvmv', 'okra_powdery_mildew', 'okra_cercospora', 'okra_enation',
    'okra_healthy', 'brassica_black_rot', 'brassica_downy_mildew',
    'brassica_alternaria', 'brassica_healthy',
]
SIGNAL_NAMES = ['DINOv3-ConvNeXt', 'EfficientNetV2-S', 'PSV', 'DINOv2 ViT-S']
SIGNAL_CACHES = ['signal1', 'signal2', 'signal3_psv', 'signal4']


# ── helpers ─────────────────────────────────────────────────────────────
def load_signal_cache(name: str):
    path = os.path.join(CACHE_DIR, f'{name}_predictions_cache.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)


def _is_ok(e):
    """Success-flag normalisation · the 4 caches use different keys:
       signal1/signal2 → inference_success
       signal3_psv     → extraction_success
       signal4         → no flag (always assume success if predictions present)
    """
    if 'inference_success' in e:
        return bool(e['inference_success'])
    if 'extraction_success' in e:
        return bool(e['extraction_success'])
    return e.get('predictions') is not None


def to_arrays(cache, split_filter='conformal', field_only=True):
    """Return (probs[N,9], labels[N]) for the requested split."""
    probs, labels = [], []
    for k, e in cache.items():
        if not _is_ok(e):
            continue
        if split_filter and e.get('split') != split_filter:
            continue
        if field_only and not e.get('is_field_photo'):
            continue
        p = e.get('predictions')
        if p is None:
            continue
        probs.append(np.asarray(p, dtype=np.float64))
        labels.append(e.get('true_class_idx', -1))
    return np.array(probs), np.array(labels)


def mean_ensemble(signal_caches):
    """Build a simple 1/4 mean ensemble.

    Aligns images across signals by their cache key (which is the per-image
    integer index). Returns (probs[N,9], labels[N], splits[N], is_field[N]).
    """
    keys = set(signal_caches[0].keys())
    for sc in signal_caches[1:]:
        keys &= set(sc.keys())
    keys = sorted(keys)
    rows = []
    for k in keys:
        ok = all(_is_ok(sc[k]) for sc in signal_caches)
        if not ok:
            continue
        # Renormalise each signal's prob vector defensively (some caches may
        # not be perfectly summed-to-1 due to dtype rounding).
        ps = []
        for sc in signal_caches:
            p = np.asarray(sc[k]['predictions'], dtype=np.float64)
            s = p.sum()
            ps.append(p / s if s > 0 else p)
        avg = np.mean(np.stack(ps, axis=0), axis=0)
        e = signal_caches[0][k]  # all signals share metadata
        rows.append((avg, e.get('true_class_idx', -1),
                     e.get('split'), e.get('is_field_photo', False),
                     e.get('class_name', '')))
    if not rows:
        return np.zeros((0, 9)), np.zeros((0,), dtype=int), [], [], []
    probs = np.stack([r[0] for r in rows])
    labels = np.array([r[1] for r in rows])
    splits = [r[2] for r in rows]
    is_field = np.array([r[3] for r in rows])
    class_names = [r[4] for r in rows]
    return probs, labels, splits, is_field, class_names


# ── C.3 · multi-alpha conformal sweep ───────────────────────────────────────
def compute_conformal_sweep(cal_probs, cal_labels, eval_probs, eval_labels,
                            alpha_grid=None):
    """Per-class APS-style conformal: tau_c(alpha) = alpha-quantile of (predicted prob
    for true class), restricted to images whose true label is c.

    Prediction set for an image x at alpha: { c : p(c|x) >= tau_c(alpha) }.

    Returns dict with:
      · alpha_grid[]
      · tau_per_alpha[n_alpha][n_classes]
      · set_size_dist[n_alpha][6]   buckets: 0, 1, 2, 3, 4, 5+
      · empirical_cov[n_alpha]      fraction of eval images whose true label is in their set
      · mean_set_size[n_alpha]
    """
    if alpha_grid is None:
        alpha_grid = [round(0.05 * i, 2) for i in range(1, 11)]  # 0.05 .. 0.50
    n_classes = cal_probs.shape[1]
    out = {
        'alpha_grid': alpha_grid,
        'tau_per_alpha': [],
        'set_size_dist': [],
        'empirical_cov': [],
        'mean_set_size': [],
        'n_cal': int(cal_probs.shape[0]),
        'n_eval': int(eval_probs.shape[0]),
    }
    for alpha in alpha_grid:
        # Per-class tau at this alpha: take alpha-quantile of p(true class) for each class
        tau = np.zeros(n_classes)
        for c in range(n_classes):
            mask = cal_labels == c
            if mask.sum() == 0:
                tau[c] = 1.0  # empty calibration → never include
                continue
            ps = cal_probs[mask, c]
            # alpha-quantile lower bound (split-conformal · finite-sample correction)
            n = len(ps)
            q_idx = max(0, int(np.floor(alpha * (n + 1))) - 1)
            tau[c] = float(np.sort(ps)[q_idx])
        # Build prediction sets on eval split
        sets = (eval_probs >= tau[None, :])
        sizes = sets.sum(axis=1)
        # Bucket sizes into 0, 1, 2, 3, 4, 5+
        buckets = [
            int(np.sum(sizes == 0)),
            int(np.sum(sizes == 1)),
            int(np.sum(sizes == 2)),
            int(np.sum(sizes == 3)),
            int(np.sum(sizes == 4)),
            int(np.sum(sizes >= 5)),
        ]
        # Coverage: fraction of images whose true class is in their set
        covered = sets[np.arange(len(eval_labels)), eval_labels].sum()
        out['tau_per_alpha'].append([float(x) for x in tau])
        out['set_size_dist'].append(buckets)
        out['empirical_cov'].append(float(covered / max(len(eval_labels), 1)))
        out['mean_set_size'].append(float(sizes.mean()))
    return out


# ── C.5 · classic 10-bin reliability ────────────────────────────────────
def compute_reliability(probs, labels, n_bins=10):
    """For each image take max softmax prob (confidence) and predicted class.
    Bin by confidence into n_bins equal-width buckets.
    Returns bins[n_bins] = {center, count, mean_conf, accuracy, gap}.
    """
    if len(probs) == 0:
        return {'bins': [], 'ece': None, 'n': 0}
    preds = probs.argmax(axis=1)
    confs = probs.max(axis=1)
    correct = (preds == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    ece = 0.0
    n = len(confs)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # Include the right edge for the last bin
        if i == n_bins - 1:
            mask = (confs >= lo) & (confs <= hi)
        else:
            mask = (confs >= lo) & (confs < hi)
        cnt = int(mask.sum())
        if cnt == 0:
            bins.append({'center': float((lo + hi) / 2), 'count': 0,
                         'mean_conf': None, 'accuracy': None, 'gap': None})
            continue
        mc = float(confs[mask].mean())
        ac = float(correct[mask].mean())
        gap = float(abs(mc - ac))
        bins.append({'center': float((lo + hi) / 2), 'count': cnt,
                     'mean_conf': mc, 'accuracy': ac, 'gap': gap})
        ece += (cnt / n) * gap
    return {'bins': bins, 'ece': float(ece), 'n': int(n)}


# ── C.6 · per-class miscalibration heatmap (5 conf bins × 9 classes) ────
def compute_per_class_miscalib(probs, labels, n_bins=5):
    """For each (predicted class, confidence bin), gap = |conf - accuracy|.
    Returns matrix[n_classes][n_bins].
    """
    n_classes = probs.shape[1]
    if len(probs) == 0:
        return {'matrix': [], 'counts': [], 'n_bins': n_bins, 'n': 0}
    preds = probs.argmax(axis=1)
    confs = probs.max(axis=1)
    correct = (preds == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    matrix = [[None] * n_bins for _ in range(n_classes)]
    counts = [[0] * n_bins for _ in range(n_classes)]
    for c in range(n_classes):
        cmask = preds == c
        for b in range(n_bins):
            lo, hi = edges[b], edges[b + 1]
            if b == n_bins - 1:
                bmask = cmask & (confs >= lo) & (confs <= hi)
            else:
                bmask = cmask & (confs >= lo) & (confs < hi)
            cnt = int(bmask.sum())
            counts[c][b] = cnt
            if cnt == 0:
                continue
            mc = float(confs[bmask].mean())
            ac = float(correct[bmask].mean())
            matrix[c][b] = {
                'conf': mc, 'acc': ac,
                'gap': float(mc - ac),  # signed: positive = over-confident
                'abs_gap': float(abs(mc - ac)),
            }
    return {
        'matrix': matrix,
        'counts': counts,
        'n_bins': n_bins,
        'bin_edges': [float(x) for x in edges],
        'class_order': CLASS_ORDER,
        'n': int(len(probs)),
    }


# ── C.4 · softmax-PCA scatter (proxy for CLS-embedding PCA) ─────────────
def compute_softmax_pca(probs, labels, class_names):
    """2D PCA of per-image 9-D softmax vectors.

    This is NOT the CLS-embedding PCA the spec calls for — that needs 384-D
    backbone outputs that aren't in the caches. The softmax-vector PCA is
    a data-grounded proxy: in-distribution well-classified images cluster
    near class corners of the simplex; high-entropy / abstain candidates
    spread toward the centroid. UI labels this clearly as proxy.

    Returns:
      · points[N] = [x, y, class_idx, true_class_idx, entropy, max_prob]
      · explained_variance_ratio[2]
      · component_axes[2][9]    (so the page can name the axes if useful)
    """
    if len(probs) == 0:
        return {'points': [], 'explained_variance_ratio': [0, 0]}
    # Center
    mu = probs.mean(axis=0, keepdims=True)
    X = probs - mu
    # SVD
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    total_var = float((S ** 2).sum())
    explained = [float(s ** 2 / total_var) for s in S[:2]] if total_var > 0 else [0, 0]
    proj = X @ Vt[:2].T  # [N, 2]
    # Entropy per image (in nats)
    eps = 1e-12
    H = -(probs * np.log(probs + eps)).sum(axis=1)
    max_p = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    points = []
    for i in range(len(probs)):
        # Downsample to ~400 points for browser snappiness if we have many
        points.append([
            float(proj[i, 0]), float(proj[i, 1]),
            int(pred[i]), int(labels[i]),
            float(H[i]), float(max_p[i]),
        ])
    # Sample if too many
    if len(points) > 500:
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(len(points), size=500, replace=False)
        points = [points[i] for i in sorted(idx)]
    return {
        'points': points,
        'explained_variance_ratio': explained,
        'component_axes': [Vt[0].tolist(), Vt[1].tolist()] if Vt.shape[0] >= 2 else [],
        'class_order': CLASS_ORDER,
        'is_proxy': True,
        'proxy_note': 'softmax-vector PCA · proxy for CLS-embedding PCA · the latter needs a backbone forward pass for 384-D embeddings',
        'n_points_after_downsample': len(points),
        'n_source': int(len(probs)),
    }


# ── orchestrator ────────────────────────────────────────────────────────
def main():
    print(f'Loading 4 signal caches from {CACHE_DIR} ...')
    caches = [load_signal_cache(name) for name in SIGNAL_CACHES]
    for nm, c in zip(SIGNAL_CACHES, caches):
        print(f'  {nm}: {len(c)} entries')

    # ── Build the mean ensemble across all 4 signals ───────────────────
    print('Building 1/4 mean ensemble ...')
    ens_probs, ens_labels, ens_splits, ens_field, ens_classes = mean_ensemble(caches)
    print(f'  ensemble entries (post-success-filter): {len(ens_probs)}')

    # Per-split masks (field-only)
    def split_mask(name):
        return np.array([s == name and f for s, f in zip(ens_splits, ens_field)])

    cal_mask = split_mask('conformal')
    eval_mask = split_mask('final_val')
    vs_mask = split_mask('val_and_soup')
    print(f'  conformal field photos: {int(cal_mask.sum())}')
    print(f'  final_val field photos: {int(eval_mask.sum())}')
    print(f'  val_and_soup field photos: {int(vs_mask.sum())}')

    # ── C.3 multi-alpha conformal sweep ────────────────────────────────────
    print('Computing C.3 multi-alpha conformal sweep ...')
    cal_probs = ens_probs[cal_mask]
    cal_labels = ens_labels[cal_mask]
    eval_probs = ens_probs[eval_mask]
    eval_labels = ens_labels[eval_mask]
    sweep = compute_conformal_sweep(cal_probs, cal_labels, eval_probs, eval_labels)
    print(f'  alphas: {sweep["alpha_grid"]}')
    print(f'  empirical coverage at alpha=0.05: {sweep["empirical_cov"][0]:.4f} '
          f'(target {1 - sweep["alpha_grid"][0]:.2f})')
    print(f'  set-size dist at alpha=0.05: {sweep["set_size_dist"][0]}')

    # ── C.5 classic 10-bin reliability (per-signal + ensemble) ─────────
    print('Computing C.5 reliability bins ...')
    per_signal_reliability = []
    for sig_name, cache in zip(SIGNAL_NAMES, caches):
        p, l = to_arrays(cache, split_filter='final_val', field_only=True)
        rel = compute_reliability(p, l, n_bins=10)
        rel['signal'] = sig_name
        rel['n'] = int(len(p))
        per_signal_reliability.append(rel)
        print(f'  {sig_name}: n={len(p)}, ECE={rel["ece"]:.4f}')
    # Ensemble post-T proxy: take ensemble mean (post-signal-softmax) on final_val
    ens_rel = compute_reliability(eval_probs, eval_labels, n_bins=10)
    ens_rel['signal'] = 'Mean Ensemble (post-signal-softmax)'
    ens_rel['n'] = int(len(eval_probs))
    print(f'  Mean Ensemble: n={ens_rel["n"]}, ECE={ens_rel["ece"]:.4f}')

    # ── C.6 per-class miscalibration heatmap ───────────────────────────
    print('Computing C.6 per-class miscalibration heatmap ...')
    vs_probs = ens_probs[vs_mask]
    vs_labels = ens_labels[vs_mask]
    heatmap = compute_per_class_miscalib(vs_probs, vs_labels, n_bins=5)
    print(f'  heatmap shape: {len(heatmap["matrix"])}×{heatmap["n_bins"]} '
          f'· n={heatmap["n"]}')

    # ── C.4 softmax-PCA scatter (proxy) ────────────────────────────────
    print('Computing C.4 softmax-PCA scatter (proxy) ...')
    pca = compute_softmax_pca(eval_probs, eval_labels,
                              [ens_classes[i] for i in range(len(ens_classes)) if eval_mask[i]])
    print(f'  PCA components explain {pca["explained_variance_ratio"][0]:.3f} + '
          f'{pca["explained_variance_ratio"][1]:.3f} = '
          f'{sum(pca["explained_variance_ratio"]):.3f} of variance')
    print(f'  scatter points after downsample: {pca["n_points_after_downsample"]}')

    # ── Write live JSON ───────────────────────────────────────────────
    out = {
        'produced_at_utc': __import__('datetime').datetime.utcnow().isoformat() + 'Z',
        'produced_by': 'scripts/apin_v2/extract_phase_d_live.py',
        'data_source': '4 per-image signal prediction caches (signal1-4 · 9,006 entries · ~736 field photos in final_val)',
        'class_order': CLASS_ORDER,
        'signal_names': SIGNAL_NAMES,
        'conformal_sweep': sweep,
        'reliability': {
            'per_signal': per_signal_reliability,
            'ensemble_post_signal_softmax': ens_rel,
        },
        'per_class_miscalib': heatmap,
        'softmax_pca_scatter': pca,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f'\nWrote {OUT}')
    sz = os.path.getsize(OUT) / 1024
    print(f'  size: {sz:.1f} KB')


if __name__ == '__main__':
    main()
