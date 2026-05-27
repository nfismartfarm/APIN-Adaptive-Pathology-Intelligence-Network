"""extract_phase_d.py — Phase 4D data extractor.

Reads existing real cached results (apin_calibration.json, confusion-matrix
fig data, benchmark reports, source_map.csv) and emits the JSON shape
the Pipeline Atlas Phase 4D UI consumes:

  _qa_tmp/_pipeline_atlas_phase_d.json

Per spec Part 6 decision matrix: where real data exists, emit it.
Where it doesn't yet, emit { "pending": true, "reason": "..." } so the
UI shows an honest empty state instead of fabricated numbers.

Data sources read (real, on-disk):
  · scripts/apin/caches/apin_calibration.json     — T values + ECE + conformal
  · report_figures/fig_9_2_apin_confusion_matrix_v2.json — 9x9 confusion + per-class F1
  · reports/apin_vs_baselines_final_*.json         — benchmark / accuracy
  · data/metadata/source_map.csv                   — lineage
  · models/temperature.pt                          — fallback T values (if api unreachable)
"""

import csv
import glob
import json
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT  = os.path.join(ROOT, '_qa_tmp', '_pipeline_atlas_phase_d.json')

# ── helpers ──────────────────────────────────────────────────────────────
def load_json(path, default=None):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default

def pending(reason):
    return {"pending": True, "reason": reason}

def latest(pattern):
    """Newest file matching a glob pattern, or None."""
    matches = glob.glob(os.path.join(ROOT, pattern))
    return max(matches, key=os.path.getmtime) if matches else None

# ── 1 · CALIBRATION (per model) ──────────────────────────────────────────
def extract_calibration():
    """Pull T values, ECE, conformal thresholds from apin_calibration.json.
    Tomato and router don't have analogous cached files yet → pending state.
    """
    out = {"router": None, "tomato": None, "apin": None, "chilli": None}

    # APIN
    apin_cal = load_json(os.path.join(ROOT, 'scripts/apin/caches/apin_calibration.json'))
    if apin_cal:
        classes = apin_cal.get('class_order', [])
        ts = apin_cal.get('temperature_scaling', {})
        t_dict = ts.get('per_class_temperatures', {})
        t_values = [t_dict.get(c, None) for c in classes]
        cf = apin_cal.get('conformal_prediction', {})
        thr_dict = cf.get('per_class_thresholds', {})
        thresholds = [thr_dict.get(c, None) for c in classes]
        cal_n = cf.get('per_class_calibration_n', {})
        field_n = cf.get('per_class_field_calibration_n', {})
        out['apin'] = {
            "n_classes":    len(classes),
            "class_labels": classes,
            "t_values":     t_values,
            "ece_before":   ts.get('ece_before'),
            "ece_after":    ts.get('ece_after_held_out'),
            "fit_samples":  ts.get('fit_samples'),
            "eval_samples": ts.get('eval_samples'),
            "before_after_per_image": pending(
                "per-image pre/post softmax requires a fresh model run on the pool images"),
            "conformal": {
                "alpha":            cf.get('alpha'),
                "target_coverage":  cf.get('target_coverage'),
                "tau_per_class":    thresholds,
                "calibration_n":    [cal_n.get(c, 0) for c in classes],
                "field_calibration_n": [field_n.get(c, 0) for c in classes],
                "alpha_grid":         pending("multi-α conformal sweep needs a fresh run"),
                "set_size_dist":      pending("multi-α set sizes need a fresh run"),
                "empirical_cov":      pending("empirical coverage needs a fresh run"),
            },
            "ood": pending("OOD CLS-embedding PCA needs a backbone forward over the calibration set"),
            "reliability": pending("classic 10-bin reliability needs pre/post softmax on the calibration set"),
            "miscalib_heatmap": pending("per-class miscalibration heatmap needs the same data as reliability"),
        }
    else:
        out['apin'] = pending("scripts/apin/caches/apin_calibration.json not found")

    # Tomato — phase3_calibration files exist
    t_cal = load_json(os.path.join(ROOT, 'data/specialist/model3/phase3_calibration.json'))
    if t_cal:
        # Just surface what's actually in the file; the rest is pending
        out['tomato'] = {
            "n_classes": 6,
            "class_labels": t_cal.get('class_order', [
                "tomato_late_blight","tomato_septoria_leaf_spot","tomato_yellow_leaf_curl_virus",
                "tomato_mosaic_virus","tomato_foliar_spot","tomato_healthy"
            ]),
            "t_values":     t_cal.get('temperature_scaling', {}).get('per_class_temperatures'),
            "ece_before":   (t_cal.get('temperature_scaling', {}) or {}).get('ece_before'),
            "ece_after":    (t_cal.get('temperature_scaling', {}) or {}).get('ece_after_held_out'),
            "before_after_per_image": pending("per-image pre/post softmax needs a fresh run"),
            "conformal":    t_cal.get('conformal_prediction') or pending("no tomato conformal cached"),
            "ood":          pending("tomato OOD scatter pending"),
            "reliability":  pending("tomato reliability pending"),
            "miscalib_heatmap": pending("tomato miscalib heatmap pending"),
        }
    else:
        out['tomato'] = pending("tomato calibration cache not found")

    # Router — no cached calibration file
    out['router'] = pending("router has no per-class temperature scaling (4-class softmax is calibrated as a whole)")

    # Chilli — router-only by design
    out['chilli'] = {"stub": True, "reason": "router-only, no calibration head"}

    return out

# ── 2 · SCATTERED · M1 router metrics ───────────────────────────────────
def extract_router_metrics():
    """Router metrics: try to read a cached confusion matrix file; else pending."""
    # The router doesn't have a single cached confusion-matrix file the same way
    # APIN does. Surface a pending state with the precise file shape needed.
    return pending(
        "scripts/apin_v2/extract_router_metrics.py needs to run · "
        "produces fig_router_confusion.json with shape "
        "{ confusion_matrix: 4x4, per_crop_metrics: [{crop, precision, recall, f1}] }"
    )

# ── 3 · SCATTERED · M2 tomato metrics ────────────────────────────────────
def extract_tomato_metrics():
    """Tomato metrics: prototype bank exists, confusion matrix doesn't have a cache yet."""
    # Look for any tomato confusion matrix figure
    tomato_cm = load_json(os.path.join(ROOT, 'report_figures/fig_tomato_confusion_matrix.json'))
    out = {}
    if tomato_cm:
        out['confusion_matrix']     = tomato_cm.get('confusion_matrix')
        out['per_class_metrics']    = tomato_cm.get('per_class_f1')  # rough
        out['overall_accuracy']     = tomato_cm.get('overall_accuracy')
        out['class_labels']         = tomato_cm.get('class_order')
        out['n_test_images']        = tomato_cm.get('n')
    else:
        out['confusion_matrix']  = pending("tomato 6x6 confusion matrix cache not found · run extract_tomato_metrics.py")
        out['locked_test_eval']  = pending("tomato locked test split eval pending")
    # Per-block CLS metrics
    out['per_block_cls_evolution'] = pending(
        "per-block CLS evolution needs a backbone forward over the 3 pool images · "
        "data is captured during forward_pass extraction but not yet surfaced separately")
    # Reliability diagrams
    out['reliability_before']      = pending("tomato classic reliability needs the same data as APIN's reliability")
    out['reliability_after']       = pending("tomato classic reliability needs the same data as APIN's reliability")
    # Tier thresholds
    out['tier_thresholds']         = {
        "tier_1a": 0.60,
        "tier_1b": 0.45,
        "lower_cuts": [0.30, 0.18, 0.10],
        "source": "v1 router design doc · static thresholds",
    }
    # Prototype bank
    proto = os.path.join(ROOT, 'data/specialist/model3/prototype_bank.pt')
    proto_lora = os.path.join(ROOT, 'data/specialist/model3/prototype_bank_sp_lora_ep13.pt')
    if os.path.exists(proto) or os.path.exists(proto_lora):
        out['prototype_bank'] = {
            "available": True,
            "file_v3":      "data/specialist/model3/prototype_bank.pt" if os.path.exists(proto) else None,
            "file_sp_lora": "data/specialist/model3/prototype_bank_sp_lora_ep13.pt" if os.path.exists(proto_lora) else None,
            "n_prototypes_per_class": 5,
            "pca_variance":  pending("PCA variance computation pending"),
        }
    else:
        out['prototype_bank'] = pending("prototype bank .pt files not found")
    return out

# ── 4 · SCATTERED · M4 APIN ensemble (real confusion + per-class F1) ────
def extract_apin_ensemble():
    """Read the rich confusion matrix file and surface everything."""
    cm = load_json(os.path.join(ROOT, 'report_figures/fig_9_2_apin_confusion_matrix_v2.json'))
    if not cm:
        return pending("APIN confusion matrix figure not found")
    out = {
        "evaluation_subset":  cm.get('evaluation_subset'),
        "n_test_images":      cm.get('n'),
        "overall_accuracy":   cm.get('overall_accuracy'),
        "overall_macro_f1":   cm.get('overall_macro_f1'),
        "overall_weighted_f1": cm.get('overall_weighted_f1'),
        "class_order":        cm.get('class_order'),
        "class_short_names":  cm.get('short_names'),
        "confusion_matrix":   cm.get('confusion_matrix'),
        "per_class_recall":   cm.get('per_class_recall'),
        "per_class_precision":cm.get('per_class_precision'),
        "per_class_f1":       cm.get('per_class_f1'),
        "major_confusions":   cm.get('major_confusions'),
        "row_sums":           cm.get('row_sums'),
        "col_sums":           cm.get('col_sums'),
    }
    # Look for per-signal benchmarks in the final report
    bench = load_json(os.path.join(ROOT, 'reports/apin_vs_baselines_final_20260523_030207.json'))
    if bench and 'apin_ensemble' in bench:
        out['apin_vs_baselines'] = bench['apin_ensemble']
    # Per-signal reliability matrix is exposed live via /api/info.reliability_matrix
    # · UI reads it directly from there. Note that fact:
    out['per_signal_reliability_source'] = "/api/info · reliability_matrix (live)"
    out['per_signal_calibration']      = pending("per-signal ECE breakdown needs split-conformal sweep on the calibration set")
    out['per_signal_ood']              = pending("per-signal OOD distributions need backbone-specific Mahalanobis runs")
    out['tier_distribution']           = pending("tier-1A/1B/2/3/4A/4B/5 counts need a router+ensemble joint run on the test set")
    return out

# ── 5 · LINEAGE (already done by _compute_lineage.py · re-run it here) ──
def extract_lineage():
    """Compute fresh source-to-split flow from source_map.csv."""
    src_csv = os.path.join(ROOT, 'data/metadata/source_map.csv')
    if not os.path.exists(src_csv):
        return pending("source_map.csv not found")
    with open(src_csv, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    # Per-source-per-split counts
    src_split = defaultdict(lambda: defaultdict(int))
    for r in rows:
        src_split[r['source_dataset']][r['split']] += 1
    splits = sorted({r['split'] for r in rows})
    # Family aggregation matches _compute_lineage.py
    FAMILY = {
        'gadde_okra':'Okra · academic','bangladesh_okra':'Okra · academic',
        'yeesi':'Okra · academic','mendeley_okra':'Okra · academic',
        'inaturalist_okra':'iNaturalist · CC-BY-NC',
        'cauliflower_noam':'Brassica · academic','cabbage_balanced':'Brassica · academic',
        'diya_broccoli':'Brassica · academic','diya_cabbage':'Brassica · academic',
        'diya_cauliflower':'Brassica · academic','mendeley_caul_leaf':'Brassica · academic',
        'mendeley_cabbage_dis':'Brassica · academic','caul_sharifashik':'Brassica · academic',
        'inaturalist_brassica':'iNaturalist · CC-BY-NC',
        'plantvillage_tomato':'PlantVillage · canonical','tomato_ashish':'Tomato · academic',
        'plantdoc_train':'PlantDoc','plantdoc_eval':'PlantDoc',
        'tomato_luisolazo':'Tomato · academic','tomato_kaustubh':'Tomato · academic',
        'inaturalist_tomato':'iNaturalist · CC-BY-NC',
        'chilli_bangladesh':'Chilli · academic','chilli_anthracnose_prudhvi':'Chilli · academic',
        'chilli_cold_karnataka':'Chilli · academic','chilli_bangladesh_2025':'Chilli · academic',
        'inaturalist_chilli':'iNaturalist · CC-BY-NC',
    }
    fam_split = defaultdict(lambda: defaultdict(int))
    for src, by_split in src_split.items():
        fam = FAMILY.get(src, 'Other')
        for sp, n in by_split.items():
            fam_split[fam][sp] += n
    families = sorted(fam_split.keys())
    flow = [[fam_split[fam].get(sp, 0) for sp in splits] for fam in families]
    # Leakage check
    split_paths = defaultdict(set)
    for r in rows:
        split_paths[r['split']].add(r['image_path'])
    pairs = []
    sn = splits
    for i, a in enumerate(sn):
        for b in sn[i+1:]:
            pairs.append({
                'a': a, 'b': b,
                'overlap_count': len(split_paths[a] & split_paths[b]),
                'a_size': len(split_paths[a]),
                'b_size': len(split_paths[b]),
            })
    return {
        "sankey": {
            "sources": [{"name": fam, "n_images": sum(fam_split[fam].values())} for fam in families],
            "splits":  [{"name": sp, "n_images": sum(fam_split[fam].get(sp, 0) for fam in families),
                         "role": {
                             "train":"model fitting",
                             "val":"hyperparameters · calibration · conformal · early stopping",
                             "test":"locked · final evaluation only",
                         }.get(sp, sp)} for sp in splits],
            "flow": flow,
            "excluded_total": 0,
        },
        "leakage": {
            "pairs": pairs,
            "total_paths": sum(len(s) for s in split_paths.values()),
            "method": "image_path overlap test on source_map.csv",
        },
        "totals": {
            "total_rows":   len(rows),
            "unique_paths": len({r['image_path'] for r in rows}),
            "n_sources":    len(src_split),
            "n_families":   len(families),
            "n_splits":     len(splits),
        },
    }

# ── orchestrator ────────────────────────────────────────────────────────
def main():
    import datetime
    out = {
        "produced_at":   datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "produced_by":   "scripts/apin_v2/extract_phase_d.py",
        "models":        extract_calibration(),
        "scattered": {
            "router":     extract_router_metrics(),
            "tomato":     extract_tomato_metrics(),
            "apin":       extract_apin_ensemble(),
        },
        "lineage":       extract_lineage(),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    # Brief summary
    print(f"Wrote {OUT}")
    print(f"  models with real T values: " + ", ".join(
        m for m, d in out['models'].items()
        if isinstance(d, dict) and not d.get('pending') and not d.get('stub')
        and d.get('t_values') and not (isinstance(d.get('t_values'), dict) and d['t_values'].get('pending'))
    ))
    print(f"  APIN confusion matrix: {out['scattered']['apin'].get('n_test_images')} test images, "
          f"accuracy {out['scattered']['apin'].get('overall_accuracy', 0):.4f}")
    print(f"  Lineage: {out['lineage']['totals']['total_rows']} images, "
          f"{out['lineage']['totals']['n_families']} families, "
          f"{out['lineage']['totals']['n_splits']} splits")
    # Count pending items
    def count_pending(o, c=0):
        if isinstance(o, dict):
            if o.get('pending'): return c + 1
            return sum(count_pending(v, 0) for v in o.values()) + c
        if isinstance(o, list):
            return sum(count_pending(v, 0) for v in o) + c
        return c
    print(f"  pending sub-items: {count_pending(out)}")

if __name__ == '__main__':
    main()
