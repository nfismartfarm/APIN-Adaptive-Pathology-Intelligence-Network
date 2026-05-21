"""
DINOv2 Linear Probe Experiment — Master Orchestration Script.

Runs the complete experiment in the correct order:
  1. Validate config and paths
  2. Extract/load DINOv2 features (30-45 min first time)
  3. Train linear probe with grid search (2-5 min)
  4. Deep analysis (5 min)
  5. OOD detection (2 min)
  6. Visualizations (5-10 min)
  7. Generate final report

Usage: python scripts/dinov2_probe/run_experiment.py
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.dinov2_probe.config import (
    PROJECT_ROOT, MODEL2_CSV, SPLIT_INDICES,
    FEATURES_CACHE_PATH, RESULTS_DIR,
    FEATURE_AGGREGATION, FEATURE_DIM,
    CLASS_NAMES, NUM_CLASSES, FAILURE_CLASSES,
    MODEL2_VAL_F1, RANDOM_SEED, TIMM_MODEL_NAME,
)


def validate_prerequisites() -> None:
    """Check all required files exist before starting."""
    checks = [
        (MODEL2_CSV, "Model 2 CSV"),
        (SPLIT_INDICES, "Split indices JSON"),
    ]
    for path, name in checks:
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")
    logger.info("All prerequisites validated")


def generate_report(
    probe_results: dict,
    analysis_results: dict,
    ood_results: dict,
    elapsed: float,
) -> str:
    """Generate the final decision-making report."""
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    val_f1 = probe_results.get('val_macro_f1', 0)
    per_class = probe_results.get('per_class', {})

    # Determine recommendation
    br_result = per_class.get('brassica_black_rot', {})
    cc_result = per_class.get('okra_cercospora', {})
    br_f1 = br_result.get('probe_f1', 0)
    cc_f1 = cc_result.get('probe_f1', 0)

    field_br = probe_results.get('field_analysis', {}).get('brassica_black_rot', {})
    field_cc = probe_results.get('field_analysis', {}).get('okra_cercospora', {})
    br_field_acc = field_br.get('field_acc')
    cc_field_acc = field_cc.get('field_acc')

    # Recommendation logic
    if br_field_acc is not None and br_field_acc > 0.70:
        recommendation = (
            "DINOv2 features ARE domain-invariant for disease classification.\n"
            "Recommended: Replace Model 2's classification head with a DINOv2-based\n"
            "linear head OR use DINOv2 as Signal 4 in the ensemble MLP. PSV remains\n"
            "useful but is not load-bearing."
        )
        verdict = "POSITIVE — DINOv2 solves the domain shift problem"
    elif br_field_acc is not None and br_field_acc >= 0.50:
        recommendation = (
            "DINOv2 provides partial signal — better than Model 2's real-world\n"
            "performance but not sufficient alone. Recommended: Use DINOv2 as Signal 4\n"
            "in the ensemble MLP. Fix PSV bugs and proceed with ensemble approach."
        )
        verdict = "PARTIAL — DINOv2 helps but needs ensemble support"
    elif br_field_acc is None:
        recommendation = (
            "INCONCLUSIVE — insufficient field photos in val set to measure\n"
            "field-only performance. Need to collect more field val images.\n"
            f"Val F1 for black_rot: {br_f1:.4f} (overall, not field-only).\n"
            "Proceed with ensemble approach (DINOv2 + PSV + Model 2 + EfficientNet)."
        )
        verdict = "INCONCLUSIVE — need more field val data"
    else:
        recommendation = (
            "DINOv2 features cannot separate these classes in a domain-invariant way.\n"
            "The domain shift runs deeper than backbone choice. PSV is the correct\n"
            "primary intervention. Proceed with PSV bug fixes."
        )
        verdict = "NEGATIVE — domain shift is deeper than backbone"

    lines = [
        "=" * 60,
        "DINOV2 LINEAR PROBE EXPERIMENT REPORT",
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "=" * 60,
        "",
        "EXPERIMENT CONFIGURATION:",
        f"  Model: {TIMM_MODEL_NAME}",
        f"  Feature dim: {FEATURE_DIM.get(FEATURE_AGGREGATION, '?')}",
        f"  Aggregation: {FEATURE_AGGREGATION}",
        f"  Best C: {probe_results.get('best_C', '?')}",
        f"  Best scaler: {probe_results.get('best_scaler', '?')}",
        f"  Total elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)",
        "",
        "=" * 60,
        "OVERALL PERFORMANCE",
        "=" * 60,
        f"DINOv2 Linear Probe val macro F1: {val_f1:.4f}",
        f"Model 2 val macro F1 (reference): {MODEL2_VAL_F1['macro']:.4f}",
        f"Delta: {val_f1 - MODEL2_VAL_F1['macro']:+.4f}",
        "",
        "=" * 60,
        "PER-CLASS COMPARISON",
        "=" * 60,
        f"{'Class':<28s} {'Probe F1':>10s} {'Model2 F1':>10s} {'Delta':>8s} {'Winner':>8s}",
        "-" * 68,
    ]
    for cls in CLASS_NAMES:
        r = per_class.get(cls, {})
        pf1 = r.get('probe_f1', 0)
        mf1 = r.get('model2_f1', 0)
        d = r.get('delta', 0)
        w = r.get('winner', '?')
        lines.append(f"{cls:<28s} {pf1:>10.4f} {mf1:>10.4f} {d:>+8.4f} {w:>8s}")

    lines += [
        "",
        "=" * 60,
        "FAILING CLASS ANALYSIS",
        "=" * 60,
    ]
    for cls in FAILURE_CLASSES:
        r = per_class.get(cls, {})
        fa = probe_results.get('field_analysis', {}).get(cls, {})
        lines.append(f"\n{cls}:")
        lines.append(f"  Val F1: {r.get('probe_f1', 0):.4f}")
        lines.append(f"  Val images: {r.get('n_val', 0)}")
        if fa.get('field_acc') is not None:
            lines.append(f"  FIELD-ONLY accuracy: {fa['field_acc']:.4f} (n={fa['n_field']})")
        else:
            lines.append(f"  FIELD-ONLY: {fa.get('note', 'N/A')}")

        # Per-source
        sources = probe_results.get('source_analysis', {}).get(cls, [])
        if sources:
            lines.append(f"  Per-source accuracy:")
            for sb in sources:
                dom = " [DOMINANT]" if sb.get('is_dominant') else ""
                lines.append(f"    {sb['source']:<30s} n={sb['n']:>4d} "
                           f"acc={sb['acc']:.4f}{dom}")

    lines += [
        "",
        "=" * 60,
        "ARCHITECTURAL RECOMMENDATION",
        "=" * 60,
        "",
        f"VERDICT: {verdict}",
        "",
        recommendation,
        "",
        "=" * 60,
        "OOD DETECTION",
        "=" * 60,
        f"Calibrated threshold: {ood_results.get('threshold', 'N/A')}",
        f"Val coverage: {ood_results.get('val_total', 0) - ood_results.get('val_ood_count', 0)}"
        f"/{ood_results.get('val_total', 0)} in-distribution",
    ]

    for name, sr in ood_results.get('synthetic_results', {}).items():
        lines.append(f"  {name}: {sr['rate']*100:.0f}% detected as OOD")

    lines += [
        "",
        "=" * 60,
        "FILES GENERATED",
        "=" * 60,
    ]
    for f in sorted(RESULTS_DIR.glob('*')):
        lines.append(f"  {f.name}")

    report_text = "\n".join(lines)

    report_path = RESULTS_DIR / f'dinov2_probe_report_{ts}.txt'
    with open(report_path, 'w') as f:
        f.write(report_text)

    print(f"\n{'='*60}", flush=True)
    print(f"FINAL REPORT: {report_path}", flush=True)
    print(f"{'='*60}", flush=True)
    print(report_text, flush=True)

    return str(report_path)


def main():
    """Run the complete DINOv2 linear probe experiment."""
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    t_start = time.time()
    print("=" * 60, flush=True)
    print("DINOv2 LINEAR PROBE EXPERIMENT", flush=True)
    print(f"Estimated runtime: ~45-60 minutes (first run)", flush=True)
    print("=" * 60, flush=True)

    # [1/7] Validate
    print("\n[1/7] Validating prerequisites...", flush=True)
    validate_prerequisites()

    # [2/7] Feature extraction
    print("\n[2/7] Feature extraction...", flush=True)
    from scripts.dinov2_probe.feature_cache import run_feature_extraction
    cache = run_feature_extraction()

    # [3/7] Linear probe
    print("\n[3/7] Training linear probe...", flush=True)
    from scripts.dinov2_probe.probe_train import prepare_data, run_probe
    probe_results = run_probe()  # [FIX Issue 11] run_probe calls prepare_data internally

    # Prepare data for analysis/visualization (separate from probe's internal data)
    data = prepare_data(cache)

    # [4/7] Analysis
    print("\n[4/7] Running deep analysis...", flush=True)
    from scripts.dinov2_probe.analysis import run_analysis
    analysis_results = run_analysis(data, probe_results)

    # [5/7] OOD detection
    print("\n[5/7] OOD detection...", flush=True)
    from scripts.dinov2_probe.ood_detection import run_ood_detection
    ood_results = run_ood_detection(data)

    # [6/7] Visualizations
    print("\n[6/7] Generating visualizations...", flush=True)
    from scripts.dinov2_probe.visualise import run_visualisations
    run_visualisations(data, probe_results)

    # [7/7] Final report
    print("\n[7/7] Generating final report...", flush=True)
    elapsed = time.time() - t_start
    report_path = generate_report(probe_results, analysis_results, ood_results, elapsed)

    print(f"\nExperiment complete in {elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)
    print(f"Report: {report_path}", flush=True)


if __name__ == '__main__':
    main()
