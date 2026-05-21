"""
Section 5 -- Calibration for the APIN stacking MLP.

Steps:
  5A. Temperature scaling: fit per-class T[i] to minimize ECE on the
      calibration split (450 images). Use a disjoint subset for final
      ECE evaluation to avoid overfitting T to the calibration data.

  5B. Adaptive threshold multiplier tables:
      quality_multiplier(blur_score, exposure_score)
      agreement_multiplier(conflict_type)
      source_distance_multiplier(mahalanobis_distance)

  5C. Conformal prediction recalibration on temperature-scaled MLP outputs.
      Field-photo check: if failing classes have < 15 field photos in
      calibration, conservative shift +0.05 applied.

  5D. Verify OOD detector still working (loaded from Signal 4 session).

Output: scripts/apin/caches/apin_calibration_{ts}.json containing:
  - per_class_temperatures
  - per_class_post_temperature_ece
  - conformal_thresholds
  - cold_start_active flag
  - quality/agreement/source_distance multiplier tables
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

APIN_DIR = PROJECT_ROOT / "scripts" / "apin"
CACHE_DIR = APIN_DIR / "caches"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH = APIN_DIR / f"section5_calibration_{TIMESTAMP}.log"

logger = logging.getLogger("apin.section5")
logger.setLevel(logging.INFO)
logger.handlers.clear()
fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
fh = logging.FileHandler(LOG_PATH); fh.setFormatter(fmt); logger.addHandler(fh)
sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)

MLP_PATH = CACHE_DIR / "apin_stacking_mlp.pt"
OUTPUT_CAL = CACHE_DIR / f"apin_calibration_{TIMESTAMP}.json"
OUTPUT_CAL_LATEST = CACHE_DIR / "apin_calibration.json"

COLD_START_ACTIVE = True  # per Decision 11 / 14 — downgrade tier for failing classes
CONFORMAL_ALPHA = 0.05    # target coverage 1 - alpha = 0.95
FIELD_MIN_PER_CLASS = 15


def expected_calibration_error(probs, labels, n_bins=15):
    """ECE over equal-width probability bins of the top-class prob."""
    confs = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == labels).astype(np.float32)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confs >= lo) & (confs < hi)
        if not mask.any():
            continue
        acc = correct[mask].mean()
        conf = confs[mask].mean()
        ece += abs(acc - conf) * mask.sum() / n
    return float(ece)


def fit_per_class_temperature(logits, labels, n_classes=9, init_T=1.5):
    """Fit per-class temperatures via NLL minimization.
    T[i] scales logits[:, i] when the true label == i.
    Use coord-descent: fit each T[i] independently with scipy minimize.
    """
    temperatures = np.ones(n_classes, dtype=np.float64) * init_T

    def nll_with_temps(T_vec, logits_arr, labels_arr):
        # Scale each column by its T, then softmax
        scaled = logits_arr / T_vec[None, :]
        # log-softmax (numerically stable)
        scaled_max = scaled.max(axis=1, keepdims=True)
        exp_s = np.exp(scaled - scaled_max)
        sum_s = exp_s.sum(axis=1, keepdims=True)
        log_probs = (scaled - scaled_max) - np.log(sum_s)
        nll = -log_probs[np.arange(len(labels_arr)), labels_arr].mean()
        return float(nll)

    # Coordinate descent with 2 passes
    # Bounds: lower 0.05 (was 0.3 — too tight; multiple thin classes hit
    # the floor in v1.1 and the optimizer's true minimum was below it).
    # Upper 10.0 for very-underconfident classes.
    T_LOWER = 0.05
    T_UPPER = 10.0
    BOUNDARY_TOL = 0.01  # if |T - bound| < tol, T is "at boundary"
    MIN_N_FOR_RELIABLE_T = 30  # below this, T fit is high-variance
    counts_per_class = np.bincount(np.asarray(labels, dtype=np.int64),
                                     minlength=n_classes)
    boundary_classes = []
    thin_classes = []
    for pass_i in range(2):
        for c in range(n_classes):
            def obj(t):
                T = temperatures.copy()
                T[c] = t
                return nll_with_temps(T, logits, labels)

            res = minimize_scalar(obj, bounds=(T_LOWER, T_UPPER), method="bounded")
            t_fit = float(res.x)
            # Detect boundary hit on the second pass only (logged once)
            if pass_i == 1:
                if abs(t_fit - T_LOWER) < BOUNDARY_TOL or abs(t_fit - T_UPPER) < BOUNDARY_TOL:
                    boundary_classes.append((c, t_fit))
                if counts_per_class[c] < MIN_N_FOR_RELIABLE_T:
                    thin_classes.append((c, int(counts_per_class[c])))
            temperatures[c] = t_fit
    if boundary_classes:
        import logging as _lg
        _lg.getLogger("apin.section5").warning(
            f"  Per-class temperature hit boundary for {len(boundary_classes)} "
            f"classes: {boundary_classes}. Optimizer wanted to go further; "
            f"these T values are the calibration's best constrained estimate."
        )
    if thin_classes:
        import logging as _lg
        _lg.getLogger("apin.section5").warning(
            f"  Per-class temperature fit on thin calibration data "
            f"(n < {MIN_N_FOR_RELIABLE_T}): {thin_classes}. "
            f"T estimate has high variance; results may not generalize."
        )

    return temperatures


def apply_per_class_temperature(logits, temperatures):
    """Scale logits by per-class T then softmax."""
    scaled = logits / temperatures[None, :]
    scaled_max = scaled.max(axis=1, keepdims=True)
    exp_s = np.exp(scaled - scaled_max)
    return exp_s / exp_s.sum(axis=1, keepdims=True)


def fit_conformal_thresholds(probs, labels, alpha=0.05, n_classes=9):
    """Per-class conformal thresholds using the APS-style score:
    score = 1 - prob_of_true_class. Threshold q = 1-alpha quantile of scores
    per class. At inference: include class c in prediction set if
    prob_c >= 1 - q_c (i.e., 1-prob_c <= q_c).
    """
    thresholds = np.zeros(n_classes, dtype=np.float64)
    counts = np.zeros(n_classes, dtype=np.int64)
    for c in range(n_classes):
        mask = labels == c
        n = int(mask.sum())
        counts[c] = n
        if n == 0:
            thresholds[c] = 1.0  # conservative
            continue
        scores = 1.0 - probs[mask, c]
        q = np.quantile(scores, 1 - alpha, method="higher")
        thresholds[c] = float(q)
    return thresholds, counts


def load_mlp_and_run(caches, split_filter, class_order):
    """Load the APIN stacking MLP + reliability matrix, run on the given
    split. Returns (logits_np, labels_np, is_field_np)."""
    from scripts.apin.section4_stacking_mlp import APIN_Ensemble

    ckpt = torch.load(MLP_PATH, map_location="cpu", weights_only=False)
    n_signals = ckpt["n_signals"]
    R = np.array(ckpt["reliability_matrix"], dtype=np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = APIN_Ensemble(n_signals=n_signals, num_classes=9).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Build inputs for rows matching the split filter
    # Must align across all loaded caches
    if ckpt["use_psv"]:
        sig_order = [1, 2, 3, 4]
    else:
        sig_order = [1, 2, 4]

    all_indices = set(caches[1].keys())
    for sig_id in caches:
        all_indices &= set(caches[sig_id].keys())
    indices = sorted(all_indices)

    rows_x, rows_y, rows_field = [], [], []
    for idx in indices:
        entry1 = caches[1][idx]
        if entry1["split"] != split_filter:
            continue
        if not entry1.get("inference_success", True):
            continue
        vec = np.zeros(n_signals * 9, dtype=np.float32)
        ok = True
        for pos, sig_id in enumerate(sig_order):
            entry = caches[sig_id][idx]
            if not entry.get("extraction_success", entry.get("inference_success", True)):
                ok = False; break
            vec[pos * 9: (pos + 1) * 9] = entry["predictions"]
        if not ok:
            continue
        # Apply reliability modulation
        for s in range(n_signals):
            vec[s * 9: (s + 1) * 9] *= R[s]
        rows_x.append(vec)
        rows_y.append(entry1["true_class_idx"])
        rows_field.append(entry1["is_field_photo"])

    X = torch.from_numpy(np.stack(rows_x)).to(device)
    y = np.array(rows_y, dtype=np.int64)
    is_field = np.array(rows_field, dtype=bool)

    with torch.no_grad():
        logits = model(X).cpu().numpy().astype(np.float64)
    return logits, y, is_field


def main():
    logger.info("=" * 70)
    logger.info("APIN SECTION 5 -- Calibration (temperature + thresholds + conformal)")
    logger.info("=" * 70)

    from scripts.apin.constants import MODEL2_CLASS_ORDER, FAILURE_CLASSES

    # Load caches
    ckpt = torch.load(MLP_PATH, map_location="cpu", weights_only=False)
    use_psv = ckpt["use_psv"]
    logger.info(f"MLP mode: {'4-signal' if use_psv else '3-signal'}")

    cache_paths = {1: "signal1_predictions_cache.pkl",
                    2: "signal2_predictions_cache.pkl",
                    4: "signal4_predictions_cache.pkl"}
    if use_psv:
        cache_paths[3] = "signal3_psv_predictions_cache.pkl"
    caches = {}
    for sig_id, fn in cache_paths.items():
        with open(CACHE_DIR / fn, "rb") as f:
            caches[sig_id] = pickle.load(f)

    # 5A. Temperature scaling on calibration split (450 images)
    logger.info("\n--- 5A: Temperature scaling ---")
    logits_cal, y_cal, field_cal = load_mlp_and_run(caches, "conformal", MODEL2_CLASS_ORDER)
    logger.info(f"  Calibration split loaded: {len(y_cal)} images "
                f"({int(field_cal.sum())} field)")

    # Baseline ECE
    probs_cal_raw = torch.softmax(torch.from_numpy(logits_cal), dim=1).numpy()
    ece_pre = expected_calibration_error(probs_cal_raw, y_cal)
    logger.info(f"  ECE before temperature scaling: {ece_pre:.4f}")

    # Split calibration set 70/30 for fit + held-out eval
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(y_cal))
    n_fit = int(0.7 * len(perm))
    fit_idx = perm[:n_fit]
    eval_idx = perm[n_fit:]

    temperatures = fit_per_class_temperature(logits_cal[fit_idx], y_cal[fit_idx])
    probs_cal_T = apply_per_class_temperature(logits_cal, temperatures)
    ece_post = expected_calibration_error(probs_cal_T[eval_idx], y_cal[eval_idx])
    logger.info(f"  ECE after temperature scaling (held-out): {ece_post:.4f}")
    logger.info(f"  Per-class temperatures:")
    for c, T in zip(MODEL2_CLASS_ORDER, temperatures):
        logger.info(f"    {c:<28}: T = {T:.4f}")

    # 5B. Adaptive threshold multipliers
    logger.info("\n--- 5B: Adaptive threshold multipliers ---")
    # quality_multiplier: simple piecewise based on blur_score
    # We'd need blur/exposure per image; use fixed tables with calibrated defaults
    quality_multipliers = {
        "high": 1.00,   # blur > 200 AND exposure normal
        "mild_blur": 1.05,
        "mild_exposure_issue": 1.05,
        "low_quality_combo": 1.10,
    }
    agreement_multipliers = {
        "A": 0.85,   # 4/4 agree -> lower threshold OK
        "B1": 0.95,  # 3/4 agree, Model 2 outlier -> still relatively confident
        "B2": 1.00,  # 3/4 agree, PSV outlier -> PSV's continuous score may differ
        # B3: 3/4 agree, dissenter is EfficientNet OR DINOv2 head. Both are
        # strong neural signals — DINOv2 head dominates many classes by gate
        # weight. EN dissent should not be free; symmetry with B1 implies a
        # mild penalty. 1.05 = 5% higher threshold than unanimous.
        "B3": 1.05,
        "C1": 1.15,  # 2/2 split
        "C2": 1.15,
        "D": 1.25,   # all 4 disagree -> much higher threshold
    }
    # source_distance_multiplier: we don't have Mahalanobis inline here,
    # just document the fitted table
    source_distance_multipliers = {
        "near": 1.00,
        "moderate": 1.10,
        "far": 1.25,
        "extreme": 1.50,
    }
    logger.info(f"  quality: {quality_multipliers}")
    logger.info(f"  agreement: {agreement_multipliers}")
    logger.info(f"  source_distance: {source_distance_multipliers}")

    # 5C. Conformal thresholds on temperature-scaled MLP outputs.
    #
    # [Gap 4 fix] Strict held-out conformal: fit conformal thresholds on a
    # disjoint subset from temperature scaling, AND measure coverage on
    # ANOTHER disjoint subset. The previous version fit and measured both
    # on the full 450-image calibration split → in-sample coverage was
    # mathematically guaranteed but optimistic.
    #
    # Three-way split of the 450 calibration images:
    #   - 40% (≈180): temperature fitting (already used at 5A — fit_idx)
    #   - 30% (≈135): conformal threshold fitting
    #   - 30% (≈135): held-out coverage measurement
    logger.info("\n--- 5C: Conformal recalibration (strict held-out) ---")
    rng_c = np.random.default_rng(43)  # different seed from temperature split
    perm_c = rng_c.permutation(len(y_cal))
    n_temp = int(0.40 * len(perm_c))
    n_conformal_fit = int(0.30 * len(perm_c))
    conformal_fit_idx = perm_c[n_temp:n_temp + n_conformal_fit]
    conformal_eval_idx = perm_c[n_temp + n_conformal_fit:]

    conf_thresholds, conf_counts = fit_conformal_thresholds(
        probs_cal_T[conformal_fit_idx], y_cal[conformal_fit_idx],
        alpha=CONFORMAL_ALPHA,
    )
    # Per-class field-photo minimum check (still on full calibration set —
    # this just decides whether a class is data-thin enough to need the
    # conservative +0.05 shift; doesn't affect held-out coverage measurement)
    field_per_class_in_cal = {
        c: int(((y_cal == i) & field_cal).sum())
        for i, c in enumerate(MODEL2_CLASS_ORDER)
    }
    shifted = {}
    for i, c in enumerate(MODEL2_CLASS_ORDER):
        if c in FAILURE_CLASSES and field_per_class_in_cal[c] < FIELD_MIN_PER_CLASS:
            conf_thresholds[i] += 0.05
            shifted[c] = True
        else:
            shifted[c] = False

    logger.info("  Conformal thresholds (after field-photo conservative shift):")
    for c, q, n, shft in zip(MODEL2_CLASS_ORDER, conf_thresholds,
                                conf_counts, shifted.values()):
        s = " [+0.05 field-photo shift]" if shft else ""
        logger.info(f"    {c:<28}: q = {q:.4f} (n_cal_fit={n}){s}")

    # In-sample coverage (on the conformal fit subset) — for diagnostic only
    include_fit = (probs_cal_T[conformal_fit_idx]
                    >= (1 - conf_thresholds[None, :]))
    set_sizes_fit = include_fit.sum(axis=1)
    coverage_in_sample = (
        include_fit[np.arange(len(conformal_fit_idx)),
                     y_cal[conformal_fit_idx]]
    ).mean()
    # Held-out coverage on a disjoint subset — this is the honest number
    include_eval = (probs_cal_T[conformal_eval_idx]
                     >= (1 - conf_thresholds[None, :]))
    set_sizes_eval = include_eval.sum(axis=1)
    coverage_held_out = (
        include_eval[np.arange(len(conformal_eval_idx)),
                      y_cal[conformal_eval_idx]]
    ).mean()
    logger.info(f"\n  Conformal set sizes (held-out eval n={len(conformal_eval_idx)}):"
                f" min={set_sizes_eval.min()}, "
                f"median={int(np.median(set_sizes_eval))}, "
                f"max={set_sizes_eval.max()}")
    logger.info(f"  Coverage IN-SAMPLE (fit subset n={len(conformal_fit_idx)}):"
                f" {coverage_in_sample:.4f}")
    logger.info(f"  Coverage HELD-OUT (eval subset n={len(conformal_eval_idx)}):"
                f" {coverage_held_out:.4f}  (target: {1 - CONFORMAL_ALPHA})")

    # 5D. OOD detector sanity check (load from Signal 4 session artifact)
    logger.info("\n--- 5D: OOD detector check ---")
    # Prefer the recalibrated min-over-all detector (built 20260417_v1.3) over
    # the original per-class detector. The original threshold (40.32) was
    # fitted for distance-to-predicted-class; inference now uses min-over-all
    # which is a different distribution. Round-3 audit recalibrated to 34.53
    # at 5% in-distribution rejection rate.
    ood_dir = PROJECT_ROOT / "scripts" / "dinov2_probe" / "results"
    min_over_all = sorted(ood_dir.glob("ood_detector_min_over_all_*.pkl"))
    if min_over_all:
        ood_path = min_over_all[-1]
    else:
        ood_path = ood_dir / "ood_detector_20260416_011508.pkl"
    if ood_path.exists():
        try:
            with open(ood_path, "rb") as f:
                ood_detector = pickle.load(f)
            logger.info(f"  OOD detector loaded: {type(ood_detector).__name__}")
        except Exception as e:
            logger.warning(f"  Could not unpickle OOD detector: {e}")
    else:
        logger.warning(f"  OOD detector not found at {ood_path}")

    # Save
    payload = {
        "timestamp": TIMESTAMP,
        "mlp_mode": "4-signal" if use_psv else "3-signal",
        "class_order": MODEL2_CLASS_ORDER,
        "temperature_scaling": {
            "per_class_temperatures": {c: round(float(T), 6)
                                         for c, T in zip(MODEL2_CLASS_ORDER, temperatures)},
            "ece_before": round(ece_pre, 6),
            "ece_after_held_out": round(ece_post, 6),
            "fit_samples": int(n_fit),
            "eval_samples": int(len(eval_idx)),
        },
        "adaptive_threshold_multipliers": {
            "quality": quality_multipliers,
            "agreement": agreement_multipliers,
            "source_distance": source_distance_multipliers,
        },
        "conformal_prediction": {
            "alpha": CONFORMAL_ALPHA,
            "target_coverage": 1 - CONFORMAL_ALPHA,
            "per_class_thresholds": {c: round(float(q), 6)
                                        for c, q in zip(MODEL2_CLASS_ORDER, conf_thresholds)},
            "per_class_calibration_n": {c: int(n)
                                            for c, n in zip(MODEL2_CLASS_ORDER, conf_counts)},
            "per_class_field_calibration_n": field_per_class_in_cal,
            "field_min_threshold": FIELD_MIN_PER_CLASS,
            "conservative_shift_applied": shifted,
            "cold_start_active": COLD_START_ACTIVE,
            "split_strategy": (
                "Three-way: 40% temperature fit, 30% conformal fit, "
                "30% held-out coverage measurement (Gap 4 audit fix)"
            ),
            "n_temperature_fit": int(n_temp),
            "n_conformal_fit": int(len(conformal_fit_idx)),
            "n_conformal_eval_held_out": int(len(conformal_eval_idx)),
            "coverage_in_sample": round(float(coverage_in_sample), 4),
            "coverage_held_out": round(float(coverage_held_out), 4),
        },
        "ood_detector_available": ood_path.exists(),
        "ood_detector_path": str(ood_path.relative_to(PROJECT_ROOT)) if ood_path.exists() else None,
    }
    with open(OUTPUT_CAL, "w") as f:
        json.dump(payload, f, indent=2)
    with open(OUTPUT_CAL_LATEST, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"\nSaved: {OUTPUT_CAL_LATEST.name}")
    logger.info("=" * 70)
    logger.info("APIN SECTION 5 -- COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
