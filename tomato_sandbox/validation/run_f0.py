"""
F.0 validation script for the Tomato 3-Signal sandbox.

Drives a held-out test set through the full pipeline (predict_single),
computes validation metrics, and emits a JSON validation report.

Entry point:
    run_f0_validation(labeled_data_path, pipeline_context, output_dir, calibration_dir)

(β) interpretation per DEC-047 / DEC-053: this script consumes pipeline outputs
via predict_single from the orchestrator.  It does NOT load model weights directly.
Zero references to torch.load, model3_production_v3.pt, or sp_lora_epoch13.

Spec references:
# spec: section 29 lines 8105-8243 — F.0 validation suite (primary contract)
# spec: section 29.3 lines 8138-8171 — F.0 procedure Step 3 (evaluate on test set)
# spec: section 29.4 lines 8173-8200 — quality bars
# spec: section 13.6 lines 3621-3635 — empirical coverage monitoring
# spec: section 17.3 lines 5966-5982 — per-disease severity thresholds
# spec: section 17.5 lines 6015-6033 — multi-class severity
# spec: section 16.2 lines 5655-5712 — response schema (the shape we parse)

DEC-053: all architectural decisions for this module.
"""

from __future__ import annotations

import csv
import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from tomato_sandbox.utils.logging import get_logger

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level import of predict_single — placed here so unittest.mock.patch
# can address it as "tomato_sandbox.validation.run_f0.predict_single".
# A deferred (in-function) import creates only a local name, not a module
# attribute, which causes AttributeError at patch time.
# DEC-053 Decision 3 — orchestrator shim path.
# ---------------------------------------------------------------------------
try:
    from tomato_sandbox.orchestrator.orchestrator import predict_single  # noqa: F401
except ImportError:  # pragma: no cover — orchestrator may not be importable in test envs
    predict_single = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Canonical class names (spec S12.1 lines 3151-3159 / S16.2)
_CLASS_NAMES: list[str] = [
    "foliar",       # 0
    "septoria",     # 1
    "late_blight",  # 2
    "ylcv",         # 3
    "mosaic",       # 4
    "healthy",      # 5
    "OOD",          # 6
]
_CLASS_TO_IDX: dict[str, int] = {n: i for i, n in enumerate(_CLASS_NAMES)}
_NUM_CLASSES: int = 7  # spec: section 13.7 line 3642

# Disease names (excludes healthy, OOD — for severity validation)
# spec: section 17.3 lines 5972-5979
_DISEASE_NAMES: list[str] = ["foliar", "septoria", "late_blight", "ylcv", "mosaic"]

# Canonical conformal coverage target
# spec: section 13.4 line 3567 — "Coverage target: 90%"
_CONFORMAL_ALPHA: float = 0.10  # spec: section 13.2 line 3538

# Quality bar for conformal coverage (spec S29.4 line 8183):
# "Conformal empirical coverage | 88-92% | 85-95%"
_COVERAGE_TARGET_LOW: float = 0.88   # spec: section 29.4 line 8183
_COVERAGE_TARGET_HIGH: float = 0.92  # spec: section 29.4 line 8183
_COVERAGE_FLOOR_LOW: float = 0.85    # spec: section 29.4 line 8183
_COVERAGE_FLOOR_HIGH: float = 0.95   # spec: section 29.4 line 8183

# Tier 4B rate quality bar (spec S29.4 line 8185):
# "Tier 4B rate | < 1% | < 3%"
_TIER4B_TARGET: float = 0.01         # spec: section 29.4 line 8185
_TIER4B_FLOOR: float = 0.03          # spec: section 29.4 line 8185

# Canonical output directory
# spec: section 13.5 line 3602 — "tomato_sandbox/phase_f0_calibration/"
_SANDBOX_ROOT: Path = Path(__file__).resolve().parents[1]
_DEFAULT_OUTPUT_DIR: Path = _SANDBOX_ROOT / "phase_f0_calibration"

# Calibration artifact filenames (spec S13.5)
_CONFORMAL_TAU_FILE: str = "conformal_tau.json"
_PSV_STANDARDIZATION_FILE: str = "psv_standardization.json"

# ---------------------------------------------------------------------------
# Internal helpers — response parsing
# ---------------------------------------------------------------------------


def _get_tier_label(response: dict) -> str:
    """Extract tier label string from S16.2 response.

    spec: section 16.2 lines 5664-5668 — 'tier': {'label': '1', ...}
    """
    return str(response.get("tier", {}).get("label", "UNKNOWN"))


def _get_prediction_set(response: dict) -> list[str]:
    """Extract prediction_set (class name list) from S16.2 response.

    spec: section 16.2 lines 5670-5675 — 'prediction': {'prediction_set': [...]}
    """
    pred = response.get("prediction", {})
    if pred is None:
        return []
    return list(pred.get("prediction_set") or [])


def _get_primary_class(response: dict) -> str | None:
    """Extract primary_class from S16.2 response.

    spec: section 16.2 line 5670 — 'prediction': {'primary_class': 'foliar', ...}
    """
    pred = response.get("prediction", {})
    if pred is None:
        return None
    return pred.get("primary_class")


def _get_rule_id_fired(response: dict) -> str | None:
    """Extract rule_id_fired from explanation.structured.

    spec: section 16.4 line 5756 — 'explanation': {'structured': {'rule_id_fired': ...}}
    """
    explanation = response.get("explanation") or {}
    structured = explanation.get("structured") or {}
    return structured.get("rule_id_fired")


def _get_severity_grade(response: dict) -> str | None:
    """Extract severity.grade from S16.2 response.

    spec: section 17.4 lines 5992-6010 — severity block in response
    """
    sev = response.get("severity")
    if sev is None:
        return None
    return sev.get("grade")


def _is_tier_4b_degraded(response: dict) -> bool:
    """Return True if this is a Tier 4B caused by degraded-mode (all signals failed).

    Detection logic per DEC-053 Decision 4:
    - tier.label == "4B"
    - rule_id_fired is one of the pipeline_failure sentinels (Rule 1 or "pipeline_failure")
    - These indicate the all-signals-failed short-circuit path
      (spec S21.5 lines 6745-6755 → _make_sentinel_classifier_result)

    spec: section 14.5 Rule 1 — pipeline failure → Tier 4B
    spec: section 21.5 lines 6745-6755 — all-signals-failed short-circuit
    """
    if _get_tier_label(response) != "4B":
        return False
    rule_id = _get_rule_id_fired(response)
    # "1" is the numeric rule id; "pipeline_failure" is the string variant used in
    # _make_sentinel_classifier_result  (spec S21.5 line 6750)
    return rule_id in {"1", "pipeline_failure"}


def _is_error_response(response: dict) -> bool:
    """Return True if the response dict is a pipeline error (not a tier response).

    spec: section 21.8 lines 6810-6826 — error responses include 'error' key
    """
    return "error" in response


# ---------------------------------------------------------------------------
# Wilson confidence interval
# spec: section 13.3 line 3561 — "The standard error of a binomial proportion at
# p=0.9, n=40 is sqrt(0.9 × 0.1 / 40) ≈ 0.047"
# DEC-053 Decision 5 — report Wilson 95% CI as informational
# ---------------------------------------------------------------------------

def _wilson_ci_95(n_success: int, n_total: int) -> tuple[float, float]:
    """Compute Wilson score 95% confidence interval for a proportion.

    Returns (lower, upper) as floats in [0, 1].
    If n_total == 0, returns (0.0, 1.0) as degenerate interval.

    DEC-053 Decision 5 — Wilson CI is informational alongside empirical rate.
    """
    if n_total == 0:
        return 0.0, 1.0
    z = 1.96  # 95% two-sided z
    p_hat = n_success / n_total
    n = n_total
    center = (p_hat + z * z / (2 * n)) / (1 + z * z / n)
    half_width = (z / (1 + z * z / n)) * math.sqrt(
        p_hat * (1 - p_hat) / n + z * z / (4 * n * n)
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


# ---------------------------------------------------------------------------
# Confusion matrix
# spec: section 29.5 line 8207 — "Confusion matrix per class"
# ---------------------------------------------------------------------------

def _build_confusion_matrix(
    true_classes: list[int],
    pred_classes: list[int],
    n_classes: int = _NUM_CLASSES,
) -> list[list[int]]:
    """Build n_classes × n_classes confusion matrix.

    Rows = true class, Columns = predicted class (argmax of prediction_set
    or -1 if no prediction made / error response).

    spec: section 29.5 line 8207 — "Confusion matrix per class"
    """
    cm = [[0] * n_classes for _ in range(n_classes)]
    for y_true, y_pred in zip(true_classes, pred_classes):
        if 0 <= y_true < n_classes and 0 <= y_pred < n_classes:
            cm[y_true][y_pred] += 1
    return cm


# ---------------------------------------------------------------------------
# Conformal coverage
# spec: section 13.4 lines 3564-3581 — coverage target 90%
# spec: section 13.6 lines 3621-3635 — empirical coverage monitoring
# ---------------------------------------------------------------------------

def _compute_conformal_coverage(
    true_classes: list[int],
    prediction_sets: list[list[str]],
) -> dict[str, Any]:
    """Compute empirical conformal coverage rate and quality assessment.

    Coverage = fraction of images where true class is in prediction_set.

    spec: section 13.4 line 3570 — "P(y_true ∈ PredSet) ≥ 1 - α = 0.90"
    spec: section 29.4 line 8183 — "Conformal empirical coverage | 88-92% | 85-95%"
    DEC-053 Decision 5 — report Wilson 95% CI as informational.

    Args:
        true_classes: list of integer class indices (0-6); -1 = unknown/skip
        prediction_sets: parallel list of prediction set (class name strings)

    Returns:
        dict with keys: n_total, n_covered, coverage_rate,
        coverage_ci_95_wilson, quality_bar_met, floor_met, target
    """
    n_total = 0
    n_covered = 0
    for y_true, pred_set in zip(true_classes, prediction_sets):
        if y_true < 0:
            continue  # unknown true class; skip
        n_total += 1
        true_name = _CLASS_NAMES[y_true] if 0 <= y_true < _NUM_CLASSES else ""
        if true_name and true_name in pred_set:
            n_covered += 1

    if n_total == 0:
        return {
            "n_total": 0,
            "n_covered": 0,
            "coverage_rate": None,
            "coverage_ci_95_wilson": [None, None],
            # spec: section 29.4 line 8183
            "quality_bar_target_range": [_COVERAGE_TARGET_LOW, _COVERAGE_TARGET_HIGH],
            "quality_bar_floor_range": [_COVERAGE_FLOOR_LOW, _COVERAGE_FLOOR_HIGH],
            "quality_bar_met": None,
            "floor_met": None,
        }

    rate = n_covered / n_total
    ci_lo, ci_hi = _wilson_ci_95(n_covered, n_total)

    # spec: section 29.4 line 8183 quality bar assessment
    quality_bar_met = _COVERAGE_TARGET_LOW <= rate <= _COVERAGE_TARGET_HIGH
    floor_met = _COVERAGE_FLOOR_LOW <= rate <= _COVERAGE_FLOOR_HIGH

    return {
        "n_total": n_total,
        "n_covered": n_covered,
        "coverage_rate": round(rate, 6),
        "coverage_ci_95_wilson": [round(ci_lo, 6), round(ci_hi, 6)],
        # spec: section 29.4 line 8183
        "quality_bar_target_range": [_COVERAGE_TARGET_LOW, _COVERAGE_TARGET_HIGH],
        "quality_bar_floor_range": [_COVERAGE_FLOOR_LOW, _COVERAGE_FLOOR_HIGH],
        "quality_bar_met": quality_bar_met,
        "floor_met": floor_met,
    }


# ---------------------------------------------------------------------------
# Per-disease severity validation
# spec: section 17.3 lines 5966-5982 — per-disease severity thresholds
# spec: section 17.5 lines 6015-6033 — multi-class severity for Tier 3A/3B
# ---------------------------------------------------------------------------

def _compute_severity_validation(
    per_image: list[dict],
) -> dict[str, Any]:
    """Validate per-disease severity predictions against ground-truth grades.

    spec: section 17.3 lines 5966-5982 — per-disease severity thresholds
    spec: section 17.5 lines 6015-6033 — multi-class severity for Tier 3A/3B

    Args:
        per_image: list of per-image dicts; each has:
            true_class (str), true_severity (str, may be ""), pred_severity (str | None)

    Returns:
        dict with status ("ok" | "skipped") and per-disease accuracy stats, or
        {"status": "skipped", "reason": "skipped_no_ground_truth"} if no GT.
    """
    # Collect rows that have valid ground-truth severity
    gt_rows = [
        r for r in per_image
        if r.get("true_severity") in ("mild", "moderate", "severe")
        and r.get("true_class") in _DISEASE_NAMES
    ]

    if not gt_rows:
        # spec DEC-053 Decision 6 — skip with explicit reason
        return {
            "status": "skipped",
            "reason": "skipped_no_ground_truth",
            "note": "No rows with true_severity in {mild, moderate, severe} for disease classes",
        }

    # Per-disease accuracy
    per_disease: dict[str, dict[str, int]] = {
        d: {"n_total": 0, "n_correct": 0} for d in _DISEASE_NAMES
    }

    for row in gt_rows:
        d = row["true_class"]
        gt = row["true_severity"]
        pred = row.get("pred_severity")
        per_disease[d]["n_total"] += 1
        if pred == gt:
            per_disease[d]["n_correct"] += 1

    per_disease_stats: dict[str, Any] = {}
    for d, counts in per_disease.items():
        n = counts["n_total"]
        correct = counts["n_correct"]
        acc = (correct / n) if n > 0 else None
        per_disease_stats[d] = {
            "n_total": n,
            "n_correct": correct,
            "accuracy": round(acc, 6) if acc is not None else None,
        }

    # Overall
    n_all = sum(c["n_total"] for c in per_disease.values())
    n_correct_all = sum(c["n_correct"] for c in per_disease.values())
    overall_acc = (n_correct_all / n_all) if n_all > 0 else None

    return {
        "status": "ok",
        # spec: section 17.3 — per-disease validation
        "per_disease": per_disease_stats,
        "overall_n": n_all,
        "overall_n_correct": n_correct_all,
        "overall_accuracy": round(overall_acc, 6) if overall_acc is not None else None,
    }


# ---------------------------------------------------------------------------
# Tier disposition tracking
# DEC-053 Decision 4 — Tier 4B degraded vs real failure
# ---------------------------------------------------------------------------

def _compute_tier_disposition(responses: list[dict]) -> dict[str, Any]:
    """Summarize tier distribution and Tier 4B classification.

    DEC-053 Decision 4:
    - tier_4b_count_degraded: rule_id_fired in {"1", "pipeline_failure"}
      (all-signals-failed short-circuit, expected in pre-F.0 mode)
    - tier_4b_count_real_failure: Tier 4B with a different rule_id (genuine
      pipeline bug post-Component-C)

    spec: section 29.4 line 8185 — "Tier 4B rate | < 1% | < 3%"
    spec: section 21.5 lines 6745-6755 — all-signals-failed → sentinel Tier 4B
    """
    tier_counts: dict[str, int] = {}
    tier_4b_total = 0
    tier_4b_degraded = 0
    tier_4b_real_failure = 0
    error_count = 0

    for resp in responses:
        if _is_error_response(resp):
            error_count += 1
            tier_counts["ERROR"] = tier_counts.get("ERROR", 0) + 1
            continue

        label = _get_tier_label(resp)
        tier_counts[label] = tier_counts.get(label, 0) + 1

        if label == "4B":
            tier_4b_total += 1
            if _is_tier_4b_degraded(resp):
                tier_4b_degraded += 1
            else:
                tier_4b_real_failure += 1

    n_total = len(responses)
    tier_4b_rate = tier_4b_total / n_total if n_total > 0 else None

    # spec: section 29.4 line 8185 quality bars
    tier_4b_target_met = (tier_4b_rate is not None and tier_4b_rate < _TIER4B_TARGET)
    tier_4b_floor_met = (tier_4b_rate is not None and tier_4b_rate < _TIER4B_FLOOR)

    return {
        "tier_counts": tier_counts,
        "tier_4b_count_total": tier_4b_total,
        # DEC-053 Decision 4 — degraded-mode 4B (expected in pre-F.0)
        "tier_4b_count_degraded": tier_4b_degraded,
        # DEC-053 Decision 4 — real pipeline failure 4B (indicates a bug post-C)
        "tier_4b_count_real_failure": tier_4b_real_failure,
        "tier_4b_rate": round(tier_4b_rate, 6) if tier_4b_rate is not None else None,
        "error_count": error_count,
        # spec: section 29.4 line 8185 quality bars
        "quality_bar_target": _TIER4B_TARGET,
        "quality_bar_floor": _TIER4B_FLOOR,
        "quality_bar_target_met": tier_4b_target_met,
        "quality_bar_floor_met": tier_4b_floor_met,
        # Informational flag for pre-F.0 detection
        "all_4b_are_degraded": (
            tier_4b_total > 0 and tier_4b_degraded == tier_4b_total
        ),
        "is_pre_f0_mode": (
            n_total > 0 and tier_4b_total == n_total
            and tier_4b_degraded == n_total
        ),
    }


# ---------------------------------------------------------------------------
# Calibration artifacts reader
# DEC-053 Decision 7
# ---------------------------------------------------------------------------

def _read_calibration_artifacts(calibration_dir: Path) -> dict[str, Any]:
    """Read conformal_tau.json and psv_standardization.json from calibration_dir.

    DEC-053 Decision 7 — surface calibration artifact contents in report metadata.
    Returns dict with "conformal_tau" and "psv_standardization" keys.
    If a file does not exist, value is "not_found".
    """
    artifacts: dict[str, Any] = {}

    for fname, key in [
        (_CONFORMAL_TAU_FILE, "conformal_tau"),
        (_PSV_STANDARDIZATION_FILE, "psv_standardization"),
    ]:
        fpath = calibration_dir / fname
        if fpath.exists():
            try:
                with fpath.open("r", encoding="utf-8") as fh:
                    artifacts[key] = json.load(fh)
            except (json.JSONDecodeError, OSError) as exc:
                artifacts[key] = {"error": f"parse_error: {exc}"}
        else:
            artifacts[key] = "not_found"

    return artifacts


# ---------------------------------------------------------------------------
# CSV loader (same schema as DEC-052 Decision 6)
# ---------------------------------------------------------------------------

def _load_labeled_csv(labeled_data_path: Path) -> list[dict[str, str]]:
    """Load labeled data CSV. Expected columns per DEC-052 Decision 6.

    Required: image_path, true_class, split
    Optional: true_severity, is_confirmed_tomato

    DEC-053 Decision 2 — same CSV schema as run_full_calibration.
    """
    records: list[dict[str, str]] = []
    with labeled_data_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            records.append(dict(row))
    return records


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_f0_validation(
    labeled_data_path: Path,
    pipeline_context: Any,
    output_dir: Optional[Path] = None,
    calibration_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Run F.0 dry-run validation.

    Drives the held-out test set through predict_single, aggregates per-image
    results, and computes:
      - Confusion matrix (spec S29.5 line 8207)
      - Conformal coverage + Wilson CI (spec S13.4 / S29.4)
      - Per-disease severity validation (spec S17.3 / S17.5)
      - Tier 4B disposition: degraded vs real-failure (DEC-053 Decision 4)
      - Calibration artifact metadata (DEC-053 Decision 7)

    Writes JSON report to <output_dir>/validation_report_<ISO_TIMESTAMP>.json.

    (β) interpretation per DEC-047 / DEC-053 Decision 9:
    - Does NOT load model weights directly.
    - Calls predict_single from the orchestrator only.
    - In pre-F.0 mode, all predictions will be Tier 4B-degraded; the report
      notes this via 'tier_disposition.is_pre_f0_mode'.

    spec: section 29.3 lines 8138-8171 — F.0 procedure (primary contract)
    spec: section 29.4 lines 8173-8200 — quality bars
    spec: section 13.4 lines 3564-3581 — coverage target 90%
    spec: section 17.3 lines 5966-5982 — per-disease severity thresholds
    spec: section 16.2 lines 5655-5712 — response schema (shape we parse)

    Args:
        labeled_data_path: Path to CSV with columns per DEC-052 Decision 6.
            Rows with split == "test" are used for evaluation.
            DEC-053 Decision 2 — same CSV schema as run_full_calibration.
        pipeline_context: PipelineContext from orchestrator.
            spec: section 21.2 line 6614 — predict_single(image_bytes, request_id, ctx)
        output_dir: Directory for validation_report_*.json. Defaults to
            phase_f0_calibration/. Tests MUST pass tmp_path here.
            DEC-053 Decision 8.
        calibration_dir: Directory containing calibration artifacts (conformal_tau.json,
            psv_standardization.json). Defaults to output_dir (same directory).
            DEC-053 Decision 7.

    Returns:
        Validation report dict with keys:
          metadata, per_image_predictions, confusion_matrix,
          conformal_coverage, severity_validation, tier_disposition.

    Raises:
        FileNotFoundError: If labeled_data_path does not exist.
        ValueError: If CSV has no test-split rows.
    """
    labeled_data_path = Path(labeled_data_path)
    if not labeled_data_path.exists():
        raise FileNotFoundError(
            f"run_f0_validation: labeled data file not found: {labeled_data_path}"
        )

    # Resolve directories
    resolved_output_dir = Path(output_dir) if output_dir is not None else _DEFAULT_OUTPUT_DIR
    resolved_calib_dir = (
        Path(calibration_dir) if calibration_dir is not None else resolved_output_dir
    )

    _log.info(
        "run_f0_validation: starting",
        labeled_data_path=str(labeled_data_path),
        output_dir=str(resolved_output_dir),
        calibration_dir=str(resolved_calib_dir),
    )

    # --- Load CSV ---
    records = _load_labeled_csv(labeled_data_path)
    test_records = [r for r in records if r.get("split") == "test"]

    if not test_records:
        raise ValueError(
            f"run_f0_validation: no rows with split='test' in {labeled_data_path}. "
            f"Total rows: {len(records)}. "
            f"DEC-053 Decision 2: use split='test' for validation rows."
        )

    _log.info(
        "run_f0_validation: test set loaded",
        n_total=len(records),
        n_test=len(test_records),
    )

    # --- Run pipeline over test set ---
    # spec: section 29.3 step 3 lines 8153-8156 — "Run pipeline over test set"
    per_image_predictions: list[dict[str, Any]] = []
    all_responses: list[dict] = []
    true_class_indices: list[int] = []
    pred_class_indices: list[int] = []
    true_classes_for_coverage: list[int] = []
    prediction_sets_for_coverage: list[list[str]] = []

    n_processed = 0
    n_errors = 0

    for i, record in enumerate(test_records):
        image_path_str: str = record.get("image_path", "")
        true_class_str: str = record.get("true_class", "")
        true_class_idx = _CLASS_TO_IDX.get(true_class_str, -1)
        true_severity_str: str = record.get("true_severity", "")

        # Resolve image path
        img_path = Path(image_path_str)
        if not img_path.is_absolute():
            img_path = labeled_data_path.parent / img_path

        try:
            image_bytes = img_path.read_bytes()
        except (OSError, FileNotFoundError) as exc:
            _log.warning(
                "run_f0_validation: cannot read image; recording as error",
                image_path=str(img_path),
                error=str(exc),
            )
            n_errors += 1
            error_row = {
                "image_path": image_path_str,
                "true_class": true_class_str,
                "true_severity": true_severity_str,
                "error": f"image_read_error: {exc}",
                "tier_label": None,
                "pred_class": None,
                "pred_severity": None,
                "prediction_set": [],
            }
            per_image_predictions.append(error_row)
            all_responses.append({"error": "image_read_error"})
            true_class_indices.append(true_class_idx)
            pred_class_indices.append(-1)
            true_classes_for_coverage.append(true_class_idx)
            prediction_sets_for_coverage.append([])
            continue

        try:
            request_id = f"f0val_{i:05d}_{uuid.uuid4().hex[:8]}"
            response = predict_single(image_bytes, request_id, pipeline_context)
        except Exception as exc:
            _log.warning(
                "run_f0_validation: predict_single raised; recording as error",
                image_path=str(img_path),
                error=str(exc),
            )
            n_errors += 1
            error_row = {
                "image_path": image_path_str,
                "true_class": true_class_str,
                "true_severity": true_severity_str,
                "error": f"predict_single_error: {exc}",
                "tier_label": None,
                "pred_class": None,
                "pred_severity": None,
                "prediction_set": [],
            }
            per_image_predictions.append(error_row)
            all_responses.append({"error": "predict_single_error"})
            true_class_indices.append(true_class_idx)
            pred_class_indices.append(-1)
            true_classes_for_coverage.append(true_class_idx)
            prediction_sets_for_coverage.append([])
            continue

        # Parse response fields
        # spec: section 16.2 lines 5655-5712 — response schema
        tier_label = _get_tier_label(response) if not _is_error_response(response) else None
        prediction_set = _get_prediction_set(response)
        primary_class = _get_primary_class(response)
        pred_severity = _get_severity_grade(response)
        rule_id = _get_rule_id_fired(response)

        # Map primary_class → int index for confusion matrix
        pred_class_idx = _CLASS_TO_IDX.get(primary_class, -1) if primary_class else -1

        per_image_row = {
            "image_path": image_path_str,
            "true_class": true_class_str,
            "true_severity": true_severity_str,
            "tier_label": tier_label,
            "pred_class": primary_class,
            "pred_severity": pred_severity,
            "prediction_set": prediction_set,
            "rule_id_fired": rule_id,
            "is_4b_degraded": _is_tier_4b_degraded(response),
        }
        per_image_predictions.append(per_image_row)
        all_responses.append(response)
        true_class_indices.append(true_class_idx)
        pred_class_indices.append(pred_class_idx)
        true_classes_for_coverage.append(true_class_idx)
        prediction_sets_for_coverage.append(prediction_set)
        n_processed += 1

        if (i + 1) % 50 == 0:
            _log.info(
                "run_f0_validation: progress",
                processed=n_processed,
                total=len(test_records),
                errors=n_errors,
            )

    _log.info(
        "run_f0_validation: pipeline pass complete",
        n_processed=n_processed,
        n_errors=n_errors,
        n_test=len(test_records),
    )

    # --- Compute metrics ---

    # Confusion matrix (spec S29.5 line 8207)
    confusion_matrix = _build_confusion_matrix(
        true_class_indices, pred_class_indices, _NUM_CLASSES
    )

    # Conformal coverage (spec S13.4, S29.4, DEC-053 Decision 5)
    conformal_coverage = _compute_conformal_coverage(
        true_classes_for_coverage,
        prediction_sets_for_coverage,
    )

    # Per-disease severity validation (spec S17.3, S17.5, DEC-053 Decision 6)
    severity_validation = _compute_severity_validation(per_image_predictions)

    # Tier 4B disposition (DEC-053 Decision 4)
    tier_disposition = _compute_tier_disposition(all_responses)

    # Calibration artifacts (DEC-053 Decision 7)
    calibration_artifacts = _read_calibration_artifacts(resolved_calib_dir)

    # --- Assemble report ---
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report: dict[str, Any] = {
        "metadata": {
            # spec: section 29.5 lines 8203-8218 — F.0 report contents
            "schema_version": "f0_validation_v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "labeled_data_path": str(labeled_data_path),
            "n_test_total": len(test_records),
            "n_processed": n_processed,
            "n_errors": n_errors,
            "class_names": _CLASS_NAMES,
            # DEC-053 Decision 7 — calibration artifact metadata
            "calibration_artifacts": calibration_artifacts,
            "calibration_dir": str(resolved_calib_dir),
        },
        # Per-image predictions (spec S29.5 line 8210 — "Failure analysis")
        "per_image_predictions": per_image_predictions,
        # spec: section 29.5 line 8207 — "Confusion matrix per class"
        "confusion_matrix": {
            "matrix": confusion_matrix,
            "class_names": _CLASS_NAMES,
            "description": (
                "rows=true_class, columns=predicted_class (argmax), "
                "indices map to class_names"
            ),
        },
        # spec: section 29.4 line 8183 — conformal coverage quality bar
        # spec: section 13.4 lines 3564-3581 — 90% coverage target
        # DEC-053 Decision 5 — Wilson CI included as informational
        "conformal_coverage": conformal_coverage,
        # spec: section 17.3 lines 5966-5982 — severity validation
        # DEC-053 Decision 6 — skip if no ground-truth severity
        "severity_validation": severity_validation,
        # DEC-053 Decision 4 — Tier 4B degraded vs real failure
        "tier_disposition": tier_disposition,
    }

    # --- Write JSON report ---
    # DEC-053 Decision 8 — write to output_dir
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    report_filename = f"validation_report_{timestamp}.json"
    report_path = resolved_output_dir / report_filename

    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=_json_default)

    _log.info(
        "run_f0_validation: report written",
        path=str(report_path),
        n_test=len(test_records),
        n_processed=n_processed,
        coverage_rate=conformal_coverage.get("coverage_rate"),
        tier_4b_total=tier_disposition["tier_4b_count_total"],
        is_pre_f0_mode=tier_disposition.get("is_pre_f0_mode"),
    )

    return report


# ---------------------------------------------------------------------------
# JSON serialization helper
# ---------------------------------------------------------------------------

def _json_default(obj: Any) -> Any:
    """Fallback serializer for JSON dump — handles numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
