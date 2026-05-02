"""
Severity grading for the Tomato 3-Signal system.

Spec section: 17 (Severity grading), lines 5941-6083.
DEC-044: canonical implementation at severity/grader.py per spec file layout (S21 line 6537).
Task-card path severity/severity.py is a re-export shim.

Key design decisions (DEC-044):
- PSV features accessed by name via FEATURE_NAMES index map (Decision 2).
- Primary grading rule: coverage_pct primary; lesion_count OR-joins at Severe boundary (Decision 3).
- YLCV and mosaic: coverage_pct only (spec 17.3 line 5980).
- Severity omitted (grade=None) for Tier 4A/4B, psv_reliability < 0.50,
  coverage_pct < 1.0 (spec 17.7 lines 6051-6058).
- Healthy and OOD: grade=None with specific human_readable (spec 17.6 lines 6036-6049).
- BLK-012: mean_lesion_intensity G3 cite in spec is wrong vs feature catalog;
  mean_lesion_size (G2 idx 3) used as proxy. No grading impact (ancillary only).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Optional

from tomato_sandbox.utils.logging import get_logger
from tomato_sandbox.signals.psv.features import FEATURE_NAMES

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Class index constants
# spec: import_contract.md "Class index reference"
# 0=foliar 1=septoria 2=late_blight 3=ylcv 4=mosaic 5=healthy 6=OOD
# ---------------------------------------------------------------------------
_CLASS_HEALTHY = 5
_CLASS_OOD = 6
# Disease classes that get severity grading (spec 17.1 lines 5943-5945)
_DISEASE_CLASSES = frozenset({0, 1, 2, 3, 4})  # foliar, septoria, late_blight, ylcv, mosaic

# Class index → disease key for threshold lookup
# spec: 17.3 lines 5972-5978 — per-disease threshold table
_CLASS_IDX_TO_DISEASE_KEY: dict[int, str] = {
    0: "foliar",
    1: "septoria",
    2: "late_blight",
    3: "ylcv",
    4: "mosaic",
}

# ---------------------------------------------------------------------------
# PSV feature name → index lookup (DEC-044 Decision 2)
# Accessed by name, never by magic number.
# spec: 17.2 lines 5954-5962
# ---------------------------------------------------------------------------
_FEAT_IDX: dict[str, int] = {name: i for i, name in enumerate(FEATURE_NAMES)}

# Validated indices — fail fast at import if feature catalog changed
_IDX_DISEASE_COVERAGE_PCT: int = _FEAT_IDX["disease_coverage_pct"]   # G1 idx 0
_IDX_LESION_COUNT: int = _FEAT_IDX["lesion_count"]                   # G1 idx 2
_IDX_MEAN_LESION_SIZE: int = _FEAT_IDX["mean_lesion_size"]           # G2 idx 3 (BLK-012 proxy)
_IDX_LESION_SIZE_STD: int = _FEAT_IDX["lesion_size_std"]             # G2 idx 4 (BLK-012 proxy)

# ---------------------------------------------------------------------------
# PSV reliability threshold for severity omit
# spec: 17.7 line 6057 — "psv_reliability < 0.50"
# ---------------------------------------------------------------------------
_PSV_RELIABILITY_MIN: float = 0.50

# Disease coverage minimum for reliable severity grading
# spec: 17.7 line 6058 — "Disease coverage < 1%: very small coverage ... unreliable"
_COVERAGE_PCT_MIN: float = 1.0

# ---------------------------------------------------------------------------
# Per-disease severity thresholds (spec 17.3 lines 5972-5978)
# Exposed as env vars; defaults are spec table values.
# spec: 17.3 lines 5984-5987 — TOMATO_SEVERITY_<DISEASE>_MILD_PCT etc.
# ---------------------------------------------------------------------------

def _env_float(name: str, default: float) -> float:
    """Read a float from an env var, falling back to default on parse error."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        _log.warning("severity_env_parse_error", env_var=name, raw=raw, default=default)
        return default


# Foliar leaf spot thresholds
# spec: 17.3 line 5974 — "< 5%, 1-5 lesions | 5-15%, 5-15 | > 15% or > 15 lesions"
_FOLIAR_MILD_MAX_PCT: float = _env_float("TOMATO_SEVERITY_FOLIAR_MILD_PCT", 5.0)
_FOLIAR_MODERATE_MAX_PCT: float = _env_float("TOMATO_SEVERITY_FOLIAR_MODERATE_PCT", 15.0)
_FOLIAR_SEVERE_MIN_LESION_COUNT: int = int(_env_float("TOMATO_SEVERITY_FOLIAR_SEVERE_MIN_LESIONS", 15))

# Septoria leaf spot thresholds
# spec: 17.3 line 5975 — "< 8%, 1-10 | 8-25%, 10-25 | > 25% or > 25 lesions"
_SEPTORIA_MILD_MAX_PCT: float = _env_float("TOMATO_SEVERITY_SEPTORIA_MILD_PCT", 8.0)
_SEPTORIA_MODERATE_MAX_PCT: float = _env_float("TOMATO_SEVERITY_SEPTORIA_MODERATE_PCT", 25.0)
_SEPTORIA_SEVERE_MIN_LESION_COUNT: int = int(_env_float("TOMATO_SEVERITY_SEPTORIA_SEVERE_MIN_LESIONS", 25))

# Late blight thresholds
# spec: 17.3 line 5976 — "< 2%, 1-3 | 2-8%, 3-8 | > 8% or > 8 lesions"
_LATE_BLIGHT_MILD_MAX_PCT: float = _env_float("TOMATO_SEVERITY_LATE_BLIGHT_MILD_PCT", 2.0)
_LATE_BLIGHT_MODERATE_MAX_PCT: float = _env_float("TOMATO_SEVERITY_LATE_BLIGHT_MODERATE_PCT", 8.0)
_LATE_BLIGHT_SEVERE_MIN_LESION_COUNT: int = int(_env_float("TOMATO_SEVERITY_LATE_BLIGHT_SEVERE_MIN_LESIONS", 8))

# YLCV thresholds (coverage only — spec 17.3 line 5977)
# spec: 17.3 line 5977 — "< 10% | 10-30% | > 30%"
_YLCV_MILD_MAX_PCT: float = _env_float("TOMATO_SEVERITY_YLCV_MILD_PCT", 10.0)
_YLCV_MODERATE_MAX_PCT: float = _env_float("TOMATO_SEVERITY_YLCV_MODERATE_PCT", 30.0)

# Mosaic thresholds (coverage only — spec 17.3 line 5978)
# spec: 17.3 line 5978 — "< 15% | 15-40% | > 40%"
_MOSAIC_MILD_MAX_PCT: float = _env_float("TOMATO_SEVERITY_MOSAIC_MILD_PCT", 15.0)
_MOSAIC_MODERATE_MAX_PCT: float = _env_float("TOMATO_SEVERITY_MOSAIC_MODERATE_PCT", 40.0)


# ---------------------------------------------------------------------------
# Output dataclass
# spec: 17.4 (lines 5991-6011) JSON block defines all fields.
# DEC-044 Decision 5.
# ---------------------------------------------------------------------------

@dataclass
class SeverityResult:
    """Result of severity grading for a single predicted class.

    spec: 17.4 lines 5991-6011 — JSON response block schema.
    DEC-044 Decision 5.

    Fields map directly to the severity block in the API response (Section 16).
    grade=None when severity is omitted (see spec 17.6, 17.7).
    """

    grade: Optional[str]                          # "mild" / "moderate" / "severe" / None
    human_readable: str                           # human-facing explanation
    disease_coverage_pct: Optional[float]         # spec: 17.4 details.disease_coverage_pct
    lesion_count: Optional[int]                   # spec: 17.4 details.lesion_count
    psv_confidence_in_severity: Optional[float]   # spec: 17.4 details.psv_confidence_in_severity
    thresholds_used: Optional[dict]               # spec: 17.4 details.thresholds_used
    recommended_action: str                       # spec: 17.4 recommended_action
    omit_reason: Optional[str] = None            # internal; why grade=None (for logging)

    # For multi-class (Tier 3A/3B) — spec 17.5 lines 6015-6032
    grade_per_class: Optional[list] = None       # list of {"class": str, "grade": str, "coverage_pct": float}


@dataclass
class _DiseaseThresholds:
    """Internal: per-disease threshold bundle."""
    mild_max_pct: float
    moderate_max_pct: float
    severe_min_lesion_count: Optional[int]   # None = coverage-only disease (ylcv, mosaic)
    disease_key: str


# ---------------------------------------------------------------------------
# Threshold table lookup
# ---------------------------------------------------------------------------

def _get_thresholds(disease_class_idx: int) -> Optional[_DiseaseThresholds]:
    """Return thresholds for a disease class index, or None for non-disease classes.

    spec: 17.3 lines 5972-5980
    """
    if disease_class_idx == 0:   # foliar
        return _DiseaseThresholds(
            mild_max_pct=_FOLIAR_MILD_MAX_PCT,
            moderate_max_pct=_FOLIAR_MODERATE_MAX_PCT,
            severe_min_lesion_count=_FOLIAR_SEVERE_MIN_LESION_COUNT,
            disease_key="foliar",
        )
    if disease_class_idx == 1:   # septoria
        return _DiseaseThresholds(
            mild_max_pct=_SEPTORIA_MILD_MAX_PCT,
            moderate_max_pct=_SEPTORIA_MODERATE_MAX_PCT,
            severe_min_lesion_count=_SEPTORIA_SEVERE_MIN_LESION_COUNT,
            disease_key="septoria",
        )
    if disease_class_idx == 2:   # late_blight
        return _DiseaseThresholds(
            mild_max_pct=_LATE_BLIGHT_MILD_MAX_PCT,
            moderate_max_pct=_LATE_BLIGHT_MODERATE_MAX_PCT,
            severe_min_lesion_count=_LATE_BLIGHT_SEVERE_MIN_LESION_COUNT,
            disease_key="late_blight",
        )
    if disease_class_idx == 3:   # ylcv — coverage only
        # spec: 17.3 line 5977 and line 5980 — "only coverage matters"
        return _DiseaseThresholds(
            mild_max_pct=_YLCV_MILD_MAX_PCT,
            moderate_max_pct=_YLCV_MODERATE_MAX_PCT,
            severe_min_lesion_count=None,
            disease_key="ylcv",
        )
    if disease_class_idx == 4:   # mosaic — coverage only
        # spec: 17.3 line 5978 and line 5980 — "only coverage matters"
        return _DiseaseThresholds(
            mild_max_pct=_MOSAIC_MILD_MAX_PCT,
            moderate_max_pct=_MOSAIC_MODERATE_MAX_PCT,
            severe_min_lesion_count=None,
            disease_key="mosaic",
        )
    return None


# ---------------------------------------------------------------------------
# Core grading logic
# ---------------------------------------------------------------------------

def _grade_from_thresholds(
    coverage_pct: float,
    lesion_count: int,
    thresholds: _DiseaseThresholds,
) -> str:
    """Apply per-disease thresholds to produce a grade string.

    DEC-044 Decision 3:
    - Mild: coverage_pct < mild_max_pct
    - Severe: coverage_pct > moderate_max_pct OR (lesion_count > severe_min_lesion_count
              for lesion-based diseases)
    - Moderate: everything in between.

    spec: 17.3 lines 5972-5980 — Severe column explicitly says "or > N lesions"
    for foliar, septoria, late_blight; YLCV and mosaic are coverage-only.
    """
    # spec: 17.3 — Mild bucket
    if coverage_pct < thresholds.mild_max_pct:
        return "mild"

    # spec: 17.3 — Severe bucket
    # "or > N lesions" triggers severe for lesion-based diseases
    # spec: 17.3 line 5972 table — explicit OR in the Severe column
    coverage_severe = coverage_pct > thresholds.moderate_max_pct
    lesion_severe = (
        thresholds.severe_min_lesion_count is not None
        and lesion_count > thresholds.severe_min_lesion_count
    )
    if coverage_severe or lesion_severe:
        return "severe"

    return "moderate"


def _make_null_result(human_readable: str, recommended_action: str, omit_reason: str) -> SeverityResult:
    """Helper: SeverityResult with grade=None (omit case).

    spec: 17.6 lines 6038-6049, 17.7 lines 6060-6071
    """
    return SeverityResult(
        grade=None,
        human_readable=human_readable,
        disease_coverage_pct=None,
        lesion_count=None,
        psv_confidence_in_severity=None,
        thresholds_used=None,
        recommended_action=recommended_action,
        omit_reason=omit_reason,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_severity(
    *,
    predicted_class: int,
    raw_features: object,          # np.ndarray of shape (26,) from SignalCResult.raw_features
    psv_reliability: float,
    tier_label: str,
) -> SeverityResult:
    """Compute severity grade for a single predicted class.

    spec: 17 (lines 5941-6083)
    DEC-044: primary grading rule, omit conditions, healthy/OOD handling.

    Args:
        predicted_class: Class index (0-6). 0-4 = disease, 5 = healthy, 6 = OOD.
        raw_features: SignalCResult.raw_features, shape (26,). May be None on
                      PSV failure; guard handles this.
        psv_reliability: SignalCResult.psv_reliability ∈ [0, 1].
        tier_label: TierAssignment.tier_label. Severity omitted for "4A" / "4B".
                    spec: 17.7 lines 6053-6056.

    Returns:
        SeverityResult dataclass. grade=None when severity is omitted.
    """
    import numpy as np
    from tomato_sandbox.utils.nan_guards import guard_array

    # ------------------------------------------------------------------
    # 17.6: Healthy and OOD have no severity
    # spec: 17.6 lines 6036-6049
    # ------------------------------------------------------------------
    if predicted_class == _CLASS_HEALTHY:
        _log.debug("severity_healthy", predicted_class=predicted_class)
        return _make_null_result(
            human_readable="Not applicable (plant appears healthy)",
            recommended_action="No action needed.",
            omit_reason="healthy",
        )

    if predicted_class == _CLASS_OOD:
        _log.debug("severity_ood", predicted_class=predicted_class)
        return _make_null_result(
            human_readable="Not applicable (image content unclear)",
            recommended_action="Please ensure the image shows a tomato leaf clearly.",
            omit_reason="ood",
        )

    # ------------------------------------------------------------------
    # 17.7: Omit for Tier 4A / 4B
    # spec: 17.7 lines 6053-6056
    # ------------------------------------------------------------------
    if tier_label in ("4A", "4B"):
        _log.debug("severity_omit_tier", tier_label=tier_label)
        return _make_null_result(
            human_readable="Severity could not be reliably graded.",
            recommended_action="Consult agronomist for detailed severity assessment.",
            omit_reason=f"tier_{tier_label}",
        )

    # ------------------------------------------------------------------
    # 17.7: Omit for low PSV reliability
    # spec: 17.7 line 6057 — "psv_reliability < 0.50"
    # ------------------------------------------------------------------
    if math.isnan(psv_reliability) or psv_reliability < _PSV_RELIABILITY_MIN:
        safe_rel = psv_reliability if not math.isnan(psv_reliability) else 0.0
        _log.debug("severity_omit_low_psv_reliability", psv_reliability=safe_rel)
        return SeverityResult(
            grade=None,
            human_readable="Severity could not be reliably graded.",
            disease_coverage_pct=None,
            lesion_count=None,
            psv_confidence_in_severity=safe_rel,
            thresholds_used={"reason": "low_psv_reliability"},
            recommended_action="Consult agronomist for detailed severity assessment.",
            omit_reason="low_psv_reliability",
        )

    # ------------------------------------------------------------------
    # Extract PSV features by name (DEC-044 Decision 2, BLK-012)
    # Guard raw_features against None / NaN / wrong shape.
    # spec: 17.2 lines 5954-5962
    # ------------------------------------------------------------------
    if raw_features is None:
        feats = guard_array([], 26, 0.0)
    else:
        feats = guard_array(raw_features, 26, 0.0)  # type: ignore[arg-type]

    coverage_pct = float(feats[_IDX_DISEASE_COVERAGE_PCT])   # spec: 17.2 line 5956
    lesion_count_raw = float(feats[_IDX_LESION_COUNT])       # spec: 17.2 line 5958
    # BLK-012: spec calls this G3 "mean_lesion_intensity" but that feature
    # does not exist in the catalog. Using G2 "mean_lesion_size" as proxy.
    # spec: 17.2 lines 5957-5960 (BLK-012 documented)
    mean_lesion_size = float(feats[_IDX_MEAN_LESION_SIZE])
    lesion_size_std_val = float(feats[_IDX_LESION_SIZE_STD])

    lesion_count = max(0, int(round(lesion_count_raw)))

    # ------------------------------------------------------------------
    # 17.7: Omit for very low coverage (noise / single lesion)
    # spec: 17.7 line 6058 — "Disease coverage < 1%"
    # ------------------------------------------------------------------
    if coverage_pct < _COVERAGE_PCT_MIN:
        _log.debug("severity_omit_low_coverage", disease_coverage_pct=coverage_pct)
        return _make_null_result(
            human_readable="Severity could not be reliably graded.",
            recommended_action="Consult agronomist for detailed severity assessment.",
            omit_reason="low_coverage",
        )

    # ------------------------------------------------------------------
    # Get per-disease thresholds (spec 17.3)
    # ------------------------------------------------------------------
    thresholds = _get_thresholds(predicted_class)
    if thresholds is None:
        # Unknown class index — defensive fallback
        _log.warning("severity_unknown_class", predicted_class=predicted_class)
        return _make_null_result(
            human_readable="Severity could not be reliably graded.",
            recommended_action="Consult agronomist for detailed severity assessment.",
            omit_reason="unknown_class",
        )

    # ------------------------------------------------------------------
    # Compute grade (DEC-044 Decision 3)
    # spec: 17.3 lines 5972-5980
    # ------------------------------------------------------------------
    grade = _grade_from_thresholds(coverage_pct, lesion_count, thresholds)

    # psv_confidence_in_severity: derived from psv_reliability as a simple proxy.
    # Spec 17.4 details field does not prescribe a precise formula beyond "PSV confidence".
    # We use psv_reliability directly as the confidence estimate.
    psv_confidence = psv_reliability

    human_readable = f"{grade.capitalize()} severity"

    thresholds_dict = {
        "mild_max": thresholds.mild_max_pct,
        "moderate_max": thresholds.moderate_max_pct,
        "disease": thresholds.disease_key,
    }

    # Recommended action placeholder (spec 17.4 line 6013 says from
    # treatment_templates.yaml; that file is a future deployment artifact).
    # Provide a sensible default keyed by (disease, grade).
    recommended_action = _get_recommended_action(thresholds.disease_key, grade)

    result = SeverityResult(
        grade=grade,
        human_readable=human_readable,
        disease_coverage_pct=round(coverage_pct, 2),
        lesion_count=lesion_count,
        psv_confidence_in_severity=round(psv_confidence, 3),
        thresholds_used=thresholds_dict,
        recommended_action=recommended_action,
    )

    _log.debug(
        "severity_computed",
        predicted_class=predicted_class,
        grade=grade,
        disease_coverage_pct=coverage_pct,
        lesion_count=lesion_count,
        psv_reliability=psv_reliability,
        tier_label=tier_label,
    )
    return result


def _get_recommended_action(disease_key: str, grade: str) -> str:
    """Return a recommended action string for (disease, grade).

    spec: 17.4 line 6008 example + line 6013 — treatment_templates.yaml (future).
    Inline defaults used until treatment_templates.yaml is deployed.
    """
    # spec: 17.4 line 6008 example: "Apply standard fungicide treatment within 48 hours."
    _ACTIONS: dict[tuple[str, str], str] = {
        ("foliar", "mild"): "Monitor closely and apply preventive fungicide if spreading.",
        ("foliar", "moderate"): "Apply standard fungicide treatment within 48 hours.",
        ("foliar", "severe"): "Apply aggressive fungicide treatment immediately. Consult agronomist.",
        ("septoria", "mild"): "Monitor and consider fungicide spray if humid conditions persist.",
        ("septoria", "moderate"): "Apply standard copper-based or systemic fungicide within 48 hours.",
        ("septoria", "severe"): "Apply aggressive fungicide. Remove severely affected leaves. Consult agronomist.",
        ("late_blight", "mild"): "Apply preventive fungicide (mancozeb or chlorothalonil) immediately.",
        ("late_blight", "moderate"): "Apply systemic fungicide (metalaxyl-M) immediately. Monitor daily.",
        ("late_blight", "severe"): "Apply systemic fungicide immediately. Consult agronomist urgently. Late blight spreads rapidly.",
        ("ylcv", "mild"): "Control whitefly vector. Monitor closely.",
        ("ylcv", "moderate"): "Apply insecticide for whitefly control. Consider removing severely affected plants.",
        ("ylcv", "severe"): "Remove and destroy severely affected plants. Apply systemic insecticide. Consult agronomist.",
        ("mosaic", "mild"): "Remove infected plants if feasible. Control aphid vectors.",
        ("mosaic", "moderate"): "Remove infected plants. Apply insecticide for aphid control. Consult agronomist.",
        ("mosaic", "severe"): "Remove infected plants immediately. Apply insecticide. Consult agronomist urgently.",
    }
    key = (disease_key, grade)
    return _ACTIONS.get(key, "Consult agronomist for treatment recommendations.")


__all__ = ["SeverityResult", "compute_severity"]
