"""
Unit tests for Section 17 — Severity grading.

spec: 17 (lines 5941-6083)
DEC-044: one test per S17 behavioral contract.

Coverage:
  - Each of the 5 disease classes: foliar, septoria, late_blight, ylcv, mosaic
  - Per-class threshold boundaries (mild, moderate, severe)
  - Healthy and OOD: grade=None (spec 17.6)
  - Tier 4A: omit (spec 17.7)
  - Tier 4B: omit (spec 17.7)
  - psv_reliability < 0.50: omit (spec 17.7)
  - disease_coverage_pct < 1.0%: omit (spec 17.7)
  - lesion_count OR-join for severe (spec 17.3 — "or > N lesions")
  - YLCV/mosaic: coverage-only, lesion_count irrelevant (spec 17.3 line 5980)
  - NaN psv_reliability: omit
  - None raw_features: omit via guard
  - grade_per_class field present in SeverityResult (init test)
"""

import math
import numpy as np
import pytest

from tomato_sandbox.severity.grader import SeverityResult, compute_severity
from tomato_sandbox.signals.psv.features import FEATURE_NAMES

# ---------------------------------------------------------------------------
# Helpers to build raw_features arrays
# ---------------------------------------------------------------------------

def _make_features(
    disease_coverage_pct: float = 10.0,
    lesion_count: float = 5.0,
    mean_lesion_size: float = 50.0,
    lesion_size_std: float = 10.0,
) -> np.ndarray:
    """Return a 26-element float32 feature array with named fields set."""
    feats = np.zeros(26, dtype=np.float32)
    idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
    feats[idx["disease_coverage_pct"]] = disease_coverage_pct
    feats[idx["lesion_count"]] = lesion_count
    feats[idx["mean_lesion_size"]] = mean_lesion_size
    feats[idx["lesion_size_std"]] = lesion_size_std
    return feats


# ---------------------------------------------------------------------------
# Import sanity
# ---------------------------------------------------------------------------

class TestImport:
    def test_severity_result_importable_from_canonical(self):
        from tomato_sandbox.severity.grader import SeverityResult, compute_severity
        assert SeverityResult is not None
        assert compute_severity is not None

    def test_severity_result_importable_from_shim(self):
        from tomato_sandbox.severity.severity import SeverityResult, compute_severity
        assert SeverityResult is not None

    def test_severity_result_importable_from_package(self):
        from tomato_sandbox.severity import SeverityResult, compute_severity
        assert SeverityResult is not None


# ---------------------------------------------------------------------------
# S17.6: Healthy and OOD have no severity
# spec: 17.6 lines 6036-6049
# ---------------------------------------------------------------------------

class TestHealthyAndOOD:
    def test_healthy_returns_null_grade(self):
        # spec: 17.6 line 6038 — "argmax = healthy or OOD" → grade = null
        result = compute_severity(
            predicted_class=5,    # healthy
            raw_features=_make_features(disease_coverage_pct=25.0),
            psv_reliability=0.90,
            tier_label="1",
        )
        assert result.grade is None

    def test_healthy_human_readable(self):
        result = compute_severity(
            predicted_class=5,
            raw_features=_make_features(),
            psv_reliability=0.90,
            tier_label="1",
        )
        assert "healthy" in result.human_readable.lower()

    def test_ood_returns_null_grade(self):
        # spec: 17.6 line 6038 — OOD → grade = null
        result = compute_severity(
            predicted_class=6,    # OOD
            raw_features=_make_features(disease_coverage_pct=40.0),
            psv_reliability=0.90,
            tier_label="1",
        )
        assert result.grade is None

    def test_ood_human_readable(self):
        result = compute_severity(
            predicted_class=6,
            raw_features=_make_features(),
            psv_reliability=0.90,
            tier_label="1",
        )
        assert "unclear" in result.human_readable.lower()

    def test_ood_recommended_action_suggests_retake(self):
        result = compute_severity(
            predicted_class=6,
            raw_features=_make_features(),
            psv_reliability=0.90,
            tier_label="1",
        )
        assert "tomato leaf" in result.recommended_action.lower()


# ---------------------------------------------------------------------------
# S17.7: Severity omit conditions
# spec: 17.7 lines 6051-6071
# ---------------------------------------------------------------------------

class TestSeverityOmitConditions:
    def test_tier_4a_omits_severity(self):
        # spec: 17.7 line 6055 — "Tier 4A: classifier too uncertain"
        result = compute_severity(
            predicted_class=0,  # foliar
            raw_features=_make_features(disease_coverage_pct=10.0),
            psv_reliability=0.90,
            tier_label="4A",
        )
        assert result.grade is None

    def test_tier_4b_omits_severity(self):
        # spec: 17.7 line 6056 — "Tier 4B: PSV may have failed"
        result = compute_severity(
            predicted_class=1,  # septoria
            raw_features=_make_features(disease_coverage_pct=15.0),
            psv_reliability=0.90,
            tier_label="4B",
        )
        assert result.grade is None

    def test_low_psv_reliability_omits_severity(self):
        # spec: 17.7 line 6057 — "psv_reliability < 0.50"
        result = compute_severity(
            predicted_class=0,
            raw_features=_make_features(disease_coverage_pct=20.0),
            psv_reliability=0.49,    # just below threshold
            tier_label="1",
        )
        assert result.grade is None
        assert "low_psv_reliability" in (result.omit_reason or "")

    def test_psv_reliability_exactly_0_5_permits_severity(self):
        # spec: 17.7 line 6057 — threshold is < 0.50 (strict); 0.50 should grade
        result = compute_severity(
            predicted_class=0,  # foliar
            raw_features=_make_features(disease_coverage_pct=10.0),
            psv_reliability=0.50,    # exactly at threshold — should grade
            tier_label="1",
        )
        assert result.grade is not None

    def test_nan_psv_reliability_omits_severity(self):
        # NaN reliability is treated as < 0.50 per guard
        result = compute_severity(
            predicted_class=0,
            raw_features=_make_features(disease_coverage_pct=20.0),
            psv_reliability=float("nan"),
            tier_label="1",
        )
        assert result.grade is None

    def test_low_coverage_omits_severity(self):
        # spec: 17.7 line 6058 — "Disease coverage < 1%: unreliable"
        result = compute_severity(
            predicted_class=2,  # late_blight
            raw_features=_make_features(disease_coverage_pct=0.9),
            psv_reliability=0.90,
            tier_label="1",
        )
        assert result.grade is None
        assert "low_coverage" in (result.omit_reason or "")

    def test_coverage_exactly_1_pct_permits_severity(self):
        # spec: 17.7 line 6058 — "< 1%" is strict; 1.0 should grade
        result = compute_severity(
            predicted_class=2,  # late_blight
            raw_features=_make_features(disease_coverage_pct=1.0),
            psv_reliability=0.90,
            tier_label="1",
        )
        assert result.grade is not None

    def test_none_raw_features_omits_severity(self):
        # Defensive: None features guard → all zeros → coverage = 0 < 1.0 → omit
        result = compute_severity(
            predicted_class=0,
            raw_features=None,
            psv_reliability=0.90,
            tier_label="1",
        )
        assert result.grade is None

    def test_omit_reason_field_set(self):
        # All omit cases must have omit_reason populated
        result = compute_severity(
            predicted_class=0,
            raw_features=_make_features(disease_coverage_pct=5.0),
            psv_reliability=0.30,
            tier_label="1",
        )
        assert result.omit_reason is not None
        assert len(result.omit_reason) > 0


# ---------------------------------------------------------------------------
# S17.3: Foliar leaf spot thresholds
# spec: 17.3 line 5974 — "< 5%, 1-5 | 5-15%, 5-15 | > 15% or > 15 lesions"
# ---------------------------------------------------------------------------

class TestFoliarThresholds:
    def test_foliar_mild_by_coverage(self):
        # spec: 17.3 line 5974 — "< 5%" → mild
        result = compute_severity(
            predicted_class=0,
            raw_features=_make_features(disease_coverage_pct=4.0, lesion_count=3.0),
            psv_reliability=0.80,
            tier_label="1",
        )
        assert result.grade == "mild"

    def test_foliar_moderate_by_coverage(self):
        # spec: 17.3 — 5-15% coverage → moderate (below severe threshold)
        result = compute_severity(
            predicted_class=0,
            raw_features=_make_features(disease_coverage_pct=10.0, lesion_count=8.0),
            psv_reliability=0.80,
            tier_label="2",
        )
        assert result.grade == "moderate"

    def test_foliar_severe_by_coverage(self):
        # spec: 17.3 line 5974 — "> 15%" → severe
        result = compute_severity(
            predicted_class=0,
            raw_features=_make_features(disease_coverage_pct=20.0, lesion_count=5.0),
            psv_reliability=0.80,
            tier_label="1",
        )
        assert result.grade == "severe"

    def test_foliar_severe_by_lesion_count(self):
        # spec: 17.3 — "or > 15 lesions" triggers severe even below coverage threshold
        result = compute_severity(
            predicted_class=0,
            raw_features=_make_features(disease_coverage_pct=8.0, lesion_count=16.0),
            psv_reliability=0.80,
            tier_label="1",
        )
        assert result.grade == "severe"

    def test_foliar_details_populated(self):
        result = compute_severity(
            predicted_class=0,
            raw_features=_make_features(disease_coverage_pct=12.0, lesion_count=8.0),
            psv_reliability=0.80,
            tier_label="1",
        )
        assert result.disease_coverage_pct == pytest.approx(12.0, abs=0.1)
        assert result.lesion_count == 8
        assert result.thresholds_used is not None
        assert result.thresholds_used["disease"] == "foliar"
        assert result.thresholds_used["mild_max"] == pytest.approx(5.0)
        assert result.thresholds_used["moderate_max"] == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# S17.3: Septoria leaf spot thresholds
# spec: 17.3 line 5975 — "< 8% | 8-25% | > 25% or > 25 lesions"
# ---------------------------------------------------------------------------

class TestSeptoriaThresholds:
    def test_septoria_mild(self):
        result = compute_severity(
            predicted_class=1,
            raw_features=_make_features(disease_coverage_pct=5.0, lesion_count=5.0),
            psv_reliability=0.75,
            tier_label="1",
        )
        assert result.grade == "mild"

    def test_septoria_moderate(self):
        result = compute_severity(
            predicted_class=1,
            raw_features=_make_features(disease_coverage_pct=15.0, lesion_count=15.0),
            psv_reliability=0.75,
            tier_label="2",
        )
        assert result.grade == "moderate"

    def test_septoria_severe_by_coverage(self):
        result = compute_severity(
            predicted_class=1,
            raw_features=_make_features(disease_coverage_pct=30.0, lesion_count=20.0),
            psv_reliability=0.75,
            tier_label="1",
        )
        assert result.grade == "severe"

    def test_septoria_severe_by_lesion_count(self):
        # spec: 17.3 — "or > 25 lesions"
        result = compute_severity(
            predicted_class=1,
            raw_features=_make_features(disease_coverage_pct=10.0, lesion_count=26.0),
            psv_reliability=0.75,
            tier_label="1",
        )
        assert result.grade == "severe"


# ---------------------------------------------------------------------------
# S17.3: Late blight thresholds
# spec: 17.3 line 5976 — "< 2% | 2-8% | > 8% or > 8 lesions"
# ---------------------------------------------------------------------------

class TestLateBlightThresholds:
    def test_late_blight_mild(self):
        # spec: 17.3 — "< 2%" is mild for late_blight (rapid progression risk)
        result = compute_severity(
            predicted_class=2,
            raw_features=_make_features(disease_coverage_pct=1.5, lesion_count=2.0),
            psv_reliability=0.85,
            tier_label="1",
        )
        assert result.grade == "mild"

    def test_late_blight_moderate(self):
        result = compute_severity(
            predicted_class=2,
            raw_features=_make_features(disease_coverage_pct=5.0, lesion_count=5.0),
            psv_reliability=0.85,
            tier_label="1",
        )
        assert result.grade == "moderate"

    def test_late_blight_severe_by_coverage(self):
        # spec: 17.3 line 5976 — "> 8%" → severe
        result = compute_severity(
            predicted_class=2,
            raw_features=_make_features(disease_coverage_pct=10.0, lesion_count=5.0),
            psv_reliability=0.85,
            tier_label="1",
        )
        assert result.grade == "severe"

    def test_late_blight_severe_by_lesion_count(self):
        # spec: 17.3 — "or > 8 lesions"
        result = compute_severity(
            predicted_class=2,
            raw_features=_make_features(disease_coverage_pct=3.0, lesion_count=9.0),
            psv_reliability=0.85,
            tier_label="1",
        )
        assert result.grade == "severe"

    def test_late_blight_threshold_lower_than_others(self):
        # Spec rationale: late_blight has stricter thresholds (rapid progression)
        # 2.0% coverage = NOT mild for late_blight (should be moderate)
        result = compute_severity(
            predicted_class=2,
            raw_features=_make_features(disease_coverage_pct=2.0, lesion_count=2.0),
            psv_reliability=0.85,
            tier_label="1",
        )
        assert result.grade == "moderate"


# ---------------------------------------------------------------------------
# S17.3: YLCV thresholds (coverage-only)
# spec: 17.3 line 5977 — "< 10% | 10-30% | > 30%"
# spec: 17.3 line 5980 — "only coverage matters" for YLCV
# ---------------------------------------------------------------------------

class TestYLCVThresholds:
    def test_ylcv_mild(self):
        result = compute_severity(
            predicted_class=3,
            raw_features=_make_features(disease_coverage_pct=5.0, lesion_count=100.0),
            psv_reliability=0.70,
            tier_label="1",
        )
        assert result.grade == "mild"

    def test_ylcv_moderate(self):
        result = compute_severity(
            predicted_class=3,
            raw_features=_make_features(disease_coverage_pct=20.0, lesion_count=100.0),
            psv_reliability=0.70,
            tier_label="2",
        )
        assert result.grade == "moderate"

    def test_ylcv_severe(self):
        result = compute_severity(
            predicted_class=3,
            raw_features=_make_features(disease_coverage_pct=35.0, lesion_count=2.0),
            psv_reliability=0.70,
            tier_label="1",
        )
        assert result.grade == "severe"

    def test_ylcv_lesion_count_does_not_cause_severe(self):
        # spec: 17.3 line 5980 — YLCV has no lesion_count threshold
        # Very high lesion count with low coverage → mild (coverage-only)
        result = compute_severity(
            predicted_class=3,
            raw_features=_make_features(disease_coverage_pct=5.0, lesion_count=999.0),
            psv_reliability=0.70,
            tier_label="1",
        )
        assert result.grade == "mild"


# ---------------------------------------------------------------------------
# S17.3: Mosaic virus thresholds (coverage-only)
# spec: 17.3 line 5978 — "< 15% | 15-40% | > 40%"
# spec: 17.3 line 5980 — "only coverage matters" for mosaic
# ---------------------------------------------------------------------------

class TestMosaicThresholds:
    def test_mosaic_mild(self):
        result = compute_severity(
            predicted_class=4,
            raw_features=_make_features(disease_coverage_pct=10.0, lesion_count=500.0),
            psv_reliability=0.65,
            tier_label="1",
        )
        assert result.grade == "mild"

    def test_mosaic_moderate(self):
        result = compute_severity(
            predicted_class=4,
            raw_features=_make_features(disease_coverage_pct=25.0, lesion_count=2.0),
            psv_reliability=0.65,
            tier_label="2",
        )
        assert result.grade == "moderate"

    def test_mosaic_severe(self):
        result = compute_severity(
            predicted_class=4,
            raw_features=_make_features(disease_coverage_pct=45.0, lesion_count=2.0),
            psv_reliability=0.65,
            tier_label="1",
        )
        assert result.grade == "severe"

    def test_mosaic_lesion_count_does_not_cause_severe(self):
        # spec: 17.3 line 5980 — mosaic is coverage-only
        result = compute_severity(
            predicted_class=4,
            raw_features=_make_features(disease_coverage_pct=5.0, lesion_count=9999.0),
            psv_reliability=0.65,
            tier_label="1",
        )
        assert result.grade == "mild"


# ---------------------------------------------------------------------------
# S17.4: Response fields present and correct types
# spec: 17.4 lines 5991-6011
# ---------------------------------------------------------------------------

class TestResponseFields:
    def test_all_fields_present_on_grade_result(self):
        result = compute_severity(
            predicted_class=0,
            raw_features=_make_features(disease_coverage_pct=12.0, lesion_count=8.0),
            psv_reliability=0.80,
            tier_label="1",
        )
        assert isinstance(result, SeverityResult)
        assert result.grade in ("mild", "moderate", "severe")
        assert isinstance(result.human_readable, str)
        assert result.disease_coverage_pct is not None
        assert result.lesion_count is not None
        assert result.psv_confidence_in_severity is not None
        assert result.thresholds_used is not None
        assert isinstance(result.recommended_action, str)

    def test_grade_per_class_default_none(self):
        # spec: 17.5 — grade_per_class is for multi-class tiers; None by default
        result = compute_severity(
            predicted_class=0,
            raw_features=_make_features(disease_coverage_pct=12.0),
            psv_reliability=0.80,
            tier_label="1",
        )
        assert result.grade_per_class is None

    def test_recommended_action_non_empty(self):
        result = compute_severity(
            predicted_class=2,
            raw_features=_make_features(disease_coverage_pct=3.0),
            psv_reliability=0.80,
            tier_label="1",
        )
        assert len(result.recommended_action) > 0

    def test_human_readable_capitalised(self):
        result = compute_severity(
            predicted_class=0,
            raw_features=_make_features(disease_coverage_pct=10.0),
            psv_reliability=0.80,
            tier_label="1",
        )
        assert result.human_readable[0].isupper()


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------

class TestBoundaryConditions:
    def test_coverage_at_foliar_mild_boundary(self):
        # spec: 17.3 — "< 5%" is mild; 5.0% is not mild → moderate
        result_below = compute_severity(
            predicted_class=0,
            raw_features=_make_features(disease_coverage_pct=4.99),
            psv_reliability=0.80,
            tier_label="1",
        )
        assert result_below.grade == "mild"

        result_at = compute_severity(
            predicted_class=0,
            raw_features=_make_features(disease_coverage_pct=5.0, lesion_count=5.0),
            psv_reliability=0.80,
            tier_label="1",
        )
        assert result_at.grade == "moderate"

    def test_lesion_count_at_severe_boundary_foliar(self):
        # spec: 17.3 line 5974 — "> 15 lesions" (strict >)
        # 15 lesions should NOT trigger severe
        result_at = compute_severity(
            predicted_class=0,
            raw_features=_make_features(disease_coverage_pct=8.0, lesion_count=15.0),
            psv_reliability=0.80,
            tier_label="1",
        )
        assert result_at.grade == "moderate"

        # 16 lesions SHOULD trigger severe
        result_above = compute_severity(
            predicted_class=0,
            raw_features=_make_features(disease_coverage_pct=8.0, lesion_count=16.0),
            psv_reliability=0.80,
            tier_label="1",
        )
        assert result_above.grade == "severe"
