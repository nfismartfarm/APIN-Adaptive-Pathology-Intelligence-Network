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


# ---------------------------------------------------------------------------
# BLK-015 fix — grade_per_class for Tier 3A/3B multi-class sets
# spec: 17.5 lines 6015-6032
# DEC-050: multi_class_set parameter populates grade_per_class.
# ---------------------------------------------------------------------------

class TestGradePerClass:
    """Tests for the multi-class severity path (Tier 3A/3B).

    spec: section 17.5 lines 6015-6032
    BLK-015 fix (DEC-050): grade_per_class must be populated for Tier 3A/3B.
    """

    def test_grade_per_class_populated_for_two_disease_classes(self):
        """grade_per_class is a list with one entry per disease class when
        multi_class_set has 2 disease classes.
        spec: 17.5 lines 6015-6025 — list with class/grade/coverage_pct entries
        """
        result = compute_severity(
            predicted_class=0,
            raw_features=_make_features(disease_coverage_pct=10.0, lesion_count=5.0),
            psv_reliability=0.80,
            tier_label="3A",
            multi_class_set=[0, 1],   # foliar and septoria
        )
        assert result.grade_per_class is not None
        assert len(result.grade_per_class) == 2

    def test_grade_per_class_entry_has_required_fields(self):
        """Each grade_per_class entry has class, grade, coverage_pct.
        spec: 17.5 lines 6022-6025 — schema per entry
        """
        result = compute_severity(
            predicted_class=0,
            raw_features=_make_features(disease_coverage_pct=10.0, lesion_count=5.0),
            psv_reliability=0.80,
            tier_label="3A",
            multi_class_set=[0, 1],
        )
        for entry in result.grade_per_class:
            assert "class" in entry
            assert "grade" in entry
            assert "coverage_pct" in entry

    def test_grade_per_class_class_names_are_canonical_short_names(self):
        """class field in each entry uses canonical short names.
        spec: 17.5 line 6023 — 'class': 'foliar'
        """
        result = compute_severity(
            predicted_class=0,
            raw_features=_make_features(disease_coverage_pct=10.0, lesion_count=5.0),
            psv_reliability=0.80,
            tier_label="3A",
            multi_class_set=[0, 1],   # foliar=0, septoria=1
        )
        class_names = {e["class"] for e in result.grade_per_class}
        assert "foliar" in class_names
        assert "septoria" in class_names

    def test_grade_per_class_healthy_excluded(self):
        """Healthy class (idx=5) excluded from grade_per_class entries.
        spec: 17.6 lines 6036-6049 — healthy has no severity
        """
        result = compute_severity(
            predicted_class=0,
            raw_features=_make_features(disease_coverage_pct=10.0, lesion_count=5.0),
            psv_reliability=0.80,
            tier_label="3A",
            multi_class_set=[0, 1, 5],   # foliar, septoria, healthy
        )
        assert result.grade_per_class is not None
        class_names = {e["class"] for e in result.grade_per_class}
        assert "healthy" not in class_names
        # Only disease classes remain
        assert len(result.grade_per_class) == 2

    def test_grade_per_class_ood_excluded(self):
        """OOD class (idx=6) excluded from grade_per_class entries.
        spec: 17.6 lines 6044-6049 — OOD has no severity
        """
        result = compute_severity(
            predicted_class=0,
            raw_features=_make_features(disease_coverage_pct=10.0, lesion_count=5.0),
            psv_reliability=0.80,
            tier_label="3B",
            multi_class_set=[0, 2, 6],   # foliar, late_blight, OOD
        )
        class_names = {e["class"] for e in result.grade_per_class}
        assert "OOD" not in class_names
        assert len(result.grade_per_class) == 2

    def test_grade_per_class_none_for_single_class_set(self):
        """grade_per_class remains None when multi_class_set has only 1 disease class.
        spec: 17.5 — only meaningful for ≥ 2 disease classes.
        """
        result = compute_severity(
            predicted_class=0,
            raw_features=_make_features(disease_coverage_pct=10.0, lesion_count=5.0),
            psv_reliability=0.80,
            tier_label="3A",
            multi_class_set=[0],   # only one disease class
        )
        assert result.grade_per_class is None

    def test_grade_per_class_same_coverage_pct_for_all_classes(self):
        """All entries in grade_per_class share the same coverage_pct value.
        SPEC-INT-003 (DEC-050): PSV computes one coverage value per image;
        same value reported for all classes in the prediction set.
        """
        feats = _make_features(disease_coverage_pct=10.0, lesion_count=5.0)
        result = compute_severity(
            predicted_class=0,
            raw_features=feats,
            psv_reliability=0.80,
            tier_label="3A",
            multi_class_set=[0, 1, 2],   # foliar, septoria, late_blight
        )
        coverages = {e["coverage_pct"] for e in result.grade_per_class}
        # All entries must share the same coverage_pct
        assert len(coverages) == 1

    def test_grade_per_class_grades_differ_by_disease_threshold(self):
        """Different diseases can have different grades for the same coverage.
        spec: 17.3 — foliar mild_max=5%, late_blight mild_max=2%
        At 3% coverage: foliar → mild, late_blight → moderate.
        """
        result = compute_severity(
            predicted_class=0,
            raw_features=_make_features(disease_coverage_pct=3.0, lesion_count=2.0),
            psv_reliability=0.80,
            tier_label="3A",
            multi_class_set=[0, 2],   # foliar (mild_max=5%) and late_blight (mild_max=2%)
        )
        grades = {e["class"]: e["grade"] for e in result.grade_per_class}
        assert grades["foliar"] == "mild"       # 3.0 < 5.0 → mild
        assert grades["late_blight"] == "moderate"  # 3.0 >= 2.0 → moderate

    def test_grade_per_class_not_none_when_no_multi_class_set(self):
        """grade_per_class remains None when multi_class_set not provided.
        Existing behavior must not break (test_grade_per_class_default_none).
        spec: 17.5 — multi_class_set is optional.
        """
        result = compute_severity(
            predicted_class=0,
            raw_features=_make_features(disease_coverage_pct=10.0, lesion_count=5.0),
            psv_reliability=0.80,
            tier_label="3A",
            # multi_class_set not passed
        )
        assert result.grade_per_class is None

    def test_grade_per_class_3b_with_three_disease_classes(self):
        """Tier 3B grade_per_class includes all 3 disease classes.
        spec: 17.5 — applies to both 3A (2 classes) and 3B (3+ classes).
        """
        result = compute_severity(
            predicted_class=4,   # mosaic argmax
            raw_features=_make_features(disease_coverage_pct=20.0, lesion_count=3.0),
            psv_reliability=0.80,
            tier_label="3B",
            multi_class_set=[1, 3, 4],   # septoria, ylcv, mosaic
        )
        assert result.grade_per_class is not None
        assert len(result.grade_per_class) == 3
        class_names = {e["class"] for e in result.grade_per_class}
        assert class_names == {"septoria", "ylcv", "mosaic"}

    def test_grade_per_class_grade_field_values_valid(self):
        """Each grade entry has a value in {mild, moderate, severe}.
        spec: 17.3 — valid grade values
        """
        result = compute_severity(
            predicted_class=0,
            raw_features=_make_features(disease_coverage_pct=10.0, lesion_count=5.0),
            psv_reliability=0.80,
            tier_label="3A",
            multi_class_set=[0, 1],
        )
        valid_grades = {"mild", "moderate", "severe"}
        for entry in result.grade_per_class:
            assert entry["grade"] in valid_grades
