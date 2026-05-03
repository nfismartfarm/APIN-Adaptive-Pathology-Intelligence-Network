"""
Unit tests for tomato_sandbox/validation/run_f0.py.

Coverage (30 tests):
  -- Fixtures and helpers --
  1.  _make_minimal_jpeg(): helper creates decodable JPEG bytes
  2.  _make_labeled_csv(): creates CSV + images that run_f0_validation accepts

  -- Schema smoke test --
  3.  run_f0_validation returns a dict with all required top-level keys
  4.  JSON file written to output_dir with valid, round-trippable JSON
  5.  Returned dict and loaded JSON file match (same content)

  -- Tier 4B disposition --
  6.  All images → Tier 4B degraded: tier_4b_count_degraded == n_total
  7.  All images → Tier 4B degraded: tier_4b_count_real_failure == 0
  8.  All images → Tier 4B degraded: is_pre_f0_mode == True
  9.  Mixed Tier 4B (degraded + real): counts split correctly
  10. No Tier 4B images: tier_4b_count_total == 0, is_pre_f0_mode == False

  -- Conformal coverage --
  11. All true classes in prediction_set → coverage_rate == 1.0
  12. No true classes in prediction_set → coverage_rate == 0.0
  13. Half in set → coverage_rate == 0.5
  14. Coverage Wilson CI: lo <= coverage_rate <= hi
  15. Empty prediction_set for each image → coverage_rate == 0.0
  16. Unknown true_class (-1): rows with unknown class are skipped in coverage

  -- Severity validation --
  17. No true_severity in manifest → severity_validation.status == "skipped"
  18. Severity_validation.reason == "skipped_no_ground_truth" when skipped
  19. With true_severity: per_disease accuracy computed; overall_accuracy in [0, 1]
  20. All severity predictions correct → overall_accuracy == 1.0
  21. All severity predictions wrong → overall_accuracy == 0.0
  22. Severity skipped when only healthy/OOD rows have true_severity

  -- Confusion matrix --
  23. Shape: confusion_matrix.matrix is 7×7 (n_classes × n_classes)
  24. Diagonal entries count: correct predictions reflected on diagonal
  25. class_names list preserved in confusion_matrix block

  -- Calibration artifacts --
  26. calibration_dir with placeholder conformal_tau.json: surfaced in metadata
  27. calibration_dir with no files: values are "not_found"
  28. psv_standardization.json placeholder: surfaced in metadata

  -- Error handling --
  29. CSV with no test-split rows → ValueError
  30. labeled_data_path does not exist → FileNotFoundError

  -- Output isolation --
  (All tests use tmp_path; never write to production calibration dir)

# spec: section 29 lines 8105-8243 — F.0 validation suite
# spec: section 13.4 lines 3564-3581 — coverage target 90%
# spec: section 17.3 lines 5966-5982 — per-disease severity thresholds
# spec: section 16.2 lines 5655-5712 — response schema
# DEC-053 — run_f0.py architectural decisions
"""

from __future__ import annotations

import csv
import io
import json
import struct
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tomato_sandbox.validation.run_f0 import (
    _build_confusion_matrix,
    _compute_conformal_coverage,
    _compute_severity_validation,
    _compute_tier_disposition,
    _get_prediction_set,
    _get_tier_label,
    _is_tier_4b_degraded,
    _is_error_response,
    _wilson_ci_95,
    run_f0_validation,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_CLASS_NAMES = [
    "foliar", "septoria", "late_blight", "ylcv", "mosaic", "healthy", "OOD",
]
_NUM_CLASSES = 7
_DISEASE_NAMES = ["foliar", "septoria", "late_blight", "ylcv", "mosaic"]


# ===========================================================================
# Test 1: minimal JPEG helper
# ===========================================================================

def _make_minimal_jpeg() -> bytes:
    """Smallest decodable JPEG (1×1 grey).

    Same pattern as test_fit_calibration.py for consistency.
    """
    return bytes([
        0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
        0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00,
        0xFF, 0xDB, 0x00, 0x43, 0x00,
        *([0x08] * 64),
        0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01, 0x11, 0x00,
        0xFF, 0xC4, 0x00, 0x1F, 0x00, 0x00, 0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01,
        0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04,
        0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B,
        0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3F, 0x00, 0xFB, 0x26,
        0xFF, 0xD9,
    ])


class TestMinimalJpeg:
    """Test 1: helper produces non-empty bytes."""

    def test_makes_bytes(self):
        data = _make_minimal_jpeg()
        assert isinstance(data, bytes)
        assert len(data) > 0


# ===========================================================================
# Test 2: CSV + image fixture helper
# ===========================================================================

def _make_labeled_csv(
    tmp_path: Path,
    n: int = 4,
    split: str = "test",
    true_class: str = "foliar",
    true_severity: str = "",
    is_confirmed_tomato: str = "1",
) -> Path:
    """Create n images + a labeled CSV with n test rows.

    Returns csv_path. Images go to tmp_path/images/.
    """
    img_dir = tmp_path / "images"
    img_dir.mkdir(exist_ok=True)
    jpeg_bytes = _make_minimal_jpeg()

    rows = []
    classes = _CLASS_NAMES[:n] if n <= _NUM_CLASSES else [_CLASS_NAMES[i % _NUM_CLASSES] for i in range(n)]
    for i in range(n):
        fname = f"img_{i:03d}.jpg"
        fpath = img_dir / fname
        fpath.write_bytes(jpeg_bytes)
        rows.append({
            "image_path": str(fpath),
            "true_class": classes[i],
            "split": split,
            "true_severity": true_severity,
            "is_confirmed_tomato": is_confirmed_tomato,
        })

    csv_path = tmp_path / "labeled.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["image_path", "true_class", "split", "true_severity", "is_confirmed_tomato"],
        )
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


class TestCsvHelper:
    """Test 2: CSV helper creates valid CSV with image paths."""

    def test_csv_created(self, tmp_path):
        csv_path = _make_labeled_csv(tmp_path, n=3)
        assert csv_path.exists()
        with csv_path.open() as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert len(rows) == 3
        for row in rows:
            assert Path(row["image_path"]).exists()


# ===========================================================================
# Mock predict_single — returns Tier 4B degraded (pre-F.0 mode)
# ===========================================================================

def _make_tier_4b_degraded_response(
    request_id: str = "test",
    prediction_set: list[str] | None = None,
    primary_class: str | None = None,
) -> dict:
    """Build a S16.2-shaped Tier 4B degraded response (Rule 1 / pipeline_failure)."""
    return {
        "request_id": request_id,
        "tier": {"label": "4B", "human_readable": "Pipeline issue", "alert_level": "error"},
        "prediction": {
            "primary_class": primary_class,
            "primary_confidence": 0.0,
            "prediction_set": prediction_set or [],
            "prediction_set_human": [],
        },
        "tier5_alert": {"fired": False, "reason": None},
        "severity": {"grade": None, "human_readable": None, "details": None},
        "explanation": {
            "user_strings": ["Pipeline issue"],
            "structured": {
                "rule_id_fired": "pipeline_failure",  # DEC-053 Decision 4 sentinel
                "sub_rule_id_fired": None,
                "psv_reliability": 0.0,
                "iqa_decision": "ACCEPTABLE",
            },
        },
        "warnings": [],
        "model_version": "test",
        "processing_time_ms": 1,
    }


def _make_tier_4b_real_failure_response(
    request_id: str = "test",
) -> dict:
    """Build a Tier 4B with a non-degraded rule_id (simulating a real bug)."""
    resp = _make_tier_4b_degraded_response(request_id)
    resp["explanation"]["structured"]["rule_id_fired"] = "rule_9_catch_all"
    return resp


def _make_tier1_response(
    request_id: str = "test",
    primary_class: str = "foliar",
    prediction_set: list[str] | None = None,
    severity_grade: str | None = "moderate",
) -> dict:
    """Build a S16.2-shaped Tier 1 response."""
    return {
        "request_id": request_id,
        "tier": {"label": "1", "human_readable": "Definitive prediction", "alert_level": "info"},
        "prediction": {
            "primary_class": primary_class,
            "primary_confidence": 0.91,
            "prediction_set": prediction_set if prediction_set is not None else [primary_class],
            "prediction_set_human": [],
        },
        "tier5_alert": {"fired": False, "reason": None},
        "severity": {"grade": severity_grade, "human_readable": "Moderate severity", "details": {}},
        "explanation": {
            "user_strings": ["Clear prediction"],
            "structured": {
                "rule_id_fired": "definitive_single_class",
                "sub_rule_id_fired": None,
                "psv_reliability": 0.78,
                "iqa_decision": "ACCEPTABLE",
            },
        },
        "warnings": [],
        "model_version": "test",
        "processing_time_ms": 10,
    }


# ===========================================================================
# Test 3 & 4 & 5: Schema smoke test
# ===========================================================================

class TestSchemaSmoke:
    """Tests 3-5: run_f0_validation returns correct structure and JSON."""

    REQUIRED_TOP_KEYS = {
        "metadata",
        "per_image_predictions",
        "confusion_matrix",
        "conformal_coverage",
        "severity_validation",
        "tier_disposition",
    }

    def _run_with_mock(self, tmp_path, side_effect_fn=None):
        """Helper: run run_f0_validation with mocked predict_single."""
        csv_path = _make_labeled_csv(tmp_path, n=3, split="test")
        mock_ctx = MagicMock()
        call_counter = [0]

        def mock_predict(image_bytes, request_id, context):
            i = call_counter[0]
            call_counter[0] += 1
            if side_effect_fn:
                return side_effect_fn(i, request_id)
            return _make_tier_4b_degraded_response(request_id)

        with patch(
            "tomato_sandbox.validation.run_f0.predict_single",
            side_effect=mock_predict,
        ):
            report = run_f0_validation(csv_path, mock_ctx, output_dir=tmp_path)

        return report

    def test_returns_dict_with_required_keys(self, tmp_path):
        """Test 3: returned dict has all required top-level keys."""
        report = self._run_with_mock(tmp_path)
        assert isinstance(report, dict)
        assert self.REQUIRED_TOP_KEYS.issubset(report.keys()), (
            f"Missing keys: {self.REQUIRED_TOP_KEYS - report.keys()}"
        )

    def test_json_file_written_and_valid(self, tmp_path):
        """Test 4: JSON file written to output_dir; can be re-loaded."""
        self._run_with_mock(tmp_path)
        json_files = list(tmp_path.glob("validation_report_*.json"))
        assert len(json_files) == 1, f"Expected 1 report JSON, found {len(json_files)}"
        with json_files[0].open() as fh:
            loaded = json.load(fh)
        assert isinstance(loaded, dict)
        assert self.REQUIRED_TOP_KEYS.issubset(loaded.keys())

    def test_returned_dict_matches_json_file(self, tmp_path):
        """Test 5: returned dict and loaded JSON file have same keys."""
        report = self._run_with_mock(tmp_path)
        json_files = list(tmp_path.glob("validation_report_*.json"))
        assert len(json_files) == 1
        with json_files[0].open() as fh:
            loaded = json.load(fh)
        # Top-level keys must match
        assert set(report.keys()) == set(loaded.keys())
        # n_processed must be consistent
        assert report["metadata"]["n_processed"] == loaded["metadata"]["n_processed"]


# ===========================================================================
# Tests 6-10: Tier 4B disposition
# ===========================================================================

class TestTier4BDisposition:
    """Tests 6-10: Tier 4B degraded vs real failure tracking."""

    def _run_with_responses(self, tmp_path, response_fn):
        """Run validation with a per-index response factory."""
        n = 4
        csv_path = _make_labeled_csv(tmp_path, n=n, split="test")
        mock_ctx = MagicMock()
        call_counter = [0]

        def mock_predict(image_bytes, request_id, context):
            i = call_counter[0]
            call_counter[0] += 1
            return response_fn(i)

        with patch(
            "tomato_sandbox.validation.run_f0.predict_single",
            side_effect=mock_predict,
        ):
            return run_f0_validation(csv_path, mock_ctx, output_dir=tmp_path)

    def test_all_4b_degraded_count(self, tmp_path):
        """Test 6: all Tier 4B degraded → tier_4b_count_degraded == n."""
        report = self._run_with_responses(
            tmp_path,
            lambda i: _make_tier_4b_degraded_response(f"req_{i}"),
        )
        td = report["tier_disposition"]
        assert td["tier_4b_count_degraded"] == 4

    def test_all_4b_real_failure_zero(self, tmp_path):
        """Test 7: all Tier 4B degraded → tier_4b_count_real_failure == 0."""
        report = self._run_with_responses(
            tmp_path,
            lambda i: _make_tier_4b_degraded_response(f"req_{i}"),
        )
        td = report["tier_disposition"]
        assert td["tier_4b_count_real_failure"] == 0

    def test_all_4b_degraded_is_pre_f0_mode(self, tmp_path):
        """Test 8: all Tier 4B degraded → is_pre_f0_mode == True."""
        report = self._run_with_responses(
            tmp_path,
            lambda i: _make_tier_4b_degraded_response(f"req_{i}"),
        )
        td = report["tier_disposition"]
        assert td["is_pre_f0_mode"] is True

    def test_mixed_4b_counts_split(self, tmp_path):
        """Test 9: 2 degraded + 2 real failure → counts split correctly."""
        def response_fn(i):
            if i < 2:
                return _make_tier_4b_degraded_response(f"req_{i}")
            else:
                return _make_tier_4b_real_failure_response(f"req_{i}")

        report = self._run_with_responses(tmp_path, response_fn)
        td = report["tier_disposition"]
        assert td["tier_4b_count_total"] == 4
        assert td["tier_4b_count_degraded"] == 2
        assert td["tier_4b_count_real_failure"] == 2

    def test_no_tier_4b_counts_zero(self, tmp_path):
        """Test 10: no Tier 4B responses → tier_4b_count_total == 0."""
        report = self._run_with_responses(
            tmp_path,
            lambda i: _make_tier1_response(f"req_{i}", primary_class=_CLASS_NAMES[i % 5]),
        )
        td = report["tier_disposition"]
        assert td["tier_4b_count_total"] == 0
        assert td["is_pre_f0_mode"] is False


# ===========================================================================
# Tests 11-16: Conformal coverage
# ===========================================================================

class TestConformalCoverage:
    """Tests 11-16: conformal coverage computation."""

    def test_all_true_in_set_coverage_one(self):
        """Test 11: all true classes in prediction_set → coverage_rate == 1.0."""
        true_classes = [0, 1, 2, 3]  # foliar, septoria, late_blight, ylcv
        prediction_sets = [
            ["foliar"],
            ["septoria"],
            ["late_blight"],
            ["ylcv"],
        ]
        result = _compute_conformal_coverage(true_classes, prediction_sets)
        assert result["coverage_rate"] == 1.0

    def test_none_in_set_coverage_zero(self):
        """Test 12: no true classes in prediction_set → coverage_rate == 0.0."""
        true_classes = [0, 1, 2, 3]
        prediction_sets = [
            ["septoria"],    # true is foliar
            ["late_blight"],  # true is septoria
            ["ylcv"],         # true is late_blight
            ["mosaic"],       # true is ylcv
        ]
        result = _compute_conformal_coverage(true_classes, prediction_sets)
        assert result["coverage_rate"] == 0.0

    def test_half_in_set_coverage_half(self):
        """Test 13: half in set → coverage_rate == 0.5."""
        true_classes = [0, 1, 0, 1]
        prediction_sets = [
            ["foliar"],   # covered
            ["mosaic"],   # not covered (true=septoria)
            ["foliar"],   # covered
            ["mosaic"],   # not covered
        ]
        result = _compute_conformal_coverage(true_classes, prediction_sets)
        assert result["coverage_rate"] == pytest.approx(0.5, abs=1e-9)

    def test_wilson_ci_bounds_coverage_rate(self):
        """Test 14: coverage Wilson CI: lo <= coverage_rate <= hi."""
        true_classes = [0, 1, 2]
        prediction_sets = [["foliar"], ["mosaic"], ["late_blight"]]
        result = _compute_conformal_coverage(true_classes, prediction_sets)
        rate = result["coverage_rate"]
        ci_lo, ci_hi = result["coverage_ci_95_wilson"]
        assert ci_lo <= rate <= ci_hi

    def test_empty_prediction_sets_coverage_zero(self):
        """Test 15: empty prediction_set for all images → coverage_rate == 0.0."""
        true_classes = [0, 1, 2]
        prediction_sets = [[], [], []]
        result = _compute_conformal_coverage(true_classes, prediction_sets)
        assert result["coverage_rate"] == 0.0

    def test_unknown_true_class_skipped(self):
        """Test 16: rows with true_class=-1 skipped; known rows counted correctly."""
        # Mix of known and unknown true classes
        true_classes = [-1, 0, -1, 1]  # 2 known: foliar, septoria
        prediction_sets = [
            ["foliar"],    # skipped (-1)
            ["foliar"],    # covered (foliar in set)
            ["septoria"],  # skipped (-1)
            ["mosaic"],    # not covered (true=septoria, set=[mosaic])
        ]
        result = _compute_conformal_coverage(true_classes, prediction_sets)
        # Only 2 rows count; 1 covered → 0.5
        assert result["n_total"] == 2
        assert result["n_covered"] == 1
        assert result["coverage_rate"] == pytest.approx(0.5, abs=1e-9)


# ===========================================================================
# Tests 17-22: Severity validation
# ===========================================================================

class TestSeverityValidation:
    """Tests 17-22: per-disease severity validation."""

    def test_no_severity_in_manifest_skipped(self):
        """Test 17: no true_severity → status == 'skipped'."""
        per_image = [
            {"true_class": "foliar", "true_severity": "", "pred_severity": "mild"},
            {"true_class": "septoria", "true_severity": "", "pred_severity": "moderate"},
        ]
        result = _compute_severity_validation(per_image)
        assert result["status"] == "skipped"

    def test_no_severity_reason_string(self):
        """Test 18: skipped reason is 'skipped_no_ground_truth'."""
        per_image = [
            {"true_class": "foliar", "true_severity": "", "pred_severity": None},
        ]
        result = _compute_severity_validation(per_image)
        assert result["reason"] == "skipped_no_ground_truth"

    def test_with_severity_accuracy_computed(self):
        """Test 19: with true_severity, per_disease accuracy in [0, 1]."""
        per_image = [
            {"true_class": "foliar", "true_severity": "mild", "pred_severity": "mild"},
            {"true_class": "foliar", "true_severity": "moderate", "pred_severity": "mild"},
        ]
        result = _compute_severity_validation(per_image)
        assert result["status"] == "ok"
        acc = result["per_disease"]["foliar"]["accuracy"]
        assert 0.0 <= acc <= 1.0

    def test_all_correct_accuracy_one(self):
        """Test 20: all correct → overall_accuracy == 1.0."""
        per_image = [
            {"true_class": "foliar", "true_severity": "mild", "pred_severity": "mild"},
            {"true_class": "septoria", "true_severity": "severe", "pred_severity": "severe"},
        ]
        result = _compute_severity_validation(per_image)
        assert result["overall_accuracy"] == pytest.approx(1.0, abs=1e-9)

    def test_all_wrong_accuracy_zero(self):
        """Test 21: all wrong → overall_accuracy == 0.0."""
        per_image = [
            {"true_class": "foliar", "true_severity": "mild", "pred_severity": "moderate"},
            {"true_class": "septoria", "true_severity": "severe", "pred_severity": "mild"},
        ]
        result = _compute_severity_validation(per_image)
        assert result["overall_accuracy"] == pytest.approx(0.0, abs=1e-9)

    def test_severity_skipped_for_healthy_ood_only(self):
        """Test 22: severity skipped when only healthy/OOD rows have true_severity."""
        per_image = [
            {"true_class": "healthy", "true_severity": "mild", "pred_severity": "mild"},
            {"true_class": "OOD", "true_severity": "moderate", "pred_severity": "moderate"},
        ]
        result = _compute_severity_validation(per_image)
        # healthy and OOD are not in _DISEASE_NAMES → no disease rows → skipped
        assert result["status"] == "skipped"


# ===========================================================================
# Tests 23-25: Confusion matrix
# ===========================================================================

class TestConfusionMatrix:
    """Tests 23-25: confusion matrix structure and values."""

    def test_confusion_matrix_shape(self):
        """Test 23: confusion_matrix.matrix is 7×7."""
        true = [0, 1, 2]
        pred = [0, 1, 2]
        cm = _build_confusion_matrix(true, pred, _NUM_CLASSES)
        assert len(cm) == _NUM_CLASSES
        for row in cm:
            assert len(row) == _NUM_CLASSES

    def test_diagonal_entries_correct(self):
        """Test 24: correct predictions appear on diagonal."""
        true = [0, 1, 0]
        pred = [0, 1, 0]
        cm = _build_confusion_matrix(true, pred, _NUM_CLASSES)
        assert cm[0][0] == 2  # 2 correct for class 0
        assert cm[1][1] == 1  # 1 correct for class 1

    def test_class_names_in_block(self, tmp_path):
        """Test 25: class_names preserved in confusion_matrix block of report."""
        csv_path = _make_labeled_csv(tmp_path, n=2, split="test")
        mock_ctx = MagicMock()
        call_counter = [0]

        def mock_predict(image_bytes, request_id, context):
            i = call_counter[0]
            call_counter[0] += 1
            return _make_tier_4b_degraded_response(request_id)

        with patch(
            "tomato_sandbox.validation.run_f0.predict_single",
            side_effect=mock_predict,
        ):
            report = run_f0_validation(csv_path, mock_ctx, output_dir=tmp_path)

        assert report["confusion_matrix"]["class_names"] == _CLASS_NAMES


# ===========================================================================
# Tests 26-28: Calibration artifacts
# ===========================================================================

class TestCalibrationArtifacts:
    """Tests 26-28: calibration artifacts surfaced in metadata."""

    def _make_placeholder_conformal_tau(self, calib_dir: Path) -> None:
        (calib_dir / "conformal_tau.json").write_text(
            json.dumps({
                "tau": 0.42,
                "alpha": 0.10,
                "n_calibration": 40,
                "calibration_timestamp": "pre-F.0-placeholder",
            })
        )

    def _make_placeholder_psv_std(self, calib_dir: Path) -> None:
        (calib_dir / "psv_standardization.json").write_text(
            json.dumps({
                "_comment": "placeholder",
                "F0_FEATURE_MEAN": [0.0] * 26,
                "F0_FEATURE_STD": [1.0] * 26,
                "T_PSV": 1.0,
            })
        )

    def test_conformal_tau_placeholder_surfaced(self, tmp_path):
        """Test 26: placeholder conformal_tau.json surfaced in metadata."""
        calib_dir = tmp_path / "calib"
        calib_dir.mkdir()
        self._make_placeholder_conformal_tau(calib_dir)

        csv_path = _make_labeled_csv(tmp_path, n=2, split="test")
        mock_ctx = MagicMock()

        with patch(
            "tomato_sandbox.validation.run_f0.predict_single",
            side_effect=lambda b, r, c: _make_tier_4b_degraded_response(r),
        ):
            report = run_f0_validation(
                csv_path, mock_ctx,
                output_dir=tmp_path,
                calibration_dir=calib_dir,
            )

        artifacts = report["metadata"]["calibration_artifacts"]
        assert isinstance(artifacts["conformal_tau"], dict)
        assert artifacts["conformal_tau"]["tau"] == pytest.approx(0.42)

    def test_no_files_yields_not_found(self, tmp_path):
        """Test 27: calibration_dir with no files → values are 'not_found'."""
        calib_dir = tmp_path / "empty_calib"
        calib_dir.mkdir()
        csv_path = _make_labeled_csv(tmp_path, n=2, split="test")
        mock_ctx = MagicMock()

        with patch(
            "tomato_sandbox.validation.run_f0.predict_single",
            side_effect=lambda b, r, c: _make_tier_4b_degraded_response(r),
        ):
            report = run_f0_validation(
                csv_path, mock_ctx,
                output_dir=tmp_path,
                calibration_dir=calib_dir,
            )

        artifacts = report["metadata"]["calibration_artifacts"]
        assert artifacts["conformal_tau"] == "not_found"
        assert artifacts["psv_standardization"] == "not_found"

    def test_psv_standardization_placeholder_surfaced(self, tmp_path):
        """Test 28: psv_standardization.json surfaced in metadata."""
        calib_dir = tmp_path / "calib2"
        calib_dir.mkdir()
        self._make_placeholder_psv_std(calib_dir)

        csv_path = _make_labeled_csv(tmp_path, n=2, split="test")
        mock_ctx = MagicMock()

        with patch(
            "tomato_sandbox.validation.run_f0.predict_single",
            side_effect=lambda b, r, c: _make_tier_4b_degraded_response(r),
        ):
            report = run_f0_validation(
                csv_path, mock_ctx,
                output_dir=tmp_path,
                calibration_dir=calib_dir,
            )

        artifacts = report["metadata"]["calibration_artifacts"]
        assert isinstance(artifacts["psv_standardization"], dict)
        # Placeholder T_PSV should be surfaced
        assert artifacts["psv_standardization"]["T_PSV"] == pytest.approx(1.0)


# ===========================================================================
# Tests 29-30: Error handling
# ===========================================================================

class TestErrorHandling:
    """Tests 29-30: error handling for bad inputs."""

    def test_no_test_rows_raises_value_error(self, tmp_path):
        """Test 29: CSV with no split='test' rows → ValueError."""
        csv_path = _make_labeled_csv(tmp_path, n=3, split="calibration")
        mock_ctx = MagicMock()

        with pytest.raises(ValueError, match="no rows with split='test'"):
            with patch(
                "tomato_sandbox.validation.run_f0.predict_single",
                side_effect=lambda b, r, c: _make_tier_4b_degraded_response(r),
            ):
                run_f0_validation(csv_path, mock_ctx, output_dir=tmp_path)

    def test_missing_path_raises_file_not_found(self, tmp_path):
        """Test 30: labeled_data_path does not exist → FileNotFoundError."""
        missing = tmp_path / "does_not_exist.csv"
        mock_ctx = MagicMock()

        with pytest.raises(FileNotFoundError, match="labeled data file not found"):
            run_f0_validation(missing, mock_ctx, output_dir=tmp_path)


# ===========================================================================
# Additional unit tests for internal helpers
# ===========================================================================

class TestWilsonCi:
    """Wilson CI edge cases (covers DEC-053 Decision 5)."""

    def test_zero_total_degenerate(self):
        lo, hi = _wilson_ci_95(0, 0)
        assert lo == 0.0
        assert hi == 1.0

    def test_all_success(self):
        lo, hi = _wilson_ci_95(100, 100)
        assert lo >= 0.90  # very high coverage
        assert hi <= 1.0

    def test_symmetry_at_half(self):
        lo, hi = _wilson_ci_95(50, 100)
        center = (lo + hi) / 2
        assert abs(center - 0.5) < 0.02  # near 0.5


class TestResponseParsers:
    """Helper function correctness."""

    def test_get_tier_label_4b(self):
        resp = _make_tier_4b_degraded_response()
        assert _get_tier_label(resp) == "4B"

    def test_get_tier_label_1(self):
        resp = _make_tier1_response()
        assert _get_tier_label(resp) == "1"

    def test_get_prediction_set_empty(self):
        resp = _make_tier_4b_degraded_response()
        assert _get_prediction_set(resp) == []

    def test_get_prediction_set_single(self):
        resp = _make_tier1_response(primary_class="foliar", prediction_set=["foliar"])
        assert _get_prediction_set(resp) == ["foliar"]

    def test_is_tier_4b_degraded_true(self):
        resp = _make_tier_4b_degraded_response()
        assert _is_tier_4b_degraded(resp) is True

    def test_is_tier_4b_degraded_false_for_tier1(self):
        resp = _make_tier1_response()
        assert _is_tier_4b_degraded(resp) is False

    def test_is_tier_4b_degraded_false_for_real_failure(self):
        resp = _make_tier_4b_real_failure_response()
        assert _is_tier_4b_degraded(resp) is False

    def test_is_error_response_true(self):
        assert _is_error_response({"error": "IMAGE_DECODE_FAILED"}) is True

    def test_is_error_response_false(self):
        assert _is_error_response(_make_tier1_response()) is False
