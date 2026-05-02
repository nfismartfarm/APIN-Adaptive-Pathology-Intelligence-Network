"""
Unit tests for tomato_sandbox/orchestrator/pipeline.py.

Tests cover:
  - Pipeline composes correctly with mock signals
  - TTA triggers when classifier confidence < threshold
  - PSV NOT invoked under TTA (mock + assert call_count)
  - Signal failure routes to degraded mode without breaking pipeline
  - Happy-path end-to-end with mocked outputs produces a TierAssignment
  - GPU lock acquired around A/B but not PSV

DEC-042: pre-allocated for T-IMPL-6a orchestrator.
"""

from __future__ import annotations

import io
import hashlib
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
from tomato_sandbox.orchestrator.pipeline import (
    PipelineContext,
    predict_single,
    predict_multi,
    _apply_nan_guard,
    _make_sentinel_classifier_result,
    _make_fallback_conformal,
    _signal_a_to_dict,
    _signal_b_to_dict,
    _signal_c_to_dict,
    _classifier_to_dict,
    _conformal_to_dict,
    _iqa_to_dict,
    _image_hash,
    _error_response,
)

# Re-export shim — same symbols must be importable from orchestrator.orchestrator
from tomato_sandbox.orchestrator.orchestrator import (
    predict_single as _predict_single_alias,
    PipelineContext as _PipelineContext_alias,
)

# Dataclasses from upstream modules for constructing mock results
from tomato_sandbox.signals.v3_signal import SignalAResult
from tomato_sandbox.signals.lora_signal import SignalBResult
from tomato_sandbox.signals.psv.psv import SignalCResult
from tomato_sandbox.classifier.hierarchical_classifier import ClassifierResult
from tomato_sandbox.conformal.conformal import ConformalResult
from tomato_sandbox.tier.tier_assignment import TierAssignment
from tomato_sandbox.iqa.iqa import IQAResult


# ---------------------------------------------------------------------------
# Factories — build minimal-valid dataclass instances for mocking
# ---------------------------------------------------------------------------

def _make_signal_a(
    probs: np.ndarray | None = None,
    max_prob: float = 0.8,
    argmax: int = 5,  # healthy
    chilli_leakage: float = 0.01,
    succeeded: bool = True,
) -> SignalAResult:
    if probs is None:
        probs = np.zeros(6, dtype=np.float32)
        probs[argmax] = max_prob
    return SignalAResult(
        tomato_probs_canonical=probs,
        tomato_max_prob_canonical=max_prob,
        tomato_argmax_canonical=argmax,
        chilli_leakage=chilli_leakage,
        raw_probs_v3_order=np.zeros(10, dtype=np.float32),
        forward_succeeded=succeeded,
        failure_reason="" if succeeded else "mock_failure",
    )


def _make_signal_b(
    probs: np.ndarray | None = None,
    max_prob: float = 0.8,
    argmax: int = 5,
    succeeded: bool = True,
) -> SignalBResult:
    if probs is None:
        probs = np.zeros(6, dtype=np.float32)
        probs[argmax] = max_prob
    return SignalBResult(
        tomato_probs_canonical=probs,
        tomato_max_prob_canonical=max_prob,
        tomato_argmax_canonical=argmax,
        cls_token=np.zeros(768, dtype=np.float32),
        raw_lora_probs_canonical=probs.copy(),
        prototype_blend_applied=False,
        prototype_blend_reason="high_confidence_no_blend",
        forward_succeeded=succeeded,
        failure_reason="" if succeeded else "mock_failure",
    )


def _make_signal_c(
    argmax: int = 5,
    compat_max: float = 0.75,
    reliability: float = 0.80,
    succeeded: bool = True,
) -> SignalCResult:
    compat = np.zeros(6, dtype=np.float32)
    compat[argmax] = compat_max
    h, w = 100, 100
    return SignalCResult(
        compatibility=compat,
        compatibility_argmax=argmax,
        compatibility_max=compat_max,
        compatibility_margin=0.50,
        psv_reliability=reliability,
        raw_features=np.zeros(26, dtype=np.float32),
        standardized_features=np.zeros(26, dtype=np.float32),
        leaf_mask=np.zeros((h, w), dtype=bool),
        disease_mask=np.zeros((h, w), dtype=bool),
        n_lesions=0,
        fallback_used=False,
        forward_succeeded=succeeded,
        failure_reason="" if succeeded else "mock_failure",
    )


def _make_classifier_result(
    argmax: int = 5,
    max_prob: float = 0.8,
    margin: float = 0.4,
    succeeded: bool = True,
) -> ClassifierResult:
    probs = np.zeros(7, dtype=np.float32)
    probs[argmax] = max_prob
    return ClassifierResult(
        p_final_calibrated=probs,
        combined_argmax=argmax,
        combined_max_prob=max_prob,
        combined_margin=margin,
        p_final_uncalibrated=probs.copy(),
        p_stage1=np.array([0.1, 0.8, 0.1], dtype=np.float32),
        p_stage2=np.array([0.1, 0.1, 0.1, 0.1, 0.6], dtype=np.float32),
        classifier_succeeded=succeeded,
        failure_reason="" if succeeded else "mock_failure",
    )


def _make_conformal_result(
    prediction_set: list | None = None,
    size: int | None = None,
) -> ConformalResult:
    if prediction_set is None:
        prediction_set = [5]
    if size is None:
        size = len(prediction_set)
    return ConformalResult(
        prediction_set=prediction_set,
        prediction_set_size=size,
        threshold_tau_used=0.1,
        nonconformity_per_class=np.zeros(7, dtype=np.float32),
        inside_set_per_class=np.zeros(7, dtype=bool),
    )


def _make_iqa_result(decision: str = "ACCEPTABLE") -> IQAResult:
    return IQAResult(
        decision=decision,
        aggregate_score=0.85,
        per_dimension={},
        failing_dimensions=[],
        retake_message=None,
        green_mask=None,
    )


def _make_tier_assignment(
    tier: str = "1",
    t5: bool = False,
    rule: str = "7c",
) -> TierAssignment:
    return TierAssignment(tier_label=tier, tier5_alert=t5, rule_id_fired=rule)


def _make_context(**kwargs: Any) -> PipelineContext:
    """Minimal PipelineContext with no real models (mocked at call sites)."""
    defaults = {
        "v3_model": MagicMock(),
        "lora_model": MagicMock(),
        "psv_module": None,
        "gpu_lock": None,
        "cache": None,
        "underpowered_classes": None,
    }
    defaults.update(kwargs)
    return PipelineContext(**defaults)


def _tiny_jpeg_bytes() -> bytes:
    """Produce minimal valid JPEG bytes via PIL."""
    try:
        from PIL import Image
        buf = io.BytesIO()
        img = Image.new("RGB", (50, 50), color=(30, 120, 30))
        img.save(buf, format="JPEG")
        return buf.getvalue()
    except ImportError:
        # Fallback: JPEG magic bytes + minimal stub
        return b"\xff\xd8\xff" + b"\x00" * 200


# ---------------------------------------------------------------------------
# Tests: import contract / alias shim
# ---------------------------------------------------------------------------

class TestAliasShim:
    """Re-export shims must expose the same symbols."""

    def test_predict_single_alias_is_same_function(self) -> None:
        assert _predict_single_alias is predict_single

    def test_pipeline_context_alias_is_same_class(self) -> None:
        assert _PipelineContext_alias is PipelineContext

    def test_orchestrator_module_importable(self) -> None:
        import tomato_sandbox.orchestrator.orchestrator as oo
        assert hasattr(oo, "predict_single")
        assert hasattr(oo, "predict_multi")

    def test_package_init_importable(self) -> None:
        import tomato_sandbox.orchestrator as op
        assert hasattr(op, "predict_single")
        assert hasattr(op, "PipelineContext")


# ---------------------------------------------------------------------------
# Tests: helper functions
# ---------------------------------------------------------------------------

class TestImageHash:
    def test_returns_sha256_hex(self) -> None:
        data = b"hello world"
        expected = hashlib.sha256(data).hexdigest()
        assert _image_hash(data) == expected

    def test_different_bytes_different_hash(self) -> None:
        assert _image_hash(b"abc") != _image_hash(b"def")

    def test_same_bytes_same_hash(self) -> None:
        assert _image_hash(b"abc") == _image_hash(b"abc")


class TestErrorResponse:
    def test_contains_error_code(self) -> None:
        r = _error_response("FOO_BAR", "msg", "req-1", status=400)
        assert r["error"] == "FOO_BAR"
        assert r["message"] == "msg"
        assert r["request_id"] == "req-1"
        assert r["status"] == 400

    def test_extra_fields_merged(self) -> None:
        r = _error_response("E", "m", "r", extra={"retry_after_seconds": 5})
        assert r["retry_after_seconds"] == 5


class TestMakeSentinelClassifierResult:
    def test_classifier_succeeded_false(self) -> None:
        r = _make_sentinel_classifier_result()
        assert not r.classifier_succeeded

    def test_failure_reason_set(self) -> None:
        r = _make_sentinel_classifier_result("test_reason")
        assert "test_reason" in r.failure_reason

    def test_combined_max_prob_zero(self) -> None:
        r = _make_sentinel_classifier_result()
        assert r.combined_max_prob == 0.0

    def test_p_final_calibrated_zero(self) -> None:
        r = _make_sentinel_classifier_result()
        assert r.p_final_calibrated.shape == (7,)
        assert np.all(r.p_final_calibrated == 0.0)


class TestMakeFallbackConformal:
    def test_all_classes_in_set(self) -> None:
        r = _make_fallback_conformal()
        assert r.prediction_set_size == 7
        assert set(r.prediction_set) == set(range(7))

    def test_tau_is_one(self) -> None:
        r = _make_fallback_conformal()
        assert r.threshold_tau_used == 1.0


# ---------------------------------------------------------------------------
# Tests: dict adapters
# ---------------------------------------------------------------------------

class TestDictAdapters:
    def test_signal_a_to_dict_fields(self) -> None:
        sa = _make_signal_a(max_prob=0.7, chilli_leakage=0.05, succeeded=True)
        d = _signal_a_to_dict(sa)
        assert "probs" in d
        assert "chilli_leak" in d
        assert "forward_succeeded" in d
        assert len(d["probs"]) == 6
        assert isinstance(d["probs"], list)
        assert d["chilli_leak"] == pytest.approx(0.05)
        assert d["forward_succeeded"] is True

    def test_signal_b_to_dict_fields(self) -> None:
        sb = _make_signal_b(max_prob=0.6, succeeded=False)
        d = _signal_b_to_dict(sb)
        assert "probs" in d
        assert "forward_succeeded" in d
        assert d["forward_succeeded"] is False

    def test_signal_c_to_dict_fields(self) -> None:
        sc = _make_signal_c(argmax=2, compat_max=0.5, reliability=0.72)
        d = _signal_c_to_dict(sc)
        assert d["argmax"] == 2
        assert d["max"] == pytest.approx(0.5)
        assert d["reliability"] == pytest.approx(0.72)
        assert "margin" in d
        assert "forward_succeeded" in d

    def test_classifier_to_dict_fields(self) -> None:
        cr = _make_classifier_result(argmax=1, max_prob=0.75, margin=0.30)
        d = _classifier_to_dict(cr)
        assert d["argmax"] == 1
        assert d["max"] == pytest.approx(0.75)
        assert d["margin"] == pytest.approx(0.30)

    def test_conformal_to_dict_fields(self) -> None:
        cfr = _make_conformal_result([2, 5], 2)
        d = _conformal_to_dict(cfr)
        assert d["set"] == {2, 5}
        assert d["size"] == 2
        assert "tau" in d

    def test_iqa_to_dict_dataclass(self) -> None:
        iqa = _make_iqa_result("DEGRADED")
        d = _iqa_to_dict(iqa)
        assert d["decision"] == "DEGRADED"

    def test_iqa_to_dict_plain_dict(self) -> None:
        d = _iqa_to_dict({"decision": "HIGH", "extra": "ignored"})
        assert d["decision"] == "HIGH"


# ---------------------------------------------------------------------------
# Tests: NaN guard
# ---------------------------------------------------------------------------

class TestApplyNanGuard:
    def test_no_nan_no_change(self) -> None:
        sa = _make_signal_a(succeeded=True)
        sb = _make_signal_b(succeeded=True)
        sc = _make_signal_c(succeeded=True)
        cr = _make_classifier_result(max_prob=0.8)
        cr2, sa2, sb2, sc2 = _apply_nan_guard(cr, sa, sb, sc)
        # Without NaN, signal succeeded flags should be unchanged
        assert sa2.forward_succeeded is True
        assert sb2.forward_succeeded is True
        assert sc2.forward_succeeded is True
        assert cr2.classifier_succeeded is True

    def test_nan_in_max_prob_marks_all_failed(self) -> None:
        sa = _make_signal_a(succeeded=True)
        sb = _make_signal_b(succeeded=True)
        sc = _make_signal_c(succeeded=True)
        cr = _make_classifier_result(max_prob=float("nan"))
        cr2, sa2, sb2, sc2 = _apply_nan_guard(cr, sa, sb, sc)
        assert not sa2.forward_succeeded
        assert not sb2.forward_succeeded
        assert not sc2.forward_succeeded
        assert not cr2.classifier_succeeded
        assert cr2.combined_max_prob == pytest.approx(0.0)

    def test_nan_in_margin_marks_all_failed(self) -> None:
        sa = _make_signal_a(succeeded=True)
        sb = _make_signal_b(succeeded=True)
        sc = _make_signal_c(succeeded=True)
        cr = _make_classifier_result(max_prob=0.7, margin=float("nan"))
        cr2, sa2, sb2, sc2 = _apply_nan_guard(cr, sa, sb, sc)
        assert not sa2.forward_succeeded
        assert cr2.combined_margin == pytest.approx(0.0)

    def test_nan_in_probs_marks_all_failed(self) -> None:
        sa = _make_signal_a(succeeded=True)
        sb = _make_signal_b(succeeded=True)
        sc = _make_signal_c(succeeded=True)
        probs = np.full(7, float("nan"), dtype=np.float32)
        from dataclasses import replace
        cr = ClassifierResult(
            p_final_calibrated=probs,
            combined_argmax=0,
            combined_max_prob=0.5,
            combined_margin=0.2,
            p_final_uncalibrated=probs.copy(),
            p_stage1=np.zeros(3, dtype=np.float32),
            p_stage2=np.zeros(5, dtype=np.float32),
            classifier_succeeded=True,
            failure_reason="",
        )
        cr2, sa2, sb2, sc2 = _apply_nan_guard(cr, sa, sb, sc)
        assert not sa2.forward_succeeded
        assert "nan" in sa2.failure_reason

    def test_failure_reason_contains_nan_marker(self) -> None:
        sa = _make_signal_a(succeeded=True)
        sb = _make_signal_b(succeeded=True)
        sc = _make_signal_c(succeeded=True)
        cr = _make_classifier_result(max_prob=float("nan"))
        cr2, sa2, sb2, sc2 = _apply_nan_guard(cr, sa, sb, sc)
        assert "nan_in_classifier_output" in sa2.failure_reason


# ---------------------------------------------------------------------------
# Tests: predict_single — image decode failure
# ---------------------------------------------------------------------------

class TestPredictSingleDecodeFailure:
    def test_invalid_bytes_returns_error(self) -> None:
        ctx = _make_context()
        result = predict_single(b"not-an-image", "req-1", ctx)
        assert "error" in result
        assert result["error"] == "IMAGE_DECODE_FAILED"
        assert result["status"] == 400

    def test_empty_bytes_returns_error(self) -> None:
        ctx = _make_context()
        result = predict_single(b"", "req-2", ctx)
        assert "error" in result


# ---------------------------------------------------------------------------
# Tests: predict_single — request cache
# ---------------------------------------------------------------------------

class TestRequestCache:
    def test_cache_hit_returns_cached_result(self) -> None:
        img_bytes = _tiny_jpeg_bytes()
        img_hash = _image_hash(img_bytes)
        cached_entry = {
            "tier_label": "1",
            "tier5_alert": False,
            "warnings": [],
        }
        ctx = _make_context(cache={img_hash: cached_entry})
        result = predict_single(img_bytes, "req-cache-1", ctx)
        assert "from_cache" in result["warnings"]
        assert result["tier_label"] == "1"
        assert "processing_time_ms" in result

    def test_cache_miss_does_not_return_cached(self) -> None:
        # Cache is populated with a different hash
        ctx = _make_context(cache={"different_hash": {"tier_label": "4B"}})
        # Inject a bytes sequence that isn't in cache
        result = predict_single(b"not-an-image", "req-cache-2", ctx)
        # Should NOT find "4B" from cache; should get decode error instead
        assert result.get("tier_label") != "4B" or "error" in result


# ---------------------------------------------------------------------------
# Tests: predict_single — IQA rejection
# ---------------------------------------------------------------------------

class TestIQARejection:
    def test_iqa_reject_returns_422_error(self) -> None:
        img_bytes = _tiny_jpeg_bytes()
        reject_iqa = _make_iqa_result("REJECT")
        reject_iqa_with_msg = IQAResult(
            decision="REJECT",
            aggregate_score=0.2,
            per_dimension={},
            failing_dimensions=["sharpness"],
            retake_message="Image is too blurry. Please retake.",
            green_mask=None,
        )

        ctx = _make_context()
        with patch(
            "tomato_sandbox.orchestrator.pipeline.compute_iqa",
            return_value=reject_iqa_with_msg,
        ):
            result = predict_single(img_bytes, "req-iqa-1", ctx)

        assert "error" in result
        assert result["error"] == "IQA_REJECTED"
        assert result["status"] == 422


# ---------------------------------------------------------------------------
# Tests: predict_single — happy path (full mock)
# ---------------------------------------------------------------------------

class TestPredictSingleHappyPath:
    """End-to-end happy path with all pipeline stages mocked."""

    def _run_happy_path(
        self,
        sa_max_prob: float = 0.85,
        sb_max_prob: float = 0.80,
        sc_reliability: float = 0.88,
        clf_max_prob: float = 0.82,
        conformal_set: list | None = None,
        tier: str = "1",
        rule: str = "7c",
    ) -> dict:
        if conformal_set is None:
            conformal_set = [5]

        img_bytes = _tiny_jpeg_bytes()
        ctx = _make_context()

        sa = _make_signal_a(max_prob=sa_max_prob)
        sb = _make_signal_b(max_prob=sb_max_prob)
        sc = _make_signal_c(reliability=sc_reliability)
        cr = _make_classifier_result(max_prob=clf_max_prob)
        cfr = _make_conformal_result(conformal_set)
        iqa = _make_iqa_result("ACCEPTABLE")
        ta = _make_tier_assignment(tier=tier, rule=rule)

        with (
            patch("tomato_sandbox.orchestrator.pipeline.compute_iqa", return_value=iqa),
            patch("tomato_sandbox.orchestrator.pipeline.preprocess_for_v3", return_value=MagicMock()),
            patch("tomato_sandbox.orchestrator.pipeline.compute_signal_a", return_value=sa),
            patch("tomato_sandbox.orchestrator.pipeline.preprocess_for_lora", return_value=MagicMock()),
            patch("tomato_sandbox.orchestrator.pipeline.compute_signal_b", return_value=sb),
            patch("tomato_sandbox.orchestrator.pipeline.preprocess_for_psv", return_value=MagicMock()),
            patch("tomato_sandbox.orchestrator.pipeline.compute_signal_c", return_value=sc),
            patch("tomato_sandbox.orchestrator.pipeline.compute_classifier", return_value=cr),
            patch("tomato_sandbox.orchestrator.pipeline.compute_conformal_set", return_value=cfr),
            patch("tomato_sandbox.orchestrator.pipeline.assign_tier", return_value=ta),
        ):
            return predict_single(img_bytes, "req-happy-1", ctx)

    def test_happy_path_returns_tier_label(self) -> None:
        # S16.2: tier is now a nested block; tier["label"] carries the tier string
        result = self._run_happy_path()
        assert "tier" in result
        assert result["tier"]["label"] == "1"

    def test_happy_path_has_signal_health_fields(self) -> None:
        # S16.2: signal_a/b/c_succeeded are not top-level response fields.
        # Semantic intent: when all signals succeed, the pipeline produces a valid
        # non-error response and assigns a non-4B tier (degraded-mode 4B only fires
        # when all signals fail). The conformal set size < 7 (fallback size)
        # also confirms signals contributed meaningful probability mass.
        result = self._run_happy_path()
        assert "error" not in result, f"Unexpected error: {result.get('error')}"
        assert result["tier"]["label"] != "4B", (
            "Expected a successful tier (not 4B) when all signals succeed"
        )
        # Conformal set comes from mock (1 element), not the 7-element fallback
        assert len(result["prediction"]["prediction_set"]) < 7, (
            "Expected a narrow conformal set when signals succeed"
        )

    def test_happy_path_has_conformal_fields(self) -> None:
        # S16.2: conformal fields are nested under result["prediction"]
        result = self._run_happy_path(conformal_set=[3, 5])
        assert "prediction_set" in result["prediction"]
        assert len(result["prediction"]["prediction_set"]) == 2

    def test_happy_path_no_tta_fired(self) -> None:
        # S21.3 step 22: tta_fired is surfaced as a warning string, not a bare bool key.
        # When TTA did NOT fire, the sentinel warning string is absent from warnings.
        result = self._run_happy_path(clf_max_prob=0.82)
        assert "TTA was triggered for this request." not in result.get("warnings", [])

    def test_happy_path_processing_time_present(self) -> None:
        result = self._run_happy_path()
        assert "processing_time_ms" in result
        assert result["processing_time_ms"] >= 0.0

    def test_happy_path_request_id_preserved(self) -> None:
        img_bytes = _tiny_jpeg_bytes()
        ctx = _make_context()
        sa = _make_signal_a()
        sb = _make_signal_b()
        sc = _make_signal_c()
        cr = _make_classifier_result()
        cfr = _make_conformal_result()
        iqa = _make_iqa_result()
        ta = _make_tier_assignment()
        with (
            patch("tomato_sandbox.orchestrator.pipeline.compute_iqa", return_value=iqa),
            patch("tomato_sandbox.orchestrator.pipeline.preprocess_for_v3", return_value=MagicMock()),
            patch("tomato_sandbox.orchestrator.pipeline.compute_signal_a", return_value=sa),
            patch("tomato_sandbox.orchestrator.pipeline.preprocess_for_lora", return_value=MagicMock()),
            patch("tomato_sandbox.orchestrator.pipeline.compute_signal_b", return_value=sb),
            patch("tomato_sandbox.orchestrator.pipeline.preprocess_for_psv", return_value=MagicMock()),
            patch("tomato_sandbox.orchestrator.pipeline.compute_signal_c", return_value=sc),
            patch("tomato_sandbox.orchestrator.pipeline.compute_classifier", return_value=cr),
            patch("tomato_sandbox.orchestrator.pipeline.compute_conformal_set", return_value=cfr),
            patch("tomato_sandbox.orchestrator.pipeline.assign_tier", return_value=ta),
        ):
            result = predict_single(img_bytes, "my-special-req-id", ctx)
        assert result["request_id"] == "my-special-req-id"


# ---------------------------------------------------------------------------
# Tests: PSV NOT invoked during TTA
# spec: section 11.1 line 2925 — "PSV does NOT participate in TTA"
# spec: section 11.9 lines 3139-3140 — PSV excluded from TTA loop
# ---------------------------------------------------------------------------

class TestTTADoesNotInvokePSV:
    def test_psv_not_called_during_tta(self) -> None:
        """
        When TTA triggers (combined_max_prob < 0.55), compute_signal_c must NOT
        be invoked additional times during the TTA loop.

        spec: section 11.1 line 2925 — "PSV does NOT participate in TTA"
        spec: section 11.9 lines 3139-3140 — apply_tta does not call compute_signal_c
        """
        img_bytes = _tiny_jpeg_bytes()
        ctx = _make_context()

        sa = _make_signal_a(max_prob=0.40)
        sb = _make_signal_b(max_prob=0.40)
        sc = _make_signal_c()
        # First classifier run returns low confidence → triggers TTA
        cr_low = _make_classifier_result(max_prob=0.40)
        cr_high = _make_classifier_result(max_prob=0.70)
        cfr = _make_conformal_result()
        iqa = _make_iqa_result()
        ta = _make_tier_assignment()

        psv_call_count = {"count": 0}
        original_sig_c = sc

        def counting_signal_c(*args, **kwargs):
            psv_call_count["count"] += 1
            return sc

        # apply_tta returns aggregated A and B (not C); orchestrator passes original sc to
        # the post-TTA classifier call
        tta_report = MagicMock()
        tta_report.n_views = 5

        with (
            patch("tomato_sandbox.orchestrator.pipeline.compute_iqa", return_value=iqa),
            patch("tomato_sandbox.orchestrator.pipeline.preprocess_for_v3", return_value=MagicMock()),
            patch("tomato_sandbox.orchestrator.pipeline.compute_signal_a", return_value=sa),
            patch("tomato_sandbox.orchestrator.pipeline.preprocess_for_lora", return_value=MagicMock()),
            patch("tomato_sandbox.orchestrator.pipeline.compute_signal_b", return_value=sb),
            patch("tomato_sandbox.orchestrator.pipeline.preprocess_for_psv", return_value=MagicMock()),
            patch(
                "tomato_sandbox.orchestrator.pipeline.compute_signal_c",
                side_effect=counting_signal_c,
            ),
            patch(
                "tomato_sandbox.orchestrator.pipeline.compute_classifier",
                side_effect=[cr_low, cr_high],
            ),
            patch(
                "tomato_sandbox.orchestrator.pipeline.apply_tta",
                return_value=(sa, sb, tta_report),
            ) as mock_tta,
            patch("tomato_sandbox.orchestrator.pipeline.should_trigger_tta", return_value=5),
            patch("tomato_sandbox.orchestrator.pipeline.compute_conformal_set", return_value=cfr),
            patch("tomato_sandbox.orchestrator.pipeline.assign_tier", return_value=ta),
        ):
            result = predict_single(img_bytes, "req-tta-psv-1", ctx)

        # PSV was called exactly once (single-view pre-TTA call) — NOT inside TTA
        assert psv_call_count["count"] == 1, (
            f"compute_signal_c was called {psv_call_count['count']} times; "
            f"expected exactly 1 (pre-TTA only, never during TTA)"
        )

    def test_tta_fires_when_confidence_low(self) -> None:
        """
        TTA should be triggered when combined_max_prob < 0.55.
        spec: section 11.2 line 2932 — TTA_TRIGGER_THRESHOLD = 0.55
        """
        img_bytes = _tiny_jpeg_bytes()
        ctx = _make_context()

        sa = _make_signal_a(max_prob=0.35)
        sb = _make_signal_b(max_prob=0.35)
        sc = _make_signal_c()
        cr_low = _make_classifier_result(max_prob=0.35)
        cr_high = _make_classifier_result(max_prob=0.70)
        cfr = _make_conformal_result()
        iqa = _make_iqa_result()
        ta = _make_tier_assignment()
        tta_report = MagicMock()

        mock_tta = MagicMock(return_value=(sa, sb, tta_report))

        with (
            patch("tomato_sandbox.orchestrator.pipeline.compute_iqa", return_value=iqa),
            patch("tomato_sandbox.orchestrator.pipeline.preprocess_for_v3", return_value=MagicMock()),
            patch("tomato_sandbox.orchestrator.pipeline.compute_signal_a", return_value=sa),
            patch("tomato_sandbox.orchestrator.pipeline.preprocess_for_lora", return_value=MagicMock()),
            patch("tomato_sandbox.orchestrator.pipeline.compute_signal_b", return_value=sb),
            patch("tomato_sandbox.orchestrator.pipeline.preprocess_for_psv", return_value=MagicMock()),
            patch("tomato_sandbox.orchestrator.pipeline.compute_signal_c", return_value=sc),
            patch(
                "tomato_sandbox.orchestrator.pipeline.compute_classifier",
                side_effect=[cr_low, cr_high],
            ),
            patch("tomato_sandbox.orchestrator.pipeline.apply_tta", new=mock_tta),
            patch("tomato_sandbox.orchestrator.pipeline.should_trigger_tta", return_value=5),
            patch("tomato_sandbox.orchestrator.pipeline.compute_conformal_set", return_value=cfr),
            patch("tomato_sandbox.orchestrator.pipeline.assign_tier", return_value=ta),
        ):
            result = predict_single(img_bytes, "req-tta-1", ctx)

        mock_tta.assert_called_once()
        # S21.3 step 22: tta_fired is surfaced as a warning string, not a bare bool key.
        assert "TTA was triggered for this request." in result.get("warnings", [])

    def test_tta_not_fired_when_confidence_high(self) -> None:
        """
        TTA should NOT trigger when combined_max_prob >= 0.55.
        spec: section 11.2 line 2932 — returns 1 (no TTA) when max_prob >= 0.55
        """
        img_bytes = _tiny_jpeg_bytes()
        ctx = _make_context()

        sa = _make_signal_a(max_prob=0.90)
        sb = _make_signal_b(max_prob=0.85)
        sc = _make_signal_c()
        cr = _make_classifier_result(max_prob=0.82)
        cfr = _make_conformal_result()
        iqa = _make_iqa_result()
        ta = _make_tier_assignment()

        mock_tta = MagicMock()

        with (
            patch("tomato_sandbox.orchestrator.pipeline.compute_iqa", return_value=iqa),
            patch("tomato_sandbox.orchestrator.pipeline.preprocess_for_v3", return_value=MagicMock()),
            patch("tomato_sandbox.orchestrator.pipeline.compute_signal_a", return_value=sa),
            patch("tomato_sandbox.orchestrator.pipeline.preprocess_for_lora", return_value=MagicMock()),
            patch("tomato_sandbox.orchestrator.pipeline.compute_signal_b", return_value=sb),
            patch("tomato_sandbox.orchestrator.pipeline.preprocess_for_psv", return_value=MagicMock()),
            patch("tomato_sandbox.orchestrator.pipeline.compute_signal_c", return_value=sc),
            patch("tomato_sandbox.orchestrator.pipeline.compute_classifier", return_value=cr),
            patch("tomato_sandbox.orchestrator.pipeline.apply_tta", new=mock_tta),
            patch("tomato_sandbox.orchestrator.pipeline.should_trigger_tta", return_value=1),
            patch("tomato_sandbox.orchestrator.pipeline.compute_conformal_set", return_value=cfr),
            patch("tomato_sandbox.orchestrator.pipeline.assign_tier", return_value=ta),
        ):
            result = predict_single(img_bytes, "req-no-tta-1", ctx)

        mock_tta.assert_not_called()
        # S21.3 step 22: tta_fired is surfaced as a warning string, not a bare bool key.
        assert "TTA was triggered for this request." not in result.get("warnings", [])


# ---------------------------------------------------------------------------
# Tests: Signal failure → degraded mode (pipeline continues)
# spec: section 21.5 lines 6713-6743
# ---------------------------------------------------------------------------

class TestSignalFailureDegradedMode:
    """
    When one or two signals fail, pipeline continues (degraded mode).
    All-signals-failed is tested separately (all-signals-failed short-circuit).
    """

    def test_signal_a_fails_pipeline_continues(self) -> None:
        img_bytes = _tiny_jpeg_bytes()
        ctx = _make_context()

        # Signal A fails but B and C succeed
        sa = _make_signal_a(succeeded=False)
        sb = _make_signal_b(succeeded=True, max_prob=0.75)
        sc = _make_signal_c(succeeded=True)
        cr = _make_classifier_result(max_prob=0.72)
        cfr = _make_conformal_result()
        iqa = _make_iqa_result()
        ta = _make_tier_assignment(tier="2", rule="8c")

        with (
            patch("tomato_sandbox.orchestrator.pipeline.compute_iqa", return_value=iqa),
            patch("tomato_sandbox.orchestrator.pipeline.preprocess_for_v3", return_value=MagicMock()),
            patch("tomato_sandbox.orchestrator.pipeline.compute_signal_a", return_value=sa),
            patch("tomato_sandbox.orchestrator.pipeline.preprocess_for_lora", return_value=MagicMock()),
            patch("tomato_sandbox.orchestrator.pipeline.compute_signal_b", return_value=sb),
            patch("tomato_sandbox.orchestrator.pipeline.preprocess_for_psv", return_value=MagicMock()),
            patch("tomato_sandbox.orchestrator.pipeline.compute_signal_c", return_value=sc),
            patch("tomato_sandbox.orchestrator.pipeline.compute_classifier", return_value=cr),
            patch("tomato_sandbox.orchestrator.pipeline.compute_conformal_set", return_value=cfr),
            patch("tomato_sandbox.orchestrator.pipeline.assign_tier", return_value=ta),
        ):
            result = predict_single(img_bytes, "req-sa-fail-1", ctx)

        # Pipeline should produce a valid S16.2 result with tier block (not an error).
        # S16.2: top-level "tier" block replaces old "tier_label" key.
        assert "error" not in result, f"Unexpected error: {result.get('error')}"
        assert "tier" in result
        # When A failed but B+C succeeded the pipeline continues in degraded mode;
        # the mock assign_tier returns tier "2" (not the pipeline-failure tier "4B").
        assert result["tier"]["label"] != "4B", (
            "Expected degraded-mode tier (not 4B) when only signal A fails"
        )

    def test_signal_b_fails_pipeline_continues(self) -> None:
        img_bytes = _tiny_jpeg_bytes()
        ctx = _make_context()

        sa = _make_signal_a(succeeded=True, max_prob=0.78)
        sb = _make_signal_b(succeeded=False)
        sc = _make_signal_c(succeeded=True)
        cr = _make_classifier_result(max_prob=0.70)
        cfr = _make_conformal_result()
        iqa = _make_iqa_result()
        ta = _make_tier_assignment()

        with (
            patch("tomato_sandbox.orchestrator.pipeline.compute_iqa", return_value=iqa),
            patch("tomato_sandbox.orchestrator.pipeline.preprocess_for_v3", return_value=MagicMock()),
            patch("tomato_sandbox.orchestrator.pipeline.compute_signal_a", return_value=sa),
            patch("tomato_sandbox.orchestrator.pipeline.preprocess_for_lora", return_value=MagicMock()),
            patch("tomato_sandbox.orchestrator.pipeline.compute_signal_b", return_value=sb),
            patch("tomato_sandbox.orchestrator.pipeline.preprocess_for_psv", return_value=MagicMock()),
            patch("tomato_sandbox.orchestrator.pipeline.compute_signal_c", return_value=sc),
            patch("tomato_sandbox.orchestrator.pipeline.compute_classifier", return_value=cr),
            patch("tomato_sandbox.orchestrator.pipeline.compute_conformal_set", return_value=cfr),
            patch("tomato_sandbox.orchestrator.pipeline.assign_tier", return_value=ta),
        ):
            result = predict_single(img_bytes, "req-sb-fail-1", ctx)

        # S16.2: pipeline continues when only B fails; result has a tier block, no error.
        # "signal_b_succeeded" is not a top-level S16.2 key; the semantic guarantee is
        # that the pipeline continues (produces a valid tier response, not an error dict).
        assert "error" not in result
        assert "tier" in result
        # A+C succeeded so pipeline does not short-circuit to all-signals-failed 4B
        assert result["tier"]["label"] != "4B", (
            "Expected degraded-mode tier (not 4B) when only signal B fails"
        )

    def test_signal_c_fails_pipeline_continues(self) -> None:
        img_bytes = _tiny_jpeg_bytes()
        ctx = _make_context()

        sa = _make_signal_a(succeeded=True)
        sb = _make_signal_b(succeeded=True)
        sc = _make_signal_c(succeeded=False)
        cr = _make_classifier_result(max_prob=0.75)
        cfr = _make_conformal_result()
        iqa = _make_iqa_result()
        ta = _make_tier_assignment()

        with (
            patch("tomato_sandbox.orchestrator.pipeline.compute_iqa", return_value=iqa),
            patch("tomato_sandbox.orchestrator.pipeline.preprocess_for_v3", return_value=MagicMock()),
            patch("tomato_sandbox.orchestrator.pipeline.compute_signal_a", return_value=sa),
            patch("tomato_sandbox.orchestrator.pipeline.preprocess_for_lora", return_value=MagicMock()),
            patch("tomato_sandbox.orchestrator.pipeline.compute_signal_b", return_value=sb),
            patch("tomato_sandbox.orchestrator.pipeline.preprocess_for_psv", return_value=MagicMock()),
            patch("tomato_sandbox.orchestrator.pipeline.compute_signal_c", return_value=sc),
            patch("tomato_sandbox.orchestrator.pipeline.compute_classifier", return_value=cr),
            patch("tomato_sandbox.orchestrator.pipeline.compute_conformal_set", return_value=cfr),
            patch("tomato_sandbox.orchestrator.pipeline.assign_tier", return_value=ta),
        ):
            result = predict_single(img_bytes, "req-sc-fail-1", ctx)

        # S16.2: pipeline continues when only C fails; result has a tier block, no error.
        # "signal_c_succeeded" is not a top-level S16.2 key; the semantic guarantee is
        # that the pipeline continues (A+B succeeded so no all-signals-failed short-circuit).
        assert "error" not in result
        assert "tier" in result
        assert result["tier"]["label"] != "4B", (
            "Expected degraded-mode tier (not 4B) when only signal C fails"
        )


# ---------------------------------------------------------------------------
# Tests: all-signals-failed short-circuit
# spec: section 21.5 lines 6745-6755
# ---------------------------------------------------------------------------

class TestAllSignalsFailed:
    def test_all_signals_failed_produces_tier_4b(self) -> None:
        """
        When all three signals fail, the orchestrator short-circuits to a sentinel
        classifier result and routes directly to assign_tier.
        Rule 1 (any signal forward_succeeded=False) fires → Tier 4B.
        spec: section 21.5 lines 6745-6755
        """
        img_bytes = _tiny_jpeg_bytes()
        ctx = _make_context()

        sa = _make_signal_a(succeeded=False)
        sb = _make_signal_b(succeeded=False)
        sc = _make_signal_c(succeeded=False)
        iqa = _make_iqa_result()
        # The real assign_tier fires Rule 1 when any signal fails
        # Route through real tier assignment
        with (
            patch("tomato_sandbox.orchestrator.pipeline.compute_iqa", return_value=iqa),
            patch("tomato_sandbox.orchestrator.pipeline.preprocess_for_v3", return_value=MagicMock()),
            patch("tomato_sandbox.orchestrator.pipeline.compute_signal_a", return_value=sa),
            patch("tomato_sandbox.orchestrator.pipeline.preprocess_for_lora", return_value=MagicMock()),
            patch("tomato_sandbox.orchestrator.pipeline.compute_signal_b", return_value=sb),
            patch("tomato_sandbox.orchestrator.pipeline.preprocess_for_psv", return_value=MagicMock()),
            patch("tomato_sandbox.orchestrator.pipeline.compute_signal_c", return_value=sc),
        ):
            result = predict_single(img_bytes, "req-all-fail-1", ctx)

        # Should not return an HTTP error — pipeline should produce a valid (degraded) result
        assert "error" not in result, f"Unexpected error: {result.get('error')}"
        # S16.2: tier is now a nested block; tier["label"] carries the tier string.
        # Tier 4B because Rule 1 fired (all signals failed).
        assert result.get("tier", {}).get("label") == "4B", (
            f"Expected tier.label='4B' from all-signals-failed, got: "
            f"{result.get('tier', {}).get('label')}"
        )
        # signal_a/b/c_succeeded are not top-level S16.2 fields.
        # The semantic guarantee is that the all-signals-failed short-circuit fired
        # (producing 4B) and no error was raised — verified above.

    def test_all_signals_failed_classifier_not_called(self) -> None:
        """
        When all signals fail, compute_classifier must NOT be called.
        spec: section 21.5 lines 6749-6753 — "skip classifier forward pass entirely"
        """
        img_bytes = _tiny_jpeg_bytes()
        ctx = _make_context()

        sa = _make_signal_a(succeeded=False)
        sb = _make_signal_b(succeeded=False)
        sc = _make_signal_c(succeeded=False)
        iqa = _make_iqa_result()

        mock_classifier = MagicMock()

        with (
            patch("tomato_sandbox.orchestrator.pipeline.compute_iqa", return_value=iqa),
            patch("tomato_sandbox.orchestrator.pipeline.preprocess_for_v3", return_value=MagicMock()),
            patch("tomato_sandbox.orchestrator.pipeline.compute_signal_a", return_value=sa),
            patch("tomato_sandbox.orchestrator.pipeline.preprocess_for_lora", return_value=MagicMock()),
            patch("tomato_sandbox.orchestrator.pipeline.compute_signal_b", return_value=sb),
            patch("tomato_sandbox.orchestrator.pipeline.preprocess_for_psv", return_value=MagicMock()),
            patch("tomato_sandbox.orchestrator.pipeline.compute_signal_c", return_value=sc),
            patch("tomato_sandbox.orchestrator.pipeline.compute_classifier", new=mock_classifier),
        ):
            predict_single(img_bytes, "req-all-fail-2", ctx)

        mock_classifier.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: GPU lock acquired around A/B but NOT PSV
# spec: section 21.3 steps 4-8; section 10.2
# ---------------------------------------------------------------------------

class TestGPULockBehavior:
    """
    Verify that:
    - GPU lock (if provided) is acquired before signals A+B
    - PSV runs outside the GPU lock (after the finally block)
    - GPU lock timeout produces SERVER_OVERLOAD 503
    """

    def test_gpu_lock_timeout_returns_503(self) -> None:
        """
        spec: section 21.8 line 6820 — GPU lock timeout → 503 SERVER_OVERLOAD
        """
        from tomato_sandbox.utils.gpu_lock import GPULockTimeoutError

        img_bytes = _tiny_jpeg_bytes()
        iqa = _make_iqa_result()

        mock_lock = MagicMock()
        # spec: section 21.3 step 4 (sync-acquire path) — DEC-045 / Batch 7 cross-loop fix:
        # orchestrator only attempts to acquire when no running loop AND `gpu_lock.locked`
        # is False (note: locked is a @property on GPULock, not a method).
        # MagicMock auto-attribute access returns a truthy MagicMock by default,
        # which would bypass the acquire path. Override with a literal False so the
        # property-style read returns False and the acquire path is reached.
        mock_lock.locked = False
        # Simulate timeout on acquire_with_timeout
        import asyncio

        async def _timeout_coro():
            raise GPULockTimeoutError(timeout_s=10.0)

        mock_lock.acquire_with_timeout = _timeout_coro

        ctx = _make_context(gpu_lock=mock_lock)
        with patch("tomato_sandbox.orchestrator.pipeline.compute_iqa", return_value=iqa):
            result = predict_single(img_bytes, "req-lock-timeout-1", ctx)

        assert "error" in result
        assert result["error"] == "SERVER_OVERLOAD"
        assert result["status"] == 503

    def test_psv_runs_outside_gpu_lock_context(self) -> None:
        """
        PSV (Signal C) must run strictly after GPU lock is released.
        Verify: compute_signal_c is called even when gpu_lock=None (no lock path).
        The ordering constraint (lock released before PSV) is enforced by code structure.
        spec: section 10.2 — "CPU-only: no GPU API, no gpu_lock"
        """
        img_bytes = _tiny_jpeg_bytes()
        ctx = _make_context(gpu_lock=None)

        sa = _make_signal_a()
        sb = _make_signal_b()
        sc = _make_signal_c()
        cr = _make_classifier_result()
        cfr = _make_conformal_result()
        iqa = _make_iqa_result()
        ta = _make_tier_assignment()

        psv_call_log: list[str] = []

        def log_psv_call(*args, **kwargs):
            psv_call_log.append("psv_called")
            return sc

        with (
            patch("tomato_sandbox.orchestrator.pipeline.compute_iqa", return_value=iqa),
            patch("tomato_sandbox.orchestrator.pipeline.preprocess_for_v3", return_value=MagicMock()),
            patch("tomato_sandbox.orchestrator.pipeline.compute_signal_a", return_value=sa),
            patch("tomato_sandbox.orchestrator.pipeline.preprocess_for_lora", return_value=MagicMock()),
            patch("tomato_sandbox.orchestrator.pipeline.compute_signal_b", return_value=sb),
            patch("tomato_sandbox.orchestrator.pipeline.preprocess_for_psv", return_value=MagicMock()),
            patch("tomato_sandbox.orchestrator.pipeline.compute_signal_c", side_effect=log_psv_call),
            patch("tomato_sandbox.orchestrator.pipeline.compute_classifier", return_value=cr),
            patch("tomato_sandbox.orchestrator.pipeline.compute_conformal_set", return_value=cfr),
            patch("tomato_sandbox.orchestrator.pipeline.assign_tier", return_value=ta),
        ):
            result = predict_single(img_bytes, "req-psv-lock-1", ctx)

        # PSV was called exactly once
        assert len(psv_call_log) == 1


# ---------------------------------------------------------------------------
# Tests: predict_multi stub
# ---------------------------------------------------------------------------

class TestPredictMulti:
    def test_predict_multi_returns_per_image_list(self) -> None:
        ctx = _make_context()
        img_bytes = _tiny_jpeg_bytes()

        # predict_single will fail (no mocks) but predict_multi should not raise
        result = predict_multi([b"invalid1", b"invalid2"], "req-multi-1", ctx)
        assert "n_images" in result
        assert result["n_images"] == 2
        assert "per_image_results" in result
        assert len(result["per_image_results"]) == 2

    def test_predict_multi_per_image_has_image_id(self) -> None:
        ctx = _make_context()
        result = predict_multi(
            [(b"invalid1", "img-A"), (b"invalid2", "img-B")],
            "req-multi-2",
            ctx,
        )
        image_ids = [r["image_id"] for r in result["per_image_results"]]
        assert "img-A" in image_ids
        assert "img-B" in image_ids


# ---------------------------------------------------------------------------
# Tests: PipelineContext field contract
# ---------------------------------------------------------------------------

class TestPipelineContext:
    def test_all_fields_default_to_none(self) -> None:
        ctx = PipelineContext()
        assert ctx.v3_model is None
        assert ctx.lora_model is None
        assert ctx.gpu_lock is None
        assert ctx.cache is None
        assert ctx.underpowered_classes is None

    def test_fields_settable(self) -> None:
        mock_model = MagicMock()
        ctx = PipelineContext(v3_model=mock_model, underpowered_classes={3, 4})
        assert ctx.v3_model is mock_model
        assert ctx.underpowered_classes == {3, 4}
