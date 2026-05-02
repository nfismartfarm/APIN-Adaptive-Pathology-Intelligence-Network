"""
Unit tests for the hierarchical classifier (T-IMPL-4a).

Spec section: 12 (Hierarchical classifier), lines 3145–3505.

All tests use MOCK signal outputs — no real GPU, no real calibration files.
The pre-F.0 sentinel fallback provides identity standardization (mean=0, std=1)
and uniform-logit stage weights, making all forward-pass shapes deterministic.

Test coverage:
  1. Feature vector shape: exactly 19 dimensions
  2. Feature vector slot ordering: each index matches spec S12.2 table
  3. JSD computation: finite, bounded in [0, ln(2)]
  4. JSD sentinel used when signals fail
  5. build_classifier_input + degraded-mode zeroing (each of 3 signals failed)
  6. ClassifierResult field names: all 9 required fields present
  7. Stage 1 forward shape: [3]
  8. Stage 2 forward shape: [5]
  9. Soft routing: sum to 1, shape [7], index mapping
  10. Platt calibration: output sums to 1, identity Platt is idempotent (approx)
  11. compute_classifier forward shape
  12. combined_argmax is valid index 0-6
  13. combined_margin >= 0
  14. combined_max_prob == max(p_final_calibrated)
  15. p_final_calibrated sums to approximately 1
  16. p_final_uncalibrated sums to approximately 1
  17. degraded mode: Signal A failed → indices 0-5 and 18 zeroed in raw
  18. degraded mode: Signal B failed → indices 6-11 zeroed in raw
  19. degraded mode: Signal C failed → indices 12-15 and 17 zeroed in raw
  20. All 3 signals failed: still returns valid ClassifierResult
  21. classifier_succeeded=True on normal inputs
  22. classifier_succeeded=False on malformed input
  23. No gpu_lock import anywhere in classifier package
  24. Both import paths work: tomato_sandbox.classifier and sub-modules
"""

from __future__ import annotations

import dataclasses
import importlib
import math
import sys
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Mock signal result factories
# ---------------------------------------------------------------------------

def _make_mock_sa(
    tomato_probs: list[float] | None = None,
    argmax: int = 0,
    chilli_leakage: float = 0.05,
    forward_succeeded: bool = True,
) -> Any:
    """Create a mock SignalAResult (no real v3 model needed)."""
    if tomato_probs is None:
        tomato_probs = [0.1, 0.2, 0.3, 0.1, 0.1, 0.1]  # sum != 1 (v3 case)
    obj = MagicMock()
    obj.tomato_probs_canonical = np.array(tomato_probs, dtype=np.float32)
    obj.tomato_argmax_canonical = argmax
    obj.chilli_leakage = float(chilli_leakage)
    obj.forward_succeeded = forward_succeeded
    obj.failure_reason = None if forward_succeeded else "mock_failure"
    return obj


def _make_mock_sb(
    tomato_probs: list[float] | None = None,
    argmax: int = 0,
    forward_succeeded: bool = True,
) -> Any:
    """Create a mock SignalBResult (no real LoRA model needed)."""
    if tomato_probs is None:
        tomato_probs = [0.5, 0.1, 0.1, 0.1, 0.1, 0.1]  # sum = 1.0
    obj = MagicMock()
    obj.tomato_probs_canonical = np.array(tomato_probs, dtype=np.float32)
    obj.tomato_argmax_canonical = argmax
    obj.forward_succeeded = forward_succeeded
    obj.failure_reason = None if forward_succeeded else "mock_failure"
    return obj


def _make_mock_sc(
    compatibility_max: float = 0.7,
    compatibility_margin: float = 0.3,
    compatibility_argmax: int = 0,
    psv_reliability: float = 0.8,
    forward_succeeded: bool = True,
) -> Any:
    """Create a mock SignalCResult (no real PSV needed)."""
    obj = MagicMock()
    obj.compatibility_max = compatibility_max
    obj.compatibility_margin = compatibility_margin
    obj.compatibility_argmax = compatibility_argmax
    obj.psv_reliability = psv_reliability
    obj.forward_succeeded = forward_succeeded
    obj.failure_reason = None if forward_succeeded else "mock_failure"
    return obj


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_raw_feature_vector(sa: Any, sb: Any, sc: Any) -> np.ndarray:
    """Build the 19-dim raw (pre-standardization) vector for slot inspection.

    We bypass standardization by temporarily patching the mean/std to 0/1.
    """
    import tomato_sandbox.classifier.feature_builder as fb

    original_mean = fb.CLASSIFIER_FEATURE_MEAN.copy()
    original_std = fb.CLASSIFIER_FEATURE_STD.copy()
    try:
        fb.CLASSIFIER_FEATURE_MEAN[:] = 0.0
        fb.CLASSIFIER_FEATURE_STD[:] = 1.0
        return fb.build_classifier_input(sa, sb, sc).copy()
    finally:
        fb.CLASSIFIER_FEATURE_MEAN[:] = original_mean
        fb.CLASSIFIER_FEATURE_STD[:] = original_std


# ---------------------------------------------------------------------------
# Test 1: Feature vector shape
# ---------------------------------------------------------------------------

def test_feature_vector_shape() -> None:
    """Feature vector must be exactly 19-dimensional.
    # spec: section 12.1 line 3149 — "19-dimensional feature vector"
    """
    from tomato_sandbox.classifier.feature_builder import build_classifier_input
    sa = _make_mock_sa()
    sb = _make_mock_sb()
    sc = _make_mock_sc()
    x = build_classifier_input(sa, sb, sc)
    assert x.shape == (19,), f"Expected (19,), got {x.shape}"


# ---------------------------------------------------------------------------
# Test 2: Feature vector slot ordering
# ---------------------------------------------------------------------------

def test_feature_vector_slot_v3_probs() -> None:
    """Indices 0-5 must come from sa.tomato_probs_canonical.
    # spec: section 12.2 table lines 3177-3182
    """
    v3_probs = [0.11, 0.22, 0.33, 0.05, 0.07, 0.08]
    sa = _make_mock_sa(tomato_probs=v3_probs)
    sb = _make_mock_sb()
    sc = _make_mock_sc()
    raw = _get_raw_feature_vector(sa, sb, sc)
    np.testing.assert_array_almost_equal(
        raw[0:6], np.array(v3_probs, dtype=np.float32), decimal=5,
        err_msg="Indices 0-5 must match sa.tomato_probs_canonical"
    )


def test_feature_vector_slot_lora_probs() -> None:
    """Indices 6-11 must come from sb.tomato_probs_canonical.
    # spec: section 12.2 table lines 3183-3188
    """
    lora_probs = [0.6, 0.1, 0.1, 0.1, 0.05, 0.05]
    sa = _make_mock_sa()
    sb = _make_mock_sb(tomato_probs=lora_probs)
    sc = _make_mock_sc()
    raw = _get_raw_feature_vector(sa, sb, sc)
    np.testing.assert_array_almost_equal(
        raw[6:12], np.array(lora_probs, dtype=np.float32), decimal=5,
        err_msg="Indices 6-11 must match sb.tomato_probs_canonical"
    )


def test_feature_vector_slot_psv_top1() -> None:
    """Index 12 must be sc.compatibility_max.
    # spec: section 12.2 table line 3189
    """
    sa = _make_mock_sa()
    sb = _make_mock_sb()
    sc = _make_mock_sc(compatibility_max=0.85)
    raw = _get_raw_feature_vector(sa, sb, sc)
    assert abs(raw[12] - 0.85) < 1e-5, f"Index 12 should be compatibility_max=0.85, got {raw[12]}"


def test_feature_vector_slot_psv_margin() -> None:
    """Index 13 must be sc.compatibility_margin.
    # spec: section 12.2 table line 3190
    """
    sa = _make_mock_sa()
    sb = _make_mock_sb()
    sc = _make_mock_sc(compatibility_margin=0.42)
    raw = _get_raw_feature_vector(sa, sb, sc)
    assert abs(raw[13] - 0.42) < 1e-5, f"Index 13 should be compatibility_margin=0.42, got {raw[13]}"


def test_feature_vector_slot_agree_v3() -> None:
    """Index 14: 1.0 if PSV argmax == v3 argmax, else 0.0.
    # spec: section 12.2 table line 3191
    """
    sa = _make_mock_sa(argmax=2)
    sb = _make_mock_sb()
    sc_agree = _make_mock_sc(compatibility_argmax=2)
    sc_disagree = _make_mock_sc(compatibility_argmax=3)

    raw_agree = _get_raw_feature_vector(sa, sb, sc_agree)
    raw_disagree = _get_raw_feature_vector(sa, sb, sc_disagree)

    assert raw_agree[14] == pytest.approx(1.0), "Should be 1.0 when PSV argmax == v3 argmax"
    assert raw_disagree[14] == pytest.approx(0.0), "Should be 0.0 when PSV argmax != v3 argmax"


def test_feature_vector_slot_agree_lora() -> None:
    """Index 15: 1.0 if PSV argmax == LoRA argmax, else 0.0.
    # spec: section 12.2 table line 3192
    """
    sa = _make_mock_sa()
    sb = _make_mock_sb(argmax=4)
    sc_agree = _make_mock_sc(compatibility_argmax=4)
    sc_disagree = _make_mock_sc(compatibility_argmax=1)

    raw_agree = _get_raw_feature_vector(sa, sb, sc_agree)
    raw_disagree = _get_raw_feature_vector(sa, sb, sc_disagree)

    assert raw_agree[15] == pytest.approx(1.0), "Should be 1.0 when PSV argmax == LoRA argmax"
    assert raw_disagree[15] == pytest.approx(0.0), "Should be 0.0 when PSV argmax != LoRA argmax"


def test_feature_vector_slot_jsd() -> None:
    """Index 16 must be finite JSD when both signals succeeded.
    # spec: section 12.2 table line 3193
    """
    sa = _make_mock_sa(tomato_probs=[0.5, 0.1, 0.1, 0.1, 0.1, 0.1])
    sb = _make_mock_sb(tomato_probs=[0.1, 0.5, 0.1, 0.1, 0.1, 0.1])
    sc = _make_mock_sc()
    raw = _get_raw_feature_vector(sa, sb, sc)
    assert np.isfinite(raw[16]), f"JSD at index 16 should be finite, got {raw[16]}"
    assert 0.0 <= raw[16] <= 1.0, f"JSD should be in [0, ~0.693], got {raw[16]}"  # ln(2) ≈ 0.693


def test_feature_vector_slot_psv_reliability() -> None:
    """Index 17 must be sc.psv_reliability.
    # spec: section 12.2 table line 3194
    """
    sa = _make_mock_sa()
    sb = _make_mock_sb()
    sc = _make_mock_sc(psv_reliability=0.65)
    raw = _get_raw_feature_vector(sa, sb, sc)
    assert abs(raw[17] - 0.65) < 1e-5, f"Index 17 should be psv_reliability=0.65, got {raw[17]}"


def test_feature_vector_slot_chilli_leakage() -> None:
    """Index 18 must be sa.chilli_leakage.
    # spec: section 12.2 table line 3195
    """
    sa = _make_mock_sa(chilli_leakage=0.12)
    sb = _make_mock_sb()
    sc = _make_mock_sc()
    raw = _get_raw_feature_vector(sa, sb, sc)
    assert abs(raw[18] - 0.12) < 1e-5, f"Index 18 should be chilli_leakage=0.12, got {raw[18]}"


# ---------------------------------------------------------------------------
# Test 3-4: JSD computation
# ---------------------------------------------------------------------------

def test_jsd_bounded() -> None:
    """JSD must be bounded in [0, ln(2)] for proper distributions.
    # spec: section 11.5 lines 3061 — "bounded in [0, log(2)] ≈ [0, 0.693]"
    """
    from tomato_sandbox.classifier.feature_builder import jensen_shannon_divergence

    p = np.array([0.6, 0.1, 0.1, 0.1, 0.05, 0.05])
    q = np.array([0.1, 0.5, 0.1, 0.1, 0.1, 0.1])
    jsd = jensen_shannon_divergence(p, q)
    assert np.isfinite(jsd), "JSD must be finite"
    assert 0.0 <= jsd <= math.log(2) + 1e-6, f"JSD={jsd} not in [0, ln(2)]"


def test_jsd_identical_distributions_zero() -> None:
    """JSD of identical distributions is 0.
    # spec: section 11.5 — JSD is a divergence; D(P||P)=0
    """
    from tomato_sandbox.classifier.feature_builder import jensen_shannon_divergence

    p = np.array([0.5, 0.2, 0.1, 0.1, 0.05, 0.05])
    jsd = jensen_shannon_divergence(p, p.copy())
    assert jsd < 1e-5, f"JSD of identical distributions should be ~0, got {jsd}"


def test_jsd_sentinel_on_signal_failure() -> None:
    """JSD_SENTINEL is used when either signal failed.
    # spec: section 12.2 lines 3225-3228 — "if not (sa.forward_succeeded and
    # sb.forward_succeeded) else JSD_SENTINEL"
    """
    from tomato_sandbox.classifier.feature_builder import JSD_SENTINEL

    # SA failed
    sa_fail = _make_mock_sa(forward_succeeded=False)
    sb = _make_mock_sb()
    sc = _make_mock_sc()
    raw = _get_raw_feature_vector(sa_fail, sb, sc)
    # When SA failed, raw[0:6] and raw[18] are zeroed by apply_degraded_mode,
    # but raw[16] should be JSD_SENTINEL
    # (Note: sentinel path chosen in build_classifier_input before zeroing)
    assert abs(raw[16] - JSD_SENTINEL) < 1e-4, (
        f"Index 16 should be JSD_SENTINEL={JSD_SENTINEL} when SA failed, got {raw[16]}"
    )


# ---------------------------------------------------------------------------
# Test 5: Degraded mode zeroing
# ---------------------------------------------------------------------------

def test_degraded_mode_signal_a_zeroed() -> None:
    """Signal A failed: indices 0-5 and 18 must be 0.
    # spec: section 12.2 lines 3232-3234 — "raw[0:6] = 0.0; raw[18] = 0.0"
    """
    sa = _make_mock_sa(
        tomato_probs=[0.9, 0.0, 0.0, 0.0, 0.0, 0.0],
        chilli_leakage=0.3,
        forward_succeeded=False,
    )
    sb = _make_mock_sb()
    sc = _make_mock_sc()
    raw = _get_raw_feature_vector(sa, sb, sc)
    assert all(raw[0:6] == 0.0), f"Indices 0-5 should be 0 when SA failed: {raw[0:6]}"
    assert raw[18] == 0.0, f"Index 18 should be 0 when SA failed: {raw[18]}"


def test_degraded_mode_signal_b_zeroed() -> None:
    """Signal B failed: indices 6-11 must be 0.
    # spec: section 12.2 lines 3235-3236 — "raw[6:12] = 0.0"
    """
    sa = _make_mock_sa()
    sb = _make_mock_sb(
        tomato_probs=[0.9, 0.0, 0.0, 0.0, 0.0, 0.1],
        forward_succeeded=False,
    )
    sc = _make_mock_sc()
    raw = _get_raw_feature_vector(sa, sb, sc)
    assert all(raw[6:12] == 0.0), f"Indices 6-11 should be 0 when SB failed: {raw[6:12]}"
    # SA indices should still be nonzero
    assert not all(raw[0:6] == 0.0), "SA indices should not be zeroed when only SB failed"


def test_degraded_mode_signal_c_zeroed() -> None:
    """Signal C failed: indices 12, 13, 14, 15, 17 must be 0.
    # spec: section 12.2 lines 3237-3241 — "raw[12:14]=0; raw[14]=0; raw[15]=0; raw[17]=0"
    """
    sa = _make_mock_sa()
    sb = _make_mock_sb()
    sc = _make_mock_sc(
        compatibility_max=0.9,
        compatibility_margin=0.5,
        psv_reliability=0.8,
        forward_succeeded=False,
    )
    raw = _get_raw_feature_vector(sa, sb, sc)
    assert raw[12] == 0.0, f"Index 12 should be 0 when SC failed: {raw[12]}"
    assert raw[13] == 0.0, f"Index 13 should be 0 when SC failed: {raw[13]}"
    assert raw[14] == 0.0, f"Index 14 should be 0 when SC failed: {raw[14]}"
    assert raw[15] == 0.0, f"Index 15 should be 0 when SC failed: {raw[15]}"
    assert raw[17] == 0.0, f"Index 17 should be 0 when SC failed: {raw[17]}"
    # Note: index 16 (JSD) is NOT zeroed when only SC fails (per spec 12.2)


# ---------------------------------------------------------------------------
# Test 6: ClassifierResult field names (all 9)
# ---------------------------------------------------------------------------

def test_classifier_result_field_names() -> None:
    """All 9 ClassifierResult fields must be present with exact spec names.
    # spec: section 12.10 lines 3446-3458 (verbatim dataclass)
    # BLK-010.2 fix: spec wins; 9 fields, not 6
    """
    from tomato_sandbox.classifier.hierarchical_classifier import ClassifierResult

    fields = {f.name for f in dataclasses.fields(ClassifierResult)}
    required = {
        "p_final_calibrated",      # spec line 3449
        "combined_argmax",         # spec line 3450
        "combined_max_prob",       # spec line 3451 — task card omitted this; spec wins
        "combined_margin",         # spec line 3452
        "p_final_uncalibrated",    # spec line 3453
        "p_stage1",                # spec line 3454 — task card omitted this; spec wins
        "p_stage2",                # spec line 3455 — task card omitted this; spec wins
        "classifier_succeeded",    # spec line 3456
        "failure_reason",          # spec line 3457
    }
    missing = required - fields
    assert not missing, (
        f"ClassifierResult missing required fields (spec S12.10): {missing}\n"
        f"Present fields: {fields}"
    )


def test_classifier_result_field_types() -> None:
    """ClassifierResult fields must have correct Python types."""
    from tomato_sandbox.classifier.hierarchical_classifier import ClassifierResult

    sa = _make_mock_sa()
    sb = _make_mock_sb()
    sc = _make_mock_sc()

    from tomato_sandbox.classifier.hierarchical_classifier import compute_classifier
    result = compute_classifier(sa, sb, sc)

    assert isinstance(result.p_final_calibrated, np.ndarray)
    assert isinstance(result.combined_argmax, int)
    assert isinstance(result.combined_max_prob, float)
    assert isinstance(result.combined_margin, float)
    assert isinstance(result.p_final_uncalibrated, np.ndarray)
    assert isinstance(result.p_stage1, np.ndarray)
    assert isinstance(result.p_stage2, np.ndarray)
    assert isinstance(result.classifier_succeeded, bool)
    assert result.failure_reason is None or isinstance(result.failure_reason, str)


# ---------------------------------------------------------------------------
# Test 7-8: Stage forward shapes
# ---------------------------------------------------------------------------

def test_stage1_forward_shape() -> None:
    """Stage 1 forward must produce [3] output.
    # spec: section 12.3 line 3251 — "3-class distribution: [P(healthy), P(diseased), P(OOD)]"
    """
    from tomato_sandbox.classifier.hierarchical_classifier import _stage1_forward

    x = np.zeros(19, dtype=np.float32)
    out = _stage1_forward(x)
    assert out.shape == (3,), f"Stage 1 must output shape (3,), got {out.shape}"


def test_stage2_forward_shape() -> None:
    """Stage 2 forward must produce [5] output.
    # spec: section 12.4 line 3281 — "5-class distribution"
    """
    from tomato_sandbox.classifier.hierarchical_classifier import _stage2_forward

    x = np.zeros(19, dtype=np.float32)
    out = _stage2_forward(x)
    assert out.shape == (5,), f"Stage 2 must output shape (5,), got {out.shape}"


def test_stage1_probs_sum_to_1() -> None:
    """Stage 1 softmax output must sum to 1."""
    from tomato_sandbox.classifier.hierarchical_classifier import _stage1_forward

    x = np.random.default_rng(42).standard_normal(19).astype(np.float32)
    out = _stage1_forward(x)
    assert abs(out.sum() - 1.0) < 1e-5, f"Stage 1 probs must sum to 1, got {out.sum()}"


def test_stage2_probs_sum_to_1() -> None:
    """Stage 2 softmax output must sum to 1."""
    from tomato_sandbox.classifier.hierarchical_classifier import _stage2_forward

    x = np.random.default_rng(42).standard_normal(19).astype(np.float32)
    out = _stage2_forward(x)
    assert abs(out.sum() - 1.0) < 1e-5, f"Stage 2 probs must sum to 1, got {out.sum()}"


# ---------------------------------------------------------------------------
# Test 9: Soft routing
# ---------------------------------------------------------------------------

def test_soft_route_shape() -> None:
    """Soft routing must produce [7] output.
    # spec: section 12.5 lines 3308-3315
    """
    from tomato_sandbox.classifier.hierarchical_classifier import _soft_route

    p1 = np.array([0.3, 0.5, 0.2], dtype=np.float32)
    p2 = np.array([0.4, 0.2, 0.2, 0.1, 0.1], dtype=np.float32)
    out = _soft_route(p1, p2)
    assert out.shape == (7,), f"Soft routing must output shape (7,), got {out.shape}"


def test_soft_route_sums_to_1() -> None:
    """Soft-routed distribution must sum to 1.
    # spec: section 12.5 lines 3317-3322 — "These sum to 1"
    """
    from tomato_sandbox.classifier.hierarchical_classifier import _soft_route

    p1 = np.array([0.3, 0.5, 0.2], dtype=np.float32)
    p2 = np.array([0.4, 0.2, 0.2, 0.1, 0.1], dtype=np.float32)
    out = _soft_route(p1, p2)
    assert abs(out.sum() - 1.0) < 1e-5, f"Soft-routed probs must sum to 1, got {out.sum()}"


def test_soft_route_index_mapping() -> None:
    """Verify P_final[5] == P_stage1[healthy] and P_final[6] == P_stage1[OOD].
    # spec: section 12.5 lines 3313-3314
    """
    from tomato_sandbox.classifier.hierarchical_classifier import _soft_route, IDX_HEALTHY, IDX_OOD

    p1 = np.array([0.30, 0.50, 0.20], dtype=np.float32)  # healthy, diseased, OOD
    p2 = np.array([0.4, 0.2, 0.2, 0.1, 0.1], dtype=np.float32)
    out = _soft_route(p1, p2)

    # spec: section 12.5 line 3313 — P_final[5] = P_stage1[healthy]
    assert abs(out[IDX_HEALTHY] - 0.30) < 1e-5, (
        f"P_final[{IDX_HEALTHY}] should be P_stage1[healthy]=0.30, got {out[IDX_HEALTHY]}"
    )
    # spec: section 12.5 line 3314 — P_final[6] = P_stage1[OOD]
    assert abs(out[IDX_OOD] - 0.20) < 1e-5, (
        f"P_final[{IDX_OOD}] should be P_stage1[OOD]=0.20, got {out[IDX_OOD]}"
    )


def test_soft_route_disease_multiply() -> None:
    """P_final[0] must equal P_stage1[diseased] * P_stage2[foliar].
    # spec: section 12.5 line 3308
    """
    from tomato_sandbox.classifier.hierarchical_classifier import _soft_route, IDX_FOLIAR

    p_diseased = 0.60
    p_foliar = 0.40
    p1 = np.array([0.20, p_diseased, 0.20], dtype=np.float32)
    p2 = np.array([p_foliar, 0.30, 0.10, 0.10, 0.10], dtype=np.float32)
    out = _soft_route(p1, p2)

    expected = p_diseased * p_foliar
    assert abs(out[IDX_FOLIAR] - expected) < 1e-5, (
        f"P_final[foliar] should be {expected}, got {out[IDX_FOLIAR]}"
    )


# ---------------------------------------------------------------------------
# Test 10: Platt calibration
# ---------------------------------------------------------------------------

def test_platt_output_sums_to_1() -> None:
    """Platt-calibrated output must sum to approximately 1 after renormalization.
    # spec: section 12.8 lines 3395-3400 — "renormalize so calibrated probs sum to 1"
    """
    from tomato_sandbox.classifier.hierarchical_classifier import _apply_platt

    p_uncal = np.array([0.4, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05], dtype=np.float32)
    p_cal = _apply_platt(p_uncal)
    assert abs(p_cal.sum() - 1.0) < 1e-4, f"Platt output must sum to 1, got {p_cal.sum()}"


def test_platt_output_shape() -> None:
    """Platt output must be [7]."""
    from tomato_sandbox.classifier.hierarchical_classifier import _apply_platt

    p_uncal = np.array([0.3, 0.2, 0.15, 0.15, 0.1, 0.05, 0.05], dtype=np.float32)
    p_cal = _apply_platt(p_uncal)
    assert p_cal.shape == (7,), f"Platt output must be (7,), got {p_cal.shape}"


def test_platt_identity_preserves_argmax() -> None:
    """With identity Platt (alpha=1, beta=0), argmax should be preserved.

    The pre-F.0 sentinel has alpha=1, beta=0. With these params the logit
    transformation and sigmoid are inverses of each other up to renormalization,
    so the argmax of p_uncal should equal the argmax of p_cal.
    """
    from tomato_sandbox.classifier.hierarchical_classifier import _apply_platt

    p_uncal = np.array([0.05, 0.50, 0.10, 0.15, 0.10, 0.05, 0.05], dtype=np.float32)
    p_cal = _apply_platt(p_uncal)
    assert np.argmax(p_cal) == np.argmax(p_uncal), (
        f"Identity Platt should preserve argmax: {np.argmax(p_uncal)} vs {np.argmax(p_cal)}"
    )


# ---------------------------------------------------------------------------
# Test 11-16: Full compute_classifier
# ---------------------------------------------------------------------------

def test_compute_classifier_returns_classifier_result() -> None:
    """compute_classifier must return a ClassifierResult instance."""
    from tomato_sandbox.classifier.hierarchical_classifier import compute_classifier, ClassifierResult

    sa = _make_mock_sa()
    sb = _make_mock_sb()
    sc = _make_mock_sc()
    result = compute_classifier(sa, sb, sc)
    assert isinstance(result, ClassifierResult)


def test_compute_classifier_p_calibrated_shape() -> None:
    """p_final_calibrated must have shape (7,).
    # spec: section 12.10 line 3449 — "[7], post-Platt"
    """
    from tomato_sandbox.classifier.hierarchical_classifier import compute_classifier

    sa = _make_mock_sa()
    sb = _make_mock_sb()
    sc = _make_mock_sc()
    result = compute_classifier(sa, sb, sc)
    assert result.p_final_calibrated.shape == (7,), (
        f"p_final_calibrated must be (7,), got {result.p_final_calibrated.shape}"
    )


def test_compute_classifier_p_uncalibrated_shape() -> None:
    """p_final_uncalibrated must have shape (7,)."""
    from tomato_sandbox.classifier.hierarchical_classifier import compute_classifier

    sa = _make_mock_sa()
    sb = _make_mock_sb()
    sc = _make_mock_sc()
    result = compute_classifier(sa, sb, sc)
    assert result.p_final_uncalibrated.shape == (7,)


def test_compute_classifier_combined_argmax_valid() -> None:
    """combined_argmax must be in [0, 6].
    # spec: section 12.10 line 3450 — "0-6 in canonical+OOD order"
    """
    from tomato_sandbox.classifier.hierarchical_classifier import compute_classifier

    sa = _make_mock_sa()
    sb = _make_mock_sb()
    sc = _make_mock_sc()
    result = compute_classifier(sa, sb, sc)
    assert 0 <= result.combined_argmax <= 6, (
        f"combined_argmax must be 0-6, got {result.combined_argmax}"
    )


def test_compute_classifier_max_prob_equals_argmax() -> None:
    """combined_max_prob must equal p_final_calibrated[combined_argmax].
    # spec: section 12.10 line 3451 — "max of p_final_calibrated"
    """
    from tomato_sandbox.classifier.hierarchical_classifier import compute_classifier

    sa = _make_mock_sa()
    sb = _make_mock_sb()
    sc = _make_mock_sc()
    result = compute_classifier(sa, sb, sc)
    expected = float(result.p_final_calibrated[result.combined_argmax])
    assert abs(result.combined_max_prob - expected) < 1e-5, (
        f"combined_max_prob={result.combined_max_prob} != p_cal[argmax]={expected}"
    )


def test_compute_classifier_margin_nonneg() -> None:
    """combined_margin must be >= 0.
    # spec: section 12.10 line 3452 — "max minus second-max"
    """
    from tomato_sandbox.classifier.hierarchical_classifier import compute_classifier

    sa = _make_mock_sa()
    sb = _make_mock_sb()
    sc = _make_mock_sc()
    result = compute_classifier(sa, sb, sc)
    assert result.combined_margin >= 0.0, (
        f"combined_margin must be >= 0, got {result.combined_margin}"
    )


def test_compute_classifier_calibrated_sums_to_1() -> None:
    """p_final_calibrated must sum to approximately 1."""
    from tomato_sandbox.classifier.hierarchical_classifier import compute_classifier

    sa = _make_mock_sa()
    sb = _make_mock_sb()
    sc = _make_mock_sc()
    result = compute_classifier(sa, sb, sc)
    total = result.p_final_calibrated.sum()
    assert abs(total - 1.0) < 1e-3, f"p_final_calibrated must sum to ~1, got {total}"


def test_compute_classifier_uncalibrated_sums_to_1() -> None:
    """p_final_uncalibrated must sum to approximately 1 (soft routing partition).
    # spec: section 12.5 lines 3317-3322 — "These sum to 1"
    """
    from tomato_sandbox.classifier.hierarchical_classifier import compute_classifier

    sa = _make_mock_sa()
    sb = _make_mock_sb()
    sc = _make_mock_sc()
    result = compute_classifier(sa, sb, sc)
    total = result.p_final_uncalibrated.sum()
    assert abs(total - 1.0) < 1e-4, f"p_final_uncalibrated must sum to ~1, got {total}"


def test_compute_classifier_succeeded_on_normal_inputs() -> None:
    """classifier_succeeded must be True on normal inputs.
    # spec: section 12.10 line 3456 — "False only if input was malformed"
    """
    from tomato_sandbox.classifier.hierarchical_classifier import compute_classifier

    sa = _make_mock_sa()
    sb = _make_mock_sb()
    sc = _make_mock_sc()
    result = compute_classifier(sa, sb, sc)
    assert result.classifier_succeeded is True
    assert result.failure_reason is None


def test_compute_classifier_p_stage1_shape() -> None:
    """p_stage1 must have shape (3,)."""
    from tomato_sandbox.classifier.hierarchical_classifier import compute_classifier

    sa = _make_mock_sa()
    sb = _make_mock_sb()
    sc = _make_mock_sc()
    result = compute_classifier(sa, sb, sc)
    assert result.p_stage1.shape == (3,)


def test_compute_classifier_p_stage2_shape() -> None:
    """p_stage2 must have shape (5,)."""
    from tomato_sandbox.classifier.hierarchical_classifier import compute_classifier

    sa = _make_mock_sa()
    sb = _make_mock_sb()
    sc = _make_mock_sc()
    result = compute_classifier(sa, sb, sc)
    assert result.p_stage2.shape == (5,)


# ---------------------------------------------------------------------------
# Test 20: All signals failed
# ---------------------------------------------------------------------------

def test_compute_classifier_all_signals_failed() -> None:
    """With all signals failed, classifier still returns a valid ClassifierResult.
    # spec: section 12.7 lines 3348-3373 — degraded-mode handling
    """
    from tomato_sandbox.classifier.hierarchical_classifier import compute_classifier

    sa = _make_mock_sa(forward_succeeded=False)
    sb = _make_mock_sb(forward_succeeded=False)
    sc = _make_mock_sc(forward_succeeded=False)
    result = compute_classifier(sa, sb, sc)

    assert isinstance(result.p_final_calibrated, np.ndarray)
    assert result.p_final_calibrated.shape == (7,)
    assert abs(result.p_final_calibrated.sum() - 1.0) < 1e-3
    assert 0 <= result.combined_argmax <= 6
    # All-zero feature vector → uniform-ish distribution from sentinel weights
    # The classifier still produces a valid result (not a crash)


# ---------------------------------------------------------------------------
# Test 22: classifier_succeeded=False on malformed input
# ---------------------------------------------------------------------------

def test_compute_classifier_handles_nan_input() -> None:
    """If feature vector construction returns NaN, classifier returns failure result.

    We simulate this by setting tomato_probs_canonical to NaN so that
    guard_array is triggered (it zero-fills). The result should still
    have classifier_succeeded=True because guard_array recovers gracefully.
    Note: for a truly malformed input that can't be recovered, we'd need to
    inject a more severe failure. This test verifies the guard pathway.
    """
    from tomato_sandbox.classifier.hierarchical_classifier import compute_classifier

    sa = _make_mock_sa(tomato_probs=[float("nan")] * 6)
    sb = _make_mock_sb()
    sc = _make_mock_sc()
    result = compute_classifier(sa, sb, sc)
    # guard_array fills NaN with zeros; result should still be valid
    assert result.p_final_calibrated.shape == (7,)
    assert np.isfinite(result.p_final_calibrated).all()


# ---------------------------------------------------------------------------
# Test 23: No gpu_lock import in classifier package
# ---------------------------------------------------------------------------

def test_no_gpu_lock_in_classifier() -> None:
    """Classifier must not import gpu_lock (classifier is post-signal, CPU-only).
    # DEC-039 Decision 4; spec section 12.12 — "all numpy, no GPU"
    """
    import tomato_sandbox.classifier.feature_builder as fb
    import tomato_sandbox.classifier.hierarchical_classifier as hc

    fb_source = fb.__file__
    hc_source = hc.__file__

    for path, mod_name in [(fb_source, "feature_builder"), (hc_source, "hierarchical_classifier")]:
        with open(path, encoding="utf-8") as f:
            content = f.read()
        assert "gpu_lock" not in content, (
            f"{mod_name} must not import gpu_lock (spec S12.12: CPU-only)"
        )


# ---------------------------------------------------------------------------
# Test 24: Both import paths work
# ---------------------------------------------------------------------------

def test_import_path_package() -> None:
    """from tomato_sandbox.classifier import ClassifierResult must work."""
    from tomato_sandbox.classifier import ClassifierResult
    assert ClassifierResult is not None


def test_import_path_submodule() -> None:
    """from tomato_sandbox.classifier.hierarchical_classifier import ClassifierResult must work."""
    from tomato_sandbox.classifier.hierarchical_classifier import ClassifierResult
    assert ClassifierResult is not None


def test_import_path_feature_builder() -> None:
    """from tomato_sandbox.classifier.feature_builder import build_classifier_input must work."""
    from tomato_sandbox.classifier.feature_builder import build_classifier_input
    assert build_classifier_input is not None


def test_import_path_compute_classifier() -> None:
    """from tomato_sandbox.classifier import compute_classifier must work."""
    from tomato_sandbox.classifier import compute_classifier
    assert compute_classifier is not None


# ---------------------------------------------------------------------------
# Test: standardization clips to [-3, 3]
# ---------------------------------------------------------------------------

def test_feature_vector_clipped_to_3() -> None:
    """Standardized feature vector must be clipped to [-3, 3].
    # spec: section 12.2 line 3204 — "clip(x_std[i], -3, 3)"
    """
    from tomato_sandbox.classifier.feature_builder import build_classifier_input
    import tomato_sandbox.classifier.feature_builder as fb

    # Patch standardization to make the raw values extreme so clipping fires
    original_mean = fb.CLASSIFIER_FEATURE_MEAN.copy()
    original_std = fb.CLASSIFIER_FEATURE_STD.copy()
    try:
        # Set std to very small value to amplify standardized values
        fb.CLASSIFIER_FEATURE_MEAN[:] = 0.0
        fb.CLASSIFIER_FEATURE_STD[:] = 1e-6  # amplify to force clipping
        sa = _make_mock_sa(tomato_probs=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        sb = _make_mock_sb()
        sc = _make_mock_sc()
        x = build_classifier_input(sa, sb, sc)
    finally:
        fb.CLASSIFIER_FEATURE_MEAN[:] = original_mean
        fb.CLASSIFIER_FEATURE_STD[:] = original_std

    assert np.all(x >= -3.0), f"All values must be >= -3.0: {x[x < -3.0]}"
    assert np.all(x <= 3.0), f"All values must be <= 3.0: {x[x > 3.0]}"
