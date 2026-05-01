"""
Unit tests for tomato_sandbox/utils/nan_guards.py.

Tests: guard_scalar, guard_array, tta_n_views, filter_finite_views,
aggregate_views, TTA_TRIGGER_THRESHOLD, TTA_ESCALATE_THRESHOLD.

# spec: 11.2 lines 2932-2951 (TTA decision + NaN guard)
# spec: 11.4 lines 3019-3035 (aggregate_views)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tomato_sandbox.utils.nan_guards import (
    TTA_ESCALATE_THRESHOLD,
    TTA_TRIGGER_THRESHOLD,
    aggregate_views,
    filter_finite_views,
    guard_array,
    guard_scalar,
    tta_n_views,
)


# ---------------------------------------------------------------------------
# guard_scalar
# ---------------------------------------------------------------------------


class TestGuardScalar:
    def test_finite_value_returned(self) -> None:
        assert guard_scalar(0.7) == 0.7

    def test_nan_returns_default_nan(self) -> None:
        result = guard_scalar(float("nan"))
        assert math.isnan(result)

    def test_nan_with_custom_default(self) -> None:
        assert guard_scalar(float("nan"), default=0.0) == 0.0

    def test_inf_returns_default(self) -> None:
        assert guard_scalar(float("inf"), default=0.0) == 0.0

    def test_neg_inf_returns_default(self) -> None:
        assert guard_scalar(float("-inf"), default=0.0) == 0.0

    def test_zero_is_finite(self) -> None:
        assert guard_scalar(0.0) == 0.0

    def test_negative_finite(self) -> None:
        assert guard_scalar(-0.5, default=99.0) == -0.5


# ---------------------------------------------------------------------------
# guard_array
# ---------------------------------------------------------------------------


class TestGuardArray:
    def test_finite_array_returned(self) -> None:
        arr = np.array([0.1, 0.2, 0.3, 0.4, 0.0, 0.0], dtype=np.float32)
        result = guard_array(arr, expected_len=6)
        np.testing.assert_array_equal(result, arr)
        assert result.dtype == np.float32

    def test_nan_in_array_returns_zeros(self) -> None:
        arr = np.array([0.1, float("nan"), 0.3, 0.4, 0.0, 0.0], dtype=np.float32)
        result = guard_array(arr, expected_len=6)
        np.testing.assert_array_equal(result, np.zeros(6, dtype=np.float32))

    def test_inf_in_array_returns_zeros(self) -> None:
        arr = np.array([0.1, 0.2, float("inf"), 0.4, 0.0, 0.0], dtype=np.float32)
        result = guard_array(arr, expected_len=6)
        np.testing.assert_array_equal(result, np.zeros(6, dtype=np.float32))

    def test_wrong_length_returns_zeros(self) -> None:
        arr = np.array([0.5, 0.5], dtype=np.float32)
        result = guard_array(arr, expected_len=6)
        assert result.shape == (6,)
        np.testing.assert_array_equal(result, np.zeros(6, dtype=np.float32))

    def test_custom_default_value(self) -> None:
        arr = np.array([float("nan")] * 6, dtype=np.float32)
        result = guard_array(arr, expected_len=6, default_value=-1.0)
        np.testing.assert_array_equal(result, np.full(6, -1.0, dtype=np.float32))

    def test_list_input_coerced(self) -> None:
        """Should accept plain Python lists."""
        result = guard_array([0.1, 0.2, 0.3, 0.4, 0.0, 0.0], expected_len=6)
        assert result.shape == (6,)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# TTA threshold constants
# ---------------------------------------------------------------------------


class TestTTAThresholds:
    def test_trigger_threshold_value(self) -> None:
        """spec: 11.2 line 2932 — TOMATO_TTA_TRIGGER_THRESHOLD default 0.55"""
        assert TTA_TRIGGER_THRESHOLD == 0.55

    def test_escalate_threshold_value(self) -> None:
        """spec: 11.2 line 2938 — TOMATO_TTA_ESCALATE_THRESHOLD default 0.45"""
        assert TTA_ESCALATE_THRESHOLD == 0.45


# ---------------------------------------------------------------------------
# tta_n_views
# ---------------------------------------------------------------------------


class TestTtaNViews:
    """spec: 11.2 lines 2932-2951"""

    def test_nan_returns_1(self) -> None:
        """NaN combined_max_prob → no TTA (n_views = 1).
        # spec: 11.2 lines 2946-2951
        """
        assert tta_n_views(float("nan")) == 1

    def test_inf_returns_1(self) -> None:
        assert tta_n_views(float("inf")) == 1

    def test_above_trigger_returns_1(self) -> None:
        """combined_max_prob >= 0.55 → no TTA."""
        assert tta_n_views(0.55) == 1
        assert tta_n_views(0.90) == 1
        assert tta_n_views(1.00) == 1

    def test_below_trigger_above_escalate_returns_2(self) -> None:
        """0.45 <= combined_max_prob < 0.55 → 2-view TTA."""
        assert tta_n_views(0.54) == 2
        assert tta_n_views(0.50) == 2
        assert tta_n_views(0.45) == 2  # boundary: escalate_threshold is inclusive

    def test_below_escalate_returns_5(self) -> None:
        """combined_max_prob < 0.45 → 5-view TTA."""
        assert tta_n_views(0.44) == 5
        assert tta_n_views(0.30) == 5
        assert tta_n_views(0.00) == 5

    def test_custom_thresholds(self) -> None:
        assert tta_n_views(0.70, trigger_threshold=0.75, escalate_threshold=0.60) == 2
        assert tta_n_views(0.55, trigger_threshold=0.75, escalate_threshold=0.60) == 5

    def test_boundary_exactly_trigger(self) -> None:
        """At exactly the trigger threshold, no TTA."""
        assert tta_n_views(TTA_TRIGGER_THRESHOLD) == 1

    def test_boundary_exactly_escalate(self) -> None:
        """At exactly the escalate threshold, 2-view TTA (inclusive lower bound)."""
        assert tta_n_views(TTA_ESCALATE_THRESHOLD) == 2


# ---------------------------------------------------------------------------
# filter_finite_views
# ---------------------------------------------------------------------------


class TestFilterFiniteViews:
    def test_all_ok_all_finite(self) -> None:
        p0 = np.array([0.5, 0.3, 0.1, 0.05, 0.04, 0.01], dtype=np.float32)
        p1 = np.array([0.4, 0.4, 0.1, 0.05, 0.03, 0.02], dtype=np.float32)
        result = filter_finite_views([p0, p1], [True, True])
        assert len(result) == 2

    def test_failed_view_excluded(self) -> None:
        p0 = np.array([0.5, 0.3, 0.1, 0.05, 0.04, 0.01], dtype=np.float32)
        p1 = np.array([0.4, 0.4, 0.1, 0.05, 0.03, 0.02], dtype=np.float32)
        result = filter_finite_views([p0, p1], [True, False])
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], p0)

    def test_nan_view_excluded(self) -> None:
        p0 = np.array([0.5, 0.3, 0.1, 0.05, 0.04, 0.01], dtype=np.float32)
        p_nan = np.array([float("nan")] * 6, dtype=np.float32)
        result = filter_finite_views([p0, p_nan], [True, True])
        assert len(result) == 1

    def test_all_failed_returns_empty(self) -> None:
        p0 = np.array([0.1] * 6, dtype=np.float32)
        result = filter_finite_views([p0], [False])
        assert result == []

    def test_empty_input(self) -> None:
        result = filter_finite_views([], [])
        assert result == []


# ---------------------------------------------------------------------------
# aggregate_views
# ---------------------------------------------------------------------------


class TestAggregateViews:
    """spec: 11.4 lines 3019-3030"""

    def test_two_ok_views_averaged(self) -> None:
        p0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        p1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        agg, n = aggregate_views([p0, p1], [True, True])
        assert n == 2
        expected = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float32)
        np.testing.assert_allclose(agg, expected, atol=1e-6)

    def test_single_ok_view(self) -> None:
        p0 = np.array([0.9, 0.05, 0.02, 0.01, 0.01, 0.01], dtype=np.float32)
        agg, n = aggregate_views([p0], [True])
        assert n == 1
        np.testing.assert_array_equal(agg, p0)

    def test_all_failed_returns_zeros(self) -> None:
        """spec: 11.4 lines 3027-3029 — all views fail → zero-filled"""
        p0 = np.array([0.1] * 6, dtype=np.float32)
        agg, n = aggregate_views([p0], [False])
        assert n == 0
        np.testing.assert_array_equal(agg, np.zeros(6, dtype=np.float32))

    def test_one_failed_one_ok(self) -> None:
        p0 = np.array([0.1] * 6, dtype=np.float32)
        p1 = np.array([0.6, 0.2, 0.1, 0.05, 0.03, 0.02], dtype=np.float32)
        agg, n = aggregate_views([p0, p1], [False, True])
        assert n == 1
        np.testing.assert_array_equal(agg, p1)

    def test_nan_view_excluded_from_average(self) -> None:
        p0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        p_nan = np.array([float("nan")] * 6, dtype=np.float32)
        agg, n = aggregate_views([p0, p_nan], [True, True])
        assert n == 1
        np.testing.assert_array_equal(agg, p0)

    def test_result_is_float32(self) -> None:
        p0 = np.array([0.5] * 6, dtype=np.float64)
        agg, _ = aggregate_views([p0], [True])
        assert agg.dtype == np.float32
