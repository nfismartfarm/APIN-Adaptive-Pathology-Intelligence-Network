"""
Unit tests for tomato_sandbox/utils/degraded_mode.py.

Tests: VECTOR_DIM, SIGNAL_A/B/C_SLICES, zero_signal_a/b/c,
       apply_degraded_mode, zeros_vector.

# spec: 12.7 lines 3348-3373
# spec: 12.2 lines 3231-3242 (build_classifier_input degraded-mode block)
"""

from __future__ import annotations

import numpy as np
import pytest

from tomato_sandbox.utils.degraded_mode import (
    SIGNAL_A_SLICES,
    SIGNAL_B_SLICES,
    SIGNAL_C_SLICES,
    VECTOR_DIM,
    apply_degraded_mode,
    zero_signal_a,
    zero_signal_b,
    zero_signal_c,
    zeros_vector,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_vector_dim_is_19(self) -> None:
        """spec: 12.1 lines 3149 — 19-dimensional feature vector"""
        assert VECTOR_DIM == 19

    def test_signal_a_slices_cover_0_to_6_and_18(self) -> None:
        """spec: 12.2 lines 3232-3234 — raw[0:6] and raw[18]"""
        flat_indices = {i for start, stop in SIGNAL_A_SLICES for i in range(start, stop)}
        assert 0 in flat_indices
        assert 5 in flat_indices
        assert 18 in flat_indices

    def test_signal_a_slices_do_not_cover_other_indices(self) -> None:
        flat_indices = {i for start, stop in SIGNAL_A_SLICES for i in range(start, stop)}
        # 6-17 must NOT be in Signal A slices
        for idx in range(6, 18):
            assert idx not in flat_indices, f"index {idx} should not be in SIGNAL_A_SLICES"

    def test_signal_b_slices_cover_6_to_12(self) -> None:
        """spec: 12.2 lines 3235-3236 — raw[6:12]"""
        flat_indices = {i for start, stop in SIGNAL_B_SLICES for i in range(start, stop)}
        assert flat_indices == {6, 7, 8, 9, 10, 11}

    def test_signal_c_slices_cover_12_13_14_15_17(self) -> None:
        """spec: 12.2 lines 3237-3241 — raw[12:14], raw[14], raw[15], raw[17]"""
        flat_indices = {i for start, stop in SIGNAL_C_SLICES for i in range(start, stop)}
        assert flat_indices == {12, 13, 14, 15, 17}

    def test_signal_c_does_not_cover_16(self) -> None:
        """Index 16 (JSD) is NOT zeroed when PSV fails — spec 12.2 confirms."""
        flat_indices = {i for start, stop in SIGNAL_C_SLICES for i in range(start, stop)}
        assert 16 not in flat_indices


# ---------------------------------------------------------------------------
# zero_signal_a
# ---------------------------------------------------------------------------


class TestZeroSignalA:
    def _make_ones(self) -> np.ndarray:
        return np.ones(VECTOR_DIM, dtype=np.float32)

    def test_zeros_indices_0_to_5(self) -> None:
        """spec: 12.2 line 3232 — raw[0:6] = 0.0"""
        raw = self._make_ones()
        zero_signal_a(raw)
        np.testing.assert_array_equal(raw[0:6], np.zeros(6))

    def test_zeros_index_18(self) -> None:
        """spec: 12.2 line 3234 — raw[18] = 0.0 (chilli_leakage from v3)"""
        raw = self._make_ones()
        zero_signal_a(raw)
        assert raw[18] == 0.0

    def test_other_indices_unchanged(self) -> None:
        raw = self._make_ones()
        zero_signal_a(raw)
        # Indices 6-17 must remain 1.0
        for idx in range(6, 18):
            assert raw[idx] == 1.0, f"index {idx} should be unchanged"

    def test_returns_same_array(self) -> None:
        raw = self._make_ones()
        result = zero_signal_a(raw)
        assert result is raw

    def test_raises_on_wrong_dim(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            zero_signal_a(np.ones(10, dtype=np.float32))


# ---------------------------------------------------------------------------
# zero_signal_b
# ---------------------------------------------------------------------------


class TestZeroSignalB:
    def _make_ones(self) -> np.ndarray:
        return np.ones(VECTOR_DIM, dtype=np.float32)

    def test_zeros_indices_6_to_11(self) -> None:
        """spec: 12.2 lines 3235-3236 — raw[6:12] = 0.0"""
        raw = self._make_ones()
        zero_signal_b(raw)
        np.testing.assert_array_equal(raw[6:12], np.zeros(6))

    def test_other_indices_unchanged(self) -> None:
        raw = self._make_ones()
        zero_signal_b(raw)
        for idx in list(range(0, 6)) + list(range(12, 19)):
            assert raw[idx] == 1.0, f"index {idx} should be unchanged"

    def test_returns_same_array(self) -> None:
        raw = self._make_ones()
        result = zero_signal_b(raw)
        assert result is raw


# ---------------------------------------------------------------------------
# zero_signal_c
# ---------------------------------------------------------------------------


class TestZeroSignalC:
    def _make_ones(self) -> np.ndarray:
        return np.ones(VECTOR_DIM, dtype=np.float32)

    def test_zeros_indices_12_13_14_15_17(self) -> None:
        """spec: 12.2 lines 3237-3241"""
        raw = self._make_ones()
        zero_signal_c(raw)
        for idx in (12, 13, 14, 15, 17):
            assert raw[idx] == 0.0, f"index {idx} should be 0.0"

    def test_index_16_unchanged(self) -> None:
        """JSD (index 16) is NOT zeroed when PSV fails."""
        raw = self._make_ones()
        zero_signal_c(raw)
        assert raw[16] == 1.0

    def test_indices_0_to_11_unchanged(self) -> None:
        raw = self._make_ones()
        zero_signal_c(raw)
        for idx in range(0, 12):
            assert raw[idx] == 1.0, f"index {idx} should be unchanged"

    def test_returns_same_array(self) -> None:
        raw = self._make_ones()
        result = zero_signal_c(raw)
        assert result is raw


# ---------------------------------------------------------------------------
# apply_degraded_mode
# ---------------------------------------------------------------------------


class TestApplyDegradedMode:
    def _make_ones(self) -> np.ndarray:
        return np.ones(VECTOR_DIM, dtype=np.float32)

    def test_all_ok_no_change(self) -> None:
        raw = self._make_ones()
        apply_degraded_mode(raw, sa_ok=True, sb_ok=True, sc_ok=True)
        np.testing.assert_array_equal(raw, np.ones(VECTOR_DIM))

    def test_sa_failed_zeros_a_block(self) -> None:
        raw = self._make_ones()
        apply_degraded_mode(raw, sa_ok=False, sb_ok=True, sc_ok=True)
        np.testing.assert_array_equal(raw[0:6], np.zeros(6))
        assert raw[18] == 0.0
        # Signal B unchanged
        np.testing.assert_array_equal(raw[6:12], np.ones(6))

    def test_sb_failed_zeros_b_block(self) -> None:
        raw = self._make_ones()
        apply_degraded_mode(raw, sa_ok=True, sb_ok=False, sc_ok=True)
        np.testing.assert_array_equal(raw[6:12], np.zeros(6))
        # Signal A unchanged
        np.testing.assert_array_equal(raw[0:6], np.ones(6))

    def test_sc_failed_zeros_c_block(self) -> None:
        raw = self._make_ones()
        apply_degraded_mode(raw, sa_ok=True, sb_ok=True, sc_ok=False)
        for idx in (12, 13, 14, 15, 17):
            assert raw[idx] == 0.0
        assert raw[16] == 1.0  # JSD not zeroed

    def test_all_failed_zeros_all_blocks(self) -> None:
        raw = self._make_ones()
        apply_degraded_mode(raw, sa_ok=False, sb_ok=False, sc_ok=False)
        # Indices zeroed: 0-5, 6-11, 12-15, 17, 18
        zeroed = set(range(0, 6)) | set(range(6, 12)) | {12, 13, 14, 15, 17, 18}
        # Only index 16 (JSD) should remain 1.0
        for idx in range(VECTOR_DIM):
            if idx in zeroed:
                assert raw[idx] == 0.0, f"index {idx} should be 0.0"
            else:
                assert raw[idx] == 1.0, f"index {idx} should be 1.0"

    def test_returns_same_array(self) -> None:
        raw = self._make_ones()
        result = apply_degraded_mode(raw, sa_ok=True, sb_ok=True, sc_ok=True)
        assert result is raw

    def test_raises_on_wrong_dim(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            apply_degraded_mode(
                np.ones(10, dtype=np.float32), sa_ok=False, sb_ok=False, sc_ok=False
            )


# ---------------------------------------------------------------------------
# zeros_vector
# ---------------------------------------------------------------------------


class TestZerosVector:
    def test_shape(self) -> None:
        v = zeros_vector()
        assert v.shape == (VECTOR_DIM,)

    def test_all_zeros(self) -> None:
        v = zeros_vector()
        np.testing.assert_array_equal(v, np.zeros(VECTOR_DIM))

    def test_dtype_float32(self) -> None:
        v = zeros_vector()
        assert v.dtype == np.float32

    def test_each_call_returns_new_array(self) -> None:
        v1 = zeros_vector()
        v2 = zeros_vector()
        v1[0] = 99.0
        assert v2[0] == 0.0  # v2 is independent
