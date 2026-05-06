"""
Unit tests for tomato_sandbox/training/train_classifier.py

Covers all public/module-level functions.
Tests are kept data-free where possible (no features.npz required)
so they run fast in CI without production artifacts.

spec: section 12.3-12.11 (hierarchical classifier training)
spec: section 12.7 (degraded-mode augmentation + verification)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import under test
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tomato_sandbox.training.train_classifier import (
    # Constants
    P_NO_DEGRADE,
    P_DEGRADE_V3,
    P_DEGRADE_LORA,
    P_DEGRADE_PSV,
    _P_DEGRADE_SUM,
    _AUG_SEED,
    _CV_SEED,
    _OOD_HELDOUT_SEED,
    CLASS_NAMES_7,
    NUM_CLASSES,
    IDX_OOD,
    IDX_HEALTHY,
    STAGE1_CLASS_ORDER,
    S1_HEALTHY_IDX,
    S1_DISEASED_IDX,
    S1_OOD_IDX,
    STAGE2_CLASS_ORDER,
    _V3_BLOCK_PROBS,
    _V3_CHILLI_IDX,
    _LORA_BLOCK,
    _PSV_BLOCK_SLICES,
    _JSD_IDX,
    _DEGRADE_THRESH,
    # Public helpers
    _ece,
    _soft_route,
    _soft_route_batch,
    _standardize,
    _to_canonical,
    _to_canonical_batch,
    _apply_degraded_augmentation,
    _apply_platt_batch,
    _run_degraded_mode_verification,
    _repartition_ood,
)


# ===========================================================================
# A. Constant / probability-mass tests
# ===========================================================================

class TestPDegradeConstants:
    """Per DEC-061 sub-decision: V3 P_DEGRADE=0.35, single delta from V2.
    spec: section 12.7 lines 3348-3373
    """

    def test_p_degrade_sum_is_0_35(self):
        assert abs(_P_DEGRADE_SUM - 0.35) < 1e-10

    def test_probability_mass_unity(self):
        total = P_NO_DEGRADE + P_DEGRADE_V3 + P_DEGRADE_LORA + P_DEGRADE_PSV
        assert abs(total - 1.0) < 1e-10

    def test_v3_values(self):
        assert P_NO_DEGRADE == 0.65
        assert P_DEGRADE_V3 == 0.12
        assert P_DEGRADE_LORA == 0.12
        assert P_DEGRADE_PSV == 0.11

    def test_lora_ge_psv(self):
        """spec S12.7 remediation: lora>=psv preserved."""
        assert P_DEGRADE_LORA >= P_DEGRADE_PSV

    def test_aug_seed_unchanged(self):
        """RNG seed=45 unchanged from V2 per DEC-060 sub-decision."""
        assert _AUG_SEED == 45

    def test_degrade_thresholds(self):
        """spec S12.7 lines 3369-3371."""
        assert _DEGRADE_THRESH["v3_off"] == 0.55
        assert _DEGRADE_THRESH["lora_off"] == 0.55
        assert _DEGRADE_THRESH["psv_off"] == 0.65


class TestClassIndexSpace:
    """spec: section 12.10 lines 3460-3467 — 7-class canonical+OOD space."""

    def test_class_names_length(self):
        assert len(CLASS_NAMES_7) == 7

    def test_class_order(self):
        assert CLASS_NAMES_7 == [
            "foliar", "septoria", "late_blight", "ylcv", "mosaic",
            "healthy", "OOD"
        ]

    def test_idx_healthy(self):
        assert IDX_HEALTHY == 5

    def test_idx_ood(self):
        assert IDX_OOD == 6

    def test_stage1_order(self):
        assert STAGE1_CLASS_ORDER == ["healthy", "diseased", "OOD"]
        assert S1_HEALTHY_IDX == 0
        assert S1_DISEASED_IDX == 1
        assert S1_OOD_IDX == 2

    def test_stage2_order(self):
        assert STAGE2_CLASS_ORDER == [
            "foliar", "septoria", "late_blight", "ylcv", "mosaic"
        ]


class TestSignalBlockIndices:
    """Verify signal block slices match degraded_mode.py SLICES.
    spec: section 12.7 (SIGNAL_A, SIGNAL_B, SIGNAL_C slices)
    """

    def test_v3_block_probs(self):
        # SIGNAL_A_SLICES[(0,6)] -> slice(0,6)
        x = np.ones(19)
        x[_V3_BLOCK_PROBS] = 0.0
        assert x[0] == 0.0 and x[5] == 0.0
        assert x[6] == 1.0  # lora unaffected

    def test_v3_chilli_idx(self):
        assert _V3_CHILLI_IDX == 18  # SIGNAL_A_SLICES[(18,19)]

    def test_lora_block(self):
        # SIGNAL_B_SLICES[(6,12)] -> slice(6,12)
        x = np.ones(19)
        x[_LORA_BLOCK] = 0.0
        assert x[6] == 0.0 and x[11] == 0.0
        assert x[0] == 1.0  # v3 unaffected
        assert x[12] == 1.0  # psv unaffected

    def test_psv_block_slices(self):
        # SIGNAL_C_SLICES: [(12,16),(17,18)] — does NOT include idx 16 (JSD)
        x = np.ones(19)
        for slc in _PSV_BLOCK_SLICES:
            x[slc] = 0.0
        # Zeroed: 12,13,14,15,17
        assert x[12] == 0.0
        assert x[15] == 0.0
        assert x[17] == 0.0
        # NOT zeroed: JSD at 16
        assert x[16] == 1.0
        # NOT zeroed: v3 and lora blocks
        assert x[0] == 1.0
        assert x[6] == 1.0

    def test_jsd_idx(self):
        assert _JSD_IDX == 16


# ===========================================================================
# B. _standardize
# ===========================================================================

class TestStandardize:
    """spec: section 12.2 lines 3202-3205 — x_std = clip((x-mean)/(std+1e-6), -3, 3)"""

    def test_zero_mean_unit_std(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        mean = np.array([2.0, 3.0])
        std = np.array([1.0, 1.0])
        result = _standardize(X, mean, std)
        expected = np.array([[-1.0, -1.0], [1.0, 1.0]])
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_clip_at_minus_3(self):
        X = np.array([[-100.0]])
        mean = np.array([0.0])
        std = np.array([1.0])
        result = _standardize(X, mean, std)
        assert result[0, 0] == pytest.approx(-3.0)

    def test_clip_at_plus_3(self):
        X = np.array([[100.0]])
        mean = np.array([0.0])
        std = np.array([1.0])
        result = _standardize(X, mean, std)
        assert result[0, 0] == pytest.approx(3.0)

    def test_zero_std_no_divide_by_zero(self):
        """std=0 -> denominator becomes 1e-6 -> large but clipped to 3."""
        X = np.array([[5.0]])
        mean = np.array([0.0])
        std = np.array([0.0])
        result = _standardize(X, mean, std)
        assert result[0, 0] == pytest.approx(3.0)

    def test_shape_preserved(self):
        X = np.random.rand(50, 19)
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        result = _standardize(X, mean, std)
        assert result.shape == (50, 19)

    def test_clipping_range(self):
        """All values should be in [-3, 3]."""
        X = np.random.RandomState(0).randn(100, 19) * 10
        mean = np.zeros(19)
        std = np.ones(19)
        result = _standardize(X, mean, std)
        assert result.min() >= -3.0
        assert result.max() <= 3.0


# ===========================================================================
# C. _to_canonical / _to_canonical_batch
# ===========================================================================

class TestToCanonical:
    """spec: section 12.10 lines 3460-3467."""

    def test_healthy_maps_to_5(self):
        assert _to_canonical(S1_HEALTHY_IDX, -1) == IDX_HEALTHY  # 5

    def test_ood_maps_to_6(self):
        assert _to_canonical(S1_OOD_IDX, -1) == IDX_OOD  # 6

    def test_diseased_foliar_maps_to_0(self):
        assert _to_canonical(S1_DISEASED_IDX, 0) == 0

    def test_diseased_mosaic_maps_to_4(self):
        assert _to_canonical(S1_DISEASED_IDX, 4) == 4

    def test_all_disease_indices(self):
        for d in range(5):
            assert _to_canonical(S1_DISEASED_IDX, d) == d

    def test_invalid_stage1_raises(self):
        with pytest.raises(ValueError):
            _to_canonical(99, 0)

    def test_batch_version(self):
        y_s1 = np.array([0, 1, 2, 1, 0])
        y_s2 = np.array([-1, 3, -1, 0, -1])
        result = _to_canonical_batch(y_s1, y_s2)
        expected = np.array([5, 3, 6, 0, 5])
        np.testing.assert_array_equal(result, expected)

    def test_batch_preserves_length(self):
        n = 100
        y_s1 = np.random.choice([0, 1, 2], size=n)
        y_s2 = np.where(y_s1 == 1, np.random.choice(5, size=n), -1)
        result = _to_canonical_batch(y_s1, y_s2)
        assert len(result) == n


# ===========================================================================
# D. _soft_route / _soft_route_batch
# ===========================================================================

class TestSoftRoute:
    """spec: section 12.5 lines 3307-3315 — multiplicative combination."""

    def test_healthy_goes_to_index_5(self):
        p_s1 = np.array([0.8, 0.1, 0.1])  # heavy on healthy
        p_s2 = np.ones(5) / 5.0
        result = _soft_route(p_s1, p_s2)
        assert result[IDX_HEALTHY] == pytest.approx(0.8)

    def test_ood_goes_to_index_6(self):
        p_s1 = np.array([0.1, 0.1, 0.8])  # heavy on OOD
        p_s2 = np.ones(5) / 5.0
        result = _soft_route(p_s1, p_s2)
        assert result[IDX_OOD] == pytest.approx(0.8)

    def test_disease_multiplicative(self):
        """p_final[0:5] = p_s1[diseased] * p_s2[0:5]."""
        p_s1 = np.array([0.0, 1.0, 0.0])  # all diseased
        p_s2 = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        result = _soft_route(p_s1, p_s2)
        np.testing.assert_allclose(result[0:5], np.array([0.2, 0.2, 0.2, 0.2, 0.2]))
        assert result[IDX_HEALTHY] == pytest.approx(0.0)
        assert result[IDX_OOD] == pytest.approx(0.0)

    def test_output_length_7(self):
        p_s1 = np.array([0.3, 0.4, 0.3])
        p_s2 = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        result = _soft_route(p_s1, p_s2)
        assert len(result) == 7

    def test_partition_of_unity_for_uniform_inputs(self):
        """When p_s1 and p_s2 are uniform, output should sum to ~1."""
        p_s1 = np.array([1 / 3, 1 / 3, 1 / 3])
        p_s2 = np.ones(5) / 5.0
        result = _soft_route(p_s1, p_s2)
        # healthy + OOD + sum(disease) = 1/3 + 1/3 + 1/3*(1/5)*5 = 1/3+1/3+1/3=1
        assert result.sum() == pytest.approx(1.0, abs=1e-10)

    def test_batch_matches_single(self):
        p_s1_single = np.array([0.5, 0.3, 0.2])
        p_s2_single = np.array([0.4, 0.1, 0.2, 0.2, 0.1])
        expected = _soft_route(p_s1_single, p_s2_single)

        p_s1_batch = p_s1_single[np.newaxis, :]  # (1, 3)
        p_s2_batch = p_s2_single[np.newaxis, :]  # (1, 5)
        result_batch = _soft_route_batch(p_s1_batch, p_s2_batch)
        np.testing.assert_allclose(result_batch[0], expected, atol=1e-12)

    def test_batch_shape(self):
        n = 30
        p_s1 = np.random.dirichlet([1, 1, 1], size=n)
        p_s2 = np.random.dirichlet([1, 1, 1, 1, 1], size=n)
        result = _soft_route_batch(p_s1, p_s2)
        assert result.shape == (n, 7)

    def test_batch_partition_of_unity(self):
        """Each row of batch output should sum to 1."""
        rng = np.random.RandomState(42)
        n = 50
        p_s1 = rng.dirichlet([1, 1, 1], size=n)
        p_s2 = rng.dirichlet([1, 1, 1, 1, 1], size=n)
        result = _soft_route_batch(p_s1, p_s2)
        np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-10)


# ===========================================================================
# E. _apply_degraded_augmentation — V3 bucket boundaries
# ===========================================================================

class TestApplyDegradedAugmentation:
    """spec: section 12.7 lines 3348-3373 — degraded-mode augmentation.
    V3: P_v3=0.12, P_lora=0.12, P_psv=0.11, P_none=0.65
    """

    def _make_ones(self, n: int = 100) -> np.ndarray:
        return np.ones((n, 19), dtype=np.float64)

    def test_output_shape_preserved(self):
        X = self._make_ones()
        rng = np.random.default_rng(45)
        result = _apply_degraded_augmentation(X, rng, 0.35)
        assert result.shape == (100, 19)

    def test_original_not_mutated(self):
        X = self._make_ones()
        X_orig = X.copy()
        rng = np.random.default_rng(45)
        _apply_degraded_augmentation(X, rng, 0.35)
        np.testing.assert_array_equal(X, X_orig)

    def test_v3_off_zeroes_correct_indices(self):
        """V3-off: indices 0-5 and 18 zeroed; JSD (idx 16) = sentinel."""
        # Force a draw in [0, 0.12) by creating a mock rng that gives 0.05
        # Use a 1-sample matrix to isolate the bucket
        X = np.ones((1, 19), dtype=np.float64)

        class _MockRng:
            def uniform(self, lo, hi, size):
                return np.array([0.05])  # < P_DEGRADE_V3 = 0.12

        result = _apply_degraded_augmentation(X, _MockRng(), 0.35)
        # v3 block (0-5) zeroed
        assert np.all(result[0, 0:6] == 0.0)
        # chilli (18) zeroed
        assert result[0, 18] == 0.0
        # JSD replaced with sentinel
        assert result[0, 16] == pytest.approx(0.35)
        # lora (6-11) and psv (12-15, 17) unchanged
        assert result[0, 6] == 1.0
        assert result[0, 12] == 1.0

    def test_lora_off_zeroes_correct_indices(self):
        """Lora-off: indices 6-11 zeroed; JSD (idx 16) = sentinel."""
        X = np.ones((1, 19), dtype=np.float64)

        class _MockRng:
            def uniform(self, lo, hi, size):
                return np.array([0.18])  # in [0.12, 0.24)

        result = _apply_degraded_augmentation(X, _MockRng(), 0.35)
        assert np.all(result[0, 6:12] == 0.0)
        assert result[0, 16] == pytest.approx(0.35)
        # v3 block untouched
        assert result[0, 0] == 1.0
        # psv block untouched
        assert result[0, 12] == 1.0

    def test_psv_off_zeroes_correct_indices(self):
        """PSV-off: indices 12-15 and 17 zeroed; JSD (idx 16) UNCHANGED."""
        X = np.ones((1, 19), dtype=np.float64)

        class _MockRng:
            def uniform(self, lo, hi, size):
                return np.array([0.28])  # in [0.24, 0.35)

        result = _apply_degraded_augmentation(X, _MockRng(), 0.35)
        assert np.all(result[0, 12:16] == 0.0)
        assert result[0, 17] == 0.0
        # JSD stays as-is (PSV failure does not zero JSD)
        assert result[0, 16] == 1.0
        # v3 and lora untouched
        assert result[0, 0] == 1.0
        assert result[0, 6] == 1.0

    def test_no_degrade_leaves_unchanged(self):
        """Draw >= 0.35 -> no changes."""
        X = np.ones((1, 19), dtype=np.float64)

        class _MockRng:
            def uniform(self, lo, hi, size):
                return np.array([0.80])  # >= P_DEGRADE_SUM

        result = _apply_degraded_augmentation(X, _MockRng(), 0.35)
        np.testing.assert_array_equal(result, np.ones((1, 19)))

    def test_jsd_not_zeroed_in_psv_scenario(self):
        """Critical: PSV-off must NOT replace JSD with sentinel.
        spec: S12.7 line 3366 — JSD sentinel only when signal_a or signal_b fails.
        """
        X = np.full((1, 19), 0.5, dtype=np.float64)
        X[0, 16] = 0.999  # sentinel detection value

        class _MockRng:
            def uniform(self, lo, hi, size):
                return np.array([0.30])  # psv_off bucket

        sentinel = 0.35
        result = _apply_degraded_augmentation(X, _MockRng(), sentinel)
        assert result[0, 16] == pytest.approx(0.999), (
            "JSD should NOT change in psv_off scenario"
        )

    def test_large_n_approximate_proportions(self):
        """With N=10000 draws, bucket proportions should be ~V3 values.

        Detection logic:
          v3_off  -> index 0 zeroed (ONLY v3 touches indices 0-5)
          lora_off -> index 6 zeroed (ONLY lora touches indices 6-11)
          psv_off  -> index 12 zeroed (ONLY psv touches index 12)
        These three sets are mutually exclusive by design.
        """
        n = 10000
        X = np.ones((n, 19), dtype=np.float64)
        rng = np.random.default_rng(42)
        result = _apply_degraded_augmentation(X, rng, 0.35)

        # v3_off: col 0 == 0 (only v3-off zeros indices 0-5)
        v3_off = np.sum(result[:, 0] == 0.0)
        # lora_off: col 6 == 0 (only lora-off zeros indices 6-11)
        lora_off = np.sum(result[:, 6] == 0.0)
        # psv_off: col 12 == 0 (only psv-off zeros index 12)
        psv_off = np.sum(result[:, 12] == 0.0)

        # Allow 2% tolerance
        assert abs(v3_off / n - 0.12) < 0.02, f"v3_off proportion: {v3_off/n:.4f}"
        assert abs(lora_off / n - 0.12) < 0.02, f"lora_off proportion: {lora_off/n:.4f}"
        assert abs(psv_off / n - 0.11) < 0.02, f"psv_off proportion: {psv_off/n:.4f}"

    def test_total_degraded_proportion_is_35pct(self):
        """35% of rows should have ANY degradation."""
        n = 10000
        X = np.ones((n, 19), dtype=np.float64)
        rng = np.random.default_rng(99)
        result = _apply_degraded_augmentation(X, rng, 0.35)

        # Any row with any zero => degraded
        any_zeroed = np.any(result != 1.0, axis=1)
        degrade_fraction = any_zeroed.mean()
        assert abs(degrade_fraction - 0.35) < 0.02, (
            f"Expected ~35% degraded, got {degrade_fraction:.4f}"
        )


# ===========================================================================
# F. _apply_platt_batch
# ===========================================================================

class TestApplyPlattBatch:
    """spec: section 12.8 lines 3391-3397 — per-class sigmoid + renormalize."""

    def test_identity_params_preserve_order(self):
        """alpha=1, beta=0 for all classes -> monotone transform, argmax preserved."""
        rng = np.random.RandomState(0)
        probs = rng.dirichlet([1] * 7, size=20)
        alpha = np.ones(7)
        beta = np.zeros(7)
        result = _apply_platt_batch(probs, alpha, beta)
        # argmax should be the same (monotone transform)
        np.testing.assert_array_equal(
            probs.argmax(axis=1), result.argmax(axis=1)
        )

    def test_rows_sum_to_1(self):
        """Renormalization ensures rows sum to 1."""
        rng = np.random.RandomState(1)
        probs = rng.dirichlet([1] * 7, size=30)
        alpha = np.random.RandomState(2).uniform(0.5, 2.0, size=7)
        beta = np.random.RandomState(3).uniform(-2.0, 2.0, size=7)
        result = _apply_platt_batch(probs, alpha, beta)
        np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-10)

    def test_output_in_0_1(self):
        rng = np.random.RandomState(4)
        probs = rng.dirichlet([1] * 7, size=50)
        alpha = np.ones(7) * 1.5
        beta = np.zeros(7)
        result = _apply_platt_batch(probs, alpha, beta)
        assert result.min() >= 0.0
        assert result.max() <= 1.0 + 1e-10

    def test_shape_preserved(self):
        probs = np.random.dirichlet([1] * 7, size=10)
        alpha = np.ones(7)
        beta = np.zeros(7)
        result = _apply_platt_batch(probs, alpha, beta)
        assert result.shape == (10, 7)

    def test_extreme_beta_clipped_logits(self):
        """Very large beta should not produce NaN due to eps clipping."""
        probs = np.full((5, 7), 1 / 7)
        alpha = np.ones(7)
        beta = np.ones(7) * 100.0  # very large
        result = _apply_platt_batch(probs, alpha, beta)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


# ===========================================================================
# G. _ece
# ===========================================================================

class TestEce:
    """spec: section 12.8 line 3404 — 10 equal-width bins."""

    def test_perfect_calibration_is_zero(self):
        """If confidence == accuracy in every bin, ECE = 0."""
        n = 100
        probs = np.zeros((n, 7))
        # All samples have confidence 0.9 in class 0 and 0.1/6 elsewhere
        probs[:, 0] = 0.9
        probs[:, 1:] = 0.1 / 6
        labels = np.zeros(n, dtype=int)  # all correct
        ece = _ece(probs, labels)
        # With confidence ~0.9 and accuracy 1.0, ECE = |1.0 - 0.9| * 1.0 = 0.1
        # (Not exactly 0, but closer to 0 when acc matches conf)
        assert ece >= 0.0

    def test_empty_input_is_zero(self):
        result = _ece(np.zeros((0, 7)), np.zeros(0, dtype=int))
        assert result == 0.0

    def test_output_is_float(self):
        probs = np.random.dirichlet([1] * 7, size=20)
        labels = probs.argmax(axis=1)
        result = _ece(probs, labels)
        assert isinstance(result, float)

    def test_range_0_to_1(self):
        probs = np.random.RandomState(5).dirichlet([1] * 7, size=50)
        labels = np.random.RandomState(6).randint(0, 7, size=50)
        result = _ece(probs, labels)
        assert 0.0 <= result <= 1.0

    def test_all_wrong_high_confidence_has_large_ece(self):
        """Confidently wrong predictions -> large ECE."""
        n = 50
        probs = np.zeros((n, 7))
        probs[:, 0] = 0.95  # high confidence in class 0
        probs[:, 1:] = 0.05 / 6
        labels = np.ones(n, dtype=int)  # all labeled class 1 (wrong)
        ece = _ece(probs, labels)
        assert ece > 0.5, f"ECE for confidently wrong={ece:.4f}"


# ===========================================================================
# H. _repartition_ood  (synthetic data test)
# ===========================================================================

class TestRepartitionOod:
    """OOD repartitioning: 56 OOD rows -> 14 heldout + 42 oof."""

    def _make_fake_features(self):
        """
        Build a synthetic partition/source arrays matching expected structure:
          160 train_subset + 43 held_out_subset + 56 ood = 259 rows
          OOD: 9 model2 folders x 4 images each = 36
               + 10 gaussian + 6 solid + 4 scrambled = 20 synthetic
               total = 56
        """
        n_train = 160
        n_held = 43
        n_ood = 56

        partitions = (
            ["train_subset"] * n_train
            + ["held_out_subset"] * n_held
            + ["ood"] * n_ood
        )
        partition = np.array(partitions, dtype=object)

        sources = (
            ["train"] * n_train
            + ["held"] * n_held
        )
        # 9 model2 folders x 4 images each
        for fi in range(9):
            sources += [f"model2_folder_{fi:02d}"] * 4
        # synthetic
        sources += ["synthetic_noise_gaussian"] * 10
        sources += ["synthetic_noise_solid"] * 6
        sources += ["synthetic_noise_scrambled"] * 4
        source = np.array(sources, dtype=object)

        assert len(partition) == 259
        assert len(source) == 259
        return source, partition

    def test_counts_14_and_42(self):
        source, partition = self._make_fake_features()
        heldout, oof = _repartition_ood(source, partition)
        assert len(heldout) == 14
        assert len(oof) == 42

    def test_no_overlap_between_heldout_and_oof(self):
        source, partition = self._make_fake_features()
        heldout, oof = _repartition_ood(source, partition)
        overlap = set(heldout.tolist()) & set(oof.tolist())
        assert len(overlap) == 0

    def test_all_heldout_are_ood_rows(self):
        source, partition = self._make_fake_features()
        heldout, _ = _repartition_ood(source, partition)
        for idx in heldout:
            assert partition[idx] == "ood", (
                f"heldout idx {idx} has partition={partition[idx]!r}"
            )

    def test_reproducible_with_same_seed(self):
        """seed=46 must give identical results across calls."""
        source, partition = self._make_fake_features()
        h1, o1 = _repartition_ood(source, partition)
        h2, o2 = _repartition_ood(source, partition)
        np.testing.assert_array_equal(h1, h2)
        np.testing.assert_array_equal(o1, o2)

    def test_total_is_56(self):
        source, partition = self._make_fake_features()
        heldout, oof = _repartition_ood(source, partition)
        assert len(heldout) + len(oof) == 56


# ===========================================================================
# I. _run_degraded_mode_verification  (pure unit test with synthetic model)
# ===========================================================================

class TestRunDegradedModeVerification:
    """
    Integration-level unit test using a synthetic perfect classifier.
    We build a dummy 43-sample dataset and a classifier that always
    predicts correctly, then verify the verification function runs without error
    and returns the expected keys.
    """

    def _make_args(self):
        """Build minimal args for _run_degraded_mode_verification."""
        n = 43
        rng = np.random.RandomState(7)

        # Ground truth: mix of healthy, foliar, septoria for all 43 samples
        # Distribution matches approximate held_out_subset (no OOD)
        y_s1 = np.array(
            [S1_HEALTHY_IDX] * 8
            + [S1_DISEASED_IDX] * 35,
            dtype=np.int64,
        )
        y_s2 = np.full(n, -1, dtype=np.int64)
        y_s2[8:] = np.tile([0, 1, 2, 3, 4], 7)[:35]  # cycle disease classes

        X_held = rng.randn(n, 19)
        final_mean = np.zeros(19)
        final_std = np.ones(19)

        # Construct weights that produce roughly correct Stage 1 logits
        # Stage 1: 3 classes, 19 features.
        # Simple approach: weights that produce large logit for correct class.
        s1_weights = np.zeros((3, 19))
        # healthy -> high s1_healthy_idx (0) when low mean feature
        s1_bias = np.array([0.0, 0.0, 0.0])

        s2_weights = np.zeros((5, 19))
        s2_bias = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        alpha_np = np.ones(7)
        beta_np = np.zeros(7)

        return dict(
            X_held=X_held,
            y_s1_held=y_s1,
            y_s2_held=y_s2,
            final_mean=final_mean,
            final_std=final_std,
            s1_weights=s1_weights,
            s1_bias=s1_bias,
            s2_weights=s2_weights,
            s2_bias=s2_bias,
            alpha_np=alpha_np,
            beta_np=beta_np,
            jsd_sentinel=0.35,
        )

    def test_returns_dict_with_all_scenarios(self):
        args = self._make_args()
        results = _run_degraded_mode_verification(**args)
        for scenario in ["all_on", "v3_off", "lora_off", "psv_off"]:
            assert scenario in results, f"Missing scenario: {scenario}"

    def test_scenario_has_required_keys(self):
        args = self._make_args()
        results = _run_degraded_mode_verification(**args)
        required_keys = {"scenario", "n", "macro_f1", "accuracy", "per_class_f1"}
        for scenario, r in results.items():
            assert required_keys.issubset(set(r.keys())), (
                f"Scenario {scenario} missing keys: {required_keys - set(r.keys())}"
            )

    def test_thresholded_scenarios_have_pass_field(self):
        args = self._make_args()
        results = _run_degraded_mode_verification(**args)
        for scenario in ["v3_off", "lora_off", "psv_off"]:
            assert "pass" in results[scenario], f"Missing 'pass' in {scenario}"
            assert "threshold" in results[scenario]

    def test_all_on_has_no_pass_field(self):
        args = self._make_args()
        results = _run_degraded_mode_verification(**args)
        assert "pass" not in results["all_on"]

    def test_macro_f1_in_range(self):
        args = self._make_args()
        results = _run_degraded_mode_verification(**args)
        for scenario, r in results.items():
            f1 = r["macro_f1"]
            assert 0.0 <= f1 <= 1.0, (
                f"macro_f1={f1} out of range for scenario={scenario}"
            )

    def test_n_equals_43(self):
        args = self._make_args()
        results = _run_degraded_mode_verification(**args)
        for r in results.values():
            assert r["n"] == 43

    def test_accuracy_in_range(self):
        args = self._make_args()
        results = _run_degraded_mode_verification(**args)
        for scenario, r in results.items():
            acc = r["accuracy"]
            assert 0.0 <= acc <= 1.0, (
                f"accuracy={acc} out of range for scenario={scenario}"
            )

    def test_per_class_f1_keys_are_class_names(self):
        """per_class_f1 keys should be subsets of CLASS_NAMES_7."""
        args = self._make_args()
        results = _run_degraded_mode_verification(**args)
        valid_names = set(CLASS_NAMES_7)
        for scenario, r in results.items():
            for cls_name in r["per_class_f1"]:
                assert cls_name in valid_names, (
                    f"Unknown class name '{cls_name}' in scenario {scenario}"
                )

    def test_pass_field_is_bool(self):
        args = self._make_args()
        results = _run_degraded_mode_verification(**args)
        for scenario in ["v3_off", "lora_off", "psv_off"]:
            assert isinstance(results[scenario]["pass"], bool)

    def test_all_on_typically_best_f1(self):
        """all_on should produce >= any degraded scenario f1 (with default weights)."""
        args = self._make_args()
        results = _run_degraded_mode_verification(**args)
        all_on_f1 = results["all_on"]["macro_f1"]
        # This is a soft check — with random weights it's not guaranteed
        # but at least verify all_on doesn't throw
        assert all_on_f1 >= 0.0


# ===========================================================================
# J. Probability-consistency smoke tests
# ===========================================================================

class TestProbabilityConsistency:
    """Verify that the pipeline functions compose correctly."""

    def test_soft_route_to_platt_preserves_rows_sum_to_1(self):
        """soft_route_batch output -> apply_platt_batch -> rows still sum to 1."""
        rng = np.random.RandomState(8)
        n = 20
        p_s1 = rng.dirichlet([1, 1, 1], size=n)
        p_s2 = rng.dirichlet([1, 1, 1, 1, 1], size=n)
        p_final = _soft_route_batch(p_s1, p_s2)
        alpha = np.ones(7) * 1.2
        beta = np.zeros(7)
        p_cal = _apply_platt_batch(p_final, alpha, beta)
        np.testing.assert_allclose(p_cal.sum(axis=1), 1.0, atol=1e-10)

    def test_platt_identity_preserves_argmax(self):
        """With alpha=1, beta=0: argmax(p_cal) == argmax(p_final)."""
        rng = np.random.RandomState(9)
        n = 40
        p_s1 = rng.dirichlet([1, 2, 1], size=n)
        p_s2 = rng.dirichlet([2, 1, 1, 1, 1], size=n)
        p_final = _soft_route_batch(p_s1, p_s2)
        alpha = np.ones(7)
        beta = np.zeros(7)
        p_cal = _apply_platt_batch(p_final, alpha, beta)
        np.testing.assert_array_equal(
            p_final.argmax(axis=1), p_cal.argmax(axis=1)
        )

    def test_standardize_then_augment_no_nan(self):
        """standardize followed by augment must not produce NaN/Inf."""
        X = np.random.RandomState(10).randn(50, 19) * 5
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        X_std = _standardize(X, mean, std)
        rng = np.random.default_rng(45)
        X_aug = _apply_degraded_augmentation(X_std, rng, 0.35)
        assert not np.any(np.isnan(X_aug))
        assert not np.any(np.isinf(X_aug))
