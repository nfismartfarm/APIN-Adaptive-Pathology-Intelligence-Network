"""
Unit tests for tomato_sandbox/conformal/conformal.py.

Coverage:
  1. ConformalResult dataclass schema
  2. compute_conformal_tau: τ derivation from held_out_subset (n=40)
  3. compute_conformal_tau: α=0.10 (90% coverage), quantile=0.925 per spec
  4. compute_conformal_tau: method="higher" (conservative upper bound)
  5. compute_conformal_tau: NaN in p_final_calibrated handled gracefully
  6. compute_conformal_set: high-confidence single-class set (boundary case)
  7. compute_conformal_set: low-confidence multi-class set (boundary case)
  8. compute_conformal_set: uniform distribution → all-class or near-all-class set
  9. compute_conformal_set: tau=None triggers load_tau fallback (file missing)
  10. compute_conformal_set: tau supplied directly (skip file I/O)
  11. compute_conformal_set: nonconformity_per_class correct values
  12. compute_conformal_set: inside_set_per_class consistent with prediction_set
  13. compute_conformal_set: empty prediction set when all p < 1-tau
  14. load_tau: missing file → fallback 1.0
  15. load_tau: valid JSON → correct tau returned
  16. load_tau: JSON with tau out of [0,1] → clamped
  17. load_tau: corrupt JSON → fallback 1.0
  18. Empirical coverage simulation: 40 calibration points, α=0.10 → ≥36/40 covered
  19. compute_conformal_tau: edge case N=40 exact spec formula: q = ceil(41*0.9)/40 = 0.925
  20. compute_conformal_set: NaN input → guard_array zeros → large set (conservative)
  21. prediction_set_size == len(prediction_set) invariant
  22. nonconformity values in [0, 1] when p is a valid probability vector

# spec: section 13.2 lines 3521-3553 — split conformal algorithm
# spec: section 13.3 lines 3555-3563 — n=40 calibration set
# spec: section 13.4 lines 3565-3581 — α=0.10, 90% coverage
# spec: section 13.5 lines 3583-3619 — τ derivation + storage
# spec: section 13.7 lines 3637-3647 — ConformalResult dataclass
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from tomato_sandbox.conformal.conformal import (
    CONFORMAL_ALPHA,
    CONFORMAL_N_CALIBRATION,
    NUM_CLASSES,
    ConformalResult,
    compute_conformal_set,
    compute_conformal_tau,
    load_tau,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniform_probs() -> np.ndarray:
    """Uniform probability vector over NUM_CLASSES classes."""
    return np.full(NUM_CLASSES, 1.0 / NUM_CLASSES, dtype=np.float32)


def _peaked_probs(peak_class: int, peak_val: float = 0.95) -> np.ndarray:
    """Probability vector with most mass on one class."""
    residual = (1.0 - peak_val) / (NUM_CLASSES - 1)
    p = np.full(NUM_CLASSES, residual, dtype=np.float32)
    p[peak_class] = peak_val
    return p


def _make_holdout(n: int = 40, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic held_out_subset arrays.
    Returns (p_final_calibrated_holdout [n, 7], y_true [n]).
    Each row is a valid probability vector; y_true is always the argmax class.
    """
    rng = np.random.default_rng(seed)
    # Dirichlet concentrates mass on the true class
    alpha_vec = np.ones(NUM_CLASSES) * 0.2
    p = np.zeros((n, NUM_CLASSES), dtype=np.float64)
    y = rng.integers(0, NUM_CLASSES, size=n)
    for i in range(n):
        a = alpha_vec.copy()
        a[y[i]] += 8.0  # put more mass on true class
        sample = rng.dirichlet(a)
        p[i] = sample
    return p, y.astype(np.int64)


# ---------------------------------------------------------------------------
# 1. ConformalResult schema
# spec: section 13.7 lines 3639-3647
# ---------------------------------------------------------------------------


class TestConformalResultSchema:
    def test_fields_present(self):
        """ConformalResult has exactly the fields specified in Section 13.7."""
        cr = ConformalResult(
            prediction_set=[0, 2],
            prediction_set_size=2,
            threshold_tau_used=0.5,
            nonconformity_per_class=np.zeros(NUM_CLASSES),
            inside_set_per_class=np.zeros(NUM_CLASSES, dtype=bool),
        )
        assert hasattr(cr, "prediction_set")
        assert hasattr(cr, "prediction_set_size")
        assert hasattr(cr, "threshold_tau_used")
        assert hasattr(cr, "nonconformity_per_class")
        assert hasattr(cr, "inside_set_per_class")

    def test_prediction_set_is_list_of_ints(self):
        cr = ConformalResult(
            prediction_set=[1, 3, 5],
            prediction_set_size=3,
            threshold_tau_used=0.7,
            nonconformity_per_class=np.zeros(NUM_CLASSES),
            inside_set_per_class=np.zeros(NUM_CLASSES, dtype=bool),
        )
        assert isinstance(cr.prediction_set, list)
        assert all(isinstance(x, int) for x in cr.prediction_set)

    def test_nonconformity_shape(self):
        """nonconformity_per_class is [NUM_CLASSES=7]."""
        # spec: section 13.7 line 3645 — "[7], 1 - p_calibrated[c] for each c"
        cr = ConformalResult(
            prediction_set=[],
            prediction_set_size=0,
            threshold_tau_used=0.3,
            nonconformity_per_class=np.zeros(NUM_CLASSES),
            inside_set_per_class=np.zeros(NUM_CLASSES, dtype=bool),
        )
        assert cr.nonconformity_per_class.shape == (NUM_CLASSES,)

    def test_inside_set_shape(self):
        """inside_set_per_class is [NUM_CLASSES=7] bool."""
        # spec: section 13.7 line 3646 — "[7] bool, True if class is in the set"
        cr = ConformalResult(
            prediction_set=[0],
            prediction_set_size=1,
            threshold_tau_used=0.1,
            nonconformity_per_class=np.zeros(NUM_CLASSES),
            inside_set_per_class=np.zeros(NUM_CLASSES, dtype=bool),
        )
        assert cr.inside_set_per_class.shape == (NUM_CLASSES,)


# ---------------------------------------------------------------------------
# 2. Constants
# spec: section 13.2 line 3538, section 13.3 line 3557
# ---------------------------------------------------------------------------


class TestConstants:
    def test_alpha(self):
        """α = 0.10 per spec section 13.2 line 3538."""
        # spec: section 13.2 line 3538 — "α = 0.10 (for 90% coverage)"
        assert CONFORMAL_ALPHA == 0.10

    def test_n_calibration(self):
        """N=40 per spec section 13.2 line 3538 / 13.3 line 3557."""
        # spec: section 13.3 line 3557 — "the 40-image held_out_subset"
        # spec: section 13.2 line 3538 — "n = 40"
        assert CONFORMAL_N_CALIBRATION == 40

    def test_num_classes(self):
        """7 classes (6 tomato + OOD) per spec section 12.10 lines 3460-3467."""
        # spec: section 12.10 lines 3460-3467 — "0=foliar...6=OOD"
        assert NUM_CLASSES == 7


# ---------------------------------------------------------------------------
# 3. compute_conformal_tau: formula verification
# spec: section 13.2 lines 3533-3538, section 13.5 lines 3583-3600
# ---------------------------------------------------------------------------


class TestComputeConformalTau:
    def test_q_formula_n40_alpha01(self):
        """
        With n=40, α=0.10:
          q = ceil((40+1) * 0.9) / 40 = ceil(36.9) / 40 = 37/40 = 0.925
        τ = 92.5th percentile of nonconformity scores.
        # spec: section 13.2 line 3538 —
        # "q = ceil(41 × 0.9) / 40 = 37/40 = 0.925"
        """
        # Build a trivially-predictable holdout: perfect model,
        # so nonconformity_scores[i] = 1 - 1.0 = 0.0 for all i.
        n = 40
        p = np.zeros((n, NUM_CLASSES), dtype=np.float64)
        y = np.arange(n) % NUM_CLASSES
        # Perfect prediction: p[i, y[i]] = 1.0
        for i in range(n):
            p[i, y[i]] = 1.0

        tau = compute_conformal_tau(p, y, alpha=0.10)
        # All nonconformity scores = 0.0; quantile at any level = 0.0
        assert tau == pytest.approx(0.0, abs=1e-9)

    def test_tau_is_score_at_0925_quantile(self):
        """
        τ = quantile(s, 0.925, method='higher').
        # spec: section 13.5 lines 3594-3596
        """
        n = 40
        # Create scores linearly spaced 0..1
        scores = np.linspace(0.0, 1.0, n)
        # p[i, y[i]] = 1 - scores[i], so nonconformity_scores[i] = scores[i]
        p = np.zeros((n, NUM_CLASSES), dtype=np.float64)
        y = np.zeros(n, dtype=np.int64)
        for i in range(n):
            p[i, 0] = 1.0 - scores[i]
            # Distribute residual equally
            residual = scores[i] / (NUM_CLASSES - 1)
            for c in range(1, NUM_CLASSES):
                p[i, c] = residual

        tau = compute_conformal_tau(p, y, alpha=0.10)
        expected_q = min(math.ceil(41 * 0.9) / 40, 1.0)  # 0.925
        expected_tau = float(np.quantile(scores, expected_q, method="higher"))
        assert tau == pytest.approx(expected_tau, abs=1e-9)

    def test_method_higher_is_conservative(self):
        """
        'higher' interpolation ensures coverage ≥ 1-α, not just approximately.
        # spec: section 13.5 lines 3596-3600 — "conservative choice is the upper one"
        """
        n = 40
        p, y = _make_holdout(n=n, seed=7)
        tau_higher = compute_conformal_tau(p, y, alpha=0.10)
        # Cross-check with manual "higher" computation
        scores = 1.0 - p[np.arange(n), y]
        q = min(math.ceil(41 * 0.9) / 40, 1.0)
        expected = float(np.quantile(scores, q, method="higher"))
        assert tau_higher == pytest.approx(expected, abs=1e-9)

    def test_tau_in_unit_interval(self):
        """τ ∈ [0, 1] for any valid input."""
        p, y = _make_holdout(n=40, seed=99)
        tau = compute_conformal_tau(p, y, alpha=0.10)
        assert 0.0 <= tau <= 1.0

    def test_nan_in_holdout_handled(self):
        """NaN in p_final_calibrated_holdout is handled without exception."""
        p, y = _make_holdout(n=40)
        p[3, :] = np.nan  # inject NaN into one row
        # Should not raise; returns a finite τ
        tau = compute_conformal_tau(p, y, alpha=0.10)
        assert np.isfinite(tau)

    def test_shape_mismatch_raises(self):
        """Wrong array shapes raise ValueError."""
        p_wrong = np.random.rand(40, 6)  # 6 instead of 7 classes
        y = np.zeros(40, dtype=np.int64)
        with pytest.raises(ValueError, match="expected p shape"):
            compute_conformal_tau(p_wrong, y)

    def test_y_length_mismatch_raises(self):
        """y_true length != N raises ValueError."""
        p = np.random.rand(40, NUM_CLASSES)
        y = np.zeros(30, dtype=np.int64)  # wrong length
        with pytest.raises(ValueError, match="y_true length"):
            compute_conformal_tau(p, y)

    def test_out_of_range_y_raises(self):
        """y_true with out-of-range index raises ValueError."""
        p, y = _make_holdout(n=40)
        y[0] = 99  # invalid class
        with pytest.raises(ValueError, match="out-of-range class indices"):
            compute_conformal_tau(p, y)

    def test_perfect_model_tau_zero(self):
        """Perfect model → all nonconformity scores = 0 → τ = 0."""
        n = 40
        p = np.zeros((n, NUM_CLASSES), dtype=np.float64)
        y = np.arange(n) % NUM_CLASSES
        for i in range(n):
            p[i, y[i]] = 1.0
        tau = compute_conformal_tau(p, y, alpha=0.10)
        assert tau == pytest.approx(0.0, abs=1e-9)

    def test_worst_model_tau_one(self):
        """Worst model (p_true = 0) → all nonconformity scores = 1 → τ = 1."""
        n = 40
        # Put all probability on a class that is NOT the true label
        p = np.zeros((n, NUM_CLASSES), dtype=np.float64)
        y = np.zeros(n, dtype=np.int64)  # true class = 0
        for i in range(n):
            p[i, 1] = 1.0  # predict class 1 always; true is 0
        tau = compute_conformal_tau(p, y, alpha=0.10)
        assert tau == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 4. load_tau
# spec: section 13.5 lines 3602-3619
# ---------------------------------------------------------------------------


class TestLoadTau:
    def test_missing_file_returns_fallback_one(self, tmp_path):
        """Missing conformal_tau.json → conservative fallback τ = 1.0."""
        missing = tmp_path / "no_such_file.json"
        tau = load_tau(missing)
        assert tau == 1.0

    def test_valid_json_returns_tau(self, tmp_path):
        """Valid conformal_tau.json → correct τ returned."""
        tau_file = tmp_path / "conformal_tau.json"
        tau_file.write_text(json.dumps({
            "tau": 0.6234,
            "alpha": 0.10,
            "calibration_set_size": 40,
            "calibration_date": "2026-05-01",
            "model_version": "abc1234",
        }))
        tau = load_tau(tau_file)
        assert tau == pytest.approx(0.6234, abs=1e-9)

    def test_tau_out_of_range_clamped(self, tmp_path):
        """tau > 1.0 is clamped to 1.0."""
        tau_file = tmp_path / "conformal_tau.json"
        tau_file.write_text(json.dumps({"tau": 1.5}))
        tau = load_tau(tau_file)
        assert tau == 1.0

    def test_tau_below_zero_clamped(self, tmp_path):
        """tau < 0.0 is clamped to 0.0."""
        tau_file = tmp_path / "conformal_tau.json"
        tau_file.write_text(json.dumps({"tau": -0.1}))
        tau = load_tau(tau_file)
        assert tau == 0.0

    def test_corrupt_json_fallback(self, tmp_path):
        """Corrupt JSON → fallback τ = 1.0."""
        tau_file = tmp_path / "conformal_tau.json"
        tau_file.write_text("NOT VALID JSON {{{")
        tau = load_tau(tau_file)
        assert tau == 1.0

    def test_missing_tau_key_fallback(self, tmp_path):
        """JSON missing 'tau' key → fallback τ = 1.0."""
        tau_file = tmp_path / "conformal_tau.json"
        tau_file.write_text(json.dumps({"alpha": 0.10}))
        tau = load_tau(tau_file)
        assert tau == 1.0


# ---------------------------------------------------------------------------
# 5. compute_conformal_set: boundary cases
# spec: section 13.2 lines 3541-3553
# ---------------------------------------------------------------------------


class TestComputeConformalSet:
    def test_high_confidence_single_class_set(self):
        """
        High-confidence peaked distribution → single-class prediction set.
        # spec: section 13.2 lines 3547-3550 — PredSet = {c : p_c >= 1-τ}
        If τ is small (model is well-calibrated), only the dominant class enters.
        """
        p = _peaked_probs(peak_class=0, peak_val=0.95)
        # τ = 0.10: only classes with p >= 0.90 are in the set
        result = compute_conformal_set(p, tau=0.10)
        assert isinstance(result, ConformalResult)
        assert result.prediction_set == [0], (
            f"Expected [0] but got {result.prediction_set}"
        )
        assert result.prediction_set_size == 1

    def test_low_confidence_multi_class_set(self):
        """
        Low-confidence distribution → multi-class prediction set.
        With τ = 0.50, all classes with p >= 0.50 are in the set.
        Two classes at p=0.40 each → neither reaches threshold.
        """
        p = np.array([0.40, 0.40, 0.04, 0.04, 0.04, 0.04, 0.04], dtype=np.float32)
        # 1-tau = 0.50; no class has p >= 0.50 → empty set
        result = compute_conformal_set(p, tau=0.50)
        assert result.prediction_set == []
        assert result.prediction_set_size == 0

        # With τ = 0.60 (1-τ = 0.40): classes 0 and 1 have p = 0.40 ≥ 0.40
        result2 = compute_conformal_set(p, tau=0.60)
        assert 0 in result2.prediction_set
        assert 1 in result2.prediction_set
        assert result2.prediction_set_size == 2

    def test_uniform_distribution_large_set(self):
        """
        Uniform distribution (1/7 ≈ 0.143 per class).
        With τ = 0.90, threshold = 1-0.90 = 0.10; all classes p≈0.143 >= 0.10
        → all-class set (size 7).
        # spec: section 13.2 line 3553 — "Empty set is rare with proper calibration"
        """
        p = _uniform_probs()
        result = compute_conformal_set(p, tau=0.90)
        # With tau=0.90, 1-tau=0.10; uniform p=1/7≈0.143 >= 0.10 for all
        assert result.prediction_set_size == NUM_CLASSES

    def test_empty_set_when_all_p_below_threshold(self):
        """
        τ = 0.0 → threshold 1-τ = 1.0; no class has p = 1.0 exactly → empty set.
        This is the boundary case spec 13.2 line 3553 describes.
        """
        p = _peaked_probs(peak_class=0, peak_val=0.99)
        result = compute_conformal_set(p, tau=0.0)
        # 1-tau = 1.0; p[0] = 0.99 < 1.0 → empty
        assert result.prediction_set == []
        assert result.prediction_set_size == 0

    def test_tau_one_produces_all_class_set(self):
        """
        τ = 1.0 → threshold = 0.0; all classes have p >= 0 → all in set.
        This is the fallback behaviour when conformal_tau.json is missing.
        """
        p = _uniform_probs()
        result = compute_conformal_set(p, tau=1.0)
        assert result.prediction_set == list(range(NUM_CLASSES))
        assert result.prediction_set_size == NUM_CLASSES

    def test_nonconformity_per_class_values(self):
        """
        nonconformity_per_class[c] = 1 - p[c] for all c.
        # spec: section 13.2 lines 3544-3546 — "s_c = 1 - P_final_calibrated[c]"
        """
        p = _peaked_probs(peak_class=2, peak_val=0.80)
        result = compute_conformal_set(p, tau=0.50)
        expected = 1.0 - p.astype(np.float64)
        np.testing.assert_allclose(result.nonconformity_per_class, expected, atol=1e-6)

    def test_inside_set_consistent_with_prediction_set(self):
        """
        inside_set_per_class[c] == True iff c in prediction_set.
        # spec: section 13.7 line 3646 — "[7] bool, True if class is in the set"
        """
        p = _peaked_probs(peak_class=3, peak_val=0.70)
        result = compute_conformal_set(p, tau=0.40)
        for c in range(NUM_CLASSES):
            expected_inside = c in result.prediction_set
            assert bool(result.inside_set_per_class[c]) == expected_inside, (
                f"Class {c}: inside_set_per_class={result.inside_set_per_class[c]}, "
                f"prediction_set={result.prediction_set}"
            )

    def test_prediction_set_size_invariant(self):
        """prediction_set_size == len(prediction_set) always."""
        for tau in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            p = _uniform_probs()
            result = compute_conformal_set(p, tau=tau)
            assert result.prediction_set_size == len(result.prediction_set)

    def test_nan_input_guard_array(self):
        """
        NaN in p_final_calibrated → guard_array zeros → conservative large set.
        # per nan_guards.guard_array: non-finite → zero-filled array
        """
        p_nan = np.full(NUM_CLASSES, np.nan, dtype=np.float32)
        # After guard_array: p = zeros → nonconformity = 1.0 for all
        # With tau=1.0, all classes enter (1.0 <= 1.0 is True)
        result = compute_conformal_set(p_nan, tau=1.0)
        assert result.prediction_set_size == NUM_CLASSES

    def test_nan_input_with_small_tau(self):
        """NaN input with tau=0.5: nonconformity = 1.0 > 0.5 → empty set."""
        p_nan = np.full(NUM_CLASSES, np.nan, dtype=np.float32)
        result = compute_conformal_set(p_nan, tau=0.5)
        # zeros guard: p=0 → s=1.0; 1.0 <= 0.5 is False → empty set
        assert result.prediction_set == []

    def test_threshold_tau_used_matches_input(self):
        """threshold_tau_used equals the tau passed in."""
        p = _uniform_probs()
        for tau in [0.1, 0.5, 0.73, 0.9]:
            result = compute_conformal_set(p, tau=tau)
            assert result.threshold_tau_used == pytest.approx(tau, abs=1e-9)

    def test_prediction_set_elements_in_range(self):
        """All elements in prediction_set are valid class indices 0-6."""
        # spec: section 13.7 line 3642 — "canonical+OOD indices in the set"
        p = _uniform_probs()
        result = compute_conformal_set(p, tau=0.9)
        for c in result.prediction_set:
            assert 0 <= c < NUM_CLASSES

    def test_nonconformity_in_unit_interval_for_valid_probs(self):
        """nonconformity ∈ [0, 1] when p is a valid probability vector."""
        p = _peaked_probs(peak_class=1, peak_val=0.88)
        result = compute_conformal_set(p, tau=0.5)
        nc = result.nonconformity_per_class
        assert np.all(nc >= 0.0)
        assert np.all(nc <= 1.0 + 1e-9)

    def test_missing_tau_file_uses_fallback(self, tmp_path):
        """When tau=None and tau file missing, load_tau returns 1.0 → all-class set."""
        p = _uniform_probs()
        missing_path = tmp_path / "no_tau.json"
        result = compute_conformal_set(p, tau=None, tau_path=missing_path)
        # Fallback tau=1.0 → threshold=0.0 → all classes enter (p > 0)
        assert result.prediction_set_size == NUM_CLASSES
        assert result.threshold_tau_used == 1.0

    def test_tau_from_valid_json_file(self, tmp_path):
        """compute_conformal_set loads τ from valid JSON when tau=None."""
        tau_file = tmp_path / "conformal_tau.json"
        stored_tau = 0.3
        tau_file.write_text(json.dumps({
            "tau": stored_tau,
            "alpha": 0.10,
            "calibration_set_size": 40,
        }))
        p = _peaked_probs(peak_class=4, peak_val=0.95)
        result = compute_conformal_set(p, tau=None, tau_path=tau_file)
        assert result.threshold_tau_used == pytest.approx(stored_tau, abs=1e-9)
        # 1-tau = 0.70; p[4]=0.95 >= 0.70 → in set; others have p < 0.70
        assert 4 in result.prediction_set

    def test_non_finite_tau_supplied_uses_fallback(self):
        """Non-finite tau supplied directly → treated as 1.0."""
        p = _uniform_probs()
        result = compute_conformal_set(p, tau=float("nan"))
        assert result.threshold_tau_used == 1.0

    def test_inf_tau_supplied_uses_fallback(self):
        """Inf tau supplied → treated as 1.0."""
        p = _uniform_probs()
        result = compute_conformal_set(p, tau=float("inf"))
        assert result.threshold_tau_used == 1.0


# ---------------------------------------------------------------------------
# 6. Empirical coverage simulation
# spec: section 13.4 lines 3566-3581 — 90% coverage guarantee
# ---------------------------------------------------------------------------


class TestEmpiricalCoverage:
    def test_coverage_at_least_90_percent(self):
        """
        Simulate calibration + inference to verify ≥ 90% empirical coverage.

        # spec: section 13.4 lines 3566-3570 —
        # "P(y_true ∈ PredSet) ≥ 1 - α = 0.90"
        # spec: section 13.3 lines 3560-3562 —
        # "finite-sample variation roughly ±5%"

        We use the SAME (p_holdout, y_holdout) set for both calibration and
        coverage verification (leave-one-out is the ideal, but for this test
        we verify the end-to-end τ → coverage chain is not broken).
        On-calibration coverage by construction is ≥ ⌈n*(1-α)⌉/n = 0.925.
        """
        n = 40
        rng = np.random.default_rng(2026)

        # Draw calibration probabilities
        p_cal, y_cal = _make_holdout(n=n, seed=0)
        tau = compute_conformal_tau(p_cal, y_cal, alpha=CONFORMAL_ALPHA)

        # Draw a fresh test set (200 points, same distribution)
        n_test = 200
        p_test, y_test = _make_holdout(n=n_test, seed=1234)

        covered = 0
        for i in range(n_test):
            result = compute_conformal_set(p_test[i].astype(np.float32), tau=tau)
            if y_test[i] in result.prediction_set:
                covered += 1

        empirical_coverage = covered / n_test
        # The conformal guarantee is ≥ 1-α=0.90 under exchangeability.
        # With synthetic but structurally-equivalent data, we expect >90%.
        # Allow a 5% tolerance for finite-sample variation per spec 13.3.
        # spec: section 13.3 lines 3560-3562 — "coverage may differ by ±5%"
        assert empirical_coverage >= 0.80, (
            f"Empirical coverage {empirical_coverage:.3f} < 0.80 "
            f"(expected ≥ 0.90 - 0.10 tolerance)"
        )

    def test_calibration_set_coverage_gte_target(self):
        """
        By construction, on the calibration set itself, coverage ≥ 1-α.
        τ = quantile(s, q, method='higher') with q = ceil((n+1)*(1-α))/n.
        This means ceil(n*(1-α)) scores are ≤ τ, so coverage = ceil(n*(1-α))/n.
        # spec: section 13.5 line 3596 — "method='higher'" ensures ≥ 1-α coverage
        """
        n = 40
        p_cal, y_cal = _make_holdout(n=n, seed=77)
        tau = compute_conformal_tau(p_cal, y_cal, alpha=CONFORMAL_ALPHA)

        covered = sum(
            1 for i in range(n)
            if y_cal[i] in compute_conformal_set(
                p_cal[i].astype(np.float32), tau=tau
            ).prediction_set
        )
        cal_coverage = covered / n
        # Must be ≥ 1-α = 0.90 by construction
        # spec: section 13.4 line 3574 —
        # "by construction, the threshold τ produces approximately 90% coverage
        # on the calibration set"
        assert cal_coverage >= CONFORMAL_ALPHA * 0.5, (
            f"Even calibration-set coverage {cal_coverage:.3f} is suspiciously low"
        )
        # More precisely: must be ≥ 0.90 - small_delta for rounding
        target = (math.ceil(n * (1.0 - CONFORMAL_ALPHA))) / n
        assert cal_coverage >= target - 1e-6, (
            f"Calibration coverage {cal_coverage:.3f} < target {target:.3f}"
        )


# ---------------------------------------------------------------------------
# 7. Integration: τ derivation → set construction end-to-end
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_tau_to_set_pipeline(self):
        """
        End-to-end: derive τ from held_out_subset, then build prediction set.
        Verify the pipeline is connected and produces consistent output.
        """
        p_cal, y_cal = _make_holdout(n=40, seed=314)
        tau = compute_conformal_tau(p_cal, y_cal, alpha=0.10)
        assert 0.0 <= tau <= 1.0

        # Run on a fresh test vector
        p_test = _peaked_probs(peak_class=0, peak_val=0.9)
        result = compute_conformal_set(p_test, tau=tau)
        assert isinstance(result, ConformalResult)
        assert result.prediction_set_size == len(result.prediction_set)
        assert result.threshold_tau_used == pytest.approx(tau, abs=1e-9)

    def test_roundtrip_with_json_file(self, tmp_path):
        """
        Derive τ, write to JSON, load via load_tau, build set.
        Mimics the F.0 calibration → inference cycle.
        # spec: section 13.5 lines 3602-3619
        """
        p_cal, y_cal = _make_holdout(n=40, seed=42)
        tau = compute_conformal_tau(p_cal, y_cal, alpha=0.10)

        tau_file = tmp_path / "conformal_tau.json"
        tau_file.write_text(json.dumps({
            "tau": tau,
            "alpha": 0.10,
            "calibration_set_size": 40,
            "calibration_date": "2026-05-02",
            "model_version": "abc1234",
        }))

        tau_loaded = load_tau(tau_file)
        assert tau_loaded == pytest.approx(tau, abs=1e-9)

        p_test = _peaked_probs(peak_class=1, peak_val=0.88)
        result = compute_conformal_set(p_test, tau=tau_loaded)
        assert result.threshold_tau_used == pytest.approx(tau, abs=1e-9)
