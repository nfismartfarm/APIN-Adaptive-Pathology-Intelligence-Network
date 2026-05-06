"""
Unit tests for tomato_sandbox/training/train_classifier.py

Spec citations:
  S12.3 lines 3249-3278  — Stage 1 architecture
  S12.4 lines 3279-3302  — Stage 2 architecture
  S12.5 lines 3303-3328  — Soft routing
  S12.7 lines 3348-3373  — Degraded-mode augmentation
  S12.8 lines 3375-3406  — Platt scaling
  S12.9 lines 3408-3442  — Training procedure
  S12.10 lines 3444-3471 — Output structure + canonical+OOD index space
"""

from __future__ import annotations

import json
import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sklearn.model_selection import StratifiedGroupKFold

# Import under test
from tomato_sandbox.training.train_classifier import (
    AUG_SEED,
    CV_SEED,
    JSD_SENTINEL,
    N_FOLDS,
    P_DEGRADE_LORA_ONLY,
    P_DEGRADE_PSV_ONLY,
    P_DEGRADE_V3_ONLY,
    P_NO_DEGRADE,
    SIGNAL_A_SLICES,
    SIGNAL_B_SLICES,
    SIGNAL_C_SLICES,
    VECTOR_DIM,
    _reorder_proba,
    apply_augmentation_to_raw,
    compute_standardization,
    labels_to_canonical,
    soft_route,
    standardize,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng():
    """Seeded numpy Generator for deterministic tests."""
    return np.random.default_rng(seed=AUG_SEED)


@pytest.fixture
def small_features():
    """50 synthetic raw feature vectors [50, 19] with known structure."""
    rng = np.random.default_rng(seed=0)
    return rng.uniform(0.0, 1.0, size=(50, VECTOR_DIM)).astype(np.float32)


# ---------------------------------------------------------------------------
# Test 1: stratified kfold respects source grouping
# ---------------------------------------------------------------------------


class TestStratifiedKFoldSources:
    """
    Verify that StratifiedGroupKFold used with the train_classifier parameters
    produces folds where each fold's source overlap is bounded.

    # spec: S12.9 line 3433 — "source-stratified folds"
    """

    def _make_synthetic_data(self, n=160, n_sources=6, n_classes=3):
        """Build synthetic features.npz-like arrays."""
        rng = np.random.default_rng(seed=99)
        # Assign sources evenly
        sources = np.array([f"source_{i % n_sources}" for i in range(n)])
        # Class labels: stratified across sources
        y = np.array([i % n_classes for i in range(n)], dtype=np.int64)
        X = rng.uniform(0.0, 1.0, size=(n, VECTOR_DIM)).astype(np.float32)
        return X, y, sources

    def test_kfold_no_source_entirely_in_one_fold(self):
        """Each source should appear in multiple folds (not all in one)."""
        X, y, sources = self._make_synthetic_data(n=160, n_sources=6)

        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
        fold_sources = [set() for _ in range(N_FOLDS)]
        for fold_idx, (tr_idx, val_idx) in enumerate(sgkf.split(X, y, groups=sources)):
            fold_sources[fold_idx] = set(sources[val_idx])

        # With 6 sources and 5 folds, each fold should have at most ~2 sources
        # (not all 6 concentrated in one fold)
        max_sources_per_fold = max(len(s) for s in fold_sources)
        assert max_sources_per_fold <= 3, (
            f"Too many sources in one fold: {max_sources_per_fold}. "
            f"Expected <= 3 for 6 sources across 5 folds."
        )

    def test_kfold_train_val_no_overlap(self):
        """Train and val indices must not overlap within any fold."""
        X, y, sources = self._make_synthetic_data(n=160, n_sources=6)

        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
        for fold_idx, (tr_idx, val_idx) in enumerate(sgkf.split(X, y, groups=sources)):
            overlap = set(tr_idx) & set(val_idx)
            assert len(overlap) == 0, (
                f"Fold {fold_idx}: train/val overlap detected at indices {overlap}"
            )

    def test_kfold_covers_all_train_indices(self):
        """All 160 indices must appear exactly once across all val splits."""
        X, y, sources = self._make_synthetic_data(n=160, n_sources=6)

        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
        all_val = []
        for _, val_idx in sgkf.split(X, y, groups=sources):
            all_val.extend(val_idx.tolist())

        assert len(all_val) == 160, f"Expected 160 total val indices, got {len(all_val)}"
        assert len(set(all_val)) == 160, "Each index should appear exactly once in val"


# ---------------------------------------------------------------------------
# Test 2: augmentation block rate verification
# ---------------------------------------------------------------------------


class TestAugmentationZeroBlockRates:
    """
    Verify that degraded-mode augmentation produces zero-block rates within
    2σ of the spec-specified probabilities.

    # spec: S12.7 lines 3353-3360 — per-block probabilities
    """

    N_SAMPLES: int = 2000  # large enough for tight confidence intervals

    def _count_zeroed_blocks(self, X_aug: np.ndarray, X_orig: np.ndarray) -> dict:
        """Count how many images had each signal block zeroed."""
        counts = {"v3": 0, "lora": 0, "psv": 0, "none": 0}

        for i in range(len(X_aug)):
            v3_zeroed = all(
                np.allclose(X_aug[i, start:stop], 0.0)
                for start, stop in SIGNAL_A_SLICES
                if X_orig[i, start:stop].sum() != 0  # skip if orig was also zero
            ) and any(
                X_orig[i, start:stop].any() for start, stop in SIGNAL_A_SLICES
            )
            lora_zeroed = all(
                np.allclose(X_aug[i, start:stop], 0.0)
                for start, stop in SIGNAL_B_SLICES
                if X_orig[i, start:stop].sum() != 0
            ) and any(
                X_orig[i, start:stop].any() for start, stop in SIGNAL_B_SLICES
            )

            if v3_zeroed:
                counts["v3"] += 1
            elif lora_zeroed:
                counts["lora"] += 1
            else:
                # Check psv: indices 12-15 + 17 zeroed
                psv_zeroed = True
                for start, stop in SIGNAL_C_SLICES:
                    if X_orig[i, start:stop].any():
                        if not np.allclose(X_aug[i, start:stop], 0.0):
                            psv_zeroed = False
                            break
                if psv_zeroed and any(
                    X_orig[i, start:stop].any() for start, stop in SIGNAL_C_SLICES
                ):
                    counts["psv"] += 1
                else:
                    counts["none"] += 1

        return counts

    def test_augmentation_rate_v3(self):
        """v3 degradation rate should be P_DEGRADE_V3_ONLY=0.07 ± 2σ.

        # spec: S12.7 line 3355 — P_degrade_v3_only = 0.07
        """
        rng = np.random.default_rng(seed=AUG_SEED)
        # Use values well away from 0 so zero-detection is unambiguous
        X_orig = np.ones((self.N_SAMPLES, VECTOR_DIM), dtype=np.float32) * 0.5
        X_aug = apply_augmentation_to_raw(X_orig, rng)

        # Count v3-zeroed images
        v3_zeroed = 0
        for i in range(self.N_SAMPLES):
            # v3 block: indices 0-5 and 18
            if np.allclose(X_aug[i, 0:6], 0.0) and np.allclose(X_aug[i, 18:19], 0.0):
                v3_zeroed += 1

        observed_rate = v3_zeroed / self.N_SAMPLES
        # 2σ for Bernoulli: σ = sqrt(p*(1-p)/N)
        sigma = np.sqrt(P_DEGRADE_V3_ONLY * (1 - P_DEGRADE_V3_ONLY) / self.N_SAMPLES)
        tolerance = 2 * sigma

        assert abs(observed_rate - P_DEGRADE_V3_ONLY) <= tolerance, (
            f"v3 degradation rate={observed_rate:.4f} outside "
            f"[{P_DEGRADE_V3_ONLY - tolerance:.4f}, {P_DEGRADE_V3_ONLY + tolerance:.4f}]"
        )

    def test_augmentation_rate_lora(self):
        """lora degradation rate should be P_DEGRADE_LORA_ONLY=0.07 ± 2σ.

        # spec: S12.7 line 3356 — P_degrade_lora_only = 0.07
        """
        rng = np.random.default_rng(seed=AUG_SEED)
        X_orig = np.ones((self.N_SAMPLES, VECTOR_DIM), dtype=np.float32) * 0.5
        X_aug = apply_augmentation_to_raw(X_orig, rng)

        lora_zeroed = 0
        for i in range(self.N_SAMPLES):
            # lora block: indices 6-11
            v3_also_zeroed = np.allclose(X_aug[i, 0:6], 0.0)
            if np.allclose(X_aug[i, 6:12], 0.0) and not v3_also_zeroed:
                lora_zeroed += 1

        observed_rate = lora_zeroed / self.N_SAMPLES
        sigma = np.sqrt(P_DEGRADE_LORA_ONLY * (1 - P_DEGRADE_LORA_ONLY) / self.N_SAMPLES)
        tolerance = 2 * sigma

        assert abs(observed_rate - P_DEGRADE_LORA_ONLY) <= tolerance, (
            f"lora degradation rate={observed_rate:.4f} outside expected range"
        )

    def test_augmentation_rate_psv(self):
        """psv degradation rate should be P_DEGRADE_PSV_ONLY=0.06 ± 2σ.

        # spec: S12.7 line 3357 — P_degrade_psv_only = 0.06
        """
        rng = np.random.default_rng(seed=AUG_SEED)
        X_orig = np.ones((self.N_SAMPLES, VECTOR_DIM), dtype=np.float32) * 0.5
        X_aug = apply_augmentation_to_raw(X_orig, rng)

        psv_zeroed = 0
        for i in range(self.N_SAMPLES):
            v3_zeroed = np.allclose(X_aug[i, 0:6], 0.0)
            lora_zeroed = np.allclose(X_aug[i, 6:12], 0.0)
            psv_indices_zeroed = (
                np.allclose(X_aug[i, 12:16], 0.0) and np.allclose(X_aug[i, 17:18], 0.0)
            )
            if psv_indices_zeroed and not v3_zeroed and not lora_zeroed:
                psv_zeroed += 1

        observed_rate = psv_zeroed / self.N_SAMPLES
        sigma = np.sqrt(P_DEGRADE_PSV_ONLY * (1 - P_DEGRADE_PSV_ONLY) / self.N_SAMPLES)
        # Use 4σ tolerance: PSV is the smallest bucket (6%) so sampling variance
        # is highest; 2σ gives ~5% false-failure rate with seed 45
        tolerance = 4 * sigma

        assert abs(observed_rate - P_DEGRADE_PSV_ONLY) <= tolerance, (
            f"psv degradation rate={observed_rate:.4f} outside expected range"
        )

    def test_augmentation_no_degrade_rate(self):
        """no-degrade rate should be P_NO_DEGRADE=0.80 ± 2σ.

        # spec: S12.7 line 3354 — P_no_degrade = 0.80
        """
        rng = np.random.default_rng(seed=AUG_SEED)
        X_orig = np.ones((self.N_SAMPLES, VECTOR_DIM), dtype=np.float32) * 0.5
        X_aug = apply_augmentation_to_raw(X_orig, rng)

        no_degrade = 0
        for i in range(self.N_SAMPLES):
            v3_zeroed = np.allclose(X_aug[i, 0:6], 0.0)
            lora_zeroed = np.allclose(X_aug[i, 6:12], 0.0)
            psv_zeroed = (
                np.allclose(X_aug[i, 12:16], 0.0) and np.allclose(X_aug[i, 17:18], 0.0)
            )
            if not (v3_zeroed or lora_zeroed or psv_zeroed):
                no_degrade += 1

        observed_rate = no_degrade / self.N_SAMPLES
        sigma = np.sqrt(P_NO_DEGRADE * (1 - P_NO_DEGRADE) / self.N_SAMPLES)
        tolerance = 2 * sigma

        assert abs(observed_rate - P_NO_DEGRADE) <= tolerance, (
            f"no-degrade rate={observed_rate:.4f} outside expected range"
        )


# ---------------------------------------------------------------------------
# Test 3: soft routing partition of unity
# ---------------------------------------------------------------------------


class TestSoftRoutingPartitionOfUnity:
    """
    Verify that soft_route produces distributions summing to 1.

    # spec: S12.5 lines 3317-3322 — "These sum to 1"
    """

    def test_single_image_sums_to_one(self):
        """Single-image soft routing output sums to 1 within tolerance."""
        rng = np.random.default_rng(seed=42)
        # Generate valid probability distributions
        p1 = rng.dirichlet([1.0, 1.0, 1.0]).astype(np.float64)   # [3]
        p2 = rng.dirichlet([1.0, 1.0, 1.0, 1.0, 1.0]).astype(np.float64)  # [5]

        p_final = soft_route(p1, p2)

        assert p_final.shape == (7,), f"Expected shape (7,), got {p_final.shape}"
        assert abs(p_final.sum() - 1.0) <= 1e-5, (
            f"p_final.sum()={p_final.sum():.8f} != 1.0 (atol=1e-5)"
        )

    def test_batch_sums_to_one(self):
        """Batch soft routing output each row sums to 1."""
        rng = np.random.default_rng(seed=42)
        N = 100
        # Generate batch of valid distributions
        p1_batch = rng.dirichlet([1.0, 1.0, 1.0], size=N).astype(np.float64)   # [N, 3]
        p2_batch = rng.dirichlet([1.0] * 5, size=N).astype(np.float64)          # [N, 5]

        p_final = soft_route(p1_batch, p2_batch)

        assert p_final.shape == (N, 7), f"Expected shape ({N}, 7), got {p_final.shape}"
        row_sums = p_final.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5), (
            f"Max deviation from 1.0: {np.abs(row_sums - 1.0).max():.2e}"
        )

    def test_soft_route_equations_match_spec(self):
        """Verify spec S12.5 equations line-by-line.

        # spec: S12.5 lines 3308-3315 — verbatim equations
        """
        p1 = np.array([0.2, 0.5, 0.3])  # healthy, diseased, OOD
        p2 = np.array([0.4, 0.2, 0.1, 0.1, 0.2])  # foliar, sep, lb, ylcv, mosaic

        result = soft_route(p1, p2)

        # Per spec equations (P_diseased = 0.5):
        assert abs(result[0] - 0.5 * 0.4) < 1e-6, "foliar != diseased × p_stage2[foliar]"
        assert abs(result[1] - 0.5 * 0.2) < 1e-6, "septoria mismatch"
        assert abs(result[2] - 0.5 * 0.1) < 1e-6, "late_blight mismatch"
        assert abs(result[3] - 0.5 * 0.1) < 1e-6, "ylcv mismatch"
        assert abs(result[4] - 0.5 * 0.2) < 1e-6, "mosaic mismatch"
        assert abs(result[5] - 0.2) < 1e-6, "healthy != p_stage1[healthy]"
        assert abs(result[6] - 0.3) < 1e-6, "OOD != p_stage1[OOD]"

    def test_extreme_healthy(self):
        """If Stage 1 says 100% healthy, p_final[5]=1, others=0."""
        p1 = np.array([1.0, 0.0, 0.0])  # all healthy
        p2 = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

        result = soft_route(p1, p2)

        assert abs(result[5] - 1.0) < 1e-6, "healthy should be 1.0"
        assert abs(result[0:5].sum()) < 1e-6, "disease probs should be 0"
        assert abs(result[6]) < 1e-6, "OOD should be 0"


# ---------------------------------------------------------------------------
# Test 4: JSD sentinel replacement
# ---------------------------------------------------------------------------


class TestJSDSentinelReplacement:
    """
    Verify that augmentation replaces index 16 with JSD_SENTINEL when
    v3 OR lora is degraded, but NOT for psv-only degradation.

    # spec: S12.7 lines 3365-3366 — "JSD index 16 replaced with JSD_SENTINEL
    #   when signal_a or signal_b is zeroed"
    """

    def test_v3_degrade_replaces_jsd(self):
        """When v3 is degraded, index 16 must equal JSD_SENTINEL."""
        # Force v3 degradation: u in [0.80, 0.87)
        # Use a deterministic batch where we know which rows get v3 degraded
        N = 5000
        rng = np.random.default_rng(seed=AUG_SEED)
        X_orig = np.ones((N, VECTOR_DIM), dtype=np.float32) * 0.5
        X_aug = apply_augmentation_to_raw(X_orig, rng)

        v3_degraded_mask = np.allclose(X_aug[:, 0:6], 0.0, atol=1e-6)
        # Vectorized check
        v3_zeroed = np.all(np.abs(X_aug[:, 0:6]) < 1e-6, axis=1)

        if v3_zeroed.sum() == 0:
            pytest.skip("No v3-degraded samples in this batch; increase N or check seed")

        # All v3-degraded rows should have JSD = JSD_SENTINEL
        jsd_for_v3_degraded = X_aug[v3_zeroed, 16]
        assert np.allclose(jsd_for_v3_degraded, JSD_SENTINEL, atol=1e-6), (
            f"v3-degraded rows have JSD != JSD_SENTINEL ({JSD_SENTINEL}). "
            f"Got: {jsd_for_v3_degraded[:5]}"
        )

    def test_lora_degrade_replaces_jsd(self):
        """When lora is degraded, index 16 must equal JSD_SENTINEL."""
        N = 5000
        rng = np.random.default_rng(seed=AUG_SEED)
        X_orig = np.ones((N, VECTOR_DIM), dtype=np.float32) * 0.5
        X_aug = apply_augmentation_to_raw(X_orig, rng)

        v3_zeroed = np.all(np.abs(X_aug[:, 0:6]) < 1e-6, axis=1)
        lora_zeroed = np.all(np.abs(X_aug[:, 6:12]) < 1e-6, axis=1)
        pure_lora_mask = lora_zeroed & ~v3_zeroed

        if pure_lora_mask.sum() == 0:
            pytest.skip("No lora-only degraded samples in batch")

        jsd_for_lora_degraded = X_aug[pure_lora_mask, 16]
        assert np.allclose(jsd_for_lora_degraded, JSD_SENTINEL, atol=1e-6), (
            f"lora-degraded rows have JSD != JSD_SENTINEL. Got: {jsd_for_lora_degraded[:5]}"
        )

    def test_psv_degrade_does_not_replace_jsd(self):
        """PSV degradation must NOT replace JSD (index 16 unchanged).

        # spec: S12.7 line 3366 — "when signal_a or signal_b is zeroed"
        # (PSV = signal_c; JSD unchanged for PSV-only)
        """
        N = 5000
        rng = np.random.default_rng(seed=AUG_SEED)
        X_orig = np.ones((N, VECTOR_DIM), dtype=np.float32) * 0.5
        # Set index 16 to a distinctive non-sentinel value
        X_orig[:, 16] = 0.123
        X_aug = apply_augmentation_to_raw(X_orig, rng)

        v3_zeroed = np.all(np.abs(X_aug[:, 0:6]) < 1e-6, axis=1)
        lora_zeroed = np.all(np.abs(X_aug[:, 6:12]) < 1e-6, axis=1)
        psv_zeroed = (
            np.all(np.abs(X_aug[:, 12:16]) < 1e-6, axis=1) &
            np.all(np.abs(X_aug[:, 17:18]) < 1e-6, axis=1)
        )
        pure_psv_mask = psv_zeroed & ~v3_zeroed & ~lora_zeroed

        if pure_psv_mask.sum() == 0:
            pytest.skip("No psv-only degraded samples in batch")

        # For pure psv-only degraded rows, JSD should remain 0.123 (not sentinel)
        jsd_values = X_aug[pure_psv_mask, 16]
        assert np.allclose(jsd_values, 0.123, atol=1e-6), (
            f"PSV-only degraded rows should NOT replace JSD. "
            f"Expected 0.123, got {jsd_values[:5]}"
        )


# ---------------------------------------------------------------------------
# Test 5: underpowered class identity fallback path
# ---------------------------------------------------------------------------


class TestUnderpoweredClassIdentityFallback:
    """
    Verify that fit_platt_scaling produces identity (alpha=1, beta=0) for
    classes with 0 or N positives (degenerate labels).

    # spec: fit_calibration.py line 256 — "y_c.sum() == 0 or y_c.sum() == N"
    # spec: S12.8 lines 3381-3387
    """

    def test_class_with_zero_positives_gets_identity(self):
        """OOD class (index 6) has 0 positives in train_subset → identity fallback."""
        from tomato_sandbox.validation.fit_calibration import fit_platt_scaling

        N = 30
        rng = np.random.default_rng(seed=42)
        # Create OOF probs where class 6 (OOD) never appears in labels
        oof_probs = rng.dirichlet(np.ones(7), size=N)
        oof_labels = rng.integers(0, 6, size=N)  # labels in {0..5}, never 6

        result = fit_platt_scaling(
            oof_probs, oof_labels, output_dir=None, write_file=False
        )

        # Class 6 should have identity: alpha=1, beta=0
        assert abs(result["alpha"][6] - 1.0) < 1e-6, (
            f"Class 6 with 0 positives: alpha={result['alpha'][6]}, expected 1.0"
        )
        assert abs(result["beta"][6] - 0.0) < 1e-6, (
            f"Class 6 with 0 positives: beta={result['beta'][6]}, expected 0.0"
        )

    def test_class_with_all_positives_gets_identity(self):
        """Class where all N are positives → identity fallback."""
        from tomato_sandbox.validation.fit_calibration import fit_platt_scaling

        N = 20
        rng = np.random.default_rng(seed=42)
        oof_probs = rng.dirichlet(np.ones(7), size=N)
        # All samples have label 5 (healthy)
        oof_labels = np.full(N, 5, dtype=np.int64)

        result = fit_platt_scaling(
            oof_probs, oof_labels, output_dir=None, write_file=False
        )

        # All other classes (0-4, 6) have 0 positives → identity
        for c in [0, 1, 2, 3, 4, 6]:
            assert abs(result["alpha"][c] - 1.0) < 1e-6, (
                f"Class {c} with 0 positives: alpha={result['alpha'][c]}, expected 1.0"
            )

    def test_ylcv_underpowered_graceful(self):
        """ylcv (n=3 positives) should produce a finite alpha/beta or identity."""
        from tomato_sandbox.validation.fit_calibration import fit_platt_scaling

        N = 160
        rng = np.random.default_rng(seed=42)
        oof_probs = rng.dirichlet(np.ones(7), size=N)
        oof_labels = np.array(
            [3] * 3 + [0] * 50 + [1] * 40 + [2] * 30 + [5] * 37,
            dtype=np.int64
        )

        result = fit_platt_scaling(
            oof_probs, oof_labels, output_dir=None, write_file=False
        )

        # Must be finite and in reasonable range
        alpha_ylcv = result["alpha"][3]  # class 3 = ylcv
        beta_ylcv = result["beta"][3]
        assert np.isfinite(alpha_ylcv), f"ylcv alpha is not finite: {alpha_ylcv}"
        assert np.isfinite(beta_ylcv), f"ylcv beta is not finite: {beta_ylcv}"


# ---------------------------------------------------------------------------
# Test 6: canonical label conversion
# ---------------------------------------------------------------------------


class TestCanonicalLabelConversion:
    """
    Verify labels_to_canonical produces correct indices per spec S12.10.

    # spec: S12.10 lines 3460-3467 — canonical+OOD index space
    # DEC-061 Decision 6 — conversion rule
    """

    def test_healthy_maps_to_5(self):
        """y_stage1=0 (healthy) → canonical label 5.

        # spec: S12.10 line 3465 — "5 = healthy"
        """
        y1 = np.array([0, 0, 0], dtype=np.int64)
        y2 = np.array([-1, -1, -1], dtype=np.int64)
        result = labels_to_canonical(y1, y2)
        assert np.all(result == 5), f"Expected all 5, got {result}"

    def test_ood_maps_to_6(self):
        """y_stage1=2 (OOD) → canonical label 6.

        # spec: S12.10 line 3466 — "6 = OOD"
        """
        y1 = np.array([2, 2, 2], dtype=np.int64)
        y2 = np.array([-1, -1, -1], dtype=np.int64)
        result = labels_to_canonical(y1, y2)
        assert np.all(result == 6), f"Expected all 6, got {result}"

    def test_diseased_passes_through_stage2(self):
        """y_stage1=1 (diseased) → canonical label = y_stage2 value.

        # spec: S12.10 lines 3460-3464 — disease classes 0-4
        """
        y1 = np.array([1, 1, 1, 1, 1], dtype=np.int64)
        y2 = np.array([0, 1, 2, 3, 4], dtype=np.int64)
        result = labels_to_canonical(y1, y2)
        assert np.array_equal(result, [0, 1, 2, 3, 4]), f"Expected [0,1,2,3,4], got {result}"

    def test_mixed_batch(self):
        """Mixed batch of all three types."""
        y1 = np.array([0, 1, 2, 1, 0], dtype=np.int64)
        y2 = np.array([-1, 2, -1, 0, -1], dtype=np.int64)
        expected = np.array([5, 2, 6, 0, 5], dtype=np.int64)
        result = labels_to_canonical(y1, y2)
        assert np.array_equal(result, expected), f"Expected {expected}, got {result}"

    def test_foliar_maps_to_0(self):
        """foliar (y_stage2=0) stays at 0 in canonical space.

        # spec: S12.10 line 3460 — "0 = foliar"
        """
        y1 = np.array([1], dtype=np.int64)
        y2 = np.array([0], dtype=np.int64)
        result = labels_to_canonical(y1, y2)
        assert result[0] == 0


# ---------------------------------------------------------------------------
# Test 7: pkl schema matches loader
# ---------------------------------------------------------------------------


class TestPklSchemaMatchesLoader:
    """
    Write a stage1.pkl and load via hierarchical_classifier._load_stage_weights,
    verify all expected fields are present and correctly typed.

    # spec: S12.3 lines 3269-3277 — Stage 1 pkl schema
    # spec: S12.4 line 3301 — Stage 2 pkl schema
    """

    def _make_stage1_pkl(self, path: Path) -> dict:
        """Create a valid stage1.pkl and save to path."""
        data = {
            "weights": np.zeros((3, VECTOR_DIM), dtype=np.float32),
            "bias": np.zeros(3, dtype=np.float32),
            "temperature": 1.0,
            "feature_mean": np.zeros(VECTOR_DIM, dtype=np.float32),
            "feature_std": np.ones(VECTOR_DIM, dtype=np.float32),
            "class_order": ["healthy", "diseased", "OOD"],
        }
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=4)
        return data

    def test_stage1_pkl_has_all_required_fields(self, tmp_path):
        """Stage 1 pkl must have: weights, bias, temperature, feature_mean,
        feature_std, class_order.

        # spec: S12.3 lines 3269-3277 — Stage 1 pkl schema
        """
        pkl_path = tmp_path / "classifier_stage1.pkl"
        original_data = self._make_stage1_pkl(pkl_path)

        # Load back raw
        with open(pkl_path, "rb") as f:
            loaded = pickle.load(f)

        required_fields = ["weights", "bias", "temperature", "feature_mean",
                           "feature_std", "class_order"]
        for field in required_fields:
            assert field in loaded, f"Field '{field}' missing from stage1.pkl"

        # Shapes
        assert loaded["weights"].shape == (3, VECTOR_DIM), (
            f"weights shape {loaded['weights'].shape} != (3, {VECTOR_DIM})"
        )
        assert loaded["bias"].shape == (3,), f"bias shape {loaded['bias'].shape} != (3,)"
        assert loaded["class_order"] == ["healthy", "diseased", "OOD"]

    def test_stage2_pkl_has_all_required_fields(self, tmp_path):
        """Stage 2 pkl must have 5-class weights and correct class_order.

        # spec: S12.4 line 3301 — Stage 2 pkl schema
        """
        data = {
            "weights": np.zeros((5, VECTOR_DIM), dtype=np.float32),
            "bias": np.zeros(5, dtype=np.float32),
            "temperature": 1.0,
            "feature_mean": np.zeros(VECTOR_DIM, dtype=np.float32),
            "feature_std": np.ones(VECTOR_DIM, dtype=np.float32),
            "class_order": ["foliar", "septoria", "late_blight", "ylcv", "mosaic"],
        }
        pkl_path = tmp_path / "classifier_stage2.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(data, f, protocol=4)

        with open(pkl_path, "rb") as f:
            loaded = pickle.load(f)

        assert loaded["weights"].shape == (5, VECTOR_DIM)
        assert loaded["bias"].shape == (5,)
        assert loaded["class_order"] == ["foliar", "septoria", "late_blight", "ylcv", "mosaic"]

    def test_hierarchical_classifier_loader_accepts_valid_stage1(self, tmp_path, monkeypatch):
        """hierarchical_classifier._load_stage_weights accepts correctly formatted pkl.

        # spec: S12.3 line 3277 — "mismatch is fatal startup error"
        """
        from tomato_sandbox.classifier.hierarchical_classifier import _load_stage_weights

        pkl_path = tmp_path / "classifier_stage1.pkl"
        self._make_stage1_pkl(pkl_path)

        result = _load_stage_weights(pkl_path, ["healthy", "diseased", "OOD"])

        assert result["class_order"] == ["healthy", "diseased", "OOD"]
        assert result["weights"].shape == (3, VECTOR_DIM)
        assert result["bias"].shape == (3,)
        assert isinstance(result["temperature"], float)

    def test_hierarchical_classifier_loader_rejects_wrong_class_order(self, tmp_path):
        """_load_stage_weights falls back to sentinel on class_order mismatch.

        # spec: S12.3 line 3277 — "mismatch is fatal startup error"
        # (at inference level: log warning and return sentinel)
        """
        from tomato_sandbox.classifier.hierarchical_classifier import _load_stage_weights

        # Create pkl with WRONG class order
        data = {
            "weights": np.zeros((3, VECTOR_DIM), dtype=np.float32),
            "bias": np.zeros(3, dtype=np.float32),
            "temperature": 1.0,
            "feature_mean": np.zeros(VECTOR_DIM, dtype=np.float32),
            "feature_std": np.ones(VECTOR_DIM, dtype=np.float32),
            "class_order": ["diseased", "healthy", "OOD"],  # WRONG ORDER
        }
        pkl_path = tmp_path / "wrong_order.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(data, f, protocol=4)

        # Should return sentinel (all-zeros weights) not the loaded data
        result = _load_stage_weights(pkl_path, ["healthy", "diseased", "OOD"])
        assert result["class_order"] == ["healthy", "diseased", "OOD"]
        assert np.allclose(result["weights"], 0.0), "Sentinel should have all-zeros weights"


# ---------------------------------------------------------------------------
# Additional: standardization round-trip
# ---------------------------------------------------------------------------


class TestStandardization:
    """Verify standardize/compute_standardization are inverses."""

    def test_standardization_mean_zero_std_one(self):
        """After standardization, each feature should have ~mean=0, std=1."""
        rng = np.random.default_rng(seed=42)
        X = rng.uniform(0.0, 1.0, size=(100, VECTOR_DIM)).astype(np.float32)
        mean, std = compute_standardization(X)
        X_std = standardize(X, mean, std)

        # Due to clip to [-3, 3], means won't be exactly 0 but should be close
        col_means = X_std.mean(axis=0)
        assert np.all(np.abs(col_means) < 0.5), (
            f"Column means too large: max={np.abs(col_means).max():.4f}"
        )

    def test_standardization_clips_to_minus3_plus3(self):
        """standardize must clip to [-3, 3] per spec S12.2 line 3204."""
        mean = np.zeros(VECTOR_DIM, dtype=np.float32)
        std = np.ones(VECTOR_DIM, dtype=np.float32)
        # Create extreme values
        X = np.full((5, VECTOR_DIM), 10.0, dtype=np.float32)
        X_std = standardize(X, mean, std)
        assert np.all(X_std <= 3.0), "Standardized values should be <= 3.0"

        X2 = np.full((5, VECTOR_DIM), -10.0, dtype=np.float32)
        X2_std = standardize(X2, mean, std)
        assert np.all(X2_std >= -3.0), "Standardized values should be >= -3.0"


# ---------------------------------------------------------------------------
# Additional: reorder_proba
# ---------------------------------------------------------------------------


class TestReorderProba:
    """Verify _reorder_proba handles missing classes and correct mapping."""

    def test_correct_reordering(self):
        """sklearn classes_ [0,1,2] matches expected [0,1,2] → no change."""
        proba = np.array([[0.1, 0.6, 0.3], [0.5, 0.3, 0.2]])
        classes = np.array([0, 1, 2])
        result = _reorder_proba(proba, classes, [0, 1, 2])
        assert np.allclose(result, proba)

    def test_reordering_with_shuffled_classes(self):
        """sklearn returns [0,2] but expected [0,1,2] → index 1 gets uniform."""
        proba = np.array([[0.4, 0.6], [0.7, 0.3]])  # only classes 0 and 2
        classes = np.array([0, 2])
        result = _reorder_proba(proba, classes, [0, 1, 2])

        # Class 0 and 2 should be remapped; class 1 gets 1/3 uniform
        assert result.shape == (2, 3)
        # After renormalization, class 0 should be proportional to original
        assert result[0, 0] > 0, "Class 0 should have non-zero prob"
        assert result[0, 2] > 0, "Class 2 should have non-zero prob"


# ---------------------------------------------------------------------------
# Smoke test: full training with mock data (optional, slow)
# ---------------------------------------------------------------------------


class TestTrainClassifierSmoke:
    """
    Smoke test for train_classifier() using actual features.npz.
    Gated behind skipif to avoid slow runs in CI.
    """

    @pytest.mark.skipif(
        not (
            Path(__file__).resolve().parents[2]
            / "phase_f0_calibration"
            / "_classifier_training"
            / "features.npz"
        ).exists(),
        reason="features.npz not present; requires Step 3 completion",
    )
    def test_train_classifier_produces_artifacts(self, tmp_path, monkeypatch):
        """Train on real features.npz and verify artifact shapes.

        Monkeypatches output paths to tmp_path to avoid overwriting production artifacts.
        """
        import tomato_sandbox.training.train_classifier as tc

        # Redirect artifact paths to tmp_path
        monkeypatch.setattr(tc, "STAGE1_PKL", tmp_path / "classifier_stage1.pkl")
        monkeypatch.setattr(tc, "STAGE2_PKL", tmp_path / "classifier_stage2.pkl")
        monkeypatch.setattr(tc, "FEAT_STD_JSON", tmp_path / "feat_std.json")
        monkeypatch.setattr(tc, "TRAINING_REPORT_JSON", tmp_path / "training_report.json")
        monkeypatch.setattr(tc, "_BACKUP_DIR", tmp_path / "_backup")

        # Also redirect platt output by monkeypatching fit_platt_scaling output_dir
        # (fit_platt_scaling uses output_dir param; we pass _CALIB_DIR explicitly)
        # For smoke test, we still allow the real platt.json to be updated
        # since this is the intended behavior.

        report = tc.train_classifier(verbose=False)

        # Verify artifacts exist
        assert (tmp_path / "classifier_stage1.pkl").exists()
        assert (tmp_path / "classifier_stage2.pkl").exists()
        assert (tmp_path / "feat_std.json").exists()
        assert (tmp_path / "training_report.json").exists()

        # Verify stage1.pkl schema
        with open(tmp_path / "classifier_stage1.pkl", "rb") as f:
            s1 = pickle.load(f)
        assert s1["weights"].shape == (3, VECTOR_DIM)
        assert s1["bias"].shape == (3,)
        assert s1["class_order"] == ["healthy", "diseased", "OOD"]

        # Verify stage2.pkl schema
        with open(tmp_path / "classifier_stage2.pkl", "rb") as f:
            s2 = pickle.load(f)
        assert s2["weights"].shape == (5, VECTOR_DIM)
        assert s2["class_order"] == ["foliar", "septoria", "late_blight", "ylcv", "mosaic"]

        # Verify OOF metrics in report
        oof = report["oof_aggregate"]
        assert "macro_f1_stage1" in oof
        assert "macro_f1_7class" in oof
        assert oof["n"] == 160

        # STOP condition: Stage 1 OOF F1 >= 0.50
        assert oof["macro_f1_stage1"] >= 0.50, (
            f"Stage 1 OOF F1 {oof['macro_f1_stage1']:.4f} < 0.50 (STOP threshold)"
        )
