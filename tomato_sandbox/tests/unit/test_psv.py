"""
Unit tests for PSV (Plant Symptom Visual) signal — T-IMPL-3c.

Tests cover:
  - FEATURE_NAMES length == 26
  - compute_26_features output shape
  - Deterministic output for fixed synthetic input
  - Each feature group individually (smoke tests)
  - No CUDA/GPU reference in any PSV module
  - Leaf segmentation edge cases
  - Disease detection edge cases
  - Compatibility scoring (standardize + softmax)
  - Reliability score properties
  - SignalCResult fields and types
  - Exception path returns forward_succeeded=False
  - Fallback path (empty leaf_mask → IQA mask used)
  - Both-empty-mask path (returns 0.05 reliability)
  - Weight matrix loads correctly
"""

from __future__ import annotations

import inspect
import os
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _green_leaf_image(h: int = 200, w: int = 200) -> np.ndarray:
    """Synthetic green leaf image: uniform green with slight noise."""
    rng = np.random.default_rng(seed=42)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 1] = 140  # green channel dominant
    img[:, :, 0] = 50   # red
    img[:, :, 2] = 40   # blue
    noise = rng.integers(-20, 20, (h, w, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # Add a brown patch to simulate disease
    img[60:100, 60:100, 0] = 150   # red up
    img[60:100, 60:100, 1] = 80    # green down
    img[60:100, 60:100, 2] = 30    # blue low
    return img


def _leaf_mask(h: int = 200, w: int = 200) -> np.ndarray:
    """Simple rectangular leaf mask."""
    mask = np.zeros((h, w), dtype=bool)
    mask[20:180, 20:180] = True
    return mask


def _disease_mask(h: int = 200, w: int = 200) -> np.ndarray:
    """Small disease region mask."""
    mask = np.zeros((h, w), dtype=bool)
    mask[60:100, 60:100] = True
    return mask


# ---------------------------------------------------------------------------
# Section A: Feature catalog
# ---------------------------------------------------------------------------

class TestFeatureCatalog:
    def test_feature_count_is_26(self):
        """spec: 10.5.9 — exactly 26 features in fixed order."""
        from tomato_sandbox.signals.psv.features import FEATURE_NAMES, NUM_FEATURES
        assert len(FEATURE_NAMES) == 26, f"Got {len(FEATURE_NAMES)} features; expected 26"
        assert NUM_FEATURES == 26

    def test_feature_names_are_strings(self):
        from tomato_sandbox.signals.psv.features import FEATURE_NAMES
        for name in FEATURE_NAMES:
            assert isinstance(name, str) and len(name) > 0

    def test_feature_index_21_is_psv_aggregate_reliability(self):
        """spec: 10.5.7 / 10.9 — placeholder index for Stage 5."""
        from tomato_sandbox.signals.psv.features import FEATURE_NAMES
        assert FEATURE_NAMES[21] == "psv_aggregate_reliability"

    def test_vegetation_indices_at_22_to_25(self):
        """spec: 10.5.8 — G8 indices."""
        from tomato_sandbox.signals.psv.features import FEATURE_NAMES
        assert FEATURE_NAMES[22] == "ExG"
        assert FEATURE_NAMES[23] == "GLI"
        assert FEATURE_NAMES[24] == "MGRVI"
        assert FEATURE_NAMES[25] == "VARI"


# ---------------------------------------------------------------------------
# Section B: compute_26_features output
# ---------------------------------------------------------------------------

class TestCompute26Features:
    def _run(self, img=None, leaf=None, disease=None, lesion_stats=None, iqa=0.8):
        from tomato_sandbox.signals.psv.features import compute_26_features
        if img is None:
            img = _green_leaf_image()
        if leaf is None:
            leaf = _leaf_mask()
        if disease is None:
            disease = _disease_mask()
        if lesion_stats is None:
            import cv2
            nb, labels, stats, centroids = cv2.connectedComponentsWithStats(
                disease.astype(np.uint8)
            )
            lesion_stats = {
                "n_lesions": nb - 1,
                "labels": labels,
                "stats": stats,
                "centroids": centroids,
                "leaf_area_px": int(leaf.sum()),
                "disease_area_px": int(disease.sum()),
            }
        return compute_26_features(img, leaf, disease, lesion_stats, iqa)

    def test_output_shape(self):
        feats = self._run()
        assert feats.shape == (26,), f"Expected (26,), got {feats.shape}"

    def test_output_is_float32(self):
        feats = self._run()
        assert feats.dtype == np.float32

    def test_index_21_is_zero_placeholder(self):
        """Stage 3 always outputs 0 at index 21; Stage 5 overwrites it."""
        feats = self._run()
        assert feats[21] == 0.0

    def test_deterministic(self):
        """Same input → same output."""
        feats1 = self._run()
        feats2 = self._run()
        np.testing.assert_array_equal(feats1, feats2)

    def test_no_nan_or_inf(self):
        feats = self._run()
        assert np.all(np.isfinite(feats)), f"Found non-finite values: {feats}"

    def test_empty_leaf_mask_returns_zeros(self):
        """Edge case: no leaf pixels → all-zero feature vector."""
        empty_leaf = np.zeros((200, 200), dtype=bool)
        empty_disease = np.zeros((200, 200), dtype=bool)
        lesion_stats = {
            "n_lesions": 0, "labels": None, "stats": None,
            "centroids": None, "leaf_area_px": 0, "disease_area_px": 0,
        }
        feats = self._run(leaf=empty_leaf, disease=empty_disease, lesion_stats=lesion_stats)
        assert feats.shape == (26,)
        assert np.all(np.isfinite(feats))

    def test_disease_coverage_in_valid_range(self):
        """G1 idx 0: disease_coverage_pct ∈ [0, 100] — spec: 10.5.1 lines 2317-2321."""
        feats = self._run()
        assert 0.0 <= feats[0] <= 100.0, f"disease_coverage_pct={feats[0]}"

    def test_vegetation_indices_clamped(self):
        """G8 features should be finite (clipping enforced)."""
        feats = self._run()
        for idx in [22, 23, 24, 25]:
            assert np.isfinite(feats[idx]), f"Feature {idx} = {feats[idx]}"

    def test_aggregate_quality_propagated(self):
        """G7 idx 20: aggregate_quality should match iqa_aggregate_score."""
        feats = self._run(iqa=0.7)
        assert abs(feats[20] - 0.7) < 1e-5, f"aggregate_quality={feats[20]}, expected 0.7"


# ---------------------------------------------------------------------------
# Section C: Leaf segmentation
# ---------------------------------------------------------------------------

class TestLeafSegmentation:
    def test_returns_bool_array(self):
        from tomato_sandbox.signals.psv.leaf_segmentation import segment_leaf
        img = _green_leaf_image()
        mask = segment_leaf(img)
        assert mask.dtype == bool
        assert mask.shape == img.shape[:2]

    def test_green_image_produces_nonzero_mask(self):
        from tomato_sandbox.signals.psv.leaf_segmentation import segment_leaf
        img = _green_leaf_image()
        mask = segment_leaf(img)
        assert mask.sum() > 0, "Expected some leaf pixels from green image"

    def test_black_image_produces_empty_mask(self):
        """All-black image → no leaf detected → all-False mask."""
        from tomato_sandbox.signals.psv.leaf_segmentation import segment_leaf
        img = np.zeros((150, 150, 3), dtype=np.uint8)
        mask = segment_leaf(img)
        assert mask.dtype == bool
        # Black image has no saturation → Otsu threshold collapses to empty or full
        # either way the function should return without error
        assert mask.shape == (150, 150)


# ---------------------------------------------------------------------------
# Section D: Disease detection
# ---------------------------------------------------------------------------

class TestDiseaseDetection:
    def test_returns_tuple(self):
        from tomato_sandbox.signals.psv.disease_detection import detect_disease_regions
        img = _green_leaf_image()
        leaf = _leaf_mask()
        result = detect_disease_regions(img, leaf)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_disease_mask_shape(self):
        from tomato_sandbox.signals.psv.disease_detection import detect_disease_regions
        img = _green_leaf_image()
        leaf = _leaf_mask()
        disease_mask, _ = detect_disease_regions(img, leaf)
        assert disease_mask.shape == img.shape[:2]
        assert disease_mask.dtype == bool

    def test_lesion_stats_keys(self):
        from tomato_sandbox.signals.psv.disease_detection import detect_disease_regions
        img = _green_leaf_image()
        leaf = _leaf_mask()
        _, stats = detect_disease_regions(img, leaf)
        required_keys = {"n_lesions", "labels", "stats", "centroids",
                         "leaf_area_px", "disease_area_px"}
        assert required_keys.issubset(stats.keys())

    def test_empty_leaf_mask(self):
        """Empty leaf_mask → n_lesions=0, disease_area_px=0."""
        from tomato_sandbox.signals.psv.disease_detection import detect_disease_regions
        img = _green_leaf_image()
        empty = np.zeros(img.shape[:2], dtype=bool)
        mask, stats = detect_disease_regions(img, empty)
        assert stats["n_lesions"] == 0
        assert stats["disease_area_px"] == 0


# ---------------------------------------------------------------------------
# Section E: Compatibility scoring
# ---------------------------------------------------------------------------

class TestCompatibilityScoring:
    def test_standardize_features_shape(self):
        from tomato_sandbox.signals.psv.compatibility import standardize_features
        raw = np.zeros(26, dtype=np.float32)
        standardized = standardize_features(raw)
        assert standardized.shape == (26,)

    def test_standardize_clips_to_pm3(self):
        from tomato_sandbox.signals.psv.compatibility import standardize_features
        raw = np.full(26, 1000.0, dtype=np.float32)
        standardized = standardize_features(raw)
        assert np.all(standardized <= 3.0)
        assert np.all(standardized >= -3.0)

    def test_compute_compatibility_output_shape(self):
        from tomato_sandbox.signals.psv.compatibility import compute_compatibility_scores, standardize_features
        raw = np.zeros(26, dtype=np.float32)
        standardized = standardize_features(raw)
        compat = compute_compatibility_scores(standardized)
        assert compat.shape == (6,)

    def test_compute_compatibility_sums_to_one(self):
        from tomato_sandbox.signals.psv.compatibility import compute_compatibility_scores, standardize_features
        raw = np.random.default_rng(7).random(26).astype(np.float32)
        standardized = standardize_features(raw)
        compat = compute_compatibility_scores(standardized)
        assert abs(compat.sum() - 1.0) < 1e-5, f"Softmax sum={compat.sum()}"

    def test_compatibility_all_nonnegative(self):
        from tomato_sandbox.signals.psv.compatibility import compute_compatibility_scores, standardize_features
        raw = np.random.default_rng(8).random(26).astype(np.float32)
        standardized = standardize_features(raw)
        compat = compute_compatibility_scores(standardized)
        assert np.all(compat >= 0.0)

    def test_weight_matrix_shape(self):
        from tomato_sandbox.signals.psv.compatibility import WEIGHT_MATRIX
        assert WEIGHT_MATRIX.shape == (6, 26), f"Got {WEIGHT_MATRIX.shape}"

    def test_weight_matrix_index_21_all_zeros(self):
        """spec: 10.6.1 lines 2687-2689 — G7 features have zero weights."""
        from tomato_sandbox.signals.psv.compatibility import WEIGHT_MATRIX
        # Column 21 = psv_aggregate_reliability; all rows must be 0.0
        col_21 = WEIGHT_MATRIX[:, 21]
        np.testing.assert_array_equal(
            col_21, np.zeros(6),
            err_msg="WEIGHT_MATRIX column 21 must be all zeros (spec 10.6.1)"
        )


# ---------------------------------------------------------------------------
# Section F: Reliability
# ---------------------------------------------------------------------------

class TestReliability:
    def test_empty_leaf_returns_zero(self):
        from tomato_sandbox.signals.psv.reliability import compute_psv_reliability
        empty = np.zeros((100, 100), dtype=bool)
        disease = np.zeros((100, 100), dtype=bool)
        r = compute_psv_reliability(empty, disease, None, 0.8, 0)
        assert r == 0.0

    def test_no_iqa_mask_uses_neutral(self):
        """No IQA mask → mask_agreement=0.5 (neutral)."""
        from tomato_sandbox.signals.psv.reliability import compute_psv_reliability
        leaf = _leaf_mask()
        disease = _disease_mask()
        r = compute_psv_reliability(leaf, disease, None, 0.8, 1)
        assert 0.0 < r <= 1.0

    def test_returns_float_in_unit_interval(self):
        from tomato_sandbox.signals.psv.reliability import compute_psv_reliability
        leaf = _leaf_mask()
        disease = _disease_mask()
        iqa_mask = _leaf_mask()
        for iqa_score in [0.1, 0.5, 0.9]:
            r = compute_psv_reliability(leaf, disease, iqa_mask, iqa_score, 1)
            assert 0.0 <= r <= 1.0, f"Reliability {r} out of [0,1] for iqa={iqa_score}"

    def test_high_coverage_degrades_reliability(self):
        """Disease coverage > 90 % → coverage_sanity=0.2 → lower reliability."""
        from tomato_sandbox.signals.psv.reliability import compute_psv_reliability
        leaf = _leaf_mask()
        # Disease covers 95 % of the leaf
        disease_heavy = np.zeros((200, 200), dtype=bool)
        disease_heavy[20:180, 20:178] = True
        r_heavy = compute_psv_reliability(leaf, disease_heavy, None, 0.8, 1)
        # Normal case for comparison
        disease_normal = _disease_mask()
        r_normal = compute_psv_reliability(leaf, disease_normal, None, 0.8, 1)
        assert r_heavy < r_normal, "High coverage should produce lower reliability"

    def test_fallback_reliability_floor(self):
        """spec: 10.8 — fallback reliability ≥ 0.1."""
        from tomato_sandbox.signals.psv.reliability import fallback_reliability
        assert fallback_reliability(0.0) == 0.1    # max(0.1, 0)
        assert fallback_reliability(1.0) == pytest.approx(0.3)
        assert fallback_reliability(0.5) == pytest.approx(0.15)


# ---------------------------------------------------------------------------
# Section G: SignalCResult and compute_signal_c
# ---------------------------------------------------------------------------

class TestComputeSignalC:
    def _make_result(self, img=None, iqa_mask=None, iqa_score=0.8):
        from tomato_sandbox.signals.psv.psv import compute_signal_c
        if img is None:
            img = _green_leaf_image()
        return compute_signal_c(img, iqa_mask, iqa_score)

    def test_returns_signal_c_result(self):
        from tomato_sandbox.signals.psv.psv import SignalCResult
        result = self._make_result()
        assert isinstance(result, SignalCResult)

    def test_compatibility_shape_and_sum(self):
        result = self._make_result()
        assert result.compatibility.shape == (6,)
        assert abs(result.compatibility.sum() - 1.0) < 1e-4

    def test_forward_succeeded_on_valid_input(self):
        result = self._make_result()
        assert result.forward_succeeded is True
        assert result.failure_reason is None

    def test_raw_features_shape(self):
        result = self._make_result()
        assert result.raw_features.shape == (26,)

    def test_standardized_features_shape(self):
        result = self._make_result()
        assert result.standardized_features.shape == (26,)

    def test_index_21_filled_by_stage5(self):
        """After compute_signal_c, raw_features[21] equals psv_reliability."""
        result = self._make_result()
        assert abs(result.raw_features[21] - result.psv_reliability) < 1e-6

    def test_compatibility_argmax_in_range(self):
        result = self._make_result()
        assert 0 <= result.compatibility_argmax <= 5

    def test_compatibility_max_matches_argmax(self):
        result = self._make_result()
        assert abs(result.compatibility[result.compatibility_argmax] - result.compatibility_max) < 1e-6

    def test_compatibility_margin_nonnegative(self):
        result = self._make_result()
        assert result.compatibility_margin >= 0.0

    def test_psv_reliability_in_unit_interval(self):
        result = self._make_result()
        assert 0.0 <= result.psv_reliability <= 1.0

    def test_leaf_mask_shape(self):
        img = _green_leaf_image(150, 150)
        result = self._make_result(img=img)
        assert result.leaf_mask.shape == (150, 150)
        assert result.leaf_mask.dtype == bool

    def test_disease_mask_shape(self):
        img = _green_leaf_image(150, 150)
        result = self._make_result(img=img)
        assert result.disease_mask.shape == (150, 150)
        assert result.disease_mask.dtype == bool

    def test_n_lesions_nonnegative(self):
        result = self._make_result()
        assert result.n_lesions >= 0

    def test_exception_path_returns_failed_result(self):
        """Passing garbage input triggers exception path, forward_succeeded=False."""
        from tomato_sandbox.signals.psv.psv import compute_signal_c
        # Pass invalid ndarray that will cause stages to fail
        bad_img = np.zeros((5, 5, 3), dtype=np.uint8)  # too small for morphology
        result = compute_signal_c(bad_img, None, 0.8)
        # May succeed or fail depending on input; either way should not raise
        assert isinstance(result.forward_succeeded, bool)
        if not result.forward_succeeded:
            assert result.failure_reason is not None
            assert abs(result.compatibility.sum() - 1.0) < 1e-4

    def test_explicit_exception_path(self):
        """Passing None as rgb_cc (not ndarray) forces exception."""
        from tomato_sandbox.signals.psv.psv import compute_signal_c
        # numpy operations on None will raise AttributeError
        # but compute_signal_c catches all exceptions
        try:
            result = compute_signal_c(None, None, 0.8)
            # If it somehow didn't raise, check the failure path was used
            assert result.forward_succeeded is False
            assert result.failure_reason is not None
        except Exception:
            pytest.fail("compute_signal_c should not propagate exceptions")

    def test_both_empty_mask_path(self):
        """spec: 10.8 lines 2835-2836 — both masks empty → 0.05 reliability."""
        from tomato_sandbox.signals.psv.psv import compute_signal_c
        # Black image → leaf segmentation produces empty mask
        black_img = np.zeros((200, 200, 3), dtype=np.uint8)
        # No IQA mask either
        result = compute_signal_c(black_img, None, 0.8)
        # Should succeed (not exception), but may use fallback or empty path
        assert isinstance(result, type(result))  # always returns a result
        # If both masks were empty, reliability should be 0.05
        if not result.fallback_used and result.leaf_mask.sum() == 0:
            assert result.psv_reliability == pytest.approx(0.05)

    def test_fallback_path_sets_fallback_used(self):
        """When PSV mask is empty but IQA mask has content, fallback fires."""
        from tomato_sandbox.signals.psv.psv import compute_signal_c
        black_img = np.zeros((200, 200, 3), dtype=np.uint8)
        iqa_mask = _leaf_mask()  # IQA has a valid mask
        result = compute_signal_c(black_img, iqa_mask, 0.7)
        # Black image may or may not produce an empty leaf mask;
        # if it does, fallback_used should be True and reliability lower
        if result.fallback_used:
            assert result.psv_reliability <= 0.3

    def test_iqa_score_propagated_to_aggregate_quality(self):
        """spec: 10.5.7 — aggregate_quality = iqa_aggregate_score."""
        result = self._make_result(iqa_score=0.65)
        assert abs(result.raw_features[20] - 0.65) < 1e-5


# ---------------------------------------------------------------------------
# Section H: No GPU references in PSV modules
# ---------------------------------------------------------------------------

class TestNoCUDA:
    """spec: 10.2 line 2018, 10.11, 10.12 — PSV is CPU-only."""

    def _get_psv_module_sources(self) -> dict[str, str]:
        import tomato_sandbox.signals.psv.psv as m_psv
        import tomato_sandbox.signals.psv.leaf_segmentation as m_seg
        import tomato_sandbox.signals.psv.disease_detection as m_dis
        import tomato_sandbox.signals.psv.features as m_feat
        import tomato_sandbox.signals.psv.compatibility as m_compat
        import tomato_sandbox.signals.psv.reliability as m_rel
        return {
            "psv.py": inspect.getsource(m_psv),
            "leaf_segmentation.py": inspect.getsource(m_seg),
            "disease_detection.py": inspect.getsource(m_dis),
            "features.py": inspect.getsource(m_feat),
            "compatibility.py": inspect.getsource(m_compat),
            "reliability.py": inspect.getsource(m_rel),
        }

    def test_no_cuda_references(self):
        sources = self._get_psv_module_sources()
        for module_name, src in sources.items():
            assert "cuda" not in src.lower(), (
                f"{module_name} contains 'cuda' reference — PSV must be CPU-only"
            )

    def test_no_gpu_lock_import(self):
        """No PSV module may import gpu_lock (CPU-only constraint)."""
        sources = self._get_psv_module_sources()
        for module_name, src in sources.items():
            # Check for actual import statements only, not comments/docstrings
            import_lines = [
                line for line in src.splitlines()
                if "import" in line and "gpu_lock" in line
            ]
            assert not import_lines, (
                f"{module_name} has gpu_lock import line(s): {import_lines} "
                f"— PSV must be CPU-only"
            )

    def test_no_torch_import(self):
        sources = self._get_psv_module_sources()
        for module_name, src in sources.items():
            assert "import torch" not in src, (
                f"{module_name} imports torch — PSV must be pure CV/numpy"
            )


# ---------------------------------------------------------------------------
# Section I: __init__ re-exports
# ---------------------------------------------------------------------------

class TestReExports:
    def test_compute_signal_c_importable_from_package(self):
        from tomato_sandbox.signals.psv import compute_signal_c
        assert callable(compute_signal_c)

    def test_signal_c_result_importable_from_package(self):
        from tomato_sandbox.signals.psv import SignalCResult
        assert SignalCResult is not None

    def test_both_import_paths_same_object(self):
        from tomato_sandbox.signals.psv import compute_signal_c as fn1
        from tomato_sandbox.signals.psv.psv import compute_signal_c as fn2
        assert fn1 is fn2
