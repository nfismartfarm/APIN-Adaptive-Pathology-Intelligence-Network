"""
Unit tests for TTA controller (tomato_sandbox/signals/tta.py).

Coverage:
  1.  should_trigger_tta: 1 returned at/above trigger_threshold (0.55)
  2.  should_trigger_tta: 2 returned in [escalate, trigger) band
  3.  should_trigger_tta: 5 returned below escalate_threshold (0.45)
  4.  should_trigger_tta: boundary at exactly 0.55 → 1
  5.  should_trigger_tta: boundary at exactly 0.45 → 2 (not 5)
  6.  should_trigger_tta: just below 0.45 → 5
  7.  should_trigger_tta: NaN → 1 (no TTA)
  8.  should_trigger_tta: +inf / -inf → 1
  9.  build_augmentations: n_views=1 → empty list
  10. build_augmentations: n_views=2 → exactly [("hflip",)]
  11. build_augmentations: n_views=5 → 4 specs in correct order
  12. apply_augmentation: hflip is its own inverse (double-flip = identity)
  13. apply_augmentation: rotate produces different pixels from original
  14. apply_augmentation: brightness>1 increases mean pixel value
  15. aggregate_views: mean computed correctly over two views
  16. aggregate_views: all-failed views → zeros, n_used=0
  17. aggregate_views: one failed view excluded from mean
  18. jensen_shannon_divergence: identical distributions → 0.0
  19. jensen_shannon_divergence: perfectly opposed → ≈ log(2)
  20. PSV NOT invoked during TTA (mock compute_signal_c; assert call_count==0)
  21. Signal B single-pass: compute_signal_b called once per view (2-view TTA → 2 calls)
  22. apply_tta n_views=1: returns result without augmentation
  23. apply_tta n_views=2: correct number of views in TTAReport
  24. apply_tta n_views=5: runs 5 views (1 original + 4 augmented)
  25. apply_tta all-failed views: returns SignalAResult.forward_succeeded=False
  26. TTAReport fields correct for 2-view TTA
  27. view_disagreement: 0.0 when all views agree
  28. view_disagreement: excluded failed views
  29. Flat-path shim import: from tomato_sandbox.tta import should_trigger_tta

29 tests total.
"""

from __future__ import annotations

import math
from unittest.mock import patch, MagicMock
from typing import Optional

import numpy as np
import pytest

try:
    from PIL import Image as _PIL_Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
from tomato_sandbox.signals.tta import (
    TTAReport,
    should_trigger_tta,
    build_augmentations,
    apply_augmentation,
    aggregate_views,
    jensen_shannon_divergence,
    apply_tta,
    _compute_view_disagreement,
    _failed_signal_a,
    _failed_signal_b,
)
from tomato_sandbox.utils.nan_guards import TTA_TRIGGER_THRESHOLD, TTA_ESCALATE_THRESHOLD
from tomato_sandbox.signals.v3_signal import SignalAResult
from tomato_sandbox.signals.lora_signal import SignalBResult


# ---------------------------------------------------------------------------
# Mock model classes — proper callables (not MagicMock.__call__ which is bypassed
# by Python's type-based dunder dispatch).
# ---------------------------------------------------------------------------

class _MockV3Model:
    """Callable mock for the v3 10-class model.

    signal_a_forward calls: model(x, crop_mode=..., domain_labels=None)
    Returns: {"logits": Tensor[B, 10]}
    """

    def __init__(self, probs_6d: Optional[np.ndarray] = None):
        if probs_6d is None:
            probs_6d = np.array([0.4, 0.1, 0.2, 0.1, 0.1, 0.1], dtype=np.float32)
        # Build [10]-class probs (6 tomato + 4 chilli)
        probs_10d = np.zeros(10, dtype=np.float32)
        probs_10d[:6] = probs_6d * 0.9
        probs_10d[6] = 0.1
        # Normalise to sum to 1.0
        probs_10d = probs_10d / probs_10d.sum()
        # Use log(probs) as logits (softmax recovers probs)
        self._logits_1d = np.log(probs_10d + 1e-8).astype(np.float32)
        self.call_count = 0

    def eval(self):
        return self

    def __call__(self, x, crop_mode=None, domain_labels=None):
        self.call_count += 1
        b = x.shape[0]
        logits = torch.from_numpy(
            np.tile(self._logits_1d, (b, 1))
        )
        return {"logits": logits}


class _MockLoRAModel:
    """Callable mock for the LoRA 6-class model.

    signal_b_forward calls: model(x)
    Returns: {"logits": Tensor[B, 6], "cls_token": Tensor[B, 768]}
    """

    def __init__(self, probs_6d: Optional[np.ndarray] = None):
        if probs_6d is None:
            probs_6d = np.array([0.3, 0.2, 0.2, 0.1, 0.1, 0.1], dtype=np.float32)
        probs_6d = np.asarray(probs_6d, dtype=np.float32)
        probs_6d = probs_6d / probs_6d.sum()
        self._logits_1d = np.log(probs_6d + 1e-8).astype(np.float32)
        self.call_count = 0

    def eval(self):
        return self

    def __call__(self, x):
        self.call_count += 1
        b = x.shape[0]
        logits = torch.from_numpy(np.tile(self._logits_1d, (b, 1)))
        cls_token = torch.zeros(b, 768, dtype=torch.float32)
        return {"logits": logits, "cls_token": cls_token}


class _AlwaysFailingModel:
    """A model that always raises RuntimeError on __call__."""

    def eval(self):
        return self

    def __call__(self, *args, **kwargs):
        raise RuntimeError("simulated model failure")


# ---------------------------------------------------------------------------
# PIL image helper
# ---------------------------------------------------------------------------

def _make_pil_image(width: int = 64, height: int = 64) -> "_PIL_Image.Image":
    """Create a textured PIL RGB image that passes blur check."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:, :, 1] = 120  # green channel
    arr[:, :, 0] = 30
    arr[:, :, 2] = 30
    rng = np.random.default_rng(42)
    noise = rng.integers(-30, 30, arr.shape)
    arr = np.clip(arr.astype(np.int32) + noise, 0, 255).astype(np.uint8)
    return _PIL_Image.fromarray(arr, mode="RGB")


# ---------------------------------------------------------------------------
# 1-8: should_trigger_tta
# ---------------------------------------------------------------------------

class TestShouldTriggerTTA:
    """Threshold boundary tests per spec 11.2 lines 2932-2951."""

    def test_above_trigger_threshold_returns_1(self):
        # spec: section 11.2 lines 2932-2933 — "combined_max_prob >= 0.55 → no TTA"
        assert should_trigger_tta(0.80) == 1

    def test_in_band_returns_2(self):
        # spec: section 11.2 lines 2935-2936 — "[0.45, 0.55) → 2-view TTA"
        assert should_trigger_tta(0.50) == 2

    def test_below_escalate_threshold_returns_5(self):
        # spec: section 11.2 lines 2938-2939 — "< 0.45 → 5-view TTA"
        assert should_trigger_tta(0.30) == 5

    def test_exactly_at_trigger_threshold_returns_1(self):
        # spec: section 11.2 line 2932 — ">= TRIGGER_THRESHOLD (0.55) → 1"
        assert should_trigger_tta(TTA_TRIGGER_THRESHOLD) == 1

    def test_just_below_trigger_threshold_returns_2(self):
        # spec: section 11.2 lines 2935-2936 — band is [escalate, trigger)
        assert should_trigger_tta(TTA_TRIGGER_THRESHOLD - 1e-9) == 2

    def test_exactly_at_escalate_threshold_returns_2(self):
        # spec: section 11.2 line 2935 — "ESCALATE_THRESHOLD <= ... < TRIGGER → 2"
        # boundary: exactly 0.45 → 2, not 5
        assert should_trigger_tta(TTA_ESCALATE_THRESHOLD) == 2

    def test_just_below_escalate_threshold_returns_5(self):
        # spec: section 11.2 line 2938 — "< ESCALATE_THRESHOLD → 5-view"
        assert should_trigger_tta(TTA_ESCALATE_THRESHOLD - 1e-9) == 5

    def test_nan_returns_1(self):
        # spec: section 11.2 lines 2946-2951 — "NaN → n_views = 1 (no TTA)"
        assert should_trigger_tta(float("nan")) == 1

    def test_pos_inf_returns_1(self):
        # spec: section 11.2 lines 2946-2951 — "not np.isfinite → n_views = 1"
        assert should_trigger_tta(float("inf")) == 1

    def test_neg_inf_returns_1(self):
        assert should_trigger_tta(float("-inf")) == 1


# ---------------------------------------------------------------------------
# 9-11: build_augmentations
# ---------------------------------------------------------------------------

class TestBuildAugmentations:
    """Spec 11.3 lines 2975-2987."""

    def test_n_views_1_returns_empty(self):
        # spec: section 11.3 line 2980 — n_views < 2 → empty
        assert build_augmentations(1) == []

    def test_n_views_2_returns_hflip_only(self):
        # spec: section 11.3 lines 2981-2982
        augs = build_augmentations(2)
        assert len(augs) == 1
        assert augs[0] == ("hflip",)

    def test_n_views_5_returns_four_specs(self):
        # spec: section 11.3 lines 2983-2986
        augs = build_augmentations(5)
        assert len(augs) == 4
        assert augs[0] == ("hflip",)
        assert augs[1] == ("rotate", +5)
        assert augs[2] == ("rotate", -5)
        assert augs[3] == ("brightness", 1.05)


# ---------------------------------------------------------------------------
# 12-14: apply_augmentation
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _PIL_AVAILABLE, reason="PIL not available")
class TestApplyAugmentation:
    """Spec 11.3 lines 2989-3002."""

    def test_hflip_is_own_inverse(self):
        # spec: section 11.3 lines 2990-2991 — FLIP_LEFT_RIGHT
        img = _make_pil_image()
        arr_orig = np.array(img)
        flipped = apply_augmentation(img, ("hflip",))
        double_flipped = apply_augmentation(flipped, ("hflip",))
        np.testing.assert_array_equal(np.array(double_flipped), arr_orig)

    def test_rotate_changes_pixels(self):
        # spec: section 11.3 lines 2992-2999 — rotate by +5 degrees
        img = _make_pil_image(64, 64)
        rotated = apply_augmentation(img, ("rotate", +5))
        assert not np.allclose(
            np.array(img, dtype=np.float32),
            np.array(rotated, dtype=np.float32),
        )

    def test_brightness_increases_mean(self):
        # spec: section 11.3 lines 3000-3001 — brightness > 1 → brighter
        img = _make_pil_image()
        brightened = apply_augmentation(img, ("brightness", 1.05))
        mean_orig = np.array(img, dtype=np.float32).mean()
        mean_bright = np.array(brightened, dtype=np.float32).mean()
        assert mean_bright > mean_orig


# ---------------------------------------------------------------------------
# 15-17: aggregate_views
# ---------------------------------------------------------------------------

class TestAggregateViews:
    """Spec 11.4 lines 3019-3035."""

    def test_mean_over_two_views(self):
        # spec: section 11.4 line 3030 — "stacked.mean(axis=0)"
        p1 = np.array([0.6, 0.1, 0.1, 0.1, 0.05, 0.05], dtype=np.float32)
        p2 = np.array([0.4, 0.2, 0.1, 0.1, 0.10, 0.10], dtype=np.float32)
        agg, n = aggregate_views([p1, p2], [True, True])
        np.testing.assert_allclose(agg, (p1 + p2) / 2, atol=1e-6)
        assert n == 2

    def test_all_failed_returns_zeros(self):
        # spec: section 11.4 lines 3027-3028 — "All views failed → zero-filled"
        p1 = np.zeros(6, dtype=np.float32)
        p2 = np.zeros(6, dtype=np.float32)
        agg, n = aggregate_views([p1, p2], [False, False])
        np.testing.assert_array_equal(agg, np.zeros(6, dtype=np.float32))
        assert n == 0

    def test_one_failed_view_excluded(self):
        # spec: section 11.4 lines 3025-3026 — "surviving = [p for p, ok in ... if ok]"
        good = np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
        bad = np.array([0.9, 0.1, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        agg, n = aggregate_views([good, bad], [True, False])
        np.testing.assert_allclose(agg, good, atol=1e-6)
        assert n == 1


# ---------------------------------------------------------------------------
# 18-19: jensen_shannon_divergence
# ---------------------------------------------------------------------------

class TestJSD:
    """Spec 11.5 lines 3046-3065."""

    def test_identical_distributions_zero(self):
        # spec: section 11.5 line 3062 — "JSD = 0 means v3 and LoRA agree perfectly"
        p = np.array([0.5, 0.2, 0.1, 0.1, 0.05, 0.05])
        assert abs(jensen_shannon_divergence(p, p.copy())) < 1e-10

    def test_opposed_distributions_near_log2(self):
        # spec: section 11.5 line 3061 — bounded by log(2) ≈ 0.693 (natural log)
        p = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        jsd = jensen_shannon_divergence(p, q)
        assert abs(jsd - math.log(2)) < 1e-6


# ---------------------------------------------------------------------------
# 20: PSV NOT invoked during TTA
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (_PIL_AVAILABLE and _TORCH_AVAILABLE),
    reason="PIL and torch required",
)
class TestPSVNotInvokedDuringTTA:
    """Spec 11.1 line 2925 / 11.9 lines 3139-3140: PSV excluded from TTA."""

    def test_psv_not_called(self):
        """compute_signal_c must have call_count == 0 after apply_tta."""
        # Mock PSV's compute_signal_c — should NEVER be called by apply_tta.
        # spec: section 11.1 line 2925 — "PSV does NOT participate in TTA"
        # spec: section 11.9 lines 3139-3140 — "TTA does not run on PSV"
        mock_psv = MagicMock(name="compute_signal_c")

        pil = _make_pil_image()
        v3_model = _MockV3Model()
        lora_model = _MockLoRAModel()

        with patch("tomato_sandbox.signals.psv.psv.compute_signal_c", mock_psv):
            apply_tta(
                pil_image=pil,
                n_views=2,
                v3_model=v3_model,
                lora_model=lora_model,
                prototype_bank=None,
                initial_combined_max_prob=0.50,
            )

        assert mock_psv.call_count == 0, (
            f"compute_signal_c was called {mock_psv.call_count} time(s) during TTA. "
            "PSV must NOT be invoked by apply_tta."
        )


# ---------------------------------------------------------------------------
# 21: Signal B single-pass constraint preserved under TTA
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (_PIL_AVAILABLE and _TORCH_AVAILABLE),
    reason="PIL and torch required",
)
class TestSignalBSinglePassUnderTTA:
    """Spec 9.2 lines 1838-1848: single-pass per call, even under TTA."""

    def test_compute_signal_b_called_once_per_view(self):
        """For 2-view TTA, compute_signal_b is called exactly 2 times.

        Each call to compute_signal_b is a single deterministic forward pass
        (model.eval() + torch.no_grad()), never a loop of stochastic passes.
        TTA calls it once per view; 2-view TTA = 2 calls total.
        # spec: section 9.2 lines 1838-1848 — single-pass constraint
        """
        pil = _make_pil_image()
        lora_model = _MockLoRAModel()
        v3_model = _MockV3Model()

        # Use a real SignalBResult-producing side_effect to count calls.
        # We wrap the real compute_signal_b via a mock's side_effect.
        from tomato_sandbox.signals.lora_signal import compute_signal_b as real_compute_b

        call_records: list[int] = []

        def counting_compute_b(lora_input, model, prototype_bank=None, **kwargs):
            call_records.append(1)
            return real_compute_b(
                lora_input, model, prototype_bank=prototype_bank, **kwargs
            )

        with patch(
            "tomato_sandbox.signals.tta.compute_signal_b",
            side_effect=counting_compute_b,
        ):
            apply_tta(
                pil_image=pil,
                n_views=2,
                v3_model=v3_model,
                lora_model=lora_model,
                prototype_bank=None,
                initial_combined_max_prob=0.50,
            )

        # 2-view TTA: 1 original + 1 augmented = 2 calls to compute_signal_b.
        assert len(call_records) == 2, (
            f"Expected 2 calls to compute_signal_b for 2-view TTA, got {len(call_records)}. "
            "Signal B must be called once per view (single-pass constraint, spec 9.2)."
        )


# ---------------------------------------------------------------------------
# 22-25: apply_tta high-level behavior
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (_PIL_AVAILABLE and _TORCH_AVAILABLE),
    reason="PIL and torch required",
)
class TestApplyTTABehavior:
    """Integration-level tests for apply_tta."""

    def test_n_views_1_no_augmentation(self):
        """n_views=1: only the original view; apply_tta returns valid results."""
        pil = _make_pil_image()
        v3_model = _MockV3Model()
        lora_model = _MockLoRAModel()

        sig_a, sig_b, report = apply_tta(
            pil_image=pil,
            n_views=1,
            v3_model=v3_model,
            lora_model=lora_model,
            prototype_bank=None,
            initial_combined_max_prob=0.70,
        )

        assert report.n_views_attempted == 1
        assert report.triggered is False  # n_views == 1 → not triggered
        assert sig_a.forward_succeeded
        assert sig_b.forward_succeeded
        assert len(report.per_view_v3_argmax) == 1
        assert len(report.per_view_lora_argmax) == 1
        assert report.n_views_succeeded_v3 == 1
        assert report.n_views_succeeded_lora == 1

    def test_n_views_2_reports_correctly(self):
        """n_views=2: TTAReport.n_views_attempted == 2, triggered == True."""
        pil = _make_pil_image()
        v3_model = _MockV3Model()
        lora_model = _MockLoRAModel()

        sig_a, sig_b, report = apply_tta(
            pil_image=pil,
            n_views=2,
            v3_model=v3_model,
            lora_model=lora_model,
            prototype_bank=None,
            initial_combined_max_prob=0.50,
        )

        assert report.n_views_attempted == 2
        assert report.triggered is True
        assert len(report.per_view_v3_argmax) == 2
        assert len(report.per_view_lora_argmax) == 2

    def test_n_views_5_runs_five_views(self):
        """n_views=5: 1 original + 4 augmented = 5 views attempted."""
        pil = _make_pil_image()
        v3_model = _MockV3Model()
        lora_model = _MockLoRAModel()

        sig_a, sig_b, report = apply_tta(
            pil_image=pil,
            n_views=5,
            v3_model=v3_model,
            lora_model=lora_model,
            prototype_bank=None,
            initial_combined_max_prob=0.30,
        )

        assert report.n_views_attempted == 5
        assert report.triggered is True
        assert len(report.per_view_v3_argmax) == 5
        assert len(report.per_view_lora_argmax) == 5

    def test_all_views_failed_forward_succeeded_false(self):
        """If all model calls raise, aggregated results have forward_succeeded=False."""
        pil = _make_pil_image()
        bad_v3 = _AlwaysFailingModel()
        bad_lora = _AlwaysFailingModel()

        sig_a, sig_b, report = apply_tta(
            pil_image=pil,
            n_views=2,
            v3_model=bad_v3,
            lora_model=bad_lora,
            prototype_bank=None,
            initial_combined_max_prob=0.50,
        )

        # spec: section 11.4 lines 3027-3028 — all failed → zero-filled distribution
        assert sig_a.forward_succeeded is False
        assert sig_b.forward_succeeded is False
        assert report.n_views_succeeded_v3 == 0
        assert report.n_views_succeeded_lora == 0


# ---------------------------------------------------------------------------
# 26: TTAReport fields
# ---------------------------------------------------------------------------

class TestTTAReportFields:
    """Spec 11.6 lines 3079-3098."""

    def test_report_dataclass_fields(self):
        """TTAReport can be constructed with all required fields."""
        report = TTAReport(
            triggered=True,
            n_views_attempted=2,
            n_views_succeeded_v3=2,
            n_views_succeeded_lora=2,
            initial_combined_max_prob=0.50,
            final_combined_max_prob=0.62,
            per_view_v3_argmax=[0, 0],
            per_view_v3_succeeded=[True, True],
            per_view_lora_argmax=[0, 0],
            per_view_lora_succeeded=[True, True],
            view_disagreement_v3=0.0,
            view_disagreement_lora=0.0,
        )
        assert report.triggered is True
        assert report.n_views_attempted == 2
        assert report.initial_combined_max_prob == 0.50
        assert report.final_combined_max_prob == 0.62
        # spec: section 11.6 line 3087 — per_view_v3_argmax uses -1 for failed views
        assert report.per_view_v3_argmax == [0, 0]


# ---------------------------------------------------------------------------
# 27-28: view_disagreement helper
# ---------------------------------------------------------------------------

class TestViewDisagreement:
    """Spec 11.6 lines 3093-3099."""

    def test_all_agree_returns_zero(self):
        # spec: section 11.6 line 3093 — "fraction ... where argmax differs from majority"
        result = _compute_view_disagreement([0, 0, 0], [True, True, True])
        assert result == 0.0

    def test_failed_views_excluded(self):
        # spec: section 11.6 line 3099 — "computed only over surviving views"
        # Views 0 and 2 succeeded (both argmax=0); view 1 failed (ignored)
        result = _compute_view_disagreement([0, 1, 0], [True, False, True])
        assert result == 0.0  # only two succeeded views, both agree

    def test_partial_disagreement(self):
        # 3 survived views: [0, 0, 1] — majority is 0; 1/3 disagree
        result = _compute_view_disagreement([0, 0, 1], [True, True, True])
        assert result == pytest.approx(1 / 3, abs=1e-9)


# ---------------------------------------------------------------------------
# 29: Flat-path shim import
# ---------------------------------------------------------------------------

class TestFlatPathShimImport:
    """DEC-037: spec 11.7 line 3103 says tomato_sandbox/tta.py; shim satisfies this."""

    def test_shim_import_resolves(self):
        """from tomato_sandbox.tta import should_trigger_tta must resolve."""
        from tomato_sandbox.tta import should_trigger_tta as _fn
        assert callable(_fn)
        assert _fn is should_trigger_tta

    def test_shim_apply_tta_resolves(self):
        from tomato_sandbox.tta import apply_tta as _at
        assert callable(_at)

    def test_shim_ttareport_resolves(self):
        from tomato_sandbox.tta import TTAReport as _TR
        assert _TR is TTAReport
