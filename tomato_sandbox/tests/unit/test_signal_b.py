"""
Unit tests for Signal B — Single-Pass LoRA wrapper.

Tests:
  1.  forward_shape_ok             — signal_b_forward returns [B,6] probs, [B,768] cls_token
  2.  no_remap_behavior            — indices 0-5 pass through unchanged (no remap)
  3.  gpu_lock_import              — acquire_gpu_lock import resolves correctly
  4.  nan_guard_logits             — NaN in logits → ok=False
  5.  inf_guard_logits             — Inf in logits → ok=False
  6.  degraded_mode_failure        — forward failure → forward_succeeded=False
  7.  degraded_mode_exception      — exception in forward → forward_succeeded=False
  8.  single_pass_only             — model.__call__ called exactly once per compute_signal_b
  9.  high_confidence_no_blend     — lora_max >= 0.60 → no prototype blending
  10. low_confidence_triggers_blend — lora_max < 0.60 → blending triggered
  11. underpopulated_all_fallback   — all classes underpopulated → raw lora returned
  12. prototype_blend_result_sums_to_1 — blended distribution sums to 1
  13. result_probs_canonical_len   — SignalBResult.tomato_probs_canonical is length 6
  14. result_cls_token_len         — SignalBResult.cls_token is length 768
  15. failure_result_forward_succeeded_false — _failure_result sets forward_succeeded=False
  16. uniform_fallback_on_failure  — _failure_result returns uniform 1/6 distribution
  17. blend_reason_strings         — blend_reason takes only allowed values
  18. no_mc_dropout_single_call    — model.eval called before forward (no MC loop)

Spec section: 9 (Signal B — Single-Pass LoRA), lines 1793-1992.
# spec: section 9.2 lines 1838-1848 — signal_b_forward
# spec: section 9.5 lines 1903-1945 — prototype_blend
# spec: section 9.6 lines 1961-1973 — SignalBResult
"""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, call, patch

# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------
from tomato_sandbox.signals.lora_signal import (
    PrototypeBank,
    SignalBResult,
    compute_signal_b,
    prototype_blend,
    signal_b_forward,
    _failure_result,
    NUM_LORA_CLASSES,
    CLS_TOKEN_DIM,
    PROTOTYPE_BLEND_THRESHOLD,
    T_PROTO,
    BLEND_WEIGHT,
)

# Verify re-export of degraded_mode helper
from tomato_sandbox.signals.lora_signal import zero_signal_b  # noqa: F401

# GPU lock import check (task-card requirement)
from tomato_sandbox.utils.gpu_lock import GPULock  # noqa: F401

# ---------------------------------------------------------------------------
# Helpers — mock model and tensors
# ---------------------------------------------------------------------------

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _TORCH_AVAILABLE, reason="torch not installed"
)


def _make_mock_model(
    logits: np.ndarray | None = None,
    force_nan: bool = False,
    raise_exc: bool = False,
) -> MagicMock:
    """Return a mock LoRA model that exposes the uniform forward dict contract.

    # spec: section 9.2 line 1842 — `out = model(x)` dict contract
    """
    mock = MagicMock()

    if raise_exc:
        mock.side_effect = RuntimeError("mock forward failure")
        return mock

    if logits is None:
        # Default: uniform logits → uniform softmax
        logits_arr = np.zeros(NUM_LORA_CLASSES, dtype=np.float32)
    else:
        logits_arr = np.array(logits, dtype=np.float32)

    if force_nan:
        logits_arr[0] = float("nan")

    logits_t = torch.tensor(logits_arr, dtype=torch.float32).unsqueeze(0)  # [1, 6]
    cls_t = torch.zeros(1, CLS_TOKEN_DIM, dtype=torch.float32)             # [1, 768]

    mock.return_value = {"logits": logits_t, "cls_token": cls_t}
    return mock


def _make_lora_input() -> "torch.Tensor":
    """Create a minimal [1, 3, 392, 392] tensor for testing.

    # spec: section 9.1 line 1800 — input shape [B, 3, 392, 392]
    """
    return torch.zeros(1, 3, 392, 392, dtype=torch.float32)


def _make_prototype_bank(
    *,
    underpopulated_classes: set[int] | None = None,
    max_prob_class: int = 0,
) -> PrototypeBank:
    """Build a minimal PrototypeBank with random prototypes.

    # spec: section 9.4 lines 1870-1876 — PrototypeBank fields
    """
    rng = np.random.default_rng(seed=42)
    if underpopulated_classes is None:
        underpopulated_classes = set()

    prototypes: dict[int, np.ndarray] = {}
    class_counts: dict[int, int] = {}

    for cls_idx in range(NUM_LORA_CLASSES):
        if cls_idx in underpopulated_classes:
            # spec: section 9.4 line 1882 — empty array placeholder
            prototypes[cls_idx] = np.empty((0, CLS_TOKEN_DIM), dtype=np.float32)
            class_counts[cls_idx] = 0
        else:
            n_protos = 5  # within [3, 10] range
            # If this is the "preferred" class, give it a prototype aligned with
            # a unit vector so cosine similarity is high.
            if cls_idx == max_prob_class:
                proto = np.ones((n_protos, CLS_TOKEN_DIM), dtype=np.float32)
            else:
                proto = rng.random((n_protos, CLS_TOKEN_DIM), dtype=np.float32)
            prototypes[cls_idx] = proto
            class_counts[cls_idx] = n_protos

    return PrototypeBank(
        prototypes=prototypes,
        class_counts=class_counts,
        underpopulated_classes=underpopulated_classes,
        model_version="abc1234",
    )


# ---------------------------------------------------------------------------
# Test 1: forward_shape_ok
# ---------------------------------------------------------------------------

def test_forward_shape_ok() -> None:
    """signal_b_forward returns correct shapes: probs [B,6], cls_token [B,768].

    # spec: section 9.2 lines 1834-1836 — return dict shape contract
    # spec: section 9.1 lines 1813,1819 — cls_token [B,768], logits [B,6]
    """
    model = _make_mock_model()
    x = _make_lora_input()
    result = signal_b_forward(model, x)

    assert result["ok"] is True
    assert result["probs"].shape == (1, NUM_LORA_CLASSES), (
        f"Expected probs shape (1, {NUM_LORA_CLASSES}), got {result['probs'].shape}"
    )
    assert result["cls_token"].shape == (1, CLS_TOKEN_DIM), (
        f"Expected cls_token shape (1, {CLS_TOKEN_DIM}), got {result['cls_token'].shape}"
    )


# ---------------------------------------------------------------------------
# Test 2: no_remap_behavior
# ---------------------------------------------------------------------------

def test_no_remap_behavior() -> None:
    """LoRA indices pass through unchanged — no remap applied.

    Signal B index ordering matches canonical; no permutation needed.

    # spec: section 9.1 line 1822 — "This ordering matches canonical, so no
    # remap is needed for LoRA → canonical."
    # spec: section 8.3 lines 1672-1678 — only Signal A (v3) has a remap
    """
    # Give class 2 (late_blight) the highest logit
    logits = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
    model = _make_mock_model(logits=logits)
    x = _make_lora_input()
    result = signal_b_forward(model, x)

    assert result["ok"] is True
    probs = result["probs"][0].numpy()
    argmax = int(np.argmax(probs))
    assert argmax == 2, (
        f"Expected argmax=2 (late_blight, no remap), got {argmax}. "
        "Signal B must NOT remap indices."
    )


# ---------------------------------------------------------------------------
# Test 3: gpu_lock_import
# ---------------------------------------------------------------------------

def test_gpu_lock_import() -> None:
    """acquire_gpu_lock import path resolves correctly.

    GPU lock is the orchestrator's responsibility (Section 21.3 step 4).
    This test verifies the import contract is satisfied.

    # spec: section 21.3 step 4 — "Acquire GPU lock (timeout per Section 20.6)"
    """
    # GPULock is already imported at module level — verify it is the right class
    assert GPULock is not None
    lock = GPULock(timeout_s=5.0)
    assert lock.timeout_s == 5.0


# ---------------------------------------------------------------------------
# Test 4: nan_guard_logits
# ---------------------------------------------------------------------------

def test_nan_guard_logits() -> None:
    """NaN in logits → ok=False, all fields None.

    # spec: section 9.2 lines 1845-1846 — NaN check returns ok=False
    """
    model = _make_mock_model(force_nan=True)
    x = _make_lora_input()
    result = signal_b_forward(model, x)

    assert result["ok"] is False
    assert result["logits"] is None
    assert result["probs"] is None
    assert result["cls_token"] is None


# ---------------------------------------------------------------------------
# Test 5: inf_guard_logits
# ---------------------------------------------------------------------------

def test_inf_guard_logits() -> None:
    """Inf in logits → ok=False.

    # spec: section 9.2 line 1845 — `torch.isinf(logits).any()` → ok=False
    """
    inf_logits = torch.full((1, NUM_LORA_CLASSES), float("inf"))
    cls_t = torch.zeros(1, CLS_TOKEN_DIM)

    model = MagicMock()
    model.return_value = {"logits": inf_logits, "cls_token": cls_t}

    x = _make_lora_input()
    result = signal_b_forward(model, x)

    assert result["ok"] is False


# ---------------------------------------------------------------------------
# Test 6: degraded_mode_failure
# ---------------------------------------------------------------------------

def test_degraded_mode_failure() -> None:
    """NaN in logits propagates to compute_signal_b as forward_succeeded=False.

    # spec: section 9.6 line 1971 — "forward_succeeded: bool"
    # spec: section 12.7 lines 3348-3364 — degraded mode: LoRA block zeroed
    """
    model = _make_mock_model(force_nan=True)
    x = _make_lora_input()
    result = compute_signal_b(lora_input=x, model=model)

    assert result.forward_succeeded is False
    assert result.failure_reason == "numerical_instability"
    assert len(result.tomato_probs_canonical) == NUM_LORA_CLASSES
    assert len(result.cls_token) == CLS_TOKEN_DIM


# ---------------------------------------------------------------------------
# Test 7: degraded_mode_exception
# ---------------------------------------------------------------------------

def test_degraded_mode_exception() -> None:
    """Exception in model forward → forward_succeeded=False, reason="exception".

    # spec: section 9.6 line 1972 — `failure_reason: "exception"`
    """
    model = _make_mock_model(raise_exc=True)
    x = _make_lora_input()
    result = compute_signal_b(lora_input=x, model=model)

    assert result.forward_succeeded is False
    assert result.failure_reason == "exception"


# ---------------------------------------------------------------------------
# Test 8: single_pass_only
# ---------------------------------------------------------------------------

def test_single_pass_only() -> None:
    """compute_signal_b calls model.__call__ exactly once — no MC Dropout loop.

    CRITICAL constraint: Signal B is single-pass. No stochastic loop.

    # spec: section 9.2 lines 1838-1848 — single eval() + no_grad() pass
    # spec: section 9.1 line 1797 — "single-pass" training/inference strategy
    """
    model = _make_mock_model()
    x = _make_lora_input()
    compute_signal_b(lora_input=x, model=model)

    # model is called exactly once (B=1 batch)
    assert model.call_count == 1, (
        f"Expected model to be called exactly once (single-pass), "
        f"but was called {model.call_count} times. "
        "Signal B must NOT run multiple stochastic passes (no MC Dropout)."
    )


# ---------------------------------------------------------------------------
# Test 9: high_confidence_no_blend
# ---------------------------------------------------------------------------

def test_high_confidence_no_blend() -> None:
    """lora_max >= PROTOTYPE_BLEND_THRESHOLD → no prototype blending applied.

    # spec: section 9.5 line 1955 — "lora_max_prob >= threshold: use raw output"
    # spec: section 9.4 line 1863 — PROTOTYPE_BLEND_THRESHOLD = 0.60
    """
    # Class 0 gets logit=10 → softmax ≈ 1.0 > 0.60
    logits = [10.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    model = _make_mock_model(logits=logits)
    bank = _make_prototype_bank()
    x = _make_lora_input()

    result = compute_signal_b(lora_input=x, model=model, prototype_bank=bank)

    assert result.forward_succeeded is True
    assert result.prototype_blend_applied is False
    assert result.prototype_blend_reason == "high_confidence_no_blend"
    # Raw and final should be identical
    np.testing.assert_array_almost_equal(
        result.tomato_probs_canonical,
        result.raw_lora_probs_canonical,
        decimal=5,
    )


# ---------------------------------------------------------------------------
# Test 10: low_confidence_triggers_blend
# ---------------------------------------------------------------------------

def test_low_confidence_triggers_blend() -> None:
    """lora_max < PROTOTYPE_BLEND_THRESHOLD → prototype blending triggered.

    # spec: section 9.5 line 1901 — "when lora_max_prob < threshold, blend"
    """
    # Uniform logits → softmax = 1/6 ≈ 0.167 < 0.60
    logits = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    model = _make_mock_model(logits=logits)
    bank = _make_prototype_bank()
    x = _make_lora_input()

    result = compute_signal_b(
        lora_input=x,
        model=model,
        prototype_bank=bank,
        blend_threshold=PROTOTYPE_BLEND_THRESHOLD,
    )

    assert result.forward_succeeded is True
    assert result.prototype_blend_applied is True
    assert result.prototype_blend_reason == "low_confidence"


# ---------------------------------------------------------------------------
# Test 11: underpopulated_all_fallback
# ---------------------------------------------------------------------------

def test_underpopulated_all_fallback() -> None:
    """All classes underpopulated → blend falls back to raw LoRA distribution.

    # spec: section 9.5 line 1937 — "fall back to LoRA if all classes underpopulated"
    # spec: section 9.5 line 1920-1921 — underpopulated → -inf in similarity
    """
    all_underpopulated = set(range(NUM_LORA_CLASSES))
    logits = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # uniform → max ≈ 0.167 < 0.60
    lora_probs = np.ones(NUM_LORA_CLASSES, dtype=np.float32) / NUM_LORA_CLASSES
    cls_token = np.zeros(CLS_TOKEN_DIM, dtype=np.float32)

    bank = _make_prototype_bank(underpopulated_classes=all_underpopulated)

    blended, reason = prototype_blend(
        lora_probs=lora_probs,
        cls_token=cls_token,
        bank=bank,
    )

    assert reason == "all_classes_underpopulated"
    np.testing.assert_array_almost_equal(blended, lora_probs, decimal=5)


# ---------------------------------------------------------------------------
# Test 12: prototype_blend_result_sums_to_1
# ---------------------------------------------------------------------------

def test_prototype_blend_result_sums_to_1() -> None:
    """Blended distribution sums to 1 (renormalization check).

    # spec: section 9.5 lines 1941-1944 — "Renormalize ... blended / blended.sum()"
    """
    rng = np.random.default_rng(seed=7)
    lora_probs = rng.dirichlet(np.ones(NUM_LORA_CLASSES)).astype(np.float32)
    cls_token = rng.standard_normal(CLS_TOKEN_DIM).astype(np.float32)

    # Partial underpopulation to exercise renorm path
    bank = _make_prototype_bank(underpopulated_classes={3, 4})

    blended, reason = prototype_blend(
        lora_probs=lora_probs,
        cls_token=cls_token,
        bank=bank,
    )

    assert abs(float(blended.sum()) - 1.0) < 1e-5, (
        f"Blended distribution does not sum to 1: sum={blended.sum():.6f}"
    )


# ---------------------------------------------------------------------------
# Test 13: result_probs_canonical_len
# ---------------------------------------------------------------------------

def test_result_probs_canonical_len() -> None:
    """SignalBResult.tomato_probs_canonical has length NUM_LORA_CLASSES (6).

    # spec: section 9.6 line 1962 — "tomato_probs_canonical: np.ndarray  # [6]"
    """
    model = _make_mock_model()
    x = _make_lora_input()
    result = compute_signal_b(lora_input=x, model=model)

    assert len(result.tomato_probs_canonical) == NUM_LORA_CLASSES


# ---------------------------------------------------------------------------
# Test 14: result_cls_token_len
# ---------------------------------------------------------------------------

def test_result_cls_token_len() -> None:
    """SignalBResult.cls_token has length CLS_TOKEN_DIM (768).

    # spec: section 9.6 line 1967 — "cls_token: np.ndarray  # [768]"
    # spec: section 9.1 line 1813 — "CLS token output: [B, 768]"
    """
    model = _make_mock_model()
    x = _make_lora_input()
    result = compute_signal_b(lora_input=x, model=model)

    assert len(result.cls_token) == CLS_TOKEN_DIM


# ---------------------------------------------------------------------------
# Test 15: failure_result_forward_succeeded_false
# ---------------------------------------------------------------------------

def test_failure_result_forward_succeeded_false() -> None:
    """_failure_result sets forward_succeeded=False and failure_reason correctly.

    # spec: section 9.6 lines 1971-1972 — forward_succeeded, failure_reason
    """
    result = _failure_result("exception")
    assert result.forward_succeeded is False
    assert result.failure_reason == "exception"

    result2 = _failure_result("numerical_instability")
    assert result2.failure_reason == "numerical_instability"


# ---------------------------------------------------------------------------
# Test 16: uniform_fallback_on_failure
# ---------------------------------------------------------------------------

def test_uniform_fallback_on_failure() -> None:
    """_failure_result returns uniform 1/6 probabilities for all classes.

    A failed Signal B should provide a valid (if uninformative) distribution
    so downstream consumers don't crash on non-finite values.

    # spec: section 12.7 — degraded mode: classifier sees zero-filled block
    # (uniform 1/6 is overridden by zero_signal_b in build_classifier_input)
    """
    result = _failure_result("exception")
    expected_prob = 1.0 / NUM_LORA_CLASSES
    for p in result.tomato_probs_canonical:
        assert abs(float(p) - expected_prob) < 1e-5, (
            f"Expected uniform {expected_prob:.4f}, got {p:.4f}"
        )
    assert abs(sum(result.tomato_probs_canonical) - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# Test 17: blend_reason_strings
# ---------------------------------------------------------------------------

def test_blend_reason_strings() -> None:
    """prototype_blend_reason takes only the three allowed values from spec.

    # spec: section 9.6 line 1970 — allowed values:
    # "low_confidence" | "high_confidence_no_blend" | "all_classes_underpopulated"
    """
    allowed_reasons = {
        "low_confidence",
        "high_confidence_no_blend",
        "all_classes_underpopulated",
    }
    model = _make_mock_model()
    x = _make_lora_input()

    # No bank → "high_confidence_no_blend"
    result_no_bank = compute_signal_b(lora_input=x, model=model, prototype_bank=None)
    assert result_no_bank.prototype_blend_reason in allowed_reasons

    # All underpopulated bank + uniform logits → "all_classes_underpopulated"
    bank_all_under = _make_prototype_bank(
        underpopulated_classes=set(range(NUM_LORA_CLASSES))
    )
    logits_uniform = [0.0] * NUM_LORA_CLASSES
    model_low = _make_mock_model(logits=logits_uniform)
    result_under = compute_signal_b(
        lora_input=x, model=model_low, prototype_bank=bank_all_under
    )
    assert result_under.prototype_blend_reason in allowed_reasons


# ---------------------------------------------------------------------------
# Test 18: no_mc_dropout_single_call
# ---------------------------------------------------------------------------

def test_no_mc_dropout_single_call() -> None:
    """model.eval() is called before the forward pass (no MC Dropout loop).

    Validates that signal_b_forward sets eval mode on the model before the
    forward pass. There is NO MC Dropout — model is in eval() deterministically.

    # spec: section 9.2 line 1838 — `model.eval()` called unconditionally
    # spec: section 9.1 line 1797 — "single-pass" means one deterministic forward
    """
    eval_called_before_forward = []

    class _TrackEvalModel:
        """Records whether eval() was called before __call__."""
        def __init__(self) -> None:
            self._eval_count = 0
            self._call_count = 0

        def eval(self) -> "_TrackEvalModel":
            self._eval_count += 1
            return self

        def __call__(self, x: "torch.Tensor") -> dict:
            eval_called_before_forward.append(self._eval_count > 0)
            self._call_count += 1
            logits = torch.zeros(1, NUM_LORA_CLASSES)
            cls = torch.zeros(1, CLS_TOKEN_DIM)
            return {"logits": logits, "cls_token": cls}

    tracking_model = _TrackEvalModel()
    x = _make_lora_input()
    signal_b_forward(tracking_model, x)

    assert tracking_model._eval_count >= 1, (
        "model.eval() was never called — spec requires eval mode before forward."
    )
    assert tracking_model._call_count == 1, (
        f"Expected exactly 1 forward call (single-pass), got {tracking_model._call_count}."
    )
    assert all(eval_called_before_forward), (
        "model.eval() must be called BEFORE the forward pass."
    )
