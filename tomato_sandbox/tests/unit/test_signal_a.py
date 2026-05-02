"""
Unit tests for Signal A (v3 model wrapper).

Spec section: 8 (Signal A), lines 1578-1789.

Design note (DEC-034, Decision 3):
  Tests use a _MockV3Model that validates input shapes and returns
  synthetic [B, 10] logits. Real weights (sacred path) are NOT loaded —
  they require CUDA and GPU memory, making them unsuitable for unit tests.
  This mock exercises the full wiring:
    preprocess→ (caller's responsibility) → forward → remap → SignalAResult.
  Weight correctness is a Phase C validation concern (spec Section 28).

Tests:
  1. forward pass shape: SignalAResult has correct array shapes.
  2. remap correctness: v3-ordered probs are correctly mapped to canonical.
  3. GPU lock: import resolves; lock context manager works around mock forward.
  4. NaN guard: NaN logits yield forward_succeeded=False, reason="numerical_instability".
  5. exception guard: forward exception yields forward_succeeded=False, reason includes "exception:".
  6. degraded-mode: zero_signal_a import resolves and zeroes correct slots.
  7. chilli_leakage: computed as sum of v3 indices 6,7,8,9.
  8. no-renorm: tomato_probs_canonical does NOT sum to 1 when chilli_leakage > 0.
  9. zero-failure fields: zero-filled probs on failure.
  10. extract_v3_outputs: standalone test of remap on known input.
  11. TOMATO_CROP_MODE_INDEX: constant equals 2 (spec section 7.2 line 1431).
  12. signal_a_forward: returns ok=True with valid model and input.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
from tomato_sandbox.signals.v3_signal import (
    CHILLI_LEAKAGE_THRESHOLD,
    SignalAResult,
    _NUM_TOMATO_CLASSES,
    _V3_NUM_OUTPUTS,
    _V3_TO_CANONICAL_REMAP,
    compute_signal_a,
    extract_v3_outputs,
    signal_a_forward,
    zero_signal_a,  # re-exported from degraded_mode
)
from tomato_sandbox.config import TOMATO_CROP_MODE_INDEX
from tomato_sandbox.utils.gpu_lock import GPULock


# ---------------------------------------------------------------------------
# Mock v3 model
# DEC-034 Decision 3: uses synthetic logits, no sacred weights loaded.
# ---------------------------------------------------------------------------

class _MockV3Model(torch.nn.Module):
    """Minimal mock that satisfies the v3 forward interface.

    Accepts (x, crop_mode, domain_labels) and returns {"logits": [B, 10]}.

    spec: section 8.2 lines 1617-1640 — interface contract
    """

    def __init__(self, logits_override: torch.Tensor | None = None) -> None:
        super().__init__()
        # logits_override: if given, returned regardless of input.
        self._logits_override = logits_override

    def forward(
        self,
        x: torch.Tensor,
        crop_mode: torch.Tensor,
        domain_labels: object,
    ) -> dict:
        B = x.shape[0]
        if self._logits_override is not None:
            logits = self._logits_override.expand(B, -1)
        else:
            # Deterministic synthetic logits: class 0 gets highest activation.
            logits = torch.zeros(B, _V3_NUM_OUTPUTS)
            logits[:, 0] = 2.0  # foliar dominant in v3 space
        return {"logits": logits}

    # eval/train are inherited from nn.Module — no override needed.


class _NaNV3Model(torch.nn.Module):
    """Returns NaN logits to trigger the numerical_instability path."""

    def forward(
        self, x: torch.Tensor, crop_mode: torch.Tensor, domain_labels: object
    ) -> dict:
        B = x.shape[0]
        return {"logits": torch.full((B, _V3_NUM_OUTPUTS), float("nan"))}


class _ExceptionV3Model(torch.nn.Module):
    """Always raises RuntimeError."""

    def forward(
        self, x: torch.Tensor, crop_mode: torch.Tensor, domain_labels: object
    ) -> dict:
        raise RuntimeError("mock CUDA OOM")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_model() -> _MockV3Model:
    return _MockV3Model()


@pytest.fixture
def input_tensor() -> torch.Tensor:
    """[3, 224, 224] CPU tensor — mimics preprocess_for_v3 output."""
    return torch.randn(3, 224, 224)


# ---------------------------------------------------------------------------
# Test 1: forward pass shape
# ---------------------------------------------------------------------------

def test_forward_pass_shape(mock_model: _MockV3Model, input_tensor: torch.Tensor) -> None:
    """SignalAResult has the correct array dimensions and types.

    spec: section 8.6 lines 1716-1726
    """
    result = compute_signal_a(mock_model, input_tensor)

    assert result.forward_succeeded is True
    assert result.failure_reason is None

    # tomato_probs_canonical: [6] float32
    assert isinstance(result.tomato_probs_canonical, np.ndarray), "must be ndarray"
    assert result.tomato_probs_canonical.shape == (_NUM_TOMATO_CLASSES,)
    assert result.tomato_probs_canonical.dtype == np.float32

    # raw_probs_v3_order: [10] float32
    assert result.raw_probs_v3_order is not None
    assert result.raw_probs_v3_order.shape == (_V3_NUM_OUTPUTS,)

    # Scalars
    assert isinstance(result.tomato_max_prob_canonical, float)
    assert isinstance(result.tomato_argmax_canonical, int)
    assert isinstance(result.chilli_leakage, float)


# ---------------------------------------------------------------------------
# Test 2: remap correctness
# ---------------------------------------------------------------------------

def test_remap_correctness() -> None:
    """v3-ordered probabilities are correctly remapped to canonical order.

    v3 ordering: [foliar=0, late_blight=1, septoria=2, ylcv=3, mosaic=4, healthy=5]
    canonical:   [foliar=0, septoria=1, late_blight=2, ylcv=3, mosaic=4, healthy=5]
    remap = [0, 2, 1, 3, 4, 5]

    spec: section 8.3 lines 1672-1678
    """
    # Construct a v3-ordered prob vector with distinct values at each position.
    # Assign p[i] = (i+1) / 60.0 for tomato slice, zero for chilli.
    probs_v3 = torch.zeros(_V3_NUM_OUTPUTS)
    for i in range(6):
        probs_v3[i] = (i + 1) / 60.0

    extracted = extract_v3_outputs(probs_v3)
    canonical: np.ndarray = extracted["tomato_probs_canonical"]

    # v3[0]=foliar → canonical[0]
    assert canonical[0] == pytest.approx(probs_v3[0].item(), abs=1e-6), "foliar remap"
    # v3[1]=late_blight → canonical[2]
    assert canonical[2] == pytest.approx(probs_v3[1].item(), abs=1e-6), "late_blight remap"
    # v3[2]=septoria → canonical[1]
    assert canonical[1] == pytest.approx(probs_v3[2].item(), abs=1e-6), "septoria remap"
    # v3[3]=ylcv → canonical[3] (unchanged)
    assert canonical[3] == pytest.approx(probs_v3[3].item(), abs=1e-6), "ylcv unchanged"
    # v3[4]=mosaic → canonical[4] (unchanged)
    assert canonical[4] == pytest.approx(probs_v3[4].item(), abs=1e-6), "mosaic unchanged"
    # v3[5]=healthy → canonical[5] (unchanged)
    assert canonical[5] == pytest.approx(probs_v3[5].item(), abs=1e-6), "healthy unchanged"


# ---------------------------------------------------------------------------
# Test 3: GPU lock acquisition import and usage
# DEC-034 Decision 4: lock is orchestrator-level; test verifies import resolves
# and lock context manager works around a synchronous mock call.
# ---------------------------------------------------------------------------

def test_gpu_lock_import_and_context() -> None:
    """GPULock can be imported and used as async context manager.

    The lock itself is owned by the orchestrator (spec Section 21).
    We verify the import chain is intact and the lock API functions.

    spec: section 20.6 lines 6579-6589
    """
    # Verify import resolves (DEC-034 Decision 4)
    from tomato_sandbox.utils.gpu_lock import GPULock  # noqa: PLC0415

    lock = GPULock(timeout_s=1.0)

    async def _run() -> None:
        async with lock.acquired():
            # Simulate a forward call inside the lock
            model = _MockV3Model()
            tensor = torch.randn(3, 224, 224)
            result = compute_signal_a(model, tensor)
            assert result.forward_succeeded is True

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Test 4: NaN guard — numerical_instability path
# ---------------------------------------------------------------------------

def test_nan_guard_triggers_failure() -> None:
    """NaN logits from the model result in forward_succeeded=False.

    spec: section 8.2 lines 1637-1638 — NaN guard
    spec: section 8.7 lines 1754-1763 — not fwd["ok"] branch
    """
    model = _NaNV3Model()
    tensor = torch.randn(3, 224, 224)

    result = compute_signal_a(model, tensor)

    assert result.forward_succeeded is False
    assert result.failure_reason == "numerical_instability"
    # Zero-filled probs
    np.testing.assert_array_equal(
        result.tomato_probs_canonical,
        np.zeros(_NUM_TOMATO_CLASSES, dtype=np.float32),
    )
    assert result.raw_probs_v3_order is None
    assert result.chilli_leakage == 0.0
    assert result.tomato_max_prob_canonical == 0.0
    assert result.tomato_argmax_canonical == 0


# ---------------------------------------------------------------------------
# Test 5: exception guard
# ---------------------------------------------------------------------------

def test_exception_guard_triggers_failure() -> None:
    """Forward exception results in forward_succeeded=False and exception reason.

    spec: section 8.7 lines 1744-1753
    """
    model = _ExceptionV3Model()
    tensor = torch.randn(3, 224, 224)

    result = compute_signal_a(model, tensor)

    assert result.forward_succeeded is False
    assert result.failure_reason is not None
    assert result.failure_reason.startswith("exception:")
    assert "RuntimeError" in result.failure_reason
    np.testing.assert_array_equal(
        result.tomato_probs_canonical,
        np.zeros(_NUM_TOMATO_CLASSES, dtype=np.float32),
    )
    assert result.raw_probs_v3_order is None


# ---------------------------------------------------------------------------
# Test 6: degraded-mode zero_signal_a re-export
# ---------------------------------------------------------------------------

def test_zero_signal_a_zeroes_correct_slots() -> None:
    """zero_signal_a zeroes indices 0-5 and 18 in the 19-dim feature vector.

    spec: section 12.2 lines 3232-3234
    The function is re-exported from v3_signal for downstream convenience.
    """
    raw = np.ones(19, dtype=np.float32)
    zero_signal_a(raw)

    # Indices 0-5 (v3 tomato probs) must be zero
    np.testing.assert_array_equal(raw[0:6], np.zeros(6, dtype=np.float32))
    # Index 18 (chilli_leakage) must be zero
    assert raw[18] == 0.0
    # Indices 6-17 must remain 1.0 (untouched)
    np.testing.assert_array_equal(raw[6:18], np.ones(12, dtype=np.float32))


# ---------------------------------------------------------------------------
# Test 7: chilli_leakage computed correctly
# ---------------------------------------------------------------------------

def test_chilli_leakage_computed_correctly() -> None:
    """chilli_leakage = sum of v3 output probs at indices 6, 7, 8, 9.

    spec: section 8.3 line 1670
    """
    probs = torch.zeros(_V3_NUM_OUTPUTS)
    probs[6] = 0.05
    probs[7] = 0.10
    probs[8] = 0.15
    probs[9] = 0.20

    extracted = extract_v3_outputs(probs)
    expected_leakage = 0.05 + 0.10 + 0.15 + 0.20

    assert extracted["chilli_leakage"] == pytest.approx(expected_leakage, abs=1e-6)


# ---------------------------------------------------------------------------
# Test 8: no renormalisation — tomato probs do NOT sum to 1
# ---------------------------------------------------------------------------

def test_tomato_probs_not_renormalized() -> None:
    """tomato_probs_canonical sum to (1 - chilli_leakage), NOT to 1.

    spec: section 8.3 lines 1687-1689
    "The 6 tomato probs do NOT sum to 1 after extraction — they sum to
     (1 - chilli_leakage). This is by design."
    """
    # Build logits such that chilli classes get 40% total probability.
    logits = torch.zeros(_V3_NUM_OUTPUTS)
    # Make chilli get 40%: softmax drives this via high chilli logits
    logits[6] = 0.4
    logits[7] = 0.4
    logits[8] = 0.3
    logits[9] = 0.3

    probs = torch.softmax(logits, dim=0)
    extracted = extract_v3_outputs(probs)

    tomato_sum = float(extracted["tomato_probs_canonical"].sum())
    chilli_leakage = extracted["chilli_leakage"]

    # tomato_sum + chilli_leakage should equal 1.0 (all probs from softmax)
    assert tomato_sum + chilli_leakage == pytest.approx(1.0, abs=1e-5)
    # tomato_sum should be LESS than 1.0 when chilli_leakage > 0
    assert tomato_sum < 1.0


# ---------------------------------------------------------------------------
# Test 9: zero-filled fields on failure
# ---------------------------------------------------------------------------

def test_zero_filled_probs_on_failure() -> None:
    """Failure results have zero-filled prob arrays, 0.0 scalars, argmax=0.

    spec: section 8.7 lines 1744-1763 — both failure branches
    """
    for model in [_NaNV3Model(), _ExceptionV3Model()]:
        result = compute_signal_a(model, torch.randn(3, 224, 224))
        np.testing.assert_array_equal(
            result.tomato_probs_canonical,
            np.zeros(_NUM_TOMATO_CLASSES, dtype=np.float32),
        )
        assert result.tomato_max_prob_canonical == 0.0
        assert result.tomato_argmax_canonical == 0
        assert result.chilli_leakage == 0.0
        assert result.raw_probs_v3_order is None


# ---------------------------------------------------------------------------
# Test 10: extract_v3_outputs standalone on known input
# ---------------------------------------------------------------------------

def test_extract_v3_outputs_standalone() -> None:
    """extract_v3_outputs returns correct dict structure on known input.

    spec: section 8.3 lines 1660-1684
    """
    probs = torch.zeros(_V3_NUM_OUTPUTS)
    probs[0] = 0.5  # foliar — should land at canonical[0]
    probs[2] = 0.3  # septoria in v3 → canonical[1]
    probs[1] = 0.2  # late_blight in v3 → canonical[2]

    result = extract_v3_outputs(probs)

    assert "tomato_probs_canonical" in result
    assert "chilli_leakage" in result
    assert "raw_probs_v3_order" in result

    canon = result["tomato_probs_canonical"]
    assert canon.shape == (_NUM_TOMATO_CLASSES,)
    assert canon.dtype == np.float32

    # foliar (v3[0]) → canonical[0]
    assert canon[0] == pytest.approx(0.5, abs=1e-6)
    # septoria (v3[2]) → canonical[1]
    assert canon[1] == pytest.approx(0.3, abs=1e-6)
    # late_blight (v3[1]) → canonical[2]
    assert canon[2] == pytest.approx(0.2, abs=1e-6)

    # raw_probs_v3_order has length 10
    assert result["raw_probs_v3_order"].shape == (_V3_NUM_OUTPUTS,)


# ---------------------------------------------------------------------------
# Test 11: TOMATO_CROP_MODE_INDEX constant
# ---------------------------------------------------------------------------

def test_tomato_crop_mode_index_is_2() -> None:
    """TOMATO_CROP_MODE_INDEX must equal 2.

    spec: section 7.2 line 1431 — "passed to v3's HardFiLM at inference"
    spec: section 8.2 line 1643 — "crop_mode = TOMATO_CROP_MODE_INDEX (=2)"
    """
    assert TOMATO_CROP_MODE_INDEX == 2


# ---------------------------------------------------------------------------
# Test 12: signal_a_forward returns ok=True with valid mock model
# ---------------------------------------------------------------------------

def test_signal_a_forward_ok_true() -> None:
    """signal_a_forward returns ok=True and correct shapes with mock model.

    spec: section 8.2 lines 1616-1641
    """
    model = _MockV3Model()
    x = torch.randn(1, 3, 224, 224)

    result = signal_a_forward(model, x)

    assert result["ok"] is True
    assert result["logits"] is not None
    assert result["probs"] is not None
    assert result["logits"].shape == (1, _V3_NUM_OUTPUTS)
    assert result["probs"].shape == (1, _V3_NUM_OUTPUTS)

    # probs must sum to ~1 (softmax)
    prob_sum = result["probs"].sum(dim=1)
    assert prob_sum.item() == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Test 13: remap constant integrity
# ---------------------------------------------------------------------------

def test_remap_constant() -> None:
    """_V3_TO_CANONICAL_REMAP equals [0, 2, 1, 3, 4, 5] exactly.

    spec: section 8.3 lines 1672-1674
    """
    expected = np.array([0, 2, 1, 3, 4, 5], dtype=np.intp)
    np.testing.assert_array_equal(_V3_TO_CANONICAL_REMAP, expected)


# ---------------------------------------------------------------------------
# Test 14: chilli leakage threshold constant
# ---------------------------------------------------------------------------

def test_chilli_leakage_threshold_default() -> None:
    """CHILLI_LEAKAGE_THRESHOLD default is 0.40.

    spec: section 8.4 line 1695 — "TOMATO_CHILLI_LEAKAGE_THRESHOLD (default 0.40)"
    """
    assert CHILLI_LEAKAGE_THRESHOLD == pytest.approx(0.40, abs=1e-9)


# ---------------------------------------------------------------------------
# Test 15: compute_signal_a argmax/max consistent with canonical probs
# ---------------------------------------------------------------------------

def test_argmax_and_max_consistent() -> None:
    """tomato_argmax_canonical and tomato_max_prob_canonical are consistent.

    spec: section 8.6 lines 1720-1721
    """
    # Force v3 to give highest logit at v3-index 2 (septoria) → canonical[1]
    logits = torch.zeros(_V3_NUM_OUTPUTS)
    logits[2] = 5.0  # septoria in v3 ordering

    probs = torch.softmax(logits, dim=0)

    model_class = type(
        "_FixedModel",
        (torch.nn.Module,),
        {
            "forward": lambda self, x, crop_mode, domain_labels: {
                "logits": probs.unsqueeze(0).expand(x.shape[0], -1)
            }
        },
    )
    model = model_class()

    result = compute_signal_a(model, torch.randn(3, 224, 224))

    assert result.forward_succeeded is True
    # argmax in canonical space should be 1 (septoria)
    assert result.tomato_argmax_canonical == 1
    # max prob should be the canonical[1] value
    assert result.tomato_max_prob_canonical == pytest.approx(
        float(result.tomato_probs_canonical[1]), abs=1e-6
    )
