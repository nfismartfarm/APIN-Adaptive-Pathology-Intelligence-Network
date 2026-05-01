"""
Degraded-mode helpers for the Tomato 3-Signal sandbox classifier input.

Spec section: 12.7 (Degraded-mode handling), lines 3348-3373.
Spec section: 12.2 (The 19-dimensional feature vector / build_classifier_input),
              lines 3169-3244.

When one or more signals fail at inference time, the corresponding block in
the 19-dim feature vector is zeroed before standardization.  This matches the
training-time degraded-mode augmentation (P_DEGRADE = 0.20), so the classifier
has seen zero-filled blocks and knows to rely on the surviving signals.

Feature vector layout (authoritative — from spec 12.2 table, lines 3175-3196):
  Index 0-5   : v3 tomato_probs_canonical [0..5]
  Index 6-11  : LoRA tomato_probs_canonical [0..5]
  Index 12    : psv compatibility_max
  Index 13    : psv compatibility_margin
  Index 14    : agree_v3  (PSV argmax == v3 argmax)
  Index 15    : agree_lora (PSV argmax == LoRA argmax)
  Index 16    : JSD between v3 and LoRA
  Index 17    : psv_reliability
  Index 18    : chilli_leakage (from v3)

Degraded zeroing rules (spec 12.2 code, lines 3231-3242):
  Signal A (v3) failed:   zero raw[0:6] AND raw[18]
  Signal B (LoRA) failed: zero raw[6:12]
  Signal C (PSV) failed:  zero raw[12:14], raw[14], raw[15], raw[17]

# spec: 12.7 lines 3348-3373
# spec: 12.2 lines 3231-3242
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VECTOR_DIM: int = 19
"""Dimension of the 19-dim classifier input feature vector.

# spec: 12.1 lines 3149 — "assembles them into a 19-dimensional feature vector"
"""

# Slice boundaries for each signal's block in the 19-dim vector.
# Expressed as (start_inclusive, stop_exclusive) pairs matching Python slice
# notation. Multiple tuples per signal indicate non-contiguous indices.
#
# spec: 12.2 code lines 3231-3242 — verbatim block assignments:
#   Signal A: raw[0:6] = 0.0; raw[18] = 0.0
#   Signal B: raw[6:12] = 0.0
#   Signal C: raw[12:14] = 0.0; raw[14] = 0.0; raw[15] = 0.0; raw[17] = 0.0

SIGNAL_A_SLICES: list[tuple[int, int]] = [
    (0, 6),    # v3_p_foliar .. v3_p_healthy  (indices 0-5)
    (18, 19),  # chilli_leakage               (index 18)
]
"""Feature-vector slices zeroed when Signal A (v3) forward fails.

# spec: 12.2 lines 3232-3234 — `raw[0:6] = 0.0; raw[18] = 0.0`
"""

SIGNAL_B_SLICES: list[tuple[int, int]] = [
    (6, 12),   # lora_p_foliar .. lora_p_healthy  (indices 6-11)
]
"""Feature-vector slices zeroed when Signal B (LoRA) forward fails.

# spec: 12.2 lines 3235-3236 — `raw[6:12] = 0.0`
"""

SIGNAL_C_SLICES: list[tuple[int, int]] = [
    (12, 14),  # psv_top1, psv_margin            (indices 12-13)
    (14, 15),  # agree_v3                         (index 14)
    (15, 16),  # agree_lora                       (index 15)
    (17, 18),  # psv_reliability                  (index 17)
]
"""Feature-vector slices zeroed when Signal C (PSV) forward fails.

Note: index 16 (JSD) is NOT zeroed here — when PSV fails it still makes sense
to compute JSD between v3 and LoRA if both succeeded.  The spec's
build_classifier_input code only zeros indices 12-15 and 17 for Signal C.

# spec: 12.2 lines 3237-3241 — `raw[12:14]=0.0; raw[14]=0.0; raw[15]=0.0;
# raw[17]=0.0`
"""


# ---------------------------------------------------------------------------
# Per-signal zero-fill helpers
# ---------------------------------------------------------------------------


def zero_signal_a(raw: np.ndarray) -> np.ndarray:
    """Zero the Signal A (v3) block in the 19-dim feature vector, in-place.

    Args:
        raw: Float32 array of shape ``[19]``.  Modified in-place.

    Returns:
        The same array (for chaining convenience).

    Raises:
        ValueError: If *raw* does not have ``VECTOR_DIM`` elements.

    # spec: 12.2 lines 3232-3234 — `raw[0:6] = 0.0; raw[18] = 0.0`
    """
    _check_dim(raw)
    for start, stop in SIGNAL_A_SLICES:
        raw[start:stop] = 0.0
    return raw


def zero_signal_b(raw: np.ndarray) -> np.ndarray:
    """Zero the Signal B (LoRA) block in the 19-dim feature vector, in-place.

    Args:
        raw: Float32 array of shape ``[19]``.  Modified in-place.

    Returns:
        The same array (for chaining convenience).

    Raises:
        ValueError: If *raw* does not have ``VECTOR_DIM`` elements.

    # spec: 12.2 lines 3235-3236 — `raw[6:12] = 0.0`
    """
    _check_dim(raw)
    for start, stop in SIGNAL_B_SLICES:
        raw[start:stop] = 0.0
    return raw


def zero_signal_c(raw: np.ndarray) -> np.ndarray:
    """Zero the Signal C (PSV) block in the 19-dim feature vector, in-place.

    Args:
        raw: Float32 array of shape ``[19]``.  Modified in-place.

    Returns:
        The same array (for chaining convenience).

    Raises:
        ValueError: If *raw* does not have ``VECTOR_DIM`` elements.

    # spec: 12.2 lines 3237-3241 — `raw[12:14]=0.0; raw[14]=0.0;
    # raw[15]=0.0; raw[17]=0.0`
    """
    _check_dim(raw)
    for start, stop in SIGNAL_C_SLICES:
        raw[start:stop] = 0.0
    return raw


# ---------------------------------------------------------------------------
# Composite helper
# ---------------------------------------------------------------------------


def apply_degraded_mode(
    raw: np.ndarray,
    *,
    sa_ok: bool,
    sb_ok: bool,
    sc_ok: bool,
) -> np.ndarray:
    """Zero all failed-signal blocks in the 19-dim feature vector.

    This is the canonical inference-time degraded-mode handler.  It combines
    the three per-signal helpers and matches the ``build_classifier_input``
    code from spec Section 12.2.

    Args:
        raw: Float32 array of shape ``[19]``.  Modified in-place.
        sa_ok: ``True`` if Signal A (v3) forward pass succeeded.
        sb_ok: ``True`` if Signal B (LoRA) forward pass succeeded.
        sc_ok: ``True`` if Signal C (PSV) forward pass succeeded.

    Returns:
        The same *raw* array after zeroing whichever blocks correspond to
        failed signals.

    # spec: 12.2 lines 3231-3242 — degraded-mode section of
    # build_classifier_input
    # spec: 12.7 lines 3364 — "At inference, signal failures are handled
    # directly in build_classifier_input: the corresponding feature block is
    # zeroed before standardization."
    """
    _check_dim(raw)
    if not sa_ok:
        zero_signal_a(raw)
    if not sb_ok:
        zero_signal_b(raw)
    if not sc_ok:
        zero_signal_c(raw)
    return raw


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_dim(raw: np.ndarray) -> None:
    if raw.shape != (VECTOR_DIM,):
        raise ValueError(
            f"degraded_mode: expected array of shape ({VECTOR_DIM},), "
            f"got {raw.shape}"
        )


# ---------------------------------------------------------------------------
# Convenience: build an all-zeros baseline vector
# ---------------------------------------------------------------------------


def zeros_vector() -> np.ndarray:
    """Return a zero-filled float32 array of shape ``[VECTOR_DIM]``.

    Useful as a starting point for building the classifier input when all
    signals have failed (Tier 4B path).

    Returns:
        ``np.zeros(VECTOR_DIM, dtype=np.float32)``
    """
    return np.zeros(VECTOR_DIM, dtype=np.float32)
