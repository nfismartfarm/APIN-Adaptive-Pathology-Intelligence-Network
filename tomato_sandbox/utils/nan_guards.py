"""
NaN / non-finite guards for TTA and signal forward passes.

Spec section: 11 (Test-Time Augmentation), specifically 11.2 (When TTA fires)
and 11.4 (Per-signal aggregation across views).

The spec mandates two distinct guard patterns:

1. TTA decision guard — spec 11.2 lines 2946-2951:
   "If the 1-view classifier itself produces a non-numeric result ...
   combined_max_prob may be NaN.  The TTA decision treats NaN as 'do not
   run TTA':
       if not np.isfinite(combined_max_prob):
           n_views = 1   # no TTA"

2. View aggregation guard — spec 11.4 lines 3025-3035:
   "Failed views are excluded.  If a view's forward pass produced NaN,
   threw an exception, or otherwise had forward_succeeded=False, that
   view's probability vector is dropped from aggregation."

# spec: 11.2 lines 2946-2951
# spec: 11.4 lines 3025-3035
"""

from __future__ import annotations

import numpy as np
from typing import Sequence

# ---------------------------------------------------------------------------
# TTA threshold defaults
# spec: 11.2 lines 2932-2939 (verbatim)
# "combined_max_prob >= TOMATO_TTA_TRIGGER_THRESHOLD (default 0.55) → no TTA"
# "TOMATO_TTA_ESCALATE_THRESHOLD <= combined_max_prob < TOMATO_TTA_TRIGGER_THRESHOLD
#  → 2-view TTA"
# "combined_max_prob < TOMATO_TTA_ESCALATE_THRESHOLD (default 0.45) → 5-view TTA"
# ---------------------------------------------------------------------------
TTA_TRIGGER_THRESHOLD: float = 0.55
TTA_ESCALATE_THRESHOLD: float = 0.45


# ---------------------------------------------------------------------------
# Scalar guard
# ---------------------------------------------------------------------------


def guard_scalar(value: float, default: float = float("nan")) -> float:
    """Return *value* if finite, else *default*.

    Used to sanitize model output scalars (e.g. ``combined_max_prob``) before
    they are used in control-flow decisions such as TTA triggering.

    Args:
        value: The raw scalar to check.
        default: Value to return if *value* is NaN, +inf, or -inf.
            Defaults to ``float("nan")`` — callers that need a safe numeric
            fallback should pass ``0.0`` or another sentinel.

    Returns:
        *value* if ``np.isfinite(value)`` is ``True``; *default* otherwise.

    # spec: 11.2 lines 2948 — pattern `if not np.isfinite(combined_max_prob)`
    """
    return value if np.isfinite(value) else default


# ---------------------------------------------------------------------------
# Array guard
# ---------------------------------------------------------------------------


def guard_array(
    arr: np.ndarray | Sequence[float],
    expected_len: int,
    default_value: float = 0.0,
) -> np.ndarray:
    """Return *arr* as a float32 array if all elements are finite and length
    matches *expected_len*; otherwise return a zero-filled array of the same
    shape.

    Used to sanitize per-view softmax probability vectors before they are
    aggregated.

    Args:
        arr: Array to check (any sequence coercible to numpy float32).
        expected_len: Expected number of elements (e.g. 6 for a 6-class
            signal).
        default_value: Fill value for the returned fallback array. Defaults
            to ``0.0``.

    Returns:
        A numpy float32 array of length *expected_len*.  Either the
        (possibly converted) *arr* — if it passes both length and finiteness
        checks — or ``np.full(expected_len, default_value, dtype=np.float32)``.

    # spec: 11.4 lines 3027-3030 — "All views failed; return zero-filled
    # distribution (caller treats as forward failure)"
    """
    a = np.asarray(arr, dtype=np.float32)
    if a.shape != (expected_len,):
        return np.full(expected_len, default_value, dtype=np.float32)
    if not np.all(np.isfinite(a)):
        return np.full(expected_len, default_value, dtype=np.float32)
    return a


# ---------------------------------------------------------------------------
# TTA decision
# ---------------------------------------------------------------------------


def tta_n_views(
    combined_max_prob: float,
    trigger_threshold: float = TTA_TRIGGER_THRESHOLD,
    escalate_threshold: float = TTA_ESCALATE_THRESHOLD,
) -> int:
    """Return the number of TTA views required given *combined_max_prob*.

    Implements the TTA decision table from spec Section 11.2 with the NaN
    guard from the same section.

    Decision table (spec 11.2 lines 2932-2939):
      - NaN / non-finite → 1 view (no TTA)
      - combined_max_prob >= trigger_threshold (default 0.55) → 1 view
      - escalate_threshold <= combined_max_prob < trigger_threshold → 2 views
      - combined_max_prob < escalate_threshold (default 0.45) → 5 views

    Args:
        combined_max_prob: The classifier's ``combined_max_prob`` from the
            1-view pass.  May be NaN.
        trigger_threshold: Threshold above which TTA is not needed.
            Default: ``TTA_TRIGGER_THRESHOLD`` (0.55).
        escalate_threshold: Threshold below which 5-view TTA fires.
            Default: ``TTA_ESCALATE_THRESHOLD`` (0.45).

    Returns:
        Integer number of views: 1, 2, or 5.

    # spec: 11.2 lines 2946-2951 — NaN guard: `if not np.isfinite: n_views=1`
    # spec: 11.2 lines 2932-2939 — TTA decision table
    """
    # NaN guard — spec 11.2 lines 2946-2951
    if not np.isfinite(combined_max_prob):
        return 1  # no TTA; pipeline proceeds with failed signals

    if combined_max_prob >= trigger_threshold:
        return 1

    if combined_max_prob >= escalate_threshold:
        return 2

    return 5


# ---------------------------------------------------------------------------
# View aggregation filter
# ---------------------------------------------------------------------------


def filter_finite_views(
    per_view_probs: Sequence[np.ndarray],
    per_view_ok: Sequence[bool],
) -> list[np.ndarray]:
    """Return only the probability arrays whose corresponding view succeeded
    and whose values are all finite.

    This is the filtering logic described in :func:`aggregate_views` in spec
    Section 11.4.  The caller is responsible for stacking and averaging the
    returned list.

    Args:
        per_view_probs: Sequence of per-view softmax arrays (one per view).
            Each element should be shape ``[n_classes]``.
        per_view_ok: Sequence of booleans, parallel to *per_view_probs*.
            ``True`` means that view's forward pass succeeded.

    Returns:
        List of surviving probability arrays — only those where
        ``per_view_ok[i]`` is ``True`` AND all values in
        ``per_view_probs[i]`` are finite.  May be empty (all views failed).

    # spec: 11.4 lines 3025-3030 — "surviving = [p for p, ok in
    # zip(per_view_probs, per_view_ok) if ok]"
    # spec: 11.4 lines 3032-3035 — "If a view's forward pass produced NaN ...
    # that view's probability vector is dropped from aggregation."
    """
    surviving: list[np.ndarray] = []
    for prob, ok in zip(per_view_probs, per_view_ok):
        if not ok:
            continue
        arr = np.asarray(prob, dtype=np.float32)
        if np.all(np.isfinite(arr)):
            surviving.append(arr)
    return surviving


def aggregate_views(
    per_view_probs: Sequence[np.ndarray],
    per_view_ok: Sequence[bool],
) -> tuple[np.ndarray, int]:
    """Aggregate per-view softmax outputs by mean over surviving views.

    A direct implementation of the ``aggregate_views`` function from spec
    Section 11.4.

    Args:
        per_view_probs: List of ``[n_classes]`` softmax outputs (zero-filled
            for failure views — the ``per_view_ok`` flag is the authoritative
            indicator of success).
        per_view_ok: Parallel list of success flags.

    Returns:
        A tuple ``(aggregated, n_views_used)`` where:
          - ``aggregated`` is the mean over surviving views, as float32
            ``[n_classes]``; or ``np.zeros(n_classes)`` if all views failed.
          - ``n_views_used`` is the number of surviving views.

    # spec: 11.4 lines 3019-3030 — aggregate_views function verbatim
    """
    surviving = filter_finite_views(per_view_probs, per_view_ok)
    if not surviving:
        # Determine n_classes from input; fall back to 6 (canonical tomato classes)
        n_classes = (
            len(per_view_probs[0])
            if per_view_probs
            else 6
        )
        return np.zeros(n_classes, dtype=np.float32), 0
    stacked = np.stack(surviving)  # [n_surviving, n_classes]
    return stacked.mean(axis=0).astype(np.float32), len(surviving)
