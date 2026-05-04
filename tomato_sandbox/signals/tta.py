"""
TTA controller — Test-Time Augmentation orchestration for Signals A and B.

Spec section: 11 (Test-Time Augmentation), lines 2919-3143.

Canonical sandbox path: tomato_sandbox/signals/tta.py  (DEC-037, task-card DEC-033)
Flat-path re-export shim:  tomato_sandbox/tta.py  (per spec 11.7 line 3103)

Public API:
    TTAReport           — dataclass, per spec 11.6 lines 3079-3098
    should_trigger_tta  — spec 11.2 lines 2932-2939 + 2946-2951
    build_augmentations — spec 11.3 lines 2975-2987
    apply_augmentation  — spec 11.3 lines 2989-3002
    aggregate_views     — spec 11.4 lines 3019-3030 (delegates to nan_guards)
    jensen_shannon_divergence — spec 11.5 lines 3046-3058
    apply_tta           — spec 11.7 line 3106 orchestrator entry point

CRITICAL RULE — PSV NOT INVOKED during TTA:
    # spec: section 11.1 lines 2925-2925 (verbatim):
    # "PSV does NOT participate in TTA."
    # spec: section 11.9 lines 3139-3140:
    # "TTA does not run on PSV. PSV's spatial and color features are not
    #  augmentation-invariant."
    Signal C (PSV) is computed ONCE by the orchestrator on the original image and
    passed in; apply_tta never calls compute_signal_c or any PSV function.

SINGLE-PASS CONSTRAINT — Signal B:
    # spec: section 9.2 lines 1838-1848 — "single-pass inference … model.eval()
    #   is called before the forward pass and Dropout … is fully disabled."
    apply_tta calls compute_signal_b once per augmented view; each call is a
    separate deterministic pass (model.eval() is called inside signal_b_forward).
    There is no loop of stochastic passes; MC Dropout is not used for Signal B.

No print() in this module.  All informational output uses get_logger.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

try:
    from PIL import Image as _PIL_Image
    import PIL.ImageEnhance as _PIL_Enhance
    _PIL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PIL_AVAILABLE = False

try:
    import torch  # type: ignore[import]
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False

from tomato_sandbox.utils.logging import get_logger
from tomato_sandbox.utils.nan_guards import (
    tta_n_views,
    TTA_TRIGGER_THRESHOLD,
    TTA_ESCALATE_THRESHOLD,
    aggregate_views as _nan_guard_aggregate_views,
)
from tomato_sandbox.signals.v3_signal import compute_signal_a, SignalAResult
from tomato_sandbox.signals.lora_signal import compute_signal_b, SignalBResult
from tomato_sandbox.preprocessing.preprocess import preprocess_for_v3, preprocess_for_lora
from tomato_sandbox.config import LORA_PAD_VALUE

_logger = get_logger(__name__)

# Number of canonical tomato classes shared by both signals.
_NUM_CLASSES: int = 6  # spec: section 8.1 line 1610 / 9.1 line 1819


# ---------------------------------------------------------------------------
# TTAReport — spec: section 11.6 lines 3079-3098
# ---------------------------------------------------------------------------

@dataclass
class TTAReport:
    """Monitoring record produced by apply_tta.

    # spec: section 11.6 lines 3079-3098
    """

    triggered: bool
    """True if TTA fired (2-view or 5-view).
    # spec: section 11.6 line 3081 — "triggered: bool"
    """

    n_views_attempted: int
    """1, 2, or 5.
    # spec: section 11.6 line 3082 — "n_views_attempted: int  # 1, 2, or 5"
    """

    n_views_succeeded_v3: int
    """How many v3 views succeeded.
    # spec: section 11.6 line 3083 — "n_views_succeeded_v3: int"
    """

    n_views_succeeded_lora: int
    """How many LoRA views succeeded.
    # spec: section 11.6 line 3084 — "n_views_succeeded_lora: int"
    """

    initial_combined_max_prob: float
    """The 1-view classifier output that triggered TTA.
    # spec: section 11.6 line 3085 — "initial_combined_max_prob: float"
    """

    final_combined_max_prob: float
    """Post-aggregation classifier output (filled by orchestrator after re-run).
    # spec: section 11.6 line 3086 — "final_combined_max_prob: float"
    """

    per_view_v3_argmax: list[int]
    """Argmax per view for v3, canonical order; -1 for failed views.
    # spec: section 11.6 lines 3087-3088
    """

    per_view_v3_succeeded: list[bool]
    """True if that view's v3 forward succeeded.
    # spec: section 11.6 line 3089 — "per_view_v3_succeeded: list[bool]"
    """

    per_view_lora_argmax: list[int]
    """Argmax per view for LoRA, canonical order; -1 for failed views.
    # spec: section 11.6 lines 3090-3091
    """

    per_view_lora_succeeded: list[bool]
    """True if that view's LoRA forward succeeded.
    # spec: section 11.6 line 3092 — "per_view_lora_succeeded: list[bool]"
    """

    view_disagreement_v3: float
    """Fraction of SUCCEEDED v3 views where argmax differs from majority.
    # spec: section 11.6 line 3093 — "view_disagreement_v3: float"
    """

    view_disagreement_lora: float
    """Fraction of SUCCEEDED LoRA views where argmax differs from majority.
    # spec: section 11.6 line 3094 — "view_disagreement_lora: float"
    """


# ---------------------------------------------------------------------------
# should_trigger_tta — spec: section 11.2 lines 2932-2951 + 11.7 line 3105
# ---------------------------------------------------------------------------

def should_trigger_tta(combined_max_prob: float) -> int:
    """Return the number of TTA views (1, 2, or 5) for combined_max_prob.

    Delegates to nan_guards.tta_n_views, which is the authoritative implementation
    of the TTA decision table.  This function is the spec-named entry point from
    Section 11.7; it exists so callers can do
        `from tomato_sandbox.signals.tta import should_trigger_tta`
    matching the spec's "TTA controller" public interface.

    Decision table (spec 11.2 lines 2932-2939):
      NaN / non-finite                              → 1  (no TTA)
      combined_max_prob >= 0.55 (TRIGGER_THRESHOLD) → 1  (no TTA)
      0.45 <= combined_max_prob < 0.55              → 2  (2-view TTA)
      combined_max_prob < 0.45  (ESCALATE_THRESHOLD)→ 5  (5-view TTA)

    Returns:
        int: one of {1, 2, 5}.

    # spec: section 11.2 lines 2932-2939 — TTA decision table
    # spec: section 11.2 lines 2946-2951 — NaN guard
    # spec: section 11.7 line 3105 — "should_trigger_tta(combined_max_prob: float) -> int"
    """
    # Delegate to nan_guards.tta_n_views, which has the exact spec-verbatim
    # implementation and inline citations.
    # spec: section 11.2 lines 2946-2951 — NaN guard: `if not np.isfinite: n_views=1`
    # spec: section 11.2 lines 2932-2939 — threshold ladder
    return tta_n_views(
        combined_max_prob,
        trigger_threshold=TTA_TRIGGER_THRESHOLD,    # 0.55
        escalate_threshold=TTA_ESCALATE_THRESHOLD,  # 0.45
    )


# ---------------------------------------------------------------------------
# build_augmentations — spec: section 11.3 lines 2975-2987
# ---------------------------------------------------------------------------

def build_augmentations(n_views: int) -> list[tuple]:
    """Return a list of augmentation specs for views 1..n-1 (view 0 = original).

    For 2-view TTA: returns [("hflip",)]
    For 5-view TTA: returns [("hflip",), ("rotate", +5), ("rotate", -5),
                              ("brightness", 1.05)]

    # spec: section 11.3 lines 2975-2987 — build_augmentations verbatim
    """
    # spec: section 11.3 line 2980 — "augs = []"
    augs: list[tuple] = []
    # spec: section 11.3 lines 2981-2982 — "if n_views >= 2: augs.append(("hflip",))"
    if n_views >= 2:
        augs.append(("hflip",))
    # spec: section 11.3 lines 2983-2986 — "if n_views >= 5: ..."
    if n_views >= 5:
        augs.append(("rotate", +5))
        augs.append(("rotate", -5))
        augs.append(("brightness", 1.05))
    return augs


# ---------------------------------------------------------------------------
# apply_augmentation — spec: section 11.3 lines 2989-3002
# ---------------------------------------------------------------------------

def apply_augmentation(pil: "_PIL_Image.Image", aug_spec: tuple) -> "_PIL_Image.Image":
    """Apply one augmentation to a PIL image.

    # spec: section 11.3 lines 2989-3002 — apply_augmentation verbatim
    # spec: section 11.3 line 2996 — fillcolor=(LORA_PAD_VALUE, ...) for rotation
    """
    # spec: section 11.3 lines 2990-2991 — hflip
    if aug_spec[0] == "hflip":
        return pil.transpose(_PIL_Image.FLIP_LEFT_RIGHT)
    # spec: section 11.3 lines 2992-2999 — rotate
    elif aug_spec[0] == "rotate":
        # fillcolor matches LORA_PAD_VALUE so rotation padding looks like
        # LoRA's expected pad value.
        # spec: section 11.3 lines 2995-2999 — "fillcolor matches LORA_PAD_VALUE"
        return pil.rotate(
            aug_spec[1],
            expand=False,
            fillcolor=(LORA_PAD_VALUE, LORA_PAD_VALUE, LORA_PAD_VALUE),
        )
    # spec: section 11.3 lines 3000-3001 — brightness
    elif aug_spec[0] == "brightness":
        return _PIL_Enhance.Brightness(pil).enhance(aug_spec[1])
    else:
        # Unknown augmentation — return image unchanged; log a warning.
        _logger.warning(
            "apply_augmentation: unknown aug_spec kind; returning original",
            step="apply_augmentation",
            aug_kind=aug_spec[0],
        )
        return pil


# ---------------------------------------------------------------------------
# aggregate_views — spec: section 11.4 lines 3019-3030
# ---------------------------------------------------------------------------

def aggregate_views(
    per_view_probs: Sequence[np.ndarray],
    per_view_ok: Sequence[bool],
) -> tuple[np.ndarray, int]:
    """Aggregate per-view softmax outputs by mean over surviving views.

    Delegates to nan_guards.aggregate_views.
    Failed / non-finite views are excluded; result averaged over survivors.
    If all views fail: returns zero-filled [6] array and n_views_used=0.

    # spec: section 11.4 lines 3019-3030 — aggregate_views verbatim
    # spec: section 11.4 lines 3032-3035 — failed views are excluded
    # spec: section 11.4 lines 3036-3038 — why mean-of-softmax not mean-of-logits
    """
    return _nan_guard_aggregate_views(per_view_probs, per_view_ok)


# ---------------------------------------------------------------------------
# jensen_shannon_divergence — spec: section 11.5 lines 3046-3058
# ---------------------------------------------------------------------------

def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """JSD between two 6-class distributions; uses natural log.

    Range: [0, log(2)] ≈ [0, 0.693].

    # spec: section 11.5 lines 3046-3058 — jensen_shannon_divergence verbatim
    """
    # spec: section 11.5 lines 3051-3052 — add 1e-12 and renormalize
    p = np.asarray(p, dtype=np.float64) + 1e-12
    q = np.asarray(q, dtype=np.float64) + 1e-12
    p = p / p.sum()
    q = q / q.sum()
    # spec: section 11.5 line 3055 — "m = 0.5 * (p + q)"
    m = 0.5 * (p + q)
    # spec: section 11.5 lines 3056-3057 — KL divergences
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    # spec: section 11.5 line 3058 — "return 0.5 * (kl_pm + kl_qm)"
    return float(0.5 * (kl_pm + kl_qm))


# ---------------------------------------------------------------------------
# _compute_view_disagreement — internal helper
# spec: section 11.6 lines 3093-3099
# ---------------------------------------------------------------------------

def _compute_view_disagreement(
    argmaxes: list[int],
    succeeded: list[bool],
) -> float:
    """Compute fraction of succeeded views that disagree with the majority argmax.

    # spec: section 11.6 lines 3093-3099 — "fraction of SUCCEEDED views where
    #   v3/LoRA argmax differs from majority"
    # spec: section 11.6 line 3099 — "Disagreement is computed only over
    #   surviving (non-failed) views."
    """
    surviving = [a for a, ok in zip(argmaxes, succeeded) if ok]
    if not surviving:
        return 0.0
    # Majority vote
    counts: dict[int, int] = {}
    for a in surviving:
        counts[a] = counts.get(a, 0) + 1
    majority = max(counts, key=lambda k: counts[k])
    disagree = sum(1 for a in surviving if a != majority)
    return float(disagree) / len(surviving)


# ---------------------------------------------------------------------------
# apply_tta — spec: section 11.7 line 3106
# ---------------------------------------------------------------------------

def apply_tta(
    pil_image: "_PIL_Image.Image",
    n_views: int,
    v3_model: object,
    lora_model: object,
    prototype_bank: Optional[object] = None,
    initial_combined_max_prob: float = 0.0,
) -> tuple[SignalAResult, SignalBResult, TTAReport]:
    """Orchestrate TTA over n_views for Signal A (v3) and Signal B (LoRA).

    PSV (Signal C) is NOT invoked here.
    # spec: section 11.1 lines 2925-2925 — "PSV does NOT participate in TTA"
    # spec: section 11.9 lines 3139-3140 — "TTA does not run on PSV"

    Signal B uses single-pass inference per view (model.eval() + no_grad).
    # spec: section 9.2 lines 1838-1848 — single-pass constraint for Signal B

    Steps:
      1. Build augmentation specs for views 1..n-1.
      2. For view 0 (original) AND each augmented view:
         a. Preprocess PIL image for v3 → tensor.
         b. Preprocess PIL image for LoRA → tensor.
         c. Call compute_signal_a(v3_model, v3_tensor) → SignalAResult.
         d. Call compute_signal_b(lora_tensor, lora_model, prototype_bank).
         e. Collect per-view probs and succeeded flags.
      3. Aggregate surviving views for both signals via aggregate_views.
      4. Construct updated SignalAResult and SignalBResult with aggregated probs.
      5. Build TTAReport.

    Args:
        pil_image:   Original PIL.Image.Image (RGB). Must already be validated.
        n_views:     Number of views to run (1, 2, or 5). 1 = no augmentation;
                     existing 1-view results should be used instead (caller
                     optimizes this path; apply_tta still handles n_views=1
                     for testing convenience).
        v3_model:    Loaded v3 model. Caller (orchestrator) holds in app.state.
        lora_model:  Loaded LoRA model. Caller holds in app.state.
        prototype_bank: PrototypeBank or None (passed to compute_signal_b).
        initial_combined_max_prob: The 1-view classifier combined_max_prob that
            triggered TTA. Stored in TTAReport for monitoring.

    Returns:
        (aggregated_signal_a, aggregated_signal_b, tta_report)

    # spec: section 11.7 line 3106 — apply_tta signature
    """
    # PSV not invoked during TTA — sentinel guard so the constraint is visible
    # at the call site without needing to read the function body.
    # spec: section 11.1 lines 2925 / 11.9 lines 3139-3140
    # (PSV is excluded from TTA; compute_signal_c is never called here.)

    aug_specs = build_augmentations(n_views)
    # views: original + len(aug_specs) augmented views
    all_pils: list["_PIL_Image.Image"] = [pil_image] + [
        apply_augmentation(pil_image, spec) for spec in aug_specs
    ]
    # We always have at least 1 view (the original).
    total_views = len(all_pils)  # == n_views when n_views in {1, 2, 5}

    # Accumulators
    per_view_v3_probs: list[np.ndarray] = []
    per_view_v3_ok: list[bool] = []
    per_view_v3_argmax: list[int] = []

    per_view_lora_probs: list[np.ndarray] = []
    per_view_lora_ok: list[bool] = []
    per_view_lora_argmax: list[int] = []

    # Temporary: save a representative succeeded result for fallback metadata
    _last_a_result: Optional[SignalAResult] = None
    _last_b_result: Optional[SignalBResult] = None

    # Detect model devices once before the view loop so we can move tensors
    # to the correct device on each view.  preprocess_for_v3 and preprocess_for_lora
    # always return CPU tensors; the models may be on CUDA.
    # Per DEC-055 / pipeline.py fix: use next(model.parameters()).device.
    _v3_device = None
    _lora_device = None
    if _TORCH_AVAILABLE:
        try:
            _v3_device = next(iter(v3_model.parameters())).device
        except (StopIteration, TypeError, AttributeError):
            _v3_device = None
        try:
            _lora_device = next(iter(lora_model.parameters())).device
        except (StopIteration, TypeError, AttributeError):
            _lora_device = None

    for view_idx, view_pil in enumerate(all_pils):
        # --- Signal A (v3) -------------------------------------------------
        # spec: section 11.3 lines 3004 — "preprocess_for_v3 and preprocess_for_lora
        #   re-run for each view because the augmented pixel content differs"
        try:
            v3_tensor = preprocess_for_v3(view_pil)
            # Move tensor to the same device as the v3 model.
            # preprocess_for_v3 returns a CPU tensor; model may be on CUDA.
            if _TORCH_AVAILABLE and _v3_device is not None:
                v3_tensor = v3_tensor.to(_v3_device)
            sig_a: SignalAResult = compute_signal_a(v3_model, v3_tensor)
        except Exception as exc:
            _logger.warning(
                "apply_tta: Signal A raised exception on view",
                step="apply_tta",
                view_idx=view_idx,
                exc_info=exc,
            )
            sig_a = _failed_signal_a()

        v3_ok = sig_a.forward_succeeded
        v3_probs = sig_a.tomato_probs_canonical  # [6]
        per_view_v3_probs.append(v3_probs)
        per_view_v3_ok.append(v3_ok)
        per_view_v3_argmax.append(int(np.argmax(v3_probs)) if v3_ok else -1)
        if v3_ok:
            _last_a_result = sig_a

        # --- Signal B (LoRA) ------------------------------------------------
        # Single-pass constraint: compute_signal_b calls signal_b_forward once.
        # spec: section 9.2 lines 1838-1848 — single deterministic pass per call
        # TTA calls compute_signal_b once per view; aggregation is our responsibility.
        try:
            lora_tensor = preprocess_for_lora(view_pil)
            if _TORCH_AVAILABLE:
                lora_tensor_batched = lora_tensor.unsqueeze(0)  # [1, 3, 392, 392]
                # Move tensor to the same device as the LoRA model.
                # preprocess_for_lora returns a CPU tensor; model may be on CUDA.
                if _lora_device is not None:
                    lora_tensor_batched = lora_tensor_batched.to(_lora_device)
            else:
                lora_tensor_batched = lora_tensor  # type: ignore[assignment]
            sig_b: SignalBResult = compute_signal_b(
                lora_tensor_batched,
                lora_model,
                prototype_bank=prototype_bank,
            )
        except Exception as exc:
            _logger.warning(
                "apply_tta: Signal B raised exception on view",
                step="apply_tta",
                view_idx=view_idx,
                exc_info=exc,
            )
            sig_b = _failed_signal_b()

        lora_ok = sig_b.forward_succeeded
        lora_probs = sig_b.tomato_probs_canonical  # [6]
        per_view_lora_probs.append(lora_probs)
        per_view_lora_ok.append(lora_ok)
        per_view_lora_argmax.append(int(np.argmax(lora_probs)) if lora_ok else -1)
        if lora_ok:
            _last_b_result = sig_b

    # ------- Aggregate across views ----------------------------------------
    # spec: section 11.4 lines 3019-3030 — mean of softmax over surviving views
    agg_v3_probs, n_v3_ok = aggregate_views(per_view_v3_probs, per_view_v3_ok)
    agg_lora_probs, n_lora_ok = aggregate_views(per_view_lora_probs, per_view_lora_ok)

    # ------- Build updated SignalAResult with aggregated probs -------------
    v3_succeeded = n_v3_ok > 0
    aggregated_signal_a = SignalAResult(
        tomato_probs_canonical=agg_v3_probs,
        tomato_max_prob_canonical=float(agg_v3_probs.max()),
        tomato_argmax_canonical=int(agg_v3_probs.argmax()),
        # Keep chilli_leakage from the last successful view, or 0.0 if all failed.
        # spec: section 11.4 lines 3033-3035 — "aggregated outputs replace the 1-view
        #   outputs in the classifier's input"
        chilli_leakage=(
            _last_a_result.chilli_leakage if _last_a_result is not None else 0.0
        ),
        raw_probs_v3_order=(
            _last_a_result.raw_probs_v3_order if _last_a_result is not None else None
        ),
        forward_succeeded=v3_succeeded,
        failure_reason=None if v3_succeeded else "all_views_failed",
    )

    # ------- Build updated SignalBResult with aggregated probs -------------
    lora_succeeded = n_lora_ok > 0
    aggregated_signal_b = SignalBResult(
        tomato_probs_canonical=agg_lora_probs,
        tomato_max_prob_canonical=float(agg_lora_probs.max()),
        tomato_argmax_canonical=int(agg_lora_probs.argmax()),
        # CLS token from last succeeded view (monitoring/debug).
        cls_token=(
            _last_b_result.cls_token
            if _last_b_result is not None
            else np.zeros(768, dtype=np.float32)
        ),
        # raw_lora_probs_canonical: mean of raw (pre-blend) probs is most
        # informative post-TTA; we approximate with the aggregated probs since
        # mixing per-view blended and raw is complex and the spec doesn't specify.
        # Spec only says "aggregated outputs replace the 1-view outputs."
        # spec: section 11.4 lines 3033
        raw_lora_probs_canonical=agg_lora_probs.copy(),
        prototype_blend_applied=(
            _last_b_result.prototype_blend_applied
            if _last_b_result is not None
            else False
        ),
        prototype_blend_reason=(
            _last_b_result.prototype_blend_reason
            if _last_b_result is not None
            else "high_confidence_no_blend"
        ),
        forward_succeeded=lora_succeeded,
        failure_reason=None if lora_succeeded else "all_views_failed",
    )

    # ------- TTAReport -------------------------------------------------------
    # spec: section 11.6 lines 3079-3098
    v3_disagreement = _compute_view_disagreement(per_view_v3_argmax, per_view_v3_ok)
    lora_disagreement = _compute_view_disagreement(per_view_lora_argmax, per_view_lora_ok)

    report = TTAReport(
        triggered=(n_views > 1),
        n_views_attempted=n_views,
        n_views_succeeded_v3=n_v3_ok,
        n_views_succeeded_lora=n_lora_ok,
        initial_combined_max_prob=initial_combined_max_prob,
        # final_combined_max_prob is set by the orchestrator after re-running the
        # classifier; we initialize to NaN as a sentinel.
        # spec: section 11.6 line 3086 — "final_combined_max_prob: float"
        final_combined_max_prob=float("nan"),
        per_view_v3_argmax=per_view_v3_argmax,
        per_view_v3_succeeded=per_view_v3_ok,
        per_view_lora_argmax=per_view_lora_argmax,
        per_view_lora_succeeded=per_view_lora_ok,
        view_disagreement_v3=v3_disagreement,
        view_disagreement_lora=lora_disagreement,
    )

    _logger.debug(
        "apply_tta complete",
        step="apply_tta",
        n_views=n_views,
        n_v3_ok=n_v3_ok,
        n_lora_ok=n_lora_ok,
        v3_disagreement=round(v3_disagreement, 4),
        lora_disagreement=round(lora_disagreement, 4),
    )

    return aggregated_signal_a, aggregated_signal_b, report


# ---------------------------------------------------------------------------
# Internal failure-result helpers
# ---------------------------------------------------------------------------

def _failed_signal_a() -> SignalAResult:
    """Return a zero-filled SignalAResult with forward_succeeded=False."""
    zeros = np.zeros(_NUM_CLASSES, dtype=np.float32)
    return SignalAResult(
        tomato_probs_canonical=zeros.copy(),
        tomato_max_prob_canonical=0.0,
        tomato_argmax_canonical=0,
        chilli_leakage=0.0,
        raw_probs_v3_order=None,
        forward_succeeded=False,
        failure_reason="exception",
    )


def _failed_signal_b() -> SignalBResult:
    """Return a uniform-filled SignalBResult with forward_succeeded=False."""
    uniform = np.full(_NUM_CLASSES, 1.0 / _NUM_CLASSES, dtype=np.float32)
    return SignalBResult(
        tomato_probs_canonical=uniform.copy(),
        tomato_max_prob_canonical=float(uniform.max()),
        tomato_argmax_canonical=0,
        cls_token=np.zeros(768, dtype=np.float32),
        raw_lora_probs_canonical=uniform.copy(),
        prototype_blend_applied=False,
        prototype_blend_reason="high_confidence_no_blend",
        forward_succeeded=False,
        failure_reason="exception",
    )


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    "TTAReport",
    "should_trigger_tta",
    "build_augmentations",
    "apply_augmentation",
    "aggregate_views",
    "jensen_shannon_divergence",
    "apply_tta",
]
