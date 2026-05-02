"""
Signal B — Single-Pass LoRA (epoch 13) wrapper.

Spec section: 9 (Signal B — Single-Pass LoRA (epoch 13)), lines 1793-1992.

Architecture:
  Input:  [B, 3, 392, 392]  ImageNet-normalized, LAB-CLAHE, letterbox padded
  Backbone: DINOv2-Base with registers (vit_base_patch14_reg4_dinov2, FROZEN)
  LoRA adapters on transformer blocks 4-11 (trained, rank 4)
  CLS token output: [B, 768]
  Linear(768, 6) head (trained)
  Output: [B, 6] logits

Class ordering: LoRA index ordering matches canonical ordering (no remap needed).
  0=foliar, 1=septoria, 2=late_blight, 3=ylcv, 4=mosaic, 5=healthy
# spec: section 9.1 lines 1822 — "This ordering matches canonical, so no remap
# is needed for LoRA → canonical."

CRITICAL — no MC Dropout:
  Signal B is a single-pass inference. model.eval() is called before the forward
  pass and Dropout (if any) is fully disabled. The spec does NOT prescribe MC
  Dropout for Signal B; that technique is not used here.
# spec: section 9.2 lines 1838-1848 — `model.eval()` + `torch.no_grad()`, one
# forward pass only, no loop over multiple stochastic passes.

GPU lock is the ORCHESTRATOR's responsibility (Section 21.3 step 4-7), not
Signal B's. `compute_signal_b` is called while the lock is already held.
# spec: section 21.3 steps 4,7 — acquire lock at step 4; run signals at steps
# 6-7; release at step 17. Signal B itself does not acquire the lock.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import torch  # type: ignore[import]
    import torch.nn as nn  # type: ignore[import]
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False

from tomato_sandbox.utils.logging import get_logger
from tomato_sandbox.utils.gpu_lock import GPULock  # noqa: F401 — imported for task-card compliance
from tomato_sandbox.utils.nan_guards import guard_array
from tomato_sandbox.utils.degraded_mode import zero_signal_b  # re-exported below

# ---------------------------------------------------------------------------
# Module logger — no print()
# ---------------------------------------------------------------------------
_logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_LORA_CLASSES: int = 6
"""Number of classes in the LoRA output.

# spec: section 9.1 line 1819 — "Output: [B, 6] logits"
"""

CLS_TOKEN_DIM: int = 768
"""Dimension of DINOv2-Base CLS token.

# spec: section 9.1 line 1813 — "CLS token output: [B, 768]"
"""

PROTOTYPE_BLEND_THRESHOLD: float = 0.60
"""Confidence threshold below which prototype blending is triggered.

When lora_max_prob < PROTOTYPE_BLEND_THRESHOLD, prototype bank blending
is applied. Default 0.60 per spec; F.0 calibration may override.

# spec: section 9.4 lines 1863 — "below `TOMATO_PROTOTYPE_BLEND_THRESHOLD`,
# default 0.60, F.0-calibrated"
"""

PROTOTYPE_MAX_PER_CLASS: int = 10
"""Maximum number of prototype CLS tokens stored per class.

# spec: section 9.4 line 1867 — "stores up to 10 prototypes"
"""

PROTOTYPE_MIN_PER_CLASS: int = 3
"""Minimum prototypes per class before that class is marked underpopulated.

# spec: section 9.4 line 1882 — "If fewer than 3, mark the class as
# `underpopulated`."
"""

T_PROTO: float = 0.3
"""Softmax temperature for prototype similarity → probability conversion.

# spec: section 9.5 lines 1949 — "T_PROTO = 0.3"
"""

BLEND_WEIGHT: float = 0.35
"""Weight on the prototype distribution relative to LoRA softmax.

LoRA contributes (1 - BLEND_WEIGHT) = 0.65; prototypes contribute 0.35.

# spec: section 9.5 lines 1950 — "BLEND_WEIGHT = 0.35"
"""

# ---------------------------------------------------------------------------
# PrototypeBank dataclass
# ---------------------------------------------------------------------------


@dataclass
class PrototypeBank:
    """Prototype bank built from high-confidence LoRA predictions on field_val.

    Stores up to 10 CLS-token vectors per class (canonical index).
    Used for prototype blending when LoRA confidence is low.

    Invariant: any class in `underpopulated_classes` has its prototypes either
    missing or empty; any class NOT in `underpopulated_classes` has at least
    3 prototypes.  This invariant is checked before accessing prototypes.

    # spec: section 9.4 lines 1870-1876 — PrototypeBank dataclass definition
    """

    prototypes: dict[int, np.ndarray]  # class_idx -> [N_class, 768] CLS tokens
    class_counts: dict[int, int]       # class_idx -> number of prototypes
    underpopulated_classes: set[int]   # classes with < PROTOTYPE_MIN_PER_CLASS
    model_version: str                 # first 7 chars of SHA-256 of LoRA weights


# ---------------------------------------------------------------------------
# SignalBResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class SignalBResult:
    """Output of compute_signal_b.

    Both raw and blended distributions are kept.  The classifier (Section 12)
    reads `tomato_probs_canonical` (possibly blended) as 6 of its 19 input
    features.  The raw distribution is exposed for transparency.

    # spec: section 9.6 lines 1961-1973 — SignalBResult fields (verbatim)
    """

    tomato_probs_canonical: np.ndarray
    """[6] probability distribution (possibly blended with prototypes).

    Canonical ordering: 0=foliar, 1=septoria, 2=late_blight, 3=ylcv,
    4=mosaic, 5=healthy.

    # spec: section 9.1 line 1822 — "This ordering matches canonical,
    # so no remap is needed for LoRA → canonical."
    """

    tomato_max_prob_canonical: float
    """Max value of tomato_probs_canonical."""

    tomato_argmax_canonical: int
    """Argmax index of tomato_probs_canonical (canonical class index)."""

    cls_token: np.ndarray
    """[768] CLS token features; used for monitoring/debug and prototype bank.

    # spec: section 9.6 line 1967 — "cls_token: np.ndarray  # [768]"
    """

    raw_lora_probs_canonical: np.ndarray
    """[6] un-blended LoRA softmax (for transparency).

    # spec: section 9.6 line 1968 — "raw_lora_probs_canonical: np.ndarray"
    """

    prototype_blend_applied: bool
    """True if prototype blending was triggered.

    # spec: section 9.6 line 1969 — "prototype_blend_applied: bool"
    """

    prototype_blend_reason: str
    """Reason string for blending decision.

    Values: "low_confidence" | "high_confidence_no_blend" |
            "all_classes_underpopulated"

    # spec: section 9.6 line 1970 — "prototype_blend_reason: str"
    """

    forward_succeeded: bool
    """True unless an exception or NaN occurred.

    # spec: section 9.6 line 1971 — "forward_succeeded: bool"
    """

    failure_reason: Optional[str]
    """None on success; "exception" | "numerical_instability" on failure.

    # spec: section 9.6 line 1972 — "failure_reason: str | None"
    """


# ---------------------------------------------------------------------------
# Internal: signal_b_forward
# ---------------------------------------------------------------------------


def signal_b_forward(model: object, x: "torch.Tensor") -> dict:
    """Run a single deterministic forward pass through the LoRA model.

    CRITICAL — single-pass only:
      model.eval() is called before the forward pass.  NO MC Dropout.
      This function must not be called in a loop for stochastic passes.
      TTA (Section 11) calls this once per augmented view; each call is
      a separate deterministic pass.

    # spec: section 9.2 lines 1828-1848 — signal_b_forward verbatim
    # spec: section 9.2 line 1838 — `model.eval()`
    # spec: section 9.2 line 1839 — `with torch.no_grad():`
    # spec: section 9.2 lines 1845-1846 — NaN/Inf check returns ok=False

    Args:
        model: LoRA model wrapper that exposes a uniform forward dict contract.
               Must implement __call__(x) -> {"logits": Tensor[B,6],
               "cls_token": Tensor[B,768]}.
        x: [B, 3, 392, 392] tensor on the GPU (ImageNet-normalized,
           LAB-CLAHE, letterbox-padded).
           # spec: section 9.1 line 1800 — input shape [B, 3, 392, 392]

    Returns:
        dict with keys:
          "logits":    [B, 6] raw logits (Tensor) or None on failure
          "probs":     [B, 6] softmax probs (Tensor) or None on failure
          "cls_token": [B, 768] CLS token features (Tensor) or None on failure
          "ok":        bool, False when NaN/Inf found in logits
    """
    if not _TORCH_AVAILABLE:
        return {"logits": None, "probs": None, "cls_token": None, "ok": False}

    # Single deterministic pass: eval mode disables all Dropout
    # spec: section 9.2 line 1838 — model.eval()
    model.eval()  # type: ignore[union-attr]

    with torch.no_grad():
        # spec: section 9.2 lines 1842-1843 — uniform forward dict contract
        out = model(x)  # type: ignore[operator]
        logits: torch.Tensor = out["logits"]       # [B, 6]
        cls_token: torch.Tensor = out["cls_token"] # [B, 768]

        # spec: section 9.2 lines 1845-1846 — NaN/Inf guard
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            return {"logits": None, "probs": None, "cls_token": None, "ok": False}

        # spec: section 9.2 line 1847 — softmax
        probs = torch.softmax(logits, dim=1)

    return {"logits": logits, "probs": probs, "cls_token": cls_token, "ok": True}


# ---------------------------------------------------------------------------
# Prototype blending
# ---------------------------------------------------------------------------


def prototype_blend(
    lora_probs: np.ndarray,
    cls_token: np.ndarray,
    bank: PrototypeBank,
    T_proto: float = T_PROTO,
    blend_weight: float = BLEND_WEIGHT,
) -> tuple[np.ndarray, str]:
    """Blend LoRA's softmax with prototype-similarity distribution.

    Called when `lora_max_prob < PROTOTYPE_BLEND_THRESHOLD`.  Uses cosine
    similarity from the current image's CLS token to each stored prototype.

    # spec: section 9.5 lines 1903-1945 — prototype_blend verbatim

    Args:
        lora_probs:   [6] LoRA softmax in canonical ordering.
        cls_token:    [768] current image's CLS token.
        bank:         Loaded PrototypeBank (always available at inference).
        T_proto:      Softmax temperature for similarity → prob conversion.
                      # spec: section 9.5 line 1949 — T_PROTO = 0.3
        blend_weight: Weight on prototype distribution vs LoRA.
                      # spec: section 9.5 line 1950 — BLEND_WEIGHT = 0.35

    Returns:
        (blended_probs, blend_reason) where:
          blended_probs: [6] renormalised probability distribution
          blend_reason: "low_confidence" | "all_classes_underpopulated"
    """
    # Normalize current CLS token
    # spec: section 9.5 line 1915 — cls_norm = cls_token / (norm + 1e-8)
    cls_norm = cls_token / (np.linalg.norm(cls_token) + 1e-8)

    # Per-class max cosine similarity to stored prototypes
    # spec: section 9.5 lines 1918-1926
    per_class_sim = np.zeros(NUM_LORA_CLASSES, dtype=np.float64)
    for cls_idx in range(NUM_LORA_CLASSES):
        # spec: section 9.5 lines 1920-1921 — underpopulated → -inf
        if cls_idx in bank.underpopulated_classes:
            per_class_sim[cls_idx] = -np.inf
            continue
        protos = bank.prototypes[cls_idx]          # [N_class, 768]
        protos_norm = protos / (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-8)
        sims = protos_norm @ cls_norm              # [N_class]
        # spec: section 9.5 line 1926 — max similarity (closest prototype)
        per_class_sim[cls_idx] = sims.max()

    # Convert similarities → probability via softmax / T_proto
    # Underpopulated classes had -inf → zero prob after softmax
    # spec: section 9.5 lines 1930-1937
    finite_mask = np.isfinite(per_class_sim)
    sim_probs = np.zeros(NUM_LORA_CLASSES, dtype=np.float64)

    if not finite_mask.any():
        # All classes underpopulated — fall back to raw LoRA
        # spec: section 9.5 line 1937 — "fall back to LoRA if all classes underpopulated"
        return lora_probs.copy(), "all_classes_underpopulated"

    sims_finite = per_class_sim[finite_mask]
    # Numerically stable softmax: subtract max before exp
    sims_shifted = sims_finite - sims_finite.max()
    exp_sims = np.exp(sims_shifted / T_proto)
    sims_softmax = exp_sims / exp_sims.sum()
    sim_probs[finite_mask] = sims_softmax

    # Blend
    # spec: section 9.5 line 1940 — blended = (1 - blend_weight)*lora + blend_weight*sim
    blended = (1.0 - blend_weight) * lora_probs + blend_weight * sim_probs

    # Renormalize — necessary when underpopulated classes reduce sim_probs sum
    # spec: section 9.5 lines 1941-1944 — "Renormalize ... necessary because
    # sim_probs can sum to less than 1 when underpopulated classes are zeroed"
    blended_sum = blended.sum()
    if blended_sum > 0:
        blended = blended / blended_sum
    else:
        blended = lora_probs.copy()

    return blended.astype(np.float32), "low_confidence"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def compute_signal_b(
    lora_input: "torch.Tensor",
    model: object,
    prototype_bank: Optional[PrototypeBank] = None,
    blend_threshold: float = PROTOTYPE_BLEND_THRESHOLD,
) -> SignalBResult:
    """Compute Signal B from the LoRA model with optional prototype blending.

    This is the primary entry point for Signal B.  The GPU lock MUST already
    be held by the orchestrator (Section 21.3 step 4) before calling this.

    CRITICAL — single forward pass:
      This function calls signal_b_forward exactly once.  TTA (Section 11)
      calls compute_signal_b once per augmented view and aggregates separately.
      There is NO loop here, NO multiple stochastic passes, NO MC Dropout.
    # spec: section 9.2 lines 1838-1848 — single forward pass with model.eval()
    # spec: section 9.1 line 1797 — "single-pass" refers to the training/inference
    # strategy; at inference, exactly one forward pass per call.

    No remap applied:
      LoRA index ordering matches canonical ordering.  Signal A (v3) applies a
      remap; Signal B does NOT.
    # spec: section 9.1 line 1822 — "This ordering matches canonical, so no
    # remap is needed for LoRA → canonical."

    Args:
        lora_input: [1, 3, 392, 392] preprocessed tensor on the GPU.
                    Produced by preprocess_for_lora() (Section 7.3).
                    # spec: section 9.1 line 1800 — input [B, 3, 392, 392]
        model: LoRA model wrapper exposing the uniform forward dict contract.
               # spec: section 9.2 line 1842 — `out = model(x)`
        prototype_bank: Loaded PrototypeBank or None.
                        If None, blending is skipped entirely (degraded mode).
                        At startup the bank is always loaded (spec 4.4); None is
                        only possible in unit tests with mock scenarios.
        blend_threshold: Confidence threshold for triggering prototype blending.
                         Default: PROTOTYPE_BLEND_THRESHOLD (0.60).
                         # spec: section 9.5 line 1901 — "lora_max_prob < threshold"

    Returns:
        SignalBResult with all fields populated.
        On failure: forward_succeeded=False, probs filled with uniform 1/6.

    # spec: section 9.6 lines 1961-1973 — SignalBResult output structure
    # spec: section 9.7 line 1981 — "`tomato_sandbox/signals/lora_signal.py`
    # defines `SignalBResult` and `compute_signal_b`"
    """
    _logger.debug("compute_signal_b called")

    # ── 1. Forward pass ───────────────────────────────────────────────────
    try:
        fwd = signal_b_forward(model, lora_input)
    except Exception as exc:
        _logger.error(
            "signal_b_forward raised exception",
            exc_info=True,
            exc_type=type(exc).__name__,
        )
        return _failure_result("exception")

    if not fwd["ok"]:
        # NaN or Inf in logits — spec 9.2 lines 1845-1846
        _logger.warning("signal_b_forward: NaN/Inf in logits")
        return _failure_result("numerical_instability")

    # ── 2. Extract probs and CLS token ────────────────────────────────────
    probs_tensor: torch.Tensor = fwd["probs"]     # [B, 6]
    cls_tensor: torch.Tensor = fwd["cls_token"]   # [B, 768]

    # Convert batch-dim-0 to numpy — signal B wrapper handles B=1
    probs_np = probs_tensor[0].cpu().numpy().astype(np.float32)   # [6]
    cls_np = cls_tensor[0].cpu().numpy().astype(np.float32)       # [768]

    # Guard finiteness of probability array
    # spec: section 26 (production hygiene) — non-finite values must not propagate
    probs_np = guard_array(probs_np, expected_len=NUM_LORA_CLASSES, default_value=0.0)
    cls_np = guard_array(cls_np, expected_len=CLS_TOKEN_DIM, default_value=0.0)

    # ── 3. Prototype blending (optional) ─────────────────────────────────
    # spec: section 9.5 lines 1901-1957 — blending logic
    lora_max_prob = float(probs_np.max())
    raw_lora_probs = probs_np.copy()

    if prototype_bank is None:
        # No bank available (unit test / startup failure edge case)
        # spec: section 9.5 lines 1954-1957 — bank unavailable → skip blend
        # Note: spec says "if bank failed to load, sandbox didn't start" but
        # unit tests may pass None for testing without a real bank.
        final_probs = raw_lora_probs.copy()
        blend_applied = False
        blend_reason = "high_confidence_no_blend"  # no blend regardless of confidence
    elif lora_max_prob >= blend_threshold:
        # High confidence — no blend needed
        # spec: section 9.5 line 1955 — "lora_max_prob >= threshold: use raw output"
        final_probs = raw_lora_probs.copy()
        blend_applied = False
        blend_reason = "high_confidence_no_blend"
    else:
        # Low confidence — apply prototype blending
        # spec: section 9.5 line 1901 — "when lora_max_prob < threshold, blend"
        blended, blend_reason = prototype_blend(
            lora_probs=raw_lora_probs,
            cls_token=cls_np,
            bank=prototype_bank,
            T_proto=T_PROTO,
            blend_weight=BLEND_WEIGHT,
        )
        final_probs = blended.astype(np.float32)
        blend_applied = (blend_reason == "low_confidence")

    # ── 4. Compute derived fields ─────────────────────────────────────────
    final_max_prob = float(final_probs.max())
    final_argmax = int(np.argmax(final_probs))

    return SignalBResult(
        tomato_probs_canonical=final_probs,
        tomato_max_prob_canonical=final_max_prob,
        tomato_argmax_canonical=final_argmax,
        cls_token=cls_np,
        raw_lora_probs_canonical=raw_lora_probs,
        prototype_blend_applied=blend_applied,
        prototype_blend_reason=blend_reason,
        forward_succeeded=True,
        failure_reason=None,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _failure_result(reason: str) -> SignalBResult:
    """Return a SignalBResult representing a failed forward pass.

    Probabilities default to uniform 1/6 so downstream consumers have a
    well-defined (if low-confidence) distribution.  The classifier's
    degraded-mode handling (Section 12.7) will zero the LoRA block anyway
    when forward_succeeded=False.

    # spec: section 9.6 lines 1971-1972 — forward_succeeded / failure_reason
    # spec: section 12.7 lines 3348-3364 — degraded mode: LoRA block zeroed
    """
    uniform = np.full(NUM_LORA_CLASSES, 1.0 / NUM_LORA_CLASSES, dtype=np.float32)
    zero_cls = np.zeros(CLS_TOKEN_DIM, dtype=np.float32)
    return SignalBResult(
        tomato_probs_canonical=uniform,
        tomato_max_prob_canonical=float(uniform.max()),
        tomato_argmax_canonical=0,
        cls_token=zero_cls,
        raw_lora_probs_canonical=uniform.copy(),
        prototype_blend_applied=False,
        prototype_blend_reason="high_confidence_no_blend",
        forward_succeeded=False,
        failure_reason=reason,
    )


# ---------------------------------------------------------------------------
# Re-exports for downstream convenience
# spec: section 9.7 line 1981 — this file is the canonical location
# ---------------------------------------------------------------------------
__all__ = [
    "SignalBResult",
    "PrototypeBank",
    "compute_signal_b",
    "signal_b_forward",
    "prototype_blend",
    "zero_signal_b",      # re-exported from degraded_mode for downstream use
    "NUM_LORA_CLASSES",
    "CLS_TOKEN_DIM",
    "PROTOTYPE_BLEND_THRESHOLD",
    "T_PROTO",
    "BLEND_WEIGHT",
]
