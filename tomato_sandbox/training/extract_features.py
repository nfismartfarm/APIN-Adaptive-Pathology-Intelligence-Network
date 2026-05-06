"""
Feature extraction entry point for classifier weight training (T-PHASE-F0-CLASSIFIER Path (a)).

Composition of existing pipeline functions:
    validate → IQA gate → preprocess → signals A/B/C → build_classifier_input

Returns the 19-dimensional classifier input vector (standardized, clipped) for
each training image.  This module contains NO new signal logic — it delegates
entirely to tomato_sandbox sub-modules.

spec: section 12.2 lines 3137-3244 — 19-dim feature vector construction and
      degraded-mode zero-filling
spec: section 12.7 lines 3348-3373 — degraded-mode handling at inference time
      (same zero-fill logic applies here; training-time augmentation is Step 4's job)
spec: section 12.9 lines 3408-3442 — training procedure context
"""

from __future__ import annotations

import io
from typing import Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Module-level imports for all pipeline sub-functions.
# Placed here so that unit tests can patch at the module boundary
# (e.g. patch("tomato_sandbox.training.extract_features.compute_signal_a", ...)).
# Pattern mirrors orchestrator/pipeline.py lines 61-82.
# ---------------------------------------------------------------------------

from tomato_sandbox.utils.logging import get_logger

# IQA gate
from tomato_sandbox.iqa.iqa import compute_iqa, IQAResult

# Preprocessing pipelines (spec section 7, lines 1394-1563)
from tomato_sandbox.preprocessing.preprocess import (
    preprocess_for_v3,
    preprocess_for_lora,
    preprocess_for_psv,
)

# Signal functions
from tomato_sandbox.signals.v3_signal import compute_signal_a, SignalAResult
from tomato_sandbox.signals.lora_signal import compute_signal_b, SignalBResult
from tomato_sandbox.signals.psv.psv import compute_signal_c, SignalCResult

# Feature vector builder
from tomato_sandbox.classifier.feature_builder import build_classifier_input

_logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_features_for_training(
    image_bytes: bytes,
    context: "PipelineContext",  # noqa: F821 — imported below; forward ref for type hint
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Run validate → IQA → preprocess → signals → build_classifier_input.

    Composes existing pipeline functions in the same order as predict_single
    (tomato_sandbox/orchestrator/pipeline.py lines 356-700), stopping before
    the classifier forward pass.  No new logic is introduced here.

    spec: section 12.2 lines 3137-3244 — feature vector composition
    spec: section 12.9 lines 3408-3442 — training procedure context
    spec: section 12.7 lines 3364 — "At inference, signal failures are handled
          directly in build_classifier_input: the corresponding feature block is
          zeroed before standardization."

    Args:
        image_bytes: Raw bytes of a JPEG/PNG/WebP image.
        context: PipelineContext holding v3_model, lora_model, prototype_bank.
                 GPU lock is NOT used here (feature extraction is training-time
                 only; serialised by the training loop caller if needed).

    Returns:
        (feature_vector_19d, None) on success — float32 ndarray shape (19,).
        (None, error_reason)  on any failure:
            "decode_failed"       — PIL could not open the bytes
            "iqa_reject:<reason>" — IQA gate rejected the image
            "all_signals_failed"  — all three signals returned forward_succeeded=False
            "feature_vector_non_finite" — build_classifier_input returned NaN/Inf
    """
    # ------------------------------------------------------------------
    # Step 1: Decode image bytes → PIL RGB
    # Mirrors orchestrator/pipeline.py line 400 step 1 (spec 21.3 step 1)
    # ------------------------------------------------------------------
    try:
        from PIL import Image as _PIL_Image  # PIL is optional dep; import here
    except ImportError:
        _logger.error("PIL_not_available", step="decode")
        return None, "decode_failed"

    try:
        pil_image = _PIL_Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        _logger.warning("image_decode_failed", step="decode", exc_info=exc)
        return None, "decode_failed"

    # ------------------------------------------------------------------
    # Step 2: IQA gate
    # Mirrors orchestrator/pipeline.py lines 522-577 step 5 (spec 21.3 step 5)
    # BLK-013 / DEC-048 fix: wrap PIL image in _PILAdapter so compute_iqa
    # can access validated_image.pil_image as expected by spec 6.6 line 1374.
    # orchestrator/pipeline.py lines 528-535 — same adapter pattern.
    # ------------------------------------------------------------------

    class _PILAdapter:
        """Minimal adapter exposing .pil_image for compute_iqa.

        spec: section 6.6 line 1374 — compute_iqa(validated_image) -> IQAResult
              where validated_image exposes .pil_image
        orchestrator/pipeline.py lines 528-535 — same adapter (BLK-013 fix)
        """
        def __init__(self, pil):
            self.pil_image = pil

    try:
        iqa_result = compute_iqa(_PILAdapter(pil_image))
    except Exception as exc:
        # IQA failure during training: treat as ACCEPTABLE and continue.
        # Prefer keeping the sample over discarding it on IQA infrastructure errors.
        _logger.warning("iqa_failed_treating_as_acceptable", step="iqa", exc_info=exc)
        iqa_result = IQAResult(
            decision="ACCEPTABLE",
            aggregate_score=0.5,
            per_dimension={},
            failing_dimensions=[],
            retake_message=None,
            green_mask=None,
        )

    if iqa_result.decision == "REJECT":
        reason = iqa_result.retake_message or "image_quality_too_low"
        _logger.debug("iqa_rejected", step="iqa", reason=reason)
        return None, f"iqa_reject:{reason}"

    # ------------------------------------------------------------------
    # Step 3: Signal A (v3) — GPU-bound
    # Mirrors orchestrator/pipeline.py lines 582-622 step 6 (spec 21.3 step 6)
    # Device placement: move tensor to same device as v3_model.
    # DEC-058 fix: detect device via next(iter(model.parameters())).device
    # orchestrator/pipeline.py lines 590-594 — same device-detection pattern.
    # ------------------------------------------------------------------
    try:
        v3_tensor = preprocess_for_v3(pil_image)
        # Move to model device (orchestrator/pipeline.py lines 590-594)
        try:
            import torch as _torch
            _v3_device = next(context.v3_model.parameters()).device
            v3_tensor = v3_tensor.to(_v3_device)
        except Exception:
            pass  # best-effort; compute_signal_a will raise on device mismatch
        signal_a: SignalAResult = compute_signal_a(context.v3_model, v3_tensor)
    except Exception as exc:
        _logger.warning("signal_a_exception", step="signal_a", exc_info=exc)
        signal_a = SignalAResult(
            tomato_probs_canonical=np.zeros(6, dtype=np.float32),
            tomato_max_prob_canonical=0.0,
            tomato_argmax_canonical=0,
            chilli_leakage=0.0,
            raw_probs_v3_order=None,
            forward_succeeded=False,
            failure_reason=f"exception:{type(exc).__name__}",
        )

    # ------------------------------------------------------------------
    # Step 4: Signal B (LoRA) — GPU-bound
    # Mirrors orchestrator/pipeline.py lines 627-666 step 7 (spec 21.3 step 7)
    # Device placement: move tensor to same device as lora_model.
    # DEC-058 fix: detect device via next(iter(model.parameters())).device
    # orchestrator/pipeline.py lines 634-638 — same device-detection pattern.
    # ------------------------------------------------------------------
    try:
        lora_tensor = preprocess_for_lora(pil_image)
        try:
            import torch as _torch2
            lora_tensor_batched = lora_tensor.unsqueeze(0)  # [1, 3, 392, 392]
            _lora_device = next(context.lora_model.parameters()).device
            lora_tensor_batched = lora_tensor_batched.to(_lora_device)
        except Exception:
            lora_tensor_batched = lora_tensor
        signal_b: SignalBResult = compute_signal_b(
            lora_tensor_batched,
            context.lora_model,
            prototype_bank=context.prototype_bank,
        )
    except Exception as exc:
        _logger.warning("signal_b_exception", step="signal_b", exc_info=exc)
        _uni = np.full(6, 1.0 / 6, dtype=np.float32)
        signal_b = SignalBResult(
            tomato_probs_canonical=_uni,
            tomato_max_prob_canonical=float(_uni.max()),
            tomato_argmax_canonical=0,
            cls_token=np.zeros(768, dtype=np.float32),
            raw_lora_probs_canonical=_uni.copy(),
            prototype_blend_applied=False,
            prototype_blend_reason="high_confidence_no_blend",
            forward_succeeded=False,
            failure_reason=f"exception:{type(exc).__name__}",
        )

    # ------------------------------------------------------------------
    # Step 5: Signal C (PSV) — CPU-only, no GPU lock needed
    # Mirrors orchestrator/pipeline.py lines 695-730 step 8 (spec 21.3 step 8)
    # spec: section 10.2 — "CPU-only: no GPU API, no gpu_lock"
    # ------------------------------------------------------------------
    try:
        psv_rgb = preprocess_for_psv(pil_image)
        signal_c: SignalCResult = compute_signal_c(
            rgb_cc=psv_rgb,
            iqa_green_mask=iqa_result.green_mask,
            iqa_aggregate_score=iqa_result.aggregate_score,
        )
    except Exception as exc:
        _logger.warning("signal_c_exception", step="signal_c", exc_info=exc)
        _h, _w = np.array(pil_image).shape[:2]
        _uni_c = np.full(6, 1.0 / 6, dtype=np.float32)
        signal_c = SignalCResult(
            compatibility=_uni_c,
            compatibility_argmax=0,
            compatibility_max=float(_uni_c[0]),
            compatibility_margin=0.0,
            psv_reliability=0.05,
            raw_features=np.zeros(26, dtype=np.float32),
            standardized_features=np.zeros(26, dtype=np.float32),
            leaf_mask=np.zeros((_h, _w), dtype=bool),
            disease_mask=np.zeros((_h, _w), dtype=bool),
            n_lesions=0,
            fallback_used=False,
            forward_succeeded=False,
            failure_reason=f"exception:{type(exc).__name__}",
        )

    # ------------------------------------------------------------------
    # Step 6: All-signals-failed guard
    # Mirrors orchestrator/pipeline.py lines 742-779 step 8b (spec 21.5 lines 6745-6755)
    # At training time: discard this sample — a feature vector with no signal
    # information provides no useful gradient signal.
    # spec: section 12.7 lines 3348-3373 — degraded-mode handling
    # ------------------------------------------------------------------
    if not (
        signal_a.forward_succeeded
        or signal_b.forward_succeeded
        or signal_c.forward_succeeded
    ):
        _logger.warning("all_signals_failed", step="build_vector")
        return None, "all_signals_failed"

    # ------------------------------------------------------------------
    # Step 7: Build 19-dim classifier input vector
    # Delegates entirely to build_classifier_input which handles:
    #   - degraded-mode zero-filling (spec 12.2 lines 3231-3242)
    #   - JSD_SENTINEL when either neural signal failed (spec 12.2 line 3227)
    #   - standardization (spec 12.2 lines 3203-3204)
    #   - clip to [-3, 3] (spec 12.2 line 3204)
    # spec: section 12.2 lines 3212-3244 — authoritative build_classifier_input
    # Standardization happens INSIDE build_classifier_input; do NOT re-standardize.
    # ------------------------------------------------------------------
    feature_vector: np.ndarray = build_classifier_input(signal_a, signal_b, signal_c)

    # Invariant check: must be float32, shape (19,), finite
    assert feature_vector.dtype == np.float32, (
        f"build_classifier_input returned dtype {feature_vector.dtype}; expected float32"
    )
    assert feature_vector.shape == (19,), (
        f"build_classifier_input returned shape {feature_vector.shape}; expected (19,)"
    )
    # NaN/Inf guard: build_classifier_input clips to [-3,3] so NaN would indicate a bug
    if not np.isfinite(feature_vector).all():
        _logger.error(
            "feature_vector_non_finite",
            step="build_vector",
            n_bad=int((~np.isfinite(feature_vector)).sum()),
        )
        return None, "feature_vector_non_finite"

    _logger.debug(
        "extract_features_success",
        step="build_vector",
        signals_ok={
            "a": signal_a.forward_succeeded,
            "b": signal_b.forward_succeeded,
            "c": signal_c.forward_succeeded,
        },
    )
    return feature_vector, None


# ---------------------------------------------------------------------------
# IQA-bypassed sibling: for OOD synthetic noise + IQA-failure fallbacks
# Added in T-PHASE-F0-CLASSIFIER Step 3 (DEC-060 sub-decision).
# ---------------------------------------------------------------------------


def extract_features_no_iqa(
    pil_image: object,  # PIL.Image.Image — typed as object to avoid hard PIL import
    context: "PipelineContext",  # noqa: F821 — same forward ref as sibling above
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Sibling of extract_features_for_training that BYPASSES the IQA gate.

    Used for:
      - Synthetic noise OOD samples (Gaussian/solid/scrambled): they would
        fail IQA blur and wetness gates by construction; bypassing preserves
        their training-time purpose per spec S12.9 line 3440.
      - OOD okra/brassica fallbacks: when a folder exhausts its IQA-accepted
        pool, the residual draws are processed without the IQA gate to
        preserve folder diversity (capped at 20% per Refinement 2).

    Composition is identical to extract_features_for_training MINUS the IQA
    step.  signal_c receives iqa_green_mask=None and iqa_aggregate_score=0.0
    (the same defaults compute_signal_c sees when IQA infrastructure errors
    out — orchestrator/pipeline.py lines 540-546).

    spec: section 12.9 lines 3437-3441 — OOD class construction (noise allowed)
    spec: section 12.7 lines 3348-3373 — same degraded-mode zero-fill semantics

    Args:
        pil_image: A PIL.Image.Image already opened and converted to RGB.
                   Caller is responsible for I/O; this function does not
                   decode bytes (different from extract_features_for_training).
        context: Same PipelineContext used by extract_features_for_training.

    Returns:
        (feature_vector_19d, None) on success.
        (None, error_reason) on failure (all signals failed, non-finite vector).
    """
    # Step 1: Signal A (v3) — same device-placement pattern as sibling
    try:
        v3_tensor = preprocess_for_v3(pil_image)
        try:
            _v3_device = next(context.v3_model.parameters()).device
            v3_tensor = v3_tensor.to(_v3_device)
        except Exception:
            pass
        signal_a: SignalAResult = compute_signal_a(context.v3_model, v3_tensor)
    except Exception as exc:
        _logger.warning("signal_a_exception_no_iqa", step="signal_a", exc_info=exc)
        signal_a = SignalAResult(
            tomato_probs_canonical=np.zeros(6, dtype=np.float32),
            tomato_max_prob_canonical=0.0,
            tomato_argmax_canonical=0,
            chilli_leakage=0.0,
            raw_probs_v3_order=None,
            forward_succeeded=False,
            failure_reason=f"exception:{type(exc).__name__}",
        )

    # Step 2: Signal B (LoRA)
    try:
        lora_tensor = preprocess_for_lora(pil_image)
        try:
            lora_tensor_batched = lora_tensor.unsqueeze(0)
            _lora_device = next(context.lora_model.parameters()).device
            lora_tensor_batched = lora_tensor_batched.to(_lora_device)
        except Exception:
            lora_tensor_batched = lora_tensor
        signal_b: SignalBResult = compute_signal_b(
            lora_tensor_batched,
            context.lora_model,
            prototype_bank=context.prototype_bank,
        )
    except Exception as exc:
        _logger.warning("signal_b_exception_no_iqa", step="signal_b", exc_info=exc)
        _uni = np.full(6, 1.0 / 6, dtype=np.float32)
        signal_b = SignalBResult(
            tomato_probs_canonical=_uni,
            tomato_max_prob_canonical=float(_uni.max()),
            tomato_argmax_canonical=0,
            cls_token=np.zeros(768, dtype=np.float32),
            raw_lora_probs_canonical=_uni.copy(),
            prototype_blend_applied=False,
            prototype_blend_reason="high_confidence_no_blend",
            forward_succeeded=False,
            failure_reason=f"exception:{type(exc).__name__}",
        )

    # Step 3: Signal C (PSV) — IQA-bypass: pass green_mask=None, aggregate=0.0
    try:
        psv_rgb = preprocess_for_psv(pil_image)
        signal_c: SignalCResult = compute_signal_c(
            rgb_cc=psv_rgb,
            iqa_green_mask=None,  # bypassed; PSV computes its own leaf mask
            iqa_aggregate_score=0.0,
        )
    except Exception as exc:
        _logger.warning("signal_c_exception_no_iqa", step="signal_c", exc_info=exc)
        import numpy as _np
        _arr = _np.asarray(pil_image)
        _h, _w = _arr.shape[:2] if _arr.ndim >= 2 else (224, 224)
        _uni_c = np.full(6, 1.0 / 6, dtype=np.float32)
        signal_c = SignalCResult(
            compatibility=_uni_c,
            compatibility_argmax=0,
            compatibility_max=float(_uni_c[0]),
            compatibility_margin=0.0,
            psv_reliability=0.05,
            raw_features=np.zeros(26, dtype=np.float32),
            standardized_features=np.zeros(26, dtype=np.float32),
            leaf_mask=np.zeros((_h, _w), dtype=bool),
            disease_mask=np.zeros((_h, _w), dtype=bool),
            n_lesions=0,
            fallback_used=False,
            forward_succeeded=False,
            failure_reason=f"exception:{type(exc).__name__}",
        )

    # Step 4: All-signals-failed guard (same semantic as sibling)
    if not (
        signal_a.forward_succeeded
        or signal_b.forward_succeeded
        or signal_c.forward_succeeded
    ):
        _logger.warning("all_signals_failed_no_iqa", step="build_vector")
        return None, "all_signals_failed"

    # Step 5: Build 19-dim feature vector
    feature_vector: np.ndarray = build_classifier_input(signal_a, signal_b, signal_c)

    if not np.isfinite(feature_vector).all():
        _logger.error("feature_vector_non_finite_no_iqa", step="build_vector")
        return None, "feature_vector_non_finite"

    _logger.debug(
        "extract_features_no_iqa_success",
        step="build_vector",
        signals_ok={
            "a": signal_a.forward_succeeded,
            "b": signal_b.forward_succeeded,
            "c": signal_c.forward_succeeded,
        },
    )
    return feature_vector, None
