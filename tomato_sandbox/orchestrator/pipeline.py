"""
Pipeline orchestrator for the Tomato 3-Signal sandbox.

spec: Section 21 (Pipeline orchestrator), lines 6604-6861

Canonical sandbox path: tomato_sandbox/orchestrator/pipeline.py
  (spec 21.1 line 6608 names this file)

The orchestrator is the integration glue that drives a single prediction request
through all pipeline stages in order. It has no signal-processing logic of its
own — it delegates to existing modules.

Function signatures (spec 21.2 lines 6614-6640):
    predict_single(image_bytes, request_id, context) -> dict
    predict_multi(images, request_id, context) -> dict  [stub — S18 task]

No print() anywhere. All output via get_logger().

Key behavioural contracts enforced here:
  - GPU lock held around Signal A and Signal B only (spec 21.3 steps 4-17)
  - Signal C (PSV) runs CPU-only, never within GPU lock (spec 10.2 / S21.3 step 8)
  - TTA: PSV NOT re-run (spec 11.1 line 2925, 11.9 lines 3139-3140)
  - Signal failure → degraded mode via utils.degraded_mode (spec 21.5, 12.7)
  - All-signals-failed → short-circuit to sentinel 4B (spec 21.5 lines 6745-6755)
  - NaN guard at classifier output boundary (spec 21.4 lines 6679-6710)
  - GPU lock timeout → SERVER_OVERLOAD error dict (spec 21.8 / 20.6)

DEC-042: orchestrator placement at tomato_sandbox/orchestrator/pipeline.py (spec-named)
  with re-export shim at orchestrator/__init__.py and flat shim at
  orchestrator/orchestrator.py (task-card path alias per DEC-033).
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

import numpy as np

try:
    from PIL import Image as _PIL_Image
    _PIL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PIL_AVAILABLE = False

from tomato_sandbox.utils.logging import get_logger
from tomato_sandbox.utils.gpu_lock import GPULock, GPULockTimeoutError
from tomato_sandbox.utils.nan_guards import guard_array
from tomato_sandbox.utils.degraded_mode import (
    apply_degraded_mode,
    zero_signal_a,
    zero_signal_b,
    zero_signal_c,
    zeros_vector,
    VECTOR_DIM,
)

# Signal modules
from tomato_sandbox.signals.v3_signal import compute_signal_a, SignalAResult
from tomato_sandbox.signals.lora_signal import compute_signal_b, SignalBResult
from tomato_sandbox.signals.psv.psv import compute_signal_c, SignalCResult
from tomato_sandbox.signals.tta import should_trigger_tta, apply_tta

# Preprocessing
from tomato_sandbox.preprocessing.preprocess import (
    preprocess_for_v3,
    preprocess_for_lora,
    preprocess_for_psv,
)

# Downstream pipeline modules
from tomato_sandbox.classifier.hierarchical_classifier import (
    compute_classifier,
    ClassifierResult,
)
from tomato_sandbox.conformal.conformal import compute_conformal_set, ConformalResult
from tomato_sandbox.tier.tier_assignment import assign_tier, TierAssignment

# IQA — imported at module level so patch("...pipeline.compute_iqa") works in tests
from tomato_sandbox.iqa.iqa import compute_iqa, IQAResult

_logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# PipelineContext
# spec: section 21.2 lines 6623-6629
# ---------------------------------------------------------------------------


@dataclass
class PipelineContext:
    """Holds all models and configuration for the pipeline.

    spec: section 21.2 lines 6623-6629 — PipelineContext fields
    """
    v3_model: Any = None             # spec: 21.2 line 6624
    lora_model: Any = None           # spec: 21.2 line 6624
    psv_module: Any = None           # spec: 21.2 line 6624  (currently unused: PSV is function-based)
    classifier: Any = None           # spec: 21.2 line 6624  (loaded calibration; compute_classifier loads lazily)
    iqa_module: Any = None           # spec: 21.2 line 6624
    conformal_calibration: Any = None  # spec: 21.2 line 6625
    iqa_thresholds: Any = None       # spec: 21.2 line 6625
    severity_thresholds: Any = None  # spec: 21.2 line 6625
    gpu_lock: Optional[GPULock] = None  # spec: 21.2 line 6626
    cache: Optional[dict] = None     # spec: 21.2 line 6627 (request cache)
    metrics: Any = None              # spec: 21.2 line 6628 (Prometheus counters)
    phase_e_logger: Any = None       # spec: 21.2 line 6629 (Section 24)
    prototype_bank: Any = None       # LoRA prototype bank for signal B blending
    underpowered_classes: Optional[set] = None  # F.0 flagged classes


# ---------------------------------------------------------------------------
# Sentinel classifier result
# spec: section 21.5 lines 6749-6755 — all-signals-failed short-circuit
# ---------------------------------------------------------------------------


def _make_sentinel_classifier_result(reason: str = "all_signals_failed") -> ClassifierResult:
    """Return a sentinel ClassifierResult when all signals fail.

    spec: section 21.5 lines 6749-6755 — "skip classifier forward pass entirely,
    set classifier output to a sentinel 'all signals failed' marker, route to Tier 4B"
    """
    zeros7 = np.zeros(7, dtype=np.float32)
    zeros3 = np.zeros(3, dtype=np.float32)
    zeros5 = np.zeros(5, dtype=np.float32)
    return ClassifierResult(
        p_final_calibrated=zeros7,
        combined_argmax=0,
        combined_max_prob=0.0,
        combined_margin=0.0,
        p_final_uncalibrated=zeros7,
        p_stage1=zeros3,
        p_stage2=zeros5,
        classifier_succeeded=False,
        failure_reason=reason,
    )


# ---------------------------------------------------------------------------
# NaN guard at classifier output boundary
# spec: section 21.4 lines 6679-6710
# ---------------------------------------------------------------------------


def _apply_nan_guard(
    classifier_result: ClassifierResult,
    signal_a: SignalAResult,
    signal_b: SignalBResult,
    signal_c: SignalCResult,
) -> tuple[ClassifierResult, SignalAResult, SignalBResult, SignalCResult]:
    """Detect NaN in classifier output; mark all signals failed if found.

    spec: section 21.4 lines 6684-6710 — apply_nan_guard pseudocode

    If any classifier output is NaN, mark all signals as failed.
    This forces tier rule chain to fire Rule 1 (pipeline failure → 4B)
    instead of falling through to Rule 9 (catch-all → 4A) on NaN inputs.
    """
    has_nan = (
        np.isnan(classifier_result.combined_max_prob)
        or np.isnan(classifier_result.combined_margin)
        or np.any(np.isnan(classifier_result.p_final_calibrated))
    )
    if has_nan:
        # spec: section 21.4 lines 6698-6706 — mark all signals failed, zero classifier
        from dataclasses import replace  # immutable-friendly approach per spec note
        signal_a = SignalAResult(
            tomato_probs_canonical=signal_a.tomato_probs_canonical,
            tomato_max_prob_canonical=signal_a.tomato_max_prob_canonical,
            tomato_argmax_canonical=signal_a.tomato_argmax_canonical,
            chilli_leakage=signal_a.chilli_leakage,
            raw_probs_v3_order=signal_a.raw_probs_v3_order,
            forward_succeeded=False,   # spec: 21.4 line 6699
            failure_reason="nan_in_classifier_output",
        )
        signal_b = SignalBResult(
            tomato_probs_canonical=signal_b.tomato_probs_canonical,
            tomato_max_prob_canonical=signal_b.tomato_max_prob_canonical,
            tomato_argmax_canonical=signal_b.tomato_argmax_canonical,
            cls_token=signal_b.cls_token,
            raw_lora_probs_canonical=signal_b.raw_lora_probs_canonical,
            prototype_blend_applied=signal_b.prototype_blend_applied,
            prototype_blend_reason=signal_b.prototype_blend_reason,
            forward_succeeded=False,   # spec: 21.4 line 6700
            failure_reason="nan_in_classifier_output",
        )
        signal_c = SignalCResult(
            compatibility=signal_c.compatibility,
            compatibility_argmax=signal_c.compatibility_argmax,
            compatibility_max=signal_c.compatibility_max,
            compatibility_margin=signal_c.compatibility_margin,
            psv_reliability=signal_c.psv_reliability,
            raw_features=signal_c.raw_features,
            standardized_features=signal_c.standardized_features,
            leaf_mask=signal_c.leaf_mask,
            disease_mask=signal_c.disease_mask,
            n_lesions=signal_c.n_lesions,
            fallback_used=signal_c.fallback_used,
            forward_succeeded=False,   # spec: 21.4 line 6701
            failure_reason="nan_in_classifier_output",
        )
        # Zero out NaN fields in classifier result so tier rules get safe defaults
        # spec: section 21.4 lines 6702-6704
        from dataclasses import replace as _replace
        classifier_result = ClassifierResult(
            p_final_calibrated=classifier_result.p_final_calibrated,
            combined_argmax=classifier_result.combined_argmax,
            combined_max_prob=0.0,    # spec: 21.4 line 6703
            combined_margin=0.0,      # spec: 21.4 line 6704
            p_final_uncalibrated=classifier_result.p_final_uncalibrated,
            p_stage1=classifier_result.p_stage1,
            p_stage2=classifier_result.p_stage2,
            classifier_succeeded=False,
            failure_reason="nan_in_classifier_output",
        )
        _logger.warning(
            "nan_in_classifier_output",
            # spec: section 21.4 line 6705-6707
            extra={"has_nan": True},
        )
    return classifier_result, signal_a, signal_b, signal_c


# ---------------------------------------------------------------------------
# Signal result → assign_tier dict adapters
# spec: section 14.8 — assign_tier keyword dict shapes
# ---------------------------------------------------------------------------


def _signal_a_to_dict(sa: SignalAResult) -> dict:
    """Convert SignalAResult to the dict shape expected by assign_tier.

    spec: section 14 / import_contract.md — v3_signal dict keys:
        "probs" (list[float] len 6), "chilli_leak" (float), "forward_succeeded" (bool)
    """
    return {
        "probs": sa.tomato_probs_canonical.tolist(),
        "chilli_leak": sa.chilli_leakage,
        "forward_succeeded": sa.forward_succeeded,
    }


def _signal_b_to_dict(sb: SignalBResult) -> dict:
    """Convert SignalBResult to the dict shape expected by assign_tier.

    spec: section 14 / import_contract.md — lora_signal dict keys:
        "probs" (list[float] len 6), "forward_succeeded" (bool)
    """
    return {
        "probs": sb.tomato_probs_canonical.tolist(),
        "forward_succeeded": sb.forward_succeeded,
    }


def _signal_c_to_dict(sc: SignalCResult) -> dict:
    """Convert SignalCResult to the dict shape expected by assign_tier.

    spec: section 14 / import_contract.md — psv_signal dict keys:
        "argmax", "max", "margin", "reliability", "forward_succeeded"
    """
    return {
        "argmax": sc.compatibility_argmax,
        "max": sc.compatibility_max,
        "margin": sc.compatibility_margin,
        "reliability": sc.psv_reliability,
        "forward_succeeded": sc.forward_succeeded,
    }


def _classifier_to_dict(cr: ClassifierResult) -> dict:
    """Convert ClassifierResult to the dict shape expected by assign_tier.

    spec: import_contract.md — classifier dict keys:
        "argmax", "max" (combined_max_prob), "margin"
    """
    return {
        "argmax": cr.combined_argmax,
        "max": cr.combined_max_prob,
        "margin": cr.combined_margin,
    }


def _conformal_to_dict(cfr: ConformalResult) -> dict:
    """Convert ConformalResult to the dict shape expected by assign_tier.

    spec: import_contract.md — conformal dict keys:
        "set" (set[int]), "size" (int), "tau" (float | None)
    """
    return {
        "set": set(cfr.prediction_set),
        "size": cfr.prediction_set_size,
        "tau": cfr.threshold_tau_used,
    }


def _iqa_to_dict(iqa_result: Any) -> dict:
    """Convert IQAResult to the dict shape expected by assign_tier.

    spec: import_contract.md — iqa dict keys:
        "decision" (str: "ACCEPTABLE" | "DEGRADED" | "HIGH")
    """
    if hasattr(iqa_result, "decision"):
        return {"decision": iqa_result.decision}
    # Fallback if raw dict passed
    return {"decision": iqa_result.get("decision", "ACCEPTABLE")}


# ---------------------------------------------------------------------------
# Helper: compute sha256 image hash
# spec: section 21.3 step 2 lines 6649
# ---------------------------------------------------------------------------

def _image_hash(data: bytes) -> str:
    """Compute SHA256 hex digest of image bytes. spec: 21.3 step 2 line 6649."""
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Error response builders
# spec: section 21.8 lines 6810-6826 — error categories
# ---------------------------------------------------------------------------


def _error_response(
    error_code: str,
    message: str,
    request_id: str,
    status: int = 400,
    extra: Optional[dict] = None,
) -> dict:
    """Build a structured error response dict.

    spec: section 21.8 lines 6810-6826 — pre-tier errors return error responses
    without a tier label.
    """
    resp = {
        "error": error_code,
        "message": message,
        "request_id": request_id,
        "status": status,
    }
    if extra:
        resp.update(extra)
    return resp


# ---------------------------------------------------------------------------
# predict_single — main pipeline entry point
# spec: section 21.2 lines 6614-6621, 21.3 lines 6644-6677
# ---------------------------------------------------------------------------


def predict_single(
    image_bytes: bytes,
    request_id: str,
    context: PipelineContext,
) -> dict:
    """Run the full pipeline for a single image and return the API response dict.

    spec: section 21.2 lines 6614-6621 — function signature
    spec: section 21.3 lines 6644-6677 — step-by-step orchestration

    Pipeline steps (spec 21.3):
      1.  Decode image bytes → PIL RGB (on failure: IMAGE_DECODE_FAILED)
      2.  Compute image_hash (sha256)
      3.  Check request cache (on hit: return cached response)
      4.  Acquire GPU lock (on timeout: SERVER_OVERLOAD 503)
      5.  IQA gate (on REJECT: release lock, IQA_REJECTED 422)
      6.  Signal A (v3) — try/except
      7.  Signal B (LoRA) — try/except
      8.  Signal C (PSV) — CPU-only, after lock-guarded A+B
      8b. Degraded-mode: zero failed signal blocks in feature vector
      9.  Build feature vector for classifier
      10. Classifier forward pass
      11. NaN guard on classifier output
      12. TTA trigger check — if fires, aggregate and re-run classifier
      13. Conformal prediction set
      14. Tier assignment
      (15-22: severity, GradCAM++, lock release, response build, log, cache — stubs)
    """
    t_start = time.perf_counter()

    # ------------------------------------------------------------------
    # Step 1: Decode image bytes → PIL RGB
    # spec: section 21.3 step 1 lines 6647-6648
    # ------------------------------------------------------------------
    if not _PIL_AVAILABLE:
        return _error_response(
            "IMAGE_DECODE_FAILED",
            "PIL not available; cannot decode image.",
            request_id,
            status=400,
        )

    try:
        import io
        pil_image = _PIL_Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        _logger.error(
            "image_decode_failed",
            step="decode",
            request_id=request_id,
            exc_info=exc,
        )
        # spec: section 21.3 step 1 line 6648 — IMAGE_DECODE_FAILED
        return _error_response(
            "IMAGE_DECODE_FAILED",
            f"Could not decode image: {exc}",
            request_id,
            status=400,
        )

    _logger.debug(
        "image_decoded",
        step="decode",
        request_id=request_id,
        size=f"{pil_image.width}x{pil_image.height}",
        succeeded=True,
    )

    # ------------------------------------------------------------------
    # Step 2: Compute image hash
    # spec: section 21.3 step 2 line 6649
    # ------------------------------------------------------------------
    image_hash = _image_hash(image_bytes)
    _logger.debug("image_hash_computed", step="hash", request_id=request_id)

    # ------------------------------------------------------------------
    # Step 3: Check request cache
    # spec: section 21.3 step 3 lines 6650 / 21.10 lines 6845-6858
    # ------------------------------------------------------------------
    if context.cache is not None:
        cached = context.cache.get(image_hash)
        if cached is not None:
            _logger.debug(
                "cache_hit",
                step="cache_check",
                request_id=request_id,
                image_hash=image_hash[:8],
            )
            # spec: 21.10 line 6858 — add from_cache: true to warnings
            cached_response = dict(cached)
            warnings = list(cached_response.get("warnings", []))
            if "from_cache" not in warnings:
                warnings.append("from_cache")
            cached_response["warnings"] = warnings
            cached_response["processing_time_ms"] = round(
                (time.perf_counter() - t_start) * 1000, 1
            )
            return cached_response

    _logger.debug("cache_miss", step="cache_check", request_id=request_id)

    # ------------------------------------------------------------------
    # Step 4: Acquire GPU lock
    # spec: section 21.3 step 4 line 6651 / 20.6 lines 6579-6589
    # spec: section 21.8 line 6820 — GPU lock timeout → 503 SERVER_OVERLOAD
    # ------------------------------------------------------------------
    gpu_lock: Optional[GPULock] = context.gpu_lock

    # Track whether THIS code path acquired the lock (for matching release).
    # See cross-loop asyncio.Lock comment below + step-17 release block.
    acquired_locally: bool = False

    try:
        if gpu_lock is not None:
            import asyncio
            # spec: section 21.3 step 4 — "Acquire GPU lock"
            # Cross-loop asyncio.Lock semantics (DEC-045 + Batch 7 smoke-test fix):
            # The server holds the lock in its FastAPI event loop, then dispatches
            # this orchestrator call via run_in_executor (worker thread, no loop).
            # The asyncio.Lock instance is bound to the server's loop; trying to
            # re-acquire it from a fresh asyncio.run() in the worker thread hangs
            # forever because the lock's internal Future is on a different loop.
            #
            # Resolution: if no running loop AND the lock is already locked, assume
            # the caller (FastAPI handler) holds it and skip re-acquisition. This is
            # safe because the server holds the lock for the full duration of the
            # executor call (per server.py's `async with gpu_lock.acquired()`).
            # Sync-context unit tests still work: lock is unlocked, acquire normally.
            try:
                running = asyncio.get_running_loop()
                _ = running  # async context — caller holds lock; skip
            except RuntimeError:
                # No running loop in this thread.
                # Note: GPULock.locked is a @property, not a method (gpu_lock.py:185).
                if gpu_lock.locked:
                    # Lock already held (in another loop, by the FastAPI handler).
                    # Assume server.py holds it for our call duration; skip.
                    pass
                else:
                    # Lock is unlocked — sync test or standalone use; acquire it.
                    asyncio.run(gpu_lock.acquire_with_timeout())
                    acquired_locally = True
    except GPULockTimeoutError as exc:
        _logger.warning(
            "gpu_lock_timeout",
            step="acquire_lock",
            request_id=request_id,
            timeout_s=exc.timeout_s,
        )
        # spec: section 21.8 line 6820 — 503 SERVER_OVERLOAD
        return _error_response(
            "SERVER_OVERLOAD",
            "Server is busy. Please retry after a few seconds.",
            request_id,
            status=503,
            extra={"retry_after_seconds": 5},
        )

    lock_held = gpu_lock is not None

    try:
        # ------------------------------------------------------------------
        # Step 5: IQA gate
        # spec: section 21.3 step 5 lines 6652-6654
        # ------------------------------------------------------------------
        iqa_result = None
        try:
            # compute_iqa expects an object with a .pil_image attribute (spec 6.6 line 1374).
            # The orchestrator receives a raw PIL.Image from the decode step; wrap it in a
            # minimal adapter so compute_iqa can access validated_image.pil_image.
            # BLK-013 / DEC-048 — raw PIL passed here caused AttributeError inside compute_iqa
            # which returned REJECT(0.0) for every real image regardless of quality.
            class _PILAdapter:
                """Minimal ValidatedImage-shaped wrapper for compute_iqa.

                spec: section 6.6 line 1374 — compute_iqa expects an object with .pil_image attr.
                BLK-013 / DEC-048 — orchestrator was passing raw PIL.Image; now wraps before call.
                """
                def __init__(self, pil):
                    self.pil_image = pil

            iqa_result = compute_iqa(_PILAdapter(pil_image))
        except Exception as exc:
            _logger.warning(
                "iqa_failed_using_acceptable",
                step="iqa",
                request_id=request_id,
                exc_info=exc,
            )
            # IQA failure: treat as ACCEPTABLE so pipeline continues
            iqa_result = IQAResult(
                decision="ACCEPTABLE",
                aggregate_score=0.5,
                per_dimension={},
                failing_dimensions=[],
                retake_message=None,
                green_mask=None,
            )

        _logger.debug(
            "iqa_complete",
            step="iqa",
            request_id=request_id,
            decision=iqa_result.decision,
            aggregate_score=round(iqa_result.aggregate_score, 3),
            succeeded=True,
        )

        if iqa_result.decision == "REJECT":
            # spec: section 21.3 step 5 line 6653 — release lock, IQA_REJECTED 422
            _logger.info(
                "iqa_rejected",
                step="iqa",
                request_id=request_id,
                decision="REJECT",
            )
            return _error_response(
                "IQA_REJECTED",
                iqa_result.retake_message or "Image quality too low.",
                request_id,
                status=422,
            )

        # ------------------------------------------------------------------
        # Step 6: Signal A (v3) — GPU-bound, lock already held
        # spec: section 21.3 step 6 line 6655
        # ------------------------------------------------------------------
        t_a = time.perf_counter()
        try:
            v3_tensor = preprocess_for_v3(pil_image)
            # Move tensor to the same device as the v3 model.
            # preprocess_for_v3 returns a CPU tensor; the model is on CUDA.
            # signal_a_forward doc: "[B, 3, 224, 224] tensor on the correct device"
            try:
                import torch as _torch_dev
                _v3_device = next(context.v3_model.parameters()).device
                v3_tensor = v3_tensor.to(_v3_device)
            except Exception:
                pass  # best-effort; signal_a_forward will raise on mismatch
            signal_a: SignalAResult = compute_signal_a(context.v3_model, v3_tensor)
        except Exception as exc:
            _logger.error(
                "signal_a_exception",
                step="signal_a",
                request_id=request_id,
                exc_info=exc,
                succeeded=False,
            )
            from tomato_sandbox.signals.v3_signal import SignalAResult as _SAR
            import numpy as _np
            signal_a = _SAR(
                tomato_probs_canonical=_np.zeros(6, dtype=_np.float32),
                tomato_max_prob_canonical=0.0,
                tomato_argmax_canonical=0,
                chilli_leakage=0.0,
                raw_probs_v3_order=None,
                forward_succeeded=False,
                failure_reason=f"exception:{type(exc).__name__}",
            )

        _logger.debug(
            "signal_a_complete",
            step="signal_a",
            duration_ms=round((time.perf_counter() - t_a) * 1000, 1),
            succeeded=signal_a.forward_succeeded,
            details={"max_prob": round(signal_a.tomato_max_prob_canonical, 4)},
        )

        # ------------------------------------------------------------------
        # Step 7: Signal B (LoRA) — GPU-bound, lock already held
        # spec: section 21.3 step 7 line 6656
        # ------------------------------------------------------------------
        t_b = time.perf_counter()
        try:
            lora_tensor = preprocess_for_lora(pil_image)
            try:
                import torch as _torch
                lora_tensor_batched = lora_tensor.unsqueeze(0)  # [1,3,392,392]
                # Move tensor to same device as the LoRA model.
                # preprocess_for_lora returns a CPU tensor; model is on CUDA.
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
            _logger.error(
                "signal_b_exception",
                step="signal_b",
                request_id=request_id,
                exc_info=exc,
                succeeded=False,
            )
            from tomato_sandbox.signals.lora_signal import SignalBResult as _SBR
            import numpy as _np2
            _uni = _np2.full(6, 1.0 / 6, dtype=_np2.float32)
            signal_b = _SBR(
                tomato_probs_canonical=_uni,
                tomato_max_prob_canonical=float(_uni.max()),
                tomato_argmax_canonical=0,
                cls_token=_np2.zeros(768, dtype=_np2.float32),
                raw_lora_probs_canonical=_uni.copy(),
                prototype_blend_applied=False,
                prototype_blend_reason="high_confidence_no_blend",
                forward_succeeded=False,
                failure_reason=f"exception:{type(exc).__name__}",
            )

        _logger.debug(
            "signal_b_complete",
            step="signal_b",
            duration_ms=round((time.perf_counter() - t_b) * 1000, 1),
            succeeded=signal_b.forward_succeeded,
            details={"max_prob": round(signal_b.tomato_max_prob_canonical, 4)},
        )

    finally:
        # ------------------------------------------------------------------
        # Step 17 (early): Release GPU lock after signals A and B
        # spec: section 21.3 step 17 line 6669
        # Only release if THIS code path acquired the lock (sync test path).
        # In production, server.py holds the lock via `async with gpu_lock.acquired()`
        # for the full executor-call duration; the server's __aexit__ releases it.
        # Releasing here would either no-op (lock not actually held) or cross-loop fail.
        # ------------------------------------------------------------------
        if acquired_locally:
            try:
                gpu_lock.release()
            except Exception:
                pass  # Best-effort release

    # ------------------------------------------------------------------
    # Step 8: Signal C (PSV) — CPU-only, OUTSIDE GPU lock
    # spec: section 21.3 step 8 line 6657 — "try/except per Section 7"
    # spec: section 10.2 — "CPU-only: no GPU API, no gpu_lock"
    # ------------------------------------------------------------------
    t_c = time.perf_counter()
    try:
        psv_rgb = preprocess_for_psv(pil_image)
        signal_c: SignalCResult = compute_signal_c(
            rgb_cc=psv_rgb,
            iqa_green_mask=iqa_result.green_mask,
            iqa_aggregate_score=iqa_result.aggregate_score,
        )
    except Exception as exc:
        _logger.error(
            "signal_c_exception",
            step="signal_c",
            request_id=request_id,
            exc_info=exc,
            succeeded=False,
        )
        import numpy as _np3
        _h, _w = np.array(pil_image).shape[:2]
        from tomato_sandbox.signals.psv.psv import SignalCResult as _SCR
        _uni_c = _np3.full(6, 1.0 / 6, dtype=_np3.float32)
        signal_c = _SCR(
            compatibility=_uni_c,
            compatibility_argmax=0,
            compatibility_max=float(_uni_c[0]),
            compatibility_margin=0.0,
            psv_reliability=0.05,
            raw_features=_np3.zeros(26, dtype=_np3.float32),
            standardized_features=_np3.zeros(26, dtype=_np3.float32),
            leaf_mask=_np3.zeros((_h, _w), dtype=bool),
            disease_mask=_np3.zeros((_h, _w), dtype=bool),
            n_lesions=0,
            fallback_used=False,
            forward_succeeded=False,
            failure_reason=f"exception:{type(exc).__name__}",
        )

    _logger.debug(
        "signal_c_complete",
        step="signal_c",
        duration_ms=round((time.perf_counter() - t_c) * 1000, 1),
        succeeded=signal_c.forward_succeeded,
        details={"reliability": round(signal_c.psv_reliability, 4)},
    )

    # ------------------------------------------------------------------
    # Step 8b: All-signals-failed short-circuit
    # spec: section 21.5 lines 6745-6755 — skip classifier on all-zero input
    # ------------------------------------------------------------------
    if not (
        signal_a.forward_succeeded
        or signal_b.forward_succeeded
        or signal_c.forward_succeeded
    ):
        _logger.warning(
            "all_signals_failed",
            step="degraded_mode",
            request_id=request_id,
        )
        # spec: section 21.5 lines 6749-6753 — set sentinel classifier result
        classifier_result = _make_sentinel_classifier_result("all_signals_failed")
        conformal_result = _make_fallback_conformal()
        tier_result = assign_tier(
            v3_signal=_signal_a_to_dict(signal_a),
            lora_signal=_signal_b_to_dict(signal_b),
            psv_signal=_signal_c_to_dict(signal_c),
            classifier=_classifier_to_dict(classifier_result),
            conformal=_conformal_to_dict(conformal_result),
            iqa=_iqa_to_dict(iqa_result),
            underpowered_classes=context.underpowered_classes,
        )
        return _build_pipeline_result(
            request_id=request_id,
            signal_a=signal_a,
            signal_b=signal_b,
            signal_c=signal_c,
            iqa_result=iqa_result,
            classifier_result=classifier_result,
            conformal_result=conformal_result,
            tier_result=tier_result,
            tta_fired=False,
            t_start=t_start,
            image_hash=image_hash,
            context=context,
        )

    # ------------------------------------------------------------------
    # Step 8b (continued): Apply degraded mode for any individual failed signals
    # spec: section 21.5 lines 6713-6743 — zero failed signal blocks
    # spec: section 12.7 lines 3348-3373 — degraded mode details
    # Note: degraded_mode.apply_degraded_mode operates on the feature vector,
    # not on the signal result objects. The classifier's build_classifier_input
    # handles this internally via the forward_succeeded flags on the signal results.
    # We do NOT need to manually zero feature vectors here; compute_classifier
    # receives the signal results and applies degraded mode internally per spec 12.7.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Steps 9-11: Classifier + calibration + NaN guard (first pass, 1-view)
    # spec: section 21.3 steps 9-11 lines 6658-6662
    # ------------------------------------------------------------------
    t_clf = time.perf_counter()
    try:
        classifier_result: ClassifierResult = compute_classifier(
            sa=signal_a,
            sb=signal_b,
            sc=signal_c,
        )
    except Exception as exc:
        _logger.error(
            "classifier_exception",
            step="classifier",
            request_id=request_id,
            exc_info=exc,
        )
        classifier_result = _make_sentinel_classifier_result(
            f"exception:{type(exc).__name__}"
        )

    # spec: section 21.4 — NaN guard at classifier output boundary
    classifier_result, signal_a, signal_b, signal_c = _apply_nan_guard(
        classifier_result, signal_a, signal_b, signal_c
    )

    _logger.debug(
        "classifier_complete",
        step="classifier",
        duration_ms=round((time.perf_counter() - t_clf) * 1000, 1),
        succeeded=classifier_result.classifier_succeeded,
        details={
            "combined_max_prob": round(classifier_result.combined_max_prob, 4),
            "combined_argmax": classifier_result.combined_argmax,
        },
    )

    # ------------------------------------------------------------------
    # Step 12: TTA trigger check
    # spec: section 21.3 step 12 lines 6663-6664
    # spec: section 11.2 — should_trigger_tta uses combined_max_prob
    # spec: section 11.1 line 2925 — PSV NOT invoked during TTA
    # ------------------------------------------------------------------
    tta_fired = False
    n_views = should_trigger_tta(classifier_result.combined_max_prob)
    # spec: section 11.2 lines 2932-2939 — returns 1, 2, or 5
    if n_views > 1:
        tta_fired = True
        _logger.info(
            "tta_triggered",
            step="tta",
            request_id=request_id,
            n_views=n_views,
            initial_max_prob=round(classifier_result.combined_max_prob, 4),
        )
        # spec: section 21.3 step 12 — re-run Signals A and B on augmented views
        # spec: section 11.9 lines 3139-3140 — PSV NOT invoked during TTA
        try:
            agg_signal_a, agg_signal_b, tta_report = apply_tta(
                pil_image=pil_image,
                n_views=n_views,
                v3_model=context.v3_model,
                lora_model=context.lora_model,
                prototype_bank=context.prototype_bank,
                initial_combined_max_prob=classifier_result.combined_max_prob,
            )
        except Exception as exc:
            _logger.error(
                "tta_exception",
                step="tta",
                request_id=request_id,
                exc_info=exc,
            )
            agg_signal_a = signal_a   # keep original on TTA failure
            agg_signal_b = signal_b

        # spec: section 21.3 step 12 line 6664 — re-run classifier with TTA-aggregated features
        # Signal C (PSV) unchanged — keep original single-view result
        try:
            classifier_result = compute_classifier(
                sa=agg_signal_a,
                sb=agg_signal_b,
                sc=signal_c,  # PSV unchanged — spec: 11.1 "PSV does NOT participate in TTA"
            )
        except Exception as exc:
            _logger.error(
                "classifier_post_tta_exception",
                step="classifier_tta",
                request_id=request_id,
                exc_info=exc,
            )
            classifier_result = _make_sentinel_classifier_result(
                f"tta_classifier_exception:{type(exc).__name__}"
            )

        # Re-apply NaN guard after TTA re-run
        # spec: section 21.3 step 12 line 6664 — "re-applying calibration and NaN guard"
        classifier_result, agg_signal_a, agg_signal_b, signal_c = _apply_nan_guard(
            classifier_result, agg_signal_a, agg_signal_b, signal_c
        )
        # Update signal_a, signal_b to TTA-aggregated versions for downstream
        signal_a = agg_signal_a
        signal_b = agg_signal_b

        _logger.debug(
            "tta_complete",
            step="tta",
            request_id=request_id,
            n_views=n_views,
            final_max_prob=round(classifier_result.combined_max_prob, 4),
            succeeded=True,
        )

    # ------------------------------------------------------------------
    # Step 13: Conformal prediction set
    # spec: section 21.3 step 13 line 6665
    # ------------------------------------------------------------------
    t_conf = time.perf_counter()
    try:
        conformal_result: ConformalResult = compute_conformal_set(
            classifier_result.p_final_calibrated
        )
    except Exception as exc:
        _logger.error(
            "conformal_exception",
            step="conformal",
            request_id=request_id,
            exc_info=exc,
        )
        conformal_result = _make_fallback_conformal()

    _logger.debug(
        "conformal_complete",
        step="conformal",
        duration_ms=round((time.perf_counter() - t_conf) * 1000, 1),
        succeeded=True,
        details={"prediction_set": conformal_result.prediction_set,
                 "size": conformal_result.prediction_set_size},
    )

    # ------------------------------------------------------------------
    # Step 14: Tier assignment
    # spec: section 21.3 step 14 line 6666 — assign_tier per Section 14.8
    # ------------------------------------------------------------------
    t_tier = time.perf_counter()
    tier_result: TierAssignment = assign_tier(
        v3_signal=_signal_a_to_dict(signal_a),
        lora_signal=_signal_b_to_dict(signal_b),
        psv_signal=_signal_c_to_dict(signal_c),
        classifier=_classifier_to_dict(classifier_result),
        conformal=_conformal_to_dict(conformal_result),
        iqa=_iqa_to_dict(iqa_result),
        underpowered_classes=context.underpowered_classes,
    )

    _logger.debug(
        "tier_assignment_complete",
        step="tier_assignment",
        duration_ms=round((time.perf_counter() - t_tier) * 1000, 1),
        tier_label=tier_result.tier_label,
        tier5_alert=tier_result.tier5_alert,
        rule_id_fired=tier_result.rule_id_fired,
        succeeded=True,
    )

    # ------------------------------------------------------------------
    # Steps 15-22: severity, GradCAM++, response build, log, cache
    # spec: section 21.3 steps 15-22 lines 6667-6674
    # Note: GradCAM++ (step 16) and severity (step 15) are Batch 6b/6c work.
    # Response building (step 18) is Batch 6b work. Stubs placed here.
    # ------------------------------------------------------------------
    result = _build_pipeline_result(
        request_id=request_id,
        signal_a=signal_a,
        signal_b=signal_b,
        signal_c=signal_c,
        iqa_result=iqa_result,
        classifier_result=classifier_result,
        conformal_result=conformal_result,
        tier_result=tier_result,
        tta_fired=tta_fired,
        t_start=t_start,
        image_hash=image_hash,
        context=context,
    )

    # spec: section 21.3 step 21 lines 6673 — cache response
    if context.cache is not None:
        context.cache[image_hash] = result

    return result


# ---------------------------------------------------------------------------
# predict_multi — multi-image stub
# spec: section 21.2 lines 6631-6640, 21.6 lines 6757-6784
# Note: full implementation is Batch 6c / Section 18 work (T-IMPL-6c)
# ---------------------------------------------------------------------------


def predict_multi(
    images: list,
    request_id: str,
    context: PipelineContext,
) -> dict:
    """Run pipeline for each image, then aggregate (Section 18).

    spec: section 21.2 lines 6631-6640 — signature
    spec: section 21.6 lines 6757-6784 — multi-image orchestration

    Note: Full aggregation logic is T-IMPL-6c work. This stub runs predict_single
    per image and returns a basic per-image list for integration testing.
    """
    _logger.debug(
        "predict_multi_called",
        step="predict_multi",
        request_id=request_id,
        n_images=len(images),
    )

    per_image_results = []
    for idx, img in enumerate(images):
        # Each element may be bytes or a (bytes, image_id) tuple
        if isinstance(img, tuple):
            img_bytes, img_id = img
        else:
            img_bytes = img
            img_id = str(idx)

        img_request_id = f"{request_id}__img{idx}"
        try:
            result = predict_single(img_bytes, img_request_id, context)
        except Exception as exc:
            _logger.error(
                "predict_multi_single_failed",
                step="predict_multi",
                request_id=request_id,
                img_idx=idx,
                exc_info=exc,
            )
            result = _error_response(
                "IMAGE_PIPELINE_FAILED",
                str(exc),
                img_request_id,
                status=500,
            )
        per_image_results.append({"image_id": img_id, "result": result})

    # spec: section 21.6 line 6777 — aggregated = aggregate_results(per_image_results)
    # Aggregation delegated to Batch 6c / multi_image module
    return {
        "request_id": request_id,
        "n_images": len(images),
        "per_image_results": per_image_results,
        "aggregated": None,  # Populated by T-IMPL-6c aggregator
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_fallback_conformal() -> ConformalResult:
    """Return a fallback ConformalResult when conformal fails.

    Default: all 7 classes in set (widest possible set, maximally conservative).
    spec: section 13 — conformal always returns a valid set; failure defaults to all-in.
    """
    return ConformalResult(
        prediction_set=list(range(7)),
        prediction_set_size=7,
        threshold_tau_used=1.0,
        nonconformity_per_class=np.zeros(7, dtype=np.float32),
        inside_set_per_class=np.ones(7, dtype=bool),
    )


def _build_pipeline_result(
    *,
    request_id: str,
    signal_a: SignalAResult,
    signal_b: SignalBResult,
    signal_c: SignalCResult,
    iqa_result: Any,
    classifier_result: ClassifierResult,
    conformal_result: ConformalResult,
    tier_result: TierAssignment,
    tta_fired: bool,
    t_start: float,
    image_hash: str,
    context: PipelineContext,
) -> dict:
    """Assemble the final pipeline result dict using build_response().

    spec: section 21.3 steps 18-22 lines 6670-6674
    spec: section 16.1 lines 5643-5644 — build_response is a pure function
    DEC-045 Decision 5: wired to build_response() from response_builder.

    After build_response() returns the 14-key spec-compliant dict, severity
    is computed via compute_severity() and merged into the 'severity' block.
    """
    import datetime as _dt

    processing_ms = round((time.perf_counter() - t_start) * 1000, 1)

    # spec: section 21.9 lines 6832-6843 — structured log at each step
    _logger.info(
        "pipeline_complete",
        step="pipeline",
        request_id=request_id,
        tier_label=tier_result.tier_label,
        tier5_alert=tier_result.tier5_alert,
        rule_id_fired=tier_result.rule_id_fired,
        combined_max_prob=round(classifier_result.combined_max_prob, 4),
        tta_fired=tta_fired,
        processing_ms=processing_ms,
        succeeded=True,
    )

    # Build request_metadata dict for build_response()
    # spec: section 16.1 line 5651 — request_id, image_hash, timestamp, client_version
    # DEC-045 Decision 5
    timestamp_iso = _dt.datetime.utcnow().isoformat() + "Z"
    request_metadata = {
        "request_id": request_id,
        "image_hash": image_hash,
        "timestamp_iso": timestamp_iso,
        "processing_time_ms": int(processing_ms),
        "client_version": "tomato-sandbox-v1.0.0",
    }

    # Build spec-compliant 14-key response using response builder
    # spec: section 16.2 (full schema); DEC-045 Decision 5
    # BLK-014 fix (DEC-049): pass signal_extra so explanation.structured can
    # report chilli_leakage_actual and psv_reliability_actual.
    # spec: section 16.4 lines 5765-5766
    _chilli_leakage_val = float(
        signal_a.chilli_leakage if hasattr(signal_a, "chilli_leakage") else 0.0
    )
    _psv_reliability_val = float(
        signal_c.psv_reliability if hasattr(signal_c, "psv_reliability") else 0.0
    )
    from tomato_sandbox.response.response_builder import build_response
    response = build_response(
        tier_result,
        classifier_result,
        conformal_result,
        iqa_result,
        request_metadata=request_metadata,
        model_version="tomato-sandbox-v1.0.0",
        signal_extra={
            "chilli_leakage_actual": _chilli_leakage_val,
            "psv_reliability_actual": _psv_reliability_val,
        },
    )

    # Merge severity into response dict's 'severity' block
    # spec: section 16.2 lines 5680-5684 — "populated per Section 17"
    # spec: section 17.7 — severity omitted for 4A/4B, healthy, OOD, low PSV
    # DEC-045 Decision 5
    try:
        from tomato_sandbox.severity.grader import compute_severity
        psv_reliability = (
            signal_c.psv_reliability
            if hasattr(signal_c, "psv_reliability")
            else 0.0
        )
        raw_features = (
            signal_c.raw_features
            if hasattr(signal_c, "raw_features")
            else None
        )
        # BLK-015 fix (DEC-050): pass multi_class_set for Tier 3A/3B so that
        # grade_per_class is populated per spec 17.5 lines 6015-6032.
        # spec: section 17.5 — multi-class severity for conformal prediction sets
        _multi_cls = (
            list(conformal_result.prediction_set)
            if tier_result.tier_label in ("3A", "3B")
            else None
        )
        sev_result = compute_severity(
            predicted_class=classifier_result.combined_argmax,
            raw_features=raw_features,
            psv_reliability=float(psv_reliability),
            tier_label=tier_result.tier_label,
            multi_class_set=_multi_cls,
        )
        # Merge computed severity into the spec severity block
        if sev_result.grade is not None:
            sev_details: dict = {
                "disease_coverage_pct": sev_result.disease_coverage_pct,
                "lesion_count": sev_result.lesion_count,
                "psv_confidence_in_severity": sev_result.psv_confidence_in_severity,
                "thresholds_used": sev_result.thresholds_used,
            }
            response["severity"] = {
                "grade": sev_result.grade,
                "human_readable": sev_result.human_readable,
                "details": sev_details,
            }
            # BLK-015: include grade_per_class when populated (Tier 3A/3B)
            # spec: 17.5 lines 6015-6032
            if sev_result.grade_per_class is not None:
                response["severity"]["grade_per_class"] = sev_result.grade_per_class
        else:
            # grade=None means severity omitted; keep null block from builder
            response["severity"] = {
                "grade": None,
                "human_readable": sev_result.human_readable,
                "details": None,
            }
    except Exception as _sev_exc:
        # Severity failure must never abort the response — log and leave null
        _logger.warning(
            "severity_computation_failed",
            request_id=request_id,
            error=str(_sev_exc),
        )
        # response["severity"] already null from build_response

    # Attach tta_fired as a warning (not a spec field but useful for debugging)
    if tta_fired:
        response.setdefault("warnings", [])
        if "TTA was triggered for this request." not in response["warnings"]:
            response["warnings"].append("TTA was triggered for this request.")

    return response
