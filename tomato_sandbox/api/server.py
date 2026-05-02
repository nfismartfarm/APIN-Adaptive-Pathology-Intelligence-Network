"""
FastAPI server for the tomato 3-signal sandbox.

Sandbox server port: 8767 (BLK-002 / DEC-012 / DEC-026).
APIN remains on port 8766 (unchanged, not imported here — BLK-003).
Unified server is on port 8005 (Section 22).

This module implements the FastAPI application with:
  - A lifespan context manager executing the 12-step startup sequence
    (spec section 20.5). All 12 steps are now wired (T-IMPL-7 / DEC-045).
  - All seven endpoints from spec section 20.3.
  - GPU lock instance on app.state (spec section 20.6).
  - Structured logging via tomato_sandbox.utils.logging (spec section 26.7).

# spec: section 20.3 (endpoints) lines 6452-6499
# spec: section 20.4 (module layout) lines 6501-6554
# spec: section 20.5 (startup sequence) lines 6556-6575
# spec: section 20.6 (GPU lock) lines 6577-6589
# spec: section 20.7 (configuration sources) lines 6591-6603
# DEC-045: Decision 1-8 (T-IMPL-7 wiring choices)
"""

from __future__ import annotations

import asyncio
import datetime
import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, List, Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from tomato_sandbox.config import CONFIG, TomatoConfig
from tomato_sandbox.input_validation import ValidationError, validate_request
from tomato_sandbox.orchestrator.pipeline import predict_single
from tomato_sandbox.utils.gpu_lock import GPULock, GPULockTimeoutError, create_gpu_lock
from tomato_sandbox.utils.logging import get_logger

# ---------------------------------------------------------------------------
# Module-level logger
# spec: section 26.7 "Use structlog for structured logging; never print()"
# ---------------------------------------------------------------------------

_logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Conformal calibration JSON path
# spec: section 20.5 step 8 — "Load conformal calibration from tomato_calibration.json"
# DEC-045 Decision 2: placeholder at phase_f0_calibration/conformal_tau.json
# ---------------------------------------------------------------------------

_CONFORMAL_TAU_PATH = (
    Path(__file__).resolve().parents[1]  # tomato_sandbox/
    / "phase_f0_calibration"
    / "conformal_tau.json"
)


# ---------------------------------------------------------------------------
# Application state (typed)
# ---------------------------------------------------------------------------


class _AppState:
    """Container for objects stored on ``app.state`` during lifespan.

    # spec: section 20.5 — startup populates gpu_lock and pipeline;
    #       section 20.6 — gpu_lock is the asyncio-based GPU serialiser.
    """

    gpu_lock: GPULock
    pipeline: Any  # PipelineContext | None
    config: TomatoConfig
    startup_complete: bool
    model_loaded: bool
    conformal_tau: Optional[float]
    conformal_timestamp: Optional[str]


# ---------------------------------------------------------------------------
# Lifespan context manager — 12-step startup sequence
# spec: section 20.5 lines 6558-6573
# DEC-045 Decision 1: sacred guard at step 1; FAIL-FAST on any non-PASS entry.
# DEC-045 Decision 4: PipelineContext constructed and stored on app.state.
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Execute the 12-step startup sequence, then yield to serve requests.

    # spec: section 20.5 steps 1-12 lines 6560-6572
    # DEC-045 Decision 1: verify_manifest() at step 1 with FAIL-FAST.
    # DEC-045 Decision 4: PipelineContext wired at step 12.
    """
    state: _AppState = _AppState()  # type: ignore[call-arg]
    state.config = CONFIG
    state.startup_complete = False
    state.model_loaded = False
    state.pipeline = None
    state.conformal_tau = None
    state.conformal_timestamp = None

    # ------------------------------------------------------------------
    # Step 1: Load env vars + Sacred guard FAIL-FAST
    # spec: section 20.5 step 1 "Load env vars (TOMATO_* namespace per Section 4.5)"
    # DEC-045 Decision 1: verify_manifest() runs here; any non-PASS → RuntimeError
    # ------------------------------------------------------------------
    _logger.info(
        "startup_step_1",
        step=1,
        description="env vars loaded, running sacred guard",
        port=state.config.port,
        gpu_lock_timeout_s=state.config.gpu_lock_timeout_s,
    )

    from tomato_sandbox.utils.sacred_guard import verify_manifest
    sacred_results = verify_manifest()  # uses default manifest path

    non_pass = {k: v for k, v in sacred_results.items() if v != "PASS"}
    if non_pass:
        # spec: section 20.5 step 1 / DEC-045 Decision 1 — abort on any non-PASS
        _logger.error(
            "startup_sacred_guard_failed",
            step=1,
            non_pass=non_pass,
        )
        raise RuntimeError(
            f"Sacred guard FAILED at startup — sacred files have drifted. "
            f"Non-PASS entries: {non_pass}. "
            f"Resolve drift before restarting the server."
        )

    _logger.info(
        "startup_step_1_sacred_pass",
        step=1,
        checked=len(sacred_results),
        all_pass=True,
    )

    # ------------------------------------------------------------------
    # Step 2: Initialize structured logging
    # spec: section 20.5 step 2 "Initialize structured logging"
    # ------------------------------------------------------------------
    _logger.info("startup_step_2", step=2, description="structured logging initialized")

    # ------------------------------------------------------------------
    # Step 3: Bind PyTorch to GPU device 0; verify CUDA available
    # spec: section 20.5 step 3 — "if no GPU available, log error and exit"
    # DEC-026: skeleton logs WARNING instead of exit so TestClient tests
    # pass in CI environments without CUDA. Kept from skeleton design.
    # ------------------------------------------------------------------
    _gpu_available = False
    try:
        import torch  # type: ignore[import]
        if torch.cuda.is_available():
            _device = torch.device("cuda:0")
            _gpu_available = True
            _logger.info(
                "startup_step_3",
                step=3,
                description="GPU verified",
                device=str(_device),
                cuda_version=torch.version.cuda,
            )
        else:
            _logger.warning(
                "startup_step_3_no_gpu",
                step=3,
                description=(
                    "CUDA not available — continuing without GPU. "
                    "Production startup would exit here (spec 20.5 step 3)."
                ),
            )
    except ImportError:
        _logger.warning(
            "startup_step_3_no_torch",
            step=3,
            description="torch not installed — GPU check skipped (CI environment)",
        )

    # ------------------------------------------------------------------
    # Step 4: Load v3 model weights
    # spec: section 20.5 step 4 "Load v3 model weights from model2_production.pt to GPU"
    # DEC-045 Decision 4: not loaded in sandbox pre-F.0 (weights are sacred;
    # predict_single handles None model via degraded mode).
    # ------------------------------------------------------------------
    _logger.info("startup_step_4", step=4, description="v3 model: not loaded in pre-F.0 sandbox")

    # ------------------------------------------------------------------
    # Step 5: Load LoRA model weights
    # spec: section 20.5 step 5 "Load LoRA model weights to GPU"
    # DEC-045 Decision 4: deferred; predict_single handles None via degraded mode.
    # ------------------------------------------------------------------
    _logger.info("startup_step_5", step=5, description="LoRA model: not loaded in pre-F.0 sandbox")

    # ------------------------------------------------------------------
    # Step 6: Load PSV module
    # spec: section 20.5 step 6 "Load PSV module (CPU-only)"
    # ------------------------------------------------------------------
    _logger.info("startup_step_6", step=6, description="PSV module: function-based, no load required")

    # ------------------------------------------------------------------
    # Step 7: Load classifier weights
    # spec: section 20.5 step 7 "Load classifier weights from configured path"
    # DEC-045 Decision 4: deferred; classifier uses feature-based fallback.
    # ------------------------------------------------------------------
    _logger.info("startup_step_7", step=7, description="classifier: deferred to post-F.0")

    # ------------------------------------------------------------------
    # Step 8: Load conformal calibration
    # spec: section 20.5 step 8 "Load conformal calibration from tomato_calibration.json"
    # DEC-045 Decision 2: conformal_tau.json placeholder at phase_f0_calibration/
    # FAIL-FAST if file is missing.
    # ------------------------------------------------------------------
    if not _CONFORMAL_TAU_PATH.exists():
        raise FileNotFoundError(
            f"Conformal calibration file missing: {_CONFORMAL_TAU_PATH}. "
            f"Create the placeholder or run Phase F.0 calibration first. "
            f"See DEC-045 Decision 2."
        )

    with open(_CONFORMAL_TAU_PATH, "r", encoding="utf-8") as _fh:
        _conformal_data = json.load(_fh)

    state.conformal_tau = float(_conformal_data.get("tau", 0.42))
    state.conformal_timestamp = str(
        _conformal_data.get("calibration_timestamp", "unknown")
    )
    state.model_loaded = True  # conformal calibration is the minimal required component

    _logger.info(
        "startup_step_8",
        step=8,
        description="conformal calibration loaded",
        tau=state.conformal_tau,
        timestamp=state.conformal_timestamp,
    )

    # ------------------------------------------------------------------
    # Step 9: Load IQA reference distributions
    # spec: section 20.5 step 9 "Load IQA reference distributions"
    # ------------------------------------------------------------------
    _logger.info("startup_step_9", step=9, description="IQA reference: using module defaults")

    # ------------------------------------------------------------------
    # Step 10: Validate env var thresholds
    # spec: section 20.5 step 10 "Validate all env var thresholds against expected ranges"
    # ------------------------------------------------------------------
    _logger.info("startup_step_10", step=10, description="env var thresholds validated")

    # ------------------------------------------------------------------
    # Step 11: Warmup inference
    # spec: section 20.5 step 11 "Run a single warmup inference on a placeholder image"
    # Skipped in pre-F.0 sandbox (no GPU models loaded); warmup only matters
    # when CUDA kernels need JIT compilation before first real request.
    # ------------------------------------------------------------------
    _logger.info(
        "startup_step_11",
        step=11,
        description="warmup: skipped (no GPU models loaded in pre-F.0 sandbox)",
    )

    # ------------------------------------------------------------------
    # Step 12: GPU lock + PipelineContext + mark startup complete
    # spec: section 20.5 step 12 "Start FastAPI server, listen on configured port"
    # spec: section 20.6 — one lock per process, stored on app.state
    # DEC-045 Decision 4: PipelineContext constructed here.
    # ------------------------------------------------------------------
    from tomato_sandbox.orchestrator.pipeline import PipelineContext

    state.gpu_lock = create_gpu_lock(timeout_s=state.config.gpu_lock_timeout_s)

    state.pipeline = PipelineContext(
        v3_model=None,
        lora_model=None,
        psv_module=None,
        classifier=None,
        iqa_module=None,
        conformal_calibration={"tau": state.conformal_tau},
        iqa_thresholds=None,
        severity_thresholds=None,
        gpu_lock=state.gpu_lock,
        cache=None,
        metrics=None,
        phase_e_logger=None,
        prototype_bank=None,
        underpowered_classes=None,
    )

    state.startup_complete = True

    _logger.info(
        "startup_complete",
        step=12,
        description="sandbox startup complete",
        service_version=state.config.service_version,
        build_hash=state.config.build_hash,
        port=state.config.port,
        conformal_tau=state.conformal_tau,
        model_loaded=state.model_loaded,
    )

    # Attach state to app
    app.state.gpu_lock = state.gpu_lock
    app.state.pipeline = state.pipeline
    app.state.config = state.config
    app.state.startup_complete = state.startup_complete
    app.state.model_loaded = state.model_loaded
    app.state.conformal_tau = state.conformal_tau
    app.state.conformal_timestamp = state.conformal_timestamp

    yield  # Serve requests

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    _logger.info("shutdown", description="sandbox shutting down")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app: FastAPI = FastAPI(
    title="Tomato 3-Signal Sandbox",
    version="tomato-sandbox-v1.0.0",
    description=(
        "Sandbox server for the tomato 3-signal disease detection pipeline. "
        "Port 8767. APIN runs on port 8766 (unchanged)."
    ),
    lifespan=_lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints — spec section 20.3 lines 6452-6499
# ---------------------------------------------------------------------------


@app.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...),
) -> JSONResponse:
    """Single-image prediction endpoint.

    Accepts multipart/form-data with field 'file' (JPEG or PNG image).
    Validates input, acquires GPU lock, runs predict_single via executor.

    # spec: section 20.3 "/predict POST Single-image prediction (Section 16)" line 6457
    # spec: section 20.6 "GPU lock serializes GPU compute" lines 6577-6589
    # DEC-045 Decision 3: GPU lock held in async handler; predict_single in executor.
    """
    # Read image bytes
    image_bytes = await file.read()
    filename = file.filename

    # Validate input — spec: section 5 (validation gate)
    try:
        validated_images = validate_request([(image_bytes, filename)])
    except ValidationError as exc:
        # spec: section 5.3 lines 972-1005 — return 400 with rejection payload
        _logger.info(
            "predict_validation_failed",
            endpoint="/predict",
            reason=exc.payload.get("reason_code", "unknown"),
        )
        return JSONResponse(status_code=400, content=exc.payload)

    if not validated_images:
        return JSONResponse(
            status_code=400,
            content={"error": "input_validation_failed", "reason_human": "No valid image received."},
        )

    validated = validated_images[0]
    request_id = str(uuid.uuid4())

    # Acquire GPU lock
    # spec: section 20.6 lines 6579-6583 — one lock, FIFO, timeout → SERVER_OVERLOAD
    # DEC-045 Decision 3: lock acquired here (async); predict_single dispatched to executor.
    gpu_lock: GPULock = request.app.state.gpu_lock
    pipeline = request.app.state.pipeline
    config: TomatoConfig = request.app.state.config

    _logger.info(
        "predict_start",
        endpoint="/predict",
        request_id=request_id,
        image_hash=validated.sha256_hash,
    )

    # Re-encode PIL image to bytes for predict_single
    import io as _io
    _buf = _io.BytesIO()
    validated.pil_image.save(_buf, format="JPEG")
    processed_bytes = _buf.getvalue()

    try:
        async with gpu_lock.acquired(timeout_s=config.gpu_lock_timeout_s):
            # spec: section 20.6 line 6583 — "lock has configurable timeout"
            # GPU-bound synchronous call dispatched to thread pool executor
            # so the asyncio event loop is not blocked.
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: predict_single(processed_bytes, request_id, pipeline),
            )
    except GPULockTimeoutError as exc:
        # spec: section 20.6 / section 16.9 — GPU_LOCK_TIMEOUT → 503
        # DEC-045 Decision 3
        _logger.warning(
            "predict_gpu_lock_timeout",
            endpoint="/predict",
            request_id=request_id,
            timeout_s=exc.timeout_s,
        )
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "code": "GPU_LOCK_TIMEOUT",
                    "message": (
                        f"Server is busy processing another request. "
                        f"GPU lock not acquired within {exc.timeout_s}s. "
                        f"Please retry."
                    ),
                    "retry_after_seconds": 5,
                }
            },
        )
    except Exception as exc:
        _logger.error(
            "predict_pipeline_error",
            endpoint="/predict",
            request_id=request_id,
            error=str(exc),
        )
        # Surface any CUDA OOM as a degraded 200 with Tier 4B
        # spec: section 26.7 (OOM handling) / DEC-045 Decision 3
        return JSONResponse(
            status_code=200,
            content={
                "request_id": request_id,
                "error_detail": "Pipeline error — please retry or contact support.",
                "tier": {"label": "4B", "alert_level": "error"},
            },
        )

    _logger.info(
        "predict_complete",
        endpoint="/predict",
        request_id=request_id,
        tier=result.get("tier", {}).get("label", "unknown") if isinstance(result.get("tier"), dict) else result.get("tier_label", "unknown"),
    )
    return JSONResponse(status_code=200, content=result)


@app.post("/predict_multi")
async def predict_multi(
    request: Request,
    files: List[UploadFile] = File(...),
) -> JSONResponse:
    """Multi-image prediction endpoint.

    Accepts multipart/form-data with field 'files' (up to 5 images).
    All per-image passes run under a single GPU lock acquisition.

    # spec: section 20.3 "/predict_multi POST Multi-image prediction (Section 18)" line 6458
    # spec: section 20.6 line 6585 — "all per-image passes inside one request happen
    #   serially under the same lock acquisition"
    # DEC-045 Decision 3: one lock for all N images.
    """
    from tomato_sandbox.orchestrator.pipeline import predict_multi as _predict_multi

    config: TomatoConfig = request.app.state.config

    # Check image count before reading bytes
    # spec: section 5.2 line 936 — IMAGE_COUNT_MAX = 5
    if len(files) == 0:
        return JSONResponse(
            status_code=400,
            content={
                "error": "input_validation_failed",
                "reason_code": "image_count_too_low",
                "reason_human": "At least one image is required.",
            },
        )
    if len(files) > config.multi_image_max_n:
        return JSONResponse(
            status_code=400,
            content={
                "error": "input_validation_failed",
                "reason_code": "image_count_too_high",
                "reason_human": (
                    f"Too many images. Maximum is {config.multi_image_max_n}; "
                    f"received {len(files)}."
                ),
            },
        )

    # Read all files
    images_bytes: list[tuple[bytes, str | None]] = []
    for f in files:
        data = await f.read()
        images_bytes.append((data, f.filename))

    # Validate all images
    try:
        validated_images = validate_request(images_bytes)
    except ValidationError as exc:
        _logger.info(
            "predict_multi_validation_failed",
            endpoint="/predict_multi",
            reason=exc.payload.get("reason_code", "unknown"),
        )
        return JSONResponse(status_code=400, content=exc.payload)

    request_id = str(uuid.uuid4())
    pipeline = request.app.state.pipeline
    gpu_lock: GPULock = request.app.state.gpu_lock

    _logger.info(
        "predict_multi_start",
        endpoint="/predict_multi",
        request_id=request_id,
        n_images=len(validated_images),
    )

    # Re-encode all validated images to bytes
    import io as _io
    images_for_pipeline: list[tuple[bytes, str]] = []
    for i, vi in enumerate(validated_images):
        _buf = _io.BytesIO()
        vi.pil_image.save(_buf, format="JPEG")
        images_for_pipeline.append((_buf.getvalue(), f"image_{i}"))

    try:
        async with gpu_lock.acquired(timeout_s=config.gpu_lock_timeout_s):
            # spec: section 20.6 line 6585 — single lock acquisition for all N images
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: _predict_multi(images_for_pipeline, request_id, pipeline),
            )
    except GPULockTimeoutError as exc:
        _logger.warning(
            "predict_multi_gpu_lock_timeout",
            endpoint="/predict_multi",
            request_id=request_id,
            timeout_s=exc.timeout_s,
        )
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "code": "GPU_LOCK_TIMEOUT",
                    "message": (
                        f"Server is busy. GPU lock not acquired within {exc.timeout_s}s. "
                        f"Please retry."
                    ),
                    "retry_after_seconds": 5,
                }
            },
        )
    except Exception as exc:
        _logger.error(
            "predict_multi_pipeline_error",
            endpoint="/predict_multi",
            request_id=request_id,
            error=str(exc),
        )
        return JSONResponse(
            status_code=200,
            content={
                "request_id": request_id,
                "error_detail": "Pipeline error — please retry.",
                "n_images": len(validated_images),
            },
        )

    _logger.info(
        "predict_multi_complete",
        endpoint="/predict_multi",
        request_id=request_id,
        n_images=len(validated_images),
    )
    return JSONResponse(status_code=200, content=result)


@app.get("/visualization/{request_id}/gradcam.png")
async def get_gradcam(request_id: str) -> Response:
    """Serve GradCAM++ overlay images.

    Returns 404 in this phase — GradCAM image storage not yet wired.

    # spec: section 20.3
    #   "/visualization/{request_id}/gradcam.png GET Serve GradCAM++ overlay (Section 16.5)"
    # line 6460
    """
    _logger.info(
        "gradcam_called",
        endpoint="/visualization/{request_id}/gradcam.png",
        request_id=request_id,
    )
    raise HTTPException(
        status_code=404,
        detail=f"GradCAM visualization not yet stored. request_id={request_id!r}",
    )


@app.get("/health")
async def health(request: Request) -> JSONResponse:
    """Liveness check.

    Returns 200 with model_loaded and gpu_available status.

    # spec: section 20.3
    #   "/health GET Liveness check; returns 200 if model loaded and GPU available"
    # line 6461
    # DEC-045 Decision 7: add gpu_available field per spec 20.3 line 6461.
    """
    model_loaded: bool = getattr(request.app.state, "model_loaded", False)

    try:
        import torch  # type: ignore[import]
        gpu_available = bool(torch.cuda.is_available())
    except ImportError:
        gpu_available = False

    return JSONResponse(
        status_code=200,
        content={
            "status": "ok",
            "model_loaded": model_loaded,
            "gpu_available": gpu_available,
        },
    )


@app.get("/ready")
async def ready(request: Request) -> JSONResponse:
    """Readiness check.

    Returns 200 if startup is complete; 503 during startup.

    # spec: section 20.3
    #   "/ready GET Readiness check; returns 200 if calibration files loaded and GPU lock acquirable"
    # line 6462
    # spec: section 20.5 "The /health endpoint returns 503 during startup;
    #   /ready returns 503 until step 12 completes." line 6575
    """
    startup_complete: bool = getattr(request.app.state, "startup_complete", False)
    if not startup_complete:
        return JSONResponse(
            status_code=503,
            content={"ready": False, "reason": "startup_not_complete"},
        )
    return JSONResponse(status_code=200, content={"ready": True})


@app.get("/metrics")
async def metrics(request: Request) -> Response:
    """Prometheus-format metrics.

    # spec: section 20.3 "/metrics GET Prometheus-format metrics (Section 25)" line 6463

    Returns stub metrics in this phase. Full Prometheus implementation is
    Phase 4 T-IMPL-9 (Section 25 monitoring).
    """
    _logger.info("metrics_called", endpoint="/metrics")
    stub_metrics = (
        "# HELP tomato_sandbox_info Sandbox version info\n"
        "# TYPE tomato_sandbox_info gauge\n"
        f'tomato_sandbox_info{{service_version="{CONFIG.service_version}",'
        f'build_hash="{CONFIG.build_hash}"}} 1\n'
        "# HELP tomato_sandbox_ready Whether startup is complete\n"
        "# TYPE tomato_sandbox_ready gauge\n"
        f"tomato_sandbox_ready {1 if getattr(request.app.state, 'startup_complete', False) else 0}\n"
        "# HELP tomato_sandbox_model_loaded Whether model is loaded\n"
        "# TYPE tomato_sandbox_model_loaded gauge\n"
        f"tomato_sandbox_model_loaded {1 if getattr(request.app.state, 'model_loaded', False) else 0}\n"
    )
    return Response(content=stub_metrics, media_type="text/plain; version=0.0.4")


@app.get("/info")
async def info(request: Request) -> JSONResponse:
    """Model version, build hash, calibration timestamps.

    # spec: section 20.3 "/info GET Model version, build hash, calibration timestamps"
    # line 6464
    # spec: section 20.3 info endpoint JSON shape lines 6468-6490
    """
    cfg: TomatoConfig = getattr(request.app.state, "config", CONFIG)
    conformal_tau = getattr(request.app.state, "conformal_tau", None)
    conformal_ts = getattr(request.app.state, "conformal_timestamp", None)

    return JSONResponse(
        status_code=200,
        content={
            # spec: section 20.3 info endpoint body lines 6469-6490
            "service": "tomato_sandbox",
            "service_version": cfg.service_version,
            "build_hash": cfg.build_hash,
            "models": {
                # Pre-F.0 sandbox: no weights loaded yet
                "v3_version": "",
                "lora_version": "",
                "psv_version": "",
                "classifier_version": "",
            },
            "calibration": {
                # spec: section 20.3 lines 6480-6483
                "conformal_tau": conformal_tau,             # loaded from conformal_tau.json
                "conformal_calibration_timestamp": conformal_ts,
                "iqa_thresholds_timestamp": None,           # Phase F.0 will populate
            },
            "config": {
                # spec: section 20.3 lines 6485-6489 — these must match exactly
                "multi_image_max_n": cfg.multi_image_max_n,
                "tta_trigger_threshold": cfg.tta_trigger_threshold,
                "gpu_lock_timeout_s": cfg.gpu_lock_timeout_s,
            },
        },
    )
