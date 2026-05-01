"""
FastAPI skeleton server for the tomato 3-signal sandbox.

Sandbox server port: 8767 (BLK-002 / DEC-012 / DEC-026).
APIN remains on port 8766 (unchanged, not imported here — BLK-003).
Unified server is on port 8005 (Section 22).

This module implements the FastAPI application with:
  - A lifespan context manager executing the 12-step startup sequence
    (spec section 20.5). Steps 4-11 are stubs in this skeleton.
  - All seven endpoints from spec section 20.3.
  - GPU lock instance on app.state (spec section 20.6).
  - Structured logging via tomato_sandbox.utils.logging (spec section 26.7).

# spec: section 20.3 (endpoints) lines 6452-6499
# spec: section 20.4 (module layout) lines 6501-6554
# spec: section 20.5 (startup sequence) lines 6556-6575
# spec: section 20.6 (GPU lock) lines 6577-6589
# spec: section 20.7 (configuration sources) lines 6591-6603
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from tomato_sandbox.config import CONFIG, TomatoConfig
from tomato_sandbox.utils.gpu_lock import GPULock, create_gpu_lock
from tomato_sandbox.utils.logging import get_logger

# ---------------------------------------------------------------------------
# Module-level logger
# spec: section 26.7 "Use structlog for structured logging; never print()"
# ---------------------------------------------------------------------------

_logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Application state (typed)
# ---------------------------------------------------------------------------


class _AppState:
    """Container for objects stored on ``app.state`` during lifespan.

    # spec: section 20.5 — startup populates gpu_lock and pipeline;
    #       section 20.6 — gpu_lock is the asyncio-based GPU serialiser.
    """

    gpu_lock: GPULock
    pipeline: Any  # None until orchestrator is wired (Phase 4 later batches)
    config: TomatoConfig
    startup_complete: bool
    model_loaded: bool


# ---------------------------------------------------------------------------
# Lifespan context manager — 12-step startup sequence
# spec: section 20.5 lines 6558-6573
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Execute the 12-step startup sequence, then yield to serve requests.

    Steps 4-11 are stubs in this skeleton (model loading deferred to
    later Phase 4 batches). Steps 1-3 and 12 execute for real.

    # spec: section 20.5 steps 1-12 lines 6560-6572
    """
    state: _AppState = _AppState()  # type: ignore[call-arg]
    state.config = CONFIG
    state.startup_complete = False
    state.model_loaded = False
    state.pipeline = None  # placeholder until orchestrator is wired

    # Step 1: Load env vars (TOMATO_* namespace)
    # spec: section 20.5 step 1 "Load env vars (TOMATO_* namespace per Section 4.5)"
    _logger.info(
        "startup_step_1",
        step=1,
        description="env vars loaded",
        port=state.config.port,
        gpu_lock_timeout_s=state.config.gpu_lock_timeout_s,
    )

    # Step 2: Initialize structured logging
    # spec: section 20.5 step 2 "Initialize structured logging"
    # Logging is configured at module import time (tomato_sandbox/utils/logging.py);
    # this step just confirms it's active.
    _logger.info("startup_step_2", step=2, description="structured logging initialized")

    # Step 3: Bind PyTorch to GPU device 0; verify CUDA available
    # spec: section 20.5 step 3
    # "if no GPU available, log error and exit"
    # NOTE (DEC-026): skeleton logs WARNING instead of exit so TestClient
    # tests pass in CI environments without CUDA. The exit behavior is wired
    # when real model loading (step 4) is implemented.
    try:
        import torch  # type: ignore[import]
        if torch.cuda.is_available():
            _device = torch.device("cuda:0")
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
                    "CUDA not available — skeleton continues for testing. "
                    "Production startup would exit here (spec 20.5 step 3)."
                ),
            )
    except ImportError:
        _logger.warning(
            "startup_step_3_no_torch",
            step=3,
            description="torch not installed — GPU check skipped (test/CI environment)",
        )

    # Step 4: Load v3 model weights (STUB)
    # spec: section 20.5 step 4 "Load v3 model weights from model2_production.pt to GPU"
    _logger.info("startup_step_4", step=4, description="stub: v3 model load deferred")

    # Step 5: Load LoRA model weights (STUB)
    # spec: section 20.5 step 5 "Load LoRA model weights to GPU"
    _logger.info("startup_step_5", step=5, description="stub: LoRA model load deferred")

    # Step 6: Load PSV module (STUB)
    # spec: section 20.5 step 6 "Load PSV module (CPU-only)"
    _logger.info("startup_step_6", step=6, description="stub: PSV module load deferred")

    # Step 7: Load classifier weights (STUB)
    # spec: section 20.5 step 7 "Load classifier weights from configured path"
    _logger.info("startup_step_7", step=7, description="stub: classifier load deferred")

    # Step 8: Load conformal calibration (STUB)
    # spec: section 20.5 step 8 "Load conformal calibration from tomato_calibration.json"
    _logger.info("startup_step_8", step=8, description="stub: conformal calibration load deferred")

    # Step 9: Load IQA reference distributions (STUB)
    # spec: section 20.5 step 9 "Load IQA reference distributions"
    _logger.info("startup_step_9", step=9, description="stub: IQA reference load deferred")

    # Step 10: Validate env var thresholds (STUB)
    # spec: section 20.5 step 10 "Validate all env var thresholds against expected ranges"
    _logger.info("startup_step_10", step=10, description="stub: threshold validation deferred")

    # Step 11: Warmup inference (STUB)
    # spec: section 20.5 step 11
    # "Run a single warmup inference on a placeholder image"
    _logger.info("startup_step_11", step=11, description="stub: warmup inference deferred")

    # Step 12: GPU lock + mark startup complete
    # spec: section 20.5 step 12 "Start FastAPI server, listen on configured port"
    # spec: section 20.6 — one lock per process, stored on app.state
    state.gpu_lock = create_gpu_lock(timeout_s=state.config.gpu_lock_timeout_s)
    state.startup_complete = True
    _logger.info(
        "startup_complete",
        step=12,
        description="sandbox startup complete",
        service_version=state.config.service_version,
        build_hash=state.config.build_hash,
        port=state.config.port,
    )

    # Attach state to app
    app.state.gpu_lock = state.gpu_lock
    app.state.pipeline = state.pipeline
    app.state.config = state.config
    app.state.startup_complete = state.startup_complete
    app.state.model_loaded = state.model_loaded

    yield  # Serve requests

    # Shutdown
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
async def predict(request: Request) -> JSONResponse:
    """Single-image prediction endpoint.

    # spec: section 20.3 "/predict POST Single-image prediction (Section 16)"
    # line 6457

    Returns HTTP 503 in this skeleton. Full implementation wired in
    Phase 4 later batches (T-IMPL-6b orchestrator).

    # DEC-026: 503 placeholder until orchestrator is wired.
    """
    _logger.info("predict_called", endpoint="/predict", note="skeleton stub — 503")
    return JSONResponse(
        status_code=503,
        content={
            "error": "pipeline_not_ready",
            "message": "Not ready — model loading deferred to later Phase 4 batch.",
        },
    )


@app.post("/predict_multi")
async def predict_multi(request: Request) -> JSONResponse:
    """Multi-image prediction endpoint.

    # spec: section 20.3 "/predict_multi POST Multi-image prediction (Section 18)"
    # line 6458

    Returns HTTP 503 in this skeleton.
    """
    _logger.info("predict_multi_called", endpoint="/predict_multi", note="skeleton stub — 503")
    return JSONResponse(
        status_code=503,
        content={
            "error": "pipeline_not_ready",
            "message": "Not ready — model loading deferred to later Phase 4 batch.",
        },
    )


@app.get("/visualization/{request_id}/gradcam.png")
async def get_gradcam(request_id: str) -> Response:
    """Serve GradCAM++ overlay images.

    # spec: section 20.3
    #   "/visualization/{request_id}/gradcam.png GET Serve GradCAM++ overlay (Section 16.5)"
    # line 6460

    Returns 404 in this skeleton. Full implementation in Phase 4 later batches.
    """
    _logger.info(
        "gradcam_called",
        endpoint="/visualization/{request_id}/gradcam.png",
        request_id=request_id,
        note="skeleton stub — 404",
    )
    raise HTTPException(
        status_code=404,
        detail=f"GradCAM visualization not available in skeleton. request_id={request_id}",
    )


@app.get("/health")
async def health(request: Request) -> JSONResponse:
    """Liveness check.

    Returns 200 with ``{"status": "ok", "model_loaded": false}`` in skeleton.

    # spec: section 20.3
    #   "/health GET Liveness check; returns 200 if model loaded and GPU available"
    # line 6461

    ``model_loaded`` is False in this skeleton because model-loading steps (4-11)
    are stubs. When the orchestrator is wired in a later Phase 4 batch, this will
    reflect the real model-loaded state.
    """
    model_loaded: bool = getattr(request.app.state, "model_loaded", False)
    return JSONResponse(
        status_code=200,
        content={"status": "ok", "model_loaded": model_loaded},
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

    In this skeleton, startup completes quickly (all model steps are stubs),
    so /ready returns 200 immediately after startup.
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

    Returns a stub placeholder in this skeleton. Full Prometheus implementation
    is Phase 4 T-IMPL-9 (Section 25 monitoring).
    """
    _logger.info("metrics_called", endpoint="/metrics", note="skeleton stub")
    stub_metrics = (
        "# HELP tomato_sandbox_info Sandbox version info\n"
        "# TYPE tomato_sandbox_info gauge\n"
        f'tomato_sandbox_info{{service_version="{CONFIG.service_version}",'
        f'build_hash="{CONFIG.build_hash}"}} 1\n"'
        "# HELP tomato_sandbox_ready Whether startup is complete\n"
        "# TYPE tomato_sandbox_ready gauge\n"
        f"tomato_sandbox_ready {1 if getattr(request.app.state, 'startup_complete', False) else 0}\n"
    )
    return Response(content=stub_metrics, media_type="text/plain; version=0.0.4")


@app.get("/info")
async def info(request: Request) -> JSONResponse:
    """Model version, build hash, calibration timestamps.

    # spec: section 20.3 "/info GET Model version, build hash, calibration timestamps"
    # line 6464
    # spec: section 20.3 info endpoint JSON shape lines 6468-6490

    Returns the spec-verbatim JSON shape with stub/placeholder values.
    Real model/calibration versions are populated when model-loading is wired.
    """
    cfg: TomatoConfig = getattr(request.app.state, "config", CONFIG)
    return JSONResponse(
        status_code=200,
        content={
            # spec: section 20.3 info endpoint body lines 6469-6490
            "service": "tomato_sandbox",
            "service_version": cfg.service_version,
            "build_hash": cfg.build_hash,
            "models": {
                # Stub values — populated after model loading is wired
                "v3_version": "",
                "lora_version": "",
                "psv_version": "",
                "classifier_version": "",
            },
            "calibration": {
                # Stub values — populated after calibration loading is wired
                "conformal_tau": None,
                "conformal_calibration_timestamp": None,
                "iqa_thresholds_timestamp": None,
            },
            "config": {
                # spec: section 20.3 lines 6485-6489 — these must match exactly
                "multi_image_max_n": cfg.multi_image_max_n,
                "tta_trigger_threshold": cfg.tta_trigger_threshold,
                "gpu_lock_timeout_s": cfg.gpu_lock_timeout_s,
            },
        },
    )
