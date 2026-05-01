"""
Unit tests for the tomato sandbox FastAPI skeleton server.

Tests use FastAPI's TestClient (Starlette synchronous test client) to call
endpoints in-process without a running uvicorn process.

Spec coverage:
  - Section 20.3: all seven endpoints verified
  - Section 20.5: startup sequence completes without error; state set correctly
  - Section 20.6: gpu_lock attached to app.state after startup
  - Section 20.7: config hierarchy (port default 8767, timeout 10.0)

# spec: section 20.3 lines 6452-6499
# spec: section 20.5 lines 6556-6575
# spec: section 20.6 lines 6577-6589
# spec: section 20.7 lines 6591-6603
"""

from __future__ import annotations

import os
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tomato_sandbox.api.server import app
from tomato_sandbox.config import TomatoConfig, load_config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Return a TestClient with the sandbox app.

    ``TestClient`` runs the lifespan context manager synchronously, so startup
    and shutdown both run inside the ``with`` block. This fixture creates one
    client per module and reuses it across all tests.
    """
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Section 20.7 — Configuration tests (no server needed)
# ---------------------------------------------------------------------------


class TestConfig:
    """Verify config hierarchy: env vars > local.yaml > default.yaml > hardcoded."""

    def test_default_port_is_8767(self) -> None:
        """Port default must be 8767 per BLK-002 / DEC-012 / DEC-026.

        # spec: section 20.5 step 12 "listen on configured port (default 8767)"
        # line 6571. MUST NOT be 8766 (APIN) or 8005 (unified server).
        """
        # Ensure no env override is present
        env_backup = os.environ.pop("TOMATO_PORT", None)
        try:
            cfg = load_config()
            assert cfg.port == 8767, (
                f"Default port must be 8767 (spec 20.5 / BLK-002). Got {cfg.port}."
            )
        finally:
            if env_backup is not None:
                os.environ["TOMATO_PORT"] = env_backup

    def test_port_not_8766(self) -> None:
        """Verify sandbox port is never 8766 (that's APIN's port).

        # spec: BLK-002 / DEC-012 resolution: "port 8767 is sandbox, port 8766 stays APIN"
        """
        env_backup = os.environ.pop("TOMATO_PORT", None)
        try:
            cfg = load_config()
            assert cfg.port != 8766, "Sandbox port must never be 8766 (APIN port)."
        finally:
            if env_backup is not None:
                os.environ["TOMATO_PORT"] = env_backup

    def test_default_gpu_lock_timeout(self) -> None:
        """GPU lock timeout default must be 10.0 s.

        # spec: section 20.6 "configurable timeout (TOMATO_GPU_LOCK_TIMEOUT_S, default 10)"
        # line 6583
        """
        env_backup = os.environ.pop("TOMATO_GPU_LOCK_TIMEOUT_S", None)
        try:
            cfg = load_config()
            assert cfg.gpu_lock_timeout_s == 10.0, (
                f"GPU lock timeout default must be 10.0. Got {cfg.gpu_lock_timeout_s}."
            )
        finally:
            if env_backup is not None:
                os.environ["TOMATO_GPU_LOCK_TIMEOUT_S"] = env_backup

    def test_default_multi_image_max_n(self) -> None:
        """multi_image_max_n must default to 5.

        # spec: section 20.3 info endpoint config.multi_image_max_n line 6487
        """
        env_backup = os.environ.pop("TOMATO_MULTI_IMAGE_MAX_N", None)
        try:
            cfg = load_config()
            assert cfg.multi_image_max_n == 5
        finally:
            if env_backup is not None:
                os.environ["TOMATO_MULTI_IMAGE_MAX_N"] = env_backup

    def test_default_tta_trigger_threshold(self) -> None:
        """tta_trigger_threshold must default to 0.55.

        # spec: section 20.3 info endpoint config.tta_trigger_threshold line 6488
        """
        env_backup = os.environ.pop("TOMATO_TTA_TRIGGER_THRESHOLD", None)
        try:
            cfg = load_config()
            assert cfg.tta_trigger_threshold == 0.55
        finally:
            if env_backup is not None:
                os.environ["TOMATO_TTA_TRIGGER_THRESHOLD"] = env_backup

    def test_env_var_overrides_port(self) -> None:
        """Env var TOMATO_PORT overrides default.

        # spec: section 20.7 "Env vars at process startup" — highest precedence
        """
        original = os.environ.get("TOMATO_PORT")
        os.environ["TOMATO_PORT"] = "9999"
        try:
            cfg = load_config()
            assert cfg.port == 9999
        finally:
            if original is None:
                del os.environ["TOMATO_PORT"]
            else:
                os.environ["TOMATO_PORT"] = original

    def test_env_var_overrides_gpu_lock_timeout(self) -> None:
        """Env var TOMATO_GPU_LOCK_TIMEOUT_S overrides default.

        # spec: section 20.7 — env vars highest precedence
        """
        original = os.environ.get("TOMATO_GPU_LOCK_TIMEOUT_S")
        os.environ["TOMATO_GPU_LOCK_TIMEOUT_S"] = "42.5"
        try:
            cfg = load_config()
            assert cfg.gpu_lock_timeout_s == 42.5
        finally:
            if original is None:
                del os.environ["TOMATO_GPU_LOCK_TIMEOUT_S"]
            else:
                os.environ["TOMATO_GPU_LOCK_TIMEOUT_S"] = original

    def test_config_dataclass_fields(self) -> None:
        """TomatoConfig has the required fields."""
        cfg = TomatoConfig()
        assert hasattr(cfg, "port")
        assert hasattr(cfg, "gpu_lock_timeout_s")
        assert hasattr(cfg, "multi_image_max_n")
        assert hasattr(cfg, "tta_trigger_threshold")
        assert hasattr(cfg, "service_version")
        assert hasattr(cfg, "build_hash")


# ---------------------------------------------------------------------------
# Section 20.5 — Startup sequence tests (via TestClient)
# ---------------------------------------------------------------------------


class TestStartupSequence:
    """Verify that the 12-step startup sequence completes and sets app.state."""

    def test_startup_complete_flag(self, client: TestClient) -> None:
        """startup_complete must be True after lifespan startup.

        # spec: section 20.5 step 12 "Start FastAPI server, listen on configured port"
        """
        assert getattr(client.app.state, "startup_complete", False) is True

    def test_gpu_lock_on_app_state(self, client: TestClient) -> None:
        """app.state.gpu_lock must be a GPULock instance after startup.

        # spec: section 20.6 — "Per-request GPU lock (Section 20.6) requires a single arbiter"
        # spec: section 20.5 step 12 — lock created as part of step 12
        """
        from tomato_sandbox.utils.gpu_lock import GPULock
        assert hasattr(client.app.state, "gpu_lock")
        assert isinstance(client.app.state.gpu_lock, GPULock)

    def test_pipeline_is_none_in_skeleton(self, client: TestClient) -> None:
        """app.state.pipeline is None in skeleton (model loading is stubbed).

        # DEC-026 — pipeline=None until orchestrator is wired in later batch.
        """
        assert client.app.state.pipeline is None

    def test_model_loaded_is_false_in_skeleton(self, client: TestClient) -> None:
        """model_loaded is False because model-loading steps are stubbed.

        # DEC-026: steps 4-11 are stubs; model_loaded remains False.
        """
        assert client.app.state.model_loaded is False


# ---------------------------------------------------------------------------
# Section 20.3 — Endpoint tests
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """/health returns 200 with correct JSON body.

    # spec: section 20.3 "/health GET Liveness check; returns 200 if model loaded and GPU available"
    # line 6461
    """

    def test_health_status_200(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"

    def test_health_body_has_status_ok(self, client: TestClient) -> None:
        body = client.get("/health").json()
        assert body["status"] == "ok", f"Expected status='ok', got {body}"

    def test_health_body_has_model_loaded(self, client: TestClient) -> None:
        """model_loaded key must be present (False in skeleton)."""
        body = client.get("/health").json()
        assert "model_loaded" in body, f"'model_loaded' missing from /health response: {body}"

    def test_health_model_loaded_is_false_in_skeleton(self, client: TestClient) -> None:
        """model_loaded is False in skeleton (no models loaded yet)."""
        body = client.get("/health").json()
        assert body["model_loaded"] is False

    def test_health_content_type_json(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert "application/json" in resp.headers.get("content-type", "")


class TestReadyEndpoint:
    """/ready returns 200 after startup completes.

    # spec: section 20.3 "/ready GET Readiness check; returns 200 if calibration files loaded"
    # spec: section 20.5 "/ready returns 503 until step 12 completes." line 6575
    # In skeleton, startup completes fast (stubs), so /ready returns 200.
    """

    def test_ready_200_after_startup(self, client: TestClient) -> None:
        """After lifespan startup completes, /ready returns 200."""
        resp = client.get("/ready")
        assert resp.status_code == 200, f"Expected 200 after startup, got {resp.status_code}: {resp.text}"

    def test_ready_body_has_ready_true(self, client: TestClient) -> None:
        body = client.get("/ready").json()
        assert body.get("ready") is True, f"Expected ready=True, got {body}"


class TestPredictEndpoint:
    """/predict returns HTTP 503 in skeleton.

    # spec: section 20.3 "/predict POST Single-image prediction (Section 16)" line 6457
    # DEC-026: 503 placeholder until orchestrator is wired.
    """

    def test_predict_503(self, client: TestClient) -> None:
        resp = client.post("/predict", content=b"fake image bytes")
        assert resp.status_code == 503, f"Expected 503, got {resp.status_code}: {resp.text}"

    def test_predict_body_has_error_key(self, client: TestClient) -> None:
        body = client.post("/predict", content=b"fake").json()
        assert "error" in body or "detail" in body, (
            f"Response should have 'error' or 'detail' key: {body}"
        )

    def test_predict_503_with_empty_body(self, client: TestClient) -> None:
        """Even with empty body, predict returns 503 (not 422 from Pydantic)."""
        resp = client.post("/predict")
        assert resp.status_code == 503


class TestPredictMultiEndpoint:
    """/predict_multi returns HTTP 503 in skeleton.

    # spec: section 20.3 "/predict_multi POST Multi-image prediction (Section 18)" line 6458
    """

    def test_predict_multi_503(self, client: TestClient) -> None:
        resp = client.post("/predict_multi", json={"images": []})
        assert resp.status_code == 503, f"Expected 503, got {resp.status_code}: {resp.text}"

    def test_predict_multi_body_has_error_key(self, client: TestClient) -> None:
        body = client.post("/predict_multi", json={}).json()
        assert "error" in body or "detail" in body


class TestGradcamEndpoint:
    """/visualization/{request_id}/gradcam.png returns 404 in skeleton.

    # spec: section 20.3
    # "/visualization/{request_id}/gradcam.png GET Serve GradCAM++ overlay" line 6460
    """

    def test_gradcam_404(self, client: TestClient) -> None:
        resp = client.get("/visualization/test-request-id-123/gradcam.png")
        assert resp.status_code == 404, f"Expected 404, got {resp.status_code}: {resp.text}"

    def test_gradcam_404_any_id(self, client: TestClient) -> None:
        """Returns 404 for any request_id in skeleton."""
        resp = client.get("/visualization/another-id/gradcam.png")
        assert resp.status_code == 404


class TestMetricsEndpoint:
    """/metrics returns text/plain Prometheus format (stub) or 503.

    # spec: section 20.3 "/metrics GET Prometheus-format metrics (Section 25)" line 6463
    """

    def test_metrics_responds(self, client: TestClient) -> None:
        """Metrics endpoint must respond (200 with stub content)."""
        resp = client.get("/metrics")
        assert resp.status_code in (200, 503), (
            f"Expected 200 or 503, got {resp.status_code}"
        )

    def test_metrics_200_content_type(self, client: TestClient) -> None:
        resp = client.get("/metrics")
        if resp.status_code == 200:
            ct = resp.headers.get("content-type", "")
            assert "text/plain" in ct, f"Expected text/plain, got {ct}"

    def test_metrics_contains_sandbox_info(self, client: TestClient) -> None:
        """Stub metrics should mention the sandbox service."""
        resp = client.get("/metrics")
        if resp.status_code == 200:
            assert "tomato_sandbox" in resp.text


class TestInfoEndpoint:
    """/info returns the spec-verbatim JSON shape.

    # spec: section 20.3 "/info GET Model version, build hash, calibration timestamps"
    # spec: section 20.3 info endpoint JSON body lines 6468-6490
    """

    def test_info_200(self, client: TestClient) -> None:
        resp = client.get("/info")
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"

    def test_info_service_key(self, client: TestClient) -> None:
        """'service' must be 'tomato_sandbox'.

        # spec: section 20.3 line 6470 {"service": "tomato_sandbox"}
        """
        body = client.get("/info").json()
        assert body.get("service") == "tomato_sandbox", (
            f"Expected service='tomato_sandbox', got {body.get('service')}"
        )

    def test_info_service_version(self, client: TestClient) -> None:
        """service_version must be 'tomato-sandbox-v1.0.0'.

        # spec: section 20.3 line 6471 {"service_version": "tomato-sandbox-v1.0.0"}
        """
        body = client.get("/info").json()
        assert body.get("service_version") == "tomato-sandbox-v1.0.0", (
            f"Expected 'tomato-sandbox-v1.0.0', got {body.get('service_version')}"
        )

    def test_info_has_models_dict(self, client: TestClient) -> None:
        """'models' key must be present and be a dict.

        # spec: section 20.3 lines 6473-6478
        """
        body = client.get("/info").json()
        assert isinstance(body.get("models"), dict), (
            f"Expected 'models' to be a dict: {body.get('models')}"
        )

    def test_info_models_has_required_keys(self, client: TestClient) -> None:
        """models dict must have v3_version, lora_version, psv_version, classifier_version.

        # spec: section 20.3 lines 6474-6477
        """
        body = client.get("/info").json()
        models = body.get("models", {})
        for key in ("v3_version", "lora_version", "psv_version", "classifier_version"):
            assert key in models, f"'models.{key}' missing: {models}"

    def test_info_has_calibration_dict(self, client: TestClient) -> None:
        """'calibration' key must be present.

        # spec: section 20.3 lines 6479-6484
        """
        body = client.get("/info").json()
        assert isinstance(body.get("calibration"), dict)

    def test_info_calibration_has_required_keys(self, client: TestClient) -> None:
        """calibration dict must have conformal_tau, timestamps.

        # spec: section 20.3 lines 6480-6483
        """
        body = client.get("/info").json()
        cal = body.get("calibration", {})
        for key in ("conformal_tau", "conformal_calibration_timestamp", "iqa_thresholds_timestamp"):
            assert key in cal, f"'calibration.{key}' missing: {cal}"

    def test_info_has_config_dict(self, client: TestClient) -> None:
        """'config' key must be present.

        # spec: section 20.3 lines 6484-6489
        """
        body = client.get("/info").json()
        assert isinstance(body.get("config"), dict)

    def test_info_config_multi_image_max_n(self, client: TestClient) -> None:
        """config.multi_image_max_n must be 5 (spec default).

        # spec: section 20.3 line 6486 "multi_image_max_n": 5
        """
        body = client.get("/info").json()
        assert body["config"]["multi_image_max_n"] == 5

    def test_info_config_tta_trigger_threshold(self, client: TestClient) -> None:
        """config.tta_trigger_threshold must be 0.55 (spec default).

        # spec: section 20.3 line 6487 "tta_trigger_threshold": 0.55
        """
        body = client.get("/info").json()
        assert body["config"]["tta_trigger_threshold"] == 0.55

    def test_info_config_gpu_lock_timeout_s(self, client: TestClient) -> None:
        """config.gpu_lock_timeout_s must be 10 (spec default).

        # spec: section 20.3 line 6488 "gpu_lock_timeout_s": 10
        # spec: section 20.6 "configurable timeout (TOMATO_GPU_LOCK_TIMEOUT_S, default 10)"
        """
        body = client.get("/info").json()
        assert body["config"]["gpu_lock_timeout_s"] == 10.0

    def test_info_build_hash_present(self, client: TestClient) -> None:
        """build_hash key must be present.

        # spec: section 20.3 line 6472 "build_hash"
        """
        body = client.get("/info").json()
        assert "build_hash" in body


# ---------------------------------------------------------------------------
# Integration smoke test (as required by task spec)
# ---------------------------------------------------------------------------


class TestClientSmokeTest:
    """Instantiate TestClient, hit /health, verify response.

    This is the TestClient smoke test required by the task acceptance criteria.
    """

    def test_smoke_health_roundtrip(self) -> None:
        """Full lifecycle smoke: create client, call /health, check response."""
        with TestClient(app) as c:
            resp = c.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body == {"status": "ok", "model_loaded": False}, (
            f"Unexpected /health response: {body}"
        )

    def test_smoke_no_apin_import(self) -> None:
        """Verify no APIN library is imported by the server module.

        # spec: BLK-003 / DEC-012 — "HTTP-only client to APIN; no APIN import"
        """
        import importlib
        import sys
        # Collect all module names imported by server module's package
        server_module = sys.modules.get("tomato_sandbox.api.server")
        assert server_module is not None
        # Verify none of the known APIN module names are in sys.modules
        apin_namespaces = ["apin", "scripts.apin", "section2d_psv"]
        for ns in apin_namespaces:
            for mod_name in sys.modules:
                assert not mod_name.startswith(ns), (
                    f"APIN module '{mod_name}' was imported — violates BLK-003 / DEC-012"
                )
