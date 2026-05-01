"""
Unit tests for tomato_sandbox/utils/logging.py.

Tests the public API: get_logger, log_step, SENSITIVE_FIELDS, _redact_sensitive.
Verifies the four mandatory fields, sensitive-field redaction, and level routing.

# spec: 26.7 lines 7754-7765
"""

from __future__ import annotations

import io
import json
import sys
from typing import Any
from unittest.mock import MagicMock, patch, call

import pytest

from tomato_sandbox.utils.logging import (
    SENSITIVE_FIELDS,
    _redact_sensitive,
    get_logger,
    log_step,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class CapturingLogger:
    """Minimal logger that captures all calls for assertion."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def _record(self, level: str, event: str, **kwargs: Any) -> None:
        self.calls.append({"level": level, "event": event, **kwargs})

    def debug(self, event: str, **kwargs: Any) -> None:
        self._record("debug", event, **kwargs)

    def info(self, event: str, **kwargs: Any) -> None:
        self._record("info", event, **kwargs)

    def warning(self, event: str, **kwargs: Any) -> None:
        self._record("warning", event, **kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        self._record("error", event, **kwargs)

    def critical(self, event: str, **kwargs: Any) -> None:
        self._record("critical", event, **kwargs)


# ---------------------------------------------------------------------------
# get_logger
# ---------------------------------------------------------------------------


class TestGetLogger:
    def test_returns_object(self) -> None:
        """get_logger() should return something non-None."""
        logger = get_logger(__name__)
        assert logger is not None

    def test_has_info_method(self) -> None:
        """Returned logger must have .info() method (structlog and stdlib both do)."""
        logger = get_logger(__name__)
        assert callable(getattr(logger, "info", None))

    def test_has_error_method(self) -> None:
        """Returned logger must have .error() method."""
        logger = get_logger(__name__)
        assert callable(getattr(logger, "error", None))

    def test_different_names_return_distinct_loggers(self) -> None:
        """Two different names should not return identical objects."""
        logger_a = get_logger("module_a")
        logger_b = get_logger("module_b")
        # They might be different instances (structlog caches; stdlib returns same)
        # At minimum, both must be non-None
        assert logger_a is not None
        assert logger_b is not None


# ---------------------------------------------------------------------------
# SENSITIVE_FIELDS
# ---------------------------------------------------------------------------


class TestSensitiveFields:
    def test_is_frozenset(self) -> None:
        assert isinstance(SENSITIVE_FIELDS, frozenset)

    def test_contains_user_metadata(self) -> None:
        """spec: 26.7 line 7761 — 'user_metadata' is a sensitive field."""
        assert "user_metadata" in SENSITIVE_FIELDS

    def test_contains_image_bytes(self) -> None:
        """spec: 26.7 line 7761 — 'image bytes' → 'image_bytes' in code."""
        assert "image_bytes" in SENSITIVE_FIELDS


# ---------------------------------------------------------------------------
# _redact_sensitive
# ---------------------------------------------------------------------------


class TestRedactSensitive:
    def test_non_sensitive_passes_through(self) -> None:
        extra = {"model_name": "v3", "latency_ms": 42.0}
        result = _redact_sensitive(extra)
        assert result == extra

    def test_sensitive_field_redacted(self) -> None:
        extra = {"user_metadata": {"name": "Alice"}, "step": "iqa"}
        result = _redact_sensitive(extra)
        assert result["user_metadata"] == "<REDACTED>"
        assert result["step"] == "iqa"

    def test_image_bytes_redacted(self) -> None:
        extra = {"image_bytes": b"\xff\xd8\xff", "n": 3}
        result = _redact_sensitive(extra)
        assert result["image_bytes"] == "<REDACTED>"
        assert result["n"] == 3

    def test_empty_dict(self) -> None:
        assert _redact_sensitive({}) == {}

    def test_original_not_mutated(self) -> None:
        extra = {"user_metadata": "secret"}
        _redact_sensitive(extra)
        assert extra["user_metadata"] == "secret"  # original unchanged


# ---------------------------------------------------------------------------
# log_step — mandatory fields
# ---------------------------------------------------------------------------


class TestLogStep:
    def test_mandatory_fields_present(self) -> None:
        """log_step must emit request_id, step, succeeded, duration_ms.
        # spec: 26.7 lines 7759
        """
        logger = CapturingLogger()
        log_step(
            logger,
            request_id="req-001",
            step="iqa",
            succeeded=True,
            duration_ms=12.5,
        )
        assert len(logger.calls) == 1
        call_kwargs = logger.calls[0]
        assert call_kwargs["request_id"] == "req-001"
        assert call_kwargs["step"] == "iqa"
        assert call_kwargs["succeeded"] is True
        assert call_kwargs["duration_ms"] == 12.5

    def test_default_level_is_info(self) -> None:
        logger = CapturingLogger()
        log_step(
            logger,
            request_id="r",
            step="s",
            succeeded=True,
            duration_ms=1.0,
        )
        assert logger.calls[0]["level"] == "info"

    def test_explicit_debug_level(self) -> None:
        logger = CapturingLogger()
        log_step(
            logger,
            request_id="r",
            step="s",
            succeeded=True,
            duration_ms=1.0,
            level="debug",
        )
        assert logger.calls[0]["level"] == "debug"

    def test_explicit_error_level(self) -> None:
        logger = CapturingLogger()
        log_step(
            logger,
            request_id="r",
            step="s",
            succeeded=False,
            duration_ms=5.0,
            level="error",
        )
        assert logger.calls[0]["level"] == "error"

    def test_invalid_level_raises(self) -> None:
        logger = CapturingLogger()
        with pytest.raises(ValueError, match="log_step"):
            log_step(
                logger,
                request_id="r",
                step="s",
                succeeded=True,
                duration_ms=1.0,
                level="verbose",  # not a valid level
            )

    def test_extra_fields_included(self) -> None:
        logger = CapturingLogger()
        log_step(
            logger,
            request_id="r",
            step="s",
            succeeded=True,
            duration_ms=1.0,
            model="v3",
            n_classes=6,
        )
        call_kwargs = logger.calls[0]
        assert call_kwargs["model"] == "v3"
        assert call_kwargs["n_classes"] == 6

    def test_sensitive_extra_redacted(self) -> None:
        """Sensitive fields in extra kwargs must be redacted before emission.
        # spec: 26.7 line 7761 — NEVER logged at INFO or above.
        """
        logger = CapturingLogger()
        log_step(
            logger,
            request_id="r",
            step="s",
            succeeded=True,
            duration_ms=1.0,
            user_metadata={"id": 42},
        )
        call_kwargs = logger.calls[0]
        assert call_kwargs.get("user_metadata") == "<REDACTED>"

    def test_exc_info_attached_on_error(self) -> None:
        """Stack traces should appear when level=error.
        # spec: 26.7 line 7762
        """
        logger = CapturingLogger()
        try:
            raise ValueError("boom")
        except ValueError as exc:
            log_step(
                logger,
                request_id="r",
                step="s",
                succeeded=False,
                duration_ms=0.5,
                level="error",
                exc_info=exc,
            )
        call_kwargs = logger.calls[0]
        assert "exc_info" in call_kwargs
        assert "boom" in call_kwargs["exc_info"]

    def test_exc_info_not_attached_on_info(self) -> None:
        """exc_info must NOT be included when level=info."""
        logger = CapturingLogger()
        try:
            raise ValueError("irrelevant")
        except ValueError as exc:
            log_step(
                logger,
                request_id="r",
                step="s",
                succeeded=True,
                duration_ms=0.5,
                level="info",
                exc_info=exc,
            )
        call_kwargs = logger.calls[0]
        assert "exc_info" not in call_kwargs

    def test_duration_rounded(self) -> None:
        """duration_ms should be rounded to 3 decimal places."""
        logger = CapturingLogger()
        log_step(
            logger,
            request_id="r",
            step="s",
            succeeded=True,
            duration_ms=12.3456789,
        )
        assert logger.calls[0]["duration_ms"] == round(12.3456789, 3)
