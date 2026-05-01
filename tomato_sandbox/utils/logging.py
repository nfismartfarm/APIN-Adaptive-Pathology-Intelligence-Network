"""
Structured logging utility for the Tomato 3-Signal sandbox.

Spec section: 26.7 (Logging and observability standards), lines 7754-7765.

All log emissions in this sandbox follow these standards:
  - Use structlog for structured logging; never print() in production code.
  - Every log line has at minimum: request_id, step, succeeded, duration_ms.
  - Sensitive fields (user_metadata, image bytes) are NEVER logged at INFO or above.
  - Stack traces are logged on ERROR; never swallow exceptions silently.
  - Sandbox emits to stdout in JSON format.

# spec: 26.7 lines 7758-7765
"""

from __future__ import annotations

import json
import logging as _stdlib_logging
import os
import sys
import traceback
from typing import Any

# ---------------------------------------------------------------------------
# Sensitive field redaction
# spec: 26.7 lines 7761 — "Sensitive fields (user_metadata, image bytes) are
# NEVER logged at INFO or above."
# ---------------------------------------------------------------------------
SENSITIVE_FIELDS: frozenset[str] = frozenset(
    {
        "user_metadata",
        "image_bytes",
        "image_data",
        "raw_image",
        "file_bytes",
        "password",
        "token",
        "secret",
    }
)

_REDACTED = "<REDACTED>"

# ---------------------------------------------------------------------------
# structlog integration (optional; falls back to stdlib logging)
# ---------------------------------------------------------------------------
try:
    import structlog  # type: ignore[import]

    _STRUCTLOG_AVAILABLE = True
except ImportError:  # pragma: no cover
    _STRUCTLOG_AVAILABLE = False


def _configure_structlog() -> None:
    """Configure structlog to emit JSON to stdout.

    Called once at module import. Idempotent.

    # spec: 26.7 lines 7765 — "the sandbox emits to stdout in JSON format"
    """
    if not _STRUCTLOG_AVAILABLE:
        return

    import structlog  # noqa: PLC0415  (conditional import)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            # NOTE: add_logger_name is intentionally omitted.
            # structlog.stdlib.add_logger_name calls logger.name which only
            # exists on stdlib Logger, not on structlog's PrintLogger.
            # Using it here raises AttributeError when structlog is configured
            # with PrintLoggerFactory (our JSON-to-stdout setup).
            # The spec (26.7) does not require a logger name field; it requires
            # request_id, step, succeeded, duration_ms.
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            _stdlib_logging.DEBUG
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )


_configure_structlog()


# ---------------------------------------------------------------------------
# Stdlib fallback logger with JSON formatter
# ---------------------------------------------------------------------------

class _StdlibJsonFormatter(_stdlib_logging.Formatter):
    """JSON formatter for stdlib logging, used when structlog is not installed."""

    def format(self, record: _stdlib_logging.LogRecord) -> str:  # noqa: A003
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "timestamp": self.formatTime(record, self.datefmt),
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        # Merge any extra keys attached to the record
        for key, val in record.__dict__.items():
            if key.startswith("_") or key in {
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "name",
                "message",
                "taskName",
            }:
                continue
            payload[key] = val
        return json.dumps(payload, default=str)


def _make_stdlib_logger(name: str) -> _stdlib_logging.Logger:
    """Return a stdlib logger configured with JSON formatter to stdout."""
    logger = _stdlib_logging.getLogger(name)
    if not logger.handlers:
        handler = _stdlib_logging.StreamHandler(sys.stdout)
        handler.setFormatter(_StdlibJsonFormatter())
        logger.addHandler(handler)
    logger.setLevel(_stdlib_logging.DEBUG)
    return logger


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_logger(name: str) -> Any:
    """Return a logger bound to the given name.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns:
        A structlog BoundLogger if structlog is installed, otherwise a stdlib
        Logger with JSON formatting. Both support the same log-level methods:
        ``debug``, ``info``, ``warning``, ``error``, ``critical``.

    # spec: 26.7 lines 7758 — "Use structlog for structured logging; never
    # print() in production code."
    """
    if _STRUCTLOG_AVAILABLE:
        import structlog  # noqa: PLC0415

        return structlog.get_logger(name)
    return _make_stdlib_logger(name)


def _redact_sensitive(extra: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *extra* with sensitive fields replaced by <REDACTED>.

    # spec: 26.7 lines 7761 — sensitive fields never logged at INFO or above.
    """
    return {
        k: (_REDACTED if k in SENSITIVE_FIELDS else v) for k, v in extra.items()
    }


def log_step(
    logger: Any,
    *,
    request_id: str,
    step: str,
    succeeded: bool,
    duration_ms: float,
    level: str = "info",
    exc_info: BaseException | None = None,
    **extra: Any,
) -> None:
    """Emit a single structured log line with mandatory fields.

    This is the primary logging helper for all pipeline steps. It enforces the
    four mandatory fields required by spec Section 26.7 and automatically
    redacts any sensitive field names from *extra*.

    Args:
        logger: Logger returned by :func:`get_logger`.
        request_id: UUID for the current request (ties log lines together).
        step: Short name of the pipeline step, e.g. ``"iqa"``, ``"signal_a"``.
        succeeded: ``True`` if the step completed without error.
        duration_ms: Wall-clock duration of the step in milliseconds.
        level: Log level string — ``"debug"``, ``"info"``, ``"warning"``,
            ``"error"``, or ``"critical"``. Defaults to ``"info"``.
        exc_info: Exception to log alongside this event. Only emitted when
            *level* is ``"error"`` or ``"critical"``; ignored otherwise.
            # spec: 26.7 lines 7762 — "Stack traces are logged on ERROR;
            # never swallow exceptions silently."
        **extra: Additional key-value pairs to include in the log event.
            Fields matching :data:`SENSITIVE_FIELDS` are redacted to
            ``"<REDACTED>"`` before emission.

    Raises:
        ValueError: If *level* is not a recognised log-level string.

    # spec: 26.7 lines 7759 — "Every log line has at minimum: request_id,
    # step, succeeded, duration_ms."
    """
    allowed_levels = {"debug", "info", "warning", "error", "critical"}
    if level not in allowed_levels:
        raise ValueError(
            f"log_step: level={level!r} is not one of {sorted(allowed_levels)}"
        )

    safe_extra = _redact_sensitive(extra)

    event_kwargs: dict[str, Any] = {
        "request_id": request_id,
        "step": step,
        "succeeded": succeeded,
        "duration_ms": round(duration_ms, 3),
        **safe_extra,
    }

    # Attach traceback when appropriate
    if exc_info is not None and level in {"error", "critical"}:
        event_kwargs["exc_info"] = "".join(
            traceback.format_exception(type(exc_info), exc_info, exc_info.__traceback__)
        )

    log_fn = getattr(logger, level)
    log_fn("pipeline_step", **event_kwargs)
