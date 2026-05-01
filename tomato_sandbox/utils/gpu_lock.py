"""
GPU concurrency lock for the Tomato 3-Signal sandbox.

Spec section: 20.6 (GPU lock), lines 6577-6589.

GPU compute (model forward passes) is serialized by a single asyncio.Lock.
Only one request holds the lock at a time. This prevents:
  - VRAM exhaustion from concurrent forward passes
  - CUDA stream contention degrading per-request latency

Requests waiting for the lock queue with FIFO ordering.
The lock has a configurable timeout (TOMATO_GPU_LOCK_TIMEOUT_S, default 10 s).
On timeout the caller should return a SERVER_OVERLOAD error with
retry_after_seconds: 5.

# spec: 20.6 lines 6579-6583
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

# ---------------------------------------------------------------------------
# Configuration
# spec: 20.6 lines 6583 — "configurable timeout (TOMATO_GPU_LOCK_TIMEOUT_S,
# default 10 seconds)"
# ---------------------------------------------------------------------------
_DEFAULT_TIMEOUT_S: float = 10.0
_ENV_VAR: str = "TOMATO_GPU_LOCK_TIMEOUT_S"


def _get_timeout_s() -> float:
    """Read TOMATO_GPU_LOCK_TIMEOUT_S from env; return default if absent/invalid."""
    raw = os.environ.get(_ENV_VAR)
    if raw is None:
        return _DEFAULT_TIMEOUT_S
    try:
        value = float(raw)
        if value <= 0:
            raise ValueError("timeout must be positive")
        return value
    except (ValueError, TypeError):
        return _DEFAULT_TIMEOUT_S


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class GPULockTimeoutError(RuntimeError):
    """Raised when :meth:`GPULock.acquire_with_timeout` times out.

    Callers catch this and return a SERVER_OVERLOAD response with
    ``retry_after_seconds: 5``.

    # spec: 20.6 lines 6583 — "On timeout, the request returns Section 16.9
    # SERVER_OVERLOAD error with retry_after_seconds: 5."
    """

    def __init__(self, timeout_s: float) -> None:
        self.timeout_s = timeout_s
        super().__init__(
            f"GPU lock not acquired within {timeout_s:.1f} s "
            f"(TOMATO_GPU_LOCK_TIMEOUT_S={timeout_s}). "
            "Return SERVER_OVERLOAD to the caller with retry_after_seconds=5."
        )


# ---------------------------------------------------------------------------
# GPULock class
# ---------------------------------------------------------------------------


class GPULock:
    """Serialises GPU forward passes across concurrent async requests.

    Wraps :class:`asyncio.Lock` with a configurable acquire timeout. FIFO
    ordering of waiters is guaranteed by CPython's asyncio.Lock implementation
    (documented since Python 3.10, de-facto true in 3.7+).

    # spec: 20.6 lines 6579 — "serialized by a single asyncio.Lock"
    # spec: 20.6 lines 6582 — "Requests waiting for the lock queue with FIFO
    # ordering."

    Typical usage in the orchestrator::

        async with gpu_lock.acquired(timeout_s=10.0):
            outputs = model(inputs)

    Or using :meth:`acquire_with_timeout` directly for manual release::

        await gpu_lock.acquire_with_timeout(timeout_s=10.0)
        try:
            outputs = model(inputs)
        finally:
            gpu_lock.release()
    """

    def __init__(self, timeout_s: float | None = None) -> None:
        """Create a GPULock.

        Args:
            timeout_s: Default acquire timeout in seconds. If ``None``, reads
                ``TOMATO_GPU_LOCK_TIMEOUT_S`` from the environment (default
                10 s).
        """
        self._lock: asyncio.Lock = asyncio.Lock()
        self._timeout_s: float = (
            timeout_s if timeout_s is not None else _get_timeout_s()
        )

    @property
    def timeout_s(self) -> float:
        """Return the configured timeout in seconds."""
        return self._timeout_s

    async def acquire_with_timeout(
        self, timeout_s: float | None = None
    ) -> None:
        """Attempt to acquire the GPU lock within the given timeout.

        Args:
            timeout_s: Override the instance-level timeout for this
                acquisition. If ``None``, uses :attr:`timeout_s`.

        Raises:
            GPULockTimeoutError: If the lock is not acquired within the
                timeout period.

        # spec: 20.6 lines 6583 — "The lock has a configurable timeout
        # (TOMATO_GPU_LOCK_TIMEOUT_S, default 10 seconds). On timeout, the
        # request returns Section 16.9 SERVER_OVERLOAD error."
        """
        effective_timeout = timeout_s if timeout_s is not None else self._timeout_s
        try:
            acquired = await asyncio.wait_for(
                self._lock.acquire(), timeout=effective_timeout
            )
            # wait_for raises TimeoutError on expiry; if it returns, lock was acquired
            _ = acquired  # asyncio.Lock.acquire() always returns True
        except asyncio.TimeoutError:
            raise GPULockTimeoutError(effective_timeout)

    def release(self) -> None:
        """Release the GPU lock.

        Must be called after :meth:`acquire_with_timeout` in a ``finally``
        block.

        Raises:
            RuntimeError: If the lock is not currently held (propagated from
                :class:`asyncio.Lock`).
        """
        self._lock.release()

    @asynccontextmanager
    async def acquired(
        self, timeout_s: float | None = None
    ) -> AsyncIterator[None]:
        """Async context manager that acquires and releases the GPU lock.

        Args:
            timeout_s: Override the instance-level timeout.

        Raises:
            GPULockTimeoutError: If the lock cannot be acquired within the
                timeout.

        Example::

            async with gpu_lock.acquired():
                result = model(inputs)
        """
        await self.acquire_with_timeout(timeout_s=timeout_s)
        try:
            yield
        finally:
            self.release()

    @property
    def locked(self) -> bool:
        """Return ``True`` if the lock is currently held by any coroutine."""
        return self._lock.locked()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_gpu_lock(timeout_s: float | None = None) -> GPULock:
    """Create a :class:`GPULock` instance for use as ``app.state.gpu_lock``.

    Reads ``TOMATO_GPU_LOCK_TIMEOUT_S`` from the environment if *timeout_s*
    is ``None``.

    Args:
        timeout_s: Explicit timeout override. If ``None``, env var is used.

    Returns:
        A new :class:`GPULock` instance.

    # spec: 20.6 lines 6582-6583
    """
    return GPULock(timeout_s=timeout_s)
