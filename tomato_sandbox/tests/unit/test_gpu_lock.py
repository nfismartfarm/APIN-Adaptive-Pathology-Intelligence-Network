"""
Unit tests for tomato_sandbox/utils/gpu_lock.py.

Tests: GPULock, GPULockTimeoutError, create_gpu_lock, timeout env var.

# spec: 20.6 lines 6577-6589
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import patch

import pytest

from tomato_sandbox.utils.gpu_lock import (
    GPULock,
    GPULockTimeoutError,
    _DEFAULT_TIMEOUT_S,
    _ENV_VAR,
    _get_timeout_s,
    create_gpu_lock,
)


# ---------------------------------------------------------------------------
# _get_timeout_s
# ---------------------------------------------------------------------------


class TestGetTimeoutS:
    def test_returns_default_when_env_absent(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            # Remove env var if present
            env = {k: v for k, v in os.environ.items() if k != _ENV_VAR}
            with patch.dict(os.environ, env, clear=True):
                assert _get_timeout_s() == _DEFAULT_TIMEOUT_S

    def test_reads_custom_value(self) -> None:
        with patch.dict(os.environ, {_ENV_VAR: "25.0"}):
            assert _get_timeout_s() == 25.0

    def test_returns_default_for_invalid_string(self) -> None:
        with patch.dict(os.environ, {_ENV_VAR: "not_a_number"}):
            assert _get_timeout_s() == _DEFAULT_TIMEOUT_S

    def test_returns_default_for_negative(self) -> None:
        with patch.dict(os.environ, {_ENV_VAR: "-5"}):
            assert _get_timeout_s() == _DEFAULT_TIMEOUT_S


# ---------------------------------------------------------------------------
# GPULockTimeoutError
# ---------------------------------------------------------------------------


class TestGPULockTimeoutError:
    def test_is_runtime_error(self) -> None:
        err = GPULockTimeoutError(10.0)
        assert isinstance(err, RuntimeError)

    def test_message_contains_timeout(self) -> None:
        err = GPULockTimeoutError(7.5)
        assert "7.5" in str(err)

    def test_timeout_s_attribute(self) -> None:
        err = GPULockTimeoutError(3.0)
        assert err.timeout_s == 3.0


# ---------------------------------------------------------------------------
# GPULock — basic acquire/release
# ---------------------------------------------------------------------------


class TestGPULockAcquireRelease:
    def test_acquire_and_release(self) -> None:
        async def run() -> None:
            lock = GPULock(timeout_s=5.0)
            assert not lock.locked
            await lock.acquire_with_timeout()
            assert lock.locked
            lock.release()
            assert not lock.locked

        asyncio.run(run())

    def test_locked_property_false_initially(self) -> None:
        lock = GPULock(timeout_s=5.0)
        assert not lock.locked

    def test_timeout_s_property(self) -> None:
        lock = GPULock(timeout_s=15.0)
        assert lock.timeout_s == 15.0

    def test_timeout_raises_on_contention(self) -> None:
        """A second acquire with short timeout should raise when lock is held."""

        async def run() -> None:
            lock = GPULock(timeout_s=5.0)
            await lock.acquire_with_timeout()  # first acquirer holds it
            # second acquire with very short timeout should timeout
            with pytest.raises(GPULockTimeoutError):
                await lock.acquire_with_timeout(timeout_s=0.05)
            lock.release()

        asyncio.run(run())


# ---------------------------------------------------------------------------
# GPULock — context manager
# ---------------------------------------------------------------------------


class TestGPULockContextManager:
    def test_acquired_context_manager(self) -> None:
        async def run() -> None:
            lock = GPULock(timeout_s=5.0)
            async with lock.acquired():
                assert lock.locked
            assert not lock.locked

        asyncio.run(run())

    def test_acquired_releases_on_exception(self) -> None:
        async def run() -> None:
            lock = GPULock(timeout_s=5.0)
            try:
                async with lock.acquired():
                    raise ValueError("test error")
            except ValueError:
                pass
            assert not lock.locked  # lock must be released even on exception

        asyncio.run(run())

    def test_acquired_timeout_raises(self) -> None:
        async def run() -> None:
            lock = GPULock(timeout_s=5.0)
            await lock.acquire_with_timeout()  # hold the lock
            with pytest.raises(GPULockTimeoutError):
                async with lock.acquired(timeout_s=0.05):
                    pass  # should not reach here
            lock.release()

        asyncio.run(run())


# ---------------------------------------------------------------------------
# create_gpu_lock factory
# ---------------------------------------------------------------------------


class TestCreateGpuLock:
    def test_returns_gpu_lock_instance(self) -> None:
        lock = create_gpu_lock(timeout_s=10.0)
        assert isinstance(lock, GPULock)

    def test_explicit_timeout_respected(self) -> None:
        lock = create_gpu_lock(timeout_s=42.0)
        assert lock.timeout_s == 42.0

    def test_none_reads_env(self) -> None:
        with patch.dict(os.environ, {_ENV_VAR: "33.0"}):
            lock = create_gpu_lock(timeout_s=None)
            assert lock.timeout_s == 33.0


# ---------------------------------------------------------------------------
# FIFO ordering hint test (behavioural)
# ---------------------------------------------------------------------------


class TestFIFOOrdering:
    def test_two_waiters_execute_in_order(self) -> None:
        """Verify that two waiters are scheduled in FIFO order.

        asyncio.Lock guarantees FIFO for waiters in CPython 3.10+.
        # spec: 20.6 line 6582 — "Requests waiting for the lock queue with
        # FIFO ordering."
        """
        completed: list[int] = []

        async def worker(lock: GPULock, label: int, hold_s: float) -> None:
            await lock.acquire_with_timeout(timeout_s=5.0)
            try:
                await asyncio.sleep(hold_s)
                completed.append(label)
            finally:
                lock.release()

        async def run() -> None:
            lock = GPULock(timeout_s=5.0)
            # Worker 1 acquires first
            t1 = asyncio.create_task(worker(lock, 1, 0.05))
            await asyncio.sleep(0.01)  # let worker 1 acquire
            # Worker 2 starts waiting
            t2 = asyncio.create_task(worker(lock, 2, 0.01))
            await asyncio.gather(t1, t2)

        asyncio.run(run())
        assert completed == [1, 2], f"Expected [1, 2], got {completed}"
