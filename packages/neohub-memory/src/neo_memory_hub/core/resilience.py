"""Lightweight circuit breaker and retry utilities for Mem0 calls."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


T = TypeVar("T")


@dataclass
class CircuitBreaker:
    failure_threshold: int = 5
    recovery_timeout: float = 60.0  # seconds

    def __post_init__(self) -> None:
        self._failures = 0
        self._opened_at: float | None = None

    def allow(self) -> bool:
        """Return True if calls are permitted."""
        if self._opened_at is None:
            return True
        if (time.time() - self._opened_at) >= self.recovery_timeout:
            # Half-open: allow a trial call
            return True
        return False

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._opened_at = time.time()


async def retry_async(
    func: Callable[[], Awaitable[T]],
    retries: int = 3,
    base_delay: float = 0.5,
    breaker: CircuitBreaker | None = None,
) -> T:
    """Retry an async callable with exponential backoff and optional circuit breaker."""
    attempt = 0
    last_exc: Exception | None = None

    while attempt <= retries:
        if breaker and not breaker.allow():
            raise RuntimeError("Circuit breaker open for Mem0 call")

        try:
            result = await func()
            if breaker:
                breaker.record_success()
            return result
        except Exception as exc:
            last_exc = exc
            if breaker:
                breaker.record_failure()
            if attempt == retries:
                break
            delay = base_delay * (2**attempt)
            await asyncio.sleep(delay)
            attempt += 1
            continue

    assert last_exc is not None
    raise last_exc
