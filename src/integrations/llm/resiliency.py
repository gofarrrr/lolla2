from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable


@dataclass
class RetryPolicy:
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 15.0
    multiplier: float = 2.0
    jitter: float = 0.1

    async def execute(
        self,
        operation: Callable[[], Awaitable[Any]],
        *,
        on_retry: Callable[[int, Exception, float], None] | None = None,
    ) -> Any:
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                return await operation()
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt >= self.max_attempts:
                    break

                delay = min(
                    self.base_delay * (self.multiplier ** (attempt - 1)),
                    self.max_delay,
                )
                delay += random.uniform(0, delay * self.jitter)
                if on_retry:
                    on_retry(attempt, exc, delay)
                await asyncio.sleep(delay)

        assert last_error is not None  # for mypy
        raise last_error


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, recovery_time: float = 60.0) -> None:
        self.failure_threshold = max(1, failure_threshold)
        self.recovery_time = max(1.0, recovery_time)
        self._state: str = "closed"
        self._failure_count = 0
        self._opened_at: float | None = None

    def allow(self) -> bool:
        if self._state == "open":
            assert self._opened_at is not None
            if time.monotonic() - self._opened_at >= self.recovery_time:
                self._state = "half-open"
                return True
            return False
        return True

    def record_success(self) -> None:
        self._failure_count = 0
        self._state = "closed"
        self._opened_at = None

    def record_failure(self) -> None:
        self._failure_count += 1
        if self._failure_count >= self.failure_threshold:
            self._state = "open"
            self._opened_at = time.monotonic()

    @property
    def state(self) -> str:
        return self._state
