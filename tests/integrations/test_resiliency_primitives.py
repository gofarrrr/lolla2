from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from src.integrations.llm.resiliency import CircuitBreaker, RetryPolicy


@pytest.mark.asyncio
async def test_retry_policy_succeeds_after_retries() -> None:
    attempts = {"count": 0}
    observed_retries: list[tuple[int, str]] = []

    async def operation() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("transient failure")
        return "ok"

    def on_retry(attempt: int, exc: Exception, delay: float) -> None:
        observed_retries.append((attempt, str(exc)))

    policy = RetryPolicy(max_attempts=3, base_delay=0.0, max_delay=0.0, jitter=0.0)
    result = await policy.execute(operation, on_retry=on_retry)

    assert result == "ok"
    assert attempts["count"] == 3
    assert observed_retries == [(1, "transient failure"), (2, "transient failure")]


@pytest.mark.asyncio
async def test_retry_policy_raises_after_max_attempts() -> None:
    async def operation() -> str:
        raise ValueError("persistent failure")

    policy = RetryPolicy(max_attempts=2, base_delay=0.0, max_delay=0.0, jitter=0.0)

    with pytest.raises(ValueError):
        await policy.execute(operation)


def test_circuit_breaker_state_transitions(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = {"value": 0.0}

    fake_time = SimpleNamespace(monotonic=lambda: clock["value"])
    monkeypatch.setattr("src.integrations.llm.resiliency.time", fake_time)

    breaker = CircuitBreaker(failure_threshold=2, recovery_time=5.0)

    assert breaker.allow()
    breaker.record_failure()
    assert breaker.allow(), "Circuit should remain closed until threshold reached"

    breaker.record_failure()
    assert not breaker.allow(), "Circuit should open after threshold failures"

    clock["value"] = 10.0  # Advance time beyond recovery window
    assert breaker.allow(), "Circuit should allow requests in half-open state after cooldown"

    breaker.record_success()
    assert breaker.allow(), "Circuit should close after successful call"
