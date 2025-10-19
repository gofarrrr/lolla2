from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class AttemptMetrics:
    """Structured record describing an individual provider attempt."""

    provider: str
    attempt: int
    latency_ms: float
    status: str
    error: Optional[str] = None
    circuit_state: Optional[str] = None


class LLMObservability:
    """Emit structured logs (and future metrics) for LLM provider attempts."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self._logger = logger or logging.getLogger("llm.observability")

    def record_attempt(self, metrics: AttemptMetrics) -> None:
        payload = {
            "provider": metrics.provider,
            "attempt": metrics.attempt,
            "latency_ms": round(metrics.latency_ms, 2),
            "status": metrics.status,
        }
        if metrics.error:
            payload["error"] = metrics.error
        if metrics.circuit_state:
            payload["circuit_state"] = metrics.circuit_state

        self._logger.info("llm.attempt %s", json.dumps(payload, sort_keys=True))

    def record_retry_scheduled(
        self,
        *,
        provider: str,
        attempt: int,
        delay_seconds: float,
        error: Exception,
    ) -> None:
        self._logger.warning(
            "llm.retry %s",
            json.dumps(
                {
                    "provider": provider,
                    "attempt": attempt,
                    "delay_seconds": round(delay_seconds, 2),
                    "error": str(error),
                },
                sort_keys=True,
            ),
        )

    def record_fallback_chain(
        self,
        *,
        initial_provider: str,
        chain: Iterable[str],
        selected: str,
    ) -> None:
        self._logger.info(
            "llm.fallback %s",
            json.dumps(
                {
                    "initial_provider": initial_provider,
                    "candidate_chain": list(chain),
                    "selected_provider": selected,
                },
                sort_keys=True,
            ),
        )
