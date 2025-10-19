from __future__ import annotations

from typing import Any, Optional, Protocol, Sequence, runtime_checkable

from src.core.services.context_metrics_service import ContextMetricsService


@runtime_checkable
class ContextMetrics(Protocol):
    """Contract for context metrics operations consumed outside core."""

    def get_relevant_context(
        self,
        for_phase: Optional[str] = None,
        min_relevance: float = 0.3,
    ) -> Sequence[Any]:
        """Return events considered relevant for the requested phase."""

    def get_recent_events(self, limit: int = 10) -> Sequence[Any]:
        """Return the most recent events captured by the stream."""

    def recalculate_relevance(self, event: Any) -> float:
        """Recalculate the relevance score for a given event."""

    def compress_old_events(self) -> None:
        """Purge or compress old events to respect memory limits."""

    def summarize_event(self, event: Any) -> str:
        """Produce a human-readable summary of a context event."""


class ContextMetricsAdapter(ContextMetrics):
    """Adapter exposing ContextMetricsService through the ContextMetrics protocol."""

    def __init__(self, service: ContextMetricsService) -> None:
        self._service = service

    @property
    def service(self) -> ContextMetricsService:
        """Return the wrapped service for advanced use cases."""
        return self._service

    def get_relevant_context(
        self,
        for_phase: Optional[str] = None,
        min_relevance: float = 0.3,
    ) -> Sequence[Any]:
        return self._service.get_relevant_context(for_phase=for_phase, min_relevance=min_relevance)

    def get_recent_events(self, limit: int = 10) -> Sequence[Any]:
        return self._service.get_recent_events(limit=limit)

    def recalculate_relevance(self, event: Any) -> float:
        return self._service.recalculate_relevance(event)

    def compress_old_events(self) -> None:
        self._service.compress_old_events()

    def summarize_event(self, event: Any) -> str:
        return self._service.summarize_event(event)


__all__ = ["ContextMetrics", "ContextMetricsAdapter"]
