from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Protocol, runtime_checkable


@runtime_checkable
class ContextStream(Protocol):
    """Contract for context streams consumed by infrastructure layers."""

    trace_id: str

    def add_event(
        self,
        event_type: str,
        data: Mapping[str, Any],
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        ...

    def get_events(self, event_type: Optional[str] = None) -> Iterable[Any]:
        ...

    def format_for_llm(self, *, compression: bool = False) -> str:
        ...

    def create_checkpoint(self) -> Mapping[str, Any]:
        ...

    def get_summary_metrics(self) -> Mapping[str, Any]:
        ...
