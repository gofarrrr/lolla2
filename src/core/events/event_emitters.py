"""Unified event emitter helpers for LLM, research, and tool events.

These helpers provide a single place to enforce payload structure and
prevent drift across modules when emitting UnifiedContextStream events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

from src.core.unified_context_stream import UnifiedContextStream, ContextEventType


@dataclass
class BaseEventEmitter:
    """Base helper with shared emit logic."""

    context_stream: UnifiedContextStream
    default_metadata: Dict[str, Any] = field(default_factory=dict)

    def _emit(self, event_type: ContextEventType, payload: Dict[str, Any]) -> None:
        data = {**payload}
        metadata = {**self.default_metadata}
        self.context_stream.add_event(event_type, data, metadata)


class ToolEventEmitter(BaseEventEmitter):
    """Emit tool-related events with consistent fields."""

    def decision(
        self,
        tool_name: str,
        selection_reasoning: str,
        alternatives: Optional[list[str]] = None,
        **extra: Any,
    ) -> None:
        self._emit(
            ContextEventType.TOOL_DECISION,
            {
                "tool_name": tool_name,
                "selection_reasoning": selection_reasoning,
                "alternatives_considered": alternatives or [],
                **extra,
            },
        )

    def call_start(
        self,
        tool_name: str,
        inputs_fingerprint: Optional[str] = None,
        **extra: Any,
    ) -> None:
        self._emit(
            ContextEventType.TOOL_CALL_START,
            {
                "tool_name": tool_name,
                "inputs_fingerprint": inputs_fingerprint,
                **extra,
            },
        )

    def call_complete(
        self,
        tool_name: str,
        latency_ms: Optional[int] = None,
        result_fingerprint: Optional[str] = None,
        **extra: Any,
    ) -> None:
        self._emit(
            ContextEventType.TOOL_CALL_COMPLETE,
            {
                "tool_name": tool_name,
                "latency_ms": latency_ms,
                "result_fingerprint": result_fingerprint,
                **extra,
            },
        )

    def execution(
        self,
        tool_name: str,
        action: str,
        latency_ms: Optional[int] = None,
        **extra: Any,
    ) -> None:
        self._emit(
            ContextEventType.TOOL_EXECUTION,
            {
                "tool_name": tool_name,
                "action": action,
                "latency_ms": latency_ms,
                **extra,
            },
        )


class LLMEventEmitter(BaseEventEmitter):
    """Emit LLM request/response events with consistent fields."""

    def request(
        self,
        provider: str,
        model: str,
        **extra: Any,
    ) -> None:
        self._emit(
            ContextEventType.LLM_PROVIDER_REQUEST,
            {
                "provider": provider,
                "model": model,
                **extra,
            },
        )

    def response(
        self,
        provider: str,
        model: str,
        latency_ms: Optional[int] = None,
        tokens_used: Optional[int] = None,
        **extra: Any,
    ) -> None:
        self._emit(
            ContextEventType.LLM_PROVIDER_RESPONSE,
            {
                "provider": provider,
                "model": model,
                "latency_ms": latency_ms,
                "response_time_ms": latency_ms,
                "tokens_used": tokens_used,
                **extra,
            },
        )


class ResearchEventEmitter(BaseEventEmitter):
    """Emit research provider events consistently."""

    def request(
        self,
        provider: str,
        query: str,
        **extra: Any,
    ) -> None:
        self._emit(
            ContextEventType.RESEARCH_PROVIDER_REQUEST,
            {
                "provider": provider,
                "query": query,
                **extra,
            },
        )

    def response(
        self,
        provider: str,
        status: str,
        latency_ms: Optional[int] = None,
        citations_count: Optional[int] = None,
        result_preview: Optional[str] = None,
        **extra: Any,
    ) -> None:
        self._emit(
            ContextEventType.RESEARCH_PROVIDER_RESPONSE,
            {
                "provider": provider,
                "status": status,
                "latency_ms": latency_ms,
                "processing_time_ms": latency_ms,
                "citations_count": citations_count,
                "result_preview": result_preview,
                **extra,
            },
        )

    def fallback(
        self,
        failed_provider: str,
        fallback_provider: str,
        failure_reason: Optional[str] = None,
        **extra: Any,
    ) -> None:
        self._emit(
            ContextEventType.RESEARCH_PROVIDER_FALLBACK,
            {
                "failed_provider": failed_provider,
                "fallback_provider": fallback_provider,
                "failure_reason": failure_reason,
                **extra,
            },
        )

