from __future__ import annotations

from src.core.unified_context_stream import (
    ContextEvent,
    ContextEventType,
    UnifiedContextStream,
    create_new_context_stream,
    get_unified_context_stream,
)
from src.core.incremental_context_manager import IncrementalContextManager
from src.interfaces import ContextStream

__all__ = [
    "ContextStream",
    "ContextEvent",
    "ContextEventType",
    "UnifiedContextStream",
    "IncrementalContextManager",
    "get_context_stream",
    "get_unified_context_stream",
    "create_new_context_stream",
]


def get_context_stream() -> ContextStream:
    """Return the shared UnifiedContextStream as a ContextStream protocol."""
    return get_unified_context_stream()
