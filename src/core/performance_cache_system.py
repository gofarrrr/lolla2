"""Compatibility shim for the performance cache system.

Legacy modules import performance cache utilities from ``src.core`` while the
current implementation lives under ``src.engine.core``. This facade simply
re-exports the canonical definitions so downstream imports continue to work
without modification.
"""

from __future__ import annotations

from src.engine.core.performance_cache_system import (  # noqa: F401
    CacheEntryType,
    get_performance_cache,
    get_smart_cache,
)

__all__ = ["CacheEntryType", "get_performance_cache", "get_smart_cache"]
