"""Compatibility wrapper for distributed Redis cache utilities.

Legacy modules import the cache helpers from ``src.core`` while the canonical
implementations live under ``src.engine.core``. This shim simply re-exports the
engine-layer constructs so both import paths remain valid during the migration.
"""

from __future__ import annotations

from src.engine.core.distributed_redis_cache import (  # noqa: F401
    DistributedRedisCache,
    CacheLevel,
    get_distributed_cache,
)

__all__ = ["DistributedRedisCache", "CacheLevel", "get_distributed_cache"]
