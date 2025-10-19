"""
Intelligent Caching System - Phase 4.1
Performance optimization through intelligent caching with adaptive strategies and predictive capabilities.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable, Tuple, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import defaultdict
from abc import ABC, abstractmethod

# Import circuit breaker for protection
try:
    from src.core.circuit_breaker import get_circuit_manager, CircuitBreakerConfig

    CIRCUIT_BREAKER_AVAILABLE = True
except ImportError:
    CIRCUIT_BREAKER_AVAILABLE = False

# Import event coordinator for integration
try:
    from src.core.event_driven_coordinator import get_event_coordinator, EventType

    EVENT_COORDINATOR_AVAILABLE = True
except ImportError:
    EVENT_COORDINATOR_AVAILABLE = False

T = TypeVar("T")


class CacheStrategy(str, Enum):
    """Caching strategies for different use cases"""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    ADAPTIVE = "adaptive"  # Adaptive based on access patterns
    PREDICTIVE = "predictive"  # Predictive pre-loading
    WRITE_THROUGH = "write_through"  # Write through to backing store
    WRITE_BACK = "write_back"  # Write back with delayed sync


class CacheEvent(str, Enum):
    """Cache events for monitoring and optimization"""

    HIT = "hit"
    MISS = "miss"
    EVICTION = "eviction"
    EXPIRATION = "expiration"
    PRELOAD = "preload"
    INVALIDATION = "invalidation"
    SYNC = "sync"


@dataclass
class CacheEntry(Generic[T]):
    """Individual cache entry with metadata"""

    key: str
    value: T
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl_seconds: Optional[float] = None
    size_bytes: int = 0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if entry has expired"""
        if self.ttl_seconds is None:
            return False
        return (datetime.now() - self.created_at).total_seconds() > self.ttl_seconds

    def touch(self):
        """Update access metadata"""
        self.last_accessed = datetime.now()
        self.access_count += 1


@dataclass
class CacheMetrics:
    """Comprehensive cache performance metrics"""

    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    evictions: int = 0
    expirations: int = 0
    preloads: int = 0
    invalidations: int = 0
    total_size_bytes: int = 0
    average_response_time: float = 0.0
    hit_rate: float = 0.0
    miss_rate: float = 0.0
    memory_usage_mb: float = 0.0
    key_access_patterns: Dict[str, int] = field(default_factory=dict)
    strategy_performance: Dict[str, float] = field(default_factory=dict)


class CacheBackend(ABC):
    """Abstract base class for cache backends"""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        pass

    @abstractmethod
    async def clear(self) -> bool:
        pass

    @abstractmethod
    async def keys(self, pattern: str = "*") -> List[str]:
        pass


class MemoryBackend(CacheBackend):
    """In-memory cache backend with size limits"""

    def __init__(self, max_size_mb: float = 100.0):
        self.storage: Dict[str, CacheEntry] = {}
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.current_size_bytes = 0
        self.lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[CacheEntry]:
        async with self.lock:
            entry = self.storage.get(key)
            if entry and not entry.is_expired():
                entry.touch()
                return entry
            elif entry and entry.is_expired():
                await self._remove_entry(key)
            return None

    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        async with self.lock:
            # Calculate entry size
            entry_size = self._calculate_size(value)

            # Remove existing entry if present
            if key in self.storage:
                await self._remove_entry(key)

            # Check if we need to evict entries
            while (
                self.current_size_bytes + entry_size > self.max_size_bytes
                and len(self.storage) > 0
            ):
                await self._evict_entry()

            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                ttl_seconds=ttl,
                size_bytes=entry_size,
            )

            self.storage[key] = entry
            self.current_size_bytes += entry_size
            return True

    async def delete(self, key: str) -> bool:
        async with self.lock:
            if key in self.storage:
                await self._remove_entry(key)
                return True
            return False

    async def clear(self) -> bool:
        async with self.lock:
            self.storage.clear()
            self.current_size_bytes = 0
            return True

    async def keys(self, pattern: str = "*") -> List[str]:
        async with self.lock:
            if pattern == "*":
                return list(self.storage.keys())
            # Simple pattern matching (could be enhanced)
            import fnmatch

            return [key for key in self.storage.keys() if fnmatch.fnmatch(key, pattern)]

    async def _remove_entry(self, key: str):
        """Remove entry and update size tracking"""
        if key in self.storage:
            entry = self.storage[key]
            self.current_size_bytes -= entry.size_bytes
            del self.storage[key]

    async def _evict_entry(self):
        """Evict least recently used entry"""
        if not self.storage:
            return

        # Find LRU entry
        lru_key = min(self.storage.keys(), key=lambda k: self.storage[k].last_accessed)
        await self._remove_entry(lru_key)

    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes"""
        try:
            # Use safer size calculation instead of pickle.dumps
            import sys

            return sys.getsizeof(value)
        except:
            return len(str(value).encode("utf-8"))


class IntelligentCache:
    """
    Intelligent caching system with adaptive strategies, predictive capabilities,
    and integrated performance optimization.
    """

    def __init__(
        self,
        name: str,
        backend: Optional[CacheBackend] = None,
        strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
        max_size_mb: float = 100.0,
        default_ttl: Optional[float] = None,
        enable_circuit_breaker: bool = True,
    ):
        self.name = name
        self.backend = backend or MemoryBackend(max_size_mb)
        self.strategy = strategy
        self.default_ttl = default_ttl
        self.metrics = CacheMetrics()
        self.logger = logging.getLogger(f"{__name__}.{name}")

        # Strategy-specific data structures
        self.access_frequency: Dict[str, int] = defaultdict(int)
        self.access_history: List[Tuple[str, datetime]] = []
        self.prediction_patterns: Dict[str, List[str]] = defaultdict(list)

        # Performance monitoring
        self.performance_history: List[Tuple[datetime, float]] = []
        self.adaptive_threshold = 0.8  # Hit rate threshold for strategy adaptation

        # Circuit breaker integration
        self.circuit_breaker_enabled = (
            enable_circuit_breaker and CIRCUIT_BREAKER_AVAILABLE
        )
        if self.circuit_breaker_enabled:
            self._setup_circuit_breaker()

        # Event coordination
        self.event_coordinator_enabled = EVENT_COORDINATOR_AVAILABLE

        # Background tasks
        self.cleanup_task: Optional[asyncio.Task] = None
        self.monitoring_task: Optional[asyncio.Task] = None
        self.running = False

        self.logger.info(
            f"Intelligent cache initialized: {name} with {strategy.value} strategy"
        )

    def _setup_circuit_breaker(self):
        """Setup circuit breaker for cache operations"""
        if not CIRCUIT_BREAKER_AVAILABLE:
            return

        manager = get_circuit_manager()
        config = CircuitBreakerConfig(
            name=f"cache_{self.name}",
            failure_threshold=5,
            timeout_seconds=5.0,
            slow_call_duration=1.0,
            slow_call_rate_threshold=0.3,
        )
        self.circuit = manager.register_circuit(config)

    async def start(self):
        """Start background tasks for cache management"""
        if not self.running:
            self.running = True
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            self.logger.info(f"Cache {self.name} started")

    async def stop(self):
        """Stop background tasks"""
        if self.running:
            self.running = False
            if self.cleanup_task:
                self.cleanup_task.cancel()
            if self.monitoring_task:
                self.monitoring_task.cancel()
            self.logger.info(f"Cache {self.name} stopped")

    async def get(self, key: str, default: Optional[T] = None) -> Optional[T]:
        """
        Get value from cache with intelligent prefetching and pattern recognition.

        Args:
            key: Cache key
            default: Default value if not found

        Returns:
            Cached value or default
        """
        start_time = time.time()

        try:
            # Use circuit breaker if enabled
            if self.circuit_breaker_enabled:
                result = await self.circuit.call(self._get_internal, key, default)
            else:
                result = await self._get_internal(key, default)

            # Update performance metrics
            response_time = time.time() - start_time
            await self._update_metrics(
                CacheEvent.HIT if result is not default else CacheEvent.MISS,
                response_time,
            )

            return result

        except Exception as e:
            self.logger.error(f"Cache get error for key {key}: {e}")
            response_time = time.time() - start_time
            await self._update_metrics(CacheEvent.MISS, response_time)
            return default

    async def _get_internal(self, key: str, default: Optional[T] = None) -> Optional[T]:
        """Internal get implementation"""
        # Record access pattern
        self.access_frequency[key] += 1
        self.access_history.append((key, datetime.now()))

        # Maintain access history size
        if len(self.access_history) > 1000:
            self.access_history = self.access_history[-500:]

        # Get from backend
        entry = await self.backend.get(key)

        if entry is not None:
            # Trigger predictive preloading
            await self._trigger_predictive_preload(key)
            return entry.value

        return default

    async def set(
        self,
        key: str,
        value: T,
        ttl: Optional[float] = None,
        tags: Optional[List[str]] = None,
    ) -> bool:
        """
        Set value in cache with intelligent strategy optimization.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            tags: Optional tags for grouping

        Returns:
            True if successful
        """
        try:
            # Use circuit breaker if enabled
            if self.circuit_breaker_enabled:
                result = await self.circuit.call(
                    self._set_internal, key, value, ttl or self.default_ttl, tags
                )
            else:
                result = await self._set_internal(
                    key, value, ttl or self.default_ttl, tags
                )

            await self._emit_cache_event(
                CacheEvent.HIT, {"key": key, "operation": "set"}
            )
            return result

        except Exception as e:
            self.logger.error(f"Cache set error for key {key}: {e}")
            return False

    async def _set_internal(
        self, key: str, value: T, ttl: Optional[float], tags: Optional[List[str]]
    ) -> bool:
        """Internal set implementation"""
        return await self.backend.set(key, value, ttl)

    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            result = await self.backend.delete(key)
            if result:
                await self._emit_cache_event(CacheEvent.INVALIDATION, {"key": key})
            return result
        except Exception as e:
            self.logger.error(f"Cache delete error for key {key}: {e}")
            return False

    async def clear(self) -> bool:
        """Clear all cache entries"""
        try:
            result = await self.backend.clear()
            if result:
                self.metrics = CacheMetrics()
                await self._emit_cache_event(
                    CacheEvent.INVALIDATION, {"operation": "clear_all"}
                )
            return result
        except Exception as e:
            self.logger.error(f"Cache clear error: {e}")
            return False

    async def invalidate_by_tags(self, tags: List[str]) -> int:
        """Invalidate all entries with matching tags"""
        invalidated = 0
        try:
            keys = await self.backend.keys()
            for key in keys:
                entry = await self.backend.get(key)
                if entry and any(tag in entry.tags for tag in tags):
                    if await self.backend.delete(key):
                        invalidated += 1

            if invalidated > 0:
                await self._emit_cache_event(
                    CacheEvent.INVALIDATION,
                    {"tags": tags, "invalidated_count": invalidated},
                )
        except Exception as e:
            self.logger.error(f"Tag invalidation error: {e}")

        return invalidated

    async def get_or_compute(
        self,
        key: str,
        compute_func: Callable[[], T],
        ttl: Optional[float] = None,
        tags: Optional[List[str]] = None,
    ) -> T:
        """
        Get from cache or compute and store the value.

        Args:
            key: Cache key
            compute_func: Function to compute value if not cached
            ttl: Time to live for computed value
            tags: Tags for the cached value

        Returns:
            Cached or computed value
        """
        # Try to get from cache first
        cached_value = await self.get(key)
        if cached_value is not None:
            return cached_value

        # Compute the value
        try:
            # Execute the compute function and handle both sync and async cases
            result = compute_func()

            # If the result is a coroutine, await it
            if asyncio.iscoroutine(result):
                computed_value = await result
            else:
                computed_value = result

            # Store in cache
            await self.set(key, computed_value, ttl, tags)
            return computed_value

        except Exception as e:
            self.logger.error(f"Compute function error for key {key}: {e}")
            raise

    async def _trigger_predictive_preload(self, accessed_key: str):
        """Trigger predictive preloading based on access patterns"""
        if self.strategy != CacheStrategy.PREDICTIVE:
            return

        # Simple pattern: if key A is accessed, preload related keys
        # This is a basic implementation - could be enhanced with ML
        if accessed_key in self.prediction_patterns:
            related_keys = self.prediction_patterns[accessed_key]
            for related_key in related_keys[:3]:  # Limit preloading
                # Check if related key is not already cached
                cached = await self.backend.get(related_key)
                if cached is None:
                    await self._emit_cache_event(
                        CacheEvent.PRELOAD,
                        {"source_key": accessed_key, "preload_key": related_key},
                    )

    async def _cleanup_loop(self):
        """Background cleanup of expired entries"""
        while self.running:
            try:
                await asyncio.sleep(60)  # Run every minute

                if isinstance(self.backend, MemoryBackend):
                    expired_keys = []
                    for key, entry in self.backend.storage.items():
                        if entry.is_expired():
                            expired_keys.append(key)

                    for key in expired_keys:
                        await self.backend.delete(key)
                        await self._emit_cache_event(
                            CacheEvent.EXPIRATION, {"key": key}
                        )
                        self.metrics.expirations += 1

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Cleanup loop error: {e}")

    async def _monitoring_loop(self):
        """Background monitoring and strategy optimization"""
        while self.running:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes

                # Calculate current hit rate
                total_requests = self.metrics.cache_hits + self.metrics.cache_misses
                if total_requests > 0:
                    hit_rate = self.metrics.cache_hits / total_requests
                    self.metrics.hit_rate = hit_rate
                    self.metrics.miss_rate = 1.0 - hit_rate

                # Adaptive strategy optimization
                if self.strategy == CacheStrategy.ADAPTIVE:
                    await self._optimize_strategy()

                # Emit monitoring event
                await self._emit_cache_event(
                    CacheEvent.HIT,
                    {
                        "monitoring": True,
                        "hit_rate": self.metrics.hit_rate,
                        "total_requests": total_requests,
                    },
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")

    async def _optimize_strategy(self):
        """Optimize caching strategy based on performance"""
        if self.metrics.hit_rate < self.adaptive_threshold:
            # Consider switching strategies or adjusting parameters
            self.logger.info(
                f"Cache {self.name} hit rate below threshold: {self.metrics.hit_rate:.2%}"
            )

            # Simple optimization: increase TTL for frequently accessed items
            if isinstance(self.backend, MemoryBackend):
                for key, entry in self.backend.storage.items():
                    if entry.access_count > 10 and entry.ttl_seconds:
                        # Extend TTL for popular items
                        entry.ttl_seconds *= 1.2

    async def _update_metrics(self, event: CacheEvent, response_time: float):
        """Update cache metrics"""
        self.metrics.total_requests += 1

        if event == CacheEvent.HIT:
            self.metrics.cache_hits += 1
        elif event == CacheEvent.MISS:
            self.metrics.cache_misses += 1

        # Update hit/miss rates
        total_requests = self.metrics.cache_hits + self.metrics.cache_misses
        if total_requests > 0:
            self.metrics.hit_rate = self.metrics.cache_hits / total_requests
            self.metrics.miss_rate = self.metrics.cache_misses / total_requests

        # Update average response time
        total = self.metrics.total_requests
        current_avg = self.metrics.average_response_time
        self.metrics.average_response_time = (
            current_avg * (total - 1) + response_time
        ) / total

        # Update memory usage for memory backend
        if isinstance(self.backend, MemoryBackend):
            self.metrics.memory_usage_mb = self.backend.current_size_bytes / (
                1024 * 1024
            )
            self.metrics.total_size_bytes = self.backend.current_size_bytes

    async def _emit_cache_event(self, event: CacheEvent, payload: Dict[str, Any]):
        """Emit cache event for coordination"""
        if not self.event_coordinator_enabled:
            return

        try:
            coordinator = await get_event_coordinator()
            await coordinator.emit_event(
                EventType.METRICS_UPDATE,
                source_component=f"cache_{self.name}",
                payload={
                    "cache_event": event.value,
                    "cache_name": self.name,
                    **payload,
                },
            )
        except Exception as e:
            self.logger.debug(f"Failed to emit cache event: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get current cache metrics"""
        return {
            "name": self.name,
            "strategy": self.strategy.value,
            "total_requests": self.metrics.total_requests,
            "cache_hits": self.metrics.cache_hits,
            "cache_misses": self.metrics.cache_misses,
            "hit_rate": self.metrics.hit_rate,
            "miss_rate": self.metrics.miss_rate,
            "average_response_time": self.metrics.average_response_time,
            "memory_usage_mb": self.metrics.memory_usage_mb,
            "total_size_bytes": self.metrics.total_size_bytes,
            "evictions": self.metrics.evictions,
            "expirations": self.metrics.expirations,
            "circuit_breaker_enabled": self.circuit_breaker_enabled,
        }

    async def warm_up(self, warm_up_data: Dict[str, Any], ttl: Optional[float] = None):
        """Warm up cache with initial data"""
        self.logger.info(
            f"Warming up cache {self.name} with {len(warm_up_data)} entries"
        )

        for key, value in warm_up_data.items():
            await self.set(key, value, ttl)

        await self._emit_cache_event(
            CacheEvent.PRELOAD, {"warm_up": True, "entries_loaded": len(warm_up_data)}
        )


class CacheManager:
    """
    Manages multiple intelligent caches with coordination and optimization.
    """

    def __init__(self):
        self.caches: Dict[str, IntelligentCache] = {}
        self.global_metrics = {
            "total_caches": 0,
            "total_memory_mb": 0.0,
            "average_hit_rate": 0.0,
            "total_requests": 0,
        }
        self.logger = logging.getLogger(__name__)

    def create_cache(
        self,
        name: str,
        strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
        max_size_mb: float = 100.0,
        default_ttl: Optional[float] = None,
    ) -> IntelligentCache:
        """Create a new intelligent cache"""
        if name in self.caches:
            self.logger.warning(f"Cache {name} already exists")
            return self.caches[name]

        cache = IntelligentCache(
            name=name,
            strategy=strategy,
            max_size_mb=max_size_mb,
            default_ttl=default_ttl,
        )

        self.caches[name] = cache
        self.global_metrics["total_caches"] = len(self.caches)

        self.logger.info(f"Created cache: {name}")
        return cache

    def get_cache(self, name: str) -> Optional[IntelligentCache]:
        """Get cache by name"""
        return self.caches.get(name)

    async def start_all(self):
        """Start all caches"""
        for cache in self.caches.values():
            await cache.start()
        self.logger.info(f"Started {len(self.caches)} caches")

    async def stop_all(self):
        """Stop all caches"""
        for cache in self.caches.values():
            await cache.stop()
        self.logger.info(f"Stopped {len(self.caches)} caches")

    def get_global_metrics(self) -> Dict[str, Any]:
        """Get global cache metrics"""
        total_memory = 0.0
        total_hit_rate = 0.0
        total_requests = 0

        for cache in self.caches.values():
            metrics = cache.get_metrics()
            total_memory += metrics["memory_usage_mb"]
            total_requests += metrics["total_requests"]
            if metrics["total_requests"] > 0:
                total_hit_rate += metrics["hit_rate"]

        avg_hit_rate = total_hit_rate / len(self.caches) if self.caches else 0.0

        self.global_metrics.update(
            {
                "total_memory_mb": total_memory,
                "average_hit_rate": avg_hit_rate,
                "total_requests": total_requests,
            }
        )

        return self.global_metrics.copy()

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get metrics for all caches"""
        cache_metrics = {}
        for name, cache in self.caches.items():
            cache_metrics[name] = cache.get_metrics()

        return {
            "global_metrics": self.get_global_metrics(),
            "cache_metrics": cache_metrics,
        }


# Global cache manager instance
_cache_manager_instance = None


def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance"""
    global _cache_manager_instance

    if _cache_manager_instance is None:
        _cache_manager_instance = CacheManager()

    return _cache_manager_instance


# Decorator for caching function results
def cached(
    cache_name: str = "default",
    ttl: Optional[float] = None,
    key_func: Optional[Callable] = None,
):
    """
    Decorator to cache function results.

    Args:
        cache_name: Name of cache to use
        ttl: Time to live for cached results
        key_func: Function to generate cache key from arguments
    """

    def decorator(func):
        async def wrapper(*args, **kwargs):
            manager = get_cache_manager()

            # Get or create cache
            cache = manager.get_cache(cache_name)
            if not cache:
                cache = manager.create_cache(cache_name)
                await cache.start()

            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = ":".join(key_parts)

            # Get or compute result
            async def compute_result():
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)

            return await cache.get_or_compute(
                key=cache_key, compute_func=compute_result, ttl=ttl
            )

        return wrapper

    return decorator
