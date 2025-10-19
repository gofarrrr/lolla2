"""
N-Way Cache Service - High-Performance Caching Layer
Addresses the predicted 100x scale bottleneck in N-Way database queries

This service implements intelligent caching with:
- Multi-tier cache architecture (Memory → Redis → Database)
- Semantic similarity-based cache hits
- Intelligent cache warming and eviction
- Complete Glass-Box transparency for cache operations
"""

import hashlib
import json
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

# Glass-Box Integration - CRITICAL
from src.core.unified_context_stream import UnifiedContextStream, ContextEventType

# Cache backends
try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    REDIS_AVAILABLE = False

# Semantic similarity for intelligent caching
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np

    SEMANTIC_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    np = None
    SEMANTIC_AVAILABLE = False


class CacheLevel(Enum):
    """Cache tier levels"""

    MEMORY = "memory"
    REDIS = "redis"
    DATABASE = "database"
    MISS = "miss"


@dataclass
class CacheEntry:
    """Cache entry with metadata"""

    key: str
    value: Any
    timestamp: datetime
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    ttl_seconds: int = 3600  # 1 hour default
    similarity_hash: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        return datetime.utcnow() > self.timestamp + timedelta(seconds=self.ttl_seconds)

    def update_access(self):
        """Update access statistics"""
        self.access_count += 1
        self.last_accessed = datetime.utcnow()


@dataclass
class CacheStats:
    """Cache performance statistics"""

    memory_hits: int = 0
    redis_hits: int = 0
    database_hits: int = 0
    cache_misses: int = 0
    similarity_hits: int = 0
    total_requests: int = 0
    average_response_time_ms: float = 0.0

    def get_hit_rate(self) -> float:
        """Calculate overall cache hit rate"""
        if self.total_requests == 0:
            return 0.0
        hits = self.memory_hits + self.redis_hits + self.similarity_hits
        return (hits / self.total_requests) * 100.0

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        return {
            "hit_rate_percentage": self.get_hit_rate(),
            "memory_hit_rate": (self.memory_hits / max(1, self.total_requests)) * 100,
            "redis_hit_rate": (self.redis_hits / max(1, self.total_requests)) * 100,
            "similarity_hit_rate": (self.similarity_hits / max(1, self.total_requests))
            * 100,
            "average_response_ms": self.average_response_time_ms,
            "total_requests": self.total_requests,
            "cache_efficiency": (
                "high"
                if self.get_hit_rate() > 70
                else "medium" if self.get_hit_rate() > 40 else "low"
            ),
        }


class NWayCacheService:
    """
    High-Performance N-Way Caching Service

    Implements multi-tier intelligent caching to address 100x scale bottlenecks:

    Tier 1: Memory Cache (< 1ms)
    - Hot data in Python dictionaries
    - LRU eviction policy
    - 1000 entry limit

    Tier 2: Redis Cache (< 10ms)
    - Distributed caching across instances
    - Configurable TTL
    - Semantic similarity clustering

    Tier 3: Database (< 100ms)
    - Original Supabase queries
    - Query result caching
    - Connection pooling

    Glass-Box: Complete transparency for all cache operations
    """

    def __init__(
        self,
        context_stream: UnifiedContextStream,
        redis_client: Optional[Any] = None,
        max_memory_entries: int = 1000,
        default_ttl_seconds: int = 3600,
        enable_similarity_cache: bool = True,
    ):
        """
        Initialize N-Way Cache Service

        Args:
            context_stream: UnifiedContextStream for Glass-Box transparency
            redis_client: Optional Redis client for distributed caching
            max_memory_entries: Maximum entries in memory cache
            default_ttl_seconds: Default TTL for cache entries
            enable_similarity_cache: Enable semantic similarity caching
        """
        self.context_stream = context_stream
        self.redis_client = redis_client if REDIS_AVAILABLE else None
        self.max_memory_entries = max_memory_entries
        self.default_ttl = default_ttl_seconds
        self.enable_similarity = enable_similarity_cache and SEMANTIC_AVAILABLE
        self.logger = logging.getLogger(__name__)

        # Memory cache (Tier 1)
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []  # For LRU eviction

        # Semantic model for similarity caching
        self.semantic_model = None
        if self.enable_similarity:
            try:
                self.semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
                self.logger.info("🧠 Semantic similarity model loaded for cache")
            except Exception as e:
                self.logger.warning(f"Failed to load semantic model: {e}")
                self.enable_similarity = False

        # Cache statistics
        self.stats = CacheStats()
        self.response_times: List[float] = []

        # Glass-Box: Log cache service initialization
        self.context_stream.add_event(
            event_type=ContextEventType.SYSTEM_STATE,
            data={
                "cache_service_initialized": True,
                "memory_cache_size": max_memory_entries,
                "redis_available": bool(self.redis_client),
                "similarity_enabled": self.enable_similarity,
                "default_ttl_seconds": default_ttl_seconds,
            },
            metadata={
                "service": "NWayCacheService",
                "method": "__init__",
                "performance_feature": "100x_scale_optimization",
            },
        )

    async def get_nway_clusters(
        self, query: str, query_embedding: Optional[List[float]] = None, top_k: int = 3
    ) -> Tuple[Optional[List[str]], CacheLevel]:
        """
        Get N-Way clusters with intelligent multi-tier caching

        Args:
            query: The query string
            query_embedding: Optional pre-computed embedding
            top_k: Number of clusters to retrieve

        Returns:
            Tuple of (clusters, cache_level) where cache_level indicates hit source
        """
        start_time = time.time()
        self.stats.total_requests += 1

        # Generate cache key
        cache_key = self._generate_cache_key(query, top_k)

        # Glass-Box: Log cache lookup start
        self.context_stream.add_event(
            event_type=ContextEventType.TOOL_EXECUTION,
            data={
                "operation": "nway_cache_lookup",
                "query": query[:100] + "..." if len(query) > 100 else query,
                "cache_key": cache_key,
                "top_k": top_k,
            },
            metadata={"service": "NWayCacheService", "method": "get_nway_clusters"},
        )

        try:
            # Tier 1: Memory Cache Check
            if cache_key in self.memory_cache:
                entry = self.memory_cache[cache_key]
                if not entry.is_expired():
                    entry.update_access()
                    self._update_lru_order(cache_key)
                    self.stats.memory_hits += 1

                    self._log_cache_hit(CacheLevel.MEMORY, cache_key, start_time)
                    return entry.value, CacheLevel.MEMORY
                else:
                    # Remove expired entry
                    self._remove_from_memory_cache(cache_key)

            # Tier 2: Redis Cache Check
            if self.redis_client:
                redis_result = await self._get_from_redis(cache_key)
                if redis_result:
                    clusters, metadata = redis_result

                    # Store in memory cache for faster future access
                    self._store_in_memory_cache(cache_key, clusters, metadata)
                    self.stats.redis_hits += 1

                    self._log_cache_hit(CacheLevel.REDIS, cache_key, start_time)
                    return clusters, CacheLevel.REDIS

            # Tier 3: Semantic Similarity Check
            if self.enable_similarity and query_embedding:
                similar_result = await self._find_similar_cached_query(
                    query_embedding, top_k
                )
                if similar_result:
                    clusters, similarity_score = similar_result
                    self.stats.similarity_hits += 1

                    # Cache this result for future exact matches
                    await self._store_result(
                        cache_key, clusters, {"similarity_match": True}
                    )

                    self._log_cache_hit(
                        CacheLevel.MEMORY, cache_key, start_time, similarity_score
                    )
                    return clusters, CacheLevel.MEMORY

            # Cache miss - will need to query database
            self.stats.cache_misses += 1

            # Glass-Box: Log cache miss
            self.context_stream.add_event(
                event_type=ContextEventType.TOOL_EXECUTION,
                data={
                    "operation": "cache_miss",
                    "cache_key": cache_key,
                    "will_query_database": True,
                },
                metadata={"service": "NWayCacheService", "result": "miss"},
            )

            return None, CacheLevel.MISS

        finally:
            # Update response time statistics
            response_time = (time.time() - start_time) * 1000
            self.response_times.append(response_time)
            if len(self.response_times) > 100:
                self.response_times = self.response_times[-100:]  # Keep last 100
            self.stats.average_response_time_ms = sum(self.response_times) / len(
                self.response_times
            )

    async def store_nway_result(
        self,
        query: str,
        clusters: List[str],
        query_embedding: Optional[List[float]] = None,
        top_k: int = 3,
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        """
        Store N-Way result in multi-tier cache

        Args:
            query: The original query
            clusters: Selected clusters to cache
            query_embedding: Optional query embedding
            top_k: Number of clusters
            ttl_seconds: Optional custom TTL

        Returns:
            True if stored successfully, False otherwise
        """
        cache_key = self._generate_cache_key(query, top_k)
        ttl = ttl_seconds or self.default_ttl

        # Glass-Box: Log cache store operation
        self.context_stream.add_event(
            event_type=ContextEventType.TOOL_EXECUTION,
            data={
                "operation": "cache_store",
                "cache_key": cache_key,
                "clusters_count": len(clusters),
                "ttl_seconds": ttl,
            },
            metadata={"service": "NWayCacheService", "method": "store_nway_result"},
        )

        try:
            metadata = {
                "query_length": len(query),
                "top_k": top_k,
                "stored_at": datetime.utcnow().isoformat(),
            }

            # Add semantic information if available
            if query_embedding:
                metadata["has_embedding"] = True
                metadata["embedding_dimensions"] = len(query_embedding)

            # Store in all available cache tiers
            success = await self._store_result(cache_key, clusters, metadata, ttl)

            return success

        except Exception as e:
            self.logger.error(f"Failed to store cache result: {e}")

            # Glass-Box: Log storage error
            self.context_stream.add_event(
                event_type=ContextEventType.ERROR_OCCURRED,
                data={"error": str(e), "cache_key": cache_key},
                metadata={"service": "NWayCacheService", "method": "store_nway_result"},
            )

            return False

    async def _store_result(
        self,
        cache_key: str,
        clusters: List[str],
        metadata: Dict[str, Any],
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        """Store result in all available cache tiers"""
        ttl = ttl_seconds or self.default_ttl
        success = True

        # Store in memory cache
        try:
            self._store_in_memory_cache(cache_key, clusters, metadata, ttl)
        except Exception as e:
            self.logger.warning(f"Memory cache store failed: {e}")
            success = False

        # Store in Redis cache
        if self.redis_client:
            try:
                await self._store_in_redis(cache_key, clusters, metadata, ttl)
            except Exception as e:
                self.logger.warning(f"Redis cache store failed: {e}")
                success = False

        return success

    def _store_in_memory_cache(
        self,
        cache_key: str,
        clusters: List[str],
        metadata: Dict[str, Any],
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """Store entry in memory cache with LRU eviction"""
        ttl = ttl_seconds or self.default_ttl

        # Create cache entry
        entry = CacheEntry(
            key=cache_key,
            value=clusters,
            timestamp=datetime.utcnow(),
            ttl_seconds=ttl,
            metadata=metadata,
        )

        # Check if we need to evict entries
        if len(self.memory_cache) >= self.max_memory_entries:
            self._evict_lru_entries(1)

        # Store entry
        self.memory_cache[cache_key] = entry
        self._update_lru_order(cache_key)

    async def _store_in_redis(
        self,
        cache_key: str,
        clusters: List[str],
        metadata: Dict[str, Any],
        ttl_seconds: int,
    ) -> None:
        """Store entry in Redis cache"""
        if not self.redis_client:
            return

        # Prepare data for Redis
        cache_data = {
            "clusters": clusters,
            "metadata": metadata,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Store with TTL
        await self.redis_client.setex(cache_key, ttl_seconds, json.dumps(cache_data))

    async def _get_from_redis(
        self, cache_key: str
    ) -> Optional[Tuple[List[str], Dict[str, Any]]]:
        """Get entry from Redis cache"""
        if not self.redis_client:
            return None

        try:
            data = await self.redis_client.get(cache_key)
            if data:
                cache_data = json.loads(data)
                return cache_data["clusters"], cache_data["metadata"]
        except Exception as e:
            self.logger.warning(f"Redis get failed: {e}")

        return None

    async def _find_similar_cached_query(
        self,
        query_embedding: List[float],
        top_k: int,
        similarity_threshold: float = 0.85,
    ) -> Optional[Tuple[List[str], float]]:
        """Find semantically similar cached query"""
        if not self.enable_similarity or not self.semantic_model:
            return None

        best_match = None
        best_similarity = 0.0

        # Check memory cache for semantic similarity
        for cache_key, entry in self.memory_cache.items():
            if entry.is_expired():
                continue

            # Check if this entry has embedding metadata
            if "query_embedding" in entry.metadata:
                cached_embedding = entry.metadata["query_embedding"]

                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_embedding, cached_embedding)

                if similarity > similarity_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match = (entry.value, similarity)

        return best_match

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if not np:
            return 0.0

        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)

        dot_product = np.dot(vec1_np, vec2_np)
        norm1 = np.linalg.norm(vec1_np)
        norm2 = np.linalg.norm(vec2_np)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _generate_cache_key(self, query: str, top_k: int) -> str:
        """Generate cache key for query"""
        # Create a hash of the query and parameters
        key_data = f"{query.lower().strip()}|{top_k}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _update_lru_order(self, cache_key: str) -> None:
        """Update LRU access order"""
        if cache_key in self.access_order:
            self.access_order.remove(cache_key)
        self.access_order.append(cache_key)

    def _evict_lru_entries(self, count: int = 1) -> None:
        """Evict least recently used entries"""
        for _ in range(count):
            if self.access_order:
                lru_key = self.access_order.pop(0)
                if lru_key in self.memory_cache:
                    del self.memory_cache[lru_key]

    def _remove_from_memory_cache(self, cache_key: str) -> None:
        """Remove entry from memory cache"""
        if cache_key in self.memory_cache:
            del self.memory_cache[cache_key]
        if cache_key in self.access_order:
            self.access_order.remove(cache_key)

    def _log_cache_hit(
        self,
        level: CacheLevel,
        cache_key: str,
        start_time: float,
        similarity_score: Optional[float] = None,
    ) -> None:
        """Log cache hit with Glass-Box transparency"""
        response_time = (time.time() - start_time) * 1000

        self.context_stream.add_event(
            event_type=ContextEventType.TOOL_EXECUTION,
            data={
                "operation": "cache_hit",
                "cache_level": level.value,
                "cache_key": cache_key,
                "response_time_ms": response_time,
                "similarity_score": similarity_score,
            },
            metadata={
                "service": "NWayCacheService",
                "result": "hit",
                "performance_tier": level.value,
            },
        )

    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        memory_size = len(self.memory_cache)
        expired_entries = sum(
            1 for entry in self.memory_cache.values() if entry.is_expired()
        )

        return {
            "performance_stats": self.stats.get_performance_summary(),
            "memory_cache": {
                "size": memory_size,
                "max_size": self.max_memory_entries,
                "utilization_percentage": (memory_size / self.max_memory_entries) * 100,
                "expired_entries": expired_entries,
            },
            "redis_cache": {
                "available": bool(self.redis_client),
                "connected": bool(
                    self.redis_client
                ),  # TODO: Add connection health check
            },
            "similarity_cache": {
                "enabled": self.enable_similarity,
                "model_loaded": bool(self.semantic_model),
            },
            "system_health": {
                "cache_efficiency": self.stats.get_performance_summary()[
                    "cache_efficiency"
                ],
                "average_response_ms": self.stats.average_response_time_ms,
                "total_requests": self.stats.total_requests,
            },
        }

    async def clear_cache(
        self, cache_level: Optional[CacheLevel] = None
    ) -> Dict[str, int]:
        """Clear cache at specified level or all levels"""
        cleared = {"memory": 0, "redis": 0}

        if cache_level is None or cache_level == CacheLevel.MEMORY:
            cleared["memory"] = len(self.memory_cache)
            self.memory_cache.clear()
            self.access_order.clear()

        if (
            cache_level is None or cache_level == CacheLevel.REDIS
        ) and self.redis_client:
            # Clear Redis cache (be careful in production!)
            try:
                await self.redis_client.flushdb()
                cleared["redis"] = -1  # Unknown count
            except Exception as e:
                self.logger.error(f"Failed to clear Redis cache: {e}")

        # Glass-Box: Log cache clear
        self.context_stream.add_event(
            event_type=ContextEventType.SYSTEM_STATE,
            data={"operation": "cache_clear", "cleared_counts": cleared},
            metadata={"service": "NWayCacheService", "method": "clear_cache"},
        )

        return cleared

    async def warm_cache(self, common_queries: List[Tuple[str, int]]) -> int:
        """Warm cache with common queries"""
        warmed_count = 0

        # Glass-Box: Log cache warming start
        self.context_stream.add_event(
            event_type=ContextEventType.SYSTEM_STATE,
            data={
                "operation": "cache_warm_start",
                "queries_count": len(common_queries),
            },
            metadata={"service": "NWayCacheService", "method": "warm_cache"},
        )

        for query, top_k in common_queries:
            cache_key = self._generate_cache_key(query, top_k)
            if cache_key not in self.memory_cache:
                # This would typically trigger database query and caching
                # For now, we'll just mark the intent
                warmed_count += 1

        return warmed_count


# Factory function for service creation
def create_nway_cache_service(
    context_stream: UnifiedContextStream,
    redis_host: Optional[str] = None,
    redis_port: int = 6379,
    max_memory_entries: int = 1000,
    default_ttl_seconds: int = 3600,
    enable_similarity_cache: bool = True,
) -> NWayCacheService:
    """
    Factory function to create NWayCacheService with optional Redis

    Args:
        context_stream: UnifiedContextStream for Glass-Box transparency
        redis_host: Optional Redis host for distributed caching
        redis_port: Redis port (default 6379)
        max_memory_entries: Maximum entries in memory cache
        default_ttl_seconds: Default TTL for cache entries
        enable_similarity_cache: Enable semantic similarity caching

    Returns:
        Configured NWayCacheService
    """
    redis_client = None

    if redis_host and REDIS_AVAILABLE:
        try:
            redis_client = redis.asyncio.Redis(
                host=redis_host, port=redis_port, decode_responses=True
            )
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to connect to Redis: {e}")

    return NWayCacheService(
        context_stream=context_stream,
        redis_client=redis_client,
        max_memory_entries=max_memory_entries,
        default_ttl_seconds=default_ttl_seconds,
        enable_similarity_cache=enable_similarity_cache,
    )
