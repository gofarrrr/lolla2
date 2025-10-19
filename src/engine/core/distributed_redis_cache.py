"""
METIS Distributed Redis Cache System - Week 3.2 Implementation
Replaces per-instance in-memory LRU cache with centralized Redis cache

Features:
- Redis-first architecture with fallback to in-memory
- Cache key versioning for deployment safety
- Distributed cache invalidation
- Memory-efficient compression
- Connection pooling and circuit breakers
- Automatic failover and recovery
"""

import asyncio
import pickle
import time
import zlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

# Redis imports with fallback
try:
    import redis.asyncio as redis
    from redis.asyncio.sentinel import Sentinel
    from redis.exceptions import ConnectionError, TimeoutError, RedisError

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Circuit breaker for Redis operations
try:
    from src.core.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

    CIRCUIT_BREAKER_AVAILABLE = True
except ImportError:
    CIRCUIT_BREAKER_AVAILABLE = False

from src.config import get_settings
from src.core.structured_logging import get_logger

settings = get_settings()
logger = get_logger(__name__, component="distributed_cache")


class CacheLevel(str, Enum):
    """Cache levels for distributed architecture"""

    L1_LOCAL = "l1_local"  # Local in-memory cache
    L2_REDIS = "l2_redis"  # Distributed Redis cache
    L3_PERSISTENT = "l3_persistent"  # Database persistent cache


class CacheOperation(str, Enum):
    """Cache operations for monitoring"""

    GET = "get"
    SET = "set"
    DELETE = "delete"
    INVALIDATE = "invalidate"
    EXPIRE = "expire"


@dataclass
class DistributedCacheEntry:
    """Cache entry optimized for distributed storage"""

    key: str
    value: Any
    created_at: datetime
    expires_at: Optional[datetime]
    version: str
    content_type: str
    compressed: bool = False
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class CacheStats:
    """Cache performance statistics"""

    total_requests: int = 0
    l1_hits: int = 0
    l2_hits: int = 0
    l3_hits: int = 0
    misses: int = 0
    errors: int = 0
    total_latency_ms: float = 0.0
    compression_ratio: float = 0.0
    memory_saved_mb: float = 0.0

    @property
    def hit_rate(self) -> float:
        """Calculate overall hit rate"""
        if self.total_requests == 0:
            return 0.0
        return (self.l1_hits + self.l2_hits + self.l3_hits) / self.total_requests

    @property
    def average_latency_ms(self) -> float:
        """Calculate average latency"""
        if self.total_requests == 0:
            return 0.0
        return self.total_latency_ms / self.total_requests


class DistributedRedisCache:
    """
    Redis-first distributed cache system with intelligent fallback

    Week 3.2 Implementation:
    - Replaces per-instance in-memory LRU with centralized Redis
    - Adds cache key versioning for safe deployments
    - Implements distributed invalidation
    - Provides automatic failover to local cache
    """

    def __init__(
        self,
        redis_url: str = None,
        key_prefix: str = "metis",
        version: str = "v1",
        compression_threshold: int = 1024,
        local_cache_size: int = 1000,
        default_ttl: int = 3600,
    ):
        """Initialize distributed cache system"""
        self.redis_url = redis_url or settings.REDIS_URL or "redis://localhost:6379"
        self.key_prefix = key_prefix
        self.version = version
        self.compression_threshold = compression_threshold
        self.local_cache_size = local_cache_size
        self.default_ttl = default_ttl

        # Redis connection pool
        self.redis_pool: Optional[redis.ConnectionPool] = None
        self.redis_client: Optional[redis.Redis] = None

        # Local L1 cache as fallback
        self.local_cache: Dict[str, DistributedCacheEntry] = {}
        self.local_cache_order: List[str] = []  # For LRU eviction

        # Circuit breaker for Redis operations
        self.circuit_breaker = None
        if CIRCUIT_BREAKER_AVAILABLE:
            self.circuit_breaker = CircuitBreaker(
                failure_threshold=5, recovery_timeout=30, expected_exception=RedisError
            )

        # Performance tracking
        self.stats = CacheStats()

        # Initialize Redis connection
        asyncio.create_task(self._initialize_redis())

        logger.info(
            "🚀 Distributed Redis cache initialized",
            redis_url=self._mask_redis_url(self.redis_url),
            version=self.version,
            compression_threshold=self.compression_threshold,
        )

    def _mask_redis_url(self, url: str) -> str:
        """Mask sensitive information in Redis URL for logging"""
        if "://" not in url:
            return url
        protocol, rest = url.split("://", 1)
        if "@" in rest:
            credentials, host = rest.split("@", 1)
            return f"{protocol}://***:***@{host}"
        return url

    async def _initialize_redis(self):
        """Initialize Redis connection with connection pooling"""
        if not REDIS_AVAILABLE:
            logger.warning("⚠️ Redis not available, using local cache only")
            return

        try:
            # Create connection pool for better performance
            self.redis_pool = redis.ConnectionPool.from_url(
                self.redis_url,
                max_connections=20,
                socket_timeout=2.0,
                socket_connect_timeout=2.0,
                health_check_interval=30,
                retry_on_timeout=True,
            )

            self.redis_client = redis.Redis(
                connection_pool=self.redis_pool,
                decode_responses=False,  # Handle binary data
            )

            # Test connection
            await self.redis_client.ping()
            logger.info("✅ Redis connection pool initialized successfully")

        except Exception as e:
            logger.error(f"❌ Redis initialization failed: {e}")
            self.redis_client = None

    def _generate_cache_key(self, key: str, content_type: str = "general") -> str:
        """Generate versioned cache key for Redis"""
        # Include version in key for deployment safety
        return f"{self.key_prefix}:{self.version}:{content_type}:{key}"

    def _compress_data(self, data: bytes) -> Tuple[bytes, bool]:
        """Compress data if it exceeds threshold"""
        if len(data) < self.compression_threshold:
            return data, False

        try:
            compressed = zlib.compress(data, level=6)  # Balanced compression
            if len(compressed) < len(data):
                self.stats.compression_ratio = len(compressed) / len(data)
                self.stats.memory_saved_mb += (len(data) - len(compressed)) / (
                    1024 * 1024
                )
                return compressed, True
            return data, False
        except Exception as e:
            logger.warning(f"⚠️ Compression failed: {e}")
            return data, False

    def _decompress_data(self, data: bytes, compressed: bool) -> bytes:
        """Decompress data if needed"""
        if not compressed:
            return data

        try:
            return zlib.decompress(data)
        except Exception as e:
            logger.error(f"❌ Decompression failed: {e}")
            raise

    def _serialize_entry(self, entry: DistributedCacheEntry) -> bytes:
        """Serialize cache entry for storage"""
        try:
            # Convert to dict with ISO format dates
            entry_dict = asdict(entry)
            entry_dict["created_at"] = entry.created_at.isoformat()
            if entry.expires_at:
                entry_dict["expires_at"] = entry.expires_at.isoformat()

            return pickle.dumps(entry_dict, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.error(f"❌ Serialization failed: {e}")
            raise

    def _deserialize_entry(self, data: bytes) -> DistributedCacheEntry:
        """Deserialize cache entry from storage"""
        try:
            entry_dict = pickle.loads(data)

            # Convert ISO format dates back to datetime
            entry_dict["created_at"] = datetime.fromisoformat(entry_dict["created_at"])
            if entry_dict.get("expires_at"):
                entry_dict["expires_at"] = datetime.fromisoformat(
                    entry_dict["expires_at"]
                )

            return DistributedCacheEntry(**entry_dict)
        except Exception as e:
            logger.error(f"❌ Deserialization failed: {e}")
            raise

    async def _redis_operation(self, operation: CacheOperation, *args, **kwargs) -> Any:
        """Execute Redis operation with circuit breaker protection"""
        if not self.redis_client:
            raise ConnectionError("Redis client not available")

        if self.circuit_breaker:
            return await self.circuit_breaker.call(
                self._execute_redis_operation, operation, *args, **kwargs
            )
        else:
            return await self._execute_redis_operation(operation, *args, **kwargs)

    async def _execute_redis_operation(
        self, operation: CacheOperation, *args, **kwargs
    ) -> Any:
        """Execute the actual Redis operation"""
        if operation == CacheOperation.GET:
            return await self.redis_client.get(args[0])
        elif operation == CacheOperation.SET:
            return await self.redis_client.setex(
                args[0], args[1], args[2]
            )  # key, ttl, value
        elif operation == CacheOperation.DELETE:
            return await self.redis_client.delete(*args)
        elif operation == CacheOperation.INVALIDATE:
            # Pattern-based invalidation
            pattern = args[0]
            keys = await self.redis_client.keys(pattern)
            if keys:
                return await self.redis_client.delete(*keys)
            return 0
        elif operation == CacheOperation.EXPIRE:
            return await self.redis_client.expire(args[0], args[1])
        else:
            raise ValueError(f"Unsupported operation: {operation}")

    def _update_local_cache(self, key: str, entry: DistributedCacheEntry):
        """Update local L1 cache with LRU eviction"""
        if key in self.local_cache:
            # Move to end (most recent)
            self.local_cache_order.remove(key)
        elif len(self.local_cache) >= self.local_cache_size:
            # Evict least recently used
            lru_key = self.local_cache_order.pop(0)
            del self.local_cache[lru_key]

        self.local_cache[key] = entry
        self.local_cache_order.append(key)

    def _get_from_local_cache(self, key: str) -> Optional[DistributedCacheEntry]:
        """Get from local L1 cache"""
        entry = self.local_cache.get(key)
        if entry:
            # Check expiration
            if entry.expires_at and datetime.utcnow() > entry.expires_at:
                del self.local_cache[key]
                self.local_cache_order.remove(key)
                return None

            # Update LRU order
            self.local_cache_order.remove(key)
            self.local_cache_order.append(key)
            return entry

        return None

    async def get(self, key: str, content_type: str = "general") -> Optional[Any]:
        """
        Get value from distributed cache with L1 -> L2 fallback

        Args:
            key: Cache key
            content_type: Type of content for key namespacing

        Returns:
            Cached value or None if not found
        """
        start_time = time.time()
        self.stats.total_requests += 1

        try:
            cache_key = self._generate_cache_key(key, content_type)

            # L1: Check local cache first
            entry = self._get_from_local_cache(cache_key)
            if entry:
                self.stats.l1_hits += 1
                logger.debug("🎯 L1 cache hit", key=key, content_type=content_type)
                return entry.value

            # L2: Check Redis cache
            try:
                redis_data = await self._redis_operation(CacheOperation.GET, cache_key)
                if redis_data:
                    # Decompress and deserialize
                    decompressed_data = self._decompress_data(redis_data, True)
                    entry = self._deserialize_entry(decompressed_data)

                    # Check expiration
                    if entry.expires_at and datetime.utcnow() > entry.expires_at:
                        await self._redis_operation(CacheOperation.DELETE, cache_key)
                        self.stats.misses += 1
                        return None

                    # Update local cache
                    self._update_local_cache(cache_key, entry)

                    self.stats.l2_hits += 1
                    logger.debug(
                        "🎯 L2 (Redis) cache hit", key=key, content_type=content_type
                    )
                    return entry.value

            except Exception as e:
                logger.warning(f"⚠️ Redis get operation failed: {e}", key=key)
                self.stats.errors += 1

            # Cache miss
            self.stats.misses += 1
            logger.debug("❌ Cache miss", key=key, content_type=content_type)
            return None

        finally:
            self.stats.total_latency_ms += (time.time() - start_time) * 1000

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        content_type: str = "general",
    ) -> bool:
        """
        Set value in distributed cache

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            content_type: Type of content for key namespacing

        Returns:
            True if successfully cached
        """
        try:
            cache_key = self._generate_cache_key(key, content_type)
            ttl = ttl or self.default_ttl
            expires_at = datetime.utcnow() + timedelta(seconds=ttl)

            # Create cache entry
            entry = DistributedCacheEntry(
                key=cache_key,
                value=value,
                created_at=datetime.utcnow(),
                expires_at=expires_at,
                version=self.version,
                content_type=content_type,
            )

            # Serialize and compress
            serialized_data = self._serialize_entry(entry)
            compressed_data, is_compressed = self._compress_data(serialized_data)
            entry.compressed = is_compressed

            # Update local cache immediately
            self._update_local_cache(cache_key, entry)

            # Update Redis cache
            try:
                await self._redis_operation(
                    CacheOperation.SET, cache_key, ttl, compressed_data
                )
                logger.debug(
                    "✅ Cached in distributed cache",
                    key=key,
                    content_type=content_type,
                    ttl=ttl,
                    compressed=is_compressed,
                )
                return True

            except Exception as e:
                logger.warning(f"⚠️ Redis set operation failed: {e}", key=key)
                self.stats.errors += 1
                # Still return True since local cache was updated
                return True

        except Exception as e:
            logger.error(
                f"❌ Cache set failed: {e}", key=key, content_type=content_type
            )
            return False

    async def delete(self, key: str, content_type: str = "general") -> bool:
        """Delete key from distributed cache"""
        try:
            cache_key = self._generate_cache_key(key, content_type)

            # Remove from local cache
            self.local_cache.pop(cache_key, None)
            if cache_key in self.local_cache_order:
                self.local_cache_order.remove(cache_key)

            # Remove from Redis
            try:
                await self._redis_operation(CacheOperation.DELETE, cache_key)
            except Exception as e:
                logger.warning(f"⚠️ Redis delete operation failed: {e}", key=key)

            logger.debug("🗑️ Deleted from cache", key=key, content_type=content_type)
            return True

        except Exception as e:
            logger.error(f"❌ Cache delete failed: {e}", key=key)
            return False

    async def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all keys matching a pattern
        Useful for cache busting after deployments
        """
        try:
            cache_pattern = self._generate_cache_key(pattern, "*")

            # Clear matching keys from local cache
            keys_to_remove = [
                k for k in self.local_cache.keys() if k.startswith(cache_pattern)
            ]
            for key in keys_to_remove:
                del self.local_cache[key]
                if key in self.local_cache_order:
                    self.local_cache_order.remove(key)

            # Clear from Redis
            try:
                deleted_count = await self._redis_operation(
                    CacheOperation.INVALIDATE, cache_pattern
                )
                logger.info(
                    "🧹 Invalidated cache pattern",
                    pattern=pattern,
                    deleted_count=deleted_count,
                    local_deleted=len(keys_to_remove),
                )
                return deleted_count

            except Exception as e:
                logger.warning(f"⚠️ Redis pattern invalidation failed: {e}")
                return len(keys_to_remove)

        except Exception as e:
            logger.error(f"❌ Pattern invalidation failed: {e}")
            return 0

    async def health_check(self) -> Dict[str, Any]:
        """Health check for cache system"""
        health = {
            "status": "healthy",
            "local_cache_size": len(self.local_cache),
            "redis_available": False,
            "stats": asdict(self.stats),
            "version": self.version,
        }

        # Test Redis connection
        if self.redis_client:
            try:
                await self.redis_client.ping()
                health["redis_available"] = True
            except Exception as e:
                health["status"] = "degraded"
                health["redis_error"] = str(e)

        return health

    async def close(self):
        """Close connections and cleanup resources"""
        if self.redis_client:
            await self.redis_client.close()

        if self.redis_pool:
            await self.redis_pool.disconnect()

        logger.info("🔌 Distributed cache connections closed")


# Global cache instance
_distributed_cache: Optional[DistributedRedisCache] = None


async def get_distributed_cache() -> DistributedRedisCache:
    """Get global distributed cache instance"""
    global _distributed_cache
    if _distributed_cache is None:
        _distributed_cache = DistributedRedisCache()
    return _distributed_cache


async def initialize_distributed_cache(
    redis_url: str = None, version: str = "v1"
) -> DistributedRedisCache:
    """Initialize distributed cache with custom configuration"""
    global _distributed_cache
    _distributed_cache = DistributedRedisCache(redis_url=redis_url, version=version)
    return _distributed_cache
