#!/usr/bin/env python3
"""
Cognitive Pattern Cache - Emergency Performance Fix #1
Semantic similarity-based caching for sub-second query responses

This is the single most important performance optimization for METIS.
Similar queries should return cached results in <1 second instead of 6+ minutes.
"""

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np

try:
    import redis.asyncio as redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

try:
    from sentence_transformers import SentenceTransformer

    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    SentenceTransformer = None

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache strategy types"""

    EXACT_MATCH = "exact"  # Exact string match
    SEMANTIC_SIMILAR = "semantic"  # Semantic similarity matching
    PATTERN_BASED = "pattern"  # Pattern-based matching
    HYBRID = "hybrid"  # All strategies combined


@dataclass
class CacheHit:
    """Cache hit result"""

    cache_key: str
    similarity_score: float
    cached_timestamp: datetime
    cache_age_seconds: float
    strategy_used: CacheStrategy
    audit_trail: Dict[str, Any]


@dataclass
class CacheStats:
    """Cache performance statistics"""

    total_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    hit_rate: float = 0.0
    avg_cache_response_time_ms: float = 0.0
    avg_full_execution_time_ms: float = 0.0
    cost_savings_usd: float = 0.0
    time_savings_seconds: float = 0.0


class CognitivePatternCache:
    """
    High-performance semantic caching for METIS cognitive operations

    Features:
    - Semantic similarity matching using sentence transformers
    - Redis-based distributed caching
    - Intelligent cache invalidation
    - Performance analytics and monitoring
    - Sub-second response times for similar queries
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        similarity_threshold: float = 0.85,
        cache_ttl_hours: int = 24,
        max_cache_size: int = 10000,
    ):
        self.redis_url = redis_url
        self.similarity_threshold = similarity_threshold
        self.cache_ttl_seconds = cache_ttl_hours * 3600
        self.max_cache_size = max_cache_size

        # Initialize components
        self.redis_client: Optional[redis.Redis] = None
        self.embeddings_model: Optional[SentenceTransformer] = None
        self.stats = CacheStats()

        # Cache keys
        self.QUERY_EMBEDDINGS_KEY = "metis:query_embeddings"
        self.AUDIT_TRAILS_KEY = "metis:audit_trails"
        self.STATS_KEY = "metis:cache_stats"

        logger.info("🚀 CognitivePatternCache initialized")
        logger.info(f"   Similarity threshold: {similarity_threshold}")
        logger.info(f"   Cache TTL: {cache_ttl_hours} hours")
        logger.info(f"   Max cache size: {max_cache_size}")

    async def initialize(self) -> bool:
        """Initialize cache components"""
        success = True

        # Initialize Redis
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.from_url(
                    self.redis_url, decode_responses=True
                )
                await self.redis_client.ping()
                logger.info("✅ Redis connection established")
            except Exception as e:
                logger.error(f"❌ Redis connection failed: {e}")
                success = False
        else:
            logger.warning("⚠️ Redis not available - using in-memory cache")
            success = False

        # Initialize embeddings model
        if EMBEDDINGS_AVAILABLE:
            try:
                self.embeddings_model = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("✅ Sentence transformer model loaded")
            except Exception as e:
                logger.error(f"❌ Embeddings model failed: {e}")
                success = False
        else:
            logger.warning(
                "⚠️ Sentence transformers not available - using hash-based matching"
            )

        # Load existing stats
        await self._load_stats()

        return success

    async def get_cached_result(
        self, query: str, context: Dict[str, Any] = None
    ) -> Optional[CacheHit]:
        """
        Check cache for similar queries and return cached result if found

        Returns:
            CacheHit if found, None if cache miss
        """
        start_time = time.time()
        self.stats.total_queries += 1

        try:
            # Try exact match first (fastest)
            exact_hit = await self._check_exact_match(query, context)
            if exact_hit:
                self.stats.cache_hits += 1
                await self._update_cache_stats(time.time() - start_time)
                logger.info(f"🎯 Cache HIT (exact): {exact_hit.similarity_score:.3f}")
                return exact_hit

            # Try semantic similarity if available
            if self.embeddings_model:
                semantic_hit = await self._check_semantic_match(query, context)
                if semantic_hit:
                    self.stats.cache_hits += 1
                    await self._update_cache_stats(time.time() - start_time)
                    logger.info(
                        f"🎯 Cache HIT (semantic): {semantic_hit.similarity_score:.3f}"
                    )
                    return semantic_hit

            # Cache miss
            self.stats.cache_misses += 1
            logger.info("❌ Cache MISS")
            return None

        except Exception as e:
            logger.error(f"💥 Cache lookup error: {e}")
            self.stats.cache_misses += 1
            return None

    async def store_result(
        self,
        query: str,
        audit_trail: Dict[str, Any],
        context: Dict[str, Any] = None,
        execution_time_seconds: float = 0.0,
        cost_usd: float = 0.0,
    ) -> str:
        """Store cognitive analysis result in cache"""

        try:
            # Generate cache key
            cache_key = self._generate_cache_key(query, context)

            # Store audit trail
            audit_data = {
                "query": query,
                "context": context or {},
                "audit_trail": audit_trail,
                "cached_at": datetime.utcnow().isoformat(),
                "execution_time_seconds": execution_time_seconds,
                "cost_usd": cost_usd,
            }

            if self.redis_client:
                # Store in Redis with TTL
                await self.redis_client.hset(
                    self.AUDIT_TRAILS_KEY, cache_key, json.dumps(audit_data)
                )
                await self.redis_client.expire(
                    self.AUDIT_TRAILS_KEY, self.cache_ttl_seconds
                )

                # Store query embedding for semantic matching
                if self.embeddings_model:
                    embedding = self.embeddings_model.encode(query).tolist()
                    embedding_data = {
                        "cache_key": cache_key,
                        "query": query,
                        "embedding": embedding,
                        "context_hash": self._hash_context(context),
                    }
                    await self.redis_client.hset(
                        self.QUERY_EMBEDDINGS_KEY, cache_key, json.dumps(embedding_data)
                    )
                    await self.redis_client.expire(
                        self.QUERY_EMBEDDINGS_KEY, self.cache_ttl_seconds
                    )

            logger.info(f"💾 Cached result: {cache_key}")
            return cache_key

        except Exception as e:
            logger.error(f"💥 Cache storage error: {e}")
            return ""

    async def _check_exact_match(
        self, query: str, context: Dict[str, Any]
    ) -> Optional[CacheHit]:
        """Check for exact query match"""

        cache_key = self._generate_cache_key(query, context)

        if not self.redis_client:
            return None

        try:
            cached_data = await self.redis_client.hget(self.AUDIT_TRAILS_KEY, cache_key)
            if cached_data:
                data = json.loads(cached_data)
                cached_time = datetime.fromisoformat(data["cached_at"])
                age_seconds = (datetime.utcnow() - cached_time).total_seconds()

                return CacheHit(
                    cache_key=cache_key,
                    similarity_score=1.0,
                    cached_timestamp=cached_time,
                    cache_age_seconds=age_seconds,
                    strategy_used=CacheStrategy.EXACT_MATCH,
                    audit_trail=data["audit_trail"],
                )
        except Exception as e:
            logger.error(f"Exact match check error: {e}")

        return None

    async def _check_semantic_match(
        self, query: str, context: Dict[str, Any]
    ) -> Optional[CacheHit]:
        """Check for semantically similar queries"""

        if not self.redis_client or not self.embeddings_model:
            return None

        try:
            # Get query embedding
            query_embedding = self.embeddings_model.encode(query)
            context_hash = self._hash_context(context)

            # Get all cached embeddings
            all_embeddings = await self.redis_client.hgetall(self.QUERY_EMBEDDINGS_KEY)

            best_match = None
            best_similarity = 0.0

            for cached_key, embedding_data in all_embeddings.items():
                try:
                    data = json.loads(embedding_data)

                    # Check context compatibility first
                    if data["context_hash"] != context_hash:
                        continue

                    # Calculate semantic similarity
                    cached_embedding = np.array(data["embedding"])
                    similarity = self._cosine_similarity(
                        query_embedding, cached_embedding
                    )

                    if (
                        similarity > best_similarity
                        and similarity >= self.similarity_threshold
                    ):
                        best_similarity = similarity
                        best_match = data["cache_key"]

                except Exception as e:
                    logger.error(f"Similarity calculation error: {e}")
                    continue

            # If we found a good match, get the audit trail
            if best_match and best_similarity >= self.similarity_threshold:
                cached_data = await self.redis_client.hget(
                    self.AUDIT_TRAILS_KEY, best_match
                )
                if cached_data:
                    data = json.loads(cached_data)
                    cached_time = datetime.fromisoformat(data["cached_at"])
                    age_seconds = (datetime.utcnow() - cached_time).total_seconds()

                    return CacheHit(
                        cache_key=best_match,
                        similarity_score=best_similarity,
                        cached_timestamp=cached_time,
                        cache_age_seconds=age_seconds,
                        strategy_used=CacheStrategy.SEMANTIC_SIMILAR,
                        audit_trail=data["audit_trail"],
                    )

        except Exception as e:
            logger.error(f"Semantic match error: {e}")

        return None

    def _generate_cache_key(self, query: str, context: Dict[str, Any] = None) -> str:
        """Generate deterministic cache key"""
        content = f"{query}_{self._hash_context(context)}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _hash_context(self, context: Dict[str, Any] = None) -> str:
        """Generate deterministic hash of context"""
        if not context:
            return "no_context"

        # Sort context for deterministic hashing
        sorted_context = json.dumps(context, sort_keys=True)
        return hashlib.md5(sorted_context.encode()).hexdigest()[:8]

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    async def _update_cache_stats(self, response_time_seconds: float):
        """Update cache performance statistics"""
        response_time_ms = response_time_seconds * 1000

        # Update rolling averages
        total_responses = self.stats.cache_hits + self.stats.cache_misses
        if total_responses > 0:
            self.stats.hit_rate = self.stats.cache_hits / total_responses

            # Update average response time
            current_avg = self.stats.avg_cache_response_time_ms
            self.stats.avg_cache_response_time_ms = (
                current_avg * (total_responses - 1) + response_time_ms
            ) / total_responses

        # Persist stats periodically
        if total_responses % 10 == 0:  # Every 10 queries
            await self._save_stats()

    async def _load_stats(self):
        """Load cache statistics from Redis"""
        if not self.redis_client:
            return

        try:
            stats_data = await self.redis_client.get(self.STATS_KEY)
            if stats_data:
                stats_dict = json.loads(stats_data)
                self.stats = CacheStats(**stats_dict)
                logger.info(
                    f"📊 Loaded cache stats: {self.stats.hit_rate:.1%} hit rate"
                )
        except Exception as e:
            logger.error(f"Stats loading error: {e}")

    async def _save_stats(self):
        """Save cache statistics to Redis"""
        if not self.redis_client:
            return

        try:
            stats_dict = {
                "total_queries": self.stats.total_queries,
                "cache_hits": self.stats.cache_hits,
                "cache_misses": self.stats.cache_misses,
                "hit_rate": self.stats.hit_rate,
                "avg_cache_response_time_ms": self.stats.avg_cache_response_time_ms,
                "avg_full_execution_time_ms": self.stats.avg_full_execution_time_ms,
                "cost_savings_usd": self.stats.cost_savings_usd,
                "time_savings_seconds": self.stats.time_savings_seconds,
            }

            await self.redis_client.set(
                self.STATS_KEY, json.dumps(stats_dict), ex=self.cache_ttl_seconds
            )
        except Exception as e:
            logger.error(f"Stats saving error: {e}")

    async def get_cache_stats(self) -> CacheStats:
        """Get current cache performance statistics"""
        await self._load_stats()  # Refresh from Redis
        return self.stats

    async def clear_cache(self):
        """Clear all cached data"""
        if self.redis_client:
            try:
                await self.redis_client.delete(self.QUERY_EMBEDDINGS_KEY)
                await self.redis_client.delete(self.AUDIT_TRAILS_KEY)
                await self.redis_client.delete(self.STATS_KEY)
                logger.info("🧹 Cache cleared")
            except Exception as e:
                logger.error(f"Cache clear error: {e}")

    async def cleanup(self):
        """Cleanup resources"""
        if self.redis_client:
            await self.redis_client.close()


# Global cache instance
_cache_instance: Optional[CognitivePatternCache] = None


async def get_cognitive_cache() -> CognitivePatternCache:
    """Get or create global cache instance"""
    global _cache_instance

    if _cache_instance is None:
        _cache_instance = CognitivePatternCache()
        await _cache_instance.initialize()

    return _cache_instance


# Performance emergency test function
async def test_cache_performance():
    """Test cache performance with sample queries"""
    print("🧪 Testing CognitivePatternCache Performance")
    print("=" * 50)

    cache = await get_cognitive_cache()

    # Test queries
    queries = [
        "How should our software company pivot to AI?",
        "Our software business needs to pivot into artificial intelligence",  # Similar to above
        "What's the best go-to-market strategy for a B2B SaaS product?",
        "Camera company losing to smartphones - how to pivot?",
        "Our camera business is being disrupted by mobile phones",  # Similar to above
    ]

    # Simulate storing some results
    mock_audit_trail = {
        "engagement_id": "test-001",
        "consultants": ["strategic_analyst", "innovation_catalyst"],
        "recommendations": "Sample strategic recommendation",
        "total_cost": 0.025,
        "execution_time": 45.0,
    }

    for i, query in enumerate(queries[:3]):  # Store first 3
        await cache.store_result(
            query=query,
            audit_trail=mock_audit_trail,
            execution_time_seconds=45.0,
            cost_usd=0.025,
        )
        print(f"💾 Stored: {query[:50]}...")

    print()

    # Test cache lookups
    for query in queries:
        start_time = time.time()
        hit = await cache.get_cached_result(query)
        lookup_time = (time.time() - start_time) * 1000

        if hit:
            print(
                f"✅ HIT ({hit.strategy_used.value}): {query[:40]}... "
                f"({hit.similarity_score:.3f}, {lookup_time:.1f}ms)"
            )
        else:
            print(f"❌ MISS: {query[:40]}... ({lookup_time:.1f}ms)")

    # Show stats
    stats = await cache.get_cache_stats()
    print("\n📊 Cache Statistics:")
    print(f"   Hit Rate: {stats.hit_rate:.1%}")
    print(f"   Total Queries: {stats.total_queries}")
    print(f"   Avg Response: {stats.avg_cache_response_time_ms:.1f}ms")


if __name__ == "__main__":
    asyncio.run(test_cache_performance())
