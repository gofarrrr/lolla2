"""
METIS Multi-Layer Performance Cache System - Week 3.2 Upgrade
Distributed Redis-first caching replacing per-instance LRU caches

Week 3.2 Enhancements:
- Redis-first architecture with intelligent fallback
- Distributed cache invalidation with versioned keys
- Connection pooling and circuit breaker protection
- Automatic compression for large objects
- Performance Target: >90% cache hit rate with <50ms average latency

Based on industry insights:
- Cognition.ai: Multi-layer caching (L1: Memory, L2: Redis, L3: Patterns)
- LangChain: Semantic context caching with vector similarity
- Manus: Context relevance scoring and intelligent compression
"""

import asyncio
import hashlib
import json
import time
import logging
import os
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
from collections import OrderedDict
import pickle

# Week 3.2: Use new distributed Redis cache system
from src.core.distributed_redis_cache import (
    get_distributed_cache,
    DistributedRedisCache,
)

# Optional vector similarity support
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    VECTOR_AVAILABLE = True
except ImportError:
    VECTOR_AVAILABLE = False

logger = logging.getLogger(__name__)


class CacheLayer(str, Enum):
    """Cache layers following industry patterns"""

    L1_MEMORY = "l1_memory"  # Hot cache - frequently used mental models
    L2_REDIS = "l2_redis"  # Distributed cache - reasoning patterns
    L3_DATABASE = "l3_database"  # Persistent cache - historical successful patterns


class CacheEntryType(str, Enum):
    """Types of cached content"""

    MENTAL_MODEL = "mental_model"
    REASONING_PATTERN = "reasoning_pattern"
    CONTEXT_COMBINATION = "context_combination"
    ANALYSIS_RESULT = "analysis_result"
    BLUEPRINT = "blueprint"
    PHASE_RESULT = "phase_result"


@dataclass
class CacheEntry:
    """Cached content with metadata"""

    key: str
    content: Any
    entry_type: CacheEntryType
    created_at: datetime
    last_accessed: datetime
    access_count: int
    confidence_score: float
    relevance_score: float
    context_hash: str
    expiry_time: Optional[datetime] = None
    metadata: Dict[str, Any] = None


class LRUCache:
    """High-performance LRU cache for L1 layer"""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[CacheEntry]:
        """Get item from cache with LRU update"""
        if key in self.cache:
            # Move to end (most recently used)
            entry = self.cache.pop(key)
            self.cache[key] = entry
            entry.last_accessed = datetime.now()
            entry.access_count += 1
            self.hits += 1
            return entry
        else:
            self.misses += 1
            return None

    def put(self, key: str, entry: CacheEntry):
        """Add item to cache with LRU eviction"""
        if key in self.cache:
            self.cache.pop(key)
        elif len(self.cache) >= self.max_size:
            # Remove least recently used
            self.cache.popitem(last=False)

        self.cache[key] = entry

    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests) if total_requests > 0 else 0

        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "utilization": len(self.cache) / self.max_size,
        }


class SemanticSimilarityCache:
    """Semantic cache using vector embeddings for similar contexts"""

    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self.model = None
        self.embeddings: Dict[str, np.ndarray] = {}
        self.entries: Dict[str, CacheEntry] = {}

        if VECTOR_AVAILABLE:
            try:
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("✅ Semantic similarity cache initialized")
            except Exception as e:
                logger.warning(f"⚠️ Could not initialize semantic model: {e}")
                self.model = None

    def _generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding for text content"""
        if not self.model:
            return None

        try:
            return self.model.encode(text, convert_to_tensor=False)
        except Exception as e:
            logger.warning(f"⚠️ Embedding generation failed: {e}")
            return None

    def find_similar_entries(self, query: str) -> List[Tuple[str, float, CacheEntry]]:
        """Find semantically similar cached entries"""
        if not self.model or not self.embeddings:
            return []

        query_embedding = self._generate_embedding(query)
        if query_embedding is None:
            return []

        similar_entries = []

        for key, embedding in self.embeddings.items():
            try:
                # Calculate cosine similarity
                similarity = np.dot(query_embedding, embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
                )

                if similarity >= self.similarity_threshold:
                    entry = self.entries.get(key)
                    if entry:
                        similar_entries.append((key, float(similarity), entry))
            except Exception as e:
                logger.warning(f"⚠️ Similarity calculation failed for {key}: {e}")

        # Sort by similarity descending
        similar_entries.sort(key=lambda x: x[1], reverse=True)
        return similar_entries

    def add_entry(self, key: str, entry: CacheEntry, text_for_embedding: str):
        """Add entry with semantic indexing"""
        self.entries[key] = entry

        if self.model:
            embedding = self._generate_embedding(text_for_embedding)
            if embedding is not None:
                self.embeddings[key] = embedding


class MultiLayerCache:
    """
    Week 3.2: Distributed Redis-first multi-layer cache system
    Replaces per-instance in-memory LRU caches with centralized Redis
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()

        # Week 3.2: Use distributed Redis cache instead of separate L1/L2
        self.distributed_cache: Optional[DistributedRedisCache] = None
        # Initialize only when enabled; avoid awaiting in constructors
        if self.config.get("enable_distributed_cache", False):
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._initialize_distributed_cache())
            except RuntimeError:
                # No running loop; defer initialization until first awaitable call path
                # The cache will operate without distributed layer until then
                self.distributed_cache = None

        # Initialize Semantic Cache for specialized semantic matching
        self.semantic_cache = SemanticSimilarityCache(
            similarity_threshold=self.config["similarity_threshold"]
        )

        # Enhanced performance tracking for distributed architecture
        self.performance_stats = {
            "total_requests": 0,
            "distributed_hits": 0,  # Combined L1 + L2 from distributed cache
            "semantic_hits": 0,
            "database_hits": 0,  # L3 database hits
            "cache_misses": 0,
            "average_retrieval_time": 0.0,
            "redis_errors": 0,
            "compression_ratio": 0.0,
        }

        self.logger.info(
            "🚀 Distributed multi-layer cache system initialized (Week 3.2)"
        )

    def _get_default_config(self) -> Dict[str, Any]:
        """Default cache configuration - Week 3.2 distributed caching"""
        test_fast = str(os.getenv("TEST_FAST", "")).lower() in {"1", "true", "yes"}
        return {
            "distributed_cache_ttl": 3600,  # 1 hour for distributed cache
            "database_cache_ttl": 86400,  # 24 hours for L3 database cache
            # Hermetic mode: disable distributed cache to avoid background async init
            "enable_distributed_cache": (not test_fast),
            "similarity_threshold": 0.85,
            "max_semantic_entries": 5000,
            "compression_enabled": True,
            "performance_tracking": True,
            "cache_version": "v3.2",  # Version for cache key namespace
        }

    async def _initialize_distributed_cache(self):
        """Initialize distributed Redis cache system"""
        try:
            self.distributed_cache = await get_distributed_cache()
            self.logger.info("✅ Distributed Redis cache system initialized")
        except Exception as e:
            self.logger.warning(f"⚠️ Distributed cache initialization failed: {e}")
            self.distributed_cache = None

    def _generate_cache_key(
        self,
        content_type: CacheEntryType,
        primary_key: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate deterministic cache key"""
        key_components = [content_type.value, primary_key]

        if context:
            # Sort context keys for deterministic hashing
            context_str = json.dumps(context, sort_keys=True)
            context_hash = hashlib.md5(context_str.encode()).hexdigest()[:8]
            key_components.append(context_hash)

        return ":".join(key_components)

    def _serialize_entry(self, entry: CacheEntry) -> bytes:
        """Serialize cache entry for storage"""
        try:
            return pickle.dumps(asdict(entry))
        except Exception as e:
            self.logger.warning(f"⚠️ Serialization failed: {e}")
            return b""

    def _deserialize_entry(self, data: bytes) -> Optional[CacheEntry]:
        """Deserialize cache entry from storage"""
        try:
            entry_dict = pickle.loads(data)
            # Convert datetime strings back to datetime objects
            entry_dict["created_at"] = datetime.fromisoformat(entry_dict["created_at"])
            entry_dict["last_accessed"] = datetime.fromisoformat(
                entry_dict["last_accessed"]
            )
            if entry_dict.get("expiry_time"):
                entry_dict["expiry_time"] = datetime.fromisoformat(
                    entry_dict["expiry_time"]
                )

            return CacheEntry(**entry_dict)
        except Exception as e:
            self.logger.warning(f"⚠️ Deserialization failed: {e}")
            return None

    async def get(
        self,
        content_type: CacheEntryType,
        primary_key: str,
        context: Optional[Dict[str, Any]] = None,
        allow_semantic_match: bool = True,
    ) -> Tuple[Optional[Any], Optional[str]]:
        """
        Week 3.2: Distributed cache retrieval with fallback strategy
        Returns: (content, cache_layer_hit)
        """
        start_time = time.time()
        self.performance_stats["total_requests"] += 1

        cache_key = self._generate_cache_key(content_type, primary_key, context)

        # Week 3.2: Use distributed cache (includes L1 + L2 Redis)
        if self.distributed_cache:
            try:
                cached_value = await self.distributed_cache.get(
                    cache_key, content_type.value
                )
                if cached_value is not None:
                    retrieval_time = time.time() - start_time
                    self.performance_stats["distributed_hits"] += 1
                    self._update_performance_stats(retrieval_time)
                    self.logger.debug(
                        f"🎯 Distributed cache hit for {cache_key} in {retrieval_time*1000:.1f}ms"
                    )
                    return cached_value, "distributed_cache"
            except Exception as e:
                self.logger.warning(f"⚠️ Distributed cache retrieval failed: {e}")
                self.performance_stats["redis_errors"] += 1

        # Semantic similarity fallback (if enabled)
        if allow_semantic_match and content_type in [
            CacheEntryType.REASONING_PATTERN,
            CacheEntryType.ANALYSIS_RESULT,
        ]:
            semantic_query = f"{primary_key} {json.dumps(context or {})}"
            similar_entries = self.semantic_cache.find_similar_entries(semantic_query)

            if similar_entries:
                best_match = similar_entries[0]  # Highest similarity
                key, similarity, entry = best_match

                retrieval_time = time.time() - start_time
                self.performance_stats["semantic_hits"] += 1
                self._update_performance_stats(retrieval_time)
                self.logger.debug(
                    f"🎯 Semantic cache hit (similarity: {similarity:.3f}) for {cache_key} in {retrieval_time*1000:.1f}ms"
                )
                return entry.content, "semantic_cache"

        # Cache miss
        retrieval_time = time.time() - start_time
        self.performance_stats["cache_misses"] += 1
        self._update_performance_stats(retrieval_time)
        self.logger.debug(
            f"❌ Cache miss for {cache_key} in {retrieval_time*1000:.1f}ms"
        )
        return None, None

    async def put(
        self,
        content_type: CacheEntryType,
        primary_key: str,
        content: Any,
        context: Optional[Dict[str, Any]] = None,
        confidence_score: float = 1.0,
        ttl_seconds: Optional[int] = None,
    ):
        """Store content in multi-layer cache"""
        cache_key = self._generate_cache_key(content_type, primary_key, context)

        # Create cache entry
        now = datetime.now()
        expiry_time = None
        if ttl_seconds:
            expiry_time = now + timedelta(seconds=ttl_seconds)

        context_hash = hashlib.md5(
            json.dumps(context or {}, sort_keys=True).encode()
        ).hexdigest()[:16]

        entry = CacheEntry(
            key=cache_key,
            content=content,
            entry_type=content_type,
            created_at=now,
            last_accessed=now,
            access_count=1,
            confidence_score=confidence_score,
            relevance_score=1.0,  # Will be updated based on usage
            context_hash=context_hash,
            expiry_time=expiry_time,
            metadata={"original_context": context},
        )

        # Store in distributed cache if available
        if self.distributed_cache:
            try:
                await self.distributed_cache.put(cache_key, entry, ttl_seconds)
            except Exception as e:
                self.logger.warning(f"Failed to store in distributed cache: {e}")

        # Add to semantic cache for certain types
        if content_type in [
            CacheEntryType.REASONING_PATTERN,
            CacheEntryType.ANALYSIS_RESULT,
        ]:
            semantic_text = f"{primary_key} {json.dumps(context or {})}"
            self.semantic_cache.add_entry(cache_key, entry, semantic_text)

        self.logger.debug(f"💾 Cached {content_type.value} with key {cache_key}")

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry has expired"""
        if not entry.expiry_time:
            return False
        return datetime.now() > entry.expiry_time

    def _adapt_cache_entry(
        self, entry: CacheEntry, new_context: Dict[str, Any]
    ) -> CacheEntry:
        """Adapt cached entry for new context (LangChain pattern)"""
        # Create new entry with updated context
        adapted_entry = CacheEntry(
            key=entry.key + "_adapted",
            content=entry.content,
            entry_type=entry.entry_type,
            created_at=entry.created_at,
            last_accessed=datetime.now(),
            access_count=entry.access_count + 1,
            confidence_score=entry.confidence_score
            * 0.9,  # Slight reduction for adaptation
            relevance_score=entry.relevance_score * 0.95,
            context_hash=hashlib.md5(
                json.dumps(new_context, sort_keys=True).encode()
            ).hexdigest()[:16],
            expiry_time=entry.expiry_time,
            metadata={"adapted_from": entry.key, "new_context": new_context},
        )
        return adapted_entry

    def _update_performance_stats(self, retrieval_time: float):
        """Update running performance statistics"""
        current_avg = self.performance_stats["average_retrieval_time"]
        total_requests = self.performance_stats["total_requests"]

        # Running average calculation
        new_avg = (
            (current_avg * (total_requests - 1)) + retrieval_time
        ) / total_requests
        self.performance_stats["average_retrieval_time"] = new_avg

    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache performance statistics"""
        # Get stats from distributed cache if available
        distributed_stats = {}
        if self.distributed_cache:
            try:
                distributed_stats = await self.distributed_cache.get_stats()
            except:
                distributed_stats = {"hit_rate": 0, "total_entries": 0}

        total_hits = (
            self.performance_stats["l1_hits"]
            + self.performance_stats["l2_hits"]
            + self.performance_stats["semantic_hits"]
        )

        total_requests = self.performance_stats["total_requests"]
        overall_hit_rate = (total_hits / total_requests) if total_requests > 0 else 0

        return {
            "overall": {
                "total_requests": total_requests,
                "overall_hit_rate": overall_hit_rate,
                "average_retrieval_time_ms": self.performance_stats[
                    "average_retrieval_time"
                ]
                * 1000,
                "cache_misses": self.performance_stats["cache_misses"],
            },
            "distributed_cache": {
                **distributed_stats,
                "hits": self.performance_stats["l1_hits"]
                + self.performance_stats["l2_hits"],
            },
            "semantic": {
                "enabled": VECTOR_AVAILABLE,
                "hits": self.performance_stats["semantic_hits"],
                "entries_count": len(self.semantic_cache.entries),
            },
            "performance_targets": {
                "hit_rate_target": 0.80,
                "hit_rate_actual": overall_hit_rate,
                "hit_rate_status": (
                    "✅ Target Met" if overall_hit_rate >= 0.80 else "⚠️ Below Target"
                ),
            },
        }

    async def clear_expired_entries(self):
        """Maintenance task to clear expired entries"""
        cleared_count = 0

        # Clear expired entries from distributed cache if available
        if self.distributed_cache:
            try:
                cleared_count = await self.distributed_cache.clear_expired()
            except Exception as e:
                self.logger.warning(
                    f"Failed to clear expired entries from distributed cache: {e}"
                )

        self.logger.info(f"🧹 Cleared {cleared_count} expired cache entries")

    async def warmup_cache(self, mental_models: List[Dict[str, Any]]):
        """Pre-populate cache with frequently used mental models"""
        warmup_start = time.time()
        cached_count = 0

        for model in mental_models:
            try:
                await self.put(
                    content_type=CacheEntryType.MENTAL_MODEL,
                    primary_key=model.get("name", "unknown"),
                    content=model,
                    confidence_score=model.get("confidence", 1.0),
                    ttl_seconds=self.config["l2_ttl_seconds"],
                )
                cached_count += 1
            except Exception as e:
                self.logger.warning(
                    f"⚠️ Warmup failed for model {model.get('name')}: {e}"
                )

        warmup_time = time.time() - warmup_start
        self.logger.info(
            f"🔥 Cache warmed up with {cached_count} mental models in {warmup_time:.2f}s"
        )


# Smart Cache Enhancements
class SmartCacheEnhancements:
    """
    Smart enhancements for the MultiLayerCache that add intelligence without skipping phases.

    These enhancements make phases execute faster by:
    1. Predictive caching of likely-needed mental models
    2. Context-aware semantic deduplication
    3. Confidence-weighted cache prioritization
    4. Pattern-based prefetching
    5. Intelligent cache warming
    """

    def __init__(self, base_cache: MultiLayerCache):
        self.base_cache = base_cache
        self.logger = logging.getLogger(__name__)

        # Smart caching components
        self.pattern_predictor = CachePatternPredictor()
        self.confidence_weights = {}

        # Enhanced performance tracking
        self.smart_stats = {
            "predictive_hits": 0,
            "pattern_matches": 0,
            "confidence_optimizations": 0,
            "intelligent_operations": 0,
        }

        self.logger.info("🧠 Smart cache enhancements initialized")

    async def smart_cache_mental_models(
        self, problem_context: Dict[str, Any]
    ) -> List[str]:
        """
        Intelligently cache and suggest mental models based on problem context.
        This doesn't skip phases but makes phase execution faster.
        """
        problem_type = problem_context.get("problem_type", "analysis")
        industry = problem_context.get("industry", "general")
        complexity = problem_context.get("complexity", "moderate")

        # Check if we have cached model suggestions for this context
        context_key = f"smart_models:{problem_type}:{industry}:{complexity}"
        cached_models = await self.base_cache.get(
            CacheEntryType.MENTAL_MODEL, context_key, problem_context
        )

        if cached_models[0]:  # cached_models is tuple (content, source)
            self.smart_stats["predictive_hits"] += 1
            self.logger.info(
                f"🎯 Smart cache provided mental models from {cached_models[1]}"
            )
            return cached_models[0].get("suggested_models", [])

        # Generate intelligent model suggestions
        suggested_models = self._generate_smart_model_suggestions(problem_context)

        # Cache the suggestions with confidence weighting
        confidence = self._calculate_suggestion_confidence(
            problem_context, suggested_models
        )

        await self.base_cache.put(
            CacheEntryType.MENTAL_MODEL,
            context_key,
            {
                "suggested_models": suggested_models,
                "generated_at": datetime.now().isoformat(),
                "confidence": confidence,
            },
            context=problem_context,
            confidence_score=confidence,
            ttl_seconds=7200,  # 2 hours
        )

        # Record pattern for future predictions
        self.pattern_predictor.record_model_selection(problem_context, suggested_models)
        self.smart_stats["intelligent_operations"] += 1
        return suggested_models

    def _generate_smart_model_suggestions(self, context: Dict[str, Any]) -> List[str]:
        """Generate intelligent mental model suggestions based on context"""
        problem_type = context.get("problem_type", "analysis")
        industry = context.get("industry", "general")
        complexity = context.get("complexity", "moderate")

        base_models = ["systems_thinking", "critical_analysis"]

        # Add problem-type specific models (enhanced intelligence)
        if "strategic" in problem_type.lower():
            base_models.extend(
                ["mece_framework", "hypothesis_testing", "decision_analysis"]
            )
            if complexity in ["high", "strategic"]:
                base_models.extend(["scenario_analysis", "monte_carlo"])
        elif "creative" in problem_type.lower():
            base_models.extend(
                ["divergent_thinking", "lateral_thinking", "scamper_method"]
            )
            base_models.extend(["design_thinking", "innovation_frameworks"])
        elif "financial" in problem_type.lower():
            base_models.extend(
                ["financial_analysis", "risk_assessment", "cost_benefit"]
            )

        # Add industry-specific intelligence
        industry_lower = industry.lower()
        if "technology" in industry_lower or "saas" in industry_lower:
            base_models.extend(
                ["innovation_frameworks", "platform_thinking", "network_effects"]
            )
        elif "healthcare" in industry_lower:
            base_models.extend(
                ["regulatory_compliance", "patient_outcomes", "clinical_pathways"]
            )
        elif "financial" in industry_lower:
            base_models.extend(
                ["regulatory_frameworks", "risk_management", "compliance_analysis"]
            )

        # Add complexity-based models
        if complexity in ["high", "strategic"]:
            base_models.extend(["decision_trees", "game_theory", "systems_dynamics"])

        # Remove duplicates and prioritize by context relevance
        unique_models = list(set(base_models))
        return self._prioritize_models_by_context(unique_models, context)[:8]

    def _prioritize_models_by_context(
        self, models: List[str], context: Dict[str, Any]
    ) -> List[str]:
        """Prioritize models based on context relevance"""
        model_scores = {}

        for model in models:
            score = 1.0  # Base score

            # Boost core analytical models
            if model in ["systems_thinking", "critical_analysis", "mece_framework"]:
                score += 0.5

            # Boost based on problem complexity
            complexity = context.get("complexity", "moderate")
            if complexity in ["high", "strategic"] and "scenario" in model:
                score += 0.3
            elif complexity == "creative" and "thinking" in model:
                score += 0.3

            model_scores[model] = score

        # Sort by score and return
        return sorted(models, key=lambda m: model_scores.get(m, 0), reverse=True)

    def _calculate_suggestion_confidence(
        self, context: Dict[str, Any], models: List[str]
    ) -> float:
        """Calculate confidence in model suggestions based on context completeness"""
        base_confidence = 0.7

        # Boost confidence for complete context
        if context.get("problem_type") and context.get("industry"):
            base_confidence += 0.1
        if context.get("complexity"):
            base_confidence += 0.1
        if len(models) >= 5:  # Good model diversity
            base_confidence += 0.1

        return min(base_confidence, 1.0)

    async def cache_phase_results_intelligently(
        self,
        phase_name: str,
        phase_result: Dict[str, Any],
        context: Dict[str, Any],
        confidence: float,
    ):
        """
        Cache phase results intelligently without skipping phases.
        This caches intermediate results for reuse in similar contexts.
        """
        phase_key = f"phase_result:{phase_name}"

        # Weight cache priority by confidence
        cache_priority = confidence * context.get("importance", 1.0)
        self.confidence_weights[phase_key] = cache_priority

        # Determine TTL based on confidence and result type
        if confidence > 0.8:
            ttl = 7200  # 2 hours for high confidence results
        elif confidence > 0.6:
            ttl = 3600  # 1 hour for medium confidence
        else:
            ttl = 1800  # 30 minutes for low confidence

        await self.base_cache.put(
            CacheEntryType.PHASE_RESULT,
            phase_key,
            {
                "result": phase_result,
                "confidence": confidence,
                "context": context,
                "cached_at": datetime.now().isoformat(),
                "cache_priority": cache_priority,
            },
            context=context,
            confidence_score=confidence,
            ttl_seconds=ttl,
        )

        # Update pattern predictor
        self.pattern_predictor.record_phase_execution(phase_name, context, confidence)
        self.smart_stats["confidence_optimizations"] += 1

        self.logger.debug(
            f"💡 Intelligently cached {phase_name} result (confidence: {confidence:.2f}, priority: {cache_priority:.2f})"
        )

    def get_smart_stats(self) -> Dict[str, Any]:
        """Get smart cache enhancement statistics"""
        base_stats = self.base_cache.get_performance_stats()

        return {
            **base_stats,
            "smart_enhancements": self.smart_stats,
            "pattern_data": {
                "recorded_patterns": len(self.pattern_predictor.pattern_history),
                "confidence_weights": len(self.confidence_weights),
            },
            "intelligence_active": True,
        }


class CachePatternPredictor:
    """Predicts cache patterns for intelligent optimization"""

    def __init__(self):
        self.pattern_history = []
        self.phase_execution_history = []
        self.model_selection_history = []
        self.max_history_size = 1000

    def record_phase_execution(
        self, phase_name: str, context: Dict[str, Any], confidence: float
    ):
        """Record phase execution for pattern analysis"""
        self.phase_execution_history.append(
            {
                "phase": phase_name,
                "context": context,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat(),
            }
        )

        self._maintain_history_size()

    def record_model_selection(
        self, context: Dict[str, Any], selected_models: List[str]
    ):
        """Record model selection for pattern analysis"""
        self.model_selection_history.append(
            {
                "context": context,
                "models": selected_models,
                "timestamp": datetime.now().isoformat(),
            }
        )

        self._maintain_history_size()

    def _maintain_history_size(self):
        """Keep history within manageable size"""
        if len(self.phase_execution_history) > self.max_history_size:
            self.phase_execution_history = self.phase_execution_history[-800:]

        if len(self.model_selection_history) > self.max_history_size:
            self.model_selection_history = self.model_selection_history[-800:]


# Enhanced cache system factory function
def get_smart_performance_cache(
    config: Optional[Dict[str, Any]] = None,
) -> SmartCacheEnhancements:
    """
    Factory function to create smart-enhanced performance cache.

    Returns:
        SmartCacheEnhancements wrapping MultiLayerCache
    """
    base_cache = MultiLayerCache(config)
    return SmartCacheEnhancements(base_cache)


# Singleton instance for global access
_performance_cache: Optional[MultiLayerCache] = None
_smart_performance_cache: Optional[SmartCacheEnhancements] = None


def get_performance_cache(config: Optional[Dict[str, Any]] = None) -> MultiLayerCache:
    """Get singleton multi-layer cache instance"""
    global _performance_cache
    if _performance_cache is None:
        _performance_cache = MultiLayerCache(config)
    return _performance_cache


def get_smart_cache() -> SmartCacheEnhancements:
    """Get singleton smart cache instance"""
    global _smart_performance_cache
    if _smart_performance_cache is None:
        _smart_performance_cache = get_smart_performance_cache()
    return _smart_performance_cache
