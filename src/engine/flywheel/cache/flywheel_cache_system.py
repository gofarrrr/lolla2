"""
METIS Production Flywheel Caching System
Intelligent caching layer that learns and optimizes from user interaction patterns
"""

import json
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging
from collections import defaultdict
import redis.asyncio as redis
import pickle
import numpy as np

logger = logging.getLogger(__name__)


class CacheLayer(Enum):
    """Cache layers in the flywheel system"""

    L1_MEMORY = "l1_memory"  # In-process memory cache
    L2_REDIS = "l2_redis"  # Redis distributed cache
    L3_PERSISTENT = "l3_persistent"  # Database persistent cache
    L4_LEARNING = "l4_learning"  # Learning pattern cache


class MemoryTier(Enum):
    """Hierarchical memory management tiers"""

    SHORT_TERM = "short_term"  # Active working memory (minutes to hours)
    MEDIUM_TERM = "medium_term"  # Intermediate consolidation (hours to days)
    LONG_TERM = "long_term"  # Permanent knowledge storage (persistent)
    EPISODIC = "episodic"  # Event-based memory (specific interactions)
    SEMANTIC = "semantic"  # Generalized knowledge (patterns and rules)


@dataclass
class HierarchicalMemoryEntry:
    """Memory entry with hierarchical memory management"""

    key: str
    data: Any
    memory_tier: MemoryTier
    importance_score: float  # 0.0-1.0 for consolidation decisions
    access_frequency: int = 0
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    created_at: datetime = field(default_factory=datetime.utcnow)
    consolidation_count: int = 0  # How many times consolidated
    related_entries: List[str] = field(default_factory=list)  # Semantic connections

    def calculate_retention_value(self) -> float:
        """Calculate value for memory consolidation decisions"""
        # Time decay factor
        age_hours = (datetime.utcnow() - self.created_at).total_seconds() / 3600
        time_decay = np.exp(-age_hours / 168)  # 1-week half-life

        # Access pattern value
        access_value = min(1.0, self.access_frequency / 10)  # Normalize to 10 accesses

        # Importance multiplier
        importance_multiplier = self.importance_score

        # Consolidation bonus (learning value)
        consolidation_bonus = min(0.3, self.consolidation_count * 0.1)

        return (
            time_decay * 0.4
            + access_value * 0.4
            + importance_multiplier * 0.2
            + consolidation_bonus
        )

    def should_promote_to_long_term(self) -> bool:
        """Determine if entry should be promoted to long-term memory"""
        retention_value = self.calculate_retention_value()
        return (
            retention_value > 0.7
            and self.access_frequency >= 3
            and self.importance_score > 0.6
        )


@dataclass
class CacheEntry:
    """Structured cache entry with metadata"""

    key: str
    data: Any
    timestamp: datetime
    ttl_seconds: int
    access_count: int
    confidence_score: float
    user_feedback_score: Optional[float] = None
    learning_metadata: Dict[str, Any] = None
    memory_entry: Optional[HierarchicalMemoryEntry] = (
        None  # Enhanced with hierarchical memory
    )

    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        return datetime.utcnow() > self.timestamp + timedelta(seconds=self.ttl_seconds)

    def update_access(self):
        """Update access patterns"""
        self.access_count += 1
        # Adaptive TTL based on access patterns
        if self.access_count > 10:
            self.ttl_seconds = min(self.ttl_seconds * 1.2, 3600)  # Max 1 hour

        # Update hierarchical memory if present
        if self.memory_entry:
            self.memory_entry.access_frequency += 1
            self.memory_entry.last_accessed = datetime.utcnow()


@dataclass
class FlywheelMetrics:
    """Flywheel performance and learning metrics"""

    cache_hit_rate: float
    average_response_time_ms: float
    learning_accuracy: float
    user_satisfaction_score: float
    prediction_confidence: float
    total_interactions: int
    successful_predictions: int


class FlywheelCacheSystem:
    """
    Intelligent multi-layer caching system that learns from user interactions
    to optimize METIS cognitive platform performance
    """

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None

        # L1 Memory Cache
        self.l1_cache: Dict[str, CacheEntry] = {}
        self.l1_max_size = 1000

        # ENHANCED: Hierarchical Memory Management
        self.memory_tiers: Dict[MemoryTier, Dict[str, HierarchicalMemoryEntry]] = {
            MemoryTier.SHORT_TERM: {},  # Active working memory
            MemoryTier.MEDIUM_TERM: {},  # Consolidation buffer
            MemoryTier.LONG_TERM: {},  # Persistent knowledge
            MemoryTier.EPISODIC: {},  # Specific events/interactions
            MemoryTier.SEMANTIC: {},  # Generalized patterns
        }

        # Memory management configuration
        self.memory_config = {
            "short_term_capacity": 500,  # Max entries in short-term
            "medium_term_capacity": 200,  # Max entries in medium-term
            "consolidation_interval": 3600,  # 1 hour consolidation cycle
            "promotion_threshold": 0.7,  # Retention value for promotion
            "semantic_similarity_threshold": 0.8,  # For semantic clustering
        }

        # Last consolidation timestamp
        self.last_consolidation = datetime.utcnow()

        # Learning components
        self.interaction_patterns = defaultdict(list)
        self.user_decision_history = []
        self.consultant_performance_matrix = defaultdict(dict)
        self.query_similarity_graph = {}

        # Flywheel metrics
        self.metrics = FlywheelMetrics(
            cache_hit_rate=0.0,
            average_response_time_ms=0.0,
            learning_accuracy=0.0,
            user_satisfaction_score=0.0,
            prediction_confidence=0.0,
            total_interactions=0,
            successful_predictions=0,
        )

    async def initialize(self):
        """Initialize the flywheel cache system"""
        try:
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info("✅ Flywheel Cache System initialized with Redis")
        except Exception as e:
            logger.warning(f"⚠️ Redis unavailable, operating in L1-only mode: {e}")

    def _generate_cache_key(self, query: str, context: Dict[str, Any]) -> str:
        """Generate semantic cache key"""
        # Create semantic fingerprint
        content = {
            "query": query.lower().strip(),
            "context_keys": sorted(context.keys()),
            "context_hash": hashlib.md5(
                json.dumps(context, sort_keys=True).encode()
            ).hexdigest()[:8],
        }
        key_string = json.dumps(content, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]

    async def get(
        self, query: str, context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Intelligent cache retrieval with learning integration"""
        cache_key = self._generate_cache_key(query, context)
        start_time = time.time()

        # L1 Memory Cache Check
        if cache_key in self.l1_cache:
            entry = self.l1_cache[cache_key]
            if not entry.is_expired():
                entry.update_access()
                self._record_cache_hit("L1_MEMORY", time.time() - start_time)
                return entry.data
            else:
                del self.l1_cache[cache_key]

        # L2 Redis Cache Check
        if self.redis_client:
            try:
                cached_data = await self.redis_client.get(f"metis:cache:{cache_key}")
                if cached_data:
                    entry_data = pickle.loads(cached_data)
                    entry = CacheEntry(**entry_data)
                    if not entry.is_expired():
                        # Promote to L1
                        self.l1_cache[cache_key] = entry
                        entry.update_access()
                        self._record_cache_hit("L2_REDIS", time.time() - start_time)
                        return entry.data
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")

        # Cache miss - will be populated after computation
        self._record_cache_miss(time.time() - start_time)
        return None

    async def set(
        self,
        query: str,
        context: Dict[str, Any],
        data: Dict[str, Any],
        confidence_score: float,
        ttl_seconds: int = 300,
    ):
        """Intelligent cache storage with learning metadata"""
        cache_key = self._generate_cache_key(query, context)

        # Create cache entry with learning metadata
        learning_metadata = {
            "query_type": self._classify_query_type(query),
            "consultant_combination": self._extract_consultant_info(data),
            "complexity_score": self._estimate_complexity(query, context),
            "timestamp": datetime.utcnow().isoformat(),
        }

        entry = CacheEntry(
            key=cache_key,
            data=data,
            timestamp=datetime.utcnow(),
            ttl_seconds=ttl_seconds,
            access_count=1,
            confidence_score=confidence_score,
            learning_metadata=learning_metadata,
        )

        # Store in L1 cache
        if len(self.l1_cache) >= self.l1_max_size:
            self._evict_l1_cache()
        self.l1_cache[cache_key] = entry

        # Store in L2 Redis cache
        if self.redis_client:
            try:
                serialized_entry = pickle.dumps(asdict(entry))
                await self.redis_client.setex(
                    f"metis:cache:{cache_key}", ttl_seconds, serialized_entry
                )
            except Exception as e:
                logger.warning(f"Redis cache storage error: {e}")

    async def record_user_decision(
        self,
        query: str,
        context: Dict[str, Any],
        chosen_consultant: str,
        user_satisfaction: float,
    ):
        """Record user decision for learning loop"""
        cache_key = self._generate_cache_key(query, context)

        decision_record = {
            "cache_key": cache_key,
            "query": query,
            "context": context,
            "chosen_consultant": chosen_consultant,
            "user_satisfaction": user_satisfaction,
            "timestamp": datetime.utcnow().isoformat(),
            "query_type": self._classify_query_type(query),
            "complexity_score": self._estimate_complexity(query, context),
        }

        self.user_decision_history.append(decision_record)

        # Update consultant performance matrix
        query_type = decision_record["query_type"]
        if query_type not in self.consultant_performance_matrix[chosen_consultant]:
            self.consultant_performance_matrix[chosen_consultant][query_type] = []

        self.consultant_performance_matrix[chosen_consultant][query_type].append(
            {
                "satisfaction": user_satisfaction,
                "complexity": decision_record["complexity_score"],
                "timestamp": decision_record["timestamp"],
            }
        )

        # Update cache entry with user feedback if exists
        if cache_key in self.l1_cache:
            self.l1_cache[cache_key].user_feedback_score = user_satisfaction

        # Store learning data in Redis
        if self.redis_client:
            try:
                await self.redis_client.lpush(
                    "metis:learning:decisions", json.dumps(decision_record)
                )
                await self.redis_client.ltrim(
                    "metis:learning:decisions", 0, 10000
                )  # Keep last 10k
            except Exception as e:
                logger.warning(f"Learning data storage error: {e}")

    def predict_optimal_consultant(
        self, query: str, context: Dict[str, Any]
    ) -> List[Tuple[str, float]]:
        """Predict optimal consultant based on learning history"""
        query_type = self._classify_query_type(query)
        complexity = self._estimate_complexity(query, context)

        consultant_scores = []

        for consultant, performance_data in self.consultant_performance_matrix.items():
            if query_type in performance_data:
                # Calculate weighted average satisfaction based on recency and complexity similarity
                recent_performances = performance_data[query_type][
                    -10:
                ]  # Last 10 interactions

                if not recent_performances:
                    continue

                # Weight by recency and complexity similarity
                total_weight = 0
                weighted_satisfaction = 0

                for perf in recent_performances:
                    # Recency weight (exponential decay)
                    days_ago = (
                        datetime.utcnow() - datetime.fromisoformat(perf["timestamp"])
                    ).days
                    recency_weight = np.exp(-days_ago / 30)  # 30-day half-life

                    # Complexity similarity weight
                    complexity_diff = abs(perf["complexity"] - complexity)
                    complexity_weight = np.exp(-complexity_diff)

                    total_weight += recency_weight * complexity_weight
                    weighted_satisfaction += (
                        perf["satisfaction"] * recency_weight * complexity_weight
                    )

                if total_weight > 0:
                    avg_satisfaction = weighted_satisfaction / total_weight
                    consultant_scores.append((consultant, avg_satisfaction))

        # Sort by predicted satisfaction
        consultant_scores.sort(key=lambda x: x[1], reverse=True)
        return consultant_scores[:5]  # Top 5 predictions

    def get_flywheel_metrics(self) -> FlywheelMetrics:
        """Get current flywheel performance metrics"""
        # Update metrics based on current state
        if hasattr(self, "_cache_hits") and hasattr(self, "_cache_misses"):
            total_requests = self._cache_hits + self._cache_misses
            self.metrics.cache_hit_rate = (
                (self._cache_hits / total_requests) if total_requests > 0 else 0.0
            )

        if self.user_decision_history:
            recent_decisions = self.user_decision_history[-100:]  # Last 100 decisions
            self.metrics.user_satisfaction_score = np.mean(
                [d["user_satisfaction"] for d in recent_decisions]
            )
            self.metrics.total_interactions = len(self.user_decision_history)

        return self.metrics

    def _record_cache_hit(self, layer: str, response_time: float):
        """Record cache hit metrics"""
        if not hasattr(self, "_cache_hits"):
            self._cache_hits = 0
        self._cache_hits += 1

        if not hasattr(self, "_response_times"):
            self._response_times = []
        self._response_times.append(response_time * 1000)  # Convert to ms

        logger.debug(f"Cache HIT [{layer}]: {response_time*1000:.1f}ms")

    def _record_cache_miss(self, response_time: float):
        """Record cache miss metrics"""
        if not hasattr(self, "_cache_misses"):
            self._cache_misses = 0
        self._cache_misses += 1

        logger.debug(f"Cache MISS: {response_time*1000:.1f}ms")

    def _evict_l1_cache(self):
        """Evict least recently used entries from L1 cache"""
        # Sort by access_count and timestamp
        entries = list(self.l1_cache.items())
        entries.sort(key=lambda x: (x[1].access_count, x[1].timestamp))

        # Remove bottom 20%
        evict_count = len(entries) // 5
        for i in range(evict_count):
            key, _ = entries[i]
            del self.l1_cache[key]

    def _classify_query_type(self, query: str) -> str:
        """Classify query type for learning purposes"""
        query_lower = query.lower()

        if any(
            word in query_lower
            for word in ["strategy", "strategic", "market", "competition"]
        ):
            return "strategic"
        elif any(
            word in query_lower
            for word in ["technical", "implementation", "architecture", "system"]
        ):
            return "technical"
        elif any(
            word in query_lower
            for word in ["financial", "budget", "cost", "revenue", "profit"]
        ):
            return "financial"
        elif any(
            word in query_lower
            for word in ["process", "operational", "efficiency", "workflow"]
        ):
            return "operational"
        elif any(
            word in query_lower
            for word in ["innovation", "creative", "new", "breakthrough"]
        ):
            return "innovation"
        else:
            return "general"

    def _estimate_complexity(self, query: str, context: Dict[str, Any]) -> float:
        """Estimate query complexity score (0.0 to 1.0)"""
        complexity = 0.0

        # Length factor
        complexity += min(len(query) / 1000, 0.3)

        # Context richness
        complexity += min(len(context) / 20, 0.3)

        # Keyword complexity indicators
        complex_words = [
            "integrate",
            "synthesize",
            "optimize",
            "transform",
            "strategic",
            "comprehensive",
        ]
        complexity += min(
            sum(1 for word in complex_words if word in query.lower())
            / len(complex_words),
            0.4,
        )

        return min(complexity, 1.0)

    def _extract_consultant_info(self, data: Dict[str, Any]) -> List[str]:
        """Extract consultant information from analysis data"""
        consultants = []

        if "cognitive_analysis" in data:
            if "selected_models" in data["cognitive_analysis"]:
                for model in data["cognitive_analysis"]["selected_models"]:
                    if isinstance(model, dict) and "name" in model:
                        consultants.append(model["name"])

        return consultants

    # ENHANCED: Hierarchical Memory Management Methods

    async def store_in_hierarchical_memory(
        self,
        key: str,
        data: Any,
        importance_score: float,
        memory_tier: MemoryTier = MemoryTier.SHORT_TERM,
    ):
        """Store data in hierarchical memory with proper tier management"""
        # Create memory entry
        memory_entry = HierarchicalMemoryEntry(
            key=key,
            data=data,
            memory_tier=memory_tier,
            importance_score=importance_score,
        )

        # Store in appropriate tier
        self.memory_tiers[memory_tier][key] = memory_entry

        # Check tier capacity and consolidate if needed
        await self._check_tier_capacity(memory_tier)

        # Trigger consolidation if interval has passed
        if self._should_consolidate():
            await self.perform_memory_consolidation()

        logger.debug(
            f"Stored {key} in {memory_tier.value} memory (importance: {importance_score:.2f})"
        )

    async def retrieve_from_hierarchical_memory(
        self, key: str
    ) -> Optional[HierarchicalMemoryEntry]:
        """Retrieve data from hierarchical memory across all tiers"""
        # Search through tiers in order of priority
        search_order = [
            MemoryTier.SHORT_TERM,
            MemoryTier.MEDIUM_TERM,
            MemoryTier.LONG_TERM,
            MemoryTier.SEMANTIC,
            MemoryTier.EPISODIC,
        ]

        for tier in search_order:
            if key in self.memory_tiers[tier]:
                entry = self.memory_tiers[tier][key]
                entry.access_frequency += 1
                entry.last_accessed = datetime.utcnow()
                return entry

        return None

    async def perform_memory_consolidation(self):
        """Perform memory consolidation across hierarchical tiers"""
        logger.info("🧠 Starting hierarchical memory consolidation")

        # Consolidate short-term to medium-term
        await self._consolidate_short_to_medium_term()

        # Consolidate medium-term to long-term
        await self._consolidate_medium_to_long_term()

        # Extract semantic patterns
        await self._extract_semantic_patterns()

        # Update consolidation timestamp
        self.last_consolidation = datetime.utcnow()

        logger.info("✅ Memory consolidation complete")

    async def _consolidate_short_to_medium_term(self):
        """Move valuable short-term memories to medium-term storage"""
        short_term_entries = list(self.memory_tiers[MemoryTier.SHORT_TERM].items())

        for key, entry in short_term_entries:
            retention_value = entry.calculate_retention_value()

            # Promote to medium-term if valuable enough
            if retention_value >= 0.5 and entry.access_frequency >= 2:
                # Move to medium-term
                entry.memory_tier = MemoryTier.MEDIUM_TERM
                entry.consolidation_count += 1

                self.memory_tiers[MemoryTier.MEDIUM_TERM][key] = entry
                del self.memory_tiers[MemoryTier.SHORT_TERM][key]

                logger.debug(
                    f"Promoted {key} to medium-term (retention: {retention_value:.2f})"
                )

            # Remove low-value entries
            elif retention_value < 0.2:
                del self.memory_tiers[MemoryTier.SHORT_TERM][key]
                logger.debug(f"Discarded low-value entry {key}")

    async def _consolidate_medium_to_long_term(self):
        """Move important medium-term memories to long-term storage"""
        medium_term_entries = list(self.memory_tiers[MemoryTier.MEDIUM_TERM].items())

        for key, entry in medium_term_entries:
            if entry.should_promote_to_long_term():
                # Move to long-term
                entry.memory_tier = MemoryTier.LONG_TERM
                entry.consolidation_count += 1

                self.memory_tiers[MemoryTier.LONG_TERM][key] = entry
                del self.memory_tiers[MemoryTier.MEDIUM_TERM][key]

                logger.info(f"🌟 Promoted {key} to long-term memory")

    async def _extract_semantic_patterns(self):
        """Extract semantic patterns from episodic memories"""
        # Analyze patterns across all memories
        all_entries = []
        for tier_entries in self.memory_tiers.values():
            all_entries.extend(tier_entries.values())

        # Group by similarity (simplified semantic clustering)
        semantic_clusters = self._cluster_by_semantic_similarity(all_entries)

        for cluster_key, cluster_entries in semantic_clusters.items():
            if len(cluster_entries) >= 3:  # Pattern emerges from multiple instances
                # Create semantic memory entry
                semantic_key = f"pattern_{cluster_key}"
                pattern_data = self._synthesize_pattern_from_cluster(cluster_entries)

                semantic_entry = HierarchicalMemoryEntry(
                    key=semantic_key,
                    data=pattern_data,
                    memory_tier=MemoryTier.SEMANTIC,
                    importance_score=min(1.0, len(cluster_entries) * 0.2),
                    access_frequency=len(cluster_entries),
                    consolidation_count=1,
                )

                self.memory_tiers[MemoryTier.SEMANTIC][semantic_key] = semantic_entry
                logger.info(f"🧩 Extracted semantic pattern: {cluster_key}")

    def _cluster_by_semantic_similarity(
        self, entries: List[HierarchicalMemoryEntry]
    ) -> Dict[str, List[HierarchicalMemoryEntry]]:
        """Group entries by semantic similarity (simplified)"""
        clusters = defaultdict(list)

        for entry in entries:
            # Simple clustering by query type and consultant patterns
            cluster_key = self._extract_cluster_key(entry)
            clusters[cluster_key].append(entry)

        return dict(clusters)

    def _extract_cluster_key(self, entry: HierarchicalMemoryEntry) -> str:
        """Extract semantic cluster key from memory entry"""
        if isinstance(entry.data, dict):
            # Extract query characteristics
            query_type = "general"
            if "cognitive_analysis" in entry.data:
                if "selected_models" in entry.data["cognitive_analysis"]:
                    models = entry.data["cognitive_analysis"]["selected_models"]
                    if models and isinstance(models[0], dict) and "name" in models[0]:
                        query_type = models[0]["name"]

            return f"{query_type}_pattern"

        return "generic_pattern"

    def _synthesize_pattern_from_cluster(
        self, cluster_entries: List[HierarchicalMemoryEntry]
    ) -> Dict[str, Any]:
        """Synthesize a semantic pattern from cluster of entries"""
        pattern = {
            "type": "semantic_pattern",
            "frequency": len(cluster_entries),
            "avg_importance": np.mean(
                [entry.importance_score for entry in cluster_entries]
            ),
            "total_accesses": sum(entry.access_frequency for entry in cluster_entries),
            "representative_keys": [entry.key for entry in cluster_entries[:3]],
            "pattern_confidence": min(1.0, len(cluster_entries) / 10),
            "created_at": datetime.utcnow().isoformat(),
        }

        return pattern

    async def _check_tier_capacity(self, tier: MemoryTier):
        """Check and manage tier capacity limits"""
        tier_entries = self.memory_tiers[tier]

        if tier == MemoryTier.SHORT_TERM:
            max_capacity = self.memory_config["short_term_capacity"]
        elif tier == MemoryTier.MEDIUM_TERM:
            max_capacity = self.memory_config["medium_term_capacity"]
        else:
            return  # No limits on long-term tiers

        if len(tier_entries) > max_capacity:
            # Remove least valuable entries
            entries_by_value = sorted(
                tier_entries.items(), key=lambda x: x[1].calculate_retention_value()
            )

            excess_count = len(tier_entries) - max_capacity
            for i in range(excess_count):
                key_to_remove = entries_by_value[i][0]
                del tier_entries[key_to_remove]
                logger.debug(
                    f"Removed {key_to_remove} from {tier.value} due to capacity"
                )

    def _should_consolidate(self) -> bool:
        """Check if memory consolidation should be performed"""
        time_since_last = (datetime.utcnow() - self.last_consolidation).total_seconds()
        return time_since_last >= self.memory_config["consolidation_interval"]

    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get hierarchical memory statistics"""
        stats = {}

        for tier, entries in self.memory_tiers.items():
            tier_stats = {
                "count": len(entries),
                "avg_importance": (
                    np.mean([e.importance_score for e in entries.values()])
                    if entries
                    else 0.0
                ),
                "total_accesses": sum(e.access_frequency for e in entries.values()),
                "avg_retention_value": (
                    np.mean([e.calculate_retention_value() for e in entries.values()])
                    if entries
                    else 0.0
                ),
            }
            stats[tier.value] = tier_stats

        stats["last_consolidation"] = self.last_consolidation.isoformat()
        stats["time_until_next_consolidation"] = max(
            0,
            self.memory_config["consolidation_interval"]
            - (datetime.utcnow() - self.last_consolidation).total_seconds(),
        )

        return stats

    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status of the Flywheel Cache System"""
        status = "healthy"
        details = []
        errors = []

        try:
            # Check L1 Memory Cache
            l1_status = "healthy" if self.l1_cache is not None else "unhealthy"
            if l1_status == "unhealthy":
                errors.append("L1 memory cache not initialized")
                status = "degraded"
            else:
                details.append(
                    f"L1 cache: {len(self.l1_cache)}/{self.l1_max_size} entries"
                )

            # Check Redis connection
            redis_status = "unavailable"
            if self.redis_client:
                try:
                    await self.redis_client.ping()
                    redis_status = "healthy"
                    details.append("Redis connection: operational")
                except Exception as e:
                    redis_status = "failed"
                    errors.append(f"Redis connection failed: {str(e)}")
                    if status == "healthy":
                        status = "degraded"
            else:
                details.append("Redis: operating in L1-only mode")

            # Check memory tiers
            memory_tier_health = True
            for tier, entries in self.memory_tiers.items():
                if entries is None:
                    memory_tier_health = False
                    errors.append(f"Memory tier {tier.value} not initialized")

            if not memory_tier_health and status == "healthy":
                status = "degraded"
            else:
                total_memory_entries = sum(
                    len(entries) for entries in self.memory_tiers.values()
                )
                details.append(
                    f"Hierarchical memory: {total_memory_entries} total entries"
                )

            # Check performance metrics
            cache_metrics = self.get_flywheel_metrics()
            if cache_metrics:
                details.append(f"Cache hit rate: {cache_metrics.cache_hit_rate:.2%}")
                details.append(
                    f"User satisfaction: {cache_metrics.user_satisfaction_score:.2f}"
                )
                details.append(
                    f"Total interactions: {cache_metrics.total_interactions}"
                )

            # Overall health assessment
            if len(errors) == 0:
                status = "healthy"
            elif len(errors) >= 3:
                status = "unhealthy"

        except Exception as e:
            status = "error"
            errors.append(f"Health check failed: {str(e)}")

        return {
            "status": status,
            "details": "; ".join(details) if details else "Cache system operational",
            "errors": errors,
            "timestamp": datetime.utcnow().isoformat(),
            "component": "FlywheelCacheSystem",
            "redis_available": redis_status == "healthy",
            "l1_cache_size": len(self.l1_cache) if self.l1_cache else 0,
            "memory_tiers_count": len(
                [t for t in self.memory_tiers.values() if t is not None]
            ),
        }


# Global flywheel cache instance
_flywheel_cache: Optional[FlywheelCacheSystem] = None


async def get_flywheel_cache() -> FlywheelCacheSystem:
    """Get or create global flywheel cache instance"""
    global _flywheel_cache

    if _flywheel_cache is None:
        _flywheel_cache = FlywheelCacheSystem()
        await _flywheel_cache.initialize()

    return _flywheel_cache
