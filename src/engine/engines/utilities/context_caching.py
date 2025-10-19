"""
METIS Phase 2.1.3: Phase-Based Context Caching System
Research Foundation: LangChain semantic caching + phase optimization

Intelligent caching with phase-specific optimization and semantic similarity
for cognitive workflow acceleration and cross-session learning.

Performance Targets:
- Cache hit rate: >85% for phase-specific contexts
- Response speed improvement: 3x through intelligent caching
- Context relevance: >88% semantic similarity threshold
- Cross-session learning: Continuous improvement of cache effectiveness
"""

import logging
import json
import hashlib
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict

# Import METIS core components
from src.engine.models.data_contracts import (
    EngagementContext,
    CognitiveState,
)


class CacheHitType(str, Enum):
    """Types of cache hits for analytics"""

    EXACT_MATCH = "exact_match"
    SEMANTIC_MATCH = "semantic_match"
    PHASE_AFFINITY = "phase_affinity"
    MISS = "miss"


class CacheEntryType(str, Enum):
    """Types of cached entries"""

    PROBLEM_STRUCTURE = "problem_structure"
    HYPOTHESIS_SET = "hypothesis_set"
    ANALYSIS_RESULT = "analysis_result"
    SYNTHESIS_OUTPUT = "synthesis_output"
    MENTAL_MODEL_SELECTION = "mental_model_selection"
    NWAY_INTERACTION = "nway_interaction"


@dataclass
class CacheKey:
    """Structured cache key with semantic components"""

    phase: str
    problem_type: str
    mental_models: List[str]
    context_signature: str
    complexity_level: str
    user_profile: str = "default"


@dataclass
class CacheEntry:
    """Individual cache entry with metadata"""

    key: CacheKey
    content: Any
    entry_type: CacheEntryType
    creation_timestamp: datetime
    last_accessed: datetime
    access_count: int
    semantic_embedding: Optional[np.ndarray]
    confidence_score: float
    phase_affinity_score: float
    success_rate: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheHitResult:
    """Result of cache lookup with performance metrics"""

    hit: bool
    hit_type: CacheHitType
    entry: Optional[CacheEntry]
    similarity_score: float
    retrieval_time_ms: float
    performance_boost: float = 0.0
    confidence: float = 0.0


@dataclass
class CachedContextResult:
    """Result container for cached context retrieval"""

    context: Any
    cache_hit: bool
    phase_optimized: bool
    confidence: float
    performance_boost: float
    hit_type: CacheHitType = CacheHitType.MISS
    retrieval_time_ms: float = 0.0


class SemanticSimilarityEngine:
    """
    Semantic similarity calculation for context matching
    Uses embeddings and contextual features for similarity scoring
    """

    def __init__(self, similarity_threshold: float = 0.88):
        self.similarity_threshold = similarity_threshold
        self.logger = logging.getLogger(__name__)
        self._embedding_cache = {}

    async def calculate_similarity(
        self,
        query_embedding: np.ndarray,
        cached_embedding: np.ndarray,
        context_factors: Dict[str, Any] = None,
    ) -> float:
        """
        Calculate semantic similarity between query and cached content
        Target: >88% similarity threshold for relevant matches
        """

        try:
            # Base cosine similarity
            cosine_sim = np.dot(query_embedding, cached_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(cached_embedding)
            )

            # Adjust for context factors if provided
            if context_factors:
                context_boost = self._calculate_context_boost(context_factors)
                adjusted_similarity = min(cosine_sim + context_boost, 1.0)
            else:
                adjusted_similarity = cosine_sim

            return float(adjusted_similarity)

        except Exception as e:
            self.logger.error(f"Similarity calculation failed: {e}")
            return 0.0

    def _calculate_context_boost(self, context_factors: Dict[str, Any]) -> float:
        """Calculate context-based similarity boost"""

        boost = 0.0

        # Phase alignment boost
        if context_factors.get("phase_match", False):
            boost += 0.05

        # Problem type alignment boost
        if context_factors.get("problem_type_match", False):
            boost += 0.03

        # Mental model overlap boost
        model_overlap = context_factors.get("model_overlap", 0.0)
        boost += model_overlap * 0.02

        # Complexity level alignment
        if context_factors.get("complexity_match", False):
            boost += 0.02

        return min(boost, 0.12)  # Cap boost at 12%

    def generate_embedding(
        self, content: Any, context: Dict[str, Any] = None
    ) -> np.ndarray:
        """
        Generate semantic embedding for content
        In production, this would use actual embedding models (e.g., OpenAI, Sentence-BERT)
        """

        # For demonstration, create synthetic embeddings based on content characteristics
        content_str = str(content)
        content_hash = hashlib.md5(content_str.encode()).hexdigest()

        # Use hash to create deterministic but varied embeddings
        np.random.seed(int(content_hash[:8], 16))
        base_embedding = np.random.normal(0, 1, 384)  # Standard embedding dimension

        # Adjust based on context if provided
        if context:
            phase_factor = hash(context.get("phase", "")) % 100 / 100.0
            problem_factor = hash(context.get("problem_type", "")) % 100 / 100.0

            base_embedding[0] += phase_factor * 0.1
            base_embedding[1] += problem_factor * 0.1

        # Normalize
        return base_embedding / np.linalg.norm(base_embedding)


class SemanticCache:
    """
    Semantic cache with similarity-based retrieval
    Implements LRU eviction and performance tracking
    """

    def __init__(
        self, threshold: float = 0.88, max_entries: int = 1000, ttl_hours: int = 24
    ):
        self.threshold = threshold
        self.max_entries = max_entries
        self.ttl = timedelta(hours=ttl_hours)
        self.entries: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []
        self.similarity_engine = SemanticSimilarityEngine(threshold)
        self.logger = logging.getLogger(__name__)
        self.performance_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_retrieval_time": 0.0,
        }

    async def get_similar_context(
        self,
        query_context: Any,
        query_key: CacheKey,
        similarity_threshold: Optional[float] = None,
    ) -> Optional[CacheHitResult]:
        """
        Retrieve similar context from cache based on semantic similarity
        Target: 85% hit rate for relevant contexts
        """

        start_time = time.time()
        threshold = similarity_threshold or self.threshold

        # Generate query embedding
        query_embedding = self.similarity_engine.generate_embedding(
            query_context,
            context={
                "phase": query_key.phase,
                "problem_type": query_key.problem_type,
                "mental_models": query_key.mental_models,
            },
        )

        best_match = None
        best_similarity = 0.0
        best_hit_type = CacheHitType.MISS

        # Check for exact matches first
        exact_key = self._generate_cache_key_string(query_key)
        if exact_key in self.entries:
            entry = self.entries[exact_key]
            if self._is_entry_valid(entry):
                best_match = entry
                best_similarity = 1.0
                best_hit_type = CacheHitType.EXACT_MATCH

        # If no exact match, search for semantic matches
        if not best_match:
            for entry_key, entry in self.entries.items():
                if not self._is_entry_valid(entry):
                    continue

                if entry.semantic_embedding is not None:
                    # Calculate context factors for similarity boost
                    context_factors = {
                        "phase_match": entry.key.phase == query_key.phase,
                        "problem_type_match": entry.key.problem_type
                        == query_key.problem_type,
                        "model_overlap": len(
                            set(entry.key.mental_models) & set(query_key.mental_models)
                        )
                        / max(
                            len(
                                set(entry.key.mental_models)
                                | set(query_key.mental_models)
                            ),
                            1,
                        ),
                        "complexity_match": entry.key.complexity_level
                        == query_key.complexity_level,
                    }

                    similarity = await self.similarity_engine.calculate_similarity(
                        query_embedding, entry.semantic_embedding, context_factors
                    )

                    if similarity > best_similarity and similarity >= threshold:
                        best_match = entry
                        best_similarity = similarity
                        best_hit_type = (
                            CacheHitType.PHASE_AFFINITY
                            if context_factors["phase_match"]
                            else CacheHitType.SEMANTIC_MATCH
                        )

        # Update access statistics
        retrieval_time = (time.time() - start_time) * 1000  # Convert to ms
        self.performance_stats["total_retrieval_time"] += retrieval_time

        if best_match:
            # Update access tracking
            self._update_access_tracking(best_match)
            self.performance_stats["hits"] += 1

            # Calculate performance boost
            performance_boost = self._calculate_performance_boost(
                best_match, best_similarity
            )

            result = CacheHitResult(
                hit=True,
                hit_type=best_hit_type,
                entry=best_match,
                similarity_score=best_similarity,
                retrieval_time_ms=retrieval_time,
                performance_boost=performance_boost,
                confidence=best_similarity,
            )

            self.logger.info(
                f"Cache HIT: {best_hit_type.value}, similarity={best_similarity:.3f}, "
                f"boost={performance_boost:.1f}x, time={retrieval_time:.1f}ms"
            )

        else:
            self.performance_stats["misses"] += 1
            result = CacheHitResult(
                hit=False,
                hit_type=CacheHitType.MISS,
                entry=None,
                similarity_score=0.0,
                retrieval_time_ms=retrieval_time,
            )

            self.logger.debug(f"Cache MISS: time={retrieval_time:.1f}ms")

        return result

    async def store_context(
        self,
        key: CacheKey,
        content: Any,
        entry_type: CacheEntryType,
        confidence_score: float = 0.85,
    ) -> bool:
        """
        Store content in semantic cache with metadata
        """

        try:
            # Generate semantic embedding
            embedding = self.similarity_engine.generate_embedding(
                content,
                context={
                    "phase": key.phase,
                    "problem_type": key.problem_type,
                    "mental_models": key.mental_models,
                },
            )

            # Calculate phase affinity score
            phase_affinity = self._calculate_phase_affinity(key, content)

            # Create cache entry
            entry = CacheEntry(
                key=key,
                content=content,
                entry_type=entry_type,
                creation_timestamp=datetime.now(),
                last_accessed=datetime.now(),
                access_count=1,
                semantic_embedding=embedding,
                confidence_score=confidence_score,
                phase_affinity_score=phase_affinity,
            )

            # Store entry
            cache_key = self._generate_cache_key_string(key)
            self.entries[cache_key] = entry
            self.access_order.append(cache_key)

            # Perform LRU eviction if needed
            await self._perform_lru_eviction()

            self.logger.debug(
                f"Cache STORE: {entry_type.value}, affinity={phase_affinity:.3f}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Cache storage failed: {e}")
            return False

    def _is_entry_valid(self, entry: CacheEntry) -> bool:
        """Check if cache entry is still valid (not expired)"""
        return datetime.now() - entry.creation_timestamp < self.ttl

    def _update_access_tracking(self, entry: CacheEntry) -> None:
        """Update access tracking for LRU management"""
        entry.last_accessed = datetime.now()
        entry.access_count += 1

        # Move to end of access order
        cache_key = self._generate_cache_key_string(entry.key)
        if cache_key in self.access_order:
            self.access_order.remove(cache_key)
        self.access_order.append(cache_key)

    def _calculate_performance_boost(
        self, entry: CacheEntry, similarity: float
    ) -> float:
        """Calculate performance boost from cache hit"""

        # Base boost from avoiding computation
        base_boost = 2.0

        # Similarity-based boost (higher similarity = more reliable boost)
        similarity_factor = similarity * 1.5

        # Phase affinity boost (same phase operations are more beneficial)
        affinity_boost = entry.phase_affinity_score * 0.5

        # Historical success rate boost
        success_boost = entry.success_rate * 0.3

        total_boost = base_boost + similarity_factor + affinity_boost + success_boost

        return min(total_boost, 5.0)  # Cap at 5x boost

    def _calculate_phase_affinity(self, key: CacheKey, content: Any) -> float:
        """Calculate how well content aligns with its phase"""

        # Phase-specific characteristics
        phase_characteristics = {
            "problem_structuring": ["structure", "decompose", "mece", "framework"],
            "hypothesis_generation": ["hypothesis", "theory", "assumption", "predict"],
            "analysis_execution": ["analyze", "evaluate", "test", "validate"],
            "synthesis_delivery": ["synthesize", "conclude", "recommend", "deliver"],
        }

        phase_keywords = phase_characteristics.get(key.phase, [])
        content_str = str(content).lower()

        # Count keyword matches
        matches = sum(1 for keyword in phase_keywords if keyword in content_str)
        affinity_score = min(
            matches / len(phase_keywords) if phase_keywords else 0.5, 1.0
        )

        return affinity_score

    def _generate_cache_key_string(self, key: CacheKey) -> str:
        """Generate string representation of cache key"""
        key_data = {
            "phase": key.phase,
            "problem_type": key.problem_type,
            "mental_models": sorted(key.mental_models),
            "context_signature": key.context_signature,
            "complexity_level": key.complexity_level,
            "user_profile": key.user_profile,
        }

        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    async def _perform_lru_eviction(self) -> None:
        """Perform LRU eviction when cache is full"""

        while len(self.entries) > self.max_entries:
            if not self.access_order:
                break

            # Remove least recently used entry
            lru_key = self.access_order.pop(0)
            if lru_key in self.entries:
                del self.entries[lru_key]
                self.performance_stats["evictions"] += 1

    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics"""

        total_requests = (
            self.performance_stats["hits"] + self.performance_stats["misses"]
        )
        hit_rate = (
            self.performance_stats["hits"] / total_requests
            if total_requests > 0
            else 0.0
        )
        avg_retrieval_time = (
            self.performance_stats["total_retrieval_time"] / total_requests
            if total_requests > 0
            else 0.0
        )

        return {
            "hit_rate": hit_rate,
            "total_entries": len(self.entries),
            "total_requests": total_requests,
            "hits": self.performance_stats["hits"],
            "misses": self.performance_stats["misses"],
            "evictions": self.performance_stats["evictions"],
            "average_retrieval_time_ms": avg_retrieval_time,
            "cache_efficiency": hit_rate
            * (3.0 if hit_rate > 0.85 else 1.0),  # 3x target
        }


class MetisPhaseBasedContextCache:
    """
    Intelligent caching with phase-specific optimization
    Implements semantic similarity with phase affinity

    Performance Targets:
    - Cache hit rate: >85% for phase-specific contexts
    - Response speed improvement: 3x through intelligent caching
    - Context relevance: >88% semantic similarity threshold
    - Cross-session learning: Continuous improvement of cache effectiveness
    """

    def __init__(self):
        self.phase_cache = {
            "problem_structuring": SemanticCache(threshold=0.88, max_entries=250),
            "hypothesis_generation": SemanticCache(threshold=0.85, max_entries=250),
            "analysis_execution": SemanticCache(threshold=0.90, max_entries=300),
            "synthesis_delivery": SemanticCache(threshold=0.87, max_entries=200),
        }

        self.global_cache = SemanticCache(threshold=0.88, max_entries=500)
        self.problem_type_classifier = ProblemTypeClassifier()
        self.logger = logging.getLogger(__name__)
        self.session_analytics = defaultdict(list)

    async def get_phase_optimized_context(
        self,
        current_phase: str,
        problem_context: str,
        mental_models: List[str],
        user_profile: str = "default",
    ) -> CachedContextResult:
        """
        Retrieve context optimized for specific cognitive phases
        Target: 85% cache hit rate, 3x response speed improvement
        """

        start_time = time.time()

        # Generate phase-specific context signature
        context_signature = await self._generate_phase_context_signature(
            phase=current_phase,
            problem_type=self.problem_type_classifier.classify_problem_type(
                problem_context
            ),
            models=mental_models,
            user_profile=user_profile,
        )

        # Create cache key
        cache_key = CacheKey(
            phase=current_phase,
            problem_type=context_signature["problem_type"],
            mental_models=mental_models,
            context_signature=context_signature["signature"],
            complexity_level=context_signature["complexity_level"],
            user_profile=user_profile,
        )

        # Check phase-specific cache first
        phase_cache = self.phase_cache.get(current_phase)
        if phase_cache:
            cached_result = await phase_cache.get_similar_context(
                problem_context, cache_key, similarity_threshold=0.88
            )

            if cached_result and cached_result.hit and cached_result.confidence > 0.88:
                retrieval_time = (time.time() - start_time) * 1000

                result = CachedContextResult(
                    context=cached_result.entry.content,
                    cache_hit=True,
                    phase_optimized=True,
                    confidence=cached_result.confidence,
                    performance_boost=cached_result.performance_boost,
                    hit_type=cached_result.hit_type,
                    retrieval_time_ms=retrieval_time,
                )

                self.logger.info(
                    f"Phase cache HIT for {current_phase}: confidence={cached_result.confidence:.3f}, "
                    f"boost={cached_result.performance_boost:.1f}x"
                )

                return result

        # Fallback to global cache
        global_result = await self.global_cache.get_similar_context(
            problem_context, cache_key, similarity_threshold=0.85
        )

        if global_result and global_result.hit:
            retrieval_time = (time.time() - start_time) * 1000

            result = CachedContextResult(
                context=global_result.entry.content,
                cache_hit=True,
                phase_optimized=False,
                confidence=global_result.confidence,
                performance_boost=global_result.performance_boost
                * 0.8,  # Reduced for global cache
                hit_type=global_result.hit_type,
                retrieval_time_ms=retrieval_time,
            )

            self.logger.info(
                f"Global cache HIT: confidence={global_result.confidence:.3f}"
            )
            return result

        # Cache miss
        retrieval_time = (time.time() - start_time) * 1000
        self.logger.debug(
            f"Cache MISS for {current_phase}: time={retrieval_time:.1f}ms"
        )

        return CachedContextResult(
            context=None,
            cache_hit=False,
            phase_optimized=False,
            confidence=0.0,
            performance_boost=0.0,
            retrieval_time_ms=retrieval_time,
        )

    async def store_phase_context(
        self,
        phase: str,
        problem_context: str,
        mental_models: List[str],
        content: Any,
        entry_type: CacheEntryType,
        confidence_score: float = 0.85,
        user_profile: str = "default",
    ) -> bool:
        """
        Store context in phase-optimized cache
        """

        try:
            # Generate context signature
            context_signature = await self._generate_phase_context_signature(
                phase=phase,
                problem_type=self.problem_type_classifier.classify_problem_type(
                    problem_context
                ),
                models=mental_models,
                user_profile=user_profile,
            )

            # Create cache key
            cache_key = CacheKey(
                phase=phase,
                problem_type=context_signature["problem_type"],
                mental_models=mental_models,
                context_signature=context_signature["signature"],
                complexity_level=context_signature["complexity_level"],
                user_profile=user_profile,
            )

            # Store in phase-specific cache
            phase_cache = self.phase_cache.get(phase)
            if phase_cache:
                phase_success = await phase_cache.store_context(
                    cache_key, content, entry_type, confidence_score
                )
            else:
                phase_success = False

            # Also store in global cache for cross-phase lookup
            global_success = await self.global_cache.store_context(
                cache_key, content, entry_type, confidence_score
            )

            # Update session analytics
            self.session_analytics[phase].append(
                {
                    "timestamp": datetime.now(),
                    "stored": phase_success or global_success,
                    "entry_type": entry_type.value,
                    "confidence": confidence_score,
                }
            )

            return phase_success or global_success

        except Exception as e:
            self.logger.error(f"Phase context storage failed: {e}")
            return False

    async def _generate_phase_context_signature(
        self,
        phase: str,
        problem_type: str,
        models: List[str],
        user_profile: str = "default",
    ) -> Dict[str, str]:
        """
        Generate phase-specific context signature
        """

        # Calculate complexity level
        complexity_level = self._assess_complexity_level(models, problem_type)

        # Create signature components
        signature_data = {
            "phase": phase,
            "problem_type": problem_type,
            "models": sorted(models),
            "complexity": complexity_level,
            "user_profile": user_profile,
            "timestamp_bucket": datetime.now().strftime(
                "%Y-%m-%d-%H"
            ),  # Hour-level bucketing
        }

        # Generate signature hash
        signature_str = json.dumps(signature_data, sort_keys=True)
        signature_hash = hashlib.md5(signature_str.encode()).hexdigest()

        return {
            "signature": signature_hash,
            "problem_type": problem_type,
            "complexity_level": complexity_level,
        }

    def _assess_complexity_level(self, models: List[str], problem_type: str) -> str:
        """Assess complexity level for caching optimization"""

        # Model count factor
        model_complexity = len(models)

        # Problem type complexity
        complex_problem_types = [
            "strategic_planning",
            "investment_analysis",
            "system_optimization",
        ]
        problem_complexity = 2 if problem_type in complex_problem_types else 1

        # Calculate total complexity
        total_complexity = model_complexity + problem_complexity

        if total_complexity <= 3:
            return "low"
        elif total_complexity <= 6:
            return "medium"
        else:
            return "high"

    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive caching statistics across all phases"""

        overall_stats = {
            "timestamp": datetime.now().isoformat(),
            "phase_statistics": {},
            "global_statistics": {},
            "cross_session_learning": {},
        }

        # Collect phase-specific statistics
        total_hit_rate = 0
        total_requests = 0

        for phase, cache in self.phase_cache.items():
            phase_stats = cache.get_cache_statistics()
            overall_stats["phase_statistics"][phase] = phase_stats

            total_hit_rate += phase_stats["hit_rate"] * phase_stats["total_requests"]
            total_requests += phase_stats["total_requests"]

        # Global cache statistics
        overall_stats["global_statistics"] = self.global_cache.get_cache_statistics()

        # Overall metrics
        overall_hit_rate = total_hit_rate / total_requests if total_requests > 0 else 0
        overall_stats["overall_hit_rate"] = overall_hit_rate
        overall_stats["target_achievement"] = {
            "hit_rate_target": overall_hit_rate >= 0.85,
            "speed_improvement_target": overall_hit_rate * 3.0,  # Approximation
            "target_met": overall_hit_rate >= 0.85,
        }

        # Cross-session learning metrics
        learning_stats = self._analyze_cross_session_learning()
        overall_stats["cross_session_learning"] = learning_stats

        return overall_stats

    def _analyze_cross_session_learning(self) -> Dict[str, Any]:
        """Analyze cross-session learning effectiveness"""

        learning_stats = {
            "session_count": len(self.session_analytics),
            "improvement_trends": {},
            "learning_effectiveness": 0.0,
        }

        # Analyze improvement trends by phase
        for phase, sessions in self.session_analytics.items():
            if len(sessions) >= 2:
                recent_success_rate = sum(
                    1 for s in sessions[-10:] if s["stored"]
                ) / min(len(sessions), 10)
                early_success_rate = sum(1 for s in sessions[:10] if s["stored"]) / min(
                    len(sessions), 10
                )

                improvement = recent_success_rate - early_success_rate
                learning_stats["improvement_trends"][phase] = {
                    "improvement": improvement,
                    "recent_success_rate": recent_success_rate,
                    "early_success_rate": early_success_rate,
                }

        # Calculate overall learning effectiveness
        if learning_stats["improvement_trends"]:
            avg_improvement = sum(
                t["improvement"] for t in learning_stats["improvement_trends"].values()
            ) / len(learning_stats["improvement_trends"])
            learning_stats["learning_effectiveness"] = max(0, avg_improvement)

        return learning_stats

    async def optimize_cache_configuration(self) -> Dict[str, Any]:
        """
        Optimize cache configuration based on usage patterns
        Implements continuous improvement through analytics
        """

        optimization_results = {
            "timestamp": datetime.now().isoformat(),
            "optimizations_applied": [],
            "performance_improvements": {},
        }

        # Analyze phase-specific performance
        for phase, cache in self.phase_cache.items():
            stats = cache.get_cache_statistics()

            # Adjust cache size based on hit rate
            if (
                stats["hit_rate"] < 0.80
                and stats["total_entries"] < cache.max_entries * 0.8
            ):
                # Increase cache size for low-performing phases
                new_size = min(cache.max_entries * 1.2, 500)
                cache.max_entries = int(new_size)
                optimization_results["optimizations_applied"].append(
                    f"Increased {phase} cache size to {new_size}"
                )

            # Adjust similarity threshold based on performance
            if stats["hit_rate"] > 0.90 and cache.threshold < 0.90:
                # Increase threshold for high-performing phases
                cache.threshold = min(cache.threshold + 0.02, 0.92)
                optimization_results["optimizations_applied"].append(
                    f"Increased {phase} similarity threshold to {cache.threshold:.3f}"
                )

        return optimization_results


class ProblemTypeClassifier:
    """Simple problem type classifier for cache optimization"""

    def classify_problem_type(self, problem_context: str) -> str:
        """Classify problem type for cache key generation"""

        problem_lower = problem_context.lower()

        # Strategic planning indicators
        if any(
            keyword in problem_lower
            for keyword in ["strategy", "planning", "roadmap", "vision"]
        ):
            return "strategic_planning"

        # Investment analysis indicators
        elif any(
            keyword in problem_lower
            for keyword in ["investment", "financial", "roi", "valuation"]
        ):
            return "investment_analysis"

        # Operational improvement indicators
        elif any(
            keyword in problem_lower
            for keyword in ["process", "efficiency", "optimization", "cost"]
        ):
            return "operational_improvement"

        # Market analysis indicators
        elif any(
            keyword in problem_lower
            for keyword in ["market", "competitive", "customer", "segment"]
        ):
            return "market_analysis"

        # Default classification
        else:
            return "general_analysis"


# Factory function for easy instantiation
def create_phase_based_cache() -> MetisPhaseBasedContextCache:
    """Create and configure phase-based context cache instance"""
    return MetisPhaseBasedContextCache()


# Integration helper for workflow engine
async def cache_workflow_context(
    phase: str,
    engagement_context: EngagementContext,
    cognitive_state: CognitiveState,
    result_content: Any,
    entry_type: CacheEntryType = CacheEntryType.ANALYSIS_RESULT,
) -> bool:
    """
    Helper function to cache workflow context during execution

    Args:
        phase: Current engagement phase
        engagement_context: Current engagement context
        cognitive_state: Current cognitive processing state
        result_content: Content to cache
        entry_type: Type of cache entry
    """

    cache = create_phase_based_cache()

    return await cache.store_phase_context(
        phase=phase,
        problem_context=engagement_context.problem_statement,
        mental_models=cognitive_state.selected_mental_models,
        content=result_content,
        entry_type=entry_type,
        confidence_score=0.85,
        user_profile="default",
    )


# Integration helper for context retrieval
async def retrieve_cached_workflow_context(
    phase: str, engagement_context: EngagementContext, cognitive_state: CognitiveState
) -> CachedContextResult:
    """
    Helper function to retrieve cached workflow context

    Args:
        phase: Current engagement phase
        engagement_context: Current engagement context
        cognitive_state: Current cognitive processing state
    """

    cache = create_phase_based_cache()

    return await cache.get_phase_optimized_context(
        current_phase=phase,
        problem_context=engagement_context.problem_statement,
        mental_models=cognitive_state.selected_mental_models,
        user_profile="default",
    )
