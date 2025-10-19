"""
Sprint 1.5: Context Intelligence Interface
F003: Interface definitions for Context Intelligence Engine integration with Factory Pattern

This module defines the contracts for context intelligence operations,
supporting the revolutionary Context Intelligence Platform with multi-layer
caching, Manus taxonomy, and KV-cache optimization.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple

# Import types to avoid circular dependencies
try:
    from src.engine.models.data_contracts import ContextElement, ContextRelevanceScore
    from src.models.context_taxonomy import ContextIntelligenceResult

    TYPES_AVAILABLE = True
except ImportError:
    TYPES_AVAILABLE = False
    # Use Any as fallback to avoid import issues
    ContextElement = Any
    ContextRelevanceScore = Any
    ContextIntelligenceResult = Any


class IContextIntelligence(ABC):
    """
    Interface for Context Intelligence Engine operations

    Defines the contract for the revolutionary Context Intelligence Platform
    that uses AI's own thinking process for context curation and optimization.
    """

    @abstractmethod
    async def get_relevant_context(
        self,
        current_query: str,
        max_contexts: int = 5,
        engagement_id: Optional[str] = None,
    ) -> List[
        Tuple[Any, Any]
    ]:  # List[Tuple[CognitiveExhaustContext, ContextRelevanceScore]]
        """
        Get most relevant contexts using Context Intelligence with multi-layer caching

        Args:
            current_query: Current user query/goal
            max_contexts: Maximum number of contexts to return
            engagement_id: Optional engagement ID for L2/L3 cache access

        Returns:
            List of tuples containing (context, relevance_score)
        """
        pass

    @abstractmethod
    async def store_cognitive_exhaust_triple_layer(
        self,
        engagement_id: str,
        phase: str,
        mental_model: str,
        thinking_process: str,
        cleaned_response: str,
        confidence: float,
    ):
        """
        Store cognitive exhaust in L1, L2, and L3 caches

        Args:
            engagement_id: Engagement identifier
            phase: Processing phase
            mental_model: Mental model applied
            thinking_process: AI's thinking process from <thinking> tags
            cleaned_response: Clean response without thinking tags
            confidence: Confidence score
        """
        pass

    @abstractmethod
    async def analyze_contexts_with_manus_taxonomy(
        self,
        context_contents: List[str],
        current_query: str,
        engagement_id: str,
        cognitive_coherence_scores: Optional[List[float]] = None,
    ) -> Any:  # ContextIntelligenceResult
        """
        Comprehensive context analysis using Manus Taxonomy

        Args:
            context_contents: Raw context content strings
            current_query: Current user query/goal
            engagement_id: Engagement identifier
            cognitive_coherence_scores: Optional cognitive coherence scores

        Returns:
            ContextIntelligenceResult with complete analysis
        """
        pass

    @abstractmethod
    async def get_engine_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive engine statistics for observability

        Returns:
            Dictionary containing cache performance, taxonomy status, and metrics
        """
        pass

    @abstractmethod
    async def close(self):
        """Clean shutdown of Context Intelligence Engine with all cache layers"""
        pass


class IKVCacheOptimizer(ABC):
    """
    Interface for KV-Cache optimization based on Manus.im insights

    Provides 10x cost reduction through intelligent caching strategies
    """

    @abstractmethod
    def create_stable_prompt_prefix(self, base_context: str) -> str:
        """
        Create stable prompt prefix for KV-cache optimization

        Args:
            base_context: Base context to stabilize

        Returns:
            Stable prefix string optimized for caching
        """
        pass

    @abstractmethod
    def calculate_cache_hit_probability(
        self, current_prompt: str, cache_history: List[str]
    ) -> float:
        """
        Calculate probability of cache hit for given prompt

        Args:
            current_prompt: Current prompt being processed
            cache_history: History of cached prompts

        Returns:
            Probability score (0.0 to 1.0) of cache hit
        """
        pass

    @abstractmethod
    def optimize_for_caching(self, prompt_components: Dict[str, str]) -> Dict[str, str]:
        """
        Optimize prompt components for maximum cache efficiency

        Args:
            prompt_components: Dictionary of prompt parts

        Returns:
            Optimized prompt components for caching
        """
        pass

    @abstractmethod
    def get_cache_metrics(self) -> Dict[str, Any]:
        """
        Get KV-cache performance metrics

        Returns:
            Dictionary containing hit rates, cost savings, and optimization stats
        """
        pass


class IContextTaxonomy(ABC):
    """
    Interface for Manus Context Taxonomy operations

    Provides systematic context classification and relevance scoring
    """

    @abstractmethod
    def classify_context_type(
        self, content: str
    ) -> Tuple[str, float]:  # Tuple[ContextType, confidence]
        """
        Classify context content into Manus taxonomy with confidence

        Args:
            content: Context content to classify

        Returns:
            Tuple of (context_type, confidence_score)
        """
        pass

    @abstractmethod
    def score_context_relevance(
        self,
        context_element: Any,  # ContextElement
        current_query: str,
        cognitive_coherence_score: float = 0.0,
    ) -> Any:  # ContextRelevanceScore
        """
        Score context element relevance using Manus enhanced methodology

        Args:
            context_element: Context element to score
            current_query: Current user query/goal
            cognitive_coherence_score: Cognitive coherence from Operation Mindforge

        Returns:
            ContextRelevanceScore with detailed breakdown
        """
        pass

    @abstractmethod
    def compress_contexts(
        self,
        context_elements: List[Any],  # List[ContextElement]
        target_compression_ratio: float,
        strategy: str = "summarization",
    ) -> Tuple[List[Any], float]:  # Tuple[List[ContextElement], actual_ratio]
        """
        Compress context elements using specified strategy

        Args:
            context_elements: Elements to compress
            target_compression_ratio: Target ratio (0.5 = 50% compression)
            strategy: Compression strategy to use

        Returns:
            Tuple of (compressed_elements, actual_compression_ratio)
        """
        pass


class IContextIntelligenceFactory(ABC):
    """
    Interface for Context Intelligence Factory operations

    Provides dependency injection and configuration for context intelligence
    """

    @abstractmethod
    def create_context_intelligence(
        self,
        settings: Any,  # CognitiveEngineSettings
        enable_cache_layers: Optional[List[str]] = None,
        enable_manus_taxonomy: bool = True,
        enable_kv_optimization: bool = True,
    ) -> IContextIntelligence:
        """
        Create configured Context Intelligence Engine instance

        Args:
            settings: Cognitive engine settings
            enable_cache_layers: List of cache layers to enable (L1, L2, L3)
            enable_manus_taxonomy: Whether to enable Manus taxonomy
            enable_kv_optimization: Whether to enable KV-cache optimization

        Returns:
            Configured Context Intelligence Engine instance
        """
        pass

    @abstractmethod
    def create_kv_cache_optimizer(self, settings: Any) -> IKVCacheOptimizer:
        """Create KV-cache optimizer instance"""
        pass

    @abstractmethod
    def create_context_taxonomy(self, settings: Any) -> IContextTaxonomy:
        """Create context taxonomy classifier instance"""
        pass


class ICacheLayer(ABC):
    """
    Interface for individual cache layer operations

    Supports L1 (memory), L2 (Redis), and L3 (Supabase) cache layers
    """

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Retrieve value from cache layer"""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Store value in cache layer"""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache layer"""
        pass

    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache layer statistics"""
        pass

    @abstractmethod
    async def close(self):
        """Close cache layer connections"""
        pass


# Factory function for creating context intelligence engines
def create_context_intelligence_engine_interface(
    settings: Any, enable_cache_layers: Optional[List[str]] = None, **kwargs
) -> IContextIntelligence:
    """
    Factory function for creating Context Intelligence Engine with interface compatibility

    This provides a clean interface-based way to create context intelligence engines
    while maintaining backward compatibility with the existing factory pattern.
    """
    try:
        from src.core.context_intelligence_engine import (
            create_context_intelligence_engine,
        )

        return create_context_intelligence_engine(settings)
    except ImportError as e:
        raise ImportError(f"Failed to create Context Intelligence Engine: {e}")


# Validation utilities
def validate_context_intelligence_config(config: Dict[str, Any]) -> bool:
    """
    Validate context intelligence configuration

    Args:
        config: Configuration dictionary

    Returns:
        True if configuration is valid
    """
    required_fields = [
        "enable_context_intelligence",
        "cache_layers",
        "manus_taxonomy",
        "kv_cache_optimization",
    ]

    for field in required_fields:
        if field not in config:
            return False

    # Validate cache layers
    valid_cache_layers = {"L1", "L2", "L3"}
    if not all(layer in valid_cache_layers for layer in config.get("cache_layers", [])):
        return False

    return True


# Constants for configuration
CONTEXT_INTELLIGENCE_DEFAULTS = {
    "enable_context_intelligence": True,
    "cache_layers": ["L1", "L2", "L3"],
    "manus_taxonomy": True,
    "kv_cache_optimization": True,
    "compression_strategy": "reversible",
    "single_agent_depth_mode": True,
    "kv_cache_hit_target": 0.8,  # 80% cache hit rate target
    "context_compression_reversible": True,
}

# Performance targets from Manus.im insights
PERFORMANCE_TARGETS = {
    "kv_cache_hit_rate": 0.8,  # 80% cache hit rate
    "cost_reduction_factor": 10,  # 10x cost reduction on cached tokens
    "cache_latency_ms": 50,  # Sub-50ms cache access
    "compression_ratio": 0.6,  # 40% compression while preserving meaning
}
