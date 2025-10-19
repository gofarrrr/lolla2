"""
N-Way Selection Service - Extracted from OptimalConsultantEngine
Handles semantic cluster selection with 3-tier intelligence (semantic → manual → keyword fallback)

This service implements the core N-Way cluster selection logic with proper dependency injection
and complete Glass-Box transparency integration.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

# Glass-Box Integration - CRITICAL
from src.core.unified_context_stream import UnifiedContextStream, ContextEventType
from src.engine.core.tool_decision_framework import (
    ToolDecisionFramework,
)

# Semantic search dependencies
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np

    SEMANTIC_SEARCH_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    np = None
    SEMANTIC_SEARCH_AVAILABLE = False

# Database integration
try:
    from supabase import Client

    SUPABASE_AVAILABLE = True
except ImportError:
    Client = None
    SUPABASE_AVAILABLE = False


@dataclass
class NWaySelectionResult:
    """Result of N-Way cluster selection"""

    selected_clusters: List[str]
    selection_method: str  # semantic, manual_similarity, keyword_fallback
    confidence_score: float
    processing_time_ms: int
    metadata: Dict[str, Any]


class NWaySelectionService:
    """
    N-Way Cluster Selection Service - Level 3 Intelligence

    Extracted from OptimalConsultantEngine monolith with dependency injection.
    Provides semantic cluster selection with graceful degradation.

    Integrates with UnifiedContextStream for complete Glass-Box transparency.
    """

    def __init__(
        self,
        context_stream: UnifiedContextStream,
        tool_framework: ToolDecisionFramework,
        supabase_client: Optional[Client] = None,
    ):
        """
        Initialize N-Way Selection Service with dependency injection

        Args:
            context_stream: UnifiedContextStream for Glass-Box transparency
            tool_framework: ToolDecisionFramework for decision auditing
            supabase_client: Optional Supabase client for database queries
        """
        self.context_stream = context_stream
        self.tool_framework = tool_framework
        self.supabase_client = supabase_client
        self.logger = logging.getLogger(__name__)

        # Initialize semantic model if available
        self.semantic_model = None
        if SEMANTIC_SEARCH_AVAILABLE:
            try:
                self.semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
                self.logger.info("🧠 Semantic model loaded: all-MiniLM-L6-v2")
            except Exception as e:
                self.logger.warning(f"Failed to load semantic model: {e}")

        self.selection_cache: Dict[str, NWaySelectionResult] = {}
        self.performance_metrics = {
            "semantic_selections": 0,
            "manual_selections": 0,
            "keyword_selections": 0,
            "cache_hits": 0,
        }

    async def select_relevant_nway_clusters(
        self, enhanced_query: str, top_k: int = 3, engagement_id: Optional[str] = None
    ) -> NWaySelectionResult:
        """
        Main entry point for N-Way cluster selection with 3-tier intelligence

        Selection Strategy:
        1. Semantic vector search (Level 3) - if available
        2. Manual similarity (Level 2) - if configured
        3. Keyword fallback (Level 1) - always available

        Args:
            enhanced_query: The enhanced query from Socratic process
            top_k: Number of clusters to select (default 3)
            engagement_id: Optional engagement ID for audit trail

        Returns:
            NWaySelectionResult with selected clusters and metadata
        """
        start_time = datetime.now()

        # Glass-Box: Log selection start
        self.context_stream.add_event(
            event_type=ContextEventType.NWAY_CLUSTER_ACTIVATED,
            data={
                "enhanced_query": (
                    enhanced_query[:200] + "..."
                    if len(enhanced_query) > 200
                    else enhanced_query
                ),
                "top_k": top_k,
                "engagement_id": engagement_id,
                "available_methods": self._get_available_methods(),
            },
            metadata={
                "service": "NWaySelectionService",
                "method": "select_relevant_nway_clusters",
            },
        )

        # Check cache first
        cache_key = f"{hash(enhanced_query)}_{top_k}"
        if cache_key in self.selection_cache:
            self.performance_metrics["cache_hits"] += 1
            cached_result = self.selection_cache[cache_key]

            # Glass-Box: Log cache hit
            self.context_stream.add_event(
                event_type=ContextEventType.TOOL_EXECUTION,
                data={"cache_hit": True, "clusters": cached_result.selected_clusters},
                metadata={
                    "service": "NWaySelectionService",
                    "method": "cache_retrieval",
                },
            )

            return cached_result

        # Determine selection method and execute
        try:
            if self.semantic_model and SEMANTIC_SEARCH_AVAILABLE:
                result = await self._select_semantic(enhanced_query, top_k)
                self.performance_metrics["semantic_selections"] += 1
            elif self._manual_similarity_available():
                result = await self._select_manual_similarity(enhanced_query, top_k)
                self.performance_metrics["manual_selections"] += 1
            else:
                result = await self._select_keyword_fallback(enhanced_query, top_k)
                self.performance_metrics["keyword_selections"] += 1

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            result.processing_time_ms = int(processing_time)

            # Cache result for future use
            self.selection_cache[cache_key] = result

            # Glass-Box: Log successful selection
            self.context_stream.add_event(
                event_type=ContextEventType.CONSULTANT_SELECTION,
                data={
                    "selected_clusters": result.selected_clusters,
                    "selection_method": result.selection_method,
                    "confidence_score": result.confidence_score,
                    "processing_time_ms": result.processing_time_ms,
                },
                metadata={
                    "service": "NWaySelectionService",
                    "method": result.selection_method,
                    "engagement_id": engagement_id,
                },
            )

            return result

        except Exception as e:
            # Glass-Box: Log selection error
            self.context_stream.add_event(
                event_type=ContextEventType.ERROR_OCCURRED,
                data={
                    "error": str(e),
                    "enhanced_query": (
                        enhanced_query[:100] + "..."
                        if len(enhanced_query) > 100
                        else enhanced_query
                    ),
                },
                metadata={
                    "service": "NWaySelectionService",
                    "method": "error_handling",
                },
            )

            self.logger.error(f"N-Way selection failed: {e}")

            # Fallback to keyword selection
            result = await self._select_keyword_fallback(enhanced_query, top_k)
            result.metadata["error_occurred"] = True
            result.metadata["error_message"] = str(e)

            return result

    async def _select_semantic(
        self, enhanced_query: str, top_k: int
    ) -> NWaySelectionResult:
        """Level 3: Semantic vector search selection"""

        # Generate query embedding
        query_embedding = self.semantic_model.encode(enhanced_query).tolist()

        # Search database for similar clusters
        if self.supabase_client:
            try:
                # Vector similarity search (requires pgvector extension)
                result = self.supabase_client.rpc(
                    "match_nway_clusters",
                    {
                        "query_embedding": query_embedding,
                        "match_threshold": 0.7,
                        "match_count": top_k,
                    },
                ).execute()

                if result.data:
                    selected_clusters = [
                        item["interaction_id"] for item in result.data[:top_k]
                    ]
                    confidence = sum(
                        item.get("similarity", 0) for item in result.data
                    ) / len(result.data)
                else:
                    # Fallback if no semantic matches
                    selected_clusters = await self._get_fallback_clusters(top_k)
                    confidence = 0.3

            except Exception as e:
                self.logger.warning(f"Semantic database search failed: {e}")
                selected_clusters = await self._get_fallback_clusters(top_k)
                confidence = 0.3
        else:
            # No database connection - use fallback
            selected_clusters = await self._get_fallback_clusters(top_k)
            confidence = 0.3

        return NWaySelectionResult(
            selected_clusters=selected_clusters,
            selection_method="semantic_vector_search",
            confidence_score=confidence,
            processing_time_ms=0,  # Will be set by caller
            metadata={
                "semantic_model": "all-MiniLM-L6-v2",
                "embedding_dimension": len(query_embedding) if query_embedding else 0,
                "database_connected": bool(self.supabase_client),
            },
        )

    async def _select_manual_similarity(
        self, enhanced_query: str, top_k: int
    ) -> NWaySelectionResult:
        """Level 2: Manual similarity patterns (if configured)"""

        # This would implement manual similarity patterns
        # For now, fallback to keyword selection
        return await self._select_keyword_fallback(enhanced_query, top_k)

    async def _select_keyword_fallback(
        self, enhanced_query: str, top_k: int
    ) -> NWaySelectionResult:
        """Level 1: Keyword-based cluster selection (always available)"""

        keywords = self._extract_keywords(enhanced_query)

        # Map keywords to clusters using hardcoded patterns
        keyword_cluster_map = {
            "strategic": ["strategic_analysis_cluster", "market_intelligence_cluster"],
            "problem": ["problem_solving_cluster", "root_cause_analysis_cluster"],
            "solution": ["solution_design_cluster", "creative_thinking_cluster"],
            "analysis": ["analytical_thinking_cluster", "data_analysis_cluster"],
            "market": ["market_intelligence_cluster", "competitive_analysis_cluster"],
            "design": ["solution_design_cluster", "systems_thinking_cluster"],
        }

        # Score clusters based on keyword matches
        cluster_scores = {}
        for keyword in keywords:
            if keyword.lower() in keyword_cluster_map:
                for cluster in keyword_cluster_map[keyword.lower()]:
                    cluster_scores[cluster] = cluster_scores.get(cluster, 0) + 1

        # Select top clusters
        if cluster_scores:
            sorted_clusters = sorted(
                cluster_scores.items(), key=lambda x: x[1], reverse=True
            )
            selected_clusters = [cluster for cluster, score in sorted_clusters[:top_k]]
            confidence = min(
                1.0, sum(cluster_scores.values()) / (len(keywords) * top_k)
            )
        else:
            # No keyword matches - use default clusters
            selected_clusters = await self._get_fallback_clusters(top_k)
            confidence = 0.2

        return NWaySelectionResult(
            selected_clusters=selected_clusters,
            selection_method="keyword_fallback",
            confidence_score=confidence,
            processing_time_ms=0,  # Will be set by caller
            metadata={
                "keywords_found": keywords,
                "cluster_scores": cluster_scores,
                "fallback_used": not bool(cluster_scores),
            },
        )

    async def _get_fallback_clusters(self, top_k: int) -> List[str]:
        """Get default fallback clusters when no selection method works"""
        default_clusters = [
            "strategic_analysis_cluster",
            "problem_solving_cluster",
            "solution_design_cluster",
            "analytical_thinking_cluster",
            "systems_thinking_cluster",
        ]
        return default_clusters[:top_k]

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract relevant keywords from query"""
        import re

        # Simple keyword extraction
        words = re.findall(r"\b[a-zA-Z]{3,}\b", query.lower())

        # Filter out common words
        stop_words = {
            "the",
            "and",
            "but",
            "for",
            "are",
            "with",
            "this",
            "that",
            "from",
            "they",
            "have",
            "been",
        }
        keywords = [word for word in words if word not in stop_words]

        return keywords[:10]  # Limit to top 10 keywords

    def _get_available_methods(self) -> List[str]:
        """Get list of available selection methods"""
        methods = ["keyword_fallback"]  # Always available

        if self.semantic_model and SEMANTIC_SEARCH_AVAILABLE:
            methods.append("semantic_vector_search")

        if self._manual_similarity_available():
            methods.append("manual_similarity")

        return methods

    def _manual_similarity_available(self) -> bool:
        """Check if manual similarity patterns are configured"""
        # For now, manual similarity is not implemented
        return False

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get service performance metrics"""
        total_selections = (
            sum(self.performance_metrics.values())
            - self.performance_metrics["cache_hits"]
        )

        return {
            "total_selections": total_selections,
            "semantic_percentage": (
                self.performance_metrics["semantic_selections"]
                / max(1, total_selections)
            )
            * 100,
            "manual_percentage": (
                self.performance_metrics["manual_selections"] / max(1, total_selections)
            )
            * 100,
            "keyword_percentage": (
                self.performance_metrics["keyword_selections"]
                / max(1, total_selections)
            )
            * 100,
            "cache_hit_rate": (
                self.performance_metrics["cache_hits"]
                / max(1, total_selections + self.performance_metrics["cache_hits"])
            )
            * 100,
            "available_methods": self._get_available_methods(),
        }

    def clear_cache(self) -> int:
        """Clear selection cache and return number of items cleared"""
        cleared_count = len(self.selection_cache)
        self.selection_cache.clear()
        return cleared_count


# Factory function for dependency injection
def create_nway_selection_service(
    context_stream: UnifiedContextStream,
    tool_framework: ToolDecisionFramework,
    supabase_client: Optional[Client] = None,
) -> NWaySelectionService:
    """
    Factory function to create NWaySelectionService with proper dependencies

    This ensures proper dependency injection and Glass-Box integration
    """
    return NWaySelectionService(
        context_stream=context_stream,
        tool_framework=tool_framework,
        supabase_client=supabase_client,
    )
