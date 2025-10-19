#!/usr/bin/env python3
"""
METIS Research Storage Manager
Handles storage and retrieval of multi-provider research results for cognitive engine context
"""

import os
import logging
from uuid import UUID, uuid4
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import asdict

from supabase import create_client

# Import our research types
from src.engine.integrations.perplexity_client_advanced import (
    AdvancedResearchResult,
    ResearchInsight,
    ResearchTemplateType,
)
from src.models.research_types import ResearchRequest

logger = logging.getLogger(__name__)


class ResearchStorageManager:
    """Manages storage and retrieval of research results for cognitive engine integration"""

    def __init__(self, enable_supabase: bool = True):
        self.enable_supabase = enable_supabase
        self.supabase_client = None
        self.logger = logger

        if self.enable_supabase:
            self._initialize_supabase()

    def _initialize_supabase(self):
        """Initialize Supabase client for research storage"""
        try:
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_ANON_KEY")

            if not supabase_url or not supabase_key:
                self.logger.warning(
                    "⚠️ Supabase credentials not found - research storage disabled"
                )
                self.enable_supabase = False
                return

            self.supabase_client = create_client(supabase_url, supabase_key)
            self.logger.info("✅ Research Storage Manager initialized with Supabase")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Supabase client: {e}")
            self.enable_supabase = False

    async def store_research_session(
        self,
        engagement_id: UUID,
        request: ResearchRequest,
        result: AdvancedResearchResult,
        providers_used: List[str],
        execution_metadata: Dict[str, Any],
    ) -> Optional[str]:
        """
        Store a complete research session with all results and metadata

        Args:
            engagement_id: The engagement this research belongs to
            request: The original research request
            result: The complete research result
            providers_used: List of providers that were used
            execution_metadata: Performance and execution data

        Returns:
            Session ID if successful, None if failed
        """
        if not self.enable_supabase:
            self.logger.warning("⚠️ Research storage disabled - not storing session")
            return None

        try:
            session_id = str(uuid4())

            # Prepare session data
            session_data = {
                "session_id": session_id,
                "engagement_id": str(engagement_id),
                "research_query": request.query,
                "research_context": request.context,
                "template_type": str(request.template_type.value),
                "research_mode": str(request.mode.value),
                "strategy_used": str(request.strategy.value),
                "providers_used": providers_used,
                "execution_phases": execution_metadata.get("total_phases", 1),
                # Results summary
                "total_sources_found": len(result.sources),
                "unique_sources_count": len(
                    set(source.url for source in result.sources if source.url)
                ),
                "confidence_score": result.overall_confidence,
                "coverage_completeness": result.coverage_completeness,
                "source_diversity_score": result.source_diversity_score,
                "fact_validation_score": result.fact_validation_score,
                # Performance metrics
                "total_cost_usd": result.cost_usd,
                "processing_time_ms": result.total_processing_time_ms,
                "tokens_consumed": result.tokens_consumed,
                # Quality indicators
                "information_gaps": result.information_gaps,
                "confidence_limitations": result.confidence_limitations,
                "additional_research_recommendations": result.additional_research_recommendations,
                # Full results
                "executive_summary": result.executive_summary,
                "detailed_findings": result.detailed_findings,
                "raw_results": (
                    asdict(result) if hasattr(result, "__dict__") else str(result)
                ),
                # Embeddings (placeholder - would need actual embedding service)
                "query_embedding": self._generate_embedding_placeholder(request.query),
                "results_embedding": self._generate_embedding_placeholder(
                    result.executive_summary
                ),
                # Metadata
                "session_metadata": execution_metadata,
                "status": "completed",
            }

            # Insert research session
            session_response = (
                self.supabase_client.table("research_sessions")
                .insert(session_data)
                .execute()
            )

            if session_response.data:
                self.logger.info(f"✅ Stored research session: {session_id}")

                # Store individual insights
                await self._store_research_insights(
                    session_id, engagement_id, result.key_insights
                )

                # Store provider performance data
                await self._store_provider_performance(
                    session_id, engagement_id, providers_used, execution_metadata
                )

                # Update research context cache
                await self._update_research_context_cache(
                    engagement_id, request, result
                )

                return session_id
            else:
                self.logger.error(
                    "❌ Failed to store research session - no data returned"
                )
                return None

        except Exception as e:
            self.logger.error(f"❌ Error storing research session: {e}")
            return None

    async def _store_research_insights(
        self, session_id: str, engagement_id: UUID, insights: List[ResearchInsight]
    ):
        """Store individual research insights"""
        if not insights:
            return

        try:
            insights_data = []

            for i, insight in enumerate(insights):
                insight_data = {
                    "insight_id": f"{session_id}_insight_{i}",
                    "session_id": session_id,
                    "engagement_id": str(engagement_id),
                    "claim": insight.claim,
                    "confidence": insight.confidence,
                    "evidence_strength": insight.evidence_strength,
                    "fact_type": insight.fact_type,
                    "verification_status": insight.verification_status,
                    "supporting_sources": insight.supporting_sources,
                    "contradicting_sources": insight.contradicting_sources,
                    "source_count": len(insight.supporting_sources),
                    # Categorization (could be enhanced with NLP)
                    "insight_category": self._categorize_insight(insight.claim),
                    "business_impact_level": self._assess_business_impact(
                        insight.confidence, insight.evidence_strength
                    ),
                    "time_sensitivity": self._assess_time_sensitivity(insight.claim),
                    # Embedding
                    "insight_embedding": self._generate_embedding_placeholder(
                        insight.claim
                    ),
                    "insight_metadata": {
                        "original_index": i,
                        "source_diversity": len(set(insight.supporting_sources)),
                    },
                }
                insights_data.append(insight_data)

            # Batch insert insights
            insights_response = (
                self.supabase_client.table("research_insights")
                .insert(insights_data)
                .execute()
            )

            if insights_response.data:
                self.logger.info(f"✅ Stored {len(insights_data)} research insights")
            else:
                self.logger.error("❌ Failed to store research insights")

        except Exception as e:
            self.logger.error(f"❌ Error storing research insights: {e}")

    async def _store_provider_performance(
        self,
        session_id: str,
        engagement_id: UUID,
        providers_used: List[str],
        execution_metadata: Dict[str, Any],
    ):
        """Store provider performance data"""
        try:
            performance_data = []

            # Extract performance data for each provider
            provider_metrics = execution_metadata.get("provider_performance", {})

            for i, provider in enumerate(providers_used):
                metrics = provider_metrics.get(provider, {})

                performance_record = {
                    "session_id": session_id,
                    "engagement_id": str(engagement_id),
                    "provider_name": provider,
                    "execution_phase": i + 1,
                    "phase_type": metrics.get("phase_type", "discovery"),
                    # Performance metrics
                    "sources_found": metrics.get("sources_found", 0),
                    "processing_time_ms": metrics.get("processing_time_ms", 0),
                    "cost_usd": metrics.get("cost_usd", 0.0),
                    "tokens_used": metrics.get("tokens_used", 0),
                    "confidence_score": metrics.get("confidence_score", 0.0),
                    # Success metrics
                    "success": metrics.get("success", True),
                    "error_message": metrics.get("error_message"),
                    "retry_count": metrics.get("retry_count", 0),
                    # Quality metrics
                    "source_quality_average": metrics.get(
                        "source_quality_average", 0.0
                    ),
                    "duplicate_sources_found": metrics.get(
                        "duplicate_sources_found", 0
                    ),
                    "unique_sources_contributed": metrics.get(
                        "unique_sources_contributed", 0
                    ),
                    "provider_metadata": metrics,
                }
                performance_data.append(performance_record)

            if performance_data:
                perf_response = (
                    self.supabase_client.table("provider_performance_logs")
                    .insert(performance_data)
                    .execute()
                )

                if perf_response.data:
                    self.logger.info(
                        f"✅ Stored performance data for {len(providers_used)} providers"
                    )
                else:
                    self.logger.error("❌ Failed to store provider performance data")

        except Exception as e:
            self.logger.error(f"❌ Error storing provider performance: {e}")

    async def _update_research_context_cache(
        self,
        engagement_id: UUID,
        request: ResearchRequest,
        result: AdvancedResearchResult,
    ):
        """Update the research context cache for fast cognitive engine retrieval"""
        try:
            # Generate semantic key
            context_key = f"{request.template_type.value}_{hash(request.query) % 10000}"

            # Prepare key insights for cache
            key_insights = [
                {
                    "claim": insight.claim,
                    "confidence": insight.confidence,
                    "sources_count": len(insight.supporting_sources),
                }
                for insight in result.key_insights[:5]  # Top 5 insights
            ]

            # Prepare relevant sources
            relevant_sources = [
                {
                    "url": source.url,
                    "title": source.title,
                    "credibility": source.credibility_score,
                    "relevance": source.relevance_score,
                }
                for source in result.sources[:10]  # Top 10 sources
            ]

            cache_data = {
                "engagement_id": str(engagement_id),
                "context_key": context_key,
                "context_summary": result.executive_summary,
                "key_insights": key_insights,
                "relevant_sources": relevant_sources,
                "research_confidence": result.overall_confidence,
                # Relevance scoring
                "semantic_similarity_scores": {},
                "recency_weight": 1.0,
                "authority_weight": result.source_diversity_score,
                # Fast retrieval
                "context_embedding": self._generate_embedding_placeholder(
                    result.executive_summary
                ),
                "tags": [request.template_type.value, request.mode.value],
                # Session reference
                "source_session_ids": [],  # Would be populated with session_id
                # Cache management
                "expires_at": (datetime.now(timezone.utc)).isoformat(),  # Set expiry
            }

            # Upsert context cache
            cache_response = (
                self.supabase_client.table("research_context_cache")
                .upsert(cache_data, on_conflict="engagement_id,context_key")
                .execute()
            )

            if cache_response.data:
                self.logger.info(f"✅ Updated research context cache: {context_key}")
            else:
                self.logger.error("❌ Failed to update research context cache")

        except Exception as e:
            self.logger.error(f"❌ Error updating research context cache: {e}")

    async def get_research_context_for_cognitive_engine(
        self,
        engagement_id: UUID,
        query: Optional[str] = None,
        template_type: Optional[ResearchTemplateType] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant research context for cognitive engine processing

        Args:
            engagement_id: The engagement to get context for
            query: Optional semantic query for similarity matching
            template_type: Optional template type filter
            limit: Maximum number of context items to return

        Returns:
            List of research context items
        """
        if not self.enable_supabase:
            self.logger.warning("⚠️ Research storage disabled - returning empty context")
            return []

        try:
            # Build query
            query_builder = self.supabase_client.table(
                "cognitive_research_context"
            ).select("*")
            query_builder = query_builder.eq("engagement_id", str(engagement_id))

            if template_type:
                query_builder = query_builder.eq("template_type", template_type.value)

            # Execute query
            query_builder = query_builder.order("confidence_score", desc=True)
            query_builder = query_builder.limit(limit)

            response = query_builder.execute()

            if response.data:
                self.logger.info(
                    f"✅ Retrieved {len(response.data)} research context items"
                )
                return response.data
            else:
                self.logger.info("ℹ️ No research context found for engagement")
                return []

        except Exception as e:
            self.logger.error(f"❌ Error retrieving research context: {e}")
            return []

    def _generate_embedding_placeholder(self, text: str) -> List[float]:
        """
        Generate placeholder embedding (in production, use real embedding service)
        """
        # Simple hash-based placeholder - replace with actual embeddings
        import hashlib

        hash_bytes = hashlib.md5(text.encode()).digest()
        # Convert to 1536-dimensional vector (OpenAI embedding size)
        vector = []
        for i in range(1536):
            vector.append((hash_bytes[i % 16] - 128) / 128.0)
        return vector

    def _categorize_insight(self, claim: str) -> str:
        """Categorize insight based on content (placeholder - use NLP in production)"""
        claim_lower = claim.lower()

        if any(word in claim_lower for word in ["market", "trend", "growth", "size"]):
            return "market_trend"
        elif any(
            word in claim_lower
            for word in ["competitor", "competitive", "vs", "versus"]
        ):
            return "competitive_advantage"
        elif any(
            word in claim_lower for word in ["risk", "threat", "challenge", "problem"]
        ):
            return "risk_factor"
        elif any(
            word in claim_lower for word in ["opportunity", "potential", "advantage"]
        ):
            return "opportunity"
        else:
            return "general_insight"

    def _assess_business_impact(self, confidence: float, evidence: float) -> str:
        """Assess business impact level based on confidence and evidence"""
        impact_score = (confidence + evidence) / 2

        if impact_score > 0.7:
            return "high"
        elif impact_score > 0.4:
            return "medium"
        else:
            return "low"

    def _assess_time_sensitivity(self, claim: str) -> str:
        """Assess time sensitivity of insight (placeholder)"""
        claim_lower = claim.lower()

        if any(
            word in claim_lower
            for word in ["urgent", "immediate", "crisis", "breaking"]
        ):
            return "urgent"
        elif any(word in claim_lower for word in ["trend", "emerging", "developing"]):
            return "important"
        else:
            return "routine"


# Global instance for easy access
research_storage = ResearchStorageManager()
