#!/usr/bin/env python3
"""
User Research Storage Integration
Handles storage and retrieval of 3-tier research results with comprehensive source attribution
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

# Supabase integration
from supabase import Client as SupabaseClient

# METIS models
from src.integrations.research.models import ResearchResult
from src.engine.integrations.perplexity_client_advanced import (
    AdvancedResearchResult,
    EnhancedSource,
    ResearchInsight,
)


@dataclass
class UserResearchRequest:
    """User research request with context"""

    user_id: str
    engagement_id: Optional[str]
    research_tier: str  # 'regular', 'premium', 'enterprise'
    question_depth: str  # 'essential', 'strategic', 'expert'
    problem_statement: str
    progressive_questions: Dict[str, Any]
    context_data: Dict[str, Any]


@dataclass
class StoredResearchResult:
    """Research result as stored in database"""

    id: str
    user_id: str
    engagement_id: Optional[str]
    research_tier: str
    question_depth: str
    problem_statement: str
    executive_summary: str
    overall_confidence: float
    total_cost_usd: float
    processing_time_ms: int
    source_count: int
    insight_count: int
    created_at: datetime


class UserResearchStorage:
    """Handles storage and retrieval of user research results"""

    def __init__(self, supabase_client: SupabaseClient):
        self.supabase = supabase_client
        self.logger = logging.getLogger(__name__)

    async def store_research_result(
        self,
        request: UserResearchRequest,
        result: ResearchResult,
        sources: List[Dict[str, Any]] = None,
        insights: List[Dict[str, Any]] = None,
        perplexity_model: str = "sonar-pro",
    ) -> str:
        """
        Store research result with comprehensive source attribution

        Args:
            request: Original user research request
            result: Research result from research manager
            sources: Enhanced source data
            insights: Extracted insights
            perplexity_model: Model used for research

        Returns:
            str: ID of stored research result
        """

        try:
            # Prepare main research result record
            research_data = {
                "user_id": request.user_id,
                "engagement_id": request.engagement_id,
                "research_tier": request.research_tier,
                "question_depth": request.question_depth,
                "problem_statement": request.problem_statement,
                "progressive_questions": request.progressive_questions,
                "context_data": request.context_data,
                # Research execution
                "research_queries": (
                    result.queries if hasattr(result, "queries") else []
                ),
                "perplexity_model": perplexity_model,
                "processing_start": (
                    datetime.utcnow() - timedelta(milliseconds=result.time_spent_ms)
                ).isoformat(),
                "processing_end": datetime.utcnow().isoformat(),
                "processing_time_ms": result.time_spent_ms,
                # Results
                "executive_summary": result.summary,
                "key_insights": (
                    [insight for insight in result.bullets] if result.bullets else []
                ),
                "detailed_findings": (
                    "\n".join(result.bullets) if result.bullets else ""
                ),
                "recommendations": [],  # TODO: Extract from results
                # Quality metrics
                "overall_confidence": result.confidence,
                "coverage_completeness": result.coverage_score,
                "source_diversity_score": getattr(
                    result, "source_diversity_score", 0.0
                ),
                "fact_validation_score": result.consistency_score,
                # Cost & usage
                "total_cost_usd": self._calculate_tier_cost(
                    request.research_tier, result.time_spent_ms
                ),
                "tokens_consumed": getattr(result, "tokens_consumed", 0),
                "queries_executed": (
                    len(result.queries) if hasattr(result, "queries") else 1
                ),
                # Status
                "status": "completed" if not result.timeout_occurred else "timeout",
                "early_stopped": result.early_stopped,
                "timeout_occurred": result.timeout_occurred,
            }

            # Insert main research result
            research_response = (
                self.supabase.table("user_research_results")
                .insert(research_data)
                .execute()
            )

            if not research_response.data:
                raise Exception("Failed to insert research result")

            research_result_id = research_response.data[0]["id"]
            self.logger.info(f"✅ Stored research result: {research_result_id}")

            # Store sources if available
            if sources:
                await self._store_research_sources(research_result_id, sources)
            elif result.sources:  # Fallback to basic sources
                await self._store_basic_sources(research_result_id, result.sources)

            # Store insights if available
            if insights:
                await self._store_research_insights(research_result_id, insights)

            # Update user research usage
            await self._update_user_usage(
                request.user_id,
                request.research_tier,
                research_data["total_cost_usd"],
                result.confidence,
            )

            return research_result_id

        except Exception as e:
            self.logger.error(f"❌ Failed to store research result: {e}")
            raise

    async def store_advanced_research_result(
        self, request: UserResearchRequest, result: AdvancedResearchResult
    ) -> str:
        """Store advanced research result from AdvancedPerplexityClient"""

        try:
            # Convert advanced result to storage format
            research_data = {
                "user_id": request.user_id,
                "engagement_id": request.engagement_id,
                "research_tier": request.research_tier,
                "question_depth": request.question_depth,
                "problem_statement": request.problem_statement,
                "progressive_questions": request.progressive_questions,
                "context_data": request.context_data,
                # Research execution
                "research_queries": result.queries_executed,
                "perplexity_model": (
                    "sonar-deep-research"
                    if request.research_tier == "enterprise"
                    else "sonar-pro"
                ),
                "processing_start": (
                    datetime.utcnow()
                    - timedelta(milliseconds=result.total_processing_time_ms)
                ).isoformat(),
                "processing_end": datetime.utcnow().isoformat(),
                "processing_time_ms": result.total_processing_time_ms,
                # Results
                "executive_summary": result.executive_summary,
                "key_insights": [
                    {"claim": insight.claim, "confidence": insight.confidence}
                    for insight in result.key_insights
                ],
                "detailed_findings": result.detailed_findings,
                "recommendations": result.additional_research_recommendations,
                # Quality metrics
                "overall_confidence": result.overall_confidence,
                "coverage_completeness": result.coverage_completeness,
                "source_diversity_score": result.source_diversity_score,
                "fact_validation_score": result.fact_validation_score,
                # Cost & usage
                "total_cost_usd": result.cost_usd,
                "tokens_consumed": result.tokens_consumed,
                "queries_executed": len(result.queries_executed),
                # Status
                "status": "completed",
                "early_stopped": False,
                "timeout_occurred": False,
            }

            # Insert main research result
            research_response = (
                self.supabase.table("user_research_results")
                .insert(research_data)
                .execute()
            )
            research_result_id = research_response.data[0]["id"]

            # Store enhanced sources
            if result.sources:
                await self._store_enhanced_sources(research_result_id, result.sources)

            # Store research insights
            if result.key_insights:
                await self._store_advanced_insights(
                    research_result_id, result.key_insights
                )

            # Update user usage
            await self._update_user_usage(
                request.user_id,
                request.research_tier,
                result.cost_usd,
                result.overall_confidence,
            )

            self.logger.info(
                f"✅ Stored advanced research result: {research_result_id}"
            )
            return research_result_id

        except Exception as e:
            self.logger.error(f"❌ Failed to store advanced research result: {e}")
            raise

    async def get_user_research_history(
        self, user_id: str, limit: int = 20, offset: int = 0
    ) -> List[StoredResearchResult]:
        """Get user's research history"""

        try:
            # Call the stored function
            response = self.supabase.rpc(
                "get_user_research_history",
                {"p_user_id": user_id, "p_limit": limit, "p_offset": offset},
            ).execute()

            results = []
            for row in response.data:
                results.append(
                    StoredResearchResult(
                        id=row["id"],
                        user_id=user_id,
                        engagement_id=row.get("engagement_id"),
                        research_tier=row["research_tier"],
                        question_depth=row["question_depth"],
                        problem_statement=row["problem_statement"],
                        executive_summary=row["executive_summary"],
                        overall_confidence=row["overall_confidence"],
                        total_cost_usd=row["total_cost_usd"],
                        processing_time_ms=row["processing_time_ms"],
                        source_count=row["source_count"],
                        insight_count=row["insight_count"],
                        created_at=datetime.fromisoformat(
                            row["created_at"].replace("Z", "+00:00")
                        ),
                    )
                )

            return results

        except Exception as e:
            self.logger.error(
                f"❌ Failed to get research history for user {user_id}: {e}"
            )
            return []

    async def get_research_result_detail(
        self, user_id: str, research_result_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get detailed research result with sources and insights"""

        try:
            # Get main research result
            result_response = (
                self.supabase.table("user_research_results")
                .select("*")
                .eq("id", research_result_id)
                .eq("user_id", user_id)
                .execute()
            )

            if not result_response.data:
                return None

            research_result = result_response.data[0]

            # Get sources
            sources_response = (
                self.supabase.table("user_research_sources")
                .select("*")
                .eq("research_result_id", research_result_id)
                .order("display_order")
                .execute()
            )

            # Get insights
            insights_response = (
                self.supabase.table("user_research_insights")
                .select("*")
                .eq("research_result_id", research_result_id)
                .order("display_order")
                .execute()
            )

            # Combine all data
            return {
                **research_result,
                "sources": sources_response.data,
                "insights": insights_response.data,
            }

        except Exception as e:
            self.logger.error(f"❌ Failed to get research result detail: {e}")
            return None

    async def check_user_research_access(
        self, user_id: str, research_tier: str
    ) -> bool:
        """Check if user can access specified research tier"""

        try:
            response = self.supabase.rpc(
                "can_user_access_research_tier",
                {"p_user_id": user_id, "p_research_tier": research_tier},
            ).execute()

            return response.data if response.data is not None else False

        except Exception as e:
            self.logger.error(f"❌ Failed to check research access: {e}")
            return False

    async def _store_research_sources(
        self, research_result_id: str, sources: List[Dict[str, Any]]
    ) -> None:
        """Store research sources with credibility assessment"""

        source_records = []
        for i, source in enumerate(sources):
            source_record = {
                "research_result_id": research_result_id,
                "url": source.get("url", ""),
                "title": source.get("title", ""),
                "domain": source.get("domain", ""),
                "content_preview": (
                    source.get("content", "")[:300] if source.get("content") else ""
                ),
                "credibility_tier": source.get("credibility_tier", "unverified"),
                "credibility_score": source.get("credibility_score", 0.0),
                "fact_density": source.get("fact_density", 0.0),
                "citation_quality": source.get("citation_quality", 0.0),
                "bias_indicators": source.get("bias_indicators", []),
                "publication_date": source.get("publication_date"),
                "author": source.get("author"),
                "display_order": i,
            }
            source_records.append(source_record)

        if source_records:
            self.supabase.table("user_research_sources").insert(
                source_records
            ).execute()

    async def _store_basic_sources(
        self, research_result_id: str, sources: List[Dict[str, Any]]
    ) -> None:
        """Store basic sources from ResearchResult"""

        source_records = []
        for i, source in enumerate(sources):
            source_record = {
                "research_result_id": research_result_id,
                "url": source.get("url", ""),
                "title": source.get("title", ""),
                "domain": source.get("domain", ""),
                "content_preview": (
                    source.get("content", "")[:300] if source.get("content") else ""
                ),
                "credibility_tier": "unverified",
                "credibility_score": 0.5,
                "display_order": i,
            }
            source_records.append(source_record)

        if source_records:
            self.supabase.table("user_research_sources").insert(
                source_records
            ).execute()

    async def _store_enhanced_sources(
        self, research_result_id: str, sources: List[EnhancedSource]
    ) -> None:
        """Store enhanced sources from AdvancedResearchResult"""

        source_records = []
        for i, source in enumerate(sources):
            source_record = {
                "research_result_id": research_result_id,
                "url": source.url,
                "title": source.title,
                "domain": source.domain,
                "content_preview": source.content[:300] if source.content else "",
                "credibility_tier": source.credibility_tier.value,
                "credibility_score": source.credibility_score,
                "fact_density": source.fact_density,
                "citation_quality": source.citation_quality,
                "bias_indicators": source.bias_indicators,
                "publication_date": source.date,
                "author": source.author,
                "display_order": i,
            }
            source_records.append(source_record)

        if source_records:
            self.supabase.table("user_research_sources").insert(
                source_records
            ).execute()

    async def _store_research_insights(
        self, research_result_id: str, insights: List[Dict[str, Any]]
    ) -> None:
        """Store research insights"""

        insight_records = []
        for i, insight in enumerate(insights):
            insight_record = {
                "research_result_id": research_result_id,
                "claim": insight.get("claim", ""),
                "confidence": insight.get("confidence", 0.0),
                "evidence_strength": insight.get("evidence_strength", 0.0),
                "fact_type": insight.get("fact_type", "categorical"),
                "verification_status": insight.get("verification_status", "unverified"),
                "supporting_source_ids": insight.get("supporting_sources", []),
                "contradicting_source_ids": insight.get("contradicting_sources", []),
                "business_impact": insight.get("business_impact", "medium"),
                "insight_category": insight.get("category", "general"),
                "display_order": i,
            }
            insight_records.append(insight_record)

        if insight_records:
            self.supabase.table("user_research_insights").insert(
                insight_records
            ).execute()

    async def _store_advanced_insights(
        self, research_result_id: str, insights: List[ResearchInsight]
    ) -> None:
        """Store advanced research insights"""

        insight_records = []
        for i, insight in enumerate(insights):
            insight_record = {
                "research_result_id": research_result_id,
                "claim": insight.claim,
                "confidence": insight.confidence,
                "evidence_strength": insight.evidence_strength,
                "fact_type": insight.fact_type,
                "verification_status": insight.verification_status,
                "supporting_source_ids": insight.supporting_sources,
                "contradicting_source_ids": insight.contradicting_sources,
                "business_impact": "medium",  # Default, could be enhanced
                "insight_category": "general",  # Could be derived from fact_type
                "display_order": i,
            }
            insight_records.append(insight_record)

        if insight_records:
            self.supabase.table("user_research_insights").insert(
                insight_records
            ).execute()

    async def _update_user_usage(
        self, user_id: str, research_tier: str, cost_usd: float, confidence_score: float
    ) -> None:
        """Update user research usage statistics"""

        try:
            self.supabase.rpc(
                "update_user_research_usage",
                {
                    "p_user_id": user_id,
                    "p_research_tier": research_tier,
                    "p_cost_usd": cost_usd,
                    "p_confidence_score": confidence_score,
                },
            ).execute()

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to update user usage: {e}")

    def _calculate_tier_cost(
        self, research_tier: str, processing_time_ms: int
    ) -> float:
        """Calculate cost based on research tier"""

        base_costs = {"regular": 0.016, "premium": 0.063, "enterprise": 0.35}

        base_cost = base_costs.get(research_tier, 0.016)

        # Time-based adjustment (very rough approximation)
        time_factor = max(
            0.5, min(2.0, processing_time_ms / 60000)
        )  # 1 minute baseline

        return base_cost * time_factor


# Global instance
_user_research_storage: Optional[UserResearchStorage] = None


def get_user_research_storage(supabase_client: SupabaseClient) -> UserResearchStorage:
    """Get or create global UserResearchStorage instance"""
    global _user_research_storage

    if _user_research_storage is None:
        _user_research_storage = UserResearchStorage(supabase_client)

    return _user_research_storage
