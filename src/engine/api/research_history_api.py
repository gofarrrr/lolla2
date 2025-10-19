#!/usr/bin/env python3
"""
Research History API
FastAPI endpoints for user research history and results management
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel

# Supabase integration
from supabase import create_client, Client as SupabaseClient
from src.persistence.user_research_storage import (
    get_user_research_storage,
)

# Configuration
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize router
router = APIRouter(prefix="/api/research", tags=["research-history"])
logger = logging.getLogger(__name__)


# Pydantic models for API
class ResearchHistoryItem(BaseModel):
    """Research history item for API response"""

    id: str
    research_tier: str
    question_depth: str
    problem_statement: str
    executive_summary: str
    overall_confidence: float
    total_cost_usd: float
    source_count: int
    insight_count: int
    processing_time_ms: int
    created_at: datetime


class ResearchHistoryResponse(BaseModel):
    """Research history API response"""

    results: List[ResearchHistoryItem]
    total_count: int
    page: int
    page_size: int
    has_more: bool


class ResearchSource(BaseModel):
    """Research source detail"""

    id: str
    url: str
    title: Optional[str]
    domain: str
    content_preview: Optional[str]
    credibility_tier: str
    credibility_score: float
    fact_density: float
    citation_quality: float
    display_order: int


class ResearchInsight(BaseModel):
    """Research insight detail"""

    id: str
    claim: str
    confidence: float
    evidence_strength: float
    fact_type: str
    verification_status: str
    business_impact: str
    insight_category: str
    display_order: int


class ResearchResultDetail(BaseModel):
    """Detailed research result with sources and insights"""

    id: str
    research_tier: str
    question_depth: str
    problem_statement: str
    progressive_questions: Dict[str, Any]
    context_data: Dict[str, Any]

    # Results
    executive_summary: str
    key_insights: List[Any]
    detailed_findings: Optional[str]
    recommendations: List[Any]

    # Quality metrics
    overall_confidence: float
    coverage_completeness: float
    source_diversity_score: float
    fact_validation_score: float

    # Processing info
    total_cost_usd: float
    processing_time_ms: int
    perplexity_model: str
    queries_executed: int

    # Related data
    sources: List[ResearchSource]
    insights: List[ResearchInsight]

    # Timestamps
    created_at: datetime
    updated_at: datetime


class UserAccessInfo(BaseModel):
    """User research access information"""

    subscription_tier: str
    monthly_research_limit: int
    current_month_usage: int
    research_credits_remaining: int
    total_research_requests: int
    total_cost_usd: float
    can_access_regular: bool
    can_access_premium: bool
    can_access_enterprise: bool


# Dependency to get Supabase client
def get_supabase_client() -> SupabaseClient:
    """Get Supabase client instance"""
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_ANON_KEY")

    if not supabase_url or not supabase_key:
        raise HTTPException(status_code=500, detail="Supabase configuration missing")

    return create_client(supabase_url, supabase_key)


# Dependency to get user ID (mock for now - integrate with auth later)
def get_current_user_id() -> str:
    """Get current user ID from auth token"""
    # TODO: Integrate with real authentication system
    return "00000000-0000-0000-0000-000000000002"  # Mock premium user


@router.get("/history", response_model=ResearchHistoryResponse)
async def get_research_history(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Results per page"),
    supabase: SupabaseClient = Depends(get_supabase_client),
    user_id: str = Depends(get_current_user_id),
):
    """Get user's research history with pagination"""

    try:
        storage = get_user_research_storage(supabase)

        offset = (page - 1) * page_size
        results = await storage.get_user_research_history(
            user_id=user_id,
            limit=page_size + 1,  # Get one extra to check if there's more
            offset=offset,
        )

        has_more = len(results) > page_size
        if has_more:
            results = results[:page_size]

        # Convert to API models
        history_items = [
            ResearchHistoryItem(
                id=result.id,
                research_tier=result.research_tier,
                question_depth=result.question_depth,
                problem_statement=result.problem_statement,
                executive_summary=result.executive_summary or "",
                overall_confidence=result.overall_confidence,
                total_cost_usd=result.total_cost_usd,
                source_count=result.source_count,
                insight_count=result.insight_count,
                processing_time_ms=result.processing_time_ms,
                created_at=result.created_at,
            )
            for result in results
        ]

        return ResearchHistoryResponse(
            results=history_items,
            total_count=len(history_items),  # Approximate
            page=page,
            page_size=page_size,
            has_more=has_more,
        )

    except Exception as e:
        logger.error(f"❌ Failed to get research history: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve research history"
        )


@router.get("/result/{result_id}", response_model=ResearchResultDetail)
async def get_research_result_detail(
    result_id: str,
    supabase: SupabaseClient = Depends(get_supabase_client),
    user_id: str = Depends(get_current_user_id),
):
    """Get detailed research result with sources and insights"""

    try:
        storage = get_user_research_storage(supabase)

        result_data = await storage.get_research_result_detail(
            user_id=user_id, research_result_id=result_id
        )

        if not result_data:
            raise HTTPException(status_code=404, detail="Research result not found")

        # Convert sources
        sources = [
            ResearchSource(
                id=source["id"],
                url=source["url"],
                title=source.get("title"),
                domain=source["domain"],
                content_preview=source.get("content_preview"),
                credibility_tier=source["credibility_tier"],
                credibility_score=source["credibility_score"],
                fact_density=source["fact_density"],
                citation_quality=source["citation_quality"],
                display_order=source["display_order"],
            )
            for source in result_data.get("sources", [])
        ]

        # Convert insights
        insights = [
            ResearchInsight(
                id=insight["id"],
                claim=insight["claim"],
                confidence=insight["confidence"],
                evidence_strength=insight["evidence_strength"],
                fact_type=insight["fact_type"],
                verification_status=insight["verification_status"],
                business_impact=insight["business_impact"],
                insight_category=insight["insight_category"],
                display_order=insight["display_order"],
            )
            for insight in result_data.get("insights", [])
        ]

        return ResearchResultDetail(
            id=result_data["id"],
            research_tier=result_data["research_tier"],
            question_depth=result_data["question_depth"],
            problem_statement=result_data["problem_statement"],
            progressive_questions=result_data.get("progressive_questions", {}),
            context_data=result_data.get("context_data", {}),
            # Results
            executive_summary=result_data.get("executive_summary", ""),
            key_insights=result_data.get("key_insights", []),
            detailed_findings=result_data.get("detailed_findings"),
            recommendations=result_data.get("recommendations", []),
            # Quality metrics
            overall_confidence=result_data["overall_confidence"],
            coverage_completeness=result_data.get("coverage_completeness", 0.0),
            source_diversity_score=result_data.get("source_diversity_score", 0.0),
            fact_validation_score=result_data.get("fact_validation_score", 0.0),
            # Processing info
            total_cost_usd=result_data["total_cost_usd"],
            processing_time_ms=result_data.get("processing_time_ms", 0),
            perplexity_model=result_data.get("perplexity_model", "sonar-pro"),
            queries_executed=result_data.get("queries_executed", 0),
            # Related data
            sources=sources,
            insights=insights,
            # Timestamps
            created_at=datetime.fromisoformat(
                result_data["created_at"].replace("Z", "+00:00")
            ),
            updated_at=datetime.fromisoformat(
                result_data["updated_at"].replace("Z", "+00:00")
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get research result detail: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve research result detail"
        )


@router.get("/access-info", response_model=UserAccessInfo)
async def get_user_access_info(
    supabase: SupabaseClient = Depends(get_supabase_client),
    user_id: str = Depends(get_current_user_id),
):
    """Get user's research access information and limits"""

    try:
        storage = get_user_research_storage(supabase)

        # Get user access record
        access_response = (
            supabase.table("user_research_access")
            .select("*")
            .eq("user_id", user_id)
            .execute()
        )

        if access_response.data:
            access_data = access_response.data[0]
        else:
            # Default for new users
            access_data = {
                "subscription_tier": "free",
                "monthly_research_limit": 10,
                "current_month_usage": 0,
                "research_credits_remaining": 0,
                "total_research_requests": 0,
                "total_cost_usd": 0.0,
            }

        # Check access for each tier
        can_access_regular = await storage.check_user_research_access(
            user_id, "regular"
        )
        can_access_premium = await storage.check_user_research_access(
            user_id, "premium"
        )
        can_access_enterprise = await storage.check_user_research_access(
            user_id, "enterprise"
        )

        return UserAccessInfo(
            subscription_tier=access_data["subscription_tier"],
            monthly_research_limit=access_data["monthly_research_limit"],
            current_month_usage=access_data["current_month_usage"],
            research_credits_remaining=access_data["research_credits_remaining"],
            total_research_requests=access_data["total_research_requests"],
            total_cost_usd=access_data["total_cost_usd"],
            can_access_regular=can_access_regular,
            can_access_premium=can_access_premium,
            can_access_enterprise=can_access_enterprise,
        )

    except Exception as e:
        logger.error(f"❌ Failed to get user access info: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve access information"
        )


@router.get("/stats")
async def get_research_stats(
    supabase: SupabaseClient = Depends(get_supabase_client),
    user_id: str = Depends(get_current_user_id),
):
    """Get user's research usage statistics"""

    try:
        # Get basic stats from user_research_access
        access_response = (
            supabase.table("user_research_access")
            .select("*")
            .eq("user_id", user_id)
            .execute()
        )

        if not access_response.data:
            return {
                "total_research_requests": 0,
                "total_cost_usd": 0.0,
                "tier_usage": {},
                "recent_activity": 0,
            }

        access_data = access_response.data[0]

        # Get tier usage distribution
        tier_usage_response = (
            supabase.table("user_research_results")
            .select("research_tier")
            .eq("user_id", user_id)
            .execute()
        )

        tier_counts = {}
        for result in tier_usage_response.data:
            tier = result["research_tier"]
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

        # Get recent activity (last 30 days)
        from datetime import datetime, timedelta, timezone

        thirty_days_ago = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()

        recent_response = (
            supabase.table("user_research_results")
            .select("id")
            .eq("user_id", user_id)
            .gte("created_at", thirty_days_ago)
            .execute()
        )

        return {
            "total_research_requests": access_data["total_research_requests"],
            "total_cost_usd": access_data["total_cost_usd"],
            "tier_usage": tier_counts,
            "recent_activity": len(recent_response.data),
            "subscription_tier": access_data["subscription_tier"],
            "current_month_usage": access_data["current_month_usage"],
            "monthly_limit": access_data["monthly_research_limit"],
        }

    except Exception as e:
        logger.error(f"❌ Failed to get research stats: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve research statistics"
        )


# Health check endpoint
@router.get("/health")
async def research_api_health():
    """Research API health check"""
    return {
        "status": "healthy",
        "service": "research-history-api",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "features": [
            "research_history",
            "result_details",
            "access_control",
            "usage_statistics",
        ],
    }
