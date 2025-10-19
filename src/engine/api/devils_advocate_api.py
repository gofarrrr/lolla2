#!/usr/bin/env python3
"""
Devil's Advocate API - Phase 2 Quality Control
REST API endpoint for challenging consultant analyses with systematic bias detection

This API provides:
1. POST /challenge-analysis - Challenge individual consultant analysis
2. POST /comprehensive-critique - Multi-engine critique system
3. GET /critique-results/{engagement_id} - Retrieve stored critique results
4. POST /bulk-challenge - Challenge multiple consultant outputs

Integrates with the Enhanced Devils Advocate System for:
- Munger Bias Detection
- Ackoff Assumption Dissolution
- Cognitive Audit Engine
- Research-grounded validation
"""

import asyncio
import time
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import logging

# Core METIS imports
from src.core.enhanced_devils_advocate_system import (
    EnhancedDevilsAdvocateSystem,
)

# Supabase integration
import os
from supabase import create_client


def get_supabase_client():
    """Get Supabase client instance"""
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_ANON_KEY")

    if not supabase_url or not supabase_key:
        raise HTTPException(status_code=500, detail="Supabase configuration missing")

    return create_client(supabase_url, supabase_key)


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/devils-advocate", tags=["Quality Control"])


# Root OPTIONS for contract guardian
@router.options("/")
async def devils_advocate_root_options():
    return {"status": "ok"}

# Request/Response Models
class ConsultantAnalysisInput(BaseModel):
    consultant_id: str = Field(
        ..., description="ID of consultant who generated analysis"
    )
    analysis_output: str = Field(
        ..., min_length=50, description="Consultant's analysis to challenge"
    )
    business_context: Dict[str, Any] = Field(
        default_factory=dict, description="Business context for analysis"
    )
    engagement_id: Optional[str] = Field(
        default=None, description="Associated engagement ID"
    )


class ChallengeAnalysisRequest(BaseModel):
    analysis: ConsultantAnalysisInput
    challenge_intensity: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Intensity of challenge (0.0-1.0)"
    )
    enable_research_grounding: bool = Field(
        default=True, description="Enable Perplexity research validation"
    )


class BulkChallengeRequest(BaseModel):
    engagement_id: str = Field(..., description="Engagement ID for bulk challenge")
    consultant_analyses: List[ConsultantAnalysisInput] = Field(
        ..., description="List of consultant analyses to challenge"
    )
    challenge_intensity: float = Field(default=0.8, ge=0.0, le=1.0)
    enable_research_grounding: bool = Field(default=True)


class ChallengeResult(BaseModel):
    challenge_id: str
    challenge_type: str
    challenge_text: str
    severity: float
    evidence: List[str]
    mitigation_strategy: str
    source_engine: str


class AnalysisCritique(BaseModel):
    consultant_id: str
    original_analysis: str
    total_challenges_found: int
    critical_challenges: List[ChallengeResult]
    overall_risk_score: float
    refined_recommendation: str
    intellectual_honesty_score: float
    system_confidence: float
    processing_time_ms: int


class ChallengeAnalysisResponse(BaseModel):
    success: bool
    engagement_id: Optional[str]
    critique: AnalysisCritique
    persistence_status: str


class BulkChallengeResponse(BaseModel):
    success: bool
    engagement_id: str
    critiques: List[AnalysisCritique]
    summary_stats: Dict[str, Any]
    total_processing_time_ms: int
    persistence_status: str


# Global instance
_devils_advocate_system: Optional[EnhancedDevilsAdvocateSystem] = None


def get_devils_advocate_system() -> EnhancedDevilsAdvocateSystem:
    """Get or create Devil's Advocate system instance"""
    global _devils_advocate_system
    if _devils_advocate_system is None:
        _devils_advocate_system = EnhancedDevilsAdvocateSystem()
    return _devils_advocate_system


@router.post("/challenge-analysis", response_model=ChallengeAnalysisResponse)
async def challenge_single_analysis(request: ChallengeAnalysisRequest):
    """
    Challenge a single consultant analysis using the Enhanced Devil's Advocate System

    Uses three specialized engines:
    1. Munger Bias Detector - Identifies cognitive biases
    2. Ackoff Assumption Dissolver - Challenges fundamental assumptions
    3. Cognitive Audit Engine - Detects motivated reasoning
    """

    start_time = time.time()

    try:
        logger.info(f"🔥 Challenging analysis from {request.analysis.consultant_id}")

        # Get Devil's Advocate system
        devils_advocate = get_devils_advocate_system()

        # Configure system based on request
        devils_advocate.enable_research_grounding = request.enable_research_grounding
        devils_advocate.severity_threshold = (
            1.0 - request.challenge_intensity
        )  # Higher intensity = lower threshold

        # Run comprehensive challenge analysis
        challenge_result = await devils_advocate.comprehensive_challenge_analysis(
            recommendation=request.analysis.analysis_output,
            business_context=request.analysis.business_context,
        )

        # Convert to API format
        critical_challenges = [
            ChallengeResult(
                challenge_id=challenge.challenge_id,
                challenge_type=challenge.challenge_type,
                challenge_text=challenge.challenge_text,
                severity=challenge.severity,
                evidence=challenge.evidence,
                mitigation_strategy=challenge.mitigation_strategy,
                source_engine=challenge.source_engine,
            )
            for challenge in challenge_result.critical_challenges
        ]

        processing_time = int((time.time() - start_time) * 1000)

        critique = AnalysisCritique(
            consultant_id=request.analysis.consultant_id,
            original_analysis=request.analysis.analysis_output,
            total_challenges_found=challenge_result.total_challenges_found,
            critical_challenges=critical_challenges,
            overall_risk_score=challenge_result.overall_risk_score,
            refined_recommendation=challenge_result.refined_recommendation,
            intellectual_honesty_score=challenge_result.intellectual_honesty_score,
            system_confidence=challenge_result.system_confidence,
            processing_time_ms=processing_time,
        )

        # Persist critique results
        persistence_status = await _persist_critique_results(
            request.analysis.engagement_id, request.analysis.consultant_id, critique
        )

        logger.info(
            f"✅ Challenge analysis complete: {challenge_result.total_challenges_found} challenges found"
        )

        return ChallengeAnalysisResponse(
            success=True,
            engagement_id=request.analysis.engagement_id,
            critique=critique,
            persistence_status=persistence_status,
        )

    except Exception as e:
        logger.error(f"❌ Challenge analysis failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to challenge analysis: {str(e)}"
        )


# Alias endpoint to satisfy API Contract Guardian: /critique
@router.post("/critique", response_model=ChallengeAnalysisResponse)
async def critique_alias(request: ChallengeAnalysisRequest):
    return await challenge_single_analysis(request)

@router.post("/bulk-challenge", response_model=BulkChallengeResponse)
async def challenge_multiple_analyses(request: BulkChallengeRequest):
    """
    Challenge multiple consultant analyses in parallel

    This is the key integration point for the complete METIS workflow:
    Analysis Execution → Devil's Advocate Critique → Senior Advisor Arbitration
    """

    start_time = time.time()

    try:
        logger.info(
            f"🔥 Bulk challenging {len(request.consultant_analyses)} analyses for {request.engagement_id}"
        )

        # Create challenge requests for each analysis
        challenge_tasks = []
        for analysis in request.consultant_analyses:
            challenge_request = ChallengeAnalysisRequest(
                analysis=analysis,
                challenge_intensity=request.challenge_intensity,
                enable_research_grounding=request.enable_research_grounding,
            )
            challenge_tasks.append(
                _challenge_single_analysis_internal(challenge_request)
            )

        # Execute all challenges in parallel
        critiques = await asyncio.gather(*challenge_tasks)

        # Calculate summary statistics
        total_challenges = sum(c.total_challenges_found for c in critiques)
        avg_risk_score = sum(c.overall_risk_score for c in critiques) / len(critiques)
        avg_honesty_score = sum(c.intellectual_honesty_score for c in critiques) / len(
            critiques
        )

        summary_stats = {
            "total_consultants_challenged": len(critiques),
            "total_challenges_found": total_challenges,
            "average_risk_score": avg_risk_score,
            "average_intellectual_honesty_score": avg_honesty_score,
            "challenges_per_consultant": total_challenges / len(critiques),
            "high_risk_analyses": len(
                [c for c in critiques if c.overall_risk_score > 0.7]
            ),
        }

        # Persist all critique results
        persistence_status = await _persist_bulk_critiques(
            request.engagement_id, critiques
        )

        total_processing_time = int((time.time() - start_time) * 1000)

        logger.info(
            f"✅ Bulk challenge complete: {total_challenges} total challenges across {len(critiques)} analyses"
        )

        return BulkChallengeResponse(
            success=True,
            engagement_id=request.engagement_id,
            critiques=critiques,
            summary_stats=summary_stats,
            total_processing_time_ms=total_processing_time,
            persistence_status=persistence_status,
        )

    except Exception as e:
        logger.error(f"❌ Bulk challenge failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to challenge analyses: {str(e)}"
        )


@router.get("/critique-results/{engagement_id}")
async def get_critique_results(engagement_id: str):
    """
    Retrieve stored critique results for an engagement

    Returns all Devil's Advocate critiques for the given engagement.
    """

    try:
        logger.info(f"📊 Retrieving critique results for engagement {engagement_id}")

        supabase = get_supabase_client()

        # Query engagement_results table for critique data
        result = (
            supabase.table("engagement_results")
            .select("*")
            .eq("engagement_id", engagement_id)
            .execute()
        )

        if not result.data:
            raise HTTPException(
                status_code=404,
                detail=f"No critique results found for engagement {engagement_id}",
            )

        # Structure the response
        critique_results = []
        for row in result.data:
            if row.get("devils_advocate_critique"):
                critique_results.append(
                    {
                        "consultant_id": row["consultant_id"],
                        "analysis_output": row["analysis_output"],
                        "devils_advocate_critique": row["devils_advocate_critique"],
                        "confidence_score": row["confidence_score"],
                        "processing_time_ms": row["processing_time_ms"],
                        "created_at": row["created_at"],
                    }
                )

        return {
            "success": True,
            "engagement_id": engagement_id,
            "critique_results": critique_results,
            "total_critiques": len(critique_results),
        }

    except Exception as e:
        logger.error(f"❌ Failed to retrieve critique results: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve critique results: {str(e)}"
        )


@router.get("/status")
async def get_devils_advocate_status():
    """Get status of the Devil's Advocate API"""

    try:
        devils_advocate = get_devils_advocate_system()

        return {
            "success": True,
            "api_endpoints": [
                "/challenge-analysis",
                "/bulk-challenge",
                "/critique-results/{engagement_id}",
                "/status",
            ],
            "system_components": {
                "munger_bias_detector": "Active",
                "ackoff_assumption_dissolver": "Active",
                "cognitive_audit_engine": "Active",
                "research_grounding": devils_advocate.enable_research_grounding,
            },
            "configuration": {
                "severity_threshold": devils_advocate.severity_threshold,
                "max_challenges_per_engine": devils_advocate.max_challenges_per_engine,
            },
            "phase_2_features": [
                "Challenge individual consultant analysis",
                "Bulk challenge multiple analyses in parallel",
                "Munger bias detection with lollapalooza effects",
                "Ackoff assumption dissolution",
                "Cognitive audit for motivated reasoning",
                "Research-grounded validation via Perplexity",
                "Intellectual honesty scoring",
                "Risk assessment and mitigation strategies",
            ],
        }

    except Exception as e:
        logger.error(f"❌ Status check failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get Devil's Advocate status: {str(e)}"
        )


# Helper functions
async def _challenge_single_analysis_internal(
    request: ChallengeAnalysisRequest,
) -> AnalysisCritique:
    """Internal helper for challenging a single analysis (used in bulk operations)"""

    start_time = time.time()

    devils_advocate = get_devils_advocate_system()
    devils_advocate.enable_research_grounding = request.enable_research_grounding
    devils_advocate.severity_threshold = 1.0 - request.challenge_intensity

    challenge_result = await devils_advocate.comprehensive_challenge_analysis(
        recommendation=request.analysis.analysis_output,
        business_context=request.analysis.business_context,
    )

    critical_challenges = [
        ChallengeResult(
            challenge_id=challenge.challenge_id,
            challenge_type=challenge.challenge_type,
            challenge_text=challenge.challenge_text,
            severity=challenge.severity,
            evidence=challenge.evidence,
            mitigation_strategy=challenge.mitigation_strategy,
            source_engine=challenge.source_engine,
        )
        for challenge in challenge_result.critical_challenges
    ]

    processing_time = int((time.time() - start_time) * 1000)

    return AnalysisCritique(
        consultant_id=request.analysis.consultant_id,
        original_analysis=request.analysis.analysis_output,
        total_challenges_found=challenge_result.total_challenges_found,
        critical_challenges=critical_challenges,
        overall_risk_score=challenge_result.overall_risk_score,
        refined_recommendation=challenge_result.refined_recommendation,
        intellectual_honesty_score=challenge_result.intellectual_honesty_score,
        system_confidence=challenge_result.system_confidence,
        processing_time_ms=processing_time,
    )


async def _persist_critique_results(
    engagement_id: Optional[str], consultant_id: str, critique: AnalysisCritique
) -> str:
    """Persist critique results to database"""

    try:
        if not engagement_id:
            return "SKIPPED: No engagement ID provided"

        # Convert engagement_id to UUID format if it's not already
        try:
            from uuid import UUID

            uuid_engagement_id = UUID(engagement_id)
        except ValueError:
            # If not a valid UUID, generate a deterministic one from the string
            from uuid import uuid5, NAMESPACE_DNS

            uuid_engagement_id = uuid5(NAMESPACE_DNS, engagement_id)
            logger.info(
                f"🔄 Converted engagement_id '{engagement_id}' to UUID: {uuid_engagement_id}"
            )

        supabase = get_supabase_client()

        # Update engagement_results table with critique data
        critique_data = {
            "total_challenges_found": critique.total_challenges_found,
            "critical_challenges": [c.dict() for c in critique.critical_challenges],
            "overall_risk_score": critique.overall_risk_score,
            "refined_recommendation": critique.refined_recommendation,
            "intellectual_honesty_score": critique.intellectual_honesty_score,
            "system_confidence": critique.system_confidence,
        }

        # Try to update existing record first
        update_result = (
            supabase.table("engagement_results")
            .update({"devils_advocate_critique": critique_data})
            .eq("engagement_id", str(uuid_engagement_id))
            .eq("consultant_id", consultant_id)
            .execute()
        )

        if update_result.data:
            logger.info(f"💾 Updated critique for {consultant_id} in {engagement_id}")
            return f"SUCCESS: Updated critique for {consultant_id}"
        else:
            # If no existing record, store in engagements metadata
            supabase.table("engagements").update(
                {
                    "metadata": {
                        "devils_advocate_critiques": {consultant_id: critique_data}
                    }
                }
            ).eq("id", str(uuid_engagement_id)).execute()

            logger.info(
                f"💾 Stored critique in engagements metadata for {consultant_id}"
            )
            return f"SUCCESS: Stored critique for {consultant_id} in metadata"

    except Exception as e:
        logger.error(f"❌ Failed to persist critique: {e}")
        return f"FAILED: {str(e)}"


async def _persist_bulk_critiques(
    engagement_id: str, critiques: List[AnalysisCritique]
) -> str:
    """Persist multiple critique results"""

    try:
        results = []
        for critique in critiques:
            result = await _persist_critique_results(
                engagement_id, critique.consultant_id, critique
            )
            results.append(result)

        success_count = len([r for r in results if "SUCCESS" in r])
        return f"SUCCESS: {success_count}/{len(critiques)} critiques persisted"

    except Exception as e:
        logger.error(f"❌ Failed to persist bulk critiques: {e}")
        return f"FAILED: {str(e)}"
