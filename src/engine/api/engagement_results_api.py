"""
Engagement Results API - METIS V5 Final Assembly
Complete implementation of the Final API Payload retrieval system

This API provides the definitive endpoint for retrieving completed cognitive analyses:
- GET /api/engagements/{id}/results - Retrieve complete analysis dossier
- GET /api/engagements/{id}/status - Check engagement completion status
- GET /api/engagements/{id}/summary - Quick overview of results

Critical Feature: Assembles the complete "Final API Payload" from persisted
consultant analyses, Devil's Advocate critiques, and Senior Advisor arbitration.
"""

import logging
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Path
from pydantic import BaseModel, Field
import os
from supabase import create_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/engagements", tags=["Engagement Results"])


def get_supabase_client():
    """Get Supabase client instance"""
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_ANON_KEY")

    if not supabase_url or not supabase_key:
        raise HTTPException(status_code=500, detail="Supabase configuration missing")

    return create_client(supabase_url, supabase_key)


# Response Models
class EngagementStatus(BaseModel):
    engagement_id: str
    status: str = Field(
        ..., description="Status: pending, processing, completed, failed"
    )
    created_at: str
    updated_at: str
    progress_percentage: float = Field(..., description="Completion percentage 0-100")


class ConsultantResult(BaseModel):
    consultant_id: str
    analysis_output: str
    devils_advocate_critique: str
    research_audit_trail: Dict[str, Any]
    processing_time_ms: int
    confidence_score: float


class SeniorAdvisorReport(BaseModel):
    arbitration_report: str
    meta_analysis: str
    recommendation_synthesis: str
    confidence_assessment: float
    processing_metadata: Dict[str, Any]


class FinalAPIPayload(BaseModel):
    """The complete, structured Final API Payload"""

    engagement_id: str
    query: str
    status: str
    created_at: str
    completed_at: str

    # Core Analysis Results
    consultant_results: List[ConsultantResult]
    senior_advisor_report: SeniorAdvisorReport

    # Processing Metadata
    classification: Dict[str, Any]
    selected_consultants: List[str]
    processing_time_seconds: float

    # Glass-Box Transparency
    research_summary: Dict[str, Any]
    audit_trail_summary: Dict[str, Any]

    # Quality Metrics
    overall_confidence: float
    analysis_completeness: float


class EngagementSummary(BaseModel):
    engagement_id: str
    query: str
    status: str
    consultant_count: int
    analysis_length: int
    confidence_score: float
    created_at: str
    completed_at: Optional[str]


@router.get("/{engagement_id}/status", response_model=EngagementStatus)
async def get_engagement_status(
    engagement_id: str = Path(..., description="UUID of the engagement")
) -> EngagementStatus:
    """
    Check the completion status of an engagement

    Returns current processing status and completion percentage.
    """

    try:
        logger.info(f"📊 Checking status for engagement {engagement_id}")

        supabase = get_supabase_client()

        # Query engagement record
        engagement_result = (
            supabase.table("engagements").select("*").eq("id", engagement_id).execute()
        )

        if not engagement_result.data:
            raise HTTPException(
                status_code=404, detail=f"Engagement {engagement_id} not found"
            )

        engagement = engagement_result.data[0]

        # Check completion by querying results tables
        results_count = (
            supabase.table("engagement_results")
            .select("consultant_id")
            .eq("engagement_id", engagement_id)
            .execute()
        )
        senior_advisor_exists = (
            supabase.table("senior_advisor_reports")
            .select("id")
            .eq("engagement_id", engagement_id)
            .execute()
        )

        consultant_count = len(results_count.data) if results_count.data else 0
        has_senior_advisor = (
            len(senior_advisor_exists.data) > 0 if senior_advisor_exists.data else False
        )

        # Calculate progress percentage
        expected_consultants = 3  # Standard three-consultant analysis
        progress = 0.0

        if consultant_count > 0:
            progress += (
                consultant_count / expected_consultants
            ) * 80  # 80% for consultant analyses

        if has_senior_advisor:
            progress += 20  # 20% for senior advisor report

        progress = min(progress, 100.0)

        # Determine status
        if progress >= 100.0:
            status = "completed"
        elif progress > 0:
            status = "processing"
        else:
            status = "pending"

        return EngagementStatus(
            engagement_id=engagement_id,
            status=status,
            created_at=engagement["created_at"],
            updated_at=engagement["updated_at"],
            progress_percentage=progress,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Status check failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to check engagement status: {str(e)}"
        )


@router.get("/{engagement_id}/summary", response_model=EngagementSummary)
async def get_engagement_summary(
    engagement_id: str = Path(..., description="UUID of the engagement")
) -> EngagementSummary:
    """
    Get a quick summary of engagement results

    Returns high-level metrics without full content.
    """

    try:
        logger.info(f"📋 Getting summary for engagement {engagement_id}")

        supabase = get_supabase_client()

        # Get engagement basic info
        engagement_result = (
            supabase.table("engagements").select("*").eq("id", engagement_id).execute()
        )

        if not engagement_result.data:
            raise HTTPException(
                status_code=404, detail=f"Engagement {engagement_id} not found"
            )

        engagement = engagement_result.data[0]

        # Get consultant results summary
        results = (
            supabase.table("engagement_results")
            .select("consultant_id, confidence_score, analysis_output")
            .eq("engagement_id", engagement_id)
            .execute()
        )

        # Calculate metrics
        consultant_count = len(results.data) if results.data else 0
        total_analysis_length = (
            sum(len(r["analysis_output"]) for r in results.data) if results.data else 0
        )
        avg_confidence = (
            sum(r["confidence_score"] for r in results.data) / len(results.data)
            if results.data
            else 0.0
        )

        # Determine completion status
        senior_advisor_exists = (
            supabase.table("senior_advisor_reports")
            .select("created_at")
            .eq("engagement_id", engagement_id)
            .execute()
        )

        status = (
            "completed"
            if consultant_count >= 3 and senior_advisor_exists.data
            else "processing"
        )
        completed_at = (
            senior_advisor_exists.data[0]["created_at"]
            if senior_advisor_exists.data
            else None
        )

        return EngagementSummary(
            engagement_id=engagement_id,
            query=engagement["query"],
            status=status,
            consultant_count=consultant_count,
            analysis_length=total_analysis_length,
            confidence_score=avg_confidence,
            created_at=engagement["created_at"],
            completed_at=completed_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Summary generation failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate engagement summary: {str(e)}"
        )


@router.get("/{engagement_id}/results", response_model=FinalAPIPayload)
async def get_engagement_results(
    engagement_id: str = Path(..., description="UUID of the engagement")
) -> FinalAPIPayload:
    """
    Retrieve the complete analysis dossier for a completed engagement

    This is the definitive endpoint that assembles the complete "Final API Payload"
    from all persisted components: consultant analyses, critiques, and arbitration.

    Returns the full, rich dossier of cognitive analysis results.
    """

    try:
        logger.info(f"🎯 Assembling complete results for engagement {engagement_id}")

        supabase = get_supabase_client()

        # 1. Get engagement basic information
        engagement_result = (
            supabase.table("engagements").select("*").eq("id", engagement_id).execute()
        )

        if not engagement_result.data:
            raise HTTPException(
                status_code=404, detail=f"Engagement {engagement_id} not found"
            )

        engagement = engagement_result.data[0]

        # 2. Get all consultant results
        consultant_results_raw = (
            supabase.table("engagement_results")
            .select("*")
            .eq("engagement_id", engagement_id)
            .execute()
        )

        if not consultant_results_raw.data:
            raise HTTPException(
                status_code=404,
                detail=f"No analysis results found for engagement {engagement_id}",
            )

        # 3. Get senior advisor report
        senior_advisor_raw = (
            supabase.table("senior_advisor_reports")
            .select("*")
            .eq("engagement_id", engagement_id)
            .execute()
        )

        if not senior_advisor_raw.data:
            raise HTTPException(
                status_code=404,
                detail=f"No senior advisor report found for engagement {engagement_id}",
            )

        # 4. Assemble consultant results
        consultant_results = []
        total_research_queries = 0
        total_processing_time = 0
        confidence_scores = []

        for result in consultant_results_raw.data:
            consultant_result = ConsultantResult(
                consultant_id=result["consultant_id"],
                analysis_output=result["analysis_output"],
                devils_advocate_critique=result["devils_advocate_critique"],
                research_audit_trail=result["research_audit_trail"] or {},
                processing_time_ms=result["processing_time_ms"],
                confidence_score=result["confidence_score"],
            )
            consultant_results.append(consultant_result)

            # Aggregate metrics
            if result["research_audit_trail"]:
                total_research_queries += result["research_audit_trail"].get(
                    "queries_executed", 0
                )
            total_processing_time += result["processing_time_ms"]
            confidence_scores.append(result["confidence_score"])

        # 5. Assemble senior advisor report
        senior_data = senior_advisor_raw.data[0]
        senior_advisor_report = SeniorAdvisorReport(
            arbitration_report=senior_data["arbitration_report"],
            meta_analysis=senior_data["meta_analysis"],
            recommendation_synthesis=senior_data["recommendation_synthesis"],
            confidence_assessment=senior_data["confidence_assessment"],
            processing_metadata=senior_data["processing_metadata"] or {},
        )

        # 6. Calculate aggregate metrics
        overall_confidence = (
            sum(confidence_scores) / len(confidence_scores)
            if confidence_scores
            else 0.0
        )

        # Analysis completeness based on content length and depth
        total_analysis_length = sum(len(r.analysis_output) for r in consultant_results)
        total_critique_length = sum(
            len(r.devils_advocate_critique) for r in consultant_results
        )
        senior_advisor_length = len(senior_advisor_report.arbitration_report)

        # Completeness score (normalized)
        expected_min_length = 5000  # Minimum expected total content length
        actual_length = (
            total_analysis_length + total_critique_length + senior_advisor_length
        )
        analysis_completeness = min(actual_length / expected_min_length, 1.0)

        # 7. Build research and audit summaries
        research_summary = {
            "total_research_queries": total_research_queries,
            "consultants_with_research": len(
                [
                    r
                    for r in consultant_results
                    if r.research_audit_trail.get("queries_executed", 0) > 0
                ]
            ),
            "research_depth_score": (
                total_research_queries / len(consultant_results)
                if consultant_results
                else 0
            ),
        }

        audit_trail_summary = {
            "total_processing_time_ms": total_processing_time,
            "consultants_analyzed": len(consultant_results),
            "critiques_completed": len(
                [r for r in consultant_results if r.devils_advocate_critique]
            ),
            "senior_advisor_completed": True,
            "glass_box_transparency": "full",
        }

        # 8. Assemble the complete Final API Payload
        final_payload = FinalAPIPayload(
            engagement_id=engagement_id,
            query=engagement["query"],
            status=engagement["status"],
            created_at=engagement["created_at"],
            completed_at=senior_data["created_at"],
            consultant_results=consultant_results,
            senior_advisor_report=senior_advisor_report,
            classification=(
                engagement["metadata"].get("classification", {})
                if engagement["metadata"]
                else {}
            ),
            selected_consultants=(
                engagement["metadata"].get("selected_consultants", [])
                if engagement["metadata"]
                else []
            ),
            processing_time_seconds=(
                engagement["metadata"].get("processing_time_seconds", 0)
                if engagement["metadata"]
                else 0
            ),
            research_summary=research_summary,
            audit_trail_summary=audit_trail_summary,
            overall_confidence=overall_confidence,
            analysis_completeness=analysis_completeness,
        )

        logger.info(
            f"✅ Complete Final API Payload assembled for engagement {engagement_id}"
        )
        logger.info(
            f"📊 Metrics: {len(consultant_results)} consultants, {overall_confidence:.2f} confidence, {analysis_completeness:.2f} completeness"
        )

        return final_payload

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Results assembly failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to assemble engagement results: {str(e)}"
        )
