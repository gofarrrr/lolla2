from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, Path
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging

from ..services.gamma import (
    GammaPresentationService,
    PresentationType,
    ValidationError,
    RateLimitError,
    AuthenticationError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/presentations", tags=["presentations"])


# Request/Response models
class PresentationRequest(BaseModel):
    """Request model for presentation generation"""

    analysis_id: str = Field(..., description="METIS analysis ID")
    presentation_type: str = Field(
        default="strategy", description="Type of presentation to generate"
    )
    export_formats: List[str] = Field(
        default=["pdf"], description="Export formats: pdf, pptx"
    )
    custom_instructions: Optional[str] = Field(
        None, description="Additional generation instructions"
    )
    theme_override: Optional[str] = Field(None, description="Override default theme")


class PresentationResponse(BaseModel):
    """Response model for presentation generation"""

    generation_id: str
    status: str
    presentation_type: str
    urls: Dict[str, str]
    timestamp: str
    analysis_id: Optional[str] = None
    message: Optional[str] = None
    error: Optional[str] = None
    errors: Optional[List[Dict[str, str]]] = None
    theme_used: Optional[str] = None
    card_count: Optional[int] = None


class BatchPresentationRequest(BaseModel):
    """Request model for batch presentation generation"""

    analysis_ids: List[str]
    presentation_type: str = "strategy"
    export_formats: List[str] = ["pdf"]


class RegenerationRequest(BaseModel):
    """Request model for regenerating with feedback"""

    feedback: str = Field(..., description="User feedback for regeneration")
    presentation_type: Optional[str] = Field(None, description="New presentation type")
    export_formats: Optional[List[str]] = Field(None, description="New export formats")


class TemplateInfo(BaseModel):
    """Template information model"""

    type: str
    name: str
    description: str
    theme: Optional[str] = None
    format: Optional[str] = None
    typical_cards: Optional[str] = None


# Service instance - will be initialized with proper config
presentation_service = GammaPresentationService()


@router.post("/generate", response_model=PresentationResponse)
async def generate_presentation(
    request: PresentationRequest, background_tasks: BackgroundTasks
) -> PresentationResponse:
    """
    Generate presentation from METIS analysis

    Creates professional presentations using Gamma AI from cognitive analysis outputs.
    Supports multiple export formats and customization options.
    """
    try:
        logger.info(f"🎯 Presentation requested for analysis: {request.analysis_id}")

        # Retrieve analysis result from METIS storage
        analysis_result = await get_analysis_result(request.analysis_id)

        if not analysis_result:
            raise HTTPException(
                status_code=404, detail=f"Analysis {request.analysis_id} not found"
            )

        # Validate presentation type
        try:
            pres_type = PresentationType(request.presentation_type)
        except ValueError:
            logger.warning(f"Invalid presentation type: {request.presentation_type}")
            pres_type = PresentationType.STRATEGY

        # Validate export formats
        valid_formats = ["pdf", "pptx"]
        export_formats = [f for f in request.export_formats if f in valid_formats]
        if not export_formats:
            export_formats = ["pdf"]  # Default fallback

        # Generate presentation
        result = await presentation_service.generate_from_analysis(
            analysis_result,
            pres_type,
            export_formats,
            request.custom_instructions,
            request.theme_override,
        )

        # Handle different result statuses
        if result["status"] == "error":
            raise HTTPException(
                status_code=500, detail=result.get("error", "Generation failed")
            )

        # Extract URLs from results
        urls = result.get("urls", {})

        response = PresentationResponse(
            generation_id=result["generation_id"],
            status=result["status"],
            presentation_type=request.presentation_type,
            urls=urls,
            timestamp=result["timestamp"],
            analysis_id=result.get("analysis_id"),
            message=result.get("message"),
            error=result.get("error"),
            errors=result.get("errors"),
            theme_used=result.get("theme_used"),
            card_count=result.get("card_count"),
        )

        # Schedule cleanup in background
        background_tasks.add_task(schedule_cleanup, result["generation_id"])

        logger.info(f"✅ Presentation generation completed: {result['generation_id']}")
        return response

    except HTTPException:
        raise
    except AuthenticationError as e:
        logger.error(f"❌ Authentication error: {e}")
        raise HTTPException(status_code=401, detail="Gamma API authentication failed")
    except RateLimitError as e:
        logger.error(f"❌ Rate limit error: {e}")
        raise HTTPException(
            status_code=429, detail="Rate limit exceeded. Please try again later."
        )
    except ValidationError as e:
        logger.error(f"❌ Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Presentation generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during presentation generation",
        )


@router.post("/generate-batch", response_model=List[PresentationResponse])
async def generate_batch_presentations(
    request: BatchPresentationRequest,
) -> List[PresentationResponse]:
    """Generate multiple presentations in batch"""

    logger.info(f"📚 Batch generation for {len(request.analysis_ids)} analyses")

    if len(request.analysis_ids) > 10:  # Reasonable batch size limit
        raise HTTPException(
            status_code=400, detail="Batch size too large (maximum 10 analyses)"
        )

    # Retrieve all analyses
    analyses = []
    missing_ids = []

    for analysis_id in request.analysis_ids:
        analysis = await get_analysis_result(analysis_id)
        if analysis:
            analyses.append(analysis)
        else:
            missing_ids.append(analysis_id)

    if not analyses:
        raise HTTPException(status_code=404, detail="No valid analyses found")

    if missing_ids:
        logger.warning(f"Missing analyses: {missing_ids}")

    # Generate presentations
    try:
        pres_type = PresentationType(request.presentation_type)
    except ValueError:
        pres_type = PresentationType.STRATEGY

    results = await presentation_service.generate_batch(analyses, pres_type)

    # Format responses
    responses = []
    for result in results:
        if isinstance(result, dict):
            urls = result.get("urls", {})

            responses.append(
                PresentationResponse(
                    generation_id=result.get("generation_id", "unknown"),
                    status=result.get("status", "error"),
                    presentation_type=request.presentation_type,
                    urls=urls,
                    timestamp=result.get("timestamp", datetime.now().isoformat()),
                    analysis_id=result.get("analysis_id"),
                    message=result.get("message"),
                    error=result.get("error"),
                    errors=result.get("errors"),
                )
            )

    successful = sum(1 for r in responses if r.status in ["success", "partial_success"])
    logger.info(
        f"✅ Batch generation complete: {successful}/{len(request.analysis_ids)} successful"
    )

    return responses


@router.get("/templates", response_model=List[TemplateInfo])
async def list_presentation_templates():
    """List available presentation templates"""

    try:
        templates = await presentation_service.list_templates()

        return [
            TemplateInfo(
                type=template["type"],
                name=template["name"],
                description=template["description"],
                theme=template.get("theme"),
                format=template.get("format"),
                typical_cards=str(template.get("typical_cards", "auto")),
            )
            for template in templates
        ]

    except Exception as e:
        logger.error(f"❌ Failed to list templates: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve templates")


@router.get("/{generation_id}")
async def get_presentation(
    generation_id: str = Path(..., description="Presentation generation ID")
):
    """Get presentation by generation ID"""

    try:
        presentation = await presentation_service.get_presentation_status(generation_id)

        if presentation.get("status") == "not_found":
            raise HTTPException(
                status_code=404, detail=f"Presentation {generation_id} not found"
            )

        return presentation

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get presentation {generation_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve presentation")


@router.post("/{generation_id}/regenerate", response_model=PresentationResponse)
async def regenerate_presentation(
    generation_id: str = Path(..., description="Presentation generation ID"),
    request: RegenerationRequest = None,
):
    """Regenerate presentation with user feedback"""

    try:
        pres_type = None
        if request.presentation_type:
            try:
                pres_type = PresentationType(request.presentation_type)
            except ValueError:
                logger.warning(
                    f"Invalid presentation type in regeneration: {request.presentation_type}"
                )

        result = await presentation_service.regenerate_with_feedback(
            generation_id, request.feedback, pres_type, request.export_formats
        )

        if result["status"] == "error":
            raise HTTPException(
                status_code=500, detail=result.get("error", "Regeneration failed")
            )

        return PresentationResponse(
            generation_id=result["generation_id"],
            status=result["status"],
            presentation_type=result.get("presentation_type", "unknown"),
            urls=result.get("urls", {}),
            timestamp=result["timestamp"],
            analysis_id=result.get("analysis_id"),
            message=result.get("message"),
            error=result.get("error"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to regenerate presentation: {e}")
        raise HTTPException(status_code=500, detail="Regeneration failed")


@router.get("/")
async def list_presentations(
    limit: int = Query(
        10, ge=1, le=50, description="Maximum number of presentations to return"
    ),
    offset: int = Query(0, ge=0, description="Number of presentations to skip"),
    status_filter: Optional[str] = Query(None, description="Filter by status"),
):
    """List generated presentations"""

    try:
        presentations = await presentation_service.storage.list_presentations(
            limit=limit, offset=offset, filter_status=status_filter
        )

        total_count = len(presentation_service.storage.metadata)

        return {
            "presentations": presentations,
            "total": total_count,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total_count,
        }

    except Exception as e:
        logger.error(f"❌ Failed to list presentations: {e}")
        raise HTTPException(status_code=500, detail="Failed to list presentations")


@router.get("/search")
async def search_presentations(
    query: str = Query(..., min_length=3, description="Search query"),
    limit: int = Query(10, ge=1, le=50, description="Maximum results to return"),
):
    """Search presentations by content"""

    try:
        results = await presentation_service.storage.search_presentations(
            query=query, limit=limit
        )

        return {"query": query, "results": results, "count": len(results)}

    except Exception as e:
        logger.error(f"❌ Search failed: {e}")
        raise HTTPException(status_code=500, detail="Search failed")


@router.get("/health")
async def health_check():
    """Health check for presentation service"""

    try:
        health = await presentation_service.health_check()

        status_code = 200
        if health["service_status"] == "degraded":
            status_code = 206  # Partial Content
        elif health["service_status"] == "unhealthy":
            status_code = 503  # Service Unavailable

        return health

    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return {
            "service_status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


@router.get("/usage/statistics")
async def get_usage_statistics():
    """Get presentation service usage statistics"""

    try:
        service_stats = presentation_service.get_usage_statistics()
        storage_stats = await presentation_service.storage.get_usage_statistics()

        return {
            "service": service_stats,
            "storage": storage_stats,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"❌ Failed to get usage statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get statistics")


@router.delete("/cleanup")
async def cleanup_old_presentations(
    days: int = Query(30, ge=1, le=365, description="Age in days for cleanup")
):
    """Clean up presentations older than specified days"""

    try:
        result = await presentation_service.cleanup_old_presentations(days)

        return {"cleanup_completed": True, "cutoff_days": days, **result}

    except Exception as e:
        logger.error(f"❌ Cleanup failed: {e}")
        raise HTTPException(status_code=500, detail="Cleanup operation failed")


# Helper functions
async def get_analysis_result(analysis_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve analysis result from METIS storage system"""
    try:
        # TODO: Integrate with actual METIS analysis storage
        # For now, return mock data for testing

        # In production, this would be:
        # from ..core.state_management import get_analysis_by_id
        # return await get_analysis_by_id(analysis_id)

        # Mock data for development/testing
        return {
            "analysis_id": analysis_id,
            "problem_statement": {
                "problem_description": "Strategic analysis for improving customer retention and market expansion in the competitive SaaS landscape.",
                "business_context": {
                    "industry": "SaaS Technology",
                    "company_stage": "Growth",
                    "timeline": "Q3-Q4 2024",
                },
            },
            "recommended_mental_models": [
                {
                    "name": "Jobs to Be Done Framework",
                    "description": "Focus on understanding what customers are trying to accomplish and why they hire your product",
                    "application": "Analyze customer churn patterns and identify unmet needs that lead to cancellations",
                    "benefits": [
                        "Customer-centric insights",
                        "Retention optimization",
                        "Product-market fit improvement",
                    ],
                    "considerations": [
                        "Requires deep customer research",
                        "May reveal uncomfortable truths about product gaps",
                    ],
                },
                {
                    "name": "Blue Ocean Strategy",
                    "description": "Create new market space by making competition irrelevant through value innovation",
                    "application": "Identify unique value propositions that differentiate from competitors while reducing costs",
                    "benefits": [
                        "Market differentiation",
                        "Pricing power",
                        "Reduced competition",
                    ],
                    "considerations": [
                        "Requires significant innovation investment",
                        "Market education may be needed",
                    ],
                },
                {
                    "name": "Customer Lifetime Value Optimization",
                    "description": "Focus on maximizing the total value derived from each customer relationship",
                    "application": "Optimize onboarding, engagement, and retention strategies to increase customer lifetime value",
                    "benefits": [
                        "Revenue optimization",
                        "Retention focus",
                        "Resource allocation efficiency",
                    ],
                    "considerations": [
                        "Complex metrics tracking",
                        "Long-term commitment required",
                    ],
                },
            ],
            "cognitive_framework": {
                "executive_summary": "The analysis reveals three critical strategic opportunities: deepening customer understanding through Jobs-to-be-Done research, creating differentiated market positioning via Blue Ocean principles, and optimizing customer lifetime value through enhanced retention strategies. These approaches work synergistically to address both immediate retention challenges and long-term growth objectives.",
                "key_insights": [
                    "Customer churn is primarily driven by unmet job-to-be-done requirements rather than product features",
                    "Significant untapped blue ocean opportunities exist in mid-market segment",
                    "Current customer lifetime value optimization is suboptimal due to poor onboarding experience",
                ],
                "synergies": [
                    {
                        "type": "Strategic Reinforcement",
                        "description": "Jobs-to-be-Done insights directly inform Blue Ocean strategy development by revealing underserved customer needs",
                        "impact": "High",
                    },
                    {
                        "type": "Operational Synergy",
                        "description": "CLV optimization naturally incorporates insights from both JTBD research and Blue Ocean positioning",
                        "impact": "Medium",
                    },
                ],
            },
            "implementation_strategy": {
                "immediate_actions": [
                    "Launch customer interview program to understand job-to-be-done motivations",
                    "Conduct competitive analysis to map current market positioning",
                    "Audit existing customer onboarding flow and retention metrics",
                ],
                "phase1": {
                    "duration": "6-8 weeks",
                    "focus": "Research and Discovery",
                    "activities": [
                        "Complete 50+ customer interviews across different segments",
                        "Map competitive landscape and identify potential blue ocean opportunities",
                        "Establish baseline metrics for customer lifetime value tracking",
                    ],
                },
                "phase2": {
                    "duration": "8-12 weeks",
                    "focus": "Strategy Development",
                    "activities": [
                        "Develop refined value proposition based on JTBD insights",
                        "Design blue ocean strategy canvas and implementation roadmap",
                        "Create optimized customer journey and retention playbooks",
                    ],
                },
                "phase3": {
                    "duration": "12-16 weeks",
                    "focus": "Implementation and Optimization",
                    "activities": [
                        "Launch new positioning and messaging across all channels",
                        "Implement enhanced onboarding and retention programs",
                        "Monitor and optimize based on customer feedback and metrics",
                    ],
                },
                "success_metrics": [
                    "Customer churn rate reduction of 25%",
                    "Net Revenue Retention improvement to 115%+",
                    "Market differentiation score increase of 40%",
                    "Customer Lifetime Value increase of 30%",
                ],
            },
        }

    except Exception as e:
        logger.error(f"❌ Failed to retrieve analysis {analysis_id}: {e}")
        return None


async def schedule_cleanup(generation_id: str):
    """Schedule background cleanup for a generation"""
    # Placeholder for cleanup scheduling
    # In production, this might involve:
    # - Scheduling file cleanup after a certain period
    # - Cleaning up temporary files
    # - Updating usage metrics
    logger.info(f"🧹 Scheduled cleanup for generation: {generation_id}")
    pass
