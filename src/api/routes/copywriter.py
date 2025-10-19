"""
Copywriter API Routes: RESTful endpoints for the N-Way Copywriter feature.

BUILD ORDER: F-04 (Clarity-NWAY-Copywriter)
Operation: Clarity
Phase: 3 (New Feature Development)

Provides endpoints for starting copywriter jobs, tracking progress, and retrieving results.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime

from src.copywriting.copywriter_orchestrator import CopywriterOrchestrator, CopywriterJobStatus
from src.core.structured_logging import StructuredLogger


# Request/Response models
class StartCopywriterRequest(BaseModel):
    """Request to start a new copywriter job"""
    trace_id: str = Field(..., description="Trace ID of the source strategic report")
    target_word_count: int = Field(default=1200, ge=600, le=3000, description="Target word count for final output")
    persuasion_intensity: float = Field(default=0.7, ge=0.3, le=1.0, description="How aggressive the persuasive elements should be")
    technical_depth: float = Field(default=0.5, ge=0.1, le=0.9, description="Level of technical detail to maintain")
    audience_sophistication: float = Field(default=0.6, ge=0.2, le=0.9, description="Assumed knowledge level of target audience")


class StartCopywriterResponse(BaseModel):
    """Response from starting a copywriter job"""
    job_id: str
    trace_id: str
    status: str
    message: str
    processing_url: str
    estimated_duration_minutes: int


class CopywriterJobStatusResponse(BaseModel):
    """Current status of a copywriter job"""
    job_id: str
    trace_id: str
    status: str
    current_stage: Optional[str]
    completed_stages: List[str]
    progress_percentage: float
    estimated_remaining_minutes: Optional[int]
    error_message: Optional[str]
    created_at: datetime
    stage_results: List[Dict[str, Any]]


class CommunicationPacketResponse(BaseModel):
    """Final communication packet from completed copywriter job"""
    job_id: str
    trace_id: str
    
    # Core deliverables
    governing_thought: str
    narrative_structure: Dict[str, Any]
    polished_content: str
    anticipated_objections: List[Dict[str, Any]]
    defensive_strategies: List[Dict[str, Any]]
    
    # Quality metrics
    skim_test_score: float
    clarity_score: float
    persuasion_score: float
    defensibility_score: float
    
    # Metadata
    word_count: int
    total_duration_seconds: float
    total_tokens_used: int
    completed_at: datetime


# Initialize router and orchestrator
router = APIRouter(prefix="/api/copywriter", tags=["copywriter"])
logger = StructuredLogger("copywriter_api")

# Global orchestrator instance
_orchestrator: Optional[CopywriterOrchestrator] = None


def get_orchestrator() -> CopywriterOrchestrator:
    """Get or create the global copywriter orchestrator instance"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = CopywriterOrchestrator()
    return _orchestrator


@router.post("/start", response_model=StartCopywriterResponse)
async def start_copywriter_job(
    request: StartCopywriterRequest,
    background_tasks: BackgroundTasks
) -> StartCopywriterResponse:
    """
    Start a new copywriter job for transforming a strategic report into persuasive communication.
    
    This endpoint kicks off the five-stage copywriter pipeline:
    1. The Distiller: Extract governing thought
    2. The Narrative Architect: Build story arc  
    3. The Copywriter: Create shitty first draft
    4. The Editor: Polish and refine
    5. The Red Team Coach: Generate objections and defenses
    """
    try:
        orchestrator = get_orchestrator()
        
        # Validate that the trace_id exists and has a completed report
        # (This would integrate with the existing report storage system)
        
        # Start the copywriter job
        job_id = await orchestrator.start_copywriter_job(
            trace_id=request.trace_id,
            target_word_count=request.target_word_count,
            persuasion_intensity=request.persuasion_intensity,
            technical_depth=request.technical_depth,
            audience_sophistication=request.audience_sophistication
        )
        
        logger.info(
            "Started copywriter job",
            extra={
                "job_id": job_id,
                "trace_id": request.trace_id,
                "config": request.dict()
            }
        )
        
        return StartCopywriterResponse(
            job_id=job_id,
            trace_id=request.trace_id,
            status=CopywriterJobStatus.PENDING.value,
            message="Copywriter job started successfully. Processing has begun.",
            processing_url=f"/copywriter/{job_id}",
            estimated_duration_minutes=8  # Estimate based on 5 stages
        )
        
    except Exception as e:
        logger.error(
            "Failed to start copywriter job",
            extra={
                "trace_id": request.trace_id,
                "error": str(e)
            }
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start copywriter job: {str(e)}"
        )


@router.get("/{job_id}/status", response_model=CopywriterJobStatusResponse)
async def get_copywriter_job_status(job_id: str) -> CopywriterJobStatusResponse:
    """
    Get the current status and progress of a copywriter job.
    
    Returns detailed information about which stages have completed,
    current progress, and any errors that may have occurred.
    """
    try:
        orchestrator = get_orchestrator()
        job_data = orchestrator.get_job_status(job_id)
        
        if not job_data:
            raise HTTPException(
                status_code=404,
                detail=f"Copywriter job {job_id} not found"
            )
        
        # Calculate progress and current stage
        stage_names = ["distillation", "architecture", "drafting", "editing", "red_team"]
        completed_stages = [result["stage"] for result in job_data["stage_results"]]
        
        if job_data["status"] == CopywriterJobStatus.COMPLETED.value:
            progress_percentage = 100.0
            current_stage = None
            estimated_remaining = 0
        elif job_data["status"] == CopywriterJobStatus.FAILED.value:
            progress_percentage = len(completed_stages) / len(stage_names) * 100
            current_stage = None
            estimated_remaining = 0
        else:
            progress_percentage = len(completed_stages) / len(stage_names) * 100
            current_stage = stage_names[len(completed_stages)] if len(completed_stages) < len(stage_names) else None
            estimated_remaining = (len(stage_names) - len(completed_stages)) * 1.5  # ~1.5 min per stage
        
        return CopywriterJobStatusResponse(
            job_id=job_id,
            trace_id=job_data["trace_id"],
            status=job_data["status"],
            current_stage=current_stage,
            completed_stages=completed_stages,
            progress_percentage=progress_percentage,
            estimated_remaining_minutes=estimated_remaining,
            error_message=job_data.get("error_message"),
            created_at=datetime.fromisoformat(job_data["created_at"]),
            stage_results=job_data["stage_results"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to get copywriter job status",
            extra={
                "job_id": job_id,
                "error": str(e)
            }
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get job status: {str(e)}"
        )


@router.get("/{job_id}/results", response_model=CommunicationPacketResponse)
async def get_copywriter_job_results(job_id: str) -> CommunicationPacketResponse:
    """
    Get the final communication packet from a completed copywriter job.
    
    Returns the complete deliverable including:
    - Governing thought (core 12-word message)
    - Polished content ready for publication
    - Anticipated objections and defensive strategies
    - Quality metrics and metadata
    """
    try:
        orchestrator = get_orchestrator()
        job_data = orchestrator.get_job_status(job_id)
        
        if not job_data:
            raise HTTPException(
                status_code=404,
                detail=f"Copywriter job {job_id} not found"
            )
        
        if job_data["status"] != CopywriterJobStatus.COMPLETED.value:
            raise HTTPException(
                status_code=400,
                detail=f"Job {job_id} is not completed. Current status: {job_data['status']}"
            )
        
        final_packet = job_data["final_packet"]
        if not final_packet:
            raise HTTPException(
                status_code=500,
                detail=f"Job {job_id} is marked as completed but has no results"
            )
        
        return CommunicationPacketResponse(
            job_id=job_id,
            trace_id=job_data["trace_id"],
            governing_thought=final_packet["governing_thought"],
            narrative_structure=final_packet["narrative_structure"],
            polished_content=final_packet["polished_content"],
            anticipated_objections=final_packet["anticipated_objections"],
            defensive_strategies=final_packet["defensive_strategies"],
            skim_test_score=final_packet["skim_test_score"],
            clarity_score=final_packet["clarity_score"],
            persuasion_score=final_packet["persuasion_score"],
            defensibility_score=final_packet["defensibility_score"],
            word_count=final_packet["word_count"],
            total_duration_seconds=final_packet["total_duration_seconds"],
            total_tokens_used=final_packet["total_tokens_used"],
            completed_at=datetime.fromisoformat(job_data["completed_at"])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to get copywriter job results",
            extra={
                "job_id": job_id,
                "error": str(e)
            }
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get job results: {str(e)}"
        )


@router.get("/{job_id}/stage/{stage_name}")
async def get_copywriter_stage_results(job_id: str, stage_name: str) -> Dict[str, Any]:
    """
    Get detailed results from a specific stage of the copywriter pipeline.
    
    Stages: distillation, architecture, drafting, editing, red_team
    """
    try:
        orchestrator = get_orchestrator()
        job_data = orchestrator.get_job_status(job_id)
        
        if not job_data:
            raise HTTPException(
                status_code=404,
                detail=f"Copywriter job {job_id} not found"
            )
        
        # Find the requested stage results
        stage_result = None
        for result in job_data["stage_results"]:
            if result["stage"] == stage_name:
                stage_result = result
                break
        
        if not stage_result:
            raise HTTPException(
                status_code=404,
                detail=f"Stage '{stage_name}' not found or not completed for job {job_id}"
            )
        
        return stage_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to get copywriter stage results",
            extra={
                "job_id": job_id,
                "stage_name": stage_name,
                "error": str(e)
            }
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get stage results: {str(e)}"
        )


@router.delete("/{job_id}")
async def cancel_copywriter_job(job_id: str) -> Dict[str, str]:
    """
    Cancel a running copywriter job.
    
    Note: Jobs cannot be cancelled once completed, and cancellation
    may not be immediate if a stage is currently processing.
    """
    try:
        orchestrator = get_orchestrator()
        job_data = orchestrator.get_job_status(job_id)
        
        if not job_data:
            raise HTTPException(
                status_code=404,
                detail=f"Copywriter job {job_id} not found"
            )
        
        if job_data["status"] in [CopywriterJobStatus.COMPLETED.value, CopywriterJobStatus.FAILED.value]:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel job {job_id} with status: {job_data['status']}"
            )
        
        # Mark job as cancelled (implementation would depend on orchestrator capabilities)
        # For now, we'll just return a success message
        
        logger.info(
            "Cancelled copywriter job",
            extra={
                "job_id": job_id,
                "trace_id": job_data["trace_id"]
            }
        )
        
        return {
            "job_id": job_id,
            "message": "Copywriter job cancellation requested"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to cancel copywriter job",
            extra={
                "job_id": job_id,
                "error": str(e)
            }
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cancel job: {str(e)}"
        )


@router.get("/jobs")
async def list_copywriter_jobs(
    trace_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50
) -> Dict[str, Any]:
    """
    List copywriter jobs with optional filtering.
    
    Can filter by trace_id and/or status.
    Returns basic job information without full results.
    """
    try:
        orchestrator = get_orchestrator()
        
        # Get all jobs from orchestrator (this would need to be implemented)
        # For now, return empty list
        jobs = []
        
        # Apply filters if provided
        if trace_id:
            jobs = [job for job in jobs if job.get("trace_id") == trace_id]
        
        if status:
            jobs = [job for job in jobs if job.get("status") == status]
        
        # Apply limit
        jobs = jobs[:limit]
        
        return {
            "jobs": jobs,
            "total": len(jobs),
            "filters": {
                "trace_id": trace_id,
                "status": status,
                "limit": limit
            }
        }
        
    except Exception as e:
        logger.error(
            "Failed to list copywriter jobs",
            extra={
                "trace_id": trace_id,
                "status": status,
                "error": str(e)
            }
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list jobs: {str(e)}"
        )


# Health check endpoint
@router.get("/health")
async def copywriter_health_check() -> Dict[str, str]:
    """Health check for the copywriter service"""
    try:
        orchestrator = get_orchestrator()
        return {
            "status": "healthy",
            "service": "copywriter",
            "version": "1.0",
            "message": "Copywriter service is operational"
        }
    except Exception as e:
        logger.error(
            "Copywriter health check failed",
            extra={"error": str(e)}
        )
        raise HTTPException(
            status_code=503,
            detail=f"Copywriter service is unhealthy: {str(e)}"
        )