"""
Ideaflow Sprint API Router for METIS Lolla Platform
==================================================

API endpoints for the Ideaflow solution-finding engine.
BUILD ORDER: F-02 (Clarity-Ideaflow-Engine-MVP)
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import uuid
import asyncio

logger = logging.getLogger(__name__)

# Router setup
router = APIRouter(prefix="/api/ideaflow", tags=["ideaflow"])


# Request/Response Models
class IdeaflowRequest(BaseModel):
    """Request model for starting an Ideaflow sprint"""
    problem_statement: str = Field(..., description="The problem statement (preferably 'How might we...' format)")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context about the problem")
    max_ideas_per_stream: Optional[int] = Field(default=10, description="Maximum ideas to generate per creative stream")
    cluster_count: Optional[int] = Field(default=5, description="Target number of idea clusters")


class IdeaCluster(BaseModel):
    """Model for a cluster of related ideas"""
    theme: str = Field(..., description="The thematic title of this cluster")
    description: str = Field(..., description="Brief description of the cluster theme")
    selected_idea: str = Field(..., description="The most promising idea from this cluster")
    all_ideas: List[str] = Field(..., description="All ideas that were grouped into this cluster")
    experiment: Dict[str, Any] = Field(..., description="Experimental design for testing the selected idea")


class IdeaflowResult(BaseModel):
    """Final result model for completed Ideaflow sprint"""
    sprint_id: str
    problem_statement: str
    generated_ideas_count: int
    clusters: List[IdeaCluster]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    completed_at: str


class IdeaflowStatus(BaseModel):
    """Status model for tracking sprint progress"""
    sprint_id: str
    status: str  # "running", "completed", "failed"
    current_step: str  # "divergent_generation", "convergent_filtering", "experimental_design"
    progress_percentage: int
    started_at: str
    estimated_completion: Optional[str] = None
    error_message: Optional[str] = None


class IdeaflowStartResponse(BaseModel):
    """Response model for sprint initiation"""
    sprint_id: str
    status: str
    message: str
    estimated_duration_minutes: int


# In-memory storage for sprint tracking (in production, use Redis/database)
_active_sprints: Dict[str, Dict[str, Any]] = {}


# Import the orchestrator
from src.ideaflow.ideaflow_orchestrator import IdeaflowOrchestrator, IdeaflowResult as OrchIdeaflowResult


async def get_ideaflow_orchestrator():
    """Get the Ideaflow orchestrator instance"""
    return IdeaflowOrchestrator()


@router.post("/start", response_model=IdeaflowStartResponse)
async def start_ideaflow_sprint(
    request: IdeaflowRequest,
    background_tasks: BackgroundTasks
):
    """
    Start a new Ideaflow sprint for solution generation
    
    This initiates a three-step pipeline:
    1. Divergent Generation - parallel creative streams
    2. Convergent Filtering - clustering and selection
    3. Experimental Design - MVS design for top ideas
    """
    try:
        # Generate unique sprint ID
        sprint_id = str(uuid.uuid4())
        
        logger.info(f"🚀 Starting Ideaflow sprint - ID: {sprint_id}")
        logger.info(f"📝 Problem: {request.problem_statement}")
        
        # Initialize sprint tracking
        _active_sprints[sprint_id] = {
            "status": "running",
            "current_step": "divergent_generation",
            "progress_percentage": 0,
            "started_at": datetime.now().isoformat(),
            "request": request.dict(),
            "result": None,
            "error": None
        }
        
        # Start the sprint in the background
        background_tasks.add_task(
            _run_ideaflow_sprint,
            sprint_id,
            request
        )
        
        return IdeaflowStartResponse(
            sprint_id=sprint_id,
            status="running",
            message="Ideaflow sprint initiated successfully",
            estimated_duration_minutes=5  # Rough estimate for 3-step pipeline
        )
        
    except Exception as e:
        logger.error(f"❌ Error starting Ideaflow sprint: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error starting Ideaflow sprint: {str(e)}"
        )


@router.get("/{sprint_id}/status", response_model=IdeaflowStatus)
async def get_sprint_status(sprint_id: str):
    """
    Get the current status of an Ideaflow sprint
    """
    try:
        if sprint_id not in _active_sprints:
            raise HTTPException(
                status_code=404,
                detail=f"Sprint {sprint_id} not found"
            )
        
        sprint_data = _active_sprints[sprint_id]
        
        return IdeaflowStatus(
            sprint_id=sprint_id,
            status=sprint_data["status"],
            current_step=sprint_data["current_step"],
            progress_percentage=sprint_data["progress_percentage"],
            started_at=sprint_data["started_at"],
            estimated_completion=sprint_data.get("estimated_completion"),
            error_message=sprint_data.get("error")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting sprint status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving sprint status: {str(e)}"
        )


@router.get("/{sprint_id}/results", response_model=IdeaflowResult)
async def get_sprint_results(sprint_id: str):
    """
    Get the results of a completed Ideaflow sprint
    """
    try:
        if sprint_id not in _active_sprints:
            raise HTTPException(
                status_code=404,
                detail=f"Sprint {sprint_id} not found"
            )
        
        sprint_data = _active_sprints[sprint_id]
        
        if sprint_data["status"] != "completed":
            raise HTTPException(
                status_code=400,
                detail=f"Sprint {sprint_id} is not completed. Current status: {sprint_data['status']}"
            )
        
        if not sprint_data.get("result"):
            raise HTTPException(
                status_code=500,
                detail=f"Sprint {sprint_id} completed but no results found"
            )
        
        return sprint_data["result"]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting sprint results: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving sprint results: {str(e)}"
        )


@router.delete("/{sprint_id}")
async def delete_sprint(sprint_id: str):
    """
    Delete a sprint and free resources
    """
    try:
        if sprint_id in _active_sprints:
            del _active_sprints[sprint_id]
            logger.info(f"🗑️ Ideaflow sprint deleted - ID: {sprint_id}")
        
        return {"message": f"Sprint {sprint_id} deleted successfully"}
        
    except Exception as e:
        logger.error(f"❌ Error deleting sprint: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting sprint: {str(e)}"
        )


@router.get("/health")
async def ideaflow_health():
    """
    Health check endpoint for Ideaflow service
    """
    return {
        "status": "healthy",
        "service": "Ideaflow Solution Engine",
        "active_sprints": len(_active_sprints),
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }


async def _run_ideaflow_sprint(sprint_id: str, request: IdeaflowRequest):
    """
    Background task to execute the full Ideaflow sprint pipeline
    """
    try:
        logger.info(f"🏃 Executing Ideaflow sprint {sprint_id}")
        
        # Get orchestrator instance
        orchestrator = await get_ideaflow_orchestrator()
        
        # Update progress
        _active_sprints[sprint_id]["progress_percentage"] = 10
        
        # Run the full sprint
        orch_result = await orchestrator.run_sprint(
            problem_statement=request.problem_statement,
            context=request.context,
            max_ideas_per_stream=request.max_ideas_per_stream,
            cluster_count=request.cluster_count,
            progress_callback=lambda step, progress: _update_sprint_progress(sprint_id, step, progress)
        )
        
        # Convert orchestrator result to API response format
        api_clusters = []
        for cluster in orch_result.clusters:
            api_cluster = IdeaCluster(
                theme=cluster.theme,
                description=cluster.description,
                selected_idea=cluster.selected_idea,
                all_ideas=cluster.all_ideas,
                experiment=cluster.experiment
            )
            api_clusters.append(api_cluster)
        
        api_result = IdeaflowResult(
            sprint_id=orch_result.sprint_id,
            problem_statement=orch_result.problem_statement,
            generated_ideas_count=orch_result.generated_ideas_count,
            clusters=api_clusters,
            metadata=orch_result.metadata,
            completed_at=orch_result.completed_at
        )
        
        # Mark as completed
        _active_sprints[sprint_id].update({
            "status": "completed",
            "current_step": "completed",
            "progress_percentage": 100,
            "result": api_result
        })
        
        logger.info(f"✅ Ideaflow sprint {sprint_id} completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Ideaflow sprint {sprint_id} failed: {e}")
        _active_sprints[sprint_id].update({
            "status": "failed",
            "error": str(e),
            "progress_percentage": 0
        })


def _update_sprint_progress(sprint_id: str, step: str, progress: int):
    """Update sprint progress tracking"""
    if sprint_id in _active_sprints:
        _active_sprints[sprint_id].update({
            "current_step": step,
            "progress_percentage": progress
        })
