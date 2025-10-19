"""
Strategic Pitch Generation API Router for METIS Lolla Platform
=============================================================

API endpoints for the strategic pitch generation engine.
BUILD ORDER: P-01 (Pitch-NWAY-Creation)
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
router = APIRouter(prefix="/pitch", tags=["pitch"])


# Request/Response Models
class PitchRequest(BaseModel):
    """Request model for starting a pitch generation"""
    strategic_content: str = Field(..., description="The strategic recommendation or content to pitch")
    audience_context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Information about the target audience")
    constraints: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Time, format, and delivery constraints")


class PitchDeck(BaseModel):
    """Model for a complete pitch deck"""
    title: str = Field(..., description="The main title of the pitch")
    executive_summary: str = Field(..., description="Executive summary of the strategic content")
    narrative_structure: Dict[str, Any] = Field(..., description="Story structure and flow")
    evidence_architecture: Dict[str, Any] = Field(..., description="Data and supporting evidence")
    objection_playbook: Dict[str, Any] = Field(..., description="Anticipated objections and responses")
    visual_guidelines: Dict[str, Any] = Field(..., description="Visual design guidelines")
    slides: List[Dict[str, Any]] = Field(..., description="Slide structure and content")


class PitchResult(BaseModel):
    """Final result model for completed pitch generation"""
    pitch_id: str
    strategic_content: str
    pitch_deck: PitchDeck
    speaker_notes: Dict[str, Any] = Field(..., description="Speaker notes and presentation guidance")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    completed_at: str


class PitchStatus(BaseModel):
    """Status model for tracking pitch generation progress"""
    pitch_id: str
    status: str  # "running", "completed", "failed"
    current_phase: str  # "foundation_development", "stress_testing", "visual_design", "integration"
    progress_percentage: int
    started_at: str
    estimated_completion: Optional[str] = None
    error_message: Optional[str] = None


class PitchStartResponse(BaseModel):
    """Response model for pitch generation initiation"""
    pitch_id: str
    status: str
    message: str
    estimated_duration_minutes: int


# In-memory storage for pitch tracking (in production, use Redis/database)
_active_pitches: Dict[str, Dict[str, Any]] = {}


# Import the orchestrator
from src.ideaflow.pitch_orchestrator import PitchOrchestrator, PitchResult as OrchPitchResult


async def get_pitch_orchestrator():
    """Get the Pitch orchestrator instance"""
    return PitchOrchestrator()


@router.post("/generate", response_model=PitchStartResponse)
async def start_pitch_generation(
    request: PitchRequest,
    background_tasks: BackgroundTasks
):
    """
    Start a new strategic pitch generation process
    
    This initiates a four-phase collaborative pipeline:
    1. Foundation Development - narrative structure and evidence architecture
    2. Stress Testing - objection analysis and red team preparation
    3. Visual Design - instant comprehension optimization  
    4. Integration - cohesive, persuasive final pitch
    """
    try:
        # Generate unique pitch ID
        pitch_id = str(uuid.uuid4())
        
        logger.info(f"🚀 Starting pitch generation - ID: {pitch_id}")
        logger.info(f"📝 Strategic Content Preview: {request.strategic_content[:100]}...")
        
        # Initialize pitch tracking
        _active_pitches[pitch_id] = {
            "status": "running",
            "current_phase": "foundation_development",
            "progress_percentage": 0,
            "started_at": datetime.now().isoformat(),
            "request": request.dict(),
            "result": None,
            "error": None
        }
        
        # Start the pitch generation in the background
        background_tasks.add_task(
            _run_pitch_generation,
            pitch_id,
            request
        )
        
        return PitchStartResponse(
            pitch_id=pitch_id,
            status="running",
            message="Strategic pitch generation initiated successfully",
            estimated_duration_minutes=8  # Rough estimate for 4-phase pipeline
        )
        
    except Exception as e:
        logger.error(f"❌ Error starting pitch generation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error starting pitch generation: {str(e)}"
        )


@router.get("/{pitch_id}/status", response_model=PitchStatus)
async def get_pitch_status(pitch_id: str):
    """
    Get the current status of a pitch generation process
    """
    try:
        if pitch_id not in _active_pitches:
            raise HTTPException(
                status_code=404,
                detail=f"Pitch {pitch_id} not found"
            )
        
        pitch_data = _active_pitches[pitch_id]
        
        return PitchStatus(
            pitch_id=pitch_id,
            status=pitch_data["status"],
            current_phase=pitch_data["current_phase"],
            progress_percentage=pitch_data["progress_percentage"],
            started_at=pitch_data["started_at"],
            estimated_completion=pitch_data.get("estimated_completion"),
            error_message=pitch_data.get("error")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting pitch status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving pitch status: {str(e)}"
        )


@router.get("/{pitch_id}/deck", response_model=PitchResult)
async def get_pitch_deck(pitch_id: str):
    """
    Get the complete pitch deck for a completed generation process
    """
    try:
        if pitch_id not in _active_pitches:
            raise HTTPException(
                status_code=404,
                detail=f"Pitch {pitch_id} not found"
            )
        
        pitch_data = _active_pitches[pitch_id]
        
        if pitch_data["status"] != "completed":
            raise HTTPException(
                status_code=400,
                detail=f"Pitch {pitch_id} is not completed. Current status: {pitch_data['status']}"
            )
        
        if not pitch_data.get("result"):
            raise HTTPException(
                status_code=500,
                detail=f"Pitch {pitch_id} completed but no results found"
            )
        
        return pitch_data["result"]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting pitch deck: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving pitch deck: {str(e)}"
        )


@router.get("/{pitch_id}/playbook")
async def get_objection_playbook(pitch_id: str):
    """
    Get the objection handling playbook for a completed pitch
    """
    try:
        if pitch_id not in _active_pitches:
            raise HTTPException(
                status_code=404,
                detail=f"Pitch {pitch_id} not found"
            )
        
        pitch_data = _active_pitches[pitch_id]
        
        if pitch_data["status"] != "completed":
            raise HTTPException(
                status_code=400,
                detail=f"Pitch {pitch_id} is not completed. Current status: {pitch_data['status']}"
            )
        
        result = pitch_data.get("result")
        if not result:
            raise HTTPException(
                status_code=500,
                detail="No pitch results found"
            )
        
        return {
            "pitch_id": pitch_id,
            "objection_playbook": result.pitch_deck.objection_playbook,
            "speaker_notes": result.speaker_notes
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting objection playbook: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving objection playbook: {str(e)}"
        )


@router.delete("/{pitch_id}")
async def delete_pitch(pitch_id: str):
    """
    Delete a pitch and free resources
    """
    try:
        if pitch_id in _active_pitches:
            del _active_pitches[pitch_id]
            logger.info(f"🗑️ Pitch deleted - ID: {pitch_id}")
        
        return {"message": f"Pitch {pitch_id} deleted successfully"}
        
    except Exception as e:
        logger.error(f"❌ Error deleting pitch: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting pitch: {str(e)}"
        )


@router.get("/health")
async def pitch_health():
    """
    Health check endpoint for Pitch service
    """
    return {
        "status": "healthy",
        "service": "Strategic Pitch Generator",
        "active_pitches": len(_active_pitches),
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }


async def _run_pitch_generation(pitch_id: str, request: PitchRequest):
    """
    Background task to execute the full pitch generation pipeline
    """
    try:
        logger.info(f"🏃 Executing pitch generation {pitch_id}")
        
        # Get orchestrator instance
        orchestrator = await get_pitch_orchestrator()
        
        # Update progress
        _active_pitches[pitch_id]["progress_percentage"] = 10
        
        # Prepare context from request
        context = {
            **request.audience_context,
            **request.constraints
        }
        
        # Run the full pitch generation
        orch_result = await orchestrator.generate_pitch(
            strategic_content=request.strategic_content,
            context=context,
            progress_callback=lambda phase, progress: _update_pitch_progress(pitch_id, phase, progress)
        )
        
        # Convert orchestrator result to API response format
        api_pitch_deck = PitchDeck(
            title=orch_result.pitch_deck.title,
            executive_summary=orch_result.pitch_deck.executive_summary,
            narrative_structure=orch_result.pitch_deck.narrative_structure,
            evidence_architecture=orch_result.pitch_deck.evidence_architecture,
            objection_playbook=orch_result.pitch_deck.objection_playbook,
            visual_guidelines=orch_result.pitch_deck.visual_guidelines,
            slides=orch_result.pitch_deck.slides
        )
        
        api_result = PitchResult(
            pitch_id=orch_result.pitch_id,
            strategic_content=orch_result.strategic_content,
            pitch_deck=api_pitch_deck,
            speaker_notes=orch_result.speaker_notes,
            metadata=orch_result.metadata,
            completed_at=orch_result.completed_at
        )
        
        # Mark as completed
        _active_pitches[pitch_id].update({
            "status": "completed",
            "current_phase": "completed",
            "progress_percentage": 100,
            "result": api_result
        })
        
        logger.info(f"✅ Pitch generation {pitch_id} completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Pitch generation {pitch_id} failed: {e}")
        _active_pitches[pitch_id].update({
            "status": "failed",
            "error": str(e),
            "progress_percentage": 0
        })


def _update_pitch_progress(pitch_id: str, phase: str, progress: int):
    """Update pitch generation progress tracking"""
    if pitch_id in _active_pitches:
        _active_pitches[pitch_id].update({
            "current_phase": phase,
            "progress_percentage": progress
        })