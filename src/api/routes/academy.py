"""
Academy API Router for METIS Lolla Academy
==========================================

API endpoints for the Mental Model Navigator conversational application.
BUILD ORDER AC-01: Backend foundation for the standalone Academy product.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import uuid

# Import Academy services
from src.academy.navigator_orchestrator import NavigatorOrchestrator
from src.services.knowledge_retrieval_service import KnowledgeRetrievalService

logger = logging.getLogger(__name__)

# Router setup
router = APIRouter(prefix="/academy", tags=["academy"])


# Request/Response Models
class ConversationRequest(BaseModel):
    """Request model for conversation endpoint"""
    session_id: str = Field(..., description="Unique session identifier")
    message: str = Field(..., description="User's latest message")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")


class ConversationResponse(BaseModel):
    """Response model for conversation endpoint"""
    session_id: str
    response: str
    state: str
    suggested_actions: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str


class SessionStatus(BaseModel):
    """Session status response model"""
    session_id: str
    state: str
    current_step: int
    total_steps: int
    last_activity: str
    created_at: str


# Global orchestrator instance (in production, this should be dependency injected)
_orchestrator_cache: Dict[str, NavigatorOrchestrator] = {}


def get_navigator_orchestrator(session_id: str) -> NavigatorOrchestrator:
    """
    Get or create NavigatorOrchestrator for the session
    
    In production, this should use proper dependency injection
    and session persistence (Redis, database, etc.)
    """
    if session_id not in _orchestrator_cache:
        # Create new orchestrator with injected services
        knowledge_service = KnowledgeRetrievalService()
        _orchestrator_cache[session_id] = NavigatorOrchestrator(
            session_id=session_id,
            knowledge_service=knowledge_service
        )
    
    return _orchestrator_cache[session_id]


@router.post("/conversation", response_model=ConversationResponse)
async def handle_conversation(request: ConversationRequest):
    """
    Main conversation endpoint for the Mental Model Navigator
    
    Handles stateful conversation flow through the 11-step Navigator logic.
    """
    try:
        logger.info(f"🎓 Academy conversation request - Session: {request.session_id}")
        
        # Get or create orchestrator for this session
        orchestrator = get_navigator_orchestrator(request.session_id)
        
        # Process the message through the Navigator state machine
        response_data = await orchestrator.process_message(
            message=request.message,
            context=request.context
        )
        
        return ConversationResponse(
            session_id=request.session_id,
            response=response_data["response"],
            state=response_data["state"],
            suggested_actions=response_data.get("suggested_actions", []),
            metadata=response_data.get("metadata", {}),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"❌ Academy conversation error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing conversation: {str(e)}"
        )


@router.get("/session/{session_id}/status", response_model=SessionStatus)
async def get_session_status(session_id: str):
    """
    Get current status of a Navigator session
    """
    try:
        if session_id not in _orchestrator_cache:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found"
            )
        
        orchestrator = _orchestrator_cache[session_id]
        status = orchestrator.get_session_status()
        
        return SessionStatus(
            session_id=session_id,
            state=status["state"],
            current_step=status["current_step"],
            total_steps=status["total_steps"],
            last_activity=status["last_activity"],
            created_at=status["created_at"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting session status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving session status: {str(e)}"
        )


@router.post("/session/{session_id}/reset")
async def reset_session(session_id: str):
    """
    Reset a Navigator session to initial state
    """
    try:
        if session_id in _orchestrator_cache:
            orchestrator = _orchestrator_cache[session_id]
            await orchestrator.reset_session()
            logger.info(f"🔄 Academy session reset - Session: {session_id}")
        
        return {"message": f"Session {session_id} reset successfully"}
        
    except Exception as e:
        logger.error(f"❌ Error resetting session: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error resetting session: {str(e)}"
        )


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """
    Delete a Navigator session and free resources
    """
    try:
        if session_id in _orchestrator_cache:
            del _orchestrator_cache[session_id]
            logger.info(f"🗑️ Academy session deleted - Session: {session_id}")
        
        return {"message": f"Session {session_id} deleted successfully"}
        
    except Exception as e:
        logger.error(f"❌ Error deleting session: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting session: {str(e)}"
        )


@router.get("/health")
async def academy_health():
    """
    Health check endpoint for Academy service
    """
    return {
        "status": "healthy",
        "service": "Academy Mental Model Navigator",
        "active_sessions": len(_orchestrator_cache),
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }