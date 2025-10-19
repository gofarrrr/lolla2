#!/usr/bin/env python3
"""
Strategic Trio + Devil's Advocate API
Implements Optional, Post-Human, Per-Consultant pattern via REST API

API Flow:
1. POST /execute - Execute Strategic Trio, return all consultant perspectives immediately
2. Human reviews Strategic Trio results
3. POST /{orchestration_id}/request-critique - Human optionally requests Devil's Advocate
4. WebSocket streams independent critique results as they complete
5. Human chooses which perspectives and critiques to act upon

Core Values Enforced:
- Post-Human: Critique only after human sees Strategic Trio results
- Optional: Human must explicitly request critique
- Per-Consultant: Each consultant gets independent critique (no synthesis)
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import (
    APIRouter,
    HTTPException,
    Depends,
    BackgroundTasks,
    WebSocket,
    WebSocketDisconnect,
)
from pydantic import BaseModel, Field

from src.engine.engines.strategic_trio_critique_orchestrator import (
    create_strategic_trio_critique_orchestrator,
)
from src.models.strategic_trio_critique_models import (
    StrategicTrioCritiqueOrchestrationResult,
    CritiqueStreamingUpdate,
)
from src.core.supabase_auth_middleware import get_current_user, SupabaseUser

logger = logging.getLogger(__name__)

# Initialize router and orchestrator
router = APIRouter(prefix="/api/strategic-trio", tags=["strategic_trio_critique"])
orchestrator = create_strategic_trio_critique_orchestrator()

# Active orchestrations for tracking
active_orchestrations: Dict[str, StrategicTrioCritiqueOrchestrationResult] = {}
active_websockets: Dict[str, List[WebSocket]] = {}

# Request/Response Models


class StrategicTrioExecuteRequest(BaseModel):
    query: str = Field(
        ..., description="Query for Strategic Trio analysis", min_length=10
    )
    context: Optional[Dict[str, Any]] = Field(
        None, description="Optional business context"
    )
    user_preferences: Optional[Dict[str, Any]] = Field(
        None, description="User preferences for analysis"
    )


class StrategicTrioExecuteResponse(BaseModel):
    orchestration_id: str
    status: str = "strategic_trio_completed"
    strategic_trio_result: Dict[str, Any]
    consultant_perspectives: Dict[str, str]
    total_consultants: int
    processing_time_seconds: float
    next_actions: List[str] = [
        "Review all consultant perspectives",
        "Optionally request Devil's Advocate critique",
        "Choose which insights to act upon",
    ]


class DevilsAdvocateCritiqueRequest(BaseModel):
    business_context: Optional[Dict[str, Any]] = Field(
        None, description="Additional business context for critique"
    )
    engines: Optional[List[str]] = Field(
        ["munger", "ackoff", "cognitive_audit"], description="Critique engines to use"
    )
    stream_updates: bool = Field(
        True, description="Stream critique updates via WebSocket"
    )


class CritiqueStatusResponse(BaseModel):
    orchestration_id: str
    status: str
    strategic_trio_completed: bool
    critique_requested: bool
    critique_in_progress: bool
    critique_completed: bool
    consultants_analyzed: int
    consultants_critiqued: int
    created_at: str
    updated_at: str


class CritiqueResultsResponse(BaseModel):
    orchestration_id: str
    consultant_critiques: Dict[str, Dict[str, Any]]
    critique_summary: str
    recommended_next_actions: List[str]
    total_processing_time: float
    critiques_completed: int
    critiques_failed: int
    independence_preserved: bool = True
    synthesis_avoided: bool = True


# API Endpoints


@router.post("/execute", response_model=StrategicTrioExecuteResponse)
async def execute_strategic_trio(
    request: StrategicTrioExecuteRequest,
    background_tasks: BackgroundTasks,
    user: SupabaseUser = Depends(get_current_user),
):
    """
    Execute Strategic Trio analysis - Phase 1 of Optional, Post-Human, Per-Consultant pattern

    Returns all consultant perspectives immediately to human.
    Human can then optionally request Devil's Advocate critique.
    """
    start_time = datetime.utcnow()

    try:
        logger.info(f"🚀 Strategic Trio execution requested by user {user_id}")
        logger.info(f"Query: {request.query[:100]}...")

        # Execute Strategic Trio (Multi-Single-Agent parallel execution)
        orchestration_result = await orchestrator.execute_strategic_trio(
            query=request.query, context=request.context
        )

        if not orchestration_result.strategic_trio_result:
            raise HTTPException(
                status_code=500, detail="Strategic Trio execution failed"
            )

        # Store orchestration for future critique requests
        active_orchestrations[orchestration_result.orchestration_id] = (
            orchestration_result
        )

        # Prepare response with all consultant perspectives
        processing_time = (datetime.utcnow() - start_time).total_seconds()

        response = StrategicTrioExecuteResponse(
            orchestration_id=orchestration_result.orchestration_id,
            strategic_trio_result={
                "execution_id": orchestration_result.strategic_trio_result.execution_id,
                "confidence_score": orchestration_result.strategic_trio_result.confidence_score,
                "clusters_activated": orchestration_result.strategic_trio_result.clusters_activated,
                "consultants_used": [
                    c.value
                    for c in orchestration_result.strategic_trio_result.consultants_used
                ],
                "final_output": orchestration_result.strategic_trio_result.final_output,
                "quality_assessment": orchestration_result.strategic_trio_result.quality_assessment,
            },
            consultant_perspectives={
                consultant.value: perspective
                for consultant, perspective in orchestration_result.strategic_trio_result.consultant_perspectives.items()
            },
            total_consultants=orchestration_result.total_consultants_analyzed,
            processing_time_seconds=processing_time,
        )

        logger.info(
            f"✅ Strategic Trio completed for user {user_id}: {orchestration_result.total_consultants_analyzed} consultant perspectives"
        )
        logger.info(
            "🎯 Human-in-the-loop: Presenting all consultant perspectives for review"
        )

        return response

    except Exception as e:
        logger.error(f"❌ Strategic Trio execution failed for user {user_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Strategic Trio execution failed: {str(e)}"
        )


@router.post("/{orchestration_id}/request-critique")
async def request_devils_advocate_critique(
    orchestration_id: str,
    request: DevilsAdvocateCritiqueRequest,
    background_tasks: BackgroundTasks,
    user: SupabaseUser = Depends(get_current_user),
):
    """
    Request Devil's Advocate critique - Phase 2 of Optional, Post-Human, Per-Consultant pattern

    IMPORTANT: Should only be called AFTER human has reviewed Strategic Trio results
    Each consultant's analysis receives INDEPENDENT critique with NO synthesis
    """
    try:
        # Validate orchestration exists and Strategic Trio is complete
        if orchestration_id not in active_orchestrations:
            raise HTTPException(status_code=404, detail="Orchestration not found")

        orchestration_result = active_orchestrations[orchestration_id]

        if not orchestration_result.strategic_trio_result:
            raise HTTPException(status_code=400, detail="Strategic Trio not completed")

        if orchestration_result.critique_requested:
            raise HTTPException(
                status_code=400, detail="Devil's Advocate critique already requested"
            )

        logger.info(
            f"🔍 Devil's Advocate critique requested by user {user_id} for orchestration {orchestration_id}"
        )
        logger.info("✅ Post-human timing: Human has reviewed Strategic Trio results")

        # Mark that human has seen original results and requested critique
        orchestration_result.human_seen_original = True
        orchestration_result.human_requested_critique = True

        if request.stream_updates:
            # Start critique in background and stream updates
            background_tasks.add_task(
                _execute_critique_with_streaming,
                orchestration_result,
                request.business_context,
                orchestration_id,
            )

            return {
                "status": "critique_started",
                "orchestration_id": orchestration_id,
                "message": "Devil's Advocate critique started - connect to WebSocket for real-time updates",
                "websocket_url": f"/ws/strategic-trio/{orchestration_id}",
                "independent_critiques": True,
                "synthesis_avoided": True,
            }
        else:
            # Execute critique synchronously
            critique_result = await orchestrator.request_devils_advocate_critique(
                orchestration_result, request.business_context, stream_updates=False
            )

            return {
                "status": "critique_completed",
                "orchestration_id": orchestration_id,
                "consultant_critiques": {
                    consultant.value: {
                        "challenges_found": (
                            result.comprehensive_challenge_result.total_challenges_found
                            if result.comprehensive_challenge_result
                            else 0
                        ),
                        "risk_score": (
                            result.comprehensive_challenge_result.overall_risk_score
                            if result.comprehensive_challenge_result
                            else 0.0
                        ),
                        "processing_time": result.processing_time_seconds,
                    }
                    for consultant, result in critique_result.consultant_critiques.items()
                },
                "critique_summary": critique_result.critique_summary,
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Devil's Advocate critique request failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Critique request failed: {str(e)}"
        )


@router.get("/{orchestration_id}/status", response_model=CritiqueStatusResponse)
async def get_orchestration_status(
    orchestration_id: str, user: SupabaseUser = Depends(get_current_user)
):
    """Get current status of Strategic Trio + Devil's Advocate orchestration"""
    if orchestration_id not in active_orchestrations:
        raise HTTPException(status_code=404, detail="Orchestration not found")

    orchestration_result = active_orchestrations[orchestration_id]

    return CritiqueStatusResponse(
        orchestration_id=orchestration_id,
        status=_get_orchestration_status_string(orchestration_result),
        strategic_trio_completed=orchestration_result.strategic_trio_result is not None,
        critique_requested=orchestration_result.critique_requested,
        critique_in_progress=orchestration_result.critique_in_progress,
        critique_completed=orchestration_result.critique_result is not None,
        consultants_analyzed=orchestration_result.total_consultants_analyzed,
        consultants_critiqued=orchestration_result.total_consultants_critiqued,
        created_at=orchestration_result.created_at.isoformat(),
        updated_at=orchestration_result.updated_at.isoformat(),
    )


@router.get("/{orchestration_id}/results", response_model=CritiqueResultsResponse)
async def get_critique_results(
    orchestration_id: str, user: SupabaseUser = Depends(get_current_user)
):
    """Get completed Devil's Advocate critique results"""
    if orchestration_id not in active_orchestrations:
        raise HTTPException(status_code=404, detail="Orchestration not found")

    orchestration_result = active_orchestrations[orchestration_id]

    if not orchestration_result.critique_result:
        raise HTTPException(
            status_code=400, detail="Devil's Advocate critique not completed"
        )

    critique_result = orchestration_result.critique_result

    # Format consultant critiques for response
    consultant_critiques = {}
    for consultant, result in critique_result.consultant_critiques.items():
        consultant_critiques[consultant.value] = {
            "original_analysis_length": len(result.original_analysis),
            "processing_time_seconds": result.processing_time_seconds,
            "challenges_by_engine": result.challenges_by_engine,
            "consultant_specific_insights": result.consultant_specific_insights,
            "comprehensive_result": (
                {
                    "total_challenges_found": (
                        result.comprehensive_challenge_result.total_challenges_found
                        if result.comprehensive_challenge_result
                        else 0
                    ),
                    "overall_risk_score": (
                        result.comprehensive_challenge_result.overall_risk_score
                        if result.comprehensive_challenge_result
                        else 0.0
                    ),
                    "intellectual_honesty_score": (
                        result.comprehensive_challenge_result.intellectual_honesty_score
                        if result.comprehensive_challenge_result
                        else 1.0
                    ),
                    "system_confidence": (
                        result.comprehensive_challenge_result.system_confidence
                        if result.comprehensive_challenge_result
                        else 0.5
                    ),
                    "refined_recommendation": (
                        result.comprehensive_challenge_result.refined_recommendation
                        if result.comprehensive_challenge_result
                        else ""
                    ),
                }
                if result.comprehensive_challenge_result
                else None
            ),
            "highest_risk_challenge": (
                {
                    "challenge_text": result.highest_risk_challenge.challenge_text,
                    "severity": result.highest_risk_challenge.severity,
                    "source_engine": result.highest_risk_challenge.source_engine,
                    "mitigation_strategy": result.highest_risk_challenge.mitigation_strategy,
                }
                if result.highest_risk_challenge
                else None
            ),
        }

    return CritiqueResultsResponse(
        orchestration_id=orchestration_id,
        consultant_critiques=consultant_critiques,
        critique_summary=critique_result.critique_summary,
        recommended_next_actions=critique_result.recommended_next_actions,
        total_processing_time=critique_result.total_processing_time,
        critiques_completed=critique_result.critiques_completed,
        critiques_failed=critique_result.critiques_failed,
    )


@router.websocket("/ws/{orchestration_id}")
async def websocket_critique_streaming(websocket: WebSocket, orchestration_id: str):
    """
    WebSocket endpoint for real-time Devil's Advocate critique updates
    Streams independent consultant critique results as they complete
    """
    await websocket.accept()

    # Add to active websockets
    if orchestration_id not in active_websockets:
        active_websockets[orchestration_id] = []
    active_websockets[orchestration_id].append(websocket)

    try:
        logger.info(f"📡 WebSocket connected for orchestration {orchestration_id}")

        # Send initial status
        if orchestration_id in active_orchestrations:
            orchestration_result = active_orchestrations[orchestration_id]
            await websocket.send_json(
                {
                    "type": "status",
                    "orchestration_id": orchestration_id,
                    "status": _get_orchestration_status_string(orchestration_result),
                    "connected": True,
                }
            )

        # Keep connection alive and handle incoming messages
        while True:
            try:
                data = await websocket.receive_text()
                # Handle any client messages if needed
                logger.info(f"📡 WebSocket message received: {data}")
            except WebSocketDisconnect:
                break

    except WebSocketDisconnect:
        logger.info(f"📡 WebSocket disconnected for orchestration {orchestration_id}")
    finally:
        # Remove from active websockets
        if orchestration_id in active_websockets:
            active_websockets[orchestration_id].remove(websocket)
            if not active_websockets[orchestration_id]:
                del active_websockets[orchestration_id]


# Background task functions


async def _execute_critique_with_streaming(
    orchestration_result: StrategicTrioCritiqueOrchestrationResult,
    business_context: Optional[Dict[str, Any]],
    orchestration_id: str,
):
    """Execute Devil's Advocate critique with real-time streaming updates"""
    try:
        logger.info(f"🔍 Starting streaming critique execution for {orchestration_id}")

        # Send streaming updates to connected WebSockets
        async def send_streaming_update(update: Dict[str, Any]):
            if orchestration_id in active_websockets:
                disconnected = []
                for websocket in active_websockets[orchestration_id]:
                    try:
                        await websocket.send_json(update)
                    except Exception:
                        disconnected.append(websocket)

                # Clean up disconnected websockets
                for ws in disconnected:
                    active_websockets[orchestration_id].remove(ws)

        # Execute critique with custom streaming
        original_send_method = orchestrator._send_streaming_update

        async def custom_send_streaming_update(
            update: CritiqueStreamingUpdate, request_id: str
        ):
            await original_send_method(update, request_id)
            await send_streaming_update(
                {
                    "type": update.update_type,
                    "consultant_role": (
                        update.consultant_role.value if update.consultant_role else None
                    ),
                    "progress_percent": update.progress_percent,
                    "current_engine": update.current_engine,
                    "timestamp": update.timestamp.isoformat(),
                    "orchestration_id": orchestration_id,
                }
            )

        # Temporarily replace streaming method
        orchestrator._send_streaming_update = custom_send_streaming_update

        # Execute critique
        critique_result = await orchestrator.request_devils_advocate_critique(
            orchestration_result, business_context, stream_updates=True
        )

        # Send final completion update
        await send_streaming_update(
            {
                "type": "critique_completed",
                "orchestration_id": orchestration_id,
                "consultant_critiques": {
                    consultant.value: {
                        "challenges_found": (
                            result.comprehensive_challenge_result.total_challenges_found
                            if result.comprehensive_challenge_result
                            else 0
                        ),
                        "risk_score": (
                            result.comprehensive_challenge_result.overall_risk_score
                            if result.comprehensive_challenge_result
                            else 0.0
                        ),
                    }
                    for consultant, result in critique_result.consultant_critiques.items()
                },
                "critique_summary": critique_result.critique_summary,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        logger.info(f"✅ Streaming critique completed for {orchestration_id}")

    except Exception as e:
        logger.error(f"❌ Streaming critique failed for {orchestration_id}: {e}")

        # Send error update
        if orchestration_id in active_websockets:
            for websocket in active_websockets[orchestration_id]:
                try:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "orchestration_id": orchestration_id,
                            "error": str(e),
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    )
                except Exception:
                    pass


# Utility functions


def _get_orchestration_status_string(
    orchestration_result: StrategicTrioCritiqueOrchestrationResult,
) -> str:
    """Get human-readable status string for orchestration"""
    if not orchestration_result.strategic_trio_result:
        return "strategic_trio_pending"
    elif not orchestration_result.critique_requested:
        return "strategic_trio_completed"
    elif orchestration_result.critique_in_progress:
        return "critique_in_progress"
    elif orchestration_result.critique_result:
        return "critique_completed"
    else:
        return "critique_requested"


@router.delete("/{orchestration_id}")
async def cleanup_orchestration(
    orchestration_id: str, user: SupabaseUser = Depends(get_current_user)
):
    """Clean up orchestration resources"""
    if orchestration_id in active_orchestrations:
        del active_orchestrations[orchestration_id]

    if orchestration_id in active_websockets:
        # Close all websockets
        for websocket in active_websockets[orchestration_id]:
            try:
                await websocket.close()
            except Exception:
                pass
        del active_websockets[orchestration_id]

    return {"status": "cleaned_up", "orchestration_id": orchestration_id}
