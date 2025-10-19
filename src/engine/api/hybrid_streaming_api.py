"""
Hybrid streaming API for real-time cognitive analysis.

Integrates the hybrid orchestrator with the existing transparency stream manager
to provide real-time updates during cognitive processing.
"""

import asyncio
import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional, AsyncGenerator
from uuid import UUID
import logging

from fastapi import APIRouter, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.core.hybrid_session_manager import (
    get_hybrid_session_manager,
)
from src.engine.api.transparency_stream_manager import get_transparency_stream_manager
from src.core.auth_foundation import get_current_user

logger = logging.getLogger(__name__)


# Request/Response Models
class HybridEngagementRequest(BaseModel):
    """Request model for hybrid engagement creation"""

    query: str
    user_id: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    stream_results: bool = True


class HybridEngagementResponse(BaseModel):
    """Response model for hybrid engagement"""

    session_id: str
    engagement_id: str
    status: str
    streaming_url: Optional[str] = None
    initial_query: str
    created_at: str


class HITLRequest(BaseModel):
    """Request model for HITL interaction"""

    session_id: str
    request_id: str
    action: str  # approve, modify, skip, pause
    modifications: Optional[Dict[str, Any]] = None
    feedback: Optional[str] = None


# Router setup
hybrid_router = APIRouter(prefix="/api/hybrid", tags=["Hybrid Cognitive Analysis"])


class HybridStreamingAPI:
    """Hybrid streaming API for real-time cognitive analysis"""

    def __init__(self):
        self.session_manager = get_hybrid_session_manager()
        self.stream_manager = get_transparency_stream_manager()
        self.active_streams: Dict[str, WebSocket] = {}

    async def create_hybrid_engagement(
        self, request: HybridEngagementRequest, current_user: Optional[Dict] = None
    ) -> HybridEngagementResponse:
        """Create new hybrid cognitive engagement with optional streaming"""

        try:
            # Create hybrid session
            session_result = await self.session_manager.create_hybrid_session(
                initial_query=request.query,
                user_id=(
                    request.user_id or current_user.get("id") if current_user else None
                ),
                session_config=request.config,
            )

            # Determine streaming URL
            streaming_url = None
            if request.stream_results:
                streaming_url = f"/api/hybrid/stream/{session_result['session_id']}"

            return HybridEngagementResponse(
                session_id=session_result["session_id"],
                engagement_id=session_result["engagement_id"],
                status=session_result["status"],
                streaming_url=streaming_url,
                initial_query=request.query,
                created_at=session_result["created_at"],
            )

        except Exception as e:
            logger.error(f"Failed to create hybrid engagement: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Engagement creation failed: {str(e)}"
            )

    async def execute_hybrid_engagement(self, session_id: str) -> Dict[str, Any]:
        """Execute hybrid engagement with full orchestration"""

        try:
            # Execute the session
            execution_result = await self.session_manager.execute_hybrid_session(
                session_id
            )

            if execution_result["execution_success"]:
                return {
                    "success": True,
                    "session_id": session_id,
                    "engagement_id": execution_result["engagement_id"],
                    "contract": execution_result["contract"],
                    "metrics": execution_result["metrics"],
                }
            else:
                return {
                    "success": False,
                    "session_id": session_id,
                    "error": execution_result["error"],
                    "recovery_result": execution_result["recovery_result"],
                }

        except Exception as e:
            logger.error(f"Hybrid engagement execution failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Execution failed: {str(e)}")

    async def stream_hybrid_engagement(
        self, session_id: str
    ) -> AsyncGenerator[str, None]:
        """Stream hybrid engagement progress in real-time"""

        try:
            # Get session status
            session_status = await self.session_manager.get_session_status(session_id)
            if session_status["status"] == "not_found":
                yield f"data: {json.dumps({'error': 'Session not found'})}\n\n"
                return

            # Stream initial status
            yield f"data: {json.dumps({'type': 'session_status', 'data': session_status})}\n\n"

            # Start session execution in background
            execution_task = asyncio.create_task(
                self.session_manager.execute_hybrid_session(session_id)
            )

            # Stream progress updates
            last_event_count = 0
            while not execution_task.done():
                try:
                    # Get current session state
                    current_status = await self.session_manager.get_session_status(
                        session_id
                    )

                    # Stream status updates
                    if current_status["status"] != session_status.get("status"):
                        session_status = current_status
                        yield f"data: {json.dumps({'type': 'status_change', 'data': current_status})}\n\n"

                    # Stream metrics updates
                    if "metrics" in current_status and current_status[
                        "metrics"
                    ] != session_status.get("metrics", {}):
                        yield f"data: {json.dumps({'type': 'metrics_update', 'data': current_status['metrics']})}\n\n"

                    # Wait before next update
                    await asyncio.sleep(1.0)

                except Exception as stream_error:
                    logger.warning(
                        f"Streaming error for session {session_id}: {str(stream_error)}"
                    )
                    yield f"data: {json.dumps({'type': 'stream_error', 'error': str(stream_error)})}\n\n"

            # Stream final results
            try:
                execution_result = await execution_task
                yield f"data: {json.dumps({'type': 'execution_completed', 'data': execution_result})}\n\n"

            except Exception as execution_error:
                yield f"data: {json.dumps({'type': 'execution_error', 'error': str(execution_error)})}\n\n"

        except Exception as e:
            logger.error(f"Streaming failed for session {session_id}: {str(e)}")
            yield f"data: {json.dumps({'type': 'fatal_error', 'error': str(e)})}\n\n"

    async def handle_hitl_interaction(
        self, hitl_request: HITLRequest
    ) -> Dict[str, Any]:
        """Handle human-in-the-loop interaction"""

        try:
            if hitl_request.action in ["approve", "modify"]:
                # Resume session with response
                resume_result = await self.session_manager.resume_session_from_hitl(
                    hitl_request.session_id,
                    {
                        "request_id": hitl_request.request_id,
                        "action": hitl_request.action,
                        "modifications": hitl_request.modifications,
                        "feedback": hitl_request.feedback,
                    },
                )

                return {
                    "success": True,
                    "action": hitl_request.action,
                    "resume_result": resume_result,
                }

            elif hitl_request.action == "pause":
                # Keep session paused
                return {
                    "success": True,
                    "action": "pause",
                    "message": "Session remains paused",
                }

            else:
                return {
                    "success": False,
                    "error": f"Unknown HITL action: {hitl_request.action}",
                }

        except Exception as e:
            logger.error(f"HITL interaction failed: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"HITL interaction failed: {str(e)}"
            )

    async def get_session_details(self, session_id: str) -> Dict[str, Any]:
        """Get detailed session information"""

        session_status = await self.session_manager.get_session_status(session_id)

        if session_status["status"] == "not_found":
            raise HTTPException(status_code=404, detail="Session not found")

        return session_status

    async def list_active_sessions(self) -> Dict[str, Any]:
        """List all active sessions"""

        active_sessions = await self.session_manager.list_active_sessions()

        return {
            "active_sessions": active_sessions,
            "total_count": len(active_sessions),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# API instance
hybrid_streaming_api = HybridStreamingAPI()


# Route definitions
@hybrid_router.post("/create", response_model=HybridEngagementResponse)
async def create_hybrid_engagement(
    request: HybridEngagementRequest,
    current_user: Optional[Dict] = Depends(get_current_user),
) -> HybridEngagementResponse:
    """Create a new hybrid cognitive engagement"""
    return await hybrid_streaming_api.create_hybrid_engagement(request, current_user)


@hybrid_router.post("/execute/{session_id}")
async def execute_hybrid_engagement(session_id: str) -> Dict[str, Any]:
    """Execute a hybrid cognitive engagement"""
    return await hybrid_streaming_api.execute_hybrid_engagement(session_id)


@hybrid_router.get("/stream/{session_id}")
async def stream_hybrid_engagement(session_id: str) -> StreamingResponse:
    """Stream hybrid engagement progress"""
    return StreamingResponse(
        hybrid_streaming_api.stream_hybrid_engagement(session_id),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        },
    )


@hybrid_router.post("/hitl")
async def handle_hitl_interaction(hitl_request: HITLRequest) -> Dict[str, Any]:
    """Handle human-in-the-loop interaction"""
    return await hybrid_streaming_api.handle_hitl_interaction(hitl_request)


@hybrid_router.get("/session/{session_id}")
async def get_session_details(session_id: str) -> Dict[str, Any]:
    """Get session details"""
    return await hybrid_streaming_api.get_session_details(session_id)


@hybrid_router.get("/sessions/active")
async def list_active_sessions() -> Dict[str, Any]:
    """List active sessions"""
    return await hybrid_streaming_api.list_active_sessions()


# WebSocket endpoint for real-time bidirectional communication
@hybrid_router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time hybrid engagement communication"""

    await websocket.accept()
    hybrid_streaming_api.active_streams[session_id] = websocket

    try:
        # Send initial session status
        session_status = await hybrid_streaming_api.get_session_details(session_id)
        await websocket.send_json({"type": "session_connected", "data": session_status})

        # Listen for messages
        while True:
            try:
                # Wait for message with timeout
                message = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)

                if message.get("type") == "execute_session":
                    # Start session execution
                    execution_task = asyncio.create_task(
                        hybrid_streaming_api.execute_hybrid_engagement(session_id)
                    )

                    # Stream progress
                    while not execution_task.done():
                        current_status = await hybrid_streaming_api.get_session_details(
                            session_id
                        )
                        await websocket.send_json(
                            {"type": "progress_update", "data": current_status}
                        )
                        await asyncio.sleep(2.0)

                    # Send final results
                    execution_result = await execution_task
                    await websocket.send_json(
                        {"type": "execution_completed", "data": execution_result}
                    )

                elif message.get("type") == "hitl_response":
                    # Handle HITL response
                    hitl_request = HITLRequest(**message.get("data", {}))
                    hitl_result = await hybrid_streaming_api.handle_hitl_interaction(
                        hitl_request
                    )

                    await websocket.send_json(
                        {"type": "hitl_processed", "data": hitl_result}
                    )

                elif message.get("type") == "get_status":
                    # Send current status
                    current_status = await hybrid_streaming_api.get_session_details(
                        session_id
                    )
                    await websocket.send_json(
                        {"type": "status_update", "data": current_status}
                    )

                else:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "error": f"Unknown message type: {message.get('type')}",
                        }
                    )

            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                await websocket.send_json({"type": "ping"})

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {str(e)}")
        await websocket.send_json({"type": "error", "error": str(e)})
    finally:
        if session_id in hybrid_streaming_api.active_streams:
            del hybrid_streaming_api.active_streams[session_id]


# Enhanced streaming with transparency integration
class HybridTransparencyStreamer:
    """Enhanced transparency streaming for hybrid engagements"""

    def __init__(self):
        self.stream_manager = get_transparency_stream_manager()
        self.session_manager = get_hybrid_session_manager()

    async def stream_with_transparency(
        self, session_id: str
    ) -> AsyncGenerator[str, None]:
        """Stream with full transparency integration"""

        try:
            # Get session info
            session_status = await self.session_manager.get_session_status(session_id)
            engagement_id = UUID(session_status.get("engagement_id", ""))

            # Initialize transparency state
            transparency_state = (
                await self.stream_manager.initialize_transparency_state(
                    engagement_id, session_status.get("initial_query", "")
                )
            )

            # Stream transparency initialization
            yield f"data: {json.dumps({'type': 'transparency_initialized', 'data': transparency_state})}\n\n"

            # Execute with transparency streaming
            execution_task = asyncio.create_task(
                self.session_manager.execute_hybrid_session(session_id)
            )

            # Stream transparency updates
            while not execution_task.done():
                try:
                    # Get transparency updates
                    transparency_updates = (
                        await self.stream_manager.get_transparency_updates(
                            engagement_id
                        )
                    )

                    if transparency_updates:
                        yield f"data: {json.dumps({'type': 'transparency_update', 'data': transparency_updates})}\n\n"

                    await asyncio.sleep(1.5)

                except Exception as transparency_error:
                    logger.warning(
                        f"Transparency streaming error: {str(transparency_error)}"
                    )

            # Stream final transparency
            execution_result = await execution_task
            final_transparency = await self.stream_manager.finalize_transparency(
                engagement_id, execution_result
            )

            yield f"data: {json.dumps({'type': 'transparency_finalized', 'data': final_transparency})}\n\n"

        except Exception as e:
            logger.error(f"Transparency streaming failed: {str(e)}")
            yield f"data: {json.dumps({'type': 'transparency_error', 'error': str(e)})}\n\n"


# Transparency streaming router
transparency_streamer = HybridTransparencyStreamer()


@hybrid_router.get("/transparency/stream/{session_id}")
async def stream_with_transparency(session_id: str) -> StreamingResponse:
    """Stream with full transparency integration"""
    return StreamingResponse(
        transparency_streamer.stream_with_transparency(session_id),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        },
    )
