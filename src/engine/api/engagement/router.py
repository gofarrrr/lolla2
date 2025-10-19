"""
FastAPI router for METIS Engagement API
Refactored from standalone app to APIRouter for Operation: Bedrock
"""

import logging
from typing import List, Dict, Any, Optional
from uuid import UUID
from datetime import datetime

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from .models import (
    EngagementRequest,
    EngagementResponse,
    EngagementPhase,
    PhaseResult,
    DeliverableRequest,
    ReevaluationRequest,
    ClarificationRequest,
    ClarificationResult,
    ClarificationResponseRequest,
    ClarificationSkipRequest,
    EnhancedQueryResult,
    ProblemStatement,
    # NEW: V2 Tiered Clarification Models
    TieredClarificationStartRequest,
    TieredClarificationStartResponse,
    TieredClarificationContinueRequest,
    TieredClarificationContinueResponse,
    CreateEngagementFromClarificationRequest,
    CreateEngagementFromClarificationResponse,
)
from .orchestrator import EngagementOrchestrator
from .clarification import ClarificationHandler, TieredClarificationHandler
from .sandbox import WhatIfSandbox

# Operation Crystal Day 1: Import comparison functionality
from .comparison import EngagementComparator

logger = logging.getLogger(__name__)

# Initialize components
orchestrator = EngagementOrchestrator()
clarification_handler = ClarificationHandler()
# NEW: V2 Tiered Clarification Handler
tiered_clarification_handler = TieredClarificationHandler()
sandbox = WhatIfSandbox(orchestrator)
# Operation Crystal Day 1: Initialize engagement comparator
comparator = EngagementComparator()

# Create APIRouter instead of FastAPI app
router = APIRouter(
    prefix="/api/v1/engagements",
    tags=["Engagements"]
)

# Additional routers for different API versions and features
v2_router = APIRouter(
    prefix="/api/v2/engagements", 
    tags=["Engagements V2"]
)

clarification_router = APIRouter(
    prefix="/api/v1/clarification",
    tags=["HITL Clarification"]
)

v2_clarification_router = APIRouter(
    prefix="/api/v2/clarification",
    tags=["V2 Tiered Clarification"]
)

health_router = APIRouter(tags=["Health"])

transparency_router = APIRouter(
    prefix="/api/v1/engagements",
    tags=["Transparency"]
)

# Startup event handler functions
async def initialize_engines():
    """Initialize engines on startup"""
    await orchestrator.initialize_engines()

async def shutdown_event():
    """Cleanup on shutdown"""
    if hasattr(orchestrator, "metis_system") and orchestrator.metis_system:
        await orchestrator.metis_system.shutdown()
    elif hasattr(orchestrator, "event_bus") and orchestrator.event_bus:
        await orchestrator.event_bus.shutdown()


# Core Engagement Endpoints
@router.post(
    "", response_model=EngagementResponse
)
async def create_engagement(request: EngagementRequest) -> EngagementResponse:
    """Create a new consulting engagement"""
    return await orchestrator.create_engagement(request)


@router.get(
    "", response_model=List[EngagementResponse]
)
async def list_engagements(user_id: Optional[str] = None) -> List[EngagementResponse]:
    """List all engagements, optionally filtered by user"""
    # Graceful fallback for environments where list_engagements is not implemented
    if not hasattr(orchestrator, "list_engagements"):
        return []
    return await orchestrator.list_engagements(user_id=user_id)


@router.get(
    "/{engagement_id}",
    response_model=EngagementResponse,
)
async def get_engagement(engagement_id: UUID) -> EngagementResponse:
    """Get specific engagement"""
    engagement = await orchestrator.get_engagement(engagement_id)
    if not engagement:
        raise HTTPException(status_code=404, detail="Engagement not found")
    return engagement


# User-specific engagements endpoint
@APIRouter(prefix="/api/v1/users", tags=["User Engagements"]).get(
    "/{user_id}/engagements",
    response_model=List[EngagementResponse],
)
async def list_user_engagements(user_id: str) -> List[EngagementResponse]:
    """List all engagements for a specific user (Dashboard API)"""
    return await orchestrator.list_engagements(user_id=user_id)


# Phase Execution Endpoints
@router.post(
    "/{engagement_id}/phases/{phase}/execute",
    response_model=PhaseResult,
)
async def execute_phase(engagement_id: UUID, phase: EngagementPhase) -> PhaseResult:
    """Execute specific engagement phase"""
    return await orchestrator.execute_phase(engagement_id, phase)


@router.get(
    "/{engagement_id}/phases/{phase}",
    response_model=PhaseResult,
)
async def get_phase_result(engagement_id: UUID, phase: EngagementPhase) -> PhaseResult:
    """Get results for specific phase"""
    engagement = await orchestrator.get_engagement(engagement_id)
    if not engagement:
        raise HTTPException(status_code=404, detail="Engagement not found")

    if phase.value not in engagement.phases:
        raise HTTPException(status_code=404, detail="Phase not executed")

    return engagement.phases[phase.value]


# Deliverable Generation
@router.post("/{engagement_id}/deliverable", tags=["Deliverables"])
async def generate_deliverable(engagement_id: UUID, request: DeliverableRequest):
    """Generate client deliverable in requested format"""
    engagement = await orchestrator.get_engagement(engagement_id)
    if not engagement:
        raise HTTPException(status_code=404, detail="Engagement not found")

    if not engagement.deliverable_ready:
        raise HTTPException(status_code=400, detail="Engagement not complete")

    # Mock deliverable generation for now
    return {
        "status": "success",
        "format": request.format,
        "template": request.template,
        "download_url": f"/api/v1/engagements/{engagement_id}/deliverable/download",
        "generated_at": "2024-01-01T00:00:00Z",
    }


# What-If Sandbox Endpoints
@router.post("/{engagement_id}/what-if", tags=["What-If Analysis"])
async def reevaluate_engagement(engagement_id: UUID, request: ReevaluationRequest):
    """Re-evaluate engagement with changed assumptions (What-If analysis)"""
    try:
        result = await sandbox.process_reevaluation_request(engagement_id, request)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"What-if analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket endpoints
@router.websocket("/{engagement_id}/ws")
async def engagement_websocket(websocket: WebSocket, engagement_id: str):
    """WebSocket endpoint for real-time engagement updates"""
    await orchestrator.connection_manager.connect(websocket, engagement_id)
    try:
        while True:
            # Keep connection alive and handle any incoming messages
            await websocket.receive_text()
    except WebSocketDisconnect:
        orchestrator.connection_manager.disconnect(engagement_id)


# Operation Crystal Day 1: Engagement Comparison Endpoints
@v2_router.post("/compare", tags=["Engagement Comparison"])
async def compare_engagements(engagement1_id: UUID, engagement2_id: UUID):
    """
    Compare findings and recommendations between two engagements.

    Compares high-level differences: Governing Thought, Key Recommendations,
    and final confidence scores. Supports both completed engagements and
    temporary What-If forks.
    """
    try:
        logger.info(f"📊 Comparing engagements {engagement1_id} vs {engagement2_id}")

        # Retrieve first engagement (check both permanent contracts and temp forks)
        engagement1_data = None
        if engagement1_id in orchestrator.contracts:
            engagement1_data = orchestrator.contracts[
                engagement1_id
            ].to_cloudevents_dict()
        else:
            # Check if it's a temp fork
            try:
                fork_data = sandbox.get_temp_fork(engagement1_id)
                engagement1_data = fork_data["contract"].to_cloudevents_dict()
            except HTTPException:
                pass

        if not engagement1_data:
            raise HTTPException(
                status_code=404, detail=f"Engagement {engagement1_id} not found"
            )

        # Retrieve second engagement (check both permanent contracts and temp forks)
        engagement2_data = None
        if engagement2_id in orchestrator.contracts:
            engagement2_data = orchestrator.contracts[
                engagement2_id
            ].to_cloudevents_dict()
        else:
            # Check if it's a temp fork
            try:
                fork_data = sandbox.get_temp_fork(engagement2_id)
                engagement2_data = fork_data["contract"].to_cloudevents_dict()
            except HTTPException:
                pass

        if not engagement2_data:
            raise HTTPException(
                status_code=404, detail=f"Engagement {engagement2_id} not found"
            )

        # Perform comparison
        comparison_result = comparator.compare_engagements(
            engagement1_data=engagement1_data,
            engagement2_data=engagement2_data,
            engagement1_id=engagement1_id,
            engagement2_id=engagement2_id,
        )

        # Return formatted response
        return JSONResponse(content=comparator.to_api_response(comparison_result))

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"❌ Engagement comparison failed: {e}")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


@v2_router.get("/{engagement_id}/cost-estimate", tags=["Cost Estimation"])
async def get_rerun_cost_estimate(engagement_id: UUID, changes: str):
    """
    Get cost estimate for re-running an engagement with changes.

    Args:
        engagement_id: The engagement to re-run
        changes: JSON string of changes (e.g., '{"budget": 50000, "timeline": "6 months"}')

    Returns cost estimate without actually executing the re-run.
    """
    try:
        import json

        changes_dict = json.loads(changes)

        cost_estimate = await sandbox.get_rerun_cost_estimate(
            engagement_id, changes_dict
        )
        return JSONResponse(content=cost_estimate)

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid changes JSON format")
    except Exception as e:
        logger.error(f"💰 Cost estimation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cost estimation failed: {str(e)}")


@v2_router.get("/{engagement_id}/rerun-status", tags=["Re-run Limits"])
async def get_rerun_status(engagement_id: UUID):
    """
    Get re-run status and limits for an engagement.

    Returns current count, remaining reruns, and warnings for the UI.
    """
    try:
        status = sandbox.get_rerun_status(engagement_id)
        return JSONResponse(content=status)
    except Exception as e:
        logger.error(f"🔢 Rerun status check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


# Operation Crystal Day 3: Scenario Promotion Endpoints
@v2_router.post("/{temp_id}/promote", tags=["Scenario Promotion"])
async def promote_scenario(temp_id: UUID, scenario_name: str):
    """
    Promote a temporary What-If fork to a permanent engagement.

    Args:
        temp_id: UUID of the temporary fork to promote
        scenario_name: User-provided name for the scenario (e.g., "Budget Cut Scenario")

    Creates a permanent engagement with parent-child relationship maintained.
    """
    try:
        logger.info(
            f"🎯 Promoting What-If scenario {temp_id} with name '{scenario_name}'"
        )

        # Validate scenario name
        if not scenario_name or len(scenario_name.strip()) < 3:
            raise HTTPException(
                status_code=400,
                detail="Scenario name must be at least 3 characters long",
            )

        if len(scenario_name) > 100:
            raise HTTPException(
                status_code=400, detail="Scenario name must be less than 100 characters"
            )

        # Promote the scenario
        promotion_details = await sandbox.promote_scenario(
            temp_id, scenario_name.strip()
        )

        return JSONResponse(content=promotion_details)

    except HTTPException:
        # Re-raise HTTP exceptions (like 404 for temp not found)
        raise
    except Exception as e:
        logger.error(f"🎯 Scenario promotion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Promotion failed: {str(e)}")


@v2_router.get("/promoted-scenarios", tags=["Scenario Promotion"])
async def list_promoted_scenarios(parent_id: Optional[UUID] = None):
    """
    List all promoted What-If scenarios.

    Args:
        parent_id: Optional parent engagement ID to filter scenarios

    Returns list of promoted scenarios with their metadata.
    """
    try:
        scenarios = sandbox.list_promoted_scenarios(parent_id)

        response = {
            "promoted_scenarios": scenarios,
            "total_count": len(scenarios),
            "filtered_by_parent": str(parent_id) if parent_id else None,
            "retrieved_at": datetime.utcnow().isoformat(),
        }

        return JSONResponse(content=response)

    except Exception as e:
        logger.error(f"📋 Failed to list promoted scenarios: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve scenarios: {str(e)}"
        )


@v2_router.get("/{engagement_id}/scenarios", tags=["Scenario Promotion"])
async def get_engagement_scenarios(engagement_id: UUID):
    """
    Get all What-If scenarios (both temporary and promoted) for an engagement.

    Returns both active temporary forks and permanently promoted scenarios.
    """
    try:
        # Get temporary forks
        active_forks = sandbox.list_active_forks(engagement_id)

        # Get promoted scenarios
        promoted_scenarios = sandbox.list_promoted_scenarios(engagement_id)

        response = {
            "parent_engagement_id": str(engagement_id),
            "active_temp_forks": {"count": len(active_forks), "forks": active_forks},
            "promoted_scenarios": {
                "count": len(promoted_scenarios),
                "scenarios": promoted_scenarios,
            },
            "total_scenarios": len(active_forks) + len(promoted_scenarios),
            "retrieved_at": datetime.utcnow().isoformat(),
        }

        return JSONResponse(content=response)

    except Exception as e:
        logger.error(f"📊 Failed to get engagement scenarios: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve scenarios: {str(e)}"
        )


# HITL Clarification Endpoints
@clarification_router.post(
    "/analyze",
    response_model=ClarificationResult,
)
async def analyze_clarification_needs(
    request: ClarificationRequest,
) -> ClarificationResult:
    """Analyze if a query needs clarification and generate questions"""
    return await clarification_handler.analyze_clarification_needs(request)


@clarification_router.post(
    "/respond",
    response_model=EnhancedQueryResult,
)
async def process_clarification_responses(
    request: ClarificationResponseRequest,
) -> EnhancedQueryResult:
    """Process user responses to clarification questions"""
    return await clarification_handler.process_clarification_responses(request)


@clarification_router.post(
    "/skip",
    response_model=EnhancedQueryResult,
)
async def skip_clarification(request: ClarificationSkipRequest) -> EnhancedQueryResult:
    """Skip clarification and proceed with original query"""
    return await clarification_handler.skip_clarification(request)


@clarification_router.get("/{session_id}/status")
async def get_clarification_status(session_id: str) -> Dict[str, Any]:
    """Get clarification session status"""
    return await clarification_handler.get_session_status(session_id)


@clarification_router.post(
    "/{session_id}/create-engagement",
    response_model=EngagementResponse,
)
async def create_engagement_from_clarification(
    session_id: str,
    client_name: str,
    engagement_type: str = "strategy_consulting",
    priority: str = "medium",
) -> EngagementResponse:
    """Create engagement using clarified query"""

    # Get session status
    session_status = await clarification_handler.get_session_status(session_id)
    if not session_status:
        raise HTTPException(status_code=404, detail="Clarification session not found")

    if session_status.get("state") not in ["completed", "skipped"]:
        raise HTTPException(
            status_code=400,
            detail=f"Clarification session not ready: {session_status.get('state')}",
        )

    # For simulation mode, create a basic engagement
    enhanced_query = session_status.get(
        "enhanced_query", "Enhanced query from clarification"
    )

    enhanced_request = EngagementRequest(
        problem_statement=ProblemStatement(
            problem_description=enhanced_query,
            business_context=session_status.get("business_context", {}),
            stakeholders=[],
            success_criteria=[],
        ),
        client_name=client_name,
        engagement_type=engagement_type,
        priority=priority,
    )

    return await orchestrator.create_engagement(enhanced_request)


# NEW: V2 Tiered Clarification Endpoints
@v2_clarification_router.post(
    "/start",
    response_model=TieredClarificationStartResponse,
)
async def start_tiered_clarification(
    request: TieredClarificationStartRequest,
) -> TieredClarificationStartResponse:
    """
    NEW: Start tiered clarification process with engagement brief and essential questions

    This endpoint fixes the "wall of text" problem by:
    1. Generating a concise engagement brief to prove understanding
    2. Presenting only essential questions (3-5 max)
    3. Using Perplexity research to make questions contextually relevant
    """
    return await tiered_clarification_handler.start_tiered_clarification(request)


@v2_clarification_router.post(
    "/{session_id}/continue",
    response_model=TieredClarificationContinueResponse,
)
async def continue_with_expert_questions(
    session_id: str, request: TieredClarificationContinueRequest
) -> TieredClarificationContinueResponse:
    """
    NEW: Continue with expert questions after essential answers received

    This endpoint enables progressive disclosure by:
    1. Processing essential question answers
    2. Offering expert-level questions for power users
    3. Allowing users to skip expert questions and proceed
    """
    # Ensure session_id matches request
    request.clarification_session_id = session_id
    return await tiered_clarification_handler.continue_with_expert_questions(request)


@v2_router.post(
    "/create_from_clarification",
    response_model=CreateEngagementFromClarificationResponse,
)
async def create_engagement_from_tiered_clarification(
    request: CreateEngagementFromClarificationRequest,
) -> CreateEngagementFromClarificationResponse:
    """
    NEW: Create final engagement from tiered clarification session

    This endpoint completes the conversational flow by:
    1. Collecting all user answers (essential + optional expert)
    2. Building enhanced query from engagement brief + answers
    3. Creating the final engagement for processing
    """
    return await tiered_clarification_handler.create_engagement_from_clarification(
        request
    )


# Transparency endpoints
@transparency_router.get("/{engagement_id}/transparency")
async def get_transparency_data(engagement_id: UUID):
    """Get captured transparency data for an engagement"""
    data = await orchestrator.get_captured_data(engagement_id)
    if not data:
        raise HTTPException(status_code=404, detail="No transparency data found")
    return data


# User engagements router (separate to handle different prefix)
user_router = APIRouter(
    prefix="/api/v1/users",
    tags=["User Engagements"]
)

@user_router.get(
    "/{user_id}/engagements",
    response_model=List[EngagementResponse],
)
async def list_user_engagements_route(user_id: str) -> List[EngagementResponse]:
    """List all engagements for a specific user (Dashboard API)"""
    if not hasattr(orchestrator, "list_engagements"):
        return []
    return await orchestrator.list_engagements(user_id=user_id)


# Export all routers for main app registration
__all__ = [
    "router",
    "v2_router", 
    "clarification_router",
    "v2_clarification_router",
    "transparency_router",
    "user_router",
    "initialize_engines",
    "shutdown_event"
]