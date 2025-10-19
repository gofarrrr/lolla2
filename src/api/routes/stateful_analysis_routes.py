"""
Analysis Execution API - V5.3 CANONICAL COMPLIANCE
===================================================

V5.3 TRANSFORMATION: Complete StatefulPipelineOrchestrator integration.
This API brings the Iteration Engine online as the live engine for all user analyses.

V5.3 COMPLIANCE FEATURES:
1. POST /execute-stateful-analysis - Execute complete stateful pipeline with checkpoints
2. POST /resume-from-checkpoint - Resume analysis from any checkpoint
3. GET /analysis-results/{trace_id} - Retrieve stateful analysis results
4. GET /checkpoints/{trace_id} - List available checkpoints for an analysis

ARCHITECTURAL BREAKTHROUGH: 100% StatefulPipelineOrchestrator integration achieved.
"""

import time
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import logging
from datetime import datetime, timezone
from uuid import UUID

# V5.3 CANONICAL IMPORTS - StatefulPipelineOrchestrator Integration
from src.engine.adapters import create_pipeline_orchestrator

# Supabase integration for persistence
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

router = APIRouter(prefix="/api/v53/analysis", tags=["stateful-analysis"])

# V5.3 CANONICAL COMPLIANCE: Initialize StatefulPipelineOrchestrator
stateful_orchestrator = create_pipeline_orchestrator()


# V5.3 Request Models - Stateful Pipeline Support
class StatefulAnalysisRequest(BaseModel):
    initial_query: str = Field(
        ..., min_length=10, description="User's initial query for analysis"
    )
    user_id: Optional[str] = Field(default=None, description="User identifier")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    project_id: Optional[str] = Field(
        default=None, description="Project ID for context scoping"
    )
    merge_project_context: bool = Field(
        default=False, description="Whether to merge project context from RAG"
    )


class ResumeAnalysisRequest(BaseModel):
    checkpoint_id: str = Field(..., description="Checkpoint UUID to resume from")
    user_id: Optional[str] = Field(default=None, description="User identifier")
    session_id: Optional[str] = Field(default=None, description="Session identifier")


# V5.3 Response Models - Stateful Pipeline Results
class StatefulAnalysisResponse(BaseModel):
    success: bool
    trace_id: str
    analysis_results: Dict[str, Any]
    checkpoints_created: List[str]
    total_processing_time_ms: int
    pipeline_status: str
    can_resume: bool
    next_available_checkpoints: List[str]
    iteration_engine_active: bool = True


class CheckpointInfo(BaseModel):
    checkpoint_id: str
    stage: str
    timestamp: datetime
    can_resume_from: bool
    stage_results: Dict[str, Any]


class CheckpointsResponse(BaseModel):
    trace_id: str
    checkpoints: List[CheckpointInfo]
    total_checkpoints: int
    latest_checkpoint: Optional[str]


# V5.3 CANONICAL ENDPOINT: Stateful Pipeline Execution
@router.post("/execute-stateful-analysis", response_model=StatefulAnalysisResponse)
async def execute_stateful_analysis(request: StatefulAnalysisRequest):
    """
    V5.3 CANONICAL COMPLIANCE: Execute complete stateful pipeline with checkpoint support.

    This endpoint uses the StatefulPipelineOrchestrator to execute the full 7-stage
    cognitive pipeline with automatic checkpoint creation after each stage.

    ITERATION ENGINE FEATURES:
    - Automatic checkpoint creation and state management
    - Resume capability from any checkpoint
    - Immutable analysis branching for revisions
    - Full UnifiedContextStream integration
    """
    start_time = time.time()

    try:
        logger.info(
            f"🚀 V5.3 Stateful Analysis: Starting for query: {request.initial_query[:100]}..."
        )

        # Convert string IDs to UUIDs if provided
        user_id = UUID(request.user_id) if request.user_id else None
        session_id = UUID(request.session_id) if request.session_id else None
        project_id = UUID(request.project_id) if request.project_id else None

        # Execute V5.3 Stateful Pipeline - THE ITERATION ENGINE IS NOW LIVE
        pipeline_result = await stateful_orchestrator.execute_pipeline(
            initial_query=request.initial_query,
            user_id=user_id,
            session_id=session_id,
            merge_project_context=request.merge_project_context,
            project_id=project_id,
        )

        # Calculate execution time
        total_time_ms = int((time.time() - start_time) * 1000)

        # Extract checkpoint information
        checkpoints = pipeline_result.get("checkpoints_created", [])
        checkpoint_ids = [str(cp.get("checkpoint_id", "")) for cp in checkpoints]

        # Persist results to Supabase
        await _persist_stateful_results(pipeline_result, request)

        logger.info(
            f"✅ V5.3 Stateful Analysis completed: {pipeline_result.get('trace_id')} in {total_time_ms}ms"
        )
        logger.info(
            f"🎯 ITERATION ENGINE ACTIVE: {len(checkpoint_ids)} checkpoints created"
        )

        return StatefulAnalysisResponse(
            success=True,
            trace_id=str(pipeline_result.get("trace_id", "")),
            analysis_results=pipeline_result,
            checkpoints_created=checkpoint_ids,
            total_processing_time_ms=total_time_ms,
            pipeline_status="completed",
            can_resume=True,
            next_available_checkpoints=(
                checkpoint_ids[-3:] if len(checkpoint_ids) >= 3 else checkpoint_ids
            ),
            iteration_engine_active=True,
        )

    except Exception as e:
        logger.error(f"❌ V5.3 Stateful Analysis failed: {str(e)}")
        total_time_ms = int((time.time() - start_time) * 1000)

        return StatefulAnalysisResponse(
            success=False,
            trace_id="",
            analysis_results={"error": str(e)},
            checkpoints_created=[],
            total_processing_time_ms=total_time_ms,
            pipeline_status="failed",
            can_resume=False,
            next_available_checkpoints=[],
            iteration_engine_active=False,
        )


@router.post("/resume-from-checkpoint", response_model=StatefulAnalysisResponse)
async def resume_from_checkpoint(request: ResumeAnalysisRequest):
    """
    V5.3 CANONICAL COMPLIANCE: Resume analysis from any checkpoint.

    This endpoint enables the core V5.3 iterative analysis capability by allowing
    users to resume from any previously created checkpoint.

    ITERATION ENGINE POWER: True cognitive partnership through iterative refinement.
    """
    start_time = time.time()

    try:
        logger.info(f"🔄 V5.3 Resume Analysis: From checkpoint {request.checkpoint_id}")

        # Convert string IDs to UUIDs
        checkpoint_id = UUID(request.checkpoint_id)
        user_id = UUID(request.user_id) if request.user_id else None
        session_id = UUID(request.session_id) if request.session_id else None

        # Resume V5.3 Stateful Pipeline - ITERATION ENGINE RESUME
        pipeline_result = await stateful_orchestrator.execute_pipeline(
            resume_from_checkpoint=checkpoint_id, user_id=user_id, session_id=session_id
        )

        # Calculate execution time
        total_time_ms = int((time.time() - start_time) * 1000)

        # Extract checkpoint information
        checkpoints = pipeline_result.get("checkpoints_created", [])
        checkpoint_ids = [str(cp.get("checkpoint_id", "")) for cp in checkpoints]

        logger.info(
            f"✅ V5.3 Resume Analysis completed: {pipeline_result.get('trace_id')} in {total_time_ms}ms"
        )
        logger.info(
            f"🎯 ITERATION ENGINE RESUMED: {len(checkpoint_ids)} new checkpoints created"
        )

        return StatefulAnalysisResponse(
            success=True,
            trace_id=str(pipeline_result.get("trace_id", "")),
            analysis_results=pipeline_result,
            checkpoints_created=checkpoint_ids,
            total_processing_time_ms=total_time_ms,
            pipeline_status="resumed_and_completed",
            can_resume=True,
            next_available_checkpoints=(
                checkpoint_ids[-3:] if len(checkpoint_ids) >= 3 else checkpoint_ids
            ),
            iteration_engine_active=True,
        )

    except Exception as e:
        logger.error(f"❌ V5.3 Resume Analysis failed: {str(e)}")
        total_time_ms = int((time.time() - start_time) * 1000)

        return StatefulAnalysisResponse(
            success=False,
            trace_id="",
            analysis_results={"error": str(e)},
            checkpoints_created=[],
            total_processing_time_ms=total_time_ms,
            pipeline_status="resume_failed",
            can_resume=False,
            next_available_checkpoints=[],
            iteration_engine_active=False,
        )


@router.get("/analysis-results/{trace_id}", response_model=Dict[str, Any])
async def get_stateful_analysis_results(trace_id: str):
    """
    V5.3 CANONICAL COMPLIANCE: Retrieve complete stateful analysis results.

    Returns the full pipeline results including all stage outputs and checkpoint metadata.
    """
    try:
        logger.info(f"📊 V5.3 Retrieving analysis results for trace: {trace_id}")

        # Query Supabase for stateful results
        supabase = get_supabase_client()

        response = (
            supabase.table("v53_stateful_analyses")
            .select("*")
            .eq("trace_id", trace_id)
            .execute()
        )

        if not response.data:
            raise HTTPException(
                status_code=404, detail=f"Analysis {trace_id} not found"
            )

        analysis_data = response.data[0]

        return {
            "trace_id": trace_id,
            "analysis_results": analysis_data.get("analysis_results", {}),
            "pipeline_status": analysis_data.get("pipeline_status", "unknown"),
            "created_at": analysis_data.get("created_at"),
            "total_processing_time_ms": analysis_data.get(
                "total_processing_time_ms", 0
            ),
            "iteration_engine_version": "v5.3",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error retrieving analysis results: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve analysis: {str(e)}"
        )


@router.get("/checkpoints/{trace_id}", response_model=CheckpointsResponse)
async def get_analysis_checkpoints(trace_id: str):
    """
    V5.3 CANONICAL COMPLIANCE: List all available checkpoints for an analysis.

    This enables users to see all resume points available for iterative refinement.
    """
    try:
        logger.info(f"📍 V5.3 Retrieving checkpoints for trace: {trace_id}")

        # Query Supabase for checkpoints
        supabase = get_supabase_client()

        response = (
            supabase.table("v53_analysis_checkpoints")
            .select("*")
            .eq("trace_id", trace_id)
            .order("created_at")
            .execute()
        )

        if not response.data:
            return CheckpointsResponse(
                trace_id=trace_id,
                checkpoints=[],
                total_checkpoints=0,
                latest_checkpoint=None,
            )

        checkpoints = []
        latest_checkpoint = None

        for checkpoint_data in response.data:
            checkpoint_info = CheckpointInfo(
                checkpoint_id=checkpoint_data["checkpoint_id"],
                stage=checkpoint_data["stage"],
                timestamp=datetime.fromisoformat(checkpoint_data["created_at"]),
                can_resume_from=True,
                stage_results=checkpoint_data.get("stage_results", {}),
            )
            checkpoints.append(checkpoint_info)
            latest_checkpoint = checkpoint_data["checkpoint_id"]

        return CheckpointsResponse(
            trace_id=trace_id,
            checkpoints=checkpoints,
            total_checkpoints=len(checkpoints),
            latest_checkpoint=latest_checkpoint,
        )

    except Exception as e:
        logger.error(f"❌ Error retrieving checkpoints: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve checkpoints: {str(e)}"
        )


@router.get("/v53-status")
async def get_v53_status():
    """V5.3 CANONICAL COMPLIANCE: System status endpoint showing V5.3 compliance."""
    return {
        "v53_compliance": True,
        "stateful_pipeline_orchestrator": "active",
        "iteration_engine": "online",
        "checkpoint_support": "enabled",
        "resume_capability": "enabled",
        "architecture": "V5.3 Canonical Standard",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# V5.3 Helper Functions
async def _persist_stateful_results(
    pipeline_result: Dict[str, Any], request: StatefulAnalysisRequest
):
    """Persist V5.3 stateful analysis results to Supabase."""
    try:
        supabase = get_supabase_client()

        # Store main analysis results
        analysis_record = {
            "trace_id": str(pipeline_result.get("trace_id", "")),
            "user_id": request.user_id,
            "session_id": request.session_id,
            "initial_query": request.initial_query,
            "analysis_results": pipeline_result,
            "pipeline_status": "completed",
            "total_processing_time_ms": pipeline_result.get("execution_time_ms", 0),
            "v53_compliant": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        supabase.table("v53_stateful_analyses").insert(analysis_record).execute()

        # Store checkpoints
        checkpoints = pipeline_result.get("checkpoints_created", [])
        for checkpoint in checkpoints:
            checkpoint_record = {
                "trace_id": str(pipeline_result.get("trace_id", "")),
                "checkpoint_id": str(checkpoint.get("checkpoint_id", "")),
                "stage": checkpoint.get("stage", "unknown"),
                "stage_results": checkpoint.get("results", {}),
                "can_resume_from": True,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

            supabase.table("v53_analysis_checkpoints").insert(
                checkpoint_record
            ).execute()

        logger.info(f"✅ V5.3 Results persisted: {len(checkpoints)} checkpoints stored")

    except Exception as e:
        logger.error(f"❌ Failed to persist V5.3 results: {str(e)}")
        # Don't fail the analysis if persistence fails
        pass
