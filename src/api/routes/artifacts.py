"""
Artifact Retrieval API - Operation "Evidence Locker"
====================================================

Provides secure, canonical access to analysis artifacts stored in context streams.
This is the official "evidence locker" for retrieving raw outputs from completed analyses.

Author: ARC Chief System Architect
Status: CRITICAL INFRASTRUCTURE
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, Dict, Any
import logging

from src.services.artifact_retrieval_service import ArtifactRetrievalService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/logs", tags=["artifacts"])

# Initialize the Evidence Locker service
artifact_service = ArtifactRetrievalService()


@router.get("/{trace_id}/artifacts")
async def get_artifact(
    trace_id: str,
    artifact_type: str = Query(
        ..., description="Type of artifact to retrieve", example="final_report"
    ),
    event_id: Optional[str] = Query(
        None, description="Specific event ID for precise artifact targeting"
    ),
) -> Dict[str, Any]:
    """
    🔐 EVIDENCE LOCKER: Retrieve Analysis Artifacts

    Fetches the raw, unabridged content of artifacts generated during analysis runs.
    This is the canonical method for accessing completed analysis outputs.

    Args:
        trace_id: Unique identifier for the analysis run
        artifact_type: Type of artifact (final_report, intermediate_analysis, etc.)
        event_id: Optional specific event ID for precise targeting

    Returns:
        Dict containing:
        - artifact_content: Raw content of the requested artifact
        - metadata: Context about the artifact (timestamp, size, type)
        - trace_context: Basic trace information

    Raises:
        404: Trace ID not found or artifact not available
        422: Invalid artifact type or malformed request
        500: Database or parsing errors
    """

    try:
        logger.info(
            f"🔐 Evidence Locker request: trace_id={trace_id}, artifact_type={artifact_type}"
        )

        # Retrieve the artifact from the Evidence Locker
        artifact_result = await artifact_service.get_artifact(
            trace_id=trace_id, artifact_type=artifact_type, event_id=event_id
        )

        if not artifact_result:
            raise HTTPException(
                status_code=404,
                detail=f"Artifact '{artifact_type}' not found for trace {trace_id}",
            )

        logger.info(
            f"✅ Evidence Locker success: retrieved {len(str(artifact_result['artifact_content']))} chars"
        )

        return {
            "trace_id": trace_id,
            "artifact_type": artifact_type,
            "artifact_content": artifact_result["artifact_content"],
            "metadata": artifact_result["metadata"],
            "trace_context": artifact_result["trace_context"],
            "retrieved_at": artifact_result["retrieved_at"],
        }

    except HTTPException:
        # Re-raise FastAPI exceptions
        raise
    except Exception as e:
        logger.error(f"❌ Evidence Locker failure: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve artifact: {str(e)}"
        )


@router.get("/{trace_id}/artifacts/types")
async def list_available_artifacts(trace_id: str) -> Dict[str, Any]:
    """
    📋 List Available Artifacts

    Returns all available artifact types for a given trace_id.
    Useful for discovery and debugging.

    Args:
        trace_id: Unique identifier for the analysis run

    Returns:
        Dict containing available artifact types and their metadata
    """

    try:
        logger.info(f"📋 Listing artifacts for trace_id={trace_id}")

        available_artifacts = await artifact_service.list_artifacts(trace_id)

        if not available_artifacts:
            raise HTTPException(
                status_code=404, detail=f"No artifacts found for trace {trace_id}"
            )

        return {
            "trace_id": trace_id,
            "available_artifacts": available_artifacts,
            "total_count": len(available_artifacts),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Artifact listing failure: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to list artifacts: {str(e)}"
        )


@router.get("/{trace_id}/health")
async def check_trace_health(trace_id: str) -> Dict[str, Any]:
    """
    🏥 Evidence Locker Health Check

    Verifies that a trace exists and provides basic health information.

    Args:
        trace_id: Unique identifier for the analysis run

    Returns:
        Dict containing trace health status and basic metadata
    """

    try:
        health_status = await artifact_service.check_trace_health(trace_id)

        return {
            "trace_id": trace_id,
            "exists": health_status["exists"],
            "status": health_status["status"],
            "total_events": health_status.get("total_events", 0),
            "completion_status": health_status.get("completion_status", "unknown"),
            "last_activity": health_status.get("last_activity"),
        }

    except Exception as e:
        logger.error(f"❌ Health check failure: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")
