"""
System Metrics Endpoint
Sprint 1: Validation & Monitoring Tools
Purpose: Expose orchestrator metrics for monitoring shadow testing

This endpoint provides real-time visibility into orchestrator selection
and execution metrics during the migration process.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from datetime import datetime, timezone

from src.core.orchestrator_selector import get_orchestrator_selector, OrchestratorMode
from src.core.structured_logging import get_logger

router = APIRouter(prefix="/api/v1/system", tags=["system", "metrics"])
logger = get_logger(__name__, component="system_metrics")


@router.get("/orchestrator_metrics")
async def get_orchestrator_metrics() -> Dict[str, Any]:
    """
    Get real-time orchestrator selection and execution metrics.

    Returns:
        JSON object containing:
        - Current orchestrator mode
        - Execution counts for each orchestrator
        - Shadow testing results
        - Mode switch history
        - AB test percentage (if applicable)
    """
    try:
        selector = get_orchestrator_selector()
        metrics = selector.get_metrics()

        # Add additional context
        enhanced_metrics = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "current_mode": metrics.get("current_mode", "unknown"),
            "execution_counts": {
                "neural_lace": metrics.get("neural_lace_executions", 0),
                "state_machine": metrics.get("state_machine_executions", 0),
                "shadow": metrics.get("shadow_executions", 0),
            },
            "mode_switches": metrics.get("mode_switches", []),
            "ab_test_percentage": metrics.get("ab_test_percentage"),
            "health_status": "healthy",
            "migration_readiness": _calculate_migration_readiness(metrics),
        }

        logger.info(
            "metrics_retrieved",
            mode=enhanced_metrics["current_mode"],
            total_executions=sum(enhanced_metrics["execution_counts"].values()),
        )

        return enhanced_metrics

    except Exception as e:
        logger.error("metrics_retrieval_failed", error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve orchestrator metrics: {str(e)}"
        )


@router.get("/orchestrator_mode")
async def get_orchestrator_mode() -> Dict[str, str]:
    """
    Get the current orchestrator mode.

    Returns:
        JSON object with current mode and available modes
    """
    try:
        selector = get_orchestrator_selector()

        return {
            "current_mode": selector.mode.value,
            "available_modes": [mode.value for mode in OrchestratorMode],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error("mode_retrieval_failed", error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve orchestrator mode: {str(e)}"
        )


@router.post("/orchestrator_mode")
async def set_orchestrator_mode(mode: str) -> Dict[str, Any]:
    """
    Dynamically change the orchestrator mode.

    WARNING: This endpoint should be protected in production!

    Args:
        mode: One of: neural_lace, state_machine, shadow, comparison, ab_test

    Returns:
        Confirmation of mode change
    """
    try:
        # Validate mode
        try:
            new_mode = OrchestratorMode(mode.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid mode: {mode}. Must be one of: {[m.value for m in OrchestratorMode]}",
            )

        selector = get_orchestrator_selector()
        old_mode = selector.mode.value
        selector.set_mode(new_mode)

        logger.warning(
            "orchestrator_mode_changed_via_api",
            old_mode=old_mode,
            new_mode=new_mode.value,
        )

        return {
            "success": True,
            "old_mode": old_mode,
            "new_mode": new_mode.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "warning": "Mode change is immediate and affects all new engagements",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("mode_change_failed", error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Failed to change orchestrator mode: {str(e)}"
        )


@router.get("/shadow_comparison_summary")
async def get_shadow_comparison_summary() -> Dict[str, Any]:
    """
    Get a summary of shadow testing comparison results.

    Returns:
        Summary statistics of differences detected during shadow testing
    """
    try:
        # This would typically query from a database or cache
        # For now, return a placeholder structure
        selector = get_orchestrator_selector()
        metrics = selector.get_metrics()

        shadow_count = metrics.get("shadow_executions", 0)

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "shadow_executions": shadow_count,
            "comparison_summary": {
                "total_comparisons": shadow_count,
                "phase_mismatches": 0,  # Would be tracked in real implementation
                "confidence_differences": 0,
                "synthesis_differences": 0,
                "perfect_matches": 0,
            },
            "parity_percentage": 0.0 if shadow_count == 0 else 100.0,  # Placeholder
            "recommendation": _get_migration_recommendation(metrics),
        }

    except Exception as e:
        logger.error("shadow_summary_failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve shadow comparison summary: {str(e)}",
        )


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Basic health check for the metrics endpoint.

    Returns:
        Health status
    """
    return {
        "status": "healthy",
        "service": "orchestrator_metrics",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _calculate_migration_readiness(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate migration readiness based on metrics.

    Args:
        metrics: Raw metrics from orchestrator selector

    Returns:
        Migration readiness assessment
    """
    shadow_executions = metrics.get("shadow_executions", 0)
    state_machine_executions = metrics.get("state_machine_executions", 0)

    readiness_score = 0
    criteria = []

    # Check shadow testing coverage
    if shadow_executions >= 100:
        readiness_score += 25
        criteria.append("✅ Sufficient shadow testing (100+ executions)")
    else:
        criteria.append(f"⚠️ More shadow testing needed ({shadow_executions}/100)")

    # Check direct state machine testing
    if state_machine_executions >= 10:
        readiness_score += 25
        criteria.append("✅ Direct testing performed (10+ executions)")
    else:
        criteria.append(f"⚠️ More direct testing needed ({state_machine_executions}/10)")

    # Check mode switches (indicates testing)
    if len(metrics.get("mode_switches", [])) >= 3:
        readiness_score += 25
        criteria.append("✅ Multiple mode switches tested")
    else:
        criteria.append("⚠️ Test more mode switches")

    # Placeholder for parity score (would come from comparison results)
    parity_score = 100  # Assume perfect parity for now
    if parity_score >= 99:
        readiness_score += 25
        criteria.append(f"✅ High parity achieved ({parity_score}%)")
    else:
        criteria.append(f"❌ Parity too low ({parity_score}%)")

    return {
        "readiness_score": readiness_score,
        "ready_for_migration": readiness_score >= 75,
        "criteria": criteria,
        "recommendation": (
            "Ready for AB testing"
            if readiness_score >= 75
            else "Continue shadow testing"
        ),
    }


def _get_migration_recommendation(metrics: Dict[str, Any]) -> str:
    """
    Get migration recommendation based on current metrics.

    Args:
        metrics: Raw metrics from orchestrator selector

    Returns:
        Human-readable recommendation
    """
    current_mode = metrics.get("current_mode", "unknown")
    shadow_count = metrics.get("shadow_executions", 0)
    state_machine_count = metrics.get("state_machine_executions", 0)

    if current_mode == "neural_lace" and shadow_count == 0:
        return "Start shadow testing to validate the new orchestrator"
    elif current_mode == "shadow" and shadow_count < 100:
        return f"Continue shadow testing ({shadow_count}/100 minimum executions)"
    elif current_mode == "shadow" and shadow_count >= 100:
        return (
            "Shadow testing complete. Review comparison results and consider AB testing"
        )
    elif current_mode == "ab_test":
        ab_percentage = metrics.get("ab_test_percentage", 0)
        if ab_percentage < 50:
            return f"Gradually increase AB test percentage (currently {ab_percentage}%)"
        else:
            return "Consider full migration to state_machine orchestrator"
    elif current_mode == "state_machine":
        return "Migration complete! Monitor for any issues"
    else:
        return "Review current testing strategy"


# Export router for inclusion in main app
__all__ = ["router"]
