"""
METIS Flywheel Management API

Internal API for administrators to monitor and manage the Flywheel system.
This is separate from the user-facing API and provides system control capabilities.
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from src.flywheel import (
    get_flywheel_cache,
    get_learning_loop,
    get_phantom_workflow_detector,
    get_unified_intelligence_dashboard,
    get_degradation_manager,
)
from src.flywheel.cache.flywheel_cache_system import MemoryTier
from src.flywheel.core.graceful_degradation import DegradationMode

logger = logging.getLogger(__name__)


# Request/Response Models
class SystemStatusResponse(BaseModel):
    system_health: str = Field(description="Overall system health status")
    components: Dict[str, Any] = Field(description="Individual component status")
    degradation_mode: str = Field(description="Current degradation mode")
    uptime_seconds: float = Field(description="System uptime")
    last_updated: str = Field(description="Last status update timestamp")


class MemoryStatistics(BaseModel):
    total_entries: int = Field(description="Total memory entries across all tiers")
    memory_tiers: Dict[str, Any] = Field(description="Statistics per memory tier")
    consolidation_info: Dict[str, Any] = Field(
        description="Memory consolidation status"
    )


class PhantomDetectionStats(BaseModel):
    total_detections: int = Field(description="Total phantom detections")
    accuracy_rate: float = Field(description="Detection accuracy rate")
    false_positive_rate: float = Field(description="False positive rate")
    recent_detections: List[Dict[str, Any]] = Field(
        description="Recent detection events"
    )


class LearningLoopMetrics(BaseModel):
    total_learning_events: int = Field(description="Total learning events processed")
    user_satisfaction_trend: float = Field(description="User satisfaction trend")
    prediction_accuracy: float = Field(description="Prediction accuracy")
    active_patterns: int = Field(description="Active learning patterns")


class SystemControlRequest(BaseModel):
    action: str = Field(description="Control action to perform")
    parameters: Dict[str, Any] = Field(default={}, description="Action parameters")


class MemoryConsolidationRequest(BaseModel):
    force_consolidation: bool = Field(
        default=False, description="Force immediate consolidation"
    )
    target_tier: Optional[str] = Field(
        default=None, description="Target specific memory tier"
    )


# Router setup
router = APIRouter(prefix="/admin/flywheel", tags=["flywheel-management"])


@router.get("/status", response_model=SystemStatusResponse)
async def get_system_status():
    """Get comprehensive Flywheel system status"""
    try:
        # Get degradation manager status
        degradation_manager = get_degradation_manager()
        degradation_status = degradation_manager.get_status_summary()

        # Get individual component health
        components = {}

        # Flywheel Cache status
        try:
            cache = await get_flywheel_cache()
            cache_metrics = cache.get_flywheel_metrics()
            components["flywheel_cache"] = {
                "status": "healthy",
                "cache_hit_rate": cache_metrics.cache_hit_rate,
                "total_interactions": cache_metrics.total_interactions,
                "user_satisfaction": cache_metrics.user_satisfaction_score,
            }
        except Exception as e:
            components["flywheel_cache"] = {"status": "error", "error": str(e)}

        # Learning Loop status
        try:
            learning_loop = await get_learning_loop()
            learning_metrics = learning_loop.get_learning_metrics()
            components["learning_loop"] = {
                "status": "healthy",
                "total_events": len(learning_loop.learning_events),
                "metrics": learning_metrics,
            }
        except Exception as e:
            components["learning_loop"] = {"status": "error", "error": str(e)}

        # Phantom Detection status
        try:
            phantom_detector = get_phantom_workflow_detector()
            detection_summary = await phantom_detector.run_diagnostic_scan()
            components["phantom_detector"] = {
                "status": "healthy",
                "recent_detections": detection_summary.get("recent_detections", 0),
                "accuracy": detection_summary.get("average_confidence", 0.0),
            }
        except Exception as e:
            components["phantom_detector"] = {"status": "error", "error": str(e)}

        # Overall health assessment
        healthy_components = sum(
            1 for comp in components.values() if comp.get("status") == "healthy"
        )
        total_components = len(components)

        if healthy_components == total_components:
            overall_health = "excellent"
        elif healthy_components >= total_components * 0.8:
            overall_health = "good"
        elif healthy_components >= total_components * 0.5:
            overall_health = "degraded"
        else:
            overall_health = "critical"

        return SystemStatusResponse(
            system_health=overall_health,
            components=components,
            degradation_mode=degradation_status["current_mode"],
            uptime_seconds=0.0,  # Would track actual uptime
            last_updated=datetime.utcnow().isoformat(),
        )

    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get system status: {str(e)}"
        )


@router.get("/memory/statistics", response_model=MemoryStatistics)
async def get_memory_statistics():
    """Get hierarchical memory system statistics"""
    try:
        cache = await get_flywheel_cache()
        memory_stats = cache.get_memory_statistics()

        # Calculate totals
        total_entries = sum(
            tier_stats["count"]
            for tier_stats in memory_stats.values()
            if isinstance(tier_stats, dict) and "count" in tier_stats
        )

        return MemoryStatistics(
            total_entries=total_entries,
            memory_tiers={
                k: v
                for k, v in memory_stats.items()
                if isinstance(v, dict) and "count" in v
            },
            consolidation_info={
                "last_consolidation": memory_stats.get("last_consolidation"),
                "time_until_next": memory_stats.get("time_until_next_consolidation"),
            },
        )

    except Exception as e:
        logger.error(f"Error getting memory statistics: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get memory statistics: {str(e)}"
        )


@router.get("/phantom-detection/stats", response_model=PhantomDetectionStats)
async def get_phantom_detection_stats():
    """Get phantom workflow detection statistics"""
    try:
        phantom_detector = get_phantom_workflow_detector()
        diagnostic_scan = await phantom_detector.run_diagnostic_scan()

        total_detections = len(phantom_detector.detection_history)
        recent_detections = phantom_detector.detection_history[-10:]  # Last 10

        # Calculate accuracy (simplified)
        accuracy_rate = diagnostic_scan.get("average_confidence", 0.75)
        false_positive_rate = 1.0 - accuracy_rate

        recent_detection_data = []
        for detection in recent_detections:
            recent_detection_data.append(
                {
                    "detection_id": detection.detection_id,
                    "phase_name": detection.phase_name,
                    "severity": detection.severity.value,
                    "confidence": detection.confidence,
                    "timestamp": detection.timestamp.isoformat(),
                }
            )

        return PhantomDetectionStats(
            total_detections=total_detections,
            accuracy_rate=accuracy_rate,
            false_positive_rate=false_positive_rate,
            recent_detections=recent_detection_data,
        )

    except Exception as e:
        logger.error(f"Error getting phantom detection stats: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get phantom detection stats: {str(e)}"
        )


@router.get("/learning/metrics", response_model=LearningLoopMetrics)
async def get_learning_metrics():
    """Get learning loop performance metrics"""
    try:
        learning_loop = await get_learning_loop()
        metrics = learning_loop.get_learning_metrics()

        return LearningLoopMetrics(
            total_learning_events=len(learning_loop.learning_events),
            user_satisfaction_trend=metrics.get("user_satisfaction_trend", 0.0),
            prediction_accuracy=metrics.get("prediction_accuracy", 0.0),
            active_patterns=metrics.get("active_patterns", 0),
        )

    except Exception as e:
        logger.error(f"Error getting learning metrics: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get learning metrics: {str(e)}"
        )


@router.post("/memory/consolidation")
async def trigger_memory_consolidation(
    request: MemoryConsolidationRequest, background_tasks: BackgroundTasks
):
    """Trigger memory consolidation process"""
    try:
        cache = await get_flywheel_cache()

        if request.force_consolidation:
            # Force immediate consolidation
            background_tasks.add_task(cache.perform_memory_consolidation)
            return {"message": "Memory consolidation triggered", "forced": True}
        else:
            # Check if consolidation is needed
            if cache._should_consolidate():
                background_tasks.add_task(cache.perform_memory_consolidation)
                return {"message": "Memory consolidation triggered", "forced": False}
            else:
                return {
                    "message": "Memory consolidation not needed at this time",
                    "forced": False,
                }

    except Exception as e:
        logger.error(f"Error triggering memory consolidation: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to trigger consolidation: {str(e)}"
        )


@router.post("/system/control")
async def system_control(request: SystemControlRequest):
    """Execute system control actions"""
    try:
        action = request.action.lower()
        parameters = request.parameters

        if action == "reset_phantom_detector":
            phantom_detector = get_phantom_workflow_detector()
            # Clear detection history
            phantom_detector.detection_history.clear()
            phantom_detector.false_positive_feedback.clear()
            return {"message": "Phantom detector reset successfully"}

        elif action == "clear_cache":
            cache = await get_flywheel_cache()
            tier = parameters.get("tier", "all")

            if tier == "all":
                cache.l1_cache.clear()
                for memory_tier in cache.memory_tiers.values():
                    memory_tier.clear()
                return {"message": "All cache layers cleared"}
            else:
                # Clear specific tier
                try:
                    target_tier = MemoryTier(tier)
                    cache.memory_tiers[target_tier].clear()
                    return {"message": f"Cache tier {tier} cleared"}
                except ValueError:
                    raise HTTPException(status_code=400, detail=f"Invalid tier: {tier}")

        elif action == "set_degradation_mode":
            degradation_manager = get_degradation_manager()
            mode = parameters.get("mode")

            try:
                target_mode = DegradationMode(mode)
                success = await degradation_manager.apply_degradation_mode(target_mode)
                if success:
                    return {"message": f"Degradation mode set to {mode}"}
                else:
                    raise HTTPException(
                        status_code=500, detail="Failed to set degradation mode"
                    )
            except ValueError:
                raise HTTPException(
                    status_code=400, detail=f"Invalid degradation mode: {mode}"
                )

        else:
            raise HTTPException(status_code=400, detail=f"Unknown action: {action}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing system control: {e}")
        raise HTTPException(status_code=500, detail=f"System control failed: {str(e)}")


@router.get("/degradation/status")
async def get_degradation_status():
    """Get current degradation status and configuration"""
    try:
        degradation_manager = get_degradation_manager()
        status = degradation_manager.get_status_summary()

        # Add available modes
        available_modes = [mode.value for mode in DegradationMode]
        status["available_modes"] = available_modes

        return status

    except Exception as e:
        logger.error(f"Error getting degradation status: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get degradation status: {str(e)}"
        )


@router.get("/health/components")
async def get_component_health():
    """Get detailed health information for all Flywheel components"""
    try:
        health_report = {"timestamp": datetime.utcnow().isoformat(), "components": {}}

        # Check each component with detailed health info
        components = [
            ("flywheel_cache", get_flywheel_cache),
            ("learning_loop", get_learning_loop),
            ("phantom_detector", get_phantom_workflow_detector),
            ("intelligence_dashboard", get_unified_intelligence_dashboard),
        ]

        for component_name, get_component_func in components:
            try:
                if component_name in ["flywheel_cache", "learning_loop"]:
                    component = await get_component_func()
                else:
                    component = get_component_func()

                health_report["components"][component_name] = {
                    "status": "healthy",
                    "initialized": component is not None,
                    "last_check": datetime.utcnow().isoformat(),
                }
            except Exception as e:
                health_report["components"][component_name] = {
                    "status": "unhealthy",
                    "error": str(e),
                    "last_check": datetime.utcnow().isoformat(),
                }

        return health_report

    except Exception as e:
        logger.error(f"Error getting component health: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get component health: {str(e)}"
        )


# Export router
flywheel_management_router = router
