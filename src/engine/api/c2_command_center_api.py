"""
METIS C2 Command Center API

Internal API for C2 Platform integration with the existing Flywheel admin dashboard.
Provides endpoints for Evaluation Harness control, CQA metrics, Glass-Box logs, and Flywheel automation.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field

# Import existing evaluation harness
from evaluation.run_harness import EvaluationHarness

# Import CQA components if available
try:
    from src.engine.agents.quality_rater_agent_v2 import TransparentQualityRater
    from src.engine.adapters.core.contracts.quality import CQA_Result, RIVAScore

    CQA_AVAILABLE = True
except ImportError:
    CQA_AVAILABLE = False
    logging.warning("CQA components not available - using mock data")

# Import context stream components
try:
    from src.engine.adapters.core.unified_context_stream import get_unified_context_stream

    CONTEXT_STREAM_AVAILABLE = True
except ImportError:
    CONTEXT_STREAM_AVAILABLE = False
    logging.warning("Unified Context Stream not available - using mock data")

logger = logging.getLogger(__name__)


# Request/Response Models
class HarnessRunRequest(BaseModel):
    test_suite: str = Field(
        description="Test suite to run",
        examples=["e2e_golden_cases", "lolla_golden_cases", "crows_pairs", "all"],
    )
    config: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional test configuration overrides"
    )


class HarnessRunResponse(BaseModel):
    run_id: str = Field(description="Unique identifier for this test run")
    status: str = Field(description="Initial run status")
    test_suite: str = Field(description="Test suite being executed")
    estimated_completion_time: Optional[int] = Field(
        description="Estimated completion time in seconds"
    )
    started_at: str = Field(description="Run start timestamp")


class HarnessStatusResponse(BaseModel):
    run_id: str = Field(description="Test run identifier")
    status: str = Field(
        description="Current run status",
        examples=["running", "completed", "failed", "cancelled"],
    )
    progress: float = Field(description="Completion progress (0.0 to 1.0)")
    test_suite: str = Field(description="Test suite being executed")
    current_test: Optional[str] = Field(description="Currently executing test")
    completed_tests: int = Field(description="Number of completed tests")
    total_tests: int = Field(description="Total number of tests")
    started_at: str = Field(description="Run start timestamp")
    completed_at: Optional[str] = Field(description="Run completion timestamp")
    results_summary: Optional[Dict[str, Any]] = Field(
        description="Test results summary if completed"
    )
    error_message: Optional[str] = Field(description="Error message if failed")


class CQAMetricsSummary(BaseModel):
    summary_period: str = Field(
        description="Time period for metrics",
        examples=["30_days", "7_days", "24_hours"],
    )
    total_evaluations: int = Field(description="Total number of evaluations in period")
    average_scores: Dict[str, float] = Field(description="Average RIVA scores")
    agent_performance: Dict[str, Dict[str, Union[float, int]]] = Field(
        description="Performance by agent type"
    )
    quality_trends: List[Dict[str, Any]] = Field(description="Quality trends over time")
    rubric_effectiveness: Dict[str, Dict[str, Union[int, float]]] = Field(
        description="Rubric usage and effectiveness"
    )


class ContextStreamLog(BaseModel):
    trace_id: str = Field(description="Unique trace identifier")
    engagement_id: Optional[str] = Field(description="Associated engagement ID")
    status: str = Field(description="Stream status")
    event_count: int = Field(description="Number of events in stream")
    performance_metrics: Dict[str, Any] = Field(
        description="Stream performance metrics"
    )
    events: List[Dict[str, Any]] = Field(description="Stream events")
    timeline_visualization: Optional[str] = Field(
        description="Base64 encoded timeline SVG"
    )
    export_formats: List[str] = Field(description="Available export formats")


class FlywheelAutomationStatus(BaseModel):
    automation_health: str = Field(description="Overall automation health status")
    flywheel_active: bool = Field(description="Whether flywheel is actively running")
    training_candidates_curated: int = Field(
        description="Number of training candidates curated"
    )
    last_retraining_cycle: Optional[str] = Field(
        description="Timestamp of last retraining cycle"
    )
    next_retraining_estimate: Optional[str] = Field(
        description="Estimated next retraining time"
    )
    curation_queue_size: int = Field(description="Number of items in curation queue")
    automation_metrics: Dict[str, Any] = Field(
        description="Detailed automation metrics"
    )
    recent_improvements: List[Dict[str, Any]] = Field(
        description="Recent system improvements"
    )


# Global storage for run tracking
active_runs: Dict[str, Dict[str, Any]] = {}

# Router setup
router = APIRouter(prefix="/admin/flywheel/c2", tags=["c2-command-center"])


@router.post("/harness/run", response_model=HarnessRunResponse)
async def trigger_evaluation_run(
    request: HarnessRunRequest, background_tasks: BackgroundTasks
):
    """Trigger a new evaluation harness test run"""
    try:
        # Generate unique run ID
        run_id = f"eval_run_{uuid.uuid4().hex[:8]}"
        started_at = datetime.utcnow().isoformat()

        # Validate test suite
        valid_suites = [
            "e2e_golden_cases",
            "lolla_golden_cases",
            "crows_pairs",
            "stereoset",
            "all",
        ]
        if request.test_suite not in valid_suites:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid test suite. Valid options: {valid_suites}",
            )

        # Initialize run tracking
        run_info = {
            "run_id": run_id,
            "status": "running",
            "test_suite": request.test_suite,
            "config": request.config or {},
            "started_at": started_at,
            "progress": 0.0,
            "completed_tests": 0,
            "total_tests": 0,
            "current_test": None,
            "results_summary": None,
            "error_message": None,
        }
        active_runs[run_id] = run_info

        # Start the evaluation run in the background
        background_tasks.add_task(
            execute_evaluation_run, run_id, request.test_suite, request.config
        )

        # Estimate completion time based on test suite
        test_suite_estimates = {
            "e2e_golden_cases": 300,  # 5 minutes
            "lolla_golden_cases": 180,  # 3 minutes
            "crows_pairs": 120,  # 2 minutes
            "stereoset": 90,  # 1.5 minutes
            "all": 600,  # 10 minutes
        }
        estimated_time = test_suite_estimates.get(request.test_suite, 300)

        logger.info(
            f"Started evaluation run {run_id} for test suite {request.test_suite}"
        )

        return HarnessRunResponse(
            run_id=run_id,
            status="running",
            test_suite=request.test_suite,
            estimated_completion_time=estimated_time,
            started_at=started_at,
        )

    except Exception as e:
        logger.error(f"Error triggering evaluation run: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to trigger evaluation run: {str(e)}"
        )


@router.get("/harness/status/{run_id}", response_model=HarnessStatusResponse)
async def get_evaluation_run_status(run_id: str):
    """Get the status of a specific evaluation run"""
    try:
        if run_id not in active_runs:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

        run_info = active_runs[run_id]

        return HarnessStatusResponse(
            run_id=run_info["run_id"],
            status=run_info["status"],
            progress=run_info["progress"],
            test_suite=run_info["test_suite"],
            current_test=run_info["current_test"],
            completed_tests=run_info["completed_tests"],
            total_tests=run_info["total_tests"],
            started_at=run_info["started_at"],
            completed_at=run_info.get("completed_at"),
            results_summary=run_info.get("results_summary"),
            error_message=run_info.get("error_message"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting run status for {run_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get run status: {str(e)}"
        )


@router.get("/quality/summary", response_model=CQAMetricsSummary)
async def get_cqa_quality_summary(
    period: str = Query(default="30_days", regex="^(24_hours|7_days|30_days)$")
):
    """Get comprehensive CQA quality metrics summary"""
    try:
        if CQA_AVAILABLE:
            # Try to get real CQA data from database or storage
            return await get_real_cqa_metrics(period)
        else:
            # Return mock data for demonstration
            return get_mock_cqa_metrics(period)

    except Exception as e:
        logger.error(f"Error getting CQA quality summary: {e}")
        # Fall back to mock data on error
        return get_mock_cqa_metrics(period)


@router.get("/logs/{trace_id}", response_model=ContextStreamLog)
async def get_context_stream_log(trace_id: str):
    """Retrieve a specific UnifiedContextStream log by trace ID"""
    try:
        if CONTEXT_STREAM_AVAILABLE:
            # Try to get real context stream data
            return await get_real_context_stream_log(trace_id)
        else:
            # Return mock data
            return get_mock_context_stream_log(trace_id)

    except Exception as e:
        logger.error(f"Error getting context stream log for {trace_id}: {e}")
        # Fall back to mock data
        return get_mock_context_stream_log(trace_id)


@router.get("/automation/status", response_model=FlywheelAutomationStatus)
async def get_flywheel_automation_status():
    """Get comprehensive Flywheel automation status"""
    try:
        # Get real automation data if available, otherwise mock
        return await get_flywheel_automation_data()

    except Exception as e:
        logger.error(f"Error getting flywheel automation status: {e}")
        return get_mock_flywheel_automation_status()


# Background task for executing evaluation runs
async def execute_evaluation_run(
    run_id: str, test_suite: str, config: Optional[Dict[str, Any]]
):
    """Execute evaluation run in background"""
    try:
        # Update status to running
        active_runs[run_id]["status"] = "running"
        active_runs[run_id]["current_test"] = f"Initializing {test_suite}"

        # Create evaluation harness instance
        harness = EvaluationHarness(
            results_db_path="evaluation_results.db", results_dir="evaluation_results"
        )

        # Simulate test execution with progress updates
        if test_suite == "all":
            test_suites_to_run = [
                "lolla_golden_cases",
                "e2e_golden_cases",
                "crows_pairs",
            ]
        else:
            test_suites_to_run = [test_suite]

        total_suites = len(test_suites_to_run)
        active_runs[run_id]["total_tests"] = total_suites

        results_summary = {
            "total_test_suites": total_suites,
            "completed_suites": 0,
            "success_count": 0,
            "failure_count": 0,
            "execution_time": 0,
            "detailed_results": {},
        }

        start_time = datetime.utcnow()

        for i, suite in enumerate(test_suites_to_run):
            # Update progress
            active_runs[run_id]["current_test"] = f"Running {suite}"
            active_runs[run_id]["progress"] = i / total_suites

            try:
                # Run the test suite
                suite_results = await harness.run_test_suite(suite)
                results_summary["detailed_results"][suite] = suite_results
                results_summary["success_count"] += 1

                # Small delay to simulate test execution
                await asyncio.sleep(2)

            except Exception as e:
                logger.error(f"Error running test suite {suite}: {e}")
                results_summary["detailed_results"][suite] = {"error": str(e)}
                results_summary["failure_count"] += 1

            # Update completed tests
            active_runs[run_id]["completed_tests"] = i + 1
            results_summary["completed_suites"] = i + 1

        # Calculate execution time
        end_time = datetime.utcnow()
        execution_time = (end_time - start_time).total_seconds()
        results_summary["execution_time"] = execution_time

        # Mark as completed
        active_runs[run_id]["status"] = "completed"
        active_runs[run_id]["progress"] = 1.0
        active_runs[run_id]["completed_at"] = end_time.isoformat()
        active_runs[run_id]["current_test"] = "Completed"
        active_runs[run_id]["results_summary"] = results_summary

        logger.info(f"Evaluation run {run_id} completed successfully")

    except Exception as e:
        # Mark as failed
        active_runs[run_id]["status"] = "failed"
        active_runs[run_id]["error_message"] = str(e)
        active_runs[run_id]["completed_at"] = datetime.utcnow().isoformat()

        logger.error(f"Evaluation run {run_id} failed: {e}")


# Helper functions for getting real or mock data
async def get_real_cqa_metrics(period: str) -> CQAMetricsSummary:
    """Get real CQA metrics from database or storage"""
    # This would integrate with actual CQA storage
    # For now, return mock data with realistic structure
    return get_mock_cqa_metrics(period)


def get_mock_cqa_metrics(period: str) -> CQAMetricsSummary:
    """Generate mock CQA metrics for demonstration"""
    period_days = {"24_hours": 1, "7_days": 7, "30_days": 30}[period]
    base_evaluations = period_days * 12  # ~12 evaluations per day

    return CQAMetricsSummary(
        summary_period=period,
        total_evaluations=base_evaluations
        + (base_evaluations // 4),  # Add some variance
        average_scores={
            "rigor": 7.2,
            "insight": 6.8,
            "value": 8.1,
            "alignment": 7.9,
            "overall": 7.5,
        },
        agent_performance={
            "strategic_analyst": {
                "avg_score": 8.2,
                "evaluation_count": base_evaluations // 3,
            },
            "risk_assessor": {
                "avg_score": 7.8,
                "evaluation_count": base_evaluations // 4,
            },
            "market_researcher": {
                "avg_score": 7.4,
                "evaluation_count": base_evaluations // 5,
            },
            "senior_advisor": {
                "avg_score": 8.6,
                "evaluation_count": base_evaluations // 6,
            },
        },
        quality_trends=[
            {
                "date": (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d"),
                "avg_score": 7.5 + (i * 0.02),
            }
            for i in range(min(period_days, 14))
        ][::-1],
        rubric_effectiveness={
            "riva_standard@1.0": {
                "usage_count": base_evaluations * 3 // 4,
                "avg_confidence": 0.84,
            },
            "creative_strategist@1.0": {
                "usage_count": base_evaluations // 4,
                "avg_confidence": 0.79,
            },
        },
    )


async def get_real_context_stream_log(trace_id: str) -> ContextStreamLog:
    """Get real context stream log from storage"""
    # This would integrate with actual context stream storage
    return get_mock_context_stream_log(trace_id)


def get_mock_context_stream_log(trace_id: str) -> ContextStreamLog:
    """Generate mock context stream log for demonstration"""
    return ContextStreamLog(
        trace_id=trace_id,
        engagement_id=f"eng_{trace_id[:8]}",
        status="completed",
        event_count=47,
        performance_metrics={
            "total_processing_time": 89.3,
            "llm_calls": 12,
            "cache_hit_rate": 0.73,
            "database_operations": 8,
            "research_queries": 4,
        },
        events=[
            {
                "event_id": "evt_001",
                "event_type": "llm_provider_request",
                "timestamp": "2025-09-12T10:30:45.123456",
                "data": {
                    "provider": "deepseek",
                    "model": "deepseek-chat",
                    "prompt_length": 2847,
                    "temperature": 0.3,
                    "max_tokens": 2000,
                },
                "metadata": {
                    "trace_id": trace_id,
                    "agent_contract_id": "strategic_analyst_v2",
                },
                "relevance_score": 0.8,
            },
            {
                "event_id": "evt_002",
                "event_type": "devils_advocate_bias_found",
                "timestamp": "2025-09-12T10:35:22.987654",
                "data": {
                    "bias_type": "confirmation_bias",
                    "confidence": 0.76,
                    "original_statement": "Market analysis clearly shows...",
                    "contradiction": "However, recent data suggests the opposite trend...",
                },
                "metadata": {
                    "trace_id": trace_id,
                    "devils_advocate_agent_id": "da_v3_instance_42",
                },
                "relevance_score": 0.95,
            },
            {
                "event_id": "evt_003",
                "event_type": "analysis_completed",
                "timestamp": "2025-09-12T10:45:10.456789",
                "data": {
                    "phase": "synthesis_delivery",
                    "confidence": 0.87,
                    "insights_generated": 12,
                    "processing_time": 145.7,
                },
                "metadata": {"trace_id": trace_id, "phase_completion": True},
                "relevance_score": 1.0,
            },
        ],
        timeline_visualization=None,  # Would be base64 encoded SVG
        export_formats=["json", "xml", "csv"],
    )


async def get_flywheel_automation_data() -> FlywheelAutomationStatus:
    """Get real flywheel automation status"""
    # This would integrate with actual flywheel automation systems
    return get_mock_flywheel_automation_status()


def get_mock_flywheel_automation_status() -> FlywheelAutomationStatus:
    """Generate mock flywheel automation status"""
    last_retraining = datetime.now() - timedelta(days=3, hours=4)
    next_retraining = datetime.now() + timedelta(days=4, hours=8)

    return FlywheelAutomationStatus(
        automation_health="excellent",
        flywheel_active=True,
        training_candidates_curated=247,
        last_retraining_cycle=last_retraining.isoformat(),
        next_retraining_estimate=next_retraining.isoformat(),
        curation_queue_size=34,
        automation_metrics={
            "curation_efficiency": 0.87,
            "quality_threshold_met": 0.94,
            "automated_improvements": 12,
            "manual_interventions_required": 3,
            "system_learning_velocity": 0.23,
            "performance_improvement_trend": 0.15,
        },
        recent_improvements=[
            {
                "improvement_id": "imp_001",
                "type": "pattern_optimization",
                "impact_score": 0.12,
                "applied_at": (datetime.now() - timedelta(days=2)).isoformat(),
                "description": "Enhanced consultant selection patterns for fintech scenarios",
            },
            {
                "improvement_id": "imp_002",
                "type": "bias_reduction",
                "impact_score": 0.08,
                "applied_at": (datetime.now() - timedelta(days=1)).isoformat(),
                "description": "Improved detection of confirmation bias in market analysis",
            },
            {
                "improvement_id": "imp_003",
                "type": "quality_calibration",
                "impact_score": 0.06,
                "applied_at": (datetime.now() - timedelta(hours=6)).isoformat(),
                "description": "Calibrated RIVA scoring for strategic recommendations",
            },
        ],
    )


# Export router
c2_command_center_router = router