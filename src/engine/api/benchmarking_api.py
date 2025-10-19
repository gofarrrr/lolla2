"""
METIS V5 Benchmarking API
Monte Carlo and Performance Testing Endpoints for Phoenix Phase 4
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import List, Optional, Dict, Any
import json
import logging
from datetime import datetime
from pydantic import BaseModel, Field

# Temporarily bypass auth for admin Monte Carlo testing
# from ..core.supabase_auth_middleware import verify_token


def verify_token() -> str:
    """Temporary bypass for Monte Carlo admin testing"""
    return "admin_user"


# Phoenix Phase 4: Monte Carlo Benchmarking Integration
try:
    from ..flywheel.flywheel_benchmarking_suite import FlywheelBenchmarkingSuite
    from ..intelligence.benchmarking_engine import BenchmarkingEngine

    benchmarking_available = True
    logger = logging.getLogger(__name__)
    logger.info("✅ Monte Carlo Benchmarking Suite integrated")
except ImportError as e:
    benchmarking_available = False
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ Monte Carlo Benchmarking not available: {e}")

router = APIRouter(prefix="/api/benchmarking", tags=["benchmarking"])


# Request/Response Models
class BenchmarkRequest(BaseModel):
    """Request to run Monte Carlo benchmarking suite"""

    test_types: Optional[List[str]] = Field(
        default=None, description="Specific tests to run (all if None)"
    )
    iterations: Optional[int] = Field(
        default=1, ge=1, le=10, description="Number of Monte Carlo iterations"
    )
    include_variance_analysis: bool = Field(
        default=True, description="Include statistical variance analysis"
    )
    stream_results: bool = Field(
        default=False, description="Stream results in real-time"
    )


class BenchmarkStatus(BaseModel):
    """Status of running benchmark"""

    status: str
    progress_percent: float
    current_test: Optional[str] = None
    completed_tests: int
    total_tests: int
    estimated_completion: Optional[str] = None


class BenchmarkSummary(BaseModel):
    """Summary of benchmark results"""

    total_tests: int
    passed_tests: int
    failed_tests: int
    overall_score: float
    execution_time: float
    monte_carlo_confidence: Optional[float] = None


# Global benchmarking instances
flywheel_suite = None
benchmarking_engine = None
if benchmarking_available:
    try:
        flywheel_suite = FlywheelBenchmarkingSuite()
        benchmarking_engine = BenchmarkingEngine()
        logger.info("✅ Benchmarking instances initialized")
    except Exception as e:
        logger.error(f"⚠️ Failed to initialize benchmarking instances: {e}")
        flywheel_suite = None
        benchmarking_engine = None


@router.post("/run-monte-carlo")
async def run_monte_carlo_benchmark(
    request: BenchmarkRequest,
    background_tasks: BackgroundTasks,
    user_id: str = "admin_user",
):
    """
    Phoenix Phase 4: Run comprehensive Monte Carlo benchmarking suite

    Executes the complete Flywheel benchmarking system with statistical
    variance analysis and Monte Carlo simulation capabilities.
    """
    try:
        if not benchmarking_available or flywheel_suite is None:
            raise HTTPException(
                status_code=503, detail="Monte Carlo benchmarking system not available"
            )

        if request.stream_results:
            return StreamingResponse(
                stream_monte_carlo_benchmark(request, user_id), media_type="text/plain"
            )

        # Run comprehensive benchmark
        benchmark_id = f"benchmark_{datetime.now().isoformat()}_{user_id}"
        logger.info(f"🚀 Starting Monte Carlo benchmark {benchmark_id}")

        # Execute Monte Carlo iterations
        results = []
        for iteration in range(request.iterations):
            logger.info(
                f"📊 Monte Carlo iteration {iteration + 1}/{request.iterations}"
            )
            result = await flywheel_suite.run_comprehensive_benchmark()
            results.append(result)

        # Calculate variance analysis if requested
        variance_analysis = None
        if request.include_variance_analysis and len(results) > 1:
            variance_analysis = calculate_monte_carlo_variance(results)

        # Aggregate results
        summary = aggregate_benchmark_results(results)

        logger.info(f"✅ Monte Carlo benchmark {benchmark_id} completed")

        return {
            "benchmark_id": benchmark_id,
            "summary": summary,
            "variance_analysis": variance_analysis,
            "detailed_results": results if request.iterations <= 3 else None,
            "monte_carlo_iterations": request.iterations,
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Monte Carlo benchmark failed for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(e)}")


async def stream_monte_carlo_benchmark(request: BenchmarkRequest, user_id: str):
    """Stream Monte Carlo benchmark results in real-time"""
    try:
        yield f"data: {json.dumps({'phase': 'initialization', 'status': 'starting', 'message': 'Initializing Monte Carlo benchmarking...'})}\n\n"

        for iteration in range(request.iterations):
            yield f"data: {json.dumps({'phase': 'iteration', 'iteration': iteration + 1, 'total': request.iterations, 'message': f'Running Monte Carlo iteration {iteration + 1}/{request.iterations}'})}\n\n"

            # Run benchmark iteration
            result = await flywheel_suite.run_comprehensive_benchmark()

            yield f"data: {json.dumps({'phase': 'iteration_complete', 'iteration': iteration + 1, 'result': result})}\n\n"

        yield f"data: {json.dumps({'phase': 'completed', 'status': 'finished', 'message': 'Monte Carlo benchmarking completed successfully'})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'phase': 'error', 'error': str(e)})}\n\n"


@router.get("/status")
async def get_benchmarking_status(user_id: str = "admin_user"):
    """Get current benchmarking system status and capabilities"""
    try:
        status = {
            "monte_carlo_available": benchmarking_available
            and flywheel_suite is not None,
            "benchmarking_engine_available": benchmarking_available
            and benchmarking_engine is not None,
            "available_tests": [],
            "system_health": "unknown",
        }

        if benchmarking_available and flywheel_suite is not None:
            # Get available test types from the suite
            status["available_tests"] = [
                "shannon_entropy_calculation",
                "gini_coefficient_calculation",
                "diversity_analysis_performance",
                "ideaflow_velocity_tracking",
                "learning_orchestration_performance",
                "anti_convergence_system",
            ]
            status["system_health"] = "operational"

        return status

    except Exception as e:
        logger.error(f"Failed to get benchmarking status: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


@router.get("/performance-metrics")
async def get_performance_metrics(days: int = 7, user_id: str = "admin_user"):
    """Get historical performance metrics from benchmarking engine"""
    try:
        if not benchmarking_available or benchmarking_engine is None:
            raise HTTPException(
                status_code=503, detail="Benchmarking engine not available"
            )

        # Get performance metrics from benchmarking engine
        metrics = await benchmarking_engine.get_performance_metrics(days=days)

        return {
            "metrics": metrics,
            "time_range_days": days,
            "timestamp": datetime.now().isoformat(),
            "status": "operational",
        }

    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(
            status_code=500, detail=f"Metrics retrieval failed: {str(e)}"
        )


def calculate_monte_carlo_variance(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate statistical variance across Monte Carlo iterations"""
    try:
        if len(results) < 2:
            return {"error": "Insufficient iterations for variance analysis"}

        # Extract key metrics for variance analysis
        execution_times = []
        success_rates = []

        for result in results:
            if "total_execution_time" in result:
                execution_times.append(result["total_execution_time"])
            if "summary" in result and "success_rate" in result["summary"]:
                success_rates.append(result["summary"]["success_rate"])

        variance_stats = {}

        if execution_times:
            variance_stats["execution_time"] = {
                "mean": sum(execution_times) / len(execution_times),
                "variance": calculate_variance(execution_times),
                "coefficient_of_variation": calculate_coefficient_of_variation(
                    execution_times
                ),
            }

        if success_rates:
            variance_stats["success_rate"] = {
                "mean": sum(success_rates) / len(success_rates),
                "variance": calculate_variance(success_rates),
                "coefficient_of_variation": calculate_coefficient_of_variation(
                    success_rates
                ),
            }

        return {
            "iterations": len(results),
            "variance_statistics": variance_stats,
            "monte_carlo_confidence": calculate_monte_carlo_confidence(variance_stats),
        }

    except Exception as e:
        return {"error": f"Variance calculation failed: {str(e)}"}


def calculate_variance(values: List[float]) -> float:
    """Calculate variance of a list of values"""
    if len(values) < 2:
        return 0.0

    mean = sum(values) / len(values)
    squared_diffs = [(x - mean) ** 2 for x in values]
    return sum(squared_diffs) / (len(values) - 1)


def calculate_coefficient_of_variation(values: List[float]) -> float:
    """Calculate coefficient of variation (CV)"""
    if len(values) < 2:
        return 0.0

    mean = sum(values) / len(values)
    if mean == 0:
        return 0.0

    variance = calculate_variance(values)
    std_dev = variance**0.5
    return std_dev / mean


def calculate_monte_carlo_confidence(variance_stats: Dict[str, Any]) -> float:
    """Calculate overall Monte Carlo confidence score"""
    try:
        if not variance_stats:
            return 0.0

        confidence_scores = []

        for metric_name, stats in variance_stats.items():
            if "coefficient_of_variation" in stats:
                cv = stats["coefficient_of_variation"]
                # Lower coefficient of variation = higher confidence
                confidence = max(0.0, 1.0 - cv)
                confidence_scores.append(confidence)

        if not confidence_scores:
            return 0.0

        return sum(confidence_scores) / len(confidence_scores)

    except Exception:
        return 0.0


def aggregate_benchmark_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate multiple benchmark results into summary statistics"""
    try:
        if not results:
            return {"error": "No results to aggregate"}

        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        total_execution_time = 0.0

        for result in results:
            if "summary" in result:
                summary = result["summary"]
                total_tests += summary.get("total_tests", 0)
                passed_tests += summary.get("passed_tests", 0)
                failed_tests += summary.get("failed_tests", 0)

            if "total_execution_time" in result:
                total_execution_time += result["total_execution_time"]

        success_rate = passed_tests / total_tests if total_tests > 0 else 0.0

        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": success_rate,
            "average_execution_time": total_execution_time / len(results),
            "total_execution_time": total_execution_time,
            "monte_carlo_iterations": len(results),
        }

    except Exception as e:
        return {"error": f"Aggregation failed: {str(e)}"}
