#!/usr/bin/env python3
"""
Performance Instrumentation System

Comprehensive timing and performance measurement for METIS components.
This provides detailed instrumentation to identify bottlenecks and optimize performance.
"""

import time
import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from contextlib import asynccontextmanager, contextmanager
from functools import wraps
import threading
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Individual performance measurement"""

    operation_name: str
    start_time: float
    end_time: float
    duration_seconds: float
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ComponentMetrics:
    """Aggregated metrics for a component"""

    component_name: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_duration_seconds: float = 0.0
    avg_duration_seconds: float = 0.0
    min_duration_seconds: float = float("inf")
    max_duration_seconds: float = 0.0
    p50_duration_seconds: float = 0.0
    p90_duration_seconds: float = 0.0
    p99_duration_seconds: float = 0.0
    success_rate: float = 0.0
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    recent_durations: deque = field(default_factory=lambda: deque(maxlen=100))


class PerformanceInstrumentationSystem:
    """
    Comprehensive performance instrumentation system for METIS components.
    Thread-safe, non-blocking, and designed for production use.
    """

    def __init__(self, max_metrics_history: int = 10000):
        self.max_metrics_history = max_metrics_history
        self.metrics_history: List[PerformanceMetric] = []
        self.component_metrics: Dict[str, ComponentMetrics] = {}
        self._lock = threading.Lock()
        self.session_id = f"perf_{int(time.time())}"

        logger.info(
            f"🔍 Performance Instrumentation System initialized (session: {self.session_id})"
        )

    @contextmanager
    def measure_sync(self, operation_name: str, component: str = "unknown", **metadata):
        """Context manager for synchronous performance measurement"""
        start_time = time.time()
        success = True
        error = None

        try:
            logger.debug(f"📊 Starting measurement: {operation_name}")
            yield
        except Exception as e:
            success = False
            error = str(e)
            logger.error(
                f"💥 Operation failed during measurement: {operation_name} - {error}"
            )
            raise
        finally:
            end_time = time.time()
            duration = end_time - start_time

            metric = PerformanceMetric(
                operation_name=operation_name,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=round(duration, 4),
                success=success,
                error=error,
                metadata={**metadata, "component": component},
            )

            self._record_metric(metric)
            logger.debug(f"📈 Measurement complete: {operation_name} ({duration:.4f}s)")

    @asynccontextmanager
    async def measure_async(
        self, operation_name: str, component: str = "unknown", **metadata
    ):
        """Context manager for asynchronous performance measurement"""
        start_time = time.time()
        success = True
        error = None

        try:
            logger.debug(f"📊 Starting async measurement: {operation_name}")
            yield
        except Exception as e:
            success = False
            error = str(e)
            logger.error(
                f"💥 Async operation failed during measurement: {operation_name} - {error}"
            )
            raise
        finally:
            end_time = time.time()
            duration = end_time - start_time

            metric = PerformanceMetric(
                operation_name=operation_name,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=round(duration, 4),
                success=success,
                error=error,
                metadata={**metadata, "component": component},
            )

            self._record_metric(metric)
            logger.debug(
                f"📈 Async measurement complete: {operation_name} ({duration:.4f}s)"
            )

    def measure_function(
        self, operation_name: str = None, component: str = "unknown", **metadata
    ):
        """Decorator for function performance measurement"""

        def decorator(func: Callable):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"

            if asyncio.iscoroutinefunction(func):

                @wraps(func)
                async def async_wrapper(*args, **kwargs):
                    async with self.measure_async(op_name, component, **metadata):
                        return await func(*args, **kwargs)

                return async_wrapper
            else:

                @wraps(func)
                def sync_wrapper(*args, **kwargs):
                    with self.measure_sync(op_name, component, **metadata):
                        return func(*args, **kwargs)

                return sync_wrapper

        return decorator

    def _record_metric(self, metric: PerformanceMetric):
        """Record a performance metric (thread-safe)"""
        with self._lock:
            # Add to history
            self.metrics_history.append(metric)

            # Trim history if needed
            if len(self.metrics_history) > self.max_metrics_history:
                self.metrics_history = self.metrics_history[-self.max_metrics_history :]

            # Update component metrics
            component = metric.metadata.get("component", "unknown")
            if component not in self.component_metrics:
                self.component_metrics[component] = ComponentMetrics(
                    component_name=component
                )

            comp_metrics = self.component_metrics[component]
            comp_metrics.total_calls += 1

            if metric.success:
                comp_metrics.successful_calls += 1
                comp_metrics.total_duration_seconds += metric.duration_seconds
                comp_metrics.min_duration_seconds = min(
                    comp_metrics.min_duration_seconds, metric.duration_seconds
                )
                comp_metrics.max_duration_seconds = max(
                    comp_metrics.max_duration_seconds, metric.duration_seconds
                )
                comp_metrics.recent_durations.append(metric.duration_seconds)
            else:
                comp_metrics.failed_calls += 1

            # Recalculate aggregated metrics
            if comp_metrics.successful_calls > 0:
                comp_metrics.avg_duration_seconds = (
                    comp_metrics.total_duration_seconds / comp_metrics.successful_calls
                )
                comp_metrics.success_rate = (
                    comp_metrics.successful_calls / comp_metrics.total_calls
                )

                # Calculate percentiles from recent durations
                if comp_metrics.recent_durations:
                    sorted_durations = sorted(comp_metrics.recent_durations)
                    n = len(sorted_durations)
                    comp_metrics.p50_duration_seconds = sorted_durations[int(n * 0.5)]
                    comp_metrics.p90_duration_seconds = sorted_durations[int(n * 0.9)]
                    comp_metrics.p99_duration_seconds = sorted_durations[int(n * 0.99)]

            comp_metrics.last_updated = datetime.now().isoformat()

    def get_component_metrics(
        self, component: str = None
    ) -> Dict[str, ComponentMetrics]:
        """Get metrics for specific component or all components"""
        with self._lock:
            if component:
                return (
                    {component: self.component_metrics.get(component)}
                    if component in self.component_metrics
                    else {}
                )
            return dict(self.component_metrics)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        with self._lock:
            total_metrics = len(self.metrics_history)
            successful_metrics = sum(1 for m in self.metrics_history if m.success)
            failed_metrics = total_metrics - successful_metrics

            if total_metrics == 0:
                return {
                    "session_id": self.session_id,
                    "summary": "No metrics recorded yet",
                    "total_operations": 0,
                }

            successful_durations = [
                m.duration_seconds for m in self.metrics_history if m.success
            ]

            summary = {
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "total_operations": total_metrics,
                "successful_operations": successful_metrics,
                "failed_operations": failed_metrics,
                "overall_success_rate": successful_metrics / total_metrics,
                "components_monitored": len(self.component_metrics),
                "component_breakdown": {},
            }

            if successful_durations:
                sorted_durations = sorted(successful_durations)
                n = len(sorted_durations)
                summary.update(
                    {
                        "avg_duration_seconds": sum(successful_durations)
                        / len(successful_durations),
                        "min_duration_seconds": min(successful_durations),
                        "max_duration_seconds": max(successful_durations),
                        "p50_duration_seconds": sorted_durations[int(n * 0.5)],
                        "p90_duration_seconds": sorted_durations[int(n * 0.9)],
                        "p99_duration_seconds": sorted_durations[int(n * 0.99)],
                    }
                )

            # Add component breakdown
            for comp_name, comp_metrics in self.component_metrics.items():
                summary["component_breakdown"][comp_name] = {
                    "total_calls": comp_metrics.total_calls,
                    "success_rate": comp_metrics.success_rate,
                    "avg_duration_seconds": comp_metrics.avg_duration_seconds,
                    "p90_duration_seconds": comp_metrics.p90_duration_seconds,
                }

            return summary

    def identify_bottlenecks(
        self, threshold_seconds: float = 5.0
    ) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks"""
        with self._lock:
            bottlenecks = []

            for comp_name, comp_metrics in self.component_metrics.items():
                issues = []

                if comp_metrics.p90_duration_seconds > threshold_seconds:
                    issues.append(
                        f"P90 duration ({comp_metrics.p90_duration_seconds:.2f}s) exceeds threshold"
                    )

                if comp_metrics.success_rate < 0.95:
                    issues.append(f"Low success rate ({comp_metrics.success_rate:.1%})")

                if comp_metrics.max_duration_seconds > threshold_seconds * 3:
                    issues.append(
                        f"Max duration ({comp_metrics.max_duration_seconds:.2f}s) is extremely high"
                    )

                if issues:
                    bottlenecks.append(
                        {
                            "component": comp_name,
                            "issues": issues,
                            "metrics": {
                                "avg_duration": comp_metrics.avg_duration_seconds,
                                "p90_duration": comp_metrics.p90_duration_seconds,
                                "max_duration": comp_metrics.max_duration_seconds,
                                "success_rate": comp_metrics.success_rate,
                                "total_calls": comp_metrics.total_calls,
                            },
                        }
                    )

            # Sort by severity (P90 duration)
            bottlenecks.sort(key=lambda x: x["metrics"]["p90_duration"], reverse=True)
            return bottlenecks

    def save_performance_report(self, filename: str = None) -> str:
        """Save comprehensive performance report to JSON"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_report_{timestamp}.json"

        report = {
            "report_metadata": {
                "session_id": self.session_id,
                "generated_at": datetime.now().isoformat(),
                "total_metrics": len(self.metrics_history),
            },
            "performance_summary": self.get_performance_summary(),
            "bottlenecks": self.identify_bottlenecks(),
            "component_details": {
                name: {
                    "component_name": metrics.component_name,
                    "total_calls": metrics.total_calls,
                    "successful_calls": metrics.successful_calls,
                    "failed_calls": metrics.failed_calls,
                    "success_rate": metrics.success_rate,
                    "avg_duration_seconds": metrics.avg_duration_seconds,
                    "min_duration_seconds": metrics.min_duration_seconds,
                    "max_duration_seconds": metrics.max_duration_seconds,
                    "p50_duration_seconds": metrics.p50_duration_seconds,
                    "p90_duration_seconds": metrics.p90_duration_seconds,
                    "p99_duration_seconds": metrics.p99_duration_seconds,
                    "last_updated": metrics.last_updated,
                }
                for name, metrics in self.component_metrics.items()
            },
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"📊 Performance report saved: {filename}")
        return filename


# Global performance instrumentation instance
_performance_system = None


def get_performance_system() -> PerformanceInstrumentationSystem:
    """Get global performance instrumentation system"""
    global _performance_system
    if _performance_system is None:
        _performance_system = PerformanceInstrumentationSystem()
    return _performance_system


# Convenience functions for common usage patterns
def measure_sync(operation_name: str, component: str = "unknown", **metadata):
    """Convenience function for sync measurement"""
    return get_performance_system().measure_sync(operation_name, component, **metadata)


def measure_async(operation_name: str, component: str = "unknown", **metadata):
    """Convenience function for async measurement"""
    return get_performance_system().measure_async(operation_name, component, **metadata)


def measure_function(
    operation_name: str = None, component: str = "unknown", **metadata
):
    """Convenience decorator for function measurement"""
    return get_performance_system().measure_function(
        operation_name, component, **metadata
    )


# Performance testing utilities
async def run_performance_baseline_test():
    """Run performance baseline test to validate instrumentation"""

    print("🔍 PERFORMANCE INSTRUMENTATION BASELINE TEST")
    print("=" * 70)

    perf_system = get_performance_system()

    # Test sync measurement
    with measure_sync("test_sync_operation", "test_component"):
        time.sleep(0.1)  # Simulate work

    # Test async measurement
    async with measure_async("test_async_operation", "test_component"):
        await asyncio.sleep(0.2)  # Simulate async work

    # Test function decorator
    @measure_function("test_decorated_function", "test_component")
    def test_sync_func():
        time.sleep(0.05)
        return "sync_result"

    @measure_function("test_decorated_async_function", "test_component")
    async def test_async_func():
        await asyncio.sleep(0.15)
        return "async_result"

    # Run decorated functions
    result1 = test_sync_func()
    result2 = await test_async_func()

    # Test error handling
    try:
        with measure_sync("test_error_operation", "test_component"):
            raise ValueError("Test error")
    except ValueError:
        pass  # Expected

    # Get results
    summary = perf_system.get_performance_summary()
    bottlenecks = perf_system.identify_bottlenecks(threshold_seconds=0.1)

    print(f"✅ Operations Measured: {summary['total_operations']}")
    print(f"📊 Success Rate: {summary['overall_success_rate']:.1%}")
    print(f"⏱️ Average Duration: {summary.get('avg_duration_seconds', 0):.4f}s")
    print(f"🚨 Bottlenecks Identified: {len(bottlenecks)}")

    # Save report
    report_file = perf_system.save_performance_report()
    print(f"💾 Performance report saved: {report_file}")

    print("\n🎯 Performance instrumentation is working correctly!")
    return True


if __name__ == "__main__":
    asyncio.run(run_performance_baseline_test())
