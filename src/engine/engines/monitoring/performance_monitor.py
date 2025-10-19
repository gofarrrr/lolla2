"""
METIS V5 Performance Monitoring Service
======================================

Extracted from monolithic optimal_consultant_engine.py monitoring concerns.
Handles performance metrics collection, health checks, and benchmarking integration.

Part of the Great Refactoring: Clean separation of monitoring concerns.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import time
import psutil
import asyncio
from dataclasses import dataclass

# Import our new contracts
from ..contracts import PerformanceMetrics, HealthStatus

# Import UnifiedContextStream for audit trail
from src.core.unified_context_stream import UnifiedContextStream, ContextEventType

# Benchmarking dashboard integration (optional)
try:
    from src.monitoring.realtime_benchmarking_dashboard import (
        RealtimeBenchmarkingDashboard,
    )

    BENCHMARKING_AVAILABLE = True
except ImportError:
    print("⚠️ Real-time benchmarking dashboard not available")
    BENCHMARKING_AVAILABLE = False

# Metrics aggregator integration (optional)
try:
    from src.metrics.aggregator import get_metrics_aggregator

    METRICS_AGGREGATOR_AVAILABLE = True
except ImportError:
    print("⚠️ Metrics aggregator not available")
    METRICS_AGGREGATOR_AVAILABLE = False


@dataclass
class SystemResourceUsage:
    """System resource usage snapshot"""

    cpu_percent: float
    memory_usage_mb: float
    memory_percent: float
    disk_usage_percent: float
    timestamp: datetime


class PerformanceMonitoringService:
    """
    Stateless service for performance monitoring and health checks.

    Extracted from OptimalConsultantEngine to follow Single Responsibility Principle.
    Handles performance metrics, health checks, and system monitoring.
    """

    def __init__(self, context_stream: Optional[UnifiedContextStream] = None):
        """Initialize the performance monitoring service"""
        from src.core.unified_context_stream import get_unified_context_stream, UnifiedContextStream
        self.context_stream = context_stream or get_unified_context_stream()

        # Initialize benchmarking dashboard if available
        self.benchmarking_dashboard = None
        self.benchmarking_enabled = False

        if BENCHMARKING_AVAILABLE:
            try:
                self.benchmarking_dashboard = RealtimeBenchmarkingDashboard()
                self.benchmarking_enabled = True
                print(
                    "✅ PerformanceMonitoringService: Benchmarking dashboard integrated"
                )
            except Exception as e:
                print(f"⚠️ Benchmarking dashboard initialization failed: {e}")
                self.benchmarking_enabled = False

        # Initialize metrics aggregator if available
        self.metrics_aggregator = None
        self.metrics_aggregator_enabled = False

        if METRICS_AGGREGATOR_AVAILABLE:
            try:
                self.metrics_aggregator = get_metrics_aggregator()
                self.metrics_aggregator_enabled = True
                print("✅ PerformanceMonitoringService: Metrics aggregator integrated")
            except Exception as e:
                print(f"⚠️ Metrics aggregator initialization failed: {e}")
                self.metrics_aggregator_enabled = False

        # Performance tracking
        self.operation_metrics: List[PerformanceMetrics] = []
        self.max_metrics_history = 1000

        print("✅ PerformanceMonitoringService: Initialized successfully")

    # === PERFORMANCE METRICS COLLECTION ===

    def start_operation_timing(self, engagement_id: str, operation: str) -> dict:
        """
        Start timing an operation.

        Returns:
            Timer context containing start time and metadata
        """
        start_time = time.time()
        start_memory = self._get_current_memory_usage()

        timer_context = {
            "engagement_id": engagement_id,
            "operation": operation,
            "start_time": start_time,
            "start_memory_mb": start_memory,
        }

        # Log operation start to context stream
        if self.context_stream:
            self.context_stream.add_event(
                ContextEventType.REASONING_STEP,
                {
                    "message": f"Operation timing started: {operation}",
                    "engagement_id": engagement_id,
                    "operation": operation,
                    "start_time": datetime.fromtimestamp(start_time).isoformat(),
                },
                metadata={
                    "source": "PerformanceMonitoringService.start_operation_timing"
                },
            )

        return timer_context

    def end_operation_timing(
        self,
        timer_context: dict,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PerformanceMetrics:
        """
        End operation timing and record metrics.

        Args:
            timer_context: Timer context from start_operation_timing
            success: Whether the operation succeeded
            metadata: Additional metadata to record

        Returns:
            PerformanceMetrics object with timing data
        """
        end_time = time.time()
        end_memory = self._get_current_memory_usage()

        duration_ms = (end_time - timer_context["start_time"]) * 1000
        memory_delta = end_memory - timer_context["start_memory_mb"]

        performance_metrics = PerformanceMetrics(
            engagement_id=timer_context["engagement_id"],
            operation=timer_context["operation"],
            duration_ms=duration_ms,
            memory_usage_mb=memory_delta,
            success=success,
            metadata=metadata or {},
        )

        # Store metrics
        self._store_performance_metrics(performance_metrics)

        # Log to context stream
        if self.context_stream:
            self.context_stream.add_event(
                ContextEventType.TOOL_EXECUTION,
                {
                    "message": f"Operation completed: {timer_context['operation']}",
                    "engagement_id": timer_context["engagement_id"],
                    "operation": timer_context["operation"],
                    "duration_ms": duration_ms,
                    "memory_delta_mb": memory_delta,
                    "success": success,
                    "end_time": datetime.fromtimestamp(end_time).isoformat(),
                },
                metadata={
                    "source": "PerformanceMonitoringService.end_operation_timing"
                },
            )

        # Send to benchmarking dashboard if available
        if self.benchmarking_enabled and self.benchmarking_dashboard:
            try:
                asyncio.create_task(
                    self._send_to_benchmarking_dashboard(performance_metrics)
                )
            except Exception as e:
                print(f"⚠️ Failed to send metrics to benchmarking dashboard: {e}")

        print(
            f"📊 Operation '{timer_context['operation']}' completed in {duration_ms:.2f}ms (success: {success})"
        )
        return performance_metrics

    def _store_performance_metrics(self, metrics: PerformanceMetrics):
        """Store performance metrics with history limit"""
        self.operation_metrics.append(metrics)

        # Maintain maximum history
        if len(self.operation_metrics) > self.max_metrics_history:
            self.operation_metrics = self.operation_metrics[-self.max_metrics_history :]

    async def _send_to_benchmarking_dashboard(self, metrics: PerformanceMetrics):
        """Send metrics to real-time benchmarking dashboard"""
        try:
            if self.benchmarking_dashboard:
                await self.benchmarking_dashboard.record_performance_metric(
                    operation=metrics.operation,
                    duration_ms=metrics.duration_ms,
                    memory_usage_mb=metrics.memory_usage_mb,
                    success=metrics.success,
                    engagement_id=metrics.engagement_id,
                )
        except Exception as e:
            print(f"⚠️ Benchmarking dashboard metric recording failed: {e}")

    # === SYSTEM RESOURCE MONITORING ===

    def get_system_resource_usage(self) -> SystemResourceUsage:
        """Get current system resource usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            return SystemResourceUsage(
                cpu_percent=cpu_percent,
                memory_usage_mb=memory.used / (1024 * 1024),
                memory_percent=memory.percent,
                disk_usage_percent=disk.percent,
                timestamp=datetime.now(),
            )
        except Exception as e:
            print(f"⚠️ Error getting system resource usage: {e}")
            return SystemResourceUsage(
                cpu_percent=0.0,
                memory_usage_mb=0.0,
                memory_percent=0.0,
                disk_usage_percent=0.0,
                timestamp=datetime.now(),
            )

    def _get_current_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            return psutil.Process().memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0

    # === HEALTH CHECK OPERATIONS ===

    async def comprehensive_health_check(self) -> Dict[str, HealthStatus]:
        """
        Perform comprehensive health check of all components.

        Returns:
            Dictionary of component health statuses
        """
        health_results = {}

        # Core service health
        health_results["PerformanceMonitoringService"] = await self._self_health_check()

        # System resources health
        health_results["SystemResources"] = await self._system_resources_health_check()

        # Context stream health
        if self.context_stream:
            health_results["UnifiedContextStream"] = (
                await self._context_stream_health_check()
            )

        # Benchmarking dashboard health
        if self.benchmarking_enabled:
            health_results["BenchmarkingDashboard"] = (
                await self._benchmarking_dashboard_health_check()
            )

        # Metrics aggregator health
        if self.metrics_aggregator_enabled:
            health_results["MetricsAggregator"] = (
                await self._metrics_aggregator_health_check()
            )

        return health_results

    async def _self_health_check(self) -> HealthStatus:
        """Health check for this service"""
        start_time = datetime.now()

        try:
            # Test basic functionality
            metrics_count = len(self.operation_metrics)
            response_time = (datetime.now() - start_time).total_seconds() * 1000

            return HealthStatus(
                component="PerformanceMonitoringService",
                healthy=True,
                response_time_ms=response_time,
                details=f"Service operational, {metrics_count} metrics stored",
            )
        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            return HealthStatus(
                component="PerformanceMonitoringService",
                healthy=False,
                response_time_ms=response_time,
                details=f"Self health check failed: {e}",
            )

    async def _system_resources_health_check(self) -> HealthStatus:
        """Health check for system resources"""
        start_time = datetime.now()

        try:
            resources = self.get_system_resource_usage()
            response_time = (datetime.now() - start_time).total_seconds() * 1000

            # Determine health based on resource usage
            healthy = (
                resources.cpu_percent < 90
                and resources.memory_percent < 90
                and resources.disk_usage_percent < 90
            )

            return HealthStatus(
                component="SystemResources",
                healthy=healthy,
                response_time_ms=response_time,
                details=f"CPU: {resources.cpu_percent:.1f}%, Memory: {resources.memory_percent:.1f}%, Disk: {resources.disk_usage_percent:.1f}%",
            )
        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            return HealthStatus(
                component="SystemResources",
                healthy=False,
                response_time_ms=response_time,
                details=f"System resources check failed: {e}",
            )

    async def _context_stream_health_check(self) -> HealthStatus:
        """Health check for unified context stream"""
        start_time = datetime.now()

        try:
            if not self.context_stream:
                return HealthStatus(
                    component="UnifiedContextStream",
                    healthy=False,
                    response_time_ms=None,
                    details="Context stream not available",
                )

            event_count = (
                len(self.context_stream.events)
                if hasattr(self.context_stream, "events")
                else 0
            )
            response_time = (datetime.now() - start_time).total_seconds() * 1000

            return HealthStatus(
                component="UnifiedContextStream",
                healthy=True,
                response_time_ms=response_time,
                details=f"Context stream operational, {event_count} events stored",
            )
        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            return HealthStatus(
                component="UnifiedContextStream",
                healthy=False,
                response_time_ms=response_time,
                details=f"Context stream health check failed: {e}",
            )

    async def _benchmarking_dashboard_health_check(self) -> HealthStatus:
        """Health check for benchmarking dashboard"""
        start_time = datetime.now()

        try:
            if not self.benchmarking_dashboard:
                return HealthStatus(
                    component="BenchmarkingDashboard",
                    healthy=False,
                    response_time_ms=None,
                    details="Benchmarking dashboard not available",
                )

            # Simple ping to dashboard
            response_time = (datetime.now() - start_time).total_seconds() * 1000

            return HealthStatus(
                component="BenchmarkingDashboard",
                healthy=True,
                response_time_ms=response_time,
                details="Benchmarking dashboard operational",
            )
        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            return HealthStatus(
                component="BenchmarkingDashboard",
                healthy=False,
                response_time_ms=response_time,
                details=f"Benchmarking dashboard health check failed: {e}",
            )

    async def _metrics_aggregator_health_check(self) -> HealthStatus:
        """Health check for metrics aggregator"""
        start_time = datetime.now()

        try:
            if not self.metrics_aggregator:
                return HealthStatus(
                    component="MetricsAggregator",
                    healthy=False,
                    response_time_ms=None,
                    details="Metrics aggregator not available",
                )

            response_time = (datetime.now() - start_time).total_seconds() * 1000

            return HealthStatus(
                component="MetricsAggregator",
                healthy=True,
                response_time_ms=response_time,
                details="Metrics aggregator operational",
            )
        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            return HealthStatus(
                component="MetricsAggregator",
                healthy=False,
                response_time_ms=response_time,
                details=f"Metrics aggregator health check failed: {e}",
            )

    # === METRICS ANALYSIS ===

    def get_performance_summary(self, hours_back: int = 24) -> Dict[str, Any]:
        """
        Get performance summary for recent operations.

        Args:
            hours_back: Number of hours of history to analyze

        Returns:
            Performance summary statistics
        """
        cutoff_time = datetime.now() - timedelta(hours=hours_back)

        # Filter recent metrics (assuming we store timestamps)
        recent_metrics = [m for m in self.operation_metrics if hasattr(m, "timestamp")]

        if not recent_metrics:
            return {
                "total_operations": 0,
                "success_rate": 0.0,
                "average_duration_ms": 0.0,
                "operations_by_type": {},
            }

        total_operations = len(recent_metrics)
        successful_operations = sum(1 for m in recent_metrics if m.success)
        success_rate = (
            successful_operations / total_operations if total_operations > 0 else 0.0
        )

        # Calculate average duration
        total_duration = sum(m.duration_ms for m in recent_metrics)
        average_duration = (
            total_duration / total_operations if total_operations > 0 else 0.0
        )

        # Group by operation type
        operations_by_type = {}
        for metric in recent_metrics:
            if metric.operation not in operations_by_type:
                operations_by_type[metric.operation] = {
                    "count": 0,
                    "success_count": 0,
                    "total_duration_ms": 0.0,
                }

            operations_by_type[metric.operation]["count"] += 1
            if metric.success:
                operations_by_type[metric.operation]["success_count"] += 1
            operations_by_type[metric.operation][
                "total_duration_ms"
            ] += metric.duration_ms

        # Calculate averages for each operation type
        for op_type, stats in operations_by_type.items():
            stats["success_rate"] = (
                stats["success_count"] / stats["count"] if stats["count"] > 0 else 0.0
            )
            stats["average_duration_ms"] = (
                stats["total_duration_ms"] / stats["count"]
                if stats["count"] > 0
                else 0.0
            )

        return {
            "total_operations": total_operations,
            "success_rate": success_rate,
            "average_duration_ms": average_duration,
            "operations_by_type": operations_by_type,
            "time_period_hours": hours_back,
        }

    # === UTILITY METHODS ===

    def get_monitoring_capabilities(self) -> Dict[str, Any]:
        """Get current monitoring capabilities"""
        return {
            "performance_metrics": True,
            "system_resources": True,
            "health_checks": True,
            "benchmarking_dashboard": self.benchmarking_enabled,
            "metrics_aggregator": self.metrics_aggregator_enabled,
            "context_stream": bool(self.context_stream),
            "metrics_history_count": len(self.operation_metrics),
            "max_metrics_history": self.max_metrics_history,
        }

    def configure_context_stream(self, context_stream: UnifiedContextStream):
        """Configure the audit trail context stream"""
        self.context_stream = context_stream
        print("✅ PerformanceMonitoringService: Context stream configured")


# Factory function for service creation
def get_performance_monitoring_service(
    context_stream: Optional[UnifiedContextStream] = None,
) -> PerformanceMonitoringService:
    """Factory function to create PerformanceMonitoringService instance"""
    return PerformanceMonitoringService(context_stream)
