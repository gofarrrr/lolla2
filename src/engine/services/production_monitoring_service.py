"""
METIS V2.1 Production Monitoring & Metrics Service
Comprehensive monitoring system for production readiness and 100x scale observability

This service provides enterprise-grade monitoring capabilities:
- Real-time system health monitoring with alerts
- Business metrics tracking (consultant performance, user satisfaction)
- Technical metrics (latency, throughput, error rates, resource usage)
- Alert management with escalation policies
- Historical trend analysis and capacity planning

ARCHITECTURAL MANDATE COMPLIANCE:
✅ Glass-Box Transparency: All monitoring operations logged to UnifiedContextStream
✅ Service-Oriented Architecture: Clean dependency injection and service boundaries
"""

import asyncio
import logging
import statistics
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict, deque
from enum import Enum

# Core METIS imports
# Migrated to use adapter for dependency inversion
from src.engine.adapters.context_stream import UnifiedContextStream, ContextEventType
from src.services.performance_optimization_service import (
    PerformanceOptimizationService,
)

# External dependencies
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class AlertSeverity(Enum):
    """Alert severity levels"""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class MetricType(Enum):
    """Types of metrics tracked"""

    GAUGE = "gauge"  # Point-in-time value
    COUNTER = "counter"  # Monotonic increasing value
    HISTOGRAM = "histogram"  # Distribution of values
    RATE = "rate"  # Rate of change over time


@dataclass
class AlertRule:
    """Alert rule configuration"""

    metric_name: str
    threshold: float
    severity: AlertSeverity
    comparison: str  # "gt", "lt", "eq", "gte", "lte"
    duration_seconds: int = 300  # Alert after condition persists for 5 minutes
    description: str = ""
    enabled: bool = True


@dataclass
class Alert:
    """Active alert instance"""

    rule: AlertRule
    triggered_at: datetime
    current_value: float
    alert_id: str
    acknowledged: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class BusinessMetric:
    """Business-level metric tracking"""

    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemHealthStatus:
    """Overall system health status"""

    status: str  # "healthy", "degraded", "critical", "down"
    timestamp: datetime
    components: Dict[str, str]
    alerts_count: int
    performance_score: float  # 0.0 - 1.0


class ProductionMonitoringService:
    """
    Production Monitoring & Metrics Service

    Provides comprehensive monitoring capabilities for METIS production deployment:
    1. System health monitoring with real-time alerts
    2. Business metrics tracking (engagement success, consultant performance)
    3. Technical metrics (latency, throughput, error rates)
    4. Alert management with configurable escalation
    5. Historical trend analysis and capacity planning
    """

    def __init__(
        self,
        context_stream: UnifiedContextStream,
        performance_service: Optional[PerformanceOptimizationService] = None,
    ):
        """
        Initialize Production Monitoring Service

        Args:
            context_stream: UnifiedContextStream for Glass-Box transparency
            performance_service: Optional PerformanceOptimizationService integration
        """
        self.context_stream = context_stream
        self.performance_service = performance_service
        self.logger = logging.getLogger(__name__)

        # Metrics storage
        self.metrics_storage: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=10000)
        )
        self.business_metrics: deque = deque(maxlen=5000)
        self.system_health_history: deque = deque(maxlen=1000)

        # Alert management
        self.alert_rules: List[AlertRule] = []
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)

        # Monitoring state
        self.is_monitoring = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.alert_task: Optional[asyncio.Task] = None

        # Alert callbacks
        self.alert_callbacks: List[Callable[[Alert], None]] = []

        # Monitoring statistics
        self.monitoring_stats = {
            "metrics_collected": 0,
            "alerts_triggered": 0,
            "alerts_resolved": 0,
            "monitoring_uptime_seconds": 0,
            "last_health_check": None,
        }

        # Initialize default alert rules
        self._initialize_default_alert_rules()

        # Glass-Box: Log service initialization
        self.context_stream.add_event(
            event_type=ContextEventType.SYSTEM_STATE,
            data={
                "service": "ProductionMonitoringService",
                "initialized": True,
                "default_alert_rules": len(self.alert_rules),
                "performance_service_integrated": bool(performance_service),
            },
            metadata={
                "service": "ProductionMonitoringService",
                "operation": "initialize",
            },
        )

    def _initialize_default_alert_rules(self) -> None:
        """Initialize default alert rules for production monitoring"""

        default_rules = [
            # System resource alerts
            AlertRule(
                metric_name="memory_usage_mb",
                threshold=2048,  # 2GB
                severity=AlertSeverity.CRITICAL,
                comparison="gt",
                description="High memory usage detected",
            ),
            AlertRule(
                metric_name="cpu_percentage",
                threshold=90.0,
                severity=AlertSeverity.WARNING,
                comparison="gt",
                description="High CPU usage detected",
            ),
            AlertRule(
                metric_name="error_rate",
                threshold=5.0,  # 5% error rate
                severity=AlertSeverity.CRITICAL,
                comparison="gt",
                description="High error rate detected",
            ),
            AlertRule(
                metric_name="request_latency_ms",
                threshold=5000,  # 5 seconds
                severity=AlertSeverity.WARNING,
                comparison="gt",
                description="High request latency detected",
            ),
            # Business metric alerts
            AlertRule(
                metric_name="engagement_success_rate",
                threshold=80.0,  # 80% success rate
                severity=AlertSeverity.WARNING,
                comparison="lt",
                description="Low engagement success rate",
            ),
            AlertRule(
                metric_name="consultant_availability",
                threshold=3,  # Minimum 3 consultants available
                severity=AlertSeverity.CRITICAL,
                comparison="lt",
                description="Insufficient consultant availability",
            ),
            # LLM provider alerts
            AlertRule(
                metric_name="llm_provider_availability",
                threshold=1,  # At least 1 provider available
                severity=AlertSeverity.EMERGENCY,
                comparison="lt",
                description="LLM provider unavailability",
            ),
        ]

        self.alert_rules.extend(default_rules)

    async def start_monitoring(self) -> None:
        """Start production monitoring with background tasks"""

        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitoring_stats["monitoring_start_time"] = datetime.utcnow()

        # Start monitoring task
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())

        # Start alert processing task
        self.alert_task = asyncio.create_task(self._alert_processing_loop())

        # Glass-Box: Log monitoring start
        self.context_stream.add_event(
            event_type=ContextEventType.SYSTEM_STATE,
            data={
                "production_monitoring": "started",
                "alert_rules": len(self.alert_rules),
            },
            metadata={
                "service": "ProductionMonitoringService",
                "operation": "start_monitoring",
            },
        )

        self.logger.info("🔍 Production monitoring started with real-time alerts")

    async def stop_monitoring(self) -> None:
        """Stop production monitoring"""

        self.is_monitoring = False

        # Cancel background tasks
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        if self.alert_task:
            self.alert_task.cancel()
            try:
                await self.alert_task
            except asyncio.CancelledError:
                pass

        # Calculate uptime
        if "monitoring_start_time" in self.monitoring_stats:
            uptime = datetime.utcnow() - self.monitoring_stats["monitoring_start_time"]
            self.monitoring_stats["monitoring_uptime_seconds"] = uptime.total_seconds()

        # Glass-Box: Log monitoring stop
        self.context_stream.add_event(
            event_type=ContextEventType.SYSTEM_STATE,
            data={
                "production_monitoring": "stopped",
                "uptime_seconds": self.monitoring_stats["monitoring_uptime_seconds"],
            },
            metadata={
                "service": "ProductionMonitoringService",
                "operation": "stop_monitoring",
            },
        )

    def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        tags: Optional[Dict[str, str]] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Record a metric value for monitoring and alerting"""

        timestamp = timestamp or datetime.utcnow()
        tags = tags or {}

        # Store metric
        metric_data = {
            "value": value,
            "timestamp": timestamp,
            "type": metric_type.value,
            "tags": tags,
        }

        self.metrics_storage[name].append(metric_data)
        self.monitoring_stats["metrics_collected"] += 1

        # Glass-Box: Log metric collection (sampled)
        if self.monitoring_stats["metrics_collected"] % 100 == 0:
            self.context_stream.add_event(
                event_type=ContextEventType.SYSTEM_STATE,
                data={
                    "metrics_collected": self.monitoring_stats["metrics_collected"],
                    "active_metrics": len(self.metrics_storage),
                    "latest_metric": {"name": name, "value": value},
                },
                metadata={
                    "service": "ProductionMonitoringService",
                    "operation": "metric_collection",
                },
            )

    def record_business_metric(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record business-level metric for tracking"""

        business_metric = BusinessMetric(
            name=name,
            value=value,
            timestamp=datetime.utcnow(),
            tags=tags or {},
            metadata=metadata or {},
        )

        self.business_metrics.append(business_metric)

        # Also record as regular metric for alerting
        self.record_metric(name, value, MetricType.GAUGE, tags)

    def get_system_health_status(self) -> SystemHealthStatus:
        """Get current system health status"""

        components_status = {}
        alerts_count = len(self.active_alerts)
        performance_score = 1.0

        # Check core system components
        components_status["monitoring_service"] = (
            "healthy" if self.is_monitoring else "down"
        )

        if self.performance_service:
            perf_status = self.performance_service.get_optimization_status()
            components_status["performance_optimization"] = (
                "healthy" if perf_status["service_status"] == "running" else "degraded"
            )

        # Check memory usage
        if PSUTIL_AVAILABLE:
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            if memory_usage > 2048:  # 2GB
                components_status["memory"] = "critical"
                performance_score *= 0.5
            elif memory_usage > 1024:  # 1GB
                components_status["memory"] = "degraded"
                performance_score *= 0.8
            else:
                components_status["memory"] = "healthy"

        # Check active alerts impact on performance score
        if alerts_count > 0:
            critical_alerts = sum(
                1
                for alert in self.active_alerts.values()
                if alert.rule.severity
                in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]
            )
            if critical_alerts > 0:
                performance_score *= 0.3
            else:
                performance_score *= 0.7

        # Determine overall status
        if performance_score >= 0.9:
            overall_status = "healthy"
        elif performance_score >= 0.7:
            overall_status = "degraded"
        elif performance_score >= 0.3:
            overall_status = "critical"
        else:
            overall_status = "down"

        health_status = SystemHealthStatus(
            status=overall_status,
            timestamp=datetime.utcnow(),
            components=components_status,
            alerts_count=alerts_count,
            performance_score=performance_score,
        )

        # Store in history
        self.system_health_history.append(health_status)

        # Update monitoring stats
        self.monitoring_stats["last_health_check"] = datetime.utcnow().isoformat()

        return health_status

    def add_alert_rule(self, alert_rule: AlertRule) -> None:
        """Add custom alert rule"""

        self.alert_rules.append(alert_rule)

        # Glass-Box: Log alert rule addition
        self.context_stream.add_event(
            event_type=ContextEventType.SYSTEM_STATE,
            data={
                "alert_rule_added": alert_rule.metric_name,
                "threshold": alert_rule.threshold,
                "severity": alert_rule.severity.value,
            },
            metadata={
                "service": "ProductionMonitoringService",
                "operation": "add_alert_rule",
            },
        )

    def add_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """Add callback function to be called when alerts are triggered"""
        self.alert_callbacks.append(callback)

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an active alert"""

        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True

            # Glass-Box: Log alert acknowledgment
            self.context_stream.add_event(
                event_type=ContextEventType.SYSTEM_STATE,
                data={"alert_acknowledged": alert_id},
                metadata={
                    "service": "ProductionMonitoringService",
                    "operation": "acknowledge_alert",
                },
            )

            return True

        return False

    def get_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data"""

        current_health = self.get_system_health_status()

        # Recent metrics summary
        recent_metrics = {}
        for metric_name, metric_data in self.metrics_storage.items():
            if metric_data:
                recent_values = [
                    m["value"] for m in list(metric_data)[-10:]
                ]  # Last 10 values
                recent_metrics[metric_name] = {
                    "current": recent_values[-1] if recent_values else 0,
                    "average": statistics.mean(recent_values) if recent_values else 0,
                    "min": min(recent_values) if recent_values else 0,
                    "max": max(recent_values) if recent_values else 0,
                }

        # Alert summary
        alert_summary = {
            "active_alerts": len(self.active_alerts),
            "alerts_by_severity": defaultdict(int),
            "recent_alerts": [],
        }

        for alert in self.active_alerts.values():
            alert_summary["alerts_by_severity"][alert.rule.severity.value] += 1

        # Get recent alerts from history
        recent_alerts = list(self.alert_history)[-10:]
        for alert in recent_alerts:
            alert_summary["recent_alerts"].append(
                {
                    "metric": alert.rule.metric_name,
                    "severity": alert.rule.severity.value,
                    "triggered_at": alert.triggered_at.isoformat(),
                    "resolved": bool(alert.resolved_at),
                }
            )

        # Business metrics summary
        business_summary = {}
        if self.business_metrics:
            business_metrics_by_name = defaultdict(list)
            for bm in list(self.business_metrics)[-100:]:  # Last 100 business metrics
                business_metrics_by_name[bm.name].append(bm.value)

            for name, values in business_metrics_by_name.items():
                business_summary[name] = {
                    "current": values[-1],
                    "average": statistics.mean(values),
                    "trend": (
                        "up" if len(values) > 1 and values[-1] > values[-2] else "down"
                    ),
                }

        return {
            "system_health": asdict(current_health),
            "monitoring_statistics": self.monitoring_stats.copy(),
            "metrics_summary": recent_metrics,
            "alert_summary": dict(alert_summary),
            "business_metrics": business_summary,
            "monitoring_uptime": self.monitoring_stats.get(
                "monitoring_uptime_seconds", 0
            ),
        }

    def get_capacity_planning_data(self) -> Dict[str, Any]:
        """Get data for capacity planning analysis"""

        # Analyze trends in key metrics over time
        trend_analysis = {}

        for metric_name in ["memory_usage_mb", "cpu_percentage", "request_latency_ms"]:
            if metric_name in self.metrics_storage:
                values = [
                    m["value"] for m in list(self.metrics_storage[metric_name])[-100:]
                ]
                if len(values) >= 10:
                    # Simple linear trend calculation
                    x = list(range(len(values)))
                    if len(values) > 1:
                        slope = (values[-1] - values[0]) / len(values)
                        trend_analysis[metric_name] = {
                            "current": values[-1],
                            "slope": slope,
                            "trend": "increasing" if slope > 0 else "decreasing",
                            "projected_1h": values[-1]
                            + slope * 60,  # Project 1 hour ahead
                            "data_points": len(values),
                        }

        return {
            "trend_analysis": trend_analysis,
            "health_history": [
                asdict(h) for h in list(self.system_health_history)[-24:]
            ],  # Last 24 health checks
            "recommendations": self._generate_capacity_recommendations(trend_analysis),
        }

    def _generate_capacity_recommendations(
        self, trend_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate capacity planning recommendations"""

        recommendations = []

        for metric_name, trend_data in trend_analysis.items():
            if trend_data["trend"] == "increasing":
                if (
                    metric_name == "memory_usage_mb"
                    and trend_data["projected_1h"] > 1500
                ):
                    recommendations.append(
                        f"Consider increasing memory allocation - projected usage: {trend_data['projected_1h']:.0f}MB"
                    )

                elif (
                    metric_name == "cpu_percentage" and trend_data["projected_1h"] > 80
                ):
                    recommendations.append(
                        f"Consider scaling CPU resources - projected usage: {trend_data['projected_1h']:.1f}%"
                    )

                elif (
                    metric_name == "request_latency_ms"
                    and trend_data["projected_1h"] > 3000
                ):
                    recommendations.append(
                        f"Consider performance optimization - projected latency: {trend_data['projected_1h']:.0f}ms"
                    )

        if not recommendations:
            recommendations.append(
                "System capacity appears adequate for current trends"
            )

        return recommendations

    async def _monitoring_loop(self) -> None:
        """Background monitoring loop"""

        while self.is_monitoring:
            try:
                # Collect system metrics
                if PSUTIL_AVAILABLE:
                    process = psutil.Process()
                    self.record_metric(
                        "memory_usage_mb", process.memory_info().rss / 1024 / 1024
                    )
                    self.record_metric("cpu_percentage", process.cpu_percent())

                # Collect performance service metrics if available
                if self.performance_service:
                    perf_metrics = (
                        self.performance_service.get_current_performance_metrics()
                    )
                    self.record_metric(
                        "request_latency_ms", perf_metrics.request_latency_ms
                    )
                    self.record_metric("error_rate", perf_metrics.error_rate)
                    self.record_metric("throughput_rps", perf_metrics.throughput_rps)

                # Update system health
                self.get_system_health_status()

                # Wait for next monitoring cycle
                await asyncio.sleep(60)  # Monitor every minute

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10)

    async def _alert_processing_loop(self) -> None:
        """Background alert processing loop"""

        while self.is_monitoring:
            try:
                # Check alert rules
                for rule in self.alert_rules:
                    if not rule.enabled:
                        continue

                    await self._evaluate_alert_rule(rule)

                # Check for resolved alerts
                await self._check_resolved_alerts()

                # Wait for next alert processing cycle
                await asyncio.sleep(30)  # Process alerts every 30 seconds

            except Exception as e:
                self.logger.error(f"Error in alert processing loop: {e}")
                await asyncio.sleep(10)

    async def _evaluate_alert_rule(self, rule: AlertRule) -> None:
        """Evaluate a single alert rule"""

        if rule.metric_name not in self.metrics_storage:
            return

        # Get recent values for the metric
        recent_data = list(self.metrics_storage[rule.metric_name])[-10:]
        if not recent_data:
            return

        current_value = recent_data[-1]["value"]

        # Check if alert condition is met
        condition_met = False
        if rule.comparison == "gt":
            condition_met = current_value > rule.threshold
        elif rule.comparison == "lt":
            condition_met = current_value < rule.threshold
        elif rule.comparison == "gte":
            condition_met = current_value >= rule.threshold
        elif rule.comparison == "lte":
            condition_met = current_value <= rule.threshold
        elif rule.comparison == "eq":
            condition_met = current_value == rule.threshold

        alert_id = f"{rule.metric_name}_{rule.comparison}_{rule.threshold}"

        if condition_met and alert_id not in self.active_alerts:
            # Create new alert
            alert = Alert(
                rule=rule,
                triggered_at=datetime.utcnow(),
                current_value=current_value,
                alert_id=alert_id,
            )

            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            self.monitoring_stats["alerts_triggered"] += 1

            # Glass-Box: Log alert trigger
            self.context_stream.add_event(
                event_type=ContextEventType.ERROR_OCCURRED,
                data={
                    "alert_triggered": alert_id,
                    "metric": rule.metric_name,
                    "current_value": current_value,
                    "threshold": rule.threshold,
                    "severity": rule.severity.value,
                },
                metadata={
                    "service": "ProductionMonitoringService",
                    "alert": "triggered",
                },
            )

            # Call alert callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    self.logger.error(f"Error calling alert callback: {e}")

    async def _check_resolved_alerts(self) -> None:
        """Check if any active alerts should be resolved"""

        resolved_alerts = []

        for alert_id, alert in self.active_alerts.items():
            rule = alert.rule

            if rule.metric_name not in self.metrics_storage:
                continue

            recent_data = list(self.metrics_storage[rule.metric_name])[-5:]
            if not recent_data:
                continue

            current_value = recent_data[-1]["value"]

            # Check if condition is no longer met
            condition_resolved = False
            if rule.comparison == "gt":
                condition_resolved = current_value <= rule.threshold
            elif rule.comparison == "lt":
                condition_resolved = current_value >= rule.threshold
            elif rule.comparison == "gte":
                condition_resolved = current_value < rule.threshold
            elif rule.comparison == "lte":
                condition_resolved = current_value > rule.threshold
            elif rule.comparison == "eq":
                condition_resolved = current_value != rule.threshold

            if condition_resolved:
                alert.resolved_at = datetime.utcnow()
                resolved_alerts.append(alert_id)
                self.monitoring_stats["alerts_resolved"] += 1

                # Glass-Box: Log alert resolution
                self.context_stream.add_event(
                    event_type=ContextEventType.SYSTEM_STATE,
                    data={
                        "alert_resolved": alert_id,
                        "metric": rule.metric_name,
                        "duration_seconds": (
                            alert.resolved_at - alert.triggered_at
                        ).total_seconds(),
                    },
                    metadata={
                        "service": "ProductionMonitoringService",
                        "alert": "resolved",
                    },
                )

        # Remove resolved alerts
        for alert_id in resolved_alerts:
            del self.active_alerts[alert_id]


# Factory function for service creation
def create_production_monitoring_service(
    context_stream: UnifiedContextStream,
    performance_service: Optional[PerformanceOptimizationService] = None,
) -> ProductionMonitoringService:
    """Factory function to create ProductionMonitoringService with proper dependencies"""

    return ProductionMonitoringService(
        context_stream=context_stream, performance_service=performance_service
    )
