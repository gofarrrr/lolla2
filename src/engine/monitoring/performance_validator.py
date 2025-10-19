#!/usr/bin/env python3
"""
METIS Performance Validation and Monitoring System
E003: Performance validation with real-time monitoring and alerting

Implements comprehensive performance validation against research targets
and provides real-time monitoring for production deployments.
"""

import logging
import statistics
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from uuid import UUID, uuid4
from collections import defaultdict, deque
import numpy as np

try:
    from src.engine.models.data_contracts import (
        MetisDataContract,
        MentalModelDefinition,
        ConfidenceLevel,
        EngagementContext,
    )
    from src.core.enhanced_event_bus import (
        EnhancedKafkaEventBus as MetisEventBus,
        CloudEvent,
    )
    from src.engine.core.state_management import DistributedStateManager, StateType

    DATA_CONTRACTS_AVAILABLE = True
except ImportError:
    DATA_CONTRACTS_AVAILABLE = False
    print("Warning: Core components not available, using mock interfaces")

# NEW: Import ideaflow metrics for enhanced creative problem tracking
try:
    from src.engine.monitoring.ideaflow_metrics import (
        get_ideaflow_tracker,
        IdeaflowMetrics,
    )

    IDEAFLOW_METRICS_AVAILABLE = True
except ImportError:
    IDEAFLOW_METRICS_AVAILABLE = False

    # Mock implementations
    class MockMetisDataContract:
        def __init__(self):
            self.engagement_context = MockEngagementContext()

    class MockEngagementContext:
        def __init__(self):
            self.engagement_id = uuid4()
            self.problem_statement = "Test problem"

    class MockStateManager:
        def __init__(self):
            self.data = {}

        async def set_state(self, key, value, state_type=None):
            self.data[key] = value

        async def get_state(self, key):
            return self.data.get(key)

    MetisDataContract = MockMetisDataContract
    EngagementContext = MockEngagementContext
    DistributedStateManager = MockStateManager
    StateType = None


class PerformanceMetricType(str, Enum):
    """Types of performance metrics"""

    RESPONSE_TIME = "response_time"
    COGNITIVE_ACCURACY = "cognitive_accuracy"
    CONSULTING_QUALITY = "consulting_quality"
    MECE_COMPLIANCE = "mece_compliance"
    SYSTEM_INTEGRATION = "system_integration"
    COST_EFFICIENCY = "cost_efficiency"
    USER_SATISFACTION = "user_satisfaction"
    THROUGHPUT = "throughput"
    MODEL_EFFECTIVENESS = "model_effectiveness"  # NEW: Individual model effectiveness


class AlertSeverity(str, Enum):
    """Alert severity levels"""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class ValidationStatus(str, Enum):
    """Performance validation status"""

    PASSING = "passing"
    WARNING = "warning"
    FAILING = "failing"
    UNKNOWN = "unknown"


@dataclass
class PerformanceTarget:
    """Performance target definition"""

    metric_type: PerformanceMetricType
    name: str
    target_value: float
    unit: str
    comparison: str = "≥"  # ≥, ≤, =, >, <
    warning_threshold: float = 0.9  # Fraction of target for warning
    critical_threshold: float = 0.8  # Fraction of target for critical
    description: str = ""
    research_basis: str = ""


@dataclass
class PerformanceMeasurement:
    """Individual performance measurement"""

    measurement_id: UUID = field(default_factory=uuid4)
    metric_type: PerformanceMetricType = PerformanceMetricType.RESPONSE_TIME
    value: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    context: Dict[str, Any] = field(default_factory=dict)
    engagement_id: Optional[UUID] = None
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    model_id: Optional[str] = None  # NEW: For model-specific effectiveness scoring


@dataclass
class PerformanceAlert:
    """Performance alert"""

    alert_id: UUID = field(default_factory=uuid4)
    metric_type: PerformanceMetricType = PerformanceMetricType.RESPONSE_TIME
    severity: AlertSeverity = AlertSeverity.WARNING
    message: str = ""
    current_value: float = 0.0
    target_value: float = 0.0
    threshold_breached: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class SystemHealthReport:
    """Comprehensive system health report"""

    report_id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    overall_status: ValidationStatus = ValidationStatus.UNKNOWN
    metrics_status: Dict[PerformanceMetricType, ValidationStatus] = field(
        default_factory=dict
    )
    performance_scores: Dict[PerformanceMetricType, float] = field(default_factory=dict)
    active_alerts: List[PerformanceAlert] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    uptime_percentage: float = 0.0
    last_24h_summary: Dict[str, Any] = field(default_factory=dict)


class PerformanceValidator:
    """
    Core performance validation engine
    Validates system performance against research-backed targets
    """

    def __init__(
        self,
        state_manager: DistributedStateManager,
        event_bus: Optional[MetisEventBus] = None,
    ):
        self.state_manager = state_manager
        self.event_bus = event_bus
        self.logger = logging.getLogger(__name__)

        # Performance targets (research-validated from PRD v7)
        self.performance_targets = self._initialize_performance_targets()

        # Measurement storage
        self.measurements: Dict[PerformanceMetricType, deque] = defaultdict(
            lambda: deque(maxlen=10000)
        )

        # Alert management
        self.active_alerts: Dict[UUID, PerformanceAlert] = {}
        self.alert_history: List[PerformanceAlert] = []

        # System state
        self.system_start_time = datetime.utcnow()
        self.last_health_check = None
        self.validation_enabled = True

        # Performance analytics
        self.analytics_cache = {}
        self.cache_ttl = timedelta(minutes=5)

        # NEW: Ideaflow metrics tracking
        if IDEAFLOW_METRICS_AVAILABLE:
            self.ideaflow_tracker = get_ideaflow_tracker()
            self.ideaflow_enabled = True
            self.logger.info("✅ Ideaflow metrics tracking enabled")
        else:
            self.ideaflow_tracker = None
            self.ideaflow_enabled = False
            self.logger.warning("⚠️ Ideaflow metrics tracking not available")

    def _initialize_performance_targets(
        self,
    ) -> Dict[PerformanceMetricType, PerformanceTarget]:
        """Initialize research-validated performance targets"""

        return {
            PerformanceMetricType.RESPONSE_TIME: PerformanceTarget(
                metric_type=PerformanceMetricType.RESPONSE_TIME,
                name="Response Time to First Insight",
                target_value=2.0,
                unit="seconds",
                comparison="≤",
                warning_threshold=1.5,  # 1.5x target for warning
                critical_threshold=2.0,  # 2x target for critical
                description="Time from request to first meaningful insight",
                research_basis="PRD v7.0 performance requirements",
            ),
            PerformanceMetricType.COGNITIVE_ACCURACY: PerformanceTarget(
                metric_type=PerformanceMetricType.COGNITIVE_ACCURACY,
                name="Cognitive Model Selection Accuracy",
                target_value=0.80,
                unit="percentage",
                comparison="≥",
                warning_threshold=0.9,
                critical_threshold=0.8,
                description="Accuracy of mental model selection for problem type",
                research_basis="6% improvement over baseline (paper 2402.18252v1)",
            ),
            PerformanceMetricType.CONSULTING_QUALITY: PerformanceTarget(
                metric_type=PerformanceMetricType.CONSULTING_QUALITY,
                name="Partner-Ready Deliverable Quality",
                target_value=0.25,
                unit="percentage",
                comparison="≥",
                warning_threshold=0.9,
                critical_threshold=0.8,
                description="Percentage of deliverables meeting partner-ready standard",
                research_basis="McKinsey-grade consulting quality benchmark",
            ),
            PerformanceMetricType.MECE_COMPLIANCE: PerformanceTarget(
                metric_type=PerformanceMetricType.MECE_COMPLIANCE,
                name="MECE Structural Validation",
                target_value=0.92,
                unit="percentage",
                comparison="≥",
                warning_threshold=0.9,
                critical_threshold=0.8,
                description="Compliance with MECE principle across problem types",
                research_basis="PRD v7.0 consulting methodology requirements",
            ),
            PerformanceMetricType.SYSTEM_INTEGRATION: PerformanceTarget(
                metric_type=PerformanceMetricType.SYSTEM_INTEGRATION,
                name="Component Coordination Success",
                target_value=0.99,
                unit="percentage",
                comparison="≥",
                warning_threshold=0.95,
                critical_threshold=0.90,
                description="Success rate of system component coordination",
                research_basis="Enterprise reliability requirements",
            ),
            PerformanceMetricType.COST_EFFICIENCY: PerformanceTarget(
                metric_type=PerformanceMetricType.COST_EFFICIENCY,
                name="Cost Per Analysis",
                target_value=5.0,
                unit="USD",
                comparison="≤",
                warning_threshold=2.0,  # 2x target
                critical_threshold=5.0,  # 5x target
                description="Average cost per strategic analysis",
                research_basis="$1-5 target vs $50K-500K traditional consulting",
            ),
            PerformanceMetricType.THROUGHPUT: PerformanceTarget(
                metric_type=PerformanceMetricType.THROUGHPUT,
                name="Concurrent Analysis Capacity",
                target_value=50.0,
                unit="concurrent users",
                comparison="≥",
                warning_threshold=0.8,
                critical_threshold=0.6,
                description="Number of concurrent analyses supported",
                research_basis="Enterprise scalability requirements",
            ),
            PerformanceMetricType.MODEL_EFFECTIVENESS: PerformanceTarget(
                metric_type=PerformanceMetricType.MODEL_EFFECTIVENESS,
                name="Mental Model Effectiveness Score",
                target_value=0.80,
                unit="effectiveness ratio",
                comparison="≥",
                warning_threshold=0.9,
                critical_threshold=0.8,
                description="Individual mental model effectiveness in problem-solving contexts",
                research_basis="6% improvement baseline + effectiveness tracking",
            ),
        }

    async def record_measurement(
        self,
        metric_type: PerformanceMetricType,
        value: float,
        context: Dict[str, Any] = None,
        engagement_id: Optional[UUID] = None,
        model_id: Optional[str] = None,
    ) -> PerformanceMeasurement:
        """Record a performance measurement"""

        measurement = PerformanceMeasurement(
            metric_type=metric_type,
            value=value,
            context=context or {},
            engagement_id=engagement_id,
            model_id=model_id,
            timestamp=datetime.utcnow(),
        )

        # Store measurement
        self.measurements[metric_type].append(measurement)

        # Store in distributed state
        await self.state_manager.set_state(
            f"measurement_{metric_type.value}_{measurement.measurement_id}",
            {
                "metric_type": metric_type.value,
                "value": value,
                "timestamp": measurement.timestamp.isoformat(),
                "context": context or {},
                "engagement_id": str(engagement_id) if engagement_id else None,
                "model_id": model_id,
            },
            StateType.PERFORMANCE,
        )

        # Check for threshold violations
        await self._check_performance_thresholds(metric_type, value, measurement)

        # Emit measurement event
        if self.event_bus:
            await self.event_bus.publish_event(
                CloudEvent(
                    type="performance.measurement.recorded",
                    source="monitoring/validator",
                    data={
                        "metric_type": metric_type.value,
                        "value": value,
                        "measurement_id": str(measurement.measurement_id),
                        "engagement_id": str(engagement_id) if engagement_id else None,
                        "model_id": model_id,
                    },
                )
            )

        self.logger.debug(f"Recorded {metric_type.value}: {value}")
        return measurement

    async def track_ideaflow_performance(
        self,
        session_id: str,
        problem_type: str,
        ideas_generated: List[str],
        generation_duration_seconds: float,
        engagement_id: Optional[UUID] = None,
    ) -> Optional[IdeaflowMetrics]:
        """
        Track ideaflow performance metrics for creative problem-solving sessions.

        Args:
            session_id: Unique identifier for the ideation session
            problem_type: Type of problem (creative_ideation, etc.)
            ideas_generated: List of ideas/solutions generated
            generation_duration_seconds: Time spent generating ideas
            engagement_id: Optional engagement context

        Returns:
            IdeaflowMetrics if tracking successful, None otherwise
        """
        if not self.ideaflow_enabled or not self.ideaflow_tracker:
            return None

        try:
            # Calculate ideaflow metrics
            metrics = self.ideaflow_tracker.calculate_session_metrics(
                session_id=session_id,
                problem_type=problem_type,
                ideas=ideas_generated,
                duration_seconds=generation_duration_seconds,
            )

            # Record key metrics in main performance system
            await self.record_measurement(
                metric_type=PerformanceMetricType.COGNITIVE_ACCURACY,
                value=metrics.diversity_score,
                context={
                    "metric_subtype": "ideaflow_diversity",
                    "session_id": session_id,
                    "problem_type": problem_type,
                    "total_ideas": metrics.total_ideas,
                },
                engagement_id=engagement_id,
            )

            # Track ideation velocity
            await self.record_measurement(
                metric_type=PerformanceMetricType.RESPONSE_TIME,
                value=metrics.ideas_per_minute,
                context={
                    "metric_subtype": "ideation_velocity",
                    "session_id": session_id,
                    "quality_score": metrics.quality_score,
                    "novelty_score": metrics.novelty_score,
                },
                engagement_id=engagement_id,
            )

            self.logger.info(
                f"Ideaflow metrics tracked: {metrics.total_ideas} ideas, "
                f"{metrics.ideas_per_minute:.1f}/min, diversity: {metrics.diversity_score:.2f}"
            )

            return metrics

        except Exception as e:
            self.logger.error(f"Failed to track ideaflow performance: {e}")
            return None

    async def _check_performance_thresholds(
        self,
        metric_type: PerformanceMetricType,
        value: float,
        measurement: PerformanceMeasurement,
    ):
        """Check if measurement violates performance thresholds"""

        target = self.performance_targets.get(metric_type)
        if not target:
            return

        # Determine if threshold is breached
        threshold_breached = None
        severity = None

        if target.comparison == "≥":
            if value < target.target_value * target.critical_threshold:
                threshold_breached = target.target_value * target.critical_threshold
                severity = AlertSeverity.CRITICAL
            elif value < target.target_value * target.warning_threshold:
                threshold_breached = target.target_value * target.warning_threshold
                severity = AlertSeverity.WARNING

        elif target.comparison == "≤":
            if value > target.target_value * target.critical_threshold:
                threshold_breached = target.target_value * target.critical_threshold
                severity = AlertSeverity.CRITICAL
            elif value > target.target_value * target.warning_threshold:
                threshold_breached = target.target_value * target.warning_threshold
                severity = AlertSeverity.WARNING

        # Create alert if threshold breached
        if threshold_breached is not None:
            await self._create_performance_alert(
                metric_type=metric_type,
                severity=severity,
                current_value=value,
                threshold_breached=threshold_breached,
                target_value=target.target_value,
                measurement=measurement,
            )

    async def _create_performance_alert(
        self,
        metric_type: PerformanceMetricType,
        severity: AlertSeverity,
        current_value: float,
        threshold_breached: float,
        target_value: float,
        measurement: PerformanceMeasurement,
    ):
        """Create and manage performance alert"""

        target = self.performance_targets[metric_type]

        alert = PerformanceAlert(
            metric_type=metric_type,
            severity=severity,
            message=f"{target.name} {severity.value}: {current_value:.2f} {target.unit} "
            f"(target: {target_value:.2f} {target.unit})",
            current_value=current_value,
            target_value=target_value,
            threshold_breached=threshold_breached,
        )

        # Store alert
        self.active_alerts[alert.alert_id] = alert
        self.alert_history.append(alert)

        # Store in distributed state
        await self.state_manager.set_state(
            f"alert_{alert.alert_id}",
            {
                "alert_id": str(alert.alert_id),
                "metric_type": metric_type.value,
                "severity": severity.value,
                "message": alert.message,
                "current_value": current_value,
                "target_value": target_value,
                "timestamp": alert.timestamp.isoformat(),
            },
            StateType.ALERTS,
        )

        # Emit alert event
        if self.event_bus:
            await self.event_bus.publish_event(
                CloudEvent(
                    type=f"performance.alert.{severity.value}",
                    source="monitoring/validator",
                    data={
                        "alert_id": str(alert.alert_id),
                        "metric_type": metric_type.value,
                        "message": alert.message,
                        "current_value": current_value,
                        "target_value": target_value,
                    },
                )
            )

        self.logger.warning(f"Performance alert: {alert.message}")

    async def validate_system_performance(self) -> SystemHealthReport:
        """Comprehensive system performance validation"""

        self.logger.info("Validating system performance against targets")

        # Calculate current performance for each metric
        metrics_status = {}
        performance_scores = {}
        recommendations = []

        for metric_type, target in self.performance_targets.items():
            status, score = await self._validate_metric_performance(metric_type)
            metrics_status[metric_type] = status
            performance_scores[metric_type] = score

            # Generate recommendations for failing metrics
            if status == ValidationStatus.FAILING:
                recommendations.extend(
                    await self._generate_performance_recommendations(metric_type, score)
                )

        # Determine overall system status
        overall_status = self._calculate_overall_status(metrics_status)

        # Calculate uptime
        uptime_percentage = await self._calculate_uptime_percentage()

        # Generate 24h summary
        last_24h_summary = await self._generate_24h_summary()

        # Get active alerts
        active_alerts = list(self.active_alerts.values())

        # Create health report
        health_report = SystemHealthReport(
            overall_status=overall_status,
            metrics_status=metrics_status,
            performance_scores=performance_scores,
            active_alerts=active_alerts,
            recommendations=recommendations,
            uptime_percentage=uptime_percentage,
            last_24h_summary=last_24h_summary,
        )

        # Store health report
        await self.state_manager.set_state(
            f"health_report_{health_report.timestamp.isoformat()}",
            {
                "report_id": str(health_report.report_id),
                "overall_status": overall_status.value,
                "metrics_count": len(metrics_status),
                "passing_metrics": sum(
                    1 for s in metrics_status.values() if s == ValidationStatus.PASSING
                ),
                "active_alerts": len(active_alerts),
                "uptime_percentage": uptime_percentage,
                "timestamp": health_report.timestamp.isoformat(),
            },
            StateType.HEALTH,
        )

        # Emit health report event
        if self.event_bus:
            await self.event_bus.publish_event(
                CloudEvent(
                    type="performance.health.report",
                    source="monitoring/validator",
                    data={
                        "report_id": str(health_report.report_id),
                        "overall_status": overall_status.value,
                        "uptime_percentage": uptime_percentage,
                        "active_alerts": len(active_alerts),
                        "recommendations_count": len(recommendations),
                    },
                )
            )

        self.last_health_check = datetime.utcnow()
        return health_report

    async def _validate_metric_performance(
        self, metric_type: PerformanceMetricType
    ) -> Tuple[ValidationStatus, float]:
        """Validate performance for specific metric"""

        measurements = list(self.measurements[metric_type])
        if not measurements:
            return ValidationStatus.UNKNOWN, 0.0

        target = self.performance_targets[metric_type]

        # Get recent measurements (last hour)
        recent_cutoff = datetime.utcnow() - timedelta(hours=1)
        recent_measurements = [m for m in measurements if m.timestamp >= recent_cutoff]

        if not recent_measurements:
            return ValidationStatus.UNKNOWN, 0.0

        # Calculate average performance
        values = [m.value for m in recent_measurements]
        avg_value = statistics.mean(values)

        # Calculate performance score (0-1)
        if target.comparison == "≥":
            score = min(1.0, avg_value / target.target_value)
        elif target.comparison == "≤":
            score = min(1.0, target.target_value / avg_value) if avg_value > 0 else 1.0
        else:
            score = 1.0 - abs(avg_value - target.target_value) / target.target_value

        # Determine status
        if score >= target.warning_threshold:
            status = ValidationStatus.PASSING
        elif score >= target.critical_threshold:
            status = ValidationStatus.WARNING
        else:
            status = ValidationStatus.FAILING

        return status, score

    def _calculate_overall_status(
        self, metrics_status: Dict[PerformanceMetricType, ValidationStatus]
    ) -> ValidationStatus:
        """Calculate overall system status from metric statuses"""

        if not metrics_status:
            return ValidationStatus.UNKNOWN

        statuses = list(metrics_status.values())

        # If any metric is failing, system is failing
        if ValidationStatus.FAILING in statuses:
            return ValidationStatus.FAILING

        # If any metric has warnings, system has warnings
        if ValidationStatus.WARNING in statuses:
            return ValidationStatus.WARNING

        # If all metrics are passing, system is passing
        if all(s == ValidationStatus.PASSING for s in statuses):
            return ValidationStatus.PASSING

        return ValidationStatus.UNKNOWN

    async def _calculate_uptime_percentage(self) -> float:
        """Calculate system uptime percentage"""

        # Simplified uptime calculation
        # In production, this would track actual outages
        total_time = (datetime.utcnow() - self.system_start_time).total_seconds()

        # Count critical alerts as downtime
        downtime = 0
        for alert in self.alert_history:
            if alert.severity == AlertSeverity.CRITICAL:
                if alert.resolved_at:
                    downtime += (alert.resolved_at - alert.timestamp).total_seconds()
                else:
                    downtime += (datetime.utcnow() - alert.timestamp).total_seconds()

        if total_time > 0:
            uptime_percentage = max(0.0, (total_time - downtime) / total_time * 100)
        else:
            uptime_percentage = 100.0

        return min(100.0, uptime_percentage)

    async def _generate_24h_summary(self) -> Dict[str, Any]:
        """Generate 24-hour performance summary"""

        cutoff = datetime.utcnow() - timedelta(hours=24)

        summary = {
            "period": "24h",
            "measurements_count": 0,
            "alerts_count": 0,
            "critical_alerts": 0,
            "average_response_time": 0.0,
            "peak_concurrent_users": 0,
            "total_analyses": 0,
        }

        # Count measurements and calculate averages
        total_measurements = 0
        response_times = []

        for metric_type, measurements in self.measurements.items():
            recent = [m for m in measurements if m.timestamp >= cutoff]
            total_measurements += len(recent)

            if metric_type == PerformanceMetricType.RESPONSE_TIME:
                response_times.extend([m.value for m in recent])
            elif metric_type == PerformanceMetricType.THROUGHPUT:
                if recent:
                    summary["peak_concurrent_users"] = max(m.value for m in recent)

        summary["measurements_count"] = total_measurements

        if response_times:
            summary["average_response_time"] = statistics.mean(response_times)

        # Count alerts
        recent_alerts = [a for a in self.alert_history if a.timestamp >= cutoff]
        summary["alerts_count"] = len(recent_alerts)
        summary["critical_alerts"] = len(
            [a for a in recent_alerts if a.severity == AlertSeverity.CRITICAL]
        )

        return summary

    async def _generate_performance_recommendations(
        self, metric_type: PerformanceMetricType, score: float
    ) -> List[str]:
        """Generate performance improvement recommendations"""

        recommendations = []
        target = self.performance_targets[metric_type]

        if metric_type == PerformanceMetricType.RESPONSE_TIME:
            recommendations.extend(
                [
                    "Optimize mental model selection algorithm",
                    "Implement result caching for common patterns",
                    "Consider distributed processing for complex analyses",
                ]
            )

        elif metric_type == PerformanceMetricType.COGNITIVE_ACCURACY:
            recommendations.extend(
                [
                    "Retrain model selection embeddings",
                    "Expand mental models library",
                    "Improve context understanding algorithms",
                ]
            )

        elif metric_type == PerformanceMetricType.CONSULTING_QUALITY:
            recommendations.extend(
                [
                    "Enhance MECE validation engine",
                    "Improve Pyramid Principle synthesis",
                    "Add more consulting framework templates",
                ]
            )

        elif metric_type == PerformanceMetricType.SYSTEM_INTEGRATION:
            recommendations.extend(
                [
                    "Review event bus reliability",
                    "Improve error handling and retries",
                    "Add circuit breaker patterns",
                ]
            )

        elif metric_type == PerformanceMetricType.COST_EFFICIENCY:
            recommendations.extend(
                [
                    "Optimize resource utilization",
                    "Implement usage-based scaling",
                    "Review infrastructure costs",
                ]
            )

        return recommendations

    async def get_performance_analytics(
        self,
        metric_types: List[PerformanceMetricType] = None,
        time_range: timedelta = timedelta(hours=24),
    ) -> Dict[str, Any]:
        """Get comprehensive performance analytics"""

        # Check cache
        metric_types_list = metric_types or []
        cache_key = (
            f"analytics_{hash(tuple(metric_types_list))}_{time_range.total_seconds()}"
        )
        if cache_key in self.analytics_cache:
            cached_time, cached_data = self.analytics_cache[cache_key]
            if datetime.utcnow() - cached_time < self.cache_ttl:
                return cached_data

        cutoff = datetime.utcnow() - time_range
        analytics = {
            "time_range_hours": time_range.total_seconds() / 3600,
            "metrics": {},
            "trends": {},
            "correlations": {},
            "summary": {},
        }

        # Analyze each metric type
        for metric_type in metric_types or self.performance_targets.keys():
            measurements = [
                m for m in self.measurements[metric_type] if m.timestamp >= cutoff
            ]

            if measurements:
                values = [m.value for m in measurements]

                analytics["metrics"][metric_type.value] = {
                    "count": len(values),
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
                    "min": min(values),
                    "max": max(values),
                    "target": self.performance_targets[metric_type].target_value,
                }

                # Calculate trends (simple linear regression)
                if len(values) >= 2:
                    x = list(range(len(values)))
                    slope = np.polyfit(x, values, 1)[0]
                    analytics["trends"][metric_type.value] = {
                        "direction": "improving" if slope < 0 else "degrading",
                        "slope": float(slope),
                    }

        # Cache results
        self.analytics_cache[cache_key] = (datetime.utcnow(), analytics)

        return analytics

    async def resolve_alert(self, alert_id: UUID, resolution_note: str = "") -> bool:
        """Resolve an active performance alert"""

        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.utcnow()

            # Remove from active alerts
            del self.active_alerts[alert_id]

            # Update in state
            await self.state_manager.set_state(
                f"alert_{alert_id}_resolved",
                {
                    "alert_id": str(alert_id),
                    "resolved_at": alert.resolved_at.isoformat(),
                    "resolution_note": resolution_note,
                },
                StateType.ALERTS,
            )

            # Emit resolution event
            if self.event_bus:
                await self.event_bus.publish_event(
                    CloudEvent(
                        type="performance.alert.resolved",
                        source="monitoring/validator",
                        data={
                            "alert_id": str(alert_id),
                            "resolution_note": resolution_note,
                        },
                    )
                )

            self.logger.info(f"Resolved alert {alert_id}: {resolution_note}")
            return True

        return False

    async def get_model_effectiveness_scores(
        self, time_range: timedelta = timedelta(days=7), min_samples: int = 3
    ) -> Dict[str, Dict[str, float]]:
        """Get effectiveness scores for individual mental models"""

        cutoff = datetime.utcnow() - time_range
        model_scores = {}

        # Get model effectiveness measurements
        effectiveness_measurements = [
            m
            for m in self.measurements[PerformanceMetricType.MODEL_EFFECTIVENESS]
            if m.timestamp >= cutoff and m.model_id
        ]

        # Group measurements by model
        model_measurements = defaultdict(list)
        for measurement in effectiveness_measurements:
            model_measurements[measurement.model_id].append(measurement)

        # Calculate effectiveness scores for each model
        for model_id, measurements in model_measurements.items():
            if len(measurements) < min_samples:
                continue

            values = [m.value for m in measurements]
            contexts = [m.context for m in measurements]

            # Calculate effectiveness metrics
            effectiveness_score = statistics.mean(values)
            effectiveness_std = statistics.stdev(values) if len(values) > 1 else 0.0
            effectiveness_trend = self._calculate_effectiveness_trend(measurements)

            # Calculate confidence based on sample size and consistency
            confidence = min(1.0, len(measurements) / 10.0) * (
                1.0 - min(0.5, effectiveness_std)
            )

            # Count problem types this model has been used for
            problem_types = set()
            for ctx in contexts:
                if "problem_type" in ctx:
                    problem_types.add(ctx["problem_type"])

            model_scores[model_id] = {
                "effectiveness_score": effectiveness_score,
                "effectiveness_std": effectiveness_std,
                "effectiveness_trend": effectiveness_trend,
                "confidence": confidence,
                "sample_count": len(measurements),
                "problem_types_count": len(problem_types),
                "last_used": max(m.timestamp for m in measurements).isoformat(),
            }

        return model_scores

    def _calculate_effectiveness_trend(
        self, measurements: List[PerformanceMeasurement]
    ) -> str:
        """Calculate effectiveness trend for a model"""
        if len(measurements) < 2:
            return "insufficient_data"

        # Sort by timestamp
        sorted_measurements = sorted(measurements, key=lambda m: m.timestamp)
        values = [m.value for m in sorted_measurements]

        # Simple trend calculation
        first_half = values[: len(values) // 2]
        second_half = values[len(values) // 2 :]

        if len(first_half) > 0 and len(second_half) > 0:
            first_avg = statistics.mean(first_half)
            second_avg = statistics.mean(second_half)

            improvement = (second_avg - first_avg) / first_avg if first_avg > 0 else 0

            if improvement > 0.05:  # 5% improvement threshold
                return "improving"
            elif improvement < -0.05:  # 5% degradation threshold
                return "degrading"
            else:
                return "stable"

        return "insufficient_data"

    async def get_top_performing_models(
        self, limit: int = 10, time_range: timedelta = timedelta(days=30)
    ) -> List[Dict[str, Any]]:
        """Get top performing mental models by effectiveness"""

        model_scores = await self.get_model_effectiveness_scores(time_range)

        # Filter models with sufficient confidence and samples
        qualified_models = [
            {"model_id": model_id, **scores}
            for model_id, scores in model_scores.items()
            if scores["confidence"] >= 0.5 and scores["sample_count"] >= 5
        ]

        # Sort by effectiveness score
        qualified_models.sort(key=lambda x: x["effectiveness_score"], reverse=True)

        return qualified_models[:limit]

    async def get_underperforming_models(
        self, threshold: float = 0.7, time_range: timedelta = timedelta(days=30)
    ) -> List[Dict[str, Any]]:
        """Get underperforming mental models that need attention"""

        model_scores = await self.get_model_effectiveness_scores(time_range)

        underperforming = []
        for model_id, scores in model_scores.items():
            if (
                scores["effectiveness_score"] < threshold
                and scores["confidence"] >= 0.4
                and scores["sample_count"] >= 3
            ):

                underperforming.append(
                    {
                        "model_id": model_id,
                        "effectiveness_gap": threshold - scores["effectiveness_score"],
                        "recommendation": self._generate_model_improvement_recommendation(
                            scores
                        ),
                        **scores,
                    }
                )

        # Sort by effectiveness gap (worst first)
        underperforming.sort(key=lambda x: x["effectiveness_gap"], reverse=True)

        return underperforming

    def _generate_model_improvement_recommendation(
        self, scores: Dict[str, float]
    ) -> str:
        """Generate improvement recommendations for underperforming models"""

        recommendations = []

        if scores["sample_count"] < 10:
            recommendations.append(
                "Increase usage frequency to gather more performance data"
            )

        if scores["effectiveness_std"] > 0.2:
            recommendations.append(
                "High variance detected - review model application contexts"
            )

        if scores["problem_types_count"] < 3:
            recommendations.append("Expand model to additional problem types")

        if scores["effectiveness_trend"] == "degrading":
            recommendations.append(
                "Model performance is declining - review and retrain"
            )

        if not recommendations:
            recommendations.append("Review model definition and application criteria")

        return "; ".join(recommendations)

    async def record_model_effectiveness(
        self,
        model_id: str,
        effectiveness_score: float,
        engagement_id: UUID = None,
        context: Dict[str, Any] = None,
    ) -> PerformanceMeasurement:
        """Record effectiveness measurement for a specific mental model"""

        return await self.record_measurement(
            PerformanceMetricType.MODEL_EFFECTIVENESS,
            effectiveness_score,
            context=context,
            engagement_id=engagement_id,
            model_id=model_id,
        )

    def get_system_status_summary(self) -> Dict[str, Any]:
        """Get quick system status summary"""

        return {
            "validation_enabled": self.validation_enabled,
            "uptime_seconds": (
                datetime.utcnow() - self.system_start_time
            ).total_seconds(),
            "active_alerts": len(self.active_alerts),
            "critical_alerts": len(
                [
                    a
                    for a in self.active_alerts.values()
                    if a.severity == AlertSeverity.CRITICAL
                ]
            ),
            "last_health_check": (
                self.last_health_check.isoformat() if self.last_health_check else None
            ),
            "total_measurements": sum(
                len(measurements) for measurements in self.measurements.values()
            ),
            "metrics_tracked": len(self.performance_targets),
        }


# Global performance validator instance
_global_validator: Optional[PerformanceValidator] = None


async def get_performance_validator() -> PerformanceValidator:
    """Get or create global performance validator instance"""
    global _global_validator

    if _global_validator is None:
        # Create mock state manager for development
        state_manager = (
            DistributedStateManager()
            if DATA_CONTRACTS_AVAILABLE
            else MockStateManager()
        )
        _global_validator = PerformanceValidator(state_manager)

    return _global_validator


# Convenience functions for performance monitoring
async def record_response_time(
    response_time_seconds: float, engagement_id: UUID = None
):
    """Record response time measurement"""
    validator = await get_performance_validator()
    await validator.record_measurement(
        PerformanceMetricType.RESPONSE_TIME,
        response_time_seconds,
        engagement_id=engagement_id,
    )


async def record_cognitive_accuracy(accuracy_score: float, engagement_id: UUID = None):
    """Record cognitive accuracy measurement"""
    validator = await get_performance_validator()
    await validator.record_measurement(
        PerformanceMetricType.COGNITIVE_ACCURACY,
        accuracy_score,
        engagement_id=engagement_id,
    )


async def record_consulting_quality(quality_score: float, engagement_id: UUID = None):
    """Record consulting quality measurement"""
    validator = await get_performance_validator()
    await validator.record_measurement(
        PerformanceMetricType.CONSULTING_QUALITY,
        quality_score,
        engagement_id=engagement_id,
    )


async def validate_system_health() -> SystemHealthReport:
    """Validate system health and get comprehensive report"""
    validator = await get_performance_validator()
    return await validator.validate_system_performance()


async def get_system_analytics(time_range_hours: int = 24) -> Dict[str, Any]:
    """Get system performance analytics"""
    validator = await get_performance_validator()
    return await validator.get_performance_analytics(
        time_range=timedelta(hours=time_range_hours)
    )


# Model effectiveness convenience functions
async def record_model_effectiveness(
    model_id: str,
    effectiveness_score: float,
    engagement_id: UUID = None,
    context: Dict[str, Any] = None,
):
    """Record effectiveness measurement for a mental model"""
    validator = await get_performance_validator()
    await validator.record_model_effectiveness(
        model_id=model_id,
        effectiveness_score=effectiveness_score,
        engagement_id=engagement_id,
        context=context,
    )


async def get_model_effectiveness_report(time_range_days: int = 7) -> Dict[str, Any]:
    """Get comprehensive model effectiveness report"""
    validator = await get_performance_validator()

    model_scores = await validator.get_model_effectiveness_scores(
        time_range=timedelta(days=time_range_days)
    )

    top_performers = await validator.get_top_performing_models(
        time_range=timedelta(days=time_range_days)
    )

    underperformers = await validator.get_underperforming_models(
        time_range=timedelta(days=time_range_days)
    )

    return {
        "summary": {
            "total_models_tracked": len(model_scores),
            "top_performers_count": len(top_performers),
            "underperformers_count": len(underperformers),
            "average_effectiveness": (
                statistics.mean(
                    [scores["effectiveness_score"] for scores in model_scores.values()]
                )
                if model_scores
                else 0.0
            ),
        },
        "model_scores": model_scores,
        "top_performers": top_performers,
        "underperformers": underperformers,
    }
