#!/usr/bin/env python3
"""
Performance Metrics Dashboard for METIS Cognitive Platform
Provides comprehensive performance monitoring and analytics for all cognitive systems
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict
import statistics


class MetricType(str, Enum):
    """Types of performance metrics"""

    ACCURACY = "accuracy"
    CONFIDENCE = "confidence"
    RESPONSE_TIME = "response_time"
    SUCCESS_RATE = "success_rate"
    PATTERN_MATCH = "pattern_match"
    MODEL_EFFECTIVENESS = "model_effectiveness"
    BAYESIAN_LEARNING = "bayesian_learning"
    CALIBRATION_QUALITY = "calibration_quality"
    SIMILARITY_DETECTION = "similarity_detection"


class ComponentType(str, Enum):
    """METIS system components being monitored"""

    COGNITIVE_ENGINE = "cognitive_engine"
    CONFIDENCE_CALIBRATION = "confidence_calibration"
    VECTOR_SIMILARITY = "vector_similarity"
    BAYESIAN_UPDATER = "bayesian_updater"
    VALUE_ASSESSMENT = "value_assessment"
    AI_AUGMENTATION = "ai_augmentation"
    LLM_VALIDATION = "llm_validation"
    MUNGER_OVERLAY = "munger_overlay"
    RESEARCH_ORCHESTRATOR = "research_orchestrator"


@dataclass
class PerformanceMetric:
    """Individual performance metric measurement"""

    component: ComponentType
    metric_type: MetricType
    value: float
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)

    # Statistical properties
    min_acceptable: float = 0.0
    max_acceptable: float = 1.0
    target_value: Optional[float] = None

    def is_within_bounds(self) -> bool:
        """Check if metric is within acceptable bounds"""
        return self.min_acceptable <= self.value <= self.max_acceptable

    def performance_score(self) -> float:
        """Calculate normalized performance score (0.0-1.0)"""
        if self.target_value is not None:
            # Distance from target (closer = better)
            distance = abs(self.value - self.target_value)
            max_distance = max(
                abs(self.max_acceptable - self.target_value),
                abs(self.min_acceptable - self.target_value),
            )
            return (
                max(0.0, 1.0 - (distance / max_distance)) if max_distance > 0 else 1.0
            )
        else:
            # Linear scaling within bounds
            range_size = self.max_acceptable - self.min_acceptable
            return (
                (self.value - self.min_acceptable) / range_size
                if range_size > 0
                else 1.0
            )


@dataclass
class ComponentHealth:
    """Health status of a system component"""

    component: ComponentType
    overall_score: float
    status: str  # "excellent", "good", "acceptable", "degraded", "critical"
    metrics_count: int
    last_updated: datetime
    alerts: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class SystemPerformanceSummary:
    """Overall system performance summary"""

    overall_score: float
    component_scores: Dict[ComponentType, float]
    critical_alerts: List[str]
    performance_trends: Dict[str, str]  # "improving", "stable", "degrading"
    system_health: str  # "excellent", "good", "acceptable", "degraded", "critical"
    last_updated: datetime
    recommendations: List[str] = field(default_factory=list)


class PerformanceMetricsDashboard:
    """
    Comprehensive performance monitoring dashboard for METIS cognitive systems

    Tracks performance metrics across all cognitive components:
    - Confidence calibration accuracy
    - Vector similarity detection effectiveness
    - Bayesian learning convergence
    - Model selection performance
    - Response times and system health
    """

    def __init__(self, retention_hours: int = 24, storage_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.retention_hours = retention_hours
        self.storage_path = storage_path or "data/performance_metrics"

        # Metrics storage
        self.metrics: List[PerformanceMetric] = []
        self.component_stats: Dict[ComponentType, Dict[str, Any]] = defaultdict(dict)

        # Performance thresholds
        self.thresholds = {
            MetricType.ACCURACY: {"min": 0.7, "max": 1.0, "target": 0.85},
            MetricType.CONFIDENCE: {"min": 0.0, "max": 1.0, "target": 0.75},
            MetricType.RESPONSE_TIME: {
                "min": 0.0,
                "max": 30000.0,
                "target": 2000.0,
            },  # ms
            MetricType.SUCCESS_RATE: {"min": 0.7, "max": 1.0, "target": 0.9},
            MetricType.PATTERN_MATCH: {"min": 0.0, "max": 1.0, "target": 0.8},
            MetricType.MODEL_EFFECTIVENESS: {"min": 0.0, "max": 1.0, "target": 0.75},
            MetricType.BAYESIAN_LEARNING: {"min": 0.0, "max": 1.0, "target": 0.8},
            MetricType.CALIBRATION_QUALITY: {
                "min": 0.0,
                "max": 0.1,
                "target": 0.05,
            },  # ECE - lower is better
            MetricType.SIMILARITY_DETECTION: {"min": 0.0, "max": 1.0, "target": 0.7},
        }

        self.logger.info("✅ Performance Metrics Dashboard initialized")

    async def record_metric(
        self,
        component: ComponentType,
        metric_type: MetricType,
        value: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Record a new performance metric

        Args:
            component: System component being measured
            metric_type: Type of metric
            value: Metric value
            context: Additional context information

        Returns:
            Metric ID for tracking
        """
        thresholds = self.thresholds.get(metric_type, {"min": 0.0, "max": 1.0})

        metric = PerformanceMetric(
            component=component,
            metric_type=metric_type,
            value=value,
            timestamp=datetime.utcnow(),
            context=context or {},
            min_acceptable=thresholds["min"],
            max_acceptable=thresholds["max"],
            target_value=thresholds.get("target"),
        )

        self.metrics.append(metric)

        # Update component statistics
        self._update_component_stats(component, metric)

        # Clean up old metrics
        await self._cleanup_old_metrics()

        metric_id = (
            f"{component.value}_{metric_type.value}_{metric.timestamp.isoformat()}"
        )
        self.logger.debug(f"📊 Recorded metric: {metric_id} = {value:.3f}")

        return metric_id

    def _update_component_stats(
        self, component: ComponentType, metric: PerformanceMetric
    ):
        """Update rolling statistics for a component"""
        if component not in self.component_stats:
            self.component_stats[component] = {
                "total_metrics": 0,
                "avg_performance": 0.0,
                "last_update": datetime.utcnow(),
                "metric_types": defaultdict(list),
            }

        stats = self.component_stats[component]
        stats["total_metrics"] += 1
        stats["last_update"] = datetime.utcnow()
        stats["metric_types"][metric.metric_type].append(metric.value)

        # Keep only recent values for rolling average
        for metric_type, values in stats["metric_types"].items():
            if len(values) > 100:  # Keep last 100 measurements
                stats["metric_types"][metric_type] = values[-100:]

        # Calculate average performance score
        all_scores = []
        for values in stats["metric_types"].values():
            if values:
                all_scores.extend(values)

        if all_scores:
            stats["avg_performance"] = statistics.mean(all_scores)

    async def _cleanup_old_metrics(self):
        """Remove metrics older than retention period"""
        cutoff_time = datetime.utcnow() - timedelta(hours=self.retention_hours)

        old_count = len(self.metrics)
        self.metrics = [m for m in self.metrics if m.timestamp > cutoff_time]
        new_count = len(self.metrics)

        if old_count > new_count:
            self.logger.debug(f"🧹 Cleaned up {old_count - new_count} old metrics")

    def get_component_health(self, component: ComponentType) -> ComponentHealth:
        """Get health assessment for a specific component"""
        if component not in self.component_stats:
            return ComponentHealth(
                component=component,
                overall_score=0.0,
                status="unknown",
                metrics_count=0,
                last_updated=datetime.utcnow(),
                alerts=["No metrics available for this component"],
                recommendations=["Start collecting metrics for this component"],
            )

        stats = self.component_stats[component]
        overall_score = stats["avg_performance"]
        metrics_count = stats["total_metrics"]

        # Determine health status
        if overall_score >= 0.9:
            status = "excellent"
        elif overall_score >= 0.8:
            status = "good"
        elif overall_score >= 0.7:
            status = "acceptable"
        elif overall_score >= 0.5:
            status = "degraded"
        else:
            status = "critical"

        # Generate alerts and recommendations
        alerts = []
        recommendations = []

        if overall_score < 0.7:
            alerts.append(
                f"Component performance below acceptable threshold: {overall_score:.3f}"
            )
            recommendations.append(f"Investigate {component.value} performance issues")

        if metrics_count < 10:
            alerts.append(f"Limited metrics available: {metrics_count} measurements")
            recommendations.append("Increase metric collection frequency")

        # Check for stale metrics
        time_since_update = datetime.utcnow() - stats["last_update"]
        if time_since_update > timedelta(hours=1):
            alerts.append(f"Metrics stale: last update {time_since_update} ago")
            recommendations.append("Verify metric collection is active")

        return ComponentHealth(
            component=component,
            overall_score=overall_score,
            status=status,
            metrics_count=metrics_count,
            last_updated=stats["last_update"],
            alerts=alerts,
            recommendations=recommendations,
        )

    def get_system_performance_summary(self) -> SystemPerformanceSummary:
        """Get overall system performance summary"""
        component_scores = {}
        all_scores = []
        critical_alerts = []
        recommendations = []

        # Assess each component
        for component in ComponentType:
            health = self.get_component_health(component)
            component_scores[component] = health.overall_score
            all_scores.append(health.overall_score)

            # Collect critical alerts
            if health.status in ["critical", "degraded"]:
                critical_alerts.extend(health.alerts)
                recommendations.extend(health.recommendations)

        # Calculate overall system score
        if all_scores:
            overall_score = statistics.mean(all_scores)
        else:
            overall_score = 0.0

        # Determine system health
        if overall_score >= 0.9:
            system_health = "excellent"
        elif overall_score >= 0.8:
            system_health = "good"
        elif overall_score >= 0.7:
            system_health = "acceptable"
        elif overall_score >= 0.5:
            system_health = "degraded"
        else:
            system_health = "critical"

        # Analyze performance trends (simplified)
        performance_trends = self._analyze_trends()

        return SystemPerformanceSummary(
            overall_score=overall_score,
            component_scores=component_scores,
            critical_alerts=list(set(critical_alerts)),  # Remove duplicates
            performance_trends=performance_trends,
            system_health=system_health,
            last_updated=datetime.utcnow(),
            recommendations=list(set(recommendations)),  # Remove duplicates
        )

    def _analyze_trends(self) -> Dict[str, str]:
        """Analyze performance trends over time"""
        trends = {}

        # Simple trend analysis - compare recent vs older metrics
        cutoff_recent = datetime.utcnow() - timedelta(hours=1)
        cutoff_older = datetime.utcnow() - timedelta(hours=6)

        for component in ComponentType:
            recent_metrics = [
                m
                for m in self.metrics
                if m.component == component and m.timestamp > cutoff_recent
            ]
            older_metrics = [
                m
                for m in self.metrics
                if m.component == component
                and cutoff_older < m.timestamp <= cutoff_recent
            ]

            if len(recent_metrics) >= 5 and len(older_metrics) >= 5:
                recent_avg = statistics.mean(
                    [m.performance_score() for m in recent_metrics]
                )
                older_avg = statistics.mean(
                    [m.performance_score() for m in older_metrics]
                )

                if recent_avg > older_avg + 0.05:
                    trends[component.value] = "improving"
                elif recent_avg < older_avg - 0.05:
                    trends[component.value] = "degrading"
                else:
                    trends[component.value] = "stable"
            else:
                trends[component.value] = "insufficient_data"

        return trends

    def get_detailed_metrics(
        self,
        component: Optional[ComponentType] = None,
        metric_type: Optional[MetricType] = None,
        hours_back: int = 24,
    ) -> List[PerformanceMetric]:
        """Get detailed metrics with optional filtering"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)

        filtered_metrics = [m for m in self.metrics if m.timestamp > cutoff_time]

        if component:
            filtered_metrics = [m for m in filtered_metrics if m.component == component]

        if metric_type:
            filtered_metrics = [
                m for m in filtered_metrics if m.metric_type == metric_type
            ]

        return sorted(filtered_metrics, key=lambda x: x.timestamp, reverse=True)

    def export_dashboard_data(self) -> Dict[str, Any]:
        """Export complete dashboard data for external analysis"""
        summary = self.get_system_performance_summary()

        component_details = {}
        for component in ComponentType:
            health = self.get_component_health(component)
            component_details[component.value] = {
                "health": {
                    "overall_score": health.overall_score,
                    "status": health.status,
                    "metrics_count": health.metrics_count,
                    "last_updated": health.last_updated.isoformat(),
                    "alerts": health.alerts,
                    "recommendations": health.recommendations,
                },
                "recent_metrics": [
                    {
                        "metric_type": m.metric_type.value,
                        "value": m.value,
                        "timestamp": m.timestamp.isoformat(),
                        "performance_score": m.performance_score(),
                        "within_bounds": m.is_within_bounds(),
                    }
                    for m in self.get_detailed_metrics(component, hours_back=1)
                ],
            }

        return {
            "system_summary": {
                "overall_score": summary.overall_score,
                "system_health": summary.system_health,
                "component_scores": {
                    k.value: v for k, v in summary.component_scores.items()
                },
                "critical_alerts": summary.critical_alerts,
                "performance_trends": summary.performance_trends,
                "recommendations": summary.recommendations,
                "last_updated": summary.last_updated.isoformat(),
            },
            "component_details": component_details,
            "metadata": {
                "total_metrics": len(self.metrics),
                "retention_hours": self.retention_hours,
                "components_monitored": len(self.component_stats),
                "export_timestamp": datetime.utcnow().isoformat(),
            },
        }

    async def generate_performance_report(self) -> str:
        """Generate a human-readable performance report"""
        summary = self.get_system_performance_summary()

        report = []
        report.append("🎯 METIS COGNITIVE PLATFORM PERFORMANCE REPORT")
        report.append("=" * 60)
        report.append(
            f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )
        report.append("")

        # System overview
        report.append("📊 SYSTEM OVERVIEW")
        report.append(f"Overall Score: {summary.overall_score:.3f}")
        report.append(f"System Health: {summary.system_health.upper()}")
        report.append(f"Total Metrics: {len(self.metrics)}")
        report.append("")

        # Component scores
        report.append("🔧 COMPONENT PERFORMANCE")
        for component, score in summary.component_scores.items():
            health = self.get_component_health(component)
            status_icon = {
                "excellent": "🟢",
                "good": "🟡",
                "acceptable": "🟠",
                "degraded": "🔴",
                "critical": "🚨",
                "unknown": "⚪",
            }.get(health.status, "❓")

            report.append(
                f"  {status_icon} {component.value.replace('_', ' ').title()}: {score:.3f} ({health.status})"
            )
            if health.alerts:
                for alert in health.alerts[:2]:  # Show top 2 alerts
                    report.append(f"    ⚠️  {alert}")

        report.append("")

        # Performance trends
        report.append("📈 PERFORMANCE TRENDS")
        for component, trend in summary.performance_trends.items():
            trend_icon = {
                "improving": "📈",
                "stable": "➡️",
                "degrading": "📉",
                "insufficient_data": "❓",
            }
            icon = trend_icon.get(trend, "❓")
            report.append(
                f"  {icon} {component.replace('_', ' ').title()}: {trend.replace('_', ' ')}"
            )

        report.append("")

        # Critical alerts
        if summary.critical_alerts:
            report.append("🚨 CRITICAL ALERTS")
            for alert in summary.critical_alerts:
                report.append(f"  ⚠️  {alert}")
            report.append("")

        # Recommendations
        if summary.recommendations:
            report.append("💡 RECOMMENDATIONS")
            for i, recommendation in enumerate(summary.recommendations[:5], 1):
                report.append(f"  {i}. {recommendation}")

        return "\n".join(report)


# Global dashboard instance
_performance_dashboard_instance: Optional[PerformanceMetricsDashboard] = None


def get_performance_dashboard() -> PerformanceMetricsDashboard:
    """Get or create global performance dashboard instance"""
    global _performance_dashboard_instance

    if _performance_dashboard_instance is None:
        _performance_dashboard_instance = PerformanceMetricsDashboard()

    return _performance_dashboard_instance


async def record_performance_metric(
    component: ComponentType,
    metric_type: MetricType,
    value: float,
    context: Optional[Dict[str, Any]] = None,
) -> str:
    """Convenience function to record a performance metric"""
    dashboard = get_performance_dashboard()
    return await dashboard.record_metric(component, metric_type, value, context)


def get_system_health() -> SystemPerformanceSummary:
    """Convenience function to get system health summary"""
    dashboard = get_performance_dashboard()
    return dashboard.get_system_performance_summary()
