"""
Real-Time Quality Monitor
Comprehensive quality monitoring system for cognitive orchestration
Provides continuous tracking, alerting, and quality assurance
"""

import logging
import time
import statistics
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque
import json


class QualityMetricType(Enum):
    """Types of quality metrics to monitor"""

    OVERALL_SCORE = "overall_score"
    SYNTACTIC_QUALITY = "syntactic_quality"
    SEMANTIC_QUALITY = "semantic_quality"
    PRAGMATIC_QUALITY = "pragmatic_quality"
    ETHICAL_QUALITY = "ethical_quality"
    CONFIDENCE_SCORE = "confidence_score"
    PROCESSING_TIME = "processing_time"
    SUCCESS_RATE = "success_rate"
    VALIDATION_PASS_RATE = "validation_pass_rate"


class AlertSeverity(Enum):
    """Alert severity levels"""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class QualityTrend(Enum):
    """Quality trend indicators"""

    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    VOLATILE = "volatile"


@dataclass
class QualityThreshold:
    """Quality threshold configuration"""

    metric_type: QualityMetricType
    warning_threshold: float
    critical_threshold: float
    emergency_threshold: Optional[float] = None
    trend_window: int = 10  # Number of samples for trend analysis
    enabled: bool = True


@dataclass
class QualityAlert:
    """Quality alert data structure"""

    alert_id: str
    severity: AlertSeverity
    metric_type: QualityMetricType
    message: str
    current_value: float
    threshold_value: float
    timestamp: datetime
    orchestration_id: Optional[str] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary for serialization"""
        return {
            "alert_id": self.alert_id,
            "severity": self.severity.value,
            "metric_type": self.metric_type.value,
            "message": self.message,
            "current_value": self.current_value,
            "threshold_value": self.threshold_value,
            "timestamp": self.timestamp.isoformat(),
            "orchestration_id": self.orchestration_id,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "metadata": self.metadata,
        }


@dataclass
class QualityMetric:
    """Individual quality metric measurement"""

    metric_type: QualityMetricType
    value: float
    timestamp: datetime
    orchestration_id: str
    model_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityReport:
    """Comprehensive quality monitoring report"""

    report_id: str
    start_time: datetime
    end_time: datetime
    total_orchestrations: int
    metrics_summary: Dict[str, Dict[str, float]]  # metric_type -> {avg, min, max, std}
    trend_analysis: Dict[str, QualityTrend]
    active_alerts: List[QualityAlert]
    resolved_alerts: List[QualityAlert]
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class QualityMonitor:
    """
    Real-time quality monitoring system for cognitive orchestration

    Provides continuous tracking of quality metrics, trend analysis,
    automated alerting, and comprehensive reporting capabilities.
    """

    def __init__(
        self,
        enable_real_time_alerts: bool = True,
        max_metric_history: int = 1000,
        alert_callback: Optional[Callable[[QualityAlert], None]] = None,
    ):
        """
        Initialize quality monitor

        Args:
            enable_real_time_alerts: Enable automatic alert generation
            max_metric_history: Maximum number of metrics to keep in memory
            alert_callback: Optional callback function for alert notifications
        """
        self.enable_real_time_alerts = enable_real_time_alerts
        self.max_metric_history = max_metric_history
        self.alert_callback = alert_callback

        self.logger = logging.getLogger(__name__)

        # Metric storage with circular buffers for memory efficiency
        self.metrics_history: Dict[QualityMetricType, deque] = {
            metric_type: deque(maxlen=max_metric_history)
            for metric_type in QualityMetricType
        }

        # Alert management
        self.active_alerts: Dict[str, QualityAlert] = {}
        self.resolved_alerts: List[QualityAlert] = []
        self.alert_counter = 0

        # Quality thresholds (can be configured)
        self.thresholds: Dict[QualityMetricType, QualityThreshold] = (
            self._create_default_thresholds()
        )

        # Aggregated statistics cache
        self._stats_cache: Dict[str, Any] = {}
        self._stats_cache_timestamp = 0
        self._stats_cache_ttl = 30  # 30 seconds cache TTL

        # Monitoring state
        self.monitoring_active = True
        self.total_orchestrations_monitored = 0
        self.start_time = datetime.now(timezone.utc)

        self.logger.info(
            "🔍 QualityMonitor initialized with real-time alerts: %s",
            enable_real_time_alerts,
        )

    def _create_default_thresholds(self) -> Dict[QualityMetricType, QualityThreshold]:
        """Create default quality thresholds"""
        return {
            QualityMetricType.OVERALL_SCORE: QualityThreshold(
                metric_type=QualityMetricType.OVERALL_SCORE,
                warning_threshold=0.7,
                critical_threshold=0.5,
                emergency_threshold=0.3,
            ),
            QualityMetricType.SYNTACTIC_QUALITY: QualityThreshold(
                metric_type=QualityMetricType.SYNTACTIC_QUALITY,
                warning_threshold=0.8,
                critical_threshold=0.6,
                emergency_threshold=0.4,
            ),
            QualityMetricType.SEMANTIC_QUALITY: QualityThreshold(
                metric_type=QualityMetricType.SEMANTIC_QUALITY,
                warning_threshold=0.7,
                critical_threshold=0.5,
                emergency_threshold=0.3,
            ),
            QualityMetricType.PRAGMATIC_QUALITY: QualityThreshold(
                metric_type=QualityMetricType.PRAGMATIC_QUALITY,
                warning_threshold=0.7,
                critical_threshold=0.5,
                emergency_threshold=0.3,
            ),
            QualityMetricType.ETHICAL_QUALITY: QualityThreshold(
                metric_type=QualityMetricType.ETHICAL_QUALITY,
                warning_threshold=0.8,
                critical_threshold=0.6,
                emergency_threshold=0.4,
            ),
            QualityMetricType.CONFIDENCE_SCORE: QualityThreshold(
                metric_type=QualityMetricType.CONFIDENCE_SCORE,
                warning_threshold=0.6,
                critical_threshold=0.4,
                emergency_threshold=0.2,
            ),
            QualityMetricType.PROCESSING_TIME: QualityThreshold(
                metric_type=QualityMetricType.PROCESSING_TIME,
                warning_threshold=5000.0,  # 5 seconds
                critical_threshold=10000.0,  # 10 seconds
                emergency_threshold=30000.0,  # 30 seconds
            ),
            QualityMetricType.SUCCESS_RATE: QualityThreshold(
                metric_type=QualityMetricType.SUCCESS_RATE,
                warning_threshold=0.9,
                critical_threshold=0.8,
                emergency_threshold=0.7,
            ),
            QualityMetricType.VALIDATION_PASS_RATE: QualityThreshold(
                metric_type=QualityMetricType.VALIDATION_PASS_RATE,
                warning_threshold=0.85,
                critical_threshold=0.7,
                emergency_threshold=0.5,
            ),
        }

    def record_orchestration_quality(
        self,
        orchestration_id: str,
        quality_validation_results: Optional[Dict[str, Any]] = None,
        confidence_score: float = 0.0,
        processing_time_ms: float = 0.0,
        success: bool = True,
        model_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record quality metrics for an orchestration

        Args:
            orchestration_id: Unique identifier for the orchestration
            quality_validation_results: Results from quality validation
            confidence_score: Confidence score from cognitive model
            processing_time_ms: Processing time in milliseconds
            success: Whether orchestration was successful
            model_id: ID of the primary model used
            metadata: Additional metadata
        """
        timestamp = datetime.now(timezone.utc)
        metadata = metadata or {}

        # Record basic metrics
        self._record_metric(
            QualityMetricType.CONFIDENCE_SCORE,
            confidence_score,
            timestamp,
            orchestration_id,
            model_id,
            metadata,
        )

        self._record_metric(
            QualityMetricType.PROCESSING_TIME,
            processing_time_ms,
            timestamp,
            orchestration_id,
            model_id,
            metadata,
        )

        # Record quality validation metrics if available
        if quality_validation_results:
            overall_score = quality_validation_results.get("overall_score", 0.0)
            self._record_metric(
                QualityMetricType.OVERALL_SCORE,
                overall_score,
                timestamp,
                orchestration_id,
                model_id,
                metadata,
            )

            # Record dimensional quality scores
            dimension_scores = quality_validation_results.get("dimension_scores", {})
            for dimension, score in dimension_scores.items():
                if dimension == "syntactic":
                    metric_type = QualityMetricType.SYNTACTIC_QUALITY
                elif dimension == "semantic":
                    metric_type = QualityMetricType.SEMANTIC_QUALITY
                elif dimension == "pragmatic":
                    metric_type = QualityMetricType.PRAGMATIC_QUALITY
                elif dimension == "ethical":
                    metric_type = QualityMetricType.ETHICAL_QUALITY
                else:
                    continue

                self._record_metric(
                    metric_type, score, timestamp, orchestration_id, model_id, metadata
                )

        self.total_orchestrations_monitored += 1

        # Update aggregated success rate
        self._update_success_rate(
            success, timestamp, orchestration_id, model_id, metadata
        )

        # Update validation pass rate
        validation_passed = (
            quality_validation_results.get("passed", False)
            if quality_validation_results
            else True
        )
        self._update_validation_pass_rate(
            validation_passed, timestamp, orchestration_id, model_id, metadata
        )

        # Clear stats cache
        self._stats_cache = {}
        self._stats_cache_timestamp = 0

        self.logger.debug(
            f"📊 Recorded quality metrics for orchestration {orchestration_id[:8]}... "
            f"(confidence: {confidence_score:.3f}, time: {processing_time_ms:.1f}ms)"
        )

    def _record_metric(
        self,
        metric_type: QualityMetricType,
        value: float,
        timestamp: datetime,
        orchestration_id: str,
        model_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record individual quality metric"""
        metric = QualityMetric(
            metric_type=metric_type,
            value=value,
            timestamp=timestamp,
            orchestration_id=orchestration_id,
            model_id=model_id,
            metadata=metadata or {},
        )

        self.metrics_history[metric_type].append(metric)

        # Check for threshold violations if alerts are enabled
        if self.enable_real_time_alerts and self.monitoring_active:
            self._check_thresholds(metric)

    def _update_success_rate(
        self,
        success: bool,
        timestamp: datetime,
        orchestration_id: str,
        model_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update running success rate metric"""
        # Calculate success rate over recent window
        recent_metrics = list(self.metrics_history[QualityMetricType.SUCCESS_RATE])[
            -100:
        ]  # Last 100

        if recent_metrics:
            successes = sum(1 for m in recent_metrics if m.value == 1.0) + (
                1 if success else 0
            )
            total = len(recent_metrics) + 1
            success_rate = successes / total
        else:
            success_rate = 1.0 if success else 0.0

        self._record_metric(
            QualityMetricType.SUCCESS_RATE,
            success_rate,
            timestamp,
            orchestration_id,
            model_id,
            metadata,
        )

    def _update_validation_pass_rate(
        self,
        validation_passed: bool,
        timestamp: datetime,
        orchestration_id: str,
        model_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update running validation pass rate metric"""
        # Calculate validation pass rate over recent window
        recent_metrics = list(
            self.metrics_history[QualityMetricType.VALIDATION_PASS_RATE]
        )[
            -100:
        ]  # Last 100

        if recent_metrics:
            passes = sum(1 for m in recent_metrics if m.value == 1.0) + (
                1 if validation_passed else 0
            )
            total = len(recent_metrics) + 1
            pass_rate = passes / total
        else:
            pass_rate = 1.0 if validation_passed else 0.0

        self._record_metric(
            QualityMetricType.VALIDATION_PASS_RATE,
            pass_rate,
            timestamp,
            orchestration_id,
            model_id,
            metadata,
        )

    def _check_thresholds(self, metric: QualityMetric) -> None:
        """Check if metric violates quality thresholds"""
        threshold = self.thresholds.get(metric.metric_type)
        if not threshold or not threshold.enabled:
            return

        severity = None
        threshold_value = None

        # Determine severity based on threshold violations
        # For processing time, higher values are worse
        if metric.metric_type == QualityMetricType.PROCESSING_TIME:
            if (
                threshold.emergency_threshold
                and metric.value >= threshold.emergency_threshold
            ):
                severity = AlertSeverity.EMERGENCY
                threshold_value = threshold.emergency_threshold
            elif metric.value >= threshold.critical_threshold:
                severity = AlertSeverity.CRITICAL
                threshold_value = threshold.critical_threshold
            elif metric.value >= threshold.warning_threshold:
                severity = AlertSeverity.WARNING
                threshold_value = threshold.warning_threshold
        else:
            # For quality metrics, lower values are worse
            if (
                threshold.emergency_threshold
                and metric.value <= threshold.emergency_threshold
            ):
                severity = AlertSeverity.EMERGENCY
                threshold_value = threshold.emergency_threshold
            elif metric.value <= threshold.critical_threshold:
                severity = AlertSeverity.CRITICAL
                threshold_value = threshold.critical_threshold
            elif metric.value <= threshold.warning_threshold:
                severity = AlertSeverity.WARNING
                threshold_value = threshold.warning_threshold

        if severity:
            self._create_alert(metric, severity, threshold_value)

    def _create_alert(
        self, metric: QualityMetric, severity: AlertSeverity, threshold_value: float
    ) -> None:
        """Create quality alert"""
        self.alert_counter += 1
        alert_id = f"QM-{self.alert_counter:06d}"

        # Create descriptive message
        if metric.metric_type == QualityMetricType.PROCESSING_TIME:
            message = f"Processing time {metric.value:.1f}ms exceeds {severity.value} threshold ({threshold_value:.1f}ms)"
        else:
            message = f"{metric.metric_type.value.replace('_', ' ').title()} {metric.value:.3f} below {severity.value} threshold ({threshold_value:.3f})"

        alert = QualityAlert(
            alert_id=alert_id,
            severity=severity,
            metric_type=metric.metric_type,
            message=message,
            current_value=metric.value,
            threshold_value=threshold_value,
            timestamp=metric.timestamp,
            orchestration_id=metric.orchestration_id,
            metadata={"model_id": metric.model_id, "metric_metadata": metric.metadata},
        )

        self.active_alerts[alert_id] = alert

        # Call alert callback if configured
        if self.alert_callback:
            try:
                self.alert_callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")

        # Log alert
        log_level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.CRITICAL: logging.ERROR,
            AlertSeverity.EMERGENCY: logging.CRITICAL,
        }.get(severity, logging.WARNING)

        self.logger.log(
            log_level, f"🚨 Quality Alert [{severity.value.upper()}]: {message}"
        )

    def resolve_alert(
        self, alert_id: str, resolution_note: Optional[str] = None
    ) -> bool:
        """
        Resolve an active alert

        Args:
            alert_id: ID of alert to resolve
            resolution_note: Optional note about resolution

        Returns:
            True if alert was resolved, False if not found
        """
        if alert_id not in self.active_alerts:
            return False

        alert = self.active_alerts.pop(alert_id)
        alert.resolved = True
        alert.resolved_at = datetime.now(timezone.utc)

        if resolution_note:
            alert.metadata["resolution_note"] = resolution_note

        self.resolved_alerts.append(alert)

        # Keep only last 500 resolved alerts for memory management
        if len(self.resolved_alerts) > 500:
            self.resolved_alerts = self.resolved_alerts[-500:]

        self.logger.info(f"✅ Resolved quality alert {alert_id}: {alert.message}")
        return True

    def get_quality_statistics(
        self, time_window_hours: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive quality statistics

        Args:
            time_window_hours: Optional time window for statistics (default: all time)

        Returns:
            Dictionary with quality statistics
        """
        # Check cache
        cache_key = f"stats_{time_window_hours}"
        current_time = time.time()

        if (
            cache_key in self._stats_cache
            and current_time - self._stats_cache_timestamp < self._stats_cache_ttl
        ):
            return self._stats_cache[cache_key]

        cutoff_time = None
        if time_window_hours:
            cutoff_time = datetime.now(timezone.utc) - timedelta(
                hours=time_window_hours
            )

        stats = {
            "monitoring_period": {
                "start_time": self.start_time.isoformat(),
                "current_time": datetime.now(timezone.utc).isoformat(),
                "total_orchestrations": self.total_orchestrations_monitored,
                "time_window_hours": time_window_hours,
            },
            "metrics_summary": {},
            "alert_summary": {
                "active_alerts": len(self.active_alerts),
                "resolved_alerts": len(self.resolved_alerts),
                "alerts_by_severity": defaultdict(int),
                "alerts_by_metric": defaultdict(int),
            },
            "trend_analysis": {},
            "recommendations": [],
        }

        # Calculate metrics summary
        for metric_type, metrics in self.metrics_history.items():
            # Filter by time window if specified
            filtered_metrics = metrics
            if cutoff_time:
                filtered_metrics = [m for m in metrics if m.timestamp >= cutoff_time]

            if not filtered_metrics:
                continue

            values = [m.value for m in filtered_metrics]

            stats["metrics_summary"][metric_type.value] = {
                "count": len(values),
                "average": statistics.mean(values),
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
                "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
                "latest": values[-1] if values else 0.0,
                "trend": self._analyze_trend(
                    values[-10:] if len(values) >= 10 else values
                ).value,
            }

        # Alert summary
        for alert in self.active_alerts.values():
            stats["alert_summary"]["alerts_by_severity"][alert.severity.value] += 1
            stats["alert_summary"]["alerts_by_metric"][alert.metric_type.value] += 1

        # Generate recommendations
        stats["recommendations"] = self._generate_recommendations(stats)

        # Cache results
        self._stats_cache[cache_key] = stats
        self._stats_cache_timestamp = current_time

        return stats

    def _analyze_trend(self, values: List[float]) -> QualityTrend:
        """Analyze trend in metric values"""
        if len(values) < 3:
            return QualityTrend.STABLE

        # Calculate moving averages for trend detection
        mid_point = len(values) // 2
        first_half_avg = statistics.mean(values[:mid_point])
        second_half_avg = statistics.mean(values[mid_point:])

        # Calculate standard deviation to detect volatility
        std_dev = statistics.stdev(values) if len(values) > 1 else 0.0
        mean_val = statistics.mean(values)

        # High volatility check
        if std_dev > mean_val * 0.3:  # High coefficient of variation
            return QualityTrend.VOLATILE

        # Trend detection
        trend_threshold = mean_val * 0.05  # 5% change threshold

        if second_half_avg > first_half_avg + trend_threshold:
            return QualityTrend.IMPROVING
        elif second_half_avg < first_half_avg - trend_threshold:
            return QualityTrend.DEGRADING
        else:
            return QualityTrend.STABLE

    def _generate_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate quality improvement recommendations"""
        recommendations = []

        metrics_summary = stats.get("metrics_summary", {})
        alert_summary = stats.get("alert_summary", {})

        # Check for quality issues
        overall_score = metrics_summary.get("overall_score")
        if overall_score and overall_score["average"] < 0.7:
            recommendations.append(
                "Overall quality score is below target (0.7). "
                "Consider reviewing cognitive model selection and quality validation thresholds."
            )

        # Check for performance issues
        processing_time = metrics_summary.get("processing_time")
        if processing_time and processing_time["average"] > 3000:  # 3 seconds
            recommendations.append(
                "Average processing time exceeds 3 seconds. "
                "Consider optimizing model selection strategy or implementing caching."
            )

        # Check for high alert volume
        if alert_summary["active_alerts"] > 10:
            recommendations.append(
                f"High number of active alerts ({alert_summary['active_alerts']}). "
                "Review alert thresholds and resolve outstanding quality issues."
            )

        # Check for trending issues
        for metric_name, metric_data in metrics_summary.items():
            if metric_data.get("trend") == "degrading":
                recommendations.append(
                    f"{metric_name.replace('_', ' ').title()} is showing a degrading trend. "
                    "Investigate recent changes to cognitive models or validation criteria."
                )

        # Check for low success rates
        success_rate = metrics_summary.get("success_rate")
        if success_rate and success_rate["latest"] < 0.9:
            recommendations.append(
                f"Success rate is {success_rate['latest']:.1%}, below target (90%). "
                "Review error handling and model reliability."
            )

        if not recommendations:
            recommendations.append(
                "Quality metrics are within acceptable ranges. Continue monitoring."
            )

        return recommendations

    def get_active_alerts(
        self, severity_filter: Optional[AlertSeverity] = None
    ) -> List[QualityAlert]:
        """Get list of active alerts, optionally filtered by severity"""
        alerts = list(self.active_alerts.values())

        if severity_filter:
            alerts = [alert for alert in alerts if alert.severity == severity_filter]

        # Sort by severity (emergency first) then by timestamp
        severity_order = {
            AlertSeverity.EMERGENCY: 0,
            AlertSeverity.CRITICAL: 1,
            AlertSeverity.WARNING: 2,
            AlertSeverity.INFO: 3,
        }

        alerts.sort(key=lambda a: (severity_order[a.severity], a.timestamp))
        return alerts

    def generate_quality_report(
        self, time_window_hours: int = 24, include_recommendations: bool = True
    ) -> QualityReport:
        """
        Generate comprehensive quality report

        Args:
            time_window_hours: Time window for the report
            include_recommendations: Whether to include recommendations

        Returns:
            QualityReport with comprehensive analysis
        """
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=time_window_hours)

        # Get statistics for the time window
        stats = self.get_quality_statistics(time_window_hours)

        # Count orchestrations in time window
        total_orchestrations = 0
        for metrics in self.metrics_history.values():
            for metric in metrics:
                if start_time <= metric.timestamp <= end_time:
                    # Count unique orchestration IDs
                    break

        # Estimate based on confidence score metrics (should be 1 per orchestration)
        confidence_metrics = [
            m
            for m in self.metrics_history[QualityMetricType.CONFIDENCE_SCORE]
            if start_time <= m.timestamp <= end_time
        ]
        total_orchestrations = len(confidence_metrics)

        # Get active and recent resolved alerts
        active_alerts = self.get_active_alerts()
        recent_resolved = [
            alert
            for alert in self.resolved_alerts
            if alert.resolved_at and start_time <= alert.resolved_at <= end_time
        ]

        # Generate trend analysis
        trend_analysis = {}
        for metric_name, metric_data in stats["metrics_summary"].items():
            trend_analysis[metric_name] = QualityTrend(metric_data["trend"])

        report = QualityReport(
            report_id=f"QR-{int(time.time())}-{time_window_hours}h",
            start_time=start_time,
            end_time=end_time,
            total_orchestrations=total_orchestrations,
            metrics_summary=stats["metrics_summary"],
            trend_analysis=trend_analysis,
            active_alerts=active_alerts,
            resolved_alerts=recent_resolved,
            recommendations=stats["recommendations"] if include_recommendations else [],
            metadata={
                "monitoring_duration_hours": time_window_hours,
                "alert_summary": stats["alert_summary"],
                "generated_at": end_time.isoformat(),
            },
        )

        self.logger.info(
            f"📋 Generated quality report {report.report_id}: "
            f"{total_orchestrations} orchestrations, {len(active_alerts)} active alerts"
        )

        return report

    def configure_threshold(
        self,
        metric_type: QualityMetricType,
        warning_threshold: Optional[float] = None,
        critical_threshold: Optional[float] = None,
        emergency_threshold: Optional[float] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        """
        Configure quality threshold for a metric

        Args:
            metric_type: Type of metric to configure
            warning_threshold: Warning threshold value
            critical_threshold: Critical threshold value
            emergency_threshold: Emergency threshold value
            enabled: Whether threshold monitoring is enabled
        """
        threshold = self.thresholds.get(metric_type)
        if not threshold:
            threshold = QualityThreshold(
                metric_type=metric_type,
                warning_threshold=0.5,
                critical_threshold=0.3,
                emergency_threshold=0.1,
            )
            self.thresholds[metric_type] = threshold

        if warning_threshold is not None:
            threshold.warning_threshold = warning_threshold
        if critical_threshold is not None:
            threshold.critical_threshold = critical_threshold
        if emergency_threshold is not None:
            threshold.emergency_threshold = emergency_threshold
        if enabled is not None:
            threshold.enabled = enabled

        self.logger.info(
            f"⚙️ Updated threshold for {metric_type.value}: "
            f"warn={threshold.warning_threshold}, crit={threshold.critical_threshold}, "
            f"emerg={threshold.emergency_threshold}, enabled={threshold.enabled}"
        )

    def export_metrics(self, format: str = "json") -> str:
        """
        Export metrics in specified format

        Args:
            format: Export format ("json", "csv")

        Returns:
            Formatted metrics data
        """
        if format == "json":
            return self._export_json()
        elif format == "csv":
            return self._export_csv()
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _export_json(self) -> str:
        """Export metrics as JSON"""
        export_data = {
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "monitoring_start": self.start_time.isoformat(),
            "total_orchestrations": self.total_orchestrations_monitored,
            "metrics": {},
            "active_alerts": [alert.to_dict() for alert in self.active_alerts.values()],
            "thresholds": {
                metric_type.value: {
                    "warning": threshold.warning_threshold,
                    "critical": threshold.critical_threshold,
                    "emergency": threshold.emergency_threshold,
                    "enabled": threshold.enabled,
                }
                for metric_type, threshold in self.thresholds.items()
            },
        }

        # Export recent metrics (last 100 per type)
        for metric_type, metrics in self.metrics_history.items():
            export_data["metrics"][metric_type.value] = [
                {
                    "value": m.value,
                    "timestamp": m.timestamp.isoformat(),
                    "orchestration_id": m.orchestration_id,
                    "model_id": m.model_id,
                }
                for m in list(metrics)[-100:]  # Last 100 metrics
            ]

        return json.dumps(export_data, indent=2)

    def _export_csv(self) -> str:
        """Export metrics as CSV"""
        lines = ["metric_type,value,timestamp,orchestration_id,model_id"]

        for metric_type, metrics in self.metrics_history.items():
            for metric in list(metrics)[-100:]:  # Last 100 metrics
                lines.append(
                    f"{metric_type.value},{metric.value},"
                    f"{metric.timestamp.isoformat()},{metric.orchestration_id},"
                    f"{metric.model_id or ''}"
                )

        return "\n".join(lines)

    def reset_monitoring(self) -> None:
        """Reset all monitoring data"""
        self.metrics_history = {
            metric_type: deque(maxlen=self.max_metric_history)
            for metric_type in QualityMetricType
        }
        self.active_alerts.clear()
        self.resolved_alerts.clear()
        self.alert_counter = 0
        self.total_orchestrations_monitored = 0
        self.start_time = datetime.now(timezone.utc)
        self._stats_cache.clear()

        self.logger.info("🔄 Quality monitoring data reset")


# Global quality monitor instance
_global_quality_monitor: Optional[QualityMonitor] = None


def get_quality_monitor() -> QualityMonitor:
    """Get or create global quality monitor instance"""
    global _global_quality_monitor
    if _global_quality_monitor is None:
        _global_quality_monitor = QualityMonitor()
    return _global_quality_monitor


def create_quality_monitor(
    enable_real_time_alerts: bool = True,
    max_metric_history: int = 1000,
    alert_callback: Optional[Callable[[QualityAlert], None]] = None,
) -> QualityMonitor:
    """Create a new quality monitor instance"""
    return QualityMonitor(
        enable_real_time_alerts=enable_real_time_alerts,
        max_metric_history=max_metric_history,
        alert_callback=alert_callback,
    )
