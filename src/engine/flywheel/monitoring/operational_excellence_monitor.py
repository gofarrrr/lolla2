"""
METIS Operational Excellence Monitor
Enterprise-grade monitoring, KPIs, dashboards and alerting for production deployment
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from collections import deque, defaultdict
import numpy as np

from src.production.flywheel_cache_system import get_flywheel_cache
from src.production.learning_loop import get_learning_loop
from src.engine.adapters.audit_trail import get_audit_manager, AuditEventType, AuditSeverity  # Migrated

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class ComponentStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class KPI:
    """Key Performance Indicator definition"""

    name: str
    current_value: float
    target_value: float
    threshold_warning: float
    threshold_critical: float
    unit: str
    description: str
    last_updated: datetime
    trend: float = 0.0  # positive = improving, negative = declining


@dataclass
class Alert:
    """System alert definition"""

    alert_id: str
    severity: AlertSeverity
    component: str
    message: str
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    metadata: Dict[str, Any] = None


@dataclass
class SystemHealth:
    """Overall system health status"""

    overall_status: ComponentStatus
    uptime_seconds: float
    components: Dict[str, ComponentStatus]
    active_alerts: List[Alert]
    performance_score: float  # 0.0 to 1.0
    reliability_score: float  # 0.0 to 1.0


class OperationalExcellenceMonitor:
    """
    Enterprise monitoring system for METIS platform
    Tracks KPIs, generates alerts, and provides operational insights
    """

    def __init__(self):
        # Core monitoring data
        self.kpis: Dict[str, KPI] = {}
        self.alerts: List[Alert] = []
        self.component_health: Dict[str, ComponentStatus] = {}
        self.metrics_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1440)
        )  # 24h at 1min resolution

        # Performance tracking
        self.engagement_metrics = deque(maxlen=10000)  # Last 10k engagements
        self.system_start_time = datetime.utcnow()

        # Alert handlers
        self.alert_handlers: List[Callable[[Alert], None]] = []

        # Monitoring configuration
        self.monitoring_enabled = True
        self.metric_collection_interval = 60  # seconds
        self.alert_cooldown_period = 300  # 5 minutes
        self.alert_history: Dict[str, datetime] = {}

        # Initialize KPIs
        self._initialize_kpis()

    async def initialize(self):
        """Initialize the monitoring system"""
        logger.info("🔍 Initializing Operational Excellence Monitor")

        # Start background monitoring tasks
        if self.monitoring_enabled:
            asyncio.create_task(self._metric_collection_loop())
            asyncio.create_task(self._health_check_loop())
            asyncio.create_task(self._alert_processing_loop())

        logger.info("✅ Operational Excellence Monitor initialized")

    def record_engagement_metrics(self, engagement_data: Dict[str, Any]):
        """Record metrics from an engagement"""
        metrics = {
            "timestamp": datetime.utcnow(),
            "processing_time_ms": engagement_data.get("processing_time_ms", 0),
            "success": engagement_data.get("status") == "completed",
            "cache_hit": engagement_data.get("cache_hit", False),
            "consultant_count": len(engagement_data.get("consultants_used", [])),
            "user_satisfaction": engagement_data.get("user_satisfaction"),
            "critique_requested": engagement_data.get("critique_requested", False),
            "arbitration_requested": engagement_data.get(
                "arbitration_requested", False
            ),
        }

        self.engagement_metrics.append(metrics)

        # Update real-time KPIs
        self._update_realtime_kpis()

    def get_system_health(self) -> SystemHealth:
        """Get comprehensive system health status"""
        # Calculate overall status
        unhealthy_components = sum(
            1
            for status in self.component_health.values()
            if status in [ComponentStatus.UNHEALTHY, ComponentStatus.UNKNOWN]
        )
        degraded_components = sum(
            1
            for status in self.component_health.values()
            if status == ComponentStatus.DEGRADED
        )

        if unhealthy_components > 0:
            overall_status = ComponentStatus.UNHEALTHY
        elif degraded_components > 2:
            overall_status = ComponentStatus.DEGRADED
        else:
            overall_status = ComponentStatus.HEALTHY

        # Calculate performance scores
        performance_score = self._calculate_performance_score()
        reliability_score = self._calculate_reliability_score()

        # Get active alerts
        active_alerts = [alert for alert in self.alerts if not alert.resolved]

        # Calculate uptime
        uptime = (datetime.utcnow() - self.system_start_time).total_seconds()

        return SystemHealth(
            overall_status=overall_status,
            uptime_seconds=uptime,
            components=dict(self.component_health),
            active_alerts=active_alerts,
            performance_score=performance_score,
            reliability_score=reliability_score,
        )

    def get_kpi_dashboard(self) -> Dict[str, Any]:
        """Get KPI dashboard data for monitoring UI"""
        dashboard_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "kpis": {},
            "trends": {},
            "alerts_summary": {
                "active": len([a for a in self.alerts if not a.resolved]),
                "critical": len(
                    [
                        a
                        for a in self.alerts
                        if not a.resolved and a.severity == AlertSeverity.CRITICAL
                    ]
                ),
                "warning": len(
                    [
                        a
                        for a in self.alerts
                        if not a.resolved and a.severity == AlertSeverity.WARNING
                    ]
                ),
            },
            "system_overview": {
                "total_engagements": len(self.engagement_metrics),
                "avg_response_time": self._calculate_avg_response_time(),
                "success_rate": self._calculate_success_rate(),
                "cache_hit_rate": self._calculate_cache_hit_rate(),
                "user_satisfaction": self._calculate_avg_user_satisfaction(),
            },
        }

        # Include KPI data
        for name, kpi in self.kpis.items():
            dashboard_data["kpis"][name] = {
                "current": kpi.current_value,
                "target": kpi.target_value,
                "unit": kpi.unit,
                "status": self._get_kpi_status(kpi),
                "trend": kpi.trend,
            }

        # Include trend data (last 24 hours)
        for metric_name, history in self.metrics_history.items():
            if history:
                recent_data = list(history)[-60:]  # Last hour
                dashboard_data["trends"][metric_name] = {
                    "data_points": len(recent_data),
                    "current": recent_data[-1] if recent_data else 0,
                    "average": np.mean(recent_data) if recent_data else 0,
                    "trend": (
                        np.polyfit(range(len(recent_data)), recent_data, 1)[0]
                        if len(recent_data) > 1
                        else 0
                    ),
                }

        return dashboard_data

    def get_performance_insights(self) -> Dict[str, Any]:
        """Get detailed performance insights and optimization recommendations"""
        if not self.engagement_metrics:
            return {"message": "Insufficient data for insights"}

        recent_engagements = list(self.engagement_metrics)[-1000:]  # Last 1000

        insights = {
            "performance_analysis": {
                "avg_processing_time": np.mean(
                    [e["processing_time_ms"] for e in recent_engagements]
                ),
                "p95_processing_time": np.percentile(
                    [e["processing_time_ms"] for e in recent_engagements], 95
                ),
                "success_rate": np.mean([e["success"] for e in recent_engagements]),
                "cache_effectiveness": np.mean(
                    [e["cache_hit"] for e in recent_engagements]
                ),
            },
            "usage_patterns": {
                "peak_hours": self._analyze_peak_hours(recent_engagements),
                "critique_usage_rate": np.mean(
                    [e["critique_requested"] for e in recent_engagements]
                ),
                "arbitration_usage_rate": np.mean(
                    [e["arbitration_requested"] for e in recent_engagements]
                ),
            },
            "optimization_recommendations": self._generate_optimization_recommendations(
                recent_engagements
            ),
            "capacity_planning": self._analyze_capacity_needs(recent_engagements),
        }

        return insights

    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add a custom alert handler"""
        self.alert_handlers.append(handler)

    async def _metric_collection_loop(self):
        """Background task for collecting metrics"""
        while self.monitoring_enabled:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(self.metric_collection_interval)
            except Exception as e:
                logger.error(f"Error in metric collection: {e}")
                await asyncio.sleep(self.metric_collection_interval)

    async def _health_check_loop(self):
        """Background task for health checking"""
        while self.monitoring_enabled:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(30)  # Health checks every 30 seconds
            except Exception as e:
                logger.error(f"Error in health check: {e}")
                await asyncio.sleep(30)

    async def _alert_processing_loop(self):
        """Background task for processing and evaluating alerts"""
        while self.monitoring_enabled:
            try:
                await self._evaluate_alert_conditions()
                await asyncio.sleep(60)  # Alert evaluation every minute
            except Exception as e:
                logger.error(f"Error in alert processing: {e}")
                await asyncio.sleep(60)

    async def _collect_system_metrics(self):
        """Collect system-wide metrics"""
        timestamp = datetime.utcnow()

        # Collect from flywheel cache
        try:
            flywheel_cache = await get_flywheel_cache()
            cache_metrics = flywheel_cache.get_flywheel_metrics()

            self.metrics_history["cache_hit_rate"].append(cache_metrics.cache_hit_rate)
            self.metrics_history["avg_response_time"].append(
                cache_metrics.average_response_time_ms
            )
            self.metrics_history["user_satisfaction"].append(
                cache_metrics.user_satisfaction_score
            )
        except Exception as e:
            logger.warning(f"Failed to collect cache metrics: {e}")

        # Collect from learning loop
        try:
            learning_loop = await get_learning_loop()
            learning_insights = learning_loop.get_learning_insights()

            self.metrics_history["learning_accuracy"].append(
                learning_insights.get("recent_learning_accuracy", 0)
            )
            self.metrics_history["total_interactions"].append(
                learning_insights.get("total_learning_events", 0)
            )
        except Exception as e:
            logger.warning(f"Failed to collect learning metrics: {e}")

        # Update KPIs based on collected metrics
        self._update_kpis_from_metrics()

    async def _perform_health_checks(self):
        """Perform health checks on system components"""
        # Check flywheel cache health
        try:
            flywheel_cache = await get_flywheel_cache()
            # Simple health check - try to get metrics
            flywheel_cache.get_flywheel_metrics()
            self.component_health["flywheel_cache"] = ComponentStatus.HEALTHY
        except Exception:
            self.component_health["flywheel_cache"] = ComponentStatus.UNHEALTHY

        # Check learning loop health
        try:
            learning_loop = await get_learning_loop()
            insights = learning_loop.get_learning_insights()
            if insights.get("learning_system_health", {}).get(
                "data_sufficiency", False
            ):
                self.component_health["learning_loop"] = ComponentStatus.HEALTHY
            else:
                self.component_health["learning_loop"] = ComponentStatus.DEGRADED
        except Exception:
            self.component_health["learning_loop"] = ComponentStatus.UNHEALTHY

        # Check audit trail health
        try:
            audit_manager = await get_audit_manager()
            health = await audit_manager.get_audit_health_status()
            if health.get("storage_health") == "healthy":
                self.component_health["audit_trail"] = ComponentStatus.HEALTHY
            else:
                self.component_health["audit_trail"] = ComponentStatus.DEGRADED
        except Exception:
            self.component_health["audit_trail"] = ComponentStatus.UNHEALTHY

    async def _evaluate_alert_conditions(self):
        """Evaluate conditions and generate alerts if needed"""
        # Check KPI thresholds
        for name, kpi in self.kpis.items():
            await self._check_kpi_thresholds(name, kpi)

        # Check system health conditions
        health = self.get_system_health()
        if health.performance_score < 0.5:
            await self._create_alert(
                AlertSeverity.WARNING,
                "system_performance",
                f"System performance score below threshold: {health.performance_score:.2f}",
                {"performance_score": health.performance_score},
            )

        if health.reliability_score < 0.7:
            await self._create_alert(
                AlertSeverity.CRITICAL,
                "system_reliability",
                f"System reliability score critically low: {health.reliability_score:.2f}",
                {"reliability_score": health.reliability_score},
            )

    async def _check_kpi_thresholds(self, name: str, kpi: KPI):
        """Check if KPI violates thresholds and create alerts"""
        if kpi.current_value <= kpi.threshold_critical:
            await self._create_alert(
                AlertSeverity.CRITICAL,
                f"kpi_{name}",
                f"KPI {name} critically low: {kpi.current_value} {kpi.unit} (threshold: {kpi.threshold_critical})",
                {
                    "kpi": name,
                    "value": kpi.current_value,
                    "threshold": kpi.threshold_critical,
                },
            )
        elif kpi.current_value <= kpi.threshold_warning:
            await self._create_alert(
                AlertSeverity.WARNING,
                f"kpi_{name}",
                f"KPI {name} below warning threshold: {kpi.current_value} {kpi.unit}",
                {
                    "kpi": name,
                    "value": kpi.current_value,
                    "threshold": kpi.threshold_warning,
                },
            )

    async def _create_alert(
        self,
        severity: AlertSeverity,
        component: str,
        message: str,
        metadata: Dict[str, Any],
    ):
        """Create and process a new alert"""
        # Check cooldown period
        alert_key = f"{component}_{severity.value}"
        if alert_key in self.alert_history:
            last_alert = self.alert_history[alert_key]
            if datetime.utcnow() - last_alert < timedelta(
                seconds=self.alert_cooldown_period
            ):
                return  # Skip duplicate alert within cooldown period

        alert = Alert(
            alert_id=f"{component}_{int(time.time())}",
            severity=severity,
            component=component,
            message=message,
            timestamp=datetime.utcnow(),
            metadata=metadata,
        )

        self.alerts.append(alert)
        self.alert_history[alert_key] = alert.timestamp

        # Log to audit trail
        try:
            audit_manager = await get_audit_manager()
            await audit_manager.log_event(
                event_type=(
                    AuditEventType.SYSTEM_ERROR
                    if severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]
                    else AuditEventType.USER_INTERACTION
                ),
                severity=(
                    AuditSeverity.CRITICAL
                    if severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]
                    else AuditSeverity.MEDIUM
                ),
                action_performed="system_alert_generated",
                event_description=message,
                metadata={
                    "alert_id": alert.alert_id,
                    "severity": severity.value,
                    "component": component,
                    **metadata,
                },
            )
        except Exception as e:
            logger.error(f"Failed to log alert to audit trail: {e}")

        # Call alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")

        logger.warning(f"🚨 ALERT [{severity.value.upper()}] {component}: {message}")

    def _initialize_kpis(self):
        """Initialize default KPIs"""
        self.kpis = {
            "avg_response_time": KPI(
                name="Average Response Time",
                current_value=0.0,
                target_value=15000.0,  # 15 seconds
                threshold_warning=25000.0,  # 25 seconds
                threshold_critical=45000.0,  # 45 seconds
                unit="ms",
                description="Average engagement processing time",
                last_updated=datetime.utcnow(),
            ),
            "success_rate": KPI(
                name="Success Rate",
                current_value=1.0,
                target_value=0.99,  # 99%
                threshold_warning=0.95,  # 95%
                threshold_critical=0.90,  # 90%
                unit="%",
                description="Percentage of successful engagements",
                last_updated=datetime.utcnow(),
            ),
            "cache_hit_rate": KPI(
                name="Cache Hit Rate",
                current_value=0.0,
                target_value=0.60,  # 60%
                threshold_warning=0.30,  # 30%
                threshold_critical=0.10,  # 10%
                unit="%",
                description="Flywheel cache effectiveness",
                last_updated=datetime.utcnow(),
            ),
            "user_satisfaction": KPI(
                name="User Satisfaction",
                current_value=0.0,
                target_value=0.80,  # 80%
                threshold_warning=0.60,  # 60%
                threshold_critical=0.40,  # 40%
                unit="score",
                description="Average user satisfaction rating",
                last_updated=datetime.utcnow(),
            ),
        }

    def _update_realtime_kpis(self):
        """Update KPIs based on latest engagement metrics"""
        if not self.engagement_metrics:
            return

        recent_engagements = list(self.engagement_metrics)[-100:]  # Last 100

        # Update response time
        avg_time = np.mean([e["processing_time_ms"] for e in recent_engagements])
        self.kpis["avg_response_time"].current_value = avg_time

        # Update success rate
        success_rate = np.mean([e["success"] for e in recent_engagements])
        self.kpis["success_rate"].current_value = success_rate

        # Update cache hit rate
        cache_rate = np.mean([e["cache_hit"] for e in recent_engagements])
        self.kpis["cache_hit_rate"].current_value = cache_rate

        # Update user satisfaction (if available)
        satisfaction_scores = [
            e["user_satisfaction"]
            for e in recent_engagements
            if e["user_satisfaction"] is not None
        ]
        if satisfaction_scores:
            avg_satisfaction = np.mean(satisfaction_scores)
            self.kpis["user_satisfaction"].current_value = avg_satisfaction

        # Update timestamps
        for kpi in self.kpis.values():
            kpi.last_updated = datetime.utcnow()

    def _update_kpis_from_metrics(self):
        """Update KPIs from collected metrics history"""
        # Calculate trends
        for name, kpi in self.kpis.items():
            metric_key = name.replace("_", "_")
            if (
                metric_key in self.metrics_history
                and len(self.metrics_history[metric_key]) > 5
            ):
                recent_values = list(self.metrics_history[metric_key])[-10:]
                x = np.arange(len(recent_values))
                trend = (
                    np.polyfit(x, recent_values, 1)[0] if len(recent_values) > 1 else 0
                )
                kpi.trend = trend

    def _calculate_performance_score(self) -> float:
        """Calculate overall performance score (0.0 to 1.0)"""
        if not self.engagement_metrics:
            return 0.5

        recent = list(self.engagement_metrics)[-100:]

        # Weighted scoring
        success_weight = 0.4
        speed_weight = 0.3
        cache_weight = 0.2
        satisfaction_weight = 0.1

        success_score = np.mean([e["success"] for e in recent])

        # Speed score (inverse of processing time, normalized)
        avg_time = np.mean([e["processing_time_ms"] for e in recent])
        speed_score = max(0, 1 - (avg_time / 60000))  # 60s = 0 score

        cache_score = np.mean([e["cache_hit"] for e in recent])

        satisfaction_scores = [
            e["user_satisfaction"] for e in recent if e["user_satisfaction"] is not None
        ]
        satisfaction_score = (
            np.mean(satisfaction_scores) if satisfaction_scores else 0.5
        )

        performance = (
            success_score * success_weight
            + speed_score * speed_weight
            + cache_score * cache_weight
            + satisfaction_score * satisfaction_weight
        )

        return min(max(performance, 0.0), 1.0)

    def _calculate_reliability_score(self) -> float:
        """Calculate system reliability score (0.0 to 1.0)"""
        # Based on uptime, error rates, and component health
        uptime_hours = (
            datetime.utcnow() - self.system_start_time
        ).total_seconds() / 3600
        uptime_score = min(uptime_hours / 24, 1.0)  # Full score after 24h uptime

        # Component health score
        healthy_components = sum(
            1
            for status in self.component_health.values()
            if status == ComponentStatus.HEALTHY
        )
        total_components = len(self.component_health) if self.component_health else 1
        health_score = healthy_components / total_components

        # Alert score (fewer alerts = better reliability)
        active_alerts = len([a for a in self.alerts if not a.resolved])
        alert_score = max(0, 1 - (active_alerts / 10))  # 10 alerts = 0 score

        reliability = uptime_score * 0.4 + health_score * 0.4 + alert_score * 0.2
        return min(max(reliability, 0.0), 1.0)

    def _get_kpi_status(self, kpi: KPI) -> str:
        """Get status string for KPI"""
        if kpi.current_value <= kpi.threshold_critical:
            return "critical"
        elif kpi.current_value <= kpi.threshold_warning:
            return "warning"
        elif kpi.current_value >= kpi.target_value:
            return "excellent"
        else:
            return "good"

    def _calculate_avg_response_time(self) -> float:
        """Calculate average response time from recent engagements"""
        if not self.engagement_metrics:
            return 0.0
        recent = list(self.engagement_metrics)[-100:]
        return np.mean([e["processing_time_ms"] for e in recent])

    def _calculate_success_rate(self) -> float:
        """Calculate success rate from recent engagements"""
        if not self.engagement_metrics:
            return 1.0
        recent = list(self.engagement_metrics)[-100:]
        return np.mean([e["success"] for e in recent])

    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate from recent engagements"""
        if not self.engagement_metrics:
            return 0.0
        recent = list(self.engagement_metrics)[-100:]
        return np.mean([e["cache_hit"] for e in recent])

    def _calculate_avg_user_satisfaction(self) -> float:
        """Calculate average user satisfaction from recent engagements"""
        if not self.engagement_metrics:
            return 0.0
        recent = list(self.engagement_metrics)[-100:]
        satisfaction_scores = [
            e["user_satisfaction"] for e in recent if e["user_satisfaction"] is not None
        ]
        return np.mean(satisfaction_scores) if satisfaction_scores else 0.0

    def _analyze_peak_hours(self, engagements: List[Dict]) -> List[int]:
        """Analyze peak usage hours"""
        hour_counts = defaultdict(int)
        for engagement in engagements:
            hour = engagement["timestamp"].hour
            hour_counts[hour] += 1

        # Return top 3 peak hours
        sorted_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)
        return [hour for hour, count in sorted_hours[:3]]

    def _generate_optimization_recommendations(
        self, engagements: List[Dict]
    ) -> List[str]:
        """Generate optimization recommendations based on performance data"""
        recommendations = []

        avg_time = np.mean([e["processing_time_ms"] for e in engagements])
        cache_rate = np.mean([e["cache_hit"] for e in engagements])
        critique_rate = np.mean([e["critique_requested"] for e in engagements])

        if avg_time > 30000:  # 30 seconds
            recommendations.append(
                "Consider optimizing consultant selection algorithms - average processing time is high"
            )

        if cache_rate < 0.4:  # 40%
            recommendations.append(
                "Improve caching strategy - cache hit rate is below optimal"
            )

        if critique_rate > 0.8:  # 80%
            recommendations.append(
                "High critique usage suggests need for improved initial consultant recommendations"
            )

        return recommendations

    def _analyze_capacity_needs(self, engagements: List[Dict]) -> Dict[str, Any]:
        """Analyze capacity planning needs"""
        timestamps = [e["timestamp"] for e in engagements]
        if not timestamps:
            return {}

        # Calculate requests per hour
        time_range = max(timestamps) - min(timestamps)
        hours = max(time_range.total_seconds() / 3600, 1)
        requests_per_hour = len(engagements) / hours

        # Estimate peak capacity needs (2x average)
        peak_capacity_needed = requests_per_hour * 2

        return {
            "current_requests_per_hour": requests_per_hour,
            "estimated_peak_capacity": peak_capacity_needed,
            "capacity_recommendation": (
                "adequate" if peak_capacity_needed < 100 else "scale_up_needed"
            ),
        }


# Global monitor instance
_monitor: Optional[OperationalExcellenceMonitor] = None


async def get_operational_monitor() -> OperationalExcellenceMonitor:
    """Get or create global operational monitor instance"""
    global _monitor

    if _monitor is None:
        _monitor = OperationalExcellenceMonitor()
        await _monitor.initialize()

    return _monitor
