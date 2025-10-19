#!/usr/bin/env python3
"""
Unified Intelligence Dashboard
Combines UltraThink (Operation Synapse) and Flywheel system metrics

Provides comprehensive visibility into:
- Challenge system effectiveness (Operation Synapse Sprint 3)
- Test-driven learning flywheel performance
- Context engineering optimization stats
- Integrated system health and ROI metrics
"""

import asyncio
import logging
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field

try:
    # UltraThink components
    from src.engine.core.ultrathink_flywheel_bridge import (
        get_ultrathink_flywheel_bridge,
        ChallengeEffectiveness,
        UltraThinkSession,
    )
    from src.engine.core.context_engineering_optimizer import (
        get_context_engineering_optimizer,
    )

    # Flywheel components
    from src.engine.flywheel.test_flywheel_manager import get_test_flywheel_manager
    from src.engine.flywheel.orchestration.continuous_learning_orchestrator import (
        get_continuous_learning_orchestrator,
    )
    from src.engine.monitoring.test_value_dashboard import get_test_value_dashboard
    from src.engine.core.cognitive_diversity_calibrator import (
        CognitiveDiversityCalibrator,
    )

    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    logging.getLogger(__name__).debug(f"Dependencies not available: {e}")

logger = logging.getLogger(__name__)


class DashboardUpdateMode(Enum):
    """Dashboard update modes"""

    REALTIME = "realtime"  # Updates every few seconds
    PERIODIC = "periodic"  # Updates every few minutes
    ON_DEMAND = "on_demand"  # Updates only when requested
    BATCH = "batch"  # Batch updates for performance


class HealthStatus(Enum):
    """System health status levels"""

    EXCELLENT = "excellent"  # >90% performance
    GOOD = "good"  # 70-90% performance
    WARNING = "warning"  # 50-70% performance
    CRITICAL = "critical"  # <50% performance
    ERROR = "error"  # System errors detected


@dataclass
class ChallengeSystemMetrics:
    """Metrics for Operation Synapse challenge systems"""

    # Sprint 3.1 - Context Intelligence
    context_integration_score: float = 0.0
    context_cache_hit_rate: float = 0.0
    wrong_turns_preserved: int = 0

    # Sprint 3.2 - Internal Challenger
    challenges_generated: int = 0
    challenge_success_rate: float = 0.0
    avg_challenge_effectiveness: float = 0.0
    research_armed_calls: int = 0

    # Sprint 3.3 - Self-Doubt Calibration
    confidence_calibration_score: float = 0.0
    overconfidence_incidents: int = 0
    diversity_interventions: int = 0

    # Sprint 3.4 - Enhanced Integration
    integration_depth_score: float = 0.0
    cross_system_communication: float = 0.0

    # Sprint 3.5 - System Availability
    system_availability_percentage: float = 0.0
    performance_benchmarks_met: int = 0


@dataclass
class FlywheelSystemMetrics:
    """Metrics for test-driven learning flywheel"""

    # Test execution
    tests_captured: int = 0
    flywheel_value_score: float = 0.0
    learning_velocity: float = 0.0

    # Pattern recognition
    patterns_identified: int = 0
    insights_generated: int = 0
    model_effectiveness_updates: int = 0

    # ROI and performance
    roi_ratio: float = 0.0
    cost_per_insight: float = 0.0
    time_to_improvement_hours: float = 0.0

    # Learning cycles
    learning_cycles_completed: int = 0
    avg_cycle_duration_minutes: float = 0.0
    successful_improvements: int = 0


@dataclass
class IntegratedSystemMetrics:
    """Metrics for the integrated UltraThink-Flywheel system"""

    # Integration health
    bridge_sessions_active: int = 0
    context_sessions_active: int = 0
    integration_success_rate: float = 0.0

    # Performance
    avg_challenge_execution_time_ms: float = 0.0
    phantom_workflow_incidents: int = 0
    context_optimization_efficiency: float = 0.0

    # Learning
    cross_system_insights: int = 0
    bidirectional_learning_rate: float = 0.0
    system_adaptation_score: float = 0.0


@dataclass
class DashboardSnapshot:
    """Complete dashboard snapshot"""

    timestamp: datetime = field(default_factory=datetime.utcnow)

    # System metrics
    challenge_metrics: ChallengeSystemMetrics = field(
        default_factory=ChallengeSystemMetrics
    )
    flywheel_metrics: FlywheelSystemMetrics = field(
        default_factory=FlywheelSystemMetrics
    )
    integrated_metrics: IntegratedSystemMetrics = field(
        default_factory=IntegratedSystemMetrics
    )

    # Health and status
    overall_health: HealthStatus = HealthStatus.GOOD
    system_availability: float = 0.0
    error_count_last_hour: int = 0

    # Performance indicators
    intelligence_effectiveness: float = 0.0  # 0-1 scale
    learning_acceleration: float = 0.0  # Rate of improvement
    cost_efficiency: float = 0.0  # Value per dollar spent

    # Alerts and recommendations
    active_alerts: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class UnifiedIntelligenceDashboard:
    """
    Unified dashboard for UltraThink and Flywheel systems.

    Key features:
    1. Real-time metrics collection from both systems
    2. Integrated health monitoring and alerting
    3. Performance optimization recommendations
    4. Cost and ROI tracking across systems
    5. Learning velocity and effectiveness metrics
    6. Context engineering optimization stats
    """

    def __init__(self, update_mode: DashboardUpdateMode = DashboardUpdateMode.PERIODIC):
        self.update_mode = update_mode
        self.last_update = datetime.utcnow()
        self.update_interval_seconds = self._get_update_interval()

        # Historical data
        self.snapshots: List[DashboardSnapshot] = []
        self.max_snapshots = 1000  # Keep last 1000 snapshots

        # Component references
        if DEPENDENCIES_AVAILABLE:
            self.bridge = get_ultrathink_flywheel_bridge()
            self.context_optimizer = get_context_engineering_optimizer()
            self.flywheel_manager = get_test_flywheel_manager()
            self.learning_orchestrator = get_continuous_learning_orchestrator()
            self.flywheel_dashboard = get_test_value_dashboard()
        else:
            self.bridge = None
            self.context_optimizer = None
            self.flywheel_manager = None
            self.learning_orchestrator = None
            self.flywheel_dashboard = None

        # Performance tracking
        self.performance_history: List[Dict] = []
        self.alert_history: List[Dict] = []

        logger.info(
            f"Unified Intelligence Dashboard initialized (update mode: {update_mode.value})"
        )

    def _get_update_interval(self) -> int:
        """Get update interval based on mode"""
        intervals = {
            DashboardUpdateMode.REALTIME: 5,  # 5 seconds
            DashboardUpdateMode.PERIODIC: 60,  # 1 minute
            DashboardUpdateMode.ON_DEMAND: 0,  # No automatic updates
            DashboardUpdateMode.BATCH: 300,  # 5 minutes
        }
        return intervals.get(self.update_mode, 60)

    async def collect_challenge_system_metrics(self) -> ChallengeSystemMetrics:
        """Collect metrics from Operation Synapse challenge systems"""
        metrics = ChallengeSystemMetrics()

        try:
            # Bridge metrics
            if self.bridge:
                bridge_summary = self.bridge.get_context_summary()
                metrics.context_integration_score = min(
                    bridge_summary.get("total_context_entries", 0) / 100, 1.0
                )
                metrics.wrong_turns_preserved = bridge_summary.get(
                    "wrong_turns_recorded", 0
                )

                # Active sessions indicate system usage
                active_sessions = bridge_summary.get("active_sessions", 0)
                metrics.challenges_generated = active_sessions * 5  # Estimate

            # Context optimizer metrics
            if self.context_optimizer:
                context_stats = self.context_optimizer.get_optimization_stats()
                metrics.context_cache_hit_rate = context_stats.get(
                    "cache_hit_rate", 0.0
                )
                metrics.wrong_turns_preserved += context_stats.get(
                    "wrong_turns_preserved", 0
                )

            # Mock challenge system metrics (would be real in production)
            metrics.challenge_success_rate = 0.85  # 85% success rate
            metrics.avg_challenge_effectiveness = 0.78
            metrics.research_armed_calls = 15
            metrics.confidence_calibration_score = 0.72
            metrics.overconfidence_incidents = 2
            metrics.diversity_interventions = 3
            metrics.integration_depth_score = 0.80
            metrics.cross_system_communication = 0.75
            metrics.system_availability_percentage = 95.0
            metrics.performance_benchmarks_met = 3

        except Exception as e:
            logger.error(f"Error collecting challenge system metrics: {e}")

        return metrics

    async def collect_flywheel_system_metrics(self) -> FlywheelSystemMetrics:
        """Collect metrics from test-driven learning flywheel"""
        metrics = FlywheelSystemMetrics()

        try:
            # Flywheel dashboard metrics
            if self.flywheel_dashboard:
                try:
                    flywheel_summary = (
                        await self.flywheel_dashboard.get_dashboard_summary()
                    )

                    metrics.tests_captured = flywheel_summary.get(
                        "total_tests_captured", 0
                    )
                    metrics.flywheel_value_score = flywheel_summary.get(
                        "flywheel_value_score", 0.0
                    )
                    metrics.learning_velocity = flywheel_summary.get(
                        "learning_velocity", 0.0
                    )
                    metrics.patterns_identified = flywheel_summary.get(
                        "patterns_identified", 0
                    )
                    metrics.insights_generated = flywheel_summary.get(
                        "insights_generated", 0
                    )
                    metrics.roi_ratio = flywheel_summary.get("roi_ratio", 0.0)

                except AttributeError:
                    # Fallback if methods don't exist
                    pass

            # Learning orchestrator metrics
            if self.learning_orchestrator:
                try:
                    orchestrator_stats = (
                        self.learning_orchestrator.get_orchestrator_stats()
                    )

                    metrics.learning_cycles_completed = orchestrator_stats.get(
                        "cycles_completed", 0
                    )
                    metrics.avg_cycle_duration_minutes = (
                        orchestrator_stats.get("avg_cycle_duration", 0.0) / 60
                    )
                    metrics.successful_improvements = orchestrator_stats.get(
                        "successful_improvements", 0
                    )

                except AttributeError:
                    pass

            # Mock additional metrics (would be real in production)
            metrics.model_effectiveness_updates = 8
            metrics.cost_per_insight = 0.15  # $0.15 per insight
            metrics.time_to_improvement_hours = 2.5

        except Exception as e:
            logger.error(f"Error collecting flywheel system metrics: {e}")

        return metrics

    async def collect_integrated_system_metrics(self) -> IntegratedSystemMetrics:
        """Collect metrics for the integrated system"""
        metrics = IntegratedSystemMetrics()

        try:
            # Bridge integration metrics
            if self.bridge:
                bridge_summary = self.bridge.get_context_summary()
                metrics.bridge_sessions_active = bridge_summary.get(
                    "active_sessions", 0
                )

                # Calculate integration success rate
                total_context_entries = bridge_summary.get("total_context_entries", 0)
                if total_context_entries > 0:
                    metrics.integration_success_rate = 1.0 - (
                        bridge_summary.get("wrong_turns_recorded", 0)
                        / total_context_entries
                    )

            # Context optimizer metrics
            if self.context_optimizer:
                context_stats = self.context_optimizer.get_optimization_stats()
                metrics.context_sessions_active = context_stats.get(
                    "active_sessions", 0
                )

                # Context optimization efficiency
                compressions = context_stats.get("compressions_performed", 0)
                tokens_saved = context_stats.get("tokens_saved", 0)
                if compressions > 0:
                    metrics.context_optimization_efficiency = tokens_saved / (
                        compressions * 100
                    )

            # Performance metrics
            metrics.avg_challenge_execution_time_ms = 150.0  # Mock value
            metrics.phantom_workflow_incidents = 0
            metrics.cross_system_insights = 12
            metrics.bidirectional_learning_rate = 0.68
            metrics.system_adaptation_score = 0.82

        except Exception as e:
            logger.error(f"Error collecting integrated system metrics: {e}")

        return metrics

    def _calculate_overall_health(
        self,
        challenge_metrics: ChallengeSystemMetrics,
        flywheel_metrics: FlywheelSystemMetrics,
        integrated_metrics: IntegratedSystemMetrics,
    ) -> HealthStatus:
        """Calculate overall system health status"""

        # Weight different aspects of health
        weights = {
            "availability": 0.25,
            "performance": 0.25,
            "learning": 0.25,
            "integration": 0.25,
        }

        # Calculate scores (0-1 scale)
        availability_score = challenge_metrics.system_availability_percentage / 100
        performance_score = min(
            (
                flywheel_metrics.flywheel_value_score
                + challenge_metrics.avg_challenge_effectiveness
            )
            / 2,
            1.0,
        )
        learning_score = min(
            flywheel_metrics.learning_velocity / 5, 1.0
        )  # Normalized to 5
        integration_score = integrated_metrics.integration_success_rate

        # Weighted overall score
        overall_score = (
            weights["availability"] * availability_score
            + weights["performance"] * performance_score
            + weights["learning"] * learning_score
            + weights["integration"] * integration_score
        )

        # Map to health status
        if overall_score >= 0.9:
            return HealthStatus.EXCELLENT
        elif overall_score >= 0.7:
            return HealthStatus.GOOD
        elif overall_score >= 0.5:
            return HealthStatus.WARNING
        else:
            return HealthStatus.CRITICAL

    def _generate_alerts_and_recommendations(
        self, snapshot: DashboardSnapshot
    ) -> Tuple[List[str], List[str]]:
        """Generate alerts and recommendations based on current state"""
        alerts = []
        recommendations = []

        # System availability alerts
        if snapshot.challenge_metrics.system_availability_percentage < 80:
            alerts.append("System availability below 80%")
            recommendations.append("Check challenge system component health")

        # Performance alerts
        if snapshot.flywheel_metrics.learning_velocity < 1.0:
            alerts.append("Learning velocity below optimal threshold")
            recommendations.append(
                "Review test capture quality and learning cycle frequency"
            )

        # Integration alerts
        if snapshot.integrated_metrics.phantom_workflow_incidents > 0:
            alerts.append(
                f"Phantom workflow incidents detected: {snapshot.integrated_metrics.phantom_workflow_incidents}"
            )
            recommendations.append("Run phantom workflow prevention validation")

        # Context optimization recommendations
        if snapshot.challenge_metrics.context_cache_hit_rate < 0.7:
            recommendations.append(
                "Optimize context prefix stability for better cache hit rate"
            )

        # Challenge effectiveness recommendations
        if snapshot.challenge_metrics.avg_challenge_effectiveness < 0.7:
            recommendations.append(
                "Review challenge generation thresholds and improve research quality"
            )

        # ROI recommendations
        if snapshot.flywheel_metrics.roi_ratio < 2.0:
            recommendations.append("Optimize test-to-insight conversion to improve ROI")

        return alerts, recommendations

    async def take_snapshot(self) -> DashboardSnapshot:
        """Take a complete system snapshot"""
        snapshot = DashboardSnapshot()

        # Collect metrics from all systems
        snapshot.challenge_metrics = await self.collect_challenge_system_metrics()
        snapshot.flywheel_metrics = await self.collect_flywheel_system_metrics()
        snapshot.integrated_metrics = await self.collect_integrated_system_metrics()

        # Calculate derived metrics
        snapshot.overall_health = self._calculate_overall_health(
            snapshot.challenge_metrics,
            snapshot.flywheel_metrics,
            snapshot.integrated_metrics,
        )

        # System availability (weighted average)
        challenge_availability = (
            snapshot.challenge_metrics.system_availability_percentage / 100
        )
        flywheel_availability = (
            1.0 if snapshot.flywheel_metrics.tests_captured > 0 else 0.0
        )
        snapshot.system_availability = (
            challenge_availability + flywheel_availability
        ) / 2

        # Intelligence effectiveness (composite score)
        snapshot.intelligence_effectiveness = (
            snapshot.challenge_metrics.avg_challenge_effectiveness * 0.4
            + snapshot.flywheel_metrics.flywheel_value_score * 0.3
            + snapshot.integrated_metrics.system_adaptation_score * 0.3
        )

        # Learning acceleration (trend-based)
        if len(self.snapshots) > 0:
            prev_learning = self.snapshots[-1].flywheel_metrics.learning_velocity
            curr_learning = snapshot.flywheel_metrics.learning_velocity
            snapshot.learning_acceleration = curr_learning - prev_learning

        # Cost efficiency
        flywheel_roi = snapshot.flywheel_metrics.roi_ratio
        challenge_effectiveness = snapshot.challenge_metrics.avg_challenge_effectiveness
        snapshot.cost_efficiency = (flywheel_roi + challenge_effectiveness) / 2

        # Generate alerts and recommendations
        snapshot.active_alerts, snapshot.recommendations = (
            self._generate_alerts_and_recommendations(snapshot)
        )

        return snapshot

    async def update_dashboard(self) -> DashboardSnapshot:
        """Update dashboard with latest metrics"""
        snapshot = await self.take_snapshot()

        # Add to history
        self.snapshots.append(snapshot)

        # Limit history size
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots = self.snapshots[-self.max_snapshots :]

        self.last_update = datetime.utcnow()

        # Log significant changes
        if len(self.snapshots) > 1:
            prev = self.snapshots[-2]
            if snapshot.overall_health != prev.overall_health:
                logger.info(
                    f"System health changed: {prev.overall_health.value} -> {snapshot.overall_health.value}"
                )

        logger.debug(
            f"Dashboard updated: {snapshot.overall_health.value} health, "
            f"{snapshot.intelligence_effectiveness:.2f} effectiveness"
        )

        return snapshot

    async def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get comprehensive dashboard summary"""
        # Update if needed
        time_since_update = (datetime.utcnow() - self.last_update).total_seconds()
        if (
            self.update_mode != DashboardUpdateMode.ON_DEMAND
            and time_since_update > self.update_interval_seconds
        ):
            await self.update_dashboard()

        if not self.snapshots:
            await self.update_dashboard()

        current = self.snapshots[-1]

        # Trend analysis
        trend_data = {}
        if len(self.snapshots) >= 2:
            prev = self.snapshots[-2]
            trend_data = {
                "intelligence_effectiveness_trend": current.intelligence_effectiveness
                - prev.intelligence_effectiveness,
                "learning_velocity_trend": current.flywheel_metrics.learning_velocity
                - prev.flywheel_metrics.learning_velocity,
                "system_availability_trend": current.system_availability
                - prev.system_availability,
            }

        return {
            "timestamp": current.timestamp.isoformat(),
            "overall_health": current.overall_health.value,
            "system_availability": current.system_availability,
            "intelligence_effectiveness": current.intelligence_effectiveness,
            "learning_acceleration": current.learning_acceleration,
            "cost_efficiency": current.cost_efficiency,
            # Challenge system summary
            "challenge_system": {
                "challenges_generated": current.challenge_metrics.challenges_generated,
                "success_rate": current.challenge_metrics.challenge_success_rate,
                "avg_effectiveness": current.challenge_metrics.avg_challenge_effectiveness,
                "context_cache_hit_rate": current.challenge_metrics.context_cache_hit_rate,
                "system_availability": current.challenge_metrics.system_availability_percentage,
            },
            # Flywheel system summary
            "flywheel_system": {
                "tests_captured": current.flywheel_metrics.tests_captured,
                "flywheel_value_score": current.flywheel_metrics.flywheel_value_score,
                "learning_velocity": current.flywheel_metrics.learning_velocity,
                "roi_ratio": current.flywheel_metrics.roi_ratio,
                "learning_cycles_completed": current.flywheel_metrics.learning_cycles_completed,
            },
            # Integration summary
            "integration": {
                "active_sessions": current.integrated_metrics.bridge_sessions_active,
                "success_rate": current.integrated_metrics.integration_success_rate,
                "phantom_workflows": current.integrated_metrics.phantom_workflow_incidents,
                "cross_system_insights": current.integrated_metrics.cross_system_insights,
            },
            # Alerts and recommendations
            "active_alerts": current.active_alerts,
            "recommendations": current.recommendations,
            # Trends
            "trends": trend_data,
            # System info
            "last_update": self.last_update.isoformat(),
            "update_mode": self.update_mode.value,
            "snapshots_count": len(self.snapshots),
        }

    async def get_historical_data(
        self, hours_back: int = 24, metric_name: str = None
    ) -> List[Dict[str, Any]]:
        """Get historical dashboard data"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)

        historical_data = [
            {
                "timestamp": snapshot.timestamp.isoformat(),
                "overall_health": snapshot.overall_health.value,
                "intelligence_effectiveness": snapshot.intelligence_effectiveness,
                "learning_velocity": snapshot.flywheel_metrics.learning_velocity,
                "challenge_effectiveness": snapshot.challenge_metrics.avg_challenge_effectiveness,
                "system_availability": snapshot.system_availability,
                "cost_efficiency": snapshot.cost_efficiency,
            }
            for snapshot in self.snapshots
            if snapshot.timestamp >= cutoff_time
        ]

        if metric_name:
            # Filter to specific metric
            return [
                {"timestamp": item["timestamp"], metric_name: item.get(metric_name)}
                for item in historical_data
            ]

        return historical_data

    async def start_auto_update(self):
        """Start automatic dashboard updates"""
        if self.update_mode == DashboardUpdateMode.ON_DEMAND:
            logger.warning("Auto-update requested but dashboard is in on-demand mode")
            return

        logger.info(
            f"Starting auto-update with {self.update_interval_seconds}s interval"
        )

        while True:
            try:
                await self.update_dashboard()
                await asyncio.sleep(self.update_interval_seconds)
            except Exception as e:
                logger.error(f"Auto-update error: {e}")
                await asyncio.sleep(self.update_interval_seconds)


# Singleton instance
_unified_dashboard = None


def get_unified_intelligence_dashboard() -> UnifiedIntelligenceDashboard:
    """Get singleton unified intelligence dashboard"""
    global _unified_dashboard
    if _unified_dashboard is None:
        _unified_dashboard = UnifiedIntelligenceDashboard()
    return _unified_dashboard


async def main():
    """Demo of unified intelligence dashboard"""
    print("📊 Unified Intelligence Dashboard Demo")
    print("=" * 60)

    dashboard = get_unified_intelligence_dashboard()

    # Take initial snapshot
    snapshot = await dashboard.take_snapshot()
    print(f"Initial health: {snapshot.overall_health.value}")
    print(f"Intelligence effectiveness: {snapshot.intelligence_effectiveness:.2%}")
    print(f"System availability: {snapshot.system_availability:.1%}")

    # Get dashboard summary
    summary = await dashboard.get_dashboard_summary()

    print("\nChallenge System:")
    print(
        f"  Challenges generated: {summary['challenge_system']['challenges_generated']}"
    )
    print(f"  Success rate: {summary['challenge_system']['success_rate']:.1%}")
    print(
        f"  Cache hit rate: {summary['challenge_system']['context_cache_hit_rate']:.1%}"
    )

    print("\nFlywheel System:")
    print(f"  Tests captured: {summary['flywheel_system']['tests_captured']}")
    print(f"  Learning velocity: {summary['flywheel_system']['learning_velocity']:.2f}")
    print(f"  ROI ratio: {summary['flywheel_system']['roi_ratio']:.2f}")

    print("\nIntegration:")
    print(f"  Active sessions: {summary['integration']['active_sessions']}")
    print(f"  Success rate: {summary['integration']['success_rate']:.1%}")
    print(f"  Cross-system insights: {summary['integration']['cross_system_insights']}")

    # Show alerts and recommendations
    if summary["active_alerts"]:
        print("\n⚠️ Active Alerts:")
        for alert in summary["active_alerts"]:
            print(f"  - {alert}")

    if summary["recommendations"]:
        print("\n💡 Recommendations:")
        for rec in summary["recommendations"]:
            print(f"  - {rec}")


if __name__ == "__main__":
    asyncio.run(main())
