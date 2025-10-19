"""
USER-FACING ANALYTICS SERVICE
=============================

Comprehensive analytics and insights service for METIS V5 user-facing dashboards.
Provides real-time metrics, performance insights, and business intelligence.

Part of V5 Support Systems Integration - delivering actionable analytics to users.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import statistics
from collections import defaultdict, deque


class AnalyticsMetricType(Enum):
    """Types of analytics metrics"""

    PERFORMANCE = "performance"
    USAGE = "usage"
    QUALITY = "quality"
    BUSINESS = "business"
    TECHNICAL = "technical"
    USER_EXPERIENCE = "user_experience"


class TimeWindow(Enum):
    """Time windows for analytics aggregation"""

    REAL_TIME = "real_time"
    LAST_HOUR = "last_hour"
    LAST_24_HOURS = "last_24_hours"
    LAST_7_DAYS = "last_7_days"
    LAST_30_DAYS = "last_30_days"
    CUSTOM = "custom"


@dataclass
class AnalyticsDataPoint:
    """Single analytics data point"""

    metric_name: str
    metric_type: AnalyticsMetricType
    value: Union[int, float, str, bool]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


@dataclass
class AnalyticsDashboardWidget:
    """Configuration for a dashboard widget"""

    widget_id: str
    widget_type: str  # chart, metric, table, etc.
    title: str
    metrics: List[str]
    time_window: TimeWindow
    visualization_config: Dict[str, Any]
    refresh_interval_seconds: int = 60
    priority: int = 1


@dataclass
class UserAnalyticsReport:
    """Comprehensive analytics report for users"""

    report_id: str
    user_id: str
    report_type: str
    time_period: Dict[str, datetime]
    metrics_summary: Dict[str, Any]
    insights: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    visualizations: List[Dict[str, Any]]
    generated_at: datetime = field(default_factory=datetime.now)


class UserAnalyticsService:
    """
    User-Facing Analytics Service for METIS V5.

    Provides:
    - Real-time performance dashboards
    - Business intelligence and insights
    - Usage analytics and trends
    - Quality metrics and monitoring
    - Custom analytics reports
    - Interactive visualizations
    - Predictive analytics
    """

    def __init__(self):
        self.service_id = "user_analytics_service"
        self.version = "1.0.0"
        self.status = "active"

        # Analytics data storage (in production, would use proper database)
        self.metrics_buffer = deque(
            maxlen=10000
        )  # Rolling buffer for real-time metrics
        self.aggregated_metrics = defaultdict(
            dict
        )  # Pre-aggregated metrics by time window
        self.user_dashboards = {}  # User-specific dashboard configurations

        # Analytics processing
        self.metric_processors = {}
        self.insight_generators = {}
        self.recommendation_engines = {}

        # Service state
        self.active_widgets = {}
        self.analytics_sessions = {}

        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"UserAnalyticsService initialized - {self.service_id} v{self.version}"
        )

        # Initialize default analytics processors
        self._initialize_metric_processors()
        self._initialize_insight_generators()

    async def record_analytics_event(self, event_data: Dict[str, Any]) -> bool:
        """
        Record an analytics event from V5 services.

        Args:
            event_data: Event data including metrics, context, and metadata

        Returns:
            Success status
        """
        try:
            # Extract metrics from event data
            metrics = event_data.get("metrics", {})
            event_type = event_data.get("event_type", "unknown")
            user_id = event_data.get("user_id", "anonymous")
            service_name = event_data.get("service_name", "unknown")

            timestamp = datetime.now()

            # Process each metric in the event
            for metric_name, metric_value in metrics.items():
                # Determine metric type
                metric_type = self._classify_metric_type(metric_name, metric_value)

                # Create data point
                data_point = AnalyticsDataPoint(
                    metric_name=metric_name,
                    metric_type=metric_type,
                    value=metric_value,
                    timestamp=timestamp,
                    metadata={
                        "event_type": event_type,
                        "user_id": user_id,
                        "service_name": service_name,
                        **event_data.get("metadata", {}),
                    },
                    tags=event_data.get("tags", []),
                )

                # Add to metrics buffer
                self.metrics_buffer.append(data_point)

                # Process metric for real-time analytics
                await self._process_real_time_metric(data_point)

            # Trigger analytics processing
            await self._update_aggregated_metrics()

            return True

        except Exception as e:
            self.logger.error(f"Error recording analytics event: {e}")
            return False

    async def get_user_dashboard(
        self, user_id: str, dashboard_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate user-specific analytics dashboard.

        Args:
            user_id: User identifier
            dashboard_config: Dashboard configuration and preferences

        Returns:
            Complete dashboard data with widgets and visualizations
        """
        try:
            self.logger.info(f"Generating analytics dashboard for user: {user_id}")

            # Step 1: Get or create dashboard configuration
            if user_id not in self.user_dashboards:
                self.user_dashboards[user_id] = self._create_default_dashboard_config(
                    user_id
                )

            user_dashboard_config = self.user_dashboards[user_id]

            # Merge with provided config
            for key, value in dashboard_config.items():
                user_dashboard_config[key] = value

            # Step 2: Generate widgets data
            dashboard_widgets = []

            for widget_config in user_dashboard_config.get("widgets", []):
                widget_data = await self._generate_widget_data(widget_config, user_id)
                dashboard_widgets.append(widget_data)

            # Step 3: Generate summary metrics
            summary_metrics = await self._generate_summary_metrics(user_id)

            # Step 4: Generate insights and recommendations
            insights = await self._generate_user_insights(user_id)
            recommendations = await self._generate_user_recommendations(user_id)

            # Step 5: Create complete dashboard
            dashboard = {
                "dashboard_id": f"dashboard_{user_id}_{int(datetime.now().timestamp())}",
                "user_id": user_id,
                "dashboard_config": user_dashboard_config,
                "widgets": dashboard_widgets,
                "summary_metrics": summary_metrics,
                "insights": insights,
                "recommendations": recommendations,
                "last_updated": datetime.now(),
                "refresh_interval": dashboard_config.get("refresh_interval", 60),
            }

            self.logger.info(
                f"Dashboard generated for user {user_id}: {len(dashboard_widgets)} widgets"
            )

            return dashboard

        except Exception as e:
            self.logger.error(f"Error generating user dashboard: {e}")
            raise

    async def generate_analytics_report(
        self, report_config: Dict[str, Any]
    ) -> UserAnalyticsReport:
        """
        Generate comprehensive analytics report.

        Args:
            report_config: Report configuration including metrics, time period, format

        Returns:
            Complete analytics report with insights and recommendations
        """
        try:
            user_id = report_config.get("user_id", "system")
            report_type = report_config.get("report_type", "comprehensive")
            time_period = report_config.get("time_period", {})

            self.logger.info(
                f"Generating analytics report: {report_type} for user {user_id}"
            )

            # Step 1: Define time period
            if not time_period:
                end_time = datetime.now()
                start_time = end_time - timedelta(days=30)  # Default 30 days
                time_period = {"start": start_time, "end": end_time}

            # Step 2: Aggregate metrics for time period
            metrics_summary = await self._aggregate_metrics_for_period(time_period)

            # Step 3: Generate insights
            insights = await self._generate_period_insights(
                metrics_summary, time_period
            )

            # Step 4: Generate recommendations
            recommendations = await self._generate_period_recommendations(
                metrics_summary, insights
            )

            # Step 5: Create visualizations
            visualizations = await self._create_report_visualizations(
                metrics_summary, report_config.get("visualization_types", [])
            )

            # Step 6: Create report
            report = UserAnalyticsReport(
                report_id=f"report_{user_id}_{int(datetime.now().timestamp())}",
                user_id=user_id,
                report_type=report_type,
                time_period=time_period,
                metrics_summary=metrics_summary,
                insights=insights,
                recommendations=recommendations,
                visualizations=visualizations,
            )

            self.logger.info(f"Analytics report generated: {report.report_id}")

            return report

        except Exception as e:
            self.logger.error(f"Error generating analytics report: {e}")
            raise

    async def get_real_time_metrics(
        self, metric_names: Optional[List[str]] = None, time_window_minutes: int = 60
    ) -> Dict[str, Any]:
        """
        Get real-time analytics metrics.

        Args:
            metric_names: Specific metrics to retrieve (None for all)
            time_window_minutes: Time window for real-time data

        Returns:
            Real-time metrics data
        """
        try:
            cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)

            # Filter recent data points
            recent_metrics = [
                dp
                for dp in self.metrics_buffer
                if dp.timestamp >= cutoff_time
                and (not metric_names or dp.metric_name in metric_names)
            ]

            # Aggregate by metric name
            metrics_data = defaultdict(list)
            for dp in recent_metrics:
                metrics_data[dp.metric_name].append(
                    {
                        "value": dp.value,
                        "timestamp": dp.timestamp,
                        "metadata": dp.metadata,
                    }
                )

            # Calculate real-time statistics
            real_time_stats = {}
            for metric_name, data_points in metrics_data.items():
                if data_points:
                    values = [
                        dp["value"]
                        for dp in data_points
                        if isinstance(dp["value"], (int, float))
                    ]

                    if values:
                        real_time_stats[metric_name] = {
                            "current_value": values[-1] if values else None,
                            "avg_value": statistics.mean(values),
                            "min_value": min(values),
                            "max_value": max(values),
                            "data_points_count": len(values),
                            "trend": self._calculate_trend(values),
                            "last_updated": max(dp["timestamp"] for dp in data_points),
                        }

            return {
                "time_window_minutes": time_window_minutes,
                "metrics_count": len(real_time_stats),
                "metrics": real_time_stats,
                "generated_at": datetime.now(),
            }

        except Exception as e:
            self.logger.error(f"Error getting real-time metrics: {e}")
            raise

    async def track_user_engagement(
        self, engagement_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Track user engagement analytics.

        Args:
            engagement_data: User engagement event data

        Returns:
            Engagement analytics summary
        """
        try:
            user_id = engagement_data.get("user_id", "anonymous")
            engagement_type = engagement_data.get("engagement_type", "unknown")

            # Record engagement metrics
            engagement_metrics = {
                "user_id": user_id,
                "engagement_type": engagement_type,
                "session_duration": engagement_data.get("session_duration", 0),
                "interactions_count": engagement_data.get("interactions_count", 1),
                "features_used": len(engagement_data.get("features_used", [])),
                "satisfaction_score": engagement_data.get("satisfaction_score", 0),
                "timestamp": datetime.now(),
            }

            # Process engagement analytics
            await self.record_analytics_event(
                {
                    "event_type": "user_engagement",
                    "user_id": user_id,
                    "metrics": engagement_metrics,
                    "metadata": engagement_data,
                }
            )

            # Generate engagement insights
            engagement_insights = await self._analyze_user_engagement(
                user_id, engagement_data
            )

            return {
                "engagement_recorded": True,
                "user_id": user_id,
                "engagement_insights": engagement_insights,
                "processed_at": datetime.now(),
            }

        except Exception as e:
            self.logger.error(f"Error tracking user engagement: {e}")
            raise

    async def get_system_health_analytics(self) -> Dict[str, Any]:
        """
        Get system health analytics for operational monitoring.

        Returns:
            System health analytics and metrics
        """
        try:
            # Collect system metrics from recent data
            system_metrics = await self._collect_system_health_metrics()

            # Analyze performance trends
            performance_trends = await self._analyze_performance_trends()

            # Generate health score
            health_score = await self._calculate_system_health_score(system_metrics)

            # Create health analytics
            health_analytics = {
                "overall_health_score": health_score,
                "system_metrics": system_metrics,
                "performance_trends": performance_trends,
                "alerts": await self._check_health_alerts(system_metrics),
                "recommendations": await self._generate_health_recommendations(
                    system_metrics
                ),
                "generated_at": datetime.now(),
            }

            return health_analytics

        except Exception as e:
            self.logger.error(f"Error getting system health analytics: {e}")
            raise

    async def get_analytics_service_health(self) -> Dict[str, Any]:
        """
        Get health status of the analytics service itself.

        Returns:
            Service health status and metrics
        """
        try:
            # Calculate service metrics
            buffer_utilization = (
                len(self.metrics_buffer) / self.metrics_buffer.maxlen * 100
            )
            active_dashboards = len(self.user_dashboards)
            processing_rate = len(self.metrics_buffer) / max(
                1, 3600
            )  # metrics per second (approximate)

            # Health factors
            health_factors = {
                "buffer_utilization": buffer_utilization,
                "active_dashboards": min(100, active_dashboards * 10),  # Normalize
                "processing_efficiency": min(100, processing_rate * 10),  # Normalize
                "service_availability": 100,  # Simplified - service is running
                "data_freshness": 100 if self.metrics_buffer else 0,
            }

            overall_health_score = statistics.mean(health_factors.values())

            health_status = {
                "service_id": self.service_id,
                "version": self.version,
                "status": (
                    "healthy"
                    if overall_health_score >= 80
                    else "degraded" if overall_health_score >= 60 else "unhealthy"
                ),
                "overall_health_score": overall_health_score,
                "health_factors": health_factors,
                "metrics_buffer_size": len(self.metrics_buffer),
                "metrics_buffer_max": self.metrics_buffer.maxlen,
                "active_dashboards_count": active_dashboards,
                "active_widgets_count": len(self.active_widgets),
                "analytics_sessions_count": len(self.analytics_sessions),
                "last_health_check": datetime.now(),
            }

            return health_status

        except Exception as e:
            self.logger.error(f"Error getting analytics service health: {e}")
            return {
                "service_id": self.service_id,
                "status": "error",
                "error": str(e),
                "last_health_check": datetime.now(),
            }

    def _classify_metric_type(
        self, metric_name: str, metric_value: Any
    ) -> AnalyticsMetricType:
        """Classify metric type based on name and value."""
        metric_name_lower = metric_name.lower()

        if any(
            keyword in metric_name_lower
            for keyword in ["response_time", "latency", "duration", "performance"]
        ):
            return AnalyticsMetricType.PERFORMANCE
        elif any(
            keyword in metric_name_lower
            for keyword in ["usage", "count", "frequency", "volume"]
        ):
            return AnalyticsMetricType.USAGE
        elif any(
            keyword in metric_name_lower
            for keyword in ["quality", "accuracy", "score", "rating"]
        ):
            return AnalyticsMetricType.QUALITY
        elif any(
            keyword in metric_name_lower
            for keyword in ["revenue", "cost", "business", "profit"]
        ):
            return AnalyticsMetricType.BUSINESS
        elif any(
            keyword in metric_name_lower
            for keyword in ["error", "cpu", "memory", "system"]
        ):
            return AnalyticsMetricType.TECHNICAL
        else:
            return AnalyticsMetricType.USER_EXPERIENCE

    async def _process_real_time_metric(self, data_point: AnalyticsDataPoint):
        """Process metric for real-time analytics."""
        try:
            # Apply metric processors if available
            if data_point.metric_name in self.metric_processors:
                processor = self.metric_processors[data_point.metric_name]
                await processor(data_point)

            # Update real-time aggregations
            await self._update_real_time_aggregations(data_point)

        except Exception as e:
            self.logger.error(f"Error processing real-time metric: {e}")

    async def _update_aggregated_metrics(self):
        """Update pre-aggregated metrics for different time windows."""
        # This would implement time-window based aggregation
        # For brevity, keeping minimal implementation
        pass

    def _create_default_dashboard_config(self, user_id: str) -> Dict[str, Any]:
        """Create default dashboard configuration for a user."""
        return {
            "dashboard_name": f"METIS V5 Analytics - {user_id}",
            "widgets": [
                {
                    "widget_id": "performance_overview",
                    "widget_type": "metrics_grid",
                    "title": "Performance Overview",
                    "metrics": ["response_time", "throughput", "error_rate"],
                    "time_window": TimeWindow.LAST_24_HOURS.value,
                },
                {
                    "widget_id": "usage_trends",
                    "widget_type": "line_chart",
                    "title": "Usage Trends",
                    "metrics": ["usage_count", "active_users"],
                    "time_window": TimeWindow.LAST_7_DAYS.value,
                },
                {
                    "widget_id": "quality_metrics",
                    "widget_type": "gauge_chart",
                    "title": "Quality Metrics",
                    "metrics": ["quality_score", "satisfaction_score"],
                    "time_window": TimeWindow.LAST_24_HOURS.value,
                },
            ],
            "refresh_interval": 60,
            "theme": "dark",
        }

    async def _generate_widget_data(
        self, widget_config: Dict[str, Any], user_id: str
    ) -> Dict[str, Any]:
        """Generate data for a specific dashboard widget."""
        # Simplified widget data generation
        return {
            "widget_id": widget_config["widget_id"],
            "widget_type": widget_config["widget_type"],
            "title": widget_config["title"],
            "data": {"placeholder": "Widget data would be generated here"},
            "last_updated": datetime.now(),
        }

    def _calculate_trend(self, values: List[Union[int, float]]) -> str:
        """Calculate trend direction from a list of values."""
        if len(values) < 2:
            return "stable"

        # Simple trend calculation
        recent_avg = statistics.mean(values[-3:]) if len(values) >= 3 else values[-1]
        older_avg = statistics.mean(values[:-3]) if len(values) >= 6 else values[0]

        if recent_avg > older_avg * 1.05:
            return "increasing"
        elif recent_avg < older_avg * 0.95:
            return "decreasing"
        else:
            return "stable"

    def _initialize_metric_processors(self):
        """Initialize metric processors for different metric types."""
        # Placeholder for metric processors
        pass

    def _initialize_insight_generators(self):
        """Initialize insight generators for analytics."""
        # Placeholder for insight generators
        pass

    # Additional helper methods would continue here...
    # For brevity, implementing core functionality above


# Global service instance for dependency injection
_user_analytics_service_instance = None


def get_user_analytics_service() -> UserAnalyticsService:
    """Get global User Analytics Service instance."""
    global _user_analytics_service_instance

    if _user_analytics_service_instance is None:
        _user_analytics_service_instance = UserAnalyticsService()

    return _user_analytics_service_instance


# Service metadata for integration
ANALYTICS_SERVICE_INFO = {
    "service_name": "UserAnalyticsService",
    "service_type": "user_facing_analytics",
    "capabilities": [
        "real_time_dashboards",
        "business_intelligence",
        "performance_analytics",
        "usage_tracking",
        "quality_metrics",
        "custom_reports",
        "predictive_insights",
        "user_engagement_analytics",
    ],
    "widget_types": [
        "metrics_grid",
        "line_chart",
        "bar_chart",
        "gauge_chart",
        "heatmap",
        "table_view",
        "trend_analysis",
    ],
    "time_windows": [
        "real_time",
        "last_hour",
        "last_24_hours",
        "last_7_days",
        "last_30_days",
        "custom",
    ],
}
