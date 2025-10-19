"""
METIS Performance Monitoring Service
Part of Application Services Cluster - Focused on model performance tracking and analytics

Extracted from model_manager.py during Phase 5.3 decomposition.
Single Responsibility: Track, analyze, and report model performance metrics.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict
import asyncio
import statistics

from src.services.contracts.application_contracts import (
    IPerformanceMonitoringService,
    PerformanceMetricsContract,
    PerformanceMetricType,
)


class PerformanceMonitoringService(IPerformanceMonitoringService):
    """
    Focused service for comprehensive model performance monitoring
    Clean extraction from model_manager.py performance tracking methods
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Performance data storage
        self.performance_metrics: Dict[str, List[PerformanceMetricsContract]] = (
            defaultdict(list)
        )
        self.performance_baselines: Dict[str, Dict[str, float]] = {}
        self.performance_trends: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        # Monitoring configuration
        self.monitoring_config = {
            "metric_retention_days": 30,
            "baseline_calculation_window_days": 7,
            "trend_analysis_window_hours": 24,
            "alert_thresholds": {
                PerformanceMetricType.ACCURACY_SCORE: {"critical": 0.5, "warning": 0.7},
                PerformanceMetricType.RESPONSE_TIME: {
                    "critical": 10000,
                    "warning": 5000,
                },  # milliseconds
                PerformanceMetricType.CONFIDENCE_LEVEL: {
                    "critical": 0.4,
                    "warning": 0.6,
                },
                PerformanceMetricType.COHERENCE_SCORE: {
                    "critical": 0.5,
                    "warning": 0.7,
                },
                PerformanceMetricType.RELEVANCE_SCORE: {
                    "critical": 0.5,
                    "warning": 0.7,
                },
                PerformanceMetricType.COMPLETENESS_SCORE: {
                    "critical": 0.5,
                    "warning": 0.7,
                },
            },
            "performance_alert_cooldown_minutes": 30,
            "baseline_update_frequency_hours": 6,
        }

        # Performance analytics cache
        self.analytics_cache = {}
        self.last_cache_update = {}
        self.cache_ttl_minutes = 10

        # Alert tracking
        self.active_alerts: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.alert_history: List[Dict[str, Any]] = []

        # Flags for background task management
        self._background_tasks_started = False

        self.logger.info("📊 PerformanceMonitoringService initialized")

    async def record_performance_metric(
        self,
        model_id: str,
        metric_type: PerformanceMetricType,
        metric_value: float,
        context: Dict[str, Any],
    ) -> PerformanceMetricsContract:
        """
        Core service method: Record a performance metric for a model
        Comprehensive metric recording with trend analysis and alerting
        """
        try:
            # Calculate baseline comparison
            baseline_comparison = await self._calculate_baseline_comparison(
                model_id, metric_type, metric_value
            )

            # Determine trend direction
            trend_direction = await self._determine_trend_direction(
                model_id, metric_type, metric_value
            )

            # Create performance metric contract
            metric_contract = PerformanceMetricsContract(
                metric_id=f"{model_id}_{metric_type.value}_{datetime.utcnow().timestamp()}",
                model_id=model_id,
                engagement_id=context.get("engagement_id", "unknown"),
                metric_type=metric_type,
                metric_value=metric_value,
                baseline_comparison=baseline_comparison,
                trend_direction=trend_direction,
                context_metadata=context,
                measurement_timestamp=datetime.utcnow(),
                service_version="v5_modular",
            )

            # Store metric
            self.performance_metrics[model_id].append(metric_contract)

            # Update trend data
            self.performance_trends[model_id][metric_type.value].append(metric_value)

            # Cleanup old metrics
            await self._cleanup_old_metrics(model_id)

            # Check for performance alerts
            await self._check_performance_alerts(
                model_id, metric_type, metric_value, baseline_comparison
            )

            # Invalidate analytics cache
            self._invalidate_analytics_cache(model_id)

            self.logger.debug(
                f"📊 Metric recorded: {model_id} - {metric_type.value}: {metric_value}"
            )
            return metric_contract

        except Exception as e:
            self.logger.error(
                f"❌ Metric recording failed: {model_id} - {metric_type.value}: {e}"
            )

            # Create fallback metric contract
            return PerformanceMetricsContract(
                metric_id=f"error_{datetime.utcnow().timestamp()}",
                model_id=model_id,
                engagement_id="error",
                metric_type=metric_type,
                metric_value=metric_value,
                baseline_comparison=0.0,
                trend_direction="unknown",
                context_metadata={"error": str(e)},
                measurement_timestamp=datetime.utcnow(),
                service_version="v5_modular_error",
            )

    async def get_performance_summary(
        self, model_id: str, time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Core service method: Get comprehensive performance summary for a model
        Detailed performance analysis with trends and comparisons
        """
        try:
            # Check cache first
            cache_key = f"{model_id}_{time_window_hours}"

            if self._is_cache_valid(cache_key):
                return self.analytics_cache[cache_key]

            # Calculate time window
            window_start = datetime.utcnow() - timedelta(hours=time_window_hours)

            # Get metrics within time window
            model_metrics = [
                metric
                for metric in self.performance_metrics[model_id]
                if metric.measurement_timestamp >= window_start
            ]

            if not model_metrics:
                return {
                    "model_id": model_id,
                    "time_window_hours": time_window_hours,
                    "metrics_count": 0,
                    "message": "No metrics found for specified time window",
                }

            # Group metrics by type
            metrics_by_type = defaultdict(list)
            for metric in model_metrics:
                metrics_by_type[metric.metric_type].append(metric)

            # Calculate summary statistics for each metric type
            metric_summaries = {}
            for metric_type, metrics in metrics_by_type.items():
                values = [m.metric_value for m in metrics]
                baseline_comparisons = [m.baseline_comparison for m in metrics]

                metric_summaries[metric_type.value] = {
                    "count": len(values),
                    "latest_value": values[-1],
                    "average": statistics.mean(values),
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values),
                    "std_deviation": (
                        statistics.stdev(values) if len(values) > 1 else 0.0
                    ),
                    "trend": self._calculate_simple_trend(values),
                    "baseline_comparison": {
                        "average": statistics.mean(baseline_comparisons),
                        "latest": baseline_comparisons[-1],
                    },
                    "alert_status": self._get_alert_status(
                        model_id, metric_type, values[-1]
                    ),
                }

            # Calculate overall performance score
            overall_score = await self._calculate_overall_performance_score(
                metric_summaries
            )

            # Generate performance insights
            insights = await self._generate_performance_insights(
                model_id, metric_summaries
            )

            # Create summary
            summary = {
                "model_id": model_id,
                "time_window_hours": time_window_hours,
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "metrics_count": len(model_metrics),
                "metric_summaries": metric_summaries,
                "overall_performance_score": overall_score,
                "performance_insights": insights,
                "active_alerts": len(self.active_alerts.get(model_id, [])),
                "baseline_info": self.performance_baselines.get(model_id, {}),
                "summary_generated": datetime.utcnow().isoformat(),
            }

            # Cache the result
            self.analytics_cache[cache_key] = summary
            self.last_cache_update[cache_key] = datetime.utcnow()

            return summary

        except Exception as e:
            self.logger.error(
                f"❌ Performance summary generation failed for {model_id}: {e}"
            )
            return {
                "model_id": model_id,
                "error": str(e),
                "time_window_hours": time_window_hours,
            }

    async def compare_model_performance(
        self, model_ids: List[str], metric_type: PerformanceMetricType
    ) -> Dict[str, Any]:
        """
        Core service method: Compare performance across multiple models
        Comprehensive multi-model performance comparison
        """
        try:
            comparison_data = {
                "comparison_type": "multi_model",
                "metric_type": metric_type.value,
                "models_compared": len(model_ids),
                "comparison_results": [],
                "performance_ranking": [],
                "statistical_analysis": {},
                "comparison_timestamp": datetime.utcnow().isoformat(),
            }

            # Collect latest metrics for each model
            model_performances = []
            all_values = []

            for model_id in model_ids:
                model_metrics = [
                    m
                    for m in self.performance_metrics[model_id]
                    if m.metric_type == metric_type
                ]

                if model_metrics:
                    # Get recent metrics (last 24 hours)
                    recent_cutoff = datetime.utcnow() - timedelta(hours=24)
                    recent_metrics = [
                        m
                        for m in model_metrics
                        if m.measurement_timestamp >= recent_cutoff
                    ]

                    if recent_metrics:
                        values = [m.metric_value for m in recent_metrics]
                        avg_performance = statistics.mean(values)
                        latest_performance = values[-1]

                        model_performances.append(
                            {
                                "model_id": model_id,
                                "latest_value": latest_performance,
                                "average_24h": avg_performance,
                                "measurements_count": len(recent_metrics),
                                "std_deviation": (
                                    statistics.stdev(values) if len(values) > 1 else 0.0
                                ),
                                "trend": self._calculate_simple_trend(
                                    values[-10:]
                                ),  # Last 10 measurements
                            }
                        )

                        all_values.extend(values)
                    else:
                        model_performances.append(
                            {
                                "model_id": model_id,
                                "latest_value": None,
                                "average_24h": None,
                                "measurements_count": 0,
                                "message": "No recent measurements",
                            }
                        )

            # Sort by average performance
            model_performances.sort(
                key=lambda x: x.get("average_24h", 0) or 0, reverse=True
            )

            comparison_data["comparison_results"] = model_performances
            comparison_data["performance_ranking"] = [
                {
                    "rank": i + 1,
                    "model_id": m["model_id"],
                    "score": m.get("average_24h"),
                }
                for i, m in enumerate(model_performances)
                if m.get("average_24h") is not None
            ]

            # Statistical analysis across all models
            if all_values:
                comparison_data["statistical_analysis"] = {
                    "total_measurements": len(all_values),
                    "overall_average": statistics.mean(all_values),
                    "overall_median": statistics.median(all_values),
                    "overall_std_dev": (
                        statistics.stdev(all_values) if len(all_values) > 1 else 0.0
                    ),
                    "performance_range": max(all_values) - min(all_values),
                    "top_performer": (
                        model_performances[0]["model_id"]
                        if model_performances
                        else None
                    ),
                    "performance_distribution": self._calculate_performance_distribution(
                        all_values
                    ),
                }

            return comparison_data

        except Exception as e:
            self.logger.error(f"❌ Model performance comparison failed: {e}")
            return {
                "error": str(e),
                "models_requested": model_ids,
                "metric_type": metric_type.value,
            }

    async def get_performance_alerts(
        self, model_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get active performance alerts for models"""
        try:
            if model_id:
                # Get alerts for specific model
                model_alerts = self.active_alerts.get(model_id, [])
                return {
                    "model_id": model_id,
                    "active_alerts": model_alerts,
                    "alert_count": len(model_alerts),
                    "alert_levels": self._categorize_alerts(model_alerts),
                }
            else:
                # Get all alerts
                all_alerts = []
                for mid, alerts in self.active_alerts.items():
                    for alert in alerts:
                        alert["model_id"] = mid
                        all_alerts.append(alert)

                return {
                    "all_models": True,
                    "active_alerts": all_alerts,
                    "total_alert_count": len(all_alerts),
                    "models_with_alerts": len(self.active_alerts),
                    "alert_levels": self._categorize_alerts(all_alerts),
                    "alert_summary_timestamp": datetime.utcnow().isoformat(),
                }

        except Exception as e:
            self.logger.error(f"❌ Alert retrieval failed: {e}")
            return {"error": str(e)}

    async def update_performance_baseline(
        self, model_id: str, force_update: bool = False
    ) -> Dict[str, Any]:
        """Update performance baseline for a model"""
        try:
            # Calculate new baselines from recent data
            baseline_window = datetime.utcnow() - timedelta(
                days=self.monitoring_config["baseline_calculation_window_days"]
            )

            recent_metrics = [
                metric
                for metric in self.performance_metrics[model_id]
                if metric.measurement_timestamp >= baseline_window
            ]

            if not recent_metrics:
                return {
                    "model_id": model_id,
                    "baseline_updated": False,
                    "error": "Insufficient recent data for baseline calculation",
                }

            # Group by metric type and calculate baselines
            new_baselines = {}
            metrics_by_type = defaultdict(list)

            for metric in recent_metrics:
                metrics_by_type[metric.metric_type].append(metric.metric_value)

            for metric_type, values in metrics_by_type.items():
                if len(values) >= 3:  # Minimum data points for baseline
                    new_baselines[metric_type.value] = {
                        "baseline_value": statistics.mean(values),
                        "baseline_std": (
                            statistics.stdev(values) if len(values) > 1 else 0.0
                        ),
                        "data_points": len(values),
                        "calculation_window_days": self.monitoring_config[
                            "baseline_calculation_window_days"
                        ],
                        "updated_timestamp": datetime.utcnow().isoformat(),
                    }

            # Update baselines
            self.performance_baselines[model_id] = new_baselines

            baseline_result = {
                "model_id": model_id,
                "baseline_updated": True,
                "baselines_calculated": len(new_baselines),
                "baseline_metrics": list(new_baselines.keys()),
                "update_timestamp": datetime.utcnow().isoformat(),
            }

            self.logger.info(
                f"📊 Baseline updated for {model_id}: {len(new_baselines)} metrics"
            )
            return baseline_result

        except Exception as e:
            self.logger.error(f"❌ Baseline update failed for {model_id}: {e}")
            return {"model_id": model_id, "baseline_updated": False, "error": str(e)}

    async def get_monitoring_analytics(self) -> Dict[str, Any]:
        """Get comprehensive monitoring analytics across all models"""
        try:
            total_metrics = sum(
                len(metrics) for metrics in self.performance_metrics.values()
            )
            models_monitored = len(self.performance_metrics)

            # Calculate metric type distribution
            metric_type_distribution = defaultdict(int)
            for metrics_list in self.performance_metrics.values():
                for metric in metrics_list:
                    metric_type_distribution[metric.metric_type.value] += 1

            # Calculate alert statistics
            total_active_alerts = sum(
                len(alerts) for alerts in self.active_alerts.values()
            )
            alert_level_counts = defaultdict(int)

            for alerts_list in self.active_alerts.values():
                for alert in alerts_list:
                    alert_level_counts[alert.get("level", "unknown")] += 1

            # Performance trends
            overall_trends = {}
            for model_id, trends in self.performance_trends.items():
                for metric_type, values in trends.items():
                    if metric_type not in overall_trends:
                        overall_trends[metric_type] = []
                    overall_trends[metric_type].extend(values[-10:])  # Last 10 values

            trend_analysis = {}
            for metric_type, values in overall_trends.items():
                if len(values) >= 3:
                    trend_analysis[metric_type] = {
                        "overall_trend": self._calculate_simple_trend(values),
                        "average_value": statistics.mean(values),
                        "data_points": len(values),
                    }

            analytics = {
                "monitoring_overview": {
                    "models_monitored": models_monitored,
                    "total_metrics_recorded": total_metrics,
                    "active_alerts": total_active_alerts,
                    "monitoring_uptime_hours": self._calculate_monitoring_uptime(),
                },
                "metric_distribution": dict(metric_type_distribution),
                "alert_statistics": {
                    "total_active": total_active_alerts,
                    "by_level": dict(alert_level_counts),
                    "models_with_alerts": len(self.active_alerts),
                },
                "performance_trends": trend_analysis,
                "system_health": {
                    "monitoring_status": (
                        "healthy" if models_monitored > 0 else "warning"
                    ),
                    "data_freshness": self._assess_data_freshness(),
                    "baseline_coverage": len(self.performance_baselines)
                    / max(models_monitored, 1)
                    * 100,
                },
                "analytics_timestamp": datetime.utcnow().isoformat(),
            }

            return analytics

        except Exception as e:
            self.logger.error(f"❌ Monitoring analytics generation failed: {e}")
            return {"error": str(e)}

    async def _calculate_baseline_comparison(
        self, model_id: str, metric_type: PerformanceMetricType, metric_value: float
    ) -> float:
        """Calculate how current metric compares to baseline"""
        try:
            baseline_info = self.performance_baselines.get(model_id, {}).get(
                metric_type.value
            )

            if not baseline_info:
                return 0.0  # No baseline available

            baseline_value = baseline_info["baseline_value"]

            if baseline_value == 0:
                return 0.0

            # Calculate percentage difference from baseline
            comparison = (metric_value - baseline_value) / baseline_value
            return comparison

        except Exception as e:
            self.logger.error(f"❌ Baseline comparison calculation failed: {e}")
            return 0.0

    async def _determine_trend_direction(
        self, model_id: str, metric_type: PerformanceMetricType, metric_value: float
    ) -> str:
        """Determine trend direction based on recent values"""
        try:
            recent_values = self.performance_trends[model_id][metric_type.value][
                -5:
            ]  # Last 5 values

            if len(recent_values) < 2:
                return "insufficient_data"

            # Add current value
            recent_values = recent_values + [metric_value]

            # Simple trend calculation
            if len(recent_values) >= 3:
                slope = self._calculate_slope(recent_values)

                if slope > 0.05:
                    return "improving"
                elif slope < -0.05:
                    return "declining"
                else:
                    return "stable"

            return "unknown"

        except Exception as e:
            self.logger.error(f"❌ Trend calculation failed: {e}")
            return "error"

    def _calculate_slope(self, values: List[float]) -> float:
        """Calculate slope of values using simple linear regression"""
        n = len(values)
        if n < 2:
            return 0.0

        x_coords = list(range(n))
        x_mean = sum(x_coords) / n
        y_mean = sum(values) / n

        numerator = sum((x_coords[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x_coords[i] - x_mean) ** 2 for i in range(n))

        return numerator / denominator if denominator != 0 else 0.0

    async def _check_performance_alerts(
        self,
        model_id: str,
        metric_type: PerformanceMetricType,
        metric_value: float,
        baseline_comparison: float,
    ):
        """Check if metric triggers performance alerts"""
        try:
            thresholds = self.monitoring_config["alert_thresholds"].get(metric_type, {})

            if not thresholds:
                return

            alert_level = None

            # Determine alert level based on thresholds
            if metric_value <= thresholds.get("critical", 0):
                alert_level = "critical"
            elif metric_value <= thresholds.get("warning", 0):
                alert_level = "warning"

            # Create alert if threshold exceeded
            if alert_level:
                alert = {
                    "alert_id": f"{model_id}_{metric_type.value}_{datetime.utcnow().timestamp()}",
                    "metric_type": metric_type.value,
                    "metric_value": metric_value,
                    "alert_level": alert_level,
                    "threshold_value": thresholds[alert_level],
                    "baseline_comparison": baseline_comparison,
                    "triggered_timestamp": datetime.utcnow().isoformat(),
                    "status": "active",
                }

                # Check cooldown period
                if not self._is_alert_in_cooldown(model_id, metric_type):
                    self.active_alerts[model_id].append(alert)
                    self.alert_history.append(alert.copy())

                    self.logger.warning(
                        f"⚠️ Performance alert: {model_id} - {metric_type.value} = {metric_value} ({alert_level})"
                    )

        except Exception as e:
            self.logger.error(f"❌ Alert checking failed: {e}")

    def _is_alert_in_cooldown(
        self, model_id: str, metric_type: PerformanceMetricType
    ) -> bool:
        """Check if alert is in cooldown period"""
        cooldown_minutes = self.monitoring_config["performance_alert_cooldown_minutes"]
        cooldown_threshold = datetime.utcnow() - timedelta(minutes=cooldown_minutes)

        for alert in self.active_alerts.get(model_id, []):
            if (
                alert["metric_type"] == metric_type.value
                and datetime.fromisoformat(alert["triggered_timestamp"])
                > cooldown_threshold
            ):
                return True

        return False

    async def _cleanup_old_metrics(self, model_id: str):
        """Clean up old performance metrics based on retention policy"""
        try:
            retention_cutoff = datetime.utcnow() - timedelta(
                days=self.monitoring_config["metric_retention_days"]
            )

            original_count = len(self.performance_metrics[model_id])
            self.performance_metrics[model_id] = [
                metric
                for metric in self.performance_metrics[model_id]
                if metric.measurement_timestamp > retention_cutoff
            ]

            cleaned_count = original_count - len(self.performance_metrics[model_id])
            if cleaned_count > 0:
                self.logger.debug(
                    f"🧹 Cleaned {cleaned_count} old metrics for {model_id}"
                )

        except Exception as e:
            self.logger.error(f"❌ Metric cleanup failed for {model_id}: {e}")

    def _invalidate_analytics_cache(self, model_id: str):
        """Invalidate analytics cache for model"""
        keys_to_remove = [
            key for key in self.analytics_cache.keys() if key.startswith(model_id)
        ]
        for key in keys_to_remove:
            self.analytics_cache.pop(key, None)
            self.last_cache_update.pop(key, None)

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid"""
        if cache_key not in self.analytics_cache:
            return False

        last_update = self.last_cache_update.get(cache_key)
        if not last_update:
            return False

        cache_age = (datetime.utcnow() - last_update).total_seconds() / 60
        return cache_age < self.cache_ttl_minutes

    def _calculate_simple_trend(self, values: List[float]) -> str:
        """Calculate simple trend from list of values"""
        if len(values) < 3:
            return "insufficient_data"

        # Compare first third vs last third
        first_third = values[: len(values) // 3]
        last_third = values[-len(values) // 3 :]

        first_avg = statistics.mean(first_third)
        last_avg = statistics.mean(last_third)

        if last_avg > first_avg * 1.05:
            return "improving"
        elif last_avg < first_avg * 0.95:
            return "declining"
        else:
            return "stable"

    def _get_alert_status(
        self, model_id: str, metric_type: PerformanceMetricType, latest_value: float
    ) -> str:
        """Get current alert status for metric"""
        thresholds = self.monitoring_config["alert_thresholds"].get(metric_type, {})

        if not thresholds:
            return "no_thresholds"

        if latest_value <= thresholds.get("critical", 0):
            return "critical"
        elif latest_value <= thresholds.get("warning", 0):
            return "warning"
        else:
            return "ok"

    async def _calculate_overall_performance_score(
        self, metric_summaries: Dict[str, Any]
    ) -> float:
        """Calculate overall performance score from metric summaries"""
        try:
            if not metric_summaries:
                return 0.0

            # Weight different metrics
            metric_weights = {
                PerformanceMetricType.ACCURACY_SCORE.value: 0.25,
                PerformanceMetricType.COHERENCE_SCORE.value: 0.20,
                PerformanceMetricType.RELEVANCE_SCORE.value: 0.20,
                PerformanceMetricType.COMPLETENESS_SCORE.value: 0.15,
                PerformanceMetricType.CONFIDENCE_LEVEL.value: 0.10,
                PerformanceMetricType.RESPONSE_TIME.value: 0.10,  # Inverted - lower is better
            }

            weighted_scores = []
            total_weight = 0.0

            for metric_type, summary in metric_summaries.items():
                if metric_type in metric_weights:
                    latest_value = summary["latest_value"]
                    weight = metric_weights[metric_type]

                    # Special handling for response time (lower is better)
                    if metric_type == PerformanceMetricType.RESPONSE_TIME.value:
                        # Normalize response time (assume 1000ms is perfect, 10000ms is poor)
                        normalized_score = max(0, min(1, (10000 - latest_value) / 9000))
                    else:
                        normalized_score = latest_value

                    weighted_scores.append(normalized_score * weight)
                    total_weight += weight

            overall_score = (
                sum(weighted_scores) / total_weight if total_weight > 0 else 0.0
            )
            return min(max(overall_score, 0.0), 1.0)  # Bound between 0 and 1

        except Exception as e:
            self.logger.error(f"❌ Overall score calculation failed: {e}")
            return 0.5

    async def _generate_performance_insights(
        self, model_id: str, metric_summaries: Dict[str, Any]
    ) -> List[str]:
        """Generate performance insights based on metrics"""
        insights = []

        try:
            for metric_type, summary in metric_summaries.items():
                trend = summary.get("trend", "unknown")
                alert_status = summary.get("alert_status", "ok")
                latest_value = summary.get("latest_value", 0)

                # Trend insights
                if trend == "improving":
                    insights.append(f"{metric_type} showing positive trend")
                elif trend == "declining":
                    insights.append(f"{metric_type} declining - requires attention")

                # Alert insights
                if alert_status == "critical":
                    insights.append(
                        f"CRITICAL: {metric_type} below acceptable threshold ({latest_value:.3f})"
                    )
                elif alert_status == "warning":
                    insights.append(
                        f"WARNING: {metric_type} approaching threshold ({latest_value:.3f})"
                    )

                # Performance insights
                baseline_comparison = summary.get("baseline_comparison", {})
                latest_baseline_comparison = baseline_comparison.get("latest", 0)

                if latest_baseline_comparison > 0.1:
                    insights.append(
                        f"{metric_type} performing {latest_baseline_comparison:.1%} above baseline"
                    )
                elif latest_baseline_comparison < -0.1:
                    insights.append(
                        f"{metric_type} performing {abs(latest_baseline_comparison):.1%} below baseline"
                    )

            # Overall insights
            if len(insights) == 0:
                insights.append("Performance metrics within normal ranges")

            return insights[:5]  # Limit to top 5 insights

        except Exception as e:
            self.logger.error(f"❌ Insights generation failed: {e}")
            return ["Error generating performance insights"]

    def _categorize_alerts(self, alerts: List[Dict[str, Any]]) -> Dict[str, int]:
        """Categorize alerts by level"""
        categories = defaultdict(int)
        for alert in alerts:
            categories[alert.get("level", "unknown")] += 1
        return dict(categories)

    def _calculate_performance_distribution(
        self, values: List[float]
    ) -> Dict[str, Any]:
        """Calculate performance value distribution"""
        try:
            if not values:
                return {}

            sorted_values = sorted(values)
            n = len(sorted_values)

            return {
                "quartiles": {
                    "q1": sorted_values[n // 4],
                    "q2_median": sorted_values[n // 2],
                    "q3": sorted_values[3 * n // 4],
                },
                "percentiles": {
                    "p10": sorted_values[n // 10],
                    "p90": sorted_values[9 * n // 10],
                },
                "distribution_analysis": "normal",  # Simplified
            }
        except Exception:
            return {}

    def _calculate_monitoring_uptime(self) -> float:
        """Calculate monitoring system uptime"""
        # Simplified calculation - in production would track actual uptime
        return 24.0  # Mock 24 hours uptime

    def _assess_data_freshness(self) -> str:
        """Assess how fresh the monitoring data is"""
        try:
            now = datetime.utcnow()
            fresh_threshold = timedelta(minutes=30)

            recent_data_count = 0
            total_models = 0

            for model_id, metrics in self.performance_metrics.items():
                total_models += 1
                if (
                    metrics
                    and (now - metrics[-1].measurement_timestamp) < fresh_threshold
                ):
                    recent_data_count += 1

            freshness_ratio = recent_data_count / max(total_models, 1)

            if freshness_ratio >= 0.8:
                return "excellent"
            elif freshness_ratio >= 0.6:
                return "good"
            elif freshness_ratio >= 0.4:
                return "moderate"
            else:
                return "stale"

        except Exception:
            return "unknown"

    async def _start_baseline_updater(self):
        """Background task to update baselines periodically"""
        try:
            while True:
                await asyncio.sleep(
                    self.monitoring_config["baseline_update_frequency_hours"] * 3600
                )

                for model_id in self.performance_metrics.keys():
                    await self.update_performance_baseline(model_id)

        except Exception as e:
            self.logger.error(f"❌ Baseline updater failed: {e}")

    async def _start_performance_analyzer(self):
        """Background task for continuous performance analysis"""
        try:
            while True:
                await asyncio.sleep(600)  # Run every 10 minutes

                # Clear expired alerts
                await self._cleanup_expired_alerts()

                # Clear old cache entries
                await self._cleanup_analytics_cache()

        except Exception as e:
            self.logger.error(f"❌ Performance analyzer failed: {e}")

    async def _cleanup_expired_alerts(self):
        """Clean up expired alerts"""
        try:
            alert_expiry = datetime.utcnow() - timedelta(hours=24)

            for model_id in list(self.active_alerts.keys()):
                self.active_alerts[model_id] = [
                    alert
                    for alert in self.active_alerts[model_id]
                    if datetime.fromisoformat(alert["triggered_timestamp"])
                    > alert_expiry
                ]

                if not self.active_alerts[model_id]:
                    del self.active_alerts[model_id]

        except Exception as e:
            self.logger.error(f"❌ Alert cleanup failed: {e}")

    async def _cleanup_analytics_cache(self):
        """Clean up old analytics cache entries"""
        try:
            cache_expiry = datetime.utcnow() - timedelta(
                minutes=self.cache_ttl_minutes * 2
            )

            expired_keys = [
                key
                for key, timestamp in self.last_cache_update.items()
                if timestamp < cache_expiry
            ]

            for key in expired_keys:
                self.analytics_cache.pop(key, None)
                self.last_cache_update.pop(key, None)

        except Exception as e:
            self.logger.error(f"❌ Cache cleanup failed: {e}")

    async def get_service_health(self) -> Dict[str, Any]:
        """Return service health status"""
        total_metrics = sum(
            len(metrics) for metrics in self.performance_metrics.values()
        )
        active_alerts = sum(len(alerts) for alerts in self.active_alerts.values())

        return {
            "service_name": "PerformanceMonitoringService",
            "status": "healthy",
            "version": "v5_modular",
            "capabilities": [
                "performance_metric_recording",
                "trend_analysis",
                "baseline_management",
                "alert_monitoring",
                "multi_model_comparison",
                "analytics_generation",
            ],
            "monitoring_statistics": {
                "models_monitored": len(self.performance_metrics),
                "total_metrics": total_metrics,
                "active_alerts": active_alerts,
                "baselines_tracked": len(self.performance_baselines),
            },
            "configuration": self.monitoring_config,
            "cache_statistics": {
                "cached_analytics": len(self.analytics_cache),
                "cache_hit_rate": "estimated_85%",  # Mock value
            },
            "last_health_check": datetime.utcnow().isoformat(),
        }


# Global service instance for dependency injection
_performance_monitoring_service: Optional[PerformanceMonitoringService] = None


def get_performance_monitoring_service() -> PerformanceMonitoringService:
    """Get or create global performance monitoring service instance"""
    global _performance_monitoring_service

    if _performance_monitoring_service is None:
        _performance_monitoring_service = PerformanceMonitoringService()

    return _performance_monitoring_service
