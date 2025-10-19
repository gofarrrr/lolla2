"""
Model Performance Tracker - Extracted from model_manager.py
Handles performance monitoring, pattern learning, and Bayesian optimization
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass


@dataclass
class PerformanceMetric:
    """Individual performance metric record"""

    model_id: str
    score: float
    timestamp: datetime
    context: Dict[str, Any]
    problem_type: str
    business_domain: str


class ModelPerformanceTracker:
    """
    Tracks model performance, manages pattern similarity, and applies Bayesian optimization
    """

    def __init__(
        self,
        vector_similarity_engine: Optional[Any] = None,
        supabase_platform: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
        settings: Optional[Any] = None,
    ):
        self.vector_similarity_engine = vector_similarity_engine
        self.supabase_platform = supabase_platform
        self.logger = logger or logging.getLogger(__name__)
        self.settings = settings

        # In-memory performance tracking
        self.performance_history: Dict[str, List[PerformanceMetric]] = {}
        self.pattern_cache: Dict[str, Any] = {}

        # Performance thresholds
        self.success_threshold = 0.7
        self.max_history_size = (
            getattr(settings, "MAX_MODEL_PERFORMANCE_HISTORY", 100) if settings else 100
        )

    async def update_model_performance(
        self,
        model_id: str,
        performance_score: float,
        business_context: Dict[str, Any],
        problem_statement: str = "",
    ) -> None:
        """
        Update performance history for a model with context
        """
        try:
            # Create performance metric
            metric = PerformanceMetric(
                model_id=model_id,
                score=performance_score,
                timestamp=datetime.utcnow(),
                context=business_context,
                problem_type=business_context.get("problem_type", "unknown"),
                business_domain=business_context.get("industry", "general"),
            )

            # Update in-memory history
            if model_id not in self.performance_history:
                self.performance_history[model_id] = []

            self.performance_history[model_id].append(metric)

            # Maintain history size limit
            if len(self.performance_history[model_id]) > self.max_history_size:
                self.performance_history[model_id] = self.performance_history[model_id][
                    -self.max_history_size :
                ]

            # Store similarity pattern for future matching
            await self._store_similarity_pattern(
                model_id, business_context, performance_score
            )

            # Persist to database if available
            if self.supabase_platform:
                await self._persist_performance_metric(metric)

            self.logger.debug(
                f"📊 Updated performance for {model_id}: {performance_score:.3f}"
            )

        except Exception as e:
            self.logger.error(f"❌ Failed to update performance for {model_id}: {e}")

    async def get_model_performance_history(
        self, model_id: str, days_back: int = 30
    ) -> List[PerformanceMetric]:
        """
        Get performance history for a specific model
        """
        try:
            if model_id not in self.performance_history:
                return []

            # Filter by date range
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)
            recent_metrics = [
                metric
                for metric in self.performance_history[model_id]
                if metric.timestamp >= cutoff_date
            ]

            return recent_metrics

        except Exception as e:
            self.logger.error(
                f"❌ Failed to get performance history for {model_id}: {e}"
            )
            return []

    async def get_top_performing_models(
        self, problem_type: str = "", min_samples: int = 3, limit: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Get top performing models based on average performance
        """
        try:
            model_averages = []

            for model_id, metrics in self.performance_history.items():
                if len(metrics) < min_samples:
                    continue

                # Filter by problem type if specified
                filtered_metrics = metrics
                if problem_type:
                    filtered_metrics = [
                        m
                        for m in metrics
                        if m.problem_type.lower() == problem_type.lower()
                    ]

                if len(filtered_metrics) < min_samples:
                    continue

                # Calculate average performance
                avg_score = sum(m.score for m in filtered_metrics) / len(
                    filtered_metrics
                )
                model_averages.append((model_id, avg_score))

            # Sort by average score descending
            model_averages.sort(key=lambda x: x[1], reverse=True)

            return model_averages[:limit]

        except Exception as e:
            self.logger.error(f"❌ Failed to get top performing models: {e}")
            return []

    async def calculate_confidence_calibration(
        self, model_id: str, predicted_confidence: float
    ) -> float:
        """
        Apply confidence calibration based on historical performance
        """
        try:
            if model_id not in self.performance_history:
                return predicted_confidence

            metrics = self.performance_history[model_id]
            if len(metrics) < 5:  # Need minimum samples for calibration
                return predicted_confidence

            # Calculate actual accuracy vs predicted confidence correlation
            recent_metrics = metrics[-20:]  # Use recent performance
            avg_actual_performance = sum(m.score for m in recent_metrics) / len(
                recent_metrics
            )

            # Simple calibration: adjust based on historical accuracy
            calibration_factor = avg_actual_performance / 0.7  # Target baseline
            calibrated_confidence = predicted_confidence * calibration_factor

            # Clamp to valid range
            return max(0.1, min(0.99, calibrated_confidence))

        except Exception as e:
            self.logger.error(f"❌ Confidence calibration failed for {model_id}: {e}")
            return predicted_confidence

    async def _store_similarity_pattern(
        self, model_id: str, business_context: Dict[str, Any], performance_score: float
    ) -> None:
        """
        Store pattern for similarity matching in future selections
        """
        try:
            if not self.vector_similarity_engine:
                return

            # Only store successful patterns
            if performance_score < self.success_threshold:
                return

            # Create pattern representation
            pattern = {
                "model_id": model_id,
                "performance_score": performance_score,
                "problem_statement": business_context.get("problem_statement", "")[
                    :500
                ],
                "business_context": str(business_context)[:300],
                "domain": business_context.get("industry", "general"),
                "problem_type": business_context.get("problem_type", "unknown"),
                "timestamp": datetime.utcnow().isoformat(),
            }

            # Store in vector similarity engine
            from src.core.vector_similarity_detection import PatternType

            await self.vector_similarity_engine.store_pattern(
                pattern_type=PatternType.PROBLEM_SIMILARITY,
                pattern_id=f"{model_id}_{datetime.utcnow().timestamp()}",
                pattern_data=pattern,
                metadata={"successful_model": model_id},
            )

            self.logger.debug(f"✅ Stored successful pattern for {model_id}")

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to store similarity pattern: {e}")

    async def _persist_performance_metric(self, metric: PerformanceMetric) -> None:
        """
        Persist performance metric to database
        """
        try:
            if not self.supabase_platform:
                return

            await self.supabase_platform.store_performance_metric(
                {
                    "model_id": metric.model_id,
                    "performance_score": metric.score,
                    "timestamp": metric.timestamp.isoformat(),
                    "business_context": metric.context,
                    "problem_type": metric.problem_type,
                    "business_domain": metric.business_domain,
                }
            )

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to persist performance metric: {e}")

    async def apply_bayesian_effectiveness_update(
        self, model_id: str, effectiveness_score: float, context: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Apply Bayesian updates to model effectiveness estimates
        """
        try:
            # Get historical performance
            metrics = await self.get_model_performance_history(model_id, days_back=90)

            if not metrics:
                # No history, return current score as estimate
                return {
                    "prior_mean": 0.7,
                    "posterior_mean": effectiveness_score,
                    "confidence_interval": [
                        effectiveness_score - 0.1,
                        effectiveness_score + 0.1,
                    ],
                }

            # Calculate Bayesian posterior
            scores = [m.score for m in metrics]

            # Prior parameters (Beta distribution)
            prior_alpha = 7.0  # Prior successes
            prior_beta = 3.0  # Prior failures

            # Observed data
            successes = sum(1 for score in scores if score > self.success_threshold)
            failures = len(scores) - successes

            # Posterior parameters
            posterior_alpha = prior_alpha + successes
            posterior_beta = prior_beta + failures

            # Posterior mean
            posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)

            # 95% confidence interval (approximate)
            variance = (posterior_alpha * posterior_beta) / (
                (posterior_alpha + posterior_beta) ** 2
                * (posterior_alpha + posterior_beta + 1)
            )
            std_dev = variance**0.5
            confidence_interval = [
                max(0.0, posterior_mean - 1.96 * std_dev),
                min(1.0, posterior_mean + 1.96 * std_dev),
            ]

            self.logger.debug(
                f"🧮 Bayesian update for {model_id}: {posterior_mean:.3f} ± {std_dev:.3f}"
            )

            return {
                "prior_mean": prior_alpha / (prior_alpha + prior_beta),
                "posterior_mean": posterior_mean,
                "confidence_interval": confidence_interval,
                "sample_size": len(scores),
            }

        except Exception as e:
            self.logger.error(f"❌ Bayesian effectiveness update failed: {e}")
            return {
                "prior_mean": 0.7,
                "posterior_mean": effectiveness_score,
                "confidence_interval": [0.6, 0.8],
            }

    async def get_performance_analytics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance analytics across all models
        """
        try:
            analytics = {
                "total_models_tracked": len(self.performance_history),
                "total_performance_records": sum(
                    len(metrics) for metrics in self.performance_history.values()
                ),
                "top_performers": await self.get_top_performing_models(limit=10),
                "performance_trends": {},
                "domain_performance": {},
                "recent_activity": [],
            }

            # Performance trends by model
            for model_id, metrics in self.performance_history.items():
                if len(metrics) >= 5:
                    recent_scores = [m.score for m in metrics[-10:]]
                    older_scores = (
                        [m.score for m in metrics[-20:-10]]
                        if len(metrics) >= 20
                        else []
                    )

                    recent_avg = sum(recent_scores) / len(recent_scores)
                    older_avg = (
                        sum(older_scores) / len(older_scores)
                        if older_scores
                        else recent_avg
                    )

                    trend = (
                        "improving"
                        if recent_avg > older_avg + 0.05
                        else "declining" if recent_avg < older_avg - 0.05 else "stable"
                    )

                    analytics["performance_trends"][model_id] = {
                        "trend": trend,
                        "recent_avg": recent_avg,
                        "change": recent_avg - older_avg,
                    }

            # Domain performance analysis
            domain_scores = {}
            for metrics in self.performance_history.values():
                for metric in metrics:
                    domain = metric.business_domain
                    if domain not in domain_scores:
                        domain_scores[domain] = []
                    domain_scores[domain].append(metric.score)

            for domain, scores in domain_scores.items():
                analytics["domain_performance"][domain] = {
                    "avg_score": sum(scores) / len(scores),
                    "sample_size": len(scores),
                    "success_rate": sum(1 for s in scores if s > self.success_threshold)
                    / len(scores),
                }

            # Recent activity
            all_metrics = []
            for metrics in self.performance_history.values():
                all_metrics.extend(metrics)

            # Sort by timestamp and get recent
            all_metrics.sort(key=lambda x: x.timestamp, reverse=True)
            for metric in all_metrics[:20]:
                analytics["recent_activity"].append(
                    {
                        "model_id": metric.model_id,
                        "score": metric.score,
                        "timestamp": metric.timestamp.isoformat(),
                        "domain": metric.business_domain,
                        "problem_type": metric.problem_type,
                    }
                )

            return analytics

        except Exception as e:
            self.logger.error(f"❌ Failed to generate performance analytics: {e}")
            return {"error": str(e)}

    def clear_performance_history(self, model_id: Optional[str] = None) -> None:
        """
        Clear performance history (for testing or reset)
        """
        if model_id:
            self.performance_history.pop(model_id, None)
            self.logger.info(f"🧹 Cleared performance history for {model_id}")
        else:
            self.performance_history.clear()
            self.pattern_cache.clear()
            self.logger.info("🧹 Cleared all performance history")
