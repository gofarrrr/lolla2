#!/usr/bin/env python3
"""
Value Assessment Database Integration
Phase 1: Foundation Systems - Systematic Intelligence Amplification

Connects ValueAssessment outcomes to database scoring for comprehensive feedback loops
between delivered business value and cognitive system performance.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict
import statistics

from src.core.value_assessment import (
    get_value_assessment_engine,
)
from src.intelligence.bayesian_effectiveness_updater import (
    get_bayesian_updater,
    EffectivenessUpdate,
)
from src.engine.monitoring.performance_validator import (
    get_performance_validator,
    PerformanceMetricType,
)


@dataclass
class ValueOutcome:
    """Represents a realized business value outcome from an engagement"""

    engagement_id: str
    assessment_id: str
    model_ids_used: List[str]

    # Value realization metrics
    planned_value_score: float  # Original assessment score
    realized_value_score: float  # Actual delivered value
    value_realization_ratio: float  # realized/planned

    # Time and effort metrics
    planned_timeline_months: float
    actual_timeline_months: float
    implementation_efficiency: float  # planned/actual timeline

    # Stakeholder feedback
    stakeholder_satisfaction: float  # 0.0-1.0
    recommendation_likelihood: float  # 0.0-1.0 (NPS-style)

    # Business impact metrics
    measurable_roi: Optional[float] = None
    cost_savings_realized: Optional[float] = None
    revenue_impact_realized: Optional[float] = None

    # Quality metrics
    insight_accuracy: float = 0.8  # How accurate were the insights
    recommendation_quality: float = 0.8  # How good were the recommendations
    strategic_alignment: float = 0.7  # How well aligned with strategy

    # Metadata
    outcome_measured_at: datetime = field(default_factory=datetime.utcnow)
    follow_up_scheduled: bool = False
    lessons_learned: List[str] = field(default_factory=list)
    improvement_opportunities: List[str] = field(default_factory=list)

    # Week 1 Day 3: Anti-Gaming Safeguards - Flag for manual review
    needs_review: bool = False
    validation_flags: List[str] = field(default_factory=list)  # Reasons for flagging
    reviewer_notes: str = ""
    approved_for_learning: bool = True  # False if flagged for review


@dataclass
class ModelValueContribution:
    """Tracks individual mental model contribution to value outcomes"""

    model_id: str
    value_contribution_score: float  # 0.0-1.0
    usage_frequency: int
    avg_confidence: float
    stakeholder_feedback: float
    domain_effectiveness: Dict[str, float] = field(default_factory=dict)

    # Learning metrics
    initial_performance: float = 0.5
    current_performance: float = 0.5
    performance_trend: float = 0.0  # Change over time

    # Context-specific performance
    complexity_performance: Dict[str, float] = field(
        default_factory=dict
    )  # low/medium/high
    industry_performance: Dict[str, float] = field(default_factory=dict)


class ValueDatabaseIntegrator:
    """
    Integrates ValueAssessment outcomes with database scoring system
    Creates feedback loops for continuous system improvement
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.value_engine = get_value_assessment_engine()

        # Core components (initialized lazily)
        self.bayesian_updater = None
        self.performance_validator = None

        # Value tracking
        self.value_outcomes: Dict[str, ValueOutcome] = {}
        self.model_contributions: Dict[str, ModelValueContribution] = defaultdict(
            lambda: ModelValueContribution(
                model_id="",
                value_contribution_score=0.5,
                usage_frequency=0,
                avg_confidence=0.5,
                stakeholder_feedback=0.5,
            )
        )

        # Analytics
        self.value_trends = defaultdict(list)
        self.feedback_patterns = defaultdict(list)

        self.logger.info(
            "✅ ValueDatabaseIntegrator initialized - Value outcome tracking active"
        )

    async def _ensure_components_initialized(self):
        """Lazy initialization of dependent components"""
        if self.bayesian_updater is None:
            try:
                self.bayesian_updater = get_bayesian_updater()
            except Exception as e:
                self.logger.warning(f"⚠️ Bayesian updater unavailable: {e}")

        if self.performance_validator is None:
            try:
                self.performance_validator = await get_performance_validator()
            except Exception as e:
                self.logger.warning(f"⚠️ Performance validator unavailable: {e}")

    async def record_value_outcome(
        self,
        engagement_id: str,
        assessment_id: str,
        model_ids_used: List[str],
        realized_metrics: Dict[str, Any],
    ) -> ValueOutcome:
        """Record actual value outcome from completed engagement"""

        await self._ensure_components_initialized()

        # Get original assessment for comparison
        original_assessment = self.value_engine.get_assessment(assessment_id)
        if not original_assessment:
            self.logger.error(f"❌ Original assessment not found: {assessment_id}")
            return None

        # Extract planned vs realized metrics
        planned_value_score = original_assessment.weighted_value_score
        realized_value_score = realized_metrics.get(
            "overall_value_realized", planned_value_score * 0.8
        )

        # Create value outcome record
        outcome = ValueOutcome(
            engagement_id=engagement_id,
            assessment_id=assessment_id,
            model_ids_used=model_ids_used,
            planned_value_score=planned_value_score,
            realized_value_score=realized_value_score,
            value_realization_ratio=(
                realized_value_score / planned_value_score
                if planned_value_score > 0
                else 1.0
            ),
            planned_timeline_months=realized_metrics.get(
                "planned_timeline_months", 6.0
            ),
            actual_timeline_months=realized_metrics.get("actual_timeline_months", 6.0),
            implementation_efficiency=realized_metrics.get(
                "planned_timeline_months", 6.0
            )
            / max(0.1, realized_metrics.get("actual_timeline_months", 6.0)),
            stakeholder_satisfaction=realized_metrics.get(
                "stakeholder_satisfaction", 0.7
            ),
            recommendation_likelihood=realized_metrics.get(
                "recommendation_likelihood", 0.6
            ),
            measurable_roi=realized_metrics.get("measurable_roi"),
            cost_savings_realized=realized_metrics.get("cost_savings_realized"),
            revenue_impact_realized=realized_metrics.get("revenue_impact_realized"),
            insight_accuracy=realized_metrics.get("insight_accuracy", 0.8),
            recommendation_quality=realized_metrics.get("recommendation_quality", 0.8),
            strategic_alignment=realized_metrics.get("strategic_alignment", 0.7),
            lessons_learned=realized_metrics.get("lessons_learned", []),
            improvement_opportunities=realized_metrics.get(
                "improvement_opportunities", []
            ),
        )

        # Week 1 Day 3: Anti-Gaming Safeguards - Validate outcome before processing
        await self._validate_value_outcome(outcome)

        # Store outcome
        self.value_outcomes[engagement_id] = outcome

        # Only update model contributions if approved for learning
        if outcome.approved_for_learning:
            await self._update_model_contributions(outcome)
        else:
            self.logger.warning(
                f"⚠️  Outcome {engagement_id} flagged for review - excluded from learning updates"
            )

        # Feed back to database scoring systems
        await self._update_database_scoring(outcome)

        # Track trends
        self._update_value_trends(outcome)

        self.logger.info(
            f"💎 Value outcome recorded: {engagement_id} | "
            f"Realization ratio: {outcome.value_realization_ratio:.2f} | "
            f"Stakeholder satisfaction: {outcome.stakeholder_satisfaction:.2f}"
        )

        return outcome

    async def _update_model_contributions(self, outcome: ValueOutcome):
        """Update individual mental model value contribution tracking"""

        # Calculate per-model value contribution
        models_used = len(outcome.model_ids_used)
        if models_used == 0:
            return

        base_contribution = outcome.realized_value_score / models_used

        for model_id in outcome.model_ids_used:
            contribution = self.model_contributions[model_id]
            contribution.model_id = model_id

            # Update metrics
            contribution.usage_frequency += 1

            # Weighted average of value contributions
            current_weight = contribution.usage_frequency
            new_weight = 1
            total_weight = current_weight + new_weight

            contribution.value_contribution_score = (
                contribution.value_contribution_score * current_weight
                + base_contribution * new_weight
            ) / total_weight

            # Update stakeholder feedback
            contribution.stakeholder_feedback = (
                contribution.stakeholder_feedback * current_weight
                + outcome.stakeholder_satisfaction * new_weight
            ) / total_weight

            # Track performance trend
            contribution.current_performance = outcome.realized_value_score
            if contribution.initial_performance == 0.5:  # First usage
                contribution.initial_performance = outcome.realized_value_score
            else:
                contribution.performance_trend = (
                    contribution.current_performance - contribution.initial_performance
                )

            self.logger.debug(
                f"📊 Updated contribution for {model_id}: {contribution.value_contribution_score:.3f}"
            )

    async def _update_database_scoring(self, outcome: ValueOutcome):
        """Feed value outcomes back to database scoring systems"""

        # Update Bayesian effectiveness with value realization data
        if self.bayesian_updater:
            for model_id in outcome.model_ids_used:
                # Convert value realization to effectiveness score
                effectiveness_score = min(1.0, outcome.value_realization_ratio)

                effectiveness_update = EffectivenessUpdate(
                    model_id=model_id,
                    effectiveness_score=effectiveness_score,
                    timestamp=outcome.outcome_measured_at,
                    context={
                        "feedback_source": "value_outcome",
                        "engagement_id": outcome.engagement_id,
                        "stakeholder_satisfaction": outcome.stakeholder_satisfaction,
                        "implementation_efficiency": outcome.implementation_efficiency,
                        "business_value_realized": True,
                    },
                    engagement_id=outcome.engagement_id,
                    problem_type="value_realization",
                    complexity_level="high",  # Real outcomes are high complexity
                )

                try:
                    await self.bayesian_updater.update_model_effectiveness(
                        effectiveness_update
                    )
                    self.logger.debug(
                        f"📈 Bayesian update from value outcome: {model_id} → {effectiveness_score:.3f}"
                    )
                except Exception as e:
                    self.logger.warning(
                        f"⚠️ Failed to update Bayesian effectiveness for {model_id}: {e}"
                    )

        # Update performance validator with value metrics
        if self.performance_validator:
            try:
                # Record overall engagement value
                await self.performance_validator.record_measurement(
                    metric_type=PerformanceMetricType.MODEL_EFFECTIVENESS,
                    value=outcome.realized_value_score,
                    context={
                        "measurement_source": "value_outcome",
                        "stakeholder_satisfaction": outcome.stakeholder_satisfaction,
                        "value_realization_ratio": outcome.value_realization_ratio,
                        "models_used": outcome.model_ids_used,
                    },
                    engagement_id=outcome.engagement_id,
                )

                # Record individual model contributions
                for model_id in outcome.model_ids_used:
                    contribution = self.model_contributions[model_id]
                    await self.performance_validator.record_model_effectiveness(
                        model_id=model_id,
                        effectiveness_score=contribution.value_contribution_score,
                        engagement_id=outcome.engagement_id,
                        context={
                            "value_contribution": True,
                            "stakeholder_feedback": contribution.stakeholder_feedback,
                            "usage_frequency": contribution.usage_frequency,
                        },
                    )

                self.logger.debug(
                    f"📊 Performance metrics updated from value outcome: {outcome.engagement_id}"
                )

            except Exception as e:
                self.logger.warning(f"⚠️ Failed to update performance metrics: {e}")

    def _update_value_trends(self, outcome: ValueOutcome):
        """Update value trend analytics"""

        # Track value realization trends
        self.value_trends["realization_ratio"].append(outcome.value_realization_ratio)
        self.value_trends["stakeholder_satisfaction"].append(
            outcome.stakeholder_satisfaction
        )
        self.value_trends["implementation_efficiency"].append(
            outcome.implementation_efficiency
        )

        # Keep only recent trends (last 100 outcomes)
        for trend_type in self.value_trends:
            if len(self.value_trends[trend_type]) > 100:
                self.value_trends[trend_type] = self.value_trends[trend_type][-100:]

    def get_model_value_performance(self, model_id: str) -> Dict[str, Any]:
        """Get comprehensive value performance metrics for a mental model"""

        if model_id not in self.model_contributions:
            return {
                "model_id": model_id,
                "value_contribution_score": 0.5,
                "usage_frequency": 0,
                "stakeholder_feedback": 0.5,
                "performance_trend": 0.0,
                "data_available": False,
            }

        contribution = self.model_contributions[model_id]

        return {
            "model_id": model_id,
            "value_contribution_score": contribution.value_contribution_score,
            "usage_frequency": contribution.usage_frequency,
            "avg_confidence": contribution.avg_confidence,
            "stakeholder_feedback": contribution.stakeholder_feedback,
            "performance_trend": contribution.performance_trend,
            "initial_performance": contribution.initial_performance,
            "current_performance": contribution.current_performance,
            "domain_effectiveness": dict(contribution.domain_effectiveness),
            "complexity_performance": dict(contribution.complexity_performance),
            "industry_performance": dict(contribution.industry_performance),
            "data_available": True,
        }

    def get_value_trends_analysis(self) -> Dict[str, Any]:
        """Get comprehensive analysis of value trends"""

        analysis = {
            "trend_periods": len(self.value_trends.get("realization_ratio", [])),
            "trends": {},
        }

        for trend_type, values in self.value_trends.items():
            if len(values) >= 5:  # Minimum for meaningful analysis
                analysis["trends"][trend_type] = {
                    "current_mean": (
                        statistics.mean(values[-10:])
                        if len(values) >= 10
                        else statistics.mean(values)
                    ),
                    "historical_mean": statistics.mean(values),
                    "trend_direction": (
                        "improving"
                        if len(values) >= 10
                        and statistics.mean(values[-5:]) > statistics.mean(values[:-5])
                        else "stable"
                    ),
                    "volatility": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "min_value": min(values),
                    "max_value": max(values),
                    "sample_count": len(values),
                }

        return analysis

    def get_top_value_contributors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top mental models by value contribution"""

        contributors = []

        for model_id, contribution in self.model_contributions.items():
            if contribution.usage_frequency > 0:  # Only models with actual usage
                contributors.append(
                    {
                        "model_id": model_id,
                        "value_contribution_score": contribution.value_contribution_score,
                        "stakeholder_feedback": contribution.stakeholder_feedback,
                        "usage_frequency": contribution.usage_frequency,
                        "performance_trend": contribution.performance_trend,
                        "combined_score": (
                            contribution.value_contribution_score * 0.5
                            + contribution.stakeholder_feedback * 0.3
                            + min(1.0, contribution.usage_frequency / 10.0) * 0.2
                        ),
                    }
                )

        # Sort by combined score
        contributors.sort(key=lambda x: x["combined_score"], reverse=True)

        return contributors[:limit]

    def simulate_value_outcome_feedback(
        self, engagement_count: int = 20
    ) -> Dict[str, Any]:
        """Simulate value outcome feedback for testing purposes"""

        import random
        from uuid import uuid4

        self.logger.info(
            f"🔄 Simulating {engagement_count} value outcomes for testing..."
        )

        # Common mental models for simulation
        test_models = [
            "critical_thinking_framework",
            "systems_thinking_analysis",
            "multi_criteria_decision_analysis",
            "hypothesis_testing_methodology",
            "scenario_planning_framework",
        ]

        simulation_results = {
            "outcomes_generated": 0,
            "avg_realization_ratio": 0.0,
            "avg_stakeholder_satisfaction": 0.0,
            "model_performance_updates": 0,
        }

        for i in range(engagement_count):
            # Generate realistic metrics with some variation
            base_performance = 0.7 + random.uniform(-0.2, 0.3)  # 0.5-1.0 range

            realized_metrics = {
                "overall_value_realized": max(
                    0.1, min(1.0, base_performance + random.uniform(-0.1, 0.2))
                ),
                "planned_timeline_months": random.uniform(3, 12),
                "actual_timeline_months": random.uniform(3, 15),
                "stakeholder_satisfaction": max(
                    0.3, min(1.0, base_performance + random.uniform(-0.15, 0.25))
                ),
                "recommendation_likelihood": max(
                    0.2, min(1.0, base_performance + random.uniform(-0.2, 0.3))
                ),
                "insight_accuracy": max(
                    0.5, min(1.0, base_performance + random.uniform(-0.1, 0.2))
                ),
                "recommendation_quality": max(
                    0.4, min(1.0, base_performance + random.uniform(-0.15, 0.25))
                ),
                "strategic_alignment": max(
                    0.4, min(1.0, base_performance + random.uniform(-0.2, 0.3))
                ),
                "lessons_learned": (
                    ["Simulation generated insight"] if random.random() > 0.5 else []
                ),
            }

            # Random model selection (2-4 models per engagement)
            models_used = random.sample(test_models, random.randint(2, 4))

            # Create value outcome
            engagement_id = str(uuid4())
            assessment_id = str(uuid4())

            # Create and record outcome
            outcome = ValueOutcome(
                engagement_id=engagement_id,
                assessment_id=assessment_id,
                model_ids_used=models_used,
                planned_value_score=0.8,  # Simulated planned value
                realized_value_score=realized_metrics["overall_value_realized"],
                value_realization_ratio=realized_metrics["overall_value_realized"]
                / 0.8,
                planned_timeline_months=realized_metrics["planned_timeline_months"],
                actual_timeline_months=realized_metrics["actual_timeline_months"],
                implementation_efficiency=realized_metrics["planned_timeline_months"]
                / realized_metrics["actual_timeline_months"],
                stakeholder_satisfaction=realized_metrics["stakeholder_satisfaction"],
                recommendation_likelihood=realized_metrics["recommendation_likelihood"],
                insight_accuracy=realized_metrics["insight_accuracy"],
                recommendation_quality=realized_metrics["recommendation_quality"],
                strategic_alignment=realized_metrics["strategic_alignment"],
            )

            # Store and process
            self.value_outcomes[engagement_id] = outcome
            asyncio.create_task(self._update_model_contributions(outcome))

            simulation_results["outcomes_generated"] += 1

        # Calculate summary statistics
        if self.value_outcomes:
            realization_ratios = [
                o.value_realization_ratio for o in self.value_outcomes.values()
            ]
            satisfaction_scores = [
                o.stakeholder_satisfaction for o in self.value_outcomes.values()
            ]

            simulation_results["avg_realization_ratio"] = statistics.mean(
                realization_ratios
            )
            simulation_results["avg_stakeholder_satisfaction"] = statistics.mean(
                satisfaction_scores
            )
            simulation_results["model_performance_updates"] = len(
                self.model_contributions
            )

        self.logger.info(f"✅ Simulation completed: {simulation_results}")

        return simulation_results

    def get_system_health_metrics(self) -> Dict[str, Any]:
        """Get overall system health from value perspective"""

        if not self.value_outcomes:
            return {
                "status": "no_data",
                "outcomes_tracked": 0,
                "avg_value_realization": 0.0,
                "avg_stakeholder_satisfaction": 0.0,
            }

        outcomes = list(self.value_outcomes.values())

        # Calculate key health metrics
        realization_ratios = [o.value_realization_ratio for o in outcomes]
        satisfaction_scores = [o.stakeholder_satisfaction for o in outcomes]
        efficiency_scores = [o.implementation_efficiency for o in outcomes]

        avg_realization = statistics.mean(realization_ratios)
        avg_satisfaction = statistics.mean(satisfaction_scores)
        avg_efficiency = statistics.mean(efficiency_scores)

        # Determine overall health status
        if avg_realization >= 0.8 and avg_satisfaction >= 0.7:
            status = "excellent"
        elif avg_realization >= 0.7 and avg_satisfaction >= 0.6:
            status = "good"
        elif avg_realization >= 0.6 and avg_satisfaction >= 0.5:
            status = "satisfactory"
        else:
            status = "needs_attention"

        return {
            "status": status,
            "outcomes_tracked": len(outcomes),
            "models_contributing": len(self.model_contributions),
            "avg_value_realization": avg_realization,
            "avg_stakeholder_satisfaction": avg_satisfaction,
            "avg_implementation_efficiency": avg_efficiency,
            "trend_periods": len(self.value_trends.get("realization_ratio", [])),
            "system_learning": len(
                [m for m in self.model_contributions.values() if m.usage_frequency > 2]
            ),
        }

    async def _validate_value_outcome(self, outcome: ValueOutcome):
        """
        Week 1 Day 3: Anti-Gaming Safeguards - Validate outcome for potential manipulation

        Performs two checks:
        a. Outlier Detection: Check if stakeholder_satisfaction is significant outlier
        b. Consistency Check: Check if satisfaction high but value_realization_ratio low
        """

        validation_flags = []

        # Check 1: Outlier Detection
        outlier_detected = await self._detect_satisfaction_outlier(outcome)
        if outlier_detected:
            validation_flags.append("satisfaction_outlier")

        # Check 2: Consistency Check
        inconsistency_detected = self._check_satisfaction_value_consistency(outcome)
        if inconsistency_detected:
            validation_flags.append("satisfaction_value_inconsistency")

        # Update outcome based on validation results
        if validation_flags:
            outcome.needs_review = True
            outcome.validation_flags = validation_flags
            outcome.approved_for_learning = False

            self.logger.warning(
                f"🚨 Week 1 Day 3: Value outcome flagged for review: {outcome.engagement_id}\n"
                f"   Flags: {', '.join(validation_flags)}\n"
                f"   Stakeholder Satisfaction: {outcome.stakeholder_satisfaction:.3f}\n"
                f"   Value Realization Ratio: {outcome.value_realization_ratio:.3f}\n"
                f"   This outcome will be excluded from Bayesian updates until manually approved."
            )
        else:
            self.logger.debug(
                f"✅ Value outcome validation passed: {outcome.engagement_id}"
            )

    async def _detect_satisfaction_outlier(self, outcome: ValueOutcome) -> bool:
        """
        Week 1 Day 3: Detect if stakeholder_satisfaction is significant outlier
        Check against user's historical average (last 20 engagements)
        """

        # For this implementation, we'll use a simplified approach
        # In production, this would query historical data by user_id

        # Get recent outcomes for comparison (simplified - using all recent outcomes)
        recent_outcomes = [
            o
            for o in self.value_outcomes.values()
            if o.engagement_id != outcome.engagement_id  # Exclude current
        ]

        if len(recent_outcomes) < 5:
            # Not enough historical data for outlier detection
            return False

        # Take last 20 outcomes as historical baseline
        historical_satisfaction = [
            o.stakeholder_satisfaction for o in recent_outcomes[-20:]
        ]

        # Calculate mean and standard deviation
        mean_satisfaction = statistics.mean(historical_satisfaction)

        if len(historical_satisfaction) < 2:
            return False

        stdev_satisfaction = statistics.stdev(historical_satisfaction)

        # Check if current satisfaction is > 3 standard deviations from mean
        z_score = abs(outcome.stakeholder_satisfaction - mean_satisfaction) / max(
            0.01, stdev_satisfaction
        )

        is_outlier = z_score > 3.0

        if is_outlier:
            self.logger.info(
                f"📊 Outlier detected: satisfaction={outcome.stakeholder_satisfaction:.3f}, "
                f"historical_mean={mean_satisfaction:.3f}, z_score={z_score:.2f}"
            )

        return is_outlier

    def _check_satisfaction_value_consistency(self, outcome: ValueOutcome) -> bool:
        """
        Week 1 Day 3: Check consistency between satisfaction and value realization
        Flag if satisfaction >= 0.9 but value_realization_ratio < 0.5
        """

        high_satisfaction = outcome.stakeholder_satisfaction >= 0.9
        low_value_realization = outcome.value_realization_ratio < 0.5

        inconsistent = high_satisfaction and low_value_realization

        if inconsistent:
            self.logger.info(
                f"📊 Consistency issue detected: satisfaction={outcome.stakeholder_satisfaction:.3f} "
                f"but value_realization={outcome.value_realization_ratio:.3f}"
            )

        return inconsistent

    def approve_flagged_outcome(
        self, engagement_id: str, reviewer_notes: str = ""
    ) -> bool:
        """
        Week 1 Day 3: Manually approve a flagged outcome for learning
        This would be called by an admin/expert reviewer
        """

        if engagement_id not in self.value_outcomes:
            self.logger.error(f"❌ Outcome not found for approval: {engagement_id}")
            return False

        outcome = self.value_outcomes[engagement_id]

        if not outcome.needs_review:
            self.logger.warning(
                f"⚠️  Outcome {engagement_id} was not flagged for review"
            )
            return False

        # Approve for learning
        outcome.approved_for_learning = True
        outcome.reviewer_notes = reviewer_notes

        # Process the previously excluded outcome
        asyncio.create_task(self._update_model_contributions(outcome))

        self.logger.info(
            f"✅ Outcome {engagement_id} manually approved for learning\n"
            f"   Reviewer notes: {reviewer_notes}\n"
            f"   Will now be included in Bayesian updates"
        )

        return True

    def get_flagged_outcomes_for_review(self) -> List[Dict[str, Any]]:
        """
        Week 1 Day 3: Get all outcomes that need manual review
        """

        flagged_outcomes = []

        for outcome in self.value_outcomes.values():
            if outcome.needs_review and not outcome.approved_for_learning:
                flagged_outcomes.append(
                    {
                        "engagement_id": outcome.engagement_id,
                        "stakeholder_satisfaction": outcome.stakeholder_satisfaction,
                        "value_realization_ratio": outcome.value_realization_ratio,
                        "validation_flags": outcome.validation_flags,
                        "outcome_measured_at": outcome.outcome_measured_at.isoformat(),
                        "models_used": outcome.model_ids_used,
                    }
                )

        self.logger.info(f"📋 Found {len(flagged_outcomes)} outcomes awaiting review")

        return flagged_outcomes


# Global ValueDatabaseIntegrator instance
_value_db_integrator_instance: Optional[ValueDatabaseIntegrator] = None


def get_value_database_integrator() -> ValueDatabaseIntegrator:
    """Get or create global ValueDatabaseIntegrator instance"""
    global _value_db_integrator_instance

    if _value_db_integrator_instance is None:
        _value_db_integrator_instance = ValueDatabaseIntegrator()

    return _value_db_integrator_instance


async def record_engagement_value_outcome(
    engagement_id: str,
    assessment_id: str,
    model_ids_used: List[str],
    realized_metrics: Dict[str, Any],
) -> ValueOutcome:
    """Convenience function to record value outcome"""
    integrator = get_value_database_integrator()
    return await integrator.record_value_outcome(
        engagement_id, assessment_id, model_ids_used, realized_metrics
    )


def get_model_value_insights(model_id: str) -> Dict[str, Any]:
    """Convenience function to get model value performance"""
    integrator = get_value_database_integrator()
    return integrator.get_model_value_performance(model_id)
