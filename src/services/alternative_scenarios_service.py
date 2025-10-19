"""
Alternative Scenarios Service - Domain Service
==============================================

Comprehensive domain service for generating alternative weighting scenarios
and implementation guidance to help users explore different decision-making approaches.

This service provides scenario planning and implementation recommendations
based on consultant outputs and user preferences.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional

from src.services.interfaces.alternative_scenarios_interface import (
    IAlternativeScenariosService,
    AlternativeScenariosError,
)
from src.arbitration.models import (
    ConsultantOutput,
    ConsultantRole,
    DifferentialAnalysis,
    UserWeightingPreferences,
)

logger = logging.getLogger(__name__)


def _safe_dict(obj: Any) -> Dict[Any, Any]:
    """Return obj if it's a dict, otherwise an empty dict.

    Defensive against loose Mock objects where attribute access returns a Mock.
    """
    return obj if isinstance(obj, dict) else {}


def _safe_len(obj: Any) -> int:
    try:
        return len(obj) if obj is not None else 0
    except Exception:
        return 0


class AlternativeScenariosService(IAlternativeScenariosService):
    """
    Domain service for generating alternative weighting scenarios and implementation guidance

    This service helps users explore different decision-making approaches by generating
    alternative weighting scenarios and comprehensive implementation guidance.
    """

    def __init__(self):
        self.logger = logger
        self.logger.info("🎯 AlternativeScenarios domain service initialized")

    async def generate_alternative_scenarios(
        self,
        consultant_outputs: List[ConsultantOutput],
        user_preferences: UserWeightingPreferences,
        merit_assessments: Optional[Dict[ConsultantRole, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Generate alternative weighting scenarios for user consideration"""
        try:
            self.logger.debug("🎭 Generating alternative scenarios")

            scenarios = []

            # Scenario 1: Equal weighting
            equal_scenario = self.generate_equal_weighting_scenario(consultant_outputs)
            scenarios.append(equal_scenario)

            # Scenario 2: Merit-based weighting
            if merit_assessments:
                merit_scenario = self.generate_merit_based_scenario(merit_assessments)
                if merit_scenario:
                    scenarios.append(merit_scenario)

            # Scenario 3: Conservative weighting
            conservative_scenario = self.generate_conservative_scenario(user_preferences)
            if conservative_scenario:
                scenarios.append(conservative_scenario)

            # Scenario 4: Expertise-focused weighting
            expertise_scenario = self._generate_expertise_focused_scenario(
                consultant_outputs, user_preferences
            )
            if expertise_scenario:
                scenarios.append(expertise_scenario)

            self.logger.debug(f"🎭 Generated {len(scenarios)} alternative scenarios")
            return scenarios

        except Exception as e:
            raise AlternativeScenariosError(
                f"Failed to generate alternative scenarios: {str(e)}",
                {
                    "consultant_count": _safe_len(consultant_outputs),
                    "has_merit_assessments": merit_assessments is not None,
                }
            )

    async def generate_implementation_guidance(
        self,
        consultant_outputs: List[ConsultantOutput],
        user_preferences: UserWeightingPreferences,
        differential_analysis: DifferentialAnalysis,
    ) -> Dict[str, Any]:
        """Generate implementation guidance based on arbitration results"""
        try:
            self.logger.debug("📋 Generating implementation guidance")

            # Generate priority order
            priority_order = self.generate_priority_order(consultant_outputs, user_preferences)

            # Generate success metrics
            success_metrics = self.generate_success_metrics(consultant_outputs, user_preferences)

            # Generate monitoring recommendations
            monitoring_recommendations = self.generate_monitoring_recommendations(
                consultant_outputs, differential_analysis
            )

            # Derive simple implementation steps from top priorities (fallback-friendly)
            implementation_steps = [
                f"Execute priority: {rec}" for rec in (priority_order[:3] or [
                    "Assign owners for key recommendations",
                    "Define KPIs and tracking cadence",
                    "Kick off Phase 1 actions",
                ])
            ]

            guidance = {
                "priority_order": priority_order,
                "success_metrics": success_metrics,
                "monitoring_recommendations": monitoring_recommendations,
                "implementation_steps": implementation_steps,
                "implementation_timeline": self._generate_implementation_timeline(priority_order),
                "risk_mitigation_steps": self._generate_risk_mitigation_steps(
                    consultant_outputs, differential_analysis
                ),
            }

            self.logger.debug("📋 Implementation guidance generated")
            return guidance

        except Exception as e:
            raise AlternativeScenariosError(
                f"Failed to generate implementation guidance: {str(e)}",
                {
                    "consultant_count": len(consultant_outputs),
                    "differential_analysis_id": getattr(differential_analysis, 'id', 'unknown'),
                }
            )

    def generate_equal_weighting_scenario(
        self, consultant_outputs: List[ConsultantOutput]
    ) -> Dict[str, Any]:
        """Generate equal weighting scenario"""
        try:
            equal_weight = 1.0 / len(consultant_outputs)
            equal_weights = {
                output.consultant_role: equal_weight for output in consultant_outputs
            }

            return {
                "scenario_name": "Equal Weighting",
                "description": "All consultants weighted equally",
                "consultant_weights": equal_weights,
                "impact_summary": "Balanced approach with no bias toward any consultant",
                "best_for": "When unsure about preferences or want comprehensive coverage",
                "confidence_level": 0.7,
                "risk_level": "low",
            }

        except Exception as e:
            raise AlternativeScenariosError(
                f"Failed to generate equal weighting scenario: {str(e)}",
                {"consultant_count": _safe_len(consultant_outputs)}
            )

    def generate_merit_based_scenario(
        self, merit_assessments: Dict[ConsultantRole, Any]
    ) -> Dict[str, Any]:
        """Generate merit-based weighting scenario"""
        try:
            merit_scores = {
                role: assessment.overall_merit_score
                for role, assessment in merit_assessments.items()
                if hasattr(assessment, "overall_merit_score")
            }

            if not merit_scores:
                return None

            total_merit = sum(merit_scores.values())
            if total_merit == 0:
                return None

            merit_weights = {
                role: score / total_merit for role, score in merit_scores.items()
            }

            return {
                "scenario_name": "Merit-Based Weighting",
                "description": "Weighting based purely on analysis quality",
                "consultant_weights": merit_weights,
                "impact_summary": "Prioritizes highest quality analysis regardless of approach",
                "best_for": "When quality and rigor are the primary concerns",
                "confidence_level": 0.85,
                "risk_level": "low",
                "merit_distribution": merit_scores,
            }

        except Exception as e:
            raise AlternativeScenariosError(
                f"Failed to generate merit-based scenario: {str(e)}",
                {"merit_assessments_count": len(merit_assessments)}
            )

    def generate_conservative_scenario(
        self, user_preferences: UserWeightingPreferences
    ) -> Dict[str, Any]:
        """Generate conservative (risk-focused) weighting scenario"""
        try:
            conservative_weights = user_preferences.consultant_weights.copy()

            if ConsultantRole.DEVIL_ADVOCATE in conservative_weights:
                # Boost devil's advocate weight
                conservative_weights[ConsultantRole.DEVIL_ADVOCATE] *= 1.5

                # Normalize
                total_weight = sum(conservative_weights.values())
                conservative_weights = {
                    role: weight / total_weight
                    for role, weight in conservative_weights.items()
                }

                return {
                    "scenario_name": "Risk-Conscious Weighting",
                    "description": "Higher weight on risk identification and mitigation",
                    "consultant_weights": conservative_weights,
                    "impact_summary": "Emphasizes thorough risk assessment and conservative approaches",
                    "best_for": "High-stakes decisions or risk-averse contexts",
                    "confidence_level": 0.8,
                    "risk_level": "very_low",
                }

            return None

        except Exception as e:
            raise AlternativeScenariosError(
                f"Failed to generate conservative scenario: {str(e)}",
                {"user_preferences_count": len(user_preferences.consultant_weights)}
            )

    def generate_priority_order(
        self,
        consultant_outputs: List[ConsultantOutput],
        user_preferences: UserWeightingPreferences,
    ) -> List[str]:
        """Generate priority order for recommendations"""
        try:
            all_recommendations = []

            for output in consultant_outputs:
                weight = user_preferences.consultant_weights.get(
                    output.consultant_role, 0.0
                )
                for rec in output.recommendations:
                    all_recommendations.append(
                        {
                            "recommendation": rec,
                            "weight": weight,
                            "consultant": output.consultant_role,
                            "confidence": output.confidence_level,
                            "combined_score": weight * output.confidence_level,
                        }
                    )

            # Sort by combined score (weight * confidence)
            all_recommendations.sort(key=lambda x: x["combined_score"], reverse=True)

            # Return top 8 recommendations
            priority_order = [item["recommendation"] for item in all_recommendations[:8]]

            self.logger.debug(f"📋 Generated priority order with {len(priority_order)} items")
            return priority_order

        except Exception as e:
            raise AlternativeScenariosError(
                f"Failed to generate priority order: {str(e)}",
                {"consultant_count": len(consultant_outputs)}
            )

    def generate_success_metrics(
        self,
        consultant_outputs: List[ConsultantOutput],
        user_preferences: UserWeightingPreferences,
    ) -> List[str]:
        """Generate success metrics based on consultant insights"""
        try:
            success_metrics = []

            # Extract metrics from highly weighted consultants
            for output in consultant_outputs:
                weight = user_preferences.consultant_weights.get(
                    output.consultant_role, 0.0
                )
                if weight > 0.3:  # Only from highly weighted consultants
                    for insight in output.key_insights[:2]:
                        if "metric" in insight.lower() or "measure" in insight.lower():
                            success_metrics.append(insight)

            # Default metrics if none found
            if not success_metrics:
                success_metrics = [
                    "Monitor implementation progress weekly",
                    "Track key performance indicators relevant to recommendations",
                    "Measure stakeholder satisfaction with changes",
                    "Assess return on investment of implemented changes",
                    "Monitor for unintended consequences",
                ]

            # Limit to top 5 metrics
            final_metrics = success_metrics[:5]

            self.logger.debug(f"📊 Generated {len(final_metrics)} success metrics")
            return final_metrics

        except Exception as e:
            raise AlternativeScenariosError(
                f"Failed to generate success metrics: {str(e)}",
                {"consultant_count": len(consultant_outputs)}
            )

    def generate_monitoring_recommendations(
        self,
        consultant_outputs: List[ConsultantOutput],
        differential_analysis: DifferentialAnalysis,
    ) -> List[str]:
        """Generate monitoring recommendations based on analysis"""
        try:
            monitoring_recommendations = []

            # Risk-based monitoring
            merit_assessments = _safe_dict(getattr(differential_analysis, "merit_assessments", {}))
            if merit_assessments:
                high_risk_consultants = [
                    role
                    for role, assessment in merit_assessments.items()
                    if any(
                        "risk" in criterion.value.lower() and score.score > 0.7
                        for criterion, score in assessment.criterion_scores.items()
                    )
                ]

                if high_risk_consultants:
                    monitoring_recommendations.append(
                        "Implement risk monitoring dashboard based on identified concerns"
                    )

            # Evidence-based monitoring
            evidence_heavy_consultants = [
                output.consultant_role
                for output in consultant_outputs
                if output.research_depth_score > 0.8
            ]

            if evidence_heavy_consultants:
                monitoring_recommendations.append(
                    "Track evidence-based metrics identified in research"
                )

            # Consensus-based monitoring
            if differential_analysis.consensus_level < 0.6:
                monitoring_recommendations.append(
                    "Monitor for divergent outcomes due to low consensus"
                )

            # Default monitoring if none specific
            if not monitoring_recommendations:
                monitoring_recommendations = [
                    "Establish baseline measurements before implementation",
                    "Set up regular progress review checkpoints",
                    "Monitor for unintended consequences of changes",
                ]

            # Add general best practices
            monitoring_recommendations.extend([
                "Create feedback loops for continuous improvement",
                "Document lessons learned throughout implementation",
            ])

            self.logger.debug(f"📹 Generated {len(monitoring_recommendations)} monitoring recommendations")
            return monitoring_recommendations

        except Exception as e:
            raise AlternativeScenariosError(
                f"Failed to generate monitoring recommendations: {str(e)}",
                {
                    "consultant_count": len(consultant_outputs),
                    "merit_assessments_count": len(_safe_dict(getattr(differential_analysis, "merit_assessments", {}))),
                }
            )

    def assess_scenario_impact(
        self,
        scenario: Dict[str, Any],
        consultant_outputs: List[ConsultantOutput],
    ) -> Dict[str, Any]:
        """Assess the impact of a specific weighting scenario"""
        try:
            scenario_weights = scenario["consultant_weights"]

            # Calculate weighted average confidence
            weighted_confidence = 0.0
            total_weight = 0.0

            for output in consultant_outputs:
                weight = scenario_weights.get(output.consultant_role, 0.0)
                weighted_confidence += weight * output.confidence_level
                total_weight += weight

            avg_confidence = weighted_confidence / total_weight if total_weight > 0 else 0.0

            # Calculate diversity score
            weight_variance = sum(
                (weight - (1.0 / len(scenario_weights))) ** 2
                for weight in scenario_weights.values()
            ) / len(scenario_weights)

            diversity_score = 1.0 - weight_variance  # Lower variance = higher diversity

            # Calculate risk assessment
            devil_advocate_weight = scenario_weights.get(ConsultantRole.DEVIL_ADVOCATE, 0.0)
            risk_coverage = min(1.0, devil_advocate_weight * 2)  # Scale devil's advocate weight

            impact_assessment = {
                "scenario_name": scenario["scenario_name"],
                "weighted_confidence": avg_confidence,
                "diversity_score": diversity_score,
                "risk_coverage": risk_coverage,
                "balance_score": (avg_confidence + diversity_score + risk_coverage) / 3,
                "strengths": self._identify_scenario_strengths(scenario, consultant_outputs),
                "potential_risks": self._identify_scenario_risks(scenario, consultant_outputs),
            }

            return impact_assessment

        except Exception as e:
            raise AlternativeScenariosError(
                f"Failed to assess scenario impact: {str(e)}",
                {"scenario_name": scenario.get("scenario_name", "unknown")}
            )

    def generate_scenario_comparison(
        self,
        scenarios: List[Dict[str, Any]],
        consultant_outputs: List[ConsultantOutput],
    ) -> Dict[str, Any]:
        """Generate comparison between multiple scenarios"""
        try:
            scenario_assessments = []

            for scenario in scenarios:
                assessment = self.assess_scenario_impact(scenario, consultant_outputs)
                scenario_assessments.append(assessment)

            # Find best scenario for different criteria
            best_confidence = max(scenario_assessments, key=lambda x: x["weighted_confidence"])
            best_diversity = max(scenario_assessments, key=lambda x: x["diversity_score"])
            best_risk_coverage = max(scenario_assessments, key=lambda x: x["risk_coverage"])
            best_overall = max(scenario_assessments, key=lambda x: x["balance_score"])

            comparison = {
                "total_scenarios": len(scenarios),
                "scenario_assessments": scenario_assessments,
                "recommendations": {
                    "highest_confidence": best_confidence["scenario_name"],
                    "most_diverse": best_diversity["scenario_name"],
                    "best_risk_coverage": best_risk_coverage["scenario_name"],
                    "best_overall_balance": best_overall["scenario_name"],
                },
                "summary": self._generate_comparison_summary(scenario_assessments),
            }

            return comparison

        except Exception as e:
            raise AlternativeScenariosError(
                f"Failed to generate scenario comparison: {str(e)}",
                {"scenarios_count": len(scenarios)}
            )

    # Helper methods

    def _generate_expertise_focused_scenario(
        self,
        consultant_outputs: List[ConsultantOutput],
        user_preferences: UserWeightingPreferences,
    ) -> Optional[Dict[str, Any]]:
        """Generate expertise-focused weighting scenario"""
        try:
            # Find consultant with highest research depth
            expertise_scores = {
                output.consultant_role: output.research_depth_score
                for output in consultant_outputs
            }

            if not expertise_scores:
                return None

            max_expertise = max(expertise_scores.values())
            if max_expertise <= 0.7:  # Only create if there's a clear expertise leader
                return None

            # Weight heavily toward highest expertise
            expertise_weights = {}
            total_score = sum(expertise_scores.values())

            for role, score in expertise_scores.items():
                # Amplify differences
                amplified_score = score ** 2
                expertise_weights[role] = amplified_score

            # Normalize
            total_amplified = sum(expertise_weights.values())
            expertise_weights = {
                role: weight / total_amplified
                for role, weight in expertise_weights.items()
            }

            return {
                "scenario_name": "Expertise-Focused Weighting",
                "description": "Weighting based on research depth and expertise",
                "consultant_weights": expertise_weights,
                "impact_summary": "Prioritizes consultants with deepest research and expertise",
                "best_for": "Complex decisions requiring deep domain knowledge",
                "confidence_level": 0.9,
                "risk_level": "medium",
                "expertise_scores": expertise_scores,
            }

        except Exception as e:
            self.logger.warning(f"Failed to generate expertise-focused scenario: {str(e)}")
            return None

    def _generate_implementation_timeline(self, priority_order: List[str]) -> Dict[str, Any]:
        """Generate implementation timeline based on priority order"""
        phases = []

        # Phase 1: Immediate actions (top 2-3 recommendations)
        immediate_actions = priority_order[:3]
        phases.append({
            "phase": "Immediate (0-30 days)",
            "actions": immediate_actions,
            "focus": "Quick wins and foundational changes",
        })

        # Phase 2: Short-term actions (next 2-3 recommendations)
        short_term_actions = priority_order[3:6]
        if short_term_actions:
            phases.append({
                "phase": "Short-term (1-3 months)",
                "actions": short_term_actions,
                "focus": "Building momentum and establishing systems",
            })

        # Phase 3: Long-term actions (remaining recommendations)
        long_term_actions = priority_order[6:]
        if long_term_actions:
            phases.append({
                "phase": "Long-term (3+ months)",
                "actions": long_term_actions,
                "focus": "Strategic implementation and optimization",
            })

        return {
            "phases": phases,
            "total_recommendations": len(priority_order),
            "estimated_duration": "3-6 months for full implementation",
        }

    def _generate_risk_mitigation_steps(
        self,
        consultant_outputs: List[ConsultantOutput],
        differential_analysis: DifferentialAnalysis,
    ) -> List[str]:
        """Generate risk mitigation steps"""
        risk_steps = []

        # Low consensus risks
        if differential_analysis.consensus_level < 0.6:
            risk_steps.append("Plan for multiple contingencies due to low consultant consensus")

        # High-risk consultant recommendations
        high_risk_outputs = [
            output for output in consultant_outputs
            if any("risk" in insight.lower() for insight in output.key_insights)
        ]

        if high_risk_outputs:
            risk_steps.append("Implement enhanced monitoring for high-risk recommendations")

        # Low confidence risks
        low_confidence_outputs = [
            output for output in consultant_outputs
            if output.confidence_level < 0.7
        ]

        if low_confidence_outputs:
            risk_steps.append("Conduct additional validation for low-confidence recommendations")

        # Default risk mitigation
        if not risk_steps:
            risk_steps = [
                "Establish clear success criteria before implementation",
                "Create rollback plans for major changes",
                "Implement gradual rollout where possible",
            ]

        return risk_steps

    def _identify_scenario_strengths(
        self, scenario: Dict[str, Any], consultant_outputs: List[ConsultantOutput]
    ) -> List[str]:
        """Identify strengths of a scenario"""
        strengths = []

        scenario_weights = scenario["consultant_weights"]

        # Check for balanced weighting
        weight_variance = sum(
            (weight - (1.0 / len(scenario_weights))) ** 2
            for weight in scenario_weights.values()
        ) / len(scenario_weights)

        if weight_variance < 0.1:
            strengths.append("Balanced approach reduces single-point-of-failure risk")

        # Check for high-confidence weighting
        high_confidence_weight = sum(
            weight for output in consultant_outputs
            for role, weight in scenario_weights.items()
            if output.consultant_role == role and output.confidence_level > 0.8
        )

        if high_confidence_weight > 0.6:
            strengths.append("Emphasizes high-confidence recommendations")

        # Check for risk awareness
        devil_advocate_weight = scenario_weights.get(ConsultantRole.DEVIL_ADVOCATE, 0.0)
        if devil_advocate_weight > 0.3:
            strengths.append("Strong risk identification and mitigation focus")

        return strengths or ["Provides alternative perspective on decision weighting"]

    def _identify_scenario_risks(
        self, scenario: Dict[str, Any], consultant_outputs: List[ConsultantOutput]
    ) -> List[str]:
        """Identify potential risks of a scenario"""
        risks = []

        scenario_weights = scenario["consultant_weights"]

        # Check for extreme weighting
        max_weight = max(scenario_weights.values())
        if max_weight > 0.7:
            risks.append("Heavy reliance on single consultant perspective")

        # Check for low risk awareness
        devil_advocate_weight = scenario_weights.get(ConsultantRole.DEVIL_ADVOCATE, 0.0)
        if devil_advocate_weight < 0.15:
            risks.append("Limited risk identification and mitigation")

        # Check for low diversity
        weight_variance = sum(
            (weight - (1.0 / len(scenario_weights))) ** 2
            for weight in scenario_weights.values()
        ) / len(scenario_weights)

        if weight_variance > 0.3:
            risks.append("Unbalanced weighting may miss important perspectives")

        return risks or ["Standard implementation risks apply"]

    def _generate_comparison_summary(self, assessments: List[Dict[str, Any]]) -> str:
        """Generate summary of scenario comparison"""
        avg_confidence = sum(a["weighted_confidence"] for a in assessments) / len(assessments)
        avg_diversity = sum(a["diversity_score"] for a in assessments) / len(assessments)
        avg_risk_coverage = sum(a["risk_coverage"] for a in assessments) / len(assessments)

        summary_parts = []

        if avg_confidence > 0.8:
            summary_parts.append("scenarios generally show high confidence")
        elif avg_confidence < 0.6:
            summary_parts.append("scenarios show moderate confidence levels")

        if avg_diversity > 0.7:
            summary_parts.append("good balance across consultant perspectives")
        elif avg_diversity < 0.5:
            summary_parts.append("some scenarios heavily favor specific consultants")

        if avg_risk_coverage > 0.6:
            summary_parts.append("adequate risk identification and mitigation")
        else:
            summary_parts.append("consider enhancing risk assessment approaches")

        return f"Analysis of {len(assessments)} scenarios shows " + ", ".join(summary_parts) + "."