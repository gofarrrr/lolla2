"""
User Satisfaction Prediction Service - Domain Service
===================================================

Comprehensive domain service for predicting user satisfaction with arbitration results
based on various analytical factors including preference alignment, consensus strength,
criteria coverage, and cognitive diversity.

This service provides intelligent prediction of user satisfaction to help
improve arbitration quality and user experience.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any
import statistics

from src.services.interfaces.user_satisfaction_prediction_interface import (
    IUserSatisfactionPredictionService,
    UserSatisfactionPredictionError,
)
from src.arbitration.models import (
    DifferentialAnalysis,
    UserWeightingPreferences,
)

logger = logging.getLogger(__name__)


def _safe_len(obj: Any) -> int:
    try:
        return len(obj) if obj is not None else 0
    except Exception:
        return 0


def _safe_dict(obj: Any) -> Dict[Any, Any]:
    return obj if isinstance(obj, dict) else {}


class UserSatisfactionPredictionService(IUserSatisfactionPredictionService):
    """
    Domain service for predicting user satisfaction with arbitration results

    This service analyzes multiple factors to predict how satisfied a user will be
    with the arbitration results, including preference alignment, consensus strength,
    criteria coverage, and cognitive diversity.
    """

    def __init__(self):
        self.logger = logger

        # Weights for different satisfaction factors
        self.factor_weights = {
            "preference_merit_alignment": 0.25,
            "cognitive_diversity": 0.20,
            "complementarity": 0.20,
            "criteria_coverage": 0.20,
            "consensus_strength": 0.15,
        }

        self.logger.info("🎯 UserSatisfactionPrediction domain service initialized")

    def predict_user_satisfaction(
        self,
        differential_analysis: DifferentialAnalysis,
        user_preferences: UserWeightingPreferences,
        perspective_analysis: Dict[str, Any],
    ) -> float:
        """Predict user satisfaction with arbitration results"""
        try:
            self.logger.debug("🔮 Predicting user satisfaction")

            satisfaction_factors = []

            # Factor 1: Alignment between user preferences and merit scores
            preference_merit_alignment = self.calculate_preference_merit_alignment(
                differential_analysis, user_preferences
            )
            satisfaction_factors.append(preference_merit_alignment)

            # Factor 2: Cognitive diversity score
            cognitive_diversity_satisfaction = self.assess_cognitive_diversity_satisfaction(
                perspective_analysis
            )
            satisfaction_factors.append(cognitive_diversity_satisfaction)

            # Factor 3: Complementarity of outputs
            complementarity_score = differential_analysis.complementarity_score
            satisfaction_factors.append(complementarity_score)

            # Factor 4: Coverage of user's decision criteria priorities
            criteria_coverage = self.assess_criteria_coverage(
                differential_analysis, user_preferences
            )
            satisfaction_factors.append(criteria_coverage)

            # Factor 5: Consensus strength on key recommendations
            consensus_strength = self.calculate_consensus_strength(differential_analysis)
            satisfaction_factors.append(consensus_strength * 0.9)  # High consensus generally positive

            # Calculate weighted satisfaction prediction
            prediction = sum(satisfaction_factors) / len(satisfaction_factors)

            # Apply confidence adjustment
            confidence_adjustment = self.calculate_confidence_adjustment(differential_analysis)
            prediction = prediction * confidence_adjustment

            final_prediction = min(1.0, prediction)

            self.logger.debug(
                f"🔮 User satisfaction predicted: {final_prediction:.3f}",
                factors={
                    "preference_merit_alignment": preference_merit_alignment,
                    "cognitive_diversity": cognitive_diversity_satisfaction,
                    "complementarity": complementarity_score,
                    "criteria_coverage": criteria_coverage,
                    "consensus_strength": consensus_strength,
                }
            )

            return final_prediction

        except Exception as e:
            safe_outputs = []
            safe_merit = {}
            try:
                safe_outputs = getattr(differential_analysis, "consultant_outputs", []) or []
            except Exception:
                safe_outputs = []
            try:
                safe_merit = getattr(differential_analysis, "merit_assessments", {}) or {}
            except Exception:
                safe_merit = {}
            raise UserSatisfactionPredictionError(
                f"Failed to predict user satisfaction: {str(e)}",
                {
                    "consultant_count": _safe_len(safe_outputs),
                    "merit_assessments_count": _safe_len(safe_merit),
                }
            )

    def assess_criteria_coverage(
        self,
        differential_analysis: DifferentialAnalysis,
        user_preferences: UserWeightingPreferences,
    ) -> float:
        """Assess how well the arbitration covers user's priority criteria"""
        try:
            total_coverage = 0.0
            total_priority = 0.0

            for criterion, priority in user_preferences.criterion_priorities.items():
                total_priority += priority

                # Check if any consultant scored well on this criterion
                best_score = 0.0
                for role, assessment in differential_analysis.merit_assessments.items():
                    if criterion in assessment.criterion_scores:
                        score = assessment.criterion_scores[criterion].score
                        user_weight = user_preferences.consultant_weights.get(role, 0.0)
                        weighted_score = score * user_weight
                        best_score = max(best_score, weighted_score)

                total_coverage += best_score * priority

            coverage_score = total_coverage / total_priority if total_priority > 0 else 0.5

            self.logger.debug(f"📊 Criteria coverage assessed: {coverage_score:.3f}")
            return coverage_score

        except Exception as e:
            safe_criteria = {}
            safe_merit = {}
            try:
                safe_criteria = getattr(user_preferences, "criterion_priorities", {}) or {}
            except Exception:
                safe_criteria = {}
            try:
                safe_merit = getattr(differential_analysis, "merit_assessments", {}) or {}
            except Exception:
                safe_merit = {}
            raise UserSatisfactionPredictionError(
                f"Failed to assess criteria coverage: {str(e)}",
                {
                    "criteria_count": _safe_len(safe_criteria),
                    "merit_assessments_count": _safe_len(safe_merit),
                }
            )

    def calculate_consensus_strength(
        self, differential_analysis: DifferentialAnalysis
    ) -> float:
        """Calculate strength of consensus across consultant outputs"""
        try:
            convergent_findings = differential_analysis.convergent_findings
            total_insights = sum(
                len(output.key_insights)
                for output in differential_analysis.consultant_outputs
            )

            if total_insights == 0:
                return 0.0

            # High-consensus findings (independently validated by multiple consultants)
            high_consensus_count = sum(
                1
                for finding in convergent_findings
                if finding.independent_validation and finding.consensus_strength > 0.7
            )

            consensus_ratio = high_consensus_count / total_insights
            consensus_strength = min(1.0, consensus_ratio * 3)  # Scale up since consensus is valuable

            self.logger.debug(f"🤝 Consensus strength calculated: {consensus_strength:.3f}")
            return consensus_strength

        except Exception as e:
            safe_findings = []
            safe_outputs = []
            try:
                safe_findings = getattr(differential_analysis, "convergent_findings", []) or []
            except Exception:
                safe_findings = []
            try:
                safe_outputs = getattr(differential_analysis, "consultant_outputs", []) or []
            except Exception:
                safe_outputs = []
            raise UserSatisfactionPredictionError(
                f"Failed to calculate consensus strength: {str(e)}",
                {
                    "convergent_findings_count": _safe_len(safe_findings),
                    "consultant_count": _safe_len(safe_outputs),
                }
            )

    def calculate_preference_merit_alignment(
        self,
        differential_analysis: DifferentialAnalysis,
        user_preferences: UserWeightingPreferences,
    ) -> float:
        """Calculate alignment between user preferences and merit scores"""
        try:
            preference_merit_alignment = 0.0
            total_weight = 0.0

            for role, user_weight in user_preferences.consultant_weights.items():
                if role in differential_analysis.merit_assessments:
                    merit_score = differential_analysis.merit_assessments[
                        role
                    ].overall_merit_score
                    preference_merit_alignment += user_weight * merit_score
                    total_weight += user_weight

            alignment_score = preference_merit_alignment / total_weight if total_weight > 0 else 0.5

            self.logger.debug(f"⚖️ Preference-merit alignment calculated: {alignment_score:.3f}")
            return alignment_score

        except Exception as e:
            safe_prefs = {}
            safe_merit = {}
            try:
                safe_prefs = getattr(user_preferences, "consultant_weights", {}) or {}
            except Exception:
                safe_prefs = {}
            try:
                safe_merit = getattr(differential_analysis, "merit_assessments", {}) or {}
            except Exception:
                safe_merit = {}
            raise UserSatisfactionPredictionError(
                f"Failed to calculate preference-merit alignment: {str(e)}",
                {
                    "user_preferences_count": _safe_len(safe_prefs),
                    "merit_assessments_count": _safe_len(safe_merit),
                }
            )

    def assess_cognitive_diversity_satisfaction(
        self, perspective_analysis: Dict[str, Any]
    ) -> float:
        """Assess user satisfaction based on cognitive diversity"""
        try:
            cognitive_diversity = perspective_analysis.get("cognitive_diversity_score", 0.5)

            # High diversity generally good, but apply diminishing returns
            diversity_satisfaction = cognitive_diversity * 0.8

            # Additional factors from perspective analysis
            if "perspective_gaps" in perspective_analysis:
                gap_count = len(perspective_analysis["perspective_gaps"])
                # Fewer gaps = higher satisfaction
                gap_penalty = min(0.2, gap_count * 0.05)
                diversity_satisfaction = max(0.0, diversity_satisfaction - gap_penalty)

            self.logger.debug(f"🧠 Cognitive diversity satisfaction: {diversity_satisfaction:.3f}")
            return diversity_satisfaction

        except Exception as e:
            raise UserSatisfactionPredictionError(
                f"Failed to assess cognitive diversity satisfaction: {str(e)}",
                {"perspective_analysis_keys": list(perspective_analysis.keys())}
            )

    def calculate_confidence_adjustment(
        self, differential_analysis: DifferentialAnalysis
    ) -> float:
        """Calculate confidence adjustment factor for satisfaction prediction"""
        try:
            # Calculate average confidence across consultant outputs
            consultant_outputs = differential_analysis.consultant_outputs

            if not consultant_outputs:
                return 0.5

            avg_confidence = sum(
                output.confidence_level for output in consultant_outputs
            ) / len(consultant_outputs)

            # Apply adjustment curve - high confidence boosts satisfaction
            confidence_adjustment = 0.7 + (avg_confidence * 0.3)  # Range: 0.7 to 1.0

            self.logger.debug(f"🎯 Confidence adjustment calculated: {confidence_adjustment:.3f}")
            return confidence_adjustment

        except Exception as e:
            safe_outputs = []
            try:
                safe_outputs = getattr(differential_analysis, "consultant_outputs", []) or []
            except Exception:
                safe_outputs = []
            raise UserSatisfactionPredictionError(
                f"Failed to calculate confidence adjustment: {str(e)}",
                {"consultant_count": _safe_len(safe_outputs)}
            )

    def generate_satisfaction_factors_breakdown(
        self,
        differential_analysis: DifferentialAnalysis,
        user_preferences: UserWeightingPreferences,
        perspective_analysis: Dict[str, Any],
    ) -> Dict[str, float]:
        """Generate detailed breakdown of satisfaction factors"""
        try:
            factors = {}

            # Calculate each factor individually
            factors["preference_merit_alignment"] = self.calculate_preference_merit_alignment(
                differential_analysis, user_preferences
            )

            factors["cognitive_diversity_satisfaction"] = self.assess_cognitive_diversity_satisfaction(
                perspective_analysis
            )

            factors["complementarity_score"] = differential_analysis.complementarity_score

            factors["criteria_coverage"] = self.assess_criteria_coverage(
                differential_analysis, user_preferences
            )

            factors["consensus_strength"] = self.calculate_consensus_strength(differential_analysis)

            factors["confidence_adjustment"] = self.calculate_confidence_adjustment(
                differential_analysis
            )

            # Calculate weighted overall score
            weighted_factors = [
                factors["preference_merit_alignment"] * self.factor_weights["preference_merit_alignment"],
                factors["cognitive_diversity_satisfaction"] * self.factor_weights["cognitive_diversity"],
                factors["complementarity_score"] * self.factor_weights["complementarity"],
                factors["criteria_coverage"] * self.factor_weights["criteria_coverage"],
                factors["consensus_strength"] * self.factor_weights["consensus_strength"],
            ]

            factors["weighted_overall_score"] = sum(weighted_factors)
            factors["final_prediction"] = min(1.0, factors["weighted_overall_score"] * factors["confidence_adjustment"])

            self.logger.debug("📋 Satisfaction factors breakdown generated")
            return factors

        except Exception as e:
            raise UserSatisfactionPredictionError(
                f"Failed to generate satisfaction factors breakdown: {str(e)}",
                {
                    "consultant_count": len(differential_analysis.consultant_outputs),
                    "perspective_analysis_keys": list(perspective_analysis.keys()),
                }
            )

    def predict_satisfaction_confidence_interval(
        self,
        differential_analysis: DifferentialAnalysis,
        user_preferences: UserWeightingPreferences,
        perspective_analysis: Dict[str, Any],
    ) -> Dict[str, float]:
        """Predict confidence interval for satisfaction prediction"""
        try:
            # Get the base prediction
            base_prediction = self.predict_user_satisfaction(
                differential_analysis, user_preferences, perspective_analysis
            )

            # Calculate uncertainty factors
            uncertainty_factors = []

            # Factor 1: Variance in consultant confidence levels
            confidence_levels = [output.confidence_level for output in differential_analysis.consultant_outputs]
            confidence_variance = statistics.variance(confidence_levels) if len(confidence_levels) > 1 else 0.0
            uncertainty_factors.append(confidence_variance)

            # Factor 2: Merit assessment variance
            merit_scores = [
                assessment.overall_merit_score
                for assessment in differential_analysis.merit_assessments.values()
            ]
            merit_variance = statistics.variance(merit_scores) if len(merit_scores) > 1 else 0.0
            uncertainty_factors.append(merit_variance)

            # Factor 3: Consensus level (low consensus = high uncertainty)
            consensus_uncertainty = 1.0 - differential_analysis.consensus_level
            uncertainty_factors.append(consensus_uncertainty)

            # Calculate overall uncertainty
            overall_uncertainty = sum(uncertainty_factors) / len(uncertainty_factors)

            # Calculate confidence interval
            margin = overall_uncertainty * 0.2  # Max margin of ±20%

            confidence_interval = {
                "prediction": base_prediction,
                "lower_bound": max(0.0, base_prediction - margin),
                "upper_bound": min(1.0, base_prediction + margin),
                "margin_of_error": margin,
                "confidence_level": 0.8,  # 80% confidence level
                "uncertainty_factors": {
                    "confidence_variance": confidence_variance,
                    "merit_variance": merit_variance,
                    "consensus_uncertainty": consensus_uncertainty,
                    "overall_uncertainty": overall_uncertainty,
                }
            }

            self.logger.debug(
                f"📊 Satisfaction confidence interval: {base_prediction:.3f} ±{margin:.3f}"
            )

            return confidence_interval

        except Exception as e:
            raise UserSatisfactionPredictionError(
                f"Failed to predict satisfaction confidence interval: {str(e)}",
                {
                    "consultant_count": len(differential_analysis.consultant_outputs),
                    "merit_assessments_count": len(differential_analysis.merit_assessments),
                }
            )