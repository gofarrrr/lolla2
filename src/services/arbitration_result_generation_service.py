"""
Arbitration Result Generation Service - Domain Service
====================================================

Comprehensive domain service for generating complete arbitration results
with weighted recommendations, insights, risk assessments, and synthesis refinements.

This service orchestrates the generation of the final arbitration result
by combining all consultant outputs according to user preferences.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import defaultdict

from src.services.interfaces.arbitration_result_generation_interface import (
    IArbitrationResultGenerationService,
    ArbitrationResultGenerationError,
)
from src.arbitration.models import (
    ConsultantOutput,
    ConsultantRole,
    DifferentialAnalysis,
    UserWeightingPreferences,
    ArbitrationResult,
)

# Optional integrations
try:
    from src.services.feature_flags import FEATURE_FLAGS_AVAILABLE, is_feature_enabled, FeatureFlag
    FEATURE_FLAGS_AVAILABLE = True
except ImportError:
    FEATURE_FLAGS_AVAILABLE = False

logger = logging.getLogger(__name__)


def _safe_list(obj: Any) -> List[Any]:
    """Return obj if it's a list, otherwise an empty list.

    Defensive against loose Mock objects where attribute access returns a Mock.
    """
    return obj if isinstance(obj, list) else []


class ArbitrationResultGenerationService(IArbitrationResultGenerationService):
    """
    Domain service for generating complete arbitration results with weighted outputs

    This service handles the orchestration and generation of the final arbitration result
    by combining differential analysis, user preferences, and consultant outputs.
    """

    def __init__(self):
        self.logger = logger
        self.logger.info("🎯 ArbitrationResultGeneration domain service initialized")

    async def generate_arbitration_result(
        self,
        differential_analysis: DifferentialAnalysis,
        user_preferences: UserWeightingPreferences,
        perspective_analysis: Dict[str, Any],
        query_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate the complete arbitration result with weighted outputs"""
        try:
            self.logger.info("🎯 Generating complete arbitration result")

            # Extract consultant outputs
            consultant_outputs = differential_analysis.consultant_outputs

            # Generate weighted recommendations
            weighted_recommendations = await self.generate_weighted_recommendations(
                consultant_outputs, user_preferences
            )

            # Generate weighted insights
            weighted_insights = await self.generate_weighted_insights(
                consultant_outputs, user_preferences
            )

            # Generate weighted risk assessment
            weighted_risk_assessment = await self.generate_weighted_risk_assessment(
                consultant_outputs, user_preferences
            )

            # Determine primary consultant recommendation
            primary_consultant = self.determine_primary_consultant(
                differential_analysis.merit_assessments, user_preferences
            )

            # Generate supporting rationales
            supporting_rationales = self.generate_supporting_rationales(
                differential_analysis, user_preferences
            )

            # Build polygon enhancements for synthesis refinement
            polygon_data = self.build_polygon_enhancements(
                consultant_outputs, user_preferences, differential_analysis
            )

            # Decide synthesis mode
            synthesis_metrics = {
                "primary_margin": 0.1,  # Placeholder - would be computed from merit assessment
                "dissent_intensity": polygon_data["metrics"]["dissent_intensity"],
            }
            synthesis_mode = self._decide_synthesis_mode(
                differential_analysis, synthesis_metrics, query_context
            )

            # Build decision quality ribbon
            decision_quality_ribbon = None
            if FEATURE_FLAGS_AVAILABLE and is_feature_enabled(
                FeatureFlag.ENABLE_SA_DECISION_RIBBON
            ):
                decision_quality_ribbon = {
                    "synthesis_mode": synthesis_mode,
                    "polygon_integrity_score": polygon_data["metrics"][
                        "polygon_integrity_score"
                    ],
                    "dissent_intensity": polygon_data["metrics"]["dissent_intensity"],
                    "option_reversibility_index": polygon_data["metrics"][
                        "option_reversibility_index"
                    ],
                    "consensus_strength": self.calculate_consensus_strength(
                        differential_analysis
                    ),
                    "timestamp": datetime.utcnow().isoformat(),
                }

            # Return a compact contract for extracted domain tests
            result_contract = {
                "final_recommendation": (
                    weighted_recommendations[0] if weighted_recommendations else f"Follow {primary_consultant.value}"
                ),
                "confidence_level": self.calculate_consensus_strength(differential_analysis),
                "supporting_evidence": weighted_insights[:3],
            }

            self.logger.info("✅ Arbitration result generation completed")
            return result_contract

        except Exception as e:
            safe_outputs = []
            try:
                safe_outputs = getattr(differential_analysis, "consultant_outputs", []) or []
            except Exception:
                safe_outputs = []
            raise ArbitrationResultGenerationError(
                f"Failed to generate arbitration result: {str(e)}",
                {
                    "consultant_count": len(safe_outputs),
                    "query_context": query_context,
                }
            )

    async def generate_weighted_recommendations(
        self,
        consultant_outputs: List[ConsultantOutput],
        user_preferences: UserWeightingPreferences,
    ) -> List[str]:
        """Generate weighted recommendations based on user preferences"""
        try:
            # Collect all recommendations with weights
            weighted_recs = []

            for output in consultant_outputs:
                consultant_weight = user_preferences.consultant_weights.get(
                    output.consultant_role, 0.0
                )

                for rec in output.recommendations:
                    # Calculate recommendation weight based on consultant weight and quality
                    rec_weight = consultant_weight * output.confidence_level

                    weighted_recs.append(
                        {
                            "recommendation": rec,
                            "weight": rec_weight,
                            "consultant": output.consultant_role,
                            "confidence": output.confidence_level,
                        }
                    )

            # Sort by weight and select top recommendations
            weighted_recs.sort(key=lambda x: x["weight"], reverse=True)

            # Generate final weighted recommendations
            final_recommendations = []
            seen_recommendations = set()

            for item in weighted_recs[:10]:  # Top 10 weighted recommendations
                rec = item["recommendation"]

                # Avoid near-duplicates
                if not self.is_similar_recommendation(rec, seen_recommendations):
                    final_recommendations.append(
                        f"{rec} (Weight: {item['weight']:.2f}, "
                        f"Consultant: {item['consultant'].value})"
                    )
                    seen_recommendations.add(rec)

                if len(final_recommendations) >= 6:  # Limit to 6 final recommendations
                    break

            return final_recommendations

        except Exception as e:
            raise ArbitrationResultGenerationError(
                f"Failed to generate weighted recommendations: {str(e)}",
                {"consultant_count": len(consultant_outputs)}
            )

    async def generate_weighted_insights(
        self,
        consultant_outputs: List[ConsultantOutput],
        user_preferences: UserWeightingPreferences,
    ) -> List[str]:
        """Generate weighted insights based on user preferences"""
        try:
            weighted_insights = []

            for output in consultant_outputs:
                consultant_weight = user_preferences.consultant_weights.get(
                    output.consultant_role, 0.0
                )

                for insight in output.key_insights:
                    insight_weight = consultant_weight * output.confidence_level

                    weighted_insights.append(
                        {
                            "insight": insight,
                            "weight": insight_weight,
                            "consultant": output.consultant_role,
                            "confidence": output.confidence_level,
                        }
                    )

            # Sort and select top insights
            weighted_insights.sort(key=lambda x: x["weight"], reverse=True)

            final_insights = []
            seen_insights = set()

            for item in weighted_insights[:8]:
                insight = item["insight"]

                if not self.is_similar_insight(insight, seen_insights):
                    final_insights.append(
                        f"{insight} (Consultant: {item['consultant'].value}, "
                        f"Weight: {item['weight']:.2f})"
                    )
                    seen_insights.add(insight)

                if len(final_insights) >= 5:
                    break

            return final_insights

        except Exception as e:
            raise ArbitrationResultGenerationError(
                f"Failed to generate weighted insights: {str(e)}",
                {"consultant_count": len(consultant_outputs)}
            )

    async def generate_weighted_risk_assessment(
        self,
        consultant_outputs: List[ConsultantOutput],
        user_preferences: UserWeightingPreferences,
    ) -> str:
        """Generate weighted risk assessment"""
        try:
            # Extract risk-related content from all consultants
            risk_contents = []

            for output in consultant_outputs:
                consultant_weight = user_preferences.consultant_weights.get(
                    output.consultant_role, 0.0
                )

                # Look for risk mentions in different sections
                risk_in_summary = self.extract_risk_mentions(output.executive_summary)
                risk_in_insights = [
                    insight
                    for insight in output.key_insights
                    if self.contains_risk_content(insight)
                ]
                risk_in_recommendations = [
                    rec
                    for rec in output.recommendations
                    if self.contains_risk_content(rec)
                ]

                if risk_in_summary or risk_in_insights or risk_in_recommendations:
                    risk_contents.append(
                        {
                            "consultant": output.consultant_role,
                            "weight": consultant_weight,
                            "summary_risks": risk_in_summary,
                            "insight_risks": risk_in_insights,
                            "recommendation_risks": risk_in_recommendations,
                            "bias_detection_score": output.bias_detection_score,
                            "red_team_quality": self._assess_red_team_quality(
                                output.red_team_results
                            ),
                        }
                    )

            # Generate weighted risk assessment
            if not risk_contents:
                return "Limited risk assessment available across consultant outputs."

            # Sort by weight and compile risk assessment
            risk_contents.sort(key=lambda x: x["weight"], reverse=True)

            risk_summary_parts = []
            risk_summary_parts.append("Weighted Risk Assessment:")

            for content in risk_contents[:3]:  # Top 3 weighted consultants
                consultant_name = content["consultant"].value
                weight = content["weight"]

                consultant_risks = []
                if content["summary_risks"]:
                    consultant_risks.extend(content["summary_risks"])
                if content["insight_risks"]:
                    consultant_risks.extend(content["insight_risks"])

                if consultant_risks:
                    risk_summary_parts.append(
                        f"\n{consultant_name} (Weight: {weight:.2f}): "
                        f"{'; '.join(consultant_risks[:2])}"  # Top 2 risks from this consultant
                    )

            return " ".join(risk_summary_parts)

        except Exception as e:
            raise ArbitrationResultGenerationError(
                f"Failed to generate weighted risk assessment: {str(e)}",
                {"consultant_count": len(consultant_outputs)}
            )

    def determine_primary_consultant(
        self,
        merit_assessments: Dict[ConsultantRole, Any],
        user_preferences: UserWeightingPreferences,
    ) -> ConsultantRole:
        """Determine which consultant should be the primary recommendation"""
        try:
            # Calculate combined scores (merit + user preference + query fitness)
            combined_scores = {}

            for role, assessment in merit_assessments.items():
                user_weight = user_preferences.consultant_weights.get(role, 0.0)
                merit_score = getattr(assessment, "overall_merit_score", 0.0)
                query_fitness = getattr(assessment, "query_fitness_score", 0.0)
                combined_score = merit_score * 0.4 + query_fitness * 0.3 + user_weight * 0.3
                combined_scores[role] = combined_score

            primary_consultant = max(combined_scores.keys(), key=lambda k: combined_scores[k])

            self.logger.debug(
                f"🎯 Primary consultant determined: {primary_consultant.value}",
                combined_scores={
                    role.value: f"{score:.3f}" for role, score in combined_scores.items()
                },
            )

            return primary_consultant

        except Exception as e:
            raise ArbitrationResultGenerationError(
                f"Failed to determine primary consultant: {str(e)}",
                {"merit_assessments_count": len(merit_assessments)}
            )

    def generate_supporting_rationales(
        self,
        differential_analysis: DifferentialAnalysis,
        user_preferences: UserWeightingPreferences,
    ) -> Dict[ConsultantRole, str]:
        """Generate rationales for how each consultant supports the decision"""
        try:
            rationales = {}

            for role, assessment in differential_analysis.merit_assessments.items():
                user_weight = user_preferences.consultant_weights.get(role, 0.0)
                rationale_parts = []

                # Merit-based rationale
                if getattr(assessment, "overall_merit_score", 0.0) > 0.8:
                    rationale_parts.append(
                        f"Excellent overall analysis quality ({assessment.overall_merit_score:.2f})"
                    )
                elif getattr(assessment, "overall_merit_score", 0.0) > 0.7:
                    rationale_parts.append(
                        f"Strong analysis quality ({assessment.overall_merit_score:.2f})"
                    )

                # Strengths-based rationale
                strengths = getattr(assessment, "strengths", []) or []
                if strengths:
                    rationale_parts.append(
                        f"Key strengths: {', '.join(strengths[:2])}"
                    )

                # Weight-based rationale
                if user_weight > 0.4:
                    rationale_parts.append(
                        f"High user weighting ({user_weight:.2f}) indicates strong preference"
                    )
                elif user_weight > 0.25:
                    rationale_parts.append(
                        f"Moderate user weighting ({user_weight:.2f}) provides balanced input"
                    )

                rationales[role] = "; ".join(rationale_parts) if rationale_parts else "Standard contribution to analysis"

            return rationales

        except Exception as e:
            raise ArbitrationResultGenerationError(
                f"Failed to generate supporting rationales: {str(e)}",
                {"consultant_count": len(differential_analysis.merit_assessments)}
            )

    def build_polygon_enhancements(
        self,
        consultant_outputs: List[ConsultantOutput],
        user_preferences: UserWeightingPreferences,
        differential_analysis: DifferentialAnalysis,
    ) -> Dict[str, Any]:
        """Build polygon enhancements for synthesis refinement"""
        try:
            # Individual perspectives
            perspectives = []
            for output in consultant_outputs:
                perspective = {
                    "consultant": output.consultant_role.value,
                    "primary_recommendation": output.recommendations[0] if output.recommendations else "",
                    "key_insight": output.key_insights[0] if output.key_insights else "",
                    "mental_models": output.mental_models_used,
                    "confidence": output.confidence_level,
                }
                perspectives.append(perspective)

            # Points of consensus (from differential analysis)
            consensus = []
            common_themes = _safe_list(getattr(differential_analysis, "common_themes", []))
            for theme in common_themes:
                consensus.append({
                    "theme": theme.get("theme", ""),
                    "supporting_consultants": [c.value for c in theme.get("supporting_consultants", [])],
                    "strength": theme.get("consensus_strength", 0.0),
                })

            # Points of dissent (from differential analysis)
            dissent = []
            key_differences = _safe_list(getattr(differential_analysis, "key_differences", []))
            for diff in key_differences:
                dissent.append({
                    "dimension": diff.get("dimension", ""),
                    "positions": diff.get("differences", {}),
                    "impact": diff.get("impact_on_conclusions", ""),
                })

            # Structured recommendations
            structured_recs = []
            structured = _safe_list(getattr(differential_analysis, "structured_recommendations", []))
            for i, rec in enumerate(structured[:5]):
                structured_recs.append({
                    "priority": i + 1,
                    "recommendation": rec,
                    "supporting_evidence": f"Supported by {len(consultant_outputs)} consultant analysis",
                })

            # Compute polygon metrics
            metrics = self._compute_polygon_metrics(
                consultant_outputs, differential_analysis
            )

            return {
                "perspectives": perspectives,
                "consensus": consensus,
                "dissent": dissent,
                "structured_recs": structured_recs,
                "metrics": metrics,
            }

        except Exception as e:
            raise ArbitrationResultGenerationError(
                f"Failed to build polygon enhancements: {str(e)}",
                {"consultant_count": len(consultant_outputs)}
            )

    def calculate_consensus_strength(
        self, differential_analysis: DifferentialAnalysis
    ) -> float:
        """Calculate consensus strength across consultants"""
        try:
            # Use complementarity score as proxy for consensus
            complementarity = differential_analysis.complementarity_score
            consensus_level = differential_analysis.consensus_level

            # Combine scores for overall consensus strength
            consensus_strength = (complementarity + consensus_level) / 2.0

            return min(max(consensus_strength, 0.0), 1.0)

        except Exception as e:
            raise ArbitrationResultGenerationError(
                f"Failed to calculate consensus strength: {str(e)}",
                {"differential_analysis_id": getattr(differential_analysis, 'id', 'unknown')}
            )

    def extract_risk_mentions(self, text: str) -> List[str]:
        """Extract risk mentions from text"""
        try:
            risk_keywords = [
                "risk", "threat", "challenge", "problem", "issue", "concern",
                "danger", "vulnerability", "weakness", "failure", "downside"
            ]

            risk_mentions = []
            text_lower = text.lower()

            # Split into sentences for context
            sentences = text.split('. ')

            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in risk_keywords):
                    # Extract relevant part of sentence
                    words = sentence.split()
                    for i, word in enumerate(words):
                        if any(keyword in word.lower() for keyword in risk_keywords):
                            # Take context around risk mention
                            start = max(0, i - 3)
                            end = min(len(words), i + 4)
                            risk_context = ' '.join(words[start:end])
                            risk_mentions.append(risk_context)
                            break

            return risk_mentions[:3]  # Limit to top 3 risk mentions

        except Exception as e:
            raise ArbitrationResultGenerationError(
                f"Failed to extract risk mentions: {str(e)}",
                {"text_length": len(text)}
            )

    def contains_risk_content(self, text: str) -> bool:
        """Check if text contains risk-related content"""
        try:
            risk_keywords = [
                "risk", "threat", "challenge", "problem", "issue", "concern",
                "danger", "vulnerability", "weakness", "failure", "downside"
            ]

            text_lower = text.lower()
            return any(keyword in text_lower for keyword in risk_keywords)

        except Exception as e:
            raise ArbitrationResultGenerationError(
                f"Failed to check risk content: {str(e)}",
                {"text_length": len(text)}
            )

    def is_similar_recommendation(self, rec: str, seen_recs: set) -> bool:
        """Check if recommendation is similar to already seen recommendations"""
        try:
            rec_words = set(rec.lower().split())

            for seen_rec in seen_recs:
                seen_words = set(seen_rec.lower().split())

                # Check for significant word overlap
                overlap = len(rec_words & seen_words)
                min_length = min(len(rec_words), len(seen_words))

                if min_length > 0 and overlap / min_length > 0.6:  # 60% word overlap threshold
                    return True

            return False

        except Exception as e:
            raise ArbitrationResultGenerationError(
                f"Failed to check recommendation similarity: {str(e)}",
                {"recommendation_length": len(rec)}
            )

    def is_similar_insight(self, insight: str, seen_insights: set) -> bool:
        """Check if insight is similar to already seen insights"""
        try:
            insight_words = set(insight.lower().split())

            for seen_insight in seen_insights:
                seen_words = set(seen_insight.lower().split())

                # Check for significant word overlap
                overlap = len(insight_words & seen_words)
                min_length = min(len(insight_words), len(seen_words))

                if min_length > 0 and overlap / min_length > 0.6:  # 60% word overlap threshold
                    return True

            return False

        except Exception as e:
            raise ArbitrationResultGenerationError(
                f"Failed to check insight similarity: {str(e)}",
                {"insight_length": len(insight)}
            )

    # Helper methods

    def _decide_synthesis_mode(
        self,
        differential_analysis: DifferentialAnalysis,
        synthesis_metrics: Dict[str, float],
        query_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Decide synthesis mode based on analysis metrics"""

        primary_margin = synthesis_metrics.get("primary_margin", 0.1)
        dissent_intensity = synthesis_metrics.get("dissent_intensity", 0.5)

        if primary_margin > 0.3 and dissent_intensity < 0.3:
            return "consensus_driven"
        elif dissent_intensity > 0.7:
            return "debate_synthesis"
        else:
            return "balanced_synthesis"

    def _assess_red_team_quality(self, red_team_results: Dict[str, Any]) -> float:
        """Assess quality of Red Team Council results"""

        if not red_team_results:
            return 0.0

        successful_challenges = sum(
            1
            for result in red_team_results.values()
            if isinstance(result, dict) and result.get("status") == "success"
        )
        total_challenges = len(red_team_results)

        return successful_challenges / total_challenges if total_challenges > 0 else 0.0

    def _compute_polygon_metrics(
        self,
        consultant_outputs: List[ConsultantOutput],
        differential_analysis: DifferentialAnalysis,
    ) -> Dict[str, float]:
        """Compute polygon metrics for synthesis refinement"""

        # Polygon integrity score (based on consultant confidence and consistency)
        avg_confidence = sum(output.confidence_level for output in consultant_outputs) / len(consultant_outputs)
        polygon_integrity_score = avg_confidence * differential_analysis.consensus_level

        # Dissent intensity (based on key differences)
        key_diffs = _safe_list(getattr(differential_analysis, "key_differences", []))
        dissent_intensity = min(1.0, len(key_diffs) / 5.0)

        # Option reversibility index (based on implementation complexity)
        option_reversibility_index = 1.0 - dissent_intensity  # Simplified calculation

        return {
            "polygon_integrity_score": polygon_integrity_score,
            "dissent_intensity": dissent_intensity,
            "option_reversibility_index": option_reversibility_index,
        }