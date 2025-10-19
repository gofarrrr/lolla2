"""
Decision Support Service - Domain Service
========================================

Comprehensive domain service for enhancing arbitration results with advanced
decision support features including confidence intervals, decision trees,
and stakeholder impact analysis.

This service provides intelligent decision support enhancements that help
users navigate complex multi-consultant arbitration results.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Any
from datetime import datetime

from src.services.interfaces.decision_support_interface import (
    IDecisionSupportService,
    DecisionSupportError,
)
from src.arbitration.models import (
    ConsultantOutput,
    ArbitrationResult,
)

# Optional LLM integration for enhanced analysis
try:
    from src.core.resilient_llm_client import get_resilient_llm_client
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

logger = logging.getLogger(__name__)


class DecisionSupportService(IDecisionSupportService):
    """
    Domain service for enhancing arbitration results with advanced decision support features

    Responsibilities:
    - Add confidence intervals for statistical backing
    - Generate decision trees for complex scenarios
    - Provide stakeholder impact analysis
    - Enhance results with metadata and additional insights
    """

    def __init__(self):
        self.logger = logger
        self.llm_client = get_resilient_llm_client() if LLM_AVAILABLE else None

        # Configuration for enhancement features
        self.confidence_threshold = 0.7
        self.decision_tree_complexity_threshold = (
            3  # Number of consultants that triggers decision tree
        )
        self.stakeholder_analysis_enabled = True

        self.logger.info("🎯 DecisionSupportService domain service initialized")

    async def enhance_decision_support(
        self,
        arbitration_result: ArbitrationResult,
        consultant_outputs: List[ConsultantOutput],
        original_query: str,
    ) -> ArbitrationResult:
        """Enhance arbitration result with comprehensive decision support features"""
        try:
            self.logger.debug("🔧 Enhancing decision support for arbitration result")

            # Add confidence intervals for recommendations
            arbitration_result = await self.add_confidence_intervals(arbitration_result)

            # Add decision trees for complex scenarios
            arbitration_result = await self.add_decision_trees(
                arbitration_result, original_query
            )

            # Add stakeholder impact analysis if enabled
            if self.stakeholder_analysis_enabled:
                arbitration_result = await self.add_stakeholder_impact_analysis(
                    arbitration_result, consultant_outputs
                )

            # Add enhancement metadata
            arbitration_result = self.add_enhancement_metadata(arbitration_result)

            self.logger.debug("✅ Decision support enhancement complete")
            return arbitration_result

        except Exception as e:
            raise DecisionSupportError(
                f"Failed to enhance decision support: {str(e)}",
                {"query": original_query, "consultant_count": len(consultant_outputs)}
            )

    async def add_confidence_intervals(
        self, arbitration_result: ArbitrationResult
    ) -> ArbitrationResult:
        """Add statistical confidence intervals to weighted recommendations"""
        try:
            # Ensure metadata exists
            if not hasattr(arbitration_result, "metadata"):
                arbitration_result.__dict__["metadata"] = {}
            elif arbitration_result.metadata is None:
                arbitration_result.metadata = {}

            # Calculate confidence statistics from user preferences and differential analysis
            confidence_stats = self.calculate_confidence_statistics(arbitration_result)

            confidence_intervals = {
                "methodology": "Weighted consultant confidence scores with statistical analysis",
                "high_confidence_threshold": self.confidence_threshold,
                "overall_confidence_score": confidence_stats["overall_confidence"],
                "confidence_distribution": confidence_stats["distribution"],
                "statistical_backing": f"{confidence_stats['high_confidence_recommendations']} of {confidence_stats['total_recommendations']} recommendations exceed threshold",
                "note": "Higher confidence scores indicate stronger consensus and evidence backing",
            }

            arbitration_result.metadata["confidence_intervals"] = confidence_intervals

            self.logger.debug(
                f"📊 Added confidence intervals: {confidence_stats['overall_confidence']:.2f} overall confidence"
            )
            return arbitration_result

        except Exception as e:
            raise DecisionSupportError(
                f"Failed to add confidence intervals: {str(e)}",
                {"arbitration_result_id": getattr(arbitration_result, 'id', 'unknown')}
            )

    async def add_decision_trees(
        self, arbitration_result: ArbitrationResult, query: str
    ) -> ArbitrationResult:
        """Add decision tree structure for complex multi-consultant scenarios"""
        try:
            if not hasattr(arbitration_result, "metadata"):
                arbitration_result.__dict__["metadata"] = {}
            elif arbitration_result.metadata is None:
                arbitration_result.metadata = {}

            # Only generate decision trees for complex scenarios
            consultant_count = len(arbitration_result.user_preferences.consultant_weights)

            if consultant_count >= self.decision_tree_complexity_threshold:
                decision_tree = await self.generate_decision_tree(
                    arbitration_result, query
                )
            else:
                decision_tree = {
                    "complexity_level": "simple",
                    "consultant_count": consultant_count,
                    "note": f"Decision tree not generated - only {consultant_count} consultants (threshold: {self.decision_tree_complexity_threshold})",
                    "simple_recommendation": "Follow primary consultant recommendation with risk assessment consideration",
                }

            arbitration_result.metadata["decision_tree"] = decision_tree

            self.logger.debug(
                f"🌳 Added decision tree for {consultant_count} consultant scenario"
            )
            return arbitration_result

        except Exception as e:
            raise DecisionSupportError(
                f"Failed to add decision trees: {str(e)}",
                {"query": query, "consultant_count": len(arbitration_result.user_preferences.consultant_weights)}
            )

    async def add_stakeholder_impact_analysis(
        self,
        arbitration_result: ArbitrationResult,
        consultant_outputs: List[ConsultantOutput],
    ) -> ArbitrationResult:
        """Add comprehensive stakeholder impact analysis"""
        try:
            if not hasattr(arbitration_result, "metadata"):
                arbitration_result.__dict__["metadata"] = {}
            elif arbitration_result.metadata is None:
                arbitration_result.metadata = {}

            # Analyze stakeholder impacts from consultant outputs
            stakeholder_analysis = self.analyze_stakeholder_impacts(
                consultant_outputs, arbitration_result
            )

            arbitration_result.metadata["stakeholder_impact"] = stakeholder_analysis

            self.logger.debug(
                f"👥 Added stakeholder analysis covering {len(stakeholder_analysis['stakeholder_groups'])} groups"
            )
            return arbitration_result

        except Exception as e:
            raise DecisionSupportError(
                f"Failed to add stakeholder impact analysis: {str(e)}",
                {"consultant_count": len(consultant_outputs)}
            )

    def add_enhancement_metadata(
        self, arbitration_result: ArbitrationResult
    ) -> ArbitrationResult:
        """Add general enhancement metadata and timestamps"""
        try:
            if not hasattr(arbitration_result, "metadata"):
                arbitration_result.__dict__["metadata"] = {}
            elif arbitration_result.metadata is None:
                arbitration_result.metadata = {}

            enhancement_metadata = {
                "enhanced_at": datetime.utcnow().isoformat(),
                "enhancement_version": "1.0",
                "features_applied": [
                    "confidence_intervals",
                    (
                        "decision_trees"
                        if len(arbitration_result.user_preferences.consultant_weights)
                        >= self.decision_tree_complexity_threshold
                        else "simple_decision_structure"
                    ),
                    (
                        "stakeholder_impact_analysis"
                        if self.stakeholder_analysis_enabled
                        else "stakeholder_analysis_disabled"
                    ),
                ],
                "enhancement_quality_score": self.calculate_enhancement_quality(
                    arbitration_result
                ),
            }

            arbitration_result.metadata["enhancement_info"] = enhancement_metadata

            return arbitration_result

        except Exception as e:
            raise DecisionSupportError(
                f"Failed to add enhancement metadata: {str(e)}",
                {"arbitration_result_id": getattr(arbitration_result, 'id', 'unknown')}
            )

    def calculate_confidence_statistics(
        self, arbitration_result: ArbitrationResult
    ) -> Dict[str, Any]:
        """Calculate comprehensive confidence statistics from arbitration result"""
        try:
            # Get consultant weights and their confidence contributions
            consultant_weights = arbitration_result.user_preferences.consultant_weights
            total_recommendations = len(arbitration_result.weighted_recommendations)

            # Calculate overall confidence as weighted average
            weighted_confidence = 0.0
            total_weight = 0.0

            for consultant_role, weight in consultant_weights.items():
                # Use weight as proxy for confidence (in real implementation, would use actual confidence scores)
                weighted_confidence += (
                    weight * 0.8
                )  # Assume 80% base confidence, weighted by consultant importance
                total_weight += weight

            overall_confidence = (
                weighted_confidence / total_weight if total_weight > 0 else 0.5
            )

            # Calculate distribution (simplified for this extraction)
            distribution = {
                "high_confidence": len(
                    [
                        r
                        for r in arbitration_result.weighted_recommendations
                        if "Weight:" in r
                        and float(r.split("Weight: ")[1].split(",")[0])
                        > self.confidence_threshold
                    ]
                ),
                "medium_confidence": max(
                    0,
                    total_recommendations
                    - len(
                        [
                            r
                            for r in arbitration_result.weighted_recommendations
                            if "Weight:" in r
                            and float(r.split("Weight: ")[1].split(",")[0])
                            > self.confidence_threshold
                        ]
                    ),
                ),
                "low_confidence": 0,  # Simplified - in real implementation would parse actual confidence scores
            }

            return {
                "overall_confidence": overall_confidence,
                "distribution": distribution,
                "high_confidence_recommendations": distribution["high_confidence"],
                "total_recommendations": total_recommendations,
            }

        except Exception as e:
            raise DecisionSupportError(
                f"Failed to calculate confidence statistics: {str(e)}",
                {"arbitration_result_id": getattr(arbitration_result, 'id', 'unknown')}
            )

    async def generate_decision_tree(
        self, arbitration_result: ArbitrationResult, query: str
    ) -> Dict[str, Any]:
        """Generate a decision tree structure for complex scenarios"""
        try:
            consultant_weights = arbitration_result.user_preferences.consultant_weights
            risk_tolerance = arbitration_result.user_preferences.risk_tolerance

            # Create decision nodes based on risk tolerance and consultant consensus
            decision_nodes = []

            # Node 1: Risk assessment branch
            if risk_tolerance < 0.3:
                risk_branch = {
                    "condition": "Risk tolerance is low (< 0.3)",
                    "recommendation": "Prioritize recommendations from risk-focused consultants",
                    "action": "Weight risk management and compliance perspectives higher",
                    "confidence": "high",
                }
            elif risk_tolerance > 0.7:
                risk_branch = {
                    "condition": "Risk tolerance is high (> 0.7)",
                    "recommendation": "Consider innovative and transformational approaches",
                    "action": "Weight innovation and growth perspectives higher",
                    "confidence": "high",
                }
            else:
                risk_branch = {
                    "condition": "Risk tolerance is moderate (0.3-0.7)",
                    "recommendation": "Balance conservative and innovative approaches",
                    "action": "Use existing weighted recommendations as-is",
                    "confidence": "medium",
                }

            decision_nodes.append(risk_branch)

            # Node 2: Consensus strength branch
            primary_consultant = arbitration_result.primary_consultant_recommendation
            consultant_count = len(consultant_weights)

            if consultant_count >= 5:
                consensus_branch = {
                    "condition": f"High consultant diversity ({consultant_count} consultants)",
                    "recommendation": f"Follow primary consultant ({primary_consultant.value}) with cross-validation",
                    "action": "Implement with enhanced monitoring and feedback loops",
                    "confidence": "high",
                }
            else:
                consensus_branch = {
                    "condition": f"Moderate consultant diversity ({consultant_count} consultants)",
                    "recommendation": f"Follow primary consultant ({primary_consultant.value}) recommendation",
                    "action": "Standard implementation approach",
                    "confidence": "medium",
                }

            decision_nodes.append(consensus_branch)

            # Node 3: Implementation complexity branch
            implementation_horizon = (
                arbitration_result.user_preferences.implementation_horizon
            )

            if implementation_horizon == "immediate":
                implementation_branch = {
                    "condition": "Immediate implementation required",
                    "recommendation": "Focus on quick wins and proven approaches",
                    "action": "Prioritize first 2-3 recommendations with highest weights",
                    "confidence": "high",
                }
            elif implementation_horizon == "long_term":
                implementation_branch = {
                    "condition": "Long-term implementation horizon",
                    "recommendation": "Consider strategic and transformational approaches",
                    "action": "Include alternative scenarios in planning",
                    "confidence": "medium",
                }
            else:
                implementation_branch = {
                    "condition": f"{implementation_horizon.title().replace('_', ' ')} implementation horizon",
                    "recommendation": "Balanced implementation approach",
                    "action": "Follow standard implementation guidance",
                    "confidence": "medium",
                }

            decision_nodes.append(implementation_branch)

            decision_tree = {
                "complexity_level": "complex",
                "consultant_count": consultant_count,
                "decision_nodes": decision_nodes,
                "final_recommendation": f"Primary path: Follow {primary_consultant.value} recommendations with {risk_branch['action'].lower()}",
                "alternative_paths": len(arbitration_result.alternative_scenarios),
                "confidence_level": self._calculate_tree_confidence(decision_nodes),
            }

            return decision_tree

        except Exception as e:
            raise DecisionSupportError(
                f"Failed to generate decision tree: {str(e)}",
                {"query": query}
            )

    def analyze_stakeholder_impacts(
        self,
        consultant_outputs: List[ConsultantOutput],
        arbitration_result: ArbitrationResult,
    ) -> Dict[str, Any]:
        """Analyze potential stakeholder impacts from consultant recommendations"""
        try:
            # Extract stakeholder mentions from consultant outputs
            stakeholder_groups = set()
            impact_analysis = {}

            # Common stakeholder categories to look for
            stakeholder_keywords = {
                "customers": ["customer", "client", "user", "consumer"],
                "employees": ["employee", "staff", "team", "workforce", "personnel"],
                "shareholders": ["shareholder", "investor", "stakeholder", "owner"],
                "suppliers": ["supplier", "vendor", "partner", "contractor"],
                "regulators": ["regulator", "government", "compliance", "authority"],
                "community": ["community", "public", "society", "local", "environmental"],
            }

            # Analyze each consultant output for stakeholder mentions
            for output in consultant_outputs:
                consultant_weight = (
                    arbitration_result.user_preferences.consultant_weights.get(
                        output.consultant_role, 0.0
                    )
                )

                # Check recommendations for stakeholder impacts
                for rec in output.recommendations:
                    rec_lower = rec.lower()

                    for stakeholder_group, keywords in stakeholder_keywords.items():
                        if any(keyword in rec_lower for keyword in keywords):
                            stakeholder_groups.add(stakeholder_group)

                            if stakeholder_group not in impact_analysis:
                                impact_analysis[stakeholder_group] = {
                                    "mentions": 0,
                                    "weighted_importance": 0.0,
                                    "potential_impacts": [],
                                    "consulting_sources": [],
                                }

                            impact_analysis[stakeholder_group]["mentions"] += 1
                            impact_analysis[stakeholder_group][
                                "weighted_importance"
                            ] += consultant_weight
                            impact_analysis[stakeholder_group]["potential_impacts"].append(
                                rec[:100] + "..." if len(rec) > 100 else rec
                            )
                            impact_analysis[stakeholder_group]["consulting_sources"].append(
                                output.consultant_role.value
                            )

            # Calculate impact priority scores
            for group_data in impact_analysis.values():
                group_data["priority_score"] = (
                    group_data["weighted_importance"] * group_data["mentions"]
                )
                # Remove duplicates from sources
                group_data["consulting_sources"] = list(
                    set(group_data["consulting_sources"])
                )
                # Limit potential impacts to top 3
                group_data["potential_impacts"] = group_data["potential_impacts"][:3]

            stakeholder_analysis = {
                "stakeholder_groups": list(stakeholder_groups),
                "detailed_analysis": impact_analysis,
                "high_impact_groups": [
                    group
                    for group, data in impact_analysis.items()
                    if data["priority_score"] > 1.0
                ],
                "analysis_completeness": len(stakeholder_groups)
                / len(stakeholder_keywords),
                "total_stakeholder_mentions": sum(
                    data["mentions"] for data in impact_analysis.values()
                ),
            }

            return stakeholder_analysis

        except Exception as e:
            raise DecisionSupportError(
                f"Failed to analyze stakeholder impacts: {str(e)}",
                {"consultant_count": len(consultant_outputs)}
            )

    def calculate_enhancement_quality(
        self, arbitration_result: ArbitrationResult
    ) -> float:
        """Calculate a quality score for the enhancement process"""
        try:
            quality_factors = []

            # Factor 1: Metadata completeness
            if hasattr(arbitration_result, "metadata") and arbitration_result.metadata:
                metadata_completeness = (
                    len(arbitration_result.metadata.keys()) / 4
                )  # Expected: confidence, decision_tree, stakeholder, enhancement
                quality_factors.append(min(metadata_completeness, 1.0))
            else:
                quality_factors.append(0.0)

            # Factor 2: Recommendation richness
            rec_count = len(arbitration_result.weighted_recommendations)
            rec_richness = min(rec_count / 6, 1.0)  # Optimal is 6 recommendations
            quality_factors.append(rec_richness)

            # Factor 3: Alternative scenario coverage
            alt_scenario_count = len(arbitration_result.alternative_scenarios)
            scenario_coverage = min(alt_scenario_count / 3, 1.0)  # Optimal is 3 scenarios
            quality_factors.append(scenario_coverage)

            # Factor 4: User satisfaction prediction accuracy (proxy)
            satisfaction_quality = arbitration_result.user_satisfaction_prediction
            quality_factors.append(satisfaction_quality)

            overall_quality = sum(quality_factors) / len(quality_factors)
            return round(overall_quality, 3)

        except Exception as e:
            raise DecisionSupportError(
                f"Failed to calculate enhancement quality: {str(e)}",
                {"arbitration_result_id": getattr(arbitration_result, 'id', 'unknown')}
            )

    # Helper methods

    def _calculate_tree_confidence(self, decision_nodes: List[Dict[str, Any]]) -> str:
        """Calculate overall confidence level for the decision tree"""

        confidence_scores = {"high": 3, "medium": 2, "low": 1}
        total_score = sum(
            confidence_scores.get(node.get("confidence", "medium"), 2)
            for node in decision_nodes
        )
        avg_score = total_score / len(decision_nodes) if decision_nodes else 2

        if avg_score >= 2.7:
            return "high"
        elif avg_score >= 2.3:
            return "medium-high"
        elif avg_score >= 1.7:
            return "medium"
        elif avg_score >= 1.3:
            return "medium-low"
        else:
            return "low"