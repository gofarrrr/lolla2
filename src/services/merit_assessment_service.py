"""
Merit Assessment Service - Domain Service
========================================

Comprehensive domain service for evaluating consultant outputs based on
multiple quality criteria and providing merit-based weighting suggestions.

This service implements context-aware quality assessment that considers both
absolute quality and query-specific fitness.
"""

from __future__ import annotations

import logging
import statistics
from typing import Dict, List, Any, Optional
from collections import defaultdict

from src.services.interfaces.merit_assessment_interface import (
    IMeritAssessmentService,
    MeritAssessmentError,
)
from src.arbitration.models import (
    ConsultantOutput,
    ConsultantRole,
    MeritCriterion,
    MeritScore,
    ConsultantMeritAssessment,
)

# Optional integrations
try:
    from src.core.resilient_llm_client import get_resilient_llm_client
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

logger = logging.getLogger(__name__)


class MeritAssessmentService(IMeritAssessmentService):
    """
    Domain service for comprehensive consultant output quality assessment

    Philosophy: Context-aware assessment that balances absolute quality
    with query-specific fitness to provide meaningful weighting guidance.
    """

    def __init__(self):
        self.logger = logger
        self.llm_client = get_resilient_llm_client() if LLM_AVAILABLE else None

        # Merit assessment configuration
        self.criterion_weights = {
            MeritCriterion.EVIDENCE_QUALITY: 0.20,
            MeritCriterion.LOGICAL_CONSISTENCY: 0.20,
            MeritCriterion.QUERY_ALIGNMENT: 0.15,
            MeritCriterion.IMPLEMENTATION_FEASIBILITY: 0.15,
            MeritCriterion.RISK_THOROUGHNESS: 0.10,
            MeritCriterion.NOVEL_INSIGHTS: 0.10,
            MeritCriterion.BIAS_RESISTANCE: 0.10,
        }

        # Quality thresholds
        self.excellence_threshold = 0.85
        self.good_threshold = 0.70
        self.acceptable_threshold = 0.55

        self.logger.info("⚖️ MeritAssessmentService domain service initialized")

    async def assess_consultant_outputs(
        self,
        consultant_outputs: List[ConsultantOutput],
        original_query: str,
        query_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[ConsultantRole, ConsultantMeritAssessment]:
        """
        Assess merit of all consultant outputs
        """
        try:
            if not consultant_outputs:
                raise MeritAssessmentError("No consultant outputs provided", {"consultant_count": 0})
            query_display = (
                original_query[:100] + "..."
                if len(original_query) > 100
                else original_query
            )
            self.logger.info(
                f"⚖️ Assessing merit for {len(consultant_outputs)} consultants - Query: {query_display}"
            )

            merit_assessments = {}

            # Assess each consultant individually
            for output in consultant_outputs:
                assessment = await self.assess_single_consultant(
                    output, original_query, query_context
                )
                merit_assessments[output.consultant_role] = assessment

            # Add comparative rankings
            self._add_comparative_rankings(merit_assessments)

            # Adjust for query-specific fitness
            if query_context:
                self._adjust_for_query_context(merit_assessments, query_context)

            self.logger.info(
                f"✅ Merit assessment completed - Assessments: {len(merit_assessments)}"
            )

            return merit_assessments

        except Exception as e:
            raise MeritAssessmentError(
                f"Failed to assess consultant outputs: {str(e)}",
                {"query": original_query, "consultant_count": len(consultant_outputs)}
            )

    async def assess_single_consultant(
        self,
        output: ConsultantOutput,
        original_query: str,
        query_context: Optional[Dict[str, Any]] = None,
    ) -> ConsultantMeritAssessment:
        """Assess merit of a single consultant output"""
        try:
            # Assess each criterion
            criterion_scores = {}

            # Evidence Quality Assessment
            evidence_score = await self.assess_evidence_quality(output)
            criterion_scores[MeritCriterion.EVIDENCE_QUALITY] = evidence_score

            # Logical Consistency Assessment
            logical_score = await self.assess_logical_consistency(output)
            criterion_scores[MeritCriterion.LOGICAL_CONSISTENCY] = logical_score

            # Query Alignment Assessment
            alignment_score = await self.assess_query_alignment(output, original_query)
            criterion_scores[MeritCriterion.QUERY_ALIGNMENT] = alignment_score

            # Implementation Feasibility Assessment
            feasibility_score = await self.assess_implementation_feasibility(output)
            criterion_scores[MeritCriterion.IMPLEMENTATION_FEASIBILITY] = feasibility_score

            # Risk Thoroughness Assessment
            risk_score = await self.assess_risk_thoroughness(output)
            criterion_scores[MeritCriterion.RISK_THOROUGHNESS] = risk_score

            # Novel Insights Assessment
            novelty_score = await self.assess_novel_insights(output)
            criterion_scores[MeritCriterion.NOVEL_INSIGHTS] = novelty_score

            # Bias Resistance Assessment
            bias_score = await self.assess_bias_resistance(output)
            criterion_scores[MeritCriterion.BIAS_RESISTANCE] = bias_score

            # Calculate overall merit score
            overall_merit = self.calculate_overall_merit(criterion_scores)

            # Calculate query fitness score
            query_fitness = await self.calculate_query_fitness_score(
                output, original_query
            )

            # Generate recommended weight
            recommended_weight = self._calculate_recommended_weight(
                overall_merit, query_fitness
            )

            # Identify strengths and weaknesses
            strengths = self._identify_strengths(criterion_scores)
            weaknesses = self._identify_weaknesses(criterion_scores)

            assessment = ConsultantMeritAssessment(
                consultant_role=output.consultant_role,
                overall_merit_score=overall_merit,
                criterion_scores=criterion_scores,
                strengths=strengths,
                weaknesses=weaknesses,
                query_fitness_score=query_fitness,
                recommended_weight=recommended_weight,
            )

            self.logger.debug(
                f"📊 Merit assessed for {output.consultant_role.value}",
                overall_merit=f"{overall_merit:.3f}",
                query_fitness=f"{query_fitness:.3f}",
                recommended_weight=f"{recommended_weight:.3f}",
            )

            return assessment

        except Exception as e:
            raise MeritAssessmentError(
                f"Failed to assess single consultant {output.consultant_role.value}: {str(e)}",
                {"consultant_role": output.consultant_role.value, "query": original_query}
            )

    async def assess_evidence_quality(self, output: ConsultantOutput) -> MeritScore:
        """Assess quality of evidence and research"""
        try:
            # Base score from output metadata
            base_score = output.research_depth_score

            # Evidence source diversity factor
            evidence_categories = self._categorize_evidence_sources(output.evidence_sources)
            diversity_factor = min(1.0, len(evidence_categories) / 5)  # Up to 5 categories

            # Source credibility factor
            credibility_factor = self._assess_source_credibility(output.evidence_sources)

            # Volume factor (diminishing returns)
            volume_factor = min(1.0, len(output.evidence_sources) / 10)

            # Fact pack quality factor
            fact_pack_factor = {"high": 1.0, "medium": 0.7, "low": 0.4}.get(
                output.fact_pack_quality, 0.5
            )

            # Weighted combination
            evidence_score = (
                base_score * 0.3
                + diversity_factor * 0.25
                + credibility_factor * 0.20
                + volume_factor * 0.15
                + fact_pack_factor * 0.10
            )

            explanation = self._generate_evidence_explanation(
                base_score,
                diversity_factor,
                credibility_factor,
                volume_factor,
                fact_pack_factor,
            )

            return MeritScore(
                criterion=MeritCriterion.EVIDENCE_QUALITY,
                score=evidence_score,
                explanation=explanation,
                supporting_evidence=[
                    f"Research depth: {output.research_depth_score:.2f}",
                    f"Source diversity: {len(evidence_categories)} categories",
                    f"Total sources: {len(output.evidence_sources)}",
                    f"Fact pack quality: {output.fact_pack_quality}",
                ],
                relative_ranking=0,  # Will be set during comparative ranking
            )

        except Exception as e:
            raise MeritAssessmentError(
                f"Failed to assess evidence quality: {str(e)}",
                {"consultant_role": output.consultant_role.value}
            )

    async def assess_logical_consistency(self, output: ConsultantOutput) -> MeritScore:
        """Assess logical consistency and reasoning quality"""
        try:
            # Base score from Red Team Council validation
            base_score = output.logical_consistency_score

            # Assumption awareness factor
            assumption_factor = min(1.0, len(output.assumptions_made) / 5)

            # Limitation recognition factor
            limitation_factor = min(1.0, len(output.limitations_identified) / 3)

            # Mental model coherence factor
            coherence_factor = self._assess_mental_model_coherence(
                output.mental_models_used
            )

            # Recommendation consistency factor
            consistency_factor = await self._assess_recommendation_consistency(output)

            logical_score = (
                base_score * 0.4
                + assumption_factor * 0.2
                + limitation_factor * 0.2
                + coherence_factor * 0.1
                + consistency_factor * 0.1
            )

            explanation = f"Strong logical consistency with {len(output.assumptions_made)} explicit assumptions and {len(output.limitations_identified)} identified limitations"

            return MeritScore(
                criterion=MeritCriterion.LOGICAL_CONSISTENCY,
                score=logical_score,
                explanation=explanation,
                supporting_evidence=[
                    f"Consistency score: {output.logical_consistency_score:.2f}",
                    f"Assumptions made: {len(output.assumptions_made)}",
                    f"Limitations identified: {len(output.limitations_identified)}",
                    f"Mental model coherence: {coherence_factor:.2f}",
                ],
                relative_ranking=0,
            )

        except Exception as e:
            raise MeritAssessmentError(
                f"Failed to assess logical consistency: {str(e)}",
                {"consultant_role": output.consultant_role.value}
            )

    async def assess_query_alignment(
        self, output: ConsultantOutput, original_query: str
    ) -> MeritScore:
        """Assess how well the output addresses the original query"""
        try:
            # Keyword overlap analysis
            query_keywords = self._extract_keywords(original_query.lower())
            output_text = " ".join(
                [
                    output.executive_summary,
                    " ".join(output.key_insights),
                    " ".join(output.recommendations),
                ]
            ).lower()

            keyword_coverage = (
                sum(1 for keyword in query_keywords if keyword in output_text)
                / len(query_keywords)
                if query_keywords
                else 0
            )

            # Query intent matching
            query_intent = self._infer_query_intent(original_query)
            output_alignment = self._assess_intent_alignment(output, query_intent)

            # Completeness factor
            completeness_factor = self._assess_response_completeness(output, original_query)

            # Focus factor (not going off-topic)
            focus_factor = self._assess_focus_maintenance(output, original_query)

            alignment_score = (
                keyword_coverage * 0.25
                + output_alignment * 0.35
                + completeness_factor * 0.25
                + focus_factor * 0.15
            )

            explanation = f"Query alignment through {keyword_coverage:.0%} keyword coverage and {query_intent} intent matching"

            return MeritScore(
                criterion=MeritCriterion.QUERY_ALIGNMENT,
                score=alignment_score,
                explanation=explanation,
                supporting_evidence=[
                    f"Keyword coverage: {keyword_coverage:.1%}",
                    f"Query intent: {query_intent}",
                    f"Completeness: {completeness_factor:.2f}",
                    f"Focus maintenance: {focus_factor:.2f}",
                ],
                relative_ranking=0,
            )

        except Exception as e:
            raise MeritAssessmentError(
                f"Failed to assess query alignment: {str(e)}",
                {"consultant_role": output.consultant_role.value, "query": original_query}
            )

    async def assess_implementation_feasibility(
        self, output: ConsultantOutput
    ) -> MeritScore:
        """Assess practical feasibility of recommendations"""
        try:
            # Recommendation specificity
            specificity_score = self._assess_recommendation_specificity(
                output.recommendations
            )

            # Resource consideration
            resource_awareness = self._assess_resource_awareness(output)

            # Timeline realism
            timeline_realism = self._assess_timeline_realism(output)

            # Stakeholder consideration
            stakeholder_factor = self._assess_stakeholder_consideration(output)

            # Risk mitigation planning
            risk_mitigation = self._assess_risk_mitigation_planning(output)

            feasibility_score = (
                specificity_score * 0.25
                + resource_awareness * 0.20
                + timeline_realism * 0.20
                + stakeholder_factor * 0.20
                + risk_mitigation * 0.15
            )

            explanation = "Implementation feasibility assessed through recommendation specificity and resource awareness"

            return MeritScore(
                criterion=MeritCriterion.IMPLEMENTATION_FEASIBILITY,
                score=feasibility_score,
                explanation=explanation,
                supporting_evidence=[
                    f"Recommendation specificity: {specificity_score:.2f}",
                    f"Resource awareness: {resource_awareness:.2f}",
                    f"Timeline realism: {timeline_realism:.2f}",
                    f"Stakeholder consideration: {stakeholder_factor:.2f}",
                ],
                relative_ranking=0,
            )

        except Exception as e:
            raise MeritAssessmentError(
                f"Failed to assess implementation feasibility: {str(e)}",
                {"consultant_role": output.consultant_role.value}
            )

    async def assess_risk_thoroughness(self, output: ConsultantOutput) -> MeritScore:
        """Assess thoroughness of risk identification and analysis"""
        try:
            # Red Team Council results
            red_team_effectiveness = self._assess_red_team_effectiveness(
                output.red_team_results
            )

            # Risk identification in recommendations
            risk_identification = self._count_risk_considerations(output)

            # Mitigation strategy presence
            mitigation_strategies = self._assess_mitigation_strategies(output)

            # Scenario consideration
            scenario_planning = self._assess_scenario_consideration(output)

            # Bias detection strength
            bias_detection = output.bias_detection_score

            risk_score = (
                red_team_effectiveness * 0.3
                + risk_identification * 0.25
                + mitigation_strategies * 0.20
                + scenario_planning * 0.15
                + bias_detection * 0.10
            )

            explanation = "Risk thoroughness through Red Team validation and comprehensive risk identification"

            return MeritScore(
                criterion=MeritCriterion.RISK_THOROUGHNESS,
                score=risk_score,
                explanation=explanation,
                supporting_evidence=[
                    f"Red Team effectiveness: {red_team_effectiveness:.2f}",
                    f"Risk considerations: {risk_identification:.2f}",
                    f"Mitigation strategies: {mitigation_strategies:.2f}",
                    f"Bias detection: {bias_detection:.2f}",
                ],
                relative_ranking=0,
            )

        except Exception as e:
            raise MeritAssessmentError(
                f"Failed to assess risk thoroughness: {str(e)}",
                {"consultant_role": output.consultant_role.value}
            )

    async def assess_novel_insights(self, output: ConsultantOutput) -> MeritScore:
        """Assess novelty and creativity of insights"""
        try:
            # Insight generation rate
            insight_rate = len(output.key_insights) / max(len(output.evidence_sources), 1)
            insight_generation = min(1.0, insight_rate / 1.5)  # Normalize to reasonable max

            # Insight sophistication
            sophistication = self._assess_insight_sophistication(output.key_insights)

            # Mental model creativity
            model_creativity = self._assess_mental_model_creativity(
                output.mental_models_used
            )

            # Non-obvious connections
            connection_creativity = self._assess_connection_creativity(output)

            # Counter-intuitive insights
            counter_intuitive = self._identify_counter_intuitive_insights(output)

            novelty_score = (
                insight_generation * 0.25
                + sophistication * 0.25
                + model_creativity * 0.20
                + connection_creativity * 0.15
                + counter_intuitive * 0.15
            )

            explanation = "Novel insights through creative mental model application and sophisticated analysis"

            return MeritScore(
                criterion=MeritCriterion.NOVEL_INSIGHTS,
                score=novelty_score,
                explanation=explanation,
                supporting_evidence=[
                    f"Insight generation rate: {insight_rate:.2f}",
                    f"Sophistication level: {sophistication:.2f}",
                    f"Model creativity: {model_creativity:.2f}",
                    f"Counter-intuitive insights: {counter_intuitive:.2f}",
                ],
                relative_ranking=0,
            )

        except Exception as e:
            raise MeritAssessmentError(
                f"Failed to assess novel insights: {str(e)}",
                {"consultant_role": output.consultant_role.value}
            )

    async def assess_bias_resistance(self, output: ConsultantOutput) -> MeritScore:
        """Assess resistance to cognitive biases"""
        try:
            # Base bias detection score
            base_score = output.bias_detection_score

            # Mental model diversity (reduces single-perspective bias)
            model_diversity = len(
                set(self._categorize_mental_models(output.mental_models_used))
            )
            diversity_factor = min(1.0, model_diversity / 5)

            # Evidence source diversity (reduces confirmation bias)
            evidence_categories = self._categorize_evidence_sources(output.evidence_sources)
            evidence_diversity = min(1.0, len(evidence_categories) / 5)

            # Assumption awareness (reduces overconfidence bias)
            assumption_awareness = min(1.0, len(output.assumptions_made) / 4)

            # Limitation recognition (reduces overconfidence bias)
            limitation_awareness = min(1.0, len(output.limitations_identified) / 3)

            bias_resistance = (
                base_score * 0.4
                + diversity_factor * 0.2
                + evidence_diversity * 0.2
                + assumption_awareness * 0.1
                + limitation_awareness * 0.1
            )

            explanation = "Bias resistance through diverse perspectives and explicit assumption awareness"

            return MeritScore(
                criterion=MeritCriterion.BIAS_RESISTANCE,
                score=bias_resistance,
                explanation=explanation,
                supporting_evidence=[
                    f"Bias detection score: {base_score:.2f}",
                    f"Mental model diversity: {model_diversity}",
                    f"Evidence diversity: {len(evidence_categories)}",
                    f"Assumptions made: {len(output.assumptions_made)}",
                ],
                relative_ranking=0,
            )

        except Exception as e:
            raise MeritAssessmentError(
                f"Failed to assess bias resistance: {str(e)}",
                {"consultant_role": output.consultant_role.value}
            )

    def calculate_overall_merit(
        self, criterion_scores: Dict[MeritCriterion, MeritScore]
    ) -> float:
        """Calculate weighted overall merit score"""
        try:
            total_score = 0.0
            total_weight = 0.0

            for criterion, merit_score in criterion_scores.items():
                weight = self.criterion_weights.get(criterion, 0.0)
                total_score += merit_score.score * weight
                total_weight += weight

            return total_score / total_weight if total_weight > 0 else 0.0

        except Exception as e:
            raise MeritAssessmentError(
                f"Failed to calculate overall merit: {str(e)}",
                {"criteria_count": len(criterion_scores)}
            )

    async def calculate_query_fitness_score(
        self, output: ConsultantOutput, query: str
    ) -> float:
        """Calculate how well suited this consultant is for this specific query"""
        try:
            # Query intent alignment
            query_intent = self._infer_query_intent(query)
            intent_alignment = self._assess_intent_alignment(output, query_intent)

            # Mental model suitability for query type
            model_suitability = self._assess_model_suitability_for_query(
                output.mental_models_used, query
            )

            # Evidence strategy fit
            evidence_fit = self._assess_evidence_strategy_fit(output, query)

            # Consultant role alignment
            role_alignment = self._assess_role_alignment_with_query(
                output.consultant_role, query
            )

            fitness_score = (
                intent_alignment * 0.3
                + model_suitability * 0.3
                + evidence_fit * 0.2
                + role_alignment * 0.2
            )

            return fitness_score

        except Exception as e:
            raise MeritAssessmentError(
                f"Failed to calculate query fitness score: {str(e)}",
                {"consultant_role": output.consultant_role.value, "query": query}
            )

    # Helper methods for specific assessments

    def _categorize_evidence_sources(self, sources: List[str]) -> Dict[str, List[str]]:
        """Categorize evidence sources by type"""
        categories = defaultdict(list)

        for source in sources:
            source_lower = source.lower()
            if any(
                keyword in source_lower for keyword in ["journal", "research", "study"]
            ):
                categories["academic"].append(source)
            elif any(keyword in source_lower for keyword in ["industry", "report"]):
                categories["industry"].append(source)
            elif any(keyword in source_lower for keyword in ["financial", "market"]):
                categories["financial"].append(source)
            elif any(keyword in source_lower for keyword in ["case", "example"]):
                categories["case_study"].append(source)
            else:
                categories["other"].append(source)

        return dict(categories)

    def _assess_source_credibility(self, sources: List[str]) -> float:
        """Assess overall credibility of sources"""
        if not sources:
            return 0.0

        credible_count = 0
        for source in sources:
            source_lower = source.lower()
            if any(
                indicator in source_lower
                for indicator in [
                    "peer-reviewed",
                    "journal",
                    "university",
                    "institute",
                    "government",
                    "established",
                    "reputable",
                ]
            ):
                credible_count += 1

        return credible_count / len(sources)

    def _generate_evidence_explanation(
        self,
        base_score: float,
        diversity: float,
        credibility: float,
        volume: float,
        fact_pack: float,
    ) -> str:
        """Generate explanation for evidence quality score"""

        strengths = []
        if base_score > 0.8:
            strengths.append("high research depth")
        if diversity > 0.8:
            strengths.append("diverse source types")
        if credibility > 0.7:
            strengths.append("credible sources")
        if volume > 0.7:
            strengths.append("comprehensive coverage")
        if fact_pack > 0.8:
            strengths.append("high-quality fact validation")

        if strengths:
            return f"Strong evidence quality with {', '.join(strengths)}"
        else:
            return "Moderate evidence quality with room for improvement"

    def _assess_mental_model_coherence(self, mental_models: List[str]) -> float:
        """Assess coherence and compatibility of mental models used"""

        if len(mental_models) <= 1:
            return 1.0  # Single model is coherent by definition

        # Simplified coherence assessment based on model categories
        categories = self._categorize_mental_models(mental_models)

        # High coherence if models complement rather than contradict
        if len(categories) > 1 and len(categories) <= 4:
            return 0.9  # Good diversity without chaos
        elif len(categories) == 1:
            return 0.8  # Consistent but narrow
        else:
            return 0.6  # Potentially scattered approach

    def _categorize_mental_models(
        self, mental_models: List[str]
    ) -> Dict[str, List[str]]:
        """Categorize mental models by type"""
        categories = defaultdict(list)

        for model in mental_models:
            model_lower = model.lower()
            if any(
                keyword in model_lower for keyword in ["risk", "inversion", "munger"]
            ):
                categories["risk_management"].append(model)
            elif any(keyword in model_lower for keyword in ["system", "ackoff"]):
                categories["systems_thinking"].append(model)
            elif any(keyword in model_lower for keyword in ["design", "user"]):
                categories["design_thinking"].append(model)
            elif any(keyword in model_lower for keyword in ["financial", "economic"]):
                categories["financial"].append(model)
            else:
                categories["general"].append(model)

        return dict(categories)

    async def _assess_recommendation_consistency(
        self, output: ConsultantOutput
    ) -> float:
        """Assess internal consistency of recommendations"""

        if len(output.recommendations) <= 1:
            return 1.0

        # Simple consistency check based on conflicting keywords
        conflicting_pairs = [
            (["aggressive", "bold", "rapid"], ["cautious", "conservative", "gradual"]),
            (["expand", "grow", "scale"], ["reduce", "cut", "minimize"]),
            (["invest", "spend"], ["save", "reduce costs"]),
        ]

        rec_text = " ".join(output.recommendations).lower()
        conflicts = 0

        for aggressive_terms, conservative_terms in conflicting_pairs:
            has_aggressive = any(term in rec_text for term in aggressive_terms)
            has_conservative = any(term in rec_text for term in conservative_terms)
            if has_aggressive and has_conservative:
                conflicts += 1

        # Penalize conflicts but allow for balanced approaches
        consistency_score = max(0.5, 1.0 - (conflicts * 0.2))
        return consistency_score

    # Query alignment helper methods

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text"""
        # Remove common words and extract meaningful terms
        common_words = set(
            [
                "the",
                "and",
                "or",
                "but",
                "in",
                "on",
                "at",
                "to",
                "for",
                "of",
                "with",
                "by",
                "is",
                "are",
                "was",
                "were",
                "be",
                "been",
                "have",
                "has",
                "had",
                "do",
                "does",
                "did",
                "will",
                "would",
                "could",
                "should",
                "can",
                "may",
            ]
        )

        words = [word.strip(".,!?;:") for word in text.split()]
        keywords = [
            word for word in words if len(word) > 3 and word not in common_words
        ]

        return keywords

    def _infer_query_intent(self, query: str) -> str:
        """Infer the intent of the query"""
        query_lower = query.lower()

        if any(word in query_lower for word in ["how", "what", "approach", "strategy"]):
            return "strategic_guidance"
        elif any(word in query_lower for word in ["should", "recommend", "suggest"]):
            return "recommendation_seeking"
        elif any(word in query_lower for word in ["analyze", "evaluate", "assess"]):
            return "analysis_request"
        elif any(word in query_lower for word in ["problem", "issue", "challenge"]):
            return "problem_solving"
        elif any(
            word in query_lower for word in ["opportunity", "potential", "growth"]
        ):
            return "opportunity_exploration"
        else:
            return "general_inquiry"

    def _assess_intent_alignment(
        self, output: ConsultantOutput, query_intent: str
    ) -> float:
        """Assess how well output aligns with query intent"""

        output_text = " ".join(
            [
                output.executive_summary,
                " ".join(output.key_insights),
                " ".join(output.recommendations),
            ]
        ).lower()

        intent_indicators = {
            "strategic_guidance": ["strategy", "approach", "framework", "method"],
            "recommendation_seeking": ["recommend", "suggest", "should", "advise"],
            "analysis_request": ["analysis", "evaluation", "assessment", "findings"],
            "problem_solving": ["solution", "solve", "address", "resolve"],
            "opportunity_exploration": [
                "opportunity",
                "potential",
                "growth",
                "leverage",
            ],
            "general_inquiry": ["overview", "understanding", "insights", "information"],
        }

        relevant_indicators = intent_indicators.get(query_intent, [])
        indicator_count = sum(
            1 for indicator in relevant_indicators if indicator in output_text
        )

        return (
            min(1.0, indicator_count / len(relevant_indicators))
            if relevant_indicators
            else 0.5
        )

    def _assess_response_completeness(
        self, output: ConsultantOutput, query: str
    ) -> float:
        """Assess completeness of response"""

        # Simple completeness indicators
        completeness_factors = [
            len(output.executive_summary) > 200,  # Substantial summary
            len(output.key_insights) >= 3,  # Multiple insights
            len(output.recommendations) >= 3,  # Multiple recommendations
            len(output.evidence_sources) >= 5,  # Adequate research
            len(output.mental_models_used) >= 2,  # Multiple perspectives
        ]

        return sum(completeness_factors) / len(completeness_factors)

    def _assess_focus_maintenance(self, output: ConsultantOutput, query: str) -> float:
        """Assess whether output maintains focus on the query"""

        # Extract main topics from query
        query_keywords = set(self._extract_keywords(query.lower()))

        # Check presence in different output sections
        sections = [
            output.executive_summary,
            " ".join(output.key_insights),
            " ".join(output.recommendations),
        ]

        focus_scores = []
        for section in sections:
            section_keywords = set(self._extract_keywords(section.lower()))
            if query_keywords and section_keywords:
                overlap = len(query_keywords & section_keywords) / len(query_keywords)
                focus_scores.append(overlap)

        return statistics.mean(focus_scores) if focus_scores else 0.5

    # Implementation feasibility helper methods

    def _assess_recommendation_specificity(self, recommendations: List[str]) -> float:
        """Assess specificity and actionability of recommendations"""

        if not recommendations:
            return 0.0

        specificity_indicators = [
            "implement",
            "create",
            "establish",
            "develop",
            "design",
            "execute",
            "launch",
            "initiate",
            "deploy",
            "install",
            "measure",
            "track",
            "monitor",
            "evaluate",
            "assess",
        ]

        specific_count = 0
        for rec in recommendations:
            rec_lower = rec.lower()
            if any(indicator in rec_lower for indicator in specificity_indicators):
                specific_count += 1

        return specific_count / len(recommendations)

    def _assess_resource_awareness(self, output: ConsultantOutput) -> float:
        """Assess awareness of resource requirements"""

        all_text = " ".join(
            [
                output.executive_summary,
                " ".join(output.key_insights),
                " ".join(output.recommendations),
            ]
        ).lower()

        resource_indicators = [
            "budget",
            "cost",
            "investment",
            "resource",
            "funding",
            "time",
            "timeline",
            "schedule",
            "team",
            "staff",
            "skills",
            "expertise",
            "technology",
            "infrastructure",
        ]

        resource_mentions = sum(
            1 for indicator in resource_indicators if indicator in all_text
        )
        return min(1.0, resource_mentions / 5)  # Normalize to max of 5 mentions

    def _assess_timeline_realism(self, output: ConsultantOutput) -> float:
        """Assess realism of implied timelines"""

        all_text = " ".join(output.recommendations).lower()

        # Look for timeline indicators
        timeline_indicators = [
            "immediate",
            "quickly",
            "asap",
            "urgent",  # Very short term
            "short term",
            "weeks",
            "month",
            "quarter",  # Short term
            "long term",
            "year",
            "years",
            "gradual",  # Long term
        ]

        has_timeline_consideration = any(
            indicator in all_text for indicator in timeline_indicators
        )

        # Check for balance between urgent and gradual recommendations
        urgent_count = sum(
            1 for word in ["immediate", "quickly", "urgent", "asap"] if word in all_text
        )
        gradual_count = sum(
            1 for word in ["gradual", "long term", "year"] if word in all_text
        )

        if has_timeline_consideration:
            if urgent_count > 0 and gradual_count > 0:
                return 0.9  # Balanced approach
            elif urgent_count > 0 or gradual_count > 0:
                return 0.7  # Some consideration
            else:
                return 0.6  # Vague consideration
        else:
            return 0.4  # Limited timeline consideration

    def _assess_stakeholder_consideration(self, output: ConsultantOutput) -> float:
        """Assess consideration of different stakeholders"""

        all_text = " ".join(
            [
                output.executive_summary,
                " ".join(output.key_insights),
                " ".join(output.recommendations),
            ]
        ).lower()

        stakeholder_terms = [
            "customer",
            "client",
            "user",
            "employee",
            "staff",
            "team",
            "investor",
            "shareholder",
            "partner",
            "vendor",
            "supplier",
            "regulator",
            "community",
            "competitor",
            "management",
            "leadership",
        ]

        stakeholder_mentions = sum(1 for term in stakeholder_terms if term in all_text)
        return min(
            1.0, stakeholder_mentions / 8
        )  # Normalize to max of 8 stakeholder types

    def _assess_risk_mitigation_planning(self, output: ConsultantOutput) -> float:
        """Assess presence of risk mitigation in recommendations"""

        rec_text = " ".join(output.recommendations).lower()

        mitigation_indicators = [
            "mitigate",
            "reduce risk",
            "contingency",
            "backup plan",
            "monitor",
            "track",
            "review",
            "assess",
            "evaluate",
            "if",
            "when",
            "should",
            "alternative",
            "fallback",
        ]

        mitigation_count = sum(
            1 for indicator in mitigation_indicators if indicator in rec_text
        )
        return min(1.0, mitigation_count / 5)

    # Risk thoroughness helper methods

    def _assess_red_team_effectiveness(self, red_team_results: Dict[str, Any]) -> float:
        """Assess effectiveness of Red Team Council validation"""

        if not red_team_results:
            return 0.0

        # Count successful challenger executions
        successful_challenges = 0
        total_challenges = 0
        total_critiques = 0

        for challenger_name, results in red_team_results.items():
            if isinstance(results, dict):
                total_challenges += 1
                if results.get("status") == "success":
                    successful_challenges += 1
                    critiques = results.get("critiques", [])
                    total_critiques += len(critiques)

        if total_challenges == 0:
            return 0.0

        success_rate = successful_challenges / total_challenges
        critique_density = total_critiques / max(successful_challenges, 1)

        # Effective Red Team should have high success rate and meaningful critiques
        effectiveness = success_rate * 0.7 + min(1.0, critique_density / 3) * 0.3

        return effectiveness

    def _count_risk_considerations(self, output: ConsultantOutput) -> float:
        """Count risk-related considerations in output"""

        all_text = " ".join(
            [
                output.executive_summary,
                " ".join(output.key_insights),
                " ".join(output.recommendations),
            ]
        ).lower()

        risk_terms = [
            "risk",
            "threat",
            "challenge",
            "problem",
            "issue",
            "concern",
            "danger",
            "vulnerability",
            "weakness",
            "failure",
            "downside",
            "limitation",
            "constraint",
        ]

        risk_mentions = sum(1 for term in risk_terms if term in all_text)
        return min(1.0, risk_mentions / 10)  # Normalize to max of 10 mentions

    def _assess_mitigation_strategies(self, output: ConsultantOutput) -> float:
        """Assess presence of mitigation strategies"""

        all_text = " ".join(output.recommendations).lower()

        mitigation_terms = [
            "mitigate",
            "prevent",
            "avoid",
            "reduce",
            "minimize",
            "contingency",
            "backup",
            "alternative",
            "fallback",
            "monitor",
            "control",
            "manage",
            "hedge",
            "diversify",
        ]

        mitigation_mentions = sum(1 for term in mitigation_terms if term in all_text)
        return min(1.0, mitigation_mentions / 5)

    def _assess_scenario_consideration(self, output: ConsultantOutput) -> float:
        """Assess consideration of different scenarios"""

        all_text = " ".join(
            [
                output.executive_summary,
                " ".join(output.key_insights),
                " ".join(output.recommendations),
            ]
        ).lower()

        scenario_indicators = [
            "scenario",
            "case",
            "situation",
            "condition",
            "if",
            "when",
            "should",
            "might",
            "could",
            "best case",
            "worst case",
            "likely",
            "unlikely",
        ]

        scenario_mentions = sum(
            1 for indicator in scenario_indicators if indicator in all_text
        )
        return min(1.0, scenario_mentions / 8)

    # Novel insights helper methods

    def _assess_insight_sophistication(self, insights: List[str]) -> float:
        """Assess sophistication of insights"""

        if not insights:
            return 0.0

        sophisticated_indicators = [
            "however",
            "paradox",
            "tension",
            "tradeoff",
            "balance",
            "systematic",
            "leverage",
            "catalyst",
            "amplify",
            "synergy",
            "counterintuitive",
            "unexpected",
            "surprising",
            "contrary",
        ]

        sophisticated_count = 0
        for insight in insights:
            insight_lower = insight.lower()
            if any(
                indicator in insight_lower for indicator in sophisticated_indicators
            ):
                sophisticated_count += 1

        return sophisticated_count / len(insights)

    def _assess_mental_model_creativity(self, mental_models: List[str]) -> float:
        """Assess creativity in mental model selection"""

        # Common models (lower creativity score)
        common_models = [
            "first_principles",
            "systems_thinking",
            "design_thinking",
            "swot_analysis",
            "financial_analysis",
            "risk_assessment",
        ]

        uncommon_count = 0
        for model in mental_models:
            if not any(common in model.lower() for common in common_models):
                uncommon_count += 1

        if not mental_models:
            return 0.0

        creativity_score = uncommon_count / len(mental_models)
        return creativity_score

    def _assess_connection_creativity(self, output: ConsultantOutput) -> float:
        """Assess creativity in making connections between concepts"""

        all_text = " ".join(
            [output.executive_summary, " ".join(output.key_insights)]
        ).lower()

        connection_indicators = [
            "connects to",
            "links with",
            "relates to",
            "similar to",
            "analogous to",
            "like",
            "as with",
            "parallels",
            "cross-cutting",
            "interdisciplinary",
            "holistic",
        ]

        connection_count = sum(
            1 for indicator in connection_indicators if indicator in all_text
        )
        return min(1.0, connection_count / 5)

    def _identify_counter_intuitive_insights(self, output: ConsultantOutput) -> float:
        """Identify counter-intuitive or surprising insights"""

        all_text = " ".join(output.key_insights).lower()

        counter_intuitive_indicators = [
            "counterintuitive",
            "surprising",
            "unexpected",
            "contrary",
            "opposite",
            "paradox",
            "ironic",
            "contradicts",
            "challenges conventional",
            "goes against",
            "defies",
        ]

        counter_intuitive_count = sum(
            1 for indicator in counter_intuitive_indicators if indicator in all_text
        )
        return min(1.0, counter_intuitive_count / 3)

    # Calculation helper methods

    def _calculate_recommended_weight(
        self, overall_merit: float, query_fitness: float
    ) -> float:
        """Calculate recommended weight for this consultant"""

        # Combined score emphasizing both quality and fitness
        combined_score = overall_merit * 0.6 + query_fitness * 0.4

        # Convert to recommended weight (will be normalized across all consultants)
        return combined_score

    def _identify_strengths(
        self, criterion_scores: Dict[MeritCriterion, MeritScore]
    ) -> List[str]:
        """Identify strengths based on criterion scores"""

        strengths = []

        for criterion, score in criterion_scores.items():
            if score.score >= self.excellence_threshold:
                strengths.append(f"Excellent {criterion.value.replace('_', ' ')}")
            elif score.score >= self.good_threshold:
                strengths.append(f"Strong {criterion.value.replace('_', ' ')}")

        return strengths[:3]  # Top 3 strengths

    def _identify_weaknesses(
        self, criterion_scores: Dict[MeritCriterion, MeritScore]
    ) -> List[str]:
        """Identify weaknesses based on criterion scores"""

        weaknesses = []

        for criterion, score in criterion_scores.items():
            if score.score < self.acceptable_threshold:
                weaknesses.append(f"Limited {criterion.value.replace('_', ' ')}")

        return weaknesses[:2]  # Top 2 weaknesses

    def _add_comparative_rankings(
        self, assessments: Dict[ConsultantRole, ConsultantMeritAssessment]
    ):
        """Add comparative rankings across all consultants"""

        # For each criterion, rank consultants
        for criterion in MeritCriterion:
            # Get scores for this criterion from all consultants
            criterion_scores = []
            for role, assessment in assessments.items():
                if criterion in assessment.criterion_scores:
                    score = assessment.criterion_scores[criterion].score
                    criterion_scores.append((score, role))

            # Sort by score (descending) and assign rankings
            criterion_scores.sort(reverse=True)

            for rank, (score, role) in enumerate(criterion_scores, 1):
                assessments[role].criterion_scores[criterion].relative_ranking = rank

    def _adjust_for_query_context(
        self,
        assessments: Dict[ConsultantRole, ConsultantMeritAssessment],
        query_context: Dict[str, Any],
    ):
        """Adjust assessments based on specific query context"""

        # Extract context factors
        urgency = query_context.get("urgency", "medium")
        complexity = query_context.get("complexity", "medium")
        risk_tolerance = query_context.get("risk_tolerance", "medium")

        for role, assessment in assessments.items():
            # Adjust weights based on context
            context_adjustment = 1.0

            if urgency == "high":
                # Favor implementation feasibility and practical recommendations
                if (
                    MeritCriterion.IMPLEMENTATION_FEASIBILITY
                    in assessment.criterion_scores
                ):
                    feasibility_score = assessment.criterion_scores[
                        MeritCriterion.IMPLEMENTATION_FEASIBILITY
                    ].score
                    context_adjustment += (feasibility_score - 0.5) * 0.2

            if complexity == "high":
                # Favor sophisticated analysis and novel insights
                if MeritCriterion.NOVEL_INSIGHTS in assessment.criterion_scores:
                    novelty_score = assessment.criterion_scores[
                        MeritCriterion.NOVEL_INSIGHTS
                    ].score
                    context_adjustment += (novelty_score - 0.5) * 0.2

            if risk_tolerance == "low":
                # Favor risk thoroughness
                if MeritCriterion.RISK_THOROUGHNESS in assessment.criterion_scores:
                    risk_score = assessment.criterion_scores[
                        MeritCriterion.RISK_THOROUGHNESS
                    ].score
                    context_adjustment += (risk_score - 0.5) * 0.3

            # Apply adjustment to recommended weight
            assessment.recommended_weight *= context_adjustment
            assessment.recommended_weight = min(1.0, assessment.recommended_weight)

    # Additional helper methods for specific model/evidence/role assessments

    def _assess_model_suitability_for_query(
        self, mental_models: List[str], query: str
    ) -> float:
        """Assess how suitable the mental models are for the query"""

        query_lower = query.lower()

        # Query-model suitability mapping
        if any(word in query_lower for word in ["risk", "failure", "problem"]):
            # Risk-focused queries benefit from risk management models
            suitable_models = ["inversion", "munger", "risk", "bias"]
            suitability = sum(
                1
                for model in mental_models
                if any(suitable in model.lower() for suitable in suitable_models)
            )
            return min(1.0, suitability / len(mental_models)) if mental_models else 0.0

        elif any(word in query_lower for word in ["user", "customer", "experience"]):
            # User-focused queries benefit from design thinking
            suitable_models = ["design", "user", "behavioral", "psychology"]
            suitability = sum(
                1
                for model in mental_models
                if any(suitable in model.lower() for suitable in suitable_models)
            )
            return min(1.0, suitability / len(mental_models)) if mental_models else 0.0

        elif any(word in query_lower for word in ["system", "process", "organization"]):
            # System-focused queries benefit from systems thinking
            suitable_models = ["system", "ackoff", "holistic", "process"]
            suitability = sum(
                1
                for model in mental_models
                if any(suitable in model.lower() for suitable in suitable_models)
            )
            return min(1.0, suitability / len(mental_models)) if mental_models else 0.0

        else:
            # General queries benefit from diverse models
            model_categories = len(set(self._categorize_mental_models(mental_models)))
            return min(1.0, model_categories / 4)  # Up to 4 categories

    def _assess_evidence_strategy_fit(
        self, output: ConsultantOutput, query: str
    ) -> float:
        """Assess how well evidence strategy fits the query"""

        query_lower = query.lower()
        evidence_categories = self._categorize_evidence_sources(output.evidence_sources)

        # Query-evidence fit assessment
        if any(
            word in query_lower for word in ["financial", "cost", "revenue", "profit"]
        ):
            return 1.0 if "financial" in evidence_categories else 0.5
        elif any(word in query_lower for word in ["market", "competition", "industry"]):
            return 1.0 if "industry" in evidence_categories else 0.5
        elif any(word in query_lower for word in ["research", "analysis", "study"]):
            return 1.0 if "academic" in evidence_categories else 0.5
        else:
            # General queries benefit from diverse evidence
            return min(1.0, len(evidence_categories) / 4)

    def _assess_role_alignment_with_query(
        self, role: ConsultantRole, query: str
    ) -> float:
        """Assess how well consultant role aligns with query"""

        query_lower = query.lower()

        role_query_fit = {
            ConsultantRole.ANALYST: [
                "analyze",
                "data",
                "research",
                "study",
                "evaluate",
            ],
            ConsultantRole.STRATEGIST: [
                "strategy",
                "plan",
                "approach",
                "direction",
                "future",
            ],
            ConsultantRole.DEVIL_ADVOCATE: [
                "risk",
                "challenge",
                "problem",
                "failure",
                "critical",
            ],
        }

        suitable_terms = role_query_fit.get(role, [])
        alignment_count = sum(1 for term in suitable_terms if term in query_lower)

        return (
            min(1.0, alignment_count / len(suitable_terms)) if suitable_terms else 0.5
        )