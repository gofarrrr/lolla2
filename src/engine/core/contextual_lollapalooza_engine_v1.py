"""
METIS V5 Operation "Calibration Duel" - V1 Non-Weighted Engine
This is a copy of the ContextualLollapaloozaEngine BEFORE Operation "N-Way Calibration"
for head-to-head comparison testing.

V1 ENGINE (BASELINE): Uses original non-weighted scoring algorithm
- No N-Way taxonomy classification
- No base_weight, applicability_score, or insight_potential_score
- Simple baseline multi-dimensional scoring
- No Operation "N-Way Calibration" enhancements
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from src.engine.contracts.analysis_contracts import (
    QueryContext,
    ConsultantProfile,
)

logger = logging.getLogger(__name__)


@dataclass
class V1ContextualLollapalozzaScore:
    """V1 Original contextual score without calibration weights."""

    consultant_id: str
    nway_interaction_id: str
    query_context: QueryContext
    base_affinity_score: float
    domain_relevance_score: float
    cognitive_style_match: float
    complexity_fit_score: float
    historical_performance: float
    dynamic_lollapalooza_score: float  # V1: Simple weighted sum
    confidence_level: float
    reasoning: str


class ContextualLollapaloozaEngineV1:
    """
    V1 BASELINE: Original ContextualLollapaloozaEngine without N-Way Calibration.
    This represents the system BEFORE Operation "N-Way Calibration".
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            "🔧 Initializing ContextualLollapaloozaEngine V1 (Non-Weighted Baseline)"
        )

        # V1 Original scoring weights (no calibration)
        self.scoring_weights = {
            "base_affinity": 0.35,
            "domain_relevance": 0.25,
            "cognitive_style_match": 0.20,
            "complexity_fit": 0.15,
            "historical_performance": 0.05,
        }

        # Cache for performance
        self.contextual_scores_cache = {}
        self.performance_history = {}

        # Initialize Supabase connection
        try:
            from src.core.supabase_platform import SupabasePlatform

            self.supabase = SupabasePlatform()
        except Exception as e:
            self.logger.warning(f"⚠️ Supabase connection failed in V1 engine: {e}")
            self.supabase = None

        # V1 Original consultant profiles (simplified)
        self.consultant_profiles = self._initialize_v1_consultant_profiles()

    def _initialize_v1_consultant_profiles(self) -> Dict[str, ConsultantProfile]:
        """Initialize V1 original consultant profiles without calibration features."""

        profiles = {
            "strategic_analyst_001": ConsultantProfile(
                consultant_id="strategic_analyst_001",
                name="Strategic Business Analyst",
                type="strategic_analyst",
                primary_specializations=[
                    "strategic_planning",
                    "market_analysis",
                    "competitive_intelligence",
                ],
                expertise_domains=[
                    "strategy",
                    "finance",
                    "market_research",
                    "competitive_analysis",
                ],
                thinking_style_strengths={
                    "systems-thinking": 0.90,
                    "outside-view": 0.85,
                    "second-order-thinking": 0.88,
                    "critical-thinking": 0.82,
                },
                cognitive_complexity_preference=4,
                mental_model_affinities=[
                    "systems-thinking",
                    "outside-view",
                    "second-order-thinking",
                    "root-cause-analysis",
                    "competitive-moats",
                ],
                avg_response_quality=0.0,  # V1: No historical tracking
                preferred_problem_types=[
                    "strategic_planning",
                    "market_analysis",
                    "competitive_strategy",
                ],
            ),
            "financial_analyst_002": ConsultantProfile(
                consultant_id="financial_analyst_002",
                name="Senior Financial Analyst",
                type="financial_analyst",
                primary_specializations=[
                    "financial_modeling",
                    "valuation",
                    "risk_assessment",
                ],
                expertise_domains=[
                    "finance",
                    "economics",
                    "investment",
                    "risk_management",
                ],
                thinking_style_strengths={
                    "outside-view": 0.85,
                    "probability-theory-probabilistic-thinking": 0.90,
                    "margin-of-safety": 0.92,
                    "inversion": 0.80,
                },
                cognitive_complexity_preference=3,
                mental_model_affinities=[
                    "outside-view",
                    "probability-theory-probabilistic-thinking",
                    "margin-of-safety",
                    "inversion",
                    "second-order-thinking",
                ],
                avg_response_quality=0.0,
                preferred_problem_types=[
                    "financial_analysis",
                    "risk_assessment",
                    "investment_evaluation",
                ],
            ),
            "market_researcher_003": ConsultantProfile(
                consultant_id="market_researcher_003",
                name="Market Research Specialist",
                type="market_researcher",
                primary_specializations=[
                    "market_research",
                    "consumer_insights",
                    "data_analysis",
                ],
                expertise_domains=[
                    "market_research",
                    "consumer_behavior",
                    "data_analytics",
                ],
                thinking_style_strengths={
                    "outside-view": 0.88,
                    "statistics-concepts": 0.85,
                    "evidence-based-reasoning": 0.90,
                    "pattern-recognition": 0.82,
                },
                cognitive_complexity_preference=3,
                mental_model_affinities=[
                    "outside-view",
                    "statistics-concepts",
                    "evidence-based-reasoning",
                    "pattern-recognition",
                    "scientific-method-evidence-testing",
                ],
                avg_response_quality=0.0,
                preferred_problem_types=[
                    "market_research",
                    "customer_analysis",
                    "trend_analysis",
                ],
            ),
        }

        return profiles

    async def calculate_contextual_lollapalooza_score(
        self, consultant_id: str, nway_interaction_id: str, query_context: QueryContext
    ) -> V1ContextualLollapalozzaScore:
        """
        V1 ORIGINAL: Calculate contextual score WITHOUT N-Way Calibration weights.
        This is the baseline algorithm before Operation "N-Way Calibration".
        """

        try:
            # Get consultant profile
            if consultant_id not in self.consultant_profiles:
                raise ValueError(f"Unknown consultant: {consultant_id}")
            consultant = self.consultant_profiles[consultant_id]

            # Get NWAY interaction data (V1: ignore calibration columns)
            nway_data = await self._get_nway_interaction_v1(nway_interaction_id)
            if not nway_data:
                raise ValueError(f"NWAY interaction not found: {nway_interaction_id}")

            # V1 ORIGINAL: Calculate multi-dimensional scores (no calibration)
            base_affinity = self._calculate_base_affinity(consultant, nway_data)
            domain_relevance = self._calculate_domain_relevance(
                consultant, query_context
            )
            cognitive_match = self._calculate_cognitive_style_match(
                consultant, nway_data
            )
            complexity_fit = self._calculate_complexity_fit(
                consultant, nway_data, query_context
            )
            historical_perf = await self._get_historical_performance(
                consultant_id, nway_interaction_id, query_context
            )

            # V1 ORIGINAL: Simple weighted final score (NO CALIBRATION)
            final_score = (
                base_affinity * self.scoring_weights["base_affinity"]
                + domain_relevance * self.scoring_weights["domain_relevance"]
                + cognitive_match * self.scoring_weights["cognitive_style_match"]
                + complexity_fit * self.scoring_weights["complexity_fit"]
                + historical_perf * self.scoring_weights["historical_performance"]
            )

            # Generate V1 reasoning (no calibration details)
            reasoning = self._generate_v1_scoring_reasoning(
                consultant,
                nway_data,
                query_context,
                base_affinity,
                domain_relevance,
                cognitive_match,
                complexity_fit,
                historical_perf,
            )

            # Calculate confidence level
            confidence = self._calculate_confidence_level(
                consultant, nway_data, query_context
            )

            contextual_score = V1ContextualLollapalozzaScore(
                consultant_id=consultant_id,
                nway_interaction_id=nway_interaction_id,
                query_context=query_context,
                base_affinity_score=base_affinity,
                domain_relevance_score=domain_relevance,
                cognitive_style_match=cognitive_match,
                complexity_fit_score=complexity_fit,
                historical_performance=historical_perf,
                dynamic_lollapalooza_score=final_score,  # V1: No calibration multipliers
                confidence_level=confidence,
                reasoning=reasoning,
            )

            # Cache result
            cache_key = f"{consultant_id}_{nway_interaction_id}_{query_context.domain}_{query_context.complexity_level}"
            self.contextual_scores_cache[cache_key] = contextual_score

            self.logger.info(
                f"🎯 V1 Score: {consultant_id} + {nway_interaction_id} + {query_context.domain} = {final_score:.3f}"
            )

            return contextual_score

        except Exception as e:
            self.logger.error(
                f"❌ V1 Error calculating contextual lollapalooza score: {e}"
            )
            # Return low-confidence default score
            return V1ContextualLollapalozzaScore(
                consultant_id=consultant_id,
                nway_interaction_id=nway_interaction_id,
                query_context=query_context,
                base_affinity_score=0.1,
                domain_relevance_score=0.1,
                cognitive_style_match=0.1,
                complexity_fit_score=0.1,
                historical_performance=0.1,
                dynamic_lollapalooza_score=0.1,
                confidence_level=0.1,
                reasoning=f"V1 Error calculating score: {e}",
            )

    async def _get_nway_interaction_v1(
        self, interaction_id: str
    ) -> Optional[Dict[str, Any]]:
        """V1 Original: Get NWAY interaction data ignoring calibration columns."""

        try:
            if self.supabase:
                # V1: Select only original columns, ignore calibration fields
                response = (
                    self.supabase.table("nway_interactions")
                    .select(
                        "interaction_id, type, models_involved, primary_model_context, "
                        "emergent_effect_summary, synergy_description, mechanism_description"
                    )
                    .eq("interaction_id", interaction_id)
                    .single()
                    .execute()
                )
                return response.data
            else:
                # Fallback for testing
                return {
                    "interaction_id": interaction_id,
                    "type": "PURE_NWAY",
                    "models_involved": [
                        "systems-thinking",
                        "outside-view",
                        "second-order-thinking",
                    ],
                    "primary_model_context": "strategic-analysis",
                }
        except Exception as e:
            self.logger.error(
                f"❌ V1 Error fetching NWAY interaction {interaction_id}: {e}"
            )
            return None

    def _calculate_base_affinity(
        self, consultant: ConsultantProfile, nway_data: Dict[str, Any]
    ) -> float:
        """V1 Original: Calculate base affinity without calibration."""

        nway_models = nway_data.get("models_involved", [])
        if not nway_models:
            return 0.1

        # V1: Simple affinity calculation
        affinity_score = 0.0
        model_count = 0

        for model in nway_models:
            if isinstance(model, str) and model in consultant.mental_model_affinities:
                affinity_score += consultant.thinking_style_strengths.get(model, 0.5)
                model_count += 1

        if model_count == 0:
            return 0.2  # V1: Low but not zero for unknown models

        return min(affinity_score / model_count, 1.0)

    def _calculate_domain_relevance(
        self, consultant: ConsultantProfile, query_context: QueryContext
    ) -> float:
        """V1 Original: Calculate domain relevance."""

        query_domain = query_context.domain
        expertise_domains = consultant.expertise_domains

        # V1: Simple domain matching
        if query_domain in expertise_domains:
            return 0.9

        # V1: Partial matches
        domain_keywords = {
            "strategy": ["strategic", "market", "competitive", "business"],
            "finance": ["financial", "economic", "investment"],
            "operations": ["operational", "process", "efficiency"],
        }

        query_keywords = domain_keywords.get(query_domain, [query_domain])

        relevance_score = 0.0
        for domain in expertise_domains:
            for keyword in query_keywords:
                if keyword.lower() in domain.lower():
                    relevance_score += 0.3

        return min(relevance_score, 0.8)

    def _calculate_cognitive_style_match(
        self, consultant: ConsultantProfile, nway_data: Dict[str, Any]
    ) -> float:
        """V1 Original: Calculate cognitive style match."""

        # V1: Simple thinking style matching
        consultant_strengths = consultant.thinking_style_strengths
        nway_models = nway_data.get("models_involved", [])

        if not nway_models:
            return 0.5

        style_match_score = 0.0
        style_count = 0

        for model in nway_models:
            if isinstance(model, str) and model in consultant_strengths:
                style_match_score += consultant_strengths[model]
                style_count += 1

        if style_count == 0:
            return 0.4

        return min(style_match_score / style_count, 1.0)

    def _calculate_complexity_fit(
        self,
        consultant: ConsultantProfile,
        nway_data: Dict[str, Any],
        query_context: QueryContext,
    ) -> float:
        """V1 Original: Calculate complexity fit."""

        consultant_preference = consultant.cognitive_complexity_preference
        query_complexity = query_context.complexity_level

        # V1: Simple complexity matching
        complexity_diff = abs(consultant_preference - query_complexity)

        if complexity_diff == 0:
            return 1.0
        elif complexity_diff == 1:
            return 0.8
        elif complexity_diff == 2:
            return 0.6
        else:
            return 0.4

    async def _get_historical_performance(
        self, consultant_id: str, nway_interaction_id: str, query_context: QueryContext
    ) -> float:
        """V1 Original: Get historical performance (minimal tracking)."""

        # V1: No historical data - return neutral
        return 0.5

    def _generate_v1_scoring_reasoning(
        self,
        consultant: ConsultantProfile,
        nway_data: Dict[str, Any],
        query_context: QueryContext,
        base_affinity: float,
        domain_relevance: float,
        cognitive_match: float,
        complexity_fit: float,
        historical_perf: float,
    ) -> str:
        """V1 Original: Generate reasoning WITHOUT calibration details."""

        reasoning_parts = []

        # V1: Basic reasoning without calibration
        if base_affinity > 0.7:
            reasoning_parts.append(f"🔥 Strong model affinity ({base_affinity:.2f})")
        else:
            reasoning_parts.append(f"⚠️ Limited model affinity ({base_affinity:.2f})")

        if domain_relevance > 0.7:
            reasoning_parts.append(f"🎯 High domain relevance ({domain_relevance:.2f})")
        else:
            reasoning_parts.append(
                f"❓ Moderate domain relevance ({domain_relevance:.2f})"
            )

        if cognitive_match > 0.7:
            reasoning_parts.append(f"🧠 Strong cognitive fit ({cognitive_match:.2f})")
        else:
            reasoning_parts.append(
                f"🤔 Cognitive adjustment needed ({cognitive_match:.2f})"
            )

        # V1: No historical data
        reasoning_parts.append("📈 No historical data - V1 baseline")

        return " | ".join(reasoning_parts)

    def _calculate_confidence_level(
        self,
        consultant: ConsultantProfile,
        nway_data: Dict[str, Any],
        query_context: QueryContext,
    ) -> float:
        """V1 Original: Calculate confidence level."""

        confidence_factors = []

        # V1: Simple confidence calculation
        models_count = len(nway_data.get("models_involved", []))
        if models_count >= 3:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.4)

        # V1: Basic profile completeness
        if len(consultant.mental_model_affinities) >= 3:
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.4)

        # V1: No historical data
        confidence_factors.append(0.4)

        return sum(confidence_factors) / len(confidence_factors)

    async def run_contextual_comparison(
        self,
        query_context: QueryContext,
        nway_interactions: List[str],
        max_consultants: int = 5,
    ) -> List[Tuple[str, str, V1ContextualLollapalozzaScore]]:
        """
        V1 Original: Run contextual comparison WITHOUT calibration weights.
        """

        self.logger.info(
            f"🔍 V1 Running contextual comparison for {query_context.domain} query"
        )

        results = []

        # Test all consultant + NWAY combinations
        for consultant_id in self.consultant_profiles.keys():
            for nway_id in nway_interactions[:5]:  # Limit for efficiency
                try:
                    score = await self.calculate_contextual_lollapalooza_score(
                        consultant_id, nway_id, query_context
                    )
                    results.append((consultant_id, nway_id, score))
                except Exception as e:
                    self.logger.error(
                        f"❌ V1 Error scoring {consultant_id} + {nway_id}: {e}"
                    )

        # Sort by V1 lollapalooza score
        results.sort(key=lambda x: x[2].dynamic_lollapalooza_score, reverse=True)

        return results[: max_consultants * len(nway_interactions)][
            :10
        ]  # Top 10 combinations

    async def process_with_forced_nway(
        self, query: str, forced_nway_id: str
    ) -> Dict[str, Any]:
        """V1 Original: Process with forced NWAY (for gauntlet testing)."""
        try:
            self.logger.info(
                f"🥊 V1 Gauntlet Mode: Forcing NWAY {forced_nway_id} for query: {query[:60]}..."
            )

            # Create query context
            from src.engine.contracts.analysis_contracts import QueryContext

            query_context = QueryContext(
                query=query,
                domain="test",
                complexity_level=3,
                urgency="medium",
                thinking_styles_required=["analytical"],
            )

            # Get best consultant for this NWAY (V1 logic)
            results = await self.run_contextual_comparison(
                query_context, [forced_nway_id], max_consultants=1
            )

            if not results:
                return {
                    "final_response": f"V1 Error: No suitable consultant found for NWAY {forced_nway_id}"
                }

            best_consultant_id, nway_id, score = results[0]

            # Generate V1 response
            response = f"""V1 BASELINE Analysis using {nway_id} methodology by {best_consultant_id}:

Score: {score.dynamic_lollapalooza_score:.3f} | Reasoning: {score.reasoning}

Query: {query}

This analysis was generated using the V1 BASELINE (non-weighted) ContextualLollapaloozaEngine. This represents the system BEFORE Operation N-Way Calibration. The consultant '{best_consultant_id}' was selected using the original multi-dimensional scoring without calibration weights:
- Base Affinity: {score.base_affinity_score:.3f}
- Domain Relevance: {score.domain_relevance_score:.3f}  
- Cognitive Style Match: {score.cognitive_style_match:.3f}

This V1 baseline uses simple weighted scoring without the sophisticated calibration system introduced in Operation N-Way Calibration."""

            return {
                "final_response": response,
                "score": score.dynamic_lollapalooza_score,
                "consultant": best_consultant_id,
                "version": "V1_BASELINE",
            }

        except Exception as e:
            self.logger.error(f"❌ V1 Forced NWAY processing failed: {e}")
            return {"final_response": f"V1 Error in forced NWAY processing: {str(e)}"}

    async def process_baseline_mode(self, query: str) -> Dict[str, Any]:
        """V1 Original: Process in baseline mode (simple)."""
        try:
            self.logger.info(f"📊 V1 Baseline Mode: Processing query: {query[:60]}...")

            # V1: Even simpler baseline response
            response = f"""V1 BASELINE Analysis (No NWAY Synergy):

Query: {query}

This is the V1 baseline analysis without N-Way mental model interactions. This represents the original system before calibration enhancements, using basic multi-dimensional scoring without sophisticated cognitive orchestration.

V1 Limitations:
- No N-Way calibration weights
- Basic consultant-NWAY matching
- Simple multi-dimensional scoring  
- No insight potential optimization
- Limited contextual adaptation

This V1 baseline serves as the comparison point for measuring the improvements achieved through Operation N-Way Calibration."""

            return {
                "final_response": response,
                "score": 0.4,
                "method": "V1_baseline",
                "version": "V1_BASELINE",
            }

        except Exception as e:
            self.logger.error(f"❌ V1 Baseline processing failed: {e}")
            return {"final_response": f"V1 Error in baseline processing: {str(e)}"}
