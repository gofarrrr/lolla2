# src/services/selection/scorer.py
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from src.services.selection.contracts import ChemistryContext
from src.core.unified_context_stream import UnifiedContextStream, ContextEventType

# Scoring and compatibility engines
from .cognitive_chemistry_scoring import get_cognitive_chemistry_scoring
from .synergistic_compatibility_matrix import (
    CompatibilityResult,
    ReactionType,
    analyze_nway_combination_compatibility,
)

# Reaction dataclass lives in legacy module
from .cognitive_chemistry_engine import CognitiveChemistryReaction

logger = logging.getLogger(__name__)


class ChemistryScorer:
    """Extracted scoring service implementing the core chemistry calculation.
    Returns a full CognitiveChemistryReaction for compatibility with callers.
    """

    def __init__(self, context_stream: Optional[UnifiedContextStream] = None) -> None:
        self.individual_scorer = get_cognitive_chemistry_scoring()
        self.context_stream = context_stream
        # Weighting factors aligned with legacy
        self.tier_weights = {
            "lollapalooza": 0.4,
            "meta_framework": 0.25,
            "cluster": 0.20,
            "contextual": 0.15,
        }
        self.compatibility_weight = 0.3
        self.stability_requirement = 0.6

    def score(self, ctx: ChemistryContext) -> CognitiveChemistryReaction:
        problem_framework = ctx.problem_framework
        nway_combination = ctx.nway_combination

        combination_ids = [
            nway.get("interaction_id", f"nway_{i}")
            for i, nway in enumerate(nway_combination)
        ]
        logger.info(
            f"🧬 CALCULATING COGNITIVE CHEMISTRY: {len(nway_combination)} NWAYs"
        )
        logger.info(f"   Combination: {combination_ids}")

        # STEP 1: Score each NWAY individually using 4-tier scoring
        individual_scores = self._score_individual_nways(
            problem_framework, nway_combination
        )

        # STEP 2: Analyze all pairwise compatibilities
        compatibility_results = self._analyze_combination_compatibility(
            nway_combination
        )

        # STEP 3: Calculate reaction probability
        reaction_probability = self._calculate_reaction_probability(
            individual_scores, compatibility_results, problem_framework
        )

        # STEP 4: Calculate amplification potential
        amplification_potential = self._calculate_amplification_potential(
            individual_scores, compatibility_results
        )

        # STEP 5: Calculate cognitive efficiency
        cognitive_efficiency = self._calculate_cognitive_efficiency(
            individual_scores, compatibility_results, amplification_potential
        )

        # STEP 6: Calculate stability rating
        stability_rating = self._calculate_stability_rating(
            individual_scores, compatibility_results
        )

        # STEP 7: Calculate overall chemistry score
        overall_chemistry_score = self._calculate_overall_chemistry_score(
            reaction_probability,
            amplification_potential,
            cognitive_efficiency,
            stability_rating,
        )

        # STEP 8: Generate comprehensive analysis
        analysis = self._generate_comprehensive_analysis(
            problem_framework,
            nway_combination,
            individual_scores,
            compatibility_results,
            reaction_probability,
            amplification_potential,
            cognitive_efficiency,
            stability_rating,
            overall_chemistry_score,
        )

        # STEP 9: Create final reaction result
        reaction = CognitiveChemistryReaction(
            nway_combination=combination_ids,
            problem_framework=problem_framework,
            individual_scores=individual_scores,
            compatibility_results={k: v for k, v in compatibility_results.items()},
            reaction_probability=reaction_probability,
            amplification_potential=amplification_potential,
            cognitive_efficiency=cognitive_efficiency,
            stability_rating=stability_rating,
            overall_chemistry_score=overall_chemistry_score,
            **analysis,
            created_at=datetime.now(),
        )

        # Glass-box evidence: record selection evidence (aligned with legacy behavior)
        self._record_consultant_selection_evidence(
            problem_framework=problem_framework,
            selected_combinations=combination_ids,
            chemistry_scores=individual_scores,
            final_score=overall_chemistry_score,
            selection_rationale=analysis["recommendation"],
            risk_factors=analysis["risk_factors"],
            success_factors=analysis["success_factors"],
            confidence_level=analysis["confidence_level"],
        )

        logger.info(
            f"⚡ REACTION COMPLETE: Overall Chemistry Score = {overall_chemistry_score:.3f}"
        )
        logger.info(f"   Quality: {analysis['recommendation']}")
        return reaction

    # ====================== Helpers (migrated from legacy) ======================
    def _score_individual_nways(
        self, problem_framework: str, nway_combination: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        individual_scores: Dict[str, Dict[str, float]] = {}
        for nway in nway_combination:
            nway_id = nway.get("interaction_id", "unknown")
            logger.info(f"  📊 Scoring individual NWAY: {nway_id}")
            scores = self.individual_scorer.score_nway_combination(
                problem_framework, nway
            )
            individual_scores[nway_id] = scores
        return individual_scores

    def _analyze_combination_compatibility(
        self, nway_combination: List[Dict[str, Any]]
    ) -> Dict[str, CompatibilityResult]:
        logger.info(f"  🧪 Analyzing {len(nway_combination)} NWAYs for compatibility")
        return analyze_nway_combination_compatibility(nway_combination)

    def _calculate_reaction_probability(
        self,
        individual_scores: Dict[str, Dict[str, float]],
        compatibility_results: Dict[str, CompatibilityResult],
        problem_framework: str,
    ) -> float:
        base_probabilities: List[float] = []
        for nway_id, scores in individual_scores.items():
            nway_type = self._classify_nway_type(nway_id, scores)
            if nway_type == "lollapalooza":
                prob = (
                    scores.get("environmental_match_score", 0.0) * 0.7
                    + scores.get("predictability", 0.0) * 0.3
                )
            elif nway_type == "meta_framework":
                prob = (
                    scores.get("problem_type_coverage", 0.0) * 0.6
                    + scores.get("transferability", 0.0) * 0.4
                )
            elif nway_type == "cluster":
                prob = (
                    scores.get("domain_specialization", 0.0) * 0.5
                    + scores.get("toolkit_completeness", 0.0) * 0.3
                    + scores.get("instructional_clarity", 0.0) * 0.2
                )
            else:
                prob = (
                    scores.get("task_specificity", 0.0) * 0.6
                    + scores.get("prerequisite_match", 0.0) * 0.4
                )
            base_probabilities.append(prob)
        avg_base_probability = (
            sum(base_probabilities) / len(base_probabilities)
            if base_probabilities
            else 0.0
        )
        compatibility_modifier = self._calculate_compatibility_modifier(
            compatibility_results
        )
        final_probability = max(
            0.0, min(1.0, avg_base_probability * compatibility_modifier)
        )
        logger.info(
            f"    📈 Reaction Probability: {final_probability:.3f} (base: {avg_base_probability:.3f}, compat: {compatibility_modifier:.3f})"
        )
        return final_probability

    def _calculate_compatibility_modifier(
        self, compatibility_results: Dict[str, CompatibilityResult]
    ) -> float:
        if not compatibility_results:
            return 1.0
        modifiers: List[float] = []
        for _, result in compatibility_results.items():
            if result.reaction_type == ReactionType.SYNERGISTIC:
                modifiers.append(1.3)
            elif result.reaction_type == ReactionType.ADDITIVE:
                modifiers.append(1.1)
            elif result.reaction_type == ReactionType.NEUTRAL:
                modifiers.append(1.0)
            elif result.reaction_type == ReactionType.CONFLICTING:
                modifiers.append(0.8)
            else:
                modifiers.append(0.5)
        avg_modifier = sum(modifiers) / len(modifiers)
        return max(0.3, min(1.5, avg_modifier))

    def _calculate_amplification_potential(
        self,
        individual_scores: Dict[str, Dict[str, float]],
        compatibility_results: Dict[str, CompatibilityResult],
    ) -> float:
        max_individual_amplification = 0.0
        total_amplification = 0.0
        for nway_id, scores in individual_scores.items():
            nway_type = self._classify_nway_type(nway_id, scores)
            if nway_type == "lollapalooza":
                individual_amp = scores.get("multiplicative_effect", 1.0)
                max_individual_amplification = max(
                    max_individual_amplification, individual_amp
                )
                total_amplification += individual_amp
            else:
                amplification_factors = {
                    "meta_framework": scores.get("foundational_depth", 0.5) * 2.0,
                    "cluster": scores.get("professional_authenticity", 0.5) * 1.5,
                    "contextual": scores.get("task_specificity", 0.5) * 1.2,
                }
                individual_amp = amplification_factors.get(nway_type, 1.0)
                total_amplification += individual_amp
        synergy_amplification = 0.0
        for _, result in compatibility_results.items():
            if result.reinforcement_effect > 0:
                synergy_amplification += (
                    result.reinforcement_effect * result.emergence_potential
                )
        if max_individual_amplification > 2.0:
            base_amplification = max_individual_amplification
        else:
            base_amplification = (
                total_amplification / len(individual_scores)
                if individual_scores
                else 1.0
            )
        final_amplification = min(10.0, base_amplification + synergy_amplification)
        logger.info(
            f"    🚀 Amplification Potential: {final_amplification:.3f} (base: {base_amplification:.3f}, synergy: {synergy_amplification:.3f})"
        )
        return final_amplification

    def _calculate_cognitive_efficiency(
        self,
        individual_scores: Dict[str, Dict[str, float]],
        compatibility_results: Dict[str, CompatibilityResult],
        amplification_potential: float,
    ) -> float:
        total_cognitive_load = 0.0
        integration_difficulty = 0.0
        for _, result in compatibility_results.items():
            total_cognitive_load += result.cognitive_load_increase
            integration_difficulty += result.integration_difficulty
        avg_cognitive_load = (
            total_cognitive_load / len(compatibility_results)
            if compatibility_results
            else 0.5
        )
        avg_integration_difficulty = (
            integration_difficulty / len(compatibility_results)
            if compatibility_results
            else 0.5
        )
        total_cognitive_cost = (avg_cognitive_load + avg_integration_difficulty) / 2
        benefit = amplification_potential / 10.0
        if total_cognitive_cost == 0:
            efficiency = benefit
        else:
            efficiency = benefit / (1 + total_cognitive_cost)
        efficiency = max(0.0, min(1.0, efficiency))
        logger.info(
            f"    ⚡ Cognitive Efficiency: {efficiency:.3f} (benefit: {benefit:.3f}, cost: {total_cognitive_cost:.3f})"
        )
        return efficiency

    def _calculate_stability_rating(
        self,
        individual_scores: Dict[str, Dict[str, float]],
        compatibility_results: Dict[str, CompatibilityResult],
    ) -> float:
        individual_stability: List[float] = []
        for nway_id, scores in individual_scores.items():
            nway_type = self._classify_nway_type(nway_id, scores)
            if nway_type == "lollapalooza":
                stability = scores.get("predictability", 0.5)
            elif nway_type == "meta_framework":
                stability = (
                    scores.get("foundational_depth", 0.5)
                    + scores.get("bias_resistance", 0.5)
                ) / 2
            elif nway_type == "cluster":
                stability = (
                    scores.get("professional_authenticity", 0.5)
                    + scores.get("synergy_coherence", 0.5)
                ) / 2
            else:
                stability = scores.get("outcome_predictability", 0.5)
            individual_stability.append(stability)
        avg_individual_stability = (
            sum(individual_stability) / len(individual_stability)
            if individual_stability
            else 0.5
        )
        compatibility_stability = (
            [r.stability for r in compatibility_results.values()]
            if compatibility_results
            else []
        )
        conflict_penalty = (
            sum(0.2 for r in compatibility_results.values() if r.conflict_risk > 0.7)
            if compatibility_results
            else 0.0
        )
        avg_compatibility_stability = (
            sum(compatibility_stability) / len(compatibility_stability)
            if compatibility_stability
            else 0.5
        )
        combined_stability = (
            avg_individual_stability + avg_compatibility_stability
        ) / 2
        final_stability = max(0.0, combined_stability - conflict_penalty)
        logger.info(
            f"    🏔️  Stability Rating: {final_stability:.3f} (individual: {avg_individual_stability:.3f}, compat: {avg_compatibility_stability:.3f})"
        )
        return final_stability

    def _calculate_overall_chemistry_score(
        self,
        reaction_probability: float,
        amplification_potential: float,
        cognitive_efficiency: float,
        stability_rating: float,
    ) -> float:
        normalized_amplification = amplification_potential / 10.0
        components = {
            "reaction_probability": reaction_probability * 0.25,
            "amplification_potential": normalized_amplification * 0.30,
            "cognitive_efficiency": cognitive_efficiency * 0.25,
            "stability_rating": stability_rating * 0.20,
        }
        base_score = sum(components.values())
        stability_bonus = (
            0.1 if stability_rating >= self.stability_requirement else -0.1
        )
        excellence_bonus = 0.0
        if (
            reaction_probability > 0.8
            and normalized_amplification > 0.7
            and cognitive_efficiency > 0.7
            and stability_rating > 0.8
        ):
            excellence_bonus = 0.1
        final_score = max(
            0.0, min(1.0, base_score + stability_bonus + excellence_bonus)
        )
        logger.info(f"    🎯 OVERALL CHEMISTRY SCORE: {final_score:.3f}")
        logger.info(f"       Components: {components}")
        logger.info(
            f"       Bonuses: stability={stability_bonus:.2f}, excellence={excellence_bonus:.2f}"
        )
        return final_score

    def _classify_nway_type(self, nway_id: str, scores: Dict[str, float]) -> str:
        if (
            any(
                term in nway_id.upper()
                for term in ["AUCTION", "TUPPERWARE", "COCACOLA"]
            )
            or scores.get("environmental_match_score", 0.0) > 0.1
        ):
            return "lollapalooza"
        if (
            any(term in nway_id.upper() for term in ["DECISION", "BIAS", "UNCERTAINTY"])
            or scores.get("problem_type_coverage", 0.0) > 0.1
        ):
            return "meta_framework"
        if (
            "CLUSTER" in nway_id.upper()
            or scores.get("domain_specialization", 0.0) > 0.1
        ):
            return "cluster"
        return "contextual"

    def _generate_comprehensive_analysis(
        self,
        problem_framework: str,
        nway_combination: List[Dict[str, Any]],
        individual_scores: Dict[str, Dict[str, float]],
        compatibility_results: Dict[str, CompatibilityResult],
        reaction_probability: float,
        amplification_potential: float,
        cognitive_efficiency: float,
        stability_rating: float,
        overall_chemistry_score: float,
    ) -> Dict[str, Any]:
        reaction_types = [
            result.reaction_type for result in compatibility_results.values()
        ]
        if reaction_types:
            dominant_reaction_type = max(set(reaction_types), key=reaction_types.count)
        else:
            dominant_reaction_type = ReactionType.NEUTRAL
        nway_types = [
            self._classify_nway_type(nway_id, scores)
            for nway_id, scores in individual_scores.items()
        ]
        primary_nway_type = (
            max(set(nway_types), key=nway_types.count) if nway_types else "contextual"
        )
        risk_factors: List[str] = []
        if stability_rating < 0.6:
            risk_factors.append("Low stability - results may be inconsistent")
        if cognitive_efficiency < 0.5:
            risk_factors.append("Poor cognitive efficiency - high mental overhead")
        if any(result.conflict_risk > 0.7 for result in compatibility_results.values()):
            risk_factors.append("High conflict risk between some NWAYs")
        if reaction_probability < 0.4:
            risk_factors.append("Low reaction probability - may not activate properly")
        success_factors: List[str] = []
        if amplification_potential > 7.0:
            success_factors.append("Exceptional amplification potential")
        if stability_rating > 0.8:
            success_factors.append("High stability and consistency")
        if cognitive_efficiency > 0.7:
            success_factors.append("Excellent cognitive efficiency")
        if any(
            result.emergence_potential > 0.8
            for result in compatibility_results.values()
        ):
            success_factors.append("Strong emergence potential for new capabilities")
        recommendation = self._generate_recommendation(
            overall_chemistry_score, risk_factors, success_factors
        )
        confidence_level = self._calculate_confidence_level(
            reaction_probability, stability_rating, len(compatibility_results)
        )
        predicted_effectiveness = min(
            1.0, overall_chemistry_score * amplification_potential / 5.0
        )
        predicted_execution_time = self._predict_execution_time(
            cognitive_efficiency, len(nway_combination)
        )
        cognitive_load_assessment = self._assess_cognitive_load(
            len(nway_combination), compatibility_results
        )
        return {
            "dominant_reaction_type": dominant_reaction_type,
            "primary_nway_type": primary_nway_type,
            "risk_factors": risk_factors,
            "success_factors": success_factors,
            "recommendation": recommendation,
            "confidence_level": confidence_level,
            "predicted_effectiveness": predicted_effectiveness,
            "predicted_execution_time": predicted_execution_time,
            "cognitive_load_assessment": cognitive_load_assessment,
        }

    def _generate_recommendation(
        self, overall_score: float, risk_factors: List[str], success_factors: List[str]
    ) -> str:
        if overall_score >= 0.9:
            return "REVOLUTIONARY - Exceptional cognitive chemistry with breakthrough potential"
        elif overall_score >= 0.8:
            return "EXCELLENT - Highly effective combination with strong benefits"
        elif overall_score >= 0.7:
            return "GOOD - Solid combination with clear advantages"
        elif overall_score >= 0.6:
            return "ACCEPTABLE - Workable combination with moderate benefits"
        elif overall_score >= 0.4:
            return "POOR - Marginal benefits, consider alternatives"
        else:
            return "HARMFUL - Avoid this combination, high risk of negative effects"

    def _calculate_confidence_level(
        self, reaction_probability: float, stability_rating: float, num_pairs: int
    ) -> float:
        base_confidence = (reaction_probability + stability_rating) / 2
        if num_pairs == 0:
            interaction_confidence = 0.3
        elif num_pairs < 3:
            interaction_confidence = 0.6
        else:
            interaction_confidence = 0.9
        final_confidence = (base_confidence + interaction_confidence) / 2
        return min(1.0, final_confidence)

    def _predict_execution_time(
        self, cognitive_efficiency: float, num_nways: int
    ) -> str:
        base_time = num_nways * 10
        if cognitive_efficiency > 0.8:
            adjusted_time = base_time * 0.7
        elif cognitive_efficiency > 0.6:
            adjusted_time = base_time * 0.9
        elif cognitive_efficiency < 0.4:
            adjusted_time = base_time * 1.5
        else:
            adjusted_time = base_time
        if adjusted_time < 20:
            return "15-30 minutes"
        elif adjusted_time < 45:
            return "30-60 minutes"
        elif adjusted_time < 90:
            return "1-2 hours"
        else:
            return "2+ hours"

    def _assess_cognitive_load(
        self, num_nways: int, compatibility_results: Dict[str, CompatibilityResult]
    ) -> str:
        base_load = num_nways / 5.0
        complexity_load = (
            sum(
                result.cognitive_load_increase
                for result in compatibility_results.values()
            )
            if compatibility_results
            else 0.0
        )
        avg_complexity_load = (
            complexity_load / len(compatibility_results)
            if compatibility_results
            else 0.0
        )
        total_load = base_load + avg_complexity_load
        if total_load < 0.5:
            return "Low - Easy to manage mentally"
        elif total_load < 1.0:
            return "Medium - Moderate mental effort required"
        elif total_load < 1.5:
            return "High - Significant mental effort required"
        else:
            return "Very High - Challenging to manage cognitively"

    def _record_consultant_selection_evidence(
        self,
        problem_framework: str,
        selected_combinations: List[str],
        chemistry_scores: Dict[str, Dict[str, float]],
        final_score: float,
        selection_rationale: str,
        risk_factors: List[str],
        success_factors: List[str],
        confidence_level: float,
    ) -> None:
        if not self.context_stream:
            return
        evidence_data = {
            "problem_framework": problem_framework,
            "selection_rationale": selection_rationale,
            "total_confidence": confidence_level,
            "consultant_count": len(selected_combinations),
            "final_chemistry_score": final_score,
            "consultants": [],
        }
        for combo_id in selected_combinations:
            scores = chemistry_scores.get(combo_id, {})
            top_features: List[str] = []
            synergy_score = 0.0
            domain_match_score = 0.0
            for score_name, score_value in scores.items():
                if score_value > 0.7:
                    top_features.append(f"{score_name}={score_value:.2f}")
                if "synergy" in score_name.lower():
                    synergy_score = max(synergy_score, score_value)
                if (
                    "domain" in score_name.lower()
                    or "specialization" in score_name.lower()
                ):
                    domain_match_score = max(domain_match_score, score_value)
            consultant_evidence = {
                "consultant_id": combo_id,
                "consultant_type": self._classify_nway_type(combo_id, scores),
                "version": "v2_chemistry",
                "synergy_score": synergy_score,
                "domain_match_score": domain_match_score,
                "why_selected": success_factors,
                "top_features": top_features,
                "chosen_nway_clusters": [combo_id],
                "cluster_rationale": f"Chemistry score: {final_score:.3f} - {selection_rationale}",
            }
            evidence_data["consultants"].append(consultant_evidence)
        evidence_data["risk_factors"] = risk_factors
        evidence_data["success_factors"] = success_factors
        self.context_stream.add_event(
            event_type=ContextEventType.MODEL_SELECTION_JUSTIFICATION,
            data=evidence_data,
            metadata={
                "evidence_type": "consultant_selection",
                "audit_level": "complete",
                "trace_id": self.context_stream.trace_id,
                "chemistry_engine": "cognitive_chemistry_v1",
            },
        )
        logger.info(
            f"🔍 Evidence recorded: Selected {len(selected_combinations)} consultant combinations with confidence {confidence_level:.2f}"
        )
