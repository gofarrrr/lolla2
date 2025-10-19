# src/services/selection/optimizer.py
import logging
from typing import Any, Dict, List, Optional, Tuple
from itertools import combinations

from src.services.selection.contracts import ChemistryContext
from src.services.selection.scorer import ChemistryScorer
from src.services.selection.cognitive_chemistry_engine import CognitiveChemistryReaction

logger = logging.getLogger(__name__)


class ChemistryOptimizer:
    """Extracted optimization service. Delegates scoring to the injected scorer.
    The optimize(ctx) method routes to the appropriate optimization strategy:
    - If ctx.available_consultants is provided: consultant-aware optimization (returns Dict[str, Any])
    - Otherwise: pattern optimization (returns CognitiveChemistryReaction)
    """

    def __init__(self, scorer: ChemistryScorer) -> None:  # type: ignore[name-defined]
        self.scorer = scorer

    def optimize(self, ctx: ChemistryContext) -> Any:
        if ctx.available_consultants:
            return self._optimize_with_consultant_selection(ctx)
        else:
            return self._optimize_patterns(ctx)

    # ================= Patterns-only optimization =================
    def _optimize_patterns(self, ctx: ChemistryContext) -> CognitiveChemistryReaction:
        problem_framework = ctx.problem_framework
        initial_nway_combination = ctx.nway_combination
        available_nway_patterns = ctx.available_nway_patterns or []
        target_score = ctx.target_score
        max_iterations = ctx.max_iterations

        logger.info(
            f"🔬 CHEMISTRY OPTIMIZATION LOOP: Starting optimization for target score {target_score}"
        )
        logger.info(
            f"   Initial combination: {[n.get('interaction_id', 'unknown') for n in initial_nway_combination]}"
        )

        # Baseline
        baseline_reaction = self.scorer.score(
            ChemistryContext(problem_framework, initial_nway_combination)
        )
        baseline_score = baseline_reaction.overall_chemistry_score
        logger.info(f"   Baseline chemistry score: {baseline_score:.3f}")
        if baseline_score >= target_score:
            logger.info(
                f"✅ OPTIMIZATION NOT NEEDED: Baseline score {baseline_score:.3f} already meets target {target_score}"
            )
            return baseline_reaction

        best_reaction = baseline_reaction
        best_score = baseline_score
        optimization_attempts: List[Tuple[str, float]] = []

        # Strategy 1: Addition
        if available_nway_patterns and len(initial_nway_combination) < 4:
            logger.info("🧪 OPTIMIZATION STRATEGY 1: Pattern Addition")
            addition_reaction = self._optimize_by_pattern_addition(
                problem_framework,
                initial_nway_combination,
                available_nway_patterns,
                target_score,
            )
            optimization_attempts.append(
                ("pattern_addition", addition_reaction.overall_chemistry_score)
            )
            if addition_reaction.overall_chemistry_score > best_score:
                best_reaction = addition_reaction
                best_score = addition_reaction.overall_chemistry_score
                logger.info(
                    f"✨ Pattern addition improved score: {baseline_score:.3f} → {best_score:.3f}"
                )

        # Strategy 2: Substitution
        if available_nway_patterns and len(initial_nway_combination) >= 2:
            logger.info("🔄 OPTIMIZATION STRATEGY 2: Pattern Substitution")
            substitution_reaction = self._optimize_by_pattern_substitution(
                problem_framework,
                initial_nway_combination,
                available_nway_patterns,
                target_score,
            )
            optimization_attempts.append(
                ("pattern_substitution", substitution_reaction.overall_chemistry_score)
            )
            if substitution_reaction.overall_chemistry_score > best_score:
                best_reaction = substitution_reaction
                best_score = substitution_reaction.overall_chemistry_score
                logger.info(f"✨ Pattern substitution improved score: {best_score:.3f}")

        # Strategy 3: Amplification boosting
        logger.info("🚀 OPTIMIZATION STRATEGY 3: Amplification Boosting")
        amplification_reaction = self._optimize_by_amplification_boosting(
            problem_framework,
            initial_nway_combination,
            available_nway_patterns or [],
            target_score,
        )
        optimization_attempts.append(
            ("amplification_boosting", amplification_reaction.overall_chemistry_score)
        )
        if amplification_reaction.overall_chemistry_score > best_score:
            best_reaction = amplification_reaction
            best_score = amplification_reaction.overall_chemistry_score
            logger.info(f"✨ Amplification boosting improved score: {best_score:.3f}")

        # Strategy 4: Iterative refinement
        if best_score < target_score and max_iterations > 1:
            logger.info("🔧 OPTIMIZATION STRATEGY 4: Iterative Refinement")
            for iteration in range(1, max_iterations):
                logger.info(f"   Iteration {iteration}/{max_iterations-1}")
                refined_reaction = self._optimize_by_iterative_refinement(
                    problem_framework,
                    best_reaction.nway_combination,
                    available_nway_patterns or [],
                    target_score,
                )
                optimization_attempts.append(
                    (
                        f"iterative_refinement_{iteration}",
                        refined_reaction.overall_chemistry_score,
                    )
                )
                if refined_reaction.overall_chemistry_score > best_score:
                    improvement = refined_reaction.overall_chemistry_score - best_score
                    best_reaction = refined_reaction
                    best_score = refined_reaction.overall_chemistry_score
                    logger.info(
                        f"✨ Iteration {iteration} improved score by {improvement:.3f}: {best_score:.3f}"
                    )
                    if best_score >= target_score:
                        logger.info(f"🎯 TARGET REACHED at iteration {iteration}!")
                        break
                else:
                    logger.info(
                        f"   Iteration {iteration} did not improve score ({refined_reaction.overall_chemistry_score:.3f})"
                    )

        improvement = best_score - baseline_score
        success_status = (
            "✅ TARGET ACHIEVED"
            if best_score >= target_score
            else "⚠️ TARGET NOT REACHED"
        )
        logger.info(f"🏁 OPTIMIZATION COMPLETE: {success_status}")
        logger.info(
            f"   Baseline: {baseline_score:.3f} → Final: {best_score:.3f} (improvement: +{improvement:.3f})"
        )
        logger.info(f"   Attempts: {len(optimization_attempts)} strategies tested")

        # Attach metadata on the returned reaction for parity
        best_reaction.optimization_metadata = {
            "baseline_score": baseline_score,
            "final_score": best_score,
            "improvement": improvement,
            "target_achieved": best_score >= target_score,
            "optimization_attempts": optimization_attempts,
            "best_strategy": (
                max(optimization_attempts, key=lambda x: x[1])[0]
                if optimization_attempts
                else "baseline"
            ),
        }
        return best_reaction

    def _optimize_by_pattern_addition(
        self,
        problem_framework: str,
        current_combination: List[Dict[str, Any]],
        available_patterns: List[Dict[str, Any]],
        target_score: float,
    ) -> CognitiveChemistryReaction:
        best_combination = current_combination.copy()
        best_score = self.scorer.score(
            ChemistryContext(problem_framework, best_combination)
        ).overall_chemistry_score
        current_pattern_ids = {p.get("interaction_id") for p in current_combination}
        for pattern in available_patterns:
            if (
                pattern.get("interaction_id") not in current_pattern_ids
                and len(best_combination) < 4
            ):
                test_combination = best_combination + [pattern]
                test_reaction = self.scorer.score(
                    ChemistryContext(problem_framework, test_combination)
                )
                if test_reaction.overall_chemistry_score > best_score:
                    best_combination = test_combination
                    best_score = test_reaction.overall_chemistry_score
                    logger.debug(
                        f"   Added {pattern.get('interaction_id')}: score improved to {best_score:.3f}"
                    )
                    if best_score >= target_score:
                        break
        return self.scorer.score(ChemistryContext(problem_framework, best_combination))

    def _optimize_by_pattern_substitution(
        self,
        problem_framework: str,
        current_combination: List[Dict[str, Any]],
        available_patterns: List[Dict[str, Any]],
        target_score: float,
    ) -> CognitiveChemistryReaction:
        best_combination = current_combination.copy()
        best_score = self.scorer.score(
            ChemistryContext(problem_framework, best_combination)
        ).overall_chemistry_score
        current_pattern_ids = {p.get("interaction_id") for p in current_combination}
        for i, current_pattern in enumerate(current_combination):
            for replacement_pattern in available_patterns:
                if replacement_pattern.get("interaction_id") not in current_pattern_ids:
                    test_combination = current_combination.copy()
                    test_combination[i] = replacement_pattern
                    test_reaction = self.scorer.score(
                        ChemistryContext(problem_framework, test_combination)
                    )
                    if test_reaction.overall_chemistry_score > best_score:
                        best_combination = test_combination
                        best_score = test_reaction.overall_chemistry_score
                        logger.debug(
                            f"   Substituted {current_pattern.get('interaction_id')} → {replacement_pattern.get('interaction_id')}: {best_score:.3f}"
                        )
                        if best_score >= target_score:
                            return self.scorer.score(
                                ChemistryContext(problem_framework, best_combination)
                            )
        return self.scorer.score(ChemistryContext(problem_framework, best_combination))

    def _optimize_by_amplification_boosting(
        self,
        problem_framework: str,
        current_combination: List[Dict[str, Any]],
        available_patterns: List[Dict[str, Any]],
        target_score: float,
    ) -> CognitiveChemistryReaction:
        pattern_amplifications: List[Tuple[Dict[str, Any], float]] = []
        for pattern in available_patterns:
            test_reaction = self.scorer.score(
                ChemistryContext(problem_framework, [pattern])
            )
            pattern_amplifications.append(
                (pattern, test_reaction.amplification_potential)
            )
        pattern_amplifications.sort(key=lambda x: x[1], reverse=True)
        optimized_combination: List[Dict[str, Any]] = []
        used_pattern_ids = set()
        for pattern, _ in pattern_amplifications:
            pattern_id = pattern.get("interaction_id")
            if pattern_id not in used_pattern_ids and len(optimized_combination) < 3:
                optimized_combination.append(pattern)
                used_pattern_ids.add(pattern_id)
        for pattern in current_combination:
            pattern_id = pattern.get("interaction_id")
            if pattern_id not in used_pattern_ids and len(optimized_combination) < 3:
                optimized_combination.append(pattern)
                used_pattern_ids.add(pattern_id)
        return self.scorer.score(
            ChemistryContext(problem_framework, optimized_combination)
        )

    def _optimize_by_iterative_refinement(
        self,
        problem_framework: str,
        current_combination_ids: List[str],
        available_patterns: List[Dict[str, Any]],
        target_score: float,
    ) -> CognitiveChemistryReaction:
        current_patterns: List[Dict[str, Any]] = []
        for pattern_id in current_combination_ids:
            for pattern in available_patterns:
                if pattern.get("interaction_id") == pattern_id:
                    current_patterns.append(pattern)
                    break
        if not current_patterns:
            current_patterns = available_patterns[: min(3, len(available_patterns))]
        best_combination = current_patterns
        best_score = self.scorer.score(
            ChemistryContext(problem_framework, best_combination)
        ).overall_chemistry_score
        if len(current_patterns) > 1:
            reordered = [current_patterns[1], current_patterns[0]] + current_patterns[
                2:
            ]
            test_reaction = self.scorer.score(
                ChemistryContext(problem_framework, reordered)
            )
            if test_reaction.overall_chemistry_score > best_score:
                best_combination = reordered
                best_score = test_reaction.overall_chemistry_score
        return self.scorer.score(ChemistryContext(problem_framework, best_combination))

    # ============== Consultant-aware optimization ==============
    def _optimize_with_consultant_selection(
        self, ctx: ChemistryContext
    ) -> Dict[str, Any]:
        problem_framework = ctx.problem_framework
        initial_nway_combination = ctx.nway_combination
        available_consultants = ctx.available_consultants or []
        available_nway_patterns = ctx.available_nway_patterns or []
        target_score = ctx.target_score

        logger.info(
            f"🧬 CONSULTANT-AWARE CHEMISTRY OPTIMIZATION: Target score {target_score}"
        )
        logger.info(f"   Available consultants: {available_consultants}")

        best_score = 0.0
        best_combination: Optional[List[Dict[str, Any]]] = None
        best_consultants: Optional[Tuple[str, str, str]] = None
        optimization_history: List[Dict[str, Any]] = []

        consultant_combinations = list(combinations(available_consultants, 3))
        logger.info(
            f"🔬 Testing {len(consultant_combinations)} consultant combinations with current patterns"
        )
        for consultant_trio in consultant_combinations:
            consultant_chemistry_factor = self._calculate_consultant_chemistry_factor(
                consultant_trio, initial_nway_combination, problem_framework
            )
            base_reaction = self.scorer.score(
                ChemistryContext(problem_framework, initial_nway_combination)
            )
            consultant_adjusted_score = (
                base_reaction.overall_chemistry_score * consultant_chemistry_factor
            )
            optimization_history.append(
                {
                    "strategy": "consultant_selection",
                    "consultants": consultant_trio,
                    "patterns": [
                        p.get("interaction_id") for p in initial_nway_combination
                    ],
                    "base_score": base_reaction.overall_chemistry_score,
                    "consultant_factor": consultant_chemistry_factor,
                    "final_score": consultant_adjusted_score,
                }
            )
            if consultant_adjusted_score > best_score:
                best_score = consultant_adjusted_score
                best_combination = initial_nway_combination
                best_consultants = consultant_trio
                logger.info(
                    f"✨ Consultant trio improved chemistry: {consultant_trio} → {consultant_adjusted_score:.3f}"
                )

        if best_score < target_score and available_nway_patterns and best_consultants:
            logger.info(
                f"🧪 Optimizing NWAY patterns for best consultants: {best_consultants}"
            )
            pattern_optimized_reaction = self._optimize_patterns_for_consultants(
                problem_framework,
                best_consultants,
                available_nway_patterns,
                target_score,
            )
            consultant_factor = self._calculate_consultant_chemistry_factor(
                best_consultants,
                pattern_optimized_reaction.nway_combination,
                problem_framework,
            )
            final_optimized_score = (
                pattern_optimized_reaction.overall_chemistry_score * consultant_factor
            )
            optimization_history.append(
                {
                    "strategy": "consultant_aware_pattern_optimization",
                    "consultants": best_consultants,
                    "patterns": pattern_optimized_reaction.nway_combination,
                    "base_score": pattern_optimized_reaction.overall_chemistry_score,
                    "consultant_factor": consultant_factor,
                    "final_score": final_optimized_score,
                }
            )
            if final_optimized_score > best_score:
                best_score = final_optimized_score
                best_combination = pattern_optimized_reaction.nway_combination
                logger.info(
                    f"✨ Pattern optimization for consultants improved score: {final_optimized_score:.3f}"
                )

        if best_score < target_score and available_nway_patterns:
            logger.info(
                "🔧 Joint consultant-pattern optimization for target {target_score}"
            )
            joint_result = self._joint_consultant_pattern_optimization(
                problem_framework,
                available_consultants,
                available_nway_patterns,
                target_score,
            )
            if joint_result["final_score"] > best_score:
                best_score = joint_result["final_score"]
                best_combination = joint_result["patterns"]
                best_consultants = joint_result["consultants"]
                optimization_history.append(joint_result)
                logger.info(f"✨ Joint optimization achieved: {best_score:.3f}")

        success_status = (
            "✅ TARGET ACHIEVED"
            if best_score >= target_score
            else "⚠️ TARGET NOT REACHED"
        )
        logger.info(f"🏁 CONSULTANT-AWARE OPTIMIZATION COMPLETE: {success_status}")
        logger.info(
            f"   Best score: {best_score:.3f} with consultants {best_consultants}"
        )
        return {
            "final_score": best_score,
            "target_achieved": best_score >= target_score,
            "best_consultants": best_consultants,
            "best_nway_patterns": [
                p.get("interaction_id") if isinstance(p, dict) else p
                for p in (best_combination or [])
            ],
            "optimization_history": optimization_history,
            "strategies_tested": len(optimization_history),
        }

    def _calculate_consultant_chemistry_factor(
        self,
        consultant_trio: Tuple[str, str, str],
        nway_patterns: List,
        problem_framework: str,
    ) -> float:
        base_factor = 1.0
        unique_consultant_types = len(set(consultant_trio))
        if unique_consultant_types == 3:
            diversity_bonus = 0.25
        elif unique_consultant_types == 2:
            diversity_bonus = 0.15
        else:
            diversity_bonus = -0.10
        perspective_breadth = 0.0
        consultant_specializations = set()
        for consultant in consultant_trio:
            if "strategist" in consultant.lower():
                consultant_specializations.add("strategic")
            elif "technical" in consultant.lower() or "architect" in consultant.lower():
                consultant_specializations.add("technical")
            elif "operations" in consultant.lower() or "expert" in consultant.lower():
                consultant_specializations.add("operational")
            else:
                consultant_specializations.add("general")
        specialization_diversity = len(consultant_specializations)
        if specialization_diversity >= 3:
            perspective_breadth = 0.20
        elif specialization_diversity == 2:
            perspective_breadth = 0.10
        complexity_match = 0.0
        if isinstance(nway_patterns, list):
            pattern_count = len(nway_patterns)
            if pattern_count >= 3 and unique_consultant_types >= 2:
                complexity_match = 0.15
            elif pattern_count >= 2 and unique_consultant_types >= 2:
                complexity_match = 0.08
        team_size_factor = 0.12 if len(consultant_trio) == 3 else 0.0
        final_factor = (
            base_factor
            + diversity_bonus
            + perspective_breadth
            + complexity_match
            + team_size_factor
        )
        logger.debug(
            f"   Analytical diversity: types={diversity_bonus:.2f}, breadth={perspective_breadth:.2f}, complexity={complexity_match:.2f}, team={team_size_factor:.2f} → {final_factor:.2f}"
        )
        return min(final_factor, 1.6)

    def _optimize_patterns_for_consultants(
        self,
        problem_framework: str,
        consultant_trio: Tuple[str, str, str],
        available_patterns: List[Dict[str, Any]],
        target_score: float,
    ) -> CognitiveChemistryReaction:
        best_patterns = available_patterns[: min(3, len(available_patterns))]
        best_reaction = self.scorer.score(
            ChemistryContext(problem_framework, best_patterns)
        )
        best_base_score = best_reaction.overall_chemistry_score
        for pattern_count in [2, 3]:
            if len(available_patterns) >= pattern_count:
                for pattern_combo in combinations(available_patterns, pattern_count):
                    test_reaction = self.scorer.score(
                        ChemistryContext(problem_framework, list(pattern_combo))
                    )
                    consultant_factor = self._calculate_consultant_chemistry_factor(
                        consultant_trio, list(pattern_combo), problem_framework
                    )
                    adjusted_score = (
                        test_reaction.overall_chemistry_score * consultant_factor
                    )
                    if (
                        adjusted_score
                        > best_base_score
                        * self._calculate_consultant_chemistry_factor(
                            consultant_trio, best_patterns, problem_framework
                        )
                    ):
                        best_patterns = list(pattern_combo)
                        best_reaction = test_reaction
                        best_base_score = test_reaction.overall_chemistry_score
                        if adjusted_score >= target_score:
                            break
        return self.scorer.score(ChemistryContext(problem_framework, best_patterns))

    def _joint_consultant_pattern_optimization(
        self,
        problem_framework: str,
        available_consultants: List[str],
        available_patterns: List[Dict[str, Any]],
        target_score: float,
    ) -> Dict[str, Any]:
        best_score = 0.0
        best_consultants: Optional[Tuple[str, str, str]] = None
        best_patterns: Optional[List[Dict[str, Any]]] = None
        consultant_combinations = list(combinations(available_consultants, 3))
        max_consultant_combos = min(10, len(consultant_combinations))
        max_pattern_combos = min(5, len(available_patterns))
        for consultant_trio in consultant_combinations[:max_consultant_combos]:
            for pattern_count in [2, 3]:
                if len(available_patterns) >= pattern_count:
                    pattern_combinations = list(
                        combinations(available_patterns, pattern_count)
                    )[:max_pattern_combos]
                    for pattern_combo in pattern_combinations:
                        base_reaction = self.scorer.score(
                            ChemistryContext(problem_framework, list(pattern_combo))
                        )
                        consultant_factor = self._calculate_consultant_chemistry_factor(
                            consultant_trio, list(pattern_combo), problem_framework
                        )
                        joint_score = (
                            base_reaction.overall_chemistry_score * consultant_factor
                        )
                        if joint_score > best_score:
                            best_score = joint_score
                            best_consultants = consultant_trio
                            best_patterns = list(pattern_combo)
                            if joint_score >= target_score:
                                break
        return {
            "strategy": "joint_optimization",
            "consultants": best_consultants,
            "patterns": [p.get("interaction_id") for p in (best_patterns or [])],
            "final_score": best_score,
            "base_score": (
                self.scorer.score(
                    ChemistryContext(problem_framework, best_patterns or [])
                ).overall_chemistry_score
                if best_patterns
                else 0.0
            ),
            "consultant_factor": (
                self._calculate_consultant_chemistry_factor(
                    best_consultants, best_patterns, problem_framework
                )
                if best_consultants and best_patterns
                else 1.0
            ),
        }
