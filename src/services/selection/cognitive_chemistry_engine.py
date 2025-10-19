#!/usr/bin/env python3
"""
COGNITIVE CHEMISTRY ENGINE
Phase 4 of Operation: Cognitive Particle Accelerator

THE MASTERPIECE ALGORITHM: calculate_cognitive_chemistry_score

This is the revolutionary heart of the entire system - the algorithm that
integrates all four tiers of scoring plus the compatibility matrix to
predict which cognitive chemistry reactions will create the most powerful,
appropriate, and stable problem-solving capabilities.

This transforms METIS from a consultant platform into a true
GENERATIVE INTELLIGENCE ENGINE.
"""

import statistics
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime

# Import glass-box evidence collection
from src.core.unified_context_stream import UnifiedContextStream

# Import our revolutionary scoring engines
try:
    from .cognitive_chemistry_scoring import (
        CognitiveChemistryScoring,
        get_cognitive_chemistry_scoring,
    )
    from .synergistic_compatibility_matrix import (
        SynergisticCompatibilityMatrix,
        CompatibilityResult,
        ReactionType,
        CompatibilityCategory,
        get_synergistic_compatibility_matrix,
        analyze_nway_combination_compatibility,
    )
except ImportError:
    # Fallback for direct execution
    from synergistic_compatibility_matrix import (
        CompatibilityResult,
        ReactionType,
    )

logger = logging.getLogger(__name__)

# ======================================================================
# COGNITIVE CHEMISTRY REACTION RESULTS
# ======================================================================


@dataclass
class CognitiveChemistryReaction:
    """Complete analysis of a cognitive chemistry reaction"""

    nway_combination: List[str]
    problem_framework: str

    # Individual NWAY scores (from 4-tier scoring)
    individual_scores: Dict[str, Dict[str, float]]

    # Compatibility matrix results
    compatibility_results: Dict[str, CompatibilityResult]

    # Final integrated scores
    reaction_probability: float  # 0.0 - 1.0: Will this reaction occur?
    amplification_potential: float  # 0.0 - 10.0: How powerful will it be?
    cognitive_efficiency: float  # 0.0 - 1.0: Benefit vs cognitive cost
    stability_rating: float  # 0.0 - 1.0: Consistent results?
    overall_chemistry_score: float  # 0.0 - 1.0: Master score

    # Analysis metadata
    dominant_reaction_type: ReactionType
    primary_nway_type: str  # "lollapalooza", "meta_framework", "cluster", "toolkit"
    risk_factors: List[str]
    success_factors: List[str]
    recommendation: str
    confidence_level: float

    # Performance predictions
    predicted_effectiveness: float
    predicted_execution_time: str
    cognitive_load_assessment: str

    created_at: datetime


class ReactionQuality(Enum):
    """Quality levels for cognitive chemistry reactions"""

    REVOLUTIONARY = "revolutionary"  # Breakthrough cognitive capability
    EXCELLENT = "excellent"  # Highly effective combination
    GOOD = "good"  # Solid, reliable combination
    ACCEPTABLE = "acceptable"  # Workable but not optimal
    POOR = "poor"  # Marginal benefit
    HARMFUL = "harmful"  # Net negative effect


# ======================================================================
# THE COGNITIVE CHEMISTRY ENGINE
# ======================================================================


class _LegacyCognitiveChemistryEngine:
    """
    LEGACY: Original engine that predicted and scored cognitive chemistry reactions.
    DECOMMISSIONED: All methods are deprecated. Use the new ChemistryScorer/Optimizer/Analytics services instead.
    """

    def __init__(self, context_stream: Optional[UnifiedContextStream] = None):
        self.context_stream = context_stream
        logger.warning(
            "_LegacyCognitiveChemistryEngine instantiated; all methods are deprecated stubs."
        )

    def calculate_cognitive_chemistry_score(
        self, problem_framework: str, nway_combination: List[Dict[str, Any]]
    ) -> CognitiveChemistryReaction:
        raise NotImplementedError(
            "This method is deprecated. Use the new ChemistryScorer/Optimizer/Analytics services instead."
        )

    def optimize_chemistry_score(
        self,
        problem_framework: str,
        initial_nway_combination: List[Dict[str, Any]],
        available_nway_patterns: List[Dict[str, Any]] = None,
        target_score: float = 0.75,
        max_iterations: int = 5,
    ) -> CognitiveChemistryReaction:
        raise NotImplementedError(
            "This method is deprecated. Use the new ChemistryScorer/Optimizer/Analytics services instead."
        )

    def _optimize_by_pattern_addition(
        self,
        problem_framework: str,
        current_combination: List[Dict[str, Any]],
        available_patterns: List[Dict[str, Any]],
        target_score: float,
    ) -> CognitiveChemistryReaction:
        raise NotImplementedError(
            "This method is deprecated. Use the new ChemistryScorer/Optimizer/Analytics services instead."
        )

    def _optimize_by_pattern_substitution(
        self,
        problem_framework: str,
        current_combination: List[Dict[str, Any]],
        available_patterns: List[Dict[str, Any]],
        target_score: float,
    ) -> CognitiveChemistryReaction:
        raise NotImplementedError(
            "This method is deprecated. Use the new ChemistryScorer/Optimizer/Analytics services instead."
        )

    def optimize_chemistry_with_consultant_selection(
        self,
        problem_framework: str,
        initial_nway_combination: List[Dict[str, Any]],
        available_consultants: List[str],
        available_nway_patterns: List[Dict[str, Any]] = None,
        target_score: float = 0.75,
    ) -> Dict[str, Any]:
        raise NotImplementedError(
            "This method is deprecated. Use the new ChemistryScorer/Optimizer/Analytics services instead."
        )

    def _calculate_consultant_chemistry_factor(
        self, consultant_trio: tuple, nway_patterns: List, problem_framework: str
    ) -> float:
        raise NotImplementedError(
            "This method is deprecated. Use the new ChemistryScorer/Optimizer/Analytics services instead."
        )

    def _optimize_patterns_for_consultants(
        self,
        problem_framework: str,
        consultant_trio: tuple,
        available_patterns: List[Dict[str, Any]],
        target_score: float,
    ) -> CognitiveChemistryReaction:
        raise NotImplementedError(
            "This method is deprecated. Use the new ChemistryScorer/Optimizer/Analytics services instead."
        )

    def _joint_consultant_pattern_optimization(
        self,
        problem_framework: str,
        available_consultants: List[str],
        available_patterns: List[Dict[str, Any]],
        target_score: float,
    ) -> Dict[str, Any]:
        raise NotImplementedError(
            "This method is deprecated. Use the new ChemistryScorer/Optimizer/Analytics services instead."
        )

    def _optimize_by_amplification_boosting(
        self,
        problem_framework: str,
        current_combination: List[Dict[str, Any]],
        available_patterns: List[Dict[str, Any]],
        target_score: float,
    ) -> CognitiveChemistryReaction:
        raise NotImplementedError(
            "This method is deprecated. Use the new ChemistryScorer/Optimizer/Analytics services instead."
        )

    def _optimize_by_iterative_refinement(
        self,
        problem_framework: str,
        current_combination: List[str],
        available_patterns: List[Dict[str, Any]],
        target_score: float,
    ) -> CognitiveChemistryReaction:
        raise NotImplementedError(
            "This method is deprecated. Use the new ChemistryScorer/Optimizer/Analytics services instead."
        )

    def _score_individual_nways(
        self, problem_framework: str, nway_combination: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        raise NotImplementedError(
            "This method is deprecated. Use the new ChemistryScorer/Optimizer/Analytics services instead."
        )

    def _analyze_combination_compatibility(
        self, nway_combination: List[Dict[str, Any]]
    ) -> Dict[str, CompatibilityResult]:
        raise NotImplementedError(
            "This method is deprecated. Use the new ChemistryScorer/Optimizer/Analytics services instead."
        )

    def _calculate_reaction_probability(
        self,
        individual_scores: Dict[str, Dict[str, float]],
        compatibility_results: Dict[str, CompatibilityResult],
        problem_framework: str,
    ) -> float:
        raise NotImplementedError(
            "This method is deprecated. Use the new ChemistryScorer/Optimizer/Analytics services instead."
        )

    def _calculate_compatibility_modifier(
        self, compatibility_results: Dict[str, CompatibilityResult]
    ) -> float:
        raise NotImplementedError(
            "This method is deprecated. Use the new ChemistryScorer/Optimizer/Analytics services instead."
        )

    def _calculate_amplification_potential(
        self,
        individual_scores: Dict[str, Dict[str, float]],
        compatibility_results: Dict[str, CompatibilityResult],
    ) -> float:
        raise NotImplementedError(
            "This method is deprecated. Use the new ChemistryScorer/Optimizer/Analytics services instead."
        )

    def _calculate_cognitive_efficiency(
        self,
        individual_scores: Dict[str, Dict[str, float]],
        compatibility_results: Dict[str, CompatibilityResult],
        amplification_potential: float,
    ) -> float:
        raise NotImplementedError(
            "This method is deprecated. Use the new ChemistryScorer/Optimizer/Analytics services instead."
        )

    def _calculate_stability_rating(
        self,
        individual_scores: Dict[str, Dict[str, float]],
        compatibility_results: Dict[str, CompatibilityResult],
    ) -> float:
        raise NotImplementedError(
            "This method is deprecated. Use the new ChemistryScorer/Optimizer/Analytics services instead."
        )

    def _calculate_overall_chemistry_score(
        self,
        reaction_probability: float,
        amplification_potential: float,
        cognitive_efficiency: float,
        stability_rating: float,
    ) -> float:
        raise NotImplementedError(
            "This method is deprecated. Use the new ChemistryScorer/Optimizer/Analytics services instead."
        )

    def _classify_nway_type(self, nway_id: str, scores: Dict[str, float]) -> str:
        raise NotImplementedError(
            "This method is deprecated. Use the new ChemistryScorer/Optimizer/Analytics services instead."
        )

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
        raise NotImplementedError(
            "This method is deprecated. Use the new ChemistryScorer/Optimizer/Analytics services instead."
        )

    def _generate_recommendation(
        self, overall_score: float, risk_factors: List[str], success_factors: List[str]
    ) -> str:
        raise NotImplementedError(
            "This method is deprecated. Use the new ChemistryScorer/Optimizer/Analytics services instead."
        )

    def _calculate_confidence_level(
        self, reaction_probability: float, stability_rating: float, num_pairs: int
    ) -> float:
        raise NotImplementedError(
            "This method is deprecated. Use the new ChemistryScorer/Optimizer/Analytics services instead."
        )

    def _predict_execution_time(
        self, cognitive_efficiency: float, num_nways: int
    ) -> str:
        raise NotImplementedError(
            "This method is deprecated. Use the new ChemistryScorer/Optimizer/Analytics services instead."
        )

    def _assess_cognitive_load(
        self, num_nways: int, compatibility_results: Dict[str, CompatibilityResult]
    ) -> str:
        raise NotImplementedError(
            "This method is deprecated. Use the new ChemistryScorer/Optimizer/Analytics services instead."
        )

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
        raise NotImplementedError(
            "This method is deprecated. Use the new ChemistryScorer/Optimizer/Analytics services instead."
        )

    def record_selection_evidence(
        self,
        problem_framework: str,
        selected_combinations: List[Dict[str, Any]],
        final_score: float,
        selection_rationale: str,
        risk_factors: List[str],
        success_factors: List[str],
        confidence_level: float,
    ) -> None:
        raise NotImplementedError(
            "This method is deprecated. Use the new ChemistryScorer/Optimizer/Analytics services instead."
        )

    # Learning and analytics era methods (all deprecated)
    def initialize_chemistry_learning_system(self):
        raise NotImplementedError(
            "This method is deprecated. Use the new ChemistryScorer/Optimizer/Analytics services instead."
        )

    async def record_chemistry_performance_feedback(
        self,
        problem_framework: str,
        nway_combination: List[Dict[str, Any]],
        consultant_team: List[str],
        chemistry_score: float,
        actual_performance: float,
        user_satisfaction: Optional[float] = None,
        analysis_quality: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        raise NotImplementedError(
            "This method is deprecated. Use the new ChemistryScorer/Optimizer/Analytics services instead."
        )

    async def get_adaptive_chemistry_score(
        self,
        problem_framework: str,
        nway_combination: List[Dict[str, Any]],
        consultant_team: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> CognitiveChemistryReaction:
        raise NotImplementedError(
            "This method is deprecated. Use the new ChemistryScorer/Optimizer/Analytics services instead."
        )

    async def optimize_chemistry_weights(
        self,
        optimization_target: str = "actual_performance",
        learning_window_days: int = 30,
    ) -> Dict[str, Any]:
        raise NotImplementedError(
            "This method is deprecated. Use the new ChemistryScorer/Optimizer/Analytics services instead."
        )

    def get_chemistry_learning_analytics(self) -> Dict[str, Any]:
        raise NotImplementedError(
            "This method is deprecated. Use the new ChemistryScorer/Optimizer/Analytics services instead."
        )

    async def _apply_learning_enhancements(
        self,
        base_reaction: CognitiveChemistryReaction,
        problem_framework: str,
        nway_combination: List[Dict[str, Any]],
        consultant_team: Optional[List[str]],
        context: Optional[Dict[str, Any]],
    ) -> CognitiveChemistryReaction:
        raise NotImplementedError(
            "This method is deprecated. Use the new ChemistryScorer/Optimizer/Analytics services instead."
        )

    async def _update_consultant_chemistry_patterns(
        self, reaction_record: Dict[str, Any]
    ):
        raise NotImplementedError(
            "This method is deprecated. Use the new ChemistryScorer/Optimizer/Analytics services instead."
        )

    async def _update_pattern_synergy_learning(self, reaction_record: Dict[str, Any]):
        raise NotImplementedError(
            "This method is deprecated. Use the new ChemistryScorer/Optimizer/Analytics services instead."
        )

    async def _trigger_chemistry_learning_if_needed(self):
        raise NotImplementedError(
            "This method is deprecated. Use the new ChemistryScorer/Optimizer/Analytics services instead."
        )

    def _create_combination_signature(
        self, nway_combination: List[Dict[str, Any]], consultant_team: List[str]
    ) -> str:
        raise NotImplementedError(
            "This method is deprecated. Use the new ChemistryScorer/Optimizer/Analytics services instead."
        )

    def _categorize_context_for_chemistry(self, context: Dict[str, Any]) -> str:
        raise NotImplementedError(
            "This method is deprecated. Use the new ChemistryScorer/Optimizer/Analytics services instead."
        )

    def _extract_domain_from_framework(self, framework: str) -> str:
        raise NotImplementedError(
            "This method is deprecated. Use the new ChemistryScorer/Optimizer/Analytics services instead."
        )

    async def _calculate_consultant_chemistry_adjustment(
        self,
        consultant_team: List[str],
        framework: str,
        context: Optional[Dict[str, Any]],
    ) -> float:
        raise NotImplementedError(
            "This method is deprecated. Use the new ChemistryScorer/Optimizer/Analytics services instead."
        )

    async def _calculate_pattern_synergy_adjustment(
        self, nway_combination: List[Dict[str, Any]], framework: str
    ) -> float:
        raise NotImplementedError(
            "This method is deprecated. Use the new ChemistryScorer/Optimizer/Analytics services instead."
        )

    async def _collect_chemistry_learning_data(self, days: int) -> List[Dict[str, Any]]:
        raise NotImplementedError(
            "This method is deprecated. Use the new ChemistryScorer/Optimizer/Analytics services instead."
        )

    def _calculate_current_chemistry_performance(
        self, learning_data: List[Dict[str, Any]]
    ) -> float:
        raise NotImplementedError(
            "This method is deprecated. Use the new ChemistryScorer/Optimizer/Analytics services instead."
        )

    async def _optimize_chemistry_weights_gradient_descent(
        self, learning_data: List[Dict[str, Any]], target: str
    ) -> Dict[str, float]:
        raise NotImplementedError(
            "This method is deprecated. Use the new ChemistryScorer/Optimizer/Analytics services instead."
        )

    async def _estimate_chemistry_improvement(
        self, learning_data: List[Dict[str, Any]], new_weights: Dict[str, float]
    ) -> float:
        raise NotImplementedError(
            "This method is deprecated. Use the new ChemistryScorer/Optimizer/Analytics services instead."
        )

    def _calculate_correlation(
        self, x_values: List[float], y_values: List[float]
    ) -> float:
        raise NotImplementedError(
            "This method is deprecated. Use the new ChemistryScorer/Optimizer/Analytics services instead."
        )

    def _calculate_trend(self, values: List[float]) -> str:
        raise NotImplementedError(
            "This method is deprecated. Use the new ChemistryScorer/Optimizer/Analytics services instead."
        )

        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values))
        sum_sq_x = sum((x - mean_x) ** 2 for x in x_values)
        sum_sq_y = sum((y - mean_y) ** 2 for y in y_values)

        denominator = (sum_sq_x * sum_sq_y) ** 0.5

        return numerator / denominator if denominator != 0 else 0.0

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from list of values"""
        if len(values) < 3:
            return "insufficient_data"

        first_half = values[: len(values) // 2]
        second_half = values[len(values) // 2 :]

        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)

        if second_avg > first_avg * 1.05:
            return "improving"
        elif second_avg < first_avg * 0.95:
            return "declining"
        else:
            return "stable"


# ======================================================================
# FACADE + FACTORY FUNCTION
# ======================================================================
from src.services.selection.contracts import ChemistryContext


class CognitiveChemistryEngine:
    """Single point of entry for the Cognitive Chemistry system.

    Responsibilities:
    - Accepts ChemistryContext and delegates to injected services:
      - ChemistryScorer (scoring)
      - ChemistryOptimizer (search/optimization)
      - ChemistryAnalytics (evidence, learning, feedback)
    - Provides backward-compatible legacy call signatures for transitional callers.

    IMPORTANT:
    - Legacy classes have been removed. This facade is the single entry point.
    - All new code should use this facade with ChemistryContext.

    Migration notes:
    - If you previously called calculate_cognitive_chemistry_score(problem_framework, nway_combination),
      you may keep that signature temporarily; the facade will adapt it by building a ChemistryContext.
    - For full capabilities, prefer constructing ChemistryContext and calling the typed methods.
    """

    def __init__(self, context_stream: Optional[UnifiedContextStream] = None, scorer: "IChemistryScorer" = None, optimizer: "IChemistryOptimizer" = None, analytics: "IChemistryAnalytics" = None):  # type: ignore[name-defined]
        # Wire scorer/optimizer/analytics from DI if not provided
        if scorer is None or optimizer is None or analytics is None:
            try:
                from src.services.container import global_container  # type: ignore

                if scorer is None:
                    scorer = (
                        global_container.get_chemistry_scorer()
                        if global_container
                        else None
                    )
                if optimizer is None:
                    optimizer = (
                        global_container.get_chemistry_optimizer()
                        if global_container
                        else None
                    )
                if analytics is None:
                    analytics = (
                        global_container.get_chemistry_analytics()
                        if global_container
                        else None
                    )
            except Exception:
                pass
        self.scorer = scorer
        self.optimizer = optimizer
        self.analytics = analytics
        self._legacy_engine = _LegacyCognitiveChemistryEngine(
            context_stream=context_stream
        )

    # New preferred API
    def calculate_cognitive_chemistry_score(
        self, ctx: ChemistryContext = None, **kwargs
    ):
        from src.services.selection.contracts import ChemistryContext as _ChemCtx
        from src.services.selection.scorer import ChemistryScorer

        if isinstance(ctx, _ChemCtx) and self.scorer is not None:
            return self.scorer.score(ctx)
        # Legacy signature support
        if (
            ctx is None
            and "problem_framework" in kwargs
            and "nway_combination" in kwargs
        ):
            built_ctx = _ChemCtx(
                problem_framework=kwargs["problem_framework"],
                nway_combination=kwargs["nway_combination"],
            )
            scorer = self.scorer or ChemistryScorer(
                context_stream=self._legacy_engine.context_stream
            )
            return scorer.score(built_ctx)
        # Fallback to legacy engine with whatever was passed (should be rare)
        return self._legacy_engine.calculate_cognitive_chemistry_score(ctx, **kwargs)

    def optimize_chemistry_score(self, ctx: ChemistryContext = None, **kwargs):
        # Legacy signature support
        if (
            ctx is None
            and "problem_framework" in kwargs
            and "initial_nway_combination" in kwargs
        ):
            if self.optimizer is not None:
                built_ctx = ChemistryContext(
                    problem_framework=kwargs["problem_framework"],
                    nway_combination=kwargs["initial_nway_combination"],
                    available_nway_patterns=kwargs.get("available_nway_patterns"),
                    target_score=kwargs.get("target_score", 0.75),
                    max_iterations=kwargs.get("max_iterations", 5),
                )
                return self.optimizer.optimize(built_ctx)
            return self._legacy_engine.optimize_chemistry_score(
                kwargs["problem_framework"],
                kwargs["initial_nway_combination"],
                kwargs.get("available_nway_patterns"),
                kwargs.get("target_score", 0.75),
                kwargs.get("max_iterations", 5),
            )
        # Preferred ChemistryContext path
        if isinstance(ctx, ChemistryContext):
            if self.optimizer is not None:
                built_ctx = ChemistryContext(
                    problem_framework=ctx.problem_framework,
                    nway_combination=(
                        kwargs.get("initial_nway_combination") or ctx.nway_combination
                    ),
                    available_nway_patterns=ctx.available_nway_patterns,
                    target_score=ctx.target_score,
                    max_iterations=ctx.max_iterations,
                )
                return self.optimizer.optimize(built_ctx)
            return self._legacy_engine.optimize_chemistry_score(
                ctx.problem_framework,
                kwargs.get("initial_nway_combination") or ctx.nway_combination,
                ctx.available_nway_patterns,
                ctx.target_score,
                ctx.max_iterations,
            )
        # Fallback
        return self._legacy_engine.optimize_chemistry_score(ctx, **kwargs)

    def optimize_chemistry_with_consultant_selection(
        self, ctx: ChemistryContext = None, **kwargs
    ):
        # Legacy signature support
        if (
            ctx is None
            and "problem_framework" in kwargs
            and "initial_nway_combination" in kwargs
            and "available_consultants" in kwargs
        ):
            if self.optimizer is not None:
                built_ctx = ChemistryContext(
                    problem_framework=kwargs["problem_framework"],
                    nway_combination=kwargs["initial_nway_combination"],
                    available_consultants=kwargs.get("available_consultants"),
                    available_nway_patterns=kwargs.get("available_nway_patterns"),
                    target_score=kwargs.get("target_score", 0.75),
                )
                return self.optimizer.optimize(built_ctx)
            return self._legacy_engine.optimize_chemistry_with_consultant_selection(
                kwargs["problem_framework"],
                kwargs["initial_nway_combination"],
                kwargs["available_consultants"],
                kwargs.get("available_nway_patterns"),
                kwargs.get("target_score", 0.75),
            )
        # Preferred ChemistryContext path
        if isinstance(ctx, ChemistryContext):
            if self.optimizer is not None:
                built_ctx = ChemistryContext(
                    problem_framework=ctx.problem_framework,
                    nway_combination=(
                        kwargs.get("initial_nway_combination") or ctx.nway_combination
                    ),
                    available_consultants=ctx.available_consultants or [],
                    available_nway_patterns=ctx.available_nway_patterns,
                    target_score=ctx.target_score,
                )
                return self.optimizer.optimize(built_ctx)
            return self._legacy_engine.optimize_chemistry_with_consultant_selection(
                ctx.problem_framework,
                kwargs.get("initial_nway_combination") or ctx.nway_combination,
                ctx.available_consultants or [],
                ctx.available_nway_patterns,
                ctx.target_score,
            )
        return self._legacy_engine.optimize_chemistry_with_consultant_selection(
            ctx, **kwargs
        )

    # Analytics and learning delegation (new)
    async def record_chemistry_performance_feedback(
        self,
        problem_framework: str,
        nway_combination: List[Dict[str, Any]],
        consultant_team: List[str],
        chemistry_score: float,
        actual_performance: float,
        user_satisfaction: Optional[float] = None,
        analysis_quality: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if self.analytics is not None:
            ctx = ChemistryContext(
                problem_framework=problem_framework,
                nway_combination=nway_combination,
                context=context or {},
            )
            return await self.analytics.record_chemistry_performance_feedback(
                ctx,
                consultant_team,
                chemistry_score,
                actual_performance,
                user_satisfaction,
                analysis_quality,
            )
        return False

    async def get_adaptive_chemistry_score(
        self,
        problem_framework: str,
        nway_combination: List[Dict[str, Any]],
        consultant_team: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        if self.analytics is not None:
            ctx = ChemistryContext(
                problem_framework=problem_framework,
                nway_combination=nway_combination,
                context=context or {},
            )
            return await self.analytics.get_adaptive_chemistry_score(
                ctx, consultant_team
            )
        # Fallback: compute base score from scorer
        if self.scorer is not None:
            from src.services.selection.contracts import ChemistryContext as _ChemCtx

            return self.scorer.score(
                _ChemCtx(
                    problem_framework=problem_framework,
                    nway_combination=nway_combination,
                )
            )
        return None

    async def optimize_chemistry_weights(
        self,
        optimization_target: str = "actual_performance",
        learning_window_days: int = 30,
    ) -> Dict[str, Any]:
        if self.analytics is not None:
            return await self.analytics.optimize_chemistry_weights(
                optimization_target, learning_window_days
            )
        return {
            "optimization_performed": False,
            "error": "ChemistryAnalytics not available",
        }

    def get_chemistry_learning_analytics(self) -> Dict[str, Any]:
        if self.analytics is not None:
            return self.analytics.get_chemistry_learning_analytics()
        return {"error": "ChemistryAnalytics not available"}

    # Keep public evidence API for compatibility
    def record_selection_evidence(self, *args, **kwargs):
        # Delegate to analytics if available, otherwise fallback
        if self.analytics is not None:
            try:
                return self.analytics.record_selection_evidence(*args, **kwargs)
            except TypeError:
                # In case args/kwargs mismatch, pass through
                return self.analytics.record_selection_evidence(*args, **kwargs)
        return self._legacy_engine.record_selection_evidence(*args, **kwargs)


def get_cognitive_chemistry_engine(
    context_stream: Optional[UnifiedContextStream] = None,
) -> CognitiveChemistryEngine:
    """Get the Cognitive Chemistry Engine facade instance"""
    from src.services.container import global_container

    scorer = None
    optimizer = None
    analytics = None
    try:
        scorer = global_container.get_chemistry_scorer() if global_container else None
        optimizer = (
            global_container.get_chemistry_optimizer() if global_container else None
        )
        analytics = (
            global_container.get_chemistry_analytics() if global_container else None
        )
    except Exception:
        pass
    return CognitiveChemistryEngine(
        context_stream=context_stream,
        scorer=scorer,
        optimizer=optimizer,
        analytics=analytics,
    )


if __name__ == "__main__":
    print("🧬 COGNITIVE CHEMISTRY ENGINE - Phase 4")
    print("   THE MASTERPIECE ALGORITHM")
    print("   calculate_cognitive_chemistry_score - Heart of the Particle Accelerator")

    # Test the complete engine
    engine = get_cognitive_chemistry_engine()

    # Test with a powerful combination
    test_problem = "We need to develop a comprehensive strategy for analyzing market disruptions while avoiding cognitive biases in our decision-making process"

    test_combination = [
        {
            "interaction_id": "NWAY_STRATEGIST_CLUSTER_009",
            "models_involved": [
                "systems-thinking",
                "second-order-thinking",
                "outside-view",
                "scenario-analysis",
            ],
            "cognitive_domain": "strategic",
            "professional_archetype": "strategist",
        },
        {
            "interaction_id": "NWAY_BIAS_MITIGATION_019",
            "models_involved": [
                "cognitive-biases",
                "intellectual-humility",
                "critical-thinking",
                "outside-view",
            ],
            "pattern_type": "meta_framework",
        },
    ]

    reaction = engine.calculate_cognitive_chemistry_score(
        test_problem, test_combination
    )

    print("\n🎯 COGNITIVE CHEMISTRY REACTION RESULTS:")
    print(f"   Overall Score: {reaction.overall_chemistry_score:.3f}")
    print(f"   Reaction Probability: {reaction.reaction_probability:.3f}")
    print(f"   Amplification: {reaction.amplification_potential:.2f}x")
    print(f"   Efficiency: {reaction.cognitive_efficiency:.3f}")
    print(f"   Stability: {reaction.stability_rating:.3f}")
    print(f"   Recommendation: {reaction.recommendation}")
    print(f"   Confidence: {reaction.confidence_level:.3f}")
    print(f"   Execution Time: {reaction.predicted_execution_time}")
    print(f"   Cognitive Load: {reaction.cognitive_load_assessment}")
