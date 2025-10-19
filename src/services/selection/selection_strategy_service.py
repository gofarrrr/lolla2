"""
METIS Selection Strategy Service
Part of Selection Services Cluster - Focused on executing different model selection strategies

Extracted from model_selector.py _apply_selection_strategy during Phase 5.2 decomposition.
Single Responsibility: Execute various selection strategies with clean, focused algorithms.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from src.services.contracts.selection_contracts import (
    ISelectionStrategyService,
    SelectionResultContract,
    ModelScoreContract,
    SelectionContextContract,
    SelectionStrategy,
)


class SelectionStrategyService(ISelectionStrategyService):
    """
    Focused service for executing different model selection strategies
    Clean extraction from model_selector.py _apply_selection_strategy method
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Strategy execution weights and parameters
        self.strategy_parameters = {
            SelectionStrategy.PERFORMANCE_OPTIMIZED: {
                "score_weight": 1.0,
                "confidence_threshold": 0.7,
                "description": "Select models with highest total scores",
            },
            SelectionStrategy.COGNITIVE_BALANCED: {
                "cognitive_penalty_factor": 0.3,
                "balance_threshold": 0.6,
                "description": "Balance performance with cognitive load",
            },
            SelectionStrategy.DIVERSITY_FOCUSED: {
                "category_diversity_weight": 0.8,
                "min_categories": 2,
                "description": "Maximize model category diversity",
            },
            SelectionStrategy.RISK_CONSERVATIVE: {
                "validation_requirement": ["production", "validated"],
                "min_confidence": 0.7,
                "description": "Prefer validated, low-risk models",
            },
            SelectionStrategy.SPEED_OPTIMIZED: {
                "criteria_weight": 0.8,
                "skip_detailed_analysis": True,
                "description": "Quick selection based on criteria matching",
            },
        }

        self.logger.info("🎯 SelectionStrategyService initialized")

    async def execute_selection_strategy(
        self,
        strategy: SelectionStrategy,
        models: List[Any],
        scores: List[ModelScoreContract],
        context: SelectionContextContract,
    ) -> SelectionResultContract:
        """
        Core service method: Execute specific selection strategy
        Clean, focused implementation with single responsibility
        """
        try:
            start_time = datetime.utcnow()

            # Create model-score pairs for strategy execution
            model_score_pairs = list(zip(models, scores))

            # Execute strategy-specific logic
            if strategy == SelectionStrategy.PERFORMANCE_OPTIMIZED:
                selected_pairs, strategy_metadata = (
                    await self._execute_performance_optimized(
                        model_score_pairs, context
                    )
                )
            elif strategy == SelectionStrategy.COGNITIVE_BALANCED:
                selected_pairs, strategy_metadata = (
                    await self._execute_cognitive_balanced(model_score_pairs, context)
                )
            elif strategy == SelectionStrategy.DIVERSITY_FOCUSED:
                selected_pairs, strategy_metadata = (
                    await self._execute_diversity_focused(model_score_pairs, context)
                )
            elif strategy == SelectionStrategy.RISK_CONSERVATIVE:
                selected_pairs, strategy_metadata = (
                    await self._execute_risk_conservative(model_score_pairs, context)
                )
            elif strategy == SelectionStrategy.SPEED_OPTIMIZED:
                selected_pairs, strategy_metadata = await self._execute_speed_optimized(
                    model_score_pairs, context
                )
            else:
                # Fallback to performance optimized
                selected_pairs, strategy_metadata = (
                    await self._execute_performance_optimized(
                        model_score_pairs, context
                    )
                )

            # Extract results
            selected_models = [pair[0] for pair in selected_pairs]
            selected_scores = [pair[1] for pair in selected_pairs]

            # Calculate timing and metrics
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            cognitive_load = self._assess_cognitive_load(selected_models, context)

            # Create selection result
            return SelectionResultContract(
                engagement_id=context.problem_statement[:50]
                + "_selection",  # Simplified ID
                selected_models=[
                    getattr(model, "model_id", str(model)) for model in selected_models
                ],
                model_scores=selected_scores,
                selection_source="strategy_execution",
                strategy_used=strategy.value,
                models_evaluated=len(models),
                selection_metadata={
                    **strategy_metadata,
                    "strategy_parameters": self.strategy_parameters.get(strategy, {}),
                    "models_considered": len(model_score_pairs),
                    "selection_efficiency": (
                        len(selected_pairs) / len(model_score_pairs)
                        if model_score_pairs
                        else 0.0
                    ),
                },
                total_selection_time_ms=execution_time,
                cognitive_load_assessment=cognitive_load,
                selection_timestamp=datetime.utcnow(),
                service_version="v5_modular",
            )

        except Exception as e:
            self.logger.error(f"❌ Strategy execution failed: {e}")
            return self._create_fallback_result(
                strategy, models, scores, context, str(e)
            )

    async def _execute_performance_optimized(
        self,
        model_score_pairs: List[Tuple[Any, ModelScoreContract]],
        context: SelectionContextContract,
    ) -> Tuple[List[Tuple[Any, ModelScoreContract]], Dict[str, Any]]:
        """Execute performance-optimized selection strategy"""

        # Sort by total score in descending order
        sorted_pairs = sorted(
            model_score_pairs, key=lambda x: x[1].total_score, reverse=True
        )

        # Select top N models
        selected_pairs = sorted_pairs[: context.max_models]

        metadata = {
            "selection_approach": "highest_scores",
            "score_threshold": (
                selected_pairs[-1][1].total_score if selected_pairs else 0.0
            ),
            "performance_focus": True,
            "optimization_target": "total_score",
        }

        return selected_pairs, metadata

    async def _execute_cognitive_balanced(
        self,
        model_score_pairs: List[Tuple[Any, ModelScoreContract]],
        context: SelectionContextContract,
    ) -> Tuple[List[Tuple[Any, ModelScoreContract]], Dict[str, Any]]:
        """Execute cognitive-balanced selection strategy"""

        penalty_factor = self.strategy_parameters[SelectionStrategy.COGNITIVE_BALANCED][
            "cognitive_penalty_factor"
        ]

        # Adjust scores based on cognitive load
        adjusted_pairs = []
        for model, score in model_score_pairs:
            cognitive_load_score = score.component_scores.get("cognitive_load", 0.5)
            cognitive_penalty = (
                1.0 - cognitive_load_score
            )  # Higher load = higher penalty
            adjusted_score = score.total_score * (
                1.0 - cognitive_penalty * penalty_factor
            )

            # Create adjusted score object
            adjusted_score_obj = ModelScoreContract(
                model_id=score.model_id,
                total_score=adjusted_score,
                component_scores={
                    **score.component_scores,
                    "cognitive_adjustment": -cognitive_penalty * penalty_factor,
                },
                rationale=score.rationale
                + f" | Cognitive adjustment: {-cognitive_penalty * penalty_factor:.3f}",
                confidence=score.confidence,
                risk_factors=score.risk_factors,
                scoring_timestamp=score.scoring_timestamp,
                service_version=score.service_version,
            )

            adjusted_pairs.append((model, adjusted_score_obj))

        # Sort by adjusted score
        sorted_pairs = sorted(
            adjusted_pairs, key=lambda x: x[1].total_score, reverse=True
        )
        selected_pairs = sorted_pairs[: context.max_models]

        metadata = {
            "selection_approach": "cognitive_balanced",
            "cognitive_penalty_factor": penalty_factor,
            "balance_optimization": True,
            "adjustment_applied": True,
        }

        return selected_pairs, metadata

    async def _execute_diversity_focused(
        self,
        model_score_pairs: List[Tuple[Any, ModelScoreContract]],
        context: SelectionContextContract,
    ) -> Tuple[List[Tuple[Any, ModelScoreContract]], Dict[str, Any]]:
        """Execute diversity-focused selection strategy"""

        # First pass: one model per category (highest scoring in each category)
        category_groups = {}
        for model, score in model_score_pairs:
            category = getattr(model, "category", "unknown")
            if isinstance(category, str):
                cat_name = category
            else:
                cat_name = getattr(category, "value", str(category))

            if cat_name not in category_groups:
                category_groups[cat_name] = []
            category_groups[cat_name].append((model, score))

        # Select best model from each category
        diverse_pairs = []
        for category, pairs in category_groups.items():
            best_pair = max(pairs, key=lambda x: x[1].total_score)
            diverse_pairs.append(best_pair)

            if len(diverse_pairs) >= context.max_models:
                break

        # Second pass: fill remaining slots with highest scoring models
        remaining_slots = context.max_models - len(diverse_pairs)
        if remaining_slots > 0:
            remaining_pairs = [
                pair for pair in model_score_pairs if pair not in diverse_pairs
            ]
            remaining_pairs.sort(key=lambda x: x[1].total_score, reverse=True)
            diverse_pairs.extend(remaining_pairs[:remaining_slots])

        metadata = {
            "selection_approach": "diversity_maximization",
            "categories_represented": len(category_groups),
            "category_diversity": len(
                set(getattr(pair[0], "category", "unknown") for pair in diverse_pairs)
            ),
            "diversity_optimization": True,
        }

        return diverse_pairs[: context.max_models], metadata

    async def _execute_risk_conservative(
        self,
        model_score_pairs: List[Tuple[Any, ModelScoreContract]],
        context: SelectionContextContract,
    ) -> Tuple[List[Tuple[Any, ModelScoreContract]], Dict[str, Any]]:
        """Execute risk-conservative selection strategy"""

        params = self.strategy_parameters[SelectionStrategy.RISK_CONSERVATIVE]
        required_statuses = params["validation_requirement"]
        min_confidence = params["min_confidence"]

        # Filter for conservative models
        conservative_pairs = []
        for model, score in model_score_pairs:
            model_status = getattr(model, "validation_status", "unknown")
            if model_status in required_statuses and score.confidence >= min_confidence:
                conservative_pairs.append((model, score))

        # If not enough conservative models, add high-confidence models
        if len(conservative_pairs) < context.max_models:
            remaining_pairs = [
                pair for pair in model_score_pairs if pair not in conservative_pairs
            ]
            high_confidence_pairs = [
                pair for pair in remaining_pairs if pair[1].confidence >= min_confidence
            ]
            high_confidence_pairs.sort(key=lambda x: x[1].confidence, reverse=True)

            additional_needed = context.max_models - len(conservative_pairs)
            conservative_pairs.extend(high_confidence_pairs[:additional_needed])

        # Sort by total score and select
        conservative_pairs.sort(key=lambda x: x[1].total_score, reverse=True)
        selected_pairs = conservative_pairs[: context.max_models]

        metadata = {
            "selection_approach": "risk_minimization",
            "validation_requirements": required_statuses,
            "min_confidence_threshold": min_confidence,
            "conservative_models_found": len(
                [
                    p
                    for p in selected_pairs
                    if getattr(p[0], "validation_status", "") in required_statuses
                ]
            ),
            "risk_mitigation": True,
        }

        return selected_pairs, metadata

    async def _execute_speed_optimized(
        self,
        model_score_pairs: List[Tuple[Any, ModelScoreContract]],
        context: SelectionContextContract,
    ) -> Tuple[List[Tuple[Any, ModelScoreContract]], Dict[str, Any]]:
        """Execute speed-optimized selection strategy"""

        # Sort by criteria matching score for quick decision
        sorted_pairs = sorted(
            model_score_pairs,
            key=lambda x: x[1].component_scores.get("criteria_matching", 0.0),
            reverse=True,
        )

        selected_pairs = sorted_pairs[: context.max_models]

        metadata = {
            "selection_approach": "speed_optimization",
            "primary_criteria": "criteria_matching",
            "detailed_analysis_skipped": True,
            "optimization_target": "selection_speed",
        }

        return selected_pairs, metadata

    async def recommend_strategy(
        self, context: SelectionContextContract
    ) -> SelectionStrategy:
        """Recommend optimal selection strategy based on context"""
        try:
            # Strategy recommendation logic based on context

            # High accuracy requirement -> risk conservative
            if context.accuracy_requirement >= 0.9:
                return SelectionStrategy.RISK_CONSERVATIVE

            # Time constraints -> speed optimized
            if context.time_constraint and context.time_constraint in [
                "urgent",
                "immediate",
            ]:
                return SelectionStrategy.SPEED_OPTIMIZED

            # Complex problems -> diversity focused
            if context.complexity_level in ["high", "very_high"]:
                return SelectionStrategy.DIVERSITY_FOCUSED

            # Cognitive load concerns -> cognitive balanced
            if context.cognitive_load_limit == "low":
                return SelectionStrategy.COGNITIVE_BALANCED

            # Default -> performance optimized
            return SelectionStrategy.PERFORMANCE_OPTIMIZED

        except Exception as e:
            self.logger.error(f"❌ Strategy recommendation failed: {e}")
            return SelectionStrategy.PERFORMANCE_OPTIMIZED  # Safe default

    def _assess_cognitive_load(
        self, selected_models: List[Any], context: SelectionContextContract
    ) -> str:
        """Assess cognitive load of selected model combination"""
        if not selected_models:
            return "low"

        # Simple heuristic based on number and complexity
        model_count = len(selected_models)

        # Estimate complexity (simplified)
        estimated_complexity = model_count * 2  # Base complexity factor

        # Add complexity for cognitive load limit
        if context.cognitive_load_limit == "high":
            complexity_threshold = 12
        elif context.cognitive_load_limit == "medium":
            complexity_threshold = 8
        else:  # low
            complexity_threshold = 4

        if estimated_complexity <= complexity_threshold * 0.6:
            return "low"
        elif estimated_complexity <= complexity_threshold:
            return "medium"
        else:
            return "high"

    def _create_fallback_result(
        self,
        strategy: SelectionStrategy,
        models: List[Any],
        scores: List[ModelScoreContract],
        context: SelectionContextContract,
        error_msg: str,
    ) -> SelectionResultContract:
        """Create fallback result when strategy execution fails"""

        # Simple fallback: select first N models
        selected_models = models[: context.max_models]
        selected_scores = scores[: context.max_models]

        return SelectionResultContract(
            engagement_id=context.problem_statement[:50] + "_fallback",
            selected_models=[
                getattr(model, "model_id", str(model)) for model in selected_models
            ],
            model_scores=selected_scores,
            selection_source="fallback",
            strategy_used=f"{strategy.value}_fallback",
            models_evaluated=len(models),
            selection_metadata={
                "fallback_triggered": True,
                "error_message": error_msg,
                "fallback_strategy": "first_n_selection",
            },
            total_selection_time_ms=0.0,
            cognitive_load_assessment="unknown",
            selection_timestamp=datetime.utcnow(),
            service_version="v5_modular_fallback",
        )

    async def get_service_health(self) -> Dict[str, Any]:
        """Return service health status"""
        return {
            "service_name": "SelectionStrategyService",
            "status": "healthy",
            "version": "v5_modular",
            "capabilities": [
                "performance_optimized_selection",
                "cognitive_balanced_selection",
                "diversity_focused_selection",
                "risk_conservative_selection",
                "speed_optimized_selection",
                "strategy_recommendation",
            ],
            "supported_strategies": [strategy.value for strategy in SelectionStrategy],
            "strategy_count": len(SelectionStrategy),
            "last_health_check": datetime.utcnow().isoformat(),
        }


# Global service instance for dependency injection
_selection_strategy_service: Optional[SelectionStrategyService] = None


def get_selection_strategy_service() -> SelectionStrategyService:
    """Get or create global selection strategy service instance"""
    global _selection_strategy_service

    if _selection_strategy_service is None:
        _selection_strategy_service = SelectionStrategyService()

    return _selection_strategy_service
