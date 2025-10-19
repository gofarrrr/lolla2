"""
METIS Exploration Strategy Service
Part of Reliability Services Cluster - Focused on exploration vs exploitation balance

Extracted from vulnerability_solutions.py CognitiveExplorationEngine during Phase 5 decomposition.
Single Responsibility: Manage exploration vs exploitation decisions to prevent local optima.
"""

import logging
import math
import random
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import defaultdict

from src.services.contracts.reliability_contracts import (
    ExplorationDecisionContract,
    IExplorationStrategyService,
    ExplorationStrategy,
)


class ExplorationStrategyService(IExplorationStrategyService):
    """
    Focused service for managing exploration vs exploitation balance
    Clean extraction from vulnerability_solutions.py CognitiveExplorationEngine
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.base_exploration_rate = 0.15  # 15% of engagements explore
        self.diversity_index_threshold = 0.6

        # Mutation strategies with weights
        self.mutation_strategies = {
            ExplorationStrategy.NOVEL_MODEL_INJECTION: 0.05,
            ExplorationStrategy.HYBRID_SYNTHESIS: 0.07,
            ExplorationStrategy.CROSS_INDUSTRY_TRANSFER: 0.03,
        }

        # Track model usage for diversity maintenance
        self.model_usage_history = defaultdict(int)
        self.total_engagements = 0

        self.logger.info("🔍 ExplorationStrategyService initialized")

    async def determine_exploration_strategy(
        self, problem_analysis: Dict[str, Any], business_context: Dict[str, Any]
    ) -> ExplorationDecisionContract:
        """
        Core service method: Determine exploration strategy for the engagement
        Clean, focused implementation with single responsibility
        """
        try:
            engagement_context = {"business_context": business_context}
            model_performance = problem_analysis.get("model_performance", {})

            # Make exploration decision using focused algorithm
            exploration_decision = self._should_explore(
                engagement_context, model_performance
            )

            # Convert to service contract
            return ExplorationDecisionContract(
                engagement_id=problem_analysis.get("engagement_id", "unknown"),
                should_explore=exploration_decision.should_explore,
                exploration_strategy=(
                    exploration_decision.exploration_strategy.value
                    if exploration_decision.exploration_strategy
                    else None
                ),
                exploration_rationale=exploration_decision.exploration_rationale,
                exploration_models=exploration_decision.exploration_models,
                confidence_adjustment=exploration_decision.confidence_adjustment,
                decision_timestamp=datetime.utcnow(),
                service_version="v5_modular",
            )

        except Exception as e:
            self.logger.error(f"❌ Exploration strategy determination failed: {e}")
            # Return safe fallback decision
            return self._create_fallback_decision(problem_analysis, str(e))

    def _should_explore(
        self, engagement_context: Dict[str, Any], model_performance: Dict[str, float]
    ) -> "ExplorationDecision":
        """Core exploration decision algorithm - clean and focused"""

        # Base exploration probability
        exploration_prob = self.base_exploration_rate

        # Adjust based on context factors
        business_context = engagement_context.get("business_context", {})

        # Higher exploration for novel industries
        industry = business_context.get("industry", "general")
        industry_novelty = self._calculate_industry_novelty(industry)
        if industry_novelty > 0.8:
            exploration_prob += 0.10  # +10% for novel industries

        # Higher exploration when top models have low confidence
        max_model_effectiveness = (
            max(model_performance.values()) if model_performance else 0.5
        )
        if max_model_effectiveness < 0.75:
            exploration_prob += 0.15  # +15% when models are uncertain

        # Higher exploration for strategic importance
        if engagement_context.get("strategic_importance") == "board_level":
            exploration_prob += 0.10  # +10% for high stakes

        # Diversity maintenance check
        diversity_index = self._calculate_model_diversity()
        if diversity_index < self.diversity_index_threshold:
            exploration_prob += 0.20  # +20% when diversity is low

        # Make exploration decision
        should_explore = random.random() < exploration_prob

        if should_explore:
            strategy = self._select_exploration_strategy(engagement_context)
            models = self._select_exploration_models(strategy, engagement_context)
            rationale = self._generate_exploration_rationale(strategy, exploration_prob)

            return ExplorationDecision(
                should_explore=True,
                exploration_strategy=strategy,
                exploration_rationale=rationale,
                exploration_models=models,
                confidence_adjustment=-0.1,  # Slight confidence penalty for exploration
            )
        else:
            return ExplorationDecision(
                should_explore=False,
                exploration_strategy=None,
                exploration_rationale="Exploitation favored based on context and model performance",
                exploration_models=[],
                confidence_adjustment=0.0,
            )

    def _calculate_industry_novelty(self, industry: str) -> float:
        """Calculate how novel this industry is for our model experience"""
        # Simple heuristic - would be replaced with actual historical data
        known_industries = [
            "manufacturing",
            "technology",
            "financial_services",
            "healthcare",
            "retail",
        ]
        return 0.2 if industry in known_industries else 0.9

    def _calculate_model_diversity(self) -> float:
        """Calculate diversity index of recent model selections"""
        if self.total_engagements == 0:
            return 1.0

        # Calculate entropy of model usage
        usage_probs = []
        for count in self.model_usage_history.values():
            prob = count / self.total_engagements
            usage_probs.append(prob)

        # Shannon entropy as diversity measure
        entropy = -sum(p * math.log2(p) for p in usage_probs if p > 0)
        max_entropy = (
            math.log2(len(self.model_usage_history))
            if len(self.model_usage_history) > 0
            else 1.0
        )

        return entropy / max_entropy if max_entropy > 0 else 1.0

    def _select_exploration_strategy(
        self, context: Dict[str, Any]
    ) -> ExplorationStrategy:
        """Select appropriate exploration strategy based on context"""
        strategies = list(self.mutation_strategies.keys())
        weights = list(self.mutation_strategies.values())

        # Adjust weights based on context
        if context.get("complexity_level") == "high":
            # Favor hybrid synthesis for complex problems
            try:
                idx = strategies.index(ExplorationStrategy.HYBRID_SYNTHESIS)
                weights[idx] *= 2.0
            except ValueError:
                pass

        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
            return random.choices(strategies, weights=weights)[0]
        else:
            return ExplorationStrategy.NOVEL_MODEL_INJECTION

    def _select_exploration_models(
        self, strategy: ExplorationStrategy, context: Dict[str, Any]
    ) -> List[str]:
        """Select models for exploration based on strategy"""
        if strategy == ExplorationStrategy.NOVEL_MODEL_INJECTION:
            # Select rarely used models
            return [
                "blue_ocean_strategy",
                "disruptive_innovation_theory",
                "platform_strategy_framework",
            ]

        elif strategy == ExplorationStrategy.HYBRID_SYNTHESIS:
            # Combine models from different categories
            return [
                "porter_5_forces",
                "lean_startup_methodology",
                "systems_thinking_framework",
            ]

        elif strategy == ExplorationStrategy.CROSS_INDUSTRY_TRANSFER:
            # Apply models from different industries
            return [
                "saas_growth_framework",
                "manufacturing_excellence",
                "healthcare_transformation",
            ]

        else:
            return ["experimental_strategic_framework"]

    def _generate_exploration_rationale(
        self, strategy: ExplorationStrategy, prob: float
    ) -> str:
        """Generate human-readable rationale for exploration decision"""
        base_rationale = f"Exploration triggered (probability: {prob:.2f}) using {strategy.value} strategy"

        if strategy == ExplorationStrategy.NOVEL_MODEL_INJECTION:
            return f"{base_rationale}. Testing less-common models to discover potential superior approaches."
        elif strategy == ExplorationStrategy.HYBRID_SYNTHESIS:
            return f"{base_rationale}. Combining models from different frameworks to find novel insights."
        elif strategy == ExplorationStrategy.CROSS_INDUSTRY_TRANSFER:
            return (
                f"{base_rationale}. Applying successful patterns from other industries."
            )
        else:
            return base_rationale

    async def record_exploration_outcome(
        self, exploration_decision: ExplorationDecisionContract, outcome: Dict[str, Any]
    ) -> None:
        """Record exploration outcome for learning and future strategy adjustment"""
        try:
            self.total_engagements += 1

            # Record model usage
            for model in exploration_decision.exploration_models:
                self.model_usage_history[model] += 1

            # Log outcome for learning (simplified version)
            outcome_quality = outcome.get("quality_score", 0.5)
            if outcome_quality > 0.8:
                self.logger.info(
                    f"✅ Excellent exploration outcome: {exploration_decision.exploration_strategy} ({outcome_quality:.2f})"
                )
            elif outcome_quality < 0.3:
                self.logger.warning(
                    f"⚠️ Poor exploration outcome: {exploration_decision.exploration_strategy} ({outcome_quality:.2f})"
                )

            # TODO: In full implementation, update strategy weights based on outcomes

        except Exception as e:
            self.logger.error(f"❌ Failed to record exploration outcome: {e}")

    def _create_fallback_decision(
        self, problem_analysis: Dict, error_msg: str
    ) -> ExplorationDecisionContract:
        """Create safe fallback decision when service fails"""
        return ExplorationDecisionContract(
            engagement_id=problem_analysis.get("engagement_id", "unknown"),
            should_explore=False,  # Safe default: no exploration
            exploration_strategy=None,
            exploration_rationale=f"Service error, defaulting to exploitation: {error_msg}",
            exploration_models=[],
            confidence_adjustment=0.0,
            decision_timestamp=datetime.utcnow(),
            service_version="v5_modular_fallback",
        )

    async def get_service_health(self) -> Dict[str, Any]:
        """Return service health status"""
        return {
            "service_name": "ExplorationStrategyService",
            "status": "healthy",
            "version": "v5_modular",
            "capabilities": [
                "exploration_decision_making",
                "strategy_selection",
                "model_diversity_tracking",
                "outcome_learning",
            ],
            "metrics": {
                "total_engagements": self.total_engagements,
                "model_diversity_index": self._calculate_model_diversity(),
                "base_exploration_rate": self.base_exploration_rate,
            },
            "last_health_check": datetime.utcnow().isoformat(),
        }


# ============================================================
# UTILITY CLASSES
# ============================================================


class ExplorationDecision:
    """Decision about whether to explore vs exploit in model selection"""

    def __init__(
        self,
        should_explore: bool,
        exploration_strategy: Optional[ExplorationStrategy],
        exploration_rationale: str,
        exploration_models: List[str],
        confidence_adjustment: float = 0.0,
    ):
        self.should_explore = should_explore
        self.exploration_strategy = exploration_strategy
        self.exploration_rationale = exploration_rationale
        self.exploration_models = exploration_models
        self.confidence_adjustment = confidence_adjustment


# Global service instance for dependency injection
_exploration_strategy_service: Optional[ExplorationStrategyService] = None


def get_exploration_strategy_service() -> ExplorationStrategyService:
    """Get or create global exploration strategy service instance"""
    global _exploration_strategy_service

    if _exploration_strategy_service is None:
        _exploration_strategy_service = ExplorationStrategyService()

    return _exploration_strategy_service
