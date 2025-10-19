"""
METIS Scoring Engine Service
Part of Selection Services Cluster - Focused on model scoring with detailed component breakdown

Extracted from model_selector.py _score_model_applicability during Phase 5.2 decomposition.
Single Responsibility: Score models for applicability with transparent component scoring.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import defaultdict

from src.services.contracts.selection_contracts import (
    IScoringEngineService,
    ModelScoreContract,
    SelectionContextContract,
    ScoringWeights,
)


class ScoringEngineService(IScoringEngineService):
    """
    Focused service for comprehensive model scoring with component breakdown
    Clean extraction from model_selector.py _score_model_applicability method
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Default scoring weights (can be customized per context)
        self.default_weights = ScoringWeights(
            criteria_matching=0.30,
            performance_history=0.20,
            validation_status=0.15,
            cognitive_load=0.10,
            diversity_bonus=0.10,
            nway_synergy=0.15,
        )

        # Performance cache for recent selections (would be database in production)
        self.performance_cache = {}
        self.selection_history = []

        # Validation status scoring mapping
        self.validation_scores = {
            "production": 1.0,
            "validated": 0.85,
            "experimental": 0.6,
            "deprecated": 0.2,
            "pending": 0.4,
        }

        # Cognitive load compatibility matrix
        self.load_compatibility = {
            "low": {"low": 1.0, "medium": 0.4, "high": 0.1},
            "medium": {"low": 1.0, "medium": 1.0, "high": 0.6},
            "high": {"low": 1.0, "medium": 1.0, "high": 1.0},
        }

        self.logger.info("📊 ScoringEngineService initialized")

    async def score_models(
        self, models: List[Any], context: SelectionContextContract
    ) -> List[ModelScoreContract]:
        """
        Core service method: Score all models for applicability
        Clean, focused implementation with single responsibility
        """
        try:
            scored_models = []

            # Use context-specific weights if available
            weights = self._get_contextual_weights(context)

            for model in models:
                try:
                    # Calculate all component scores
                    component_scores = await self.calculate_component_scores(
                        model, context
                    )

                    # Calculate weighted total score
                    total_score = self._calculate_total_score(component_scores, weights)

                    # Generate scoring rationale
                    rationale = self._generate_scoring_rationale(
                        model, component_scores, total_score
                    )

                    # Calculate confidence in scoring
                    confidence = self._calculate_score_confidence(component_scores)

                    # Identify risk factors
                    risk_factors = self._identify_risk_factors(
                        model, context, component_scores
                    )

                    # Create score contract
                    model_score = ModelScoreContract(
                        model_id=getattr(model, "model_id", str(model)),
                        total_score=total_score,
                        component_scores=component_scores,
                        rationale=rationale,
                        confidence=confidence,
                        risk_factors=risk_factors,
                        scoring_timestamp=datetime.utcnow(),
                        service_version="v5_modular",
                    )

                    scored_models.append(model_score)

                except Exception as e:
                    self.logger.warning(
                        f"⚠️ Failed to score model {getattr(model, 'model_id', 'unknown')}: {e}"
                    )
                    # Create minimal fallback score
                    fallback_score = self._create_fallback_score(model, str(e))
                    scored_models.append(fallback_score)

            self.logger.info(f"✅ Scored {len(scored_models)} models successfully")
            return scored_models

        except Exception as e:
            self.logger.error(f"❌ Model scoring failed: {e}")
            # Return fallback scores for all models
            return [self._create_fallback_score(model, str(e)) for model in models]

    async def calculate_component_scores(
        self, model: Any, context: SelectionContextContract
    ) -> Dict[str, float]:
        """Calculate detailed component scores for transparency"""
        component_scores = {}

        try:
            # 1. Criteria Matching Score
            component_scores["criteria_matching"] = self._score_criteria_matching(
                model, context
            )

            # 2. Performance History Score
            component_scores["performance_history"] = self._score_performance_history(
                model, context
            )

            # 3. Validation Status Score
            component_scores["validation_status"] = self._score_validation_status(model)

            # 4. Cognitive Load Score
            component_scores["cognitive_load"] = self._score_cognitive_load(
                model, context
            )

            # 5. Diversity Bonus Score
            component_scores["diversity_bonus"] = self._score_diversity_bonus(
                model, context
            )

            # 6. N-Way Synergy Score (placeholder - would integrate with N-Way service)
            component_scores["nway_synergy"] = self._score_nway_synergy(model, context)

            return component_scores

        except Exception as e:
            self.logger.error(
                f"❌ Component scoring failed for {getattr(model, 'model_id', 'unknown')}: {e}"
            )
            # Return neutral scores
            return {
                "criteria_matching": 0.5,
                "performance_history": 0.5,
                "validation_status": 0.5,
                "cognitive_load": 0.5,
                "diversity_bonus": 0.5,
                "nway_synergy": 0.5,
            }

    def _score_criteria_matching(
        self, model: Any, context: SelectionContextContract
    ) -> float:
        """Score how well model criteria match the problem context"""
        try:
            # Extract model criteria
            model_criteria = getattr(model, "application_criteria", [])
            if not model_criteria:
                return 0.3  # Low score for models without clear criteria

            # Extract context keywords
            context_keywords = set()

            # Add problem type keywords
            if hasattr(context, "problem_type"):
                context_keywords.update(context.problem_type.replace("_", " ").split())

            # Add complexity level
            context_keywords.add(context.complexity_level)

            # Add business context keywords
            for key, value in context.business_context.items():
                if isinstance(value, str):
                    context_keywords.update(value.lower().split())
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, str):
                            context_keywords.update(item.lower().split())

            # Add problem statement keywords (key terms only)
            problem_words = context.problem_statement.lower().split()
            significant_words = [word for word in problem_words if len(word) > 4]
            context_keywords.update(significant_words)

            # Score against model criteria
            matching_criteria = []
            for criterion in model_criteria:
                criterion_words = set(criterion.lower().split())
                if context_keywords.intersection(criterion_words):
                    matching_criteria.append(criterion)

            # Calculate match ratio
            match_ratio = len(matching_criteria) / len(model_criteria)

            # Bonus for exact problem type match
            problem_type_bonus = 0.0
            if hasattr(context, "problem_type"):
                problem_type = context.problem_type.replace("_", " ")
                if any(
                    problem_type in criterion.lower() for criterion in model_criteria
                ):
                    problem_type_bonus = 0.2

            return min(1.0, match_ratio + problem_type_bonus)

        except Exception as e:
            self.logger.warning(f"⚠️ Criteria matching scoring failed: {e}")
            return 0.5

    def _score_performance_history(
        self, model: Any, context: SelectionContextContract
    ) -> float:
        """Score based on model's performance history"""
        try:
            model_id = getattr(model, "model_id", "unknown")

            # Check performance cache first
            if model_id in self.performance_cache:
                cached_performance = self.performance_cache[model_id]
                return min(1.0, cached_performance)

            # Use model's built-in performance metrics
            performance_metrics = getattr(model, "performance_metrics", {})

            if not performance_metrics:
                return 0.5  # Neutral score for no history

            # Weight key performance indicators
            weighted_score = 0.0
            weights = {
                "accuracy": 0.4,
                "consistency": 0.3,
                "cognitive_load": 0.15,  # Lower is better
                "time_efficiency": 0.15,
            }

            for metric, weight in weights.items():
                if metric in performance_metrics:
                    value = performance_metrics[metric]
                    if metric == "cognitive_load":
                        # Invert cognitive load (lower is better)
                        weighted_score += (1.0 - value) * weight
                    else:
                        weighted_score += value * weight

            return min(1.0, weighted_score)

        except Exception as e:
            self.logger.warning(f"⚠️ Performance history scoring failed: {e}")
            return 0.5

    def _score_validation_status(self, model: Any) -> float:
        """Score based on model validation and maturity"""
        try:
            validation_status = getattr(model, "validation_status", "unknown")
            base_score = self.validation_scores.get(validation_status, 0.5)

            # Bonus for research validation (expected_improvement metric)
            expected_improvement = getattr(model, "expected_improvement", 0)
            if expected_improvement > 0:
                research_bonus = min(0.15, expected_improvement / 100.0)
                base_score += research_bonus

            return min(1.0, base_score)

        except Exception as e:
            self.logger.warning(f"⚠️ Validation status scoring failed: {e}")
            return 0.5

    def _score_cognitive_load(
        self, model: Any, context: SelectionContextContract
    ) -> float:
        """Score cognitive load compatibility (higher is better)"""
        try:
            # Estimate model complexity
            model_complexity = self._estimate_model_complexity(model)

            # Get context cognitive load limit
            load_limit = getattr(context, "cognitive_load_limit", "medium")

            # Score compatibility
            compatibility_score = self.load_compatibility.get(load_limit, {}).get(
                model_complexity, 0.5
            )

            return compatibility_score

        except Exception as e:
            self.logger.warning(f"⚠️ Cognitive load scoring failed: {e}")
            return 0.5

    def _score_diversity_bonus(
        self, model: Any, context: SelectionContextContract
    ) -> float:
        """Score diversity bonus for model category distribution"""
        try:
            # Check if this category is underrepresented in recent selections
            recent_selections = (
                self.selection_history[-10:] if self.selection_history else []
            )
            category_counts = defaultdict(int)

            for selection in recent_selections:
                for selected_model_id in selection.get("selected_model_ids", []):
                    # In production, would query model catalog
                    category_counts["general"] += 1  # Simplified

            # Give bonus to underrepresented categories
            model_category = getattr(model, "category", "general")
            if hasattr(model_category, "value"):
                category_name = model_category.value
            else:
                category_name = str(model_category)

            category_frequency = category_counts.get(category_name, 0)

            if not recent_selections:
                return 0.5  # Neutral when no history

            # Higher bonus for less frequently used categories
            max_frequency = max(category_counts.values()) if category_counts else 1
            diversity_score = 1.0 - (category_frequency / max_frequency)

            return diversity_score

        except Exception as e:
            self.logger.warning(f"⚠️ Diversity bonus scoring failed: {e}")
            return 0.5

    def _score_nway_synergy(
        self, model: Any, context: SelectionContextContract
    ) -> float:
        """Score N-Way synergy potential (placeholder for N-Way service integration)"""
        try:
            # Placeholder implementation - in production would integrate with N-Way service
            model_id = getattr(model, "model_id", "unknown")

            # Simple heuristic based on model type
            if "systems" in model_id.lower() or "strategic" in model_id.lower():
                return 0.8  # High synergy potential for strategic models
            elif "analytical" in model_id.lower() or "framework" in model_id.lower():
                return 0.7  # Medium-high synergy for analytical models
            else:
                return 0.5  # Neutral synergy score

        except Exception as e:
            self.logger.warning(f"⚠️ N-Way synergy scoring failed: {e}")
            return 0.5

    def _estimate_model_complexity(self, model: Any) -> str:
        """Estimate cognitive complexity of applying this model"""
        try:
            # Use criteria count as complexity indicator
            criteria_count = len(getattr(model, "application_criteria", []))

            # Simple heuristic mapping
            if criteria_count <= 2:
                return "low"
            elif criteria_count <= 4:
                return "medium"
            else:
                return "high"

        except Exception as e:
            self.logger.warning(f"⚠️ Model complexity estimation failed: {e}")
            return "medium"

    def _calculate_total_score(
        self, component_scores: Dict[str, float], weights: ScoringWeights
    ) -> float:
        """Calculate weighted total score from components"""
        try:
            total_score = (
                component_scores.get("criteria_matching", 0.5)
                * weights.criteria_matching
                + component_scores.get("performance_history", 0.5)
                * weights.performance_history
                + component_scores.get("validation_status", 0.5)
                * weights.validation_status
                + component_scores.get("cognitive_load", 0.5) * weights.cognitive_load
                + component_scores.get("diversity_bonus", 0.5) * weights.diversity_bonus
                + component_scores.get("nway_synergy", 0.5) * weights.nway_synergy
            )

            return min(1.0, max(0.0, total_score))

        except Exception as e:
            self.logger.warning(f"⚠️ Total score calculation failed: {e}")
            return 0.5

    def _generate_scoring_rationale(
        self, model: Any, component_scores: Dict[str, float], total_score: float
    ) -> str:
        """Generate human-readable rationale for model score"""
        try:
            model_name = getattr(
                model, "name", getattr(model, "model_id", "Unknown Model")
            )

            # Identify strongest and weakest components
            sorted_components = sorted(
                component_scores.items(), key=lambda x: x[1], reverse=True
            )
            strongest = sorted_components[0] if sorted_components else ("unknown", 0.5)
            weakest = sorted_components[-1] if sorted_components else ("unknown", 0.5)

            rationale_parts = [f"Total Score: {total_score:.2f}/1.0 for {model_name}"]

            if strongest[1] > 0.8:
                component_name = strongest[0].replace("_", " ").title()
                rationale_parts.append(f"✓ Strong {component_name}: {strongest[1]:.2f}")

            if weakest[1] < 0.3:
                component_name = weakest[0].replace("_", " ").title()
                rationale_parts.append(f"⚠ Weak {component_name}: {weakest[1]:.2f}")

            # Add validation status note
            validation_status = getattr(model, "validation_status", "unknown")
            if validation_status == "production":
                rationale_parts.append("✓ Production-validated model")
            elif validation_status == "experimental":
                rationale_parts.append("⚠ Experimental model")

            return " | ".join(rationale_parts)

        except Exception as e:
            self.logger.warning(f"⚠️ Rationale generation failed: {e}")
            return f"Score: {total_score:.2f} (rationale generation error)"

    def _calculate_score_confidence(self, component_scores: Dict[str, float]) -> float:
        """Calculate confidence in the scoring based on component consistency"""
        try:
            scores = list(component_scores.values())
            if not scores:
                return 0.5

            # Calculate variance in component scores
            mean_score = sum(scores) / len(scores)
            variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)

            # High variance = low confidence
            confidence = 1.0 - min(variance, 0.5)

            return confidence

        except Exception as e:
            self.logger.warning(f"⚠️ Confidence calculation failed: {e}")
            return 0.5

    def _identify_risk_factors(
        self,
        model: Any,
        context: SelectionContextContract,
        component_scores: Dict[str, float],
    ) -> List[str]:
        """Identify potential risk factors for model selection"""
        risk_factors = []

        try:
            validation_status = getattr(model, "validation_status", "unknown")
            if validation_status in ["experimental", "pending"]:
                risk_factors.append("Unvalidated model")

            if component_scores.get("criteria_matching", 0.5) < 0.3:
                risk_factors.append("Poor criteria match")

            if component_scores.get("performance_history", 0.5) < 0.4:
                risk_factors.append("Limited performance history")

            if component_scores.get("cognitive_load", 0.5) < 0.3:
                risk_factors.append("High cognitive load")

            if (
                context.accuracy_requirement > 0.9
                and getattr(model, "expected_improvement", 0) < 5.0
            ):
                risk_factors.append(
                    "High accuracy requirement, low improvement expectation"
                )

        except Exception as e:
            self.logger.warning(f"⚠️ Risk factor identification failed: {e}")
            risk_factors.append("Risk assessment error")

        return risk_factors

    def _get_contextual_weights(
        self, context: SelectionContextContract
    ) -> ScoringWeights:
        """Get context-specific scoring weights"""
        try:
            # Adjust weights based on context
            weights = ScoringWeights(
                criteria_matching=self.default_weights.criteria_matching,
                performance_history=self.default_weights.performance_history,
                validation_status=self.default_weights.validation_status,
                cognitive_load=self.default_weights.cognitive_load,
                diversity_bonus=self.default_weights.diversity_bonus,
                nway_synergy=self.default_weights.nway_synergy,
            )

            # High accuracy requirement -> increase validation weight
            if context.accuracy_requirement >= 0.9:
                weights.validation_status *= 1.5
                weights.performance_history *= 1.3

            # Low cognitive load limit -> increase cognitive load weight
            if context.cognitive_load_limit == "low":
                weights.cognitive_load *= 2.0

            # Complex problems -> increase criteria matching weight
            if context.complexity_level in ["high", "very_high"]:
                weights.criteria_matching *= 1.3

            # Normalize weights to sum to 1.0
            total_weight = (
                weights.criteria_matching
                + weights.performance_history
                + weights.validation_status
                + weights.cognitive_load
                + weights.diversity_bonus
                + weights.nway_synergy
            )

            if total_weight > 0:
                weights.criteria_matching /= total_weight
                weights.performance_history /= total_weight
                weights.validation_status /= total_weight
                weights.cognitive_load /= total_weight
                weights.diversity_bonus /= total_weight
                weights.nway_synergy /= total_weight

            return weights

        except Exception as e:
            self.logger.warning(f"⚠️ Contextual weight calculation failed: {e}")
            return self.default_weights

    def _create_fallback_score(self, model: Any, error_msg: str) -> ModelScoreContract:
        """Create fallback score when scoring fails"""
        model_id = getattr(model, "model_id", "unknown")

        return ModelScoreContract(
            model_id=model_id,
            total_score=0.3,  # Low fallback score
            component_scores={
                "criteria_matching": 0.3,
                "performance_history": 0.3,
                "validation_status": 0.3,
                "cognitive_load": 0.3,
                "diversity_bonus": 0.3,
                "nway_synergy": 0.3,
                "error": 1.0,
            },
            rationale=f"Scoring failed for {model_id}: {error_msg}",
            confidence=0.1,
            risk_factors=["Scoring service error"],
            scoring_timestamp=datetime.utcnow(),
            service_version="v5_modular_fallback",
        )

    async def update_performance_cache(self, model_id: str, performance_score: float):
        """Update performance cache with recent model performance"""
        try:
            self.performance_cache[model_id] = performance_score
            self.logger.debug(
                f"📊 Updated performance cache for {model_id}: {performance_score:.3f}"
            )
        except Exception as e:
            self.logger.warning(f"⚠️ Performance cache update failed: {e}")

    async def get_service_health(self) -> Dict[str, Any]:
        """Return service health status"""
        return {
            "service_name": "ScoringEngineService",
            "status": "healthy",
            "version": "v5_modular",
            "capabilities": [
                "comprehensive_model_scoring",
                "component_score_breakdown",
                "contextual_weight_adjustment",
                "performance_history_integration",
                "risk_factor_identification",
            ],
            "scoring_components": list(self.default_weights.__dict__.keys()),
            "validation_status_mapping": self.validation_scores,
            "performance_cache_size": len(self.performance_cache),
            "last_health_check": datetime.utcnow().isoformat(),
        }


# Global service instance for dependency injection
_scoring_engine_service: Optional[ScoringEngineService] = None


def get_scoring_engine_service() -> ScoringEngineService:
    """Get or create global scoring engine service instance"""
    global _scoring_engine_service

    if _scoring_engine_service is None:
        _scoring_engine_service = ScoringEngineService()

    return _scoring_engine_service
