"""
METIS Bayesian Learning Service
Part of Selection Services Cluster - Focused on context-specific model effectiveness learning

Extracted from model_selector.py during Phase 5.2 decomposition.
Single Responsibility: Learn and update Bayesian model effectiveness for specific contexts.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

from src.services.contracts.selection_contracts import (
    IBayesianLearningService,
    BayesianUpdateContract,
    SelectionContextContract,
)


class BayesianLearningService(IBayesianLearningService):
    """
    Focused service for Bayesian model effectiveness learning
    Clean extraction from model_selector.py Bayesian learning methods
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Bayesian learning parameters
        self.default_alpha = 1.0  # Prior successes
        self.default_beta = 1.0  # Prior failures
        self.min_observations = 3  # Minimum observations for reliable estimates
        self.context_specificity_threshold = 0.7
        self.learning_rate = 0.1
        self.decay_factor = 0.95  # Weekly decay for temporal relevance

        # In-memory effectiveness cache (production would use database)
        self.effectiveness_cache = defaultdict(dict)
        self.observation_history = defaultdict(list)

        # Context similarity thresholds
        self.context_similarity_weights = {
            "problem_type": 0.3,
            "complexity_level": 0.25,
            "business_context": 0.2,
            "accuracy_requirement": 0.15,
            "time_constraint": 0.1,
        }

        self.logger.info("🧠 BayesianLearningService initialized")

    async def get_learned_effectiveness(
        self, model_id: str, context: SelectionContextContract
    ) -> Optional[BayesianUpdateContract]:
        """
        Core service method: Get learned model effectiveness for specific context
        Context-aware Bayesian effectiveness retrieval
        """
        try:
            # Generate context key for similarity matching
            context_key = self._generate_context_key(context)

            # Find most similar contexts with effectiveness data
            similar_contexts = await self._find_similar_contexts(model_id, context)

            if not similar_contexts:
                self.logger.info(
                    f"📊 No learned effectiveness for {model_id} in context {context_key}"
                )
                return None

            # Aggregate effectiveness from similar contexts
            aggregated_effectiveness = await self._aggregate_context_effectiveness(
                model_id, similar_contexts, context
            )

            if aggregated_effectiveness is None:
                return None

            # Apply temporal decay
            temporal_effectiveness = self._apply_temporal_decay(
                aggregated_effectiveness
            )

            return BayesianUpdateContract(
                model_id=model_id,
                context_key=context_key,
                effectiveness_score=temporal_effectiveness["effectiveness_score"],
                observations_count=temporal_effectiveness["observations_count"],
                posterior_alpha=temporal_effectiveness["posterior_alpha"],
                posterior_beta=temporal_effectiveness["posterior_beta"],
                context_specificity=temporal_effectiveness["context_specificity"],
                update_timestamp=datetime.utcnow(),
                service_version="v5_modular",
            )

        except Exception as e:
            self.logger.error(
                f"❌ Failed to get learned effectiveness for {model_id}: {e}"
            )
            return None

    async def update_model_effectiveness(
        self,
        model_id: str,
        effectiveness_score: float,
        context: SelectionContextContract,
    ) -> BayesianUpdateContract:
        """
        Core service method: Update Bayesian effectiveness with new observation
        Context-specific Bayesian learning with temporal weighting
        """
        try:
            context_key = self._generate_context_key(context)

            # Get current effectiveness data
            current_data = self.effectiveness_cache.get(model_id, {}).get(
                context_key,
                {
                    "alpha": self.default_alpha,
                    "beta": self.default_beta,
                    "observations": [],
                    "last_update": None,
                    "context_specificity": {},
                },
            )

            # Create observation record
            observation = {
                "effectiveness_score": effectiveness_score,
                "timestamp": datetime.utcnow(),
                "context": context.__dict__,
                "success": effectiveness_score >= 0.7,  # Binary success threshold
            }

            # Update Bayesian parameters
            if observation["success"]:
                new_alpha = current_data["alpha"] + self.learning_rate
                new_beta = current_data["beta"]
            else:
                new_alpha = current_data["alpha"]
                new_beta = current_data["beta"] + self.learning_rate

            # Add observation to history
            current_data["observations"].append(observation)

            # Keep only recent observations (sliding window)
            cutoff_date = datetime.utcnow() - timedelta(days=30)
            current_data["observations"] = [
                obs
                for obs in current_data["observations"]
                if obs["timestamp"] > cutoff_date
            ]

            # Calculate context specificity
            context_specificity = await self._calculate_context_specificity(
                model_id, context, current_data["observations"]
            )

            # Update effectiveness cache
            updated_data = {
                "alpha": new_alpha,
                "beta": new_beta,
                "observations": current_data["observations"],
                "last_update": datetime.utcnow(),
                "context_specificity": context_specificity,
            }

            self.effectiveness_cache[model_id][context_key] = updated_data

            # Calculate posterior effectiveness
            posterior_mean = new_alpha / (new_alpha + new_beta)

            # Create Bayesian update contract
            update_contract = BayesianUpdateContract(
                model_id=model_id,
                context_key=context_key,
                effectiveness_score=posterior_mean,
                observations_count=len(current_data["observations"]),
                posterior_alpha=new_alpha,
                posterior_beta=new_beta,
                context_specificity=context_specificity,
                update_timestamp=datetime.utcnow(),
                service_version="v5_modular",
            )

            self.logger.info(
                f"🧠 Updated Bayesian effectiveness for {model_id}: {posterior_mean:.3f}"
            )
            return update_contract

        except Exception as e:
            self.logger.error(f"❌ Failed to update effectiveness for {model_id}: {e}")

            # Return fallback contract
            return BayesianUpdateContract(
                model_id=model_id,
                context_key=self._generate_context_key(context),
                effectiveness_score=0.5,  # Neutral default
                observations_count=0,
                posterior_alpha=self.default_alpha,
                posterior_beta=self.default_beta,
                context_specificity={"fallback": True},
                update_timestamp=datetime.utcnow(),
                service_version="v5_modular_fallback",
            )

    async def get_model_learning_summary(self, model_id: str) -> Dict[str, Any]:
        """Get comprehensive learning summary for a model across all contexts"""
        try:
            model_data = self.effectiveness_cache.get(model_id, {})

            if not model_data:
                return {
                    "model_id": model_id,
                    "total_contexts": 0,
                    "total_observations": 0,
                    "average_effectiveness": 0.5,
                    "learning_status": "no_data",
                }

            # Aggregate statistics across contexts
            total_observations = sum(
                len(context_data["observations"])
                for context_data in model_data.values()
            )

            effectiveness_scores = []
            context_summaries = []

            for context_key, context_data in model_data.items():
                if context_data["observations"]:
                    posterior_mean = context_data["alpha"] / (
                        context_data["alpha"] + context_data["beta"]
                    )
                    effectiveness_scores.append(posterior_mean)

                    context_summaries.append(
                        {
                            "context_key": context_key,
                            "effectiveness": posterior_mean,
                            "observations": len(context_data["observations"]),
                            "last_update": (
                                context_data["last_update"].isoformat()
                                if context_data["last_update"]
                                else None
                            ),
                            "confidence": self._calculate_confidence(
                                context_data["alpha"], context_data["beta"]
                            ),
                        }
                    )

            average_effectiveness = (
                sum(effectiveness_scores) / len(effectiveness_scores)
                if effectiveness_scores
                else 0.5
            )

            return {
                "model_id": model_id,
                "total_contexts": len(model_data),
                "total_observations": total_observations,
                "average_effectiveness": average_effectiveness,
                "context_summaries": context_summaries,
                "learning_status": self._determine_learning_status(
                    total_observations, average_effectiveness
                ),
                "summary_timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"❌ Failed to get learning summary for {model_id}: {e}")
            return {"error": str(e)}

    async def prune_stale_observations(
        self, days_threshold: int = 90
    ) -> Dict[str, Any]:
        """Remove stale observations to maintain cache efficiency"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_threshold)
            pruned_count = 0

            for model_id, contexts in list(self.effectiveness_cache.items()):
                for context_key, context_data in list(contexts.items()):
                    original_count = len(context_data["observations"])
                    context_data["observations"] = [
                        obs
                        for obs in context_data["observations"]
                        if obs["timestamp"] > cutoff_date
                    ]
                    pruned_count += original_count - len(context_data["observations"])

                    # Remove empty contexts
                    if not context_data["observations"]:
                        del contexts[context_key]

                # Remove empty models
                if not contexts:
                    del self.effectiveness_cache[model_id]

            self.logger.info(f"🧹 Pruned {pruned_count} stale observations")

            return {
                "pruned_observations": pruned_count,
                "cutoff_date": cutoff_date.isoformat(),
                "remaining_models": len(self.effectiveness_cache),
            }

        except Exception as e:
            self.logger.error(f"❌ Failed to prune observations: {e}")
            return {"error": str(e)}

    def _generate_context_key(self, context: SelectionContextContract) -> str:
        """Generate consistent context key for caching"""
        key_components = [
            context.problem_type,
            context.complexity_level,
            f"acc_{context.accuracy_requirement:.1f}",
            f"max_{context.max_models}",
        ]

        if context.time_constraint:
            key_components.append(f"time_{context.time_constraint}")

        return "_".join(key_components)

    async def _find_similar_contexts(
        self, model_id: str, target_context: SelectionContextContract
    ) -> List[Tuple[str, float]]:
        """Find contexts similar to target context with similarity scores"""
        model_data = self.effectiveness_cache.get(model_id, {})
        similar_contexts = []

        for context_key, context_data in model_data.items():
            if not context_data["observations"]:
                continue

            # Calculate similarity using latest observation context
            latest_obs = max(context_data["observations"], key=lambda x: x["timestamp"])
            stored_context = latest_obs["context"]

            similarity = self._calculate_context_similarity(
                target_context.__dict__, stored_context
            )

            if similarity >= self.context_specificity_threshold:
                similar_contexts.append((context_key, similarity))

        # Sort by similarity (highest first)
        similar_contexts.sort(key=lambda x: x[1], reverse=True)
        return similar_contexts[:5]  # Top 5 similar contexts

    def _calculate_context_similarity(self, context1: Dict, context2: Dict) -> float:
        """Calculate similarity between two contexts"""
        total_similarity = 0.0

        for feature, weight in self.context_similarity_weights.items():
            if feature in context1 and feature in context2:
                if feature == "accuracy_requirement":
                    # Numerical similarity
                    diff = abs(context1[feature] - context2[feature])
                    similarity = max(0, 1 - diff)
                elif feature == "business_context":
                    # Business context similarity (simplified)
                    similarity = 0.8  # Default high similarity
                else:
                    # Categorical similarity
                    similarity = 1.0 if context1[feature] == context2[feature] else 0.0

                total_similarity += similarity * weight

        return total_similarity

    async def _aggregate_context_effectiveness(
        self,
        model_id: str,
        similar_contexts: List[Tuple[str, float]],
        target_context: SelectionContextContract,
    ) -> Optional[Dict[str, Any]]:
        """Aggregate effectiveness from similar contexts"""
        if not similar_contexts:
            return None

        model_data = self.effectiveness_cache[model_id]
        weighted_alpha = 0.0
        weighted_beta = 0.0
        total_weight = 0.0
        total_observations = 0
        context_specificity = {}

        for context_key, similarity in similar_contexts:
            context_data = model_data[context_key]
            weight = similarity  # Use similarity as weight

            weighted_alpha += context_data["alpha"] * weight
            weighted_beta += context_data["beta"] * weight
            total_weight += weight
            total_observations += len(context_data["observations"])

            # Aggregate context specificity
            for key, value in context_data["context_specificity"].items():
                if key not in context_specificity:
                    context_specificity[key] = []
                context_specificity[key].append(value)

        if total_weight == 0:
            return None

        # Normalize by total weight
        final_alpha = weighted_alpha / total_weight
        final_beta = weighted_beta / total_weight
        effectiveness_score = final_alpha / (final_alpha + final_beta)

        # Average context specificity values
        for key, values in context_specificity.items():
            if isinstance(values[0], (int, float)):
                context_specificity[key] = sum(values) / len(values)
            else:
                context_specificity[key] = values[0]  # Take first for non-numeric

        return {
            "effectiveness_score": effectiveness_score,
            "posterior_alpha": final_alpha,
            "posterior_beta": final_beta,
            "observations_count": total_observations,
            "context_specificity": context_specificity,
            "similarity_weighted": True,
        }

    def _apply_temporal_decay(
        self, effectiveness_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply temporal decay to effectiveness data"""
        # Simple temporal decay (could be more sophisticated)
        decay_applied = effectiveness_data.copy()

        # Reduce confidence over time (simplified)
        alpha_decay = effectiveness_data["posterior_alpha"] * self.decay_factor
        beta_decay = effectiveness_data["posterior_beta"] * self.decay_factor

        # Recalculate effectiveness with temporal decay
        new_effectiveness = alpha_decay / (alpha_decay + beta_decay)

        decay_applied.update(
            {
                "effectiveness_score": new_effectiveness,
                "posterior_alpha": alpha_decay,
                "posterior_beta": beta_decay,
                "temporal_decay_applied": True,
            }
        )

        return decay_applied

    async def _calculate_context_specificity(
        self,
        model_id: str,
        context: SelectionContextContract,
        observations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Calculate how context-specific the model's effectiveness is"""
        try:
            if len(observations) < self.min_observations:
                return {
                    "insufficient_data": True,
                    "observations_needed": self.min_observations,
                }

            # Calculate variance in effectiveness across observations
            effectiveness_scores = [obs["effectiveness_score"] for obs in observations]
            mean_effectiveness = sum(effectiveness_scores) / len(effectiveness_scores)
            variance = sum(
                (score - mean_effectiveness) ** 2 for score in effectiveness_scores
            ) / len(effectiveness_scores)

            # Calculate context specificity metrics
            specificity = {
                "effectiveness_variance": variance,
                "observation_count": len(observations),
                "effectiveness_stability": 1.0
                - min(variance, 1.0),  # Higher stability = lower variance
                "context_consistency": self._calculate_context_consistency(
                    observations
                ),
                "temporal_trend": self._calculate_temporal_trend(observations),
            }

            return specificity

        except Exception as e:
            self.logger.error(f"❌ Failed to calculate context specificity: {e}")
            return {"error": str(e)}

    def _calculate_context_consistency(
        self, observations: List[Dict[str, Any]]
    ) -> float:
        """Calculate how consistent the context features are across observations"""
        if len(observations) < 2:
            return 1.0

        # Check consistency of key context features
        consistent_features = 0
        total_features = 0

        reference_context = observations[0]["context"]
        for feature in ["problem_type", "complexity_level"]:
            if feature in reference_context:
                total_features += 1
                feature_values = [obs["context"].get(feature) for obs in observations]
                unique_values = len(set(feature_values))

                if unique_values == 1:  # All observations have same value
                    consistent_features += 1

        return consistent_features / total_features if total_features > 0 else 1.0

    def _calculate_temporal_trend(self, observations: List[Dict[str, Any]]) -> str:
        """Calculate temporal trend in effectiveness"""
        if len(observations) < 3:
            return "insufficient_data"

        # Sort by timestamp
        sorted_obs = sorted(observations, key=lambda x: x["timestamp"])

        # Calculate trend using simple linear regression approach
        effectiveness_scores = [obs["effectiveness_score"] for obs in sorted_obs]
        n = len(effectiveness_scores)

        # Simple trend detection
        recent_half = effectiveness_scores[n // 2 :]
        early_half = effectiveness_scores[: n // 2]

        recent_avg = sum(recent_half) / len(recent_half)
        early_avg = sum(early_half) / len(early_half)

        diff = recent_avg - early_avg

        if abs(diff) < 0.05:
            return "stable"
        elif diff > 0:
            return "improving"
        else:
            return "declining"

    def _calculate_confidence(self, alpha: float, beta: float) -> float:
        """Calculate confidence in effectiveness estimate using Beta distribution"""
        # Confidence based on total observations and concentration
        total_observations = alpha + beta - 2  # Subtract prior
        concentration = alpha + beta

        # Higher concentration and more observations = higher confidence
        base_confidence = min(total_observations / 20.0, 1.0)  # Max at 20 observations
        concentration_bonus = min(
            concentration / 10.0, 0.2
        )  # Small bonus for concentration

        return min(base_confidence + concentration_bonus, 1.0)

    def _determine_learning_status(
        self, observations: int, effectiveness: float
    ) -> str:
        """Determine learning status based on observations and effectiveness"""
        if observations == 0:
            return "no_data"
        elif observations < self.min_observations:
            return "insufficient_data"
        elif observations < 10:
            return "early_learning"
        elif effectiveness >= 0.8:
            return "high_performer"
        elif effectiveness >= 0.6:
            return "moderate_performer"
        elif effectiveness >= 0.4:
            return "low_performer"
        else:
            return "poor_performer"

    async def get_service_health(self) -> Dict[str, Any]:
        """Return service health status"""
        total_models = len(self.effectiveness_cache)
        total_contexts = sum(
            len(contexts) for contexts in self.effectiveness_cache.values()
        )
        total_observations = sum(
            len(context_data["observations"])
            for contexts in self.effectiveness_cache.values()
            for context_data in contexts.values()
        )

        return {
            "service_name": "BayesianLearningService",
            "status": "healthy",
            "version": "v5_modular",
            "capabilities": [
                "context_specific_learning",
                "bayesian_effectiveness_tracking",
                "temporal_decay_handling",
                "context_similarity_matching",
                "observation_pruning",
            ],
            "cache_statistics": {
                "models_tracked": total_models,
                "contexts_tracked": total_contexts,
                "total_observations": total_observations,
                "memory_usage": "in_memory_cache",
            },
            "learning_parameters": {
                "default_alpha": self.default_alpha,
                "default_beta": self.default_beta,
                "min_observations": self.min_observations,
                "decay_factor": self.decay_factor,
            },
            "last_health_check": datetime.utcnow().isoformat(),
        }


# Global service instance for dependency injection
_bayesian_learning_service: Optional[BayesianLearningService] = None


def get_bayesian_learning_service() -> BayesianLearningService:
    """Get or create global Bayesian learning service instance"""
    global _bayesian_learning_service

    if _bayesian_learning_service is None:
        _bayesian_learning_service = BayesianLearningService()

    return _bayesian_learning_service
