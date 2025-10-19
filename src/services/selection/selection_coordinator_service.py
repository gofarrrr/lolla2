"""
METIS Selection Coordinator Service
Part of Selection Services Cluster - Master orchestrator for all selection services

Extracted from model_selector.py during Phase 5.2 decomposition.
Single Responsibility: Coordinate all selection services to provide unified model selection.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.services.contracts.selection_contracts import (
    ISelectionCoordinatorService,
    SelectionCoordinationContract,
    SelectionResultContract,
    SelectionContextContract,
    SelectionSource,
    MergeStrategy,
)

# Import all selection services
from src.services.selection.selection_strategy_service import (
    get_selection_strategy_service,
)
from src.services.selection.scoring_engine_service import get_scoring_engine_service
from src.services.selection.nway_pattern_service import get_nway_pattern_service
from src.services.selection.bayesian_learning_service import (
    get_bayesian_learning_service,
)
from src.services.selection.zero_shot_selection_service import (
    get_zero_shot_selection_service,
)


class SelectionCoordinatorService(ISelectionCoordinatorService):
    """
    Master coordinator service for all selection services
    Clean orchestration with single responsibility for unified selection
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Initialize all selection services
        self.strategy_service = get_selection_strategy_service()
        self.scoring_service = get_scoring_engine_service()
        self.nway_service = get_nway_pattern_service()
        self.bayesian_service = get_bayesian_learning_service()
        self.zero_shot_service = get_zero_shot_selection_service()

        # Coordination parameters
        self.selection_orchestration_config = {
            "min_historical_observations": 5,
            "zero_shot_confidence_threshold": 0.7,
            "hybrid_merge_strategy": MergeStrategy.WEIGHTED_CONFIDENCE,
            "nway_enhancement_enabled": True,
            "bayesian_learning_enabled": True,
            "service_health_check_interval": 300,  # 5 minutes
        }

        # Service performance tracking
        self.service_performance_metrics = {
            "total_selections_coordinated": 0,
            "average_coordination_time_ms": 0.0,
            "service_failure_counts": {
                "strategy": 0,
                "scoring": 0,
                "nway": 0,
                "bayesian": 0,
                "zero_shot": 0,
            },
        }

        self.logger.info(
            "🎭 SelectionCoordinatorService initialized - All services orchestrated"
        )

    async def coordinate_model_selection(
        self, context: SelectionContextContract
    ) -> SelectionCoordinationContract:
        """
        Core coordination method: Orchestrate all selection services for unified result
        Master orchestration with intelligent service coordination
        """
        try:
            start_time = datetime.utcnow()

            # Phase 1: Determine selection approach based on context
            selection_approach = await self._determine_selection_approach(context)

            # Phase 2: Execute coordinated selection based on approach
            if selection_approach == SelectionSource.DATABASE_ONLY:
                coordination_result = await self._coordinate_database_selection(context)
            elif selection_approach == SelectionSource.ZERO_SHOT_ONLY:
                coordination_result = await self._coordinate_zero_shot_selection(
                    context
                )
            else:  # HYBRID
                coordination_result = await self._coordinate_hybrid_selection(context)

            # Phase 3: Apply post-selection enhancements
            enhanced_result = await self._apply_post_selection_enhancements(
                coordination_result, context
            )

            # Phase 4: Calculate coordination metrics
            coordination_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            overall_confidence = self._calculate_overall_confidence(enhanced_result)

            # Phase 5: Update performance metrics
            await self._update_coordination_metrics(coordination_time)

            # Phase 6: Create comprehensive coordination contract
            final_coordination = SelectionCoordinationContract(
                engagement_id=context.problem_statement[:50] + "_coordinated",
                selection_result=enhanced_result["selection_result"],
                nway_interactions=enhanced_result.get("nway_interactions", []),
                bayesian_updates=enhanced_result.get("bayesian_updates", []),
                zero_shot_result=enhanced_result.get("zero_shot_result"),
                coordination_metadata={
                    "selection_approach": selection_approach.value,
                    "services_orchestrated": enhanced_result["services_used"],
                    "coordination_time_ms": coordination_time,
                    "enhancement_applied": enhanced_result.get(
                        "enhancements_applied", []
                    ),
                    "service_performance": self._get_service_performance_summary(),
                },
                overall_confidence=overall_confidence,
                coordination_timestamp=datetime.utcnow(),
                service_version="v5_modular_coordinator",
            )

            self.logger.info(
                f"🎭 Coordination completed in {coordination_time:.0f}ms with {overall_confidence:.2f} confidence"
            )
            return final_coordination

        except Exception as e:
            self.logger.error(f"❌ Coordination failed: {e}")
            return await self._create_fallback_coordination(context, str(e))

    async def _determine_selection_approach(
        self, context: SelectionContextContract
    ) -> SelectionSource:
        """Intelligently determine the best selection approach based on context"""
        try:
            # Check if we have sufficient historical data
            historical_data_check = await self._assess_historical_data_availability(
                context
            )

            # Analyze context complexity and novelty
            complexity_analysis = await self._analyze_context_complexity(context)

            # Decision logic for selection approach
            if (
                historical_data_check["sufficient_data"]
                and complexity_analysis["novelty_level"] < 0.7
            ):
                # Standard database-driven selection
                return SelectionSource.DATABASE_ONLY

            elif (
                not historical_data_check["sufficient_data"]
                or complexity_analysis["novelty_level"] > 0.8
            ):
                # Zero-shot selection for novel/unprecedented scenarios
                return SelectionSource.ZERO_SHOT_ONLY

            else:
                # Hybrid approach for balanced scenarios
                return SelectionSource.HYBRID

        except Exception as e:
            self.logger.error(f"❌ Failed to determine approach: {e}")
            return SelectionSource.HYBRID  # Safe default

    async def _coordinate_database_selection(
        self, context: SelectionContextContract
    ) -> Dict[str, Any]:
        """Coordinate database-driven selection using historical data"""
        try:
            coordination_result = {
                "services_used": ["strategy", "scoring", "nway", "bayesian"],
                "enhancements_applied": [],
            }

            # Step 1: Get available models (simplified for this implementation)
            available_models = await self._get_available_models(context)

            # Step 2: Score models using scoring engine
            model_scores = await self.scoring_service.score_models(
                available_models, context
            )

            # Step 3: Apply Bayesian learning adjustments
            bayesian_updates = []
            for model, score in zip(available_models, model_scores):
                bayesian_effectiveness = (
                    await self.bayesian_service.get_learned_effectiveness(
                        getattr(model, "model_id", str(model)), context
                    )
                )
                if bayesian_effectiveness:
                    bayesian_updates.append(bayesian_effectiveness)
                    # Adjust score based on learned effectiveness
                    score.total_score = (score.total_score * 0.7) + (
                        bayesian_effectiveness.effectiveness_score * 0.3
                    )

            coordination_result["bayesian_updates"] = bayesian_updates

            # Step 4: Recommend and execute selection strategy
            recommended_strategy = await self.strategy_service.recommend_strategy(
                context
            )
            selection_result = await self.strategy_service.execute_selection_strategy(
                recommended_strategy, available_models, model_scores, context
            )

            coordination_result["selection_result"] = selection_result

            self.logger.info("📊 Database-driven selection coordinated successfully")
            return coordination_result

        except Exception as e:
            self.logger.error(f"❌ Database selection coordination failed: {e}")
            self.service_performance_metrics["service_failure_counts"]["strategy"] += 1
            raise

    async def _coordinate_zero_shot_selection(
        self, context: SelectionContextContract
    ) -> Dict[str, Any]:
        """Coordinate zero-shot selection for novel scenarios"""
        try:
            coordination_result = {
                "services_used": ["zero_shot"],
                "enhancements_applied": [],
            }

            # Perform zero-shot selection using MeMo methodology
            zero_shot_result = await self.zero_shot_service.perform_zero_shot_selection(
                context
            )
            coordination_result["zero_shot_result"] = zero_shot_result

            # Convert zero-shot result to selection result format
            selection_result = self._convert_zero_shot_to_selection_result(
                zero_shot_result, context
            )
            coordination_result["selection_result"] = selection_result

            self.logger.info("🎯 Zero-shot selection coordinated successfully")
            return coordination_result

        except Exception as e:
            self.logger.error(f"❌ Zero-shot selection coordination failed: {e}")
            self.service_performance_metrics["service_failure_counts"]["zero_shot"] += 1
            raise

    async def _coordinate_hybrid_selection(
        self, context: SelectionContextContract
    ) -> Dict[str, Any]:
        """Coordinate hybrid selection combining database and zero-shot approaches"""
        try:
            coordination_result = {
                "services_used": [
                    "strategy",
                    "scoring",
                    "zero_shot",
                    "nway",
                    "bayesian",
                ],
                "enhancements_applied": [],
            }

            # Run both approaches in parallel
            database_task = asyncio.create_task(
                self._coordinate_database_selection(context)
            )
            zero_shot_task = asyncio.create_task(
                self._coordinate_zero_shot_selection(context)
            )

            database_result, zero_shot_result = await asyncio.gather(
                database_task, zero_shot_task, return_exceptions=True
            )

            # Handle potential failures
            if isinstance(database_result, Exception):
                self.logger.warning(
                    f"Database selection failed in hybrid mode: {database_result}"
                )
                return (
                    zero_shot_result
                    if not isinstance(zero_shot_result, Exception)
                    else {"error": "Both approaches failed"}
                )

            if isinstance(zero_shot_result, Exception):
                self.logger.warning(
                    f"Zero-shot selection failed in hybrid mode: {zero_shot_result}"
                )
                return database_result

            # Merge results using configured strategy
            merged_selection = (
                await self.zero_shot_service.merge_with_database_selection(
                    zero_shot_result["zero_shot_result"],
                    database_result["selection_result"],
                    self.selection_orchestration_config["hybrid_merge_strategy"],
                )
            )

            coordination_result.update(
                {
                    "selection_result": merged_selection,
                    "zero_shot_result": zero_shot_result["zero_shot_result"],
                    "bayesian_updates": database_result.get("bayesian_updates", []),
                    "enhancements_applied": ["hybrid_merge"],
                }
            )

            self.logger.info("🔄 Hybrid selection coordinated successfully")
            return coordination_result

        except Exception as e:
            self.logger.error(f"❌ Hybrid selection coordination failed: {e}")
            raise

    async def _apply_post_selection_enhancements(
        self, coordination_result: Dict[str, Any], context: SelectionContextContract
    ) -> Dict[str, Any]:
        """Apply post-selection enhancements like N-Way patterns"""
        try:
            enhanced_result = coordination_result.copy()

            # Apply N-Way pattern enhancement if enabled
            if (
                self.selection_orchestration_config["nway_enhancement_enabled"]
                and "selection_result" in coordination_result
            ):

                try:
                    selection_result = coordination_result["selection_result"]
                    nway_interactions = (
                        await self.nway_service.detect_nway_interactions(
                            selection_result.selected_models, context
                        )
                    )

                    if nway_interactions:
                        enhanced_selection = (
                            await self.nway_service.enhance_with_nway_patterns(
                                selection_result, context
                            )
                        )
                        enhanced_result["selection_result"] = enhanced_selection
                        enhanced_result["nway_interactions"] = nway_interactions
                        enhanced_result["enhancements_applied"].append("nway_patterns")

                        self.logger.info(
                            f"✨ Applied N-Way enhancements: {len(nway_interactions)} interactions"
                        )

                except Exception as e:
                    self.logger.warning(f"⚠️ N-Way enhancement failed: {e}")
                    self.service_performance_metrics["service_failure_counts"][
                        "nway"
                    ] += 1

            return enhanced_result

        except Exception as e:
            self.logger.error(f"❌ Post-selection enhancement failed: {e}")
            return coordination_result  # Return original result

    async def _assess_historical_data_availability(
        self, context: SelectionContextContract
    ) -> Dict[str, Any]:
        """Assess if we have sufficient historical data for database-driven selection"""
        try:
            # Simplified assessment - in production, would query actual database

            # Heuristic based on context specificity
            specificity_score = 0.0

            if context.problem_type and context.problem_type != "general":
                specificity_score += 0.3

            if context.complexity_level and context.complexity_level != "medium":
                specificity_score += 0.2

            if (
                isinstance(context.business_context, dict)
                and len(context.business_context) > 2
            ):
                specificity_score += 0.2

            # Mock historical data availability (would be real database query)
            sufficient_data = specificity_score >= 0.5

            return {
                "sufficient_data": sufficient_data,
                "specificity_score": specificity_score,
                "estimated_historical_cases": int(
                    specificity_score * 20
                ),  # Mock estimate
                "data_quality": "high" if sufficient_data else "low",
            }

        except Exception as e:
            self.logger.error(f"❌ Historical data assessment failed: {e}")
            return {"sufficient_data": False, "error": str(e)}

    async def _analyze_context_complexity(
        self, context: SelectionContextContract
    ) -> Dict[str, Any]:
        """Analyze context complexity and novelty"""
        try:
            novelty_indicators = 0

            # Check for novelty keywords in problem statement
            novelty_keywords = [
                "new",
                "novel",
                "unprecedented",
                "first",
                "innovative",
                "unique",
            ]
            problem_text = context.problem_statement.lower()

            for keyword in novelty_keywords:
                if keyword in problem_text:
                    novelty_indicators += 1

            # Complexity factors
            complexity_factors = []
            if context.complexity_level in ["high", "very_high"]:
                complexity_factors.append("high_complexity_level")

            if context.accuracy_requirement >= 0.9:
                complexity_factors.append("high_accuracy_requirement")

            if context.time_constraint in ["urgent", "immediate"]:
                complexity_factors.append("time_pressure")

            novelty_level = min(
                novelty_indicators * 0.2 + len(complexity_factors) * 0.15, 1.0
            )

            return {
                "novelty_level": novelty_level,
                "novelty_indicators": novelty_indicators,
                "complexity_factors": complexity_factors,
                "complexity_assessment": (
                    "high"
                    if novelty_level > 0.7
                    else "medium" if novelty_level > 0.4 else "low"
                ),
            }

        except Exception as e:
            self.logger.error(f"❌ Complexity analysis failed: {e}")
            return {"novelty_level": 0.5, "error": str(e)}

    async def _get_available_models(
        self, context: SelectionContextContract
    ) -> List[Any]:
        """Get available models for selection (mock implementation)"""

        # Mock model objects - in production, would query from database or model registry
        class MockModel:
            def __init__(self, model_id: str, category: str):
                self.model_id = model_id
                self.category = category

        # Return mock models based on context
        available_models = [
            MockModel("deepseek_chat", "reasoning"),
            MockModel("claude_sonnet", "analysis"),
            MockModel("gpt4", "general"),
        ]

        return available_models

    def _convert_zero_shot_to_selection_result(
        self, zero_shot_result: Any, context: SelectionContextContract
    ) -> SelectionResultContract:
        """Convert zero-shot result to standard selection result format"""
        from src.services.contracts.selection_contracts import ModelScoreContract

        # Create mock scores for selected models
        model_scores = []
        for i, model_id in enumerate(zero_shot_result.selected_models):
            score = ModelScoreContract(
                model_id=model_id,
                total_score=0.8 - (i * 0.1),  # Decreasing scores
                component_scores={
                    "zero_shot_preference": 1.0,
                    "reasoning_based": zero_shot_result.confidence_score,
                },
                rationale="Zero-shot selection based on MeMo reasoning",
                confidence=zero_shot_result.confidence_score,
                risk_factors=["zero_shot_uncertainty"],
                scoring_timestamp=datetime.utcnow(),
                service_version="v5_modular_zeroshot",
            )
            model_scores.append(score)

        return SelectionResultContract(
            engagement_id=zero_shot_result.engagement_id + "_converted",
            selected_models=zero_shot_result.selected_models,
            model_scores=model_scores,
            selection_source="zero_shot_converted",
            strategy_used="memo_methodology",
            models_evaluated=len(zero_shot_result.selected_models),
            selection_metadata={
                "conversion_from": "zero_shot",
                "original_confidence": zero_shot_result.confidence_score,
                "reasoning_steps": len(zero_shot_result.reasoning_process),
            },
            total_selection_time_ms=100.0,  # Mock timing
            cognitive_load_assessment="medium",
            selection_timestamp=datetime.utcnow(),
            service_version="v5_modular_converted",
        )

    def _calculate_overall_confidence(
        self, coordination_result: Dict[str, Any]
    ) -> float:
        """Calculate overall confidence in coordinated selection"""
        try:
            confidence_factors = []

            # Selection result confidence
            if "selection_result" in coordination_result:
                selection_result = coordination_result["selection_result"]
                if (
                    hasattr(selection_result, "model_scores")
                    and selection_result.model_scores
                ):
                    avg_confidence = sum(
                        score.confidence for score in selection_result.model_scores
                    ) / len(selection_result.model_scores)
                    confidence_factors.append(avg_confidence)

            # Zero-shot confidence
            if (
                "zero_shot_result" in coordination_result
                and coordination_result["zero_shot_result"]
            ):
                confidence_factors.append(
                    coordination_result["zero_shot_result"].confidence_score
                )

            # Bayesian learning confidence
            if (
                "bayesian_updates" in coordination_result
                and coordination_result["bayesian_updates"]
            ):
                bayesian_confidences = [
                    update.effectiveness_score
                    for update in coordination_result["bayesian_updates"]
                ]
                if bayesian_confidences:
                    avg_bayesian_confidence = sum(bayesian_confidences) / len(
                        bayesian_confidences
                    )
                    confidence_factors.append(avg_bayesian_confidence)

            # Enhancement confidence bonus
            if "enhancements_applied" in coordination_result:
                enhancement_bonus = (
                    len(coordination_result["enhancements_applied"]) * 0.05
                )
                confidence_factors.append(min(0.8 + enhancement_bonus, 1.0))

            # Calculate weighted average
            if confidence_factors:
                overall_confidence = sum(confidence_factors) / len(confidence_factors)
            else:
                overall_confidence = 0.5  # Neutral default

            return max(0.1, min(overall_confidence, 0.95))  # Bound between 0.1 and 0.95

        except Exception as e:
            self.logger.error(f"❌ Confidence calculation failed: {e}")
            return 0.5

    async def _update_coordination_metrics(self, coordination_time_ms: float):
        """Update coordination performance metrics"""
        try:
            self.service_performance_metrics["total_selections_coordinated"] += 1

            # Update running average
            current_avg = self.service_performance_metrics[
                "average_coordination_time_ms"
            ]
            total_selections = self.service_performance_metrics[
                "total_selections_coordinated"
            ]

            new_avg = (
                (current_avg * (total_selections - 1)) + coordination_time_ms
            ) / total_selections
            self.service_performance_metrics["average_coordination_time_ms"] = new_avg

        except Exception as e:
            self.logger.error(f"❌ Metrics update failed: {e}")

    def _get_service_performance_summary(self) -> Dict[str, Any]:
        """Get summary of service performance metrics"""
        return {
            "total_coordinated": self.service_performance_metrics[
                "total_selections_coordinated"
            ],
            "avg_time_ms": round(
                self.service_performance_metrics["average_coordination_time_ms"], 2
            ),
            "failure_rates": {
                service: f"{failures}/{self.service_performance_metrics['total_selections_coordinated']}"
                for service, failures in self.service_performance_metrics[
                    "service_failure_counts"
                ].items()
            },
        }

    async def _create_fallback_coordination(
        self, context: SelectionContextContract, error_msg: str
    ) -> SelectionCoordinationContract:
        """Create fallback coordination result when coordination fails"""
        # Create minimal fallback selection result
        from src.services.contracts.selection_contracts import ModelScoreContract

        fallback_score = ModelScoreContract(
            model_id="deepseek_chat",
            total_score=0.5,
            component_scores={"fallback": 0.5},
            rationale=f"Fallback selection due to coordination failure: {error_msg}",
            confidence=0.3,
            risk_factors=["coordination_failure"],
            scoring_timestamp=datetime.utcnow(),
            service_version="v5_modular_fallback",
        )

        fallback_selection = SelectionResultContract(
            engagement_id=context.problem_statement[:50] + "_fallback",
            selected_models=["deepseek_chat"],
            model_scores=[fallback_score],
            selection_source="fallback",
            strategy_used="emergency_fallback",
            models_evaluated=1,
            selection_metadata={"fallback_triggered": True, "error_message": error_msg},
            total_selection_time_ms=50.0,
            cognitive_load_assessment="low",
            selection_timestamp=datetime.utcnow(),
            service_version="v5_modular_fallback",
        )

        return SelectionCoordinationContract(
            engagement_id=context.problem_statement[:50] + "_coord_fallback",
            selection_result=fallback_selection,
            nway_interactions=[],
            bayesian_updates=[],
            zero_shot_result=None,
            coordination_metadata={
                "fallback_triggered": True,
                "error_message": error_msg,
                "services_orchestrated": [],
            },
            overall_confidence=0.3,
            coordination_timestamp=datetime.utcnow(),
            service_version="v5_modular_fallback",
        )

    async def get_cluster_health(self) -> Dict[str, Any]:
        """Get comprehensive health status of entire selection cluster"""
        try:
            # Get health from all constituent services
            service_healths = {}

            try:
                service_healths["strategy"] = (
                    await self.strategy_service.get_service_health()
                )
            except Exception as e:
                service_healths["strategy"] = {"status": "error", "error": str(e)}

            try:
                service_healths["scoring"] = (
                    await self.scoring_service.get_service_health()
                )
            except Exception as e:
                service_healths["scoring"] = {"status": "error", "error": str(e)}

            try:
                service_healths["nway"] = await self.nway_service.get_service_health()
            except Exception as e:
                service_healths["nway"] = {"status": "error", "error": str(e)}

            try:
                service_healths["bayesian"] = (
                    await self.bayesian_service.get_service_health()
                )
            except Exception as e:
                service_healths["bayesian"] = {"status": "error", "error": str(e)}

            try:
                service_healths["zero_shot"] = (
                    await self.zero_shot_service.get_service_health()
                )
            except Exception as e:
                service_healths["zero_shot"] = {"status": "error", "error": str(e)}

            # Calculate cluster health
            healthy_services = sum(
                1
                for health in service_healths.values()
                if health.get("status") == "healthy"
            )
            total_services = len(service_healths)
            cluster_health_percentage = (healthy_services / total_services) * 100

            return {
                "cluster_name": "SelectionServicesCluster",
                "cluster_status": (
                    "healthy"
                    if cluster_health_percentage >= 80
                    else "degraded" if cluster_health_percentage >= 50 else "critical"
                ),
                "cluster_health_percentage": cluster_health_percentage,
                "coordinator_version": "v5_modular",
                "services_count": total_services,
                "healthy_services": healthy_services,
                "service_details": service_healths,
                "coordinator_metrics": self.service_performance_metrics,
                "orchestration_config": self.selection_orchestration_config,
                "capabilities": [
                    "unified_model_selection",
                    "hybrid_selection_strategies",
                    "intelligent_approach_determination",
                    "post_selection_enhancement",
                    "service_health_monitoring",
                ],
                "last_health_check": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            return {
                "cluster_name": "SelectionServicesCluster",
                "cluster_status": "error",
                "error": str(e),
                "last_health_check": datetime.utcnow().isoformat(),
            }


# Global service instance for dependency injection
_selection_coordinator_service: Optional[SelectionCoordinatorService] = None


def get_selection_coordinator_service() -> SelectionCoordinatorService:
    """Get or create global selection coordinator service instance"""
    global _selection_coordinator_service

    if _selection_coordinator_service is None:
        _selection_coordinator_service = SelectionCoordinatorService()

    return _selection_coordinator_service
