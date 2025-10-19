"""
METIS Model Registry Service
Part of Application Services Cluster - Focused on model registration and discovery

Extracted from model_manager.py during Phase 5.3 decomposition.
Single Responsibility: Manage model registry, capabilities, and discovery.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import defaultdict

from src.services.contracts.application_contracts import (
    IModelRegistryService,
    ModelRegistryEntry,
    ApplicationStrategy,
)


class ModelRegistryService(IModelRegistryService):
    """
    Focused service for model registry and discovery management
    Clean extraction from model_manager.py model registration methods
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # In-memory registry (production would use persistent storage)
        self.model_registry: Dict[str, ModelRegistryEntry] = {}
        self.strategy_index: Dict[ApplicationStrategy, List[str]] = defaultdict(list)
        self.capability_index: Dict[str, List[str]] = defaultdict(list)

        # Registry management parameters
        self.registry_config = {
            "max_models_per_strategy": 10,
            "performance_baseline_retention_days": 30,
            "auto_retire_inactive_days": 90,
            "validation_status_hierarchy": [
                "experimental",
                "validated",
                "production",
                "deprecated",
            ],
        }

        # Performance tracking
        self.registry_metrics = {
            "total_registrations": 0,
            "active_models": 0,
            "registrations_by_strategy": defaultdict(int),
            "average_model_age_days": 0.0,
        }

        # Flag to track if defaults have been initialized
        self._defaults_initialized = False

        self.logger.info("📋 ModelRegistryService initialized")

    async def ensure_defaults_initialized(self):
        """Ensure default models are initialized (called on first use)"""
        if not self._defaults_initialized:
            await self._initialize_default_models()
            self._defaults_initialized = True

    async def register_model(self, model_entry: ModelRegistryEntry) -> Dict[str, Any]:
        """
        Core service method: Register a new model in the registry
        Comprehensive model registration with validation and indexing
        """
        try:
            # Validate model entry
            validation_result = await self._validate_model_entry(model_entry)

            if not validation_result["valid"]:
                return {
                    "success": False,
                    "error": validation_result["error"],
                    "registration_id": None,
                }

            # Check for existing model
            if model_entry.model_id in self.model_registry:
                return await self._update_existing_model(model_entry)

            # Register new model
            self.model_registry[model_entry.model_id] = model_entry

            # Update indexes
            await self._update_registry_indexes(model_entry)

            # Update metrics
            self.registry_metrics["total_registrations"] += 1
            self.registry_metrics["active_models"] += 1
            self.registry_metrics["registrations_by_strategy"][
                model_entry.application_strategy
            ] += 1

            registration_result = {
                "success": True,
                "registration_id": model_entry.model_id,
                "model_id": model_entry.model_id,
                "strategy": model_entry.application_strategy.value,
                "capabilities_count": len(model_entry.capabilities),
                "registration_timestamp": model_entry.registration_timestamp.isoformat(),
                "validation_status": model_entry.validation_status,
            }

            self.logger.info(
                f"📋 Model registered: {model_entry.model_id} ({model_entry.application_strategy.value})"
            )
            return registration_result

        except Exception as e:
            self.logger.error(f"❌ Model registration failed: {e}")
            return {"success": False, "error": str(e), "registration_id": None}

    async def get_model(self, model_id: str) -> Optional[ModelRegistryEntry]:
        """
        Core service method: Retrieve model information from registry
        Fast model lookup with comprehensive information
        """
        try:
            model_entry = self.model_registry.get(model_id)

            if model_entry:
                # Update last accessed timestamp (simplified)
                self.logger.debug(f"📋 Model retrieved: {model_id}")
                return model_entry
            else:
                self.logger.warning(f"⚠️ Model not found: {model_id}")
                return None

        except Exception as e:
            self.logger.error(f"❌ Model retrieval failed for {model_id}: {e}")
            return None

    async def list_models_by_strategy(
        self, strategy: ApplicationStrategy
    ) -> List[ModelRegistryEntry]:
        """
        Core service method: List all models supporting a specific application strategy
        Efficient strategy-based model discovery
        """
        try:
            model_ids = self.strategy_index.get(strategy, [])
            models = []

            for model_id in model_ids:
                model_entry = self.model_registry.get(model_id)
                if model_entry and model_entry.validation_status != "deprecated":
                    models.append(model_entry)

            # Sort by validation status and performance
            models.sort(
                key=lambda m: (
                    self.registry_config["validation_status_hierarchy"].index(
                        m.validation_status
                    ),
                    -m.performance_baseline.get("overall_score", 0.0),
                )
            )

            self.logger.info(
                f"📋 Found {len(models)} models for strategy: {strategy.value}"
            )
            return models

        except Exception as e:
            self.logger.error(f"❌ Strategy-based listing failed for {strategy}: {e}")
            return []

    async def search_models_by_capability(
        self, capability: str
    ) -> List[ModelRegistryEntry]:
        """Search models by specific capability"""
        try:
            model_ids = self.capability_index.get(capability, [])
            models = []

            for model_id in model_ids:
                model_entry = self.model_registry.get(model_id)
                if model_entry and model_entry.validation_status != "deprecated":
                    models.append(model_entry)

            self.logger.info(
                f"📋 Found {len(models)} models with capability: {capability}"
            )
            return models

        except Exception as e:
            self.logger.error(f"❌ Capability search failed for {capability}: {e}")
            return []

    async def get_model_performance_comparison(
        self, model_ids: List[str], metric_type: str = "overall_score"
    ) -> Dict[str, Any]:
        """Compare performance across multiple models"""
        try:
            comparison_data = {
                "models_compared": len(model_ids),
                "metric_type": metric_type,
                "comparison_results": [],
                "best_performer": None,
                "performance_spread": 0.0,
            }

            model_performances = []

            for model_id in model_ids:
                model_entry = self.model_registry.get(model_id)
                if model_entry:
                    performance_score = model_entry.performance_baseline.get(
                        metric_type, 0.0
                    )
                    model_performances.append(
                        {
                            "model_id": model_id,
                            "model_name": model_entry.model_name,
                            "performance_score": performance_score,
                            "validation_status": model_entry.validation_status,
                            "capabilities_count": len(model_entry.capabilities),
                        }
                    )

            if model_performances:
                # Sort by performance
                model_performances.sort(
                    key=lambda x: x["performance_score"], reverse=True
                )

                comparison_data["comparison_results"] = model_performances
                comparison_data["best_performer"] = model_performances[0]["model_id"]

                # Calculate performance spread
                scores = [m["performance_score"] for m in model_performances]
                comparison_data["performance_spread"] = max(scores) - min(scores)

            return comparison_data

        except Exception as e:
            self.logger.error(f"❌ Performance comparison failed: {e}")
            return {"error": str(e)}

    async def update_model_performance(
        self, model_id: str, performance_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Update model performance baseline with new metrics"""
        try:
            model_entry = self.model_registry.get(model_id)

            if not model_entry:
                return {"success": False, "error": f"Model {model_id} not found"}

            # Update performance baseline
            original_baseline = model_entry.performance_baseline.copy()
            model_entry.performance_baseline.update(performance_metrics)
            model_entry.last_updated = datetime.utcnow()

            # Calculate performance trend
            trend_analysis = self._calculate_performance_trend(
                original_baseline, model_entry.performance_baseline
            )

            update_result = {
                "success": True,
                "model_id": model_id,
                "metrics_updated": list(performance_metrics.keys()),
                "performance_trend": trend_analysis,
                "updated_timestamp": model_entry.last_updated.isoformat(),
            }

            self.logger.info(
                f"📈 Performance updated for {model_id}: {trend_analysis['overall_trend']}"
            )
            return update_result

        except Exception as e:
            self.logger.error(f"❌ Performance update failed for {model_id}: {e}")
            return {"success": False, "error": str(e)}

    async def retire_model(
        self, model_id: str, reason: str = "Manual retirement"
    ) -> Dict[str, Any]:
        """Retire a model from active registry"""
        try:
            model_entry = self.model_registry.get(model_id)

            if not model_entry:
                return {"success": False, "error": f"Model {model_id} not found"}

            # Update model status
            model_entry.validation_status = "deprecated"
            model_entry.last_updated = datetime.utcnow()

            # Remove from active indexes
            await self._remove_from_indexes(model_entry)

            # Update metrics
            self.registry_metrics["active_models"] -= 1

            retirement_result = {
                "success": True,
                "model_id": model_id,
                "retirement_reason": reason,
                "retired_timestamp": model_entry.last_updated.isoformat(),
                "active_models_remaining": self.registry_metrics["active_models"],
            }

            self.logger.info(f"🏁 Model retired: {model_id} - {reason}")
            return retirement_result

        except Exception as e:
            self.logger.error(f"❌ Model retirement failed for {model_id}: {e}")
            return {"success": False, "error": str(e)}

    async def get_registry_analytics(self) -> Dict[str, Any]:
        """Get comprehensive registry analytics"""
        try:
            # Calculate model age statistics
            current_time = datetime.utcnow()
            model_ages = []

            for model_entry in self.model_registry.values():
                age_days = (current_time - model_entry.registration_timestamp).days
                model_ages.append(age_days)

            avg_age = sum(model_ages) / len(model_ages) if model_ages else 0.0
            self.registry_metrics["average_model_age_days"] = avg_age

            # Strategy distribution
            strategy_distribution = {}
            for strategy, model_ids in self.strategy_index.items():
                active_models = [
                    mid
                    for mid in model_ids
                    if self.model_registry.get(mid, {}).validation_status
                    != "deprecated"
                ]
                strategy_distribution[strategy.value] = len(active_models)

            # Capability coverage
            capability_coverage = {
                capability: len(model_ids)
                for capability, model_ids in self.capability_index.items()
            }

            # Performance statistics
            performance_stats = self._calculate_performance_statistics()

            analytics = {
                "registry_summary": {
                    "total_models": len(self.model_registry),
                    "active_models": self.registry_metrics["active_models"],
                    "deprecated_models": len(self.model_registry)
                    - self.registry_metrics["active_models"],
                    "average_model_age_days": round(avg_age, 1),
                },
                "strategy_distribution": strategy_distribution,
                "capability_coverage": capability_coverage,
                "performance_statistics": performance_stats,
                "registry_health": {
                    "status": (
                        "healthy"
                        if self.registry_metrics["active_models"] > 0
                        else "warning"
                    ),
                    "coverage_completeness": len(strategy_distribution)
                    / len(ApplicationStrategy)
                    * 100,
                },
                "analytics_timestamp": datetime.utcnow().isoformat(),
            }

            return analytics

        except Exception as e:
            self.logger.error(f"❌ Analytics generation failed: {e}")
            return {"error": str(e)}

    async def _validate_model_entry(
        self, model_entry: ModelRegistryEntry
    ) -> Dict[str, Any]:
        """Validate model entry before registration"""
        try:
            validation_errors = []

            # Required field validation
            if not model_entry.model_id:
                validation_errors.append("model_id is required")

            if not model_entry.model_name:
                validation_errors.append("model_name is required")

            if not model_entry.application_strategy:
                validation_errors.append("application_strategy is required")

            # Capabilities validation
            if not model_entry.capabilities or len(model_entry.capabilities) == 0:
                validation_errors.append("At least one capability is required")

            # Performance baseline validation
            if not model_entry.performance_baseline:
                validation_errors.append("performance_baseline is required")

            # Validation status check
            valid_statuses = self.registry_config["validation_status_hierarchy"]
            if model_entry.validation_status not in valid_statuses:
                validation_errors.append(
                    f"validation_status must be one of: {valid_statuses}"
                )

            return {
                "valid": len(validation_errors) == 0,
                "error": "; ".join(validation_errors) if validation_errors else None,
                "errors": validation_errors,
            }

        except Exception as e:
            return {"valid": False, "error": f"Validation failed: {str(e)}"}

    async def _update_existing_model(
        self, model_entry: ModelRegistryEntry
    ) -> Dict[str, Any]:
        """Update existing model registration"""
        existing_entry = self.model_registry[model_entry.model_id]

        # Update fields
        existing_entry.model_name = model_entry.model_name
        existing_entry.model_version = model_entry.model_version
        existing_entry.capabilities = model_entry.capabilities
        existing_entry.performance_baseline.update(model_entry.performance_baseline)
        existing_entry.resource_requirements.update(model_entry.resource_requirements)
        existing_entry.validation_status = model_entry.validation_status
        existing_entry.last_updated = datetime.utcnow()

        # Re-index
        await self._update_registry_indexes(existing_entry)

        return {
            "success": True,
            "registration_id": model_entry.model_id,
            "model_id": model_entry.model_id,
            "action": "updated",
            "updated_timestamp": existing_entry.last_updated.isoformat(),
        }

    async def _update_registry_indexes(self, model_entry: ModelRegistryEntry):
        """Update registry indexes for efficient lookup"""
        try:
            # Strategy index
            if (
                model_entry.model_id
                not in self.strategy_index[model_entry.application_strategy]
            ):
                self.strategy_index[model_entry.application_strategy].append(
                    model_entry.model_id
                )

            # Capability index
            for capability in model_entry.capabilities:
                if model_entry.model_id not in self.capability_index[capability]:
                    self.capability_index[capability].append(model_entry.model_id)

        except Exception as e:
            self.logger.error(f"❌ Index update failed: {e}")

    async def _remove_from_indexes(self, model_entry: ModelRegistryEntry):
        """Remove model from registry indexes"""
        try:
            # Remove from strategy index
            strategy_models = self.strategy_index.get(
                model_entry.application_strategy, []
            )
            if model_entry.model_id in strategy_models:
                strategy_models.remove(model_entry.model_id)

            # Remove from capability indexes
            for capability in model_entry.capabilities:
                capability_models = self.capability_index.get(capability, [])
                if model_entry.model_id in capability_models:
                    capability_models.remove(model_entry.model_id)

        except Exception as e:
            self.logger.error(f"❌ Index removal failed: {e}")

    def _calculate_performance_trend(
        self, original_baseline: Dict[str, float], updated_baseline: Dict[str, float]
    ) -> Dict[str, Any]:
        """Calculate performance trend from baseline comparison"""
        try:
            improvements = 0
            degradations = 0
            total_comparisons = 0

            for metric, new_value in updated_baseline.items():
                if metric in original_baseline:
                    old_value = original_baseline[metric]

                    if new_value > old_value:
                        improvements += 1
                    elif new_value < old_value:
                        degradations += 1

                    total_comparisons += 1

            if total_comparisons == 0:
                overall_trend = "no_comparison_data"
            elif improvements > degradations:
                overall_trend = "improving"
            elif degradations > improvements:
                overall_trend = "declining"
            else:
                overall_trend = "stable"

            return {
                "overall_trend": overall_trend,
                "improvements": improvements,
                "degradations": degradations,
                "stable_metrics": total_comparisons - improvements - degradations,
                "total_comparisons": total_comparisons,
            }

        except Exception as e:
            return {"overall_trend": "error", "error": str(e)}

    def _calculate_performance_statistics(self) -> Dict[str, Any]:
        """Calculate performance statistics across all models"""
        try:
            if not self.model_registry:
                return {"no_models": True}

            all_scores = []
            metric_aggregates = defaultdict(list)

            for model_entry in self.model_registry.values():
                if model_entry.validation_status != "deprecated":
                    for metric, value in model_entry.performance_baseline.items():
                        metric_aggregates[metric].append(value)
                        if metric == "overall_score":
                            all_scores.append(value)

            # Calculate statistics
            stats = {}
            for metric, values in metric_aggregates.items():
                if values:
                    stats[metric] = {
                        "average": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                        "count": len(values),
                    }

            # Overall statistics
            if all_scores:
                stats["overall_performance"] = {
                    "average_score": sum(all_scores) / len(all_scores),
                    "top_performer_score": max(all_scores),
                    "performance_range": max(all_scores) - min(all_scores),
                }

            return stats

        except Exception as e:
            return {"error": str(e)}

    async def _initialize_default_models(self):
        """Initialize registry with default models"""
        try:
            default_models = [
                ModelRegistryEntry(
                    model_id="deepseek_chat",
                    model_name="DeepSeek Chat V3.1",
                    model_version="v3.1",
                    application_strategy=ApplicationStrategy.SYSTEMS_THINKING,
                    capabilities=[
                        "reasoning",
                        "analysis",
                        "code_generation",
                        "technical_thinking",
                    ],
                    performance_baseline={
                        "overall_score": 0.85,
                        "accuracy_score": 0.82,
                        "coherence_score": 0.88,
                        "completeness_score": 0.85,
                    },
                    resource_requirements={"memory_mb": 1024, "cpu_cores": 2},
                    validation_status="production",
                    registration_timestamp=datetime.utcnow(),
                    last_updated=datetime.utcnow(),
                    service_version="v5_modular",
                ),
                ModelRegistryEntry(
                    model_id="claude_sonnet",
                    model_name="Claude 3.5 Sonnet",
                    model_version="3.5",
                    application_strategy=ApplicationStrategy.CRITICAL_THINKING,
                    capabilities=["analysis", "writing", "reasoning", "communication"],
                    performance_baseline={
                        "overall_score": 0.83,
                        "accuracy_score": 0.85,
                        "coherence_score": 0.87,
                        "completeness_score": 0.78,
                    },
                    resource_requirements={"memory_mb": 1024, "cpu_cores": 2},
                    validation_status="production",
                    registration_timestamp=datetime.utcnow(),
                    last_updated=datetime.utcnow(),
                    service_version="v5_modular",
                ),
                ModelRegistryEntry(
                    model_id="generic_llm",
                    model_name="Generic LLM",
                    model_version="1.0",
                    application_strategy=ApplicationStrategy.GENERIC_APPLICATION,
                    capabilities=["general_reasoning", "text_generation"],
                    performance_baseline={
                        "overall_score": 0.70,
                        "accuracy_score": 0.68,
                        "coherence_score": 0.72,
                        "completeness_score": 0.70,
                    },
                    resource_requirements={"memory_mb": 512, "cpu_cores": 1},
                    validation_status="validated",
                    registration_timestamp=datetime.utcnow(),
                    last_updated=datetime.utcnow(),
                    service_version="v5_modular",
                ),
            ]

            for model_entry in default_models:
                await self.register_model(model_entry)

            self.logger.info(
                f"📋 Initialized registry with {len(default_models)} default models"
            )

        except Exception as e:
            self.logger.error(f"❌ Default models initialization failed: {e}")

    async def get_service_health(self) -> Dict[str, Any]:
        """Return service health status"""
        return {
            "service_name": "ModelRegistryService",
            "status": "healthy",
            "version": "v5_modular",
            "capabilities": [
                "model_registration",
                "model_discovery",
                "performance_tracking",
                "capability_indexing",
                "analytics_generation",
            ],
            "registry_statistics": {
                "total_models": len(self.model_registry),
                "active_models": self.registry_metrics["active_models"],
                "strategies_covered": len(self.strategy_index),
                "capabilities_indexed": len(self.capability_index),
            },
            "configuration": self.registry_config,
            "last_health_check": datetime.utcnow().isoformat(),
        }


# Global service instance for dependency injection
_model_registry_service: Optional[ModelRegistryService] = None


def get_model_registry_service() -> ModelRegistryService:
    """Get or create global model registry service instance"""
    global _model_registry_service

    if _model_registry_service is None:
        _model_registry_service = ModelRegistryService()

    return _model_registry_service
