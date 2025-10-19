"""
Unified Model Manager - Main orchestrator for model operations
Coordinates model selection, application, performance tracking, and configuration
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.engine.models.data_contracts import MentalModelDefinition
from src.interfaces.model_manager_interface import IModelManager
from src.engine.engines.model_selection import ModelSelector
from src.engine.engines.model_application.model_applicator import ModelApplicationEngine
from src.engine.engines.model_performance import ModelPerformanceTracker
from src.engine.engines.model_config import ModelConfigurator
from src.intelligence.model_catalog import MentalModelCatalog


# Import selection criteria with fallback
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.engine.engines.cognitive_engine import ModelSelectionCriteria
else:
    try:
        from src.engine.engines.cognitive_engine import ModelSelectionCriteria
    except ImportError:

        class ModelSelectionCriteria:
            def __init__(self):
                self.problem_type = ""
                self.complexity_level = "medium"
                self.accuracy_requirement = 0.8
                self.business_context = {}


class UnifiedModelManager(IModelManager):
    """
    Unified model management orchestrator that coordinates all model operations
    """

    def __init__(
        self,
        model_catalog: MentalModelCatalog,
        model_selector: Optional[ModelSelector] = None,
        model_applicator: Optional[ModelApplicationEngine] = None,
        performance_tracker: Optional[ModelPerformanceTracker] = None,
        model_configurator: Optional[ModelConfigurator] = None,
        llm_provider: Optional[Any] = None,
        vector_similarity_engine: Optional[Any] = None,
        supabase_platform: Optional[Any] = None,
        settings: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.model_catalog = model_catalog
        self.settings = settings
        self.logger = logger or logging.getLogger(__name__)

        # Initialize components with dependency injection
        self.configurator = model_configurator or ModelConfigurator(
            model_catalog=model_catalog, settings=settings, logger=self.logger
        )

        self.selector = model_selector or ModelSelector(
            model_catalog=model_catalog,
            vector_similarity_engine=vector_similarity_engine,
            logger=self.logger,
            settings=settings,
        )

        self.applicator = model_applicator or ModelApplicationEngine(
            llm_provider=llm_provider,
            model_catalog=model_catalog,
            logger=self.logger,
            settings=settings,
        )

        self.tracker = performance_tracker or ModelPerformanceTracker(
            vector_similarity_engine=vector_similarity_engine,
            supabase_platform=supabase_platform,
            logger=self.logger,
            settings=settings,
        )

        # Orchestration state
        self.current_models_in_use = {}
        self.active_processing_sessions = {}

        self.logger.info("🧠 UnifiedModelManager initialized with all components")

    async def select_and_apply_models(
        self,
        criteria: "ModelSelectionCriteria",
        problem_statement: str,
        business_context: Dict[str, Any],
        max_models: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Complete workflow: select optimal models and apply them to the problem
        """
        try:
            session_id = f"session_{datetime.utcnow().timestamp()}"
            self.active_processing_sessions[session_id] = {
                "start_time": datetime.utcnow(),
                "criteria": criteria,
                "status": "selecting_models",
            }

            self.logger.info(
                f"🚀 Starting complete model workflow for session {session_id}"
            )

            # Step 1: Model selection
            selected_models = await self.selector.select_optimal_models(
                criteria=criteria,
                business_context=business_context,
                max_models=max_models,
            )

            if not selected_models:
                self.logger.warning("⚠️ No models selected, using fallback")
                selected_models = self._get_emergency_fallback_models()

            self.active_processing_sessions[session_id]["status"] = "applying_models"
            self.active_processing_sessions[session_id]["selected_models"] = [
                m.model_id for m in selected_models
            ]

            # Step 2: Model application
            results = await self.apply_selected_models(
                models=selected_models,
                problem_statement=problem_statement,
                business_context=business_context,
                session_id=session_id,
            )

            # Step 3: Performance tracking
            await self._track_session_performance(
                session_id, results, criteria, business_context
            )

            # Clean up session
            self.active_processing_sessions[session_id]["status"] = "completed"
            self.active_processing_sessions[session_id]["end_time"] = datetime.utcnow()

            self.logger.info(
                f"✅ Completed model workflow for session {session_id}: {len(results)} results"
            )
            return results

        except Exception as e:
            self.logger.error(f"❌ Model workflow failed: {e}")
            # Return emergency fallback result
            return await self._emergency_fallback_processing(
                problem_statement, business_context
            )

    async def apply_selected_models(
        self,
        models: List[MentalModelDefinition],
        problem_statement: str,
        business_context: Dict[str, Any],
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Apply a list of pre-selected models to the problem
        """
        try:
            results = []

            # Process models concurrently for better performance
            if (
                len(models) > 1
                and self.configurator.model_settings.enable_nway_patterns
            ):
                # Parallel processing for multiple models
                tasks = []
                for model in models:
                    task = self.apply_single_model(
                        model=model,
                        problem_statement=problem_statement,
                        business_context=business_context,
                        session_id=session_id,
                    )
                    tasks.append(task)

                # Wait for all models to complete
                parallel_results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process results and handle exceptions
                for i, result in enumerate(parallel_results):
                    if isinstance(result, Exception):
                        self.logger.error(
                            f"❌ Model {models[i].model_id} failed: {result}"
                        )
                        # Create error result
                        results.append(
                            {
                                "model_id": models[i].model_id,
                                "error": str(result),
                                "timestamp": datetime.utcnow().isoformat(),
                                "success": False,
                            }
                        )
                    else:
                        results.append(result)

            else:
                # Sequential processing
                for model in models:
                    try:
                        result = await self.apply_single_model(
                            model=model,
                            problem_statement=problem_statement,
                            business_context=business_context,
                            session_id=session_id,
                        )
                        results.append(result)

                    except Exception as e:
                        self.logger.error(
                            f"❌ Model {model.model_id} application failed: {e}"
                        )
                        results.append(
                            {
                                "model_id": model.model_id,
                                "error": str(e),
                                "timestamp": datetime.utcnow().isoformat(),
                                "success": False,
                            }
                        )

            # Filter successful results
            successful_results = [r for r in results if r.get("success", False)]
            self.logger.info(
                f"📊 Applied {len(models)} models: {len(successful_results)} successful, {len(results) - len(successful_results)} failed"
            )

            return results

        except Exception as e:
            self.logger.error(f"❌ Model application batch failed: {e}")
            return []

    async def apply_single_model(
        self,
        model: MentalModelDefinition,
        problem_statement: str,
        business_context: Dict[str, Any],
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Apply a single model to the problem with performance tracking
        """
        try:
            step_id = f"step_{datetime.utcnow().timestamp()}"

            # Track model usage
            self.current_models_in_use[model.model_id] = {
                "start_time": datetime.utcnow(),
                "session_id": session_id,
                "step_id": step_id,
            }

            # Apply the model
            result = await self.applicator.apply_model(
                model=model,
                problem_statement=problem_statement,
                business_context=business_context,
                step_id=step_id,
            )

            # Apply confidence calibration if enabled
            if self.configurator.model_settings.enable_confidence_calibration:
                if "confidence_score" in result:
                    calibrated_confidence = (
                        await self.tracker.calculate_confidence_calibration(
                            model_id=model.model_id,
                            predicted_confidence=result["confidence_score"],
                        )
                    )
                    result["calibrated_confidence"] = calibrated_confidence

            # Clean up tracking
            self.current_models_in_use.pop(model.model_id, None)

            self.logger.debug(f"✅ Applied model {model.model_id} successfully")
            return result

        except Exception as e:
            self.logger.error(
                f"❌ Single model application failed for {model.model_id}: {e}"
            )
            # Clean up tracking
            self.current_models_in_use.pop(model.model_id, None)

            return {
                "model_id": model.model_id,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
                "success": False,
            }

    async def _track_session_performance(
        self,
        session_id: str,
        results: List[Dict[str, Any]],
        criteria: "ModelSelectionCriteria",
        business_context: Dict[str, Any],
    ) -> None:
        """
        Track performance for all models in the session
        """
        try:
            for result in results:
                if not result.get("success", False):
                    continue

                model_id = result.get("model_id")
                if not model_id:
                    continue

                # Calculate performance score based on result quality
                performance_score = self._calculate_result_quality(result)

                # Update model performance
                await self.tracker.update_model_performance(
                    model_id=model_id,
                    performance_score=performance_score,
                    business_context=business_context,
                    problem_statement=business_context.get("problem_statement", ""),
                )

            self.logger.debug(
                f"📈 Updated performance tracking for session {session_id}"
            )

        except Exception as e:
            self.logger.error(
                f"❌ Performance tracking failed for session {session_id}: {e}"
            )

    def _calculate_result_quality(self, result: Dict[str, Any]) -> float:
        """
        Calculate quality score for a model result
        """
        try:
            score = 0.5  # Base score

            # Content quality indicators
            content = result.get("content", "")
            if len(content) > 100:  # Substantial content
                score += 0.2

            if len(content) > 500:  # Comprehensive content
                score += 0.1

            # Confidence score factor
            if "confidence_score" in result:
                confidence = result["confidence_score"]
                score += confidence * 0.3

            # Reasoning quality
            if "reasoning" in result and result["reasoning"]:
                score += 0.1

            # Error indicators
            if "error" in result or "warning" in result:
                score -= 0.2

            # Time factor (penalize very slow responses)
            processing_time = result.get("processing_time_seconds", 30)
            if processing_time > 120:  # Over 2 minutes
                score -= 0.1

            return max(0.1, min(1.0, score))

        except Exception as e:
            self.logger.warning(f"⚠️ Quality calculation failed: {e}")
            return 0.5

    def _get_emergency_fallback_models(self) -> List[MentalModelDefinition]:
        """
        Get basic fallback models when selection fails
        """
        try:
            available_models = self.configurator.get_available_models()

            # Try to get basic problem-solving models
            fallback_ids = [
                "systems_thinking",
                "critical_thinking",
                "problem_solving",
                "analytical_thinking",
            ]
            fallback_models = []

            for model_id in fallback_ids:
                if model_id in available_models:
                    fallback_models.append(available_models[model_id])
                    if len(fallback_models) >= 2:  # Get at least 2 models
                        break

            # If still no models, take any available
            if not fallback_models and available_models:
                fallback_models = list(available_models.values())[:2]

            self.logger.warning(
                f"🚨 Using {len(fallback_models)} emergency fallback models"
            )
            return fallback_models

        except Exception as e:
            self.logger.error(f"❌ Emergency fallback failed: {e}")
            return []

    async def _emergency_fallback_processing(
        self, problem_statement: str, business_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Emergency fallback when everything fails
        """
        try:
            self.logger.warning("🚨 Executing emergency fallback processing")

            return [
                {
                    "model_id": "emergency_fallback",
                    "content": f"I understand you're asking about: {problem_statement[:200]}...\n\nUnfortunately, I'm experiencing technical difficulties with my advanced reasoning models. Please try again in a moment, or contact support if the issue persists.",
                    "timestamp": datetime.utcnow().isoformat(),
                    "success": True,
                    "confidence_score": 0.3,
                    "is_fallback": True,
                    "reasoning": "Emergency fallback response due to system issues",
                }
            ]

        except Exception as e:
            self.logger.error(f"❌ Emergency fallback processing failed: {e}")
            return [
                {
                    "model_id": "system_error",
                    "content": "System error occurred. Please try again later.",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                    "success": False,
                }
            ]

    # Public utility methods

    def get_available_models(self) -> Dict[str, MentalModelDefinition]:
        """Get currently available models"""
        return self.configurator.get_available_models()

    async def get_performance_analytics(self) -> Dict[str, Any]:
        """Get comprehensive performance analytics"""
        return await self.tracker.get_performance_analytics()

    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get current configuration summary"""
        return self.configurator.get_configuration_summary()

    def update_performance_history(
        self, model_id: str, performance_score: float
    ) -> None:
        """Update model performance (legacy compatibility)"""
        asyncio.create_task(
            self.tracker.update_model_performance(
                model_id=model_id,
                performance_score=performance_score,
                business_context={},
                problem_statement="",
            )
        )

    async def health_check(self) -> Dict[str, Any]:
        """Check health of all components"""
        try:
            health_status = {
                "unified_manager": "healthy",
                "configurator": "healthy" if self.configurator else "missing",
                "selector": "healthy" if self.selector else "missing",
                "applicator": "healthy" if self.applicator else "missing",
                "tracker": "healthy" if self.tracker else "missing",
                "available_models": len(self.get_available_models()),
                "active_sessions": len(self.active_processing_sessions),
                "models_in_use": len(self.current_models_in_use),
                "timestamp": datetime.utcnow().isoformat(),
            }

            # Validate configuration
            config_validation = self.configurator.validate_configuration()
            health_status["configuration_valid"] = config_validation["valid"]

            if not config_validation["valid"]:
                health_status["configuration_issues"] = config_validation["issues"]

            overall_health = "healthy"
            if health_status["available_models"] == 0:
                overall_health = "degraded"

            missing_components = [k for k, v in health_status.items() if v == "missing"]
            if missing_components:
                overall_health = "degraded"

            health_status["overall"] = overall_health

            return health_status

        except Exception as e:
            self.logger.error(f"❌ Health check failed: {e}")
            return {
                "overall": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }
