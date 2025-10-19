"""
Backward Compatibility Wrapper for model_manager.py
Maintains existing API while redirecting to new modular components
"""

import logging
from typing import Dict, List, Optional, Any

# Import new modular components
from src.engine.engines.model_management import UnifiedModelManager
from src.engine.engines.model_selection import ModelSelector
from src.engine.engines.model_application.model_applicator import ModelApplicationEngine
from src.engine.engines.model_performance import ModelPerformanceTracker
from src.engine.engines.model_config import ModelConfigurator

# Import original interfaces
from src.interfaces.model_manager_interface import (
    IModelManager,
    IModelApplicationStrategy,
)
from src.engine.models.data_contracts import MentalModelDefinition
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


class ModelManager(IModelManager):
    """
    BACKWARD COMPATIBILITY WRAPPER

    This class maintains the original ModelManager interface while delegating
    all functionality to the new modular components.

    ⚠️ DEPRECATED: Use UnifiedModelManager directly for new code
    """

    def __init__(
        self,
        model_catalog: MentalModelCatalog,
        llm_provider=None,
        vector_similarity_engine=None,
        supabase_platform=None,
        logger=None,
        settings=None,
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.logger.warning(
            "⚠️ ModelManager is deprecated. Use UnifiedModelManager instead."
        )

        # Initialize the new unified manager
        self._unified_manager = UnifiedModelManager(
            model_catalog=model_catalog,
            llm_provider=llm_provider,
            vector_similarity_engine=vector_similarity_engine,
            supabase_platform=supabase_platform,
            settings=settings,
            logger=self.logger,
        )

        # Legacy properties for backward compatibility
        self.model_catalog = model_catalog
        self.performance_history = {}  # Will be synced with tracker
        self.settings = settings

    # BACKWARD COMPATIBILITY METHODS

    async def select_optimal_models(
        self,
        criteria: "ModelSelectionCriteria",
        business_context: Dict[str, Any],
        max_models: int = 3,
    ) -> List[MentalModelDefinition]:
        """Legacy compatibility for model selection"""
        return await self._unified_manager.selector.select_optimal_models(
            criteria=criteria, business_context=business_context, max_models=max_models
        )

    async def apply_selected_models(
        self,
        models: List[MentalModelDefinition],
        problem_statement: str,
        business_context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Legacy compatibility for model application"""
        return await self._unified_manager.apply_selected_models(
            models=models,
            problem_statement=problem_statement,
            business_context=business_context,
        )

    async def apply_single_model(
        self,
        model: MentalModelDefinition,
        problem_statement: str,
        business_context: Dict[str, Any],
        step_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Legacy compatibility for single model application"""
        return await self._unified_manager.apply_single_model(
            model=model,
            problem_statement=problem_statement,
            business_context=business_context,
            session_id=step_id,
        )

    def update_performance_history(self, model_id: str, performance_score: float):
        """Legacy compatibility for performance updates"""
        # Update local cache for backward compatibility
        if model_id not in self.performance_history:
            self.performance_history[model_id] = []
        self.performance_history[model_id].append(performance_score)

        # Update new tracker
        self._unified_manager.update_performance_history(model_id, performance_score)

    def get_available_models(self) -> Dict[str, MentalModelDefinition]:
        """Legacy compatibility for getting available models"""
        return self._unified_manager.get_available_models()


class ModelApplicationStrategy(IModelApplicationStrategy):
    """
    BACKWARD COMPATIBILITY WRAPPER

    This class maintains the original ModelApplicationStrategy interface
    while delegating to the new ModelApplicationEngine.

    ⚠️ DEPRECATED: Use ModelApplicationEngine directly for new code
    """

    def __init__(
        self,
        llm_provider=None,
        model_catalog: Optional[MentalModelCatalog] = None,
        logger=None,
        settings=None,
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.logger.warning(
            "⚠️ ModelApplicationStrategy is deprecated. Use ModelApplicationEngine instead."
        )

        # Initialize the new application engine
        self._application_engine = ModelApplicationEngine(
            llm_provider=llm_provider,
            model_catalog=model_catalog,
            logger=self.logger,
            settings=settings,
        )

    async def apply_model(
        self,
        model: MentalModelDefinition,
        problem_statement: str,
        business_context: Dict[str, Any],
        step_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Legacy compatibility for model application"""
        return await self._application_engine.apply_model(
            model=model,
            problem_statement=problem_statement,
            business_context=business_context,
            step_id=step_id,
        )

    # Delegate all specific model methods

    async def apply_systems_thinking(
        self, problem_statement: str, business_context: Dict[str, Any], step_id: str
    ) -> Dict[str, Any]:
        return await self._application_engine.apply_systems_thinking(
            problem_statement, business_context, step_id
        )

    async def apply_critical_thinking(
        self, problem_statement: str, business_context: Dict[str, Any], step_id: str
    ) -> Dict[str, Any]:
        return await self._application_engine.apply_critical_thinking(
            problem_statement, business_context, step_id
        )

    async def apply_mece_structuring(
        self, problem_statement: str, business_context: Dict[str, Any], step_id: str
    ) -> Dict[str, Any]:
        return await self._application_engine.apply_mece_structuring(
            problem_statement, business_context, step_id
        )

    async def apply_hypothesis_testing(
        self, problem_statement: str, business_context: Dict[str, Any], step_id: str
    ) -> Dict[str, Any]:
        return await self._application_engine.apply_hypothesis_testing(
            problem_statement, business_context, step_id
        )

    async def apply_decision_framework(
        self, problem_statement: str, business_context: Dict[str, Any], step_id: str
    ) -> Dict[str, Any]:
        return await self._application_engine.apply_decision_framework(
            problem_statement, business_context, step_id
        )


# Factory function for backward compatibility
def create_model_manager(
    model_catalog: MentalModelCatalog,
    llm_provider=None,
    vector_similarity_engine=None,
    supabase_platform=None,
    logger=None,
    settings=None,
) -> ModelManager:
    """
    Factory function to create ModelManager with backward compatibility

    ⚠️ DEPRECATED: Use UnifiedModelManager directly for new code
    """
    if logger:
        logger.warning(
            "⚠️ create_model_manager is deprecated. Use UnifiedModelManager directly."
        )

    return ModelManager(
        model_catalog=model_catalog,
        llm_provider=llm_provider,
        vector_similarity_engine=vector_similarity_engine,
        supabase_platform=supabase_platform,
        logger=logger,
        settings=settings,
    )


# Export original class names for compatibility
__all__ = [
    "ModelManager",
    "ModelApplicationStrategy",
    "create_model_manager",
    # Also export new classes for migration
    "UnifiedModelManager",
    "ModelSelector",
    "ModelApplicationEngine",
    "ModelPerformanceTracker",
    "ModelConfigurator",
]
