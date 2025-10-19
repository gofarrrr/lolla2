"""
Model Manager Interfaces for Dependency Injection
Defines abstract base classes for model management functionality extracted from cognitive engine
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

from src.engine.models.data_contracts import MentalModelDefinition

# Note: ModelSelectionCriteria is defined in cognitive_engine.py
# This will need to be moved to data_contracts.py or imported from there
try:
    from src.engine.engines.cognitive_engine import ModelSelectionCriteria
except ImportError:
    # Fallback definition if not available
    class ModelSelectionCriteria:
        """Criteria for selecting appropriate mental models"""

        def __init__(self):
            self.problem_type: str = ""
            self.complexity_level: str = "medium"
            self.time_constraint: Optional[int] = None
            self.accuracy_requirement: float = 0.8
            self.business_context: Dict[str, Any] = {}


class IModelManager(ABC):
    """Abstract interface for model management functionality"""

    @abstractmethod
    def get_available_models(self) -> List[MentalModelDefinition]:
        """Get available models from catalog"""
        pass

    @abstractmethod
    async def select_optimal_models(
        self,
        selection_criteria: "ModelSelectionCriteria",
        research_intelligence: Optional[Any] = None,  # SynthesizedIntelligence
        problem_statement: str = "",
    ) -> List[MentalModelDefinition]:
        """Select optimal mental models with N-WAY pattern recognition and research enhancement"""
        pass

    @abstractmethod
    async def apply_selected_models(
        self,
        models: List[MentalModelDefinition],
        problem_statement: str,
        business_context: Dict[str, Any],
        research_intelligence: Optional[Any] = None,  # SynthesizedIntelligence
    ) -> List[Dict[str, Any]]:
        """Apply selected mental models to generate reasoning"""
        pass

    @abstractmethod
    async def apply_single_model(
        self,
        model: MentalModelDefinition,
        problem_statement: str,
        business_context: Dict[str, Any],
        step_id: str,
    ) -> Dict[str, Any]:
        """Apply single mental model to generate reasoning step"""
        pass

    @abstractmethod
    async def get_model_performance_history(self, model_id: str) -> List[float]:
        """Get performance history for specific model"""
        pass

    @abstractmethod
    async def update_model_performance(self, model_id: str, performance_score: float):
        """Update model performance based on feedback"""
        pass


class IModelApplicationStrategy(ABC):
    """Abstract interface for model-specific application strategies"""

    @abstractmethod
    async def apply_systems_thinking(
        self, problem_statement: str, business_context: Dict[str, Any], step_id: str
    ) -> Dict[str, Any]:
        """Apply systems thinking mental model with N-WAY enhanced prompting"""
        pass

    @abstractmethod
    async def apply_critical_thinking(
        self, problem_statement: str, business_context: Dict[str, Any], step_id: str
    ) -> Dict[str, Any]:
        """Apply critical thinking mental model"""
        pass

    @abstractmethod
    async def apply_mece_structuring(
        self, problem_statement: str, business_context: Dict[str, Any], step_id: str
    ) -> Dict[str, Any]:
        """Apply MECE structuring mental model"""
        pass

    @abstractmethod
    async def apply_hypothesis_testing(
        self, problem_statement: str, business_context: Dict[str, Any], step_id: str
    ) -> Dict[str, Any]:
        """Apply hypothesis testing mental model"""
        pass

    @abstractmethod
    async def apply_decision_framework(
        self, problem_statement: str, business_context: Dict[str, Any], step_id: str
    ) -> Dict[str, Any]:
        """Apply decision framework mental model"""
        pass

    @abstractmethod
    async def apply_generic_model(
        self,
        model: MentalModelDefinition,
        problem_statement: str,
        business_context: Dict[str, Any],
        step_id: str,
    ) -> Dict[str, Any]:
        """Generic model application for new or undefined models"""
        pass


class IModelPerformanceTracker(ABC):
    """Abstract interface for model performance tracking"""

    @abstractmethod
    async def record_model_performance(
        self, model_id: str, performance_score: float, context: Dict[str, Any]
    ) -> None:
        """Record performance data for a model application"""
        pass

    @abstractmethod
    async def get_performance_statistics(self, model_id: str) -> Dict[str, Any]:
        """Get performance statistics for a model"""
        pass

    @abstractmethod
    async def update_bayesian_effectiveness(
        self, model_id: str, effectiveness_score: float, context: Dict[str, Any]
    ) -> None:
        """Update Bayesian effectiveness tracking if enabled"""
        pass
