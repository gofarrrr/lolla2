"""
Model Catalog and Selector Interfaces for Dependency Injection
"""

from typing import Dict, List, Protocol, Any
from src.engine.models.data_contracts import MentalModelDefinition


class IModelCatalog(Protocol):
    """Interface for mental model catalog"""

    @property
    def models(self) -> Dict[str, MentalModelDefinition]:
        """Get all available models"""
        ...

    def get_model(self, model_id: str) -> MentalModelDefinition:
        """Get specific model by ID"""
        ...

    def get_models_by_category(self, category: str) -> List[MentalModelDefinition]:
        """Get models by category"""
        ...


class IModelSelector(Protocol):
    """Interface for model selection logic"""

    def select_models(self, selection_context: Any) -> Any:
        """Select optimal models based on context"""
        ...
