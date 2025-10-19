"""
METIS Interfaces and Protocols
Dependency injection interfaces for decoupled architecture
"""

from .database_interfaces import INwayManager
from .event_interfaces import IEventBus
from .model_interfaces import IModelCatalog, IModelSelector
from .workflow_interfaces import IWorkflowEngine
from .problem_analyzer_interface import IProblemAnalyzer, ModelSelectionCriteria
from .integration_orchestrator_interface import (
    IIntegrationOrchestrator,
    IServiceRegistry,
    ServiceInfo,
    ServiceStatus,
)

__all__ = [
    "INwayManager",
    "IEventBus",
    "IModelCatalog",
    "IModelSelector",
    "IWorkflowEngine",
    "IProblemAnalyzer",
    "ModelSelectionCriteria",
    "IIntegrationOrchestrator",
    "IServiceRegistry",
    "ServiceInfo",
    "ServiceStatus",
]
