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
from .context_stream_interface import ContextStream
from .pipeline_orchestrator_interface import PipelineOrchestrator
from .llm_manager_interface import LLMManagerInterface
from .context_metrics import ContextMetrics, ContextMetricsAdapter
from .evidence import EvidenceExtractor, EvidenceExtractionAdapter

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
    "ContextStream",
    "PipelineOrchestrator",
    "LLMManagerInterface",
    "ContextMetrics",
    "ContextMetricsAdapter",
    "EvidenceExtractor",
    "EvidenceExtractionAdapter",
]
