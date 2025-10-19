"""
Integration Orchestrator Interface Module

This module defines the abstract interfaces for integration orchestration functionality,
including service registry patterns and external service coordination.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, TypeVar, Generic
from enum import Enum

# Type variables for generic service registry
ServiceType = TypeVar("ServiceType")


class ServiceStatus(Enum):
    """Service status enumeration"""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    INITIALIZING = "initializing"
    ERROR = "error"


class ServiceInfo:
    """Information about a registered service"""

    def __init__(
        self,
        service_name: str,
        service_instance: Any,
        status: ServiceStatus,
        initialization_error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.service_name = service_name
        self.service_instance = service_instance
        self.status = status
        self.initialization_error = initialization_error
        self.metadata = metadata or {}


class IServiceRegistry(ABC, Generic[ServiceType]):
    """
    Abstract interface for service registry implementing service locator pattern
    """

    @abstractmethod
    def register_service(
        self,
        service_name: str,
        service_instance: ServiceType,
        status: ServiceStatus = ServiceStatus.AVAILABLE,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Register a service instance with the registry

        Args:
            service_name: Unique name for the service
            service_instance: The service instance to register
            status: Initial status of the service
            metadata: Additional metadata about the service
        """
        pass

    @abstractmethod
    def get_service(self, service_name: str) -> Optional[ServiceType]:
        """
        Retrieve a service instance by name

        Args:
            service_name: Name of the service to retrieve

        Returns:
            Service instance if available and healthy, None otherwise
        """
        pass

    @abstractmethod
    def get_service_info(self, service_name: str) -> Optional[ServiceInfo]:
        """
        Get detailed information about a service

        Args:
            service_name: Name of the service

        Returns:
            ServiceInfo object with status and metadata
        """
        pass

    @abstractmethod
    def is_service_available(self, service_name: str) -> bool:
        """
        Check if a service is available and healthy

        Args:
            service_name: Name of the service to check

        Returns:
            True if service is available, False otherwise
        """
        pass

    @abstractmethod
    def get_available_services(self) -> List[str]:
        """
        Get list of all available service names

        Returns:
            List of service names that are currently available
        """
        pass

    @abstractmethod
    def update_service_status(
        self, service_name: str, status: ServiceStatus, error: Optional[str] = None
    ) -> None:
        """
        Update the status of a registered service

        Args:
            service_name: Name of the service
            status: New status
            error: Error message if status is ERROR
        """
        pass


class IIntegrationOrchestrator(ABC):
    """
    Abstract interface for integration orchestration functionality
    """

    @property
    @abstractmethod
    def service_registry(self) -> IServiceRegistry:
        """Get the service registry instance"""
        pass

    @abstractmethod
    async def initialize_all_integrations(self) -> Dict[str, bool]:
        """
        Initialize all available integrations

        Returns:
            Dictionary mapping service names to initialization success status
        """
        pass

    @abstractmethod
    async def initialize_dual_intelligence(self) -> bool:
        """
        Initialize Dual Intelligence Orchestrator for RAG + Supabase integration

        Returns:
            True if initialization successful, False otherwise
        """
        pass

    @abstractmethod
    async def initialize_pattern_detector(self) -> bool:
        """
        Initialize Emergent Pattern Detector for cross-engagement analysis

        Returns:
            True if initialization successful, False otherwise
        """
        pass

    @abstractmethod
    async def initialize_nway_patterns(self) -> bool:
        """
        Initialize N-Way patterns from cognitive engine's database

        Returns:
            True if patterns loaded successfully, False otherwise
        """
        pass

    @abstractmethod
    async def conduct_problem_research(
        self,
        problem_statement: str,
        business_context: Dict[str, Any],
        problem_analysis: Any,  # ModelSelectionCriteria type
    ) -> Optional[Any]:  # Optional[SynthesizedIntelligence] type
        """
        Conduct research intelligence to enhance problem understanding

        Args:
            problem_statement: The problem to research
            business_context: Business context for the problem
            problem_analysis: Analysis criteria for the problem

        Returns:
            Synthesized intelligence if successful, None otherwise
        """
        pass

    @abstractmethod
    def enhance_reasoning_with_research(
        self,
        reasoning_results: List[Any],  # List[ReasoningStep] type
        research_intelligence: Optional[Any],  # Optional[SynthesizedIntelligence] type
    ) -> List[Any]:  # List[ReasoningStep] type
        """
        Enhance reasoning steps with research intelligence

        Args:
            reasoning_results: List of reasoning steps to enhance
            research_intelligence: Research intelligence to integrate

        Returns:
            Enhanced reasoning steps with research context
        """
        pass

    @abstractmethod
    async def apply_ai_augmentation(
        self,
        problem_statement: str,
        business_context: Dict[str, Any],
        reasoning_steps: List[Any],  # List[ReasoningStep] type
    ) -> Dict[str, Any]:
        """
        Apply AI Augmentation Engine for bias detection and intellectual honesty

        Args:
            problem_statement: The problem being analyzed
            business_context: Business context information
            reasoning_steps: Reasoning steps to augment

        Returns:
            Augmentation results with bias detection and honesty scores
        """
        pass

    @abstractmethod
    def get_integration_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get comprehensive status of all integrations

        Returns:
            Dictionary with status information for each integration
        """
        pass

    @abstractmethod
    def calculate_honesty_score(self, augmentation_result: Dict[str, Any]) -> float:
        """
        Calculate intellectual honesty score from augmentation results

        Args:
            augmentation_result: Results from AI augmentation analysis

        Returns:
            Honesty score between 0.0 and 1.0
        """
        pass

    @abstractmethod
    def extract_bias_flags(self, augmentation_result: Dict[str, Any]) -> List[str]:
        """
        Extract bias warning flags from augmentation analysis

        Args:
            augmentation_result: Results from AI augmentation analysis

        Returns:
            List of bias warning flags detected
        """
        pass
