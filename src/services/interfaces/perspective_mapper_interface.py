"""
PerspectiveMapperService Interface
=================================

Interface definition for PerspectiveMapperService domain service.
This interface establishes the contract for perspective mapping functionality
while enabling multiple implementations and testing flexibility.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from src.arbitration.models import (
    ConsultantOutput,
    ConsultantRole,
    PerspectiveType,
)


class IPerspectiveMapperService(ABC):
    """
    Interface for PerspectiveMapperService domain service

    This interface defines the contract for perspective mapping coordination,
    enabling clean separation between orchestration and domain logic.
    """

    @abstractmethod
    def map_consultant_perspectives(
        self,
        consultant_outputs: List[ConsultantOutput],
    ) -> Dict[str, Any]:
        """
        Map cognitive approaches and mental models used by consultants

        Args:
            consultant_outputs: List of consultant analyses

        Returns:
            Dict[str, Any]: Comprehensive perspective mapping

        Raises:
            PerspectiveMappingError: If perspective mapping fails
        """
        pass

    @abstractmethod
    def identify_cognitive_approaches(
        self,
        consultant_outputs: List[ConsultantOutput],
    ) -> Dict[ConsultantRole, List[str]]:
        """
        Identify cognitive approaches used by each consultant

        Args:
            consultant_outputs: List of consultant analyses

        Returns:
            Dict[ConsultantRole, List[str]]: Cognitive approaches per consultant

        Raises:
            PerspectiveMappingError: If cognitive approach identification fails
        """
        pass

    @abstractmethod
    def extract_mental_models(
        self,
        consultant_outputs: List[ConsultantOutput],
    ) -> Dict[ConsultantRole, List[str]]:
        """
        Extract mental models and frameworks used by consultants

        Args:
            consultant_outputs: List of consultant analyses

        Returns:
            Dict[ConsultantRole, List[str]]: Mental models per consultant

        Raises:
            PerspectiveMappingError: If mental model extraction fails
        """
        pass

    @abstractmethod
    def analyze_perspective_patterns(
        self,
        consultant_outputs: List[ConsultantOutput],
    ) -> Dict[str, Any]:
        """
        Analyze patterns in how consultants approach problems

        Args:
            consultant_outputs: List of consultant analyses

        Returns:
            Dict[str, Any]: Perspective pattern analysis

        Raises:
            PerspectiveMappingError: If pattern analysis fails
        """
        pass

    @abstractmethod
    def map_cognitive_diversity(
        self,
        consultant_outputs: List[ConsultantOutput],
    ) -> Dict[str, float]:
        """
        Map cognitive diversity across the consultant team

        Args:
            consultant_outputs: List of consultant analyses

        Returns:
            Dict[str, float]: Cognitive diversity metrics

        Raises:
            PerspectiveMappingError: If diversity mapping fails
        """
        pass

    @abstractmethod
    def identify_perspective_gaps(
        self,
        consultant_outputs: List[ConsultantOutput],
        target_perspectives: Optional[List[PerspectiveType]] = None,
    ) -> List[PerspectiveType]:
        """
        Identify gaps in perspective coverage

        Args:
            consultant_outputs: List of consultant analyses
            target_perspectives: Optional list of target perspectives to check

        Returns:
            List[PerspectiveType]: Missing or underrepresented perspectives

        Raises:
            PerspectiveMappingError: If gap identification fails
        """
        pass


class PerspectiveMappingError(Exception):
    """Exception for perspective mapping related errors"""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


# Factory pattern for dependency injection
class PerspectiveMapperServiceFactory:
    """Factory for creating PerspectiveMapperService instances"""

    def create_perspective_mapper_service(self) -> IPerspectiveMapperService:
        """Create default PerspectiveMapperService instance"""
        from src.services.perspective_mapper_service import PerspectiveMapperService
        return PerspectiveMapperService()


class MockPerspectiveMapperServiceFactory:
    """Mock factory for testing purposes"""

    def create_perspective_mapper_service(self) -> IPerspectiveMapperService:
        """Create mock PerspectiveMapperService instance for testing"""
        from tests.mocks.mock_perspective_mapper_service import MockPerspectiveMapperService
        return MockPerspectiveMapperService()


# Convenience function for creating default instance
def create_perspective_mapper_service() -> IPerspectiveMapperService:
    """
    Convenience function to create a PerspectiveMapperService instance

    Returns:
        IPerspectiveMapperService: Configured instance
    """
    factory = PerspectiveMapperServiceFactory()
    return factory.create_perspective_mapper_service()