"""
DifferentialAnalyzer Service Interface
====================================

Interface definition for DifferentialAnalyzer domain service.
This interface establishes the contract for differential analysis functionality
while enabling multiple implementations and testing flexibility.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from src.arbitration.models import (
    ConsultantOutput,
    DifferentialAnalysis,
    UniqueInsight,
    ConvergentFinding,
    PerspectiveDifference,
    SynergyOpportunity,
)


class IDifferentialAnalyzer(ABC):
    """
    Interface for DifferentialAnalyzer domain service

    This interface defines the contract for differential analysis coordination,
    enabling clean separation between orchestration and domain logic.
    """

    @abstractmethod
    async def analyze_consultant_outputs(
        self,
        consultant_outputs: List[ConsultantOutput],
        original_query: str,
    ) -> DifferentialAnalysis:
        """
        Main analysis method - creates comprehensive differential analysis

        Args:
            consultant_outputs: List of independent consultant analyses
            original_query: Original user query for context

        Returns:
            DifferentialAnalysis: Complete comparative breakdown

        Raises:
            DifferentialAnalysisError: If analysis generation fails
        """
        pass

    @abstractmethod
    async def identify_unique_insights(
        self,
        consultant_outputs: List[ConsultantOutput],
    ) -> List[UniqueInsight]:
        """
        Identify insights that are unique to individual consultants

        Args:
            consultant_outputs: List of consultant analyses

        Returns:
            List[UniqueInsight]: Insights unique to specific consultants

        Raises:
            DifferentialAnalysisError: If unique insight identification fails
        """
        pass

    @abstractmethod
    async def find_convergent_areas(
        self,
        consultant_outputs: List[ConsultantOutput],
    ) -> List[ConvergentFinding]:
        """
        Find areas where consultants converge or agree

        Args:
            consultant_outputs: List of consultant analyses

        Returns:
            List[ConvergentFinding]: Areas of consultant agreement

        Raises:
            DifferentialAnalysisError: If convergent analysis fails
        """
        pass

    @abstractmethod
    async def analyze_perspective_differences(
        self,
        consultant_outputs: List[ConsultantOutput],
    ) -> List[PerspectiveDifference]:
        """
        Analyze how consultants differ in their perspectives

        Args:
            consultant_outputs: List of consultant analyses

        Returns:
            List[PerspectiveDifference]: Perspective variations between consultants

        Raises:
            DifferentialAnalysisError: If perspective analysis fails
        """
        pass

    @abstractmethod
    async def identify_synergy_opportunities(
        self,
        consultant_outputs: List[ConsultantOutput],
        unique_insights: List[UniqueInsight],
        convergent_findings: List[ConvergentFinding],
    ) -> List[SynergyOpportunity]:
        """
        Identify opportunities for synergistic combinations

        Args:
            consultant_outputs: List of consultant analyses
            unique_insights: Previously identified unique insights
            convergent_findings: Previously identified convergent findings

        Returns:
            List[SynergyOpportunity]: Potential synergistic combinations

        Raises:
            DifferentialAnalysisError: If synergy identification fails
        """
        pass

    @abstractmethod
    def calculate_complementarity_score(
        self,
        consultant_outputs: List[ConsultantOutput],
        unique_insights: List[UniqueInsight],
        convergent_findings: List[ConvergentFinding],
    ) -> float:
        """
        Calculate how well consultants complement each other

        Args:
            consultant_outputs: List of consultant analyses
            unique_insights: Identified unique insights
            convergent_findings: Identified convergent findings

        Returns:
            float: Complementarity score (0.0-1.0)
        """
        pass


class DifferentialAnalysisError(Exception):
    """Exception for differential analysis related errors"""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


# Factory pattern for dependency injection
class DifferentialAnalyzerFactory:
    """Factory for creating DifferentialAnalyzer instances"""

    def create_differential_analyzer(self) -> IDifferentialAnalyzer:
        """Create default DifferentialAnalyzer instance"""
        from src.services.differential_analyzer import DifferentialAnalyzer
        return DifferentialAnalyzer()


class MockDifferentialAnalyzerFactory:
    """Mock factory for testing purposes"""

    def create_differential_analyzer(self) -> IDifferentialAnalyzer:
        """Create mock DifferentialAnalyzer instance for testing"""
        from tests.mocks.mock_differential_analyzer import MockDifferentialAnalyzer
        return MockDifferentialAnalyzer()


# Convenience function for creating default instance
def create_differential_analyzer() -> IDifferentialAnalyzer:
    """
    Convenience function to create a DifferentialAnalyzer instance

    Returns:
        IDifferentialAnalyzer: Configured instance
    """
    factory = DifferentialAnalyzerFactory()
    return factory.create_differential_analyzer()