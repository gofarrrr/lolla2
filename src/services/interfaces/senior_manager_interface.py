"""
Senior Manager Service Interface
==============================

Interface definition for Senior Manager domain service.
This interface establishes the contract for Two-Brain strategic synthesis
while enabling multiple implementations and testing flexibility.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Protocol

from src.orchestration.contracts import (
    ConsultantAnalysisResult,
    AnalysisCritique,
    SeniorAdvisorReport,
    TwoBrainInsight,
)


class ISeniorManager(ABC):
    """
    Interface for Senior Manager domain service

    This interface defines the contract for Two-Brain strategic synthesis,
    enabling clean separation between orchestration and domain logic.
    """

    @abstractmethod
    async def initialize_clients(self) -> None:
        """
        Initialize LLM clients for two-brain synthesis

        Raises:
            SeniorAdvisorError: If client initialization fails
        """
        pass

    @abstractmethod
    async def synthesize_strategic_report(
        self,
        analyses: List[ConsultantAnalysisResult],
        critiques: List[AnalysisCritique],
    ) -> SeniorAdvisorReport:
        """
        Main entry point for Two-Brain strategic synthesis

        Args:
            analyses: Results from consultant analyses
            critiques: Results from devil's advocate critiques

        Returns:
            SeniorAdvisorReport: Comprehensive strategic report

        Raises:
            SeniorAdvisorError: If synthesis fails
        """
        pass

    @abstractmethod
    async def execute_analytical_brain(
        self,
        analyses: List[ConsultantAnalysisResult],
        critiques: List[AnalysisCritique],
    ) -> TwoBrainInsight:
        """
        Execute analytical brain (DeepSeek) for data synthesis

        Args:
            analyses: Consultant analysis results
            critiques: Devil's advocate critiques

        Returns:
            TwoBrainInsight: Analytical brain output

        Raises:
            SeniorAdvisorError: If analytical brain execution fails
        """
        pass

    @abstractmethod
    async def execute_strategic_brain(
        self,
        analyses: List[ConsultantAnalysisResult],
        critiques: List[AnalysisCritique],
        analytical_insight: TwoBrainInsight,
    ) -> TwoBrainInsight:
        """
        Execute strategic brain (Claude) for synthesis and recommendations

        Args:
            analyses: Consultant analysis results
            critiques: Devil's advocate critiques
            analytical_insight: Output from analytical brain

        Returns:
            TwoBrainInsight: Strategic brain output

        Raises:
            SeniorAdvisorError: If strategic brain execution fails
        """
        pass


class SeniorManagerFactory(Protocol):
    """Factory protocol for creating SeniorManager instances"""

    def create_senior_manager(self) -> ISeniorManager:
        """
        Create a SeniorManager instance

        Returns:
            ISeniorManager: Configured senior manager instance
        """
        ...


class DefaultSeniorManagerFactory:
    """Default factory implementation for SeniorManager"""

    def create_senior_manager(self) -> ISeniorManager:
        """Create default SeniorManager instance"""
        from src.services.senior_manager import SeniorManager
        return SeniorManager()


class MockSeniorManagerFactory:
    """Mock factory for testing purposes"""

    def create_senior_manager(self) -> ISeniorManager:
        """Create mock SeniorManager instance for testing"""
        from tests.mocks.mock_senior_manager import MockSeniorManager
        return MockSeniorManager()