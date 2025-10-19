"""
Analytical Brain Service Interface
=================================

Interface definition for Analytical Brain Service domain service.
This interface establishes the contract for complex multi-layered analysis,
cognitive pattern recognition, and sophisticated analytical reasoning capabilities.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple

from src.arbitration.models import (
    ConsultantOutput,
    UserWeightingPreferences,
    DifferentialAnalysis,
)


class IAnalyticalBrainService(ABC):
    """
    Interface for Analytical Brain Service domain service

    This interface defines the contract for advanced analytical capabilities
    including complex pattern recognition, multi-layered analysis, and
    sophisticated reasoning processes.
    """

    @abstractmethod
    async def perform_complex_analysis(
        self,
        consultant_outputs: List[ConsultantOutput],
        user_preferences: UserWeightingPreferences,
        differential_analysis: DifferentialAnalysis,
        analysis_depth: str = "deep",
    ) -> Dict[str, Any]:
        """
        Perform complex multi-layered analysis

        Args:
            consultant_outputs: List of consultant analyses
            user_preferences: User weighting preferences
            differential_analysis: Differential analysis results
            analysis_depth: Depth of analysis ('surface', 'deep', 'comprehensive')

        Returns:
            Complex analysis results dictionary

        Raises:
            AnalyticalBrainError: If complex analysis fails
        """
        pass

    @abstractmethod
    def analyze_cognitive_patterns(
        self,
        consultant_outputs: List[ConsultantOutput],
        perspective_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Analyze cognitive patterns across consultant outputs

        Args:
            consultant_outputs: List of consultant analyses
            perspective_analysis: Perspective mapping analysis

        Returns:
            Cognitive patterns analysis dictionary

        Raises:
            AnalyticalBrainError: If cognitive pattern analysis fails
        """
        pass

    @abstractmethod
    def detect_analytical_gaps(
        self,
        consultant_outputs: List[ConsultantOutput],
        differential_analysis: DifferentialAnalysis,
    ) -> List[Dict[str, Any]]:
        """
        Detect gaps in analytical coverage

        Args:
            consultant_outputs: List of consultant analyses
            differential_analysis: Differential analysis results

        Returns:
            List of detected analytical gaps

        Raises:
            AnalyticalBrainError: If gap detection fails
        """
        pass

    @abstractmethod
    def generate_meta_insights(
        self,
        consultant_outputs: List[ConsultantOutput],
        differential_analysis: DifferentialAnalysis,
        user_preferences: UserWeightingPreferences,
    ) -> Dict[str, Any]:
        """
        Generate meta-level insights about the analysis

        Args:
            consultant_outputs: List of consultant analyses
            differential_analysis: Differential analysis results
            user_preferences: User weighting preferences

        Returns:
            Meta-insights dictionary

        Raises:
            AnalyticalBrainError: If meta-insight generation fails
        """
        pass

    @abstractmethod
    def perform_reasoning_chain_analysis(
        self,
        consultant_outputs: List[ConsultantOutput],
    ) -> Dict[str, Any]:
        """
        Analyze reasoning chains and logical structures

        Args:
            consultant_outputs: List of consultant analyses

        Returns:
            Reasoning chain analysis dictionary

        Raises:
            AnalyticalBrainError: If reasoning chain analysis fails
        """
        pass

    @abstractmethod
    def assess_analytical_coherence(
        self,
        consultant_outputs: List[ConsultantOutput],
        differential_analysis: DifferentialAnalysis,
    ) -> float:
        """
        Assess overall analytical coherence

        Args:
            consultant_outputs: List of consultant analyses
            differential_analysis: Differential analysis results

        Returns:
            Coherence score (0.0 to 1.0)

        Raises:
            AnalyticalBrainError: If coherence assessment fails
        """
        pass

    @abstractmethod
    def identify_emergent_themes(
        self,
        consultant_outputs: List[ConsultantOutput],
        user_preferences: UserWeightingPreferences,
    ) -> List[Dict[str, Any]]:
        """
        Identify emergent themes across analyses

        Args:
            consultant_outputs: List of consultant analyses
            user_preferences: User weighting preferences

        Returns:
            List of emergent themes

        Raises:
            AnalyticalBrainError: If theme identification fails
        """
        pass

    @abstractmethod
    def generate_synthesis_insights(
        self,
        consultant_outputs: List[ConsultantOutput],
        differential_analysis: DifferentialAnalysis,
        meta_insights: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate synthesis insights from multiple analysis layers

        Args:
            consultant_outputs: List of consultant analyses
            differential_analysis: Differential analysis results
            meta_insights: Meta-level insights

        Returns:
            Synthesis insights dictionary

        Raises:
            AnalyticalBrainError: If synthesis generation fails
        """
        pass

    @abstractmethod
    def evaluate_decision_complexity(
        self,
        consultant_outputs: List[ConsultantOutput],
        user_preferences: UserWeightingPreferences,
    ) -> Dict[str, Any]:
        """
        Evaluate the complexity of the decision being analyzed

        Args:
            consultant_outputs: List of consultant analyses
            user_preferences: User weighting preferences

        Returns:
            Decision complexity evaluation

        Raises:
            AnalyticalBrainError: If complexity evaluation fails
        """
        pass

    @abstractmethod
    def generate_analytical_confidence_map(
        self,
        consultant_outputs: List[ConsultantOutput],
        differential_analysis: DifferentialAnalysis,
    ) -> Dict[str, Any]:
        """
        Generate confidence mapping across analytical dimensions

        Args:
            consultant_outputs: List of consultant analyses
            differential_analysis: Differential analysis results

        Returns:
            Analytical confidence map

        Raises:
            AnalyticalBrainError: If confidence mapping fails
        """
        pass


class AnalyticalBrainError(Exception):
    """Exception for analytical brain related errors"""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


# Factory pattern for dependency injection
class AnalyticalBrainServiceFactory:
    """Factory for creating Analytical Brain Service instances"""

    def create_analytical_brain_service(self) -> IAnalyticalBrainService:
        """Create default Analytical Brain Service instance"""
        from src.services.analytical_brain_service import AnalyticalBrainService
        return AnalyticalBrainService()


class MockAnalyticalBrainServiceFactory:
    """Mock factory for testing purposes"""

    def create_analytical_brain_service(self) -> IAnalyticalBrainService:
        """Create mock Analytical Brain Service instance for testing"""
        from tests.mocks.mock_analytical_brain_service import MockAnalyticalBrainService
        return MockAnalyticalBrainService()


# Convenience function for creating default instance
def create_analytical_brain_service() -> IAnalyticalBrainService:
    """
    Convenience function to create an Analytical Brain Service instance

    Returns:
        IAnalyticalBrainService: Configured instance
    """
    factory = AnalyticalBrainServiceFactory()
    return factory.create_analytical_brain_service()