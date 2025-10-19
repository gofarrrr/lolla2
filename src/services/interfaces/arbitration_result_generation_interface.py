"""
Arbitration Result Generation Service Interface
=============================================

Interface definition for Arbitration Result Generation Service domain service.
This interface establishes the contract for generating complete arbitration results
with weighted outputs while enabling multiple implementations and testing flexibility.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from src.arbitration.models import (
    ConsultantOutput,
    ConsultantRole,
    DifferentialAnalysis,
    UserWeightingPreferences,
    ArbitrationResult,
)


class IArbitrationResultGenerationService(ABC):
    """
    Interface for Arbitration Result Generation Service domain service

    This interface defines the contract for generating complete arbitration results
    with weighted recommendations, insights, risk assessments, and other components.
    """

    @abstractmethod
    async def generate_arbitration_result(
        self,
        differential_analysis: DifferentialAnalysis,
        user_preferences: UserWeightingPreferences,
        perspective_analysis: Dict[str, Any],
        query_context: Optional[Dict[str, Any]] = None,
    ) -> ArbitrationResult:
        """
        Generate the complete arbitration result with weighted outputs

        Args:
            differential_analysis: Differential analysis results
            user_preferences: User weighting preferences
            perspective_analysis: Perspective mapping analysis
            query_context: Additional context about the query

        Returns:
            Complete arbitration result with all components

        Raises:
            ArbitrationResultGenerationError: If arbitration result generation fails
        """
        pass

    @abstractmethod
    async def generate_weighted_recommendations(
        self,
        consultant_outputs: List[ConsultantOutput],
        user_preferences: UserWeightingPreferences,
    ) -> List[str]:
        """
        Generate weighted recommendations based on user preferences

        Args:
            consultant_outputs: List of consultant analyses
            user_preferences: User weighting preferences

        Returns:
            List of weighted recommendations

        Raises:
            ArbitrationResultGenerationError: If recommendation generation fails
        """
        pass

    @abstractmethod
    async def generate_weighted_insights(
        self,
        consultant_outputs: List[ConsultantOutput],
        user_preferences: UserWeightingPreferences,
    ) -> List[str]:
        """
        Generate weighted insights based on user preferences

        Args:
            consultant_outputs: List of consultant analyses
            user_preferences: User weighting preferences

        Returns:
            List of weighted insights

        Raises:
            ArbitrationResultGenerationError: If insight generation fails
        """
        pass

    @abstractmethod
    async def generate_weighted_risk_assessment(
        self,
        consultant_outputs: List[ConsultantOutput],
        user_preferences: UserWeightingPreferences,
    ) -> str:
        """
        Generate weighted risk assessment

        Args:
            consultant_outputs: List of consultant analyses
            user_preferences: User weighting preferences

        Returns:
            Weighted risk assessment string

        Raises:
            ArbitrationResultGenerationError: If risk assessment generation fails
        """
        pass

    @abstractmethod
    def determine_primary_consultant(
        self,
        merit_assessments: Dict[ConsultantRole, Any],
        user_preferences: UserWeightingPreferences,
    ) -> ConsultantRole:
        """
        Determine which consultant should be the primary recommendation

        Args:
            merit_assessments: Merit assessments for each consultant
            user_preferences: User weighting preferences

        Returns:
            Primary consultant role

        Raises:
            ArbitrationResultGenerationError: If primary consultant determination fails
        """
        pass

    @abstractmethod
    def generate_supporting_rationales(
        self,
        differential_analysis: DifferentialAnalysis,
        user_preferences: UserWeightingPreferences,
    ) -> Dict[ConsultantRole, str]:
        """
        Generate rationales for how each consultant supports the decision

        Args:
            differential_analysis: Differential analysis results
            user_preferences: User weighting preferences

        Returns:
            Supporting rationales for each consultant

        Raises:
            ArbitrationResultGenerationError: If rationale generation fails
        """
        pass

    @abstractmethod
    def build_polygon_enhancements(
        self,
        consultant_outputs: List[ConsultantOutput],
        user_preferences: UserWeightingPreferences,
        differential_analysis: DifferentialAnalysis,
    ) -> Dict[str, Any]:
        """
        Build polygon enhancements for synthesis refinement

        Args:
            consultant_outputs: List of consultant analyses
            user_preferences: User weighting preferences
            differential_analysis: Differential analysis results

        Returns:
            Polygon enhancement data

        Raises:
            ArbitrationResultGenerationError: If polygon enhancement fails
        """
        pass

    @abstractmethod
    def calculate_consensus_strength(
        self, differential_analysis: DifferentialAnalysis
    ) -> float:
        """
        Calculate consensus strength across consultants

        Args:
            differential_analysis: Differential analysis results

        Returns:
            Consensus strength score (0.0 to 1.0)

        Raises:
            ArbitrationResultGenerationError: If consensus calculation fails
        """
        pass

    @abstractmethod
    def extract_risk_mentions(self, text: str) -> List[str]:
        """
        Extract risk mentions from text

        Args:
            text: Text to analyze for risk mentions

        Returns:
            List of extracted risk mentions

        Raises:
            ArbitrationResultGenerationError: If risk extraction fails
        """
        pass

    @abstractmethod
    def contains_risk_content(self, text: str) -> bool:
        """
        Check if text contains risk-related content

        Args:
            text: Text to check

        Returns:
            True if text contains risk content

        Raises:
            ArbitrationResultGenerationError: If risk content check fails
        """
        pass

    @abstractmethod
    def is_similar_recommendation(self, rec: str, seen_recs: set) -> bool:
        """
        Check if recommendation is similar to already seen recommendations

        Args:
            rec: Recommendation to check
            seen_recs: Set of previously seen recommendations

        Returns:
            True if recommendation is similar to existing ones

        Raises:
            ArbitrationResultGenerationError: If similarity check fails
        """
        pass

    @abstractmethod
    def is_similar_insight(self, insight: str, seen_insights: set) -> bool:
        """
        Check if insight is similar to already seen insights

        Args:
            insight: Insight to check
            seen_insights: Set of previously seen insights

        Returns:
            True if insight is similar to existing ones

        Raises:
            ArbitrationResultGenerationError: If similarity check fails
        """
        pass


class ArbitrationResultGenerationError(Exception):
    """Exception for arbitration result generation related errors"""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


# Factory pattern for dependency injection
class ArbitrationResultGenerationServiceFactory:
    """Factory for creating Arbitration Result Generation Service instances"""

    def create_arbitration_result_generation_service(self) -> IArbitrationResultGenerationService:
        """Create default Arbitration Result Generation Service instance"""
        from src.services.arbitration_result_generation_service import ArbitrationResultGenerationService
        return ArbitrationResultGenerationService()


class MockArbitrationResultGenerationServiceFactory:
    """Mock factory for testing purposes"""

    def create_arbitration_result_generation_service(self) -> IArbitrationResultGenerationService:
        """Create mock Arbitration Result Generation Service instance for testing"""
        from tests.mocks.mock_arbitration_result_generation_service import MockArbitrationResultGenerationService
        return MockArbitrationResultGenerationService()


# Convenience function for creating default instance
def create_arbitration_result_generation_service() -> IArbitrationResultGenerationService:
    """
    Convenience function to create an Arbitration Result Generation Service instance

    Returns:
        IArbitrationResultGenerationService: Configured instance
    """
    factory = ArbitrationResultGenerationServiceFactory()
    return factory.create_arbitration_result_generation_service()