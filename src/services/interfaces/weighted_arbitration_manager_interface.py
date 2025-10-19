"""
WeightedArbitrationManager Service Interface
==========================================

Interface definition for WeightedArbitrationManager domain service.
This interface establishes the contract for weighted arbitration and decision-making
while enabling multiple implementations and testing flexibility.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Protocol

from src.arbitration.models import (
    DifferentialAnalysis,
    UserWeightingPreferences,
    ArbitrationResult,
    ConsultantRole,
    ConsultantOutput,
)


class IWeightedArbitrationManager(ABC):
    """
    Interface for WeightedArbitrationManager domain service

    This interface defines the contract for weighted arbitration coordination,
    enabling clean separation between orchestration and domain logic.
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
        Generate complete arbitration result with weighted outputs

        Args:
            differential_analysis: Analysis results from all consultants
            user_preferences: User's weighting preferences
            perspective_analysis: Perspective mapping results
            query_context: Optional query context data

        Returns:
            ArbitrationResult: Complete arbitration with weighted recommendations

        Raises:
            ArbitrationError: If arbitration generation fails
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
            merit_assessments: Merit scores for each consultant
            user_preferences: User's weighting preferences

        Returns:
            ConsultantRole: Primary consultant recommendation

        Raises:
            ArbitrationError: If primary consultant determination fails
        """
        pass

    @abstractmethod
    async def generate_supporting_rationales(
        self,
        differential_analysis: DifferentialAnalysis,
        user_preferences: UserWeightingPreferences,
    ) -> Dict[ConsultantRole, str]:
        """
        Generate rationales for how each consultant supports the decision

        Args:
            differential_analysis: Complete differential analysis
            user_preferences: User's weighting preferences

        Returns:
            Dict mapping consultant roles to their supporting rationales

        Raises:
            ArbitrationError: If rationale generation fails
        """
        pass

    @abstractmethod
    async def predict_user_satisfaction(
        self,
        consultant_outputs: List[ConsultantOutput],
        user_preferences: UserWeightingPreferences,
        primary_consultant: ConsultantRole,
    ) -> float:
        """
        Predict user satisfaction based on arbitration results

        Args:
            consultant_outputs: All consultant outputs
            user_preferences: User's weighting preferences
            primary_consultant: Determined primary consultant

        Returns:
            float: Predicted satisfaction score (0.0-1.0)
        """
        pass

    @abstractmethod
    def calculate_arbitration_confidence(
        self,
        differential_analysis: DifferentialAnalysis,
        user_preferences: UserWeightingPreferences,
        primary_consultant: ConsultantRole,
    ) -> float:
        """
        Calculate confidence in the arbitration decision

        Args:
            differential_analysis: Complete differential analysis
            user_preferences: User's weighting preferences
            primary_consultant: Determined primary consultant

        Returns:
            float: Arbitration confidence score (0.0-1.0)
        """
        pass

    @abstractmethod
    def calculate_decision_quality_score(
        self,
        differential_analysis: DifferentialAnalysis,
        weighted_recommendations: List[str],
        weighted_insights: List[str],
    ) -> float:
        """
        Calculate overall decision quality score

        Args:
            differential_analysis: Complete differential analysis
            weighted_recommendations: Generated weighted recommendations
            weighted_insights: Generated weighted insights

        Returns:
            float: Decision quality score (0.0-1.0)
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
            consultant_outputs: All consultant outputs
            user_preferences: User's weighting preferences

        Returns:
            List[str]: Weighted and formatted recommendations

        Raises:
            ArbitrationError: If recommendation generation fails
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
            consultant_outputs: All consultant outputs
            user_preferences: User's weighting preferences

        Returns:
            List[str]: Weighted and formatted insights

        Raises:
            ArbitrationError: If insight generation fails
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
            consultant_outputs: All consultant outputs
            user_preferences: User's weighting preferences

        Returns:
            str: Comprehensive weighted risk assessment

        Raises:
            ArbitrationError: If risk assessment generation fails
        """
        pass


class WeightedArbitrationManagerFactory(Protocol):
    """Factory protocol for creating WeightedArbitrationManager instances"""

    def create_weighted_arbitration_manager(self) -> IWeightedArbitrationManager:
        """
        Create a WeightedArbitrationManager instance

        Returns:
            IWeightedArbitrationManager: Configured weighted arbitration manager instance
        """
        ...


class DefaultWeightedArbitrationManagerFactory:
    """Default factory implementation for WeightedArbitrationManager"""

    def create_weighted_arbitration_manager(self) -> IWeightedArbitrationManager:
        """Create default WeightedArbitrationManager instance"""
        from src.services.weighted_arbitration_manager import WeightedArbitrationManager
        return WeightedArbitrationManager()


class MockWeightedArbitrationManagerFactory:
    """Mock factory for testing purposes"""

    def create_weighted_arbitration_manager(self) -> IWeightedArbitrationManager:
        """Create mock WeightedArbitrationManager instance for testing"""
        from tests.mocks.mock_weighted_arbitration_manager import MockWeightedArbitrationManager
        return MockWeightedArbitrationManager()


# Convenience function for creating default instance
def create_weighted_arbitration_manager() -> IWeightedArbitrationManager:
    """
    Convenience function to create a WeightedArbitrationManager instance

    Returns:
        IWeightedArbitrationManager: Configured instance
    """
    factory = DefaultWeightedArbitrationManagerFactory()
    return factory.create_weighted_arbitration_manager()