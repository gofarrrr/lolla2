"""
Alternative Scenarios Service Interface
=====================================

Interface definition for Alternative Scenarios Service domain service.
This interface establishes the contract for generating alternative weighting scenarios
and implementation guidance while enabling multiple implementations and testing flexibility.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from src.arbitration.models import (
    ConsultantOutput,
    ConsultantRole,
    DifferentialAnalysis,
    UserWeightingPreferences,
)


class IAlternativeScenariosService(ABC):
    """
    Interface for Alternative Scenarios Service domain service

    This interface defines the contract for generating alternative weighting scenarios
    and implementation guidance to help users explore different decision-making approaches.
    """

    @abstractmethod
    async def generate_alternative_scenarios(
        self,
        consultant_outputs: List[ConsultantOutput],
        user_preferences: UserWeightingPreferences,
        merit_assessments: Optional[Dict[ConsultantRole, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate alternative weighting scenarios for user consideration

        Args:
            consultant_outputs: List of consultant analyses
            user_preferences: User weighting preferences
            merit_assessments: Merit assessments for each consultant

        Returns:
            List of alternative scenario dictionaries

        Raises:
            AlternativeScenariosError: If scenario generation fails
        """
        pass

    @abstractmethod
    async def generate_implementation_guidance(
        self,
        consultant_outputs: List[ConsultantOutput],
        user_preferences: UserWeightingPreferences,
        differential_analysis: DifferentialAnalysis,
    ) -> Dict[str, Any]:
        """
        Generate implementation guidance based on arbitration results

        Args:
            consultant_outputs: List of consultant analyses
            user_preferences: User weighting preferences
            differential_analysis: Differential analysis results

        Returns:
            Implementation guidance dictionary

        Raises:
            AlternativeScenariosError: If implementation guidance generation fails
        """
        pass

    @abstractmethod
    def generate_equal_weighting_scenario(
        self, consultant_outputs: List[ConsultantOutput]
    ) -> Dict[str, Any]:
        """
        Generate equal weighting scenario

        Args:
            consultant_outputs: List of consultant analyses

        Returns:
            Equal weighting scenario dictionary

        Raises:
            AlternativeScenariosError: If equal weighting scenario generation fails
        """
        pass

    @abstractmethod
    def generate_merit_based_scenario(
        self, merit_assessments: Dict[ConsultantRole, Any]
    ) -> Dict[str, Any]:
        """
        Generate merit-based weighting scenario

        Args:
            merit_assessments: Merit assessments for each consultant

        Returns:
            Merit-based scenario dictionary

        Raises:
            AlternativeScenariosError: If merit-based scenario generation fails
        """
        pass

    @abstractmethod
    def generate_conservative_scenario(
        self, user_preferences: UserWeightingPreferences
    ) -> Dict[str, Any]:
        """
        Generate conservative (risk-focused) weighting scenario

        Args:
            user_preferences: User weighting preferences

        Returns:
            Conservative scenario dictionary

        Raises:
            AlternativeScenariosError: If conservative scenario generation fails
        """
        pass

    @abstractmethod
    def generate_priority_order(
        self,
        consultant_outputs: List[ConsultantOutput],
        user_preferences: UserWeightingPreferences,
    ) -> List[str]:
        """
        Generate priority order for recommendations

        Args:
            consultant_outputs: List of consultant analyses
            user_preferences: User weighting preferences

        Returns:
            List of prioritized recommendations

        Raises:
            AlternativeScenariosError: If priority order generation fails
        """
        pass

    @abstractmethod
    def generate_success_metrics(
        self,
        consultant_outputs: List[ConsultantOutput],
        user_preferences: UserWeightingPreferences,
    ) -> List[str]:
        """
        Generate success metrics based on consultant insights

        Args:
            consultant_outputs: List of consultant analyses
            user_preferences: User weighting preferences

        Returns:
            List of success metrics

        Raises:
            AlternativeScenariosError: If success metrics generation fails
        """
        pass

    @abstractmethod
    def generate_monitoring_recommendations(
        self,
        consultant_outputs: List[ConsultantOutput],
        differential_analysis: DifferentialAnalysis,
    ) -> List[str]:
        """
        Generate monitoring recommendations based on analysis

        Args:
            consultant_outputs: List of consultant analyses
            differential_analysis: Differential analysis results

        Returns:
            List of monitoring recommendations

        Raises:
            AlternativeScenariosError: If monitoring recommendations generation fails
        """
        pass

    @abstractmethod
    def assess_scenario_impact(
        self,
        scenario: Dict[str, Any],
        consultant_outputs: List[ConsultantOutput],
    ) -> Dict[str, Any]:
        """
        Assess the impact of a specific weighting scenario

        Args:
            scenario: Weighting scenario to assess
            consultant_outputs: List of consultant analyses

        Returns:
            Impact assessment dictionary

        Raises:
            AlternativeScenariosError: If scenario impact assessment fails
        """
        pass

    @abstractmethod
    def generate_scenario_comparison(
        self,
        scenarios: List[Dict[str, Any]],
        consultant_outputs: List[ConsultantOutput],
    ) -> Dict[str, Any]:
        """
        Generate comparison between multiple scenarios

        Args:
            scenarios: List of scenarios to compare
            consultant_outputs: List of consultant analyses

        Returns:
            Scenario comparison analysis

        Raises:
            AlternativeScenariosError: If scenario comparison fails
        """
        pass


class AlternativeScenariosError(Exception):
    """Exception for alternative scenarios related errors"""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


# Factory pattern for dependency injection
class AlternativeScenariosServiceFactory:
    """Factory for creating Alternative Scenarios Service instances"""

    def create_alternative_scenarios_service(self) -> IAlternativeScenariosService:
        """Create default Alternative Scenarios Service instance"""
        from src.services.alternative_scenarios_service import AlternativeScenariosService
        return AlternativeScenariosService()


class MockAlternativeScenariosServiceFactory:
    """Mock factory for testing purposes"""

    def create_alternative_scenarios_service(self) -> IAlternativeScenariosService:
        """Create mock Alternative Scenarios Service instance for testing"""
        from tests.mocks.mock_alternative_scenarios_service import MockAlternativeScenariosService
        return MockAlternativeScenariosService()


# Convenience function for creating default instance
def create_alternative_scenarios_service() -> IAlternativeScenariosService:
    """
    Convenience function to create an Alternative Scenarios Service instance

    Returns:
        IAlternativeScenariosService: Configured instance
    """
    factory = AlternativeScenariosServiceFactory()
    return factory.create_alternative_scenarios_service()