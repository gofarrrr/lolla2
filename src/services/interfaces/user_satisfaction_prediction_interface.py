"""
User Satisfaction Prediction Service Interface
============================================

Interface definition for User Satisfaction Prediction Service domain service.
This interface establishes the contract for predicting user satisfaction with
arbitration results based on various factors and analytics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict, Any

from src.arbitration.models import (
    DifferentialAnalysis,
    UserWeightingPreferences,
)


class IUserSatisfactionPredictionService(ABC):
    """
    Interface for User Satisfaction Prediction Service domain service

    This interface defines the contract for predicting user satisfaction with
    arbitration results using various analytical factors and user preferences.
    """

    @abstractmethod
    def predict_user_satisfaction(
        self,
        differential_analysis: DifferentialAnalysis,
        user_preferences: UserWeightingPreferences,
        perspective_analysis: Dict[str, Any],
    ) -> float:
        """
        Predict user satisfaction with arbitration results

        Args:
            differential_analysis: Differential analysis results
            user_preferences: User weighting preferences
            perspective_analysis: Perspective mapping analysis

        Returns:
            User satisfaction prediction score (0.0 to 1.0)

        Raises:
            UserSatisfactionPredictionError: If satisfaction prediction fails
        """
        pass

    @abstractmethod
    def assess_criteria_coverage(
        self,
        differential_analysis: DifferentialAnalysis,
        user_preferences: UserWeightingPreferences,
    ) -> float:
        """
        Assess how well the arbitration covers user's priority criteria

        Args:
            differential_analysis: Differential analysis results
            user_preferences: User weighting preferences

        Returns:
            Criteria coverage score (0.0 to 1.0)

        Raises:
            UserSatisfactionPredictionError: If criteria coverage assessment fails
        """
        pass

    @abstractmethod
    def calculate_consensus_strength(
        self, differential_analysis: DifferentialAnalysis
    ) -> float:
        """
        Calculate strength of consensus across consultant outputs

        Args:
            differential_analysis: Differential analysis results

        Returns:
            Consensus strength score (0.0 to 1.0)

        Raises:
            UserSatisfactionPredictionError: If consensus calculation fails
        """
        pass

    @abstractmethod
    def calculate_preference_merit_alignment(
        self,
        differential_analysis: DifferentialAnalysis,
        user_preferences: UserWeightingPreferences,
    ) -> float:
        """
        Calculate alignment between user preferences and merit scores

        Args:
            differential_analysis: Differential analysis results
            user_preferences: User weighting preferences

        Returns:
            Preference-merit alignment score (0.0 to 1.0)

        Raises:
            UserSatisfactionPredictionError: If alignment calculation fails
        """
        pass

    @abstractmethod
    def assess_cognitive_diversity_satisfaction(
        self, perspective_analysis: Dict[str, Any]
    ) -> float:
        """
        Assess user satisfaction based on cognitive diversity

        Args:
            perspective_analysis: Perspective mapping analysis

        Returns:
            Cognitive diversity satisfaction score (0.0 to 1.0)

        Raises:
            UserSatisfactionPredictionError: If diversity assessment fails
        """
        pass

    @abstractmethod
    def calculate_confidence_adjustment(
        self, differential_analysis: DifferentialAnalysis
    ) -> float:
        """
        Calculate confidence adjustment factor for satisfaction prediction

        Args:
            differential_analysis: Differential analysis results

        Returns:
            Confidence adjustment factor (0.0 to 1.0)

        Raises:
            UserSatisfactionPredictionError: If confidence calculation fails
        """
        pass

    @abstractmethod
    def generate_satisfaction_factors_breakdown(
        self,
        differential_analysis: DifferentialAnalysis,
        user_preferences: UserWeightingPreferences,
        perspective_analysis: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Generate detailed breakdown of satisfaction factors

        Args:
            differential_analysis: Differential analysis results
            user_preferences: User weighting preferences
            perspective_analysis: Perspective mapping analysis

        Returns:
            Dictionary of satisfaction factors and their scores

        Raises:
            UserSatisfactionPredictionError: If breakdown generation fails
        """
        pass

    @abstractmethod
    def predict_satisfaction_confidence_interval(
        self,
        differential_analysis: DifferentialAnalysis,
        user_preferences: UserWeightingPreferences,
        perspective_analysis: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Predict confidence interval for satisfaction prediction

        Args:
            differential_analysis: Differential analysis results
            user_preferences: User weighting preferences
            perspective_analysis: Perspective mapping analysis

        Returns:
            Dictionary with prediction confidence bounds

        Raises:
            UserSatisfactionPredictionError: If confidence interval calculation fails
        """
        pass


class UserSatisfactionPredictionError(Exception):
    """Exception for user satisfaction prediction related errors"""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


# Factory pattern for dependency injection
class UserSatisfactionPredictionServiceFactory:
    """Factory for creating User Satisfaction Prediction Service instances"""

    def create_user_satisfaction_prediction_service(self) -> IUserSatisfactionPredictionService:
        """Create default User Satisfaction Prediction Service instance"""
        from src.services.user_satisfaction_prediction_service import UserSatisfactionPredictionService
        return UserSatisfactionPredictionService()


class MockUserSatisfactionPredictionServiceFactory:
    """Mock factory for testing purposes"""

    def create_user_satisfaction_prediction_service(self) -> IUserSatisfactionPredictionService:
        """Create mock User Satisfaction Prediction Service instance for testing"""
        from tests.mocks.mock_user_satisfaction_prediction_service import MockUserSatisfactionPredictionService
        return MockUserSatisfactionPredictionService()


# Convenience function for creating default instance
def create_user_satisfaction_prediction_service() -> IUserSatisfactionPredictionService:
    """
    Convenience function to create a User Satisfaction Prediction Service instance

    Returns:
        IUserSatisfactionPredictionService: Configured instance
    """
    factory = UserSatisfactionPredictionServiceFactory()
    return factory.create_user_satisfaction_prediction_service()