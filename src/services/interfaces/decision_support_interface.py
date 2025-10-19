"""
Decision Support Service Interface
================================

Interface definition for Decision Support Service domain service.
This interface establishes the contract for enhancing arbitration results
with advanced decision support features while enabling multiple implementations
and testing flexibility.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict, Any

from src.arbitration.models import (
    ConsultantOutput,
    ArbitrationResult,
)


class IDecisionSupportService(ABC):
    """
    Interface for Decision Support Service domain service

    This interface defines the contract for enhancing arbitration results
    with comprehensive decision support features including confidence intervals,
    decision trees, and stakeholder impact analysis.
    """

    @abstractmethod
    async def enhance_decision_support(
        self,
        arbitration_result: ArbitrationResult,
        consultant_outputs: List[ConsultantOutput],
        original_query: str,
    ) -> ArbitrationResult:
        """
        Enhance arbitration result with comprehensive decision support features

        Args:
            arbitration_result: Arbitration result to enhance
            consultant_outputs: List of consultant analyses
            original_query: Original user query

        Returns:
            Enhanced arbitration result with decision support features

        Raises:
            DecisionSupportError: If decision support enhancement fails
        """
        pass

    @abstractmethod
    async def add_confidence_intervals(
        self, arbitration_result: ArbitrationResult
    ) -> ArbitrationResult:
        """
        Add statistical confidence intervals to weighted recommendations

        Args:
            arbitration_result: Arbitration result to enhance

        Returns:
            Arbitration result with confidence intervals

        Raises:
            DecisionSupportError: If confidence interval calculation fails
        """
        pass

    @abstractmethod
    async def add_decision_trees(
        self, arbitration_result: ArbitrationResult, query: str
    ) -> ArbitrationResult:
        """
        Add decision tree structure for complex multi-consultant scenarios

        Args:
            arbitration_result: Arbitration result to enhance
            query: Original user query for context

        Returns:
            Arbitration result with decision tree

        Raises:
            DecisionSupportError: If decision tree generation fails
        """
        pass

    @abstractmethod
    async def add_stakeholder_impact_analysis(
        self,
        arbitration_result: ArbitrationResult,
        consultant_outputs: List[ConsultantOutput],
    ) -> ArbitrationResult:
        """
        Add comprehensive stakeholder impact analysis

        Args:
            arbitration_result: Arbitration result to enhance
            consultant_outputs: List of consultant analyses

        Returns:
            Arbitration result with stakeholder impact analysis

        Raises:
            DecisionSupportError: If stakeholder analysis fails
        """
        pass

    @abstractmethod
    def add_enhancement_metadata(
        self, arbitration_result: ArbitrationResult
    ) -> ArbitrationResult:
        """
        Add general enhancement metadata and timestamps

        Args:
            arbitration_result: Arbitration result to enhance

        Returns:
            Arbitration result with enhancement metadata

        Raises:
            DecisionSupportError: If metadata enhancement fails
        """
        pass

    @abstractmethod
    def calculate_confidence_statistics(
        self, arbitration_result: ArbitrationResult
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive confidence statistics from arbitration result

        Args:
            arbitration_result: Arbitration result to analyze

        Returns:
            Confidence statistics dictionary

        Raises:
            DecisionSupportError: If confidence calculation fails
        """
        pass

    @abstractmethod
    async def generate_decision_tree(
        self, arbitration_result: ArbitrationResult, query: str
    ) -> Dict[str, Any]:
        """
        Generate a decision tree structure for complex scenarios

        Args:
            arbitration_result: Arbitration result to analyze
            query: Original user query for context

        Returns:
            Decision tree structure

        Raises:
            DecisionSupportError: If decision tree generation fails
        """
        pass

    @abstractmethod
    def analyze_stakeholder_impacts(
        self,
        consultant_outputs: List[ConsultantOutput],
        arbitration_result: ArbitrationResult,
    ) -> Dict[str, Any]:
        """
        Analyze potential stakeholder impacts from consultant recommendations

        Args:
            consultant_outputs: List of consultant analyses
            arbitration_result: Arbitration result for context

        Returns:
            Stakeholder impact analysis

        Raises:
            DecisionSupportError: If stakeholder analysis fails
        """
        pass

    @abstractmethod
    def calculate_enhancement_quality(
        self, arbitration_result: ArbitrationResult
    ) -> float:
        """
        Calculate a quality score for the enhancement process

        Args:
            arbitration_result: Enhanced arbitration result

        Returns:
            Enhancement quality score (0.0 to 1.0)

        Raises:
            DecisionSupportError: If quality calculation fails
        """
        pass


class DecisionSupportError(Exception):
    """Exception for decision support related errors"""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


# Factory pattern for dependency injection
class DecisionSupportServiceFactory:
    """Factory for creating Decision Support Service instances"""

    def create_decision_support_service(self) -> IDecisionSupportService:
        """Create default Decision Support Service instance"""
        from src.services.decision_support_service import DecisionSupportService
        return DecisionSupportService()


class MockDecisionSupportServiceFactory:
    """Mock factory for testing purposes"""

    def create_decision_support_service(self) -> IDecisionSupportService:
        """Create mock Decision Support Service instance for testing"""
        from tests.mocks.mock_decision_support_service import MockDecisionSupportService
        return MockDecisionSupportService()


# Convenience function for creating default instance
def create_decision_support_service() -> IDecisionSupportService:
    """
    Convenience function to create a Decision Support Service instance

    Returns:
        IDecisionSupportService: Configured instance
    """
    factory = DecisionSupportServiceFactory()
    return factory.create_decision_support_service()