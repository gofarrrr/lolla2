"""
Merit Assessment Service Interface
===============================

Interface definition for Merit Assessment Service domain service.
This interface establishes the contract for comprehensive consultant output
quality assessment while enabling multiple implementations and testing flexibility.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from src.arbitration.models import (
    ConsultantOutput,
    ConsultantRole,
    MeritCriterion,
    MeritScore,
    ConsultantMeritAssessment,
)


class IMeritAssessmentService(ABC):
    """
    Interface for Merit Assessment Service domain service

    This interface defines the contract for comprehensive quality assessment
    of consultant outputs across multiple merit criteria.
    """

    @abstractmethod
    async def assess_consultant_outputs(
        self,
        consultant_outputs: List[ConsultantOutput],
        original_query: str,
        query_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[ConsultantRole, ConsultantMeritAssessment]:
        """
        Assess merit of all consultant outputs

        Args:
            consultant_outputs: List of consultant analyses
            original_query: Original user query for context-specific assessment
            query_context: Additional context about the query (industry, urgency, etc.)

        Returns:
            Merit assessments for each consultant

        Raises:
            MeritAssessmentError: If merit assessment fails
        """
        pass

    @abstractmethod
    async def assess_single_consultant(
        self,
        output: ConsultantOutput,
        original_query: str,
        query_context: Optional[Dict[str, Any]] = None,
    ) -> ConsultantMeritAssessment:
        """
        Assess merit of a single consultant output

        Args:
            output: Consultant analysis to assess
            original_query: Original user query for context
            query_context: Additional query context

        Returns:
            Merit assessment for the consultant

        Raises:
            MeritAssessmentError: If single consultant assessment fails
        """
        pass

    @abstractmethod
    async def assess_evidence_quality(self, output: ConsultantOutput) -> MeritScore:
        """
        Assess quality of evidence and research

        Args:
            output: Consultant output to assess

        Returns:
            Merit score for evidence quality

        Raises:
            MeritAssessmentError: If evidence quality assessment fails
        """
        pass

    @abstractmethod
    async def assess_logical_consistency(self, output: ConsultantOutput) -> MeritScore:
        """
        Assess logical consistency and reasoning quality

        Args:
            output: Consultant output to assess

        Returns:
            Merit score for logical consistency

        Raises:
            MeritAssessmentError: If logical consistency assessment fails
        """
        pass

    @abstractmethod
    async def assess_query_alignment(
        self, output: ConsultantOutput, original_query: str
    ) -> MeritScore:
        """
        Assess how well the output addresses the original query

        Args:
            output: Consultant output to assess
            original_query: Original user query

        Returns:
            Merit score for query alignment

        Raises:
            MeritAssessmentError: If query alignment assessment fails
        """
        pass

    @abstractmethod
    async def assess_implementation_feasibility(
        self, output: ConsultantOutput
    ) -> MeritScore:
        """
        Assess practical feasibility of recommendations

        Args:
            output: Consultant output to assess

        Returns:
            Merit score for implementation feasibility

        Raises:
            MeritAssessmentError: If feasibility assessment fails
        """
        pass

    @abstractmethod
    async def assess_risk_thoroughness(self, output: ConsultantOutput) -> MeritScore:
        """
        Assess thoroughness of risk identification and analysis

        Args:
            output: Consultant output to assess

        Returns:
            Merit score for risk thoroughness

        Raises:
            MeritAssessmentError: If risk assessment fails
        """
        pass

    @abstractmethod
    async def assess_novel_insights(self, output: ConsultantOutput) -> MeritScore:
        """
        Assess novelty and creativity of insights

        Args:
            output: Consultant output to assess

        Returns:
            Merit score for novel insights

        Raises:
            MeritAssessmentError: If novelty assessment fails
        """
        pass

    @abstractmethod
    async def assess_bias_resistance(self, output: ConsultantOutput) -> MeritScore:
        """
        Assess resistance to cognitive biases

        Args:
            output: Consultant output to assess

        Returns:
            Merit score for bias resistance

        Raises:
            MeritAssessmentError: If bias resistance assessment fails
        """
        pass

    @abstractmethod
    def calculate_overall_merit(
        self, criterion_scores: Dict[MeritCriterion, MeritScore]
    ) -> float:
        """
        Calculate weighted overall merit score

        Args:
            criterion_scores: Scores for each merit criterion

        Returns:
            Overall merit score

        Raises:
            MeritAssessmentError: If overall merit calculation fails
        """
        pass

    @abstractmethod
    async def calculate_query_fitness_score(
        self, output: ConsultantOutput, query: str
    ) -> float:
        """
        Calculate how well suited this consultant is for this specific query

        Args:
            output: Consultant output to assess
            query: Original user query

        Returns:
            Query fitness score

        Raises:
            MeritAssessmentError: If fitness calculation fails
        """
        pass


class MeritAssessmentError(Exception):
    """Exception for merit assessment related errors"""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


# Factory pattern for dependency injection
class MeritAssessmentServiceFactory:
    """Factory for creating Merit Assessment Service instances"""

    def create_merit_assessment_service(self) -> IMeritAssessmentService:
        """Create default Merit Assessment Service instance"""
        from src.services.merit_assessment_service import MeritAssessmentService
        return MeritAssessmentService()


class MockMeritAssessmentServiceFactory:
    """Mock factory for testing purposes"""

    def create_merit_assessment_service(self) -> IMeritAssessmentService:
        """Create mock Merit Assessment Service instance for testing"""
        from tests.mocks.mock_merit_assessment_service import MockMeritAssessmentService
        return MockMeritAssessmentService()


# Convenience function for creating default instance
def create_merit_assessment_service() -> IMeritAssessmentService:
    """
    Convenience function to create a Merit Assessment Service instance

    Returns:
        IMeritAssessmentService: Configured instance
    """
    factory = MeritAssessmentServiceFactory()
    return factory.create_merit_assessment_service()