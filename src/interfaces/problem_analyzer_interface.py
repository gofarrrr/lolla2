"""
Problem Analyzer Interface
Abstract base class for problem analysis functionality extracted from cognitive engine
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class ModelSelectionCriteria:
    """Criteria for selecting appropriate mental models"""

    problem_type: str
    complexity_level: str  # low, medium, high
    time_constraint: Optional[int] = None  # seconds
    accuracy_requirement: float = 0.8
    business_context: Dict[str, Any] = None


class IProblemAnalyzer(ABC):
    """
    Abstract interface for problem analysis functionality

    Defines the contract for analyzing problem context, complexity,
    stakeholder relationships, and system characteristics.
    """

    @abstractmethod
    async def analyze_problem_context(
        self, problem_statement: str, business_context: Dict[str, Any]
    ) -> ModelSelectionCriteria:
        """
        Analyze problem context to determine model selection criteria

        Args:
            problem_statement: The problem to be analyzed
            business_context: Additional business context and constraints

        Returns:
            ModelSelectionCriteria with analyzed problem characteristics
        """
        pass

    @abstractmethod
    def assess_problem_complexity(self, problem_statement: str) -> str:
        """
        Assess problem complexity based on linguistic indicators

        Args:
            problem_statement: The problem to analyze

        Returns:
            Complexity level: 'low', 'medium', or 'high'
        """
        pass

    @abstractmethod
    def classify_problem_type(
        self, problem_statement: str, business_context: Dict[str, Any]
    ) -> str:
        """
        Classify problem type for model selection

        Args:
            problem_statement: The problem to classify
            business_context: Additional context for classification

        Returns:
            Problem type category for model selection
        """
        pass

    @abstractmethod
    def assess_stakeholder_complexity(self, business_context: Dict[str, Any]) -> str:
        """
        Assess stakeholder complexity from business context

        Args:
            business_context: Business context containing stakeholder information

        Returns:
            Stakeholder complexity level: 'low', 'medium', or 'high'
        """
        pass

    @abstractmethod
    def identify_system_elements(
        self, problem_statement: str, business_context: Dict[str, Any]
    ) -> List[str]:
        """
        Identify key system elements in the problem context

        Args:
            problem_statement: The problem to analyze
            business_context: Additional context for element identification

        Returns:
            List of identified system elements
        """
        pass

    @abstractmethod
    def identify_relationships(self, elements: List[str]) -> List[str]:
        """
        Identify relationships between system elements

        Args:
            elements: List of system elements

        Returns:
            List of identified relationships
        """
        pass

    @abstractmethod
    def identify_feedback_loops(self, relationships: List[str]) -> List[str]:
        """
        Identify feedback loops in the system

        Args:
            relationships: List of system relationships

        Returns:
            List of identified feedback loops
        """
        pass

    @abstractmethod
    def identify_emergent_properties(
        self, elements: List[str], relationships: List[str]
    ) -> List[str]:
        """
        Identify emergent system properties

        Args:
            elements: List of system elements
            relationships: List of system relationships

        Returns:
            List of identified emergent properties
        """
        pass
