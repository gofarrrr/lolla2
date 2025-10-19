"""
Problem Analyzer Component
Extracted from MetisCognitiveEngine for modular problem analysis functionality
"""

import logging
from typing import Dict, List, Any, Optional

# Import the interface and data contracts
from src.interfaces.problem_analyzer_interface import (
    IProblemAnalyzer,
    ModelSelectionCriteria,
)


class ProblemAnalyzer(IProblemAnalyzer):
    """
    Problem analysis component implementing systematic problem context analysis,
    complexity assessment, and stakeholder relationship identification.

    Extracted from MetisCognitiveEngine for modular architecture.
    """

    def __init__(self, settings: Optional["CognitiveEngineSettings"] = None):
        """
        Initialize problem analyzer

        Args:
            settings: Configuration settings for analysis behavior
        """
        self.logger = logging.getLogger(__name__)

        # Import and initialize settings
        if settings is not None:
            self.settings = settings
        else:
            from src.config import get_cognitive_settings

            self.settings = get_cognitive_settings()

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
        # Extract key characteristics from problem statement
        problem_indicators = {
            "complexity_level": self.assess_problem_complexity(problem_statement),
            "problem_type": self.classify_problem_type(
                problem_statement, business_context
            ),
            "stakeholder_complexity": self.assess_stakeholder_complexity(
                business_context
            ),
            "time_sensitivity": self._safe_get(
                business_context, "time_constraint", "medium"
            ),
            "accuracy_requirements": self._safe_get(
                business_context,
                "accuracy_requirement",
                self.settings.DEFAULT_ACCURACY_REQUIREMENT,
            ),
        }

        return ModelSelectionCriteria(
            problem_type=problem_indicators["problem_type"],
            complexity_level=problem_indicators["complexity_level"],
            accuracy_requirement=problem_indicators.get(
                "accuracy_requirements", self.settings.DEFAULT_ACCURACY_REQUIREMENT
            ),
            business_context=business_context,
        )

    def assess_problem_complexity(self, problem_statement: str) -> str:
        """
        Assess problem complexity based on linguistic indicators

        Args:
            problem_statement: The problem to analyze

        Returns:
            Complexity level: 'low', 'medium', or 'high'
        """
        complexity_indicators = {
            "high": [
                "strategic",
                "enterprise",
                "transformation",
                "multiple stakeholders",
                "long-term",
            ],
            "medium": [
                "operational",
                "process",
                "departmental",
                "quarterly",
                "improvement",
            ],
            "low": [
                "tactical",
                "specific",
                "individual",
                "immediate",
                "routine",
                "schedule",
                "meeting",
                "fix",
                "bug",
            ],
        }

        statement_lower = problem_statement.lower()

        # Check in order: high -> low -> medium to catch edge cases first
        for level in ["high", "low", "medium"]:
            indicators = complexity_indicators[level]
            if any(indicator in statement_lower for indicator in indicators):
                return level

        return "medium"  # Default

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
        problem_types = {
            "strategic_planning": [
                "strategy",
                "planning",
                "direction",
                "vision",
                "competitive",
            ],
            "operational_optimization": [
                "efficiency",
                "process",
                "optimization",
                "workflow",
                "operations",
            ],
            "decision_analysis": [
                "decision",
                "choice",
                "alternative",
                "option",
                "selection",
            ],
            "risk_assessment": [
                "risk",
                "uncertainty",
                "threat",
                "vulnerability",
                "mitigation",
            ],
            "market_analysis": [
                "market",
                "customer",
                "competitor",
                "segment",
                "positioning",
            ],
            "organizational_change": [
                "change",
                "transformation",
                "reorganization",
                "culture",
                "structure",
            ],
        }

        statement_lower = problem_statement.lower()
        context_text = " ".join(str(v) for v in business_context.values()).lower()

        combined_text = statement_lower + " " + context_text

        for problem_type, keywords in problem_types.items():
            if any(keyword in combined_text for keyword in keywords):
                return problem_type

        return "general_analysis"  # Default

    def assess_stakeholder_complexity(self, business_context: Dict[str, Any]) -> str:
        """
        Assess stakeholder complexity from business context

        Args:
            business_context: Business context containing stakeholder information

        Returns:
            Stakeholder complexity level: 'low', 'medium', or 'high'
        """
        stakeholder_indicators = self._safe_get(business_context, "stakeholders", [])

        if isinstance(stakeholder_indicators, list):
            if len(stakeholder_indicators) > 5:
                return "high"
            elif len(stakeholder_indicators) > 2:
                return "medium"

        return "low"

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
        # Simplified extraction - in production would use NLP
        elements = [
            "stakeholders",
            "processes",
            "resources",
            "constraints",
            "objectives",
        ]
        return elements

    def identify_relationships(self, elements: List[str]) -> List[str]:
        """
        Identify relationships between system elements

        Args:
            elements: List of system elements

        Returns:
            List of identified relationships
        """
        return [
            "stakeholder-process interactions",
            "resource-constraint dependencies",
            "objective-outcome linkages",
        ]

    def identify_feedback_loops(self, relationships: List[str]) -> List[str]:
        """
        Identify feedback loops in the system

        Args:
            relationships: List of system relationships

        Returns:
            List of identified feedback loops
        """
        return [
            "performance-resource allocation loop",
            "stakeholder satisfaction-engagement loop",
        ]

    def _safe_get(self, obj, key: str, default=None):
        """Safely get value from dict or MetisDataContract object"""
        if hasattr(obj, "get"):  # It's a dict
            return obj.get(key, default)
        elif hasattr(obj, key):  # It's an object with the attribute
            return getattr(obj, key, default)
        elif hasattr(obj, "dict"):  # It's a Pydantic model
            obj_dict = obj.dict() if callable(obj.dict) else obj.dict
            return obj_dict.get(key, default)
        else:
            return default

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
        return ["organizational culture", "system resilience", "adaptive capacity"]


# Factory function for creating problem analyzer instances
def get_problem_analyzer(
    settings: Optional["CognitiveEngineSettings"] = None,
) -> ProblemAnalyzer:
    """
    Factory function for creating ProblemAnalyzer instances

    Args:
        settings: Optional configuration settings

    Returns:
        Configured ProblemAnalyzer instance
    """
    return ProblemAnalyzer(settings=settings)
