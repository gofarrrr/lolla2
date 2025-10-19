"""
Reasoning Synthesizer Interface
Abstract base classes for reasoning synthesis functionality
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any
from src.engine.models.data_contracts import ReasoningStep


class IReasoningSynthesizer(ABC):
    """Interface for reasoning synthesis operations"""

    @abstractmethod
    async def synthesize_reasoning(
        self, reasoning_results: List[Dict[str, Any]]
    ) -> List[ReasoningStep]:
        """
        Synthesize reasoning from multiple mental model applications

        Args:
            reasoning_results: List of reasoning results from mental model applications

        Returns:
            List of ReasoningStep objects including synthesis step
        """
        pass

    @abstractmethod
    async def create_synthesis_text(self, reasoning_steps: List[ReasoningStep]) -> str:
        """
        Create synthesis text combining insights from multiple models using real LLM generation

        Args:
            reasoning_steps: List of reasoning steps to synthesize

        Returns:
            Synthesized text combining all insights
        """
        pass

    @abstractmethod
    def calculate_synthesis_confidence(
        self, reasoning_steps: List[ReasoningStep]
    ) -> float:
        """
        Calculate confidence score for synthesis

        Args:
            reasoning_steps: List of reasoning steps to analyze

        Returns:
            Confidence score between 0.0 and 1.0
        """
        pass


class IReasoningValidator(ABC):
    """Interface for reasoning quality validation"""

    @abstractmethod
    async def validate_reasoning_quality(
        self, reasoning_steps: List[ReasoningStep]
    ) -> Dict[str, Any]:
        """
        Validate quality of reasoning process and outputs

        Args:
            reasoning_steps: List of reasoning steps to validate

        Returns:
            Dictionary containing validation results including:
            - overall_confidence: float
            - confidence_scores: Dict[str, float]
            - quality_metrics: Dict[str, Any]
            - validation_flags: List[str]
        """
        pass

    @abstractmethod
    def assess_cognitive_load(self, reasoning_steps: List[ReasoningStep]) -> str:
        """
        Assess cognitive load for progressive disclosure

        Args:
            reasoning_steps: List of reasoning steps to analyze

        Returns:
            Cognitive load level: 'low', 'medium', or 'high'
        """
        pass


class IReasoningSynthesizerFactory(ABC):
    """Factory interface for creating reasoning synthesizer instances"""

    @abstractmethod
    def create_synthesizer(self, settings: Any, logger: Any) -> IReasoningSynthesizer:
        """Create a reasoning synthesizer instance"""
        pass

    @abstractmethod
    def create_validator(self, settings: Any, logger: Any) -> IReasoningValidator:
        """Create a reasoning validator instance"""
        pass
