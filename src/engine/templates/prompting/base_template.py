#!/usr/bin/env python3
"""
Base Template System for DeepSeek V3.1 Prompting Strategies
Implements research-backed prompting optimization
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class PromptContext:
    """Context information for prompt generation"""

    task_type: str
    complexity_score: float
    time_constraints: str
    engagement_id: Optional[str] = None
    phase: Optional[str] = None
    quality_threshold: float = 0.8
    business_context: Optional[Dict[str, Any]] = None
    additional_context: Optional[Dict[str, Any]] = None


class BasePromptTemplate(ABC):
    """Base class for prompting strategy implementations"""

    def __init__(self):
        self.template_name = self.__class__.__name__
        self.research_basis = "DeepSeek V3.1 optimization study"
        self.performance_characteristics = {}

    @abstractmethod
    def generate_prompt(self, original_prompt: str, context: PromptContext) -> str:
        """
        Generate optimized prompt based on strategy

        Args:
            original_prompt: The original user prompt
            context: Context information for optimization

        Returns:
            Optimized prompt string
        """
        pass

    @abstractmethod
    def get_expected_performance_improvement(self) -> Dict[str, float]:
        """
        Get expected performance improvements from research

        Returns:
            Dict with performance metrics (speed, cost, quality improvements)
        """
        pass

    def get_template_info(self) -> Dict[str, Any]:
        """Get template information and characteristics"""
        return {
            "name": self.template_name,
            "research_basis": self.research_basis,
            "performance_characteristics": self.performance_characteristics,
            "expected_improvements": self.get_expected_performance_improvement(),
        }

    def add_metis_context(self, prompt: str, context: PromptContext) -> str:
        """Add METIS-specific context to prompt"""

        context_parts = []

        # Base METIS context
        context_parts.append(
            "You are part of the METIS Cognitive Intelligence Platform, a sophisticated reasoning system "
            "that applies mental models for strategic analysis."
        )

        # Phase context
        if context.phase:
            context_parts.append(f"Current analysis phase: {context.phase}")

        # Engagement context
        if context.engagement_id:
            context_parts.append(f"Engagement ID: {context.engagement_id}")

        # Business context
        if context.business_context:
            if context.business_context.get("industry"):
                context_parts.append(
                    f"Industry context: {context.business_context['industry']}"
                )
            if context.business_context.get("company_size"):
                context_parts.append(
                    f"Company size: {context.business_context['company_size']}"
                )
            if context.business_context.get("stakeholders"):
                stakeholders = ", ".join(context.business_context["stakeholders"])
                context_parts.append(f"Key stakeholders: {stakeholders}")

        # Quality expectations
        if context.quality_threshold > 0.8:
            context_parts.append(
                "Focus on high-quality, evidence-based analysis with detailed reasoning."
            )
        elif context.time_constraints == "urgent":
            context_parts.append(
                "Provide focused, actionable insights optimized for time constraints."
            )
        else:
            context_parts.append(
                "Focus on systematic reasoning, evidence-based analysis, and actionable insights."
            )

        metis_context = "\n\n".join(context_parts)

        return f"{metis_context}\n\n{prompt}"

    def validate_context(self, context: PromptContext) -> bool:
        """Validate that context is appropriate for this template"""
        return (
            context.task_type is not None
            and isinstance(context.complexity_score, (int, float))
            and 0.0 <= context.complexity_score <= 1.0
            and context.time_constraints in ["urgent", "normal", "thorough"]
        )
