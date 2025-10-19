#!/usr/bin/env python3
"""
Direct Prompting Template for DeepSeek V3.1
Minimal overhead for fast response tasks
"""

from typing import Dict
from .base_template import BasePromptTemplate, PromptContext


class DirectTemplate(BasePromptTemplate):
    """
    Direct prompting strategy for DeepSeek V3.1 fast responses

    Characteristics:
    - Minimal prompting overhead
    - Optimized for speed over complexity
    - 5.3x faster than reasoning mode
    - 38x more cost-effective
    - Ideal for simple tasks and high-volume applications
    """

    def __init__(self):
        super().__init__()
        self.performance_characteristics = {
            "speed_advantage": 5.3,  # 5.3x faster than reasoning mode
            "cost_advantage": 38.0,  # 38x cheaper
            "response_time": 31.8,  # Average 31.8 seconds
            "cost_per_query": 0.0002,  # $0.0002 per query
            "optimal_for": ["fast_response", "high_volume", "simple_tasks"],
        }

    def generate_prompt(self, original_prompt: str, context: PromptContext) -> str:
        """Generate direct, minimal prompt"""

        if not self.validate_context(context):
            raise ValueError("Invalid context for Direct template")

        # Minimal METIS context for speed
        minimal_context = self._get_minimal_metis_context(context)

        # Add simple task clarification if needed
        clarification = self._get_task_clarification(context)

        # Build direct template with minimal overhead
        direct_prompt = f"""
{minimal_context}

{original_prompt}

{clarification}
""".strip()

        return direct_prompt

    def _get_minimal_metis_context(self, context: PromptContext) -> str:
        """Get minimal METIS context for speed"""

        # Only include essential context
        context_parts = []

        # Core METIS identity (brief)
        context_parts.append(
            "You are part of the METIS platform. Provide focused, actionable insights."
        )

        # Essential business context only
        if context.business_context and context.business_context.get("industry"):
            context_parts.append(f"Industry: {context.business_context['industry']}")

        # Time urgency indicator
        if context.time_constraints == "urgent":
            context_parts.append("Priority: Urgent response needed.")

        return " ".join(context_parts)

    def _get_task_clarification(self, context: PromptContext) -> str:
        """Get minimal task clarification based on type"""

        clarifications = {
            "summary_generation": "Provide a concise summary.",
            "problem_classification": "Classify and briefly explain the problem type.",
            "query_generation": "Generate relevant questions for further analysis.",
            "pattern_recognition": "Identify key patterns and briefly explain their significance.",
            "context_understanding": "Explain the key context factors.",
            "quick_insights": "Provide 3-5 key insights in bullet points.",
        }

        # Get task-specific clarification
        clarification = clarifications.get(context.task_type, "")

        # Add urgency modifier if needed
        if context.time_constraints == "urgent" and clarification:
            clarification = f"Quick response needed: {clarification}"

        # Add format preference for structured tasks
        if context.task_type in ["problem_classification", "quick_insights"]:
            clarification += " Use clear, structured format."

        return clarification

    def get_expected_performance_improvement(self) -> Dict[str, float]:
        """Expected performance characteristics"""
        return {
            "response_time_seconds": 31.8,  # Average response time
            "cost_usd": 0.0002,  # Average cost
            "speed_vs_reasoning": 5.3,  # 5.3x faster than reasoning mode
            "cost_vs_reasoning": 38.0,  # 38x cheaper than reasoning mode
            "success_rate": 1.0,  # 100% success for appropriate tasks
            "quality_score": 0.33,  # Lower but sufficient for simple tasks
        }

    def is_optimal_for_context(self, context: PromptContext) -> bool:
        """Check if Direct template is optimal for given context"""

        optimal_conditions = [
            # Low complexity tasks
            context.complexity_score <= 0.4,
            # Urgent time constraints
            context.time_constraints == "urgent",
            # Simple task types
            context.task_type
            in [
                "summary_generation",
                "problem_classification",
                "query_generation",
                "pattern_recognition",
                "context_understanding",
                "quick_insights",
            ],
            # Cost-sensitive environments
            context.additional_context
            and context.additional_context.get("cost_sensitive", False),
            # High-volume applications
            context.additional_context
            and context.additional_context.get("high_volume", False),
            # Speed-critical applications
            context.additional_context
            and context.additional_context.get("speed_critical", False),
        ]

        return any(optimal_conditions)

    def get_optimization_principles(self) -> Dict[str, str]:
        """Get optimization principles for direct prompting"""
        return {
            "minimize_overhead": "Reduce all unnecessary prompt elements",
            "clear_instructions": "Use simple, direct language",
            "avoid_examples": "No few-shot examples (they slow DeepSeek down)",
            "single_focus": "One clear task per prompt",
            "structured_output": "Request specific output format when needed",
            "context_minimal": "Include only essential context information",
            "avoid_meta": "No meta-instructions about thinking or reasoning",
        }

    def validate_for_direct_use(self, context: PromptContext) -> Dict[str, bool]:
        """Validate that context is appropriate for direct template"""

        validation_checks = {
            "low_complexity": context.complexity_score <= 0.5,
            "simple_task": context.task_type
            in [
                "summary_generation",
                "problem_classification",
                "query_generation",
                "pattern_recognition",
                "context_understanding",
                "quick_insights",
            ],
            "acceptable_quality": context.quality_threshold <= 0.6,
            "speed_priority": context.time_constraints in ["urgent", "normal"],
            "not_business_critical": not (
                context.additional_context
                and context.additional_context.get("business_critical", False)
            ),
        }

        return validation_checks

    def estimate_cost_savings(self, baseline_cost: float) -> Dict[str, float]:
        """Estimate cost savings vs other approaches"""

        return {
            "vs_reasoning_mode": baseline_cost * 37,  # 38x savings vs reasoning
            "vs_standard_complex": baseline_cost * 25,  # ~25x vs standard complex
            "vs_ultra_complex": baseline_cost * 45,  # ~45x vs ultra complex
            "absolute_cost": 0.0002,  # Absolute cost per query
            "cost_per_1000_queries": 0.20,  # Cost for 1000 queries
        }
