"""
Reasoning Mode Selector - Phase 3 Optimization

Intelligent triage for Grok 4 Fast reasoning mode activation.

Based on research: Simple tasks don't need reasoning overhead,
complex tasks benefit from reasoning mode's enhanced capabilities.

Part of Research-Grounded Improvement Plan Phase 3.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ReasoningModeSelector:
    """
    Intelligent triage for Grok 4 Fast reasoning mode.

    Based on research findings:
    - Simple tasks: reasoning_enabled=False (faster, cheaper)
    - Complex tasks: reasoning_enabled=True (better quality)
    - Expected cost reduction: 20-30% on LLM calls
    - Expected token reduction: 40% on thinking tokens
    """

    # Research-validated task categories
    REASONING_TASKS = {
        'consultant_analysis',
        'problem_structuring',
        'senior_advisor_synthesis',
        'strategic_recommendation',
        'devils_advocate_critique',
        'minority_report_generation',
    }

    SIMPLE_TASKS = {
        'evidence_extraction',
        'format_conversion',
        'simple_synthesis',
        'brief_summarization',
        's2a_dimension_filter',  # S2A filtering is straightforward
    }

    def __init__(self):
        self._reasoning_enabled_count = 0
        self._reasoning_disabled_count = 0

    def should_enable_reasoning(
        self,
        task_type: str,
        prompt_length: int = 0,
        requires_multi_step: bool = False,
        stakeholder_impact: str = "medium"
    ) -> bool:
        """
        Decide if reasoning mode is worth the computational cost.

        Research-based heuristics:
        - Enable for: strategic analysis, multi-step problems, high-stakes decisions
        - Disable for: simple extraction, formatting, quick synthesis

        Args:
            task_type: Type of task being performed
            prompt_length: Length of the prompt in characters
            requires_multi_step: Whether the task requires multi-step reasoning
            stakeholder_impact: Impact level ("low", "medium", "high")

        Returns:
            True if reasoning mode should be enabled
        """
        # High-stakes decisions ALWAYS use reasoning
        if stakeholder_impact == "high":
            logger.info(f"🧠 Reasoning ENABLED: High-stakes decision ({task_type})")
            self._reasoning_enabled_count += 1
            return True

        # Complex multi-step problems need reasoning
        if requires_multi_step or prompt_length > 800:
            logger.info(f"🧠 Reasoning ENABLED: Multi-step/long prompt ({task_type}, {prompt_length} chars)")
            self._reasoning_enabled_count += 1
            return True

        # Strategic analysis tasks benefit from reasoning
        if task_type in self.REASONING_TASKS:
            logger.info(f"🧠 Reasoning ENABLED: Strategic task ({task_type})")
            self._reasoning_enabled_count += 1
            return True

        # Simple tasks don't need reasoning overhead
        if task_type in self.SIMPLE_TASKS:
            logger.info(f"⚡ Reasoning DISABLED: Simple task ({task_type})")
            self._reasoning_disabled_count += 1
            return False

        # Default: enable reasoning for medium+ complexity
        # Conservative approach - prefer quality over cost
        logger.info(f"🧠 Reasoning ENABLED: Default (medium+ complexity, {task_type})")
        self._reasoning_enabled_count += 1
        return True

    def get_stats(self) -> dict:
        """Get reasoning mode usage statistics"""
        total = self._reasoning_enabled_count + self._reasoning_disabled_count
        enabled_pct = (self._reasoning_enabled_count / total * 100) if total > 0 else 0

        return {
            "reasoning_enabled_count": self._reasoning_enabled_count,
            "reasoning_disabled_count": self._reasoning_disabled_count,
            "total_calls": total,
            "reasoning_enabled_pct": round(enabled_pct, 1),
            "estimated_cost_savings_pct": round((100 - enabled_pct) * 0.4, 1)  # 40% token reduction when disabled
        }


# Singleton instance for global access
_reasoning_mode_selector = None


def get_reasoning_mode_selector() -> ReasoningModeSelector:
    """Get the global reasoning mode selector instance"""
    global _reasoning_mode_selector
    if _reasoning_mode_selector is None:
        _reasoning_mode_selector = ReasoningModeSelector()
    return _reasoning_mode_selector
