"""
Reasoning Mode Selection Pipeline Stage

Determines if extended reasoning should be enabled for Grok models.

Extracted from: unified_client.py OpenRouter reasoning logic (lines 604-663)
Design: Single Responsibility - Only handles reasoning mode selection
Complexity Target: CC < 5
"""

from typing import Optional, Any
import logging

from ..context import LLMCallContext
from ..stage import PipelineStage


class ReasoningModeStage(PipelineStage):
    """
    Pipeline stage that selects reasoning mode for OpenRouter/Grok.

    This stage:
    1. Checks if provider is OpenRouter (Grok models)
    2. Analyzes task complexity (type, length, multi-step, impact)
    3. Decides if extended reasoning should be enabled
    4. Adds reasoning_enabled parameter to kwargs

    Behavior (extracted from unified_client.py):
    - Only applies to OpenRouter provider
    - Uses ReasoningModeSelector to make decision
    - Supports explicit override via reasoning_enabled_override
    - Logs decision to UnifiedContextStream
    - Tracks stats (% enabled, cost savings)

    Attributes:
        reasoning_selector: ReasoningModeSelector instance
        enabled: Whether stage is enabled

    References:
        - Original: unified_client.py lines 604-663
    """

    def __init__(
        self,
        reasoning_selector: Optional[Any] = None,
        enabled: bool = True
    ):
        super().__init__(enabled=enabled)
        self.reasoning_selector = reasoning_selector

    @property
    def name(self) -> str:
        return "ReasoningMode"

    async def execute(self, context: LLMCallContext) -> LLMCallContext:
        """Select reasoning mode for OpenRouter/Grok."""
        # Only applies to OpenRouter
        provider = context.get_effective_provider()
        if provider != "openrouter":
            return context

        # No-op if selector not provided
        if not self.reasoning_selector:
            self.logger.debug("Reasoning selector not configured, skipping")
            return context

        try:
            kwargs = context.get_effective_kwargs()
            messages = context.get_effective_messages()

            # Extract task metadata
            task_type = kwargs.get("task_type", "general")
            prompt_length = len(messages[0].get("content", "")) if messages else 0
            requires_multi_step = kwargs.get("requires_multi_step", False)
            stakeholder_impact = kwargs.get("stakeholder_impact", "medium")

            # Check for explicit override
            explicit_override = kwargs.get("reasoning_enabled_override")

            # Determine reasoning mode
            if explicit_override is not None:
                reasoning_enabled = explicit_override
                self.logger.info(f"🎯 Reasoning mode EXPLICIT OVERRIDE: {reasoning_enabled}")
            else:
                reasoning_enabled = self.reasoning_selector.should_enable_reasoning(
                    task_type=task_type,
                    prompt_length=prompt_length,
                    requires_multi_step=requires_multi_step,
                    stakeholder_impact=stakeholder_impact
                )

            # Add to kwargs
            kwargs["reasoning_enabled"] = reasoning_enabled

            # Add metadata
            metadata = {
                "reasoning_enabled": reasoning_enabled,
                "task_type": task_type,
                "prompt_length": prompt_length,
                "requires_multi_step": requires_multi_step,
                "stakeholder_impact": stakeholder_impact,
                "explicit_override": explicit_override is not None,
            }

            new_context = context.with_stage_metadata(self.name, metadata)
            new_context = new_context.with_modified_kwargs(kwargs)

            # Emit glass-box event
            try:
                from src.core.unified_context_stream import get_unified_context_stream, ContextEventType
                cs = get_unified_context_stream()
                cs.add_event(ContextEventType.REASONING_MODE_DECISION, metadata)
            except Exception:
                pass

            # Log stats periodically
            try:
                stats = self.reasoning_selector.get_stats()
                if stats["total_calls"] % 10 == 0:
                    self.logger.info(
                        f"📊 Reasoning Mode Stats: {stats['reasoning_enabled_pct']}% enabled, "
                        f"~{stats['estimated_cost_savings_pct']}% cost savings"
                    )
            except Exception:
                pass

            return new_context

        except Exception as e:
            self.logger.warning(f"⚠️ Reasoning mode selector failed: {e}")
            # Fallback: enable reasoning by default
            kwargs = context.get_effective_kwargs()
            kwargs["reasoning_enabled"] = True
            return context.with_modified_kwargs(kwargs)
