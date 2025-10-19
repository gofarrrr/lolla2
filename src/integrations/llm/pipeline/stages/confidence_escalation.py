"""
Confidence Escalation Pipeline Stage

Escalates to stronger provider if confidence is too low.

Extracted from: unified_client.py confidence escalation (lines 889-891, 404-440)
Design: Single Responsibility - Only handles confidence-based escalation
Complexity Target: CC < 5
"""

from typing import Optional, Any, Callable
import logging

from ..context import LLMCallContext
from ..stage import PipelineStage


class ConfidenceEscalationStage(PipelineStage):
    """
    Pipeline stage that escalates on low confidence.

    This stage:
    1. Checks response confidence score
    2. If below threshold (default 0.85), escalates
    3. Retries with stronger provider (Anthropic → OpenAI)
    4. Returns escalated response or original

    Attributes:
        escalate_func: Function to retry with different provider
        threshold: Confidence threshold (default 0.85)
        enabled: Whether stage is enabled

    References:
        - Original: unified_client.py lines 889-891, 404-440
    """

    def __init__(
        self,
        escalate_func: Optional[Callable] = None,
        threshold: float = 0.85,
        enabled: bool = True
    ):
        super().__init__(enabled=enabled)
        self.escalate_func = escalate_func
        self.threshold = threshold

    @property
    def name(self) -> str:
        return "ConfidenceEscalation"

    async def execute(self, context: LLMCallContext) -> LLMCallContext:
        """Escalate on low confidence."""
        # No-op if not configured
        if not self.escalate_func:
            return context

        # Need response to check confidence
        if not context.has_response():
            return context

        try:
            response = context.response
            confidence = response.confidence if hasattr(response, 'confidence') else 1.0

            # Check threshold
            if confidence >= self.threshold:
                return context.with_stage_metadata(self.name, {
                    "action": "pass",
                    "confidence": confidence,
                    "threshold": self.threshold,
                })

            # Escalate
            self.logger.warning(
                f"⚠️ Low confidence ({confidence:.2f} < {self.threshold}), escalating..."
            )

            # Call escalation function (typically retries with Anthropic)
            escalated_response = await self.escalate_func(
                context.get_effective_messages(),
                response,
                context.get_effective_provider(),
                context.get_effective_model(),
                context.get_effective_kwargs()
            )

            # Update context with escalated response
            return context.with_response(escalated_response).with_stage_metadata(self.name, {
                "action": "escalated",
                "original_confidence": confidence,
                "threshold": self.threshold,
            })

        except Exception as e:
            self.logger.warning(f"⚠️ Confidence escalation error: {e}")
            return context.with_error(f"Confidence escalation error: {str(e)}")
