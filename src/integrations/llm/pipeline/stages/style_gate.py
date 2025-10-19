"""
Style Gate Pipeline Stage

Scores response style and applies policy gates (allow/warn/block).

Extracted from: unified_client.py style scoring logic (lines 676-694)
Design: Single Responsibility - Only handles style scoring and gating
Complexity Target: CC < 5
"""

from typing import Optional, Any
import logging

from ..context import LLMCallContext
from ..stage import PipelineStage


class StyleGateStage(PipelineStage):
    """
    Pipeline stage that scores and gates response style.

    This stage:
    1. Scores response style (0.0-1.0)
    2. Evaluates against style policy for current phase
    3. Takes action: ALLOW, WARN, or BLOCK
    4. If BLOCK: replaces content and lowers confidence

    Attributes:
        style_scorer: Function to score style
        style_gate: StyleGate policy evaluator
        enabled: Whether stage is enabled

    References:
        - Original: unified_client.py lines 676-694
    """

    def __init__(
        self,
        style_scorer: Optional[Any] = None,
        style_gate: Optional[Any] = None,
        enabled: bool = True
    ):
        super().__init__(enabled=enabled)
        self.style_scorer = style_scorer
        self.style_gate = style_gate

    @property
    def name(self) -> str:
        return "StyleGate"

    async def execute(self, context: LLMCallContext) -> LLMCallContext:
        """Score style and apply gate policy."""
        # No-op if not configured
        if not self.style_scorer or not self.style_gate:
            return context

        # Need response to score
        if not context.has_response():
            return context

        try:
            response = context.response
            content = response.content if hasattr(response, 'content') else str(response)

            # Score style
            style_score = self.style_scorer(content)

            # Evaluate gate policy
            phase = context.kwargs.get("phase", "general")
            gate_result = self.style_gate.evaluate(style_score, phase)

            # Add metadata
            metadata = {
                "style_score": style_score,
                "gate_action": gate_result.action.value if hasattr(gate_result.action, 'value') else str(gate_result.action),
                "phase": phase,
            }

            new_context = context.with_stage_metadata(self.name, metadata)

            # Apply gate action
            if str(gate_result.action).lower() == "block":
                # Block: replace content and lower confidence
                if hasattr(response, 'content'):
                    response.content = "Output blocked by style policy"
                if hasattr(response, 'confidence'):
                    response.confidence = 0.2

                self.logger.warning(f"🚫 Style gate BLOCKED (score={style_score:.2f})")

            return new_context

        except Exception as e:
            self.logger.warning(f"⚠️ Style gate error: {e}")
            return context.with_error(f"Style gate error: {str(e)}")
