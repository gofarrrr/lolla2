"""
Sensitivity Routing Pipeline Stage

Routes requests to appropriate providers based on data sensitivity level.

Extracted from: unified_client.py::_apply_sensitivity_routing() (lines 1171-1229)
Design: Single Responsibility - Only handles sensitivity-based provider routing
Complexity Target: CC < 5
"""

from typing import List, Dict, Optional, Any
import logging

from ..context import LLMCallContext
from ..stage import PipelineStage


class SensitivityRoutingStage(PipelineStage):
    """
    Pipeline stage that routes requests based on data sensitivity.

    This stage:
    1. Analyzes message content for sensitivity level
    2. Checks if requested provider is allowed for this sensitivity
    3. Overrides provider if necessary (e.g., critical data → Anthropic only)

    Behavior (extracted verbatim from unified_client.py):
    - Extracts content from messages for sensitivity detection
    - Routes via SensitivityRouter
    - Overrides provider if requested provider not in allowed list
    - Logs override with sensitivity level and reasons
    - Emits glass-box event for transparency

    Dependencies:
        - src.engine.security.sensitivity_router (SensitivityRouter instance)
        - available_providers: List of provider keys available in system

    Attributes:
        sensitivity_router: SensitivityRouter instance (optional, can be None if disabled)
        available_providers: List of available provider keys
        enabled: Whether stage is enabled (inherited from PipelineStage)

    Example:
        ```python
        from src.engine.security.sensitivity_router import get_sensitivity_router

        router = get_sensitivity_router()
        stage = SensitivityRoutingStage(
            sensitivity_router=router,
            available_providers=["deepseek", "anthropic", "openai"]
        )

        context = LLMCallContext(
            messages=[{"role": "user", "content": "Analyze confidential data"}],
            model="deepseek-chat",
            provider="deepseek",  # Requested
            kwargs={}
        )

        new_context = await stage.execute(context)
        # May override: new_context.modified_provider == "anthropic" (higher security)
        ```

    References:
        - Original: unified_client.py lines 519-522, 1171-1229
        - ADR: docs/adr/ADR-001-unified-client-pipeline-refactoring.md
    """

    def __init__(
        self,
        sensitivity_router: Optional[Any] = None,
        available_providers: Optional[List[str]] = None,
        enabled: bool = True
    ):
        """
        Initialize sensitivity routing stage.

        Args:
            sensitivity_router: SensitivityRouter instance (or None to disable)
            available_providers: List of available provider keys (e.g., ["deepseek", "anthropic"])
            enabled: Whether stage is enabled

        Note:
            If sensitivity_router is None, stage becomes a no-op (returns context unchanged).
        """
        super().__init__(enabled=enabled)
        self.sensitivity_router = sensitivity_router
        self.available_providers = available_providers or []

    @property
    def name(self) -> str:
        """Stage name for logging and metadata."""
        return "SensitivityRouting"

    async def execute(self, context: LLMCallContext) -> LLMCallContext:
        """
        Route request to appropriate provider based on sensitivity.

        This implementation is extracted verbatim from unified_client.py
        with minimal changes to adapt to pipeline context pattern.

        Args:
            context: Input context with messages and provider

        Returns:
            New context with modified_provider set (if override occurred)

        Workflow:
            1. Extract content from messages for sensitivity analysis
            2. Call sensitivity router to get routing decision
            3. Check if requested provider is in allowed list
            4. Override provider if not allowed
            5. Log and emit glass-box event if override occurred
        """
        # No-op if router not provided
        if not self.sensitivity_router:
            self.logger.debug("Sensitivity router not configured, skipping")
            return context

        try:
            messages = context.get_effective_messages()
            requested_provider = context.get_effective_provider()

            # Extract content for sensitivity detection
            content = " ".join(
                msg.get("content", "") for msg in messages if isinstance(msg.get("content"), str)
            )

            # Get sensitivity override from kwargs if provided
            sensitivity_override = context.kwargs.get("sensitivity_override")

            # Make routing decision
            decision = self.sensitivity_router.route(
                content=content,
                context=context.kwargs,  # Pass kwargs as context
                sensitivity_override=sensitivity_override,
                available_providers=self.available_providers
            )

            # Check if requested provider is allowed
            if requested_provider not in decision.allowed_providers:
                # Override to first allowed provider
                overridden_provider = (
                    decision.allowed_providers[0] if decision.allowed_providers else requested_provider
                )

                self.logger.warning(
                    f"🔐 SENSITIVITY ROUTING: {requested_provider} → {overridden_provider} "
                    f"(level={decision.sensitivity_level.value}, "
                    f"reason={decision.reasons[0] if decision.reasons else 'policy'})"
                )

                # Add stage metadata
                metadata = {
                    "action": "override",
                    "original_provider": requested_provider,
                    "routed_provider": overridden_provider,
                    "sensitivity_level": decision.sensitivity_level.value,
                    "reasons": decision.reasons,
                    "restrictions": decision.restrictions,
                }

                new_context = context.with_stage_metadata(self.name, metadata)

                # Override provider
                new_context = new_context.with_modified_provider(overridden_provider)

                # Emit glass-box event
                self._emit_glass_box_event(
                    requested_provider, overridden_provider, decision
                )

                return new_context

            else:
                # Provider allowed, no override needed
                self.logger.debug(
                    f"✅ Sensitivity routing: {requested_provider} allowed "
                    f"(level={decision.sensitivity_level.value})"
                )

                # Add stage metadata (no override)
                metadata = {
                    "action": "pass",
                    "provider": requested_provider,
                    "sensitivity_level": decision.sensitivity_level.value,
                }

                return context.with_stage_metadata(self.name, metadata)

        except Exception as e:
            # Fail open: Log error but don't halt pipeline
            self.logger.warning(f"⚠️ Sensitivity routing error: {e}", exc_info=True)

            # Add error to context (non-fatal)
            error_context = context.with_error(f"Sensitivity routing error: {str(e)}")

            # Add failure metadata
            error_context = error_context.with_stage_metadata(self.name, {
                "action": "error_fail_open",
                "error": str(e),
            })

            return error_context

    def _emit_glass_box_event(
        self,
        original_provider: str,
        routed_provider: str,
        decision: Any
    ) -> None:
        """
        Emit glass-box transparency event for provider override.

        Args:
            original_provider: Originally requested provider
            routed_provider: Provider after override
            decision: SensitivityRoutingDecision object

        Note:
            This is a best-effort operation. Failures are logged but not propagated.
        """
        try:
            from src.core.unified_context_stream import get_unified_context_stream, ContextEventType

            cs = get_unified_context_stream()
            cs.add_event(ContextEventType.ERROR, {  # Using ERROR as proxy for security event
                "event_type": "sensitivity_routing_override",
                "original_provider": original_provider,
                "routed_provider": routed_provider,
                "sensitivity_level": decision.sensitivity_level.value,
                "reasons": decision.reasons,
                "restrictions": decision.restrictions
            })

        except Exception as e:
            # Glass-box logging is best-effort, don't fail stage
            self.logger.debug(f"Failed to emit glass-box event: {e}")
