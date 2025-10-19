"""
Provider Adapter Pipeline Stage

Filters and adapts parameters for provider-specific API requirements.

Extracted from: unified_client.py provider-specific branches (lines 567-603)
Design: Single Responsibility - Only handles provider parameter adaptation
Complexity Target: CC < 10 (4 providers)
"""

from typing import Dict, Optional, Any
import logging

from ..context import LLMCallContext
from ..stage import PipelineStage


class ProviderAdapterStage(PipelineStage):
    """
    Pipeline stage that adapts parameters for each provider's API.

    This stage:
    1. Checks which provider is being used
    2. Filters parameters based on provider support
    3. Returns context with provider-specific kwargs

    Provider Support Matrix:
    - DeepSeek: ✅ functions, ✅ response_format
    - Anthropic: ❌ functions, ❌ response_format
    - OpenAI: ✅ functions, ✅ response_format
    - OpenRouter: ❌ functions, ✅ response_format

    Behavior (extracted verbatim from unified_client.py):
    - Filters unsupported parameters per provider
    - Logs parameter filtering for transparency
    - Returns modified kwargs dict

    Attributes:
        enabled: Whether stage is enabled (inherited from PipelineStage)

    Example:
        ```python
        stage = ProviderAdapterStage()

        context = LLMCallContext(
            messages=[{"role": "user", "content": "Test"}],
            model="claude-3-5-sonnet-20241022",
            provider="anthropic",
            kwargs={"functions": [...], "response_format": {...}}  # Not supported
        )

        new_context = await stage.execute(context)
        # functions and response_format removed from kwargs
        ```

    References:
        - Original: unified_client.py lines 567-603
        - ADR: docs/adr/ADR-001-unified-client-pipeline-refactoring.md
    """

    @property
    def name(self) -> str:
        """Stage name for logging and metadata."""
        return "ProviderAdapter"

    async def execute(self, context: LLMCallContext) -> LLMCallContext:
        """
        Adapt parameters for provider-specific API requirements.

        Args:
            context: Input context with provider and kwargs

        Returns:
            New context with filtered kwargs

        Workflow:
            1. Get effective provider
            2. Get kwargs from context
            3. Filter parameters based on provider support
            4. Return new context with modified kwargs
        """
        provider = context.get_effective_provider()
        kwargs = context.get_effective_kwargs()  # Returns copy

        # Extract common parameters that may need filtering
        functions = kwargs.get("functions")
        response_format = kwargs.get("response_format")

        # Track what was filtered
        filtered_params = []
        kept_params = []

        # Provider-specific parameter filtering
        if provider == "deepseek":
            # DeepSeek supports functions and response_format
            if functions is not None:
                kept_params.append("functions")
            if response_format is not None:
                kept_params.append("response_format")
                self.logger.info(
                    f"📋 JSON response format enforced: {response_format}"
                )

        elif provider == "anthropic":
            # Anthropic does NOT support functions or response_format
            if functions is not None:
                del kwargs["functions"]
                filtered_params.append("functions")
            if response_format is not None:
                del kwargs["response_format"]
                filtered_params.append("response_format")

            self.logger.info(
                "🎭 Claude provider: functions and response_format parameters excluded"
            )

        elif provider == "openai":
            # OpenAI supports functions and response_format
            if functions is not None:
                kept_params.append("functions")
            if response_format is not None:
                kept_params.append("response_format")

        elif provider == "openrouter":
            # OpenRouter supports response_format but not functions
            if functions is not None:
                del kwargs["functions"]
                filtered_params.append("functions")

            if response_format is not None:
                kept_params.append("response_format")
                self.logger.info(
                    f"📋 Grok structured output enforced: {response_format.get('type', 'json')}"
                )

            self.logger.info(
                "🚀 OpenRouter provider: Using Grok-4-Fast for strategic analysis"
            )

        else:
            # Unknown provider - pass through (fail permissive)
            self.logger.warning(
                f"⚠️ Unknown provider '{provider}', passing parameters unchanged"
            )

        # Add stage metadata
        metadata = {
            "provider": provider,
            "filtered_params": filtered_params,
            "kept_params": kept_params,
            "action": "filtered" if filtered_params else "pass",
        }

        new_context = context.with_stage_metadata(self.name, metadata)

        # Update kwargs if any filtering occurred
        if filtered_params:
            new_context = new_context.with_modified_kwargs(kwargs)
            self.logger.debug(
                f"✅ Provider adapter: {len(filtered_params)} param(s) filtered for {provider}"
            )
        else:
            self.logger.debug(f"✅ Provider adapter: No filtering needed for {provider}")

        return new_context
