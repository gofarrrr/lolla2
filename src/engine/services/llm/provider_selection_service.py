"""
Provider Selection Service - LLM Provider Chain Strategy
=======================================================

REFACTORING TARGET: Extract Grade D complexity from call_best_available_provider()
PATTERN: Strategy Pattern with Provider Chain
GOAL: Reduce call_best_available_provider() from D (23) to B (≤10)

Architecture:
- ProviderSelectionStrategy: Interface for provider selection
- ProviderChainOrchestrator: Manages provider fallback chain
- CacheStrategy: Handles intelligent response caching
- Provider-specific execution strategies

Benefits:
- Single Responsibility Principle per strategy
- Easily testable provider selection logic
- Clear fallback chain management
- Pluggable caching and model selection
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Protocol

# Import domain models from original module
from src.integrations.llm.provider_interface import (
    LLMResponse,
    ProviderUnavailableError,
)

logger = logging.getLogger(__name__)


class ProviderSelectionStrategy(Protocol):
    """Strategy interface for provider selection"""

    def get_providers_to_try(self, available_providers: Dict[str, Any]) -> List[str]:
        """Get ordered list of providers to try"""
        ...


class CacheStrategy(Protocol):
    """Strategy interface for response caching"""

    async def get_cached_response(
        self, cache_key: str, providers: Dict
    ) -> Optional[LLMResponse]:
        """Attempt to get cached response"""
        ...

    async def cache_response(
        self, cache_key: str, response: LLMResponse, provider_name: str, **kwargs
    ) -> None:
        """Cache response for future use"""
        ...


@dataclass
class ProviderCallContext:
    """Context for provider call execution"""

    messages: List[Dict]
    phase: Optional[str]
    engagement_id: Optional[str]
    context_data: Optional[Dict[str, Any]]
    model: Optional[str]
    use_cache: bool
    kwargs: Dict[str, Any]


@dataclass
class ProviderCallResult:
    """Result of provider call attempt"""

    success: bool
    response: Optional[LLMResponse]
    provider_name: str
    error: Optional[Exception]
    cached: bool = False


class CostOptimizedProviderStrategy:
    """
    Cost-optimized provider selection strategy

    Responsibility: Order providers by cost efficiency (DeepSeek first)
    Complexity Target: Grade B (≤10)
    """

    def get_providers_to_try(self, available_providers: Dict[str, Any]) -> List[str]:
        """
        Get cost-optimized provider order (OpenRouter/Grok-4-Fast priority)

        Complexity: Target B (≤10)
        """
        providers_to_try = []

        # OpenRouter/Grok-4-Fast first (PRIMARY - User requested Grok-4-Fast only)
        if "openrouter" in available_providers:
            providers_to_try.append("openrouter")

        # DeepSeek as fallback (if OpenRouter fails)
        if "deepseek" in available_providers:
            providers_to_try.append("deepseek")

        # Anthropic as high-quality fallback
        if "anthropic" in available_providers:
            providers_to_try.append("anthropic")

        # OpenAI as final fallback
        if "openai" in available_providers:
            providers_to_try.append("openai")

        return providers_to_try


class IntelligentCacheStrategy:
    """
    Intelligent response caching strategy

    Responsibility: Cache and retrieve LLM responses efficiently
    Complexity Target: Grade B (≤10)
    """

    def generate_cache_key(self, context: ProviderCallContext) -> str:
        """
        Generate deterministic cache key

        Complexity: Target B (≤10)
        """
        cache_data = {
            "messages": context.messages,
            "phase": context.phase,
            "model": context.model,
            "temperature": context.kwargs.get("temperature", 0.3),
            "max_tokens": context.kwargs.get("max_tokens", 2000),
            "requires_reasoning": (
                context.context_data.get("requires_reasoning", False)
                if context.context_data
                else False
            ),
        }

        json_str = json.dumps(cache_data, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    async def get_cached_response(
        self, cache_key: str, providers: Dict
    ) -> Optional[LLMResponse]:
        """
        Attempt to get cached response from providers

        Complexity: Target B (≤10)
        """
        try:
            # Try different provider caches based on likely provider
            for provider_name in ["deepseek", "anthropic", "openai"]:
                if provider_name in providers:
                    provider = providers[provider_name]
                    if hasattr(provider, "get_cached_response"):
                        cached = await provider.get_cached_response(cache_key)
                        if cached:
                            logger.info(f"✅ Cache hit for {provider_name}")
                            return cached

            return None

        except Exception as e:
            logger.warning(f"⚠️ Cache retrieval error: {e}")
            return None

    async def cache_response(
        self,
        cache_key: str,
        response: LLMResponse,
        provider_name: str,
        providers: Dict,
        **kwargs,
    ) -> None:
        """
        Cache response with provider

        Complexity: Target B (≤10)
        """
        try:
            if provider_name in providers:
                provider = providers[provider_name]
                if hasattr(provider, "cache_response"):
                    await provider.cache_response(cache_key, response, **kwargs)

        except Exception as e:
            logger.warning(f"⚠️ Cache storage error: {e}")


class ModelSelectionStrategy:
    """
    Model selection strategy per provider

    Responsibility: Select optimal model based on provider and task
    Complexity Target: Grade B (≤10)
    """

    def select_model(self, provider_name: str, context: ProviderCallContext) -> str:
        """
        Select optimal model for provider and context

        Complexity: Target B (≤10)
        """
        if context.model:
            return context.model

        if provider_name == "openrouter":
            return "x-ai/grok-4-fast"  # Default to paid x-ai/grok-4-fast
        elif provider_name == "anthropic":
            return "claude-3-5-sonnet-20241022"
        elif provider_name == "deepseek":
            return self._select_deepseek_model(context)
        elif provider_name == "openai":
            return "gpt-4o-mini"
        else:
            return "default-model"

    def _select_deepseek_model(self, context: ProviderCallContext) -> str:
        """
        Select DeepSeek V3.1 model variant

        Complexity: Target B (≤10)
        """
        complex_phases = [
            "hypothesis_generation",
            "analysis_execution",
            "research_synthesis",
        ]

        # Use reasoning mode for complex tasks
        if context.phase in complex_phases or (
            context.context_data
            and context.context_data.get("requires_reasoning", False)
        ):
            return "deepseek-reasoner"  # V3.1 thinking mode
        else:
            return "deepseek-chat"  # V3.1 non-thinking mode (faster)


class ProviderCallExecutor:
    """
    Individual provider call executor

    Responsibility: Execute call with specific provider
    Complexity Target: Grade B (≤10)
    """

    def __init__(self, model_selector: ModelSelectionStrategy):
        self.model_selector = model_selector

    async def execute_provider_call(
        self, provider_name: str, provider: Any, context: ProviderCallContext, *, is_fallback: bool = False
    ) -> ProviderCallResult:
        """
        Execute call with specific provider

        Complexity: Target B (≤10)
        """
        try:
            # Check provider availability
            if not await provider.is_available():
                logger.warning(f"⚠️ {provider_name} provider not available")
                return ProviderCallResult(
                    success=False,
                    response=None,
                    provider_name=provider_name,
                    error=Exception("Provider not available"),
                )

            # Select model
            model = self.model_selector.select_model(provider_name, context)

            # Prepare kwargs and apply fallback cost controls for DeepSeek
            call_kwargs = dict(context.kwargs or {})
            if provider_name == "deepseek" and is_fallback:
                # Cap tokens/temperature on fallback to control cost
                max_tokens = call_kwargs.get("max_tokens", 2000)
                temperature = call_kwargs.get("temperature", 0.7)
                call_kwargs["max_tokens"] = min(int(max_tokens or 2000), 800)
                call_kwargs["temperature"] = min(float(temperature or 0.7), 0.4)
                logger.info(
                    f"💸 DeepSeek fallback: capping params max_tokens={call_kwargs['max_tokens']}, "
                    f"temperature={call_kwargs['temperature']}"
                )

            # Make provider call
            if provider_name in ["anthropic", "deepseek"]:
                response = await provider.call_llm(
                    context.messages,
                    model,
                    phase=context.phase,
                    engagement_id=context.engagement_id,
                    context_data=context.context_data,
                    **call_kwargs,
                )
            elif provider_name == "openrouter":
                # OpenRouter uses simple call_llm interface
                response = await provider.call_llm(
                    context.messages, model, **call_kwargs
                )
            else:
                response = await provider.call_llm(
                    context.messages, model, **call_kwargs
                )

            logger.info(f"✅ {provider_name} call successful")

            return ProviderCallResult(
                success=True, response=response, provider_name=provider_name, error=None
            )

        except Exception as e:
            logger.error(f"❌ {provider_name} provider failed: {e}")
            return ProviderCallResult(
                success=False, response=None, provider_name=provider_name, error=e
            )


class ProviderChainOrchestrator:
    """
    Provider chain orchestrator using strategy composition

    Responsibility: Coordinate provider selection, caching, and execution
    Complexity Target: Grade B (≤10)
    """

    def __init__(self):
        self.provider_strategy = CostOptimizedProviderStrategy()
        self.cache_strategy = IntelligentCacheStrategy()
        self.model_selector = ModelSelectionStrategy()
        self.call_executor = ProviderCallExecutor(self.model_selector)

    async def call_best_available_provider(
        self, providers: Dict[str, Any], context: ProviderCallContext
    ) -> LLMResponse:
        """
        Execute provider chain with fallback logic

        Complexity: Target B (≤10) - Pure orchestration
        """
        # Try cache first if enabled
        if context.use_cache:
            cache_key = self.cache_strategy.generate_cache_key(context)
            cached_response = await self.cache_strategy.get_cached_response(
                cache_key, providers
            )
            if cached_response:
                return cached_response

        # Get provider chain (respect explicit preference from context if provided)
        preferred_chain = context.kwargs.get("provider_preference") if context.kwargs else None
        if preferred_chain and isinstance(preferred_chain, list):
            # Filter to available providers, preserve order
            providers_to_try = [p for p in preferred_chain if p in providers]
        else:
            providers_to_try = self.provider_strategy.get_providers_to_try(providers)

        if not providers_to_try:
            raise ProviderUnavailableError("No LLM providers available")

        def _classify_reason(err: Exception) -> str:
            msg = str(err).lower() if err else ""
            if "429" in msg or "rate limit" in msg:
                return "rate_limit"
            if "timeout" in msg or "timed out" in msg:
                return "timeout"
            if "model" in msg and ("not found" in msg or "invalid" in msg):
                return "invalid_model"
            if "unavailable" in msg or "service" in msg:
                return "unavailable"
            return "error"

        # Execute provider chain
        last_error = None
        from src.telemetry.counters import fallback_counter

        for idx, provider_name in enumerate(providers_to_try):
            provider = providers[provider_name]
            result = await self.call_executor.execute_provider_call(
                provider_name, provider, context, is_fallback=(idx > 0)
            )

            if result.success:
                # Cache response if enabled
                if context.use_cache:
                    await self.cache_strategy.cache_response(
                        cache_key,
                        result.response,
                        provider_name,
                        providers,
                        **context.kwargs,
                    )

                # If succeeded on a fallback provider, record success
                if idx > 0:
                    fallback_counter.increment("succeeded")

                return result.response
            else:
                # Log fallback intent if more providers remain
                last_error = result.error
                if idx < len(providers_to_try) - 1:
                    next_provider = providers_to_try[idx + 1]
                    reason = _classify_reason(last_error)
                    logger.warning(
                        f"🔁 Fallback: {provider_name} failed (reason={reason}). Trying {next_provider}..."
                    )
                    fallback_counter.increment("attempted")

        # All providers failed
        raise ProviderUnavailableError(
            f"All providers failed. Last error: {last_error}"
        )


# Singleton instance for injection
_orchestrator_instance = None


def get_provider_chain_orchestrator() -> ProviderChainOrchestrator:
    """Factory function for dependency injection"""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = ProviderChainOrchestrator()
    return _orchestrator_instance
