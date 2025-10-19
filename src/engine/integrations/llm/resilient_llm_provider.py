#!/usr/bin/env python3
"""
Resilient LLM Provider with DeepSeek → Claude 3.5 Sonnet Fallback

This provider implements intelligent fallback logic:
1. Try DeepSeek V3.1 with reasonable timeouts (30s)
2. If timeout/failure, fallback to Claude 3.5 Sonnet
3. Capture all attempts and reasons for fallback
4. Maintain optimization quality regardless of provider used
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from .deepseek_v31_optimized_provider import (
    DeepSeekV31OptimizedProvider,
)
from .openrouter_deepseek_provider import OpenRouterDeepSeekProvider
from .provider_interface import (
    LLMResponse,
    ProviderError,
    ProviderAPIError,
)


class FallbackReason(str, Enum):
    """Reasons for fallback activation"""

    TIMEOUT = "timeout"
    API_ERROR = "api_error"
    RATE_LIMIT = "rate_limit"
    SERVICE_UNAVAILABLE = "service_unavailable"
    CONNECTION_ERROR = "connection_error"
    PROVIDER_OVERRIDE = "provider_override"


@dataclass
class FallbackAttempt:
    """Record of fallback attempt"""

    timestamp: str
    primary_provider: str
    fallback_provider: str
    reason: FallbackReason
    primary_error: Optional[str]
    success: bool
    response_time_ms: int


class ResilientLLMProvider:
    """
    Resilient LLM Provider with intelligent DeepSeek → Claude fallback

    Features:
    - Fast timeouts with immediate fallback (30s DeepSeek, 60s Claude)
    - Intelligent error classification and retry logic
    - Optimization preservation across providers
    - Complete audit trail of fallback decisions
    - Graceful degradation under any conditions
    """

    def __init__(
        self,
        deepseek_api_key: str = None,
        anthropic_api_key: str = None,
        openrouter_api_key: str = None,
    ):
        # Initialize logger first
        self.logger = logging.getLogger(__name__)

        self.deepseek_key = deepseek_api_key
        self.anthropic_key = anthropic_api_key
        self.openrouter_key = openrouter_api_key

        # Configuration - Claude as primary with DeepSeek optimizations for reliability
        self.config = {
            "deepseek_timeout": 15.0,  # DeepSeek servers overloaded
            "openrouter_timeout": 20.0,  # OpenRouter also slow due to DeepSeek backend
            "claude_timeout": 60.0,  # Primary provider gets more time
            "max_retries": 1,
            "enable_fallback": True,
            "prefer_claude": True,  # Claude most reliable with DeepSeek optimizations
            "prefer_openrouter": False,  # Deprioritize due to server issues
            "prefer_deepseek": False,  # Deprioritize direct DeepSeek due to server load
        }

        # Tracking
        self.fallback_history = []
        self.provider_stats = {
            "deepseek": {
                "calls": 0,
                "successes": 0,
                "failures": 0,
                "avg_response_time": 0,
            },
            "openrouter": {
                "calls": 0,
                "successes": 0,
                "failures": 0,
                "avg_response_time": 0,
            },
            "claude": {
                "calls": 0,
                "successes": 0,
                "failures": 0,
                "avg_response_time": 0,
            },
        }

        # Initialize providers
        self.deepseek_provider = None
        self.openrouter_provider = None
        self.claude_provider = None

        self._initialize_providers()

        self.logger.info(
            "🚀 Resilient LLM Provider initialized - Claude primary with DeepSeek optimizations, 3-provider fallback"
        )

    def _initialize_providers(self):
        """Initialize all three providers with error handling"""
        # DeepSeek Direct
        try:
            if self.deepseek_key:
                self.deepseek_provider = DeepSeekV31OptimizedProvider(self.deepseek_key)
                self.logger.info("✅ DeepSeek V3.1 Direct provider initialized")
            else:
                self.logger.warning("⚠️ DeepSeek API key not provided")
        except Exception as e:
            self.logger.error(f"❌ DeepSeek provider initialization failed: {e}")

        # OpenRouter DeepSeek
        try:
            if self.openrouter_key:
                self.openrouter_provider = OpenRouterDeepSeekProvider(
                    self.openrouter_key
                )
                self.logger.info("✅ OpenRouter DeepSeek provider initialized")
            else:
                self.logger.warning("⚠️ OpenRouter API key not provided")
        except Exception as e:
            self.logger.error(f"❌ OpenRouter provider initialization failed: {e}")

        # Claude 3.5 Sonnet
        try:
            if self.anthropic_key:
                self.claude_provider = self._create_claude_provider()
                self.logger.info("✅ Claude 3.5 Sonnet provider initialized")
            else:
                self.logger.warning("⚠️ Anthropic API key not provided")
        except Exception as e:
            self.logger.error(f"❌ Claude provider initialization failed: {e}")

    def _create_claude_provider(self):
        """Create Claude provider with compatible interface"""
        import anthropic

        class ClaudeProvider:
            def __init__(self, api_key: str):
                self.client = anthropic.Anthropic(api_key=api_key)
                self.model = "claude-3-5-sonnet-20241022"

            async def call_optimized_llm(
                self,
                prompt: str,
                task_type: str = "balanced_chat",
                consultant_role: str = None,
                complexity_score: float = 0.5,
                custom_system_prompt: str = None,
                **kwargs,
            ) -> LLMResponse:
                """Claude call with DeepSeek-compatible interface"""
                start_time = datetime.now()

                try:
                    # Build system prompt
                    if custom_system_prompt:
                        system_prompt = custom_system_prompt
                    else:
                        system_prompt = self._build_claude_system_prompt(
                            task_type, consultant_role
                        )

                    # Make Claude API call
                    response = await asyncio.to_thread(
                        self.client.messages.create,
                        model=self.model,
                        max_tokens=4000,
                        temperature=self._get_temperature_for_task(task_type),
                        system=system_prompt,
                        messages=[{"role": "user", "content": prompt}],
                    )

                    # Calculate metrics
                    end_time = datetime.now()
                    response_time = (end_time - start_time).total_seconds() * 1000

                    # Estimate cost (Claude 3.5 Sonnet pricing)
                    input_tokens = response.usage.input_tokens
                    output_tokens = response.usage.output_tokens
                    cost = (
                        input_tokens * 3.0 + output_tokens * 15.0
                    ) / 1_000_000  # $3/$15 per 1M tokens

                    return LLMResponse(
                        content=response.content[0].text,
                        provider="claude-3.5-sonnet",
                        model=self.model,
                        tokens_used=input_tokens + output_tokens,
                        cost_usd=cost,
                        response_time_ms=int(response_time),
                        reasoning_steps=[],
                        mental_models=[],
                        confidence=0.85,  # Claude generally reliable
                    )

                except Exception as e:
                    response_time = (datetime.now() - start_time).total_seconds() * 1000
                    raise ProviderAPIError(f"Claude API error: {e}")

            def _build_claude_system_prompt(
                self, task_type: str, consultant_role: str
            ) -> str:
                """Build system prompt optimized for Claude"""
                base_prompt = "You are an expert strategic consultant providing high-quality analysis."

                if consultant_role == "strategic_analyst":
                    base_prompt += " You specialize in strategic analysis using frameworks like Porter's Five Forces, BCG Matrix, and McKinsey problem-solving methodologies."
                elif consultant_role == "implementation_driver":
                    base_prompt += " You specialize in execution excellence, project management, and operational implementation using Bain & Company methodologies."
                elif consultant_role == "devils_advocate":
                    base_prompt += " You are a critical thinker who challenges assumptions, identifies blind spots, and stress-tests strategic recommendations."
                elif consultant_role == "senior_advisor":
                    base_prompt += " You are a Senior Managing Partner who synthesizes multiple perspectives into definitive executive recommendations."

                if task_type == "strategic_analysis":
                    base_prompt += " Focus on strategic clarity, market dynamics, competitive positioning, and long-term value creation."
                elif task_type == "implementation":
                    base_prompt += " Focus on practical execution, resource requirements, timeline feasibility, and operational excellence."
                elif task_type == "assumption_challenge":
                    base_prompt += " Focus on rigorous critique, identifying flawed assumptions, and stress-testing logical consistency."

                return base_prompt

            def _get_temperature_for_task(self, task_type: str) -> float:
                """Get optimal temperature for task type"""
                temperature_map = {
                    "strategic_analysis": 0.3,
                    "implementation": 0.2,
                    "assumption_challenge": 0.6,
                    "balanced_chat": 0.4,
                    "innovation": 0.7,
                    "research_synthesis": 0.3,
                }
                return temperature_map.get(task_type, 0.4)

        return ClaudeProvider(self.anthropic_key)

    async def call_optimized_llm(
        self,
        prompt: str,
        task_type: str = "balanced_chat",
        consultant_role: str = None,
        complexity_score: float = 0.5,
        custom_system_prompt: str = None,
        force_provider: str = None,
        **kwargs,
    ) -> Tuple[LLMResponse, List[FallbackAttempt]]:
        """
        Make resilient LLM call with intelligent fallback

        Returns:
            Tuple of (LLMResponse, List[FallbackAttempt])
        """

        attempts = []

        # Determine provider order - 3-provider fallback chain
        if force_provider == "claude":
            providers = ["claude"]
        elif force_provider == "deepseek":
            providers = ["deepseek"]
        elif force_provider == "openrouter":
            providers = ["openrouter"]
        else:
            # Intelligent provider ordering - Claude first with DeepSeek optimizations
            available_providers = []

            # Claude first - most reliable with DeepSeek prompt optimizations
            if self.claude_provider and self.config.get("prefer_claude", True):
                available_providers.append("claude")

            # OpenRouter as secondary (when servers stabilize)
            if self.openrouter_provider and self.config.get("prefer_openrouter", False):
                available_providers.append("openrouter")

            # DeepSeek direct as tertiary (if preferred and available)
            if self.deepseek_provider and self.config.get("prefer_deepseek", False):
                available_providers.append("deepseek")

            # Add remaining providers not already included
            if self.openrouter_provider and "openrouter" not in available_providers:
                available_providers.append("openrouter")
            if self.deepseek_provider and "deepseek" not in available_providers:
                available_providers.append("deepseek")

            providers = available_providers if available_providers else ["claude"]

        last_error = None

        for i, provider_name in enumerate(providers):
            is_fallback = i > 0

            try:
                self.logger.info(
                    f"🔄 Attempting {provider_name} {'(fallback)' if is_fallback else '(primary)'}"
                )

                if provider_name == "deepseek":
                    response = await self._call_deepseek_with_timeout(
                        prompt,
                        task_type,
                        consultant_role,
                        complexity_score,
                        custom_system_prompt,
                        **kwargs,
                    )
                elif provider_name == "openrouter":
                    response = await self._call_openrouter_with_timeout(
                        prompt,
                        task_type,
                        consultant_role,
                        complexity_score,
                        custom_system_prompt,
                        **kwargs,
                    )
                else:  # claude
                    response = await self._call_claude_with_timeout(
                        prompt,
                        task_type,
                        consultant_role,
                        complexity_score,
                        custom_system_prompt,
                        **kwargs,
                    )

                # Success!
                self._record_success(provider_name, response.response_time_ms)

                if is_fallback:
                    fallback_reason = self._classify_error(last_error)
                    attempt = FallbackAttempt(
                        timestamp=datetime.now().isoformat(),
                        primary_provider=providers[0],
                        fallback_provider=provider_name,
                        reason=fallback_reason,
                        primary_error=str(last_error),
                        success=True,
                        response_time_ms=response.response_time_ms,
                    )
                    attempts.append(attempt)
                    self.fallback_history.append(attempt)
                    self.logger.info(
                        f"✅ Fallback to {provider_name} successful ({fallback_reason})"
                    )

                return response, attempts

            except Exception as e:
                self.logger.warning(f"⚠️ {provider_name} failed: {e}")
                self._record_failure(provider_name)
                last_error = e

                if is_fallback:
                    # Final fallback failed
                    attempt = FallbackAttempt(
                        timestamp=datetime.now().isoformat(),
                        primary_provider=providers[0],
                        fallback_provider=provider_name,
                        reason=self._classify_error(e),
                        primary_error=str(last_error),
                        success=False,
                        response_time_ms=0,
                    )
                    attempts.append(attempt)
                    self.fallback_history.append(attempt)

                continue

        # All providers failed
        self.logger.error(f"❌ All providers failed. Last error: {last_error}")

        # Return graceful failure response
        fallback_response = LLMResponse(
            content=f"Analysis unavailable due to provider failures. Last error: {str(last_error)[:200]}",
            provider="fallback_failure",
            model="unavailable",
            tokens_used=0,
            cost_usd=0.0,
            response_time_ms=0,
            reasoning_steps=[],
            mental_models=[],
            confidence=0.0,
        )

        return fallback_response, attempts

    async def _call_deepseek_with_timeout(
        self,
        prompt: str,
        task_type: str,
        consultant_role: str,
        complexity_score: float,
        custom_system_prompt: str,
        **kwargs,
    ) -> LLMResponse:
        """Call DeepSeek with fast timeout"""
        try:
            return await asyncio.wait_for(
                self.deepseek_provider.call_optimized_llm(
                    prompt=prompt,
                    task_type=task_type,
                    consultant_role=consultant_role,
                    complexity_score=complexity_score,
                    custom_system_prompt=custom_system_prompt,
                    **kwargs,
                ),
                timeout=self.config["deepseek_timeout"],
            )
        except asyncio.TimeoutError:
            raise ProviderError(
                f"DeepSeek timeout after {self.config['deepseek_timeout']}s"
            )

    async def _call_openrouter_with_timeout(
        self,
        prompt: str,
        task_type: str,
        consultant_role: str,
        complexity_score: float,
        custom_system_prompt: str,
        **kwargs,
    ) -> LLMResponse:
        """Call OpenRouter DeepSeek with fast timeout"""
        try:
            return await asyncio.wait_for(
                self.openrouter_provider.call_optimized_llm(
                    prompt=prompt,
                    task_type=task_type,
                    consultant_role=consultant_role,
                    complexity_score=complexity_score,
                    custom_system_prompt=custom_system_prompt,
                    **kwargs,
                ),
                timeout=self.config["openrouter_timeout"],
            )
        except asyncio.TimeoutError:
            raise ProviderError(
                f"OpenRouter timeout after {self.config['openrouter_timeout']}s"
            )

    async def _call_claude_with_timeout(
        self,
        prompt: str,
        task_type: str,
        consultant_role: str,
        complexity_score: float,
        custom_system_prompt: str,
        **kwargs,
    ) -> LLMResponse:
        """Call Claude with timeout"""
        try:
            return await asyncio.wait_for(
                self.claude_provider.call_optimized_llm(
                    prompt=prompt,
                    task_type=task_type,
                    consultant_role=consultant_role,
                    complexity_score=complexity_score,
                    custom_system_prompt=custom_system_prompt,
                    **kwargs,
                ),
                timeout=self.config["claude_timeout"],
            )
        except asyncio.TimeoutError:
            raise ProviderError(
                f"Claude timeout after {self.config['claude_timeout']}s"
            )

    def _classify_error(self, error: Exception) -> FallbackReason:
        """Classify error for fallback reasoning"""
        error_str = str(error).lower()

        if "timeout" in error_str or isinstance(error, asyncio.TimeoutError):
            return FallbackReason.TIMEOUT
        elif "429" in error_str or "rate limit" in error_str:
            return FallbackReason.RATE_LIMIT
        elif "503" in error_str or "service unavailable" in error_str:
            return FallbackReason.SERVICE_UNAVAILABLE
        elif "connection" in error_str or "network" in error_str:
            return FallbackReason.CONNECTION_ERROR
        else:
            return FallbackReason.API_ERROR

    def _record_success(self, provider: str, response_time_ms: int):
        """Record successful call"""
        if provider in self.provider_stats:
            stats = self.provider_stats[provider]
            stats["calls"] += 1
            stats["successes"] += 1
            stats["avg_response_time"] = (
                stats["avg_response_time"] * (stats["successes"] - 1) + response_time_ms
            ) / stats["successes"]

    def _record_failure(self, provider: str):
        """Record failed call"""
        if provider in self.provider_stats:
            stats = self.provider_stats[provider]
            stats["calls"] += 1
            stats["failures"] += 1

    def get_provider_stats(self) -> Dict[str, Any]:
        """Get provider performance statistics"""
        stats = {}

        for provider, data in self.provider_stats.items():
            if data["calls"] > 0:
                success_rate = data["successes"] / data["calls"]
                stats[provider] = {
                    "total_calls": data["calls"],
                    "success_rate": success_rate,
                    "average_response_time_ms": data["avg_response_time"],
                    "status": (
                        "healthy"
                        if success_rate > 0.8
                        else "degraded" if success_rate > 0.5 else "unhealthy"
                    ),
                }

        stats["fallback_events"] = len(self.fallback_history)
        stats["recent_fallbacks"] = (
            self.fallback_history[-5:] if len(self.fallback_history) > 0 else []
        )

        return stats

    def get_recommended_provider(self) -> str:
        """Get recommended provider based on recent performance"""
        stats = self.get_provider_stats()

        deepseek_health = stats.get("deepseek", {}).get("success_rate", 0)
        openrouter_health = stats.get("openrouter", {}).get("success_rate", 0)
        claude_health = stats.get("claude", {}).get("success_rate", 0)

        # Prefer providers in order of health and speed
        if openrouter_health > 0.8:
            return "openrouter"
        elif deepseek_health > 0.8 and self.config["prefer_deepseek"]:
            return "deepseek"
        elif claude_health > 0.7:
            return "claude"
        elif openrouter_health > max(deepseek_health, claude_health):
            return "openrouter"
        elif deepseek_health > claude_health:
            return "deepseek"
        else:
            return "claude"
