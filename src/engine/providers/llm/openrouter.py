"""
OpenRouter LLM Provider Implementation
Handles OpenRouter API interactions for multi-model access
Operation Bedrock: Task 14.0 - Provider Adapters
"""

import httpx
import logging
import time
import os
from typing import Dict, Optional, Any

from .base import LLMProvider, LLMResponse


class OpenRouterProvider(LLMProvider):
    """
    OpenRouter provider for accessing multiple LLM models through a single API

    Features:
    - Access to 100+ models from different providers
    - Unified pricing across providers
    - Automatic fallback to alternative models
    - Support for streaming responses
    """

    def __init__(self, api_key: Optional[str] = None, default_model: str = "x-ai/grok-4-fast"):
        super().__init__("openrouter")
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"
        self.logger = logging.getLogger(__name__)

        # Default model configuration
        self.default_model = default_model
        self.max_tokens = 4096

        # Common model aliases - all resolve to x-ai/grok-4-fast paid tier
        self.model_aliases = {
            "grok": "x-ai/grok-4-fast",
            "grok-4-fast": "x-ai/grok-4-fast",
            "grok4fast": "x-ai/grok-4-fast",
            "grok-beta": "x-ai/grok-4-fast",  # Redirect legacy alias to Grok 4 Fast
            "claude": "anthropic/claude-3.5-sonnet",
            "gpt4": "openai/gpt-4-turbo",
            "llama": "meta-llama/llama-3.1-405b-instruct",
        }

        # Approximate pricing (USD per 1M tokens) - varies by model
        self.pricing_estimates = {
            "x-ai/grok-4-fast": {"input": 0.2, "output": 0.5},
            "x-ai/grok-2-latest": {"input": 0.3, "output": 0.6},
            "anthropic/claude-3.5-sonnet": {"input": 3.0, "output": 15.0},
            "openai/gpt-4-turbo": {"input": 10.0, "output": 30.0},
        }

        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable."
            )

        self.logger.info(
            f"🚀 OpenRouter provider initialized with default model: {default_model}"
        )

    def _resolve_model(self, model: str) -> str:
        """Resolve model alias to full model identifier"""
        return self.model_aliases.get(model, model)

    def _estimate_cost(
        self, model: str, prompt_tokens: int, completion_tokens: int
    ) -> float:
        """Estimate cost based on model pricing"""
        pricing = self.pricing_estimates.get(model, {"input": 5.0, "output": 15.0})
        return (prompt_tokens * pricing["input"] / 1_000_000) + (
            completion_tokens * pricing["output"] / 1_000_000
        )

    async def complete(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.3,
        max_tokens: int = 2000,
        model: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Execute completion with OpenRouter

        Args:
            prompt: User prompt text
            system_prompt: System instructions
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum completion tokens
            model: Model identifier (supports aliases like 'grok', 'claude')
            **kwargs: Additional provider-specific parameters
        """
        start_time = time.time()

        # Use provided model or default
        model_id = self._resolve_model(model or self.default_model)

        try:
            response = await self._make_api_call(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=min(max_tokens, self.max_tokens),
                model=model_id,
                **kwargs,
            )

            # Extract response data (OpenAI-compatible format)
            response_text = response["choices"][0]["message"]["content"]
            usage = response.get("usage", {})

            # Get token counts
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            # Estimate cost (OpenRouter doesn't always provide cost in response)
            total_cost = self._estimate_cost(model_id, prompt_tokens, completion_tokens)

            self.logger.info(f"✅ OpenRouter call succeeded with model: {model_id}")

            return self._create_response(
                raw_text=response_text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost=total_cost,
                start_time=start_time,
                model_name=model_id,
                metadata={
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "provider": "openrouter",
                },
                raw_provider_response=response,
            )

        except Exception as e:
            error_msg = f"OpenRouter API failed: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            raise Exception(error_msg)

    async def _make_api_call(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        model: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Make HTTP request to OpenRouter API"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://lolla.ai",  # Required by OpenRouter - standardized
            "X-Title": "METIS V5.3 Cognitive Platform",  # Standardized app title
        }

        # Prepare messages (OpenAI-compatible format)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Add any additional parameters
        payload.update(kwargs)

        # OpenRouter typically responds quickly
        timeout = httpx.Timeout(60.0)

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )

            if response.status_code == 200:
                return response.json()
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                raise Exception(error_msg)

    async def is_available(self) -> bool:
        """Check if OpenRouter API is responding"""
        try:
            # Quick test completion to verify API access
            timeout = httpx.Timeout(10.0)
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://lolla.ai",
                "X-Title": "METIS V5.3 Cognitive Platform",
            }

            # Use a cheap, fast model for availability check
            test_payload = {
                "model": "openai/gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 5,
            }

            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=test_payload,
                )
                return response.status_code == 200

        except Exception:
            return False

    def get_available_models(self) -> list[str]:
        """Get list of commonly used models"""
        return list(self.model_aliases.keys()) + [
            "x-ai/grok-beta",
            "x-ai/grok-2-1212",
            "anthropic/claude-3.5-sonnet",
            "openai/gpt-4-turbo",
            "openai/gpt-3.5-turbo",
            "meta-llama/llama-3.1-405b-instruct",
            "google/gemini-pro-1.5",
            "mistralai/mixtral-8x7b-instruct",
        ]
