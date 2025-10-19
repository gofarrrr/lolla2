#!/usr/bin/env python3
"""
OpenAI Provider Implementation
Handles all OpenAI-specific API interactions
"""

import httpx
from datetime import datetime
from typing import Dict, List

from .provider_interface import (
    BaseLLMProvider,
    LLMResponse,
    ProviderError,
    ProviderAPIError,
)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider implementation"""

    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1"):
        super().__init__(api_key, base_url)
        self.provider_name = "openai"
        self._models = [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
        ]

        # Cost per token mapping (approximate)
        self._cost_per_token = {
            "gpt-4o": 0.00015,
            "gpt-4o-mini": 0.00005,
            "gpt-4-turbo": 0.00010,
            "gpt-4": 0.00015,
            "gpt-3.5-turbo": 0.00002,
        }

    def get_available_models(self) -> List[str]:
        """Get list of available OpenAI models"""
        return self._models.copy()

    async def is_available(self) -> bool:
        """Check if OpenAI is available"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/models",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=10.0,
                )
                response.raise_for_status()
                return True
        except Exception as e:
            self.logger.warning(f"OpenAI availability check failed: {e}")
            return False

    async def call_llm(
        self,
        messages: List[Dict],
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs,
    ) -> LLMResponse:
        """Call OpenAI API"""
        start_time = datetime.now()

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                    timeout=30.0,
                )

            response.raise_for_status()
            result = response.json()

            response_time = int((datetime.now() - start_time).total_seconds() * 1000)
            tokens_used = result["usage"]["total_tokens"]

            # Calculate cost based on model
            cost_per_token = self._cost_per_token.get(model, 0.0001)  # Default fallback
            cost_usd = tokens_used * cost_per_token

            llm_response = LLMResponse(
                content=result["choices"][0]["message"]["content"],
                provider="openai",
                model=model,
                tokens_used=tokens_used,
                cost_usd=cost_usd,
                response_time_ms=response_time,
                reasoning_steps=[],  # Will be parsed from content if needed
                mental_models=[],  # Will be parsed from content if needed
                confidence=0.8,  # Default confidence for OpenAI
            )

            return llm_response

        except httpx.HTTPStatusError as e:
            error_detail = ""
            try:
                error_data = e.response.json()
                error_detail = error_data.get("error", {}).get("message", str(e))
            except:
                error_detail = str(e)

            raise ProviderAPIError(
                f"OpenAI API error: {e.response.status_code} - {error_detail}"
            )
        except httpx.TimeoutException:
            raise ProviderAPIError("OpenAI API timeout")
        except Exception as e:
            raise ProviderError(f"OpenAI call failed: {e}")

    def get_estimated_cost(self, tokens: int, model: str) -> float:
        """Get estimated cost for token usage"""
        cost_per_token = self._cost_per_token.get(model, 0.0001)
        return tokens * cost_per_token
