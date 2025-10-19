"""
Anthropic Claude LLM Provider Implementation
Handles Claude-3.5-Sonnet API interactions as fallback provider
"""

import httpx
import logging
import time
import os
from typing import Dict, Optional, Any

from .base import LLMProvider, LLMResponse


class AnthropicProvider(LLMProvider):
    """Anthropic Claude-3.5-Sonnet provider for fallback capabilities"""

    def __init__(self, api_key: Optional[str] = None):
        super().__init__("anthropic")
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.base_url = "https://api.anthropic.com"
        self.logger = logging.getLogger(__name__)

        # Claude-3.5-Sonnet configuration
        self.model = "claude-3-5-sonnet-20241022"
        self.pricing = {"input": 3.0, "output": 15.0}  # USD per 1M tokens
        self.max_tokens = 4096  # Claude's typical max

        if not self.api_key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable."
            )

        self.logger.info(f"🚀 Anthropic provider initialized with model: {self.model}")

    async def complete(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.3,
        max_tokens: int = 2000,
        **kwargs,
    ) -> LLMResponse:
        """
        Execute completion with Claude-3.5-Sonnet
        """
        start_time = time.time()

        try:
            response = await self._make_api_call(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=min(max_tokens, self.max_tokens),
                **kwargs,
            )

            # Extract response data
            response_text = response["content"][0]["text"]
            usage = response["usage"]

            # Calculate cost
            prompt_tokens = usage["input_tokens"]
            completion_tokens = usage["output_tokens"]
            total_cost = (prompt_tokens * self.pricing["input"] / 1_000_000) + (
                completion_tokens * self.pricing["output"] / 1_000_000
            )

            return self._create_response(
                raw_text=response_text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost=total_cost,
                start_time=start_time,
                model_name=self.model,
                metadata={"temperature": temperature, "max_tokens": max_tokens},
            )

        except Exception as e:
            error_msg = f"Anthropic API failed: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            raise Exception(error_msg)

    async def _make_api_call(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        **kwargs,
    ) -> Dict[str, Any]:
        """Make HTTP request to Anthropic API"""

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        # Prepare messages
        messages = [{"role": "user", "content": prompt}]

        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }

        # Add system prompt if provided
        if system_prompt:
            payload["system"] = system_prompt

        # Add any additional parameters
        payload.update(kwargs)

        timeout = httpx.Timeout(60.0)  # Claude typically responds quickly

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{self.base_url}/v1/messages", headers=headers, json=payload
            )

            if response.status_code == 200:
                return response.json()
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                raise Exception(error_msg)

    async def is_available(self) -> bool:
        """Check if Anthropic API is responding"""
        try:
            # Simple test completion to verify API access
            timeout = httpx.Timeout(5.0)
            headers = {
                "x-api-key": self.api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
            }

            test_payload = {
                "model": self.model,
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "test"}],
            }

            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{self.base_url}/v1/messages", headers=headers, json=test_payload
                )
                return response.status_code == 200

        except Exception:
            return False
