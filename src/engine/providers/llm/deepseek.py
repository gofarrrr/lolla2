"""
DeepSeek V3.1 LLM Provider Implementation
Handles DeepSeek API interactions with optimized configuration
Operation Bedrock: Task 14.0 - Provider Adapters
"""

import httpx
import logging
import time
import os
import asyncio
from typing import Dict, Optional, Any
from enum import Enum

from .base import LLMProvider, LLMResponse


class DeepSeekMode(Enum):
    """DeepSeek V3.1 API modes"""
    CHAT = "deepseek-chat"  # Fast, direct responses
    REASONER = "deepseek-reasoner"  # Chain-of-thought reasoning


class DeepSeekProvider(LLMProvider):
    """
    DeepSeek V3.1 provider with retry logic and optimized configuration

    Features:
    - Automatic retry on 503 (server overload) and 429 (rate limit)
    - Exponential backoff strategy
    - Support for both chat and reasoner modes
    - Cost-effective: ~$2.19 per 1M tokens (585% savings vs Claude)
    """

    def __init__(self, api_key: Optional[str] = None):
        super().__init__("deepseek")
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = "https://api.deepseek.com"
        self.logger = logging.getLogger(__name__)

        # DeepSeek V3.1 configuration
        self.model = DeepSeekMode.CHAT.value
        self.pricing = {"input": 0.27, "output": 1.10}  # USD per 1M tokens
        self.max_tokens = 8192
        self.default_timeout = 150  # DeepSeek can be slower due to high load

        # Retry configuration for DeepSeek V3.1 (high load issues)
        self.max_retries = 3
        self.retry_delays = [5, 10, 15]  # Exponential backoff in seconds

        if not self.api_key:
            raise ValueError(
                "DeepSeek API key not found. Set DEEPSEEK_API_KEY environment variable."
            )

        self.logger.info(f"🚀 DeepSeek provider initialized with model: {self.model}")

    async def complete(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.3,
        max_tokens: int = 2000,
        use_reasoner: bool = False,
        **kwargs,
    ) -> LLMResponse:
        """
        Execute completion with DeepSeek V3.1

        Args:
            prompt: User prompt text
            system_prompt: System instructions
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum completion tokens
            use_reasoner: If True, use deepseek-reasoner mode for chain-of-thought
            **kwargs: Additional provider-specific parameters
        """
        start_time = time.time()

        # Select model based on mode
        model = DeepSeekMode.REASONER.value if use_reasoner else DeepSeekMode.CHAT.value

        # Retry logic for DeepSeek server overload
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = await self._make_api_call(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=min(max_tokens, self.max_tokens),
                    model=model,
                    **kwargs,
                )

                # Extract response data
                response_text = response["choices"][0]["message"]["content"]
                usage = response["usage"]

                # Calculate cost
                prompt_tokens = usage["prompt_tokens"]
                completion_tokens = usage["completion_tokens"]
                total_cost = (prompt_tokens * self.pricing["input"] / 1_000_000) + (
                    completion_tokens * self.pricing["output"] / 1_000_000
                )

                # Success - log and return
                self.logger.info(
                    f"✅ DeepSeek call succeeded (attempt {attempt + 1}/{self.max_retries})"
                )

                return self._create_response(
                    raw_text=response_text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cost=total_cost,
                    start_time=start_time,
                    model_name=model,
                    metadata={
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "mode": "reasoner" if use_reasoner else "chat",
                        "attempts": attempt + 1,
                    },
                    raw_provider_response=response,
                )

            except httpx.HTTPStatusError as e:
                last_error = e
                # Retry on 503 (server overload) or 429 (rate limit)
                if e.response.status_code in [503, 429] and attempt < self.max_retries - 1:
                    delay = self.retry_delays[attempt]
                    self.logger.warning(
                        f"⚠️ DeepSeek {e.response.status_code} error (attempt {attempt + 1}/{self.max_retries}), "
                        f"retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    # Not retryable or final attempt
                    break

            except Exception as e:
                last_error = e
                self.logger.error(
                    f"❌ DeepSeek API failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                # Don't retry on other errors
                break

        # All retries exhausted
        error_msg = f"DeepSeek API failed after {self.max_retries} attempts: {str(last_error)}"
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
        """Make HTTP request to DeepSeek API"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
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

        # DeepSeek needs longer timeout due to high load
        timeout = httpx.Timeout(self.default_timeout, read=self.default_timeout)

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{self.base_url}/v1/chat/completions",
                headers=headers,
                json=payload,
            )

            # Raise for HTTP errors (will be caught for retry logic)
            response.raise_for_status()

            return response.json()

    async def is_available(self) -> bool:
        """Check if DeepSeek API is responding"""
        try:
            # Quick test completion to verify API access
            timeout = httpx.Timeout(10.0)
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            test_payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 10,
            }

            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers=headers,
                    json=test_payload,
                )
                return response.status_code == 200

        except Exception:
            return False
