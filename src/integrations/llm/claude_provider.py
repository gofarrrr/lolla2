#!/usr/bin/env python3
"""
Claude/Anthropic Provider Implementation
Handles all Claude/Anthropic-specific API interactions and integrations
"""

import httpx
from datetime import datetime
from typing import Dict, List, Optional, Any

from .provider_interface import (
    BaseLLMProvider,
    LLMResponse,
    ProviderError,
    ProviderAPIError,
)


class ClaudeProvider(BaseLLMProvider):
    """Claude/Anthropic provider implementation"""

    def __init__(self, api_key: str, base_url: str = "https://api.anthropic.com/v1"):
        super().__init__(api_key, base_url)
        self.provider_name = "claude"
        self._claude_client = None
        self._models = [
            "claude-3-5-sonnet-20241022",
            "claude-3-haiku-20240307",
            "claude-3-sonnet-20240229",
            "claude-3-opus-20240229",
        ]

    def get_available_models(self) -> List[str]:
        """Get list of available Claude models"""
        return self._models.copy()

    async def is_available(self) -> bool:
        """Check if Claude is available via the working claude_client system"""
        try:
            # Use the existing working claude_client for availability check
            from ..claude_client import get_claude_client

            claude_client = await get_claude_client()
            return await claude_client.is_available()
        except Exception as e:
            self.logger.warning(f"Claude availability check failed: {e}")
            return False

    async def call_llm(
        self,
        messages: List[Dict],
        model: str = "claude-3-5-sonnet-20241022",
        phase: Optional[str] = None,
        engagement_id: Optional[str] = None,
        context_data: Optional[Dict[str, Any]] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ) -> LLMResponse:
        """Call Claude API using the working claude_client system"""
        start_time = datetime.now()

        try:
            # Import and use the working Claude client
            from ..claude_client import get_claude_client, LLMCallType

            claude_client = await get_claude_client()

            # Convert messages format to what Claude client expects
            system_message = ""
            user_content = ""

            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                elif msg["role"] == "user":
                    user_content = msg["content"]
                elif msg["role"] == "assistant":
                    # Add assistant context to user message
                    user_content += f"\n\nPrevious assistant response: {msg['content']}"

            # Handle prompt phase mapping
            prompt_phase = None
            try:
                from src.engine.core.prompt_capture import PromptPhase

                # Map method name to prompt phase
                phase_mapping = {
                    "analyze_problem_structure": PromptPhase.PROBLEM_STRUCTURING,
                    "analyze_problem_structure_with_research": PromptPhase.PROBLEM_STRUCTURING,
                    "generate_hypotheses": PromptPhase.HYPOTHESIS_GENERATION,
                    "execute_analysis": PromptPhase.ANALYSIS_EXECUTION,
                    "synthesize_deliverable": PromptPhase.SYNTHESIS_DELIVERY,
                }

                # Determine phase from context or use provided phase
                if phase:
                    if hasattr(PromptPhase, phase.upper()):
                        prompt_phase = getattr(PromptPhase, phase.upper())
                    else:
                        # Try to match from our mapping
                        for method_name, mapped_phase in phase_mapping.items():
                            if phase in method_name:
                                prompt_phase = mapped_phase
                                break
                        if not prompt_phase:
                            prompt_phase = PromptPhase.OTHER
                else:
                    # Default to problem structuring if no phase specified
                    prompt_phase = PromptPhase.PROBLEM_STRUCTURING

            except ImportError:
                prompt_phase = None

            # Make the call using the working Claude client
            claude_response = await claude_client.call_claude(
                prompt=user_content,
                call_type=LLMCallType.MENTAL_MODEL,
                max_tokens=max_tokens,
                temperature=temperature,
                system_prompt=system_message if system_message else None,
                phase=prompt_phase,
                engagement_id=engagement_id,
                context_data=context_data,
            )

            # Convert Claude client response to LLMProvider format
            response_time = int((datetime.now() - start_time).total_seconds() * 1000)

            response = LLMResponse(
                content=claude_response.content,
                provider="anthropic",
                model=claude_response.model_version,
                tokens_used=claude_response.tokens_used,
                cost_usd=claude_response.cost_usd,
                response_time_ms=response_time,
                reasoning_steps=claude_response.reasoning_steps,
                mental_models=[],  # Could extract from reasoning_steps if needed
                confidence=claude_response.confidence,
            )

            return response

        except Exception as e:
            self.logger.error(f"Claude client delegation failed: {e}")
            # Fallback to HTTP if delegation fails
            return await self._call_claude_http_fallback(
                messages, model, temperature, max_tokens
            )

    async def _call_claude_http_fallback(
        self,
        messages: List[Dict],
        model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ) -> LLMResponse:
        """HTTP fallback for Claude API calls"""
        start_time = datetime.now()

        try:
            # Convert OpenAI format to Anthropic format
            system_message = ""
            user_messages = []

            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    user_messages.append(msg)

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/messages",
                    headers={
                        "x-api-key": self.api_key,
                        "Content-Type": "application/json",
                        "anthropic-version": "2023-06-01",
                    },
                    json={
                        "model": model,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "system": system_message,
                        "messages": user_messages,
                    },
                    timeout=30.0,
                )

            response.raise_for_status()
            result = response.json()

            response_time = int((datetime.now() - start_time).total_seconds() * 1000)
            tokens_used = (
                result["usage"]["input_tokens"] + result["usage"]["output_tokens"]
            )

            # Simple cost estimation (approximate)
            cost_usd = tokens_used * 0.0001

            response = LLMResponse(
                content=result["content"][0]["text"],
                provider="anthropic",
                model=model,
                tokens_used=tokens_used,
                cost_usd=cost_usd,
                response_time_ms=response_time,
                reasoning_steps=[],
                mental_models=[],
                confidence=0.8,
            )

            return response

        except httpx.HTTPStatusError as e:
            raise ProviderAPIError(
                f"Claude API error: {e.response.status_code} - {e.response.text}"
            )
        except httpx.TimeoutException:
            raise ProviderAPIError("Claude API timeout")
        except Exception as e:
            raise ProviderError(f"Claude HTTP fallback failed: {e}")
