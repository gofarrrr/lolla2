#!/usr/bin/env python3
"""
OpenRouter Provider for Unified LLM Client
Wraps the OpenRouter client to match the provider interface
"""

import asyncio
import logging
import os
from typing import Dict, List, Optional, Any

from .provider_interface import BaseLLMProvider, LLMResponse
from src.engine.integrations.openrouter_client import OpenRouterClient


class OpenRouterProvider(BaseLLMProvider):
    """OpenRouter provider wrapper for unified client interface"""

    def __init__(self, api_key: str = None, global_semaphore: asyncio.Semaphore = None):
        super().__init__(api_key=api_key or "")  # Initialize base class including _call_history
        self.logger = logging.getLogger(__name__)
        self.client = OpenRouterClient(api_key, global_semaphore=global_semaphore)
        self.provider_name = "openrouter"
        
        # OPERATION GHOST HUNT: Ensure _call_history exists (fallback in case inheritance fails)
        if not hasattr(self, '_call_history'):
            self._call_history = []
            self.logger.info("🎯 GHOST HUNT: Manually initialized _call_history fallback")

    def __getattr__(self, name):
        """OPERATION GHOST HUNT: Dynamic attribute access for _call_history"""
        if name == '_call_history':
            if not hasattr(self, '_call_history'):
                self._call_history = []
                self.logger.debug("🎯 GHOST HUNT: _call_history created dynamically via __getattr__")
            return self._call_history
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def get_call_history(self) -> List[Dict]:
        """Get call history for this provider - Override with defensive check"""
        if not hasattr(self, '_call_history'):
            self._call_history = []
        return self._call_history.copy()

    def get_total_cost(self) -> float:
        """Get total cost for this provider - Override with defensive check"""
        if not hasattr(self, '_call_history'):
            self._call_history = []
        return sum(call.get("cost", 0) for call in self._call_history)

    def record_call(self, response: LLMResponse, method: str):
        """Record a call in the history - Override with defensive check"""
        try:
            # Multiple defensive layers
            if not hasattr(self, '_call_history'):
                self._call_history = []
                self.logger.info("🎯 GHOST HUNT: _call_history created in record_call")
            
            # Verify attribute exists
            assert hasattr(self, '_call_history'), "GHOST HUNT: _call_history missing despite creation"
            
            from datetime import datetime
            self._call_history.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "method": method,
                    "provider": response.provider,
                    "model": response.model,
                    "tokens": response.tokens_used,
                    "cost": response.cost_usd,
                    "response_time_ms": response.response_time_ms,
                }
            )
            self.logger.info(f"🎯 GHOST HUNT: Call recorded successfully, history length: {len(self._call_history)}")
        except Exception as e:
            self.logger.error(f"❌ GHOST HUNT: record_call failed: {e}")
            # Don't let this break the whole flow
            pass

    def _should_use_reasoning_mode(self, messages: List[Dict[str, str]], kwargs: Dict) -> bool:
        """
        Decide if reasoning mode is needed based on task characteristics.

        Heuristics from research:
        - Enable for strategic analysis, consultant analysis, problem structuring
        - Enable for multi-step problems (long prompts >500 chars)
        - Enable for complex synthesis tasks
        """
        task_type = kwargs.get('task_type', '')

        # Calculate prompt length
        prompt_length = 0
        for msg in messages:
            prompt_length += len(msg.get('content', ''))

        # Enable reasoning for specific task types
        reasoning_tasks = ['strategic', 'analysis', 'consultant', 'synthesis', 'structuring']
        if any(indicator in task_type.lower() for indicator in reasoning_tasks):
            return True

        # Enable reasoning for complex prompts
        if prompt_length > 500:
            return True

        return False  # Simple tasks don't need reasoning overhead

    def _resolve_model_aliases(self, requested_model: str) -> List[str]:
        """
        Resolve Grok model aliases to OpenRouter slugs with graceful degradation.

        Priority order:
        1. Explicit environment override (OPENROUTER_GROK_FAST_MODEL)
        2. Caller-specified model (if full slug)
        3. Preferred Grok 4 Fast aliases
        4. Grok 2 / Grok Beta fallbacks
        """
        aliases: List[str] = []

        # Environment override allows quick hotfixes without code changes
        env_override = os.getenv("OPENROUTER_GROK_FAST_MODEL")
        if env_override:
            aliases.append(env_override.strip())
        else:
            # Default to x-ai/grok-4-fast paid tier for reliability
            aliases.append("x-ai/grok-4-fast")

        normalized = (requested_model or "grok-4-fast").strip()

        # Ensure canonical slug is attempted first unless an override already provided it
        canonical_slug = "x-ai/grok-4-fast"
        if canonical_slug not in aliases:
            aliases.append(canonical_slug)

        # Preserve the caller's explicit model next (avoids duplicates later)
        if normalized and normalized not in aliases:
            aliases.append(normalized)

        # Canonical Grok 4 Fast aliases (paid + public) - ensure all variants attempted
        grok_fast_aliases = [
            "grok-4-fast",
            "x-ai/grok-4-fast:free",
            "grok-4-fast:free",
        ]

        # Broader Grok family fallbacks (ensure we stay on Grok before DeepSeek fallback)
        grok_family_fallbacks = [
            "x-ai/grok-2-latest",
            "grok-2-latest",
            "x-ai/grok-2",
            "x-ai/grok-1.5",
            "x-ai/grok-1.5-mini",
            "x-ai/grok-beta",
            "grok-beta",
        ]

        # Merge lists while preserving order and removing duplicates
        for candidate in grok_fast_aliases + grok_family_fallbacks:
            if candidate not in aliases:
                aliases.append(candidate)

        return aliases

    @staticmethod
    def _is_model_not_found_error(error: Exception) -> bool:
        """Detect OpenRouter errors that indicate a missing model endpoint."""
        message = str(error).lower()
        return (
            "no endpoints found" in message
            or "model does not exist" in message
            or "invalid model" in message
        )

    async def call_llm(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Call OpenRouter LLM with unified interface"""

        # Determine requested model (default to Grok-4-Fast)
        requested_model = model or "grok-4-fast"

        # Pop explicit reasoning directive if provided upstream
        explicit_reasoning = kwargs.pop("reasoning_enabled", None)

        # Determine if reasoning mode should be enabled
        if explicit_reasoning is not None:
            requires_reasoning = explicit_reasoning
            self.logger.info(
                f"🎯 Reasoning mode override received: {requires_reasoning} "
                f"(requested_model={requested_model})"
            )
        else:
            requires_reasoning = self._should_use_reasoning_mode(messages, kwargs)

        # Ensure reasoning flag is passed downstream
        kwargs["reasoning_enabled"] = requires_reasoning

        if requires_reasoning:
            self.logger.info("🧠 Grok 4 Fast: Reasoning mode ENABLED for complex analysis")
        else:
            self.logger.info("⚡ Grok 4 Fast: Non-reasoning mode selected")

        # Resolve candidate OpenRouter slugs for Grok 4 Fast
        candidate_models = self._resolve_model_aliases(requested_model)

        # Convert messages to prompt format for OpenRouter client
        system_prompt = ""
        conversation_parts = []
        
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            
            if role == "system":
                system_prompt = content
            elif role == "user":
                conversation_parts.append(f"User: {content}")
            elif role == "assistant":
                conversation_parts.append(f"Assistant: {content}")
        
        # Combine all conversation parts into user_prompt
        user_prompt = "\n\n".join(conversation_parts)
        
        # Call OpenRouter client with comprehensive error logging
        last_error: Optional[Exception] = None
        for openrouter_model in candidate_models:
            try:
                self.logger.info(
                    "🔄 OpenRouter API call starting: model=%s, user_prompt_length=%d, "
                    "system_prompt_length=%d, reasoning_enabled=%s",
                    openrouter_model,
                    len(user_prompt),
                    len(system_prompt),
                    requires_reasoning,
                )

                response = await self.client.complete(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    model=openrouter_model,
                    **kwargs,
                )

                self.logger.info(
                    "✅ OpenRouter API call successful: response_type=%s, has_raw_text=%s",
                    type(response),
                    hasattr(response, "raw_text"),
                )
                break  # Success - exit loop

            except Exception as e:
                error_type = type(e).__name__
                self.logger.error(
                    "❌ OpenRouter API call failed (%s): %s", openrouter_model, e
                )
                self.logger.error(f"   User prompt: {user_prompt[:200]}...")
                self.logger.error(f"   System prompt: {system_prompt[:200]}...")

                last_error = e

                # Retry with next alias if the model slug is unavailable
                if self._is_model_not_found_error(e) and openrouter_model != candidate_models[-1]:
                    self.logger.warning(
                        "🔄 OpenRouter model alias unavailable (%s). Trying next alias...",
                        openrouter_model,
                    )
                    continue

                # For timeout errors, provide guidance
                if "timeout" in str(e).lower() or "TimeoutError" in error_type:
                    self.logger.warning(
                        "⚠️ OpenRouter timeout detected - consider shorter prompts or paid tier."
                    )

                raise
        else:
            # If loop exits without break, raise last error
            if last_error:
                raise last_error

        llm_response = LLMResponse(
            content=response.raw_text,
            model=openrouter_model,
            provider="openrouter", 
            tokens_used=response.total_tokens,
            cost_usd=response.cost,
            response_time_ms=int(response.time_seconds * 1000),
            reasoning_steps=[],
            mental_models=[],
            confidence=0.8,
            metadata={
                "original_model": model,
                "mapped_model": openrouter_model,
                "provider_metadata": getattr(response, 'metadata', {}),
                "raw_provider_response": getattr(response, 'raw_provider_response', {})
            }
        )
        
        # Record the call for history tracking
        self.record_call(llm_response, "call_llm")
        
        return llm_response

    async def test_connectivity(self) -> bool:
        """Test OpenRouter connectivity"""
        try:
            test_messages = [{"role": "user", "content": "Hello"}]
            response = await self.call_llm(test_messages)
            return response.content is not None
        except Exception as e:
            self.logger.warning(f"OpenRouter connectivity test failed: {e}")
            return False

    def get_available_models(self) -> List[str]:
        """Get list of available OpenRouter models"""
        # Provide unique list of Grok aliases we automatically probe
        return self._resolve_model_aliases("grok-4-fast")

    async def is_available(self) -> bool:
        """Check if OpenRouter provider is available"""
        return self.client is not None

    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Complete method expected by LLMManager - converts to call_llm format"""
        
        # Convert prompt and system_prompt to messages format
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        messages.append({"role": "user", "content": prompt})
        
        # Use call_llm with the converted messages
        return await self.call_llm(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
