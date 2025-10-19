"""
LLM Manager - Multi-Provider Orchestration with Automatic Fallback
Provides resilient LLM access with sequential provider fallback and comprehensive logging
"""

import asyncio
import hashlib
import logging
import re
import time
import os
from typing import List, Dict, Any, Optional

from ..providers.llm import (
    LLMProvider,
    LLMResponse,
    DeepSeekProvider,
    AnthropicProvider,
    OpenRouterProvider,
)
from src.core.unified_context_stream import ContextEventType


class ContextCompiler:
    """
    Phase 5: Context compiler for stable prompt prefixes and KV cache optimization.

    Centralizes context engineering at the LLMManager boundary to:
    1. Generate stable prompt prefixes for better KV cache utilization
    2. Apply context optimizations and token reduction strategies
    3. Emit CONTEXT_ENGINEERING_* events for glass-box transparency
    """

    def __init__(self, context_stream=None):
        self.logger = logging.getLogger(__name__)
        self.context_stream = context_stream
        self.prefix_cache = {}  # Cache stable prefixes by stage/type
        self.optimization_stats = {"tokens_saved": 0, "cache_hits": 0}

        self.logger.info("🔧 ContextCompiler initialized for stable prompt prefixes")

    def compile_context(
        self,
        prompt: str,
        system_prompt: str = "",
        stage: str = "default",
        context_vars: Dict[str, Any] = None,
    ) -> Dict[str, str]:
        """
        Compile context with stable prefixes and optimizations.

        Args:
            prompt: User prompt text
            system_prompt: System/instructions prompt
            stage: Pipeline stage for prefix caching (e.g., "analysis", "synthesis")
            context_vars: Variable data for prompt templating

        Returns:
            Dict with compiled_prompt, compiled_system_prompt, and optimization metadata
        """
        start_time = time.time()
        context_vars = context_vars or {}

        # Emit context engineering started event
        self._emit_context_event(
            ContextEventType.CONTEXT_ENGINEERING_STARTED,
            {
                "stage": stage,
                "original_prompt_length": len(prompt),
                "original_system_length": len(system_prompt),
                "has_context_vars": bool(context_vars),
            },
        )

        # Generate stable prefix for this stage
        stable_prefix = self._get_stable_prefix(stage, context_vars)

        # Apply context optimizations
        optimized_prompt = self._optimize_prompt(prompt, context_vars)
        optimized_system = self._optimize_system_prompt(system_prompt, stable_prefix)

        # Calculate token savings
        original_tokens = len(prompt.split()) + len(system_prompt.split())
        optimized_tokens = len(optimized_prompt.split()) + len(optimized_system.split())
        tokens_saved = original_tokens - optimized_tokens
        self.optimization_stats["tokens_saved"] += tokens_saved

        compilation_time_ms = int((time.time() - start_time) * 1000)

        # Emit context compiled event
        self._emit_context_event(
            ContextEventType.CONTEXT_COMPILED,
            {
                "stage": stage,
                "stable_prefix_applied": bool(stable_prefix),
                "original_tokens": original_tokens,
                "optimized_tokens": optimized_tokens,
                "tokens_saved": tokens_saved,
                "compilation_time_ms": compilation_time_ms,
                "cache_hit": stage in self.prefix_cache,
            },
        )

        return {
            "compiled_prompt": optimized_prompt,
            "compiled_system_prompt": optimized_system,
            "stable_prefix": stable_prefix,
            "tokens_saved": tokens_saved,
            "compilation_time_ms": compilation_time_ms,
        }

    def _get_stable_prefix(self, stage: str, context_vars: Dict[str, Any]) -> str:
        """Generate or retrieve stable prefix for KV cache optimization"""
        cache_key = f"{stage}_{hash(str(sorted(context_vars.keys())))}"

        if cache_key in self.prefix_cache:
            self.optimization_stats["cache_hits"] += 1
            return self.prefix_cache[cache_key]

        # Generate stable prefix based on stage
        prefix_templates = {
            "analysis": "You are analyzing a strategic business question. Focus on:",
            "synthesis": "You are synthesizing multiple perspectives. Combine insights from:",
            "research": "You are conducting research. Gather information about:",
            "evaluation": "You are evaluating options. Consider trade-offs and:",
            "default": "You are a strategic advisor. Provide thoughtful analysis of:",
        }

        base_prefix = prefix_templates.get(stage, prefix_templates["default"])

        # Add context variables to prefix if available
        if context_vars:
            var_hints = []
            if "domain" in context_vars:
                var_hints.append(f"domain expertise in {context_vars['domain']}")
            if "urgency" in context_vars:
                var_hints.append(f"urgency level: {context_vars['urgency']}")
            if "stakeholders" in context_vars:
                var_hints.append("stakeholder considerations")

            if var_hints:
                base_prefix += f" {', '.join(var_hints)}."

        self.prefix_cache[cache_key] = base_prefix
        return base_prefix

    def _optimize_prompt(self, prompt: str, context_vars: Dict[str, Any]) -> str:
        """Apply prompt optimization techniques"""
        optimized = prompt

        # Remove redundant phrases
        redundant_phrases = [
            "please help me",
            "i need assistance with",
            "can you help me",
            "i would like to",
            "i want to understand",
            "tell me about",
        ]

        for phrase in redundant_phrases:
            optimized = re.sub(
                rf"\b{re.escape(phrase)}\b", "", optimized, flags=re.IGNORECASE
            )

        # Remove explicit CoT trigger phrases per Prompt Policy
        banned_cot_triggers = [
            "let's think step by step",
            "lets think step by step",
            "think step by step",
            "think step-by-step",
            "chain of thought",
            "cot reasoning",
        ]
        for phrase in banned_cot_triggers:
            optimized = re.sub(rf"{re.escape(phrase)}", "", optimized, flags=re.IGNORECASE)

        # Clean up extra whitespace
        optimized = re.sub(r"\s+", " ", optimized).strip()

        return optimized

    def _optimize_system_prompt(self, system_prompt: str, stable_prefix: str) -> str:
        """Optimize system prompt with stable prefix"""
        if not stable_prefix:
            return system_prompt

        # If system prompt is empty or very short, use stable prefix
        if len(system_prompt.strip()) < 50:
            return stable_prefix

        # Combine stable prefix with existing system prompt
        return f"{stable_prefix}\n\n{system_prompt}"

    def _emit_context_event(self, event_type: ContextEventType, data: Dict[str, Any]):
        """Emit context engineering event to unified context stream"""
        if self.context_stream:
            try:
                self.context_stream.add_event(event_type, data)
            except Exception as e:
                self.logger.warning(f"Failed to emit context event {event_type}: {e}")

    def apply_optimization(
        self, optimization_type: str, metadata: Dict[str, Any] = None
    ):
        """Apply and track context optimization"""
        metadata = metadata or {}

        self._emit_context_event(
            ContextEventType.CONTEXT_OPTIMIZATION_APPLIED,
            {
                "optimization_type": optimization_type,
                "total_tokens_saved": self.optimization_stats["tokens_saved"],
                "total_cache_hits": self.optimization_stats["cache_hits"],
                **metadata,
            },
        )

        self.logger.debug(f"Applied context optimization: {optimization_type}")

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get context optimization statistics"""
        return {
            "total_tokens_saved": self.optimization_stats["tokens_saved"],
            "cache_hits": self.optimization_stats["cache_hits"],
            "cached_prefixes": len(self.prefix_cache),
            "optimization_types_applied": [
                "stable_prefix",
                "redundancy_removal",
                "whitespace_cleanup",
            ],
        }


from src.engine.core.contracts import IResearchProvider, ResearchResult


class LLMManager:
    """
    Orchestrates multiple LLM providers with automatic fallback logic
    Ensures high availability by attempting providers in order of preference

    Also supports research provider abstraction via execute_research().
    """

    def __init__(
        self,
        context_stream=None,
        providers: Optional[List[LLMProvider]] = None,
        research_providers: Optional[Dict[str, IResearchProvider]] = None,
    ):
        self.logger = logging.getLogger(__name__)
        self.context_stream = context_stream

        # OPERATION FINAL COMPLIANCE: Initialize providers in correct order - Grok-4-Fast FIRST
        test_fast = str(os.getenv("TEST_FAST", "")).lower() in {"1", "true", "yes"}
        if test_fast:
            # Strict quarantine: never allow default or external providers in TEST_FAST
            self.logger.info("🧪 TEST_FAST hermetic mode active: default/external providers disabled")
            if providers is None:
                raise ValueError(
                    "Default providers are disabled in TEST_FAST mode. Inject fake providers explicitly."
                )
            # If providers were explicitly supplied, ensure none are external network providers
            for p in providers:
                if isinstance(p, (OpenRouterProvider, DeepSeekProvider, AnthropicProvider)):
                    raise ValueError(
                        "External LLM providers are forbidden in TEST_FAST mode. Use fakes/noop providers."
                    )

        if providers is None:
            # In test environments (outside TEST_FAST), default providers are disabled unless explicitly allowed
            allow_default = str(os.getenv("LLM_MANAGER_ALLOW_DEFAULT_PROVIDERS", "")).lower() in {"1", "true", "yes"}
            if os.getenv("PYTEST_CURRENT_TEST") and not allow_default:
                raise ValueError(
                    "Default providers are disabled during tests. Set LLM_MANAGER_ALLOW_DEFAULT_PROVIDERS=1 to enable."
                )

            # Require API keys for default providers; fail fast if missing
            missing = []
            if not os.getenv("OPENROUTER_API_KEY"):
                missing.append("OPENROUTER_API_KEY")
            if not os.getenv("DEEPSEEK_API_KEY"):
                missing.append("DEEPSEEK_API_KEY")
            if not os.getenv("ANTHROPIC_API_KEY"):
                missing.append("ANTHROPIC_API_KEY")
            if missing:
                raise ValueError(f"Missing required API keys: {', '.join(missing)}")
            providers = [
                OpenRouterProvider(),  # PRIMARY: Grok-4-Fast via OpenRouter (mandated by architecture)
                DeepSeekProvider(),    # SECONDARY: Cost-effective fallback
                AnthropicProvider(),   # TERTIARY: Reliable enterprise fallback
            ]

        self.providers = providers
        # Research providers (abstraction over external research clients like Perplexity)
        self.research_providers: Dict[str, IResearchProvider] = research_providers or {}

        # DEBUG: Check each provider for provider_name attribute
        provider_names = []
        for i, p in enumerate(self.providers):
            if hasattr(p, "provider_name"):
                provider_names.append(p.provider_name)
            else:
                provider_names.append(f"UNKNOWN_PROVIDER_{i}_{type(p).__name__}")
                self.logger.error(
                    f"❌ Provider {i} ({type(p).__name__}) missing provider_name attribute! Available attrs: {[attr for attr in dir(p) if not attr.startswith('_')]}"
                )

        self.logger.info(
            f"🚀 LLMManager initialized with {len(self.providers)} providers: {provider_names}"
        )

        # Performance tracking
        self.call_count = 0
        self.fallback_count = 0
        self.total_cost = 0.0

        # Phase 5: Initialize context compiler for stable prefixes
        self.context_compiler = ContextCompiler(context_stream=self.context_stream)
        self.logger.info(
            "🔧 Phase 5: Context compiler integrated for stable prompt prefixes"
        )

    def _generate_content_fingerprint(self, content: str) -> str:
        """Generate safe fingerprint for content without exposing content"""
        if not content:
            return "empty_content"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]

    def _extract_safe_variables(self, content: str) -> Dict[str, Any]:
        """Extract safe statistical variables from content without exposing text"""
        if not content:
            return {
                "char_count": 0,
                "word_count": 0,
                "line_count": 0,
                "has_questions": False,
                "has_numbers": False,
                "structure_type": "empty",
            }

        # Safe statistical analysis
        char_count = len(content)
        word_count = len(content.split())
        line_count = len(content.split("\n"))

        # Pattern detection (no content exposure)
        has_questions = "?" in content
        has_numbers = bool(re.search(r"\d", content))
        has_json = content.strip().startswith("{") and content.strip().endswith("}")
        has_code = bool(re.search(r"```|def |function |class |import ", content))

        # Determine structure type
        if has_json:
            structure_type = "json"
        elif has_code:
            structure_type = "code"
        elif has_questions:
            structure_type = "interrogative"
        elif word_count < 10:
            structure_type = "brief"
        else:
            structure_type = "narrative"

        return {
            "char_count": char_count,
            "word_count": word_count,
            "line_count": line_count,
            "has_questions": has_questions,
            "has_numbers": has_numbers,
            "structure_type": structure_type,
        }

    def _sanitize_for_glass_box(
        self, content: str, content_type: str
    ) -> Dict[str, Any]:
        """Convert raw content to glass-box safe representation"""
        return {
            f"{content_type}_fingerprint": self._generate_content_fingerprint(content),
            f"{content_type}_variables": self._extract_safe_variables(content),
            f"{content_type}_redacted": True,
            "glass_box_compliant": True,
        }

    def add_provider(self, provider: LLMProvider) -> None:
        """Add a new provider to the end of the provider list"""
        self.providers.append(provider)
        self.logger.info(f"➕ Added provider: {provider.provider_name}")

    async def execute_completion(
        self,
        prompt: str,
        system_prompt: str = "",
        timeout: int = 60,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        **kwargs,
    ) -> LLMResponse:
        """
        Execute LLM completion with automatic fallback

        Args:
            prompt: User prompt text
            system_prompt: System/instructions prompt
            timeout: Per-provider timeout in seconds
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse: Standardized response from successful provider

        Raises:
            RuntimeError: If all providers fail
        """
        self.call_count += 1
        overall_start_time = time.time()
        last_error = None

        self._log_context_event(
            "llm_manager_request",
            {
                "call_count": self.call_count,
                "providers_available": len(self.providers),
                "timeout": timeout,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "prompt_length": len(prompt),
                "system_prompt_length": len(system_prompt) if system_prompt else 0,
            },
        )

        # Phase 5: Apply context compilation for stable prefixes
        stage = kwargs.get("stage", "default")
        context_vars = kwargs.get("context_vars", {})

        compiled_context = self.context_compiler.compile_context(
            prompt=prompt,
            system_prompt=system_prompt,
            stage=stage,
            context_vars=context_vars,
        )

        # Use compiled prompts for LLM calls
        compiled_prompt = compiled_context["compiled_prompt"]
        compiled_system_prompt = compiled_context["compiled_system_prompt"]

        self.logger.debug(
            f"Phase 5: Context compiled - {compiled_context['tokens_saved']} tokens saved"
        )

        for idx, provider in enumerate(self.providers):
            provider_name = provider.provider_name
            is_primary = idx == 0

            try:
                # Create sanitized request data for glass-box transparency
                sanitized_request = {
                    "provider": provider_name,
                    "is_primary": is_primary,
                    "timeout": timeout,
                    "attempt": idx + 1,
                    "total_providers": len(self.providers),
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }

                # Add glass-box compliant prompt data
                sanitized_request.update(self._sanitize_for_glass_box(prompt, "prompt"))
                if system_prompt:
                    sanitized_request.update(
                        self._sanitize_for_glass_box(system_prompt, "system_prompt")
                    )

                # Add sanitized request params
                if kwargs:
                    sanitized_request["request_params_count"] = len(kwargs)
                    sanitized_request["request_params_keys"] = list(kwargs.keys())

                self._log_context_event("llm_provider_request", sanitized_request)

                # Execute with timeout protection using compiled prompts
                # OPERATION REDO: Use provider-specific timeout for OpenRouter
                provider_timeout = timeout
                if hasattr(provider, 'provider_name') and provider.provider_name == 'openrouter':
                    provider_timeout = 180  # 3 minutes for OpenRouter/Grok-4-Fast
                
                response = await asyncio.wait_for(
                    provider.complete(
                        prompt=compiled_prompt,
                        system_prompt=compiled_system_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs,
                    ),
                    timeout=provider_timeout,
                )

                # Track successful response
                self.total_cost += getattr(response, 'cost_usd', getattr(response, 'cost', 0.0))
                overall_time = time.time() - overall_start_time

                # Create sanitized response data for glass-box transparency
                sanitized_response = {
                    "provider": provider_name,
                    "status": "success",
                    "is_primary": is_primary,
                    "cost": getattr(response, 'cost_usd', getattr(response, 'cost', 0.0)),
                    "tokens": getattr(response, 'tokens_used', getattr(response, 'total_tokens', 0)),
                    "provider_time": getattr(response, 'response_time_ms', getattr(response, 'time_seconds', 0)) / 1000.0,
                    "overall_time": overall_time,
                    "response_length": len(getattr(response, 'raw_text', getattr(response, 'content', ''))),
                    "response_metadata": {
                        "model": getattr(response, "model", None),
                        "finish_reason": getattr(response, "finish_reason", None),
                        "usage_prompt_tokens": getattr(
                            getattr(response, "usage", {}), "prompt_tokens", None
                        ),
                        "usage_completion_tokens": getattr(
                            getattr(response, "usage", {}), "completion_tokens", None
                        ),
                        "usage_total_tokens": getattr(
                            getattr(response, "usage", {}), "total_tokens", None
                        ),
                    },
                }

                # Add glass-box compliant response content
                sanitized_response.update(
                    self._sanitize_for_glass_box(getattr(response, 'raw_text', getattr(response, 'content', '')), "response")
                )

                # Add safe provider response metadata (no raw content)
                raw_provider_response = getattr(response, "raw_provider_response", None)
                if raw_provider_response:
                    sanitized_response["provider_response_has_data"] = True
                    sanitized_response["provider_response_type"] = type(
                        raw_provider_response
                    ).__name__
                    if isinstance(raw_provider_response, dict):
                        sanitized_response["provider_response_keys"] = list(
                            raw_provider_response.keys()
                        )

                self._log_context_event("llm_provider_response", sanitized_response)

                self._log_context_event(
                    "llm_manager_response",
                    {
                        "status": "success",
                        "provider_used": provider_name,
                        "fallback_attempts": idx,
                        "total_cost": self.total_cost,
                        "call_count": self.call_count,
                    },
                )

                # Track fallback usage
                if not is_primary:
                    self.fallback_count += 1

                self.logger.info(
                    f"✅ LLM completion successful via {provider_name} "
                    f"(attempt {idx + 1}/{len(self.providers)}, "
                    f"cost: ${getattr(response, 'cost_usd', getattr(response, 'cost', 0.0)):.4f}, "
                    f"tokens: {getattr(response, 'tokens_used', getattr(response, 'total_tokens', 0))}"
                )

                return response

            except asyncio.TimeoutError:
                timeout_error = f"Provider {provider_name} timed out after {timeout}s"
                last_error = timeout_error

                self._log_context_event(
                    "llm_provider_response",
                    {
                        "provider": provider_name,
                        "status": "timeout",
                        "is_primary": is_primary,
                        "timeout": timeout,
                        "error": timeout_error,
                    },
                )

                self.logger.warning(f"⏱️ {timeout_error}")

            except Exception as e:
                error_msg = str(e)
                last_error = error_msg

                self._log_context_event(
                    "llm_provider_response",
                    {
                        "provider": provider_name,
                        "status": "failure",
                        "is_primary": is_primary,
                        "error": error_msg[:500],  # Truncate long error messages
                    },
                )

                self.logger.warning(f"❌ Provider {provider_name} failed: {error_msg}")

            # Log fallback attempt (if not the last provider)
            if idx < len(self.providers) - 1:
                next_provider = self.providers[idx + 1]

                self._log_context_event(
                    "llm_provider_fallback",
                    {
                        "failed_provider": provider_name,
                        "next_provider": next_provider.provider_name,
                        "fallback_attempt": idx + 1,
                        "remaining_providers": len(self.providers) - idx - 1,
                    },
                )

                self.logger.info(
                    f"🔄 Falling back from {provider_name} to {next_provider.provider_name}"
                )

        # All providers failed
        overall_time = time.time() - overall_start_time
        failure_msg = (
            f"All {len(self.providers)} LLM providers failed. Last error: {last_error}"
        )

        self._log_context_event(
            "llm_manager_response",
            {
                "status": "complete_failure",
                "providers_attempted": len(self.providers),
                "total_time": overall_time,
                "last_error": str(last_error)[:500],
            },
        )

        self.logger.error(f"💥 {failure_msg}")
        raise RuntimeError(failure_msg)

    async def call_llm(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        provider: str = None,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        response_format: Dict[str, str] = None,
        **kwargs,
    ) -> Any:
        """
        Compatibility method for senior advisor's call_llm interface.

        Converts messages format to prompt/system_prompt and calls execute_completion.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            model: Model name (currently ignored, provider preference used)
            provider: Provider name (currently ignored, fallback order used)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            response_format: Response format specification
            **kwargs: Additional parameters

        Returns:
            LLMResponse with .content attribute containing the response text
        """
        # Convert messages to prompt and system_prompt
        system_prompt = ""
        user_messages = []

        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")

            if role == "system":
                system_prompt = content
            elif role == "user":
                user_messages.append(content)
            elif role == "assistant":
                # Handle assistant messages if needed
                user_messages.append(f"Assistant: {content}")

        # Combine user messages into a single prompt
        prompt = "\n\n".join(user_messages)

        # Handle JSON response format by adding instruction to system prompt
        if response_format and response_format.get("type") == "json_object":
            system_prompt += "\n\nIMPORTANT: Respond only with valid JSON. Do not include any text before or after the JSON object."

        # Call the underlying execute_completion method
        response = await self.execute_completion(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        # Create a wrapper object with .content attribute for compatibility
        # Since LLMResponse is a Pydantic model, we can't directly add attributes
        class ResponseWrapper:
            def __init__(self, llm_response):
                self.llm_response = llm_response
                self.content = getattr(llm_response, 'raw_text', getattr(llm_response, 'content', ''))
                # Forward all other attributes to the wrapped response
                for attr in [
                    "raw_text",
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                    "cost",
                    "time_seconds",
                    "model_name",
                    "provider_name",
                    "metadata",
                    "raw_provider_response",
                ]:
                    setattr(self, attr, getattr(llm_response, attr, None))

        return ResponseWrapper(response)

    def _log_context_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log event to context stream if available"""
        if self.context_stream:
            try:
                # Map string event types to ContextEventType enums
                event_type_map = {
                    "llm_manager_request": ContextEventType.TOOL_EXECUTION,
                    "llm_provider_request": ContextEventType.LLM_PROVIDER_REQUEST,
                    "llm_provider_response": ContextEventType.LLM_PROVIDER_RESPONSE,
                    "llm_manager_response": ContextEventType.PROCESSING_COMPLETE,
                    "llm_provider_fallback": ContextEventType.ERROR_RECOVERED,
                }

                context_event_type = event_type_map.get(event_type)
                if context_event_type:
                    self.context_stream.add_event(context_event_type, data)
                else:
                    self.logger.warning(
                        f"Unknown event type for context logging: {event_type}"
                    )
            except Exception as e:
                # Don't fail the LLM call due to logging issues
                self.logger.debug(f"Context stream logging failed: {e}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for monitoring"""
        return {
            "total_calls": self.call_count,
            "fallback_calls": self.fallback_count,
            "fallback_rate": (
                (self.fallback_count / self.call_count) if self.call_count > 0 else 0.0
            ),
            "total_cost": self.total_cost,
            "average_cost_per_call": (
                (self.total_cost / self.call_count) if self.call_count > 0 else 0.0
            ),
            "providers_count": len(self.providers),
            "providers": [p.provider_name for p in self.providers],
        }

    async def execute_research(
        self,
        provider_name: str,
        query_text: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> ResearchResult:
        """Execute research via the selected provider abstraction."""
        if not self.research_providers or provider_name not in self.research_providers:
            raise RuntimeError(f"Research provider '{provider_name}' is not configured")
        provider = self.research_providers[provider_name]
        start = time.time()
        result = await provider.query(query_text, config or {})
        # Emit a glass-box event with safe metadata
        self._log_context_event(
            "llm_manager_response",
            {
                "status": "research_success",
                "provider": provider_name,
                "overall_time": time.time() - start,
                "sources": len(result.sources),
                "confidence": result.confidence,
            },
        )
        return result

    async def health_check(self) -> Dict[str, Any]:
        """Check health status of all providers"""
        health_results = {}

        for provider in self.providers:
            try:
                is_available = await provider.is_available()
                health_results[provider.provider_name] = {
                    "status": "healthy" if is_available else "unhealthy",
                    "available": is_available,
                }
            except Exception as e:
                health_results[provider.provider_name] = {
                    "status": "error",
                    "available": False,
                    "error": str(e),
                }

        # Overall health assessment
        healthy_providers = sum(
            1 for result in health_results.values() if result["available"]
        )
        overall_healthy = healthy_providers > 0

        return {
            "overall_status": "healthy" if overall_healthy else "unhealthy",
            "healthy_providers": healthy_providers,
            "total_providers": len(self.providers),
            "providers": health_results,
        }


# Global instance for dependency injection
_global_llm_manager: Optional[LLMManager] = None


def get_llm_manager(context_stream=None) -> LLMManager:
    """Get or create global LLMManager instance"""
    global _global_llm_manager

    if _global_llm_manager is None:
        import os as _os
        # During TEST_FAST, always use a noop provider (ignore LLM_MANAGER_ALLOW_DEFAULT_PROVIDERS)
        if str(_os.getenv("TEST_FAST", "")).lower() in {"1", "true", "yes"}:
            class _NoopProvider(LLMProvider):
                def __init__(self):
                    super().__init__("noop")
                async def complete(self, prompt: str, system_prompt: str = "", **kwargs) -> LLMResponse:
                    return LLMResponse(
                        raw_text="",
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        cost=0.0,
                        time_seconds=0.0,
                        model_name="noop-model",
                        provider_name=self.provider_name,
                        metadata={"test_mode": True, "mode": "TEST_FAST"},
                    )
            _global_llm_manager = LLMManager(context_stream=context_stream, providers=[_NoopProvider()])
        # Otherwise, if under pytest (but not TEST_FAST), optionally allow defaults based on flag
        elif _os.getenv("PYTEST_CURRENT_TEST") and str(_os.getenv("LLM_MANAGER_ALLOW_DEFAULT_PROVIDERS", "")).lower() not in {"1", "true", "yes"}:
            class _NoopProvider(LLMProvider):
                def __init__(self):
                    super().__init__("noop")
                async def complete(self, prompt: str, system_prompt: str = "", **kwargs) -> LLMResponse:
                    return LLMResponse(
                        raw_text="",
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        cost=0.0,
                        time_seconds=0.0,
                        model_name="noop-model",
                        provider_name=self.provider_name,
                        metadata={"test_mode": True, "mode": "PYTEST"},
                    )
            _global_llm_manager = LLMManager(context_stream=context_stream, providers=[_NoopProvider()])
        else:
            _global_llm_manager = LLMManager(context_stream=context_stream)

    # Update context stream if provided
    if context_stream and _global_llm_manager.context_stream != context_stream:
        _global_llm_manager.context_stream = context_stream

    return _global_llm_manager


def reset_llm_manager() -> None:
    """Reset global LLMManager instance (primarily for testing)"""
    global _global_llm_manager
    _global_llm_manager = None
