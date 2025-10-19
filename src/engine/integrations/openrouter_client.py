#!/usr/bin/env python3
"""
OpenRouter Client for Grok-4 and Grok-4-Fast Integration

Provides access to xAI's Grok models via OpenRouter's API:
- Grok-4-Fast: Free, cost-efficient for easy tasks
- Grok-4: Advanced reasoning for demanding tasks

Model specifications:
- Grok-4-Fast: $0.20/1M input, $0.50/1M output, 2M context, 40% more efficient
- Grok-4: $3/1M input, $15/1M output, full reasoning capabilities
"""

import asyncio
import aiohttp
import logging
import time
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from ..providers.llm.base import LLMProvider, LLMResponse


@dataclass
class OpenRouterConfig:
    """Configuration for OpenRouter API requests"""
    base_url: str = "https://openrouter.ai/api/v1"
    timeout: int = 180  # Increased to 3 minutes for complex queries
    max_retries: int = 2  # Reduced retries to avoid long waits
    
    # Grok model configurations - USING PAID TIER FOR RELIABLE PERFORMANCE
    grok_4_fast_model: str = "x-ai/grok-4-fast"  # PAID VERSION - More reliable
    grok_4_model: str = "x-ai/grok-4"  # Paid version with full reasoning
    
    # Pricing (per 1M tokens) - PAID TIER PRICING
    grok_4_fast_input_cost: float = 0.20  # Paid tier: $0.20/1M input tokens
    grok_4_fast_output_cost: float = 0.50  # Paid tier: $0.50/1M output tokens
    grok_4_input_cost: float = 3.0
    grok_4_output_cost: float = 15.0


class OpenRouterClient(LLMProvider):
    """
    OpenRouter client for accessing Grok models with policy-based selection.
    
    Supports both regular Grok-4 and Grok-4-Fast variants with reasoning control.
    """

    def __init__(self, api_key: str = None, global_semaphore: asyncio.Semaphore = None):
        super().__init__("openrouter")
        
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable.")
        
        self.config = OpenRouterConfig()
        self.logger = logging.getLogger(__name__)
        
        # Global concurrency control for batch processing
        self.global_semaphore = global_semaphore or asyncio.Semaphore(5)
        
        # Rate limiting: 60 requests per minute (1 per second)
        self.rate_limiter = asyncio.Semaphore(10)  # Allow 10 concurrent API calls
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
        # Session for connection pooling
        self._session: Optional[aiohttp.ClientSession] = None
        
        self.logger.info("🚀 OpenRouter client initialized for Grok-4 models with concurrency control")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session with connection pooling"""
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(
                limit=50,  # Increased connection pool size for batch processing
                limit_per_host=20,  # Per-host connection limit
                ttl_dns_cache=300,  # DNS cache TTL
                use_dns_cache=True,
                enable_cleanup_closed=True,  # Clean up closed connections
            )
            
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://lolla.ai",  # Optional: For analytics
                    "X-Title": "METIS V5.3 Cognitive Platform",  # Optional: App identification
                }
            )
        
        return self._session
    
    async def complete(
        self,
        prompt: str,
        system_prompt: str = "",
        model: str = None,
        policy: str = "fast",
        reasoning_enabled: bool = None,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        **kwargs
    ) -> LLMResponse:
        """
        Execute completion with policy-based model selection.
        
        Args:
            prompt: User prompt text
            system_prompt: System/instructions prompt  
            model: Explicit model name (overrides policy)
            policy: Selection policy ("fast", "quality", "default")
            reasoning_enabled: Enable reasoning for demanding tasks
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse: Standardized response object
        """
        start_time = time.time()
        
        # Model selection based on policy
        selected_model = self._select_model(model, policy, reasoning_enabled)
        
        # Determine if reasoning should be enabled
        use_reasoning = self._should_enable_reasoning(policy, reasoning_enabled, selected_model)
        
        self.logger.info(f"🔄 OpenRouter request: model={selected_model}, reasoning={use_reasoning}")
        
        # Build request payload
        payload = self._build_request_payload(
            prompt=prompt,
            system_prompt=system_prompt,
            model=selected_model,
            reasoning_enabled=use_reasoning,
            temperature=temperature,
            max_tokens=max_tokens,
            policy=policy,
            **kwargs
        )
        
        # Execute with global concurrency control and retry logic
        async with self.global_semaphore:  # Global concurrency control
            async with self.rate_limiter:  # Rate limiting
                # Enforce minimum interval between requests
                current_time = time.time()
                time_since_last = current_time - self.last_request_time
                if time_since_last < self.min_request_interval:
                    await asyncio.sleep(self.min_request_interval - time_since_last)
                
                self.last_request_time = time.time()
                
                for attempt in range(self.config.max_retries):
                    try:
                        response = await self._execute_request(payload)
                        
                        # Parse and return standardized response
                        return self._parse_response(
                            response=response,
                            model=selected_model,
                            start_time=start_time,
                            use_reasoning=use_reasoning
                        )
                        
                    except Exception as e:
                        self.logger.warning(f"⚠️ OpenRouter attempt {attempt + 1} failed: {e}")
                        
                        # Check for rate limit errors
                        if "rate limit" in str(e).lower() or "429" in str(e):
                            # Longer backoff for rate limits
                            backoff_time = min(60, 5 * (2 ** attempt))
                            self.logger.warning(f"🚧 Rate limit detected, backing off for {backoff_time}s")
                            await asyncio.sleep(backoff_time)
                        elif attempt == self.config.max_retries - 1:
                            raise
                        else:
                            # Standard exponential backoff
                            await asyncio.sleep(2 ** attempt)
    
    def _select_model(self, explicit_model: str, policy: str, reasoning_enabled: bool) -> str:
        """Select appropriate Grok model based on policy and requirements"""
        
        # Explicit model overrides policy
        if explicit_model:
            return explicit_model
        
        # Policy-based selection - DEFAULT TO GROK-4-FAST for cost efficiency
        if policy == "quality":
            # Only use expensive Grok-4 for quality policy
            return self.config.grok_4_model
        else:  # "fast", "default", or any other policy
            # Always use cost-efficient Grok-4-Fast unless explicitly requesting quality
            return self.config.grok_4_fast_model
    
    def _should_enable_reasoning(self, policy: str, explicit_reasoning: bool, model: str) -> bool:
        """Determine if reasoning should be enabled based on context"""
        
        # Explicit setting overrides
        if explicit_reasoning is not None:
            return explicit_reasoning
        
        # Policy-based reasoning - Conservative approach for cost efficiency
        if policy == "quality":
            # Enable reasoning for quality policy (using expensive Grok-4)
            return True
        else:  # "fast", "default", or any other policy
            # Disable reasoning by default to keep costs low with Grok-4-Fast
            # Users can explicitly enable if needed
            return False
    
    def _build_request_payload(
        self,
        prompt: str,
        system_prompt: str,
        model: str,
        reasoning_enabled: bool,
        temperature: float,
        max_tokens: int,
        policy: str = "default",
        **kwargs
    ) -> Dict[str, Any]:
        """Build OpenRouter API request payload"""
        
        # Build messages array
        messages = []
        
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        messages.append({
            "role": "user", 
            "content": prompt
        })
        
        # Base payload
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        # Add reasoning control for Grok models
        # OpenRouter expects reasoning as an object with effort level
        if reasoning_enabled and ("grok" in model.lower()):
            # Use "high" effort for quality policy, "medium" for others
            effort = "high" if policy == "quality" else "medium"
            payload["reasoning"] = {"effort": effort}
        
        # Add optional parameters
        if "top_p" in kwargs:
            payload["top_p"] = kwargs["top_p"]
        if "frequency_penalty" in kwargs:
            payload["frequency_penalty"] = kwargs["frequency_penalty"]
        if "presence_penalty" in kwargs:
            payload["presence_penalty"] = kwargs["presence_penalty"]
        
        return payload
    
    async def _execute_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute HTTP request to OpenRouter API"""
        
        session = await self._get_session()
        url = f"{self.config.base_url}/chat/completions"
        
        try:
            async with session.post(url, json=payload) as response:
                
                # Check for HTTP errors
                if response.status != 200:
                    error_text = await response.text()
                    
                    # T-13 UNICODE HARDENING: Capture corrupted JSON payload for "dangling surrogate" errors
                    if response.status == 400 and "low surrogate" in error_text:
                        import json
                        import os
                        from datetime import datetime
                        
                        # Capture the corrupted payload for debugging
                        corrupted_filename = f"corrupted_payload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                        corrupted_path = os.path.join(os.getcwd(), corrupted_filename)
                        
                        try:
                            with open(corrupted_path, 'w', encoding='utf-8') as f:
                                json.dump(payload, f, indent=2, ensure_ascii=True)
                            
                            self.logger.error(f"🐛 T-13 UNICODE HARDENING: Dangling surrogate detected! Corrupted payload saved to {corrupted_path}")
                            self.logger.error(f"🐛 Error details: {error_text}")
                            
                        except Exception as capture_error:
                            self.logger.error(f"🐛 T-13 UNICODE HARDENING: Failed to capture corrupted payload: {capture_error}")
                    
                    raise Exception(f"OpenRouter API error {response.status}: {error_text}")
                
                # Parse JSON response
                result = await response.json()
                
                # Check for API-level errors
                if "error" in result:
                    raise Exception(f"OpenRouter API error: {result['error']}")
                
                return result
                
        except Exception as e:
            # T-13 UNICODE HARDENING: Additional logging for JSON encoding errors
            if "surrogate" in str(e).lower():
                self.logger.error(f"🐛 T-13 UNICODE HARDENING: Unicode surrogate error during request: {e}")
                
                # Also try to save the payload in this case
                try:
                    import json
                    import os
                    from datetime import datetime
                    
                    corrupted_filename = f"corrupted_payload_exception_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    corrupted_path = os.path.join(os.getcwd(), corrupted_filename)
                    
                    with open(corrupted_path, 'w', encoding='utf-8') as f:
                        json.dump(payload, f, indent=2, ensure_ascii=True)
                    
                    self.logger.error(f"🐛 T-13 UNICODE HARDENING: Exception payload saved to {corrupted_path}")
                    
                except Exception as capture_error:
                    self.logger.error(f"🐛 T-13 UNICODE HARDENING: Failed to capture exception payload: {capture_error}")
            
            raise
    
    def _parse_response(
        self,
        response: Dict[str, Any],
        model: str,
        start_time: float,
        use_reasoning: bool
    ) -> LLMResponse:
        """Parse OpenRouter response into standardized format"""
        
        # Extract response content
        choices = response.get("choices", [])
        if not choices:
            raise Exception("No choices in OpenRouter response")
        
        message = choices[0].get("message", {})
        content = message.get("content", "")
        
        # Extract usage information
        usage = response.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
        
        # Calculate cost based on model
        cost = self._calculate_cost(model, prompt_tokens, completion_tokens)
        
        # Build metadata
        metadata = {
            "reasoning_enabled": use_reasoning,
            "finish_reason": choices[0].get("finish_reason"),
            "model": response.get("model", model),
            "provider_id": response.get("id"),
        }
        
        # Add reasoning tokens if available (Grok-specific)
        if "reasoning_tokens" in usage:
            metadata["reasoning_tokens"] = usage["reasoning_tokens"]
            metadata["thinking_time"] = response.get("thinking_time", 0)
        
        return self._create_response(
            raw_text=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=cost,
            start_time=start_time,
            model_name=model,
            metadata=metadata,
            raw_provider_response=response
        )
    
    def _calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate request cost based on model pricing"""
        
        if "grok-4-fast" in model.lower():
            # Currently free during beta
            input_cost = prompt_tokens * self.config.grok_4_fast_input_cost / 1_000_000
            output_cost = completion_tokens * self.config.grok_4_fast_output_cost / 1_000_000
        else:  # grok-4
            input_cost = prompt_tokens * self.config.grok_4_input_cost / 1_000_000
            output_cost = completion_tokens * self.config.grok_4_output_cost / 1_000_000
        
        return input_cost + output_cost
    
    async def is_available(self) -> bool:
        """Check if OpenRouter API is available"""
        try:
            session = await self._get_session()
            url = f"{self.config.base_url}/models"
            
            async with session.get(url) as response:
                return response.status == 200
                
        except Exception as e:
            self.logger.warning(f"OpenRouter availability check failed: {e}")
            return False
    
    async def close(self):
        """Close the HTTP session"""
        if self._session and not self._session.closed:
            await self._session.close()
    
    def __del__(self):
        """Cleanup on deletion"""
        if hasattr(self, '_session') and self._session and not self._session.closed:
            # Note: This will show a warning about unclosed session
            # In production, always call close() explicitly
            pass