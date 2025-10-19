"""
Base LLM Provider Interface
Standardizes the interface for all LLM providers with fallback capabilities
"""

import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pydantic import BaseModel


class LLMResponse(BaseModel):
    """Standardized response format for all LLM providers"""

    raw_text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float
    time_seconds: float
    model_name: str
    provider_name: str
    metadata: Optional[Dict[str, Any]] = None
    raw_provider_response: Optional[Dict[str, Any]] = (
        None  # RADICAL TRANSPARENCY: Complete API response
    )


class LLMProvider(ABC):
    """Abstract base class for all LLM providers"""

    def __init__(self, provider_name: str):
        self.provider_name = provider_name

    @abstractmethod
    async def complete(
        self, prompt: str, system_prompt: str = "", **kwargs
    ) -> LLMResponse:
        """
        Execute a completion and return a standardized response

        Args:
            prompt: The user prompt text
            system_prompt: System/instructions prompt
            **kwargs: Provider-specific parameters

        Returns:
            LLMResponse: Standardized response object

        Raises:
            Exception: On API errors, timeouts, or other failures
        """
        pass

    def _create_response(
        self,
        raw_text: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost: float,
        start_time: float,
        model_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        raw_provider_response: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        """Helper method to create standardized response"""
        return LLMResponse(
            raw_text=raw_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost=cost,
            time_seconds=time.time() - start_time,
            model_name=model_name,
            provider_name=self.provider_name,
            metadata=metadata or {},
            raw_provider_response=raw_provider_response,
        )
