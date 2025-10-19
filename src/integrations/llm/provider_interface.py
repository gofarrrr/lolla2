#!/usr/bin/env python3
"""
Provider Interface for LLM Integrations
Abstract base classes and common data structures for LLM providers
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime


class LLMProvider(str, Enum):
    """Supported LLM providers"""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    CLAUDE = "claude"  # Alias for Anthropic
    DEEPSEEK = "deepseek"


@dataclass
class LLMResponse:
    """Structured LLM response"""

    content: str
    provider: str
    model: str
    tokens_used: int
    cost_usd: float
    response_time_ms: int
    reasoning_steps: List[Dict[str, Any]]
    mental_models: List[str]
    confidence: float
    metadata: Optional[Dict[str, Any]] = (
        None  # NATIVE INTEGRATION: Function calls and tool usage
    )


@dataclass
class CognitiveAnalysisResult:
    """Result of cognitive analysis phase"""

    mental_models_selected: List[str]
    reasoning_description: str
    key_insights: List[str]
    confidence_score: float
    research_requirements: List[str]
    raw_response: str
    # Neural Lace tracking fields
    tokens_used: int = 0
    cost_usd: float = 0.0
    response_time_ms: int = 0


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers"""

    def __init__(self, api_key: str, base_url: Optional[str] = None):
        self.api_key = api_key
        self.base_url = base_url
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._call_history = []

    @abstractmethod
    async def call_llm(self, messages: List[Dict], model: str, **kwargs) -> LLMResponse:
        """Make LLM API call - must be implemented by provider"""
        pass

    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Get list of available models for this provider"""
        pass

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if provider is available and properly configured"""
        pass

    def get_call_history(self) -> List[Dict]:
        """Get call history for this provider"""
        return self._call_history.copy()

    def get_total_cost(self) -> float:
        """Get total cost for this provider"""
        return sum(call.get("cost", 0) for call in self._call_history)

    def record_call(self, response: LLMResponse, method: str):
        """Record a call in the history"""
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


class ProviderError(Exception):
    """Base exception for provider errors"""

    pass


class ProviderUnavailableError(ProviderError):
    """Raised when a provider is unavailable"""

    pass


class ProviderAPIError(ProviderError):
    """Raised when a provider API call fails"""

    pass


class InvalidResponseError(ProviderError):
    """Raised when a provider returns invalid response"""

    pass
