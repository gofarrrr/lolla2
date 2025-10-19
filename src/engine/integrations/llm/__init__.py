#!/usr/bin/env python3
"""
LLM Integration Package
Unified interface for multiple LLM providers with cognitive analysis capabilities
"""

from .provider_interface import (
    LLMProvider,
    LLMResponse,
    CognitiveAnalysisResult,
    BaseLLMProvider,
    ProviderError,
    ProviderUnavailableError,
    ProviderAPIError,
    InvalidResponseError,
)

from .claude_provider import ClaudeProvider
from .openai_provider import OpenAIProvider
from .cognitive_analyzer import CognitiveAnalyzer

# NOTE: UnifiedLLMClient removed from __init__ to avoid circular import
# Import directly from src.integrations.llm.unified_client if needed

__all__ = [
    # Core interfaces and data classes
    "LLMProvider",
    "LLMResponse",
    "CognitiveAnalysisResult",
    "BaseLLMProvider",
    # Exceptions
    "ProviderError",
    "ProviderUnavailableError",
    "ProviderAPIError",
    "InvalidResponseError",
    # Provider implementations
    "ClaudeProvider",
    "OpenAIProvider",
    # Core components
    "CognitiveAnalyzer",
]

# Package metadata
__version__ = "2.0.0"
__author__ = "METIS Cognitive Platform"
__description__ = "Unified LLM provider interface with cognitive analysis capabilities"
