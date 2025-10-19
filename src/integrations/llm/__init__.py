"""
LLM Integration Package
Provides unified LLM providers and interfaces
"""

# Import core provider classes and enums
from .provider_interface import (
    LLMProvider,
    BaseLLMProvider,
    LLMResponse,
    CognitiveAnalysisResult,
)
from .unified_client import UnifiedLLMClient, get_unified_llm_client
from .claude_provider import ClaudeProvider
from src.engine.integrations.llm.deepseek_provider import DeepSeekProvider
from .openai_provider import OpenAIProvider

# Export main classes for external use
__all__ = [
    "LLMProvider",
    "BaseLLMProvider",
    "LLMResponse",
    "CognitiveAnalysisResult",
    "UnifiedLLMClient",
    "get_unified_llm_client",
    "ClaudeProvider",
    "DeepSeekProvider",
    "OpenAIProvider",
]
