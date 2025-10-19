"""
Adapter for resilient_llm_client to fix import path issues.
This bridges the import from src.core to src.engine.core for backward compatibility.
"""

# Re-export everything from the actual resilient_llm_client
from src.engine.core.resilient_llm_client import (
    CognitiveCallContext,
    LLMCallResult,
    get_resilient_llm_client,
)

# For any legacy imports that expect these
__all__ = ["CognitiveCallContext", "LLMCallResult", "get_resilient_llm_client"]
