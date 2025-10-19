#!/usr/bin/env python3
"""
LLM Provider Integration for METIS Cognitive Engine
REFACTORED: Now imports from modular provider-specific implementations

This file maintains backward compatibility while delegating to the new modular architecture:
- src/integrations/llm/provider_interface.py - Abstract interfaces and data classes
- src/integrations/llm/claude_provider.py - Claude/Anthropic implementation
- src/integrations/llm/openai_provider.py - OpenAI implementation
- src/integrations/llm/cognitive_analyzer.py - Provider-agnostic cognitive analysis
- src/integrations/llm/unified_client.py - Main orchestration client

For new code, import directly from src.integrations.llm package.
"""

import logging
from pathlib import Path
from typing import Optional

# Import refactored components
from .llm import (
    UnifiedLLMClient as RefactoredLLMProviderClient,
)

# Load environment variables for backward compatibility
try:
    from dotenv import load_dotenv
    import os

    env_path = Path(__file__).parent.parent.parent / ".env"

    if env_path.exists():
        load_dotenv(env_path, override=True)

        # Verify key environment variables were loaded
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            print(
                f"✅ ANTHROPIC_API_KEY loaded: {anthropic_key[:10]}...{anthropic_key[-4:]}"
            )
        else:
            print("❌ ANTHROPIC_API_KEY not found after loading .env")
    else:
        print(f"❌ .env file not found at {env_path}")

except ImportError:
    print("⚠️ python-dotenv not available - environment loading may fail")

logger = logging.getLogger(__name__)


# Backward compatibility wrapper
class LLMProviderClient(RefactoredLLMProviderClient):
    """
    Backward compatibility wrapper for the refactored LLMProviderClient

    This class delegates all functionality to the new UnifiedLLMClient while
    maintaining the exact same interface as the original implementation.
    """

    def __init__(self):
        super().__init__()
        self.logger.info(
            "🔄 LLMProviderClient: Using refactored modular implementation"
        )

        # Legacy compatibility - expose internal state for any code that might access it
        self._clients = self._get_legacy_clients_dict()

    def _get_legacy_clients_dict(self):
        """Create legacy-compatible _clients dict for backward compatibility"""
        legacy_clients = {}

        for provider_name, provider in self._providers.items():
            if provider_name == "anthropic":
                legacy_clients["anthropic"] = {
                    "api_key": provider.api_key,
                    "base_url": provider.base_url,
                    "models": provider.get_available_models(),
                }
            elif provider_name == "openai":
                legacy_clients["openai"] = {
                    "api_key": provider.api_key,
                    "base_url": provider.base_url,
                    "models": provider.get_available_models(),
                }

        return legacy_clients

    # Legacy method aliases for backward compatibility
    async def _call_best_available_provider(self, messages, **kwargs):
        """Legacy method alias"""
        return await self.call_best_available_provider(messages, **kwargs)

    async def _call_anthropic(self, messages, **kwargs):
        """Legacy method alias - delegates to Claude provider"""
        if "anthropic" in self._providers:
            provider = self._providers["anthropic"]
            return await provider.call_llm(messages, **kwargs)
        else:
            raise Exception("Anthropic provider not available")

    async def _call_openai(self, messages, **kwargs):
        """Legacy method alias - delegates to OpenAI provider"""
        if "openai" in self._providers:
            provider = self._providers["openai"]
            return await provider.call_llm(messages, **kwargs)
        else:
            raise Exception("OpenAI provider not available")


# Global instance for backward compatibility
_llm_client_instance: Optional[LLMProviderClient] = None


def get_llm_client() -> LLMProviderClient:
    """Get or create global LLM client instance (backward compatibility)"""
    global _llm_client_instance

    if _llm_client_instance is None:
        _llm_client_instance = LLMProviderClient()

    return _llm_client_instance


def get_llm_provider() -> LLMProviderClient:
    """Alias for get_llm_client() for backward compatibility"""
    return get_llm_client()
