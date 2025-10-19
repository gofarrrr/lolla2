"""
COMPATIBILITY BRIDGE: DeepSeek Provider Import Redirection
==========================================================

This module provides backward compatibility for imports expecting DeepSeek provider
at the old location (src.integrations.llm.deepseek_provider) by redirecting to the
canonical location (src.engine.integrations.llm.deepseek_provider).

This bridge is part of Operation Unification cleanup to ensure zero regression.
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "Importing from 'src.integrations.llm.deepseek_provider' is deprecated. "
    "Please use 'src.engine.integrations.llm.deepseek_provider' instead. "
    "This compatibility bridge will be removed in V6.0.",
    DeprecationWarning,
    stacklevel=2,
)

# Import from canonical location and re-export
from src.engine.integrations.llm.deepseek_provider import (
    DeepSeekProvider,
    EnhancedDeepSeekProvider,
    DeepSeekV31OptimizedProvider,
)

# Re-export for compatibility
__all__ = [
    "DeepSeekProvider",
    "EnhancedDeepSeekProvider",
    "DeepSeekV31OptimizedProvider",
]
