"""
Configuration compatibility layer
Provides backward compatibility imports for src.config
"""

# Import from engine config and re-export
from src.engine.config import (
    PRIMARY_PROVIDER,
    FALLBACK_PROVIDER,
    PROVIDER_TIMEOUTS,
    CognitiveEngineSettings,
    DeepSeekConfig,
    ULTRA_COMPLEX_CONFIG,
    STANDARD_COMPLEX_CONFIG,
    FAST_RESPONSE_CONFIG,
    get_settings,
    MetisSettings,
)

# Legacy compatibility exports
__all__ = [
    "PRIMARY_PROVIDER",
    "FALLBACK_PROVIDER",
    "PROVIDER_TIMEOUTS",
    "CognitiveEngineSettings",
    "DeepSeekConfig",
    "ULTRA_COMPLEX_CONFIG",
    "STANDARD_COMPLEX_CONFIG",
    "FAST_RESPONSE_CONFIG",
    "get_settings",
    "MetisSettings",
]
