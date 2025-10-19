#!/usr/bin/env python3
"""
Configuration package for METIS DeepSeek optimization
"""

from .deepseek_configs import (
    DeepSeekConfig,
    ULTRA_COMPLEX_CONFIG,
    STANDARD_COMPLEX_CONFIG,
    FAST_RESPONSE_CONFIG,
    CONFIG_PROFILES,
    select_optimal_config,
    get_config_by_name,
    get_all_configs,
    calculate_complexity_multiplier,
)

# Re-export main config constants for backward compatibility
import sys
import importlib.util
from pathlib import Path

# Import from parent config.py file
config_module_path = Path(__file__).parent.parent / "config.py"
spec = importlib.util.spec_from_file_location("main_config", config_module_path)
main_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main_config)

# Re-export required constants
PRIMARY_PROVIDER = main_config.PRIMARY_PROVIDER
FALLBACK_PROVIDER = main_config.FALLBACK_PROVIDER
PROVIDER_TIMEOUTS = main_config.PROVIDER_TIMEOUTS
CognitiveEngineSettings = main_config.CognitiveEngineSettings
get_settings = main_config.get_settings
MetisSettings = main_config.MetisSettings

__all__ = [
    "DeepSeekConfig",
    "ULTRA_COMPLEX_CONFIG",
    "STANDARD_COMPLEX_CONFIG",
    "FAST_RESPONSE_CONFIG",
    "CONFIG_PROFILES",
    "select_optimal_config",
    "get_config_by_name",
    "get_all_configs",
    "calculate_complexity_multiplier",
    "PRIMARY_PROVIDER",
    "FALLBACK_PROVIDER",
    "PROVIDER_TIMEOUTS",
    "CognitiveEngineSettings",
    "get_settings",
    "MetisSettings",
]
