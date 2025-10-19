"""
DeepSeek Provider - Compatibility wrapper for the canonical optimized provider
This wrapper maintains backward compatibility while using the optimized V3.1 provider
"""

from .deepseek_v31_optimized_provider import DeepSeekV31OptimizedProvider
import os


class DeepSeekProvider(DeepSeekV31OptimizedProvider):
    """Compatibility wrapper for legacy DeepSeekProvider imports"""

    def __init__(self, api_key: str = None):
        # Use environment variable if not provided
        if not api_key:
            api_key = os.getenv("DEEPSEEK_API_KEY")
        super().__init__(api_key)
        self.provider_name = "deepseek"


# Also provide EnhancedDeepSeekProvider for compatibility
class EnhancedDeepSeekProvider(DeepSeekV31OptimizedProvider):
    """Compatibility wrapper for EnhancedDeepSeekProvider imports"""

    def __init__(self, api_key: str = None):
        if not api_key:
            api_key = os.getenv("DEEPSEEK_API_KEY")
        super().__init__(api_key)
        self.provider_name = "deepseek"


# Export the optimized provider as the default
__all__ = [
    "DeepSeekProvider",
    "EnhancedDeepSeekProvider",
    "DeepSeekV31OptimizedProvider",
]
