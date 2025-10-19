"""
Model Selector - Simple Implementation
Provides ModelSelector class for Senior Advisor API compatibility
"""


class ModelSelector:
    """Simple ModelSelector implementation for compatibility"""

    def __init__(self):
        self.name = "ModelSelector"
        self.initialized = True

    def select_model(self, context=None, **kwargs):
        """Simple model selection - returns default model"""
        return {
            "model_name": "default",
            "confidence": 0.8,
            "reasoning": "Default selection for compatibility",
        }

    def get_available_models(self):
        """Return list of available models"""
        return ["default", "cognitive", "analytical"]


# Also create direct export
__all__ = ["ModelSelector"]
