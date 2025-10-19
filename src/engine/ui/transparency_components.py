"""
METIS Transparency Components
Compatibility shim for backward compatibility

This module provides backward compatibility by importing all transparency
engine components. Existing imports will continue to work while new code
can import components directly from their dedicated modules.
"""

# Import all transparency engine components
from .scaffolding_engine import CognitiveScaffoldingEngine
from .expertise_assessor import UserExpertiseAssessor
from .validation_evidence import ValidationEvidenceEngine
from .reasoning_visualizer import ReasoningVisualizationEngine
from .adaptive_transparency import (
    AdaptiveTransparencyEngine,
    get_transparency_engine,
    generate_user_transparency,
    update_transparency_preferences,
)

# Re-export all components for backward compatibility
__all__ = [
    "CognitiveScaffoldingEngine",
    "UserExpertiseAssessor",
    "ValidationEvidenceEngine",
    "ReasoningVisualizationEngine",
    "AdaptiveTransparencyEngine",
    "get_transparency_engine",
    "generate_user_transparency",
    "update_transparency_preferences",
]
