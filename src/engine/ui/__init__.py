# UI package

# Progressive Transparency Engine Components
from .scaffolding_engine import CognitiveScaffoldingEngine
from .expertise_assessor import UserExpertiseAssessor
from .validation_evidence import ValidationEvidenceEngine
from .reasoning_visualizer import ReasoningVisualizationEngine

# Temporarily disabled due to circular import issue
# from .adaptive_transparency import (
#     AdaptiveTransparencyEngine,
#     get_transparency_engine,
#     generate_user_transparency,
#     update_transparency_preferences
# )

# Backward compatibility shim
from .transparency_components import *

__all__ = [
    "CognitiveScaffoldingEngine",
    "UserExpertiseAssessor",
    "ValidationEvidenceEngine",
    "ReasoningVisualizationEngine",
    # Temporarily disabled due to circular import issue
    # 'AdaptiveTransparencyEngine',
    # 'get_transparency_engine',
    # 'generate_user_transparency',
    # 'update_transparency_preferences'
]
