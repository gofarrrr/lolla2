"""
DEPRECATED: Use src.integrations.odr package instead
This file maintained for backward compatibility
"""

# Re-export everything from the refactored modules
from .odr.config import ODRConfiguration
from .odr.agent import ODRResearchAgent
from .odr.autonomous import AutonomousODRClient
from .odr.client import OpenDeepResearchClient
from .odr.context import ContextGapDetector
from .odr.demo import get_autonomous_odr_client, generate_demo_search_results
from .odr import ODR_AVAILABLE

# Re-export for backward compatibility
__all__ = [
    "ODRConfiguration",
    "ODRResearchAgent",
    "AutonomousODRClient",
    "OpenDeepResearchClient",
    "ContextGapDetector",
    "get_autonomous_odr_client",
    "generate_demo_search_results",
    "ODR_AVAILABLE",
]
