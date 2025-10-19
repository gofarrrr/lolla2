"""
METIS v7.0 - Cognitive Intelligence Platform
Enterprise-grade AI-native consulting operating system
"""

__version__ = "7.0.0"
__author__ = "METIS Development Team"
__email__ = "dev@metis.ai"
__description__ = "AI-native cognitive intelligence platform for enterprise consulting"

# Core exports
from ..core.enhanced_event_bus import (
    get_event_bus,
    CloudEvent,
    EventCategory,
    EventPriority,
)
from .models.data_contracts import MetisDataContract
from .api.enterprise_gateway import (
    get_api_gateway,
    create_api_key,
    add_custom_rate_limit,
)

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__description__",
    "get_event_bus",
    "CloudEvent",
    "EventCategory",
    "EventPriority",
    "MetisDataContract",
    "get_api_gateway",
    "create_api_key",
    "add_custom_rate_limit",
]
