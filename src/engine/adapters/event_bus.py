"""Event bus adapter - bridges src.core event bus systems to src.engine"""

from src.core.enhanced_event_bus import (
    MetisEventBus,
    CloudEvent,
    create_metis_cloud_event,
)

# Also bridge the simpler event_bus
from src.core.event_bus import get_event_bus as _get_core_event_bus

def get_event_bus():
    """Get event bus - bridges both enhanced and core event bus"""
    return _get_core_event_bus()

__all__ = ["MetisEventBus", "CloudEvent", "get_event_bus", "create_metis_cloud_event"]
