"""
Event Bus Interfaces for Dependency Injection
"""

from typing import Any, Callable, Protocol
from src.engine.models.data_contracts import MetisDataContract


class IEventBus(Protocol):
    """Interface for event bus operations"""

    async def publish(self, event: Any) -> None:
        """Publish an event to the bus"""
        ...

    async def subscribe(self, event_type: str, handler: Callable) -> str:
        """Subscribe to an event type"""
        ...

    async def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe from events"""
        ...

    async def publish_cognitive_event(self, contract: MetisDataContract) -> None:
        """Publish cognitive processing event"""
        ...
