# src/core/persistence/contracts.py
from typing import Protocol, List, Dict, Any


class IEventPersistence(Protocol):
    """Defines the contract for persisting context events/records."""

    async def persist(self, batch: List[Dict[str, Any]]) -> None:
        """Persist a batch of event/record dictionaries."""
        ...
