"""
Simplified state management services used in tests and local benchmarks.

The production implementation persists engagement state in Supabase.  For
lightweight testing we expose a compatible interface that keeps everything
in-process.
"""

from enum import Enum
from typing import Any, Dict, Optional


class EngagementPhase(str, Enum):
    DISCOVERY = "discovery"
    STRUCTURING = "structuring"
    EXECUTION = "execution"
    SYNTHESIS = "synthesis"


class StateManager:
    """Simple in-memory state bag used by the facade."""

    def __init__(self):
        self._store: Dict[str, Any] = {}

    async def set_state(self, key: str, value: Any, *, namespace: Optional[str] = None) -> None:
        composite = f"{namespace}:{key}" if namespace else key
        self._store[composite] = value

    async def get_state(self, key: str, *, namespace: Optional[str] = None) -> Any:
        composite = f"{namespace}:{key}" if namespace else key
        return self._store.get(composite)

    async def close(self) -> None:
        self._store.clear()


class StateManagementService(StateManager):
    """Compatibility shim mirroring the production service interface."""

    async def detect_engagement_schema_version(self, engagement_id: str) -> str:
        return "v2.1"

    async def mark_engagement_failed(self, engagement_id: str, reason: str) -> None:
        await self.set_state(f"{engagement_id}:status", {"status": "failed", "reason": reason})

    async def load_recovery_state(self, engagement_id: str) -> Optional[Dict[str, Any]]:
        return await self.get_state(f"{engagement_id}:recovery")

    def can_recover_from_checkpoint(self, recovery_state: Dict[str, Any]) -> bool:
        return bool(recovery_state)

    async def initialize_v21_engagement_state(self, engagement_id: str) -> None:
        await self.set_state(f"{engagement_id}:schema", "v2.1")

    async def start_phase(self, engagement_id: str, phase: EngagementPhase) -> None:
        await self.set_state(f"{engagement_id}:phase", phase.value)

    async def create_checkpoint(self, engagement_id: str, stage: str, payload: Dict[str, Any]) -> None:
        await self.set_state(f"{engagement_id}:checkpoint:{stage}", payload)

    async def mark_engagement_completed(self, engagement_id: str) -> None:
        await self.set_state(f"{engagement_id}:status", {"status": "completed"})


def get_state_management_service(*_, **__) -> StateManagementService:
    return StateManagementService()


__all__ = [
    "StateManager",
    "StateManagementService",
    "get_state_management_service",
    "EngagementPhase",
]
