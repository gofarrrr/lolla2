"""
Database Interfaces for Dependency Injection
"""

from typing import Dict, List, Any, Optional, Protocol


class INwayManager(Protocol):
    """Interface for N-way database manager"""

    async def initialize(self) -> bool:
        """Initialize the N-way manager"""
        ...

    async def find_optimal_interaction(
        self, problem_keywords: List[str], domain: str, complexity: str = "High"
    ) -> Optional[Dict[str, Any]]:
        """Find the best N-way interaction for a given problem"""
        ...

    async def get_interaction_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        ...


class IDistributedStateManager(Protocol):
    """Interface for distributed state management"""

    async def get_state(self, key: str) -> Any:
        """Get state value by key"""
        ...

    async def set_state(self, key: str, value: Any) -> None:
        """Set state value"""
        ...

    async def update_state(self, updates: Dict[str, Any]) -> None:
        """Batch update state"""
        ...

    async def get_snapshot(self) -> Dict[str, Any]:
        """Get complete state snapshot"""
        ...
