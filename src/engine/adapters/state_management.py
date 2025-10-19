"""State management adapter - bridges src.core.state_management to src.engine"""

from src.core.state_management import DistributedStateManager, StateType

__all__ = ["DistributedStateManager", "StateType"]
