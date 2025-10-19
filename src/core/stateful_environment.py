"""Compatibility facade for the stateful environment utilities."""

from __future__ import annotations

from src.engine.core.stateful_environment import (  # noqa: F401
    StatefulEnvironment,
    CheckpointType,
    StateSnapshot,
    RecoveryPlan,
    ExecutionState,
)


def get_stateful_environment(engagement_id: str, config=None) -> StatefulEnvironment:
    """Legacy factory retained for backward compatibility."""
    return StatefulEnvironment(engagement_id=engagement_id, config=config)


__all__ = [
    "StatefulEnvironment",
    "CheckpointType",
    "StateSnapshot",
    "RecoveryPlan",
    "ExecutionState",
    "get_stateful_environment",
]
