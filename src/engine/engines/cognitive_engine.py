"""
Compatibility shim for legacy import path: src.engine.engines.cognitive_engine

Provides get_cognitive_engine and ModelSelectionCriteria by delegating to the
current modular engine factory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import logging

# Re-export factory symbols for convenience/backward-compatibility
from src.factories.engine_factory import (
    CognitiveEngineFactory,
    ModularCognitiveEngine,
    ICognitiveEngine,
    create_cognitive_engine as _create_cognitive_engine,
)

_logger = logging.getLogger(__name__)
_engine_instance: Optional[ICognitiveEngine] = None


@dataclass
class ModelSelectionCriteria:
    """Criteria for selecting appropriate mental models (compat layer).

    Note: Several interfaces import this from the legacy path. Keep fields
    superset-compatible with prior usages found in the codebase.
    """

    problem_type: str = ""
    complexity_level: str = "medium"
    accuracy_requirement: float = 0.8
    business_context: Dict[str, Any] = field(default_factory=dict)
    time_constraint: Optional[int] = None


def get_cognitive_engine_sync(
    config: Optional[Dict[str, Any]] = None,
) -> ICognitiveEngine:
    """Synchronous accessor for the modular cognitive engine instance.

    Creates the engine once and returns a cached instance for subsequent calls.
    """
    global _engine_instance
    if _engine_instance is None:
        _logger.info(
            "Initializing ModularCognitiveEngine via CognitiveEngineFactory (sync)"
        )
        _engine_instance = CognitiveEngineFactory.create_engine(config)
    return _engine_instance


async def get_cognitive_engine(
    config: Optional[Dict[str, Any]] = None,
) -> ICognitiveEngine:
    """Async accessor for the modular cognitive engine instance.

    Returns the same cached engine as the sync accessor.
    """
    return get_cognitive_engine_sync(config)


# Backwards-compatibility alias
create_cognitive_engine = _create_cognitive_engine

__all__ = [
    "ModelSelectionCriteria",
    "get_cognitive_engine",
    "get_cognitive_engine_sync",
    "CognitiveEngineFactory",
    "ModularCognitiveEngine",
    "ICognitiveEngine",
    "create_cognitive_engine",
]
