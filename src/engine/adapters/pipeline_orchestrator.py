from __future__ import annotations

from typing import Any

from src.core.stateful_pipeline_orchestrator import StatefulPipelineOrchestrator
from src.interfaces import PipelineOrchestrator


def create_pipeline_orchestrator(**kwargs: Any) -> PipelineOrchestrator:
    """Factory producing a pipeline orchestrator that satisfies the Protocol."""
    return StatefulPipelineOrchestrator(**kwargs)
