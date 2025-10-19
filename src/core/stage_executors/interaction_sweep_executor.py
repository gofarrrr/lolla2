# src/core/stage_executors/interaction_sweep_executor.py
"""
Interaction Sweep Stage Executor

Simplified executor achieving CC ≤12 by delegating to InteractionSweepOrchestrator.
Maintains full backward compatibility with existing API.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from src.core.pipeline_contracts import IStageExecutor, PipelineState
from src.core.interaction_sweep_orchestrator import InteractionSweepOrchestrator

logger = logging.getLogger(__name__)


class InteractionSweepStageExecutor(IStageExecutor):
    """Refactored interaction sweep executor with modular orchestration."""

    def __init__(self, context_stream: Optional[Any] = None) -> None:
        """Initialize executor with dependency injection."""
        self._context_stream = context_stream

        # Initialize orchestrator
        self.orchestrator = InteractionSweepOrchestrator(
            context_stream=self._context_stream
        )

        logger.info("✨ Initialized interaction sweep executor with modular orchestrator")

    async def execute(self, state: PipelineState) -> PipelineState:
        """
        Execute interaction sweep stage using orchestrator.

        Simple delegation pattern achieving CC ≤12.
        """
        return await self.orchestrator.execute_interaction_sweep(state)
