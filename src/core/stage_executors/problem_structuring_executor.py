# src/core/stage_executors/problem_structuring_executor.py
"""
Problem Structuring Stage Executor

Simplified executor achieving CC ≤12 by delegating to ProblemStructuringOrchestrator.
Maintains full backward compatibility with existing API.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from src.core.pipeline_contracts import IStageExecutor, PipelineState
from src.core.problem_structuring_orchestrator import ProblemStructuringOrchestrator

logger = logging.getLogger(__name__)


class ProblemStructuringStageExecutor(IStageExecutor):
    """Refactored problem structuring executor with modular orchestration."""

    def __init__(
        self,
        problem_structuring_agent: Any,
        context_stream: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize executor with dependency injection."""
        self._agent = problem_structuring_agent
        self._context_stream = context_stream
        self._context = context or {}

        # Initialize orchestrator
        self.orchestrator = ProblemStructuringOrchestrator(
            problem_structuring_agent=self._agent,
            context_stream=self._context_stream,
            context=self._context,
        )

        logger.info("✨ Initialized problem structuring executor with modular orchestrator")

    def set_context(self, context: Dict[str, Any]) -> None:
        """Set context (backward compatibility)."""
        self._context = context
        self.orchestrator.context = context

    async def execute(self, state: PipelineState) -> PipelineState:
        """
        Execute problem structuring stage using orchestrator.

        Simple delegation pattern achieving CC ≤12.
        """
        return await self.orchestrator.execute_problem_structuring(state)
