# src/core/stage_executors/consultant_selection_executor.py
"""
Consultant Selection Stage Executor

Simplified executor achieving CC ≤12 by delegating to ConsultantSelectionOrchestrator.
Maintains full backward compatibility with existing API.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from src.core.pipeline_contracts import IStageExecutor, PipelineState
from src.core.consultant_selection_orchestrator import ConsultantSelectionOrchestrator

logger = logging.getLogger(__name__)


class ConsultantSelectionStageExecutor(IStageExecutor):
    """Refactored consultant selection executor with modular orchestration."""

    def __init__(
        self,
        dispatch_orchestrator: Any,
        context_stream: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize executor with dependency injection."""
        self._dispatch = dispatch_orchestrator
        self._context_stream = context_stream
        self._context = context or {}

        # Initialize orchestrator
        self.orchestrator = ConsultantSelectionOrchestrator(
            dispatch_orchestrator=self._dispatch,
            context_stream=self._context_stream,
            context=self._context,
        )

        logger.info("✨ Initialized consultant selection executor with modular orchestrator")

    def set_context(self, context: Dict[str, Any]) -> None:
        """Set context (backward compatibility)."""
        self._context = context
        self.orchestrator.context = context

    async def execute(self, state: PipelineState) -> PipelineState:
        """
        Execute consultant selection stage using orchestrator.

        Simple delegation pattern achieving CC ≤12.
        """
        return await self.orchestrator.execute_consultant_selection(state)
