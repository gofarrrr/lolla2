"""
Socratic Stage Executor

Simplified executor achieving CC ≤12 by delegating to SocraticOrchestrator.
Maintains full backward compatibility with existing API.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from src.core.pipeline_contracts import IStageExecutor, PipelineState
from src.core.socratic_orchestrator import SocraticOrchestrator

logger = logging.getLogger(__name__)


class SocraticStageExecutor(IStageExecutor):
    """
    Refactored Socratic stage executor with modular orchestration.

    Delegates all complexity to SocraticOrchestrator, achieving CC ≤12.
    """

    def __init__(
        self,
        progressive_question_engine: Any,
        context_stream: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize executor with dependency injection."""
        self._engine = progressive_question_engine
        self._context_stream = context_stream
        self._context = context or {}

        # Initialize orchestrator
        self.orchestrator = SocraticOrchestrator(
            progressive_question_engine=self._engine,
            context_stream=self._context_stream,
            context=self._context,
        )

        logger.info("✨ Initialized Socratic stage executor (Pilot B) with modular orchestrator")

    def set_context(self, context: Dict[str, Any]) -> None:
        """Set context (backward compatibility)."""
        self._context = context
        self.orchestrator.context = context

    async def execute(self, state: PipelineState) -> PipelineState:
        """
        Execute Socratic questions stage using orchestrator.

        Simple delegation pattern achieving CC ≤12.
        """
        return await self.orchestrator.execute_socratic_stage(state)
