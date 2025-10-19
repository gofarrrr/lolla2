"""
Refactored Parallel Analysis Stage Executor

Simplified executor achieving CC ≤12 by delegating to ParallelAnalysisOrchestrator.
Maintains full backward compatibility with existing API.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from src.core.pipeline_contracts import IStageExecutor, PipelineState
from src.core.stage_executors.parallel_analysis import (
    StandardPromptBuilder,
    ParallelRunner,
    StandardAggregator,
    StandardDepthPack,
)
from src.core.parallel_analysis_orchestrator import ParallelAnalysisOrchestrator

logger = logging.getLogger(__name__)


class ParallelAnalysisStageExecutor(IStageExecutor):
    """Refactored parallel analysis executor with modular orchestration."""

    def __init__(
        self,
        llm_client_getter: Optional[Callable[[], Any]] = None,
        persona_loader: Optional[Any] = None,
        context_stream: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
        prompt_builder: Optional[StandardPromptBuilder] = None,
        runner: Optional[ParallelRunner] = None,
        aggregator: Optional[StandardAggregator] = None,
        depth_pack: Optional[StandardDepthPack] = None,
    ) -> None:
        """Initialize executor with dependency injection."""
        self._llm_client_getter = llm_client_getter
        self._context_stream = context_stream
        self._context = context or {}

        # Acquire LLM client
        llm_client = None
        if llm_client_getter:
            try:
                llm_client = llm_client_getter()
            except Exception as e:
                logger.warning(f"Failed to acquire LLM client: {e}")

        # Initialize modular components
        self.prompt_builder = prompt_builder or StandardPromptBuilder()
        self.runner = runner or (
            ParallelRunner(llm_client=llm_client) if llm_client else None
        )
        self.aggregator = aggregator or StandardAggregator()
        self.depth_pack = depth_pack or StandardDepthPack()

        # Initialize orchestrator
        self.orchestrator = ParallelAnalysisOrchestrator(
            prompt_builder=self.prompt_builder,
            runner=self.runner,
            aggregator=self.aggregator,
            depth_pack=self.depth_pack,
            context_stream=context_stream,
            context=self._context,
        )

        logger.info(
            "✨ Initialized parallel analysis executor (Pilot B) with modular orchestrator"
        )

    def set_context(self, context: Dict[str, Any]) -> None:
        """Set context (backward compatibility)."""
        self._context = context
        self.orchestrator.context = context

    async def execute(self, state: PipelineState) -> PipelineState:
        """
        Execute parallel analysis stage using orchestrator.

        Simple delegation pattern achieving CC ≤12.
        """
        return await self.orchestrator.execute_parallel_analysis(state)
