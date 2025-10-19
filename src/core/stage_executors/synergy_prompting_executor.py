# src/core/stage_executors/synergy_prompting_executor.py
from __future__ import annotations

from typing import Any, Dict, Optional
import logging

from src.core.pipeline_contracts import IStageExecutor, PipelineState

logger = logging.getLogger(__name__)


class SynergyPromptingStageExecutor(IStageExecutor):
    """Executes the Synergy Prompting stage.

    Currently minimal logic, but isolated for architectural consistency and
    future extensibility. Provides get_legacy_result() to maintain
    orchestrator compatibility during the transition.
    """

    def __init__(self, context: Optional[Dict[str, Any]] = None) -> None:
        self._context: Dict[str, Any] = context or {}
        self._legacy_result: Optional[Dict[str, Any]] = None

    def set_context(self, context: Dict[str, Any]) -> None:
        self._context = context

    async def execute(self, state: PipelineState) -> PipelineState:
        # Read consultants from PipelineState contracts instead of legacy context
        consultants = []
        if state.selection_results and state.selection_results.selected_consultants:
            consultants = state.selection_results.selected_consultants

        # Legacy context writes removed - minimal functionality preserved
        # This stage currently has no defined contract and minimal logic
        # Future enhancement: implement synergy prompting with proper contracts

        return state
