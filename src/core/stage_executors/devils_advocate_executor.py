"""
Devils Advocate Stage Executor - Pilot B (Orchestrator Extension Pattern)
==========================================================================

EXPERIMENT: P0 #3 Phase 3 - Devils Advocate Refactoring (Pilot B Implementation)

Pattern: Orchestrator Extension
- Orchestrator accepts PipelineState directly
- Executor becomes trivial (single delegation)
- Conversion logic lives in orchestrator

Expected Metrics:
- Executor CC: 1-2 (from 74 in original)
- Total LOC: ~50 lines (89% reduction!)

Evaluation Criteria:
1. Complexity: CC <10 per method ✅ (CC = 1-2)
2. LOC Reduction: 40%+ ✅ (89%)
3. Integration tests: 100% pass
4. No >5% performance regression

Trade-offs:
- Pros: Minimal executor (trivial), no new files
- Cons: Orchestrator coupled to PipelineState
"""

import logging
from typing import Any, Dict, Optional

from src.core.pipeline_contracts import IStageExecutor, PipelineState

logger = logging.getLogger(__name__)


class DevilsAdvocateStageExecutor(IStageExecutor):
    """
    Pilot B: Orchestrator Extension Pattern

    This executor delegates everything to the orchestrator's
    PipelineState-native method.

    Complexity Target: CC = 1-2 (trivial)
    """

    def __init__(
        self,
        devils_advocate_system: Any,  # EnhancedDevilsAdvocateSystemPilotB
        context_stream: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._da = devils_advocate_system
        self._context_stream = context_stream
        self._context: Dict[str, Any] = context or {}

    def set_context(self, context: Dict[str, Any]) -> None:
        self._context = context

    async def execute(self, state: PipelineState) -> PipelineState:
        """
        Execute Devils Advocate stage via orchestrator delegation.

        Pilot B Pattern:
        1. Single delegation to orchestrator's PipelineState-native method
        2. Orchestrator handles ALL conversion and processing
        3. Return updated PipelineState

        Complexity: CC = 1 (trivial)
        """
        return await self._da.run_critique_from_state(
            state=state,
            context_stream=self._context_stream,
        )
