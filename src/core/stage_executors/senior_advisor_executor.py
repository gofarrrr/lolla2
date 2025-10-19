"""
Senior Advisor Stage Executor - Pilot B (Orchestrator Extension Pattern)
=========================================================================

EXPERIMENT: P0 #3 Dual Pilot - Pilot B Implementation

Pattern: Orchestrator Extension
- Orchestrator accepts PipelineState directly
- Executor becomes trivial (single delegation)
- Conversion logic lives in orchestrator

Expected Metrics:
- Executor CC: 1 (from 82 in original)
- Total LOC: ~30 lines (96% reduction!)

Evaluation Criteria:
1. Complexity: CC <10 per method ✅ (CC = 1)
2. LOC Reduction: 96% (executor only) ✅
3. Golden Fixture: 100% match
4. Test Coverage: ≥80%
5. Performance: No >5% regression

Trade-offs:
- Pros: Minimal executor (trivial), no new files
- Cons: Orchestrator coupled to PipelineState
"""

import logging
from typing import Any, Dict, Optional

from src.core.pipeline_contracts import IStageExecutor, PipelineState

logger = logging.getLogger(__name__)


class SeniorAdvisorStageExecutor(IStageExecutor):
    """
    Pilot B: Orchestrator Extension Pattern

    This executor delegates everything to the orchestrator's
    PipelineState-native method.

    Complexity Target: CC = 1 (trivial)
    """

    def __init__(
        self,
        senior_advisor: Any,  # SeniorAdvisorOrchestratorPilotB
        context_stream: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._senior_advisor = senior_advisor
        self._context_stream = context_stream
        self._context: Dict[str, Any] = context or {}

    def set_context(self, context: Dict[str, Any]) -> None:
        self._context = context

    async def execute(self, state: PipelineState) -> PipelineState:
        """
        Execute Senior Advisor stage via orchestrator delegation.

        Pilot B Pattern:
        1. Single delegation to orchestrator's PipelineState-native method
        2. Orchestrator handles ALL conversion and processing
        3. Return updated PipelineState

        Complexity: CC = 1 (trivial)
        """
        return await self._senior_advisor.conduct_two_brain_analysis_from_state(
            state=state,
            engagement_id=state.trace_id or "unknown",
            context_stream=self._context_stream,
        )
