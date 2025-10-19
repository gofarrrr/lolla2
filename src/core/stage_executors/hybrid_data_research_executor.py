"""
Hybrid Data Research Stage Executor

Simplified executor achieving CC ≤12 by delegating to HybridDataResearchOrchestrator.
Maintains full backward compatibility with existing API.
"""

import logging
from typing import Any, Dict, Optional

from src.core.pipeline_contracts import IStageExecutor, PipelineState
from src.core.hybrid_data_research_orchestrator import HybridDataResearchOrchestrator
from src.clients.oracle_client import OracleClient

logger = logging.getLogger(__name__)


class HybridDataResearchExecutor(IStageExecutor):
    """Refactored hybrid data research executor with modular orchestration."""

    def __init__(
        self,
        context_stream: Optional[Any] = None,
        oracle_client: Optional[OracleClient] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize executor with dependency injection."""
        self._context_stream = context_stream
        self._oracle_client = oracle_client or OracleClient()
        self._oracle_client_provided = oracle_client is not None
        self._context = context or {}

        # Initialize orchestrator
        self.orchestrator = HybridDataResearchOrchestrator(
            context_stream=self._context_stream,
            oracle_client=self._oracle_client,
            oracle_client_provided=self._oracle_client_provided,
            context=self._context,
        )

        logger.info("✨ Initialized hybrid data research executor with modular orchestrator")

    def set_context(self, context: Dict[str, Any]) -> None:
        """Set context (backward compatibility)."""
        self._context = context
        self.orchestrator.context = context

    async def execute(self, state: PipelineState) -> PipelineState:
        """
        Execute hybrid data research stage using orchestrator.

        Simple delegation pattern achieving CC ≤12.
        """
        return await self.orchestrator.execute_hybrid_research(state)
