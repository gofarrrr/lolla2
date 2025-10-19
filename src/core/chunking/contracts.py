# src/core/chunking/contracts.py
from __future__ import annotations

from typing import Protocol, Dict, Any, Optional
from enum import Enum

from src.core.strategic_query_decomposer import MECEDecomposition
from src.core.chunking_quality_monitor import ProcessingContext, QualityAssessment


class CurrencyRequirement(str, Enum):
    """Signals how time-sensitive a research ticket is."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class IChunkingStrategy(Protocol):
    """Produces a strategic decomposition for a query.

    This is an architectural seam. Implementations can be legacy-backed, hybrid,
    or fully new algorithms. The facade will initially delegate to the legacy
    chunker to preserve behavior during the migration.
    """

    async def decompose(
        self, query: str, user_context: Optional[Dict[str, Any]] = None
    ) -> MECEDecomposition:
        ...


class IChunkingEvaluator(Protocol):
    """Evaluates decomposition quality and produces recommendations."""

    async def assess(
        self, decomposition: MECEDecomposition, context: Optional[ProcessingContext] = None
    ) -> QualityAssessment:
        ...


class IChunkingFinalizer(Protocol):
    """Finalizes chunking outputs (integration data, confidence, recommendations).

    Uses broad typing to avoid import cycles with the facade and result types.
    """

    def finalize(self, result: Any) -> Any:
        """Mutate and/or return the finalized result (integration_data, confidence, recommendations)."""
        ...
