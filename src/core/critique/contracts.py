# src/core/critique/contracts.py
from __future__ import annotations

from typing import Protocol, Dict, Any, List


class ICritiquePreparer(Protocol):
    """Prepares inputs and context for a devils-advocate critique run.

    This protocol intentionally uses broad types to provide a stable seam.
    Downstream PRs will introduce richer contracts once extraction proceeds.
    """

    async def prepare(
        self, analysis_results: List[Any], context_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build a prepared payload for critique execution."""
        ...


class ICritiqueRunner(Protocol):
    """Executes the core critique logic (engine orchestration)."""

    async def run(self, prepared_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Run the critique and return raw results."""
        ...


class ICritiqueSynthesizer(Protocol):
    """Synthesizes raw critique results into a refined, consumer-ready output."""

    def synthesize(self, raw_results: Dict[str, Any]) -> Dict[str, Any]:
        """Produce the final synthesized critique payload."""
        ...
