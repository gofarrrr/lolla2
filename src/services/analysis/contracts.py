# src/services/analysis/contracts.py
from typing import Protocol, Dict, Any, List


class IPromptBuilder(Protocol):
    """Defines the contract for building a prompt for a consultant."""

    def build(self, consultant: Dict[str, Any], dispatch_info: Dict[str, Any]) -> str:
        """Build a complete prompt from raw inputs (consultant and dispatch_info)."""
        ...

    def build_context_section(
        self, consultant: Dict[str, Any], dispatch_info: Dict[str, Any]
    ) -> str:
        """Build only the context section (used for ancillary needs like concept derivation)."""
        ...


class IConsultantRunner(Protocol):
    """Defines the contract for executing an analysis with a single consultant."""

    async def run(
        self, consultant: Dict[str, Any], prompt: str, context: Dict[str, Any]
    ) -> Any: ...


class IResultAggregator(Protocol):
    """Defines the contract for aggregating results from multiple consultants."""

    def aggregate(self, results: List[Dict[str, Any]]) -> Dict[str, Any]: ...


class IEvidenceEmitter(Protocol):
    """Defines the contract for emitting structured events to the context stream."""

    def emit(self, event_type: str, data: Dict[str, Any]) -> None: ...
