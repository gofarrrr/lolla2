from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class PipelineOrchestrator(Protocol):
    """Contract for orchestrators that drive the cognitive pipeline."""

    async def run_pipeline(self, payload: Any) -> Any:
        ...

    async def resume_from_checkpoint(self, checkpoint: Any) -> Any:
        ...
