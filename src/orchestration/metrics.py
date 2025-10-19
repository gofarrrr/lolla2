from __future__ import annotations

from typing import Optional

from .flow_contracts import RunId, StageId, StageStatus


class MetricsSink:
    """No-op metrics sink; replace with real metrics/telemetry as needed."""

    def stage_start(self, run: RunId, stage: StageId) -> None:
        pass

    def stage_finish(self, run: RunId, stage: StageId, status: StageStatus, duration_ms: int) -> None:
        pass

    def stage_retry(self, run: RunId, stage: StageId, attempt: int) -> None:
        pass

    def checkpoint_event(self, run: RunId, stage: StageId, op: str) -> None:
        pass
