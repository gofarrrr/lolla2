from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence


class StageStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    RETRIED = "RETRIED"
    ABORTED = "ABORTED"


@dataclass(frozen=True)
class RunId:
    value: str


@dataclass(frozen=True)
class StageId:
    value: str


@dataclass(frozen=True)
class StageSpec:
    id: StageId
    name: str
    executor_key: str
    inputs: Mapping[str, Any]
    requires: Sequence[StageId]
    retry_policy: Optional[Dict[str, Any]] = None  # e.g. {max_retries, backoff_ms, jitter_ms, retry_on}


@dataclass(frozen=True)
class StageInput:
    run_id: RunId
    stage: StageSpec
    context: Mapping[str, Any]


@dataclass
class StageResult:
    status: StageStatus
    output: Mapping[str, Any]
    metrics: Mapping[str, float]
    error: Optional[str] = None


@dataclass
class Snapshot:
    schema_version: int
    timestamp: datetime
    run_id: RunId
    stage_id: StageId
    status: StageStatus
    input_hash: str
    context_version: str
    result_digest: Optional[str]
    resume_token: Optional[str]
    payload: Mapping[str, Any]


class StageExecutor(Protocol):
    async def execute(self, sinput: StageInput) -> StageResult: ...

    @property
    def idempotent(self) -> bool: ...


class CheckpointService(Protocol):
    async def load(self, run_id: RunId, stage_id: StageId) -> Optional[Snapshot]: ...

    async def save(self, snap: Snapshot) -> None: ...

    async def mark_final(self, snap: Snapshot) -> None: ...

    async def purge_run(self, run_id: RunId) -> None: ...


class ContextOrchestrator(Protocol):
    async def assemble(
        self,
        run_id: RunId,
        stage: StageSpec,
        prior: Mapping[StageId, StageResult],
    ) -> Mapping[str, Any]: ...

    def version(self) -> str: ...


class PipelineManager(Protocol):
    async def run(
        self, plan: Sequence[StageSpec], run_id: RunId, resume: bool = True
    ) -> Dict[StageId, StageResult]: ...
