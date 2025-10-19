from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import Any, Dict, Mapping, Optional, Sequence

from .flow_contracts import (
    CheckpointService,
    ContextOrchestrator,
    PipelineManager,
    RunId,
    Snapshot,
    StageId,
    StageInput,
    StageResult,
    StageSpec,
    StageStatus,
)
from .errors import FatalStageError, PipelineExecutionError, RetryableError
from .execution_plan import topological_sort
from .metrics import MetricsSink
from .telemetry import traced_span


class DefaultPipelineManager(PipelineManager):
    def __init__(
        self,
        *,
        checkpoint_service: CheckpointService,
        context_orchestrator: ContextOrchestrator,
        executor_registry: Mapping[str, Any],
        metrics: Optional[MetricsSink] = None,
        schema_version: int = 1,
    ) -> None:
        self.cp = checkpoint_service
        self.cx = context_orchestrator
        self.registry = executor_registry
        self.metrics = metrics or MetricsSink()
        self.schema_version = schema_version

    async def run(
        self, plan: Sequence[StageSpec], run_id: RunId, resume: bool = True
    ) -> Dict[StageId, StageResult]:
        ordered = topological_sort(plan)
        results: Dict[StageId, StageResult] = {}

        for spec in ordered:
            prior = {sid: results[sid] for sid in spec.requires if sid in results}
            stage_id = spec.id

            # Assemble context
            ctx = await self.cx.assemble(run_id, spec, prior)
            context_version = self.cx.version()
            input_hash = _stable_hash({"inputs": spec.inputs, "context": ctx})

            # Check checkpoint
            if resume:
                snap = await self.cp.load(run_id, stage_id)
                if snap and self._is_fresh(snap, input_hash, context_version):
                    # Reuse prior success
                    results[stage_id] = StageResult(
                        status=StageStatus.SKIPPED, output={}, metrics={}
                    )
                    self.metrics.stage_finish(run_id, stage_id, StageStatus.SKIPPED, 0)
                    continue

            # Execute stage
            self.metrics.stage_start(run_id, stage_id)
            started = time.time()
            resume_token = None

            # Save RUNNING snapshot
            running_snap = await self._new_snapshot(
                run_id=run_id,
                stage_id=stage_id,
                status=StageStatus.RUNNING,
                input_hash=input_hash,
                context_version=context_version,
                result_digest=None,
                resume_token=resume_token,
            )
            await self.cp.save(running_snap)
            self.metrics.checkpoint_event(run_id, stage_id, "pre")

            attempt = 0
            while True:
                attempt += 1
                try:
                    async with _stage_span(spec):
                        executor = self.registry[spec.executor_key]
                        stage_input = StageInput(run_id=run_id, stage=spec, context=ctx)
                        result = await executor.execute(stage_input)

                    # Save success
                    duration_ms = int((time.time() - started) * 1000)
                    digest = _stable_hash(result.output)
                    success_snap = await self._new_snapshot(
                        run_id=run_id,
                        stage_id=stage_id,
                        status=StageStatus.SUCCEEDED,
                        input_hash=input_hash,
                        context_version=context_version,
                        result_digest=digest,
                    )
                    await self.cp.save(success_snap)
                    self.metrics.checkpoint_event(run_id, stage_id, "post")

                    results[stage_id] = result
                    self.metrics.stage_finish(
                        run_id, stage_id, StageStatus.SUCCEEDED, duration_ms
                    )
                    break

                except RetryableError:
                    max_retries = int((spec.retry_policy or {}).get("max_retries", 0))
                    if attempt > max_retries:
                        raise
                    self.metrics.stage_retry(run_id, stage_id, attempt)
                    await asyncio.sleep(min(2 ** (attempt - 1), 8))

                except FatalStageError as e:
                    # Mark failed and abort dependents
                    fail_snap = await self._new_snapshot(
                        run_id=run_id,
                        stage_id=stage_id,
                        status=StageStatus.FAILED,
                        input_hash=input_hash,
                        context_version=context_version,
                        result_digest=None,
                    )
                    await self.cp.save(fail_snap)
                    self.metrics.stage_finish(run_id, stage_id, StageStatus.FAILED, 0)
                    raise PipelineExecutionError(str(e)) from e

        return results

    async def _new_snapshot(
        self,
        *,
        run_id: RunId,
        stage_id: StageId,
        status: StageStatus,
        input_hash: str,
        context_version: str,
        result_digest: Optional[str],
        resume_token: Optional[str] = None,
    ) -> Snapshot:
        # Prefer cp-provided factory if available
        if hasattr(self.cp, "new_snapshot"):
            return self.cp.new_snapshot(  # type: ignore[attr-defined]
                run_id=run_id,
                stage_id=stage_id,
                status=status,
                input_hash=input_hash,
                context_version=context_version,
                result_digest=result_digest,
                resume_token=resume_token,
                payload={},
            )
        from datetime import datetime, timezone

        return Snapshot(
            schema_version=self.schema_version,
            timestamp=datetime.now(timezone.utc),
            run_id=run_id,
            stage_id=stage_id,
            status=status,
            input_hash=input_hash,
            context_version=context_version,
            result_digest=result_digest,
            resume_token=resume_token,
            payload={},
        )

    def _is_fresh(self, snap: Snapshot, input_hash: str, context_version: str) -> bool:
        return (
            snap.status == StageStatus.SUCCEEDED
            and snap.input_hash == input_hash
            and snap.context_version == context_version
        )


def _stable_hash(obj: object) -> str:
    try:
        data = json.dumps(obj, sort_keys=True, default=str).encode()
    except Exception:
        data = str(obj).encode()
    return hashlib.sha256(data).hexdigest()[:16]


class _stage_span:
    def __init__(self, spec: StageSpec) -> None:
        self.spec = spec
        self._cm = traced_span(f"stage:{spec.id.value}")

    async def __aenter__(self):
        self._ctx = self._cm.__enter__()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self._cm.__exit__(exc_type, exc, tb)
        return False
