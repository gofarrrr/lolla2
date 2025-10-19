from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

from .flow_contracts import CheckpointService, RunId, Snapshot, StageId, StageStatus


class InMemoryCheckpointService(CheckpointService):
    """Simple in-memory snapshot store for development and tests."""

    def __init__(self, schema_version: int = 1) -> None:
        self._store: Dict[Tuple[str, str], Snapshot] = {}
        self._schema_version = schema_version

    async def load(self, run_id: RunId, stage_id: StageId) -> Optional[Snapshot]:
        return self._store.get((run_id.value, stage_id.value))

    async def save(self, snap: Snapshot) -> None:
        self._store[(snap.run_id.value, snap.stage_id.value)] = snap

    async def mark_final(self, snap: Snapshot) -> None:
        # no-op for in-memory store
        pass

    async def purge_run(self, run_id: RunId) -> None:
        keys = [k for k in self._store.keys() if k[0] == run_id.value]
        for k in keys:
            self._store.pop(k, None)

    @property
    def schema_version(self) -> int:
        return self._schema_version

    def new_snapshot(
        self,
        *,
        run_id: RunId,
        stage_id: StageId,
        status: StageStatus,
        input_hash: str,
        context_version: str,
        result_digest: str | None,
        resume_token: str | None = None,
        payload: dict | None = None,
    ) -> Snapshot:
        return Snapshot(
            schema_version=self._schema_version,
            timestamp=datetime.now(timezone.utc),
            run_id=run_id,
            stage_id=stage_id,
            status=status,
            input_hash=input_hash,
            context_version=context_version,
            result_digest=result_digest,
            resume_token=resume_token,
            payload=payload or {},
        )
