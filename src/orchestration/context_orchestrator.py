from __future__ import annotations

from typing import Any, Dict, Mapping

from .flow_contracts import ContextOrchestrator, RunId, StageId, StageResult, StageSpec


class DefaultContextOrchestrator(ContextOrchestrator):
    """Assembles per-stage context by merging prior stage outputs with static inputs.

    This implementation is intentionally conservative and deterministic.
    """

    def __init__(self, static_base: Mapping[str, Any] | None = None) -> None:
        self._base = dict(static_base or {})
        self._version = "v1"

    async def assemble(
        self,
        run_id: RunId,
        stage: StageSpec,
        prior: Mapping[StageId, StageResult],
    ) -> Mapping[str, Any]:
        ctx: Dict[str, Any] = dict(self._base)
        # Merge prior stage outputs in topological order (by key string)
        for sid in sorted(prior.keys(), key=lambda s: s.value):
            out = prior[sid].output or {}
            for k, v in out.items():
                ctx[k] = v
        # Merge stage declarative inputs last
        for k, v in (stage.inputs or {}).items():
            ctx.setdefault(k, v)
        return ctx

    def version(self) -> str:
        return self._version
