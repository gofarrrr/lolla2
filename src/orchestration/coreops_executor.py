# CoreOps Executor (Walking Skeleton)
# Executes a linear CoreProgram using CognitiveCoreService

from __future__ import annotations

from typing import Dict, Any, Optional
import asyncio
import time
from datetime import datetime

from src.services.coreops_dsl import CoreProgram
from src.services.cognitive_core_service import CognitiveCoreService
from src.models.cognitive_core import InferenceType
from src.core.unified_context_stream import UnifiedContextStream, ContextEventType


async def execute_core_program(
    program: CoreProgram,
    core: CognitiveCoreService,
    trace_id: str,
    context_stream: Optional[UnifiedContextStream] = None,
) -> Dict[str, Any]:
    # Phase 7: Track program execution start time
    program_start_time = time.time()

    results: Dict[str, Any] = {}
    id_map: Dict[str, str] = {}  # op id -> argument id

    for step in program.steps:
        # Phase 7: Track step execution start time
        step_start_time = time.time()

        if step.op == "core.create_argument":
            args = step.args
            claim = args.get("claim")
            inference_type = InferenceType(args.get("inference_type", "abductive"))
            premises_ids = [id_map.get(p, p) for p in args.get("premises", [])]
            evidence_ids = args.get("evidence_ids", [])
            evidence_snippets = args.get("evidence_snippets")

            # Budgets and retry policy
            latency_ms: Optional[int] = None
            if step.budgets:
                latency_ms = step.budgets.get("latency_ms")
            retries = int(step.on_error.get("retry", 0)) if step.on_error else 0

            async def do_call():
                return await core.create_argument(
                    trace_id=trace_id,
                    claim=claim,
                    inference_type=inference_type,
                    premises=premises_ids,
                    evidence_ids=evidence_ids,
                    evidence_snippets=evidence_snippets,
                )

            attempt = 0
            last_exc: Optional[Exception] = None
            step_success = False
            step_result = None

            while attempt <= retries:
                try:
                    if latency_ms:
                        arg = await asyncio.wait_for(
                            do_call(), timeout=latency_ms / 1000.0
                        )
                    else:
                        arg = await do_call()
                    id_map[step.id] = arg.id
                    results[step.id] = arg
                    step_success = True
                    step_result = arg
                    break
                except Exception as e:
                    last_exc = e
                    attempt += 1
                    if attempt > retries:
                        raise

            # Phase 7: Emit COREOPS_STEP_EXECUTED event with sanitized data
            step_duration_ms = int((time.time() - step_start_time) * 1000)

            if context_stream:
                # Sanitize arguments - no raw YAML content, only variable names and types
                sanitized_args = {
                    "claim_provided": bool(claim),
                    "inference_type": (
                        inference_type.value if inference_type else "unknown"
                    ),
                    "premises_count": len(premises_ids),
                    "evidence_count": len(evidence_ids),
                }

                evidence_ids_list = []
                if step_result and hasattr(step_result, "id"):
                    evidence_ids_list = [step_result.id]

                context_stream.add_event(
                    ContextEventType.COREOPS_STEP_EXECUTED,
                    {
                        "step_id": step.id,
                        "op": step.op,
                        "duration_ms": step_duration_ms,
                        "evidence_ids": evidence_ids_list,
                        "status": "success" if step_success else "failed",
                        "sanitized_args": sanitized_args,
                        "attempt_count": attempt,
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                )
        else:
            # Phase 7: Emit event for unsupported operations too
            step_duration_ms = int((time.time() - step_start_time) * 1000)

            if context_stream:
                context_stream.add_event(
                    ContextEventType.COREOPS_STEP_EXECUTED,
                    {
                        "step_id": step.id,
                        "op": step.op,
                        "duration_ms": step_duration_ms,
                        "evidence_ids": [],
                        "status": "unsupported",
                        "sanitized_args": {"op_type": step.op},
                        "attempt_count": 0,
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                )

            # Unsupported op in skeleton
            raise NotImplementedError(f"Unsupported op: {step.op}")

    # Phase 7: Emit COREOPS_RUN_SUMMARY event with sanitized data
    program_duration_ms = int((time.time() - program_start_time) * 1000)

    if context_stream:
        # Generate program fingerprint (no actual YAML content)
        import hashlib

        program_fingerprint = hashlib.md5(
            f"{program.name}_{program.version}_{len(program.steps)}".encode()
        ).hexdigest()[:8]

        # Sample claims from successful results (sanitized)
        sample_claims = []
        for result in results.values():
            if hasattr(result, "claim") and result.claim:
                # Sanitize claim - show structure but not content
                claim_summary = (
                    f"claim_{len(result.claim)}_chars"
                    if len(result.claim) > 20
                    else "short_claim"
                )
                sample_claims.append(claim_summary)
                if len(sample_claims) >= 3:  # Limit sample size
                    break

        context_stream.add_event(
            ContextEventType.COREOPS_RUN_SUMMARY,
            {
                "system_contract_id": f"coreops_{program_fingerprint}",
                "program_path": f"programs/{program.name}.yaml",  # Relative path only
                "step_count": len(program.steps),
                "argument_count": len(results),
                "sample_claims": sample_claims,
                "processing_time_ms": program_duration_ms,
                "execution_mode": "coreops_executor",
                "program_version": program.version,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    return results
