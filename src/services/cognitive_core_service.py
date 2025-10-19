# CognitiveCoreService (Walking Skeleton)
# MVP implementation: create_argument, find_contradictions_in_trace, synthesize_contradiction

from __future__ import annotations

import logging
from typing import List, Dict, Optional, Any

from src.models.cognitive_core import (
    Argument,
    ArgumentStep,
    InferenceType,
    ArgumentStatus,
    TripwireType,
    new_argument_id,
)
from src.core.unified_context_stream import get_unified_context_stream, ContextEventType
from src.services.evidence_manager import EvidenceManager

logger = logging.getLogger(__name__)


class CognitiveCoreService:
    """Central System 2 reasoning engine (walking skeleton)."""

    def __init__(
        self,
        context_stream: Optional[Any] = None,
        mock_mode: bool = True,
        use_llm_mode: bool = False,
        evidence_manager: Optional[EvidenceManager] = None,
    ):
        self.context_stream = context_stream or get_unified_context_stream()
        self.mock_mode = mock_mode
        self.use_llm_mode = use_llm_mode
        self.evidence_manager = evidence_manager or EvidenceManager()
        # In-memory store for MVP
        self._arguments: Dict[str, Argument] = {}

    # --- Core creation ---
    async def create_argument(
        self,
        *,
        trace_id: str,
        claim: str,
        inference_type: InferenceType,
        premises: Optional[List[str]] = None,
        evidence_ids: Optional[List[str]] = None,
        evidence_snippets: Optional[List[Dict[str, Any]]] = None,
        parent_argument_id: Optional[str] = None,
    ) -> Argument:
        """
        Create a single-step argument. In mock mode, we construct deterministic content
        and confidence; no external LLM call. Emits REASONING_STEP event.
        """
        # If LLM mode is enabled, call manager (skeleton keeps mock behavior if disabled)
        if self.use_llm_mode and not self.mock_mode:
            # Placeholder prompt assembly (will be expanded in next increment)
            from src.engine.core.llm_integration_adapter import get_unified_llm_adapter

            adapter = get_unified_llm_adapter()
            await adapter.initialize()
            system = "System: Senior Analyst. Use proof obligations. Return JSON with inference, premises, conclusion."
            prompt = f"Claim: {claim}\nInference: {inference_type.value}\nPremises: {premises or []}\nEvidence: {evidence_ids or []}"
            try:
                _ = await adapter.call_llm_unified(
                    prompt=prompt, task_name="complex_reasoning", system_prompt=system
                )
            except Exception:
                # Fall back to mock path if LLM call fails
                pass

        # Normalize evidence: upsert any raw snippets to canonical evidence ids
        canonical_evidence_ids: List[str] = list(evidence_ids or [])
        if evidence_snippets:
            for snip in evidence_snippets:
                content = snip.get("content") if isinstance(snip, dict) else str(snip)
                if not content:
                    continue
                source_ref = (
                    snip.get("source_ref", "inline")
                    if isinstance(snip, dict)
                    else "inline"
                )
                source_type = (
                    snip.get("source_type", "doc") if isinstance(snip, dict) else "doc"
                )
                ev = self.evidence_manager.upsert_evidence(
                    content=content, source_ref=source_ref, source_type=source_type
                )
                canonical_evidence_ids.append(ev.id)
        # de-dup
        canonical_evidence_ids = list(dict.fromkeys(canonical_evidence_ids))

        arg_id = new_argument_id()
        step = ArgumentStep(
            inference_type=inference_type,
            premises=premises or [],
            conclusion=claim,
            evidence_ids=canonical_evidence_ids,
        )
        arg = Argument(
            id=arg_id,
            trace_id=trace_id,
            claim=claim,
            status=ArgumentStatus.HYPOTHESIS,
            parent_argument_id=parent_argument_id,
            reasoning_trace=[step],
            confidence_score=self._mock_confidence(
                inference_type, canonical_evidence_ids
            ),
            tripwires_triggered=[],
            evidence_ids=canonical_evidence_ids,
        )

        # Tripwires: simple overconfidence check
        if arg.confidence_score > 0.9 and not arg.evidence_ids:
            arg.tripwires_triggered.append(TripwireType.OVERCONFIDENCE)

        self._arguments[arg.id] = arg

        # Log to context stream
        self.context_stream.add_event(
            ContextEventType.REASONING_STEP,
            data={"argument": arg.to_dict()},
            metadata={"phase": "core.create_argument"},
        )
        return arg

    # --- Contradiction detection ---
    async def find_contradictions_in_trace(self, trace_id: str) -> List[Argument]:
        """Naive contradiction detection: look for pairs with opposing claims patterns."""
        args = [a for a in self._arguments.values() if a.trace_id == trace_id]
        contradictions: List[Argument] = []
        # naive: if one claim starts with "not " of the other, mark contradiction
        claims_map = {a.id: a.claim.strip().lower() for a in args}
        for a in args:
            for b in args:
                if a.id == b.id:
                    continue
                if claims_map[b.id] == f"not {claims_map[a.id]}":
                    # mark a as contradicting b
                    if not a.is_contradiction:
                        a.is_contradiction = True
                        a.contradicts_argument_id = b.id
                        self.context_stream.add_event(
                            ContextEventType.CONTRADICTION_DETECTED,
                            data={"argument_a_id": a.id, "argument_b_id": b.id},
                            metadata={"phase": "core.find_contradictions"},
                        )
                        contradictions.append(a)
        return contradictions

    # --- Synthesis ---
    async def synthesize_contradiction(self, a_id: str, b_id: str) -> Argument:
        """
        Believing Game mock: generate a bounded synthesis argument whose claim reconciles A and B
        via boundary conditions. Emits SYNTHESIS_CREATED event.
        """
        a = self._arguments.get(a_id)
        b = self._arguments.get(b_id)
        if not a or not b:
            raise ValueError("Arguments not found for synthesis")

        # Optionally use LLM to generate synthesis text in llm_mode
        if self.use_llm_mode and not self.mock_mode:
            try:
                from src.engine.core.llm_integration_adapter import (
                    get_unified_llm_adapter,
                )

                adapter = get_unified_llm_adapter()
                await adapter.initialize()
                system = "Believing Game: inhabit both positions charitably; propose synthesis or boundary conditions."
                prompt = (
                    f"A: {a.claim}\nB: {b.claim}\nReturn a concise synthesis sentence."
                )
                _ = await adapter.call_llm_unified(
                    prompt=prompt, task_name="strategic_synthesis", system_prompt=system
                )
            except Exception:
                pass

        synthesis_claim = f"Synthesis: {a.claim} holds under conditions X; {b.claim} holds under conditions Y"
        synth = await self.create_argument(
            trace_id=a.trace_id,
            claim=synthesis_claim,
            inference_type=InferenceType.ABDUCTIVE,
            premises=[a.id, b.id],
            evidence_ids=list(set(a.evidence_ids + b.evidence_ids)),
            parent_argument_id=None,
        )
        synth.status = ArgumentStatus.BOUNDED

        # lineage: set both as children by pointing their parent to synthesis
        a.parent_argument_id = synth.id
        b.parent_argument_id = synth.id

        self.context_stream.add_event(
            ContextEventType.SYNTHESIS_CREATED,
            data={"synthesis": synth.to_dict(), "a_id": a.id, "b_id": b.id},
            metadata={"phase": "core.synthesize_contradiction"},
        )
        return synth

    # --- Helpers ---
    def _mock_confidence(
        self, inference_type: InferenceType, evidence_ids: List[str]
    ) -> float:
        base = {
            InferenceType.DEDUCTIVE: 0.85,
            InferenceType.INDUCTIVE: 0.7,
            InferenceType.ABDUCTIVE: 0.6,
        }[inference_type]
        bonus = min(0.15, 0.05 * len(evidence_ids))
        return min(0.99, base + bonus)

    # Expose arguments for test assertions
    def get_arguments_by_trace(self, trace_id: str) -> List[Argument]:
        return [a for a in self._arguments.values() if a.trace_id == trace_id]
