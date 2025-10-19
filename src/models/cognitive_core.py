# Cognitive Core Models (Walking Skeleton)
# Minimal, typed representations for Arguments and related enums

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict
import uuid


class InferenceType(str, Enum):
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"


class ArgumentStatus(str, Enum):
    HYPOTHESIS = "hypothesis"
    SUPPORTED = "supported_conclusion"
    REFUTED = "refuted_claim"
    BOUNDED = "bounded_validity"  # for contradictions resolved via boundary conditions


class TripwireType(str, Enum):
    OVERCONFIDENCE = "overconfidence"
    MOTIVATED_REASONING = "motivated_reasoning"
    SUBSTITUTION = "substitution"
    ZERO_VALIDITY_ENV = "zero_validity_env"


@dataclass
class ArgumentStep:
    inference_type: InferenceType
    premises: List[str]
    conclusion: str
    evidence_ids: List[str] = field(default_factory=list)
    notes: Optional[str] = None


@dataclass
class Argument:
    id: str
    trace_id: str
    claim: str
    status: ArgumentStatus = ArgumentStatus.HYPOTHESIS

    # Lineage
    parent_argument_id: Optional[str] = None

    # Reasoning
    reasoning_trace: List[ArgumentStep] = field(default_factory=list)
    next_step_question: Optional[str] = None

    # Self-critique
    confidence_score: float = 0.0
    tripwires_triggered: List[TripwireType] = field(default_factory=list)

    # Contradiction relationship facet
    is_contradiction: bool = False
    contradicts_argument_id: Optional[str] = None
    relation_type: Optional[str] = None  # e.g., "contradicts" | "refutes" | "supports"

    # Evidence
    evidence_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "trace_id": self.trace_id,
            "claim": self.claim,
            "status": self.status.value,
            "parent_argument_id": self.parent_argument_id,
            "reasoning_trace": [
                {
                    "inference_type": s.inference_type.value,
                    "premises": s.premises,
                    "conclusion": s.conclusion,
                    "evidence_ids": s.evidence_ids,
                    "notes": s.notes,
                }
                for s in self.reasoning_trace
            ],
            "next_step_question": self.next_step_question,
            "confidence_score": self.confidence_score,
            "tripwires_triggered": [t.value for t in self.tripwires_triggered],
            "is_contradiction": self.is_contradiction,
            "contradicts_argument_id": self.contradicts_argument_id,
            "relation_type": self.relation_type,
            "evidence_ids": self.evidence_ids,
        }


def new_argument_id() -> str:
    return str(uuid.uuid4())
