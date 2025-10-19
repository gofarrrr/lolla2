"""
Evidence Collection Service - Glass-box Transparency Pipeline
============================================================

Collects, stores, and manages auditable evidence for "what we used" and "how we picked"
mental models, consultants, and decision rationale.

Core Evidence Types:
1. MODEL_SELECTION_JUSTIFICATION - Why consultants were selected
2. SYNERGY_META_DIRECTIVE - How mental models combine
3. COREOPS_RUN_SUMMARY - Evidence of V2 CoreOps execution
4. CONTRADICTION_AUDIT - ULTRATHINK synthesis results
"""

from dataclasses import dataclass
from typing import Dict, List, Any
from datetime import datetime, timezone
import logging

from src.core.unified_context_stream import (
    UnifiedContextStream,
    ContextEventType,
    ContextEvent,
)

logger = logging.getLogger(__name__)


@dataclass
class ConsultantSelectionEvidence:
    """Evidence for why specific consultants were selected"""

    consultant_id: str
    consultant_type: str
    version: str
    synergy_score: float
    domain_match_score: float
    why_selected: List[str]
    top_features: List[str]
    chosen_nway_clusters: List[str]
    cluster_rationale: str


@dataclass
class SynergyMetaDirective:
    """Evidence for how mental models combine"""

    meta_directive: str
    synergy_insight: str
    conflict_insight: str
    confidence_score: float
    participating_models: List[str]
    instructional_cue_apce: str
    emergent_effect_summary: str


@dataclass
class CoreOpsRunSummary:
    """Evidence of V2 CoreOps execution"""

    system_contract_id: str
    program_path: str
    step_count: int
    argument_count: int
    sample_claims: List[str]
    rag_evidence_ids: List[str]
    processing_time_ms: int
    execution_mode: str


@dataclass
class ContradictionAuditResult:
    """Evidence from ULTRATHINK synthesis"""

    contradiction_count: int
    synthesis_count: int
    example_contradiction: str
    example_synthesis: str
    confidence_score: float
    bias_mitigation_applied: bool


class EvidenceCollectionService:
    """
    Service for collecting and managing glass-box evidence throughout the pipeline.

    Provides full audit trail for:
    - Consultant selection rationale
    - Mental model activation and synergy
    - V2 CoreOps execution proof
    - ULTRATHINK contradiction analysis
    """

    def __init__(self, context_stream: UnifiedContextStream):
        self.context_stream = context_stream
        self.session_evidence: Dict[str, Any] = {}

    def record_consultant_selection(
        self,
        selected_consultants: List[ConsultantSelectionEvidence],
        selection_rationale: str,
        total_confidence: float,
    ) -> None:
        """Record evidence for consultant selection decisions"""

        # Prepare evidence data
        evidence_data = {
            "selection_rationale": selection_rationale,
            "total_confidence": total_confidence,
            "consultant_count": len(selected_consultants),
            "consultants": [],
        }

        for consultant in selected_consultants:
            consultant_data = {
                "consultant_id": consultant.consultant_id,
                "consultant_type": consultant.consultant_type,
                "version": consultant.version,
                "synergy_score": consultant.synergy_score,
                "domain_match_score": consultant.domain_match_score,
                "why_selected": consultant.why_selected,
                "top_features": consultant.top_features,
                "chosen_nway_clusters": consultant.chosen_nway_clusters,
                "cluster_rationale": consultant.cluster_rationale,
            }
            evidence_data["consultants"].append(consultant_data)

        # Log to UnifiedContextStream
        event = ContextEvent(
            event_type=ContextEventType.MODEL_SELECTION_JUSTIFICATION,
            data=evidence_data,
            metadata={
                "evidence_type": "consultant_selection",
                "audit_level": "complete",
                "trace_id": self.context_stream.trace_id,
            },
        )

        self.context_stream.add_event(event)
        self.session_evidence["consultant_selection"] = evidence_data

        logger.info(
            f"🔍 Evidence recorded: Selected {len(selected_consultants)} consultants with confidence {total_confidence:.2f}"
        )

    def record_synergy_meta_directive(
        self, meta_directive: SynergyMetaDirective, activated_clusters: List[str]
    ) -> None:
        """Record evidence for mental model synergy and meta-directive"""

        evidence_data = {
            "meta_directive": meta_directive.meta_directive,
            "synergy_insight": meta_directive.synergy_insight,
            "conflict_insight": meta_directive.conflict_insight,
            "confidence_score": meta_directive.confidence_score,
            "participating_models": meta_directive.participating_models,
            "instructional_cue_apce": meta_directive.instructional_cue_apce,
            "emergent_effect_summary": meta_directive.emergent_effect_summary,
            "activated_clusters": activated_clusters,
            "model_count": len(meta_directive.participating_models),
        }

        # Log to UnifiedContextStream
        event = ContextEvent(
            event_type=ContextEventType.SYNERGY_META_DIRECTIVE,
            data=evidence_data,
            metadata={
                "evidence_type": "synergy_directive",
                "audit_level": "complete",
                "trace_id": self.context_stream.trace_id,
            },
        )

        self.context_stream.add_event(event)
        self.session_evidence["synergy_meta_directive"] = evidence_data

        logger.info(
            f"🔍 Evidence recorded: Synergy directive with {len(meta_directive.participating_models)} models, confidence {meta_directive.confidence_score:.2f}"
        )

    def record_coreops_execution(self, execution_summary: CoreOpsRunSummary) -> None:
        """Record evidence of V2 CoreOps execution"""

        evidence_data = {
            "system_contract_id": execution_summary.system_contract_id,
            "program_path": execution_summary.program_path,
            "step_count": execution_summary.step_count,
            "argument_count": execution_summary.argument_count,
            "sample_claims": execution_summary.sample_claims,
            "rag_evidence_ids": execution_summary.rag_evidence_ids,
            "processing_time_ms": execution_summary.processing_time_ms,
            "execution_mode": execution_summary.execution_mode,
            "v2_proof": True,
        }

        # Log to UnifiedContextStream
        event = ContextEvent(
            event_type=ContextEventType.COREOPS_RUN_SUMMARY,
            data=evidence_data,
            metadata={
                "evidence_type": "coreops_execution",
                "audit_level": "complete",
                "trace_id": self.context_stream.trace_id,
                "contract_id": execution_summary.system_contract_id,
            },
        )

        self.context_stream.add_event(event)

        # Store in session evidence by contract_id
        if "coreops_executions" not in self.session_evidence:
            self.session_evidence["coreops_executions"] = {}
        self.session_evidence["coreops_executions"][
            execution_summary.system_contract_id
        ] = evidence_data

        logger.info(
            f"🔍 Evidence recorded: CoreOps execution {execution_summary.system_contract_id} with {execution_summary.argument_count} arguments"
        )

    def record_contradiction_audit(
        self, audit_result: ContradictionAuditResult
    ) -> None:
        """Record evidence from ULTRATHINK contradiction analysis"""

        evidence_data = {
            "contradiction_count": audit_result.contradiction_count,
            "synthesis_count": audit_result.synthesis_count,
            "example_contradiction": audit_result.example_contradiction,
            "example_synthesis": audit_result.example_synthesis,
            "confidence_score": audit_result.confidence_score,
            "bias_mitigation_applied": audit_result.bias_mitigation_applied,
            "ultrathink_active": True,
        }

        # Log to UnifiedContextStream
        event = ContextEvent(
            event_type=ContextEventType.CONTRADICTION_AUDIT,
            data=evidence_data,
            metadata={
                "evidence_type": "contradiction_audit",
                "audit_level": "complete",
                "trace_id": self.context_stream.trace_id,
            },
        )

        self.context_stream.add_event(event)
        self.session_evidence["contradiction_audit"] = evidence_data

        logger.info(
            f"🔍 Evidence recorded: ULTRATHINK found {audit_result.contradiction_count} contradictions, {audit_result.synthesis_count} syntheses"
        )

    def finalize_evidence_collection(self) -> Dict[str, Any]:
        """Complete evidence collection and return comprehensive audit trail"""

        # Create final evidence package
        evidence_package = {
            "trace_id": self.context_stream.trace_id,
            "collection_timestamp": datetime.now(timezone.utc).isoformat(),
            "evidence_completeness": self._assess_completeness(),
            "session_evidence": self.session_evidence,
            "audit_summary": self._generate_audit_summary(),
        }

        # Log completion
        event = ContextEvent(
            event_type=ContextEventType.EVIDENCE_COLLECTION_COMPLETE,
            data=evidence_package,
            metadata={
                "evidence_type": "collection_complete",
                "audit_level": "summary",
                "trace_id": self.context_stream.trace_id,
            },
        )

        self.context_stream.add_event(event)

        logger.info(
            f"🔍 Evidence collection finalized for trace {self.context_stream.trace_id}"
        )
        return evidence_package

    def _assess_completeness(self) -> Dict[str, bool]:
        """Assess completeness of evidence collection"""
        return {
            "consultant_selection_recorded": "consultant_selection"
            in self.session_evidence,
            "synergy_directive_recorded": "synergy_meta_directive"
            in self.session_evidence,
            "coreops_executions_recorded": "coreops_executions"
            in self.session_evidence,
            "contradiction_audit_recorded": "contradiction_audit"
            in self.session_evidence,
        }

    def _generate_audit_summary(self) -> Dict[str, Any]:
        """Generate high-level audit summary"""
        summary = {
            "total_consultants_selected": 0,
            "total_mental_models_activated": 0,
            "total_coreops_executions": 0,
            "total_contradictions_found": 0,
            "glass_box_completeness": 0.0,
        }

        if "consultant_selection" in self.session_evidence:
            summary["total_consultants_selected"] = self.session_evidence[
                "consultant_selection"
            ]["consultant_count"]

        if "synergy_meta_directive" in self.session_evidence:
            summary["total_mental_models_activated"] = self.session_evidence[
                "synergy_meta_directive"
            ]["model_count"]

        if "coreops_executions" in self.session_evidence:
            summary["total_coreops_executions"] = len(
                self.session_evidence["coreops_executions"]
            )

        if "contradiction_audit" in self.session_evidence:
            summary["total_contradictions_found"] = self.session_evidence[
                "contradiction_audit"
            ]["contradiction_count"]

        # Calculate completeness score
        completeness_checks = self._assess_completeness()
        summary["glass_box_completeness"] = sum(completeness_checks.values()) / len(
            completeness_checks
        )

        return summary
