"""
Evidence Extraction Service

Extracted from UnifiedContextStream (Task 4.0)
Handles extraction and summarization of glass-box evidence events.

Created: 2025-10-18
Campaign: Operation Lean
Original Lines: 629-1203 from unified_context_stream.py
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class EvidenceExtractionService:
    """
    Evidence extraction and summarization service for glass-box transparency.

    Responsibilities:
    - Extract evidence events by type
    - Generate evidence summaries
    - Create API-friendly evidence exports
    - Summarize individual evidence events
    """

    def __init__(self, events: List[Any], trace_id: Optional[str] = None):
        """
        Initialize evidence extraction service.

        Args:
            events: List of context events to extract evidence from
            trace_id: Optional trace identifier for evidence grouping
        """
        self.events = events
        self.trace_id = trace_id or "unknown"

    def get_evidence_events(
        self, evidence_types: Optional[List[Any]] = None
    ) -> List[Any]:
        """
        Get all glass-box evidence events.

        Args:
            evidence_types: Optional list of specific evidence types to filter

        Returns:
            List of evidence events in chronological order
        """
        # Import here to avoid circular dependency
        try:
            from src.core.unified_context_stream import ContextEventType
        except ImportError:
            # Fallback: use provided evidence_types or return empty
            if evidence_types is None:
                return []

        # Default evidence event types
        if evidence_types is None:
            evidence_types = [
                ContextEventType.MODEL_SELECTION_JUSTIFICATION,
                ContextEventType.SYNERGY_META_DIRECTIVE,
                ContextEventType.COREOPS_RUN_SUMMARY,
                ContextEventType.CONTRADICTION_AUDIT,
                ContextEventType.MENTAL_MODEL_ACTIVATION,
                ContextEventType.EVIDENCE_COLLECTION_COMPLETE,
            ]

        evidence_events = [
            event for event in self.events if event.event_type in evidence_types
        ]

        return sorted(evidence_events, key=lambda e: e.timestamp)

    def get_consultant_selection_evidence(self) -> List[Any]:
        """
        Get all consultant selection evidence events.

        Returns:
            List of MODEL_SELECTION_JUSTIFICATION events
        """
        try:
            from src.core.unified_context_stream import ContextEventType

            return [
                event
                for event in self.events
                if event.event_type == ContextEventType.MODEL_SELECTION_JUSTIFICATION
            ]
        except ImportError:
            return []

    def get_synergy_evidence(self) -> List[Any]:
        """
        Get all mental model synergy evidence events.

        Returns:
            List of SYNERGY_META_DIRECTIVE events
        """
        try:
            from src.core.unified_context_stream import ContextEventType

            return [
                event
                for event in self.events
                if event.event_type == ContextEventType.SYNERGY_META_DIRECTIVE
            ]
        except ImportError:
            return []

    def get_coreops_evidence(self) -> List[Any]:
        """
        Get all V2 CoreOps execution evidence events.

        Returns:
            List of COREOPS_RUN_SUMMARY events
        """
        try:
            from src.core.unified_context_stream import ContextEventType

            return [
                event
                for event in self.events
                if event.event_type == ContextEventType.COREOPS_RUN_SUMMARY
            ]
        except ImportError:
            return []

    def get_contradiction_evidence(self) -> List[Any]:
        """
        Get all contradiction audit evidence events.

        Returns:
            List of CONTRADICTION_AUDIT events
        """
        try:
            from src.core.unified_context_stream import ContextEventType

            return [
                event
                for event in self.events
                if event.event_type == ContextEventType.CONTRADICTION_AUDIT
            ]
        except ImportError:
            return []

    def get_evidence_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of all evidence collected.

        Returns:
            Dictionary with evidence statistics and key insights
        """
        evidence_events = self.get_evidence_events()

        summary = {
            "trace_id": self.trace_id,
            "evidence_collection_timestamp": datetime.now(timezone.utc).isoformat(),
            "total_evidence_events": len(evidence_events),
            "evidence_types": {},
            "glass_box_completeness": 0.0,
            "consultant_selections": 0,
            "synergy_directives": 0,
            "coreops_executions": 0,
            "contradiction_audits": 0,
            "key_decisions": [],
            "evidence_timeline": [],
        }

        # Count evidence by type
        for event in evidence_events:
            event_type_str = event.event_type.value
            if event_type_str not in summary["evidence_types"]:
                summary["evidence_types"][event_type_str] = 0
            summary["evidence_types"][event_type_str] += 1

        # Extract key metrics
        summary["consultant_selections"] = len(self.get_consultant_selection_evidence())
        summary["synergy_directives"] = len(self.get_synergy_evidence())
        summary["coreops_executions"] = len(self.get_coreops_evidence())
        summary["contradiction_audits"] = len(self.get_contradiction_evidence())

        # Calculate glass-box completeness
        total_events = len(self.events)
        if total_events > 0:
            summary["glass_box_completeness"] = len(evidence_events) / total_events

        # Extract key decisions from evidence events
        try:
            from src.core.unified_context_stream import ContextEventType

            for event in evidence_events:
                if event.event_type == ContextEventType.MODEL_SELECTION_JUSTIFICATION:
                    selection_data = event.data
                    summary["key_decisions"].append(
                        {
                            "type": "consultant_selection",
                            "timestamp": event.timestamp.isoformat(),
                            "rationale": selection_data.get("selection_rationale", ""),
                            "confidence": selection_data.get("total_confidence", 0),
                            "consultant_count": selection_data.get("consultant_count", 0),
                        }
                    )

                elif event.event_type == ContextEventType.SYNERGY_META_DIRECTIVE:
                    synergy_data = event.data
                    summary["key_decisions"].append(
                        {
                            "type": "synergy_directive",
                            "timestamp": event.timestamp.isoformat(),
                            "meta_directive": synergy_data.get("meta_directive", ""),
                            "confidence": synergy_data.get("confidence_score", 0),
                            "model_count": synergy_data.get("model_count", 0),
                        }
                    )
        except ImportError:
            pass

        # Create evidence timeline (last 10 evidence events)
        for event in evidence_events[-10:]:
            summary["evidence_timeline"].append(
                {
                    "timestamp": event.timestamp.isoformat(),
                    "type": event.event_type.value,
                    "description": self.summarize_evidence_event(event),
                }
            )

        return summary

    def summarize_evidence_event(self, event: Any) -> str:
        """
        Create human-readable summary of an evidence event.

        Args:
            event: Context event to summarize

        Returns:
            Human-readable summary string
        """
        try:
            from src.core.unified_context_stream import ContextEventType
        except ImportError:
            return f"Evidence event: {event.event_type}"

        if event.event_type == ContextEventType.MODEL_SELECTION_JUSTIFICATION:
            data = event.data
            consultant_count = data.get("consultant_count", 0)
            confidence = data.get("total_confidence", 0)
            return f"Selected {consultant_count} consultants with {confidence:.1%} confidence"

        elif event.event_type == ContextEventType.SYNERGY_META_DIRECTIVE:
            data = event.data
            model_count = data.get("model_count", 0)
            confidence = data.get("confidence_score", 0)
            return f"Generated meta-directive from {model_count} models with {confidence:.1%} confidence"

        elif event.event_type == ContextEventType.COREOPS_RUN_SUMMARY:
            data = event.data
            contract_id = data.get("system_contract_id", "unknown")
            argument_count = data.get("argument_count", 0)
            return f"Executed {contract_id} generating {argument_count} arguments"

        elif event.event_type == ContextEventType.COREOPS_STEP_EXECUTED:
            data = event.data
            step_id = data.get("step_id", "unknown")
            op = data.get("op", "unknown")
            status = data.get("status", "unknown")
            duration_ms = data.get("duration_ms", 0)
            return f"Executed step {step_id} ({op}) in {duration_ms}ms - {status}"

        elif event.event_type == ContextEventType.CONTRADICTION_AUDIT:
            data = event.data
            contradiction_count = data.get("contradiction_count", 0)
            synthesis_count = data.get("synthesis_count", 0)
            return f"Found {contradiction_count} contradictions, {synthesis_count} syntheses"

        # Learning Systems Events
        elif event.event_type == ContextEventType.LEARNING_CYCLE_STARTED:
            data = event.data
            cycle_id = data.get("cycle_id", "unknown")
            system_type = data.get("system_type", "unknown")
            return f"Started learning cycle {cycle_id} for {system_type}"

        elif event.event_type == ContextEventType.LEARNING_CYCLE_COMPLETED:
            data = event.data
            cycle_id = data.get("cycle_id", "unknown")
            duration_ms = data.get("duration_ms", 0)
            improvements = data.get("improvements_count", 0)
            return f"Completed learning cycle {cycle_id} in {duration_ms}ms with {improvements} improvements"

        elif event.event_type == ContextEventType.PATTERN_EFFECTIVENESS_UPDATE:
            data = event.data
            pattern_id = data.get("pattern_id", "unknown")
            old_score = data.get("old_score", 0)
            new_score = data.get("new_score", 0)
            evidence_count = data.get("evidence_count", 0)
            return f"Updated pattern {pattern_id}: {old_score:.2f}→{new_score:.2f} (evidence: {evidence_count})"

        elif event.event_type == ContextEventType.DIVERSITY_POLICY_ENFORCED:
            data = event.data
            policy_id = data.get("policy_id", "unknown")
            action = data.get("action", "unknown")
            before_metric = data.get("before_metric", 0)
            after_metric = data.get("after_metric", 0)
            return f"Applied policy {policy_id}: {action} ({before_metric:.2f}→{after_metric:.2f})"

        elif event.event_type == ContextEventType.OPTIMIZATION_ACTION_TAKEN:
            data = event.data
            action_id = data.get("action_id", "unknown")
            reason = data.get("reason", "unknown")
            guardrails_passed = data.get("guardrails_passed", True)
            return f"Optimization {action_id}: {reason} (guardrails: {guardrails_passed})"

        elif event.event_type == ContextEventType.FEEDBACK_INGESTED:
            data = event.data
            source = data.get("source", "unknown")
            items_count = data.get("items_count", 0)
            return f"Ingested {items_count} feedback items from {source}"

        elif event.event_type == ContextEventType.DASHBOARD_METRICS_UPDATED:
            data = event.data
            kpis = data.get("kpis", {})
            kpi_count = len(kpis)
            return f"Updated {kpi_count} dashboard KPIs"

        else:
            return f"Evidence event: {event.event_type.value}"

    def export_evidence_for_api(self) -> Dict[str, Any]:
        """
        Export evidence in API-friendly format for frontend consumption.

        Returns:
            Structured evidence data optimized for frontend display
        """
        evidence_events = self.get_evidence_events()

        # Calculate session duration (minutes)
        session_duration_minutes = 0.0
        if self.events and len(self.events) > 1:
            first_event = self.events[0]
            last_event = self.events[-1]
            duration_seconds = (last_event.timestamp - first_event.timestamp).total_seconds()
            session_duration_minutes = duration_seconds / 60.0

        api_evidence = {
            "metadata": {
                "trace_id": self.trace_id,
                "total_evidence_events": len(evidence_events),
                "collection_timestamp": datetime.now(timezone.utc).isoformat(),
                "session_duration_minutes": round(session_duration_minutes, 2),
            },
            "consultant_selections": [],
            "synergy_directives": [],
            "coreops_executions": [],
            "contradiction_audits": [],
            "evidence_timeline": [],
        }

        # Process consultant selection evidence
        for event in self.get_consultant_selection_evidence():
            data = event.data
            api_evidence["consultant_selections"].append(
                {
                    "timestamp": event.timestamp.isoformat(),
                    "selection_rationale": data.get("selection_rationale", ""),
                    "total_confidence": data.get("total_confidence", 0),
                    "consultant_count": data.get("consultant_count", 0),
                    "consultants": data.get("consultants", []),
                }
            )

        # Process synergy directives
        for event in self.get_synergy_evidence():
            data = event.data
            api_evidence["synergy_directives"].append(
                {
                    "timestamp": event.timestamp.isoformat(),
                    "meta_directive": data.get("meta_directive", ""),
                    "confidence_score": data.get("confidence_score", 0),
                    "model_count": data.get("model_count", 0),
                    "models_involved": data.get("models_involved", []),
                }
            )

        # Process CoreOps executions
        for event in self.get_coreops_evidence():
            data = event.data
            api_evidence["coreops_executions"].append(
                {
                    "timestamp": event.timestamp.isoformat(),
                    "system_contract_id": data.get("system_contract_id", ""),
                    "argument_count": data.get("argument_count", 0),
                    "runtime_ms": data.get("runtime_ms", 0),
                    "success": data.get("success", True),
                }
            )

        # Process contradiction audits
        for event in self.get_contradiction_evidence():
            data = event.data
            api_evidence["contradiction_audits"].append(
                {
                    "timestamp": event.timestamp.isoformat(),
                    "contradiction_count": data.get("contradiction_count", 0),
                    "synthesis_count": data.get("synthesis_count", 0),
                    "resolution_strategy": data.get("resolution_strategy", ""),
                }
            )

        # Create evidence timeline (all evidence events, chronologically)
        for event in evidence_events:
            api_evidence["evidence_timeline"].append(
                {
                    "timestamp": event.timestamp.isoformat(),
                    "event_type": event.event_type.value,
                    "summary": self.summarize_evidence_event(event),
                    "confidence": event.data.get("confidence", event.data.get("total_confidence", 0)),
                }
            )

        return api_evidence
