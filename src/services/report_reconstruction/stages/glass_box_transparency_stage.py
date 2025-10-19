"""
Glass-Box Transparency Stage
============================

Extracts all glass-box transparency data from UnifiedContextStream events.

Responsibility:
- Extract human interactions (HITL, clarifications, query enhancements)
- Extract research provider events (requests, responses, fallbacks)
- Extract evidence trail (citations, sources)
- Build quality ribbon (research depth, dissent level, stability)
- Build plan overview (stage performance, timeline)
- Extract dissent signals (devils advocate, contradictions, tensions)
- Build quality metrics (confidence, diversity, execution time)

Complexity: CC<8 per method (Multiple extraction methods, each focused)
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Optional
from uuid import UUID

from src.core.unified_context_stream import ContextEventType
from src.services.report_reconstruction.reconstruction_state import ReconstructionState
from src.services.report_reconstruction.reconstruction_stage import (
    ReconstructionStage,
    ReconstructionError,
)

logger = logging.getLogger(__name__)


class GlassBoxTransparencyStage(ReconstructionStage):
    """
    Stage 5: Glass-Box Transparency

    Extracts all transparency and evidence data from UnifiedContextStream events.
    """

    @property
    def name(self) -> str:
        return "glass_box_transparency"

    @property
    def description(self) -> str:
        return "Extract transparency data (evidence, quality, plan, dissent)"

    def process(self, state: ReconstructionState) -> ReconstructionState:
        """
        Extract all transparency data from events.

        Args:
            state: Current reconstruction state with events populated

        Returns:
            Updated state with all transparency fields populated
        """
        try:
            events = state.events

            # Extract transparency data components
            human_interactions = self._extract_human_interactions(events)
            research_providers = self._extract_research_provider_events(events)
            evidence_trail = self._extract_evidence_trail(events)
            quality_ribbon = self._build_quality_ribbon(events)
            plan_overview = self._build_plan_overview(events)
            dissent_signals = self._extract_dissent_signals(events)

            # Build quality metrics from states and events
            quality_metrics = self._build_quality_metrics(state.cognitive_states, events)

            return state.with_transparency(
                human_interactions=human_interactions,
                research_providers=research_providers,
                evidence_trail=evidence_trail,
                quality_ribbon=quality_ribbon,
                plan_overview=plan_overview,
                dissent_signals=dissent_signals,
                quality_metrics=quality_metrics,
            )

        except Exception as e:
            raise ReconstructionError(
                self.name,
                f"Failed to extract transparency data for trace_id={state.trace_id}",
                cause=e,
            )

    def _extract_human_interactions(self, events: List[Any]) -> Optional[Dict[str, Any]]:
        """Extract human interaction data from events."""
        human_interactions = {
            "answered_questions": [],
            "hitl_interactions": [],
            "query_enhancements": [],
        }

        for event in events:
            event_type = self._get_event_type(event)
            event_data = self._get_event_data(event)

            if event_type == "hitl_request":
                human_interactions["hitl_interactions"].append(
                    {
                        "interaction_id": event_data.get("interaction_id"),
                        "timestamp": self._get_timestamp(event),
                        "decision_point": event_data.get("decision_point"),
                        "request": event_data.get("request"),
                        "response": event_data.get("response", ""),
                        "impact": event_data.get("impact", {}),
                    }
                )

            if event_type == "query_enhanced_from_clarification":
                human_interactions["query_enhancements"].append(
                    {
                        "original_query": event_data.get("original_query"),
                        "enhanced_query": event_data.get("enhanced_query"),
                        "enhancement_source": "clarification_answers",
                        "timestamp": self._get_timestamp(event),
                    }
                )

        # Return None if no interactions found
        if not any(human_interactions.values()):
            return None

        return human_interactions

    def _extract_research_provider_events(
        self, events: List[Any]
    ) -> Optional[List[Dict[str, Any]]]:
        """Extract research provider transparency data from events."""
        research_events = []

        for event in events:
            event_type = self._get_event_type(event)
            event_data = self._get_event_data(event)

            if event_type in [
                "research_provider_request",
                "research_provider_response",
                "research_provider_fallback",
            ]:
                research_events.append(
                    {
                        "event_id": self._get_event_id(event),
                        "event_type": event_type,
                        "timestamp": self._get_timestamp(event),
                        "provider": event_data.get("provider"),
                        "query": event_data.get("query", ""),
                        "latency_ms": event_data.get("latency_ms"),
                        "citations_count": event_data.get("citations_count", 0),
                        "result_preview": event_data.get("result_preview", ""),
                        "confidence": event_data.get("confidence", 0.0),
                        "failure_reason": event_data.get("failure_reason"),
                        "fallback_provider": event_data.get("fallback_provider"),
                    }
                )

        return research_events if research_events else None

    def _extract_evidence_trail(self, events: List[Any]) -> Optional[List[Dict[str, Any]]]:
        """Extract evidence items from events into a normalized trail."""
        evidence: List[Dict[str, Any]] = []

        for event in events:
            event_type = self._get_event_type(event)
            event_data = self._get_event_data(event)

            # Evidence collection complete events
            if event_type in [
                ContextEventType.EVIDENCE_COLLECTION_COMPLETE.value,
                "evidence_collection_complete",
            ]:
                items = event_data.get("evidence") or []
                if isinstance(items, list):
                    for it in items:
                        try:
                            evidence.append(self._build_evidence_item(it, event))
                        except Exception:
                            continue

            # Oracle research citations
            if event_type == "oracle_research_complete":
                citations = self._extract_citations(event_data)
                for c in citations:
                    try:
                        evidence.append(self._build_citation_evidence(c, event))
                    except Exception:
                        continue

        return evidence if evidence else None

    def _build_quality_ribbon(self, events: List[Any]) -> Dict[str, Any]:
        """Compute quality ribbon heuristics from raw events."""
        research_resps = [
            e for e in events if self._get_event_type(e) == ContextEventType.RESEARCH_PROVIDER_RESPONSE.value
        ]
        fallbacks = [
            e for e in events if self._get_event_type(e) == ContextEventType.RESEARCH_PROVIDER_FALLBACK.value
        ]
        errors = [e for e in events if self._get_event_type(e) == ContextEventType.ERROR_OCCURRED.value]
        context_ok = [
            e for e in events if self._get_event_type(e) == ContextEventType.CONTEXT_PRESERVATION_VALIDATED.value
        ]
        contradictions = [
            e
            for e in events
            if self._get_event_type(e)
            in (
                ContextEventType.CONTRADICTION_DETECTED.value,
                ContextEventType.CONTRADICTION_AUDIT.value,
            )
        ]
        da_bias = [
            e for e in events if self._get_event_type(e) == ContextEventType.DEVILS_ADVOCATE_BIAS_FOUND.value
        ]

        # Extract citation counts
        citations = []
        for e in research_resps:
            d = self._get_event_data(e)
            v = d.get("citations_count") or d.get("sources_count") or 0
            try:
                citations.append(int(v))
            except Exception:
                pass

        # Compute heuristics scaled to [0,1]
        research_depth = (
            min(1.0, (sum(citations) / max(1, len(citations))) / 5.0) if citations else 0.3
        )
        dissent_events = len(contradictions) + len(da_bias)
        dissent_level = min(1.0, dissent_events / 5.0)
        provider_stability = 1.0 - min(
            1.0, (len(fallbacks) / max(1, len(research_resps) + len(fallbacks)))
        )
        data_consistency = min(1.0, len(context_ok) / max(1, len(context_ok) + len(errors)))

        return {
            "research_depth": research_depth,
            "dissent_level": dissent_level,
            "data_consistency": data_consistency,
            "provider_stability": provider_stability,
        }

    def _build_plan_overview(self, events: List[Any]) -> Dict[str, Any]:
        """Construct plan overview from stage performance events."""
        perf: Dict[str, Dict[str, Any]] = {}
        starts: Dict[str, str] = {}
        completes: Dict[str, str] = {}
        total_ms = 0

        for e in events:
            etv = self._get_event_type(e)
            d = self._get_event_data(e)

            if etv == ContextEventType.STAGE_PERFORMANCE_PROFILE_RECORDED.value:
                name = d.get("stage_name") or d.get("stage") or "stage"
                dur = int(d.get("wall_time_ms", d.get("duration_ms", 0)) or 0)
                perf[name] = {
                    "duration_ms": dur,
                    "token_count": d.get("token_count", 0),
                }
                total_ms += dur

            elif etv == ContextEventType.PIPELINE_STAGE_STARTED.value:
                name = d.get("stage_name") or d.get("stage") or "stage"
                starts[name] = d.get("start_time") or self._get_timestamp(e)

            elif etv == ContextEventType.PIPELINE_STAGE_COMPLETED.value:
                name = d.get("stage_name") or d.get("stage") or "stage"
                completes[name] = d.get("complete_time") or self._get_timestamp(e)

        # Merge into steps
        stage_names = set(perf.keys()) | set(starts.keys()) | set(completes.keys())
        steps: List[Dict[str, Any]] = []
        for name in stage_names:
            p = perf.get(name, {})
            steps.append(
                {
                    "stage_name": name,
                    "duration_ms": p.get("duration_ms", 0),
                    "token_count": p.get("token_count", 0),
                    "start_time": starts.get(name),
                    "complete_time": completes.get(name),
                }
            )

        return {"steps": steps, "total_duration_ms": total_ms}

    def _extract_dissent_signals(self, events: List[Any]) -> List[Dict[str, Any]]:
        """Derive dissent signals from event stream. Refactored: CC 25→6"""
        signals: List[Dict[str, Any]] = []

        for ev in events:
            etv = str(self._get_event_type(ev) or "").lower()
            data = self._get_event_data(ev)
            ts = self._get_timestamp(ev)

            # Process different signal types
            signal = self._process_dissent_event(ev, etv, data, ts)
            if signal:
                signals.append(signal)

        return signals

    def _process_dissent_event(self, ev: Any, etv: str, data: Dict, ts: str) -> Optional[Dict[str, Any]]:
        """Process a single event into a dissent signal. Dispatches to specific handlers. CC=5"""
        # Devils advocate bias
        if etv in (str(ContextEventType.DEVILS_ADVOCATE_BIAS_FOUND.value), "devils_advocate_bias_found"):
            return self._process_da_bias_signal(ev, etv, data, ts)

        # Devils advocate complete
        if etv in (str(ContextEventType.DEVILS_ADVOCATE_ANALYSIS_COMPLETE.value), "devils_advocate_complete"):
            return self._process_da_complete_signal(ev, etv, data, ts)

        # Contradictions
        if etv in (str(ContextEventType.CONTRADICTION_DETECTED.value), str(ContextEventType.CONTRADICTION_AUDIT.value),
                   "contradiction_detected", "contradiction_audit"):
            return self._process_contradiction_signal(ev, etv, data, ts)

        # Senior advisor tensions
        if etv in (str(ContextEventType.SENIOR_ADVISOR_TENSION_IDENTIFIED.value), "senior_advisor_tension_identified"):
            return self._process_tension_signal(ev, etv, data, ts)

        return None

    def _process_da_bias_signal(self, ev: Any, etv: str, data: Dict, ts: str) -> Dict[str, Any]:
        """Process devils advocate bias signal. CC=2"""
        message = data.get("bias_note") or data.get("message") or "Devil's advocate bias found"
        severity = data.get("severity") or "medium"
        return self._build_dissent_signal(ev, etv, ts, kind="da_bias", message=message, severity=severity)

    def _process_da_complete_signal(self, ev: Any, etv: str, data: Dict, ts: str) -> Dict[str, Any]:
        """Process devils advocate completion signal. CC=3"""
        count = data.get("challenges_count") or len(data.get("challenges") or data.get("key_challenges") or [])
        severity = "high" if (count or 0) >= 5 else "medium" if (count or 0) >= 1 else "low"
        message = data.get("summary") or f"Devil's advocate completed with {count} challenges"
        return self._build_dissent_signal(ev, etv, ts, kind="da_complete", message=message, severity=severity)

    def _process_contradiction_signal(self, ev: Any, etv: str, data: Dict, ts: str) -> Dict[str, Any]:
        """Process contradiction signal. CC=3"""
        message = data.get("example_contradiction") or data.get("contradiction") or "Contradiction detected"
        severity = data.get("severity") or ("high" if data.get("contradiction_count", 0) >= 3 else "medium")
        return self._build_dissent_signal(ev, etv, ts, kind="contradiction", message=message, severity=severity)

    def _process_tension_signal(self, ev: Any, etv: str, data: Dict, ts: str) -> Dict[str, Any]:
        """Process senior advisor tension signal. CC=2"""
        message = data.get("tension") or data.get("message") or "Senior advisor tension identified"
        severity = data.get("severity") or "medium"
        return self._build_dissent_signal(ev, etv, ts, kind="tension", message=message, severity=severity)

    def _build_dissent_signal(self, ev: Any, etv: str, ts: str, kind: str, message: str, severity: str) -> Dict[str, Any]:
        """Build dissent signal object. CC=1"""
        return {
            "id": self._generate_signal_id(ev, etv, message, ts),
            "kind": kind,
            "severity": severity,
            "message": message,
            "timestamp": ts,
        }

    def _build_quality_metrics(
        self, states: List[Dict[str, Any]], events: List[Any]
    ) -> Dict[str, Any]:
        """Build quality metrics from states and events."""
        # Sum processing time from states
        total_ms = 0
        try:
            for row in states:
                total_ms += int(row.get("processing_time_ms") or 0)
        except Exception:
            total_ms = 0

        # Count tokens from events
        token_count = 0
        try:
            for ev in events:
                et = self._get_event_type(ev)
                if et == ContextEventType.LLM_PROVIDER_RESPONSE.value:
                    data = self._get_event_data(ev)
                    token_count += int(data.get("tokens_used", 0) or 0)
        except Exception:
            token_count = 0

        return {
            "overall_confidence": 0.85,  # Placeholder
            "cognitive_diversity": 0.75,  # Placeholder
            "evidence_strength": 0.8,
            "execution_time_ms": total_ms,
            "total_tokens": token_count,
        }

    # Helper methods

    def _get_event_type(self, event: Any) -> Optional[str]:
        """Extract event type from event object."""
        if hasattr(event, "event_type"):
            et = event.event_type
            return et.value if hasattr(et, "value") else et
        return event.get("event_type") if isinstance(event, dict) else None

    def _get_event_data(self, event: Any) -> Dict[str, Any]:
        """Extract event data from event object."""
        if hasattr(event, "data"):
            return event.data or {}
        return event.get("data", {}) if isinstance(event, dict) else {}

    def _get_timestamp(self, event: Any) -> str:
        """Extract timestamp from event object."""
        if hasattr(event, "timestamp"):
            t = event.timestamp
            try:
                return t.isoformat() if t else datetime.now().isoformat()
            except Exception:
                return datetime.now().isoformat()
        return datetime.now().isoformat()

    def _get_event_id(self, event: Any) -> str:
        """Extract event ID from event object."""
        if hasattr(event, "event_id"):
            return str(event.event_id)
        return str(UUID.uuid4())

    def _generate_signal_id(
        self, event: Any, event_type: str, message: str, timestamp: str
    ) -> str:
        """Generate unique signal ID."""
        if hasattr(event, "event_id"):
            return str(event.event_id)
        content = f"{event_type}{message or ''}{timestamp}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _build_evidence_item(
        self, item: Dict[str, Any], event: Any
    ) -> Dict[str, Any]:
        """Build evidence item from evidence collection event."""
        evid_id = item.get("fetch_hash") or hashlib.sha256(
            (str(item.get("source_url", "")) + str(item.get("source_quote", ""))).encode("utf-8")
        ).hexdigest()

        snippet = item.get("source_quote") or item.get("claim") or "Evidence summary pending."
        source = (
            item.get("source_title")
            or item.get("source_url")
            or item.get("source_type")
            or "Unattributed source"
        )

        return {
            "id": evid_id,
            "content": snippet,
            "source": source,
            "provenance": self._provenance_from_source_type(item.get("source_type")),
            "confidence": item.get("credibility_weight"),
            "url": item.get("source_url"),
            "timestamp": item.get("fetch_timestamp") or self._get_timestamp(event),
        }

    def _extract_citations(self, event_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract citations from oracle research event."""
        citations = []
        briefing = event_data.get("briefing_memo", {}) or {}

        if isinstance(briefing.get("citations"), list):
            citations = briefing.get("citations")
        elif isinstance(event_data.get("citations"), list):
            citations = event_data.get("citations")

        return citations if isinstance(citations, list) else []

    def _build_citation_evidence(
        self, citation: Dict[str, Any], event: Any
    ) -> Dict[str, Any]:
        """Build evidence item from citation."""
        url = citation.get("source_url") or citation.get("url")
        excerpt = citation.get("excerpt") or ""
        evid_id = hashlib.sha256((str(url or "") + excerpt).encode("utf-8")).hexdigest()

        snippet = excerpt or "Evidence excerpt pending."
        source = citation.get("source_title") or citation.get("title") or url or "Unattributed source"

        return {
            "id": evid_id,
            "content": snippet,
            "source": source,
            "provenance": None,
            "confidence": citation.get("relevance_score") or citation.get("relevance"),
            "url": url,
            "timestamp": citation.get("timestamp") or self._get_timestamp(event),
        }

    def _provenance_from_source_type(self, source_type: Optional[str]) -> Optional[str]:
        """Map source type to provenance category."""
        if not source_type:
            return None
        s = str(source_type).lower()
        if s == "primary" or "gov" in s:
            return "real"
        if s == "secondary" or "wiki" in s:
            return "derived"
        return None
