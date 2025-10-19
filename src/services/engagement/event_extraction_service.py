"""
Event Extraction Service - Business Logic Layer
================================================

Extracts event processing logic from engagements.py route handlers.

This service handles:
- Extracting final output from context stream events
- Extracting consultant selection data
- Extracting human interactions
- Extracting research provider events

Target Complexity: CC ≤ 10 per method
Reduces CC=30 hotspot to manageable methods
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from uuid import uuid4
from datetime import datetime

from src.api.event_extractors import extract_all_fields

logger = logging.getLogger(__name__)


class EventExtractionService:
    """
    Service for extracting structured data from UnifiedContextStream events.

    Responsibilities:
    - Extract final report output from event streams
    - Extract consultant selection data and methodology
    - Extract human interaction data (HITL, query enhancements)
    - Extract research provider transparency data
    """

    def extract_final_output(self, events: List) -> Dict[str, Any]:
        """
        Extract enriched report fields from context stream events.

        Complexity: CC ≤ 3 (delegates to helper methods)

        Args:
            events: List of UnifiedContextStream events

        Returns:
            Dictionary with extracted report data
        """
        logger.info(f"🔍 EXTRACT_FINAL_OUTPUT(REGISTRY): Called with {len(events)} events")

        # Run registry extraction
        enriched, provenance = extract_all_fields(events)

        # Build result with minimal skeleton
        result = self._build_base_result(provenance)

        # Merge enriched fields
        result.update(enriched)

        # Derive additional fields
        result = self._derive_totals_from_stage_profiles(result)
        result = self._derive_orthogonality_index(result)
        result = self._synthesize_strategic_recommendations(result)
        result = self._synthesize_analysis_confidence(result)

        return result

    def _build_base_result(self, provenance: Dict[str, Any]) -> Dict[str, Any]:
        """Build minimal base result skeleton. CC = 1"""
        return {
            "executive_summary": "Strategic analysis completed. Transparency data attached.",
            "pipeline_efficiency_score": 0.0,
            "generated_at": datetime.now().isoformat(),
            "_extraction_provenance": provenance,
        }

    def _derive_totals_from_stage_profiles(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Derive total time and tokens from stage profiles. CC = 3"""
        if "total_processing_time_ms" in result:
            return result

        stage_profiles = result.get("stage_profiles")
        if not isinstance(stage_profiles, list):
            return result

        total_time_ms = sum(int(s.get("wall_time_ms", 0) or 0) for s in stage_profiles)
        total_tokens = sum(int(s.get("token_count", 0) or 0) for s in stage_profiles)

        result["total_processing_time_ms"] = total_time_ms
        result["total_time"] = total_time_ms
        result["total_tokens"] = total_tokens

        return result

    def _derive_orthogonality_index(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Derive orthogonality index from parallel_analysis. CC = 4"""
        if "orthogonality_index" in result:
            return result

        parallel_analysis = result.get("parallel_analysis")
        if not isinstance(parallel_analysis, dict):
            return result

        oi = parallel_analysis.get("orthogonality_index")
        if oi is not None:
            result["orthogonality_index"] = oi

        return result

    def _synthesize_strategic_recommendations(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize strategic recommendations from consultant analyses. CC = 5"""
        if "strategic_recommendations" in result:
            return result

        pa = result.get("parallel_analysis")
        if not isinstance(pa, dict):
            return result

        analyses = pa.get("consultant_analyses") or []
        recs = self._collect_unique_recommendations(analyses)

        if recs:
            result["strategic_recommendations"] = recs
            result.setdefault("_extraction_provenance", {})["strategic_recommendations"] = {
                "data_source": "derived",
                "extracted_from": ["consultant_analysis_complete"],
                "confidence": 0.8,
                "metadata": {"derived_from": "parallel_analysis.consultant_analyses.recommendations"},
            }

        return result

    def _collect_unique_recommendations(self, analyses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Collect unique recommendations from analyses. CC = 6"""
        recs: List[Dict[str, Any]] = []
        seen: set[str] = set()

        for a in analyses:
            for r in (a.get("recommendations") or []):
                if not isinstance(r, str):
                    continue

                key = r.strip()
                if key and key.lower() not in seen:
                    seen.add(key.lower())
                    recs.append({
                        "recommendation": key,
                        "priority": "important",
                        "rationale": None,
                        "implementation_complexity": "medium",
                    })

                if len(recs) >= 8:
                    break

            if len(recs) >= 8:
                break

        return recs

    def _synthesize_analysis_confidence(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize analysis_confidence from confidence_assessment. CC = 6"""
        if "analysis_confidence" in result:
            return result

        ca = result.get("confidence_assessment")
        if not isinstance(ca, dict):
            return result

        try:
            score = float(ca.get("probability_of_success", 0.0))
            band = float(ca.get("confidence_band", 0.0))
        except Exception:
            score = float(ca.get("probability_of_success", 0.0) or 0.0)
            band = float(ca.get("confidence_band", 0.0) or 0.0)

        label = "High" if score >= 0.75 else ("Medium" if score >= 0.5 else "Low")

        result["analysis_confidence"] = {
            "confidence_score": round(score, 2),
            "confidence_level": label,
            "confidence_band": band,
        }

        result.setdefault("_extraction_provenance", {})["analysis_confidence"] = {
            "data_source": "derived",
            "extracted_from": ["probability_assessment_reported"],
            "confidence": 0.9,
        }

        return result

    def extract_consultant_selection_data(self, events: List) -> Dict[str, Any]:
        """
        Extract consultant selection data from UnifiedContextStream events.

        Complexity: CC ≤ 7

        Args:
            events: List of UnifiedContextStream events

        Returns:
            Dictionary with consultant selection data
        """
        consultant_data = {
            "selected_consultants": [],
            "methodology": None,
            "team_size": 0
        }

        try:
            # Try new format first
            found = self._extract_consultant_selection_complete(events, consultant_data)

            # Fallback to old format
            if not found:
                self._extract_contextual_consultant_selection(events, consultant_data)

        except Exception as e:
            logger.warning(f"⚠️ Failed to extract consultant selection data: {e}")

        return consultant_data

    def _extract_consultant_selection_complete(
        self,
        events: List,
        consultant_data: Dict[str, Any]
    ) -> bool:
        """Extract CONSULTANT_SELECTION_COMPLETE events. CC = 6"""
        for event in events:
            if not hasattr(event, 'event_type') or not hasattr(event, 'data'):
                continue

            event_type = event.event_type.value if hasattr(event.event_type, 'value') else str(event.event_type)

            if event_type != "CONSULTANT_SELECTION_COMPLETE":
                continue

            event_data = event.data or {}

            # Extract selected consultants with rationales
            selected_consultants = event_data.get("selected_consultants", [])
            for consultant in selected_consultants:
                if isinstance(consultant, dict):
                    consultant_data["selected_consultants"].append({
                        "consultant_id": consultant.get("consultant_id", "unknown"),
                        "consultant_type": consultant.get("consultant_type", "unknown"),
                        "specialization": consultant.get("specialization", ""),
                        "predicted_effectiveness": consultant.get("predicted_effectiveness", 0.0),
                        "selection_rationale": consultant.get("selection_rationale", ""),
                        "assigned_dimensions": consultant.get("assigned_dimensions", [])
                    })

            # Extract methodology and metadata
            consultant_data["methodology"] = event_data.get(
                "selection_methodology",
                "YAML-enhanced cognitive profile scoring with 5-factor weighting"
            )
            consultant_data["team_size"] = event_data.get("team_size", len(selected_consultants))

            return True

        return False

    def _extract_contextual_consultant_selection(
        self,
        events: List,
        consultant_data: Dict[str, Any]
    ) -> None:
        """Extract CONTEXTUAL_CONSULTANT_SELECTION_V1 events. CC = 6"""
        for event in events:
            if not hasattr(event, 'event_type') or not hasattr(event, 'data'):
                continue

            event_type = event.event_type.value if hasattr(event.event_type, 'value') else str(event.event_type)

            if event_type != "CONTEXTUAL_CONSULTANT_SELECTION_V1":
                continue

            event_data = event.data or {}
            selected_team = event_data.get("selected_team", [])

            for consultant in selected_team:
                if isinstance(consultant, dict):
                    consultant_data["selected_consultants"].append({
                        "consultant_id": consultant.get("consultant_id", "unknown"),
                        "consultant_type": consultant.get("consultant_type", "unknown"),
                        "specialization": consultant.get("specialization", ""),
                        "predicted_effectiveness": consultant.get("predicted_effectiveness", 0.0),
                        "selection_rationale": consultant.get(
                            "selection_rationale",
                            f"Selected for {consultant.get('specialization', 'expertise')}"
                        ),
                        "assigned_dimensions": consultant.get("assigned_dimensions", [])
                    })

            consultant_data["methodology"] = "Smart GM v2.0 with synergy analysis"
            consultant_data["team_size"] = len(selected_team)
            break

    def extract_human_interactions(
        self,
        events: List,
        engagement_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract human interaction data from UnifiedContextStream events.

        Complexity: CC ≤ 6

        Args:
            events: List of UnifiedContextStream events
            engagement_state: Engagement state dict

        Returns:
            Dictionary with human interaction data
        """
        human_interactions = {
            "answered_questions": [],
            "hitl_interactions": [],
            "query_enhancements": []
        }

        # Get answered questions from engagement state
        answered_questions = engagement_state.get("answered_questions", [])
        if answered_questions:
            for q in answered_questions:
                human_interactions["answered_questions"].append({
                    "question_id": q.get("question_id"),
                    "question_text": q.get("question_text"),
                    "answer": q.get("answer"),
                    "timestamp": q.get("timestamp", datetime.now().isoformat()),
                    "impact": {
                        "recommendations_affected": [],
                        "assumptions_updated": [],
                        "confidence_change": 0.0
                    }
                })

        # Extract HITL interactions and query enhancements from events
        for event in events:
            event_type, event_data = self._get_event_type_and_data(event)

            if event_type == "hitl_request":
                human_interactions["hitl_interactions"].append({
                    "interaction_id": event_data.get("interaction_id"),
                    "timestamp": self._get_event_timestamp(event),
                    "decision_point": event_data.get("decision_point"),
                    "request": event_data.get("request"),
                    "response": event_data.get("response", ""),
                    "impact": event_data.get("impact", {})
                })

            if event_type == "query_enhanced_from_clarification":
                human_interactions["query_enhancements"].append({
                    "original_query": event_data.get("original_query"),
                    "enhanced_query": event_data.get("enhanced_query"),
                    "enhancement_source": "clarification_answers",
                    "timestamp": self._get_event_timestamp(event)
                })

        return human_interactions

    def extract_research_provider_events(self, events: List) -> List[Dict[str, Any]]:
        """
        Extract research provider transparency data from events.

        Complexity: CC ≤ 4

        Args:
            events: List of UnifiedContextStream events

        Returns:
            List of research provider events
        """
        research_events = []

        for event in events:
            event_type, event_data = self._get_event_type_and_data(event)

            if event_type in ["research_provider_request", "research_provider_response", "research_provider_fallback"]:
                research_events.append({
                    "event_id": event.event_id if hasattr(event, 'event_id') else str(uuid4()),
                    "event_type": event_type,
                    "timestamp": self._get_event_timestamp(event),
                    "provider": event_data.get("provider"),
                    "query": event_data.get("query", ""),
                    "latency_ms": event_data.get("latency_ms"),
                    "citations_count": event_data.get("citations_count", 0),
                    "result_preview": event_data.get("result_preview", ""),
                    "confidence": event_data.get("confidence", 0.0),
                    "failure_reason": event_data.get("failure_reason"),
                    "fallback_provider": event_data.get("fallback_provider")
                })

        return research_events

    def _get_event_type_and_data(self, event) -> Tuple[Optional[str], Dict[str, Any]]:
        """Extract event type and data from event object. CC = 4"""
        if hasattr(event, 'event_type'):
            et = event.event_type
            event_type = et.value if hasattr(et, 'value') else et
        else:
            event_type = event.get('event_type')

        event_data = event.data if hasattr(event, 'data') else event.get('data', {})

        return event_type, event_data

    def _get_event_timestamp(self, event) -> str:
        """Extract timestamp from event object. CC = 2"""
        if hasattr(event, 'timestamp'):
            return event.timestamp.isoformat()
        return datetime.now().isoformat()
