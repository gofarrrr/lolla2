"""
Engagements API - Shared Helper Functions
==========================================

Utility functions for engagement data extraction, report generation,
and context stream event processing.

Operation Bedrock: Task 10.0 - API Decomposition
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from uuid import UUID, uuid4
from datetime import datetime

from fastapi import Request
from pydantic import BaseModel

from src.api.event_extractors import extract_all_fields

logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Models for Helper Functions
# ============================================================================

class TraceCitation(BaseModel):
    """Citation from research providers"""
    url: str
    title: str | None = None
    domain: str | None = None


class TraceSummary(BaseModel):
    """PII-safe trace summary"""
    trace_id: str
    event_count: int
    stages_detected: list[str]
    oracle_briefing: dict | None = None
    stage0_enrichment: dict | None = None
    recent_events: list[dict]
    citations: list[TraceCitation] = []
    devils_advocate: list[dict] = []
    # Operation Glass Box UI fields
    orthogonality_index: float | None = None
    stage_profiles: list[dict] = []
    da_transcript: str | None = None
    assumption_diff: list[dict] = []
    evidence: list[dict] = []


# ============================================================================
# Context Stream Loading
# ============================================================================

def load_context_stream_events(trace_id: str) -> Optional[List[Dict[str, Any]]]:
    """
    Load context_stream events from persistence (JSONL file or Supabase).
    Returns list of event dicts, or None if trace_id not found.

    NOTE: Due to global singleton context_stream, the top-level trace_id may differ
    from the engagement trace_id. We search both the top-level field AND inside the record.
    """
    # Try FileAdapter first (local_context_stream.jsonl)
    jsonl_path = os.getenv("CONTEXT_STREAM_JSONL", "local_context_stream.jsonl")
    try:
        if os.path.exists(jsonl_path):
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        # BUGFIX: Check both top-level trace_id AND if trace appears anywhere in record
                        # This handles the case where global singleton context_stream has a different trace_id
                        if record.get("trace_id") == trace_id or trace_id in json.dumps(record):
                            context_stream = record.get("context_stream", {})
                            events = context_stream.get("events", [])
                            logger.info(f"✅ Loaded {len(events)} events from {jsonl_path} for trace {trace_id}")
                            return events
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        logger.warning(f"⚠️ Failed to load from {jsonl_path}: {e}")

    # TODO: Add Supabase fallback here if needed
    logger.warning(f"⚠️ No persisted context_stream found for trace {trace_id}")
    return None


# ============================================================================
# Final Output Extraction
# ============================================================================

def extract_final_output(events: list) -> Dict[str, Any]:
    """
    Extract enriched report fields from context stream events.

    Delegates to EventExtractionService for business logic.
    Complexity: CC = 1
    """
    from src.services.engagement import EventExtractionService
    service = EventExtractionService()
    return service.extract_final_output(events)


# ============================================================================
# Markdown Report Generation
# ============================================================================

def generate_markdown_report(final_output: Dict[str, Any]) -> str:
    """
    Generate markdown content from the final output.

    Delegates to ReportFormattingService for business logic.
    Complexity: CC = 1
    """
    from src.services.engagement import ReportFormattingService
    service = ReportFormattingService()
    return service.generate_markdown(final_output)


# ============================================================================
# Event Data Extraction
# ============================================================================

def extract_consultant_selection_data(events: list) -> Dict[str, Any]:
    """
    Extract consultant selection data from UnifiedContextStream events.

    Delegates to EventExtractionService for business logic.
    Complexity: CC = 1
    """
    from src.services.engagement import EventExtractionService
    service = EventExtractionService()
    return service.extract_consultant_selection_data(events)


def extract_human_interactions(events: list, engagement_state: dict) -> Dict[str, Any]:
    """
    Extract human interaction data from UnifiedContextStream events

    Returns:
        {
            "answered_questions": [...],
            "hitl_interactions": [...],
            "query_enhancements": [...]
        }
    """
    human_interactions = {
        "answered_questions": [],
        "hitl_interactions": [],
        "query_enhancements": []
    }

    # Get answered questions from engagement state (passed from start_engagement)
    answered_questions = engagement_state.get("answered_questions", [])
    if answered_questions:
        for q in answered_questions:
            human_interactions["answered_questions"].append({
                "question_id": q.get("question_id"),
                "question_text": q.get("question_text"),
                "answer": q.get("answer"),
                "timestamp": q.get("timestamp", datetime.now().isoformat()),
                "impact": {
                    "recommendations_affected": [],  # TODO: Track via events
                    "assumptions_updated": [],
                    "confidence_change": 0.0  # TODO: Calculate from events
                }
            })

    # Extract HITL interactions from events
    for event in events:
        # Handle event_type as enum or string
        if hasattr(event, 'event_type'):
            et = event.event_type
            event_type = et.value if hasattr(et, 'value') else et
        else:
            event_type = event.get('event_type')

        event_data = event.data if hasattr(event, 'data') else event.get('data', {})

        if event_type == "hitl_request":
            human_interactions["hitl_interactions"].append({
                "interaction_id": event_data.get("interaction_id"),
                "timestamp": event.timestamp.isoformat() if hasattr(event, 'timestamp') else datetime.now().isoformat(),
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
                "timestamp": event.timestamp.isoformat() if hasattr(event, 'timestamp') else datetime.now().isoformat()
            })

    return human_interactions


def extract_research_provider_events(events: list) -> list:
    """
    Extract research provider transparency data from UnifiedContextStream events

    Returns list of research provider events showing Oracle/Perplexity calls with citations

    OPERATION CRYSTAL PALACE: Enhanced to include detailed citation URLs
    """
    research_events = []
    detailed_citations = []

    for event in events:
        # Handle event_type as enum or string
        if hasattr(event, 'event_type'):
            et = event.event_type
            event_type = et.value if hasattr(et, 'value') else et
        else:
            event_type = event.get('event_type')

        event_data = event.data if hasattr(event, 'data') else event.get('data', {})

        # Extract research provider events
        if event_type in ["research_provider_request", "research_provider_response", "research_provider_fallback"]:
            research_events.append({
                "event_id": event.event_id if hasattr(event, 'event_id') else str(uuid4()),
                "event_type": event_type,
                "timestamp": event.timestamp.isoformat() if hasattr(event, 'timestamp') else datetime.now().isoformat(),
                "provider": event_data.get("provider"),
                "query": event_data.get("query", ""),
                "latency_ms": event_data.get("latency_ms"),
                "citations_count": event_data.get("citations_count", 0),
                "result_preview": event_data.get("result_preview", ""),
                "confidence": event_data.get("confidence", 0.0),
                "failure_reason": event_data.get("failure_reason"),  # For fallback events
                "fallback_provider": event_data.get("fallback_provider")  # For fallback events
            })

        # OPERATION CRYSTAL PALACE: Extract detailed citations from Oracle briefing memo
        if event_type == "oracle_research_complete":
            briefing_memo = event_data.get("briefing_memo", {})
            citations = briefing_memo.get("citations", [])

            for citation in citations:
                detailed_citations.append({
                    "source_url": citation.get("source_url", ""),
                    "source_title": citation.get("source_title", ""),
                    "excerpt": citation.get("excerpt", "")[:200],  # Limit excerpt length
                    "relevance_score": citation.get("relevance_score", 0.0),
                    "timestamp": citation.get("timestamp", "")
                })

    # Add detailed citations to the first research response event
    if detailed_citations and research_events:
        for event in research_events:
            if event["event_type"] == "research_provider_response":
                event["detailed_citations"] = detailed_citations
                break

    return research_events


def extract_mece_framework(events: list) -> dict:
    """
    Extract MECE framework and problem structuring data from UnifiedContextStream events

    Returns MECE framework with dimensions, assumptions, and constraints

    OPERATION CRYSTAL PALACE: Stage 2 transparency
    """
    mece_data = {
        "mece_framework": [],
        "core_assumptions": [],
        "critical_constraints": [],
        "success_criteria": []
    }

    for event in events:
        event_type = event.get("event_type", "")
        event_data = event.get("data", {})

        # Extract from problem_structuring_complete event
        if event_type == "problem_structuring_complete":
            problem_output = event_data.get("problem_structuring_output", {})

            # MECE framework components
            mece_framework = problem_output.get("mece_framework", [])
            if mece_framework:
                for component in mece_framework:
                    mece_data["mece_framework"].append({
                        "dimension": component.get("dimension", ""),
                        "key_considerations": component.get("key_considerations", []),
                        "priority_level": component.get("priority_level", 2)
                    })

            # Core assumptions
            core_assumptions = problem_output.get("core_assumptions", [])
            if core_assumptions:
                mece_data["core_assumptions"].extend(core_assumptions)

            # Critical constraints
            critical_constraints = problem_output.get("critical_constraints", [])
            if critical_constraints:
                mece_data["critical_constraints"].extend(critical_constraints)

            # Success criteria
            success_criteria = problem_output.get("success_criteria", [])
            if success_criteria:
                mece_data["success_criteria"].extend(success_criteria)

    return mece_data


# ============================================================================
# Cache Management
# ============================================================================

def _get_cache(request: Request) -> Dict[str, Any]:
    """Get or create report cache on app state"""
    if not hasattr(request.app.state, "report_cache"):
        request.app.state.report_cache = {}
    return request.app.state.report_cache
