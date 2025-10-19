"""
Event schema and normalization utilities.

Goals:
- Prevent event contract drift by centralizing allowed types and required fields
- Provide a normalization function that adds a stable `timestamp` alias and
  exposes a small set of safe, non-sensitive fields for Forensics
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
from datetime import datetime

try:
    from src.core.unified_context_stream import ContextEventType
except Exception:
    # Fallback for type checking in isolated environments
    class _Dummy:
        value: str

    class ContextEventType:  # type: ignore
        LLM_PROVIDER_REQUEST = _Dummy()
        LLM_PROVIDER_RESPONSE = _Dummy()
        TOOL_CALL_START = _Dummy()
        TOOL_CALL_COMPLETE = _Dummy()
        TOOL_EXECUTION = _Dummy()
        TOOL_DECISION = _Dummy()
        RESEARCH_PROVIDER_REQUEST = _Dummy()
        RESEARCH_PROVIDER_RESPONSE = _Dummy()
        RESEARCH_PROVIDER_FALLBACK = _Dummy()
        RESEARCH_QUERY = _Dummy()
        RESEARCH_RESULT = _Dummy()
        SOURCE_VALIDATED = _Dummy()


# Groupings for readability (not exhaustive – we focus on high‑value categories)
EVENT_CATEGORIES: Dict[str, Tuple[str, ...]] = {
    "llm": (
        "llm_provider_request",
        "llm_provider_response",
    ),
    "tool": (
        "tool_call_start",
        "tool_call_complete",
        "tool_execution",
        "tool_decision",
    ),
    "research": (
        "research_provider_request",
        "research_provider_response",
        "research_provider_fallback",
        "research_query",
        "research_result",
        "source_validated",
    ),
}


# Minimal required fields per event type for sanity (best‑effort)
REQUIRED_FIELDS: Dict[str, Tuple[str, ...]] = {
    "llm_provider_request": ("model", "provider"),
    "llm_provider_response": ("model", "provider"),  # latency/tokens optional
    "tool_execution": ("tool_name",),
    "tool_call_start": ("tool_name",),
    "tool_call_complete": ("tool_name",),
    "tool_decision": ("tool_name",),
    "research_provider_request": ("provider",),
    "research_provider_response": ("provider",),
    "research_provider_fallback": ("original_provider", "fallback_provider"),
}


def _iso(ts: Any) -> str:
    if isinstance(ts, datetime):
        return ts.isoformat()
    try:
        # Already ISO string
        return str(ts)
    except Exception:
        return ""


def normalize_event(event: Any) -> Dict[str, Any]:
    """
    Produce a sanitized, UI‑friendly shape with a stable timestamp alias and
    a small set of safe fields copied from event.data.

    The caller is responsible for any deeper PII/redaction rules.
    """
    et_val = getattr(getattr(event, "event_type", None), "value", None) or str(
        getattr(event, "event_type", "")
    )
    data = getattr(event, "data", {}) or {}

    out: Dict[str, Any] = {
        "event_type": et_val,
        # Provide both `timestamp` and `ts` for compatibility
        "timestamp": _iso(getattr(event, "timestamp", None)),
        "ts": getattr(event, "timestamp", None),
    }

    # Common convenience fields (safe)
    stage = data.get("stage") or data.get("stage_name")
    if stage:
        out["stage"] = stage

    # LLM
    if et_val in EVENT_CATEGORIES["llm"]:
        for k in ("model", "provider", "latency_ms", "tokens", "tokens_used"):
            if k in data:
                out[k] = data[k]
        # Alias common field names
        if "latency_ms" not in out and "response_time_ms" in data:
            out["latency_ms"] = data["response_time_ms"]

    # Tools
    if et_val in EVENT_CATEGORIES["tool"]:
        for k in (
            "tool_name",
            "latency_ms",
            "inputs_fingerprint",
            "result_fingerprint",
            "selection_reasoning",
            "alternatives_considered",
        ):
            if k in data:
                out[k] = data[k]
        # Alias common field names
        if "tool_name" not in out and "tool" in data:
            out["tool_name"] = data["tool"]

    # Research
    if et_val in EVENT_CATEGORIES["research"]:
        for k in (
            "provider",
            "query",
            "citations_count",
            "result_preview",
            "latency_ms",
            "original_provider",
            "fallback_provider",
            "failure_reason",
        ):
            if k in data:
                out[k] = data[k]
        # Alias common field names used by emitters
        if "latency_ms" not in out and "processing_time_ms" in data:
            out["latency_ms"] = data["processing_time_ms"]
        if "citations_count" not in out and "sources_count" in data:
            out["citations_count"] = data["sources_count"]
        if "original_provider" not in out and "failed_provider" in data:
            out["original_provider"] = data["failed_provider"]
        if "fallback_provider" not in out and "next_provider" in data:
            out["fallback_provider"] = data["next_provider"]
        if "failure_reason" not in out and ("fallback_reason" in data or "error" in data):
            out["failure_reason"] = data.get("fallback_reason") or data.get("error")

    return out


def validate_event_schema(event: Any) -> List[str]:
    """Return a list of schema errors for the event (empty if valid enough)."""
    et_val = getattr(getattr(event, "event_type", None), "value", None) or str(
        getattr(event, "event_type", "")
    )
    data = getattr(event, "data", {}) or {}

    errors: List[str] = []

    # Allowlist validation — verify event type exists in ContextEventType
    try:
        valid_types = {e.value for e in ContextEventType}  # type: ignore
        if et_val and et_val not in valid_types:
            errors.append(
                f"Unknown event type: '{et_val}' is not in ContextEventType"
            )
    except Exception:
        # Fail-open in environments where ContextEventType import is stubbed
        pass

    required = REQUIRED_FIELDS.get(et_val)
    if not required:
        # No specific requirements — treat as OK
        return errors

    for f in required:
        if f not in data:
            errors.append(f"Missing required field '{f}' for event '{et_val}'")

    return errors
