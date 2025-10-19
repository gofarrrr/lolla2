"""
Event Extraction Registry
=========================

Robust, extensible event extraction system for reconstructing enriched
reports from UnifiedContextStream events. Replaces ad‑hoc if/elif chains
with a small, composable registry that is easy to extend and test.

Usage:
    from src.api.event_extractors import registry, extract_all_fields
    enriched, provenance = extract_all_fields(events)

Add a new extractor by registering a function:

    @registry.register('quality_validation_complete')
    def extract_quality(events):
        # return list[ExtractionResult] or a single ExtractionResult
        ...

Contract:
    - Events are normalized to dicts:
        { 'event_type': 'lower_snake', 'data': {...}, 'timestamp': iso }
    - Extractors receive a list[dict] for a specific event_type
    - Extractors return one or many ExtractionResult objects
    - The aggregator merges results into a single dict and builds provenance
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from datetime import datetime


@dataclass
class ExtractionResult:
    field_name: str
    value: Any
    data_source: str = "real"  # real | mock | partial
    extracted_from: List[str] = field(default_factory=list)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class EventExtractionRegistry:
    def __init__(self) -> None:
        self._extractors: Dict[str, List[Callable[[List[Dict[str, Any]]], Union[ExtractionResult, List[ExtractionResult], None]]]] = {}

    def register(self, event_type: str):
        """Decorator to register an extractor for a given event type."""

        et = event_type.strip().lower()

        def _decorator(func: Callable[[List[Dict[str, Any]]], Union[ExtractionResult, List[ExtractionResult], None]]):
            self._extractors.setdefault(et, []).append(func)
            return func

        return _decorator

    def extract(self, events_by_type: Dict[str, List[Dict[str, Any]]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Run all extractors and return (enriched_fields, provenance)."""
        results: Dict[str, Any] = {}
        provenance: Dict[str, Any] = {}

        for et, extractor_list in self._extractors.items():
            items = events_by_type.get(et, [])
            if not items:
                continue
            for extractor in extractor_list:
                try:
                    out = extractor(items)
                    if out is None:
                        continue
                    if isinstance(out, list):
                        extracted = out
                    else:
                        extracted = [out]
                    for res in extracted:
                        # Merge arrays when appropriate; otherwise last write wins
                        if res.field_name in results and isinstance(results[res.field_name], list) and isinstance(res.value, list):
                            results[res.field_name] = results[res.field_name] + res.value
                        elif res.field_name in results and isinstance(results[res.field_name], dict) and isinstance(res.value, dict):
                            # Shallow update for dict payloads
                            merged = results[res.field_name].copy()
                            merged.update(res.value)
                            results[res.field_name] = merged
                        else:
                            results[res.field_name] = res.value

                        provenance[res.field_name] = {
                            "data_source": res.data_source,
                            "extracted_from": res.extracted_from,
                            "confidence": res.confidence,
                            "metadata": res.metadata,
                        }
                except Exception:
                    # Extractors must never break enrichment flow
                    continue

        return results, provenance


def normalize_events(raw_events: Iterable[Any]) -> List[Dict[str, Any]]:
    """Normalize mixed object/dict events to a unified dict shape."""
    norm: List[Dict[str, Any]] = []
    for ev in raw_events:
        try:
            if isinstance(ev, dict):
                et = str(ev.get("event_type", ""))
                data = ev.get("data", {}) or {}
                ts = ev.get("timestamp") or ev.get("ts")
            else:
                et = getattr(getattr(ev, "event_type", None), "value", None) or str(getattr(ev, "event_type", ""))
                data = getattr(ev, "data", {}) or {}
                ts_obj = getattr(ev, "timestamp", None)
                ts = ts_obj.isoformat() if hasattr(ts_obj, "isoformat") else None

            et = (et or "").strip().lower()
            if not et:
                continue
            norm.append({"event_type": et, "data": data, "timestamp": ts})
        except Exception:
            continue
    return norm


def index_by_type(events: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    idx: Dict[str, List[Dict[str, Any]]] = {}
    for ev in events:
        et = ev.get("event_type")
        if not et:
            continue
        idx.setdefault(et, []).append(ev)
    return idx


# Singleton registry with core extractors
registry = EventExtractionRegistry()


@registry.register("consultant_analysis_complete")
def extract_parallel_analysis(events: List[Dict[str, Any]]) -> Optional[ExtractionResult]:
    # Use the last occurrence as authoritative
    if not events:
        return None
    last = events[-1]
    data = last.get("data", {})
    contract = data.get("contract_data", {}) if isinstance(data, dict) else {}
    analyses = contract.get("consultant_analyses", [])
    parallel: Dict[str, Any] = {}
    if analyses:
        parallel["consultant_analyses"] = analyses
    if "orthogonality_index" in contract:
        parallel["orthogonality_index"] = contract.get("orthogonality_index", 0.0)
    if not parallel:
        return None
    return ExtractionResult(
        field_name="parallel_analysis",
        value=parallel,
        extracted_from=["consultant_analysis_complete"],
        confidence=1.0,
    )


@registry.register("stage_performance_profile_recorded")
def extract_stage_profiles(events: List[Dict[str, Any]]) -> List[ExtractionResult]:
    profiles: List[Dict[str, Any]] = []
    total_time = 0
    total_tokens = 0
    for ev in events:
        d = ev.get("data", {})
        name = d.get("stage_name", "")
        t = int(d.get("wall_time_ms", 0) or 0)
        tok = int(d.get("token_count", 0) or 0)
        profiles.append({"stage_name": name, "wall_time_ms": t, "token_count": tok})
        total_time += t
        total_tokens += tok
    out = [
        ExtractionResult("stage_profiles", profiles, extracted_from=["stage_performance_profile_recorded"]),
        ExtractionResult("total_processing_time_ms", total_time, extracted_from=["stage_performance_profile_recorded"]),
        ExtractionResult("total_time", total_time, extracted_from=["stage_performance_profile_recorded"]),
        ExtractionResult("total_tokens", total_tokens, extracted_from=["stage_performance_profile_recorded"]),
    ]
    return out


@registry.register("evidence_collection_complete")
def extract_evidence(events: List[Dict[str, Any]]) -> Optional[ExtractionResult]:
    # Prefer last event
    data = events[-1].get("data", {})
    evidence = data.get("evidence", [])
    if not evidence:
        return None
    return ExtractionResult(
        field_name="evidence",
        value=evidence,
        extracted_from=["evidence_collection_complete"],
        confidence=1.0,
    )


@registry.register("socratic_questions_generated")
def extract_socratic_questions(events: List[Dict[str, Any]]) -> Optional[ExtractionResult]:
    data = events[-1].get("data", {})
    contract = data.get("contract_data", {})
    if not contract:
        return None
    return ExtractionResult(
        field_name="socratic_questions",
        value=contract,
        extracted_from=["socratic_questions_generated"],
    )


@registry.register("model_selection_justification")
def extract_selection_methodology(events: List[Dict[str, Any]]) -> Optional[ExtractionResult]:
    items: List[Dict[str, Any]] = []
    for ev in events:
        d = ev.get("data", {})
        items.append(
            {
                "selected_consultants": d.get("selected_consultants", []),
                "chosen_nway_patterns": d.get("chosen_nway_patterns", []),
                "nway_selection_rationale": d.get("nway_selection_rationale", ""),
            }
        )
    if not items:
        return None
    return ExtractionResult(
        field_name="consultant_selection_methodology",
        value=items,
        extracted_from=["model_selection_justification"],
    )


@registry.register("research_result")
def extract_research_results(events: List[Dict[str, Any]]) -> Optional[ExtractionResult]:
    results: List[Dict[str, Any]] = []
    for ev in events:
        d = ev.get("data", {})
        results.append(
            {
                "consultant_id": d.get("consultant_id", ""),
                "query": d.get("query", ""),
                "sources_count": int(d.get("sources_count", 0) or 0),
                "cost_usd": float(d.get("cost_usd", 0) or 0.0),
                "success": bool(d.get("success", True)),
            }
        )
    if not results:
        return None
    return ExtractionResult(
        field_name="research_grounding",
        value=results,
        extracted_from=["research_result"],
    )


@registry.register("devils_advocate_complete")
def extract_da(events: List[Dict[str, Any]]) -> List[ExtractionResult]:
    d = events[-1].get("data", {})
    out: List[ExtractionResult] = []
    if "da_transcript" in d:
        out.append(
            ExtractionResult(
                field_name="da_transcript",
                value=d.get("da_transcript"),
                extracted_from=["devils_advocate_complete"],
            )
        )
    if "assumption_diff" in d:
        out.append(
            ExtractionResult(
                field_name="assumption_diff",
                value=d.get("assumption_diff"),
                extracted_from=["devils_advocate_complete"],
            )
        )
    return out


@registry.register("probability_assessment_reported")
def extract_confidence_assessment(events: List[Dict[str, Any]]) -> Optional[ExtractionResult]:
    d = events[-1].get("data", {})
    ca = d.get("confidence_assessment")
    if not ca:
        return None
    return ExtractionResult(
        field_name="confidence_assessment",
        value=ca,
        extracted_from=["probability_assessment_reported"],
    )


@registry.register("orthogonality_index_computed")
def extract_orthogonality(events: List[Dict[str, Any]]) -> Optional[ExtractionResult]:
    d = events[-1].get("data", {})
    if "orthogonality_index" not in d:
        return None
    return ExtractionResult(
        field_name="orthogonality_index",
        value=float(d.get("orthogonality_index", 0.0) or 0.0),
        extracted_from=["orthogonality_index_computed"],
    )


@registry.register("consultant_memo_produced")
def extract_consultant_memos(events: List[Dict[str, Any]]) -> Optional[ExtractionResult]:
    memos: List[Dict[str, Any]] = []
    for ev in events:
        d = ev.get("data", {})
        memo = {
            "consultant_id": d.get("consultant_id"),
            "consultant_type": d.get("consultant_type"),
            "specialization": d.get("specialization"),
            "analysis_memo": d.get("analysis_memo"),
            "analysis_length": d.get("analysis_length"),
        }
        # Only keep memos with text
        if memo.get("analysis_memo"):
            memos.append(memo)
    if not memos:
        return None
    return ExtractionResult(
        field_name="consultant_memos",
        value=memos,
        extracted_from=["consultant_memo_produced"],
    )


@registry.register("senior_advisor_complete")
def extract_senior_advisor_metrics(events: List[Dict[str, Any]]) -> Optional[ExtractionResult]:
    """Summarize Senior Advisor synthesis metrics from the last completion event."""
    if not events:
        return None
    d = events[-1].get("data", {})
    metrics = {
        "recommendations_count": d.get("recommendations_count"),
        "synthesis_quality": d.get("synthesis_quality"),
        "overall_confidence": d.get("overall_confidence"),
        "processing_time_ms": d.get("processing_time_ms"),
        "two_brain_completed": d.get("two_brain_completed", False),
    }
    return ExtractionResult(
        field_name="senior_advisor_metrics",
        value=metrics,
        extracted_from=["senior_advisor_complete"],
    )


def extract_all_fields(raw_events: Iterable[Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Helper to normalize, index and extract in one call."""
    norm = normalize_events(raw_events)
    idx = index_by_type(norm)
    return registry.extract(idx)
