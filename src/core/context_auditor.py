"""
ContextAuditor - V6 Context Hygiene and Integrity Checks

Runs before and after each stage to:
- Enforce canonical shapes (StageKey keys, no nested wrappers/aliases)
- Bound context size growth and flag spikes
- Sanitize text (control chars, unsafe substrings)
- Verify referential integrity (question IDs, citations present when claimed)
- Verify flags (recency/synthesis honored)
- Attach/log AEGIS config (min_ratio, genre)
- Compute and log deltas (added/modified/removed keys)
- Validate stage payloads against Pydantic contracts
"""
from __future__ import annotations

import os
import re
import copy
import logging
from typing import Any, Dict, List, Optional, Tuple, Set

from src.core.unified_context_stream import UnifiedContextStream, ContextEventType
from src.core.stage_keys import StageKey, validate_no_legacy_keys
from src.core import pipeline_contracts as pc

logger = logging.getLogger(__name__)


_UNSAFE_SUBSTRINGS = [
    "\nrole:",  # guard against prompt role injection
    "ignore previous instruction",
]

_CONTROL_CHARS_PATTERN = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")


def _estimate_tokens(obj: Any) -> int:
    try:
        s = str(obj)
        return max(1, len(s) // 4)
    except Exception:
        return 1


def _sanitize_text(value: Any) -> Any:
    if isinstance(value, str):
        v = _CONTROL_CHARS_PATTERN.sub(" ", value)
        lower = v.lower()
        for bad in _UNSAFE_SUBSTRINGS:
            if bad in lower:
                # Remove the substring conservatively
                pattern = re.compile(re.escape(bad), re.IGNORECASE)
                v = pattern.sub(" ", v)
        return v
    if isinstance(value, list):
        return [_sanitize_text(x) for x in value]
    if isinstance(value, dict):
        return {k: _sanitize_text(v) for k, v in value.items()}
    return value


def _flatten_wrappers(context: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Flatten nested wrappers like {"problem_structuring": {"problem_structuring": {...}}}.
    Returns (new_context, changed).
    """
    changed = False
    new_ctx = dict(context)

    for sk in StageKey:
        key = sk.value
        if key in new_ctx and isinstance(new_ctx[key], dict):
            inner = new_ctx[key]
            if key in inner and isinstance(inner[key], dict):
                new_ctx[key] = inner[key]
                changed = True
    return new_ctx, changed


def _canonicalize_shapes(context: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Ensure only canonical StageKey keys (metadata keys starting with '_' allowed)."""
    issues: List[str] = []
    try:
        # Will raise if legacy keys are present
        validate_no_legacy_keys(context)
    except Exception as e:
        issues.append(str(e))

    # No enforcement of extra keys other than allowing metadata keys '_' and known flags
    return context, issues


def _diff_keys(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, List[str]]:
    b_keys = set(before.keys())
    a_keys = set(after.keys())
    added = sorted(list(a_keys - b_keys))
    removed = sorted(list(b_keys - a_keys))
    modified: List[str] = []
    for k in sorted(list(b_keys & a_keys)):
        try:
            if str(before.get(k)) != str(after.get(k)):
                modified.append(k)
        except Exception:
            modified.append(k)
    return {"added": added, "removed": removed, "modified": modified}


def _validate_stage_payload(stage_key: StageKey, payload: Dict[str, Any]) -> Optional[str]:
    """Validate stage payload against Pydantic contracts. Returns error string or None."""
    try:
        if stage_key == StageKey.SOCRATIC:
            pc.SocraticOutput.model_validate(payload)
        elif stage_key == StageKey.STRUCTURING:
            pc.ProblemStructuringOutput.model_validate(payload)
        elif stage_key == StageKey.ORACLE:
            pc.BriefingMemo.model_validate(payload)
        elif stage_key == StageKey.SELECTION:
            pc.ConsultantSelectionOutput.model_validate(payload)
        elif stage_key == StageKey.ANALYSIS:
            pc.ParallelAnalysisOutput.model_validate(payload)
        elif stage_key == StageKey.DEVILS_ADVOCATE:
            pc.DevilsAdvocateOutput.model_validate(payload)
        elif stage_key == StageKey.SENIOR_ADVISOR:
            pc.SeniorAdvisorOutput.model_validate(payload)
        return None
    except Exception as e:
        return f"Payload validation failed for {stage_key.value}: {e}"


def _collect_socratic_question_ids(context: Dict[str, Any]) -> Set[str]:
    try:
        soc = context.get(StageKey.SOCRATIC.value) or {}
        questions = soc.get("key_strategic_questions") or []
        ids = set()
        for q in questions:
            if isinstance(q, dict) and q.get("id"):
                ids.add(str(q["id"]))
        return ids
    except Exception:
        return set()


def _check_referential_integrity(context_after: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    # answered_questions ids must exist in socratic questions
    try:
        answered = None
        enh_ctx = context_after.get("enhancement_context")
        if isinstance(enh_ctx, dict):
            answered = enh_ctx.get("answered_questions")
        if answered:
            valid_ids = _collect_socratic_question_ids(context_after)
            missing = [a.get("question_id") for a in answered if a.get("question_id") and str(a.get("question_id")) not in valid_ids]
            if missing:
                issues.append(f"answered_questions refer to missing Socratic IDs: {missing[:5]}")
    except Exception:
        pass

    # Oracle citations present when claimed
    try:
        oracle = context_after.get(StageKey.ORACLE.value)
        if isinstance(oracle, dict):
            qind = str(oracle.get("quality_indicator", "")).upper()
            cits = oracle.get("citations", []) or []
            if qind in {"GREEN", "YELLOW"} and len(cits) == 0:
                issues.append("Oracle quality indicates research but citations array is empty")
    except Exception:
        pass

    return issues


def _check_flags(context_after: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    # Recency/synthesis flags
    try:
        oracle = context_after.get(StageKey.ORACLE.value)
        if isinstance(oracle, dict):
            qa = oracle.get("query_audit") or []
            if isinstance(qa, list) and qa:
                qa0 = qa[0]
                synth_enabled = bool(qa0.get("synthesis_enabled", False))
                recency = qa0.get("recency_filter") or qa0.get("recency")
                # If enhancement flag requested forced synthesis
                forced = bool(context_after.get("force_synthesis")) or bool((context_after.get("enhancement_metadata") or {}).get("force_synthesis"))
                if forced and not synth_enabled:
                    issues.append("forced_synthesis flag present but synthesis_enabled=false in query_audit")
                if recency and recency not in {"hour", "day", "week", "month", "year"}:
                    issues.append(f"Unexpected recency flag: {recency}")
    except Exception:
        pass

    return issues


def _attach_aegis_config(context_after: Dict[str, Any], stage_name: str) -> None:
    try:
        min_ratio = float(os.getenv("AEGIS_MIN_GROUNDING_RATIO", os.getenv("AEGIS_MIN_GROUNDING_RATIO_SA", "0.45")))
    except Exception:
        min_ratio = 0.45
    genre = os.getenv("AEGIS_GROUNDING_GENRE", os.getenv("AEGIS_MIN_GROUNDING_GENRE", "strategic_analysis")).strip().lower()
    context_after.setdefault("_aegis", {})[stage_name] = {
        "min_ratio": min_ratio,
        "genre": genre,
    }


class ContextAuditor:
    def __init__(self, context_stream: UnifiedContextStream):
        self.context_stream = context_stream
        self._last_size_tokens: Optional[int] = None

    def audit_pre(self, stage_name: str, context_before: Dict[str, Any]) -> Dict[str, Any]:
        """Run pre-stage audits and return possibly normalized/sanitized context."""
        ctx = copy.deepcopy(context_before) if isinstance(context_before, dict) else {}

        # Sanitize text
        sanitized = _sanitize_text(ctx)

        # Flatten wrappers
        flattened, changed = _flatten_wrappers(sanitized)

        # Canonicalize shapes and collect shape issues
        canon_ctx, issues = _canonicalize_shapes(flattened)

        # Record size
        size_tokens = _estimate_tokens(canon_ctx)
        self._last_size_tokens = size_tokens

        # Emit QA event for issues
        if issues:
            try:
                self.context_stream.add_event(
                    ContextEventType.QA_SELF_CHECK,
                    {
                        "phase": "pre",
                        "stage": stage_name,
                        "issues": issues[:5],
                    },
                )
            except Exception:
                pass

        # Log wrapper normalization
        if changed:
            try:
                self.context_stream.add_event(
                    ContextEventType.CONTEXT_PRESERVATION_VALIDATED,
                    {
                        "phase": "pre",
                        "stage": stage_name,
                        "normalization": "flattened_nested_wrapper",
                    },
                )
            except Exception:
                pass

        return canon_ctx

    def audit_post(
        self,
        stage_key: StageKey,
        context_before: Dict[str, Any],
        context_after: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run post-stage audits, possibly normalize output, and emit events. Returns normalized context_after."""
        before = copy.deepcopy(context_before) if isinstance(context_before, dict) else {}
        after = copy.deepcopy(context_after) if isinstance(context_after, dict) else {}

        # Sanitize text again on new content
        after = _sanitize_text(after)

        # Flatten wrappers
        after, _ = _flatten_wrappers(after)

        # Compute size spike
        prev_tokens = self._last_size_tokens or _estimate_tokens(before)
        post_tokens = _estimate_tokens(after)
        try:
            if post_tokens > prev_tokens * 1.5:
                self.context_stream.add_event(
                    ContextEventType.QA_SELF_CHECK,
                    {
                        "phase": "post",
                        "stage": stage_key.value,
                        "warning": "context_size_spike",
                        "before_tokens": prev_tokens,
                        "after_tokens": post_tokens,
                        "growth_pct": round(((post_tokens - prev_tokens) / max(prev_tokens, 1)) * 100, 2),
                    },
                )
        except Exception:
            pass

        # Referential integrity checks
        ref_issues = _check_referential_integrity(after)
        flag_issues = _check_flags(after)

        # Attach/log AEGIS config
        _attach_aegis_config(after, stage_key.value)

        # Validate stage payload if present
        validation_error = None
        try:
            if stage_key.value in after and isinstance(after[stage_key.value], dict):
                validation_error = _validate_stage_payload(stage_key, after[stage_key.value])
        except Exception as e:
            validation_error = str(e)

        # Emit QA event for issues
        qa_payload: Dict[str, Any] = {
            "phase": "post",
            "stage": stage_key.value,
            "referential_issues": ref_issues[:5],
            "flag_issues": flag_issues[:5],
        }
        if validation_error:
            qa_payload["validation_error"] = validation_error
        try:
            if ref_issues or flag_issues or validation_error:
                self.context_stream.add_event(ContextEventType.QA_SELF_CHECK, qa_payload)
        except Exception:
            pass

        # Log deltas
        delta = _diff_keys(before, after)
        try:
            self.context_stream.add_event(
                ContextEventType.CONTEXT_PRESERVATION_VALIDATED,
                {
                    "phase": "post",
                    "stage": stage_key.value,
                    "delta": delta,
                },
            )
        except Exception:
            pass

        return after
