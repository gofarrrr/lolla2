#!/usr/bin/env python3
"""
Prompt Upgrade Canary Evaluator
- Splits trace snapshots by variant (control vs upgrade) using context events
- Runs binary judges on each cohort and compares failure rates
"""
from typing import Dict, Any, List, Tuple

from src.evaluation.runner.run_binary_judges import run_binary_judges
from src.core.unified_context_stream import ContextEventType


def split_traces_by_variant(trace_snapshots: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    control, upgrade = [], []
    for snap in trace_snapshots:
        variant = _get_variant_from_snapshot(snap)
        if variant == "upgrade":
            upgrade.append(snap)
        else:
            control.append(snap)
    return control, upgrade


def _get_variant_from_snapshot(snap: Dict[str, Any]) -> str:
    cs = snap.get("context_stream", {})
    events = cs.get("events", [])
    for ev in events:
        if ev.get("event_type") == ContextEventType.PROMPT_POLICY_VARIANT_ASSIGNED.value:
            data = ev.get("data", {})
            return data.get("variant", "control")
    return "control"


def evaluate_canary(trace_snapshots: List[Dict[str, Any]]) -> Dict[str, Any]:
    control, upgrade = split_traces_by_variant(trace_snapshots)

    control_rates = run_binary_judges(control) if control else {}
    upgrade_rates = run_binary_judges(upgrade) if upgrade else {}

    # Compute deltas (upgrade - control)
    deltas: Dict[str, float] = {}
    for judge in set(control_rates.keys()) | set(upgrade_rates.keys()):
        c = control_rates.get(judge, 0.0)
        u = upgrade_rates.get(judge, 0.0)
        deltas[judge] = u - c

    return {
        "control_count": len(control),
        "upgrade_count": len(upgrade),
        "control_failure_rates": control_rates,
        "upgrade_failure_rates": upgrade_rates,
        "deltas": deltas,
    }
