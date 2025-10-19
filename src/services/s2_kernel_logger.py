#!/usr/bin/env python3
"""
S2 Kernel Logger — persists per-request S2 tier, rationale, and token/latency budgets
This is a lightweight helper for dashboards/observability.
"""
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import json
import os

LOG_DIR = os.environ.get("S2_KERNEL_LOG_DIR", "./logs")


@dataclass
class S2KernelLogRecord:
    timestamp: str
    request_id: str
    persona: Optional[str]
    s2_tier: str
    rationale: str
    token_overhead: int
    latency_overhead_ms: int
    budget_ok: bool
    metadata: Dict[str, Any]


def ensure_log_dir():
    os.makedirs(LOG_DIR, exist_ok=True)


def log_s2_kernel_event(
    request_id: str,
    s2_tier: str,
    rationale: str,
    token_overhead: int,
    latency_overhead_ms: int,
    persona: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Write a structured JSON log for dashboards.
    Returns the path written.
    """
    ensure_log_dir()
    path = os.path.join(LOG_DIR, f"s2_kernel_{datetime.now(timezone.utc).date()}.jsonl")

    # Determine budget by tier
    tier_budgets = {
        "S2_DISABLED": (0, 0),
        "S2_TIER_1": (150, 25000),  # tokens, ms
        "S2_TIER_2": (400, 60000),
        "S2_TIER_3": (700, 120000),
    }
    t_budget, l_budget = tier_budgets.get(s2_tier, (400, 60000))
    budget_ok = token_overhead <= t_budget and latency_overhead_ms <= l_budget

    rec = S2KernelLogRecord(
        timestamp=datetime.now(timezone.utc).isoformat(),
        request_id=request_id,
        persona=persona,
        s2_tier=s2_tier,
        rationale=rationale,
        token_overhead=token_overhead,
        latency_overhead_ms=latency_overhead_ms,
        budget_ok=budget_ok,
        metadata=metadata or {},
    )
    with open(path, "a") as f:
        f.write(json.dumps(asdict(rec)) + "\n")
    return path
