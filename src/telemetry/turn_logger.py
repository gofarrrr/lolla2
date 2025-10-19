"""
Turn-level telemetry logger
Logs per-LLM-call envelope with prompt hash, context IDs, model, cost, latency,
confidence, and validation verdicts for dashboards and monitoring.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

LOG_DIR = os.environ.get("TURN_LOG_DIR", "./logs")


@dataclass
class ValidationVerdicts:
    contract_valid: Optional[bool] = None
    groundedness: Optional[float] = None
    self_verification: Optional[float] = None
    style_score: Optional[float] = None
    notes: Optional[str] = None


@dataclass
class TurnLogRecord:
    timestamp: str
    prompt_hash: str
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: float
    confidence: Optional[float] = None
    context_ids: List[str] = field(default_factory=list)
    engagement_id: Optional[str] = None
    phase: Optional[str] = None
    validation: ValidationVerdicts = field(default_factory=ValidationVerdicts)
    extra: Dict[str, Any] = field(default_factory=dict)


def _ensure_log_dir() -> str:
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, f"turns_{datetime.now(timezone.utc).date()}.jsonl")
    return path


def log_turn(record: TurnLogRecord) -> str:
    """Append a turn log JSON line. Returns the file path written."""
    path = _ensure_log_dir()
    with open(path, "a") as f:
        f.write(json.dumps(asdict(record)) + "\n")
    return path
