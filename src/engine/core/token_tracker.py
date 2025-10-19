"""
METIS Token & Cost Tracking System
Centralized tracking for all LLM usage across the platform
"""

import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
import logging


@dataclass
class TokenUsageRecord:
    """Record of token usage for a single LLM call with enhanced cognitive exhaust capture"""

    timestamp: str
    engagement_id: Optional[str]
    phase: str
    provider: str
    model: str
    tokens_input: int
    tokens_output: int
    tokens_total: int
    cost_usd: float
    response_time_ms: int
    call_type: str
    success: bool
    # Enhanced cognitive exhaust fields
    raw_llm_output: Optional[Dict[str, Any]] = None
    prompt_template_used: Optional[str] = None
    reasoning_content: Optional[str] = None


class TokenTracker:
    """
    Singleton token tracker for system-wide usage monitoring
    Thread-safe implementation for concurrent access
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.logger = logging.getLogger(__name__)
        self.usage_records = []
        self.total_tokens = 0
        self.total_cost_usd = 0.0
        self.usage_by_phase = {}
        self.usage_by_provider = {}

        # Load persistent data if exists
        self.data_file = Path("data/token_usage.json")
        self._load_usage_data()
        self._initialized = True

    def track_usage(
        self,
        phase: str,
        provider: str,
        model: str,
        tokens_used: int,
        cost_usd: float,
        response_time_ms: int,
        engagement_id: Optional[str] = None,
        call_type: str = "analysis",
        success: bool = True,
        tokens_input: Optional[int] = None,
        tokens_output: Optional[int] = None,
        # Enhanced cognitive exhaust capture
        raw_llm_output: Optional[Dict[str, Any]] = None,
        prompt_template_used: Optional[str] = None,
        reasoning_content: Optional[str] = None,
    ):
        """Track token usage for an LLM call"""

        # Handle token breakdown
        if tokens_input is None or tokens_output is None:
            # Estimate if not provided (rough 3:1 output:input ratio)
            tokens_output = int(tokens_used * 0.75)
            tokens_input = tokens_used - tokens_output

        record = TokenUsageRecord(
            timestamp=datetime.now().isoformat(),
            engagement_id=engagement_id,
            phase=phase,
            provider=provider,
            model=model,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            tokens_total=tokens_used,
            cost_usd=cost_usd,
            response_time_ms=response_time_ms,
            call_type=call_type,
            success=success,
            # Enhanced cognitive exhaust fields
            raw_llm_output=raw_llm_output,
            prompt_template_used=prompt_template_used,
            reasoning_content=reasoning_content,
        )

        with self._lock:
            self.usage_records.append(record)
            self.total_tokens += tokens_used
            self.total_cost_usd += cost_usd

            # Update phase tracking
            if phase not in self.usage_by_phase:
                self.usage_by_phase[phase] = {"tokens": 0, "cost": 0.0, "calls": 0}
            self.usage_by_phase[phase]["tokens"] += tokens_used
            self.usage_by_phase[phase]["cost"] += cost_usd
            self.usage_by_phase[phase]["calls"] += 1

            # Update provider tracking
            if provider not in self.usage_by_provider:
                self.usage_by_provider[provider] = {
                    "tokens": 0,
                    "cost": 0.0,
                    "calls": 0,
                }
            self.usage_by_provider[provider]["tokens"] += tokens_used
            self.usage_by_provider[provider]["cost"] += cost_usd
            self.usage_by_provider[provider]["calls"] += 1

            # Persist data every 10 calls
            if len(self.usage_records) % 10 == 0:
                self._save_usage_data()

        # Log the tracking
        self.logger.info(
            f"💰 Token Usage - {phase}: {tokens_used} tokens, ${cost_usd:.4f} ({provider}/{model})"
        )

    def get_engagement_usage(self, engagement_id: str) -> Dict[str, Any]:
        """Get token usage for a specific engagement"""

        engagement_records = [
            r for r in self.usage_records if r.engagement_id == engagement_id
        ]

        if not engagement_records:
            return {"tokens": 0, "cost": 0.0, "calls": 0}

        return {
            "tokens": sum(r.tokens_total for r in engagement_records),
            "cost": sum(r.cost_usd for r in engagement_records),
            "calls": len(engagement_records),
            "avg_response_time_ms": sum(r.response_time_ms for r in engagement_records)
            / len(engagement_records),
            "by_phase": self._group_by_phase(engagement_records),
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get overall usage summary"""
        return {
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
            "total_calls": len(self.usage_records),
            "by_phase": self.usage_by_phase,
            "by_provider": self.usage_by_provider,
            "avg_tokens_per_call": (
                self.total_tokens / len(self.usage_records) if self.usage_records else 0
            ),
            "avg_cost_per_call": (
                self.total_cost_usd / len(self.usage_records)
                if self.usage_records
                else 0
            ),
        }

    def get_cognitive_exhaust(self, engagement_id: str) -> List[Dict[str, Any]]:
        """Get all cognitive exhaust data for a specific engagement"""
        with self._lock:
            exhaust_data = []
            for record in self.usage_records:
                if record.engagement_id == engagement_id and (
                    record.raw_llm_output
                    or record.prompt_template_used
                    or record.reasoning_content
                ):
                    exhaust_data.append(
                        {
                            "timestamp": record.timestamp,
                            "phase": record.phase,
                            "model": record.model,
                            "prompt_template": record.prompt_template_used,
                            "reasoning_content": record.reasoning_content,
                            "raw_output": record.raw_llm_output,
                            "call_type": record.call_type,
                            "tokens_used": record.tokens_total,
                            "cost_usd": record.cost_usd,
                        }
                    )
            return exhaust_data

    def _group_by_phase(self, records: list) -> Dict[str, Any]:
        """Group records by phase"""
        by_phase = {}
        for record in records:
            if record.phase not in by_phase:
                by_phase[record.phase] = {"tokens": 0, "cost": 0.0, "calls": 0}
            by_phase[record.phase]["tokens"] += record.tokens_total
            by_phase[record.phase]["cost"] += record.cost_usd
            by_phase[record.phase]["calls"] += 1
        return by_phase

    def _save_usage_data(self):
        """Persist usage data to disk"""
        try:
            self.data_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.data_file, "w") as f:
                json.dump(
                    {
                        "total_tokens": self.total_tokens,
                        "total_cost_usd": self.total_cost_usd,
                        "usage_by_phase": self.usage_by_phase,
                        "usage_by_provider": self.usage_by_provider,
                        "records": [
                            asdict(r) for r in self.usage_records[-1000:]
                        ],  # Keep last 1000 records
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            self.logger.warning(f"Could not save token usage data: {e}")

    def _load_usage_data(self):
        """Load persisted usage data"""
        try:
            if self.data_file.exists():
                with open(self.data_file, "r") as f:
                    data = json.load(f)
                    self.total_tokens = data.get("total_tokens", 0)
                    self.total_cost_usd = data.get("total_cost_usd", 0.0)
                    self.usage_by_phase = data.get("usage_by_phase", {})
                    self.usage_by_provider = data.get("usage_by_provider", {})
                    self.logger.info(
                        f"✅ Loaded token usage history: {self.total_tokens} tokens, ${self.total_cost_usd:.2f}"
                    )
        except Exception as e:
            self.logger.warning(f"Could not load token usage data: {e}")


# Global instance getter
def get_token_tracker() -> TokenTracker:
    """Get the global token tracker instance"""
    return TokenTracker()
