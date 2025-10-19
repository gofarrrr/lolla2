#!/usr/bin/env python3
"""
METIS System-Wide Recording Infrastructure
Centralized recording system for all API calls and system interactions
"""

import logging
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class LLMCallRecord:
    """Record of LLM API call"""

    timestamp: str
    provider: str
    model: str
    method: str
    tokens_used: int
    cost_usd: float
    response_time_ms: int
    research_enhanced: bool = False


@dataclass
class PerplexityCallRecord:
    """Record of Perplexity research call"""

    timestamp: str
    query: str
    sources_count: int
    cost_usd: float
    response_time_ms: int
    mode: str
    success: bool = True


@dataclass
class SystemRecords:
    """Container for all system call records"""

    llm_calls: List[LLMCallRecord] = field(default_factory=list)
    perplexity_calls: List[PerplexityCallRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for JSON serialization"""
        return {
            "llm_calls": [
                {
                    "timestamp": call.timestamp,
                    "provider": call.provider,
                    "model": call.model,
                    "method": call.method,
                    "tokens": call.tokens_used,
                    "cost_usd": call.cost_usd,
                    "response_time_ms": call.response_time_ms,
                    "research_enhanced": call.research_enhanced,
                }
                for call in self.llm_calls
            ],
            "perplexity_calls": [
                {
                    "timestamp": call.timestamp,
                    "query": (
                        call.query[:200] + "..."
                        if len(call.query) > 200
                        else call.query
                    ),
                    "sources_count": call.sources_count,
                    "cost_usd": call.cost_usd,
                    "response_time_ms": call.response_time_ms,
                    "mode": call.mode,
                    "success": call.success,
                }
                for call in self.perplexity_calls
            ],
        }


class SystemRecorder:
    """Global system recorder for all API calls and interactions"""

    _instance: Optional["SystemRecorder"] = None
    _lock = threading.Lock()

    def __init__(self):
        self.records = SystemRecords()
        self.enabled = True
        self.external_recorders: List[Callable] = []

    @classmethod
    def get_instance(cls) -> "SystemRecorder":
        """Get or create global recorder instance"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset global instance (for testing)"""
        with cls._lock:
            cls._instance = None

    def add_external_recorder(self, recorder_func: Callable):
        """Add external recording function to be called for all events"""
        self.external_recorders.append(recorder_func)

    def remove_external_recorder(self, recorder_func: Callable):
        """Remove external recording function"""
        if recorder_func in self.external_recorders:
            self.external_recorders.remove(recorder_func)

    def record_llm_call(
        self,
        provider: str,
        model: str,
        method: str,
        tokens_used: int,
        cost_usd: float,
        response_time_ms: int,
        research_enhanced: bool = False,
    ):
        """Record LLM API call"""
        if not self.enabled:
            return

        record = LLMCallRecord(
            timestamp=datetime.now().isoformat(),
            provider=provider,
            model=model,
            method=method,
            tokens_used=tokens_used,
            cost_usd=cost_usd,
            response_time_ms=response_time_ms,
            research_enhanced=research_enhanced,
        )

        self.records.llm_calls.append(record)

        # Notify external recorders
        for recorder in self.external_recorders:
            try:
                recorder("llm_call", record)
            except Exception as e:
                logger.warning(f"External recorder failed: {e}")

        logger.debug(
            f"Recorded LLM call: {provider}/{model} - {tokens_used} tokens, ${cost_usd:.4f}"
        )

    def record_perplexity_call(
        self,
        query: str,
        sources_count: int,
        cost_usd: float,
        response_time_ms: int,
        mode: str = "moderate",
        success: bool = True,
    ):
        """Record Perplexity research call"""
        if not self.enabled:
            return

        record = PerplexityCallRecord(
            timestamp=datetime.now().isoformat(),
            query=query,
            sources_count=sources_count,
            cost_usd=cost_usd,
            response_time_ms=response_time_ms,
            mode=mode,
            success=success,
        )

        self.records.perplexity_calls.append(record)

        # Notify external recorders
        for recorder in self.external_recorders:
            try:
                recorder("perplexity_call", record)
            except Exception as e:
                logger.warning(f"External recorder failed: {e}")

        logger.debug(
            f"Recorded Perplexity call: {sources_count} sources, ${cost_usd:.4f}"
        )

    def get_records(self) -> SystemRecords:
        """Get current system records"""
        return self.records

    def get_records_dict(self) -> Dict[str, Any]:
        """Get records as dictionary"""
        return self.records.to_dict()

    def clear_records(self):
        """Clear all recorded data"""
        self.records = SystemRecords()

    def disable_recording(self):
        """Disable recording"""
        self.enabled = False

    def enable_recording(self):
        """Enable recording"""
        self.enabled = True

    def get_total_cost(self) -> float:
        """Get total cost across all recorded calls"""
        llm_cost = sum(call.cost_usd for call in self.records.llm_calls)
        perplexity_cost = sum(call.cost_usd for call in self.records.perplexity_calls)
        return llm_cost + perplexity_cost

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of recorded activity"""
        return {
            "llm_calls_count": len(self.records.llm_calls),
            "perplexity_calls_count": len(self.records.perplexity_calls),
            "total_llm_tokens": sum(
                call.tokens_used for call in self.records.llm_calls
            ),
            "total_cost_usd": self.get_total_cost(),
            "research_enhanced_calls": sum(
                1 for call in self.records.llm_calls if call.research_enhanced
            ),
            "successful_perplexity_calls": sum(
                1 for call in self.records.perplexity_calls if call.success
            ),
        }


# Global convenience functions
def get_system_recorder() -> SystemRecorder:
    """Get the global system recorder instance"""
    return SystemRecorder.get_instance()


def record_llm_call(
    provider: str,
    model: str,
    method: str,
    tokens: int,
    cost: float,
    time_ms: int,
    research_enhanced: bool = False,
):
    """Convenience function to record LLM call"""
    get_system_recorder().record_llm_call(
        provider, model, method, tokens, cost, time_ms, research_enhanced
    )


def record_perplexity_call(
    query: str,
    sources: int,
    cost: float,
    time_ms: int,
    mode: str = "moderate",
    success: bool = True,
):
    """Convenience function to record Perplexity call"""
    get_system_recorder().record_perplexity_call(
        query, sources, cost, time_ms, mode, success
    )


def get_recorded_calls() -> Dict[str, Any]:
    """Convenience function to get all recorded calls"""
    return get_system_recorder().get_records_dict()


def clear_recordings():
    """Convenience function to clear all recordings"""
    get_system_recorder().clear_records()
