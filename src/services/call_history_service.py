#!/usr/bin/env python3
"""
Call History Service - Clean, Decoupled LLM Call Tracking
Replaces the tightly-coupled _call_history attribute pattern
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class CallRecord:
    """Individual LLM call record"""
    timestamp: str
    provider: str
    model: str
    method: str
    tokens_used: int
    cost_usd: float
    response_time_ms: int
    engagement_id: Optional[str] = None
    phase: Optional[str] = None


class CallHistoryService:
    """
    Clean, service-oriented call history tracking
    Replaces the old _call_history attribute pattern with proper dependency injection
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._call_history: List[CallRecord] = []
        self._provider_stats: Dict[str, Dict[str, Any]] = {}

    def record_call(
        self,
        provider: str,
        model: str,
        method: str,
        tokens_used: int,
        cost_usd: float,
        response_time_ms: int,
        engagement_id: Optional[str] = None,
        phase: Optional[str] = None,
    ):
        """Record an LLM call in the service history"""
        record = CallRecord(
            timestamp=datetime.now().isoformat(),
            provider=provider,
            model=model,
            method=method,
            tokens_used=tokens_used,
            cost_usd=cost_usd,
            response_time_ms=response_time_ms,
            engagement_id=engagement_id,
            phase=phase,
        )
        
        self._call_history.append(record)
        self._update_provider_stats(provider, cost_usd, response_time_ms, True)
        
        self.logger.debug(f"📊 Call recorded: {provider}/{model} - ${cost_usd:.4f}, {tokens_used} tokens")

    def get_call_history(self, provider: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get call history, optionally filtered by provider"""
        history = self._call_history
        
        if provider:
            history = [record for record in history if record.provider == provider]
        
        return [
            {
                "timestamp": record.timestamp,
                "provider": record.provider,
                "model": record.model,
                "method": record.method,
                "tokens": record.tokens_used,
                "cost": record.cost_usd,
                "response_time_ms": record.response_time_ms,
                "engagement_id": record.engagement_id,
                "phase": record.phase,
            }
            for record in history
        ]

    def get_total_cost(self, provider: Optional[str] = None) -> float:
        """Get total cost, optionally filtered by provider"""
        if provider:
            return sum(
                record.cost_usd for record in self._call_history 
                if record.provider == provider
            )
        return sum(record.cost_usd for record in self._call_history)

    def get_provider_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get comprehensive provider statistics"""
        stats = {}
        
        for provider in set(record.provider for record in self._call_history):
            provider_calls = [r for r in self._call_history if r.provider == provider]
            
            if provider_calls:
                total_cost = sum(r.cost_usd for r in provider_calls)
                total_tokens = sum(r.tokens_used for r in provider_calls)
                avg_response_time = sum(r.response_time_ms for r in provider_calls) / len(provider_calls)
                
                stats[provider] = {
                    "total_calls": len(provider_calls),
                    "total_cost": total_cost,
                    "total_tokens": total_tokens,
                    "average_response_time": avg_response_time,
                    "success_rate": 1.0,  # All recorded calls are successful
                }

        return stats

    def _update_provider_stats(
        self, provider: str, cost: float, response_time: float, success: bool
    ):
        """Update internal provider statistics"""
        if provider not in self._provider_stats:
            self._provider_stats[provider] = {
                "calls": 0,
                "failures": 0,
                "total_cost": 0.0,
                "total_response_time": 0.0,
            }

        stats = self._provider_stats[provider]
        stats["calls"] += 1
        
        if success:
            stats["total_cost"] += cost
            stats["total_response_time"] += response_time
        else:
            stats["failures"] += 1

    def clear_history(self):
        """Clear all call history (for testing/cleanup)"""
        self._call_history.clear()
        self._provider_stats.clear()
        self.logger.info("📊 Call history cleared")


# Global service instance
_call_history_service: Optional[CallHistoryService] = None


def get_call_history_service() -> CallHistoryService:
    """Get the global call history service instance"""
    global _call_history_service
    if _call_history_service is None:
        _call_history_service = CallHistoryService()
    return _call_history_service