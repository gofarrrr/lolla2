"""
LLM Cost Repository - Cost Tracking and Analytics

Provides database interface for tracking LLM API costs and performance metrics.
Supports cost dashboard and budget monitoring.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class LLMCostEvent:
    """Single LLM cost event"""
    engagement_id: Optional[str]
    phase: str
    provider: str
    model: str
    tokens_input: int
    tokens_output: int
    cost_usd: float
    latency_ms: int
    reasoning_enabled: bool
    success: bool
    error_message: Optional[str] = None
    request_metadata: Optional[Dict[str, Any]] = None
    ts: Optional[datetime] = None


class LLMCostRepository:
    """Repository for LLM cost events"""

    def __init__(self, supabase_client=None):
        """
        Initialize repository with optional Supabase client.

        Args:
            supabase_client: Optional Supabase client for database operations
        """
        self.client = supabase_client
        self.logger = logging.getLogger(__name__)

        if not self.client:
            self.logger.warning("⚠️ No Supabase client provided - cost tracking disabled")

    def insert_event(
        self,
        engagement_id: Optional[str],
        phase: str,
        provider: str,
        model: str,
        tokens_input: int,
        tokens_output: int,
        cost_usd: float,
        latency_ms: int,
        reasoning_enabled: bool = False,
        success: bool = True,
        error_message: Optional[str] = None,
        request_metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Insert a single LLM cost event.

        Args:
            engagement_id: Optional engagement ID
            phase: Pipeline phase
            provider: LLM provider
            model: Model name
            tokens_input: Input token count
            tokens_output: Output token count
            cost_usd: Cost in USD
            latency_ms: Response latency in milliseconds
            reasoning_enabled: Whether reasoning mode was enabled
            success: Whether call succeeded
            error_message: Optional error message
            request_metadata: Optional request metadata

        Returns:
            True if insert succeeded, False otherwise
        """
        if not self.client:
            return False

        try:
            event_data = {
                "ts": datetime.now().isoformat(),
                "engagement_id": engagement_id,
                "phase": phase,
                "provider": provider,
                "model": model,
                "tokens_input": tokens_input,
                "tokens_output": tokens_output,
                "cost_usd": cost_usd,
                "latency_ms": latency_ms,
                "reasoning_enabled": reasoning_enabled,
                "success": success,
                "error_message": error_message,
                "request_metadata": request_metadata
            }

            self.client.table("llm_cost_events").insert(event_data).execute()
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to insert cost event: {e}")
            return False

    def get_cost_rollup(self, days: int = 7) -> Dict[str, Any]:
        """
        Get cost rollup for last N days.

        Args:
            days: Number of days to look back

        Returns:
            Dict with cost aggregates
        """
        if not self.client:
            return self._get_empty_rollup()

        try:
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()

            # Total cost
            total_result = (
                self.client.table("llm_cost_events")
                .select("cost_usd")
                .gte("ts", cutoff)
                .execute()
            )

            total_cost = sum(row["cost_usd"] for row in total_result.data)

            # Cost by provider
            provider_result = (
                self.client.rpc("get_cost_by_provider", {"days_back": days})
                .execute()
            )
            cost_by_provider = {
                row["provider"]: row["total_cost"]
                for row in (provider_result.data or [])
            }

            # Cost by phase
            phase_result = (
                self.client.rpc("get_cost_by_phase", {"days_back": days})
                .execute()
            )
            cost_by_phase = {
                row["phase"]: row["total_cost"]
                for row in (phase_result.data or [])
            }

            # Reasoning mix
            reasoning_result = (
                self.client.rpc("get_reasoning_mix", {"days_back": days})
                .execute()
            )
            reasoning_data = reasoning_result.data[0] if reasoning_result.data else {}

            # Daily series
            daily_result = (
                self.client.rpc("get_daily_cost_series", {"days_back": days})
                .execute()
            )
            daily_series = [
                {"date": row["date"], "cost": row["total_cost"]}
                for row in (daily_result.data or [])
            ]

            return {
                "total_cost": round(total_cost, 4),
                "cost_by_provider": cost_by_provider,
                "cost_by_phase": cost_by_phase,
                "reasoning_mix": reasoning_data,
                "daily_series": daily_series,
                "period_days": days
            }

        except Exception as e:
            self.logger.error(f"❌ Failed to get cost rollup: {e}")
            return self._get_empty_rollup()

    def _get_empty_rollup(self) -> Dict[str, Any]:
        """Return empty rollup structure"""
        return {
            "total_cost": 0.0,
            "cost_by_provider": {},
            "cost_by_phase": {},
            "reasoning_mix": {},
            "daily_series": [],
            "period_days": 0
        }

    def get_average_latency(self, provider: Optional[str] = None, days: int = 7) -> Dict[str, float]:
        """
        Get average latency by provider.

        Args:
            provider: Optional provider filter
            days: Number of days to look back

        Returns:
            Dict with average latency by provider
        """
        if not self.client:
            return {}

        try:
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()

            query = (
                self.client.table("llm_cost_events")
                .select("provider, latency_ms")
                .gte("ts", cutoff)
                .eq("success", True)
            )

            if provider:
                query = query.eq("provider", provider)

            result = query.execute()

            # Calculate averages by provider
            latency_by_provider = {}
            for row in result.data:
                p = row["provider"]
                if p not in latency_by_provider:
                    latency_by_provider[p] = []
                latency_by_provider[p].append(row["latency_ms"])

            return {
                p: round(sum(latencies) / len(latencies), 2)
                for p, latencies in latency_by_provider.items()
            }

        except Exception as e:
            self.logger.error(f"❌ Failed to get average latency: {e}")
            return {}

    def get_provider_health(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Aggregate provider health metrics over a period.

        Returns a list of dicts with keys:
        - provider
        - success_rate (0..1)
        - avg_latency_ms
        - total_calls
        - total_cost
        - reasoning_calls_pct (0..100)
        """
        if not self.client:
            return []
        try:
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()
            result = (
                self.client.table("llm_cost_events")
                .select("provider, success, latency_ms, cost_usd, reasoning_enabled")
                .gte("ts", cutoff)
                .execute()
            )
            by_provider: Dict[str, Dict[str, Any]] = {}
            for row in result.data or []:
                p = row.get("provider")
                if not p:
                    continue
                agg = by_provider.setdefault(
                    p,
                    {
                        "provider": p,
                        "success_count": 0,
                        "total_calls": 0,
                        "latencies": [],
                        "total_cost": 0.0,
                        "reasoning_count": 0,
                    },
                )
                agg["total_calls"] += 1
                if bool(row.get("success")):
                    agg["success_count"] += 1
                lat = row.get("latency_ms")
                if isinstance(lat, (int, float)):
                    agg["latencies"].append(float(lat))
                cost = row.get("cost_usd")
                if isinstance(cost, (int, float)):
                    agg["total_cost"] += float(cost)
                if bool(row.get("reasoning_enabled")):
                    agg["reasoning_count"] += 1
            health: List[Dict[str, Any]] = []
            for p, agg in by_provider.items():
                calls = max(agg["total_calls"], 1)
                latency_avg = (
                    sum(agg["latencies"]) / len(agg["latencies"]) if agg["latencies"] else 0.0
                )
                health.append(
                    {
                        "provider": p,
                        "success_rate": round(agg["success_count"] / calls, 4),
                        "avg_latency_ms": round(latency_avg, 2),
                        "total_calls": agg["total_calls"],
                        "total_cost": round(agg["total_cost"], 4),
                        "reasoning_calls_pct": round(
                            (agg["reasoning_count"] / calls) * 100.0, 1
                        ),
                    }
                )
            return sorted(health, key=lambda x: x["total_cost"], reverse=True)
        except Exception as e:
            self.logger.error(f"❌ Failed to get provider health: {e}")
            return []


# Global instance
_cost_repository: Optional[LLMCostRepository] = None


def get_llm_cost_repository(supabase_client=None) -> LLMCostRepository:
    """Get or create global LLM cost repository"""
    global _cost_repository

    if _cost_repository is None or supabase_client is not None:
        _cost_repository = LLMCostRepository(supabase_client)

    return _cost_repository
