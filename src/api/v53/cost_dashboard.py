"""
Cost Dashboard API - LLM Cost Monitoring and Analytics

Provides real-time visibility into LLM API costs and performance metrics.
Supports budget tracking, provider comparison, and optimization insights.
"""

from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from src.engine.persistence.llm_cost_repository import get_llm_cost_repository


router = APIRouter(prefix="/api/v53", tags=["Cost Dashboard"])
logger = logging.getLogger(__name__)


class CostDashboardResponse(BaseModel):
    """Cost dashboard response model"""
    total_cost: float
    cost_by_provider: Dict[str, float]
    cost_by_phase: Dict[str, float]
    reasoning_mix: Dict[str, Any]
    daily_series: List[Dict[str, Any]]
    average_latency: Dict[str, float]
    period_days: int
    timestamp: str


class ProviderHealthResponse(BaseModel):
    """Provider health metrics"""
    provider: str
    success_rate: float
    avg_latency_ms: float
    total_calls: int
    total_cost: float
    reasoning_calls_pct: float


@router.get("/cost-dashboard", response_model=CostDashboardResponse)
async def get_cost_dashboard(
    days: int = Query(7, ge=1, le=90, description="Number of days to look back")
):
    """
    Get comprehensive cost dashboard metrics.

    Returns:
    - Total cost across all providers
    - Cost breakdown by provider
    - Cost breakdown by phase
    - Reasoning mode utilization
    - Daily cost time series
    - Average latency by provider

    Example:
        GET /api/v53/cost-dashboard?days=7

    Response:
        {
          "total_cost": 15.43,
          "cost_by_provider": {
            "openrouter": 8.21,
            "deepseek": 4.12,
            "anthropic": 3.10
          },
          "cost_by_phase": {
            "hypothesis_generation": 5.20,
            "analysis_execution": 7.15,
            "synthesis": 3.08
          },
          "reasoning_mix": {
            "total_calls": 142,
            "reasoning_enabled_count": 87,
            "reasoning_enabled_pct": 61.3
          },
          "daily_series": [
            {"date": "2025-10-05", "cost": 2.15},
            {"date": "2025-10-06", "cost": 2.41},
            ...
          ],
          "average_latency": {
            "openrouter": 1245.3,
            "deepseek": 892.1,
            "anthropic": 1503.7
          },
          "period_days": 7,
          "timestamp": "2025-10-11T14:23:45Z"
        }
    """
    try:
        # Get cost repository
        repo = get_llm_cost_repository()

        # Get cost rollup
        rollup = repo.get_cost_rollup(days=days)

        # Get average latency
        avg_latency = repo.get_average_latency(days=days)

        return CostDashboardResponse(
            total_cost=rollup["total_cost"],
            cost_by_provider=rollup["cost_by_provider"],
            cost_by_phase=rollup["cost_by_phase"],
            reasoning_mix=rollup["reasoning_mix"],
            daily_series=rollup["daily_series"],
            average_latency=avg_latency,
            period_days=days,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"❌ Cost dashboard error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate cost dashboard: {str(e)}"
        )


@router.get("/cost-dashboard/provider-health")
async def get_provider_health(
    provider: Optional[str] = Query(None, description="Filter by provider"),
    days: int = Query(7, ge=1, le=90, description="Number of days to look back")
):
    """
    Get provider health metrics.

    Returns success rates, latency, and cost by provider.

    Example:
        GET /api/v53/cost-dashboard/provider-health?days=7

    Response:
        {
          "providers": [
            {
              "provider": "openrouter",
              "success_rate": 0.98,
              "avg_latency_ms": 1245.3,
              "total_calls": 87,
              "total_cost": 8.21,
              "reasoning_calls_pct": 68.5
            },
            ...
          ]
        }
    """
    try:
        # TODO: Implement provider health tracking
        # This requires additional queries to llm_cost_events table

        return {
            "providers": [],
            "message": "Provider health tracking coming soon"
        }

    except Exception as e:
        logger.error(f"❌ Provider health error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get provider health: {str(e)}"
        )


@router.get("/cost-dashboard/budget-alert")
async def check_budget_alert(
    daily_budget: float = Query(..., gt=0, description="Daily budget limit in USD"),
    days: int = Query(1, ge=1, le=7, description="Number of days to check")
):
    """
    Check if daily budget has been exceeded.

    Example:
        GET /api/v53/cost-dashboard/budget-alert?daily_budget=10.0&days=1

    Response:
        {
          "alert": true,
          "current_daily_cost": 12.43,
          "budget_limit": 10.0,
          "overage": 2.43,
          "overage_pct": 24.3
        }
    """
    try:
        repo = get_llm_cost_repository()
        rollup = repo.get_cost_rollup(days=days)

        # Calculate daily average
        daily_cost = rollup["total_cost"] / days if days > 0 else 0

        alert = daily_cost > daily_budget
        overage = max(0, daily_cost - daily_budget)
        overage_pct = (overage / daily_budget * 100) if daily_budget > 0 else 0

        return {
            "alert": alert,
            "current_daily_cost": round(daily_cost, 2),
            "budget_limit": daily_budget,
            "overage": round(overage, 2),
            "overage_pct": round(overage_pct, 1),
            "period_days": days
        }

    except Exception as e:
        logger.error(f"❌ Budget alert error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to check budget alert: {str(e)}"
        )
