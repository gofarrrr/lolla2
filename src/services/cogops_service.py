#!/usr/bin/env python3
"""
CogOps Service - Cognitive Operations Metrics
Provides complex aggregations and analytics on context_streams logs for operational intelligence
Reference: C2_Backend_Blueprint.md Part 3
"""

import os
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from src.services.persistence.database_service import (
    DatabaseService,
    DatabaseServiceConfig,
    DatabaseOperationError,
)

logger = logging.getLogger(__name__)


class CogOpsService:
    """Service for calculating operational metrics from context_streams data"""

    def __init__(self, database_service: DatabaseService | None = None):
        # Resolve unified DatabaseService (facade)
        self.database_service: DatabaseService | None = database_service
        if self.database_service is None:
            try:
                self.database_service = DatabaseService(DatabaseServiceConfig.from_env())
            except Exception:
                self.database_service = None
        logger.info("🚀 CogOps Service initialized (DatabaseService=%s)", bool(self.database_service))

    def _get_time_range(self, timeframe: str) -> Tuple[datetime, datetime]:
        """Get start and end times for a given timeframe"""
        end_time = datetime.utcnow()

        if timeframe == "1h":
            start_time = end_time - timedelta(hours=1)
        elif timeframe == "24h":
            start_time = end_time - timedelta(hours=24)
        elif timeframe == "7d":
            start_time = end_time - timedelta(days=7)
        elif timeframe == "30d":
            start_time = end_time - timedelta(days=30)
        else:
            # Default to 24h
            start_time = end_time - timedelta(hours=24)

        return start_time, end_time

    async def calculate_cost_metrics(self, timeframe: str = "24h") -> Dict[str, Any]:
        """Calculate cost metrics with time-bucketed data"""
        start_time, end_time = self._get_time_range(timeframe)

        # Get granularity based on timeframe
        if timeframe == "1h":
            granularity = "minute"
            trunc_format = "minute"
        elif timeframe in ["24h", "7d"]:
            granularity = "hour"
            trunc_format = "hour"
        else:
            granularity = "day"
            trunc_format = "day"

        try:
            # Execute the cost metrics query from blueprint
            if not self.database_service:
                raise DatabaseOperationError("DatabaseService unavailable")
            result = await self.database_service.execute_sql_async(
                f"""
                WITH cost_metrics AS (
                  SELECT 
                    DATE_TRUNC('{trunc_format}', started_at) as time_bucket,
                    SUM(total_cost) as total_cost,
                    AVG(total_cost) as avg_cost,
                    COUNT(*) as engagement_count
                  FROM context_streams
                  WHERE started_at >= '{start_time.isoformat()}'::timestamp
                    AND started_at <= '{end_time.isoformat()}'::timestamp
                    AND final_status = 'completed'
                  GROUP BY time_bucket
                )
                SELECT * FROM cost_metrics ORDER BY time_bucket DESC;
                """
            )

            if result.data:
                return {
                    "timeframe": {
                        "start": start_time.isoformat(),
                        "end": end_time.isoformat(),
                        "granularity": granularity,
                    },
                    "buckets": result.data,
                    "total_cost": sum(row.get("total_cost", 0) for row in result.data),
                    "total_engagements": sum(
                        row.get("engagement_count", 0) for row in result.data
                    ),
                }
            else:
                return {
                    "timeframe": {
                        "start": start_time.isoformat(),
                        "end": end_time.isoformat(),
                        "granularity": granularity,
                    },
                    "buckets": [],
                    "total_cost": 0,
                    "total_engagements": 0,
                }

        except Exception as e:
            logger.error(f"❌ Cost metrics calculation failed: {e}")
            # Return mock data structure for now
            return {
                "timeframe": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                    "granularity": granularity,
                },
                "buckets": [],
                "total_cost": 0,
                "total_engagements": 0,
                "error": str(e),
            }

    async def calculate_latency_by_consultant(
        self, timeframe: str = "24h"
    ) -> Dict[str, Any]:
        """Calculate average latency metrics by consultant"""
        start_time, end_time = self._get_time_range(timeframe)

        try:
            # Execute latency metrics query from blueprint
            if not self.database_service:
                raise DatabaseOperationError("DatabaseService unavailable")
            result = await self.database_service.execute_sql_async(
                f"""
                WITH latency_metrics AS (
                  SELECT
                    UNNEST(consultants_used) as consultant,
                    AVG(duration_ms) as avg_latency_ms,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY duration_ms) as median_latency_ms,
                    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms) as p95_latency_ms,
                    COUNT(*) as invocation_count
                  FROM context_streams
                  WHERE started_at >= '{start_time.isoformat()}'::timestamp
                    AND started_at <= '{end_time.isoformat()}'::timestamp
                    AND final_status = 'completed'
                    AND consultants_used IS NOT NULL
                    AND array_length(consultants_used, 1) > 0
                  GROUP BY consultant
                )
                SELECT * FROM latency_metrics ORDER BY invocation_count DESC;
                """
            )

            consultants = {}
            if result.data:
                for row in result.data:
                    consultants[row["consultant"]] = {
                        "invocations": row["invocation_count"],
                        "avg_latency_ms": (
                            float(row["avg_latency_ms"]) if row["avg_latency_ms"] else 0
                        ),
                        "median_latency_ms": (
                            float(row["median_latency_ms"])
                            if row["median_latency_ms"]
                            else 0
                        ),
                        "p95_latency_ms": (
                            float(row["p95_latency_ms"]) if row["p95_latency_ms"] else 0
                        ),
                    }

            return {
                "timeframe": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                },
                "consultants": consultants,
            }

        except Exception as e:
            logger.error(f"❌ Latency metrics calculation failed: {e}")
            return {
                "timeframe": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                },
                "consultants": {},
                "error": str(e),
            }

    async def calculate_fallback_rate(self, timeframe: str = "24h") -> Dict[str, Any]:
        """Calculate fallback rate metrics"""
        start_time, end_time = self._get_time_range(timeframe)

        try:
            # Execute fallback rate query from blueprint
            if not self.database_service:
                raise DatabaseOperationError("DatabaseService unavailable")
            result = await self.database_service.execute_sql_async(
                f"""
                WITH fallback_events AS (
                  SELECT
                    trace_id,
                    context_stream->'events' as events,
                    (
                      SELECT COUNT(*)
                      FROM jsonb_array_elements(context_stream->'events') e
                      WHERE e->>'event_type' LIKE '%FALLBACK%' OR e->>'event_type' LIKE '%RETRY%'
                    ) as fallback_count
                  FROM context_streams
                  WHERE started_at >= '{start_time.isoformat()}'::timestamp
                    AND started_at <= '{end_time.isoformat()}'::timestamp
                )
                SELECT
                  COUNT(*) as total_engagements,
                  SUM(CASE WHEN fallback_count > 0 THEN 1 ELSE 0 END) as engagements_with_fallback,
                  AVG(fallback_count) as avg_fallbacks_per_engagement,
                  ROUND(
                    SUM(CASE WHEN fallback_count > 0 THEN 1 ELSE 0 END)::numeric / 
                    NULLIF(COUNT(*)::numeric, 0) * 100, 2
                  ) as fallback_rate_percentage
                FROM fallback_events;
                """
            )

            if result.data and result.data[0]:
                row = result.data[0]
                return {
                    "timeframe": {
                        "start": start_time.isoformat(),
                        "end": end_time.isoformat(),
                    },
                    "total_engagements": row.get("total_engagements", 0),
                    "engagements_with_fallback": row.get(
                        "engagements_with_fallback", 0
                    ),
                    "avg_fallbacks_per_engagement": float(
                        row.get("avg_fallbacks_per_engagement", 0)
                    ),
                    "fallback_rate_percentage": float(
                        row.get("fallback_rate_percentage", 0)
                    ),
                }
            else:
                return {
                    "timeframe": {
                        "start": start_time.isoformat(),
                        "end": end_time.isoformat(),
                    },
                    "total_engagements": 0,
                    "engagements_with_fallback": 0,
                    "avg_fallbacks_per_engagement": 0.0,
                    "fallback_rate_percentage": 0.0,
                }

        except Exception as e:
            logger.error(f"❌ Fallback rate calculation failed: {e}")
            return {
                "timeframe": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                },
                "total_engagements": 0,
                "engagements_with_fallback": 0,
                "avg_fallbacks_per_engagement": 0.0,
                "fallback_rate_percentage": 0.0,
                "error": str(e),
            }

    async def calculate_model_usage(self, timeframe: str = "7d") -> Dict[str, Any]:
        """Calculate model usage distribution"""
        start_time, end_time = self._get_time_range(timeframe)

        try:
            # Execute model usage query from blueprint
            if not self.database_service:
                raise DatabaseOperationError("DatabaseService unavailable")
            result = await self.database_service.execute_sql_async(
                f"""
                WITH model_usage AS (
                  SELECT
                    UNNEST(models_used) as model,
                    COUNT(*) as usage_count,
                    SUM(total_tokens) as total_tokens,
                    SUM(total_cost) as total_cost,
                    AVG(overall_quality_score) as avg_quality
                  FROM context_streams
                  WHERE started_at >= '{start_time.isoformat()}'::timestamp
                    AND started_at <= '{end_time.isoformat()}'::timestamp
                    AND models_used IS NOT NULL
                    AND array_length(models_used, 1) > 0
                  GROUP BY model
                )
                SELECT 
                  model,
                  usage_count,
                  total_tokens,
                  total_cost,
                  avg_quality,
                  ROUND(usage_count::numeric / SUM(usage_count) OVER() * 100, 2) as usage_percentage
                FROM model_usage
                ORDER BY usage_count DESC;
                """
            )

            models = {}
            if result.data:
                for row in result.data:
                    models[row["model"]] = {
                        "usage_count": row["usage_count"],
                        "total_tokens": row["total_tokens"] or 0,
                        "total_cost": (
                            float(row["total_cost"]) if row["total_cost"] else 0
                        ),
                        "avg_quality": (
                            float(row["avg_quality"]) if row["avg_quality"] else 0
                        ),
                        "usage_percentage": (
                            float(row["usage_percentage"])
                            if row["usage_percentage"]
                            else 0
                        ),
                    }

            return {
                "timeframe": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                },
                "models": models,
            }

        except Exception as e:
            logger.error(f"❌ Model usage calculation failed: {e}")
            return {
                "timeframe": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                },
                "models": {},
                "error": str(e),
            }

    async def calculate_error_rate(self, timeframe: str = "24h") -> Dict[str, Any]:
        """Calculate error rate by time bucket"""
        start_time, end_time = self._get_time_range(timeframe)

        # Get granularity based on timeframe
        if timeframe == "1h":
            trunc_format = "minute"
        elif timeframe in ["24h", "7d"]:
            trunc_format = "hour"
        else:
            trunc_format = "day"

        try:
            # Execute error analysis query from blueprint
            result = self.supabase.rpc(
                "execute_sql",
                {
                    "sql": f"""
                WITH error_analysis AS (
                  SELECT
                    DATE_TRUNC('{trunc_format}', started_at) as time_bucket,
                    COUNT(*) as total_engagements,
                    SUM(error_count) as total_errors,
                    SUM(CASE WHEN final_status = 'failed' THEN 1 ELSE 0 END) as failed_engagements,
                    AVG(error_count) as avg_errors_per_engagement
                  FROM context_streams
                  WHERE started_at >= '{start_time.isoformat()}'::timestamp
                    AND started_at <= '{end_time.isoformat()}'::timestamp
                  GROUP BY time_bucket
                )
                SELECT 
                  time_bucket,
                  total_engagements,
                  total_errors,
                  failed_engagements,
                  ROUND(failed_engagements::numeric / NULLIF(total_engagements::numeric, 0) * 100, 2) as failure_rate,
                  avg_errors_per_engagement
                FROM error_analysis
                ORDER BY time_bucket DESC;
                """
                },
            ).execute()

            buckets = []
            if result.data:
                buckets = [
                    {
                        "time_bucket": row["time_bucket"],
                        "total_engagements": row["total_engagements"],
                        "total_errors": row["total_errors"],
                        "failed_engagements": row["failed_engagements"],
                        "failure_rate": (
                            float(row["failure_rate"]) if row["failure_rate"] else 0
                        ),
                        "avg_errors_per_engagement": (
                            float(row["avg_errors_per_engagement"])
                            if row["avg_errors_per_engagement"]
                            else 0
                        ),
                    }
                    for row in result.data
                ]

            return {
                "timeframe": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                },
                "buckets": buckets,
            }

        except Exception as e:
            logger.error(f"❌ Error rate calculation failed: {e}")
            return {
                "timeframe": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                },
                "buckets": [],
                "error": str(e),
            }

    async def get_operational_summary(
        self, timeframe: str = "24h", granularity: str = "hour"
    ) -> Dict[str, Any]:
        """Get comprehensive operational summary"""
        start_time, end_time = self._get_time_range(timeframe)

        # Get all metrics in parallel
        try:
            cost_metrics = await self.calculate_cost_metrics(timeframe)
            latency_metrics = await self.calculate_latency_by_consultant(timeframe)
            fallback_metrics = await self.calculate_fallback_rate(timeframe)
            model_metrics = await self.calculate_model_usage(timeframe)
            error_metrics = await self.calculate_error_rate(timeframe)

            # Calculate summary statistics
            total_engagements = cost_metrics.get("total_engagements", 0)
            total_cost = cost_metrics.get("total_cost", 0)
            failed_engagements = sum(
                bucket.get("failed_engagements", 0)
                for bucket in error_metrics.get("buckets", [])
            )

            # Calculate performance metrics
            all_latencies = []
            for consultant_data in latency_metrics.get("consultants", {}).values():
                if consultant_data.get("avg_latency_ms"):
                    all_latencies.append(consultant_data["avg_latency_ms"])

            avg_latency = (
                sum(all_latencies) / len(all_latencies) if all_latencies else 0
            )

            return {
                "timeframe": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                    "duration": timeframe,
                },
                "summary": {
                    "total_engagements": total_engagements,
                    "successful_engagements": total_engagements - failed_engagements,
                    "failed_engagements": failed_engagements,
                    "success_rate": (total_engagements - failed_engagements)
                    / max(total_engagements, 1),
                    "total_cost": total_cost,
                    "avg_cost_per_engagement": total_cost / max(total_engagements, 1),
                    "total_tokens": sum(
                        model["total_tokens"]
                        for model in model_metrics.get("models", {}).values()
                    ),
                    "avg_tokens_per_engagement": sum(
                        model["total_tokens"]
                        for model in model_metrics.get("models", {}).values()
                    )
                    / max(total_engagements, 1),
                },
                "performance": {
                    "avg_latency_ms": avg_latency,
                    "median_latency_ms": avg_latency * 0.8,  # Approximation
                    "p95_latency_ms": avg_latency * 1.8,  # Approximation
                    "p99_latency_ms": avg_latency * 2.5,  # Approximation
                },
                "reliability": {
                    "error_rate": failed_engagements / max(total_engagements, 1),
                    "fallback_rate": fallback_metrics.get("fallback_rate_percentage", 0)
                    / 100,
                    "avg_retries": fallback_metrics.get(
                        "avg_fallbacks_per_engagement", 0
                    ),
                },
                "resource_usage": {
                    "models": model_metrics.get("models", {}),
                    "consultants": latency_metrics.get("consultants", {}),
                },
            }

        except Exception as e:
            logger.error(f"❌ Operational summary calculation failed: {e}")
            return {
                "timeframe": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                    "duration": timeframe,
                },
                "summary": {
                    "total_engagements": 0,
                    "successful_engagements": 0,
                    "failed_engagements": 0,
                    "success_rate": 0.0,
                    "total_cost": 0.0,
                    "avg_cost_per_engagement": 0.0,
                    "total_tokens": 0,
                    "avg_tokens_per_engagement": 0,
                },
                "error": str(e),
            }


# Global instance
_cogops_service: Optional[CogOpsService] = None


def get_cogops_service() -> CogOpsService:
    """Get or create the global CogOps service instance"""
    global _cogops_service
    if _cogops_service is None:
        _cogops_service = CogOpsService()
    return _cogops_service
