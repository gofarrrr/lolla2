#!/usr/bin/env python3
"""
METIS Research Health Monitor - Proactive Research System Monitoring
Prevents cognitive collapse by ensuring research grounding availability

Author: METIS Cognitive Platform
Date: 2025
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ResearchHealthStatus(str, Enum):
    """Research system health status"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass
class ResearchHealthMetrics:
    """Health metrics for research systems"""

    status: ResearchHealthStatus
    response_time_ms: float
    success_rate: float
    last_success: Optional[datetime]
    last_failure: Optional[datetime]
    consecutive_failures: int
    total_attempts: int
    total_successes: int


class ResearchHealthMonitor:
    """Proactive monitoring of research system health"""

    def __init__(self):
        self.logger = logger
        self.perplexity_metrics = ResearchHealthMetrics(
            status=ResearchHealthStatus.UNKNOWN,
            response_time_ms=0.0,
            success_rate=0.0,
            last_success=None,
            last_failure=None,
            consecutive_failures=0,
            total_attempts=0,
            total_successes=0,
        )

        self.health_check_cache = {}
        self.cache_ttl = 300  # 5 minutes

    async def check_research_system_health(self) -> Dict[str, Any]:
        """
        Comprehensive health check of research systems

        Returns:
            Dict containing health status and recommendations
        """
        cache_key = "health_check"
        now = time.time()

        # Check cache
        if cache_key in self.health_check_cache:
            cached_result, cache_time = self.health_check_cache[cache_key]
            if now - cache_time < self.cache_ttl:
                return cached_result

        self.logger.info("🔍 Running research system health check...")

        health_report = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": ResearchHealthStatus.UNKNOWN,
            "perplexity_health": await self._check_perplexity_health(),
            "recommendations": [],
            "fallback_available": self._check_fallback_availability(),
        }

        # Determine overall status
        perplexity_status = health_report["perplexity_health"]["status"]

        if perplexity_status == ResearchHealthStatus.HEALTHY:
            health_report["overall_status"] = ResearchHealthStatus.HEALTHY
        elif perplexity_status == ResearchHealthStatus.DEGRADED:
            health_report["overall_status"] = ResearchHealthStatus.DEGRADED
            health_report["recommendations"].append(
                "Consider using cached research data"
            )
            health_report["recommendations"].append("Reduce research query frequency")
        else:
            health_report["overall_status"] = ResearchHealthStatus.FAILED
            health_report["recommendations"].append(
                "CRITICAL: Enable fallback mode immediately"
            )
            health_report["recommendations"].append("Alert system administrators")
            if not health_report["fallback_available"]:
                health_report["recommendations"].append(
                    "WARNING: No fallback available - system may fail"
                )

        # Cache result
        self.health_check_cache[cache_key] = (health_report, now)

        self.logger.info(f"✅ Health check complete: {health_report['overall_status']}")
        return health_report

    async def _check_perplexity_health(self) -> Dict[str, Any]:
        """Check Perplexity API health"""
        try:
            from src.engine.integrations.perplexity_client import PerplexityClient
            from src.engine.integrations.perplexity_client import KnowledgeQueryType

            client = PerplexityClient()
            start_time = time.time()

            # Simple health check query
            test_query = "What is the current date?"

            try:
                response = await asyncio.wait_for(
                    client.query_knowledge(
                        query=test_query,
                        query_type=KnowledgeQueryType.FACT_CHECKING,
                        max_tokens=50,
                    ),
                    timeout=10.0,  # Short timeout for health check
                )

                response_time = (time.time() - start_time) * 1000

                # Update metrics
                self.perplexity_metrics.total_attempts += 1
                self.perplexity_metrics.total_successes += 1
                self.perplexity_metrics.consecutive_failures = 0
                self.perplexity_metrics.last_success = datetime.now()
                self.perplexity_metrics.response_time_ms = response_time
                self.perplexity_metrics.success_rate = (
                    self.perplexity_metrics.total_successes
                    / self.perplexity_metrics.total_attempts
                )

                # Determine status based on response time
                if response_time < 5000:  # < 5 seconds
                    self.perplexity_metrics.status = ResearchHealthStatus.HEALTHY
                elif response_time < 15000:  # < 15 seconds
                    self.perplexity_metrics.status = ResearchHealthStatus.DEGRADED
                else:
                    self.perplexity_metrics.status = ResearchHealthStatus.DEGRADED

                return {
                    "status": self.perplexity_metrics.status,
                    "response_time_ms": response_time,
                    "success_rate": self.perplexity_metrics.success_rate,
                    "consecutive_failures": self.perplexity_metrics.consecutive_failures,
                    "sources_returned": (
                        len(response.sources) if hasattr(response, "sources") else 0
                    ),
                }

            except asyncio.TimeoutError:
                return await self._record_perplexity_failure("Timeout after 10 seconds")
            except Exception as e:
                return await self._record_perplexity_failure(str(e))

        except ImportError:
            return {
                "status": ResearchHealthStatus.FAILED,
                "error": "Perplexity client not available",
                "response_time_ms": 0,
                "success_rate": 0.0,
            }

    async def _record_perplexity_failure(self, error: str) -> Dict[str, Any]:
        """Record a Perplexity failure and update metrics"""
        self.perplexity_metrics.total_attempts += 1
        self.perplexity_metrics.consecutive_failures += 1
        self.perplexity_metrics.last_failure = datetime.now()
        self.perplexity_metrics.success_rate = (
            self.perplexity_metrics.total_successes
            / self.perplexity_metrics.total_attempts
        )

        # Determine status based on consecutive failures
        if self.perplexity_metrics.consecutive_failures >= 3:
            self.perplexity_metrics.status = ResearchHealthStatus.FAILED
        else:
            self.perplexity_metrics.status = ResearchHealthStatus.DEGRADED

        return {
            "status": self.perplexity_metrics.status,
            "error": error,
            "response_time_ms": 0,
            "success_rate": self.perplexity_metrics.success_rate,
            "consecutive_failures": self.perplexity_metrics.consecutive_failures,
        }

    def _check_fallback_availability(self) -> bool:
        """Check if fallback research options are available"""
        # Check for cached research data
        try:
            from pathlib import Path

            cache_dir = Path(__file__).parent.parent.parent / "data" / "research_cache"
            return cache_dir.exists() and len(list(cache_dir.glob("*.json"))) > 0
        except:
            return False

    def should_halt_on_research_failure(self) -> bool:
        """Determine if system should halt on research failure"""
        if self.perplexity_metrics.status == ResearchHealthStatus.FAILED:
            # Halt if no recent successes and no fallback
            time_since_success = None
            if self.perplexity_metrics.last_success:
                time_since_success = (
                    datetime.now() - self.perplexity_metrics.last_success
                )

            if (
                time_since_success is None or time_since_success > timedelta(hours=1)
            ) and not self._check_fallback_availability():
                return True

        return False

    def get_research_retry_strategy(self) -> Dict[str, Any]:
        """Get recommended retry strategy based on health metrics"""
        if self.perplexity_metrics.status == ResearchHealthStatus.HEALTHY:
            return {"max_retries": 3, "backoff_seconds": 1, "timeout_seconds": 30}
        elif self.perplexity_metrics.status == ResearchHealthStatus.DEGRADED:
            return {"max_retries": 2, "backoff_seconds": 5, "timeout_seconds": 60}
        else:
            return {"max_retries": 1, "backoff_seconds": 10, "timeout_seconds": 120}


# Global instance
_research_health_monitor = None


def get_research_health_monitor() -> ResearchHealthMonitor:
    """Get or create global research health monitor"""
    global _research_health_monitor
    if _research_health_monitor is None:
        _research_health_monitor = ResearchHealthMonitor()
    return _research_health_monitor
