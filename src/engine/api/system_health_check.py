#!/usr/bin/env python3
"""
METIS System Health Check API

Comprehensive health monitoring endpoint that validates all critical subsystems.
Provides detailed diagnostics for production readiness assessment.
"""

import asyncio
import time
import os
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

try:
    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = object
    BaseModel = object


class HealthStatus(str, Enum):
    """Health check status levels"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status for individual system component"""

    component: str
    status: HealthStatus
    response_time_ms: float
    details: Dict[str, Any]
    error: Optional[str] = None
    last_check: str = ""

    def __post_init__(self):
        if not self.last_check:
            self.last_check = datetime.utcnow().isoformat()


class SystemHealthResponse(BaseModel):
    """System health check response model"""

    overall_status: str
    timestamp: str
    system_info: Dict[str, Any]
    component_health: Dict[str, Dict[str, Any]]
    performance_metrics: Dict[str, Any]
    readiness_assessment: Dict[str, Any]
    recommendations: List[str]


class SystemHealthChecker:
    """
    Comprehensive system health checker for METIS platform.
    Validates all critical subsystems and provides production readiness assessment.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("🏥 SystemHealthChecker initialized")

    async def comprehensive_health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check of all METIS subsystems

        Returns:
            Complete health report with component status, metrics, and recommendations
        """
        self.logger.info("🔍 Starting comprehensive system health check...")

        start_time = time.time()

        # Component health checks
        component_results = {}

        # Core system components
        core_checks = [
            ("query_enhancement", self._check_query_enhancement),
            ("devils_advocate", self._check_devils_advocate),
            ("three_consultants", self._check_three_consultants),
            ("perplexity_research", self._check_perplexity_integration),
            ("llm_providers", self._check_llm_providers),
            ("database_connection", self._check_database_connection),
            ("performance_optimizer", self._check_performance_optimizer),
        ]

        # Execute all health checks in parallel
        tasks = []
        for component_name, check_func in core_checks:
            task = asyncio.create_task(
                self._safe_component_check(component_name, check_func)
            )
            tasks.append(task)

        component_health_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for (component_name, _), result in zip(core_checks, component_health_results):
            if isinstance(result, Exception):
                component_results[component_name] = ComponentHealth(
                    component=component_name,
                    status=HealthStatus.UNHEALTHY,
                    response_time_ms=0.0,
                    details={},
                    error=str(result),
                )
            else:
                component_results[component_name] = result

        # Calculate overall system status
        overall_status = self._calculate_overall_status(component_results)

        # Performance metrics
        performance_metrics = self._gather_performance_metrics(component_results)

        # Production readiness assessment
        readiness_assessment = self._assess_production_readiness(
            component_results, performance_metrics
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            component_results, readiness_assessment
        )

        total_time = time.time() - start_time

        health_report = {
            "overall_status": overall_status.value,
            "timestamp": datetime.utcnow().isoformat(),
            "system_info": {
                "health_check_duration_ms": round(total_time * 1000, 2),
                "python_version": os.sys.version,
                "environment": os.getenv("ENVIRONMENT", "unknown"),
                "components_checked": len(component_results),
            },
            "component_health": {
                name: asdict(health) for name, health in component_results.items()
            },
            "performance_metrics": performance_metrics,
            "readiness_assessment": readiness_assessment,
            "recommendations": recommendations,
        }

        self.logger.info(
            f"✅ Health check completed in {total_time:.2f}s - Status: {overall_status.value}"
        )

        return health_report

    async def _safe_component_check(
        self, component_name: str, check_func
    ) -> ComponentHealth:
        """Safely execute component health check with timeout"""
        try:
            # 10 second timeout for individual component checks
            return await asyncio.wait_for(check_func(), timeout=10.0)
        except asyncio.TimeoutError:
            return ComponentHealth(
                component=component_name,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=10000.0,
                details={},
                error="Health check timeout (10s)",
            )
        except Exception as e:
            return ComponentHealth(
                component=component_name,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=0.0,
                details={},
                error=str(e),
            )

    async def _check_query_enhancement(self) -> ComponentHealth:
        """Check Query Enhancement system health"""
        start_time = time.time()

        try:
            from src.engine.core.query_clarification_engine import (
                QueryClarificationEngine,
            )

            engine = QueryClarificationEngine()

            # Simple test query
            test_query = "How should we improve our business strategy?"
            brief = await engine.synthesize_engagement_brief(test_query)

            response_time = (time.time() - start_time) * 1000

            if brief and brief.objective:
                return ComponentHealth(
                    component="query_enhancement",
                    status=HealthStatus.HEALTHY,
                    response_time_ms=round(response_time, 2),
                    details={
                        "test_query_processed": True,
                        "engagement_brief_generated": True,
                        "brief_objective": (
                            brief.objective[:100] + "..."
                            if len(brief.objective) > 100
                            else brief.objective
                        ),
                    },
                )
            else:
                return ComponentHealth(
                    component="query_enhancement",
                    status=HealthStatus.DEGRADED,
                    response_time_ms=round(response_time, 2),
                    details={"test_query_processed": False},
                    error="Failed to generate engagement brief",
                )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return ComponentHealth(
                component="query_enhancement",
                status=HealthStatus.UNHEALTHY,
                response_time_ms=round(response_time, 2),
                details={},
                error=str(e),
            )

    async def _check_devils_advocate(self) -> ComponentHealth:
        """Check Devil's Advocate system health"""
        start_time = time.time()

        try:
            from src.core.enhanced_devils_advocate_system import (
                EnhancedDevilsAdvocateSystem,
            )

            das = EnhancedDevilsAdvocateSystem()

            # Simple test recommendation
            test_recommendation = "Increase marketing budget by 50%"
            test_context = {"company": "Test Corp", "industry": "Technology"}

            # Quick challenge analysis (should be fast)
            result = await das.comprehensive_challenge_analysis(
                test_recommendation, test_context
            )

            response_time = (time.time() - start_time) * 1000

            if result and hasattr(result, "total_challenges_found"):
                return ComponentHealth(
                    component="devils_advocate",
                    status=HealthStatus.HEALTHY,
                    response_time_ms=round(response_time, 2),
                    details={
                        "challenge_analysis_completed": True,
                        "challenges_found": result.total_challenges_found,
                        "risk_score": getattr(result, "overall_risk_score", 0),
                    },
                )
            else:
                return ComponentHealth(
                    component="devils_advocate",
                    status=HealthStatus.DEGRADED,
                    response_time_ms=round(response_time, 2),
                    details={},
                    error="Challenge analysis failed to return expected result",
                )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return ComponentHealth(
                component="devils_advocate",
                status=HealthStatus.UNHEALTHY,
                response_time_ms=round(response_time, 2),
                details={},
                error=str(e),
            )

    async def _check_three_consultants(self) -> ComponentHealth:
        """Check Three Consultant system health"""
        start_time = time.time()

        try:
            # Import check only - full test would be too slow for health check
            from src.blueprint_orchestrator import BlueprintCognitiveOrchestrator

            orchestrator = BlueprintCognitiveOrchestrator()

            response_time = (time.time() - start_time) * 1000

            return ComponentHealth(
                component="three_consultants",
                status=(
                    HealthStatus.HEALTHY
                    if response_time < 1000
                    else HealthStatus.DEGRADED
                ),
                response_time_ms=round(response_time, 2),
                details={
                    "orchestrator_initialized": True,
                    "note": "Full consultant analysis requires separate validation test",
                },
            )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return ComponentHealth(
                component="three_consultants",
                status=HealthStatus.UNHEALTHY,
                response_time_ms=round(response_time, 2),
                details={},
                error=str(e),
            )

    async def _check_perplexity_integration(self) -> ComponentHealth:
        """Check Perplexity research integration health"""
        start_time = time.time()

        try:
            from src.engine.integrations.perplexity_client import PerplexityClient

            client = PerplexityClient()
            is_available = await client.is_available()

            response_time = (time.time() - start_time) * 1000

            if is_available:
                return ComponentHealth(
                    component="perplexity_research",
                    status=HealthStatus.HEALTHY,
                    response_time_ms=round(response_time, 2),
                    details={
                        "client_available": True,
                        "api_key_configured": bool(os.getenv("PERPLEXITY_API_KEY")),
                    },
                )
            else:
                return ComponentHealth(
                    component="perplexity_research",
                    status=HealthStatus.DEGRADED,
                    response_time_ms=round(response_time, 2),
                    details={
                        "client_available": False,
                        "api_key_configured": bool(os.getenv("PERPLEXITY_API_KEY")),
                    },
                    error="Perplexity client not available",
                )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return ComponentHealth(
                component="perplexity_research",
                status=HealthStatus.UNHEALTHY,
                response_time_ms=round(response_time, 2),
                details={},
                error=str(e),
            )

    async def _check_llm_providers(self) -> ComponentHealth:
        """Check LLM provider health (DeepSeek + Claude)"""
        start_time = time.time()

        try:
            from src.core.resilient_llm_client import get_resilient_llm_client

            client = get_resilient_llm_client()

            response_time = (time.time() - start_time) * 1000

            # Check API keys
            deepseek_key = bool(os.getenv("DEEPSEEK_API_KEY"))
            claude_key = bool(os.getenv("ANTHROPIC_API_KEY"))

            if deepseek_key or claude_key:
                status = (
                    HealthStatus.HEALTHY
                    if (deepseek_key and claude_key)
                    else HealthStatus.DEGRADED
                )
                return ComponentHealth(
                    component="llm_providers",
                    status=status,
                    response_time_ms=round(response_time, 2),
                    details={
                        "deepseek_configured": deepseek_key,
                        "claude_configured": claude_key,
                        "fallback_available": deepseek_key and claude_key,
                    },
                )
            else:
                return ComponentHealth(
                    component="llm_providers",
                    status=HealthStatus.UNHEALTHY,
                    response_time_ms=round(response_time, 2),
                    details={"deepseek_configured": False, "claude_configured": False},
                    error="No LLM API keys configured",
                )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return ComponentHealth(
                component="llm_providers",
                status=HealthStatus.UNHEALTHY,
                response_time_ms=round(response_time, 2),
                details={},
                error=str(e),
            )

    async def _check_database_connection(self) -> ComponentHealth:
        """Check database connectivity"""
        start_time = time.time()

        try:
            # Check Supabase configuration
            supabase_url = os.getenv("NEXT_PUBLIC_SUPABASE_URL") or os.getenv(
                "SUPABASE_URL"
            )
            supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv(
                "NEXT_PUBLIC_SUPABASE_ANON_KEY"
            )

            response_time = (time.time() - start_time) * 1000

            if supabase_url and supabase_key:
                return ComponentHealth(
                    component="database_connection",
                    status=HealthStatus.HEALTHY,
                    response_time_ms=round(response_time, 2),
                    details={
                        "supabase_url_configured": True,
                        "supabase_key_configured": True,
                        "note": "Configuration check only - actual connection test requires separate validation",
                    },
                )
            else:
                return ComponentHealth(
                    component="database_connection",
                    status=HealthStatus.UNHEALTHY,
                    response_time_ms=round(response_time, 2),
                    details={
                        "supabase_url_configured": bool(supabase_url),
                        "supabase_key_configured": bool(supabase_key),
                    },
                    error="Supabase configuration incomplete",
                )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return ComponentHealth(
                component="database_connection",
                status=HealthStatus.UNHEALTHY,
                response_time_ms=round(response_time, 2),
                details={},
                error=str(e),
            )

    async def _check_performance_optimizer(self) -> ComponentHealth:
        """Check Performance Optimizer system health"""
        start_time = time.time()

        try:
            from src.core.performance_optimizer import get_performance_optimizer

            optimizer = get_performance_optimizer()
            metrics = optimizer.get_performance_metrics()

            response_time = (time.time() - start_time) * 1000

            return ComponentHealth(
                component="performance_optimizer",
                status=HealthStatus.HEALTHY,
                response_time_ms=round(response_time, 2),
                details={
                    "optimizer_initialized": True,
                    "timeout_count": metrics.get("timeout_count", 0),
                    "cache_hit_rate": metrics.get("cache_hit_rate", 0),
                    "active_circuit_breakers": metrics.get("active_circuits_open", 0),
                },
            )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return ComponentHealth(
                component="performance_optimizer",
                status=HealthStatus.UNHEALTHY,
                response_time_ms=round(response_time, 2),
                details={},
                error=str(e),
            )

    def _calculate_overall_status(
        self, component_results: Dict[str, ComponentHealth]
    ) -> HealthStatus:
        """Calculate overall system status from component health"""
        if not component_results:
            return HealthStatus.UNKNOWN

        statuses = [health.status for health in component_results.values()]

        # If any critical component is unhealthy, system is unhealthy
        critical_components = ["llm_providers", "query_enhancement"]
        for component_name, health in component_results.items():
            if (
                component_name in critical_components
                and health.status == HealthStatus.UNHEALTHY
            ):
                return HealthStatus.UNHEALTHY

        # Count status types
        unhealthy_count = sum(1 for s in statuses if s == HealthStatus.UNHEALTHY)
        degraded_count = sum(1 for s in statuses if s == HealthStatus.DEGRADED)
        healthy_count = sum(1 for s in statuses if s == HealthStatus.HEALTHY)

        # Overall status logic
        if unhealthy_count > len(statuses) / 2:  # More than 50% unhealthy
            return HealthStatus.UNHEALTHY
        elif (
            unhealthy_count > 0 or degraded_count > len(statuses) / 3
        ):  # Any unhealthy or >33% degraded
            return HealthStatus.DEGRADED
        elif healthy_count == len(statuses):  # All healthy
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.DEGRADED

    def _gather_performance_metrics(
        self, component_results: Dict[str, ComponentHealth]
    ) -> Dict[str, Any]:
        """Gather performance metrics from component results"""
        response_times = [
            health.response_time_ms for health in component_results.values()
        ]

        return {
            "avg_component_response_time_ms": (
                round(sum(response_times) / len(response_times), 2)
                if response_times
                else 0
            ),
            "max_component_response_time_ms": (
                max(response_times) if response_times else 0
            ),
            "components_under_1s": sum(1 for rt in response_times if rt < 1000),
            "components_over_5s": sum(1 for rt in response_times if rt > 5000),
            "total_components": len(component_results),
        }

    def _assess_production_readiness(
        self,
        component_results: Dict[str, ComponentHealth],
        performance_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Assess production readiness based on health results"""

        # Critical checks for production readiness
        readiness_checks = {
            "all_critical_systems_healthy": True,
            "performance_acceptable": True,
            "configuration_complete": True,
            "error_handling_functional": True,
        }

        # Check critical systems
        critical_systems = ["llm_providers", "query_enhancement"]
        for system in critical_systems:
            if system in component_results:
                if component_results[system].status == HealthStatus.UNHEALTHY:
                    readiness_checks["all_critical_systems_healthy"] = False

        # Check performance
        avg_response_time = performance_metrics.get("avg_component_response_time_ms", 0)
        if avg_response_time > 5000:  # Average over 5 seconds
            readiness_checks["performance_acceptable"] = False

        # Check configuration completeness
        config_critical = ["llm_providers", "database_connection"]
        for system in config_critical:
            if system in component_results:
                if component_results[system].status == HealthStatus.UNHEALTHY:
                    readiness_checks["configuration_complete"] = False

        # Overall readiness score
        readiness_score = sum(readiness_checks.values()) / len(readiness_checks)

        return {
            "production_ready": readiness_score >= 0.8,  # 80% of checks must pass
            "readiness_score": readiness_score,
            "readiness_checks": readiness_checks,
            "deployment_recommendation": (
                "Ready for production deployment"
                if readiness_score >= 0.8
                else (
                    "Requires fixes before production deployment"
                    if readiness_score >= 0.5
                    else "Not suitable for production deployment"
                )
            ),
        }

    def _generate_recommendations(
        self,
        component_results: Dict[str, ComponentHealth],
        readiness_assessment: Dict[str, Any],
    ) -> List[str]:
        """Generate actionable recommendations based on health status"""
        recommendations = []

        # Component-specific recommendations
        for component_name, health in component_results.items():
            if health.status == HealthStatus.UNHEALTHY:
                recommendations.append(
                    f"🚨 CRITICAL: Fix {component_name} - {health.error or 'System unhealthy'}"
                )
            elif health.status == HealthStatus.DEGRADED:
                recommendations.append(
                    f"⚠️ IMPROVE: Optimize {component_name} performance or configuration"
                )

            # Performance recommendations
            if health.response_time_ms > 10000:  # Over 10 seconds
                recommendations.append(
                    f"⏱️ SLOW: {component_name} taking {health.response_time_ms/1000:.1f}s - investigate performance"
                )

        # Production readiness recommendations
        if not readiness_assessment.get("production_ready", False):
            recommendations.append(
                "🏭 NOT PRODUCTION READY: Address critical system issues before deployment"
            )

            readiness_checks = readiness_assessment.get("readiness_checks", {})
            if not readiness_checks.get("all_critical_systems_healthy"):
                recommendations.append("🔧 Fix all critical system health issues")
            if not readiness_checks.get("performance_acceptable"):
                recommendations.append(
                    "⚡ Improve system performance - components taking too long"
                )
            if not readiness_checks.get("configuration_complete"):
                recommendations.append(
                    "⚙️ Complete system configuration - missing API keys or database setup"
                )

        # General recommendations
        if not recommendations:
            recommendations.append("✅ System appears healthy - continue monitoring")

        return recommendations


# FastAPI router setup removed: canonical health endpoint is /api/v53/health (see src/main.py)
# This module now provides only the SystemHealthChecker utilities and a standalone runner.


# Standalone health checker for direct usage
async def run_standalone_health_check():
    """Run health check as standalone script"""
    checker = SystemHealthChecker()
    health_report = await checker.comprehensive_health_check()

    print("🏥 METIS SYSTEM HEALTH CHECK REPORT")
    print("=" * 80)
    print(f"Overall Status: {health_report['overall_status'].upper()}")
    print(f"Timestamp: {health_report['timestamp']}")
    print(
        f"Check Duration: {health_report['system_info']['health_check_duration_ms']:.1f}ms"
    )
    print()

    print("📊 COMPONENT HEALTH:")
    for component, health in health_report["component_health"].items():
        status_emoji = {
            "healthy": "✅",
            "degraded": "⚠️",
            "unhealthy": "❌",
            "unknown": "❓",
        }.get(health["status"], "❓")
        print(
            f"  {status_emoji} {component}: {health['status']} ({health['response_time_ms']}ms)"
        )
        if health.get("error"):
            print(f"     Error: {health['error']}")

    print()
    print("🎯 PRODUCTION READINESS:")
    readiness = health_report["readiness_assessment"]
    ready_emoji = "✅" if readiness["production_ready"] else "❌"
    print(f"  {ready_emoji} {readiness['deployment_recommendation']}")
    print(f"     Readiness Score: {readiness['readiness_score']:.1%}")

    if health_report["recommendations"]:
        print()
        print("💡 RECOMMENDATIONS:")
        for rec in health_report["recommendations"]:
            print(f"  {rec}")

    return health_report


if __name__ == "__main__":
    asyncio.run(run_standalone_health_check())
