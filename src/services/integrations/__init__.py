"""
METIS V5 Support System Integrations
====================================

Integration services for connecting V5 modular architecture with support systems:
- Flywheel System Integration - Memory consolidation and continuous learning
- Benchmarking Harness Integration (Project Monte Carlo) - Performance validation
- User-Facing Analytics Service - Real-time dashboards and business intelligence

Part of the Final Mandate - completing the V5 ecosystem integration.
"""

from .flywheel_integration_service import (
    FlywheelIntegrationService,
    get_flywheel_integration_service,
    FLYWHEEL_SERVICE_INFO,
)

from .benchmarking_harness_service import (
    ProjectMonteCarloService,
    get_benchmarking_harness_service,
    BENCHMARKING_SERVICE_INFO,
)

from .user_analytics_service import (
    UserAnalyticsService,
    get_user_analytics_service,
    ANALYTICS_SERVICE_INFO,
)

__all__ = [
    # Flywheel Integration
    "FlywheelIntegrationService",
    "get_flywheel_integration_service",
    "FLYWHEEL_SERVICE_INFO",
    # Benchmarking Harness (Project Monte Carlo)
    "ProjectMonteCarloService",
    "get_benchmarking_harness_service",
    "BENCHMARKING_SERVICE_INFO",
    # User Analytics Service
    "UserAnalyticsService",
    "get_user_analytics_service",
    "ANALYTICS_SERVICE_INFO",
]

# Integration cluster information
INTEGRATION_CLUSTER_INFO = {
    "cluster_name": "V5 Support Systems Integration",
    "services_count": 3,
    "services": {
        "flywheel_integration": FLYWHEEL_SERVICE_INFO,
        "benchmarking_harness": BENCHMARKING_SERVICE_INFO,
        "user_analytics": ANALYTICS_SERVICE_INFO,
    },
    "integration_capabilities": [
        "flywheel_memory_consolidation",
        "monte_carlo_performance_testing",
        "real_time_analytics_dashboards",
        "continuous_learning_integration",
        "statistical_validation",
        "user_engagement_tracking",
    ],
    "cluster_version": "1.0.0",
    "production_ready": True,
}


def get_all_integration_services():
    """Get all integration services as a collection."""
    return {
        "flywheel_integration": get_flywheel_integration_service(),
        "benchmarking_harness": get_benchmarking_harness_service(),
        "user_analytics": get_user_analytics_service(),
    }


async def get_integration_cluster_health():
    """Get comprehensive health status of the integration cluster."""
    try:
        services = get_all_integration_services()

        health_statuses = {}
        for service_name, service in services.items():
            if hasattr(service, "get_flywheel_health_status"):
                health_statuses[service_name] = (
                    await service.get_flywheel_health_status()
                )
            elif hasattr(service, "get_benchmarking_service_health"):
                health_statuses[service_name] = (
                    await service.get_benchmarking_service_health()
                )
            elif hasattr(service, "get_analytics_service_health"):
                health_statuses[service_name] = (
                    await service.get_analytics_service_health()
                )
            else:
                health_statuses[service_name] = {"status": "unknown"}

        # Calculate overall cluster health
        health_scores = [
            status.get("overall_health_score", 0)
            for status in health_statuses.values()
            if "overall_health_score" in status
        ]

        overall_health_score = (
            sum(health_scores) / len(health_scores) if health_scores else 0
        )

        return {
            "cluster_name": "V5 Support Systems Integration",
            "overall_health_score": overall_health_score,
            "cluster_status": (
                "healthy"
                if overall_health_score >= 80
                else "degraded" if overall_health_score >= 60 else "unhealthy"
            ),
            "services_count": len(services),
            "services_health": health_statuses,
            "integration_capabilities": INTEGRATION_CLUSTER_INFO[
                "integration_capabilities"
            ],
            "last_health_check": None,  # Will be set by caller
        }

    except Exception as e:
        return {
            "cluster_name": "V5 Support Systems Integration",
            "cluster_status": "error",
            "error": str(e),
            "last_health_check": None,
        }
