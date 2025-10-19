"""
METIS Flywheel System - Graceful Degradation Manager

Enhanced graceful degradation for missing dependencies and system failures.
Implements fallback strategies to maintain core functionality even when
external services or optional components are unavailable.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from functools import wraps
import json

logger = logging.getLogger(__name__)


class DegradationMode(Enum):
    """Available degradation modes for different scenarios"""

    FULL_OPERATION = "full_operation"  # All systems operational
    REDIS_UNAVAILABLE = "redis_unavailable"  # Redis cache unavailable
    DATABASE_LIMITED = "database_limited"  # Database with limited access
    LEARNING_DISABLED = "learning_disabled"  # Learning components disabled
    MINIMAL_MODE = "minimal_mode"  # Only core functionality
    EMERGENCY_MODE = "emergency_mode"  # Maximum degradation


@dataclass
class DegradationStrategy:
    """Configuration for graceful degradation strategy"""

    mode: DegradationMode
    description: str
    fallback_operations: Dict[str, Any]
    retry_strategy: Dict[str, Any] = field(default_factory=dict)
    health_check_interval: int = 30  # seconds
    auto_recovery: bool = True
    notification_required: bool = False


class GracefulDegradationManager:
    """
    Central manager for graceful degradation across the Flywheel system.

    Monitors component health and automatically switches to appropriate
    fallback modes when dependencies become unavailable.
    """

    def __init__(self):
        self.current_mode = DegradationMode.FULL_OPERATION
        self.component_health: Dict[str, bool] = {}
        self.degradation_history: List[Dict[str, Any]] = []
        self.strategy_registry: Dict[DegradationMode, DegradationStrategy] = {}
        self.active_fallbacks: Dict[str, Any] = {}

        # Initialize degradation strategies
        self._initialize_degradation_strategies()

        # Health monitoring
        self.last_health_check = datetime.utcnow()
        self.health_check_interval = 30  # seconds

        logger.info("GracefulDegradationManager initialized")

    def _initialize_degradation_strategies(self):
        """Initialize all degradation strategies"""

        # Redis unavailable strategy
        self.strategy_registry[DegradationMode.REDIS_UNAVAILABLE] = DegradationStrategy(
            mode=DegradationMode.REDIS_UNAVAILABLE,
            description="Redis cache unavailable - using in-memory cache only",
            fallback_operations={
                "cache_type": "memory_only",
                "max_cache_size": 1000,
                "eviction_policy": "lru",
                "persistence": "none",
            },
            retry_strategy={
                "max_retries": 3,
                "backoff_multiplier": 2,
                "retry_interval": 5,
            },
            health_check_interval=15,
            auto_recovery=True,
            notification_required=False,
        )

        # Database limited strategy
        self.strategy_registry[DegradationMode.DATABASE_LIMITED] = DegradationStrategy(
            mode=DegradationMode.DATABASE_LIMITED,
            description="Database with limited access - local storage fallback",
            fallback_operations={
                "storage_type": "local_files",
                "data_retention": "24h",
                "sync_on_recovery": True,
                "read_only_mode": False,
            },
            retry_strategy={
                "max_retries": 5,
                "backoff_multiplier": 1.5,
                "retry_interval": 10,
            },
            health_check_interval=20,
            auto_recovery=True,
            notification_required=True,
        )

        # Learning disabled strategy
        self.strategy_registry[DegradationMode.LEARNING_DISABLED] = DegradationStrategy(
            mode=DegradationMode.LEARNING_DISABLED,
            description="Learning components disabled - using static configurations",
            fallback_operations={
                "learning_mode": "disabled",
                "use_static_recommendations": True,
                "pattern_detection": "basic",
                "feedback_collection": "store_only",
            },
            retry_strategy={
                "max_retries": 2,
                "backoff_multiplier": 3,
                "retry_interval": 30,
            },
            health_check_interval=60,
            auto_recovery=True,
            notification_required=False,
        )

        # Minimal mode strategy
        self.strategy_registry[DegradationMode.MINIMAL_MODE] = DegradationStrategy(
            mode=DegradationMode.MINIMAL_MODE,
            description="Minimal functionality - core operations only",
            fallback_operations={
                "cache_disabled": True,
                "learning_disabled": True,
                "monitoring_reduced": True,
                "phantom_detection": "basic",
                "performance_tracking": "minimal",
            },
            retry_strategy={
                "max_retries": 1,
                "backoff_multiplier": 5,
                "retry_interval": 60,
            },
            health_check_interval=120,
            auto_recovery=False,
            notification_required=True,
        )

    async def check_component_health(
        self, component_name: str, health_check_func: Callable
    ) -> bool:
        """Check health of a specific component"""
        try:
            is_healthy = await health_check_func()
            previous_health = self.component_health.get(component_name, True)

            # Update health status
            self.component_health[component_name] = is_healthy

            # Log health changes
            if previous_health != is_healthy:
                status = "HEALTHY" if is_healthy else "UNHEALTHY"
                logger.info(f"Component {component_name} status changed to {status}")

                # Record health change
                self.degradation_history.append(
                    {
                        "timestamp": datetime.utcnow().isoformat(),
                        "component": component_name,
                        "status": status,
                        "previous_status": (
                            "HEALTHY" if previous_health else "UNHEALTHY"
                        ),
                    }
                )

            return is_healthy

        except Exception as e:
            logger.error(f"Health check failed for {component_name}: {e}")
            self.component_health[component_name] = False
            return False

    def determine_degradation_mode(self) -> DegradationMode:
        """Determine appropriate degradation mode based on component health"""
        unhealthy_components = [
            name for name, health in self.component_health.items() if not health
        ]

        if not unhealthy_components:
            return DegradationMode.FULL_OPERATION

        # Check for specific degradation scenarios
        if "redis" in unhealthy_components and "database" not in unhealthy_components:
            return DegradationMode.REDIS_UNAVAILABLE

        if "database" in unhealthy_components and len(unhealthy_components) == 1:
            return DegradationMode.DATABASE_LIMITED

        if (
            "learning_orchestrator" in unhealthy_components
            or "flywheel_manager" in unhealthy_components
        ):
            return DegradationMode.LEARNING_DISABLED

        # Multiple critical components down
        if len(unhealthy_components) >= 3:
            return DegradationMode.MINIMAL_MODE

        # Default to learning disabled for other scenarios
        return DegradationMode.LEARNING_DISABLED

    async def apply_degradation_mode(self, mode: DegradationMode) -> bool:
        """Apply specific degradation mode configuration"""
        if mode == self.current_mode:
            return True  # Already in this mode

        strategy = self.strategy_registry.get(mode)
        if not strategy:
            logger.error(f"No strategy found for degradation mode: {mode}")
            return False

        try:
            # Apply fallback operations
            self.active_fallbacks = strategy.fallback_operations.copy()

            # Update current mode
            previous_mode = self.current_mode
            self.current_mode = mode

            # Log mode change
            logger.warning(
                f"Degradation mode changed: {previous_mode.value} → {mode.value}"
            )
            logger.info(f"Degradation strategy: {strategy.description}")

            # Record mode change
            self.degradation_history.append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "action": "mode_change",
                    "previous_mode": previous_mode.value,
                    "new_mode": mode.value,
                    "strategy": strategy.description,
                    "fallback_operations": self.active_fallbacks,
                }
            )

            # Send notification if required
            if strategy.notification_required:
                await self._send_degradation_notification(strategy)

            return True

        except Exception as e:
            logger.error(f"Failed to apply degradation mode {mode.value}: {e}")
            return False

    async def _send_degradation_notification(self, strategy: DegradationStrategy):
        """Send notification about degradation mode activation"""
        # This would integrate with notification systems
        logger.critical(f"DEGRADATION ALERT: {strategy.description}")
        logger.critical(f"Mode: {strategy.mode.value}")
        logger.critical(
            f"Auto-recovery: {'enabled' if strategy.auto_recovery else 'disabled'}"
        )

    def get_fallback_config(self, operation: str) -> Any:
        """Get fallback configuration for a specific operation"""
        return self.active_fallbacks.get(operation, None)

    def is_operation_available(self, operation: str) -> bool:
        """Check if a specific operation is available in current degradation mode"""
        if self.current_mode == DegradationMode.FULL_OPERATION:
            return True

        # Check operation-specific availability
        disabled_operations = {
            DegradationMode.REDIS_UNAVAILABLE: ["distributed_cache"],
            DegradationMode.DATABASE_LIMITED: ["complex_queries", "bulk_operations"],
            DegradationMode.LEARNING_DISABLED: [
                "pattern_learning",
                "user_adaptation",
                "recommendation_updates",
            ],
            DegradationMode.MINIMAL_MODE: [
                "caching",
                "learning",
                "advanced_monitoring",
                "phantom_detection",
            ],
        }

        return operation not in disabled_operations.get(self.current_mode, [])

    async def attempt_recovery(self) -> bool:
        """Attempt to recover from current degradation mode"""
        if self.current_mode == DegradationMode.FULL_OPERATION:
            return True

        strategy = self.strategy_registry.get(self.current_mode)
        if not strategy or not strategy.auto_recovery:
            return False

        # Re-check component health
        recovery_possible = True
        for component_name in self.component_health.keys():
            # This would trigger actual health checks
            logger.debug(f"Checking recovery possibility for {component_name}")

        if recovery_possible:
            # Attempt to return to full operation
            new_mode = self.determine_degradation_mode()
            if new_mode != self.current_mode:
                await self.apply_degradation_mode(new_mode)
                return True

        return False

    def get_status_summary(self) -> Dict[str, Any]:
        """Get current degradation status summary"""
        return {
            "current_mode": self.current_mode.value,
            "component_health": self.component_health.copy(),
            "active_fallbacks": self.active_fallbacks.copy(),
            "last_health_check": self.last_health_check.isoformat(),
            "degradation_events": len(self.degradation_history),
            "recent_changes": (
                self.degradation_history[-5:] if self.degradation_history else []
            ),
        }


# Global degradation manager instance
_degradation_manager: Optional[GracefulDegradationManager] = None


def get_degradation_manager() -> GracefulDegradationManager:
    """Get or create global degradation manager instance"""
    global _degradation_manager

    if _degradation_manager is None:
        _degradation_manager = GracefulDegradationManager()

    return _degradation_manager


def with_graceful_degradation(component_name: str, fallback_result: Any = None):
    """Decorator for graceful degradation of function calls"""

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            degradation_manager = get_degradation_manager()

            if not degradation_manager.is_operation_available(component_name):
                logger.info(
                    f"Operation {component_name} unavailable in current mode, using fallback"
                )
                return fallback_result

            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.warning(
                    f"Operation {component_name} failed: {e}, using fallback"
                )
                return fallback_result

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            degradation_manager = get_degradation_manager()

            if not degradation_manager.is_operation_available(component_name):
                logger.info(
                    f"Operation {component_name} unavailable in current mode, using fallback"
                )
                return fallback_result

            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(
                    f"Operation {component_name} failed: {e}, using fallback"
                )
                return fallback_result

        # Return appropriate wrapper based on function type
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


# Health check helpers
async def redis_health_check() -> bool:
    """Health check for Redis connectivity"""
    try:
        import redis.asyncio as redis

        client = redis.from_url("redis://localhost:6379")
        await client.ping()
        return True
    except Exception:
        return False


async def database_health_check() -> bool:
    """Health check for database connectivity"""
    try:
        # This would implement actual database health check
        return True
    except Exception:
        return False


async def learning_orchestrator_health_check() -> bool:
    """Health check for learning orchestrator"""
    try:
        # This would check learning orchestrator functionality
        return True
    except Exception:
        return False


if __name__ == "__main__":

    async def demo():
        """Demonstrate graceful degradation functionality"""
        manager = get_degradation_manager()

        print("🚀 METIS Flywheel Graceful Degradation Demo")
        print("=" * 50)

        # Check initial status
        status = manager.get_status_summary()
        print(f"Current mode: {status['current_mode']}")

        # Simulate Redis failure
        await manager.check_component_health("redis", redis_health_check)
        manager.component_health["redis"] = False  # Force failure for demo

        # Determine and apply degradation mode
        new_mode = manager.determine_degradation_mode()
        await manager.apply_degradation_mode(new_mode)

        print(f"New mode after Redis failure: {manager.current_mode.value}")
        print(
            f"Cache operation available: {manager.is_operation_available('distributed_cache')}"
        )
        print(f"Fallback cache config: {manager.get_fallback_config('cache_type')}")

        # Show final status
        final_status = manager.get_status_summary()
        print("\n📊 Final Status:")
        print(json.dumps(final_status, indent=2))

    asyncio.run(demo())
