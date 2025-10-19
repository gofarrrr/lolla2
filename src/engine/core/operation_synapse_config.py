"""
Operation Synapse: Feature Flags and Monitoring Configuration

Controls rollout of new Operation Synapse components with gradual activation,
monitoring, and safe rollback capabilities.

Key Features:
- Feature flag system with percentage-based rollout
- Component health monitoring with circuit breakers
- Comprehensive logging and metrics collection
- Safe fallback mechanisms for all new components
- Environment-based configuration overrides
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from enum import Enum
from datetime import datetime, timedelta


class OperationSynapsePhase(Enum):
    """Rollout phases for Operation Synapse components"""

    DISABLED = "disabled"
    CANARY = "canary"  # 5% of traffic
    GRADUAL = "gradual"  # 25% of traffic
    MAJORITY = "majority"  # 75% of traffic
    FULL = "full"  # 100% of traffic


@dataclass
class ComponentConfig:
    """Configuration for individual Operation Synapse component"""

    enabled: bool = False
    rollout_percentage: float = 0.0
    phase: OperationSynapsePhase = OperationSynapsePhase.DISABLED
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 300  # 5 minutes
    failure_count: int = 0
    last_failure: Optional[datetime] = None
    health_check_interval: int = 60  # seconds
    fallback_enabled: bool = True


@dataclass
class OperationSynapseConfig:
    """Complete Operation Synapse configuration with feature flags and monitoring"""

    # Core component configurations
    state_machine_orchestrator: ComponentConfig = field(default_factory=ComponentConfig)
    complexity_assessor: ComponentConfig = field(default_factory=ComponentConfig)
    nway_vector_selector: ComponentConfig = field(default_factory=ComponentConfig)
    contradiction_resolver: ComponentConfig = field(default_factory=ComponentConfig)
    transparency_contradictions: ComponentConfig = field(
        default_factory=ComponentConfig
    )

    # Global settings
    monitoring_enabled: bool = True
    metrics_collection: bool = True
    audit_trail_enabled: bool = True
    performance_tracking: bool = True

    # Rollout control
    global_kill_switch: bool = False
    max_concurrent_experiments: int = 3
    experiment_duration_hours: int = 24

    # Safety thresholds
    max_error_rate: float = 0.05  # 5% error rate threshold
    max_latency_ms: int = 30000  # 30 second timeout
    min_success_rate: float = 0.90  # 90% success rate minimum


class OperationSynapseManager:
    """
    Manages Operation Synapse rollout with feature flags, monitoring, and safety controls.

    Implements conservative rollout strategy per architectural decisions:
    - Start with 5% canary deployment
    - Gradual increase based on success metrics
    - Circuit breakers for component failures
    - Automatic fallback to legacy systems
    """

    def __init__(self, config: Optional[OperationSynapseConfig] = None):
        self.logger = logging.getLogger(__name__)

        # Load configuration (with environment overrides)
        self.config = config or self._load_default_config()
        self._apply_environment_overrides()

        # Component registry for monitoring
        self.component_metrics: Dict[str, Dict[str, Any]] = {}
        self.component_health: Dict[str, bool] = {}

        # Initialize monitoring
        self._initialize_monitoring()

        self.logger.info("🎛️ Operation Synapse Manager initialized with feature flags")

    def _load_default_config(self) -> OperationSynapseConfig:
        """Load default configuration based on environment"""

        # Determine initial phase based on environment
        env = os.getenv("ENVIRONMENT", "development").lower()

        if env == "production":
            # Conservative production rollout
            phase = OperationSynapsePhase.DISABLED
            percentage = 0.0
        elif env == "staging":
            # More aggressive staging rollout
            phase = OperationSynapsePhase.CANARY
            percentage = 5.0
        else:
            # Development - enable for testing
            phase = OperationSynapsePhase.FULL
            percentage = 100.0

        # Create component configs
        component_config = ComponentConfig(
            enabled=(phase != OperationSynapsePhase.DISABLED),
            rollout_percentage=percentage,
            phase=phase,
            circuit_breaker_threshold=5,
            fallback_enabled=True,
        )

        return OperationSynapseConfig(
            state_machine_orchestrator=component_config,
            complexity_assessor=component_config,
            nway_vector_selector=component_config,
            contradiction_resolver=component_config,
            transparency_contradictions=component_config,
        )

    def _apply_environment_overrides(self) -> None:
        """Apply environment variable overrides to configuration"""

        # Global kill switch
        if os.getenv("OPERATION_SYNAPSE_DISABLED", "").lower() == "true":
            self.config.global_kill_switch = True
            self.logger.warning(
                "🛑 Operation Synapse globally disabled via environment"
            )

        # Component-specific overrides
        component_overrides = {
            "state_machine_orchestrator": "STATE_MACHINE_ENABLED",
            "complexity_assessor": "COMPLEXITY_ASSESSOR_ENABLED",
            "nway_vector_selector": "NWAY_SELECTOR_ENABLED",
            "contradiction_resolver": "CONTRADICTION_RESOLVER_ENABLED",
            "transparency_contradictions": "TRANSPARENCY_CONTRADICTIONS_ENABLED",
        }

        for component_name, env_var in component_overrides.items():
            env_value = os.getenv(env_var, "").lower()
            if env_value in ["true", "false"]:
                component = getattr(self.config, component_name)
                component.enabled = env_value == "true"
                self.logger.info(f"🎛️ {component_name} override: {component.enabled}")

        # Rollout percentage overrides
        rollout_percentage = os.getenv("OPERATION_SYNAPSE_ROLLOUT_PERCENTAGE")
        if rollout_percentage:
            try:
                percentage = float(rollout_percentage)
                if 0.0 <= percentage <= 100.0:
                    for component_name in component_overrides.keys():
                        component = getattr(self.config, component_name)
                        component.rollout_percentage = percentage
                    self.logger.info(
                        f"🎛️ Global rollout percentage set to {percentage}%"
                    )
            except ValueError:
                self.logger.warning(
                    f"⚠️ Invalid rollout percentage: {rollout_percentage}"
                )

    def _initialize_monitoring(self) -> None:
        """Initialize monitoring and metrics collection"""

        components = [
            "state_machine_orchestrator",
            "complexity_assessor",
            "nway_vector_selector",
            "contradiction_resolver",
            "transparency_contradictions",
        ]

        for component in components:
            self.component_metrics[component] = {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "average_latency_ms": 0.0,
                "last_success": None,
                "last_failure": None,
                "circuit_breaker_trips": 0,
            }
            self.component_health[component] = True

    def should_use_component(self, component_name: str, engagement_id: str) -> bool:
        """
        Determine if a component should be used based on feature flags and health.

        Args:
            component_name: Name of the component to check
            engagement_id: Unique identifier for consistent routing

        Returns:
            True if component should be used, False for fallback
        """

        # Global kill switch check
        if self.config.global_kill_switch:
            self.logger.debug(f"🛑 {component_name} blocked by global kill switch")
            return False

        # Get component configuration
        if not hasattr(self.config, component_name):
            self.logger.warning(f"⚠️ Unknown component: {component_name}")
            return False

        component_config = getattr(self.config, component_name)

        # Component enabled check
        if not component_config.enabled:
            self.logger.debug(f"🚫 {component_name} disabled by feature flag")
            return False

        # Circuit breaker check
        if not self.component_health.get(component_name, True):
            self.logger.warning(f"⚡ {component_name} blocked by circuit breaker")
            return False

        # Rollout percentage check (consistent hashing based on engagement_id)
        if component_config.rollout_percentage < 100.0:
            hash_value = hash(f"{engagement_id}_{component_name}") % 100
            if hash_value >= component_config.rollout_percentage:
                self.logger.debug(
                    f"🎲 {component_name} not selected for rollout "
                    f"(hash: {hash_value}, threshold: {component_config.rollout_percentage})"
                )
                return False

        self.logger.debug(f"✅ {component_name} enabled for engagement {engagement_id}")
        return True

    def record_component_success(self, component_name: str, latency_ms: float) -> None:
        """Record successful component execution"""

        if component_name not in self.component_metrics:
            return

        metrics = self.component_metrics[component_name]
        metrics["total_requests"] += 1
        metrics["successful_requests"] += 1
        metrics["last_success"] = datetime.utcnow()

        # Update average latency (simple moving average)
        total_requests = metrics["total_requests"]
        current_avg = metrics["average_latency_ms"]
        metrics["average_latency_ms"] = (
            current_avg * (total_requests - 1) + latency_ms
        ) / total_requests

        # Reset circuit breaker if component is healthy
        component_config = getattr(self.config, component_name)
        component_config.failure_count = 0
        self.component_health[component_name] = True

        if self.config.performance_tracking:
            self.logger.debug(
                f"📊 {component_name} success: {latency_ms:.1f}ms "
                f"(success rate: {self._get_success_rate(component_name):.1%})"
            )

    def record_component_failure(self, component_name: str, error: str) -> None:
        """Record component failure and potentially trip circuit breaker"""

        if component_name not in self.component_metrics:
            return

        metrics = self.component_metrics[component_name]
        component_config = getattr(self.config, component_name)

        # Update metrics
        metrics["total_requests"] += 1
        metrics["failed_requests"] += 1
        metrics["last_failure"] = datetime.utcnow()

        # Update component config
        component_config.failure_count += 1
        component_config.last_failure = datetime.utcnow()

        # Check circuit breaker threshold
        if component_config.failure_count >= component_config.circuit_breaker_threshold:
            self.component_health[component_name] = False
            metrics["circuit_breaker_trips"] += 1

            self.logger.error(
                f"⚡ Circuit breaker tripped for {component_name} "
                f"({component_config.failure_count} failures)"
            )

        self.logger.warning(
            f"❌ {component_name} failure: {error} "
            f"(failure count: {component_config.failure_count})"
        )

    def _get_success_rate(self, component_name: str) -> float:
        """Calculate success rate for component"""

        metrics = self.component_metrics.get(component_name, {})
        total = metrics.get("total_requests", 0)
        successful = metrics.get("successful_requests", 0)

        return successful / total if total > 0 else 1.0

    def reset_circuit_breaker(self, component_name: str) -> bool:
        """Manually reset circuit breaker for component"""

        if not hasattr(self.config, component_name):
            return False

        component_config = getattr(self.config, component_name)

        # Check if enough time has passed since last failure
        if component_config.last_failure:
            timeout = timedelta(seconds=component_config.circuit_breaker_timeout)
            if datetime.utcnow() - component_config.last_failure < timeout:
                self.logger.warning(
                    f"⏰ Circuit breaker reset denied for {component_name} "
                    f"(timeout not reached)"
                )
                return False

        # Reset circuit breaker
        component_config.failure_count = 0
        self.component_health[component_name] = True

        self.logger.info(f"🔄 Circuit breaker reset for {component_name}")
        return True

    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health report"""

        component_health = {}
        overall_health = True

        for component_name in self.component_metrics.keys():
            metrics = self.component_metrics[component_name]
            config = getattr(self.config, component_name)

            success_rate = self._get_success_rate(component_name)
            is_healthy = (
                self.component_health[component_name]
                and success_rate >= self.config.min_success_rate
                and metrics.get("average_latency_ms", 0) <= self.config.max_latency_ms
            )

            component_health[component_name] = {
                "enabled": config.enabled,
                "healthy": is_healthy,
                "phase": config.phase.value,
                "rollout_percentage": config.rollout_percentage,
                "success_rate": success_rate,
                "average_latency_ms": metrics.get("average_latency_ms", 0),
                "total_requests": metrics.get("total_requests", 0),
                "circuit_breaker_trips": metrics.get("circuit_breaker_trips", 0),
                "failure_count": config.failure_count,
            }

            if config.enabled and not is_healthy:
                overall_health = False

        return {
            "overall_health": overall_health,
            "global_kill_switch": self.config.global_kill_switch,
            "components": component_health,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def update_rollout_phase(
        self, component_name: str, new_phase: OperationSynapsePhase
    ) -> bool:
        """Update rollout phase for component"""

        if not hasattr(self.config, component_name):
            return False

        component_config = getattr(self.config, component_name)

        # Map phases to rollout percentages
        phase_percentages = {
            OperationSynapsePhase.DISABLED: 0.0,
            OperationSynapsePhase.CANARY: 5.0,
            OperationSynapsePhase.GRADUAL: 25.0,
            OperationSynapsePhase.MAJORITY: 75.0,
            OperationSynapsePhase.FULL: 100.0,
        }

        # Update configuration
        component_config.phase = new_phase
        component_config.rollout_percentage = phase_percentages[new_phase]
        component_config.enabled = new_phase != OperationSynapsePhase.DISABLED

        self.logger.info(
            f"🎛️ {component_name} rollout updated to {new_phase.value} "
            f"({component_config.rollout_percentage}%)"
        )

        return True

    def get_component_metrics(self, component_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed metrics for specific component"""

        if component_name not in self.component_metrics:
            return None

        metrics = self.component_metrics[component_name]
        config = getattr(self.config, component_name)

        return {
            "component": component_name,
            "configuration": {
                "enabled": config.enabled,
                "phase": config.phase.value,
                "rollout_percentage": config.rollout_percentage,
                "circuit_breaker_threshold": config.circuit_breaker_threshold,
            },
            "metrics": metrics,
            "health": {
                "is_healthy": self.component_health.get(component_name, False),
                "success_rate": self._get_success_rate(component_name),
                "failure_count": config.failure_count,
                "last_failure": (
                    config.last_failure.isoformat() if config.last_failure else None
                ),
            },
        }


# Global instance for shared configuration
_operation_synapse_manager: Optional[OperationSynapseManager] = None


def get_operation_synapse_manager() -> OperationSynapseManager:
    """Get global Operation Synapse manager instance"""
    global _operation_synapse_manager

    if _operation_synapse_manager is None:
        _operation_synapse_manager = OperationSynapseManager()

    return _operation_synapse_manager


def should_use_state_machine(engagement_id: str) -> bool:
    """Check if state machine orchestrator should be used"""
    manager = get_operation_synapse_manager()
    return manager.should_use_component("state_machine_orchestrator", engagement_id)


def should_use_complexity_assessor(engagement_id: str) -> bool:
    """Check if complexity assessor should be used"""
    manager = get_operation_synapse_manager()
    return manager.should_use_component("complexity_assessor", engagement_id)


def should_use_nway_vector_selector(engagement_id: str) -> bool:
    """Check if N-WAY vector selector should be used"""
    manager = get_operation_synapse_manager()
    return manager.should_use_component("nway_vector_selector", engagement_id)


def should_use_contradiction_resolver(engagement_id: str) -> bool:
    """Check if contradiction resolver should be used"""
    manager = get_operation_synapse_manager()
    return manager.should_use_component("contradiction_resolver", engagement_id)


def should_display_contradictions_in_transparency(engagement_id: str) -> bool:
    """Check if contradictions should be displayed in transparency layer"""
    manager = get_operation_synapse_manager()
    return manager.should_use_component("transparency_contradictions", engagement_id)
