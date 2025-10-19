"""
Circuit Breaker System - Phase 3.1
Error resilience and system protection through circuit breaker patterns.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import deque

# Centralized configuration (Clarity & Consolidation Sprint)
from src.config import get_settings

settings = get_settings()

# Import event coordinator for integration
try:
    from src.core.event_driven_coordinator import (
        get_event_coordinator,
        EventType,
        CoordinationEvent,
    )

    EVENT_COORDINATOR_AVAILABLE = True
except ImportError:
    EVENT_COORDINATOR_AVAILABLE = False

# Import orchestrator for integration
try:
    from src.core.consolidated_neural_lace_orchestrator import (
        get_consolidated_neural_lace_orchestrator,
    )

    ORCHESTRATOR_AVAILABLE = True
except ImportError:
    ORCHESTRATOR_AVAILABLE = False


class CircuitState(str, Enum):
    """States of a circuit breaker"""

    CLOSED = "closed"  # Normal operation, requests pass through
    OPEN = "open"  # Circuit tripped, requests fail fast
    HALF_OPEN = "half_open"  # Testing if service recovered


class FailureType(str, Enum):
    """Types of failures that can trip circuit"""

    TIMEOUT = "timeout"
    ERROR = "error"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    RATE_LIMIT = "rate_limit"
    DEPENDENCY_FAILURE = "dependency_failure"
    VALIDATION_ERROR = "validation_error"


@dataclass
class CircuitBreakerConfig:
    """Configuration for a circuit breaker"""

    name: str
    failure_threshold: int = (
        None  # Will use settings.circuit_breaker.failure_threshold if None
    )
    success_threshold: int = (
        None  # Will use settings.circuit_breaker.success_threshold if None
    )
    timeout_seconds: float = (
        None  # Will use settings.circuit_breaker.timeout_seconds if None
    )
    reset_timeout_seconds: float = (
        None  # Will use settings.circuit_breaker.reset_timeout_seconds if None
    )
    sliding_window_size: int = 100  # Size of sliding window for metrics
    failure_rate_threshold: float = 0.5  # Failure rate to trip circuit
    slow_call_duration: float = 10.0  # Duration to consider call slow
    slow_call_rate_threshold: float = 0.5  # Slow call rate to trip
    excluded_exceptions: List[type] = field(default_factory=list)
    fallback_handler: Optional[Callable] = None

    def __post_init__(self):
        """Apply defaults from centralized configuration"""
        if self.failure_threshold is None:
            self.failure_threshold = settings.circuit_breaker.failure_threshold
        if self.success_threshold is None:
            self.success_threshold = settings.circuit_breaker.success_threshold
        if self.timeout_seconds is None:
            self.timeout_seconds = settings.circuit_breaker.timeout_seconds
        if self.reset_timeout_seconds is None:
            self.reset_timeout_seconds = settings.circuit_breaker.reset_timeout_seconds


@dataclass
class CircuitMetrics:
    """Metrics for circuit breaker monitoring"""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    timeout_calls: int = 0
    slow_calls: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    state_changes: List[Tuple[CircuitState, datetime]] = field(default_factory=list)
    call_durations: deque = field(default_factory=lambda: deque(maxlen=100))
    error_types: Dict[str, int] = field(default_factory=dict)


class CircuitBreaker:
    """
    Circuit breaker implementation for individual components or services.
    Provides fail-fast behavior and automatic recovery detection.
    """

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.metrics = CircuitMetrics()
        self.last_state_change = datetime.now()
        self.half_open_test_running = False
        self.logger = logging.getLogger(f"{__name__}.{config.name}")

        # Sliding window for recent calls
        self.sliding_window = deque(maxlen=config.sliding_window_size)

        self.logger.info(f"Circuit breaker initialized: {config.name}")

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function through the circuit breaker.

        Args:
            func: Function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Result from function or fallback

        Raises:
            Exception if circuit is open and no fallback available
        """
        # Check circuit state
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                await self._transition_to_half_open()
            else:
                return await self._handle_open_circuit()

        # Execute the call
        start_time = time.time()
        try:
            # Add timeout protection
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(
                    func(*args, **kwargs), timeout=self.config.timeout_seconds
                )
            else:
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, func, *args, **kwargs
                    ),
                    timeout=self.config.timeout_seconds,
                )

            duration = time.time() - start_time
            await self._record_success(duration)

            return result

        except asyncio.TimeoutError as e:
            duration = time.time() - start_time
            await self._record_failure(FailureType.TIMEOUT, duration, e)
            raise

        except Exception as e:
            duration = time.time() - start_time

            # Check if exception should be excluded
            if type(e) in self.config.excluded_exceptions:
                await self._record_success(duration)
                raise

            await self._record_failure(FailureType.ERROR, duration, e)
            raise

    async def _record_success(self, duration: float):
        """Record a successful call"""
        self.metrics.total_calls += 1
        self.metrics.successful_calls += 1
        self.metrics.consecutive_successes += 1
        self.metrics.consecutive_failures = 0
        self.metrics.last_success_time = datetime.now()
        self.metrics.call_durations.append(duration)

        # Update sliding window
        self.sliding_window.append((True, duration, datetime.now()))

        # Check if call was slow
        if duration > self.config.slow_call_duration:
            self.metrics.slow_calls += 1

        # Handle state transitions
        if self.state == CircuitState.HALF_OPEN:
            if self.metrics.consecutive_successes >= self.config.success_threshold:
                await self._transition_to_closed()

        self.logger.debug(f"Success recorded: {duration:.2f}s")

    async def _record_failure(
        self, failure_type: FailureType, duration: float, exception: Exception
    ):
        """Record a failed call"""
        self.metrics.total_calls += 1
        self.metrics.failed_calls += 1
        self.metrics.consecutive_failures += 1
        self.metrics.consecutive_successes = 0
        self.metrics.last_failure_time = datetime.now()
        self.metrics.call_durations.append(duration)

        # Track error types
        error_type = type(exception).__name__
        self.metrics.error_types[error_type] = (
            self.metrics.error_types.get(error_type, 0) + 1
        )

        # Update sliding window
        self.sliding_window.append((False, duration, datetime.now()))

        # Track specific failure types
        if failure_type == FailureType.TIMEOUT:
            self.metrics.timeout_calls += 1

        # Check if circuit should open
        if self.state == CircuitState.CLOSED:
            if self._should_open_circuit():
                await self._transition_to_open()
        elif self.state == CircuitState.HALF_OPEN:
            # Single failure in half-open state reopens circuit
            await self._transition_to_open()

        self.logger.warning(f"Failure recorded: {failure_type.value} - {error_type}")

    def _should_open_circuit(self) -> bool:
        """Determine if circuit should open based on metrics"""
        # Check consecutive failures
        if self.metrics.consecutive_failures >= self.config.failure_threshold:
            return True

        # Check failure rate in sliding window
        if len(self.sliding_window) >= self.config.sliding_window_size:
            failures = sum(1 for success, _, _ in self.sliding_window if not success)
            failure_rate = failures / len(self.sliding_window)
            if failure_rate >= self.config.failure_rate_threshold:
                return True

        # Check slow call rate
        if len(self.sliding_window) >= self.config.sliding_window_size:
            slow_calls = sum(
                1
                for _, duration, _ in self.sliding_window
                if duration > self.config.slow_call_duration
            )
            slow_rate = slow_calls / len(self.sliding_window)
            if slow_rate >= self.config.slow_call_rate_threshold:
                return True

        return False

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.half_open_test_running:
            return False

        time_since_change = (datetime.now() - self.last_state_change).total_seconds()
        return time_since_change >= self.config.reset_timeout_seconds

    async def _transition_to_open(self):
        """Transition circuit to open state"""
        self.state = CircuitState.OPEN
        self.last_state_change = datetime.now()
        self.metrics.state_changes.append((CircuitState.OPEN, self.last_state_change))

        self.logger.error(f"Circuit OPENED: {self.config.name}")

        # Emit event if coordinator available
        if EVENT_COORDINATOR_AVAILABLE:
            coordinator = await get_event_coordinator()
            await coordinator.emit_event(
                EventType.COORDINATION_ALERT,
                source_component=f"circuit_breaker_{self.config.name}",
                payload={
                    "alert_type": "circuit_opened",
                    "circuit_name": self.config.name,
                    "consecutive_failures": self.metrics.consecutive_failures,
                    "failure_rate": self._calculate_failure_rate(),
                },
                priority=2,
            )

    async def _transition_to_half_open(self):
        """Transition circuit to half-open state"""
        self.state = CircuitState.HALF_OPEN
        self.last_state_change = datetime.now()
        self.metrics.state_changes.append(
            (CircuitState.HALF_OPEN, self.last_state_change)
        )
        self.metrics.consecutive_successes = 0
        self.metrics.consecutive_failures = 0

        self.logger.info(f"Circuit HALF-OPEN: {self.config.name}")

    async def _transition_to_closed(self):
        """Transition circuit to closed state"""
        self.state = CircuitState.CLOSED
        self.last_state_change = datetime.now()
        self.metrics.state_changes.append((CircuitState.CLOSED, self.last_state_change))

        self.logger.info(f"Circuit CLOSED: {self.config.name}")

        # Emit recovery event if coordinator available
        if EVENT_COORDINATOR_AVAILABLE:
            coordinator = await get_event_coordinator()
            await coordinator.emit_event(
                EventType.COMPONENT_COMPLETED,
                source_component=f"circuit_breaker_{self.config.name}",
                payload={
                    "event_type": "circuit_recovered",
                    "circuit_name": self.config.name,
                    "recovery_time": (
                        (
                            datetime.now() - self.metrics.last_failure_time
                        ).total_seconds()
                        if self.metrics.last_failure_time
                        else 0
                    ),
                },
                priority=5,
            )

    async def _handle_open_circuit(self):
        """Handle request when circuit is open"""
        self.logger.warning(f"Circuit open, failing fast: {self.config.name}")

        # Use fallback if available
        if self.config.fallback_handler:
            try:
                return await self.config.fallback_handler()
            except Exception as e:
                self.logger.error(f"Fallback handler failed: {e}")

        raise CircuitOpenException(f"Circuit breaker {self.config.name} is OPEN")

    def _calculate_failure_rate(self) -> float:
        """Calculate current failure rate"""
        if not self.sliding_window:
            return 0.0

        failures = sum(1 for success, _, _ in self.sliding_window if not success)
        return failures / len(self.sliding_window)

    def get_state(self) -> CircuitState:
        """Get current circuit state"""
        return self.state

    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics"""
        return {
            "name": self.config.name,
            "state": self.state.value,
            "total_calls": self.metrics.total_calls,
            "successful_calls": self.metrics.successful_calls,
            "failed_calls": self.metrics.failed_calls,
            "timeout_calls": self.metrics.timeout_calls,
            "slow_calls": self.metrics.slow_calls,
            "failure_rate": self._calculate_failure_rate(),
            "consecutive_failures": self.metrics.consecutive_failures,
            "consecutive_successes": self.metrics.consecutive_successes,
            "average_duration": (
                sum(self.metrics.call_durations) / len(self.metrics.call_durations)
                if self.metrics.call_durations
                else 0
            ),
            "error_types": dict(self.metrics.error_types),
        }

    async def reset(self):
        """Manually reset the circuit breaker"""
        self.state = CircuitState.CLOSED
        self.metrics = CircuitMetrics()
        self.last_state_change = datetime.now()
        self.sliding_window.clear()

        self.logger.info(f"Circuit manually reset: {self.config.name}")


class CircuitOpenException(Exception):
    """Exception raised when circuit is open"""

    pass


class CircuitBreakerManager:
    """
    Manages multiple circuit breakers for different components and services.
    Provides centralized monitoring and coordination.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.global_config = {
            "max_open_circuits": 3,  # Maximum circuits that can be open simultaneously
            "cascade_protection": True,  # Protect dependent services when upstream fails
            "health_check_interval": 30.0,  # Seconds between health checks
            "auto_recovery": True,  # Automatically attempt recovery
        }

        # Integration with orchestrator and event coordinator
        self.orchestrator_integrated = ORCHESTRATOR_AVAILABLE
        self.event_coordinator_integrated = EVENT_COORDINATOR_AVAILABLE

        # Health monitoring
        self.health_check_task: Optional[asyncio.Task] = None
        self.monitoring_active = False

        self.logger.info("Circuit breaker manager initialized")

    def register_circuit(self, config: CircuitBreakerConfig) -> CircuitBreaker:
        """Register a new circuit breaker"""
        if config.name in self.circuit_breakers:
            self.logger.warning(f"Circuit breaker {config.name} already registered")
            return self.circuit_breakers[config.name]

        circuit = CircuitBreaker(config)
        self.circuit_breakers[config.name] = circuit

        self.logger.info(f"Registered circuit breaker: {config.name}")
        return circuit

    def get_circuit(self, name: str) -> Optional[CircuitBreaker]:
        """Get a circuit breaker by name"""
        return self.circuit_breakers.get(name)

    async def call_with_circuit(
        self, circuit_name: str, func: Callable, *args, **kwargs
    ) -> Any:
        """Execute a function through a named circuit breaker"""
        circuit = self.get_circuit(circuit_name)
        if not circuit:
            raise ValueError(f"Circuit breaker {circuit_name} not found")

        return await circuit.call(func, *args, **kwargs)

    def get_all_states(self) -> Dict[str, CircuitState]:
        """Get states of all circuit breakers"""
        return {
            name: circuit.get_state() for name, circuit in self.circuit_breakers.items()
        }

    def get_open_circuits(self) -> List[str]:
        """Get list of open circuits"""
        return [
            name
            for name, circuit in self.circuit_breakers.items()
            if circuit.get_state() == CircuitState.OPEN
        ]

    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health based on circuit states"""
        states = self.get_all_states()
        open_circuits = self.get_open_circuits()

        # Calculate health score
        total_circuits = len(self.circuit_breakers)
        open_count = len(open_circuits)
        health_score = (
            (total_circuits - open_count) / total_circuits
            if total_circuits > 0
            else 1.0
        )

        # Determine system status
        if open_count == 0:
            status = "healthy"
        elif open_count <= self.global_config["max_open_circuits"]:
            status = "degraded"
        else:
            status = "critical"

        return {
            "status": status,
            "health_score": health_score,
            "total_circuits": total_circuits,
            "open_circuits": open_circuits,
            "circuit_states": {name: state.value for name, state in states.items()},
            "cascade_protection": self.global_config["cascade_protection"],
            "auto_recovery": self.global_config["auto_recovery"],
        }

    async def start_health_monitoring(self):
        """Start health monitoring task"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.health_check_task = asyncio.create_task(self._health_monitoring_loop())
            self.logger.info("Health monitoring started")

    async def stop_health_monitoring(self):
        """Stop health monitoring task"""
        if self.monitoring_active:
            self.monitoring_active = False
            if self.health_check_task:
                self.health_check_task.cancel()
                try:
                    await self.health_check_task
                except asyncio.CancelledError:
                    pass
            self.logger.info("Health monitoring stopped")

    async def _health_monitoring_loop(self):
        """Periodic health monitoring"""
        while self.monitoring_active:
            try:
                await asyncio.sleep(self.global_config["health_check_interval"])

                # Check system health
                health = self.get_system_health()

                # Emit health event if coordinator available
                if self.event_coordinator_integrated:
                    coordinator = await get_event_coordinator()
                    await coordinator.emit_event(
                        EventType.METRICS_UPDATE,
                        source_component="circuit_breaker_manager",
                        payload={
                            "health_status": health["status"],
                            "health_score": health["health_score"],
                            "open_circuits": health["open_circuits"],
                        },
                    )

                # Handle cascade protection if needed
                if (
                    self.global_config["cascade_protection"]
                    and health["status"] == "critical"
                ):
                    await self._handle_cascade_protection()

            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")

    async def _handle_cascade_protection(self):
        """Handle cascade protection when system is critical"""
        self.logger.warning("Cascade protection activated due to critical system state")

        # Implement cascade protection logic
        # This could involve opening downstream circuits, reducing load, etc.
        for name, circuit in self.circuit_breakers.items():
            if circuit.get_state() == CircuitState.CLOSED:
                # Check if this circuit depends on open circuits
                # For now, we'll use a simple heuristic
                if self._should_protect_circuit(name):
                    await circuit._transition_to_open()
                    self.logger.info(f"Cascade protection: Opened circuit {name}")

    def _should_protect_circuit(self, circuit_name: str) -> bool:
        """Determine if a circuit should be protected in cascade"""
        # Simple heuristic: protect if more than half of circuits are open
        open_count = len(self.get_open_circuits())
        total_count = len(self.circuit_breakers)
        return open_count > total_count / 2

    async def reset_all_circuits(self):
        """Reset all circuit breakers"""
        for circuit in self.circuit_breakers.values():
            await circuit.reset()

        self.logger.info("All circuits reset")

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary metrics for all circuit breakers"""
        all_metrics = {}
        total_calls = 0
        total_failures = 0
        total_timeouts = 0

        for name, circuit in self.circuit_breakers.items():
            metrics = circuit.get_metrics()
            all_metrics[name] = metrics
            total_calls += metrics["total_calls"]
            total_failures += metrics["failed_calls"]
            total_timeouts += metrics["timeout_calls"]

        return {
            "circuit_metrics": all_metrics,
            "total_calls": total_calls,
            "total_failures": total_failures,
            "total_timeouts": total_timeouts,
            "overall_failure_rate": (
                total_failures / total_calls if total_calls > 0 else 0
            ),
            "system_health": self.get_system_health(),
        }


# Global circuit breaker manager instance
_manager_instance = None


def get_circuit_manager() -> CircuitBreakerManager:
    """Get the global circuit breaker manager instance"""
    global _manager_instance

    if _manager_instance is None:
        _manager_instance = CircuitBreakerManager()

    return _manager_instance


# Decorator for circuit breaker protection
def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    timeout_seconds: float = 30.0,
    fallback_handler: Optional[Callable] = None,
):
    """
    Decorator to add circuit breaker protection to a function.

    Args:
        name: Name of the circuit breaker
        failure_threshold: Number of failures before opening
        timeout_seconds: Timeout for operations
        fallback_handler: Optional fallback function
    """

    def decorator(func):
        async def wrapper(*args, **kwargs):
            manager = get_circuit_manager()

            # Get or create circuit
            circuit = manager.get_circuit(name)
            if not circuit:
                config = CircuitBreakerConfig(
                    name=name,
                    failure_threshold=failure_threshold,
                    timeout_seconds=timeout_seconds,
                    fallback_handler=fallback_handler,
                )
                circuit = manager.register_circuit(config)

            # Execute through circuit breaker
            return await circuit.call(func, *args, **kwargs)

        return wrapper

    return decorator


# Week 2 Day 5: Manual Override API compatibility
def get_circuit_breaker():
    """Get circuit breaker for manual override API compatibility"""
    manager = get_circuit_manager()

    # Create system-level circuit breaker if it doesn't exist
    system_circuit = manager.get_circuit("system")
    if not system_circuit:
        config = CircuitBreakerConfig(
            name="system", failure_threshold=10, timeout_seconds=60.0
        )
        system_circuit = manager.register_circuit(config)

    return system_circuit
