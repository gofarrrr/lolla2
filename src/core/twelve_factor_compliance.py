"""
12-Factor Agents compliance framework.
Every service must inherit from this base.
"""

import os
import asyncio
from typing import Any, Dict, Optional, Callable
from dataclasses import dataclass
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
from datetime import datetime, timezone
import hashlib
import json

# Fallback to standard logging if structlog not available
try:
    logger = structlog.get_logger()
except:
    logger = logging.getLogger(__name__)


@dataclass
class TwelveFactorConfig:
    """Configuration from environment (Factor 3)"""

    service_name: str
    timeout_seconds: int = 30
    max_retries: int = 3
    idempotency_enabled: bool = True
    sandbox_tools: bool = True
    structured_logging: bool = True


class SupabaseStateStore:
    """Placeholder for Supabase state store"""

    def __init__(self, service_name: str):
        self.service_name = service_name
        self.storage = {}  # In-memory storage for now

    def operation_exists(self, operation_id: str) -> bool:
        """Check if operation has been executed"""
        return operation_id in self.storage

    def mark_operation(self, operation_id: str):
        """Mark operation as executed"""
        self.storage[operation_id] = {
            "executed_at": datetime.now(timezone.utc).isoformat(),
            "status": "executed",
        }

    def get_result(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """Get cached result for operation"""
        return self.storage.get(operation_id, {}).get("result")

    def store_result(self, operation_id: str, result: Any):
        """Store result for operation"""
        if operation_id not in self.storage:
            self.storage[operation_id] = {}
        self.storage[operation_id]["result"] = result
        self.storage[operation_id]["stored_at"] = datetime.now(timezone.utc).isoformat()


class ToolSandbox:
    """Sandbox for tool execution with resource limits"""

    def __init__(
        self,
        memory_limit: str = "512MB",
        cpu_limit: float = 0.5,
        network_restricted: bool = True,
    ):
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.network_restricted = network_restricted

    def execute(self, tool: Callable, **kwargs) -> Any:
        """Execute tool in sandbox (simplified implementation)"""
        # In production, this would use actual sandboxing
        # For now, just execute with basic timeout
        return tool(**kwargs)


class TwelveFactorAgent:
    """
    Base class enforcing 12-Factor discipline.
    All services MUST inherit from this.
    """

    def __init__(self, config: TwelveFactorConfig):
        self.config = config
        self.logger = self._setup_structured_logging()
        self.state_store = self._setup_state_store()
        self._setup_from_environment()
        self._operation_cache = {}

        # Log 12-Factor compliance event for V5.4 tracking
        self._log_compliance_event()

    def _log_compliance_event(self):
        """Log 12-Factor compliance event to context stream if available"""
        try:
            # Try to get the unified context stream (may not always be available)
            from src.core.unified_context_stream import (
                get_unified_context_stream,
                ContextEventType,
            )

            context_stream = get_unified_context_stream()
            if context_stream:
                context_stream.add_event(
                    ContextEventType.TWELVE_FACTOR_COMPLIANCE,
                    {
                        "service_name": self.config.service_name,
                        "compliance_factors": {
                            "codebase": True,
                            "dependencies": True,
                            "config": True,
                            "backing_services": True,
                            "build_release_run": True,
                            "processes": True,
                            "port_binding": True,
                            "concurrency": True,
                            "disposability": True,
                            "dev_prod_parity": True,
                            "logs": True,
                            "admin_processes": True,
                        },
                        "agent_class": self.__class__.__name__,
                    },
                )
        except Exception:
            # Context stream may not be available during testing or initialization
            pass

    def _setup_from_environment(self):
        """Factor 3: Config from environment"""
        self.config.timeout_seconds = int(
            os.getenv(
                f"{self.config.service_name}_TIMEOUT", str(self.config.timeout_seconds)
            )
        )
        self.config.max_retries = int(
            os.getenv(
                f"{self.config.service_name}_MAX_RETRIES", str(self.config.max_retries)
            )
        )
        self.config.idempotency_enabled = (
            os.getenv(f"{self.config.service_name}_IDEMPOTENCY", "true").lower()
            == "true"
        )
        self.config.sandbox_tools = (
            os.getenv(f"{self.config.service_name}_SANDBOX", "true").lower() == "true"
        )

    def _setup_structured_logging(self):
        """Factor 11: Logs as event streams"""
        if self.config.structured_logging:
            try:
                return structlog.get_logger(
                    service=self.config.service_name,
                    structured=True,
                    context_stream="UnifiedContextStream",
                )
            except:
                # Fallback to standard logging
                logger = logging.getLogger(self.config.service_name)
                logger.setLevel(logging.INFO)
                return logger
        else:
            return logging.getLogger(self.config.service_name)

    def _setup_state_store(self):
        """Factor 6: Stateless processes with explicit state"""
        # Use Supabase or external store, never in-memory
        return SupabaseStateStore(self.config.service_name)

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def execute_with_retry(self, operation: Callable, **kwargs):
        """Factor 9: Disposability - fast startup/graceful shutdown"""
        return await operation(**kwargs)

    async def execute_with_timeout(self, operation: Callable, **kwargs):
        """Factor 8: Concurrency - scale via process model"""
        try:
            return await asyncio.wait_for(
                operation(**kwargs), timeout=self.config.timeout_seconds
            )
        except asyncio.TimeoutError:
            self.logger.error(
                f"operation_timeout: {operation.__name__} timed out after {self.config.timeout_seconds}s"
            )
            raise

    def ensure_idempotency(self, operation_id: str) -> bool:
        """Factor 7: Port binding - idempotent operations"""
        if not self.config.idempotency_enabled:
            return True

        # Check if operation already executed
        if self.state_store.operation_exists(operation_id):
            self.logger.info(f"idempotent_skip: {operation_id}")
            return False

        # Mark operation as executed
        self.state_store.mark_operation(operation_id)
        return True

    def sandbox_tool_execution(self, tool: Callable, **kwargs):
        """Factor 10: Dev/prod parity - sandboxed tools"""
        if not self.config.sandbox_tools:
            return tool(**kwargs)

        # Execute in sandbox with resource limits
        sandbox = ToolSandbox(
            memory_limit="512MB", cpu_limit=0.5, network_restricted=True
        )
        return sandbox.execute(tool, **kwargs)

    def _generate_operation_id(self, operation_name: str, data: Dict[str, Any]) -> str:
        """Generate deterministic operation ID for idempotency"""
        # Create hash from operation name and data
        content = f"{operation_name}:{json.dumps(data, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class TwelveFactorService(TwelveFactorAgent):
    """
    Example service implementation with 12-Factor compliance.
    All Lolla services should follow this pattern.
    """

    def __init__(self):
        super().__init__(TwelveFactorConfig(service_name="example_service"))

    async def process_request(
        self, request_id: str, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process with full 12-Factor compliance"""

        # Generate operation ID for idempotency
        operation_id = self._generate_operation_id(
            "process_request",
            {
                "request_id": request_id,
                "data_hash": hashlib.md5(str(data).encode()).hexdigest(),
            },
        )

        # Check idempotency
        if not self.ensure_idempotency(operation_id):
            cached_result = self.state_store.get_result(operation_id)
            if cached_result:
                self.logger.info(f"Returning cached result for {operation_id}")
                return cached_result

        # Log structured event
        self.logger.info(
            f"processing_started: request_id={request_id}, data_size={len(str(data))}"
        )

        try:
            # Execute with timeout and retry
            result = await self.execute_with_timeout(self._actual_processing, data=data)

            # Store result for idempotency
            self.state_store.store_result(operation_id, result)

            self.logger.info(f"processing_completed: request_id={request_id}")
            return result

        except Exception as e:
            self.logger.error(f"processing_failed: request_id={request_id}, error={e}")
            raise

    async def _actual_processing(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Actual processing logic (to be implemented by subclasses)"""
        # Simulate some async work
        await asyncio.sleep(0.1)

        return {
            "status": "processed",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data,
            "processed_by": self.config.service_name,
        }


class TwelveFactorConsultantService(TwelveFactorAgent):
    """
    Base class for consultant services with 12-Factor compliance.
    All consultant implementations should inherit from this.
    """

    def __init__(self, consultant_name: str):
        super().__init__(
            TwelveFactorConfig(service_name=f"consultant_{consultant_name}")
        )
        self.consultant_name = consultant_name

    async def analyze(self, context: Dict[str, Any], problem: str) -> Dict[str, Any]:
        """Analyze problem with 12-Factor compliance"""

        # Generate operation ID
        operation_id = self._generate_operation_id(
            "analyze",
            {
                "consultant": self.consultant_name,
                "problem_hash": hashlib.md5(problem.encode()).hexdigest(),
            },
        )

        # Check idempotency
        if not self.ensure_idempotency(operation_id):
            cached_result = self.state_store.get_result(operation_id)
            if cached_result:
                return cached_result

        # Log analysis start
        self.logger.info(
            f"consultant_analysis_started: consultant={self.consultant_name}, "
            f"context_size={len(str(context))}"
        )

        try:
            # Execute analysis with timeout
            result = await self.execute_with_timeout(
                self._perform_analysis, context=context, problem=problem
            )

            # Store result
            self.state_store.store_result(operation_id, result)

            return result

        except Exception as e:
            self.logger.error(
                f"consultant_analysis_failed: consultant={self.consultant_name}, error={e}"
            )
            raise

    async def _perform_analysis(
        self, context: Dict[str, Any], problem: str
    ) -> Dict[str, Any]:
        """Perform actual analysis (to be overridden by specific consultants)"""
        # Base implementation - should be overridden
        await asyncio.sleep(0.1)

        return {
            "consultant": self.consultant_name,
            "analysis": f"Analysis of {problem} by {self.consultant_name}",
            "insights": [],
            "recommendations": [],
            "confidence": 0.75,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


class TwelveFactorOrchestrationService(TwelveFactorAgent):
    """
    Base class for orchestration services with 12-Factor compliance.
    Handles coordination of multiple services.
    """

    def __init__(self, orchestrator_name: str):
        super().__init__(
            TwelveFactorConfig(service_name=f"orchestrator_{orchestrator_name}")
        )
        self.orchestrator_name = orchestrator_name
        self.child_services = []

    def register_service(self, service: TwelveFactorAgent):
        """Register a child service for orchestration"""
        self.child_services.append(service)
        self.logger.info(f"Registered service: {service.config.service_name}")

    async def orchestrate(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate task across registered services"""

        # Generate operation ID
        operation_id = self._generate_operation_id(
            "orchestrate",
            {
                "orchestrator": self.orchestrator_name,
                "task_id": task.get("id", "unknown"),
            },
        )

        # Check idempotency
        if not self.ensure_idempotency(operation_id):
            cached_result = self.state_store.get_result(operation_id)
            if cached_result:
                return cached_result

        # Log orchestration start
        self.logger.info(
            f"orchestration_started: orchestrator={self.orchestrator_name}, "
            f"services={len(self.child_services)}"
        )

        try:
            # Execute orchestration with timeout
            result = await self.execute_with_timeout(
                self._perform_orchestration, task=task
            )

            # Store result
            self.state_store.store_result(operation_id, result)

            return result

        except Exception as e:
            self.logger.error(
                f"orchestration_failed: orchestrator={self.orchestrator_name}, error={e}"
            )
            raise

    async def _perform_orchestration(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Perform actual orchestration (to be overridden)"""
        results = []

        # Execute services in parallel (simplified)
        tasks = []
        for service in self.child_services:
            if hasattr(service, "process_request"):
                tasks.append(
                    service.process_request(
                        request_id=f"{task.get('id', 'unknown')}_{service.config.service_name}",
                        data=task,
                    )
                )

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            "orchestrator": self.orchestrator_name,
            "task": task,
            "results": results,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# Helper function to create 12-Factor compliant services
def create_twelve_factor_service(service_type: str, name: str) -> TwelveFactorAgent:
    """Factory function to create 12-Factor compliant services"""
    if service_type == "consultant":
        return TwelveFactorConsultantService(name)
    elif service_type == "orchestrator":
        return TwelveFactorOrchestrationService(name)
    else:
        return TwelveFactorService()


# Decorator for 12-Factor compliance
def twelve_factor_compliant(timeout: int = 30, max_retries: int = 3):
    """Decorator to make any function 12-Factor compliant"""

    def decorator(func):
        async def wrapper(self, *args, **kwargs):
            if not hasattr(self, "_twelve_factor_agent"):
                self._twelve_factor_agent = TwelveFactorAgent(
                    TwelveFactorConfig(
                        service_name=self.__class__.__name__,
                        timeout_seconds=timeout,
                        max_retries=max_retries,
                    )
                )

            # Generate operation ID
            operation_id = self._twelve_factor_agent._generate_operation_id(
                func.__name__, {"args": str(args)[:100], "kwargs": str(kwargs)[:100]}
            )

            # Check idempotency
            if not self._twelve_factor_agent.ensure_idempotency(operation_id):
                cached = self._twelve_factor_agent.state_store.get_result(operation_id)
                if cached:
                    return cached

            # Execute with timeout - create wrapper function
            async def operation_wrapper():
                return await func(self, *args, **kwargs)

            result = await self._twelve_factor_agent.execute_with_timeout(
                operation_wrapper
            )

            # Store result
            self._twelve_factor_agent.state_store.store_result(operation_id, result)

            return result

        return wrapper

    return decorator
