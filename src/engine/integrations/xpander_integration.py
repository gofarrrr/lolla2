#!/usr/bin/env python3
"""
METIS Xpander.ai Production Integration
I009: Production component integration with Xpander.ai patterns

Implements Xpander.ai @on_task decorators and production patterns
for enterprise-grade task orchestration and scaling.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from uuid import UUID, uuid4
from functools import wraps
import inspect

try:
    from src.engine.models.data_contracts import MetisDataContract, EngagementContext
    from src.core.enhanced_event_bus import (
        EnhancedKafkaEventBus as MetisEventBus,
        CloudEvent,
    )
    from src.core.state_management import DistributedStateManager, StateType
    from src.engine.agents.specialized_workers import (
        AgentTask,
        AgentRole,
        TaskPriority,
        AgentStatus,
    )

    DATA_CONTRACTS_AVAILABLE = True
except ImportError:
    DATA_CONTRACTS_AVAILABLE = False
    print("Warning: Core components not available, using mock interfaces")

    # Mock implementations
    class MockStateManager:
        def __init__(self):
            self.data = {}

        async def set_state(self, key, value, state_type=None):
            self.data[key] = value

        async def get_state(self, key):
            return self.data.get(key)

    DistributedStateManager = MockStateManager
    StateType = None

    class MockRedisConnection:
        """Mock Redis connection for testing"""

        def __init__(self):
            self.data = {}

        async def set(self, key, value):
            self.data[key] = value

        async def get(self, key):
            return self.data.get(key)


class TaskExecutionMode(str, Enum):
    """Task execution modes for Xpander.ai patterns"""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    PIPELINE = "pipeline"
    DISTRIBUTED = "distributed"
    ADAPTIVE = "adaptive"


class TaskPriority(str, Enum):
    """Task priority levels for Xpander.ai orchestration"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class ScalingPolicy(str, Enum):
    """Scaling policies for task execution"""

    AUTO = "auto"
    MANUAL = "manual"
    REACTIVE = "reactive"
    PREDICTIVE = "predictive"


@dataclass
class XpanderTaskMetadata:
    """Metadata for Xpander.ai task execution"""

    task_id: UUID = field(default_factory=uuid4)
    task_name: str = ""
    execution_mode: TaskExecutionMode = TaskExecutionMode.SEQUENTIAL
    priority: TaskPriority = TaskPriority.MEDIUM

    # Scaling configuration
    min_workers: int = 1
    max_workers: int = 10
    scaling_policy: ScalingPolicy = ScalingPolicy.AUTO

    # Execution constraints
    timeout_seconds: int = 300
    max_retries: int = 3
    retry_delay_seconds: float = 2.0

    # Dependencies and prerequisites
    dependencies: List[UUID] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)

    # Performance targets
    target_completion_time: Optional[timedelta] = None
    resource_requirements: Dict[str, Any] = field(default_factory=dict)

    # Monitoring and observability
    telemetry_enabled: bool = True
    logging_level: str = "INFO"
    metrics_collection: bool = True


@dataclass
class TaskExecutionContext:
    """Execution context for Xpander.ai tasks"""

    context_id: UUID = field(default_factory=uuid4)
    engagement_id: Optional[UUID] = None
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None

    # Execution environment
    worker_pool_id: str = ""
    execution_node: str = ""
    allocated_resources: Dict[str, Any] = field(default_factory=dict)

    # State and data
    input_data: Dict[str, Any] = field(default_factory=dict)
    shared_state: Dict[str, Any] = field(default_factory=dict)
    intermediate_results: List[Dict[str, Any]] = field(default_factory=list)

    # Timing and performance
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_duration: Optional[timedelta] = None

    # Error handling
    errors: List[Dict[str, Any]] = field(default_factory=list)
    retry_count: int = 0
    last_error: Optional[str] = None


class XpanderTaskRegistry:
    """
    Registry for Xpander.ai task decorators and handlers
    Implements production patterns for task orchestration
    """

    def __init__(self):
        self.registered_tasks: Dict[str, Callable] = {}
        self.task_metadata: Dict[str, XpanderTaskMetadata] = {}
        self.execution_stats: Dict[str, Dict[str, Any]] = {}
        self.active_executions: Dict[UUID, TaskExecutionContext] = {}
        self.logger = logging.getLogger(__name__)

    def register_task(
        self, task_name: str, handler: Callable, metadata: XpanderTaskMetadata
    ):
        """Register a task handler with metadata"""
        self.registered_tasks[task_name] = handler
        self.task_metadata[task_name] = metadata
        self.execution_stats[task_name] = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_duration": 0.0,
            "last_execution": None,
        }

        self.logger.info(f"Registered Xpander task: {task_name}")

    def get_task_handler(self, task_name: str) -> Optional[Callable]:
        """Get registered task handler"""
        return self.registered_tasks.get(task_name)

    def get_task_metadata(self, task_name: str) -> Optional[XpanderTaskMetadata]:
        """Get task metadata"""
        return self.task_metadata.get(task_name)

    def update_execution_stats(self, task_name: str, success: bool, duration: float):
        """Update execution statistics"""
        stats = self.execution_stats.get(task_name, {})

        stats["total_executions"] = stats.get("total_executions", 0) + 1
        if success:
            stats["successful_executions"] = stats.get("successful_executions", 0) + 1
        else:
            stats["failed_executions"] = stats.get("failed_executions", 0) + 1

        # Update average duration
        total = stats["total_executions"]
        current_avg = stats.get("average_duration", 0.0)
        stats["average_duration"] = (current_avg * (total - 1) + duration) / total
        stats["last_execution"] = datetime.utcnow().isoformat()

        self.execution_stats[task_name] = stats


# Global task registry
_task_registry = XpanderTaskRegistry()


def on_task(
    name: str = None,
    execution_mode: TaskExecutionMode = TaskExecutionMode.SEQUENTIAL,
    priority: TaskPriority = TaskPriority.MEDIUM,
    max_workers: int = 5,
    timeout_seconds: int = 300,
    max_retries: int = 3,
    scaling_policy: ScalingPolicy = ScalingPolicy.AUTO,
):
    """
    Xpander.ai @on_task decorator for production task orchestration

    Usage:
        @on_task(name="problem_structuring", execution_mode=TaskExecutionMode.PARALLEL)
        async def structure_problem(context: TaskExecutionContext) -> Dict[str, Any]:
            # Task implementation
            return {"result": "structured"}
    """

    def decorator(func: Callable) -> Callable:
        # Generate task name if not provided
        task_name = name or f"{func.__module__}.{func.__name__}"

        # Create task metadata
        metadata = XpanderTaskMetadata(
            task_name=task_name,
            execution_mode=execution_mode,
            priority=priority,
            max_workers=max_workers,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            scaling_policy=scaling_policy,
        )

        # Register task
        _task_registry.register_task(task_name, func, metadata)

        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create execution context
            context = TaskExecutionContext(
                input_data=kwargs, started_at=datetime.utcnow()
            )

            # Execute with Xpander patterns
            return await execute_xpander_task(task_name, context, func, *args, **kwargs)

        # Attach metadata to function
        wrapper._xpander_task_name = task_name
        wrapper._xpander_metadata = metadata

        return wrapper

    return decorator


async def execute_xpander_task(
    task_name: str, context: TaskExecutionContext, handler: Callable, *args, **kwargs
) -> Any:
    """Execute task with Xpander.ai production patterns"""

    logger = logging.getLogger(__name__)
    metadata = _task_registry.get_task_metadata(task_name)

    if not metadata:
        raise ValueError(f"Task {task_name} not registered")

    # Add context to active executions
    _task_registry.active_executions[context.context_id] = context

    start_time = time.time()
    success = False
    result = None

    try:
        logger.info(
            f"Executing Xpander task: {task_name} (mode: {metadata.execution_mode.value})"
        )

        # Apply execution mode patterns
        if metadata.execution_mode == TaskExecutionMode.PARALLEL:
            result = await _execute_parallel_task(handler, context, *args, **kwargs)
        elif metadata.execution_mode == TaskExecutionMode.PIPELINE:
            result = await _execute_pipeline_task(handler, context, *args, **kwargs)
        elif metadata.execution_mode == TaskExecutionMode.DISTRIBUTED:
            result = await _execute_distributed_task(handler, context, *args, **kwargs)
        elif metadata.execution_mode == TaskExecutionMode.ADAPTIVE:
            result = await _execute_adaptive_task(handler, context, *args, **kwargs)
        else:  # SEQUENTIAL
            result = await _execute_sequential_task(handler, context, *args, **kwargs)

        success = True
        context.completed_at = datetime.utcnow()
        context.execution_duration = context.completed_at - context.started_at

        logger.info(
            f"Task {task_name} completed successfully in {context.execution_duration.total_seconds():.2f}s"
        )

    except Exception as e:
        logger.error(f"Task {task_name} failed: {str(e)}")
        context.last_error = str(e)
        context.errors.append(
            {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
                "retry_count": context.retry_count,
            }
        )

        # Retry logic
        if context.retry_count < metadata.max_retries:
            context.retry_count += 1
            logger.info(
                f"Retrying task {task_name} (attempt {context.retry_count}/{metadata.max_retries})"
            )

            await asyncio.sleep(metadata.retry_delay_seconds)
            return await execute_xpander_task(
                task_name, context, handler, *args, **kwargs
            )

        raise

    finally:
        # Update execution statistics
        duration = time.time() - start_time
        _task_registry.update_execution_stats(task_name, success, duration)

        # Remove from active executions
        if context.context_id in _task_registry.active_executions:
            del _task_registry.active_executions[context.context_id]

    return result


async def _execute_sequential_task(
    handler: Callable, context: TaskExecutionContext, *args, **kwargs
) -> Any:
    """Execute task in sequential mode"""

    # Add context as first argument if handler expects it
    sig = inspect.signature(handler)
    if "context" in sig.parameters:
        return await handler(context, *args, **kwargs)
    else:
        return await handler(*args, **kwargs)


async def _execute_parallel_task(
    handler: Callable, context: TaskExecutionContext, *args, **kwargs
) -> Any:
    """Execute task in parallel mode with worker scaling"""

    # For parallel execution, split input data if possible
    input_chunks = _split_input_for_parallel_execution(context.input_data)

    if len(input_chunks) <= 1:
        # No parallelization possible, fall back to sequential
        return await _execute_sequential_task(handler, context, *args, **kwargs)

    # Execute chunks in parallel
    tasks = []
    for chunk in input_chunks:
        chunk_context = TaskExecutionContext(
            engagement_id=context.engagement_id,
            input_data=chunk,
            shared_state=context.shared_state,
        )

        task = asyncio.create_task(
            _execute_sequential_task(handler, chunk_context, *args, **kwargs)
        )
        tasks.append(task)

    # Wait for all parallel tasks to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Combine results
    combined_result = _combine_parallel_results(results)
    return combined_result


async def _execute_pipeline_task(
    handler: Callable, context: TaskExecutionContext, *args, **kwargs
) -> Any:
    """Execute task in pipeline mode with streaming"""

    # Pipeline execution processes data in stages
    # For now, implement as enhanced sequential with intermediate results

    context.intermediate_results = []

    # Execute with intermediate result capture
    result = await _execute_sequential_task(handler, context, *args, **kwargs)

    # Store intermediate results for pipeline optimization
    context.intermediate_results.append(
        {
            "stage": "pipeline_execution",
            "result": result,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )

    return result


async def _execute_distributed_task(
    handler: Callable, context: TaskExecutionContext, *args, **kwargs
) -> Any:
    """Execute task in distributed mode across multiple nodes"""

    # Distributed execution would typically involve:
    # 1. Task distribution across worker nodes
    # 2. Load balancing and resource allocation
    # 3. Result aggregation from multiple nodes

    # For now, implement as optimized parallel execution
    return await _execute_parallel_task(handler, context, *args, **kwargs)


async def _execute_adaptive_task(
    handler: Callable, context: TaskExecutionContext, *args, **kwargs
) -> Any:
    """Execute task in adaptive mode with dynamic optimization"""

    # Adaptive execution analyzes task characteristics and chooses optimal mode
    input_size = len(str(context.input_data))

    if input_size > 10000:  # Large input, use parallel
        return await _execute_parallel_task(handler, context, *args, **kwargs)
    elif input_size > 1000:  # Medium input, use pipeline
        return await _execute_pipeline_task(handler, context, *args, **kwargs)
    else:  # Small input, use sequential
        return await _execute_sequential_task(handler, context, *args, **kwargs)


def _split_input_for_parallel_execution(
    input_data: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Split input data for parallel execution"""

    # Look for arrays or lists that can be chunked
    chunks = []

    for key, value in input_data.items():
        if isinstance(value, list) and len(value) > 1:
            # Split list into chunks
            chunk_size = max(1, len(value) // 3)  # Create up to 3 chunks
            for i in range(0, len(value), chunk_size):
                chunk = input_data.copy()
                chunk[key] = value[i : i + chunk_size]
                chunks.append(chunk)
            break

    return chunks if chunks else [input_data]


def _combine_parallel_results(results: List[Any]) -> Any:
    """Combine results from parallel execution"""

    # Filter out exceptions
    valid_results = [r for r in results if not isinstance(r, Exception)]

    if not valid_results:
        raise Exception("All parallel tasks failed")

    # If results are dictionaries, merge them
    if all(isinstance(r, dict) for r in valid_results):
        combined = {}
        for result in valid_results:
            combined.update(result)
        return combined

    # If results are lists, concatenate them
    if all(isinstance(r, list) for r in valid_results):
        combined = []
        for result in valid_results:
            combined.extend(result)
        return combined

    # Otherwise, return the first valid result
    return valid_results[0]


class XpanderTaskOrchestrator:
    """
    Production task orchestrator implementing Xpander.ai patterns
    Handles task scheduling, scaling, and resource management
    """

    def __init__(
        self,
        state_manager: DistributedStateManager,
        event_bus: Optional[MetisEventBus] = None,
    ):
        self.state_manager = state_manager
        self.event_bus = event_bus
        self.logger = logging.getLogger(__name__)

        # Task management
        self.task_queue: List[Dict[str, Any]] = []
        self.worker_pools: Dict[str, List[str]] = {}
        self.resource_allocations: Dict[str, Dict[str, Any]] = {}

        # Scaling and optimization
        self.auto_scaling_enabled = True
        self.performance_metrics: Dict[str, List[float]] = {}
        self.optimization_history: List[Dict[str, Any]] = []

        # Start background optimization
        asyncio.create_task(self._optimization_loop())

    async def schedule_task(
        self,
        task_name: str,
        input_data: Dict[str, Any],
        engagement_id: Optional[UUID] = None,
        priority: TaskPriority = TaskPriority.MEDIUM,
    ) -> UUID:
        """Schedule task for execution with Xpander.ai orchestration"""

        task_id = uuid4()

        # Create task execution context
        context = TaskExecutionContext(
            context_id=task_id, engagement_id=engagement_id, input_data=input_data
        )

        # Get task metadata
        metadata = _task_registry.get_task_metadata(task_name)
        if not metadata:
            raise ValueError(f"Unknown task: {task_name}")

        # Schedule for execution
        scheduled_task = {
            "task_id": task_id,
            "task_name": task_name,
            "context": context,
            "metadata": metadata,
            "priority": priority,
            "scheduled_at": datetime.utcnow(),
            "status": "scheduled",
        }

        self.task_queue.append(scheduled_task)

        # Sort queue by priority
        self._sort_task_queue_by_priority()

        # Store in state
        await self.state_manager.set_state(
            f"xpander_task_{task_id}",
            {
                "task_id": str(task_id),
                "task_name": task_name,
                "engagement_id": str(engagement_id) if engagement_id else None,
                "priority": priority.value,
                "scheduled_at": scheduled_task["scheduled_at"].isoformat(),
                "status": "scheduled",
            },
            StateType.WORKFLOW if StateType else "workflow",
        )

        # Emit scheduling event
        if self.event_bus:
            await self.event_bus.publish_event(
                CloudEvent(
                    type="xpander.task.scheduled",
                    source="xpander/orchestrator",
                    data={
                        "task_id": str(task_id),
                        "task_name": task_name,
                        "priority": priority.value,
                        "execution_mode": metadata.execution_mode.value,
                    },
                )
            )

        self.logger.info(f"Scheduled Xpander task: {task_name} (ID: {task_id})")

        # Trigger immediate execution if high priority
        if priority in [TaskPriority.CRITICAL, TaskPriority.EMERGENCY]:
            asyncio.create_task(self._execute_next_task())

        return task_id

    def _sort_task_queue_by_priority(self):
        """Sort task queue by priority"""
        priority_order = {
            TaskPriority.EMERGENCY: 0,
            TaskPriority.CRITICAL: 1,
            TaskPriority.HIGH: 2,
            TaskPriority.MEDIUM: 3,
            TaskPriority.LOW: 4,
        }

        self.task_queue.sort(key=lambda t: priority_order.get(t["priority"], 999))

    async def _execute_next_task(self):
        """Execute next task in queue"""

        if not self.task_queue:
            return

        task = self.task_queue.pop(0)
        task_id = task["task_id"]
        task_name = task["task_name"]
        context = task["context"]

        try:
            # Update status
            task["status"] = "executing"
            task["started_at"] = datetime.utcnow()

            # Get task handler
            handler = _task_registry.get_task_handler(task_name)
            if not handler:
                raise ValueError(f"No handler for task: {task_name}")

            # Execute task
            result = await execute_xpander_task(task_name, context, handler)

            # Update status and store result
            task["status"] = "completed"
            task["completed_at"] = datetime.utcnow()
            task["result"] = result

            # Store completion in state
            await self.state_manager.set_state(
                f"xpander_task_{task_id}_result",
                {
                    "task_id": str(task_id),
                    "status": "completed",
                    "result": result,
                    "completed_at": task["completed_at"].isoformat(),
                },
                StateType.WORKFLOW if StateType else "workflow",
            )

            # Emit completion event
            if self.event_bus:
                await self.event_bus.publish_event(
                    CloudEvent(
                        type="xpander.task.completed",
                        source="xpander/orchestrator",
                        data={
                            "task_id": str(task_id),
                            "task_name": task_name,
                            "execution_duration": (
                                task["completed_at"] - task["started_at"]
                            ).total_seconds(),
                        },
                    )
                )

            self.logger.info(f"Completed Xpander task: {task_name} (ID: {task_id})")

        except Exception as e:
            # Update status with error
            task["status"] = "failed"
            task["error"] = str(e)
            task["failed_at"] = datetime.utcnow()

            self.logger.error(
                f"Failed Xpander task: {task_name} (ID: {task_id}) - {str(e)}"
            )

            # Emit failure event
            if self.event_bus:
                await self.event_bus.publish_event(
                    CloudEvent(
                        type="xpander.task.failed",
                        source="xpander/orchestrator",
                        data={
                            "task_id": str(task_id),
                            "task_name": task_name,
                            "error": str(e),
                        },
                    )
                )

    async def _optimization_loop(self):
        """Background loop for performance optimization"""

        while True:
            try:
                # Execute pending tasks
                while self.task_queue:
                    await self._execute_next_task()

                # Perform optimization analysis
                await self._analyze_performance_metrics()

                # Sleep before next iteration
                await asyncio.sleep(5)  # Check every 5 seconds

            except Exception as e:
                self.logger.error(f"Optimization loop error: {str(e)}")
                await asyncio.sleep(30)

    async def _analyze_performance_metrics(self):
        """Analyze performance metrics for optimization"""

        # Collect performance data from task registry
        for task_name, stats in _task_registry.execution_stats.items():
            if stats["total_executions"] > 0:
                # Track average duration
                if task_name not in self.performance_metrics:
                    self.performance_metrics[task_name] = []

                self.performance_metrics[task_name].append(stats["average_duration"])

                # Keep only recent metrics (last 100 executions)
                self.performance_metrics[task_name] = self.performance_metrics[
                    task_name
                ][-100:]

        # Auto-scaling analysis
        if self.auto_scaling_enabled:
            await self._auto_scale_workers()

    async def _auto_scale_workers(self):
        """Auto-scale worker pools based on demand"""

        # Simple auto-scaling logic based on queue length and performance
        queue_length = len(self.task_queue)

        if queue_length > 10:  # High demand
            self.logger.info("High task demand detected - scaling up workers")
        elif queue_length == 0:  # Low demand
            self.logger.debug("Low task demand - maintaining current capacity")

    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get orchestrator status and metrics"""

        return {
            "task_queue_length": len(self.task_queue),
            "registered_tasks": len(_task_registry.registered_tasks),
            "active_executions": len(_task_registry.active_executions),
            "worker_pools": len(self.worker_pools),
            "auto_scaling_enabled": self.auto_scaling_enabled,
            "performance_metrics": {
                task: {
                    "avg_duration": sum(durations) / len(durations) if durations else 0,
                    "recent_executions": len(durations),
                }
                for task, durations in self.performance_metrics.items()
            },
            "execution_stats": _task_registry.execution_stats,
        }


# Global orchestrator instance
_global_orchestrator: Optional[XpanderTaskOrchestrator] = None


async def get_xpander_orchestrator() -> XpanderTaskOrchestrator:
    """Get or create global Xpander orchestrator instance"""
    global _global_orchestrator

    if _global_orchestrator is None:
        state_manager = (
            DistributedStateManager()
            if DATA_CONTRACTS_AVAILABLE
            else MockStateManager()
        )
        _global_orchestrator = XpanderTaskOrchestrator(state_manager)

    return _global_orchestrator


# Convenience functions for Xpander.ai integration
async def execute_task(
    task_name: str,
    input_data: Dict[str, Any],
    priority: TaskPriority = TaskPriority.MEDIUM,
) -> UUID:
    """Execute task using Xpander.ai orchestration"""
    orchestrator = await get_xpander_orchestrator()
    return await orchestrator.schedule_task(task_name, input_data, priority=priority)


def get_task_registry_status() -> Dict[str, Any]:
    """Get task registry status"""
    return {
        "registered_tasks": list(_task_registry.registered_tasks.keys()),
        "execution_stats": _task_registry.execution_stats,
        "active_executions": len(_task_registry.active_executions),
    }
