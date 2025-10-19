"""
Parallel Execution Engine
Enables concurrent execution of independent operations in Neural Lace

Features:
- Dependency graph analysis and execution planning
- Parallel phase execution for independent operations
- Resource management and concurrency control
- Performance optimization through parallelization
- Deadlock detection and prevention
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import uuid4

from src.core.neural_lace_error_framework import NeuralLaceErrorFramework, ErrorContext


class TaskStatus(Enum):
    """Status of execution tasks"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class ExecutionTask:
    """Represents a task that can be executed"""

    task_id: str
    name: str
    operation: callable
    dependencies: Set[str] = field(default_factory=set)
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    execution_time_ms: float = 0.0


@dataclass
class DependencyGraph:
    """Represents dependency relationships between tasks"""

    tasks: Dict[str, ExecutionTask] = field(default_factory=dict)
    dependencies: Dict[str, Set[str]] = field(default_factory=dict)

    def add_task(self, task: ExecutionTask):
        """Add a task to the dependency graph."""
        self.tasks[task.task_id] = task
        self.dependencies[task.task_id] = task.dependencies.copy()

    def add_dependency(self, task_id: str, depends_on: str):
        """Add a dependency relationship."""
        if task_id not in self.dependencies:
            self.dependencies[task_id] = set()
        self.dependencies[task_id].add(depends_on)

        if task_id in self.tasks:
            self.tasks[task_id].dependencies.add(depends_on)

    def get_ready_tasks(self) -> List[str]:
        """Get tasks that are ready to execute (no pending dependencies)."""
        ready_tasks = []

        for task_id, task in self.tasks.items():
            if task.status == TaskStatus.PENDING:
                # Check if all dependencies are completed
                dependencies_completed = all(
                    self.tasks[dep_id].status == TaskStatus.COMPLETED
                    for dep_id in self.dependencies[task_id]
                    if dep_id in self.tasks
                )

                if dependencies_completed:
                    ready_tasks.append(task_id)

        return ready_tasks

    def detect_cycles(self) -> List[List[str]]:
        """Detect circular dependencies in the graph."""

        def dfs(node, visited, rec_stack, path):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in self.dependencies.get(node, []):
                if neighbor not in visited:
                    cycle = dfs(neighbor, visited, rec_stack, path)
                    if cycle:
                        return cycle
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    return path[cycle_start:] + [neighbor]

            rec_stack.remove(node)
            path.pop()
            return None

        visited = set()
        cycles = []

        for task_id in self.tasks:
            if task_id not in visited:
                cycle = dfs(task_id, visited, set(), [])
                if cycle:
                    cycles.append(cycle)

        return cycles


class ParallelExecutionEngine:
    """
    Parallel execution engine for Neural Lace operations.

    Features:
    - Dependency-aware parallel execution
    - Resource management and concurrency limits
    - Deadlock detection and prevention
    - Performance monitoring and optimization
    - Error handling with graceful degradation
    """

    def __init__(self, error_framework: NeuralLaceErrorFramework):
        self.logger = logging.getLogger(__name__)
        self.error_framework = error_framework

        # Configuration
        self.max_concurrent_tasks = 5
        self.task_timeout_seconds = 300  # 5 minutes
        self.enable_deadlock_detection = True

        # Execution tracking
        self.active_tasks: Dict[str, ExecutionTask] = {}
        self.completed_tasks: Dict[str, ExecutionTask] = {}
        self.execution_statistics: Dict[str, Any] = {
            "total_executions": 0,
            "parallel_efficiency": [],
            "average_speedup": 0.0,
        }

        self.logger.info("⚡ Parallel Execution Engine initialized")

    async def execute_parallel_tasks(
        self, dependency_graph: DependencyGraph
    ) -> Dict[str, Any]:
        """
        Execute tasks in parallel based on dependency graph.

        Args:
            dependency_graph: Graph of tasks with their dependencies

        Returns:
            Dictionary with execution results and performance metrics
        """

        execution_id = str(uuid4())[:8]
        start_time = time.time()

        self.logger.info(f"⚡ Starting parallel execution: {execution_id}")
        self.logger.info(
            f"📊 Tasks: {len(dependency_graph.tasks)}, Dependencies: {sum(len(deps) for deps in dependency_graph.dependencies.values())}"
        )

        # Validate dependency graph
        validation_result = await self._validate_dependency_graph(dependency_graph)
        if not validation_result["valid"]:
            raise Exception(f"Invalid dependency graph: {validation_result['errors']}")

        # Execute tasks in parallel waves
        execution_result = await self._execute_dependency_graph(
            dependency_graph, execution_id
        )

        total_time = time.time() - start_time

        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(
            dependency_graph, execution_result, total_time
        )

        self.logger.info(f"⚡ Parallel execution complete: {execution_id}")
        self.logger.info(
            f"📊 Total time: {total_time:.2f}s, Parallel efficiency: {performance_metrics['parallel_efficiency']:.2f}"
        )

        return {
            "execution_id": execution_id,
            "success": execution_result["success"],
            "total_execution_time": total_time,
            "tasks_executed": len(dependency_graph.tasks),
            "tasks_succeeded": execution_result["succeeded"],
            "tasks_failed": execution_result["failed"],
            "parallel_efficiency": performance_metrics["parallel_efficiency"],
            "speedup_factor": performance_metrics["speedup_factor"],
            "task_results": {
                task_id: task.result for task_id, task in dependency_graph.tasks.items()
            },
            "performance_metrics": performance_metrics,
        }

    async def execute_independent_operations(
        self,
        operations: List[Tuple[str, callable, Dict[str, Any]]],
        max_concurrent: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute independent operations in parallel (no dependencies).

        Args:
            operations: List of (name, operation, context) tuples
            max_concurrent: Maximum concurrent operations (uses default if None)

        Returns:
            Dictionary with execution results
        """

        if max_concurrent is None:
            max_concurrent = self.max_concurrent_tasks

        self.logger.info(
            f"⚡ Executing {len(operations)} independent operations (max concurrent: {max_concurrent})"
        )

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)

        async def execute_with_semaphore(
            name: str, operation: callable, context: Dict[str, Any]
        ):
            async with semaphore:
                return await self._execute_single_operation(name, operation, context)

        # Execute all operations concurrently
        start_time = time.time()

        tasks = [
            execute_with_semaphore(name, operation, context)
            for name, operation, context in operations
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        total_time = time.time() - start_time

        # Process results
        success_count = sum(
            1 for result in results if not isinstance(result, Exception)
        )
        failure_count = len(results) - success_count

        self.logger.info(
            f"⚡ Independent operations complete: {success_count} succeeded, {failure_count} failed in {total_time:.2f}s"
        )

        return {
            "total_operations": len(operations),
            "succeeded": success_count,
            "failed": failure_count,
            "total_execution_time": total_time,
            "results": [
                result if not isinstance(result, Exception) else str(result)
                for result in results
            ],
            "parallel_efficiency": (
                success_count / len(operations) if operations else 0.0
            ),
        }

    async def _validate_dependency_graph(
        self, dependency_graph: DependencyGraph
    ) -> Dict[str, Any]:
        """Validate dependency graph for execution."""

        errors = []
        warnings = []

        # Check for circular dependencies
        if self.enable_deadlock_detection:
            cycles = dependency_graph.detect_cycles()
            if cycles:
                errors.append(f"Circular dependencies detected: {cycles}")

        # Check for missing dependencies
        for task_id, deps in dependency_graph.dependencies.items():
            missing_deps = [dep for dep in deps if dep not in dependency_graph.tasks]
            if missing_deps:
                errors.append(
                    f"Task {task_id} has missing dependencies: {missing_deps}"
                )

        # Check for orphaned tasks (no path to completion)
        ready_initially = dependency_graph.get_ready_tasks()
        if not ready_initially and dependency_graph.tasks:
            warnings.append(
                "No tasks are initially ready - may indicate dependency issues"
            )

        # Performance warnings
        max_dependency_depth = self._calculate_dependency_depth(dependency_graph)
        if max_dependency_depth > 10:
            warnings.append(
                f"Deep dependency chain ({max_dependency_depth} levels) may impact parallelism"
            )

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "dependency_depth": max_dependency_depth,
            "parallelizable_tasks": len(ready_initially),
        }

    async def _execute_dependency_graph(
        self, dependency_graph: DependencyGraph, execution_id: str
    ) -> Dict[str, Any]:
        """Execute the dependency graph in parallel waves."""

        total_tasks = len(dependency_graph.tasks)
        succeeded = 0
        failed = 0
        wave_number = 0

        while True:
            # Get tasks ready for execution
            ready_task_ids = dependency_graph.get_ready_tasks()

            if not ready_task_ids:
                # Check if we're done or stuck
                pending_tasks = [
                    task
                    for task in dependency_graph.tasks.values()
                    if task.status == TaskStatus.PENDING
                ]
                if not pending_tasks:
                    break  # All tasks completed
                else:
                    # Tasks are blocked - this shouldn't happen with proper validation
                    self.logger.error(
                        f"❌ Execution stuck with {len(pending_tasks)} pending tasks"
                    )
                    for task in pending_tasks:
                        task.status = TaskStatus.FAILED
                        task.error = "Execution blocked by dependencies"
                        failed += 1
                    break

            wave_number += 1
            self.logger.info(
                f"⚡ Wave {wave_number}: Executing {len(ready_task_ids)} ready tasks"
            )

            # Execute ready tasks in parallel
            wave_tasks = []
            for task_id in ready_task_ids:
                task = dependency_graph.tasks[task_id]
                task.status = TaskStatus.RUNNING
                task.start_time = time.time()

                wave_task = self._execute_task_with_error_handling(task, execution_id)
                wave_tasks.append(wave_task)

            # Wait for all tasks in this wave to complete
            wave_results = await asyncio.gather(*wave_tasks, return_exceptions=True)

            # Process wave results
            for i, (task_id, result) in enumerate(zip(ready_task_ids, wave_results)):
                task = dependency_graph.tasks[task_id]
                task.end_time = time.time()
                task.execution_time_ms = (task.end_time - task.start_time) * 1000

                if isinstance(result, Exception):
                    task.status = TaskStatus.FAILED
                    task.error = str(result)
                    failed += 1
                    self.logger.error(f"❌ Task {task_id} failed: {result}")
                else:
                    task.status = TaskStatus.COMPLETED
                    task.result = result
                    succeeded += 1
                    self.logger.info(
                        f"✅ Task {task_id} completed in {task.execution_time_ms:.2f}ms"
                    )

        return {
            "success": failed == 0,
            "succeeded": succeeded,
            "failed": failed,
            "total_waves": wave_number,
        }

    async def _execute_task_with_error_handling(
        self, task: ExecutionTask, execution_id: str
    ):
        """Execute a single task with comprehensive error handling."""

        error_context = ErrorContext(
            operation_name=f"parallel_task_{task.name}",
            component="parallel_execution",
            engagement_id=execution_id,
            max_retries=1,  # Limited retries for parallel execution
        )

        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                self.error_framework.execute_with_retries(
                    operation=task.operation, context=error_context
                ),
                timeout=self.task_timeout_seconds,
            )

            return result

        except asyncio.TimeoutError:
            raise Exception(
                f"Task {task.name} timed out after {self.task_timeout_seconds} seconds"
            )
        except Exception as e:
            raise Exception(f"Task {task.name} failed: {e}")

    async def _execute_single_operation(
        self, name: str, operation: callable, context: Dict[str, Any]
    ):
        """Execute a single independent operation."""

        start_time = time.time()

        try:
            result = await operation(**context)
            execution_time = (time.time() - start_time) * 1000

            return {
                "name": name,
                "success": True,
                "result": result,
                "execution_time_ms": execution_time,
            }

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000

            self.logger.error(f"❌ Operation {name} failed: {e}")

            return {
                "name": name,
                "success": False,
                "error": str(e),
                "execution_time_ms": execution_time,
            }

    def _calculate_dependency_depth(self, dependency_graph: DependencyGraph) -> int:
        """Calculate the maximum depth of dependency chains."""

        def calculate_depth(task_id, visited):
            if task_id in visited:
                return 0  # Circular dependency handling

            visited.add(task_id)

            deps = dependency_graph.dependencies.get(task_id, set())
            if not deps:
                visited.remove(task_id)
                return 1

            max_depth = 0
            for dep in deps:
                if dep in dependency_graph.tasks:
                    depth = calculate_depth(dep, visited)
                    max_depth = max(max_depth, depth)

            visited.remove(task_id)
            return max_depth + 1

        max_depth = 0
        for task_id in dependency_graph.tasks:
            depth = calculate_depth(task_id, set())
            max_depth = max(max_depth, depth)

        return max_depth

    def _calculate_performance_metrics(
        self,
        dependency_graph: DependencyGraph,
        execution_result: Dict[str, Any],
        total_time: float,
    ) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""

        # Calculate sequential time (sum of all task times)
        sequential_time = sum(
            task.execution_time_ms / 1000.0
            for task in dependency_graph.tasks.values()
            if task.status == TaskStatus.COMPLETED
        )

        # Calculate parallel efficiency
        parallel_efficiency = (
            min(sequential_time / total_time, 1.0) if total_time > 0 else 0.0
        )

        # Calculate speedup factor
        speedup_factor = sequential_time / total_time if total_time > 0 else 1.0

        # Task timing statistics
        completed_tasks = [
            task
            for task in dependency_graph.tasks.values()
            if task.status == TaskStatus.COMPLETED
        ]
        task_times = [task.execution_time_ms for task in completed_tasks]

        avg_task_time = sum(task_times) / len(task_times) if task_times else 0.0
        max_task_time = max(task_times) if task_times else 0.0
        min_task_time = min(task_times) if task_times else 0.0

        # Update statistics
        self.execution_statistics["total_executions"] += 1
        self.execution_statistics["parallel_efficiency"].append(parallel_efficiency)

        # Calculate rolling average speedup
        recent_efficiencies = self.execution_statistics["parallel_efficiency"][
            -20:
        ]  # Last 20 executions
        self.execution_statistics["average_speedup"] = sum(recent_efficiencies) / len(
            recent_efficiencies
        )

        return {
            "sequential_time_seconds": sequential_time,
            "parallel_time_seconds": total_time,
            "parallel_efficiency": parallel_efficiency,
            "speedup_factor": speedup_factor,
            "total_waves": execution_result["total_waves"],
            "task_timing": {
                "average_ms": avg_task_time,
                "maximum_ms": max_task_time,
                "minimum_ms": min_task_time,
                "completed_tasks": len(completed_tasks),
            },
            "resource_utilization": {
                "max_concurrent_tasks": self.max_concurrent_tasks,
                "average_concurrent_tasks": (
                    len(dependency_graph.tasks) / execution_result["total_waves"]
                    if execution_result["total_waves"] > 0
                    else 0
                ),
            },
        }

    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get comprehensive execution statistics."""

        return {
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "max_concurrent_tasks": self.max_concurrent_tasks,
                "task_timeout_seconds": self.task_timeout_seconds,
                "deadlock_detection_enabled": self.enable_deadlock_detection,
            },
            "statistics": self.execution_statistics,
            "performance_summary": {
                "total_executions": self.execution_statistics["total_executions"],
                "average_parallel_efficiency": self.execution_statistics[
                    "average_speedup"
                ],
                "recent_efficiency_trend": (
                    "improving"
                    if len(self.execution_statistics["parallel_efficiency"]) > 1
                    and self.execution_statistics["parallel_efficiency"][-1]
                    > self.execution_statistics["parallel_efficiency"][-2]
                    else "stable"
                ),
            },
        }


def get_parallel_execution_engine(
    error_framework: NeuralLaceErrorFramework,
) -> ParallelExecutionEngine:
    """Factory function for Parallel Execution Engine."""
    return ParallelExecutionEngine(error_framework)
