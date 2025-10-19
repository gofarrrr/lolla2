"""
EnhancedParallelCognitiveForges - Dependency-Aware Parallel Execution System
Implements sophisticated parallel processing from IMPLEMENTATION_HANDOVER.md
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import networkx as nx

from src.core.unified_context_stream import UnifiedContextStream, ContextEventType
from src.core.cognitive_consultant_router import ConsultantProfile
from src.core.twelve_factor_compliance import TwelveFactorAgent
from src.core.parallel_forges_breadth_mode import SupervisedBreadthOrchestrator
from src.engine.core.feature_flags import FeatureFlagService as FeatureFlagManager
from src.engine.core.llm_manager import get_llm_manager
from src.engine.core.structured_logging import get_logger

logger = get_logger(__name__)


@dataclass
class ConsultantTask:
    """Task definition for a consultant analysis"""

    task_id: str
    consultant: ConsultantProfile
    context: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    priority: int = 1
    timeout_seconds: int = 120  # Increased for complex tasks
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsultantResult:
    """Result from a consultant analysis"""

    task_id: str
    consultant_id: str
    success: bool
    output: Any
    duration_ms: int
    error: Optional[str] = None
    dependencies_met: bool = True
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class ParallelExecutionResult:
    """Aggregated result from parallel execution"""

    results: List[ConsultantResult]
    aggregated_output: Dict[str, Any]
    total_duration_ms: int
    parallelism_achieved: float
    dependency_violations: int
    success_rate: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class EnhancedParallelCognitiveForges(TwelveFactorAgent):
    """
    Enhanced parallel execution system with:
    - Dependency-aware scheduling
    - Topological sorting for execution order
    - Dynamic parallelism adjustment
    - Result aggregation and synthesis
    - Breadth mode integration
    """

    def __init__(
        self,
        context_stream: UnifiedContextStream,
        feature_flags: Optional[FeatureFlagManager] = None,
        max_parallel_tasks: int = 5,
        enable_breadth_mode: bool = False,
    ):
        """Initialize the enhanced parallel forges"""
        from src.core.twelve_factor_compliance import TwelveFactorConfig

        twelve_factor_config = TwelveFactorConfig(
            service_name="enhanced_parallel_cognitive_forges"
        )
        super().__init__(twelve_factor_config)

        self.context_stream = context_stream
        self.feature_flags = feature_flags or FeatureFlagManager()
        self.max_parallel_tasks = max_parallel_tasks
        self.enable_breadth_mode = enable_breadth_mode

        # Breadth mode orchestrator
        self.breadth_orchestrator = (
            SupervisedBreadthOrchestrator(context_stream=context_stream)
            if enable_breadth_mode
            else None
        )

        # Execution tracking
        self._execution_history: List[ParallelExecutionResult] = []
        self._task_cache: Dict[str, ConsultantResult] = {}

        # Performance metrics
        self._metrics = {
            "total_executions": 0,
            "total_tasks": 0,
            "average_parallelism": 0.0,
            "cache_hits": 0,
            "dependency_violations": 0,
            "average_duration_ms": 0,
        }

        logger.info(
            f"EnhancedParallelCognitiveForges initialized with "
            f"max_parallel={max_parallel_tasks}, breadth_mode={enable_breadth_mode}"
        )

    async def execute_parallel_analysis(
        self,
        consultants: List[ConsultantProfile],
        problem_context: Dict[str, Any],
        dependencies: Optional[Dict[str, List[str]]] = None,
        use_breadth_mode: Optional[bool] = None,
    ) -> ParallelExecutionResult:
        """
        Execute parallel consultant analyses with dependency management.

        Args:
            consultants: List of consultants to execute
            problem_context: Context for analysis
            dependencies: Optional dependency graph (task_id -> [dependency_ids])
            use_breadth_mode: Override for breadth mode usage

        Returns:
            ParallelExecutionResult with all results and metrics
        """
        start_time = time.time()
        self._metrics["total_executions"] += 1

        # Check if breadth mode should be used
        if use_breadth_mode is None:
            use_breadth_mode = self.enable_breadth_mode

        if use_breadth_mode and self.breadth_orchestrator:
            # Execute using breadth mode if eligible
            logger.info("Attempting breadth mode for parallel execution")
            breadth_result = await self.breadth_orchestrator.execute_if_eligible(
                problem_context, consultants
            )
            if breadth_result.get("breadth_mode_used", False):
                logger.info("Successfully used breadth mode for parallel execution")
                return self._convert_breadth_result(breadth_result, consultants)

        # Create tasks for each consultant
        tasks = self._create_consultant_tasks(
            consultants, problem_context, dependencies
        )
        self._metrics["total_tasks"] += len(tasks)

        # Build dependency graph
        execution_order = self._compute_execution_order(tasks)

        # Log execution plan
        self.context_stream.add_event(
            ContextEventType.ENGAGEMENT_STARTED,
            {
                "stage": "parallel_analysis",
                "consultant_count": len(consultants),
                "dependency_levels": len(execution_order),
                "breadth_mode": False,
            },
        )

        # Execute tasks in dependency order
        results = await self._execute_with_dependencies(
            tasks, execution_order, problem_context
        )

        # Aggregate results
        aggregated = self._aggregate_results(results)

        # Calculate metrics
        duration_ms = int((time.time() - start_time) * 1000)
        parallelism = self._calculate_parallelism(results, duration_ms)
        dependency_violations = sum(1 for r in results if not r.dependencies_met)
        success_rate = sum(1 for r in results if r.success) / max(len(results), 1)

        # Update metrics
        self._metrics["average_duration_ms"] = (
            self._metrics["average_duration_ms"]
            * (self._metrics["total_executions"] - 1)
            + duration_ms
        ) / self._metrics["total_executions"]
        self._metrics["dependency_violations"] += dependency_violations

        # Create execution result
        execution_result = ParallelExecutionResult(
            results=results,
            aggregated_output=aggregated,
            total_duration_ms=duration_ms,
            parallelism_achieved=parallelism,
            dependency_violations=dependency_violations,
            success_rate=success_rate,
            metadata={
                "execution_order": [
                    [t.task_id for t in level] for level in execution_order
                ],
                "cache_hits": self._metrics["cache_hits"],
            },
        )

        # Track execution
        self._execution_history.append(execution_result)

        # Log completion
        self.context_stream.add_event(
            ContextEventType.PHASE_COMPLETED,
            {
                "stage": "parallel_analysis",
                "duration_ms": duration_ms,
                "success_rate": success_rate,
                "parallelism": parallelism,
            },
        )

        logger.info(
            f"Parallel execution completed: {len(results)} tasks, "
            f"parallelism={parallelism:.2f}, success_rate={success_rate:.2f}"
        )

        return execution_result

    async def _execute_breadth_mode(
        self, consultants: List[ConsultantProfile], problem_context: Dict[str, Any]
    ) -> ParallelExecutionResult:
        """Execute using breadth mode orchestrator"""
        if not self.breadth_orchestrator:
            raise ValueError("Breadth mode not enabled")

        # Convert consultants to subagent configs for breadth mode
        subagent_configs = [
            {
                "agent_id": c.id,
                "agent_type": c.thinking_style,
                "capabilities": c.expertise,
                "context": problem_context,
            }
            for c in consultants
        ]

        # Execute through breadth mode
        breadth_result = await self.breadth_orchestrator.execute_breadth_mode(
            subagent_configs, problem_context
        )

        # Convert breadth mode results to our format
        results = []
        for agent_id, output in breadth_result["agent_outputs"].items():
            consultant = next((c for c in consultants if c.id == agent_id), None)
            if consultant:
                results.append(
                    ConsultantResult(
                        task_id=f"breadth_{agent_id}",
                        consultant_id=agent_id,
                        success=output.get("success", True),
                        output=output.get("analysis"),
                        duration_ms=output.get("duration_ms", 0),
                        dependencies_met=True,
                    )
                )

        return ParallelExecutionResult(
            results=results,
            aggregated_output=breadth_result["synthesis"],
            total_duration_ms=breadth_result["total_duration_ms"],
            parallelism_achieved=len(consultants),  # Full parallelism in breadth mode
            dependency_violations=0,
            success_rate=breadth_result["success_rate"],
            metadata={"breadth_mode": True, "metrics": breadth_result["metrics"]},
        )

    def _convert_breadth_result(
        self, breadth_result: Dict[str, Any], consultants: List[ConsultantProfile]
    ) -> ParallelExecutionResult:
        """Convert breadth mode result to standard parallel execution result format"""
        consultant_results = []
        for i, consultant in enumerate(consultants):
            consultant_results.append(
                ConsultantResult(
                    consultant=consultant,
                    analysis=breadth_result.get(
                        "final_synthesis",
                        f"Breadth mode analysis for {consultant.name}",
                    ),
                    confidence=0.85,  # Default confidence for breadth mode
                    processing_time=breadth_result.get("execution_time", 30.0),
                    sources=breadth_result.get("sources", []),
                    metadata={"breadth_mode": True},
                )
            )

        aggregated_output = {
            "execution_mode": "breadth_mode_synthesis_polygon_preserved",
            "final_analysis": breadth_result.get(
                "final_synthesis", "Breadth mode synthesis completed"
            ),
            "polygon_preservation": breadth_result.get("points_of_dissent", []),
            "consensus_points": breadth_result.get("consensus_points", []),
        }

        return ParallelExecutionResult(
            consultant_results=consultant_results,
            aggregated_output=aggregated_output,
            success_rate=1.0,  # Breadth mode succeeded if it ran
            parallelism_achieved=1.0,  # Full parallelism in breadth mode
            execution_time=breadth_result.get("execution_time", 30.0),
            metadata={"breadth_mode_used": True},
        )

    def _create_consultant_tasks(
        self,
        consultants: List[ConsultantProfile],
        problem_context: Dict[str, Any],
        dependencies: Optional[Dict[str, List[str]]],
    ) -> List[ConsultantTask]:
        """Create task definitions for each consultant"""
        tasks = []

        for i, consultant in enumerate(consultants):
            task_id = f"{consultant.id}_{i}"

            # Get dependencies for this consultant
            task_dependencies = []
            if dependencies and task_id in dependencies:
                task_dependencies = dependencies[task_id]

            # Create consultant-specific context
            consultant_context = problem_context.copy()
            consultant_context["consultant_role"] = consultant.name
            consultant_context["consultant_expertise"] = consultant.expertise
            consultant_context["consultant_approach"] = consultant.typical_approach

            task = ConsultantTask(
                task_id=task_id,
                consultant=consultant,
                context=consultant_context,
                dependencies=task_dependencies,
                priority=i,
                timeout_seconds=120,
            )

            tasks.append(task)

        return tasks

    def _compute_execution_order(
        self, tasks: List[ConsultantTask]
    ) -> List[List[ConsultantTask]]:
        """
        Compute execution order using topological sort.
        Returns list of task levels that can be executed in parallel.
        """
        # Build dependency graph
        graph = nx.DiGraph()
        task_map = {task.task_id: task for task in tasks}

        for task in tasks:
            graph.add_node(task.task_id)
            for dep in task.dependencies:
                if dep in task_map:
                    graph.add_edge(dep, task.task_id)

        # Check for cycles
        if not nx.is_directed_acyclic_graph(graph):
            logger.warning("Dependency cycle detected, removing cycles")
            # Remove cycles by breaking back edges
            cycles = list(nx.simple_cycles(graph))
            for cycle in cycles:
                if len(cycle) > 1:
                    graph.remove_edge(cycle[-1], cycle[0])

        # Compute levels for parallel execution
        levels = []
        remaining = set(task.task_id for task in tasks)

        while remaining:
            # Find tasks with no remaining dependencies
            current_level = []
            for task_id in remaining:
                predecessors = set(graph.predecessors(task_id))
                if not predecessors or predecessors.isdisjoint(remaining):
                    current_level.append(task_map[task_id])

            if not current_level:
                # Shouldn't happen if we removed cycles correctly
                logger.error("Unable to compute execution order")
                current_level = [task_map[tid] for tid in list(remaining)[:1]]

            levels.append(current_level)
            remaining -= set(task.task_id for task in current_level)

        logger.info(f"Computed {len(levels)} execution levels from {len(tasks)} tasks")
        return levels

    async def _execute_with_dependencies(
        self,
        tasks: List[ConsultantTask],
        execution_order: List[List[ConsultantTask]],
        problem_context: Dict[str, Any],
    ) -> List[ConsultantResult]:
        """Execute tasks respecting dependency order"""
        all_results = []
        completed_tasks = set()

        for level_idx, level_tasks in enumerate(execution_order):
            logger.info(
                f"Executing level {level_idx + 1}/{len(execution_order)} with {len(level_tasks)} tasks"
            )

            # Execute tasks in this level in parallel
            level_results = await self._execute_parallel_tasks(
                level_tasks, problem_context, completed_tasks
            )

            all_results.extend(level_results)
            completed_tasks.update(r.task_id for r in level_results if r.success)

        return all_results

    async def _execute_parallel_tasks(
        self,
        tasks: List[ConsultantTask],
        problem_context: Dict[str, Any],
        completed_tasks: Set[str],
    ) -> List[ConsultantResult]:
        """Execute multiple tasks in parallel"""
        # Check cache first
        results = []
        tasks_to_execute = []

        for task in tasks:
            cache_key = f"{task.consultant.id}_{hash(str(task.context))}"
            if cache_key in self._task_cache:
                logger.info(f"Cache hit for task {task.task_id}")
                self._metrics["cache_hits"] += 1
                results.append(self._task_cache[cache_key])
            else:
                tasks_to_execute.append(task)

        if not tasks_to_execute:
            return results

        # Create async tasks with rate limiting
        semaphore = asyncio.Semaphore(self.max_parallel_tasks)

        async def execute_with_limit(task):
            async with semaphore:
                return await self._execute_single_task(task, completed_tasks)

        # Execute in parallel
        task_coroutines = [execute_with_limit(task) for task in tasks_to_execute]
        new_results = await asyncio.gather(*task_coroutines, return_exceptions=True)

        # Process results
        for task, result in zip(tasks_to_execute, new_results):
            if isinstance(result, Exception):
                logger.error(f"Task {task.task_id} failed with exception: {result}")
                result = ConsultantResult(
                    task_id=task.task_id,
                    consultant_id=task.consultant.id,
                    success=False,
                    output=None,
                    duration_ms=0,
                    error=str(result),
                    dependencies_met=False,
                )

            # Cache successful results
            if result.success:
                cache_key = f"{task.consultant.id}_{hash(str(task.context))}"
                self._task_cache[cache_key] = result

            results.append(result)

        return results

    async def _execute_single_task(
        self, task: ConsultantTask, completed_tasks: Set[str]
    ) -> ConsultantResult:
        """Execute a single consultant task"""
        start_time = time.time()

        # Phase 6: Emit CONSULTANT_ANALYSIS_START event
        self.context_stream.add_event(
            ContextEventType.CONSULTANT_ANALYSIS_START,
            {
                "task_id": task.task_id,
                "consultant_name": task.consultant.name,
                "consultant_type": task.consultant.typical_approach,
                "dependencies": task.dependencies,
                "dependencies_met": all(
                    dep in completed_tasks for dep in task.dependencies
                ),
                "start_time": datetime.utcnow().isoformat(),
                "timeout_seconds": task.timeout_seconds,
            },
        )

        # Check dependencies
        dependencies_met = all(dep in completed_tasks for dep in task.dependencies)
        if not dependencies_met:
            logger.warning(f"Task {task.task_id} executed with unmet dependencies")

        try:
            # Build consultant prompt
            prompt = self._build_consultant_prompt(task)

            # Execute with timeout
            llm_manager = get_llm_manager(context_stream=self.context_stream)

            # Use asyncio timeout
            async with asyncio.timeout(task.timeout_seconds):
                response = await llm_manager.call_llm(
                    messages=[
                        {
                            "role": "system",
                            "content": f"You are {task.consultant.name}, a {task.consultant.typical_approach} expert.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.7,
                )

            output = response.get("content", {})

            duration_ms = int((time.time() - start_time) * 1000)

            # Phase 6: Emit CONSULTANT_ANALYSIS_COMPLETE event for success
            result = ConsultantResult(
                task_id=task.task_id,
                consultant_id=task.consultant.id,
                success=True,
                output=output,
                duration_ms=duration_ms,
                dependencies_met=dependencies_met,
            )

            self.context_stream.add_event(
                ContextEventType.CONSULTANT_ANALYSIS_COMPLETE,
                {
                    "task_id": task.task_id,
                    "consultant_name": task.consultant.name,
                    "success": True,
                    "duration_ms": duration_ms,
                    "dependencies_met": dependencies_met,
                    "has_output": output is not None,
                    "end_time": datetime.utcnow().isoformat(),
                },
            )

            return result

        except asyncio.TimeoutError:
            logger.error(f"Task {task.task_id} timed out after {task.timeout_seconds}s")
            duration_ms = task.timeout_seconds * 1000

            # Phase 6: Emit CONSULTANT_ANALYSIS_COMPLETE event for timeout
            self.context_stream.add_event(
                ContextEventType.CONSULTANT_ANALYSIS_COMPLETE,
                {
                    "task_id": task.task_id,
                    "consultant_name": task.consultant.name,
                    "success": False,
                    "duration_ms": duration_ms,
                    "dependencies_met": dependencies_met,
                    "error": "Timeout",
                    "end_time": datetime.utcnow().isoformat(),
                },
            )

            return ConsultantResult(
                task_id=task.task_id,
                consultant_id=task.consultant.id,
                success=False,
                output=None,
                duration_ms=duration_ms,
                error="Timeout",
                dependencies_met=dependencies_met,
            )

        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {e}")
            duration_ms = int((time.time() - start_time) * 1000)

            # Phase 6: Emit CONSULTANT_ANALYSIS_COMPLETE event for exception
            self.context_stream.add_event(
                ContextEventType.CONSULTANT_ANALYSIS_COMPLETE,
                {
                    "task_id": task.task_id,
                    "consultant_name": task.consultant.name,
                    "success": False,
                    "duration_ms": duration_ms,
                    "dependencies_met": dependencies_met,
                    "error": str(e),
                    "end_time": datetime.utcnow().isoformat(),
                },
            )

            return ConsultantResult(
                task_id=task.task_id,
                consultant_id=task.consultant.id,
                success=False,
                output=None,
                duration_ms=duration_ms,
                error=str(e),
                dependencies_met=dependencies_met,
            )

    def _build_consultant_prompt(self, task: ConsultantTask) -> str:
        """Build prompt for consultant analysis"""
        prompt = f"""
        As {task.consultant.name}, analyze the following problem using your expertise in {', '.join(task.consultant.expertise)}.
        
        Problem Context:
        {task.context.get('problem_statement', 'No problem statement provided')}
        
        Your Approach: {task.consultant.typical_approach}
        
        Your Strengths: {', '.join(task.consultant.strengths)}
        
        Please provide your analysis focusing on:
        1. Key insights from your perspective
        2. Specific recommendations based on your expertise
        3. Potential risks or concerns you identify
        4. How your expertise applies to this problem
        
        Structure your response as a comprehensive analysis.
        """

        # Add dependency context if available
        if task.dependencies:
            prompt += f"\n\nNote: Your analysis may build upon insights from: {', '.join(task.dependencies)}"

        return prompt

    def _aggregate_results(self, results: List[ConsultantResult]) -> Dict[str, Any]:
        """Aggregate results from all consultants"""
        successful_results = [r for r in results if r.success]

        if not successful_results:
            return {
                "status": "no_successful_analyses",
                "consultant_count": 0,
                "insights": [],
            }

        # Extract insights from each consultant
        all_insights = []
        recommendations = []
        risks = []

        for result in successful_results:
            if isinstance(result.output, str):
                # Simple text output
                all_insights.append(
                    {
                        "consultant": result.consultant_id,
                        "insight": result.output[:500],  # Truncate for aggregation
                    }
                )
            elif isinstance(result.output, dict):
                # Structured output
                if "insights" in result.output:
                    all_insights.extend(result.output["insights"])
                if "recommendations" in result.output:
                    recommendations.extend(result.output["recommendations"])
                if "risks" in result.output:
                    risks.extend(result.output["risks"])

        return {
            "status": "aggregated",
            "consultant_count": len(successful_results),
            "total_insights": len(all_insights),
            "key_insights": all_insights[:10],  # Top insights
            "recommendations": recommendations[:5],  # Top recommendations
            "identified_risks": risks[:5],  # Top risks
            "consensus_areas": self._identify_consensus(successful_results),
            "divergent_views": self._identify_divergence(successful_results),
        }

    def _identify_consensus(self, results: List[ConsultantResult]) -> List[str]:
        """Identify areas of consensus among consultants"""
        # Simple implementation - would be more sophisticated in production
        consensus = []

        # Count common themes (simplified)
        theme_counts = defaultdict(int)
        for result in results:
            if isinstance(result.output, str):
                # Extract key phrases (simplified)
                if "risk" in result.output.lower():
                    theme_counts["risk_awareness"] += 1
                if "opportunity" in result.output.lower():
                    theme_counts["opportunity_identification"] += 1
                if "recommend" in result.output.lower():
                    theme_counts["actionable_recommendations"] += 1

        # Identify consensus (>60% agreement)
        threshold = len(results) * 0.6
        for theme, count in theme_counts.items():
            if count >= threshold:
                consensus.append(theme.replace("_", " ").title())

        return consensus

    def _identify_divergence(self, results: List[ConsultantResult]) -> List[str]:
        """Identify areas of divergent views"""
        # Simple implementation - would be more sophisticated in production
        divergence = []

        # Look for contrasting viewpoints (simplified)
        has_optimistic = any("opportunity" in str(r.output).lower() for r in results)
        has_pessimistic = any(
            "risk" in str(r.output).lower() or "concern" in str(r.output).lower()
            for r in results
        )

        if has_optimistic and has_pessimistic:
            divergence.append("Mixed views on risk vs opportunity")

        return divergence

    def _calculate_parallelism(
        self, results: List[ConsultantResult], total_duration_ms: int
    ) -> float:
        """Calculate achieved parallelism"""
        if not results or total_duration_ms == 0:
            return 0.0

        # Sum of all task durations
        total_task_time = sum(r.duration_ms for r in results)

        # Parallelism = total task time / actual wall time
        parallelism = total_task_time / total_duration_ms

        # Update running average
        self._metrics["average_parallelism"] = (
            self._metrics["average_parallelism"]
            * (self._metrics["total_executions"] - 1)
            + parallelism
        ) / self._metrics["total_executions"]

        return min(parallelism, self.max_parallel_tasks)  # Cap at max

    def get_execution_metrics(self) -> Dict[str, Any]:
        """Get execution performance metrics"""
        return {
            "total_executions": self._metrics["total_executions"],
            "total_tasks": self._metrics["total_tasks"],
            "average_parallelism": self._metrics["average_parallelism"],
            "cache_hit_rate": (
                self._metrics["cache_hits"] / max(self._metrics["total_tasks"], 1)
            ),
            "dependency_violation_rate": (
                self._metrics["dependency_violations"]
                / max(self._metrics["total_tasks"], 1)
            ),
            "average_duration_ms": self._metrics["average_duration_ms"],
            "execution_history_size": len(self._execution_history),
        }
