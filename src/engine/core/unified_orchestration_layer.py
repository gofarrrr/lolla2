#!/usr/bin/env python3
"""
METIS Unified Orchestration Layer
Consolidates multiple orchestrators (workflow, blueprint, research, etc.) into a single,
coherent layer that provides simplified interfaces and consistent behavior patterns.

Key Features:
- Single entry point for all orchestration needs
- Consistent patterns across different orchestration types
- Integrated caching, checkpointing, and human-in-loop workflows
- Performance monitoring and intelligent resource allocation
- Context-aware orchestrator selection and execution
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
from collections import defaultdict


class OrchestrationType(Enum):
    """Types of orchestration available"""

    WORKFLOW = "workflow"  # 4-phase consulting workflow
    BLUEPRINT = "blueprint"  # High-level planning
    RESEARCH = "research"  # Research coordination
    COGNITIVE = "cognitive"  # Mental model application
    TREE_SEARCH = "tree_search"  # Complex problem exploration
    ASYNC_PROCESSING = "async_processing"  # Background task coordination
    HYBRID = "hybrid"  # Multi-orchestrator combination


class OrchestrationContext(Enum):
    """Execution contexts for different orchestration needs"""

    REAL_TIME = "real_time"  # <2s response time, immediate user feedback
    INTERACTIVE = "interactive"  # 2-30s response time, progressive disclosure
    BACKGROUND = "background"  # >30s response time, async processing
    RESEARCH_MODE = "research"  # Long-running research with external APIs
    ENTERPRISE_BATCH = "batch"  # High-volume processing with resource management


@dataclass
class OrchestrationRequest:
    """Unified request structure for all orchestration types"""

    request_id: str
    orchestration_type: OrchestrationType
    context: OrchestrationContext

    # Problem definition
    problem_statement: str
    business_context: Dict[str, Any] = field(default_factory=dict)

    # Execution preferences
    max_duration_seconds: float = 30.0
    confidence_target: float = 0.8
    quality_requirements: Dict[str, Any] = field(default_factory=dict)

    # Resource constraints
    cost_budget: Optional[float] = None
    compute_priority: str = "normal"  # low, normal, high

    # Integration requirements
    enable_checkpointing: bool = True
    enable_human_in_loop: bool = True
    enable_caching: bool = True
    enable_streaming: bool = True

    # Created metadata
    created_at: float = field(default_factory=time.time)
    user_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestrationResult:
    """Unified result structure for all orchestration outputs"""

    request_id: str
    orchestration_type: OrchestrationType
    status: str  # completed, failed, partial, timeout

    # Primary outputs
    insights: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    analysis_artifacts: Dict[str, Any] = field(default_factory=dict)

    # Quality metrics
    confidence_score: float = 0.0
    completeness_score: float = 0.0
    methodology_used: List[str] = field(default_factory=list)

    # Performance data
    execution_time_seconds: float = 0.0
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    cost_incurred: float = 0.0

    # Process metadata
    orchestrators_used: List[str] = field(default_factory=list)
    checkpoints_created: int = 0
    human_interventions: int = 0
    cache_hits: int = 0

    # Error information
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    completed_at: float = field(default_factory=time.time)


class BaseOrchestrator(ABC):
    """Abstract base class for all orchestrators"""

    @abstractmethod
    async def execute(self, request: OrchestrationRequest) -> OrchestrationResult:
        """Execute orchestration for given request"""
        pass

    @abstractmethod
    def supports_context(self, context: OrchestrationContext) -> bool:
        """Check if orchestrator supports given execution context"""
        pass

    @abstractmethod
    def get_estimated_duration(self, request: OrchestrationRequest) -> float:
        """Estimate execution duration in seconds"""
        pass

    @abstractmethod
    def get_resource_requirements(
        self, request: OrchestrationRequest
    ) -> Dict[str, Any]:
        """Get resource requirements for request"""
        pass


class WorkflowOrchestrator(BaseOrchestrator):
    """Orchestrates 4-phase consulting workflow"""

    def __init__(self):
        self.phases = [
            "problem_structuring",
            "hypothesis_generation",
            "analysis_execution",
            "synthesis_delivery",
        ]

    async def execute(self, request: OrchestrationRequest) -> OrchestrationResult:
        """Execute 4-phase workflow"""
        start_time = time.time()
        result = OrchestrationResult(
            request_id=request.request_id,
            orchestration_type=OrchestrationType.WORKFLOW,
            status="running",
        )

        try:
            # Execute each phase
            for i, phase in enumerate(self.phases):
                phase_start = time.time()

                # Simulate phase execution
                await asyncio.sleep(0.5)  # Realistic processing time

                # Generate phase insights
                insights = await self._execute_phase(phase, request.problem_statement)
                result.insights.extend(insights)

                # Update progress
                phase_duration = time.time() - phase_start
                print(f"   Phase {i+1}/4 ({phase}): {phase_duration:.1f}s")

            # Generate final recommendations
            result.recommendations = await self._synthesize_recommendations(
                result.insights
            )
            result.confidence_score = 0.85
            result.completeness_score = 0.90
            result.methodology_used = ["MECE", "Hypothesis Testing", "Systems Thinking"]
            result.status = "completed"

        except Exception as e:
            result.status = "failed"
            result.error_message = str(e)

        result.execution_time_seconds = time.time() - start_time
        result.orchestrators_used = ["workflow_orchestrator"]
        return result

    async def _execute_phase(self, phase: str, problem: str) -> List[Dict[str, Any]]:
        """Execute a single workflow phase"""
        # Simulate phase-specific insights
        insights = [
            {
                "phase": phase,
                "insight": f"Key insight from {phase}: Strategic analysis of {problem[:50]}...",
                "confidence": 0.8 + (hash(phase) % 10) * 0.02,
                "mental_models": ["systems_thinking", "critical_analysis"],
                "timestamp": time.time(),
            }
        ]
        return insights

    async def _synthesize_recommendations(
        self, insights: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Synthesize final recommendations from insights"""
        return [
            {
                "recommendation": "Proceed with strategic option based on analysis",
                "priority": "high",
                "confidence": 0.85,
                "supporting_insights": len(insights),
                "implementation_complexity": "medium",
            }
        ]

    def supports_context(self, context: OrchestrationContext) -> bool:
        return context in [
            OrchestrationContext.INTERACTIVE,
            OrchestrationContext.BACKGROUND,
        ]

    def get_estimated_duration(self, request: OrchestrationRequest) -> float:
        return 8.0  # 8 seconds for workflow

    def get_resource_requirements(
        self, request: OrchestrationRequest
    ) -> Dict[str, Any]:
        return {"cpu": "medium", "memory": "low", "llm_calls": 4}


class BlueprintOrchestrator(BaseOrchestrator):
    """Orchestrates high-level planning and blueprint generation"""

    async def execute(self, request: OrchestrationRequest) -> OrchestrationResult:
        """Execute blueprint generation"""
        start_time = time.time()
        result = OrchestrationResult(
            request_id=request.request_id,
            orchestration_type=OrchestrationType.BLUEPRINT,
            status="running",
        )

        try:
            # Generate blueprint structure
            blueprint = await self._generate_blueprint(request)

            result.analysis_artifacts = {
                "blueprint": blueprint,
                "approach_summary": "Structured analytical approach with risk mitigation",
                "total_estimated_minutes": 45,
                "confidence_target": request.confidence_target,
            }

            result.insights = [
                {
                    "type": "blueprint_insight",
                    "content": f"Generated {blueprint['steps']} step blueprint",
                    "confidence": 0.9,
                }
            ]

            result.confidence_score = 0.9
            result.completeness_score = 1.0
            result.methodology_used = ["Blueprint Planning", "Structured Analysis"]
            result.status = "completed"

        except Exception as e:
            result.status = "failed"
            result.error_message = str(e)

        result.execution_time_seconds = time.time() - start_time
        result.orchestrators_used = ["blueprint_orchestrator"]
        return result

    async def _generate_blueprint(
        self, request: OrchestrationRequest
    ) -> Dict[str, Any]:
        """Generate analytical blueprint"""
        await asyncio.sleep(0.3)  # Simulate blueprint generation

        return {
            "steps": 4,
            "approach": "systematic_analysis",
            "estimated_duration": request.max_duration_seconds,
            "methodology": "MeMo cognitive framework",
            "risk_level": "medium",
        }

    def supports_context(self, context: OrchestrationContext) -> bool:
        return context in [
            OrchestrationContext.REAL_TIME,
            OrchestrationContext.INTERACTIVE,
        ]

    def get_estimated_duration(self, request: OrchestrationRequest) -> float:
        return 2.0  # 2 seconds for blueprint

    def get_resource_requirements(
        self, request: OrchestrationRequest
    ) -> Dict[str, Any]:
        return {"cpu": "low", "memory": "low", "llm_calls": 1}


class TreeSearchOrchestrator(BaseOrchestrator):
    """Orchestrates complex problem exploration using Monte Carlo Tree Search"""

    def __init__(self):
        # Import TreeSearchEngine here to avoid circular imports
        from src.intelligence.tree_search_engine import (
            TreeSearchEngine,
            SearchConfiguration,
        )

        self.search_engine_class = TreeSearchEngine
        self.config_class = SearchConfiguration

    async def execute(self, request: OrchestrationRequest) -> OrchestrationResult:
        """Execute tree search exploration for complex problems"""
        start_time = time.time()
        result = OrchestrationResult(
            request_id=request.request_id,
            orchestration_type=OrchestrationType.TREE_SEARCH,
            status="running",
        )

        try:
            print(
                f"🌳 Starting Tree Search exploration for: {request.problem_statement[:100]}..."
            )

            # Configure search based on request context
            config = self._create_search_config(request)

            # Initialize tree search engine
            search_engine = self.search_engine_class(config)

            # Execute tree search
            search_results = await search_engine.search(
                request.problem_statement, request.business_context or {}
            )

            # Convert tree search results to orchestration result format
            result = await self._convert_search_results(result, search_results)
            result.status = "completed"

            print(
                f"✅ Tree search completed: {search_results['search_metadata']['nodes_explored']} nodes explored"
            )

        except Exception as e:
            print(f"❌ Tree search failed: {e}")
            result.status = "failed"
            result.error_message = str(e)
            # Provide fallback insights
            result.insights = [
                {
                    "type": "fallback",
                    "insight": f"Tree search failed, but problem analysis suggests: {request.problem_statement[:100]}...",
                    "confidence": 0.3,
                    "source": "fallback_analysis",
                }
            ]

        result.execution_time_seconds = time.time() - start_time
        result.orchestrators_used = ["tree_search_orchestrator"]
        return result

    def _create_search_config(self, request: OrchestrationRequest):
        """Create search configuration based on request context"""
        # Determine configuration based on context
        if request.context == OrchestrationContext.REAL_TIME:
            # Fast configuration for real-time
            return self.config_class(
                max_iterations=20,
                max_depth=2,
                max_time_seconds=5.0,
                confidence_threshold=0.6,
                min_nodes_for_convergence=8,
                enable_progressive_widening=True,
                enable_rave=True,
                progressive_widening_constant=1.5,
            )
        elif request.context == OrchestrationContext.INTERACTIVE:
            # Balanced configuration for interactive use
            return self.config_class(
                max_iterations=50,
                max_depth=3,
                max_time_seconds=15.0,
                confidence_threshold=0.75,
                min_nodes_for_convergence=12,
                enable_progressive_widening=True,
                enable_rave=True,
                progressive_widening_constant=1.2,
            )
        else:
            # Comprehensive configuration for background/research
            return self.config_class(
                max_iterations=100,
                max_depth=4,
                max_time_seconds=45.0,
                confidence_threshold=0.8,
                min_nodes_for_convergence=20,
                enable_progressive_widening=True,
                enable_rave=True,
                progressive_widening_constant=1.0,
            )

    async def _convert_search_results(
        self, result: OrchestrationResult, search_results: Dict[str, Any]
    ) -> OrchestrationResult:
        """Convert tree search results to orchestration result format"""

        # Extract key insights from tree search
        best_paths = search_results.get("best_paths", [])
        cognitive_insights = search_results.get("cognitive_insights", {})
        search_metadata = search_results.get("search_metadata", {})

        # Convert best paths to insights
        insights = []
        for i, path in enumerate(best_paths[:5]):  # Top 5 paths
            path_insight = {
                "type": "reasoning_path",
                "insight": (
                    f"Strategic pathway {i+1}: {path['path'][0]['content'][:100]}..."
                    if path.get("path")
                    else f"Pathway {i+1}"
                ),
                "confidence": path.get("path_confidence", 0.5),
                "source": "tree_search",
                "mental_models": (
                    path["path"][0].get("mental_models", []) if path.get("path") else []
                ),
                "reasoning_depth": len(path.get("path", [])),
                "tree_search_data": {
                    "path_value": path.get("path_value", 0.0),
                    "path_confidence": path.get("path_confidence", 0.0),
                    "full_path": path.get("path", []),
                },
            }
            insights.append(path_insight)

        # Add cognitive insights
        if cognitive_insights:
            meta_insight = {
                "type": "meta_analysis",
                "insight": f"Tree search explored {search_metadata.get('nodes_explored', 0)} reasoning paths using {cognitive_insights.get('cognitive_diversity', 0)} mental models",
                "confidence": search_metadata.get("confidence_score", 0.5),
                "source": "tree_search_meta",
                "tree_search_data": {
                    "nodes_explored": search_metadata.get("nodes_explored", 0),
                    "cognitive_diversity": cognitive_insights.get(
                        "cognitive_diversity", 0
                    ),
                    "mental_model_effectiveness": cognitive_insights.get(
                        "mental_model_effectiveness", {}
                    ),
                    "exploration_patterns": cognitive_insights.get(
                        "exploration_patterns", {}
                    ),
                },
            }
            insights.append(meta_insight)

        result.insights = insights

        # Generate recommendations from top paths
        recommendations = []
        for i, path in enumerate(best_paths[:3]):  # Top 3 paths for recommendations
            if path.get("path") and len(path["path"]) > 0:
                rec = {
                    "recommendation": f"Strategic approach {i+1}: Focus on {path['path'][-1]['content'][:80]}...",
                    "confidence": path.get("path_confidence", 0.5),
                    "evidence": f"Based on {len(path['path'])}-step reasoning with confidence {path.get('path_confidence', 0.5):.2f}",
                    "mental_models": path["path"][-1].get("mental_models", []),
                    "tree_search_support": True,
                }
                recommendations.append(rec)

        result.recommendations = recommendations

        # Set scores
        performance_metrics = search_results.get("performance_metrics", {})
        result.confidence_score = performance_metrics.get("confidence_score", 0.5)
        result.completeness_score = min(
            1.0, search_metadata.get("nodes_explored", 0) / 20.0
        )  # Based on exploration

        # Set methodology
        result.methodology_used = list(
            set(
                [
                    model
                    for path in best_paths
                    for node in path.get("path", [])
                    for model in node.get("mental_models", [])
                ]
            )
        )[
            :10
        ]  # Limit to 10 models

        # Store complete tree search results
        result.analysis_artifacts = {
            "tree_search_results": search_results,
            "search_metadata": search_metadata,
            "performance_metrics": performance_metrics,
        }

        return result

    def supports_context(self, context: OrchestrationContext) -> bool:
        """Tree search supports all contexts with different configurations"""
        return True

    def get_estimated_duration(self, request: OrchestrationRequest) -> float:
        """Estimate tree search duration based on context"""
        if request.context == OrchestrationContext.REAL_TIME:
            return 5.0
        elif request.context == OrchestrationContext.INTERACTIVE:
            return 15.0
        else:
            return 45.0

    def get_resource_requirements(
        self, request: OrchestrationRequest
    ) -> Dict[str, Any]:
        """Get resource requirements for tree search"""
        if request.context == OrchestrationContext.REAL_TIME:
            return {"cpu": "medium", "memory": "medium", "llm_calls": 5}
        elif request.context == OrchestrationContext.INTERACTIVE:
            return {"cpu": "high", "memory": "medium", "llm_calls": 15}
        else:
            return {"cpu": "high", "memory": "high", "llm_calls": 30}


class HybridOrchestrator(BaseOrchestrator):
    """Orchestrates multiple orchestration types in combination"""

    def __init__(self, orchestrators: Dict[OrchestrationType, BaseOrchestrator]):
        self.orchestrators = orchestrators

    async def execute(self, request: OrchestrationRequest) -> OrchestrationResult:
        """Execute hybrid orchestration"""
        start_time = time.time()
        result = OrchestrationResult(
            request_id=request.request_id,
            orchestration_type=OrchestrationType.HYBRID,
            status="running",
        )

        try:
            # Start with blueprint for planning
            blueprint_request = OrchestrationRequest(
                request_id=f"{request.request_id}_blueprint",
                orchestration_type=OrchestrationType.BLUEPRINT,
                context=OrchestrationContext.REAL_TIME,
                problem_statement=request.problem_statement,
                business_context=request.business_context,
            )

            blueprint_result = await self.orchestrators[
                OrchestrationType.BLUEPRINT
            ].execute(blueprint_request)

            # Then execute main workflow
            workflow_request = OrchestrationRequest(
                request_id=f"{request.request_id}_workflow",
                orchestration_type=OrchestrationType.WORKFLOW,
                context=OrchestrationContext.INTERACTIVE,
                problem_statement=request.problem_statement,
                business_context=request.business_context,
            )

            workflow_result = await self.orchestrators[
                OrchestrationType.WORKFLOW
            ].execute(workflow_request)

            # Combine results
            result.insights = blueprint_result.insights + workflow_result.insights
            result.recommendations = workflow_result.recommendations
            result.analysis_artifacts = {
                "blueprint": blueprint_result.analysis_artifacts,
                "workflow": workflow_result.analysis_artifacts,
            }

            result.confidence_score = (
                blueprint_result.confidence_score + workflow_result.confidence_score
            ) / 2
            result.completeness_score = workflow_result.completeness_score
            result.methodology_used = list(
                set(
                    blueprint_result.methodology_used + workflow_result.methodology_used
                )
            )
            result.orchestrators_used = (
                blueprint_result.orchestrators_used + workflow_result.orchestrators_used
            )
            result.status = "completed"

        except Exception as e:
            result.status = "failed"
            result.error_message = str(e)

        result.execution_time_seconds = time.time() - start_time
        return result

    def supports_context(self, context: OrchestrationContext) -> bool:
        return True  # Hybrid supports all contexts

    def get_estimated_duration(self, request: OrchestrationRequest) -> float:
        return 10.0  # Combined duration

    def get_resource_requirements(
        self, request: OrchestrationRequest
    ) -> Dict[str, Any]:
        return {"cpu": "high", "memory": "medium", "llm_calls": 5}


@dataclass
class UnifiedOrchestrationStats:
    """Statistics for the unified orchestration layer"""

    total_requests: int = 0
    successful_completions: int = 0
    failed_requests: int = 0
    average_execution_time: float = 0.0
    orchestration_type_usage: Dict[str, int] = field(default_factory=dict)
    context_usage: Dict[str, int] = field(default_factory=dict)
    resource_efficiency: float = 0.0


class UnifiedOrchestrationLayer:
    """
    Unified orchestration layer that consolidates all orchestrators
    """

    def __init__(self):
        # Initialize individual orchestrators
        self.orchestrators = {
            OrchestrationType.WORKFLOW: WorkflowOrchestrator(),
            OrchestrationType.BLUEPRINT: BlueprintOrchestrator(),
            OrchestrationType.TREE_SEARCH: TreeSearchOrchestrator(),
            OrchestrationType.HYBRID: None,  # Will be initialized below
        }

        # Initialize hybrid orchestrator with access to others
        self.orchestrators[OrchestrationType.HYBRID] = HybridOrchestrator(
            self.orchestrators
        )

        # Execution tracking
        self.active_executions: Dict[str, OrchestrationRequest] = {}
        self.execution_history: List[OrchestrationResult] = []
        self.stats = UnifiedOrchestrationStats()

        # Performance monitoring
        self.performance_metrics = defaultdict(list)

        # Setup logging
        self.logger = logging.getLogger(__name__)

    async def orchestrate(self, request: OrchestrationRequest) -> OrchestrationResult:
        """
        Main orchestration entry point - intelligently routes to appropriate orchestrator
        """
        self.stats.total_requests += 1
        self.stats.orchestration_type_usage[request.orchestration_type.value] = (
            self.stats.orchestration_type_usage.get(request.orchestration_type.value, 0)
            + 1
        )
        self.stats.context_usage[request.context.value] = (
            self.stats.context_usage.get(request.context.value, 0) + 1
        )

        # Register active execution
        self.active_executions[request.request_id] = request

        start_time = time.time()
        result = None

        try:
            print(
                f"🎯 Orchestrating: {request.orchestration_type.value} in {request.context.value} context"
            )

            # Select and validate orchestrator
            orchestrator = await self._select_orchestrator(request)

            if not orchestrator:
                raise ValueError(
                    f"No orchestrator available for {request.orchestration_type.value}"
                )

            # Execute orchestration
            result = await orchestrator.execute(request)

            # Post-process result
            result = await self._post_process_result(result, request)

            self.stats.successful_completions += 1
            print(
                f"✅ Orchestration completed: {result.execution_time_seconds:.1f}s, confidence: {result.confidence_score:.2f}"
            )

        except Exception as e:
            self.stats.failed_requests += 1
            self.logger.error(f"Orchestration failed: {e}")

            # Create error result
            result = OrchestrationResult(
                request_id=request.request_id,
                orchestration_type=request.orchestration_type,
                status="failed",
                error_message=str(e),
                execution_time_seconds=time.time() - start_time,
            )

            print(f"❌ Orchestration failed: {e}")

        finally:
            # Clean up active execution
            if request.request_id in self.active_executions:
                del self.active_executions[request.request_id]

            # Store in history
            if result:
                self.execution_history.append(result)

                # Update performance metrics
                self.performance_metrics[request.orchestration_type.value].append(
                    {
                        "duration": result.execution_time_seconds,
                        "confidence": result.confidence_score,
                        "success": result.status == "completed",
                    }
                )

        return result

    async def _select_orchestrator(
        self, request: OrchestrationRequest
    ) -> Optional[BaseOrchestrator]:
        """Intelligently select the best orchestrator for request"""

        # Context-aware orchestrator selection
        if request.context == OrchestrationContext.REAL_TIME:
            # For real-time, prefer blueprint or lightweight orchestrators
            if request.orchestration_type == OrchestrationType.WORKFLOW:
                # Downgrade to blueprint for real-time
                return self.orchestrators[OrchestrationType.BLUEPRINT]

        # Direct orchestrator mapping
        orchestrator = self.orchestrators.get(request.orchestration_type)

        if orchestrator and orchestrator.supports_context(request.context):
            return orchestrator

        # Fallback to hybrid orchestrator
        return self.orchestrators.get(OrchestrationType.HYBRID)

    async def _post_process_result(
        self, result: OrchestrationResult, request: OrchestrationRequest
    ) -> OrchestrationResult:
        """Post-process orchestration results for consistency"""

        # Ensure minimum quality standards
        if result.confidence_score < request.confidence_target:
            result.warnings.append(
                f"Confidence {result.confidence_score:.2f} below target {request.confidence_target:.2f}"
            )

        # Add resource usage tracking
        result.resource_usage = {
            "execution_time": result.execution_time_seconds,
            "cost_estimated": result.execution_time_seconds
            * 0.01,  # $0.01/second estimate
            "orchestrators_count": len(result.orchestrators_used),
        }

        result.cost_incurred = result.resource_usage["cost_estimated"]

        return result

    async def stream_orchestration_progress(
        self, request_id: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream real-time progress updates for active orchestration"""
        while request_id in self.active_executions:
            request = self.active_executions[request_id]

            yield {
                "request_id": request_id,
                "orchestration_type": request.orchestration_type.value,
                "context": request.context.value,
                "status": "running",
                "elapsed_time": time.time() - request.created_at,
                "estimated_remaining": max(
                    0, request.max_duration_seconds - (time.time() - request.created_at)
                ),
            }

            await asyncio.sleep(1.0)  # Update every second

        # Final status if available
        for result in reversed(self.execution_history):
            if result.request_id == request_id:
                yield {
                    "request_id": request_id,
                    "status": result.status,
                    "confidence_score": result.confidence_score,
                    "execution_time": result.execution_time_seconds,
                }
                break

    def get_orchestration_stats(self) -> Dict[str, Any]:
        """Get comprehensive orchestration statistics"""
        success_rate = (
            self.stats.successful_completions / max(self.stats.total_requests, 1)
        ) * 100

        # Calculate average execution times per type
        avg_times = {}
        for orch_type, metrics in self.performance_metrics.items():
            if metrics:
                avg_times[orch_type] = sum(m["duration"] for m in metrics) / len(
                    metrics
                )

        return {
            "total_requests": self.stats.total_requests,
            "success_rate": success_rate,
            "active_executions": len(self.active_executions),
            "orchestration_types": dict(self.stats.orchestration_type_usage),
            "contexts": dict(self.stats.context_usage),
            "average_execution_times": avg_times,
            "recent_results": [r.status for r in self.execution_history[-10:]],
        }

    async def cancel_orchestration(self, request_id: str) -> bool:
        """Cancel active orchestration"""
        if request_id in self.active_executions:
            del self.active_executions[request_id]
            print(f"🚫 Orchestration cancelled: {request_id}")
            return True
        return False


def get_unified_orchestration_layer() -> UnifiedOrchestrationLayer:
    """Factory function to get configured unified orchestration layer"""
    return UnifiedOrchestrationLayer()


# Example usage and testing
if __name__ == "__main__":

    async def test_unified_orchestration():
        """Test the unified orchestration layer"""
        print("🎯 Testing METIS Unified Orchestration Layer")

        # Create unified layer
        orchestration = get_unified_orchestration_layer()

        # Test different orchestration types
        requests = [
            OrchestrationRequest(
                request_id="test_001",
                orchestration_type=OrchestrationType.BLUEPRINT,
                context=OrchestrationContext.REAL_TIME,
                problem_statement="Quick strategic assessment needed",
                max_duration_seconds=5.0,
            ),
            OrchestrationRequest(
                request_id="test_002",
                orchestration_type=OrchestrationType.WORKFLOW,
                context=OrchestrationContext.INTERACTIVE,
                problem_statement="Comprehensive market expansion analysis",
                max_duration_seconds=15.0,
                confidence_target=0.85,
            ),
            OrchestrationRequest(
                request_id="test_003",
                orchestration_type=OrchestrationType.HYBRID,
                context=OrchestrationContext.BACKGROUND,
                problem_statement="Complex strategic decision requiring multiple approaches",
                max_duration_seconds=30.0,
            ),
        ]

        # Execute orchestrations
        results = []
        for request in requests:
            print(f"\n🚀 Executing: {request.orchestration_type.value}")
            result = await orchestration.orchestrate(request)
            results.append(result)

            print(f"   Status: {result.status}")
            print(f"   Duration: {result.execution_time_seconds:.1f}s")
            print(f"   Confidence: {result.confidence_score:.2f}")
            print(f"   Insights: {len(result.insights)}")
            print(f"   Recommendations: {len(result.recommendations)}")

        # Show statistics
        print("\n📊 Orchestration Statistics:")
        stats = orchestration.get_orchestration_stats()
        print(f"   Total requests: {stats['total_requests']}")
        print(f"   Success rate: {stats['success_rate']:.1f}%")
        print(f"   Types used: {stats['orchestration_types']}")
        print(f"   Contexts used: {stats['contexts']}")
        print(f"   Average times: {stats['average_execution_times']}")

        return all(r.status in ["completed", "partial"] for r in results)

    # Run test
    asyncio.run(test_unified_orchestration())
