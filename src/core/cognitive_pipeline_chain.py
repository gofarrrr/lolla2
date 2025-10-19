"""
CognitivePipelineChain - Advanced Pipeline with Context Engineering and Reflection
Implements the sophisticated 7-stage pipeline from IMPLEMENTATION_HANDOVER.md
"""

import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from src.core.unified_context_stream import UnifiedContextStream, ContextEventType
from src.core.context_engineering.context_compiler import (
    get_stage_compiler,
    CompiledContext,
)
from src.engine.core.feature_flags import FeatureFlagService as FeatureFlagManager
from src.core.twelve_factor_compliance import TwelveFactorAgent
from src.engine.core.llm_manager import get_llm_manager
from src.engine.core.structured_logging import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineStageResult:
    """Result from a pipeline stage execution"""

    stage_name: str
    success: bool
    output: Any
    context_metrics: Optional[Dict[str, Any]] = None
    duration_ms: int = 0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfiguration:
    """Configuration for the cognitive pipeline"""

    enable_context_engineering: bool = True
    enable_reflection: bool = True
    enable_breadth_mode: bool = False
    max_iterations: int = 3
    context_token_limit: int = 4000
    parallel_timeout_seconds: int = 60
    cache_ttl_seconds: int = 300
    enable_performance_monitoring: bool = True


class CognitivePipelineChain(TwelveFactorAgent):
    """
    Advanced cognitive pipeline with context engineering and reflection loops.
    Implements the sophisticated pipeline architecture from IMPLEMENTATION_HANDOVER.md.
    """

    def __init__(
        self,
        context_stream: UnifiedContextStream,
        feature_flags: Optional[FeatureFlagManager] = None,
        config: Optional[PipelineConfiguration] = None,
    ):
        """Initialize the cognitive pipeline chain"""
        from src.core.twelve_factor_compliance import TwelveFactorConfig

        twelve_factor_config = TwelveFactorConfig(
            service_name="cognitive_pipeline_chain"
        )
        super().__init__(twelve_factor_config)

        self.context_stream = context_stream
        self.feature_flags = feature_flags or FeatureFlagManager()
        self.config = config or PipelineConfiguration()

        # Pipeline stages
        self.stages = [
            "socratic",
            "problem_structuring",
            "consultant_selection",
            "parallel_analysis",
            "devils_advocate",
            "senior_advisor",
        ]

        # Stage results cache for reflection
        self._stage_results_cache: Dict[str, PipelineStageResult] = {}
        self._reflection_history: List[Dict[str, Any]] = []

        # Performance metrics
        self._metrics = {
            "total_executions": 0,
            "context_compressions": 0,
            "average_compression_ratio": 0.0,
            "reflection_loops": 0,
            "total_duration_ms": 0,
        }

        logger.info(f"CognitivePipelineChain initialized with config: {self.config}")

    async def execute(
        self,
        problem_statement: str,
        context: Optional[Dict[str, Any]] = None,
        iteration: int = 0,
    ) -> Dict[str, Any]:
        """
        Execute the full cognitive pipeline with context engineering and reflection.

        Args:
            problem_statement: The problem to analyze
            context: Optional initial context
            iteration: Current iteration number (for reflection loops)

        Returns:
            Dict containing final analysis results and metadata
        """
        start_time = time.time()
        self._metrics["total_executions"] += 1

        # Initialize context
        pipeline_context = context or {}
        pipeline_context["problem_statement"] = problem_statement
        pipeline_context["iteration"] = iteration

        # Log pipeline start
        self.context_stream.add_event(
            ContextEventType.ENGAGEMENT_STARTED,
            {
                "problem": problem_statement,
                "iteration": iteration,
                "config": self.config.__dict__,
            },
        )

        results = {}

        try:
            # Execute each stage with context engineering
            for stage_name in self.stages:
                logger.info(f"Executing stage: {stage_name}")

                # Apply context engineering if enabled
                if self.config.enable_context_engineering:
                    pipeline_context = await self._apply_context_engineering(
                        stage_name, pipeline_context
                    )

                # Execute the stage
                stage_result = await self._execute_stage(stage_name, pipeline_context)

                # Store result
                results[stage_name] = stage_result
                self._stage_results_cache[stage_name] = stage_result

                # Update context with stage output
                if stage_result.success and stage_result.output:
                    pipeline_context[f"{stage_name}_output"] = stage_result.output

                # Check for early termination
                if not stage_result.success and stage_name in [
                    "socratic",
                    "problem_structuring",
                ]:
                    logger.warning(
                        f"Critical stage {stage_name} failed, terminating pipeline"
                    )
                    break

            # Apply reflection if enabled and not at max iterations
            if (
                self.config.enable_reflection
                and iteration < self.config.max_iterations
                and await self._should_reflect(results)
            ):

                reflection_result = await self._apply_reflection(
                    results, pipeline_context
                )

                if reflection_result.get("needs_iteration"):
                    logger.info(f"Reflection triggered iteration {iteration + 1}")
                    self._reflection_history.append(reflection_result)

                    # Recursive call for next iteration
                    return await self.execute(
                        problem_statement,
                        reflection_result.get("refined_context", pipeline_context),
                        iteration + 1,
                    )

            # Calculate final metrics
            duration_ms = int((time.time() - start_time) * 1000)
            self._metrics["total_duration_ms"] += duration_ms

            # Log pipeline completion
            self.context_stream.add_event(
                ContextEventType.PROCESSING_COMPLETE,
                {
                    "iteration": iteration,
                    "duration_ms": duration_ms,
                    "stages_completed": len(results),
                    "success": all(r.success for r in results.values()),
                },
            )

            return {
                "success": all(r.success for r in results.values()),
                "results": {k: v.output for k, v in results.items()},
                "metadata": {
                    "iteration": iteration,
                    "duration_ms": duration_ms,
                    "context_metrics": self._get_context_metrics(results),
                    "reflection_history": self._reflection_history,
                    "performance_metrics": self._metrics,
                },
            }

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            self.context_stream.add_event(
                ContextEventType.ERROR_OCCURRED,
                {
                    "error": str(e),
                    "stage": "pipeline_execution",
                    "iteration": iteration,
                },
            )
            raise

    async def _apply_context_engineering(
        self, stage_name: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply context engineering to optimize context for a specific stage.

        Args:
            stage_name: Name of the pipeline stage
            context: Current pipeline context

        Returns:
            Engineered context optimized for the stage
        """
        try:
            # Get appropriate compiler for the stage
            compiler = get_stage_compiler(
                stage_name, token_limit=self.config.context_token_limit
            )

            # Compile the context
            compiled_context: CompiledContext = compiler.compile(context)

            # Update metrics
            self._metrics["context_compressions"] += 1
            self._metrics["average_compression_ratio"] = (
                self._metrics["average_compression_ratio"]
                * (self._metrics["context_compressions"] - 1)
                + compiled_context.compression_ratio
            ) / self._metrics["context_compressions"]

            # Log compression
            logger.info(
                f"Context engineered for {stage_name}: "
                f"compression_ratio={compiled_context.compression_ratio:.2f}, "
                f"relevance={compiled_context.relevance_score:.2f}"
            )

            # Log V5.4 context engineering event
            self.context_stream.add_event(
                ContextEventType.SYSTEM_STATE,
                {
                    "event": "context_engineered",
                    "stage": stage_name,
                    "compression_ratio": compiled_context.compression_ratio,
                    "relevance_score": compiled_context.relevance_score,
                    "tokens_saved": compiled_context.tokens_saved,
                    "lossless_compression": True,
                },
            )

            # Return engineered context
            engineered = compiled_context.structured_data.copy()
            engineered["_context_metrics"] = {
                "raw_size": compiled_context.raw_size,
                "compiled_size": compiled_context.compiled_size,
                "compression_ratio": compiled_context.compression_ratio,
                "relevance_score": compiled_context.relevance_score,
            }

            return engineered

        except Exception as e:
            logger.warning(f"Context engineering failed for {stage_name}: {e}")
            # Return original context on failure
            return context

    async def _execute_stage(
        self, stage_name: str, context: Dict[str, Any]
    ) -> PipelineStageResult:
        """
        Execute a single pipeline stage.

        Args:
            stage_name: Name of the stage to execute
            context: Context for the stage

        Returns:
            PipelineStageResult with execution details
        """
        start_time = time.time()

        # Phase 6: Emit PIPELINE_STAGE_STARTED event
        self.context_stream.add_event(
            ContextEventType.PIPELINE_STAGE_STARTED,
            {
                "stage_name": stage_name,
                "iteration": context.get("iteration", 0),
                "start_time": datetime.utcnow().isoformat(),
                "context_keys": list(context.keys()) if context else [],
            },
        )

        try:
            # Stage-specific execution logic
            if stage_name == "socratic":
                output = await self._execute_socratic_engine(context)
            elif stage_name == "problem_structuring":
                output = await self._execute_problem_structuring(context)
            elif stage_name == "consultant_selection":
                output = await self._execute_consultant_selection(context)
            elif stage_name == "parallel_analysis":
                output = await self._execute_parallel_analysis(context)
            elif stage_name == "devils_advocate":
                output = await self._execute_devils_advocate(context)
            elif stage_name == "senior_advisor":
                output = await self._execute_senior_advisor(context)
            else:
                raise ValueError(f"Unknown stage: {stage_name}")

            duration_ms = int((time.time() - start_time) * 1000)

            # Phase 6: Emit PIPELINE_STAGE_COMPLETED event for success
            result = PipelineStageResult(
                stage_name=stage_name,
                success=True,
                output=output,
                context_metrics=context.get("_context_metrics"),
                duration_ms=duration_ms,
                metadata={"timestamp": datetime.utcnow().isoformat()},
            )

            self.context_stream.add_event(
                ContextEventType.PIPELINE_STAGE_COMPLETED,
                {
                    "stage_name": stage_name,
                    "success": True,
                    "duration_ms": duration_ms,
                    "iteration": context.get("iteration", 0),
                    "has_output": output is not None,
                    "end_time": datetime.utcnow().isoformat(),
                },
            )

            return result

        except Exception as e:
            logger.error(f"Stage {stage_name} failed: {e}")
            duration_ms = int((time.time() - start_time) * 1000)

            # Phase 6: Emit PIPELINE_STAGE_COMPLETED event for failure
            self.context_stream.add_event(
                ContextEventType.PIPELINE_STAGE_COMPLETED,
                {
                    "stage_name": stage_name,
                    "success": False,
                    "duration_ms": duration_ms,
                    "iteration": context.get("iteration", 0),
                    "error": str(e),
                    "end_time": datetime.utcnow().isoformat(),
                },
            )

            return PipelineStageResult(
                stage_name=stage_name,
                success=False,
                output=None,
                error=str(e),
                duration_ms=duration_ms,
            )

    async def _should_reflect(self, results: Dict[str, PipelineStageResult]) -> bool:
        """
        Determine if reflection should be triggered based on results.

        Args:
            results: Stage results from the current iteration

        Returns:
            True if reflection should occur
        """
        # Check for conflicting recommendations
        if "parallel_analysis" in results and "devils_advocate" in results:
            parallel_output = results["parallel_analysis"].output
            devils_output = results["devils_advocate"].output

            if parallel_output and devils_output:
                # Simple heuristic: reflect if devil's advocate found significant issues
                if isinstance(devils_output, dict):
                    bias_count = len(devils_output.get("biases_detected", []))
                    assumption_count = len(
                        devils_output.get("assumptions_challenged", [])
                    )

                    if bias_count > 2 or assumption_count > 3:
                        logger.info(
                            f"Reflection triggered: {bias_count} biases, {assumption_count} assumptions"
                        )
                        return True

        # Check for low confidence in senior advisor
        if "senior_advisor" in results:
            advisor_output = results["senior_advisor"].output
            if isinstance(advisor_output, dict):
                confidence = advisor_output.get("confidence_level", 1.0)
                if confidence < 0.7:
                    logger.info(f"Reflection triggered: low confidence {confidence}")
                    return True

        return False

    async def _apply_reflection(
        self, results: Dict[str, PipelineStageResult], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply reflection to refine the analysis based on current results.

        Args:
            results: Current pipeline results
            context: Current pipeline context

        Returns:
            Reflection result with refined context
        """
        self._metrics["reflection_loops"] += 1

        # Phase 6: Emit PIPELINE_REFLECTION_TRIGGERED event
        self.context_stream.add_event(
            ContextEventType.PIPELINE_REFLECTION_TRIGGERED,
            {
                "iteration": context.get("iteration", 0),
                "stages_completed": len(results),
                "successful_stages": len([r for r in results.values() if r.success]),
                "failed_stages": len([r for r in results.values() if not r.success]),
                "reflection_count": self._metrics["reflection_loops"],
                "rationale": "Pipeline results evaluation triggered reflection loop",
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

        reflection_prompt = self._build_reflection_prompt(results, context)

        # Use LLM for reflection
        llm_manager = get_llm_manager(context_stream=self.context_stream)

        reflection_response = await llm_manager.call_llm(
            messages=[
                {
                    "role": "system",
                    "content": "You are a strategic advisor performing critical reflection on analysis results.",
                },
                {"role": "user", "content": reflection_prompt},
            ],
            temperature=0.3,
        )

        # Parse reflection response
        try:
            reflection_data = json.loads(reflection_response.get("content", "{}"))
        except:
            reflection_data = {"needs_iteration": False}

        # Build refined context
        refined_context = context.copy()
        if reflection_data.get("refinements"):
            refined_context.update(reflection_data["refinements"])

        return {
            "needs_iteration": reflection_data.get("needs_iteration", False),
            "refined_context": refined_context,
            "reflection_insights": reflection_data.get("insights", []),
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _build_reflection_prompt(
        self, results: Dict[str, PipelineStageResult], context: Dict[str, Any]
    ) -> str:
        """Build a prompt for reflection based on current results"""
        prompt = f"""
        Analyze the following pipeline results and determine if another iteration would improve the analysis:
        
        Problem: {context.get('problem_statement', 'Unknown')}
        Current Iteration: {context.get('iteration', 0)}
        
        Stage Results Summary:
        """

        for stage_name, result in results.items():
            if result.success and result.output:
                prompt += f"\n{stage_name}:\n"
                if isinstance(result.output, dict):
                    prompt += (
                        f"  - Key findings: {result.output.get('summary', 'N/A')}\n"
                    )
                else:
                    prompt += f"  - Output: {str(result.output)[:200]}...\n"

        prompt += """
        
        Please provide your reflection in JSON format:
        {
            "needs_iteration": true/false,
            "refinements": {
                // Key-value pairs to add/update in context
            },
            "insights": [
                // List of key insights from reflection
            ],
            "rationale": "Explanation for the decision"
        }
        """

        return prompt

    def _get_context_metrics(
        self, results: Dict[str, PipelineStageResult]
    ) -> Dict[str, Any]:
        """Aggregate context metrics from all stages"""
        metrics = {
            "total_compression_ratio": 0.0,
            "average_relevance_score": 0.0,
            "total_tokens_saved": 0,
        }

        count = 0
        for result in results.values():
            if result.context_metrics:
                metrics["total_compression_ratio"] += result.context_metrics.get(
                    "compression_ratio", 0
                )
                metrics["average_relevance_score"] += result.context_metrics.get(
                    "relevance_score", 0
                )
                metrics["total_tokens_saved"] += result.context_metrics.get(
                    "raw_size", 0
                ) - result.context_metrics.get("compiled_size", 0)
                count += 1

        if count > 0:
            metrics["total_compression_ratio"] /= count
            metrics["average_relevance_score"] /= count

        return metrics

    # Stage execution methods (simplified implementations)
    async def _execute_socratic_engine(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Socratic questioning stage"""
        # This would integrate with existing Socratic engine
        return {
            "questions": ["What are the key assumptions?", "What alternatives exist?"],
            "summary": "Generated Socratic questions for deeper exploration",
        }

    async def _execute_problem_structuring(
        self, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute problem structuring stage"""
        # This would integrate with existing problem structuring
        return {
            "structure": {"core_issue": "Main problem", "sub_issues": []},
            "summary": "Structured problem into analyzable components",
        }

    async def _execute_consultant_selection(
        self, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute consultant selection stage"""
        # This would use the CognitiveConsultantRouter (to be implemented)
        return {
            "selected_consultants": ["Strategic Analyst", "Risk Assessor"],
            "summary": "Selected optimal consultant mix",
        }

    async def _execute_parallel_analysis(
        self, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute parallel consultant analysis"""
        # This would use EnhancedParallelCognitiveForges (to be implemented)
        return {
            "analyses": [{"consultant": "Strategic Analyst", "findings": {}}],
            "summary": "Completed parallel consultant analyses",
        }

    async def _execute_devils_advocate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute devil's advocate critique"""
        # This would integrate with existing devils advocate system
        return {
            "biases_detected": [],
            "assumptions_challenged": [],
            "summary": "Performed critical analysis",
        }

    async def _execute_senior_advisor(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute senior advisor synthesis"""
        # This would integrate with existing senior advisor
        return {
            "recommendation": "Strategic recommendation",
            "confidence_level": 0.85,
            "summary": "Synthesized final recommendation",
        }
