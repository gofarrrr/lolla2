"""
METIS Dynamic N-Way Execution Engine
The "CPU" that executes your sophisticated cognitive recipes

This engine takes N-Way Interaction Clusters and executes their instructional_cue_apce
using intelligent orchestration, breaking complex cognitive tasks into manageable steps.

Architecture:
1. Query Analysis → Best N-Way Cluster Selection
2. Consultant Selection based on effectiveness scores
3. Multi-Step Execution of instructional_cue_apce
4. Quality validation and result synthesis

Based on your existing IP:
- src/cognitive_architecture/nway_interactions_system.py (14+ clusters)
- Database instructional_cue_apce fields
- Consultant effectiveness scoring
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass

from src.cognitive_architecture.cognitive_clusters import (
    NWayInteractionCluster,
    CognitiveExecutionPlan,
    CognitiveExecutionResult,
    CognitiveExecutionMode,
    CognitiveStep,
    rank_clusters_by_activation,
)
from src.cognitive_architecture.mental_models_system import (
    ConsultantRole,
    MentalModelsLibrary,
)
from src.cognitive_architecture.enhanced_nway_interactions_system import (
    Enhanced21ClusterNWayLibrary,
)
from src.engine.adapters.llm_client import get_resilient_llm_client, CognitiveCallContext  # Migrated

logger = logging.getLogger(__name__)


@dataclass
class CognitiveOrchestratorConfig:
    """Configuration for the N-Way execution engine"""

    max_execution_time_seconds: int = 300
    enable_parallel_execution: bool = True
    quality_threshold: float = 0.7
    max_retry_attempts: int = 3
    enable_devils_advocate: bool = True
    research_integration_enabled: bool = True
    verbose_logging: bool = False


class DynamicNWayExecutionEngine:
    """
    The cognitive "CPU" that executes sophisticated N-Way interaction patterns

    This is the core engine that:
    1. Analyzes queries to select optimal cognitive clusters
    2. Chooses the most effective consultant for execution
    3. Executes instructional_cue_apce as multi-step cognitive processes
    4. Validates results and provides comprehensive tracing
    """

    def __init__(self, config: Optional[CognitiveOrchestratorConfig] = None):
        self.config = config or CognitiveOrchestratorConfig()
        self.mental_models_library = MentalModelsLibrary()
        self.nway_library = Enhanced21ClusterNWayLibrary()
        self.llm_client = get_resilient_llm_client()

        # Load and convert existing N-Way clusters to new format
        self.cognitive_clusters = self._convert_existing_nway_clusters()

        # Execution state
        self.active_executions: Dict[str, CognitiveExecutionPlan] = {}
        self.execution_history: List[CognitiveExecutionResult] = []

        logger.info(
            f"DynamicNWayExecutionEngine initialized with {len(self.cognitive_clusters)} cognitive clusters"
        )

    def _convert_existing_nway_clusters(self) -> List[NWayInteractionCluster]:
        """Convert existing NWayInteraction objects to new NWayInteractionCluster format"""
        converted_clusters = []

        for cluster_id, nway_interaction in self.nway_library.interactions.items():
            # Convert to new enhanced format
            cluster = NWayInteractionCluster(
                cluster_id=cluster_id,
                name=nway_interaction.name,
                description=nway_interaction.research_basis,
                participating_models=nway_interaction.participating_models,
                consultant_effectiveness={
                    role: effectiveness
                    for role, effectiveness in nway_interaction.consultant_effectiveness.items()
                },
                enhanced_capabilities=nway_interaction.enhanced_capabilities,
                cognitive_multiplier=nway_interaction.complexity_multiplier,
                research_basis=nway_interaction.research_basis,
            )

            # Create activation conditions from existing data
            for condition_text in nway_interaction.activation_conditions:
                from src.cognitive_architecture.cognitive_clusters import (
                    ActivationConditionType,
                    create_activation_condition,
                )

                condition = create_activation_condition(
                    ActivationConditionType.QUERY_KEYWORD,
                    condition_text,
                    weight=1.0,
                    description=f"Activation condition: {condition_text}",
                )
                cluster.activation_conditions.append(condition)

            # Create instructional cue - this is the golden field!
            from src.cognitive_architecture.cognitive_clusters import (
                create_instructional_cue_from_text,
            )

            cue_text = getattr(
                nway_interaction, "instructional_cue_apce", cluster.research_basis
            )
            instructional_cue = create_instructional_cue_from_text(
                raw_cue=cue_text,  # Use the actual instructional_cue_apce from database
                execution_mode=CognitiveExecutionMode.MULTI_STEP,
            )
            cluster.instructional_cue = instructional_cue

            converted_clusters.append(cluster)

        return converted_clusters

    async def execute_cognitive_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        preferred_consultant: Optional[ConsultantRole] = None,
    ) -> CognitiveExecutionResult:
        """
        Main execution method: Analyze query and execute optimal cognitive approach

        This is the core method that implements the Interaction-Led architecture:
        Query → Select Best N-Way → Select Best Consultant → Execute Multi-Step
        """
        execution_id = f"exec_{int(time.time())}"
        start_time = datetime.utcnow()

        try:
            logger.info(f"🧠 Starting cognitive execution for query: '{query[:50]}...'")

            # Step 1: Analyze query and select optimal cognitive cluster
            selected_cluster, activation_score = (
                await self._select_optimal_cognitive_cluster(query, context)
            )
            if not selected_cluster:
                raise ValueError("No suitable cognitive cluster found for query")

            logger.info(
                f"🎯 Selected cluster: {selected_cluster.name} (score: {activation_score:.3f})"
            )

            # Step 2: Select Strategic Trio - Multiple consultants for Multi-Single-Agent execution
            strategic_trio = selected_cluster.get_strategic_trio(max_consultants=3)
            selected_consultants = [consultant for consultant, _ in strategic_trio]

            # Optional: Allow user to override with preferred consultant(s)
            if (
                preferred_consultant
                and preferred_consultant in selected_cluster.consultant_effectiveness
            ):
                if preferred_consultant not in selected_consultants:
                    selected_consultants = [
                        preferred_consultant
                    ] + selected_consultants[:2]

            logger.info(
                f"🎯 Strategic Trio selected: {', '.join([c.value for c in selected_consultants])} (Multi-Single-Agent execution)"
            )

            # Step 3: Create execution plan with Strategic Trio
            execution_plan = self._create_execution_plan(
                execution_id,
                query,
                context or {},
                selected_cluster,
                selected_consultants,
            )
            self.active_executions[execution_id] = execution_plan

            # Step 4: Execute the cognitive process
            result = await self._execute_cognitive_plan(execution_plan)

            # Step 5: Quality validation and enhancement
            enhanced_result = await self._validate_and_enhance_result(
                result, selected_cluster
            )

            # Store in execution history
            self.execution_history.append(enhanced_result)

            # Clean up active execution
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]

            total_time = (datetime.utcnow() - start_time).total_seconds()
            logger.info(f"✅ Cognitive execution completed in {total_time:.1f}s")

            return enhanced_result

        except Exception as e:
            logger.error(f"❌ Cognitive execution failed: {e}")

            # Return error result
            error_result = CognitiveExecutionResult(
                plan_id="",
                execution_id=execution_id,
                final_output=f"Execution failed: {str(e)}",
                confidence_score=0.0,
                errors_encountered=[str(e)],
                fallback_used=True,
                completed_at=datetime.utcnow(),
            )

            return error_result

    async def _select_optimal_cognitive_cluster(
        self, query: str, context: Optional[Dict[str, Any]]
    ) -> Tuple[Optional[NWayInteractionCluster], float]:
        """
        Select the best cognitive cluster for the query using activation scoring

        This implements the core intelligence: matching queries to cognitive approaches
        """
        query_context = {
            "query": query,
            "problem_type": context.get("problem_type", "") if context else "",
            "complexity_score": context.get("complexity_score", 5) if context else 5,
            "domain": context.get("domain", "") if context else "",
        }

        # Rank all clusters by activation score
        ranked_clusters = rank_clusters_by_activation(
            self.cognitive_clusters, query_context
        )

        if not ranked_clusters:
            return None, 0.0

        # Log top candidates for transparency
        logger.info("🔍 Top cognitive cluster candidates:")
        for i, (cluster, score) in enumerate(ranked_clusters[:3]):
            logger.info(f"  {i+1}. {cluster.name}: {score:.3f}")

        # Return best match
        best_cluster, best_score = ranked_clusters[0]

        # Only return if score meets minimum threshold
        if best_score >= best_cluster.activation_threshold:
            return best_cluster, best_score
        else:
            logger.warning(
                f"Best cluster score {best_score:.3f} below threshold {best_cluster.activation_threshold}"
            )
            return None, best_score

    def _create_execution_plan(
        self,
        execution_id: str,
        query: str,
        context: Dict[str, Any],
        cluster: NWayInteractionCluster,
        consultants: List[ConsultantRole],
    ) -> CognitiveExecutionPlan:
        """Create detailed execution plan for the selected cluster"""

        plan = CognitiveExecutionPlan(
            plan_id=f"plan_{execution_id}",
            query=query,
            query_context=context,
            primary_cluster=cluster,
            selected_consultants=consultants,  # Strategic Trio
            execution_mode=(
                cluster.instructional_cue.execution_mode
                if cluster.instructional_cue
                else CognitiveExecutionMode.MULTI_STEP
            ),
            confidence_score=sum(
                cluster.consultant_effectiveness.get(c, 0.5) for c in consultants
            )
            / len(consultants),  # Average confidence
        )

        # Calculate execution parameters
        if cluster.instructional_cue:
            plan.total_steps = len(cluster.instructional_cue.cognitive_steps)
            plan.estimated_duration_seconds = (
                cluster.instructional_cue.estimated_processing_time
            )
        else:
            plan.total_steps = len(cluster.participating_models)
            plan.estimated_duration_seconds = (
                plan.total_steps * 30
            )  # 30 seconds per model

        # Create success criteria
        plan.success_criteria = cluster.enhanced_capabilities.copy()
        plan.validation_checkpoints = [
            "logical_consistency",
            "evidence_quality",
            "actionability",
        ]

        return plan

    async def _execute_cognitive_plan(
        self, plan: CognitiveExecutionPlan
    ) -> CognitiveExecutionResult:
        """
        Execute Strategic Trio - Multiple consultants in parallel (Multi-Single-Agent paradigm)

        Key principles:
        - NO coordination between consultants
        - Each consultant works independently with same cluster/query
        - ALL perspectives preserved without synthesis
        - Human receives multiple independent analyses
        """
        result = CognitiveExecutionResult(
            plan_id=plan.plan_id,
            execution_id=plan.plan_id.replace("plan_", "exec_"),
            final_output="",  # Will be summary of all consultants
            confidence_score=plan.confidence_score,
            consultants_used=plan.selected_consultants,  # Multiple consultants
            consultant_perspectives={},  # Individual outputs preserved
            clusters_activated=[plan.primary_cluster.cluster_id],
        )

        try:
            logger.info(
                f"🚀 Executing {len(plan.selected_consultants)} consultants in parallel (Multi-Single-Agent)"
            )

            # Execute all consultants in parallel - NO coordination between them
            consultant_tasks = []
            for consultant in plan.selected_consultants:
                task = self._execute_consultant_independently(
                    consultant=consultant, plan=plan, execution_mode=plan.execution_mode
                )
                consultant_tasks.append((consultant, task))

            # Wait for ALL consultants to complete independently
            consultant_outputs = {}
            total_tokens = 0
            total_cost = 0.0

            for consultant, task in consultant_tasks:
                try:
                    consultant_output = await task
                    consultant_outputs[consultant] = consultant_output

                    # Track performance metrics
                    total_tokens += consultant_output.get("tokens", 0)
                    total_cost += consultant_output.get("cost", 0.0)

                    # Store individual perspective (NO synthesis)
                    result.consultant_perspectives[consultant] = consultant_output.get(
                        "output", ""
                    )

                    logger.info(
                        f"✅ {consultant.value} analysis completed independently"
                    )

                except Exception as e:
                    error_msg = f"{consultant.value} execution failed: {e}"
                    result.errors_encountered.append(error_msg)
                    result.consultant_perspectives[consultant] = f"Analysis failed: {e}"
                    logger.error(f"❌ {error_msg}")

            # Update performance metrics
            result.token_usage = {
                "total_tokens": total_tokens,
                "prompt_tokens": total_tokens // 2,  # Approximate
                "completion_tokens": total_tokens // 2,
            }
            result.cost_breakdown = {"total_cost": total_cost}

            # Create final output: Present ALL perspectives without synthesis
            result.final_output = self._format_multi_consultant_output(
                consultant_outputs, plan.primary_cluster
            )

            # Calculate overall confidence (average of successful consultants)
            successful_consultants = [
                c
                for c in plan.selected_consultants
                if c not in [e.split()[0] for e in result.errors_encountered]
            ]
            if successful_consultants:
                result.confidence_score = sum(
                    plan.primary_cluster.consultant_effectiveness.get(c, 0.5)
                    for c in successful_consultants
                ) / len(successful_consultants)

            logger.info(
                f"🎯 Multi-Single-Agent execution completed: {len(successful_consultants)}/{len(plan.selected_consultants)} consultants successful"
            )

            return result

        except Exception as e:
            result.errors_encountered.append(
                f"Multi-consultant execution failed: {str(e)}"
            )
            result.fallback_used = True
            result.final_output = f"Strategic Trio execution failed: {str(e)}"
            return result

    async def _execute_consultant_independently(
        self,
        consultant: ConsultantRole,
        plan: CognitiveExecutionPlan,
        execution_mode: CognitiveExecutionMode,
    ) -> Dict[str, Any]:
        """
        Execute single consultant analysis independently - NO coordination with others

        This is the core of Multi-Single-Agent: each consultant works alone
        """
        cluster = plan.primary_cluster
        consultant_role = consultant.value

        try:
            if execution_mode == CognitiveExecutionMode.SINGLE_PROMPT:
                # Single comprehensive analysis
                output = await self._execute_consultant_single_prompt(consultant, plan)
            else:
                # Multi-step analysis (default)
                output = await self._execute_consultant_multi_step(consultant, plan)

            return {
                "consultant": consultant,
                "output": output["content"],
                "tokens": output.get("tokens", 0),
                "cost": output.get("cost", 0.0),
                "success": True,
            }

        except Exception as e:
            return {
                "consultant": consultant,
                "output": f"Independent analysis failed: {str(e)}",
                "tokens": 0,
                "cost": 0.0,
                "success": False,
            }

    async def _execute_consultant_single_prompt(
        self, consultant: ConsultantRole, plan: CognitiveExecutionPlan
    ) -> Dict[str, Any]:
        """Execute consultant analysis as single comprehensive prompt"""
        cluster = plan.primary_cluster
        consultant_role = consultant.value

        # Build consultant-specific system prompt
        system_prompt = f"""
You are a {consultant_role} working independently on a sophisticated cognitive analysis.

COGNITIVE APPROACH: {cluster.name}
PARTICIPATING MENTAL MODELS: {', '.join(cluster.participating_models)}
ENHANCED CAPABILITIES: {', '.join(cluster.enhanced_capabilities)}

INSTRUCTIONAL CUE:
{cluster.instructional_cue.raw_cue if cluster.instructional_cue else cluster.research_basis}

IMPORTANT: You are working independently. Provide your unique {consultant_role} perspective.
Other consultants will provide their own independent analyses - do not try to cover their roles.
Focus on what makes your perspective valuable and distinct.

Please apply this cognitive approach to analyze the following query comprehensively from your {consultant_role} viewpoint.
Provide structured output with clear reasoning, evidence, and actionable insights specific to your expertise.
"""

        # Execute LLM call
        call_context = CognitiveCallContext(
            consultant_role=consultant_role,
            cognitive_function="independent_nway_analysis",
            complexity_level="high",
        )

        response = await self.llm_client.call(
            system_prompt=system_prompt,
            user_prompt=plan.query,
            call_context=call_context,
        )

        return {
            "content": response.content,
            "tokens": response.total_tokens,
            "cost": response.cost_usd,
            "model": response.model_name,
        }

    async def _execute_consultant_multi_step(
        self, consultant: ConsultantRole, plan: CognitiveExecutionPlan
    ) -> Dict[str, Any]:
        """Execute consultant analysis as multi-step process"""
        cluster = plan.primary_cluster
        consultant_role = consultant.value

        # Get or generate cognitive steps for this consultant
        if cluster.instructional_cue and cluster.instructional_cue.cognitive_steps:
            steps = cluster.instructional_cue.cognitive_steps
        else:
            steps = self._generate_steps_from_models(cluster.participating_models)

        step_outputs = []
        total_tokens = 0
        total_cost = 0.0

        for i, step in enumerate(steps, 1):
            try:
                # Build step-specific prompt for this consultant
                step_prompt = f"""
You are a {consultant_role} executing step {i} of a multi-step analysis.

STEP: {step.step_name}
MENTAL MODEL: {step.mental_model}
INSTRUCTION: {step.instruction}
EXPECTED OUTPUT: {step.expected_output}

Apply the {step.mental_model} framework from your {consultant_role} perspective.
Focus on insights unique to your expertise and this mental model.

QUERY TO ANALYZE:
{plan.query}
"""

                call_context = CognitiveCallContext(
                    consultant_role=consultant_role,
                    cognitive_function=f"step_{i}_{step.mental_model.lower().replace(' ', '_')}",
                    complexity_level="medium",
                )

                response = await self.llm_client.call(
                    system_prompt=step_prompt,
                    user_prompt=plan.query,
                    call_context=call_context,
                )

                step_outputs.append(
                    f"### Step {i}: {step.step_name}\n{response.content}\n"
                )
                total_tokens += response.total_tokens
                total_cost += response.cost_usd

            except Exception as e:
                step_outputs.append(
                    f"### Step {i}: {step.step_name}\nStep failed: {e}\n"
                )

        # Combine all steps for this consultant
        combined_output = f"# {consultant_role} Independent Analysis\n\n" + "\n".join(
            step_outputs
        )

        return {"content": combined_output, "tokens": total_tokens, "cost": total_cost}

    def _format_multi_consultant_output(
        self,
        consultant_outputs: Dict[ConsultantRole, Dict[str, Any]],
        cluster: NWayInteractionCluster,
    ) -> str:
        """
        Format multiple consultant perspectives WITHOUT synthesis

        Key principle: Present all perspectives independently - NO merging or consensus
        """
        if not consultant_outputs:
            return "No consultant analyses completed successfully."

        # Header with cluster information
        formatted_output = f"# {cluster.name} - Multi-Single-Agent Analysis\n\n"
        formatted_output += f"**Cognitive Approach**: {cluster.research_basis}\n\n"
        formatted_output += f"**Strategic Trio Execution**: {len(consultant_outputs)} independent consultants analyzed this query in parallel\n\n"

        formatted_output += "---\n\n"

        # Present each consultant's perspective independently
        for consultant, output in consultant_outputs.items():
            if output.get("success", False):
                formatted_output += f"## {consultant.value} Perspective\n\n"
                formatted_output += "**Independent Analysis** (No coordination with other consultants):\n\n"
                formatted_output += output.get("output", "No output available") + "\n\n"
                formatted_output += f"*Tokens: {output.get('tokens', 0)} | Cost: ${output.get('cost', 0.0):.4f}*\n\n"
                formatted_output += "---\n\n"

        # Footer emphasizing Multi-Single-Agent approach
        formatted_output += "## Multi-Single-Agent Execution Notes\n\n"
        formatted_output += (
            "- Each consultant worked **independently** with no coordination\n"
        )
        formatted_output += (
            "- All perspectives are preserved without synthesis or merging\n"
        )
        formatted_output += (
            "- You can choose which consultant perspectives to act upon\n"
        )
        formatted_output += (
            "- Consider using Senior Advisor for optional synthesis if desired\n\n"
        )

        formatted_output += f"**Enhanced Capabilities Activated**: {', '.join(cluster.enhanced_capabilities)}\n"

        return formatted_output

    async def _execute_single_prompt(
        self, plan: CognitiveExecutionPlan, result: CognitiveExecutionResult
    ) -> CognitiveExecutionResult:
        """Execute cognitive process as single comprehensive LLM call"""

        # Build comprehensive system prompt
        cluster = plan.primary_cluster
        consultant_role = plan.selected_consultant.value

        system_prompt = f"""
You are a {consultant_role} executing a sophisticated cognitive analysis.

COGNITIVE APPROACH: {cluster.name}
PARTICIPATING MENTAL MODELS: {', '.join(cluster.participating_models)}
ENHANCED CAPABILITIES: {', '.join(cluster.enhanced_capabilities)}

INSTRUCTIONAL CUE:
{cluster.instructional_cue.raw_cue if cluster.instructional_cue else cluster.research_basis}

Please apply this cognitive approach to analyze the following query comprehensively.
Provide structured output with clear reasoning, evidence, and actionable insights.
"""

        try:
            # Execute LLM call
            call_context = CognitiveCallContext(
                consultant_role=plan.selected_consultant.value,
                cognitive_function="nway_analysis",
                complexity_level="high",
            )

            response = await self.llm_client.call(
                system_prompt=system_prompt,
                user_prompt=plan.query,
                call_context=call_context,
            )

            result.final_output = response.content
            result.confidence_score = response.confidence_score
            result.token_usage = {
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "total_tokens": response.total_tokens,
            }
            result.cost_breakdown = {"total_cost": response.cost_usd}

            # Record execution step
            result.steps_executed.append(
                {
                    "step": "single_prompt_execution",
                    "model_used": response.model_name,
                    "tokens": response.total_tokens,
                    "success": True,
                }
            )

            logger.info(
                f"✅ Single-prompt execution completed: {response.total_tokens} tokens, ${response.cost_usd:.4f}"
            )

        except Exception as e:
            result.errors_encountered.append(f"Single prompt execution failed: {e}")
            result.fallback_used = True

        return result

    async def _execute_multi_step(
        self, plan: CognitiveExecutionPlan, result: CognitiveExecutionResult
    ) -> CognitiveExecutionResult:
        """Execute cognitive process as orchestrated multi-step sequence"""

        cluster = plan.primary_cluster
        consultant_role = plan.selected_consultant.value

        # If cluster has defined cognitive steps, use those
        if cluster.instructional_cue and cluster.instructional_cue.cognitive_steps:
            steps = cluster.instructional_cue.cognitive_steps
        else:
            # Generate steps from participating models
            steps = self._generate_steps_from_models(cluster.participating_models)

        step_outputs = {}

        logger.info(f"🔄 Executing {len(steps)} cognitive steps")

        for i, step in enumerate(steps, 1):
            try:
                logger.info(f"  Step {i}: {step.step_name}")

                # Build step-specific prompt
                step_prompt = self._build_step_prompt(
                    step, plan.query, step_outputs, consultant_role
                )

                # Execute step
                call_context = CognitiveCallContext(
                    consultant_role=consultant_role,
                    cognitive_function=f"step_{i}_{step.mental_model.lower().replace(' ', '_')}",
                    complexity_level="medium",
                )

                response = await self.llm_client.call(
                    system_prompt=step_prompt,
                    user_prompt=plan.query,
                    call_context=call_context,
                )

                # Store step output
                step_outputs[step.step_id] = response.content
                result.intermediate_outputs[step.step_id] = response.content

                # Record execution
                result.steps_executed.append(
                    {
                        "step": step.step_name,
                        "mental_model": step.mental_model,
                        "tokens": response.total_tokens,
                        "cost": response.cost_usd,
                        "success": True,
                    }
                )

                # Update token usage
                if "prompt_tokens" not in result.token_usage:
                    result.token_usage = {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    }
                result.token_usage["prompt_tokens"] += response.prompt_tokens
                result.token_usage["completion_tokens"] += response.completion_tokens
                result.token_usage["total_tokens"] += response.total_tokens

                # Update costs
                if "total_cost" not in result.cost_breakdown:
                    result.cost_breakdown["total_cost"] = 0.0
                result.cost_breakdown["total_cost"] += response.cost_usd

            except Exception as e:
                logger.error(f"❌ Step {i} failed: {e}")
                result.errors_encountered.append(
                    f"Step {i} ({step.step_name}) failed: {e}"
                )
                step_outputs[step.step_id] = f"Step failed: {e}"

        # Synthesize final output from all steps
        result.final_output = self._synthesize_step_outputs(step_outputs, cluster)

        logger.info(
            f"✅ Multi-step execution completed: {len(steps)} steps, {result.token_usage.get('total_tokens', 0)} tokens"
        )

        return result

    def _generate_steps_from_models(self, models: List[str]) -> List[CognitiveStep]:
        """Generate execution steps from mental models when no explicit steps defined"""
        steps = []

        for i, model in enumerate(models):
            step = CognitiveStep(
                step_id=f"step_{i+1}",
                step_name=f"Apply {model}",
                mental_model=model,
                instruction=f"Apply {model} to analyze the problem and provide insights",
                expected_output=f"Analysis using {model} framework",
            )
            steps.append(step)

        return steps

    def _build_step_prompt(
        self,
        step: CognitiveStep,
        query: str,
        previous_outputs: Dict[str, str],
        consultant_role: str,
    ) -> str:
        """Build prompt for individual cognitive step"""

        prompt = f"""
You are a {consultant_role} applying the {step.mental_model} mental model.

STEP: {step.step_name}
INSTRUCTION: {step.instruction}
EXPECTED OUTPUT: {step.expected_output}

"""

        # Add context from previous steps if available
        if previous_outputs and step.dependencies:
            prompt += "CONTEXT FROM PREVIOUS STEPS:\n"
            for dep in step.dependencies:
                if dep in previous_outputs:
                    prompt += f"- {dep}: {previous_outputs[dep][:200]}...\n"
            prompt += "\n"

        prompt += f"""
Apply the {step.mental_model} framework to provide focused analysis for this specific step.
Be concise but thorough. Focus on insights unique to this mental model.

QUERY TO ANALYZE:
{query}
"""

        return prompt

    def _synthesize_step_outputs(
        self, step_outputs: Dict[str, str], cluster: NWayInteractionCluster
    ) -> str:
        """Synthesize individual step outputs into coherent final analysis"""

        if not step_outputs:
            return "No analysis steps completed successfully."

        synthesis = f"# {cluster.name} Analysis\n\n"
        synthesis += f"**Cognitive Approach**: {cluster.research_basis}\n\n"

        synthesis += "## Analysis by Mental Model\n\n"
        for step_id, output in step_outputs.items():
            if "failed" not in output.lower():
                synthesis += f"### {step_id}\n{output}\n\n"

        synthesis += "## Integrated Insights\n"
        synthesis += "Based on the multi-model analysis above, key integrated insights emerge from the synergistic application of these mental models.\n\n"

        synthesis += "## Enhanced Capabilities Activated\n"
        for capability in cluster.enhanced_capabilities:
            synthesis += f"- {capability}\n"

        return synthesis

    async def _validate_and_enhance_result(
        self, result: CognitiveExecutionResult, cluster: NWayInteractionCluster
    ) -> CognitiveExecutionResult:
        """Apply quality validation and enhancement to execution result"""

        # Calculate quality scores
        result.internal_consistency_score = self._assess_internal_consistency(
            result.final_output
        )
        result.actionability_score = self._assess_actionability(result.final_output)
        result.novelty_score = self._assess_novelty(result.final_output)
        result.evidence_strength = self._assess_evidence_strength(result.final_output)

        # Overall quality assessment
        quality_scores = [
            result.internal_consistency_score,
            result.actionability_score,
            result.evidence_strength,
        ]
        result.quality_assessment = {
            "overall_quality": sum(quality_scores) / len(quality_scores),
            "internal_consistency": result.internal_consistency_score,
            "actionability": result.actionability_score,
            "novelty": result.novelty_score,
            "evidence_strength": result.evidence_strength,
        }

        # Apply quality threshold
        if result.quality_assessment["overall_quality"] < self.config.quality_threshold:
            result.warnings.append(
                f"Quality score {result.quality_assessment['overall_quality']:.2f} below threshold {self.config.quality_threshold}"
            )

        return result

    def _assess_internal_consistency(self, output: str) -> float:
        """Assess internal consistency of the analysis"""
        # Simple heuristic: check for contradictions and logical flow
        consistency_indicators = [
            "however" not in output.lower()
            or "but" not in output.lower(),  # Fewer contradictions
            len(output.split("\n")) > 5,  # Structured output
            output.count(".") > 3,  # Multiple complete thoughts
        ]
        return sum(consistency_indicators) / len(consistency_indicators)

    def _assess_actionability(self, output: str) -> float:
        """Assess how actionable the insights are"""
        actionable_indicators = [
            "recommend" in output.lower() or "suggest" in output.lower(),
            "should" in output.lower() or "could" in output.lower(),
            "next step" in output.lower() or "action" in output.lower(),
        ]
        return sum(actionable_indicators) / len(actionable_indicators)

    def _assess_novelty(self, output: str) -> float:
        """Assess novelty and insight quality"""
        novelty_indicators = [
            "insight" in output.lower() or "perspective" in output.lower(),
            "unique" in output.lower() or "innovative" in output.lower(),
            len(set(output.lower().split())) / len(output.split())
            > 0.7,  # Vocabulary diversity
        ]
        return sum(novelty_indicators) / len(novelty_indicators)

    def _assess_evidence_strength(self, output: str) -> float:
        """Assess strength of evidence and reasoning"""
        evidence_indicators = [
            "because" in output.lower() or "since" in output.lower(),
            "evidence" in output.lower() or "data" in output.lower(),
            "therefore" in output.lower() or "thus" in output.lower(),
        ]
        return sum(evidence_indicators) / len(evidence_indicators)

    # Public API methods for monitoring and management

    def get_available_clusters(self) -> List[Dict[str, Any]]:
        """Get summary of available cognitive clusters"""
        return [
            {
                "cluster_id": cluster.cluster_id,
                "name": cluster.name,
                "models": cluster.participating_models,
                "capabilities": cluster.enhanced_capabilities,
                "complexity_multiplier": cluster.cognitive_multiplier,
            }
            for cluster in self.cognitive_clusters
        ]

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics and performance metrics"""
        if not self.execution_history:
            return {"total_executions": 0}

        total_executions = len(self.execution_history)
        successful_executions = sum(
            1 for result in self.execution_history if not result.errors_encountered
        )

        avg_quality = (
            sum(
                result.quality_assessment.get("overall_quality", 0.0)
                for result in self.execution_history
            )
            / total_executions
            if total_executions > 0
            else 0.0
        )

        return {
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "success_rate": successful_executions / total_executions,
            "average_quality_score": avg_quality,
            "active_executions": len(self.active_executions),
        }


# Factory function for easy instantiation
def create_nway_execution_engine(
    enable_research: bool = True,
    enable_devils_advocate: bool = True,
    quality_threshold: float = 0.7,
) -> DynamicNWayExecutionEngine:
    """Factory function to create configured N-Way execution engine"""
    config = CognitiveOrchestratorConfig(
        research_integration_enabled=enable_research,
        enable_devils_advocate=enable_devils_advocate,
        quality_threshold=quality_threshold,
    )
    return DynamicNWayExecutionEngine(config)
