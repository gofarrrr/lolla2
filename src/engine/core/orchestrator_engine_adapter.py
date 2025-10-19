"""
Orchestrator Engine Adapter
Sprint 1: Shadow Testing Infrastructure
Purpose: Bridges the state_machine_orchestrator to actual engine implementations

This adapter provides the actual engine implementations for the state machine
orchestrator, extracted from the neural_lace_orchestrator.
"""

import json
from datetime import datetime

from src.config import get_settings
from src.core.structured_logging import get_logger
from src.core.enhanced_llm_manager import get_enhanced_llm_manager
from src.core.immutable_state_manager import get_immutable_state_manager
from src.core.neural_lace_error_framework import get_neural_lace_error_framework
from dataclasses import asdict
from src.engine.models.data_contracts import (
    MetisDataContract,
    ReasoningStep,
    EngagementPhase,
)

settings = get_settings()
logger = get_logger(__name__, component="engine_adapter")


class OrchestratorEngineAdapter:
    """
    Adapter that provides actual engine implementations for the state machine orchestrator.
    Extracted from neural_lace_orchestrator to maintain behavioral parity.
    """

    def __init__(self):
        """Initialize the adapter with required services"""
        self.logger = logger.with_component("orchestrator_engine_adapter")

        # Initialize core services
        self.error_framework = get_neural_lace_error_framework()
        self.llm_manager = get_enhanced_llm_manager(self.error_framework)
        self.state_manager = get_immutable_state_manager()

        self.logger.info("Engine adapter initialized")

    async def execute_problem_structuring(
        self, contract: MetisDataContract
    ) -> MetisDataContract:
        """
        Execute actual problem structuring logic.
        Extracted from neural_lace_orchestrator._execute_phase_1
        """
        self.logger.info("Executing problem structuring engine")

        # Create phase-specific prompt (exact copy from neural_lace)
        problem_structuring_prompt = f"""
You are a strategic problem structuring expert. Your task is to break down and structure this complex problem systematically.

PROBLEM STATEMENT:
{contract.engagement_context.problem_statement}

COMPANY: {contract.engagement_context.client_name}
INDUSTRY: {getattr(contract.engagement_context, 'industry', 'Not specified')}

Your task:
1. Apply MECE (Mutually Exclusive, Collectively Exhaustive) framework
2. Identify 3-5 main problem components
3. Determine key stakeholders and their perspectives
4. Identify critical constraints and dependencies
5. Suggest mental models that would be most relevant

Respond in JSON format:
{{
    "problem_breakdown": {{
        "main_components": ["component1", "component2", ...],
        "stakeholders": ["stakeholder1", "stakeholder2", ...],
        "constraints": ["constraint1", "constraint2", ...],
        "dependencies": ["dep1", "dep2", ...]
    }},
    "mental_models_recommended": ["model1", "model2", ...],
    "key_insights": ["insight1", "insight2", ...],
    "confidence_score": 0.8,
    "reasoning_summary": "Clear explanation of the structured approach"
}}
"""

        try:
            # OPERATION ILLUMINATE: Make LLM call and capture interaction
            llm_interaction = await self.llm_manager.execute_optimal_llm_call(
                operation="problem_structuring",
                prompt_context={
                    "full_prompt": problem_structuring_prompt,
                    "system_prompt": "You are a strategic problem structuring expert.",
                    "expected_format": "json",
                    "temperature": 0.3,
                    "max_tokens": 2000,
                },
            )

            # OPERATION ILLUMINATE: Store raw interaction in contract
            if not hasattr(contract, "raw_outputs"):
                contract.raw_outputs = []
            contract.raw_outputs.append(asdict(llm_interaction))

            self.logger.info(
                f"ILLUMINATE: Captured interaction {llm_interaction.interaction_id[:8]} - {llm_interaction.tokens_used} tokens"
            )

            # Parse LLM response from interaction
            analysis_content = llm_interaction.raw_response
            analysis = {}
            if isinstance(analysis_content, str):
                try:
                    analysis = json.loads(analysis_content)
                except:
                    analysis = {}

            confidence = analysis.get("confidence_score", 0.7)

            # Create reasoning step
            phase1_step = ReasoningStep(
                step="problem_structuring",
                description=f"Problem structuring analysis: {analysis.get('reasoning_summary', 'Structured problem breakdown')}",
                confidence=confidence,
                timestamp=datetime.now(),
                llm_enhanced=True,
                research_enhanced=False,
                neural_lace_capture=True,
                accumulated_intelligence=True,
                context_phases=0,
                key_insights=analysis.get("key_insights", []),
                step_id="step_1",
                mental_model_applied="mece_structuring",
                reasoning_text=llm_interaction.raw_response,
                confidence_score=confidence,
                evidence_sources=[],
                assumptions_made=[],
            )

            # Update contract state
            updates = {
                "cognitive_state.reasoning_steps": contract.cognitive_state.reasoning_steps
                + [phase1_step],
                "cognitive_state.confidence_scores.problem_structuring": confidence,
                "workflow_state.current_phase": EngagementPhase.HYPOTHESIS_GENERATION,
                "workflow_state.completed_phases": contract.workflow_state.completed_phases
                + [EngagementPhase.PROBLEM_STRUCTURING],
            }

            # Store analysis results
            if not hasattr(contract, "analysis_results"):
                contract.analysis_results = {}
            contract.analysis_results["problem_structuring"] = analysis

            new_state = self.state_manager.create_new_state(
                current_state=contract,
                updates=updates,
                transition_type="problem_structuring_completion",
            )

            self.logger.info(
                f"Problem structuring completed with confidence {confidence}"
            )
            return new_state

        except Exception as e:
            self.logger.error(f"Problem structuring failed: {str(e)}")
            raise

    async def execute_hypothesis_generation(
        self, contract: MetisDataContract
    ) -> MetisDataContract:
        """
        Execute actual hypothesis generation logic.
        Extracted from neural_lace_orchestrator._execute_phase_2
        """
        self.logger.info("Executing hypothesis generation engine")

        # Get insights from Phase 1 for context
        phase1_insights = []
        if contract.cognitive_state.reasoning_steps:
            phase1_step = contract.cognitive_state.reasoning_steps[-1]
            phase1_insights = (
                phase1_step.key_insights if hasattr(phase1_step, "key_insights") else []
            )

        hypothesis_prompt = f"""
You are a strategic hypothesis generation expert. Based on the structured problem analysis, generate testable hypotheses.

ORIGINAL PROBLEM:
{contract.engagement_context.problem_statement}

PHASE 1 INSIGHTS:
{phase1_insights}

Your task:
1. Generate 4-6 strategic hypotheses that could explain the core challenges
2. For each hypothesis, identify what evidence would prove/disprove it
3. Prioritize hypotheses by potential impact and testability
4. Suggest research directions to validate each hypothesis

Respond in JSON format:
{{
    "hypotheses": [
        {{
            "hypothesis": "Clear hypothesis statement",
            "reasoning": "Why this hypothesis makes sense",
            "evidence_needed": ["evidence1", "evidence2"],
            "impact_if_true": "high/medium/low",
            "testability": "high/medium/low"
        }}
    ],
    "priority_ranking": [1, 2, 3, 4],
    "research_directions": ["direction1", "direction2"],
    "confidence_score": 0.8,
    "reasoning_summary": "Overall hypothesis generation approach"
}}
"""

        try:
            # OPERATION ILLUMINATE: Make LLM call and capture interaction
            llm_interaction = await self.llm_manager.execute_optimal_llm_call(
                operation="hypothesis_generation",
                prompt_context={
                    "full_prompt": hypothesis_prompt,
                    "system_prompt": "You are a strategic hypothesis generation expert.",
                    "expected_format": "json",
                    "temperature": 0.5,
                    "max_tokens": 2000,
                },
            )

            # OPERATION ILLUMINATE: Store raw interaction in contract
            if not hasattr(contract, "raw_outputs"):
                contract.raw_outputs = []
            contract.raw_outputs.append(asdict(llm_interaction))

            self.logger.info(
                f"ILLUMINATE: Captured interaction {llm_interaction.interaction_id[:8]} - {llm_interaction.tokens_used} tokens"
            )

            # Parse LLM response from interaction
            analysis_content = llm_interaction.raw_response
            analysis = {}
            if isinstance(analysis_content, str):
                try:
                    analysis = json.loads(analysis_content)
                except:
                    analysis = {}

            confidence = analysis.get("confidence_score", 0.7)

            # Create reasoning step
            phase2_step = ReasoningStep(
                step="hypothesis_generation",
                description=f"Hypothesis generation: {analysis.get('reasoning_summary', 'Generated testable hypotheses')}",
                confidence=confidence,
                timestamp=datetime.now(),
                llm_enhanced=True,
                research_enhanced=False,
                neural_lace_capture=True,
                accumulated_intelligence=True,
                context_phases=1,
                key_insights=[
                    f"Generated {len(analysis.get('hypotheses', []))} testable hypotheses"
                ],
                step_id="step_2",
                mental_model_applied="hypothesis_testing",
                reasoning_text=llm_interaction.raw_response,
                confidence_score=confidence,
                evidence_sources=[],
                assumptions_made=[],
            )

            # Update contract state
            updates = {
                "cognitive_state.reasoning_steps": contract.cognitive_state.reasoning_steps
                + [phase2_step],
                "cognitive_state.confidence_scores.hypothesis_generation": confidence,
                "workflow_state.current_phase": EngagementPhase.ANALYSIS_EXECUTION,
                "workflow_state.completed_phases": contract.workflow_state.completed_phases
                + [EngagementPhase.HYPOTHESIS_GENERATION],
            }

            # Store analysis results
            if not hasattr(contract, "analysis_results"):
                contract.analysis_results = {}
            contract.analysis_results["hypothesis_generation"] = analysis

            new_state = self.state_manager.create_new_state(
                current_state=contract,
                updates=updates,
                transition_type="hypothesis_generation_completion",
            )

            self.logger.info(
                f"Hypothesis generation completed with confidence {confidence}"
            )
            return new_state

        except Exception as e:
            self.logger.error(f"Hypothesis generation failed: {str(e)}")
            raise

    async def execute_analysis(self, contract: MetisDataContract) -> MetisDataContract:
        """
        Execute actual analysis logic.
        Extracted from neural_lace_orchestrator._execute_phase_3
        """
        self.logger.info("Executing analysis engine")

        # Get context from previous phases
        previous_work = ""
        if len(contract.cognitive_state.reasoning_steps) >= 2:
            phase1 = contract.cognitive_state.reasoning_steps[-2]
            phase2 = contract.cognitive_state.reasoning_steps[-1]
            previous_work = f"Phase 1 Insights: {phase1.description[:200]}...\nPhase 2 Insights: {phase2.description[:200]}..."

        analysis_prompt = f"""
You are a strategic analysis expert. Execute deep analysis based on the structured problem and generated hypotheses.

ORIGINAL PROBLEM:
{contract.engagement_context.problem_statement}

PREVIOUS PHASES WORK:
{previous_work}

Your task:
1. Evaluate each hypothesis against available evidence and logic
2. Identify critical success factors and failure points
3. Assess risks and opportunities across different scenarios
4. Apply systems thinking to understand interdependencies
5. Generate actionable insights with clear reasoning

<thinking>
The user is asking me to provide deep strategic analysis. I need to:
1. Evaluate hypotheses systematically
2. Apply frameworks like SWOT, Porter's Five Forces where relevant
3. Consider multiple scenarios and their implications
4. Identify key risks and opportunities
5. Generate clear, actionable insights
</thinking>

Respond in JSON format:
{{
    "hypothesis_evaluations": [
        {{
            "hypothesis_reference": "brief reference",
            "evaluation": "strong/moderate/weak support",
            "supporting_evidence": ["evidence1", "evidence2"],
            "contradicting_evidence": ["counter1", "counter2"],
            "confidence": 0.8
        }}
    ],
    "critical_factors": ["factor1", "factor2"],
    "risk_assessment": {{
        "high_risks": ["risk1", "risk2"],
        "opportunities": ["opp1", "opp2"]
    }},
    "systems_analysis": "How different components interact",
    "actionable_insights": ["insight1", "insight2"],
    "confidence_score": 0.8,
    "reasoning_summary": "Analysis methodology and conclusions"
}}
"""

        try:
            # Check if we should use tree search enhancement (optional)
            tree_search_results = None
            if hasattr(self, "_should_use_tree_search"):
                use_tree_search = self._should_use_tree_search(contract)
                if use_tree_search:
                    self.logger.info("Enhancing with tree search exploration")
                    tree_search_results = await self._execute_tree_search_analysis(
                        contract
                    )

            # OPERATION ILLUMINATE: Make LLM call and capture interaction
            llm_interaction = await self.llm_manager.execute_optimal_llm_call(
                operation="analysis_execution",
                prompt_context={
                    "full_prompt": analysis_prompt,
                    "system_prompt": "You are a strategic analysis expert.",
                    "expected_format": "json",
                    "temperature": 0.2,
                    "max_tokens": 2500,
                },
            )

            # OPERATION ILLUMINATE: Store raw interaction in contract
            if not hasattr(contract, "raw_outputs"):
                contract.raw_outputs = []
            contract.raw_outputs.append(asdict(llm_interaction))

            self.logger.info(
                f"ILLUMINATE: Captured interaction {llm_interaction.interaction_id[:8]} - {llm_interaction.tokens_used} tokens"
            )

            # Parse LLM response from interaction
            analysis_content = llm_interaction.raw_response
            analysis = {}
            if isinstance(analysis_content, str):
                try:
                    analysis = json.loads(analysis_content)
                except:
                    analysis = {}

            confidence = analysis.get("confidence_score", 0.7)

            # Create reasoning step
            phase3_step = ReasoningStep(
                step="analysis_execution",
                description=f"Deep analysis execution: {analysis.get('reasoning_summary', 'Comprehensive analysis completed')}",
                confidence=confidence,
                timestamp=datetime.now(),
                llm_enhanced=True,
                research_enhanced=False,
                neural_lace_capture=True,
                accumulated_intelligence=True,
                context_phases=2,
                key_insights=analysis.get("actionable_insights", []),
                step_id="step_3",
                mental_model_applied="systems_analysis",
                reasoning_text=llm_interaction.raw_response,
                confidence_score=confidence,
                evidence_sources=[],
                assumptions_made=[],
            )

            # Update contract state
            updates = {
                "cognitive_state.reasoning_steps": contract.cognitive_state.reasoning_steps
                + [phase3_step],
                "cognitive_state.confidence_scores.analysis_execution": confidence,
                "workflow_state.current_phase": EngagementPhase.RESEARCH_GROUNDING,
                "workflow_state.completed_phases": contract.workflow_state.completed_phases
                + [EngagementPhase.ANALYSIS_EXECUTION],
            }

            # Store analysis results - this is critical for downstream phases
            if not hasattr(contract, "analysis_results"):
                contract.analysis_results = {}
            contract.analysis_results["initial_analysis"] = analysis

            new_state = self.state_manager.create_new_state(
                current_state=contract,
                updates=updates,
                transition_type="analysis_execution_completion",
            )

            self.logger.info(
                f"Analysis execution completed with confidence {confidence}"
            )
            return new_state

        except Exception as e:
            self.logger.error(f"Analysis execution failed: {str(e)}")
            raise


# Factory function
def get_orchestrator_engine_adapter() -> OrchestratorEngineAdapter:
    """Get or create orchestrator engine adapter instance"""
    return OrchestratorEngineAdapter()
