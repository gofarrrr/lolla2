#!/usr/bin/env python3
"""
Three-Consultant Multi-Single-Agent API
Implements the Multi-Single-Agent paradigm with three independent consultants.

Following principles from:
1. Cognition.ai "Don't Build Multi-Agents" - Context preservation
2. Anthropic "Multi-Agent Research System" - Parallel processing
3. Manus "Context Engineering for AI Agents" - Dynamic context management
4. 12-Factor Agents - Production-ready agent principles

One query → 3 separate context-preserving agents with NO coordination
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from uuid import uuid4

try:
    from fastapi import FastAPI, HTTPException, Depends
    from pydantic import BaseModel, Field

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    BaseModel = object

from src.cognitive_architecture import (
    initialize_three_consultant_system,
    ConsultantRole,
    IssueTreeType,
    DeepSeekMode,
    PromptComplexity,
)
from src.integrations.llm_provider import get_llm_provider
from src.core.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from src.core.audit_trail import AuditEventType
from src.core.context_engineering_optimizer import ContextEngineeringOptimizer
from src.engine.monitoring.coordination_detector import CoordinationDetector

logger = logging.getLogger(__name__)


class ThreeConsultantRequest(BaseModel):
    """Request model for three-consultant analysis"""

    query: str = Field(..., description="The business question to analyze")
    context: Optional[str] = Field(None, description="Additional context or background")
    complexity: Optional[str] = Field(
        "moderate",
        description="Query complexity: simple, moderate, complex, ultra_complex",
    )
    include_research: Optional[bool] = Field(
        True, description="Include external research"
    )
    max_tokens: Optional[int] = Field(
        4000, description="Maximum tokens per consultant response"
    )


class ConsultantResponse(BaseModel):
    """Individual consultant response"""

    role: str
    analysis: str
    mental_models_used: List[str]
    nway_interactions: List[str]
    issue_tree: Optional[Dict[str, Any]]
    confidence_score: float
    processing_time_seconds: float
    reasoning_trace: List[str]


class ThreeConsultantResponse(BaseModel):
    """Complete three-consultant response following Multi-Single-Agent paradigm"""

    engagement_id: str
    timestamp: datetime
    query: str
    consultants: Dict[str, ConsultantResponse]
    total_processing_time_seconds: float
    query_enhancement_applied: str
    metadata: Dict[str, Any]


class ThreeConsultantOrchestrator:
    """
    Orchestrates three independent consultants following Multi-Single-Agent paradigm.
    Each consultant operates independently with no coordination or synthesis.
    """

    def __init__(self):
        self.cognitive_system = initialize_three_consultant_system()
        self.llm_provider = get_llm_provider()
        # Note: get_audit_manager is async, set to None for synchronous init
        # Will be initialized properly in async context when needed
        self.audit_manager = None
        # Create circuit breaker with proper configuration object
        circuit_config = CircuitBreakerConfig(
            name="three_consultant_api", failure_threshold=3, reset_timeout_seconds=60.0
        )
        self.circuit_breaker = CircuitBreaker(circuit_config)

        # PHASE 1 ENHANCEMENT: Context Engineering Integration
        self.context_optimizer = ContextEngineeringOptimizer(
            max_context_tokens=8000, compression_threshold=0.8, enable_file_context=True
        )
        logger.info(
            "Context Engineering Optimizer integrated with Three-Consultant system"
        )

        # PHASE 2 ENHANCEMENT: Coordination Detection & Monitoring
        self.coordination_detector = CoordinationDetector(enable_monitoring=True)
        logger.info("Coordination Detector integrated for independence validation")

        # OPTIMAL CONSULTANT ENGINE: Intelligent consultant selection
        # Note: OptimalConsultantEngine requires async initialization
        # Disabled for tracer bullet - will initialize in async context when needed
        self.optimal_engine = None
        logger.info("OptimalConsultantEngine disabled for tracer bullet mode")

    def _calculate_context_window(
        self, role: ConsultantRole, complexity: PromptComplexity
    ) -> int:
        """Calculate optimal context window per consultant based on role and complexity"""
        base_window = 4000

        # Role-specific multipliers based on cognitive requirements
        role_multipliers = {
            ConsultantRole.STRATEGIC_ANALYST: 1.2,  # Needs broader context
            ConsultantRole.SYNTHESIS_ARCHITECT: 1.1,  # Moderate context needs
            ConsultantRole.IMPLEMENTATION_DRIVER: 0.9,  # More focused, less context
        }

        # Complexity multipliers
        complexity_multipliers = {
            PromptComplexity.SIMPLE: 0.8,
            PromptComplexity.MODERATE: 1.0,
            PromptComplexity.COMPLEX: 1.3,
            PromptComplexity.ULTRA_COMPLEX: 1.6,
        }

        optimized_window = int(
            base_window * role_multipliers[role] * complexity_multipliers[complexity]
        )

        return min(optimized_window, 12000)  # Cap at 12k tokens

    async def _get_optimal_consultant_roles(
        self, query: str, context: Optional[str] = None
    ) -> Tuple[List[str], Dict[str, Any]]:
        """Get optimal consultant roles using OptimalConsultantEngine"""
        try:
            # Use optimal engine to get consultant recommendations
            optimal_result = await self.optimal_engine.process_query(
                query=query,
                context={"additional_context": context} if context else None,
            )

            # Extract consultant IDs and metadata
            consultant_ids = [
                selection.consultant_id
                for selection in optimal_result.selected_consultants
            ]

            # Map consultant IDs to ConsultantRole enum (with fallback mapping)
            role_mapping = {
                "market_analyst": ConsultantRole.STRATEGIC_ANALYST,
                "strategic_synthesizer": ConsultantRole.SYNTHESIS_ARCHITECT,
                "problem_solver": ConsultantRole.STRATEGIC_ANALYST,  # Fallback to strategic
                "solution_architect": ConsultantRole.SYNTHESIS_ARCHITECT,  # Fallback to synthesis
                "execution_specialist": ConsultantRole.IMPLEMENTATION_DRIVER,
                "strategic_implementer": ConsultantRole.IMPLEMENTATION_DRIVER,
                "tactical_builder": ConsultantRole.IMPLEMENTATION_DRIVER,
                "process_expert": ConsultantRole.IMPLEMENTATION_DRIVER,
                "operational_integrator": ConsultantRole.IMPLEMENTATION_DRIVER,
            }

            consultant_roles = []
            for consultant_id in consultant_ids:
                role = role_mapping.get(
                    consultant_id, ConsultantRole.STRATEGIC_ANALYST
                )  # Default fallback
                if role not in consultant_roles:  # Avoid duplicates
                    consultant_roles.append(role)

            # Ensure we have exactly 3 roles
            all_roles = list(ConsultantRole)
            while len(consultant_roles) < 3:
                for role in all_roles:
                    if role not in consultant_roles:
                        consultant_roles.append(role)
                        break
                if len(consultant_roles) >= 3:
                    break

            consultant_roles = consultant_roles[:3]  # Ensure max 3

            # Return roles and metadata for logging
            selection_metadata = {
                "optimal_selections": [
                    {
                        "consultant_id": selection.consultant_id,
                        "name": selection.blueprint.name,
                        "confidence": selection.confidence_score,
                        "frameworks": selection.frameworks_used,
                    }
                    for selection in optimal_result.selected_consultants
                ],
                "classification": optimal_result.processing_metadata.get(
                    "classification", {}
                ),
                "processing_time": optimal_result.processing_metadata.get(
                    "processing_time_seconds", 0
                ),
            }

            logger.info(
                f"Optimal consultant selection: {consultant_ids} -> {[r.value for r in consultant_roles]}"
            )
            return consultant_roles, selection_metadata

        except Exception as e:
            logger.warning(
                f"Optimal consultant selection failed, using default roles: {e}"
            )
            # Fallback to default roles
            return list(ConsultantRole), {}

    async def enhance_query(self, query: str, context: Optional[str] = None) -> str:
        """Enhance the query with additional context for better consultant processing"""
        enhancement_prompt = f"""
        <query_enhancement>
        Original Query: {query}
        {f"Context: {context}" if context else ""}
        
        Enhance this query to make it more specific and actionable for business consultants.
        Add relevant business context, clarify ambiguous terms, and suggest specific angles of analysis.
        Keep the enhanced query concise but comprehensive.
        </query_enhancement>
        """

        try:
            enhanced = await self.llm_provider.generate_response(
                prompt=enhancement_prompt, max_tokens=500, temperature=0.3
            )
            return enhanced.strip()
        except Exception as e:
            logger.warning(f"Query enhancement failed: {e}")
            return query  # Fall back to original query

    async def process_consultant(
        self,
        role: ConsultantRole,
        enhanced_query: str,
        complexity: PromptComplexity,
        max_tokens: int,
    ) -> ConsultantResponse:
        """Process query through single consultant - independent operation with context optimization"""
        start_time = time.time()
        reasoning_trace = []

        # PHASE 1 ENHANCEMENT: Create optimized context session per consultant
        session_id = await self.context_optimizer.create_session(
            initial_context=f"CONSULTANT_ROLE: {role.value}\nQUERY: {enhanced_query}",
            session_config={
                "max_tokens": self._calculate_context_window(role, complexity),
                "prefix_locked": complexity == PromptComplexity.ULTRA_COMPLEX,
            },
        )
        reasoning_trace.append(f"[{role.value}] Context session created: {session_id}")

        try:
            consultant_components = self.cognitive_system["consultants"][role]

            # Step 1: Mental Model Selection (Independent)
            reasoning_trace.append(f"[{role.value}] Starting mental model selection")
            selected_models = consultant_components[
                "mental_model_selector"
            ].select_models(query=enhanced_query, top_k=5)
            model_names = [model.name for model in selected_models]
            reasoning_trace.append(f"[{role.value}] Selected models: {model_names}")

            # PHASE 1 ENHANCEMENT: Enhanced context tracking
            await self.context_optimizer.append_context(
                session_id,
                f"MENTAL_MODELS_SELECTED: {model_names}",
                content_type="cognitive_state",
            )

            # Step 2: N-Way Interaction Activation (Independent)
            reasoning_trace.append(f"[{role.value}] Activating N-way interactions")
            activated_interactions = consultant_components[
                "nway_activation_engine"
            ].activate_synergies(query=enhanced_query, selected_models=selected_models)
            interaction_names = [
                interaction.name for interaction in activated_interactions
            ]
            reasoning_trace.append(
                f"[{role.value}] Activated interactions: {interaction_names}"
            )

            # Step 3: MECE Issue Tree Construction (Independent)
            reasoning_trace.append(f"[{role.value}] Building MECE issue tree")
            issue_tree = consultant_components["mece_application"].build_issue_tree(
                query=enhanced_query,
                tree_type=(
                    IssueTreeType.STRATEGIC
                    if role == ConsultantRole.STRATEGIC_ANALYST
                    else (
                        IssueTreeType.SYNTHESIS
                        if role == ConsultantRole.SYNTHESIS_ARCHITECT
                        else IssueTreeType.OPERATIONAL
                    )
                ),
            )
            reasoning_trace.append(
                f"[{role.value}] Issue tree built with {len(issue_tree.nodes)} nodes"
            )

            # Step 4: DeepSeek Prompt Generation (Independent)
            reasoning_trace.append(
                f"[{role.value}] Generating optimized DeepSeek prompt"
            )
            consultant_prompt = consultant_components[
                "prompt_optimizer"
            ].generate_prompt(
                query=enhanced_query,
                mental_models=selected_models,
                nway_interactions=activated_interactions,
                issue_tree=issue_tree,
                mode=(
                    DeepSeekMode.THINKING
                    if complexity
                    in [PromptComplexity.COMPLEX, PromptComplexity.ULTRA_COMPLEX]
                    else DeepSeekMode.NON_THINKING
                ),
                complexity=complexity,
            )
            reasoning_trace.append(
                f"[{role.value}] Prompt generated, length: {len(consultant_prompt.content)} chars"
            )

            # Step 5: LLM Processing (Independent)
            reasoning_trace.append(f"[{role.value}] Processing with LLM")
            response = await self.llm_provider.generate_response(
                prompt=consultant_prompt.content,
                max_tokens=max_tokens,
                temperature=consultant_prompt.temperature,
            )
            reasoning_trace.append(f"[{role.value}] LLM processing completed")

            # Step 6: Confidence Scoring (Independent)
            confidence_score = self._calculate_confidence(
                response=response,
                models_used=len(selected_models),
                interactions_used=len(activated_interactions),
                issue_tree_depth=len(issue_tree.nodes),
            )
            reasoning_trace.append(
                f"[{role.value}] Confidence score: {confidence_score}"
            )

            processing_time = time.time() - start_time

            return ConsultantResponse(
                role=role.value,
                analysis=response,
                mental_models_used=model_names,
                nway_interactions=interaction_names,
                issue_tree={
                    "root_question": issue_tree.root_question,
                    "nodes": [
                        {"question": node.question, "type": node.node_type.value}
                        for node in issue_tree.nodes
                    ],
                    "is_mece_valid": issue_tree.is_mece_valid,
                },
                confidence_score=confidence_score,
                processing_time_seconds=processing_time,
                reasoning_trace=reasoning_trace,
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Consultant {role.value} processing failed: {e}")
            reasoning_trace.append(f"[{role.value}] ERROR: {str(e)}")

            # Return error response but don't fail entire request
            return ConsultantResponse(
                role=role.value,
                analysis=f"ERROR: Processing failed for {role.value} - {str(e)}",
                mental_models_used=[],
                nway_interactions=[],
                issue_tree=None,
                confidence_score=0.0,
                processing_time_seconds=processing_time,
                reasoning_trace=reasoning_trace,
            )

    def _calculate_confidence(
        self,
        response: str,
        models_used: int,
        interactions_used: int,
        issue_tree_depth: int,
    ) -> float:
        """Calculate confidence score based on cognitive processing depth"""
        base_confidence = 0.6  # Base confidence

        # Adjust based on mental models used
        model_boost = min(0.2, models_used * 0.04)

        # Adjust based on N-way interactions
        interaction_boost = min(0.15, interactions_used * 0.05)

        # Adjust based on MECE structure depth
        structure_boost = min(0.05, issue_tree_depth * 0.01)

        confidence = base_confidence + model_boost + interaction_boost + structure_boost
        return min(1.0, confidence)

    async def process_three_consultants(
        self, request: ThreeConsultantRequest
    ) -> ThreeConsultantResponse:
        """
        Process query through three independent consultants.
        Multi-Single-Agent paradigm: NO coordination, NO synthesis.
        """
        start_time = time.time()
        engagement_id = str(uuid4())

        # Log engagement start
        await self.audit_manager.log_event(
            event_type=AuditEventType.ENGAGEMENT_STARTED,
            user_id="system",
            details={
                "engagement_id": engagement_id,
                "query": request.query,
                "consultant_approach": "multi-single-agent",
            },
        )

        try:
            # Step 1: Query Enhancement (shared input preparation)
            enhanced_query = await self.enhance_query(request.query, request.context)

            # Step 2: Get optimal consultant roles using OptimalConsultantEngine
            optimal_roles, selection_metadata = (
                await self._get_optimal_consultant_roles(request.query, request.context)
            )

            # Step 3: Map complexity
            complexity_map = {
                "simple": PromptComplexity.SIMPLE,
                "moderate": PromptComplexity.MODERATE,
                "complex": PromptComplexity.COMPLEX,
                "ultra_complex": PromptComplexity.ULTRA_COMPLEX,
            }
            complexity = complexity_map.get(
                request.complexity, PromptComplexity.MODERATE
            )

            # Step 4: Process optimal consultants in PARALLEL (Independent)
            # This is the key Multi-Single-Agent implementation with optimal selection
            consultant_tasks = []
            for role in optimal_roles:
                task = self.process_consultant(
                    role=role,
                    enhanced_query=enhanced_query,
                    complexity=complexity,
                    max_tokens=request.max_tokens,
                )
                consultant_tasks.append((role, task))

            # Wait for all consultants to complete independently
            consultant_responses = {}
            completed_tasks = await asyncio.gather(
                *[task for _, task in consultant_tasks]
            )

            for (role, _), response in zip(consultant_tasks, completed_tasks):
                consultant_responses[role.value] = response

            total_processing_time = time.time() - start_time

            # Log engagement completion
            await self.audit_manager.log_event(
                event_type=AuditEventType.ENGAGEMENT_COMPLETED,
                user_id="system",
                details={
                    "engagement_id": engagement_id,
                    "processing_time": total_processing_time,
                    "consultants_processed": len(consultant_responses),
                },
            )

            return ThreeConsultantResponse(
                engagement_id=engagement_id,
                timestamp=datetime.now(),
                query=request.query,
                consultants=consultant_responses,
                total_processing_time_seconds=total_processing_time,
                query_enhancement_applied=enhanced_query,
                metadata={
                    "complexity_level": request.complexity,
                    "include_research": request.include_research,
                    "max_tokens_per_consultant": request.max_tokens,
                    "paradigm": "multi-single-agent",
                    "coordination": "none",
                    "synthesis": "none",
                    "optimal_selection": selection_metadata,
                    "consultant_selection_engine": "OptimalConsultantEngine",
                },
            )

        except Exception as e:
            logger.error(f"Three-consultant processing failed: {e}")
            await self.audit_manager.log_event(
                event_type=AuditEventType.ENGAGEMENT_ERROR,
                user_id="system",
                details={"engagement_id": engagement_id, "error": str(e)},
            )
            raise HTTPException(
                status_code=500, detail=f"Three-consultant processing failed: {str(e)}"
            )


# Global orchestrator instance - disabled for tracer bullet mode
# orchestrator = ThreeConsultantOrchestrator()
orchestrator = None  # Will be initialized when needed

# FastAPI routes (only if FastAPI is available)
if FASTAPI_AVAILABLE:
    app = FastAPI(title="Three-Consultant Multi-Single-Agent API")

    @app.post("/api/three-consultants/analyze", response_model=ThreeConsultantResponse)
    async def analyze_with_three_consultants(request: ThreeConsultantRequest):
        """
        Process business query through three independent consultants.

        Implements Multi-Single-Agent paradigm:
        - One query input
        - Three separate context-preserving agents
        - NO coordination between consultants
        - NO synthesis of responses
        - Parallel processing for efficiency
        """
        return await orchestrator.process_three_consultants(request)

    @app.get("/api/three-consultants/health")
    async def health_check():
        """Health check for three-consultant system"""
        return {
            "status": "healthy",
            "system": "three-consultant-multi-single-agent",
            "cognitive_components": len(orchestrator.cognitive_system["consultants"]),
            "paradigm": "multi-single-agent",
            "timestamp": datetime.now(),
        }


async def analyze_query(
    query: str, context: str = None, complexity: str = "moderate"
) -> ThreeConsultantResponse:
    """Direct function interface for three-consultant analysis"""
    request = ThreeConsultantRequest(
        query=query, context=context, complexity=complexity
    )
    return await orchestrator.process_three_consultants(request)


if __name__ == "__main__":
    # Test the system
    import uvicorn

    logger.info("Starting Three-Consultant Multi-Single-Agent API server")
    uvicorn.run(app, host="0.0.0.0", port=8002)
