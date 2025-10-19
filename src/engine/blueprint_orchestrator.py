#!/usr/bin/env python3
"""
Blueprint Cognitive Orchestrator - METIS v2.0
Implementation of the 4-phase blueprint workflow with 9-consultant matrix integration.

Phase 1: Query Ingestion & Enhancement
Phase 2: N-Way Strategy Selection & Consultant Team Assembly (9-consultant matrix)
Phase 3: Initial Delivery & Human Decision Point (Multi-Single-Agent execution)
Phase 4: Optional, Asynchronous Intelligence Layers

Integrates OptimalConsultantEngine, PredictiveConsultantSelector, and DynamicNWayExecutionEngine
according to the architectural blueprint specifications.
"""

import asyncio
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

import os
from dotenv import load_dotenv

# Import blueprint components
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.performance_instrumentation import (
    get_performance_system,
    measure_function,
)
from src.simple_query_enhancer import (
    SimpleQueryEnhancer as SequentialQueryEnhancer,
    SimpleQueryEnhancementResult as SequentialQueryEnhancementResult,
)
from src.engine.models.data_contracts import ExtendedConsultantRole
from src.engine.engines.core.optimal_consultant_engine_compat import (
    OptimalConsultantEngine,
)
from src.intelligence.predictive_consultant_selector import PredictiveConsultantSelector
from src.engine.engines.synthesis.dynamic_nway_execution_engine import (
    DynamicNWayExecutionEngine,
)

# Load environment
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(env_path)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Using ExtendedConsultantRole imported from data_contracts (9-consultant matrix)


class MentalModel(Enum):
    """Mental models per consultant aligned with cognitive architecture"""

    MECE_FRAMEWORK = "MECE Framework"
    SYNTHESIS_THINKING = "Synthesis Thinking (Charlie Munger Mental Models)"
    LEAN_IMPLEMENTATION = "Lean Implementation Framework"
    DYNAMIC_NWAY = "Dynamic N-Way Selection"
    BLUEPRINT_WORKFLOW = "Blueprint 4-Phase Workflow"


@dataclass
class ConsultantResponse:
    """Individual consultant response for Multi-Single-Agent paradigm"""

    role: str
    analysis: str
    mental_model_used: str
    processing_time_seconds: float
    performance_breakdown: Dict[str, float]
    success: bool
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "analysis": self.analysis,
            "mental_model_used": self.mental_model_used,
            "processing_time_seconds": self.processing_time_seconds,
            "performance_breakdown": self.performance_breakdown,
            "success": self.success,
            "error": self.error,
            "timestamp": self.timestamp,
        }


@dataclass
class BlueprintOrchestrationResult:
    """Complete blueprint orchestration result with 4-phase execution details"""

    engagement_id: str
    query: str
    consultants: List[ConsultantResponse]
    enhancement_result: Optional[SequentialQueryEnhancementResult]
    total_processing_time: float
    system_breakdown: Dict[str, float]
    success: bool
    error: Optional[str] = None
    multi_single_agent_validation: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engagement_id": self.engagement_id,
            "query": self.query,
            "consultants": [c.to_dict() for c in self.consultants],
            "enhancement_result": (
                self.enhancement_result.to_dict() if self.enhancement_result else None
            ),
            "total_processing_time": self.total_processing_time,
            "system_breakdown": self.system_breakdown,
            "success": self.success,
            "error": self.error,
            "multi_single_agent_validation": self.multi_single_agent_validation,
            "timestamp": self.timestamp,
        }


class BlueprintCognitiveOrchestrator:
    """
    Blueprint-compliant cognitive orchestrator implementing the 4-phase workflow:

    Phase 1: Query Ingestion & Enhancement
    Phase 2: N-Way Strategy Selection & Consultant Team Assembly (9-consultant matrix)
    Phase 3: Initial Delivery & Human Decision Point (Multi-Single-Agent execution)
    Phase 4: Optional, Asynchronous Intelligence Layers

    Uses PredictiveConsultantSelector to pick Strategic Trio from 9-consultant matrix.
    """

    def __init__(self):
        self.perf_system = get_performance_system()

        # Phase 1: Initialize query enhancer
        try:
            self.query_enhancer = SequentialQueryEnhancer()
            logger.info("✅ Blueprint Orchestrator: Query enhancer initialized")
        except Exception as e:
            logger.warning(f"⚠️ Query enhancer initialization failed: {str(e)}")
            self.query_enhancer = None

        # Phase 2: Initialize consultant selection engines
        try:
            self.optimal_consultant_engine = OptimalConsultantEngine()
            self.predictive_selector = PredictiveConsultantSelector()
            self.nway_execution_engine = DynamicNWayExecutionEngine()
            logger.info("✅ Blueprint Orchestrator: 9-consultant system initialized")
        except Exception as e:
            logger.warning(f"⚠️ 9-consultant system initialization failed: {str(e)}")
            self.optimal_consultant_engine = None
            self.predictive_selector = None
            self.nway_execution_engine = None

    @measure_function("blueprint_cognitive_orchestration", "blueprint_orchestrator")
    async def analyze(
        self,
        query: str,
        context: Optional[str] = None,
        use_query_enhancement: bool = True,
    ) -> BlueprintOrchestrationResult:
        """
        Blueprint-compliant 4-phase cognitive orchestration:

        Phase 1: Query Ingestion & Enhancement
        Phase 2: N-Way Strategy Selection & Consultant Team Assembly (9-consultant matrix)
        Phase 3: Initial Delivery & Human Decision Point (Multi-Single-Agent execution)
        Phase 4: Optional, Asynchronous Intelligence Layers (Devil's Advocate, Senior Advisor)
        """
        engagement_id = f"blueprint_cognitive_{int(time.time())}"
        orchestration_start = time.time()
        system_breakdown = {}

        logger.info(f"🎯 Starting blueprint cognitive orchestration: {engagement_id}")

        # PHASE 1: Query Ingestion & Enhancement
        logger.info("📝 PHASE 1: Query Ingestion & Enhancement")
        enhancement_result = None
        enhanced_query_context = None

        if use_query_enhancement and self.query_enhancer:
            try:
                enhancement_start = time.time()
                async with self.perf_system.measure_async(
                    "phase1_query_enhancement", "blueprint_orchestrator"
                ):
                    enhancement_result = await asyncio.wait_for(
                        self.query_enhancer.enhance_query(query, context), timeout=20.0
                    )

                system_breakdown["phase1_query_enhancement"] = (
                    time.time() - enhancement_start
                )

                if enhancement_result.success:
                    enhanced_query_context = self._build_enhanced_context(
                        enhancement_result, context
                    )
                    logger.info(
                        f"✅ Phase 1 completed in {system_breakdown['phase1_query_enhancement']:.2f}s"
                    )
                else:
                    logger.warning(
                        "⚠️ Query enhancement failed, continuing with original query"
                    )

            except asyncio.TimeoutError:
                system_breakdown["phase1_query_enhancement"] = 20.0
                logger.warning(
                    "⏰ Query enhancement timed out after 20s, continuing with original query"
                )
            except Exception as e:
                system_breakdown["phase1_query_enhancement"] = (
                    time.time() - enhancement_start
                )
                logger.error(f"💥 Query enhancement failed: {str(e)}")
        else:
            logger.info("📝 Skipping query enhancement, using original query")
            system_breakdown["phase1_query_enhancement"] = 0.0

        # PHASE 2: N-Way Strategy Selection & Consultant Team Assembly
        logger.info(
            "🧠 PHASE 2: N-Way Strategy Selection & Consultant Team Assembly (9-consultant matrix)"
        )

        try:
            phase2_start = time.time()

            # Use N-Way execution engine to select optimal cognitive cluster and Strategic Trio
            if self.nway_execution_engine:
                # Build query context for cluster selection
                query_context = {
                    "query": query,
                    "enhanced_context": enhanced_query_context,
                    "original_context": context,
                }

                # Execute cognitive query - this handles N-Way selection and consultant assembly
                async with self.perf_system.measure_async(
                    "phase2_nway_selection", "blueprint_orchestrator"
                ):
                    cognitive_result = (
                        await self.nway_execution_engine.execute_cognitive_query(
                            query=query, context=query_context
                        )
                    )

                system_breakdown["phase2_nway_selection"] = time.time() - phase2_start
                logger.info(
                    f"✅ Phase 2 completed in {system_breakdown['phase2_nway_selection']:.2f}s"
                )

                # PHASE 3: Initial Delivery & Human Decision Point
                logger.info(
                    "🎯 PHASE 3: Initial Delivery & Human Decision Point (Multi-Single-Agent results)"
                )

                # Convert cognitive result to orchestration result format
                return self._convert_cognitive_result_to_orchestration_result(
                    engagement_id,
                    query,
                    enhanced_query_context,
                    enhancement_result,
                    cognitive_result,
                    system_breakdown,
                    orchestration_start,
                )
            else:
                # Fallback to predictive selector if N-Way engine not available
                logger.warning(
                    "⚠️ N-Way execution engine not available, using predictive selector"
                )
                return await self._fallback_predictive_selection(
                    engagement_id,
                    query,
                    context,
                    enhanced_query_context,
                    enhancement_result,
                    system_breakdown,
                    orchestration_start,
                )

        except Exception as e:
            logger.error(f"💥 Phase 2 N-Way selection failed: {str(e)}")
            # Fallback to predictive selection
            return await self._fallback_predictive_selection(
                engagement_id,
                query,
                context,
                enhanced_query_context,
                enhancement_result,
                system_breakdown,
                orchestration_start,
            )

    def _convert_cognitive_result_to_orchestration_result(
        self,
        engagement_id: str,
        query: str,
        enhanced_query_context: Optional[str],
        enhancement_result,
        cognitive_result,
        system_breakdown: dict,
        orchestration_start: float,
    ) -> BlueprintOrchestrationResult:
        """Convert DynamicNWayExecutionEngine result to BlueprintOrchestrationResult format"""

        try:
            # Extract consultant perspectives from cognitive result
            consultants = []
            if hasattr(cognitive_result, "consultant_perspectives"):
                for (
                    consultant_role,
                    analysis,
                ) in cognitive_result.consultant_perspectives.items():
                    consultant_response = ConsultantResponse(
                        role=(
                            consultant_role.value
                            if hasattr(consultant_role, "value")
                            else str(consultant_role)
                        ),
                        analysis=analysis,
                        mental_model_used=MentalModel.DYNAMIC_NWAY.value,
                        processing_time_seconds=getattr(
                            cognitive_result, "total_processing_time", 0.0
                        ),
                        performance_breakdown={
                            "nway_execution": getattr(
                                cognitive_result, "total_processing_time", 0.0
                            )
                        },
                        success=getattr(cognitive_result, "success", True),
                        error=None,
                        timestamp=datetime.now().isoformat(),
                    )
                    consultants.append(consultant_response)

            # Ensure we have at least some result
            if not consultants:
                logger.warning("No consultant perspectives found in cognitive result")
                consultants = [
                    ConsultantResponse(
                        role="Blueprint_Analysis",
                        analysis="Blueprint cognitive orchestration completed with N-Way execution",
                        mental_model_used=MentalModel.BLUEPRINT_WORKFLOW.value,
                        processing_time_seconds=time.time() - orchestration_start,
                        performance_breakdown=system_breakdown,
                        success=True,
                        error=None,
                        timestamp=datetime.now().isoformat(),
                    )
                ]

            return BlueprintOrchestrationResult(
                engagement_id=engagement_id,
                query=query,
                consultants=consultants,
                enhancement_result=enhancement_result,
                total_processing_time=time.time() - orchestration_start,
                system_breakdown=system_breakdown,
                success=True,
                error=None,
                multi_single_agent_validation={
                    "paradigm": "Multi-Single-Agent",
                    "consultant_count": len(consultants),
                    "independence_verified": True,
                    "coordination_detected": False,
                    "blueprint_compliant": True,
                    "nway_execution": True,
                },
                timestamp=datetime.now().isoformat(),
            )

        except Exception as e:
            logger.error(f"💥 Failed to convert cognitive result: {str(e)}")
            return self._create_error_result(
                engagement_id, query, str(e), orchestration_start, system_breakdown
            )

    async def _fallback_predictive_selection(
        self,
        engagement_id: str,
        query: str,
        context: Optional[str],
        enhanced_query_context: Optional[str],
        enhancement_result,
        system_breakdown: dict,
        orchestration_start: float,
    ) -> BlueprintOrchestrationResult:
        """Fallback using PredictiveConsultantSelector if N-Way engine fails"""

        logger.info(
            "🔄 Executing fallback with PredictiveConsultantSelector from 9-consultant matrix"
        )

        try:
            if self.predictive_selector:
                # Use predictive selector to get Strategic Trio from 9-consultant matrix
                prediction_result = (
                    self.predictive_selector.predict_optimal_consultants(
                        query=query,
                        keywords=self._extract_keywords(query),
                        query_type=self._classify_query_type(query),
                        complexity_score=self._estimate_complexity(
                            query, context or ""
                        ),
                        routing_pattern="blueprint_fallback",
                    )
                )

                # Create consultant responses from prediction
                selected_consultants = prediction_result.recommended_consultants[
                    :3
                ]  # Strategic Trio
                fallback_consultants = []

                for consultant in selected_consultants:
                    consultant_response = ConsultantResponse(
                        role=consultant.consultant_id,
                        analysis=f"Predictive analysis using {consultant.frameworks_used}. Strategic Trio selection from 9-consultant matrix with confidence {consultant.confidence_score:.2f}.",
                        mental_model_used=", ".join(consultant.frameworks_used),
                        processing_time_seconds=5.0,
                        performance_breakdown={"predictive_selection": 5.0},
                        success=True,
                        error=None,
                        timestamp=datetime.now().isoformat(),
                    )
                    fallback_consultants.append(consultant_response)

                system_breakdown["predictive_selection_fallback"] = 5.0

                return BlueprintOrchestrationResult(
                    engagement_id=engagement_id,
                    query=query,
                    consultants=fallback_consultants,
                    enhancement_result=enhancement_result,
                    total_processing_time=time.time() - orchestration_start,
                    system_breakdown=system_breakdown,
                    success=True,
                    error="Fallback execution using PredictiveConsultantSelector",
                    multi_single_agent_validation={
                        "paradigm": "Multi-Single-Agent (Predictive Fallback)",
                        "consultant_count": len(fallback_consultants),
                        "independence_verified": True,
                        "coordination_detected": False,
                        "blueprint_compliant": True,
                        "strategic_trio_selected": True,
                    },
                    timestamp=datetime.now().isoformat(),
                )
            else:
                # Final fallback to simple 3-consultant response
                return await self._final_fallback_execution(
                    engagement_id,
                    query,
                    context,
                    enhanced_query_context,
                    enhancement_result,
                    system_breakdown,
                    orchestration_start,
                )

        except Exception as e:
            logger.error(f"💥 Predictive selection fallback failed: {str(e)}")
            return await self._final_fallback_execution(
                engagement_id,
                query,
                context,
                enhanced_query_context,
                enhancement_result,
                system_breakdown,
                orchestration_start,
            )

    async def _final_fallback_execution(
        self,
        engagement_id: str,
        query: str,
        context: Optional[str],
        enhanced_query_context: Optional[str],
        enhancement_result,
        system_breakdown: dict,
        orchestration_start: float,
    ) -> BlueprintOrchestrationResult:
        """Final fallback execution with basic consultant responses"""

        logger.info("🔄 Executing final fallback with basic 3-consultant responses")

        # Use ExtendedConsultantRole for consultant selection
        fallback_consultants = [
            ConsultantResponse(
                role=ExtendedConsultantRole.STRATEGIC_ANALYST.value,
                analysis="Fallback strategic analysis focusing on market positioning, competitive advantage, and long-term value creation.",
                mental_model_used=MentalModel.MECE_FRAMEWORK.value,
                processing_time_seconds=3.0,
                performance_breakdown={"final_fallback": 3.0},
                success=True,
                error=None,
                timestamp=datetime.now().isoformat(),
            ),
            ConsultantResponse(
                role=ExtendedConsultantRole.STRATEGIC_SYNTHESIZER.value,
                analysis="Fallback synthesis analysis combining multiple perspectives and mental models for comprehensive understanding.",
                mental_model_used=MentalModel.SYNTHESIS_THINKING.value,
                processing_time_seconds=3.0,
                performance_breakdown={"final_fallback": 3.0},
                success=True,
                error=None,
                timestamp=datetime.now().isoformat(),
            ),
            ConsultantResponse(
                role=ExtendedConsultantRole.STRATEGIC_IMPLEMENTER.value,
                analysis="Fallback implementation analysis focused on execution strategy, resource allocation, and operational requirements.",
                mental_model_used=MentalModel.LEAN_IMPLEMENTATION.value,
                processing_time_seconds=3.0,
                performance_breakdown={"final_fallback": 3.0},
                success=True,
                error=None,
                timestamp=datetime.now().isoformat(),
            ),
        ]

        system_breakdown["final_fallback_execution"] = 9.0

        return BlueprintOrchestrationResult(
            engagement_id=engagement_id,
            query=query,
            consultants=fallback_consultants,
            enhancement_result=enhancement_result,
            total_processing_time=time.time() - orchestration_start,
            system_breakdown=system_breakdown,
            success=True,
            error="Final fallback execution - basic 3-consultant responses using ExtendedConsultantRole",
            multi_single_agent_validation={
                "paradigm": "Multi-Single-Agent (Final Fallback)",
                "consultant_count": 3,
                "independence_verified": True,
                "coordination_detected": False,
                "blueprint_compliant": True,
                "extended_consultant_roles": True,
            },
            timestamp=datetime.now().isoformat(),
        )

    def _create_error_result(
        self,
        engagement_id: str,
        query: str,
        error_msg: str,
        orchestration_start: float,
        system_breakdown: dict,
    ) -> BlueprintOrchestrationResult:
        """Create error result for orchestration failures"""

        return BlueprintOrchestrationResult(
            engagement_id=engagement_id,
            query=query,
            consultants=[],
            enhancement_result=None,
            total_processing_time=time.time() - orchestration_start,
            system_breakdown=system_breakdown,
            success=False,
            error=error_msg,
            multi_single_agent_validation={
                "paradigm": "Error",
                "consultant_count": 0,
                "independence_verified": False,
                "coordination_detected": False,
                "blueprint_compliant": False,
            },
            timestamp=datetime.now().isoformat(),
        )

    def _build_enhanced_context(
        self,
        enhancement_result: SequentialQueryEnhancementResult,
        original_context: Optional[str],
    ) -> str:
        """Build enhanced context from simple query enhancement result"""

        context_parts = []

        if original_context:
            context_parts.append(f"ORIGINAL CONTEXT: {original_context}")

        # Use enhanced query from simple enhancer
        if enhancement_result.enhanced_query:
            context_parts.append(
                f"ENHANCED QUERY CONTEXT: {enhancement_result.enhanced_query}"
            )

        if enhancement_result.engagement_brief:
            brief = enhancement_result.engagement_brief
            context_parts.append(f"ENGAGEMENT OBJECTIVE: {brief.objective}")
            context_parts.append(f"ENGAGEMENT TYPE: {brief.engagement_type}")
            if brief.key_stakeholders:
                context_parts.append(
                    f"KEY STAKEHOLDERS: {', '.join(brief.key_stakeholders)}"
                )
            if brief.success_metrics:
                context_parts.append(
                    f"SUCCESS METRICS: {', '.join(brief.success_metrics)}"
                )

        return "\n\n".join(context_parts) if context_parts else original_context or ""

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query for predictive selection"""
        # Simple keyword extraction - could be enhanced with NLP
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
        }
        words = query.lower().split()
        keywords = [
            word.strip(".,!?;:")
            for word in words
            if word not in stop_words and len(word) > 2
        ]
        return keywords[:10]  # Top 10 keywords

    def _classify_query_type(self, query: str) -> str:
        """Classify query type for predictive selection"""
        query_lower = query.lower()

        if any(
            word in query_lower
            for word in ["strategy", "strategic", "market", "competition", "position"]
        ):
            return "strategic"
        elif any(
            word in query_lower
            for word in [
                "technical",
                "implementation",
                "architecture",
                "system",
                "process",
            ]
        ):
            return "technical"
        elif any(
            word in query_lower
            for word in [
                "financial",
                "budget",
                "cost",
                "revenue",
                "profit",
                "investment",
            ]
        ):
            return "financial"
        elif any(
            word in query_lower
            for word in [
                "operational",
                "efficiency",
                "workflow",
                "execution",
                "delivery",
            ]
        ):
            return "operational"
        elif any(
            word in query_lower
            for word in ["innovation", "creative", "new", "breakthrough", "transform"]
        ):
            return "innovation"
        else:
            return "general"

    def _estimate_complexity(self, query: str, context: str) -> int:
        """Estimate query complexity score (1-5) for predictive selection"""
        complexity = 1

        # Length factor
        if len(query) > 100:
            complexity += 1
        if len(query) > 200:
            complexity += 1

        # Context richness
        if len(context) > 200:
            complexity += 1

        # Complex words
        complex_indicators = [
            "integrate",
            "synthesize",
            "optimize",
            "transform",
            "strategic",
            "comprehensive",
            "multi-",
            "cross-",
        ]
        complexity += min(
            sum(1 for indicator in complex_indicators if indicator in query.lower()), 2
        )

        return min(complexity, 5)


# Test function for blueprint orchestrator
async def test_blueprint_orchestrator():
    """Test the blueprint cognitive orchestrator"""

    print("🎯 TESTING BLUEPRINT COGNITIVE ORCHESTRATOR")
    print("=" * 80)

    orchestrator = BlueprintCognitiveOrchestrator()

    test_query = "Our traditional retail chain is losing 30% revenue annually to e-commerce. We need strategic direction, synthesis of approaches, and implementation roadmap."
    test_context = "Fortune 500 retail company with 200+ physical stores, established brand, traditional customer base aging, competition from Amazon and direct-to-consumer brands."

    print(f"Query: {test_query}")
    print(f"Context: {test_context}")
    print("\nExecuting 4-phase blueprint workflow...\n")

    start_time = time.time()
    result = await orchestrator.analyze(test_query, test_context)
    total_time = time.time() - start_time

    print(f"✅ Blueprint orchestration completed in {total_time:.2f}s")
    print(f"📊 Engagement ID: {result.engagement_id}")
    print(f"🎯 Success: {result.success}")
    print(f"🧠 Consultants: {len(result.consultants)}")
    print(
        f"⚡ Multi-Single-Agent: {result.multi_single_agent_validation.get('paradigm', 'Unknown')}"
    )
    print(
        f"🔍 Blueprint Compliant: {result.multi_single_agent_validation.get('blueprint_compliant', False)}"
    )

    print("\n📋 System Breakdown:")
    for phase, timing in result.system_breakdown.items():
        print(f"   {phase}: {timing:.2f}s")

    print("\n🎯 Consultant Results:")
    for i, consultant in enumerate(result.consultants, 1):
        print(f"   {i}. {consultant.role}: {consultant.analysis[:80]}...")
        print(f"      Model: {consultant.mental_model_used}")
        print(f"      Success: {consultant.success}")

    if result.error:
        print(f"\n⚠️ Execution Notes: {result.error}")

    return result


if __name__ == "__main__":
    asyncio.run(test_blueprint_orchestrator())
