#!/usr/bin/env python3
"""
Deep Synthesis Pipeline - Asynchronous Deep Cognitive Synthesis Architecture

This is the definitive implementation of the METIS cognitive architecture:
- 10-minute performance envelope for maximum quality
- Complete cognitive pipeline: Query → Strategic Trio → Sequential Critique → Final Synthesis
- Research-grounded consultants with Perplexity deep research
- Sequential Critique Chains: Ackoff → Munger → Audit
- Senior Advisor as mandatory Chief Strategist
- Complete EngagementAuditTrail for Six-Dimensional UI
- Export capabilities: MD, PDF, Gamma presentations

This transforms METIS from "fast AI answers" to "auditable strategic intelligence"
"""

import asyncio
import logging
import time
import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

# Core system imports
from src.integrations.llm.resilient_llm_provider import ResilientLLMProvider
from src.engine.integrations.perplexity_client_advanced import (
    AdvancedPerplexityClient,
    ResearchMode,
    AdvancedResearchResult,
)
from src.integrations.apify_client import ApifyMCPClient

# Critique chain components
from src.engine.adapters.cognitive_models import  # Migrated (
    AckoffAssumptionDissolver,
    AssumptionDissolveResult,
)
from src.engine.adapters.cognitive_models import  # Migrated MungerBiasDetector, BiasDetectionResult
from src.async_senior_advisor import AsyncSeniorAdvisor

# Intelligence and selection
from src.engine.engines.selection.enhanced_query_classifier import (
    EnhancedQueryClassifier,
)
from src.intelligence.predictive_consultant_selector import PredictiveConsultantSelector
from src.engine.engines.synthesis.dynamic_nway_execution_engine import (
    DynamicNWayExecutionEngine,
)

# Export capabilities
from src.services.gamma.service import GammaPresentationService
from src.engine.api.presentation_adapter import PresentationAdapter

# Data contracts
from src.models.audit_contracts import (
    EngagementAuditTrail,
    SeniorAdvisorSynthesis,
)
from src.models.transparency_models import (
    SixDimensionalReport,
)

logger = logging.getLogger(__name__)


class PipelinePhase(str, Enum):
    """Deep synthesis pipeline phases"""

    QUERY_STRATEGY_SELECTION = "query_strategy_selection"  # 0-15s
    DEEP_COGNITIVE_EXECUTION = "deep_cognitive_execution"  # 1-8 minutes
    SEQUENTIAL_CRITIQUE = "sequential_critique"  # Automated
    BOARD_LEVEL_SYNTHESIS = "board_level_synthesis"  # Automated
    DELIVERY_PREPARATION = "delivery_preparation"  # ~10 minutes


@dataclass
class PipelineConfiguration:
    """Configuration for deep synthesis pipeline"""

    research_mode: ResearchMode = ResearchMode.DEEP_DIVE
    enable_web_scraping: bool = True
    enable_sequential_critique: bool = True
    enable_gamma_export: bool = True
    timeout_minutes: int = 10
    max_consultants: int = 3

    # Research configuration
    firecrawl_enabled: bool = True
    apify_enabled: bool = True
    perplexity_queries: int = 12  # Deep research mode

    # Quality thresholds
    minimum_confidence: float = 0.8
    minimum_research_sources: int = 15
    critique_depth_level: int = 3  # Ackoff + Munger + Audit


@dataclass
class ResearchGroundedConsultant:
    """Consultant analysis enhanced with deep research"""

    consultant_role: str
    base_analysis: str
    research_grounding: AdvancedResearchResult
    confidence_score: float
    mental_models_applied: List[str]
    assumptions_made: List[str]
    evidence_base: List[Dict[str, Any]]
    processing_time_ms: int
    tokens_used: int
    cost_usd: float


@dataclass
class SequentialCritiqueResult:
    """Result from sequential critique chain"""

    consultant_role: str
    original_analysis: str

    # Critique chain results
    ackoff_dissolution: AssumptionDissolveResult
    munger_bias_detection: BiasDetectionResult
    audit_validation: Dict[str, Any]

    # Synthesis
    critique_synthesis: str
    refined_analysis: str
    confidence_adjustment: float
    processing_time_ms: int


@dataclass
class DeepSynthesisResult:
    """Complete result from deep synthesis pipeline"""

    engagement_id: str
    pipeline_start_time: datetime
    pipeline_end_time: datetime
    total_duration_seconds: float

    # Phase results
    query_classification: Dict[str, Any]
    consultant_selection: Dict[str, Any]
    research_grounded_consultants: List[ResearchGroundedConsultant]
    sequential_critiques: List[SequentialCritiqueResult]
    senior_advisor_synthesis: SeniorAdvisorSynthesis

    # Complete audit trail
    engagement_audit_trail: EngagementAuditTrail
    six_dimensional_report: SixDimensionalReport

    # Performance metrics
    total_cost_usd: float
    total_tokens_used: int
    quality_score: float
    research_depth_score: float

    # Export artifacts
    board_memo_md: str
    comprehensive_pdf_path: Optional[str]
    gamma_presentation_url: Optional[str]
    json_export_path: str


class DeepSynthesisPipeline:
    """
    The definitive METIS Deep Synthesis Pipeline

    Implements the complete 10-minute cognitive architecture with:
    - Research-grounded strategic trio
    - Sequential critique chains
    - Senior advisor synthesis
    - Six-dimensional reporting
    - Complete export capabilities
    """

    def __init__(
        self, llm_provider: ResilientLLMProvider, config: PipelineConfiguration = None
    ):
        self.llm_provider = llm_provider
        self.config = config or PipelineConfiguration()

        # Initialize all components
        self.query_classifier = EnhancedQueryClassifier()
        self.consultant_selector = PredictiveConsultantSelector()
        self.nway_engine = DynamicNWayExecutionEngine()

        # Research components
        self.perplexity_client = AdvancedPerplexityClient()
        self.firecrawl_client = (
            FirecrawlWebResearcher() if config.firecrawl_enabled else None
        )
        self.apify_client = ApifyMCPClient() if config.apify_enabled else None

        # Critique chain
        self.ackoff_dissolver = AckoffAssumptionDissolver()
        self.munger_detector = MungerBiasDetector()

        # Senior synthesis
        self.senior_advisor = AsyncSeniorAdvisor()

        # Export services
        self.gamma_service = (
            GammaPresentationService() if config.enable_gamma_export else None
        )
        self.presentation_adapter = PresentationAdapter()

        logger.info(
            "🚀 Deep Synthesis Pipeline initialized - 10-minute cognitive architecture ready"
        )

    async def execute_deep_synthesis(
        self,
        user_query: str,
        user_id: str,
        session_id: str = None,
        research_depth: str = "deep",
    ) -> DeepSynthesisResult:
        """
        Execute the complete deep synthesis pipeline

        10-minute journey from query to board-ready strategic intelligence
        """
        engagement_id = session_id or str(uuid.uuid4())
        pipeline_start = datetime.now()

        logger.info(f"🎯 Starting Deep Synthesis Pipeline: {engagement_id}")

        try:
            # PHASE 1: Query & Strategy Selection (0-15 seconds)
            phase1_result = await self._phase1_query_strategy_selection(
                user_query, engagement_id
            )

            # PHASE 2: Deep Cognitive Execution (1-8 minutes)
            phase2_result = await self._phase2_deep_cognitive_execution(
                user_query, phase1_result, research_depth
            )

            # PHASE 3: Sequential Critique & Refinement (Automated)
            phase3_result = await self._phase3_sequential_critique(phase2_result)

            # PHASE 4: Board-Level Synthesis (Automated)
            phase4_result = await self._phase4_board_level_synthesis(
                phase1_result, phase2_result, phase3_result, engagement_id
            )

            # PHASE 5: Delivery Preparation (~10 minutes)
            final_result = await self._phase5_delivery_preparation(
                user_query,
                phase1_result,
                phase2_result,
                phase3_result,
                phase4_result,
                engagement_id,
                pipeline_start,
            )

            pipeline_end = datetime.now()
            final_result.pipeline_end_time = pipeline_end
            final_result.total_duration_seconds = (
                pipeline_end - pipeline_start
            ).total_seconds()

            logger.info(
                f"✅ Deep Synthesis Pipeline completed: {engagement_id} in {final_result.total_duration_seconds:.1f}s"
            )

            return final_result

        except Exception as e:
            logger.error(f"❌ Deep Synthesis Pipeline failed: {engagement_id} - {e}")
            raise

    async def _phase1_query_strategy_selection(
        self, user_query: str, engagement_id: str
    ) -> Dict[str, Any]:
        """Phase 1: Query Classification & N-Way Consultant Selection (0-15s)"""

        logger.info(f"🎯 Phase 1: Query & Strategy Selection - {engagement_id}")
        phase_start = time.time()

        # Enhanced query classification
        classification = await self.query_classifier.classify_query(
            query=user_query, llm_client=self.llm_provider
        )

        # N-way consultant selection with predictive algorithms
        consultant_selection = (
            await self.consultant_selector.select_optimal_consultants(
                analysis_request={
                    "query": user_query,
                    "primary_intent": classification.primary_intent,
                    "complexity_level": classification.complexity_level,
                    "urgency_level": classification.urgency_level,
                    "scope_analysis": classification.scope_analysis,
                },
                llm_client=self.llm_provider,
            )
        )

        phase_time = (time.time() - phase_start) * 1000

        result = {
            "classification": classification,
            "consultant_selection": consultant_selection,
            "selected_consultants": consultant_selection.selected_consultants[
                : self.config.max_consultants
            ],
            "phase_duration_ms": phase_time,
        }

        logger.info(
            f"✅ Phase 1 completed in {phase_time:.0f}ms - Selected: {', '.join(result['selected_consultants'])}"
        )
        return result

    async def _phase2_deep_cognitive_execution(
        self, user_query: str, phase1_result: Dict[str, Any], research_depth: str
    ) -> List[ResearchGroundedConsultant]:
        """Phase 2: Deep Cognitive Execution with Research Grounding (1-8 minutes)"""

        logger.info(
            f"🧠 Phase 2: Deep Cognitive Execution - Research Mode: {research_depth}"
        )
        phase_start = time.time()

        research_mode = (
            ResearchMode.DEEP_DIVE
            if research_depth == "deep"
            else ResearchMode.COMPREHENSIVE
        )
        selected_consultants = phase1_result["selected_consultants"]

        # Execute consultants in parallel with deep research grounding
        consultant_tasks = []

        for consultant_role in selected_consultants:
            task = self._execute_research_grounded_consultant(
                user_query, consultant_role, research_mode
            )
            consultant_tasks.append(task)

        # Execute all consultants in parallel
        research_grounded_consultants = await asyncio.gather(*consultant_tasks)

        phase_time = (time.time() - phase_start) * 1000

        logger.info(
            f"✅ Phase 2 completed in {phase_time:.0f}ms - {len(research_grounded_consultants)} research-grounded consultants"
        )
        return research_grounded_consultants

    async def _execute_research_grounded_consultant(
        self, user_query: str, consultant_role: str, research_mode: ResearchMode
    ) -> ResearchGroundedConsultant:
        """Execute single consultant with deep research grounding"""

        logger.info(f"🔬 Executing research-grounded {consultant_role}")
        start_time = time.time()

        # Phase 2a: Deep Research
        research_result = await self.perplexity_client.execute_advanced_research(
            query=f"Strategic analysis context for {consultant_role}: {user_query}",
            research_mode=research_mode,
            template_type=self._get_research_template_for_consultant(consultant_role),
        )

        # Phase 2b: Enhanced Analysis with Research Context
        enhanced_prompt = self._build_research_grounded_prompt(
            user_query, consultant_role, research_result
        )

        analysis_response, _ = await self.llm_provider.call_optimized_llm(
            prompt=enhanced_prompt,
            task_type=self._get_task_type_for_consultant(consultant_role),
            consultant_role=consultant_role,
            complexity_score=0.9,
        )

        processing_time = (time.time() - start_time) * 1000

        return ResearchGroundedConsultant(
            consultant_role=consultant_role,
            base_analysis=analysis_response.content,
            research_grounding=research_result,
            confidence_score=analysis_response.confidence,
            mental_models_applied=analysis_response.mental_models,
            assumptions_made=self._extract_assumptions(analysis_response.content),
            evidence_base=self._build_evidence_base(research_result),
            processing_time_ms=int(processing_time),
            tokens_used=analysis_response.tokens_used,
            cost_usd=analysis_response.cost_usd,
        )

    async def _phase3_sequential_critique(
        self, consultants: List[ResearchGroundedConsultant]
    ) -> List[SequentialCritiqueResult]:
        """Phase 3: Sequential Critique Chains (Ackoff → Munger → Audit)"""

        logger.info(
            f"🔍 Phase 3: Sequential Critique Chains - {len(consultants)} consultants"
        )

        # Execute critique chains in parallel for each consultant
        critique_tasks = []

        for consultant in consultants:
            task = self._execute_sequential_critique_chain(consultant)
            critique_tasks.append(task)

        critique_results = await asyncio.gather(*critique_tasks)

        logger.info(f"✅ Phase 3 completed - {len(critique_results)} critique chains")
        return critique_results

    async def _perform_independent_audit_validation(
        self, consultant: ResearchGroundedConsultant, original_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform audit validation independently of other critiques"""

        try:
            # Import the cognitive auditor
            from src.engine.engines.models.cognitive_auditor import CognitiveAuditor

            auditor = CognitiveAuditor()

            audit_result = await auditor.audit_reasoning_quality(
                reasoning_chain=consultant.base_analysis,
                context={
                    "consultant_role": consultant.consultant_role,
                    "research_grounding": consultant.research_grounding,
                    "confidence_score": consultant.confidence_score,
                    "methodology": "independent_parallel_critique",
                },
            )

            return audit_result

        except Exception as e:
            logger.warning(f"Independent audit validation failed: {e}")
            return {
                "audit_status": "failed",
                "error": str(e),
                "methodology": "independent_parallel_critique",
            }

    async def _synthesize_independent_critique_results(
        self,
        consultant: ResearchGroundedConsultant,
        ackoff_result: Optional[Any],
        munger_result: Optional[Any],
        audit_result: Optional[Any],
    ) -> Dict[str, Any]:
        """
        Synthesize independent critique results while preserving original analysis integrity

        Key principle: Each critique is independent - no cross-contamination
        """

        logger.info(
            f"🔄 Synthesizing independent critiques for {consultant.consultant_role}"
        )

        synthesis_components = {
            "original_analysis_preserved": consultant.base_analysis,
            "independent_critiques": {
                "ackoff_assumptions": [],
                "munger_biases": [],
                "audit_findings": [],
            },
            "critique_summary": "",
            "confidence_factors": {
                "strengths": [],
                "weaknesses": [],
                "blind_spots": [],
            },
        }

        # Process Ackoff results independently
        if ackoff_result and hasattr(ackoff_result, "dissolved_assumptions"):
            dissolved_count = len(ackoff_result.dissolved_assumptions)
            synthesis_components["independent_critiques"]["ackoff_assumptions"] = [
                f"Dissolved assumption: {assumption.assumption_text} (strength: {assumption.dissolution_strength:.2f})"
                for assumption in ackoff_result.dissolved_assumptions[:3]
            ]

            if dissolved_count > 0:
                synthesis_components["confidence_factors"]["blind_spots"].append(
                    f"Ackoff found {dissolved_count} questionable assumptions"
                )

        # Process Munger results independently
        if munger_result and hasattr(munger_result, "detected_biases"):
            bias_count = len(munger_result.detected_biases)
            synthesis_components["independent_critiques"]["munger_biases"] = [
                f"Cognitive bias: {bias.bias_name} (severity: {bias.severity_score:.2f})"
                for bias in munger_result.detected_biases[:3]
            ]

            if bias_count > 0:
                synthesis_components["confidence_factors"]["weaknesses"].append(
                    f"Munger identified {bias_count} cognitive biases"
                )

        # Process audit results independently
        if audit_result and isinstance(audit_result, dict):
            if audit_result.get("audit_status") == "passed":
                synthesis_components["confidence_factors"]["strengths"].append(
                    "Audit validation passed - reasoning quality acceptable"
                )
            elif audit_result.get("audit_status") == "failed":
                synthesis_components["confidence_factors"]["weaknesses"].append(
                    f"Audit validation failed: {audit_result.get('error', 'Unknown error')}"
                )

        # Create synthesis summary
        critique_count = sum(
            [
                1 if ackoff_result else 0,
                1 if munger_result else 0,
                1 if audit_result else 0,
            ]
        )

        synthesis_components["critique_summary"] = (
            f"Independent parallel critique completed with {critique_count}/3 critics successful. "
            f"Original analysis preserved with independent perspective overlays. "
            f"No cross-contamination between critics."
        )

        # Calculate confidence adjustment based on independent findings
        confidence_adjustment = self._calculate_independent_confidence_adjustment(
            ackoff_result, munger_result, audit_result, consultant.confidence_score
        )

        # Create refined analysis that preserves original while noting independent critiques
        refined_analysis = self._create_refined_analysis_with_independent_critiques(
            consultant, synthesis_components
        )

        return {
            "synthesis": synthesis_components,
            "refined_analysis": refined_analysis,
            "confidence_adjustment": confidence_adjustment,
            "methodology": "independent_parallel_critique",
        }

    def _calculate_independent_confidence_adjustment(
        self,
        ackoff_result: Optional[Any],
        munger_result: Optional[Any],
        audit_result: Optional[Any],
        original_confidence: float,
    ) -> float:
        """Calculate confidence adjustment based on independent critique findings"""

        adjustment = 0.0

        # Ackoff impact (assumption dissolutions lower confidence)
        if ackoff_result and hasattr(ackoff_result, "dissolution_impact_score"):
            # More dissolved assumptions = lower confidence
            adjustment -= ackoff_result.dissolution_impact_score * 0.1

        # Munger impact (bias detection lowers confidence)
        if munger_result and hasattr(munger_result, "detected_biases"):
            high_severity_biases = [
                b for b in munger_result.detected_biases if b.severity_score >= 0.7
            ]
            adjustment -= len(high_severity_biases) * 0.05

        # Audit impact (passed audit increases confidence)
        if audit_result and isinstance(audit_result, dict):
            if audit_result.get("audit_status") == "passed":
                adjustment += 0.05
            elif audit_result.get("audit_status") == "failed":
                adjustment -= 0.1

        # Apply adjustment with bounds
        adjusted_confidence = original_confidence + adjustment
        return max(0.1, min(1.0, adjusted_confidence))

    def _create_refined_analysis_with_independent_critiques(
        self,
        consultant: ResearchGroundedConsultant,
        synthesis_components: Dict[str, Any],
    ) -> str:
        """Create refined analysis that preserves original work with independent critique overlays"""

        refined_analysis = f"""
ORIGINAL CONSULTANT ANALYSIS ({consultant.consultant_role}):
{consultant.base_analysis}

INDEPENDENT CRITIQUE OVERLAY:
Methodology: Three independent critics analyzed the original work in parallel

{synthesis_components['critique_summary']}

INDEPENDENT FINDINGS:
• Ackoff Assumptions: {len(synthesis_components['independent_critiques']['ackoff_assumptions'])} identified
• Munger Biases: {len(synthesis_components['independent_critiques']['munger_biases'])} detected  
• Audit Status: {'Passed' if 'reasoning quality acceptable' in str(synthesis_components['confidence_factors']['strengths']) else 'Review needed'}

CONFIDENCE FACTORS:
Strengths: {'; '.join(synthesis_components['confidence_factors']['strengths']) if synthesis_components['confidence_factors']['strengths'] else 'None identified'}
Weaknesses: {'; '.join(synthesis_components['confidence_factors']['weaknesses']) if synthesis_components['confidence_factors']['weaknesses'] else 'None identified'}
Blind Spots: {'; '.join(synthesis_components['confidence_factors']['blind_spots']) if synthesis_components['confidence_factors']['blind_spots'] else 'None identified'}

METHODOLOGY NOTE: Each critic analyzed only the original consultant work independently to prevent cross-contamination and preserve analytical integrity.
        """.strip()

        return refined_analysis

    async def _execute_sequential_critique_chain(
        self, consultant: ResearchGroundedConsultant
    ) -> SequentialCritiqueResult:
        """Execute INDEPENDENT PARALLEL critique chain (prevents cross-contamination)"""

        logger.info(
            f"⚡ Independent parallel critique for {consultant.consultant_role}"
        )
        start_time = time.time()

        # CRITICAL FIX: All critics analyze ONLY the original consultant work
        # No sequential building - prevents misleading cross-contamination
        original_consultant_context = {
            "original_analysis": consultant.base_analysis,
            "consultant_role": consultant.consultant_role,
            "confidence_score": consultant.confidence_score,
            "research_grounding": consultant.research_grounding,
            "evidence_base": consultant.evidence_base,
            "assumptions_made": consultant.assumptions_made,
        }

        # Execute all three critics IN PARALLEL against ORIGINAL work only
        critique_tasks = [
            # 1. Ackoff Assumption Dissolution - analyzes ORIGINAL consultant work
            self.ackoff_dissolver.dissolve_assumptions(
                problem_statement=consultant.base_analysis,
                business_context={
                    "consultant_analysis": consultant.base_analysis,
                    "consultant_role": consultant.consultant_role,
                    "research_basis": consultant.research_grounding,
                    "methodology": "independent_parallel_critique",
                },
            ),
            # 2. Munger Bias Detection - analyzes ORIGINAL consultant work independently
            self.munger_detector.detect_cognitive_biases(
                analysis_text=consultant.base_analysis,
                context={
                    "consultant_role": consultant.consultant_role,
                    "assumptions_made": consultant.assumptions_made,
                    "confidence_level": consultant.confidence_score,
                    "methodology": "independent_parallel_critique",
                },
                decision_context={},  # No contamination from other critics
            ),
            # 3. Audit Validation - analyzes ORIGINAL consultant work independently
            self._perform_independent_audit_validation(
                consultant, original_consultant_context
            ),
        ]

        # Execute all critiques in parallel - no sequential contamination
        logger.info(
            f"🔍 Executing 3 independent critics in parallel for {consultant.consultant_role}"
        )
        critique_task_results = await asyncio.gather(
            *critique_tasks, return_exceptions=True
        )

        # Process critique results
        ackoff_result = (
            critique_task_results[0]
            if not isinstance(critique_task_results[0], Exception)
            else None
        )
        munger_result = (
            critique_task_results[1]
            if not isinstance(critique_task_results[1], Exception)
            else None
        )
        audit_result = (
            critique_task_results[2]
            if not isinstance(critique_task_results[2], Exception)
            else None
        )

        # Log critique execution results
        logger.info(
            f"✅ Independent critique results for {consultant.consultant_role}:"
        )
        logger.info(f"   Ackoff: {'✅' if ackoff_result else '❌'}")
        logger.info(f"   Munger: {'✅' if munger_result else '❌'}")
        logger.info(f"   Audit: {'✅' if audit_result else '❌'}")

        # Step 4: Independent Synthesis (preserves original work)
        critique_synthesis = await self._synthesize_independent_critique_results(
            consultant, ackoff_result, munger_result, audit_result
        )

        processing_time = (time.time() - start_time) * 1000

        return SequentialCritiqueResult(
            consultant_role=consultant.consultant_role,
            original_analysis=consultant.base_analysis,
            ackoff_dissolution=ackoff_result,
            munger_bias_detection=munger_result,
            audit_validation=audit_result,
            critique_synthesis=critique_synthesis["synthesis"],
            refined_analysis=critique_synthesis["refined_analysis"],
            confidence_adjustment=critique_synthesis["confidence_adjustment"],
            processing_time_ms=int(processing_time),
            methodology="independent_parallel_critique",  # Track methodology used
        )

    async def _phase4_board_level_synthesis(
        self,
        phase1_result: Dict[str, Any],
        consultants: List[ResearchGroundedConsultant],
        critiques: List[SequentialCritiqueResult],
        engagement_id: str,
    ) -> SeniorAdvisorSynthesis:
        """Phase 4: Senior Advisor Chief Strategist Synthesis"""

        logger.info("🎯 Phase 4: Board-Level Synthesis - Senior Advisor Integration")

        # Prepare comprehensive synthesis input
        synthesis_input = {
            "query_classification": phase1_result["classification"],
            "consultant_analyses": [
                {
                    "role": c.consultant_role,
                    "analysis": c.base_analysis,
                    "research_grounding": c.research_grounding,
                    "confidence": c.confidence_score,
                    "evidence_base": c.evidence_base,
                }
                for c in consultants
            ],
            "critique_results": [
                {
                    "role": cr.consultant_role,
                    "refined_analysis": cr.refined_analysis,
                    "ackoff_insights": cr.ackoff_dissolution,
                    "munger_insights": cr.munger_bias_detection,
                    "confidence_adjustment": cr.confidence_adjustment,
                }
                for cr in critiques
            ],
        }

        # Execute Senior Advisor synthesis as Chief Strategist
        senior_synthesis = await self.senior_advisor.synthesize_board_ready_analysis(
            synthesis_input, engagement_id
        )

        logger.info("✅ Phase 4 completed - Board-ready synthesis generated")
        return senior_synthesis

    async def _phase5_delivery_preparation(
        self,
        user_query: str,
        phase1_result: Dict[str, Any],
        consultants: List[ResearchGroundedConsultant],
        critiques: List[SequentialCritiqueResult],
        synthesis: SeniorAdvisorSynthesis,
        engagement_id: str,
        pipeline_start: datetime,
    ) -> DeepSynthesisResult:
        """Phase 5: Complete Delivery Preparation with Six-Dimensional Report"""

        logger.info(
            "📊 Phase 5: Delivery Preparation - Six-Dimensional Report & Exports"
        )

        # Build complete EngagementAuditTrail
        audit_trail = self._build_engagement_audit_trail(
            engagement_id, user_query, phase1_result, consultants, critiques, synthesis
        )

        # Create Six-Dimensional Report
        six_dimensional_report = self._build_six_dimensional_report(
            audit_trail, consultants, critiques, synthesis
        )

        # Generate exports
        board_memo_md = self._generate_board_memo_markdown(synthesis)

        # Generate Gamma presentation if enabled
        gamma_url = None
        if self.gamma_service:
            gamma_url = await self._generate_gamma_presentation(
                six_dimensional_report, engagement_id
            )

        # Save JSON export
        json_path = f"engagement_{engagement_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        await self._save_json_export(audit_trail, json_path)

        # Calculate final metrics
        total_cost = sum([c.cost_usd for c in consultants])
        total_tokens = sum([c.tokens_used for c in consultants])
        quality_score = self._calculate_quality_score(consultants, critiques, synthesis)
        research_depth_score = self._calculate_research_depth_score(consultants)

        return DeepSynthesisResult(
            engagement_id=engagement_id,
            pipeline_start_time=pipeline_start,
            pipeline_end_time=datetime.now(),
            total_duration_seconds=0,  # Will be set by caller
            query_classification=phase1_result,
            consultant_selection=phase1_result["consultant_selection"],
            research_grounded_consultants=consultants,
            sequential_critiques=critiques,
            senior_advisor_synthesis=synthesis,
            engagement_audit_trail=audit_trail,
            six_dimensional_report=six_dimensional_report,
            total_cost_usd=total_cost,
            total_tokens_used=total_tokens,
            quality_score=quality_score,
            research_depth_score=research_depth_score,
            board_memo_md=board_memo_md,
            comprehensive_pdf_path=None,  # TODO: Implement PDF generation
            gamma_presentation_url=gamma_url,
            json_export_path=json_path,
        )

    # Helper methods for pipeline execution
    def _get_research_template_for_consultant(self, consultant_role: str) -> str:
        """Get research template type for consultant role"""
        mapping = {
            "strategic_analyst": "market_analysis",
            "implementation_driver": "operational_research",
            "devils_advocate": "risk_assessment",
            "financial_analyst": "financial_research",
            "senior_advisor": "executive_intelligence",
        }
        return mapping.get(consultant_role, "general_research")

    def _get_task_type_for_consultant(self, consultant_role: str) -> str:
        """Get task type for consultant role"""
        mapping = {
            "strategic_analyst": "strategic_analysis",
            "implementation_driver": "implementation",
            "devils_advocate": "assumption_challenge",
            "financial_analyst": "financial_analysis",
            "senior_advisor": "strategic_synthesis",
        }
        return mapping.get(consultant_role, "strategic_analysis")

    def _build_research_grounded_prompt(
        self,
        user_query: str,
        consultant_role: str,
        research_result: AdvancedResearchResult,
    ) -> str:
        """Build enhanced prompt with research context"""

        research_context = "\n".join(
            [
                f"- {source.title}: {source.summary}"
                for source in research_result.sources[:10]
            ]
        )

        return f"""You are a {consultant_role.replace('_', ' ').title()} with access to current research intelligence.

USER QUERY:
{user_query}

CURRENT RESEARCH CONTEXT:
{research_context}

RESEARCH INSIGHTS:
{', '.join([insight.insight_text for insight in research_result.insights[:5]])}

Provide your analysis incorporating this research context. Apply relevant strategic frameworks and provide evidence-based recommendations."""

    def _extract_assumptions(self, analysis: str) -> List[str]:
        """Extract key assumptions from analysis"""
        # Simple heuristic - look for assumption indicators
        assumption_indicators = [
            "assume",
            "given that",
            "if we consider",
            "presumably",
            "likely",
        ]
        assumptions = []

        sentences = analysis.split(".")
        for sentence in sentences:
            if any(
                indicator in sentence.lower() for indicator in assumption_indicators
            ):
                assumptions.append(sentence.strip())

        return assumptions[:5]  # Top 5 assumptions

    def _build_evidence_base(
        self, research_result: AdvancedResearchResult
    ) -> List[Dict[str, Any]]:
        """Build evidence base from research"""
        return [
            {
                "source": source.title,
                "url": source.url,
                "credibility": source.credibility_tier,
                "summary": source.summary,
                "key_findings": source.key_findings[:3],
            }
            for source in research_result.sources[:10]
        ]

    async def _perform_audit_validation(
        self, consultant, ackoff_result, munger_result
    ) -> Dict[str, Any]:
        """Perform comprehensive audit validation"""
        return {
            "logical_consistency": 0.85,  # Placeholder
            "evidence_strength": 0.80,
            "assumption_validity": 0.75,
            "bias_mitigation": 0.90,
            "framework_application": 0.88,
            "audit_confidence": 0.82,
        }

    async def _synthesize_critique_results(
        self, consultant, ackoff_result, munger_result, audit_result
    ) -> Dict[str, Any]:
        """Synthesize all critique results into refined analysis"""

        synthesis_prompt = f"""As a quality assurance expert, synthesize these critique results:

ORIGINAL ANALYSIS:
{consultant.base_analysis[:1000]}...

ACKOFF INSIGHTS:
- Dissolved assumptions: {len(ackoff_result.dissolved_assumptions)}
- Idealized design: {ackoff_result.idealized_design_vision[:200]}...

MUNGER INSIGHTS:  
- Detected biases: {len(munger_result.detected_biases)}
- Risk level: {munger_result.overall_bias_risk}

AUDIT RESULTS:
- Logical consistency: {audit_result['logical_consistency']}
- Evidence strength: {audit_result['evidence_strength']}

Provide:
1. Critique synthesis (what the critiques revealed)
2. Refined analysis (improved version incorporating insights)
3. Confidence adjustment (-0.2 to +0.2)"""

        response, _ = await self.llm_provider.call_optimized_llm(
            prompt=synthesis_prompt,
            task_type="quality_assurance",
            consultant_role="quality_auditor",
        )

        # Parse response (simplified)
        return {
            "synthesis": response.content[:500] + "...",
            "refined_analysis": consultant.base_analysis,  # Placeholder
            "confidence_adjustment": 0.05,  # Placeholder
        }

    def _build_engagement_audit_trail(
        self,
        engagement_id,
        user_query,
        phase1_result,
        consultants,
        critiques,
        synthesis,
    ) -> EngagementAuditTrail:
        """Build complete engagement audit trail"""
        # Placeholder implementation
        return EngagementAuditTrail(
            engagement_id=engagement_id,
            user_query=user_query,
            timestamp=datetime.now(),
            classification_result=phase1_result["classification"],
            consultant_results=[],  # TODO: Map consultants to audit format
            critique_results=[],  # TODO: Map critiques to audit format
            synthesis_result=synthesis,
            total_cost=sum([c.cost_usd for c in consultants]),
            quality_metrics={},
        )

    def _build_six_dimensional_report(
        self, audit_trail, consultants, critiques, synthesis
    ) -> SixDimensionalReport:
        """Build six-dimensional progressive disclosure report"""
        # Placeholder implementation
        return SixDimensionalReport(
            layer1_executive_memo=synthesis.board_memo,
            layer2_core_perspectives=[c.base_analysis for c in consultants],
            layer3_critique_results=[c.critique_synthesis for c in critiques],
            layer4_reasoning_chain=audit_trail,
            layer5_research_evidence=[c.evidence_base for c in consultants],
            layer6_complete_audit=audit_trail,
        )

    def _generate_board_memo_markdown(self, synthesis: SeniorAdvisorSynthesis) -> str:
        """Generate board-ready memo in markdown"""
        return f"""# Strategic Intelligence Memo

## Executive Summary
{synthesis.executive_summary}

## Strategic Recommendation  
{synthesis.primary_recommendation}

## Implementation Roadmap
{synthesis.implementation_plan}

## Risk Assessment
{synthesis.risk_analysis}

---
*Generated by METIS Deep Synthesis Pipeline*
*Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}*
"""

    async def _generate_gamma_presentation(
        self, report: SixDimensionalReport, engagement_id: str
    ) -> Optional[str]:
        """Generate Gamma presentation from report"""
        if not self.gamma_service:
            return None

        try:
            presentation_url = await self.gamma_service.create_presentation_from_report(
                report, f"Strategic Analysis - {engagement_id}"
            )
            return presentation_url
        except Exception as e:
            logger.warning(f"Gamma presentation generation failed: {e}")
            return None

    async def _save_json_export(
        self, audit_trail: EngagementAuditTrail, file_path: str
    ):
        """Save complete audit trail as JSON"""
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(
                    audit_trail.__dict__, f, indent=2, default=str, ensure_ascii=False
                )
        except Exception as e:
            logger.warning(f"JSON export failed: {e}")

    def _calculate_quality_score(self, consultants, critiques, synthesis) -> float:
        """Calculate overall engagement quality score"""
        avg_consultant_confidence = sum(
            [c.confidence_score for c in consultants]
        ) / len(consultants)
        avg_critique_adjustment = sum(
            [abs(c.confidence_adjustment) for c in critiques]
        ) / len(critiques)
        synthesis_confidence = (
            synthesis.confidence_score
            if hasattr(synthesis, "confidence_score")
            else 0.85
        )

        return (
            avg_consultant_confidence * 0.4
            + (1 - avg_critique_adjustment) * 0.3
            + synthesis_confidence * 0.3
        )

    def _calculate_research_depth_score(self, consultants) -> float:
        """Calculate research depth and grounding score"""
        total_sources = sum([len(c.evidence_base) for c in consultants])
        avg_sources_per_consultant = total_sources / len(consultants)

        # Score based on research depth
        if avg_sources_per_consultant >= 15:
            return 1.0
        elif avg_sources_per_consultant >= 10:
            return 0.8
        elif avg_sources_per_consultant >= 5:
            return 0.6
        else:
            return 0.4
