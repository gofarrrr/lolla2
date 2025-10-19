"""
Socratic Cognitive Forge API - METIS V5 Week 3 Final Implementation
REST API endpoints for the intelligent question-based user intake system

This API provides:
1. POST /generate-questions - Generate Socratic questions for a problem
2. POST /enhance-query - Build enhanced query from responses
3. POST /complete-analysis - Full Socratic → Enhanced Query → Consultant Selection flow
"""

import asyncio
import time
import uuid
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import logging

from src.engine.engines.core.socratic_cognitive_forge import (
    SocraticCognitiveForge,
    UserResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/socratic-forge", tags=["Socratic Cognitive Forge"])

# Global forge instance (could be dependency injected)
_forge_instance: Optional[SocraticCognitiveForge] = None


def get_forge() -> SocraticCognitiveForge:
    """Get or create Socratic Cognitive Forge instance"""
    global _forge_instance
    if _forge_instance is None:
        _forge_instance = SocraticCognitiveForge()
        logger.info("🎭 Socratic Cognitive Forge initialized for API")
    return _forge_instance


# Request/Response Models
class ProblemRequest(BaseModel):
    statement: str = Field(
        ..., min_length=10, max_length=1000, description="Problem statement to analyze"
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional context (industry, company size, etc.)"
    )
    user_id: Optional[str] = Field(
        default=None, description="Optional user ID for tracking"
    )


class SocraticQuestionModel(BaseModel):
    question_id: str
    text: str
    tier: str
    reasoning: str
    expected_improvement: str
    category: str
    is_required: bool


class QuestionSetModel(BaseModel):
    tier: str
    title: str
    description: str
    quality_target: int
    questions: List[SocraticQuestionModel]
    expected_benefit: str


class GenerateQuestionsResponse(BaseModel):
    success: bool
    engagement_id: str
    problem_statement: str
    question_sets: List[QuestionSetModel]
    total_questions: int
    generation_time_ms: int
    forge_status: Dict[str, Any]


class UserResponseModel(BaseModel):
    question_id: str
    answer: str = Field(..., min_length=1, max_length=2000)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class EnhanceQueryRequest(BaseModel):
    original_statement: str
    user_responses: List[UserResponseModel]
    context: Optional[Dict[str, Any]] = None


class EnhanceQueryResponse(BaseModel):
    success: bool
    engagement_id: str
    original_statement: str
    enhanced_statement: str
    quality_level: int
    confidence_score: float
    context_enrichment: Dict[str, Any]
    processing_time_ms: int


class CompleteAnalysisRequest(BaseModel):
    problem_statement: str
    user_responses: List[UserResponseModel]
    context: Optional[Dict[str, Any]] = None
    request_consultant_selection: bool = Field(
        default=True, description="Whether to run consultant selection"
    )
    run_full_pipeline: bool = Field(
        default=False,
        description="Whether to run the complete cognitive pipeline (Analysis → Critique → Senior Advisor)",
    )


class ConsultantSelectionModel(BaseModel):
    consultant_id: str
    name: str
    specialization: str
    selection_reason: str
    confidence_score: float


class ConsultantAnalysisModel(BaseModel):
    consultant_id: str
    analysis: str
    tokens_used: int
    processing_time_seconds: float


class DevilsAdvocateCritiqueModel(BaseModel):
    consultant_id: str
    critique: str
    tokens_used: int
    processing_time_seconds: float


class SeniorAdvisorModel(BaseModel):
    rapporteur_analysis: str
    tokens_used: int
    context_preservation_score: float
    processing_time_seconds: float


class CompleteAnalysisResponse(BaseModel):
    success: bool
    engagement_id: str
    socratic_phase: Dict[str, Any]
    enhanced_query: EnhanceQueryResponse
    consultant_selection: Optional[List[ConsultantSelectionModel]] = None
    # Full Pipeline Results (when run_full_pipeline=True) - Defensive Contract
    consultant_analyses: List[ConsultantAnalysisModel] = Field(
        default_factory=list,
        description="Consultant analyses (empty list if not requested)",
    )
    devils_advocate_critiques: List[DevilsAdvocateCritiqueModel] = Field(
        default_factory=list,
        description="Devil's advocate critiques (empty list if not requested)",
    )
    senior_advisor_rapporteur: Optional[SeniorAdvisorModel] = None
    pipeline_metrics: Dict[str, Any] = Field(
        default_factory=dict, description="Pipeline performance metrics"
    )
    # Standard fields
    audit_trail: List[Dict[str, Any]]
    total_processing_time_ms: int


# API Endpoints
@router.post("/generate-questions", response_model=GenerateQuestionsResponse)
async def generate_socratic_questions(request: ProblemRequest):
    """
    Generate intelligent Socratic questions for a problem statement

    This endpoint analyzes the problem and creates tiered questions:
    - Essential (60% quality): Must answer to proceed
    - Strategic (85% quality): Significant improvement
    - Expert (95% quality): Maximum sophistication
    """

    start_time = time.time()
    forge = get_forge()

    try:
        logger.info(
            f"🎭 Generating Socratic questions for: {request.statement[:50]}..."
        )

        # Generate questions using the forge
        question_sets = await forge.forge_questions(
            problem_statement=request.statement, context=request.context
        )

        # Convert to response models
        response_sets = []
        total_questions = 0

        for qs in question_sets:
            questions_models = [
                SocraticQuestionModel(
                    question_id=q.question_id,
                    text=q.text,
                    tier=q.tier.value,
                    reasoning=q.reasoning,
                    expected_improvement=q.expected_improvement,
                    category=q.category,
                    is_required=q.is_required,
                )
                for q in qs.questions
            ]

            response_sets.append(
                QuestionSetModel(
                    tier=qs.tier.value,
                    title=qs.title,
                    description=qs.description,
                    quality_target=qs.quality_target,
                    questions=questions_models,
                    expected_benefit=qs.expected_benefit,
                )
            )

            total_questions += len(questions_models)

        processing_time = int((time.time() - start_time) * 1000)

        return GenerateQuestionsResponse(
            success=True,
            engagement_id=str(uuid.uuid4()),
            problem_statement=request.statement,
            question_sets=response_sets,
            total_questions=total_questions,
            generation_time_ms=processing_time,
            forge_status=forge.get_forge_status(),
        )

    except Exception as e:
        logger.error(f"❌ Question generation failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate Socratic questions: {str(e)}"
        )


@router.post("/enhance-query", response_model=EnhanceQueryResponse)
async def enhance_query(request: EnhanceQueryRequest):
    """
    Build enhanced query from original statement and user responses to Socratic questions

    This endpoint takes user responses and creates a rich, contextual query
    that enables optimal consultant selection and analysis.
    """

    start_time = time.time()
    forge = get_forge()

    try:
        logger.info(f"🔧 Enhancing query with {len(request.user_responses)} responses")

        # Convert request responses to internal format
        user_responses = [
            UserResponse(
                question_id=r.question_id, answer=r.answer, confidence=r.confidence
            )
            for r in request.user_responses
        ]

        # Build enhanced query
        enhanced_query = await forge.forge_enhanced_query(
            original_statement=request.original_statement,
            user_responses=user_responses,
            context=request.context,
        )

        processing_time = int((time.time() - start_time) * 1000)

        return EnhanceQueryResponse(
            success=True,
            engagement_id=str(uuid.uuid4()),
            original_statement=enhanced_query.original_statement,
            enhanced_statement=enhanced_query.enhanced_statement,
            quality_level=enhanced_query.quality_level,
            confidence_score=enhanced_query.confidence_score,
            context_enrichment=enhanced_query.context_enrichment,
            processing_time_ms=processing_time,
        )

    except Exception as e:
        logger.error(f"❌ Query enhancement failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to enhance query: {str(e)}"
        )


@router.post("/complete-analysis", response_model=CompleteAnalysisResponse)
async def complete_socratic_analysis(request: CompleteAnalysisRequest):
    """
    Complete end-to-end Socratic analysis flow

    This endpoint provides the full "Socratic Cognitive Forge" experience:
    1. Enhances the query from user responses
    2. Selects optimal consultants using the enhanced context
    3. Returns complete audit trail of the process
    """

    start_time = time.time()
    forge = get_forge()

    try:
        logger.info(
            f"🎯 Running complete Socratic analysis with {len(request.user_responses)} responses"
        )

        # Step 1: Convert responses and enhance query
        user_responses = [
            UserResponse(
                question_id=r.question_id, answer=r.answer, confidence=r.confidence
            )
            for r in request.user_responses
        ]

        enhanced_query = await forge.forge_enhanced_query(
            original_statement=request.problem_statement,
            user_responses=user_responses,
            context=request.context,
        )

        # Step 2: Consultant selection (if requested)
        consultant_selection = None
        audit_trail = []
        selection_result = None

        if request.request_consultant_selection:
            logger.info("🧠 Selecting optimal consultants with enhanced context")

            selection_result, combined_audit_trail = (
                await forge.integrate_with_consultant_engine(enhanced_query)
            )

            # Convert consultant selection to response format
            consultant_selection = [
                ConsultantSelectionModel(
                    consultant_id=c.consultant_id,
                    name=c.blueprint.name,
                    specialization=c.blueprint.specialization,
                    selection_reason=c.selection_reason,
                    confidence_score=getattr(c, "confidence_score", 0.8),
                )
                for c in selection_result.selected_consultants
            ]

            audit_trail = combined_audit_trail

        # Step 3: Full Progressive Assembly Pipeline (if requested) - Defensive Contract
        consultant_analyses = []  # Always initialize as empty list, never None
        devils_advocate_critiques = []  # Always initialize as empty list, never None
        senior_advisor_rapporteur = (
            None  # Initialize as None, will be SeniorAdvisorModel if pipeline runs
        )
        pipeline_metrics = {}

        if request.run_full_pipeline and selection_result:
            logger.info("🔥 Running full Progressive Assembly pipeline")

            # Import Progressive Assembly pipeline components
            from src.integrations.llm.unified_client import UnifiedLLMClient
            from src.core.unified_context_stream import (
                UnifiedContextStream,
            )
            from src.engine.services.research_brief_service import (
                ResearchBriefService,
                ResearchBriefConfig,
            )
            from src.engine.core.feature_flags import FeatureFlagService, FeatureFlag

            llm_client = UnifiedLLMClient()
            from src.core.unified_context_stream import get_unified_context_stream
            pipeline_context_stream = get_unified_context_stream()

            # Optional: Generate and attach a neutral Research Brief (feature-flagged)
            research_brief = None
            try:
                ff = FeatureFlagService()
                if ff.is_enabled(FeatureFlag.ENABLE_RESEARCH_BRIEF):
                    rb_service = ResearchBriefService(
                        context_stream=pipeline_context_stream,
                        flags=ff,
                        config=ResearchBriefConfig(),
                    )
                    research_brief = await rb_service.generate_brief(
                        enhanced_query.enhanced_statement,
                        business_context=request.context,
                    )
            except Exception as rb_err:
                logger.warning(
                    f"⚠️ Research Brief generation failed or skipped: {rb_err}"
                )

            # Pipeline Phase 1: Parallel Consultant Analyses
            logger.info("🔥 Phase 1: Executing parallel consultant analyses...")
            analysis_start_time = time.time()

            # Create mock consultant data for Progressive Assembly pattern
            mock_consultants = []
            for consultant in selection_result.selected_consultants:
                mock_consultants.append(
                    {
                        "consultant_id": consultant.consultant_id,
                        "persona_prompt": f"You are a {consultant.consultant_id} consultant. Provide strategic analysis from your specialized perspective.",
                    }
                )

            # Execute parallel analyses using Progressive Assembly pattern with N-Way Infusion
            analysis_tasks = []
            for consultant in mock_consultants:
                task = analyze_consultant_progressive(
                    llm_client,
                    pipeline_context_stream,
                    consultant,
                    enhanced_query.enhanced_statement,
                    selected_nway_clusters=selection_result.selected_nway_clusters,  # Level 3 Enhancement
                    research_brief=research_brief,
                )
                analysis_tasks.append(task)

            analysis_results = await asyncio.gather(
                *analysis_tasks, return_exceptions=True
            )

            # Process analysis results
            consultant_analyses = []
            successful_analyses = []
            for result in analysis_results:
                if isinstance(result, Exception):
                    logger.error(f"Analysis task failed: {result}")
                elif result and result.get("success", False):
                    analysis_model = ConsultantAnalysisModel(
                        consultant_id=result["consultant_id"],
                        analysis=result["analysis"],
                        tokens_used=result["tokens_used"],
                        processing_time_seconds=result["duration"],
                    )
                    consultant_analyses.append(analysis_model)
                    successful_analyses.append(result)

            # Pipeline Phase 2: Parallel Devil's Advocate Critiques
            if successful_analyses:
                logger.info(
                    "👹 Phase 2: Executing parallel Devil's Advocate critiques..."
                )

                critique_tasks = []
                for analysis in successful_analyses:
                    task = critique_analysis_progressive(
                        llm_client,
                        pipeline_context_stream,
                        analysis,
                        enhanced_query.enhanced_statement,
                        research_brief=research_brief,
                    )
                    critique_tasks.append(task)

                critique_results = await asyncio.gather(
                    *critique_tasks, return_exceptions=True
                )

                # Process critique results
                devils_advocate_critiques = []
                successful_critiques = []
                for result in critique_results:
                    if isinstance(result, Exception):
                        logger.error(f"Critique task failed: {result}")
                    elif result and result.get("success", False):
                        critique_model = DevilsAdvocateCritiqueModel(
                            consultant_id=result["consultant_id"],
                            critique=result["critique"],
                            tokens_used=result["tokens_used"],
                            processing_time_seconds=result["duration"],
                        )
                        devils_advocate_critiques.append(critique_model)
                        successful_critiques.append(result)

            # Pipeline Phase 3: Senior Advisor Rapporteur
            if successful_analyses and successful_critiques:
                logger.info("🎭 Phase 3: Executing Senior Advisor rapporteur...")

                # Prepare cognitive outputs for Senior Advisor
                cognitive_outputs = {
                    "analyses": successful_analyses,
                    "critiques": successful_critiques,
                }

                rapporteur_result = await senior_advisor_rapporteur_progressive(
                    llm_client,
                    pipeline_context_stream,
                    cognitive_outputs,
                    enhanced_query.enhanced_statement,
                    research_brief=research_brief,
                )

                if rapporteur_result and rapporteur_result.get("success", False):
                    senior_advisor_rapporteur = SeniorAdvisorModel(
                        rapporteur_analysis=rapporteur_result["rapporteur_analysis"],
                        tokens_used=rapporteur_result["tokens_used"],
                        context_preservation_score=1.0,  # Assume perfect preservation
                        processing_time_seconds=rapporteur_result["duration"],
                    )

            pipeline_end_time = time.time()
            pipeline_duration = pipeline_end_time - analysis_start_time

            # Pipeline metrics
            pipeline_metrics = {
                "total_pipeline_time_seconds": pipeline_duration,
                "phases_completed": (
                    3
                    if senior_advisor_rapporteur
                    else (2 if devils_advocate_critiques else 1)
                ),
                "total_analyses": (
                    len(consultant_analyses) if consultant_analyses else 0
                ),
                "total_critiques": (
                    len(devils_advocate_critiques) if devils_advocate_critiques else 0
                ),
                "senior_advisor_completed": senior_advisor_rapporteur is not None,
                "total_pipeline_tokens": sum(
                    [
                        (
                            sum(a.tokens_used for a in consultant_analyses)
                            if consultant_analyses
                            else 0
                        ),
                        (
                            sum(c.tokens_used for c in devils_advocate_critiques)
                            if devils_advocate_critiques
                            else 0
                        ),
                        (
                            senior_advisor_rapporteur.tokens_used
                            if senior_advisor_rapporteur
                            else 0
                        ),
                    ]
                ),
            }

            logger.info(
                f"🏆 Progressive Assembly pipeline completed in {pipeline_duration:.1f}s"
            )

        total_processing_time = int((time.time() - start_time) * 1000)

        # Create persistent engagement ID
        persistent_engagement_id = str(uuid.uuid4())

        # CRITICAL FIX: Persist results to database if pipeline was run
        if request.run_full_pipeline and consultant_analyses:
            await _persist_complete_analysis_results(
                persistent_engagement_id,
                request.problem_statement,
                enhanced_query,
                consultant_analyses,
                devils_advocate_critiques,
                senior_advisor_rapporteur,
            )

        return CompleteAnalysisResponse(
            success=True,
            engagement_id=persistent_engagement_id,
            socratic_phase={
                "questions_answered": len(request.user_responses),
                "quality_achieved": enhanced_query.quality_level,
                "confidence": enhanced_query.confidence_score,
            },
            enhanced_query=EnhanceQueryResponse(
                success=True,
                engagement_id=persistent_engagement_id,
                original_statement=enhanced_query.original_statement,
                enhanced_statement=enhanced_query.enhanced_statement,
                quality_level=enhanced_query.quality_level,
                confidence_score=enhanced_query.confidence_score,
                context_enrichment=enhanced_query.context_enrichment,
                processing_time_ms=0,  # Part of total time
            ),
            consultant_selection=consultant_selection,
            # Progressive Assembly Pipeline Results
            consultant_analyses=consultant_analyses,
            devils_advocate_critiques=devils_advocate_critiques,
            senior_advisor_rapporteur=senior_advisor_rapporteur,
            pipeline_metrics=pipeline_metrics,
            # Standard fields
            audit_trail=audit_trail[:20],  # Limit audit trail size
            total_processing_time_ms=total_processing_time,
        )

    except Exception as e:
        logger.error(f"❌ Complete Socratic analysis failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to complete Socratic analysis: {str(e)}"
        )


@router.get("/status")
async def get_forge_status():
    """Get status of the Socratic Cognitive Forge"""

    try:
        forge = get_forge()
        status = forge.get_forge_status()

        return {
            "success": True,
            "forge_status": status,
            "api_endpoints": [
                "/generate-questions",
                "/enhance-query",
                "/complete-analysis",
                "/status",
            ],
            "v5_features": [
                "Intelligent question generation by tier",
                "Context-aware problem analysis",
                "Progressive quality enhancement (60% → 85% → 95%)",
                "Query building from user responses",
                "V4 audit trail integration",
                "OptimalConsultantEngine integration",
            ],
        }

    except Exception as e:
        logger.error(f"❌ Status check failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get forge status: {str(e)}"
        )


# Progressive Assembly Pipeline Helper Functions
async def analyze_consultant_progressive(
    llm_client,
    context_stream,
    consultant,
    query,
    selected_nway_clusters=None,
    research_brief=None,
):
    """Progressive Assembly: Single consultant analysis task with N-Way Infusion"""
    from src.core.unified_context_stream import ContextEventType
    from datetime import datetime

    consultant_id = consultant["consultant_id"]
    persona_prompt = consultant["persona_prompt"]

    # N-Way Infusion: Inject cognitive directives into consultant prompt
    infused_persona_prompt = persona_prompt
    if selected_nway_clusters:
        try:
            # PROJECT LOLLAPALOOZA: Use the new Synergy Engine
            from src.engine.utils.nway_prompt_infuser_synergy_engine import (
                get_nway_synergy_engine,
            )
            from src.core.supabase_platform import MetisSupabasePlatform

            platform = MetisSupabasePlatform()
            synergy_engine = get_nway_synergy_engine(platform.supabase)

            # Apply SYNERGY ENGINE infusion to enhance the persona prompt
            infusion_result = (
                await synergy_engine.infuse_consultant_prompt_with_synergy_engine(
                    original_prompt=persona_prompt,
                    selected_nway_clusters=selected_nway_clusters,
                    consultant_id=consultant_id,
                )
            )

            if infusion_result.success:
                infused_persona_prompt = infusion_result.infused_prompt
                print(f"✅ N-Way cognitive directives injected for {consultant_id}")
            else:
                print(
                    f"⚠️ N-Way infusion failed for {consultant_id}: {infusion_result.error_message}"
                )
        except Exception as e:
            print(f"⚠️ N-Way infusion error for {consultant_id}: {e}")

    try:
        # Neutral Research Brief inclusion (if available) without contaminating perspectives
        rb_section = ""
        if research_brief and getattr(research_brief, "neutral_summary", None):
            rb_section = (
                f"\n\nNeutral Research Brief (shared context):\n{research_brief.neutral_summary[:800]}\n\nKey facts:\n- "
                + "\n- ".join(research_brief.key_facts[:5])
            )

        analysis_prompt = f"{infused_persona_prompt}\n\nQuery: {query}{rb_section}\n\nProvide a focused analysis from your perspective (150-200 words). Do not assume other consultants' conclusions; remain independent."

        context_stream.add_event(
            event_type=ContextEventType.REASONING_STEP,
            data={
                "action": "consultant_analysis_started",
                "consultant_id": consultant_id,
            },
            timestamp=datetime.now(),
        )

        start_time = time.time()
        messages = [{"role": "user", "content": analysis_prompt}]
        response = await asyncio.wait_for(
            llm_client.call_llm(
                messages=messages, model="deepseek-chat", max_tokens=250
            ),
            timeout=30.0,
        )

        duration = time.time() - start_time

        context_stream.add_event(
            event_type=ContextEventType.REASONING_STEP,
            data={
                "action": "consultant_analysis_completed",
                "consultant_id": consultant_id,
                "tokens": response.tokens_used,
            },
            timestamp=datetime.now(),
        )

        return {
            "consultant_id": consultant_id,
            "analysis": response.content,
            "tokens_used": response.tokens_used,
            "duration": duration,
            "success": True,
        }

    except Exception as e:
        return {
            "consultant_id": consultant_id,
            "analysis": f"Analysis failed: {str(e)}",
            "tokens_used": 0,
            "duration": 0.0,
            "success": False,
            "error": str(e),
        }


async def critique_analysis_progressive(
    llm_client, context_stream, analysis_data, query, research_brief=None
):
    """Progressive Assembly: Devil's Advocate critique task"""
    from src.core.unified_context_stream import ContextEventType
    from datetime import datetime

    consultant_id = analysis_data["consultant_id"]
    analysis_content = analysis_data["analysis"]

    try:
        rb_section = ""
        if research_brief and getattr(research_brief, "neutral_summary", None):
            rb_section = (
                f"\n\nNeutral Research Brief (shared context):\n{research_brief.neutral_summary[:600]}\n\nKey facts:\n- "
                + "\n- ".join(research_brief.key_facts[:4])
            )

        critique_prompt = f"""You are a Devil's Advocate challenger. Rigorously critique this consultant analysis.

Original Query: {query}{rb_section}

Consultant Analysis: {analysis_content}

Identify flaws, oversights, biases, and weaknesses. Challenge assumptions and highlight risks. Be constructive but rigorous (100-150 words):"""

        context_stream.add_event(
            event_type=ContextEventType.REASONING_STEP,
            data={"action": "devils_advocate_started", "consultant_id": consultant_id},
            timestamp=datetime.now(),
        )

        start_time = time.time()
        messages = [{"role": "user", "content": critique_prompt}]
        response = await asyncio.wait_for(
            llm_client.call_llm(
                messages=messages, model="deepseek-chat", max_tokens=200
            ),
            timeout=30.0,
        )

        duration = time.time() - start_time

        context_stream.add_event(
            event_type=ContextEventType.REASONING_STEP,
            data={
                "action": "devils_advocate_completed",
                "consultant_id": consultant_id,
                "tokens": response.tokens_used,
            },
            timestamp=datetime.now(),
        )

        return {
            "consultant_id": consultant_id,
            "critique": response.content,
            "tokens_used": response.tokens_used,
            "duration": duration,
            "success": True,
        }

    except Exception as e:
        return {
            "consultant_id": consultant_id,
            "critique": f"Critique failed: {str(e)}",
            "tokens_used": 0,
            "duration": 0.0,
            "success": False,
            "error": str(e),
        }


async def senior_advisor_rapporteur_progressive(
    llm_client, context_stream, cognitive_outputs, query, research_brief=None
):
    """Progressive Assembly: Senior Advisor rapporteur meta-analysis"""
    from src.core.unified_context_stream import ContextEventType
    from datetime import datetime

    try:
        # Construct comprehensive input for rapporteur
        analyses_section = "\n\n".join(
            [
                f"=== {a['consultant_id'].upper()} ANALYSIS ===\n{a['analysis']}"
                for a in cognitive_outputs["analyses"]
            ]
        )

        critiques_section = "\n\n".join(
            [
                f"=== DEVIL'S ADVOCATE CRITIQUE OF {c['consultant_id'].upper()} ===\n{c['critique']}"
                for c in cognitive_outputs["critiques"]
            ]
        )

        rb_section = ""
        if research_brief and getattr(research_brief, "neutral_summary", None):
            rb_section = (
                f"\n\nNeutral Research Brief (shared context):\n{research_brief.neutral_summary[:800]}\n\nKey facts:\n- "
                + "\n- ".join(research_brief.key_facts[:6])
            )

        rapporteur_prompt = f"""You are a Senior Advisor serving as Rapporteur. Provide meta-analysis WITHOUT synthesizing into a single recommendation.

Original Query: {query}{rb_section}

CONSULTANT ANALYSES:
{analyses_section}

DEVIL'S ADVOCATE CRITIQUES:
{critiques_section}

Your Task:
1. Assess QUALITY of each analysis
2. Evaluate how well critiques identified weaknesses  
3. Identify GAPS all consultants missed
4. Provide INDEPENDENT meta-perspective

CRITICAL: Do NOT synthesize. Preserve all consultant perspectives. Provide meta-analysis and quality assessment only.

Rapporteur assessment (200-250 words):"""

        context_stream.add_event(
            event_type=ContextEventType.REASONING_STEP,
            data={
                "action": "senior_advisor_started",
                "input_analyses": len(cognitive_outputs["analyses"]),
            },
            timestamp=datetime.now(),
        )

        start_time = time.time()
        messages = [{"role": "user", "content": rapporteur_prompt}]
        response = await asyncio.wait_for(
            llm_client.call_llm(
                messages=messages, model="deepseek-chat", max_tokens=350
            ),
            timeout=45.0,
        )

        duration = time.time() - start_time

        context_stream.add_event(
            event_type=ContextEventType.REASONING_STEP,
            data={"action": "senior_advisor_completed", "tokens": response.tokens_used},
            timestamp=datetime.now(),
        )

        return {
            "rapporteur_analysis": response.content,
            "tokens_used": response.tokens_used,
            "duration": duration,
            "success": True,
        }

    except Exception as e:
        return {
            "rapporteur_analysis": f"Rapporteur failed: {str(e)}",
            "tokens_used": 0,
            "duration": 0.0,
            "success": False,
            "error": str(e),
        }


async def _persist_complete_analysis_results(
    engagement_id: str,
    problem_statement: str,
    enhanced_query,
    consultant_analyses: List[ConsultantAnalysisModel],
    devils_advocate_critiques: List[DevilsAdvocateCritiqueModel],
    senior_advisor_rapporteur: Optional[SeniorAdvisorModel],
) -> str:
    """Persist complete analysis results to database - CRITICAL DATA PIPELINE FIX"""

    try:
        # Import Supabase client
        import os
        from supabase import create_client

        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_ANON_KEY")

        if not supabase_url or not supabase_key:
            logger.error("❌ Supabase configuration missing")
            return "FAILED: Supabase configuration missing"

        supabase = create_client(supabase_url, supabase_key)

        # Step 1: Create engagement record
        engagement_data = {
            "id": engagement_id,  # Use the UUID as primary key
            "problem_statement": problem_statement,
            "business_context": {
                "enhanced_query": enhanced_query.enhanced_statement,
                "quality_level": enhanced_query.quality_level,
                "confidence_score": enhanced_query.confidence_score,
                "context_enrichment": enhanced_query.context_enrichment,
            },
            "status": "completed",
            "metadata": {
                "socratic_forge_processing": True,
                "consultant_count": len(consultant_analyses),
                "pipeline_phases_completed": 3 if senior_advisor_rapporteur else 2,
            },
        }

        # Insert engagement record
        engagement_result = (
            supabase.table("engagements").insert(engagement_data).execute()
        )
        logger.info(f"💾 Created engagement record for {engagement_id}")

        # Step 2: Create consultant analysis records
        analysis_records_inserted = 0
        for analysis in consultant_analyses:
            analysis_data = {
                "engagement_id": engagement_id,
                "consultant_id": analysis.consultant_id,
                "analysis_output": analysis.analysis,
                "confidence_score": 0.8,  # Default confidence
                "processing_time_ms": int(analysis.processing_time_seconds * 1000),
                "tokens_used": analysis.tokens_used,
            }

            # Find matching critique
            matching_critique = next(
                (
                    c
                    for c in devils_advocate_critiques
                    if c.consultant_id == analysis.consultant_id
                ),
                None,
            )
            if matching_critique:
                analysis_data["devils_advocate_critique"] = matching_critique.critique

            try:
                supabase.table("engagement_results").insert(analysis_data).execute()
                analysis_records_inserted += 1
            except Exception as analysis_error:
                logger.warning(
                    f"⚠️ Failed to insert analysis for {analysis.consultant_id}: {analysis_error}"
                )

        # Step 3: Create Senior Advisor record (if available)
        senior_advisor_inserted = False
        if senior_advisor_rapporteur:
            senior_advisor_data = {
                "engagement_id": engagement_id,
                "meta_analysis_report": senior_advisor_rapporteur.rapporteur_analysis,
                "context_preservation_score": senior_advisor_rapporteur.context_preservation_score,
                "processing_time_ms": int(
                    senior_advisor_rapporteur.processing_time_seconds * 1000
                ),
                "tokens_used": senior_advisor_rapporteur.tokens_used,
                "consultant_comparisons": {},  # Could be enhanced later
                "decision_points": [],  # Could be enhanced later
            }

            try:
                supabase.table("senior_advisor_reports").insert(
                    senior_advisor_data
                ).execute()
                senior_advisor_inserted = True
                logger.info(f"💾 Created Senior Advisor report for {engagement_id}")
            except Exception as senior_error:
                logger.warning(
                    f"⚠️ Failed to insert Senior Advisor report: {senior_error}"
                )

        success_message = f"SUCCESS: Engagement={engagement_id}, Analyses={analysis_records_inserted}, SeniorAdvisor={'Yes' if senior_advisor_inserted else 'No'}"
        logger.info(f"✅ Complete analysis persisted: {success_message}")
        return success_message

    except Exception as e:
        error_message = f"FAILED: Complete analysis persistence error: {str(e)}"
        logger.error(f"❌ {error_message}")
        return error_message
