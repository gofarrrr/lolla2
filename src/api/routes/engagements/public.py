"""
Engagements API - Public Lifecycle Endpoints
=============================================

Public and authenticated endpoints for managing strategic analysis engagements
through the 8-stage pipeline orchestrator.

Operation Bedrock: Task 10.0 - API Decomposition
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple
from uuid import UUID, uuid4
from datetime import datetime

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Request, Header
from fastapi.responses import Response as FastAPIResponse

from src.core.stateful_pipeline_orchestrator import StatefulPipelineOrchestrator
from src.core.unified_context_stream import get_unified_context_stream
from src.services.persistence import DatabaseService, DatabaseOperationError
from src.services.orchestration_infra.supabase_checkpoint_repository import SupabaseCheckpointRepository
from src.services.orchestration_infra.revision_service import V1RevisionService
from src.core.checkpoint_service import CheckpointService
from src.api.auth import require_user

from .helpers import (
    load_context_stream_events,
    extract_final_output,
    generate_markdown_report,
    extract_consultant_selection_data,
    extract_human_interactions,
    extract_research_provider_events,
    extract_mece_framework,
    TraceSummary,
)
from .models import (
    StartEngagementRequest,
    EngagementStatusResponse,
    QuestionsResponse,
    SubmitAnswersRequest,
    OutcomeReportRequest,
    EngagementReportResponse,
    AnsweredQuestion,
    ResearchQuestion,
)

logger = logging.getLogger(__name__)

# Authenticated router for protected endpoints
router = APIRouter(
    prefix="/api/engagements",
    tags=["Engagements"],
    dependencies=[Depends(require_user)]
)

# Public router for unauthenticated endpoints (status polling, etc.)
public_router = APIRouter(
    prefix="/api/engagements",
    tags=["Engagements (Public)"]
)

# Global storage for active engagements (in production, use proper database)
active_engagements: Dict[str, Dict[str, Any]] = {}


# ============================================================================
# Helper Functions
# ============================================================================

def _generate_ux_metadata(questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate UX metadata for progressive disclosure based on question tiers.

    Organizes questions into 3-4-3 tiers and provides messaging for each tier.
    """
    # Group questions by tier
    tier_1 = [q for q in questions if q.get("tier") == "essential"]
    tier_2 = [q for q in questions if q.get("tier") == "strategic"]
    tier_3 = [q for q in questions if q.get("tier") == "expert"]

    return {
        "header": {
            "title": "Strategic Discovery Questions",
            "subtitle": "Stage 1 of 10-Stage Analysis Pipeline",
            "icon": "🎯",
            "description": "Surgical questions targeting YOUR specific context"
        },
        "tiers": {
            "tier_1": {
                "title": "Strategic Foundation",
                "subtitle": "Stage 1: Socratic Discovery",
                "badge": "Essential",
                "question_count": len(tier_1),
                "quality_impact": "50% baseline analysis",
                "unlock_threshold": 2,
                "unlock_message": "Answer 2 of 3 to unlock deeper analysis",
                "why": "Establishes foundation for 10-stage analysis. Without clear context, subsequent stages lack direction.",
                "next_stages": ["Problem Structuring", "Research", "Multi-Consultant Analysis"]
            },
            "tier_2": {
                "title": "Strategic Depth",
                "subtitle": "Stage 1: Deeper Context",
                "badge": "+30% Quality Boost",
                "question_count": len(tier_2),
                "quality_impact": "80% analysis depth (50% + 30%)",
                "unlock_threshold": len(tier_1) + 2,  # After tier 1
                "unlock_message": "Answer 4 more for richer strategic insights",
                "why": "Enables 9 downstream stages to produce personalized recommendations vs generic advice.",
                "next_stages": ["Consultant Selection", "Synergy Analysis", "Devil's Advocate"]
            },
            "tier_3": {
                "title": "Expert Mode",
                "subtitle": "Stage 1: Complete Context",
                "badge": "+20% Quality Boost",
                "question_count": len(tier_3),
                "quality_impact": "100% full strategic depth",
                "unlock_threshold": len(tier_1) + len(tier_2) + 2,
                "unlock_message": "Final 3 questions for 100% expert-level analysis",
                "why": "Maximizes value from Senior Advisor review and final arbitration stages.",
                "next_stages": ["Devil's Advocate", "Senior Advisor", "Final Strategic Report"]
            }
        },
        "methodology": {
            "title": "Why these questions drive better analysis",
            "points": [
                "10 Strategic Lenses: GOAL, CONSTRAINTS, STAKEHOLDERS, RISKS, OPTIONS, etc.",
                "Surgical Targeting: Asks about YOUR approach, not generic information",
                "Personalized Analysis: Powers 9 downstream stages with rich context",
                "Proven Framework: Research-validated 5x information value vs basic queries"
            ]
        },
        "quality_progress": {
            "0_answered": {"quality": 0, "label": "No analysis baseline", "message": "Generic recommendations (like ChatGPT)"},
            "tier_1_complete": {"quality": 50, "label": "50% baseline", "message": "Contextual analysis begins"},
            "tier_2_complete": {"quality": 80, "label": "80% depth", "message": "Personalized multi-consultant insights"},
            "tier_3_complete": {"quality": 100, "label": "100% expert", "message": "Full strategic depth across all stages"}
        },
        "pipeline_preview": [
            {"stage": 1, "name": "Socratic Questions", "status": "current", "icon": "🎯"},
            {"stage": 2, "name": "Problem Structuring", "status": "pending", "icon": "🏗️"},
            {"stage": 3, "name": "Data Research", "status": "pending", "icon": "🔬"},
            {"stage": 4, "name": "Consultant Selection", "status": "pending", "icon": "👥"},
            {"stage": 5, "name": "Parallel Analysis", "status": "pending", "icon": "⚡"},
            {"stage": 6, "name": "Devil's Advocate", "status": "pending", "icon": "🔍"},
            {"stage": 7, "name": "Senior Review", "status": "pending", "icon": "👔"},
            {"stage": 8, "name": "Final Report", "status": "pending", "icon": "📊"}
        ],
        "total_questions": len(questions)
    }


# ============================================================================
# Public Lifecycle Endpoints
# ============================================================================

@public_router.post("/start", response_model=Dict[str, str])
async def start_engagement(
    engagement_request: StartEngagementRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
) -> Dict[str, str]:
    """
    Start a new strategic analysis engagement

    This endpoint:
    1. Creates a new trace_id
    2. Initializes the StatefulPipelineOrchestrator
    3. Starts the 8-stage pipeline in the background
    4. Returns the trace_id immediately
    """
    try:
        # Generate new trace ID
        trace_id = str(uuid4())

        # Determine if this is an enhanced analysis (came from query enhancement flow)
        is_enhanced = bool(engagement_request.enhanced_context)
        quality_level = engagement_request.quality_requested or 60

        logger.info(f"🚀 Starting new engagement - Trace ID: {trace_id}")
        logger.info(f"📝 User Query: {engagement_request.user_query[:100]}...")

        # Log progressive questions enhancement
        if engagement_request.answered_questions or engagement_request.research_questions:
            answered_count = len(engagement_request.answered_questions or [])
            research_count = len(engagement_request.research_questions or [])
            logger.info(f"✨ Progressive Questions Enhancement:")
            logger.info(f"   📋 {answered_count} questions answered by user")
            logger.info(f"   🔍 {research_count} questions flagged for research")
            if engagement_request.quality_target:
                logger.info(f"   🎯 Target quality: {int(engagement_request.quality_target * 100)}%")

            # Log research questions to be answered
            if engagement_request.research_questions:
                logger.info("🔬 Research Questions to Answer During Analysis:")
                for idx, rq in enumerate(engagement_request.research_questions[:3], 1):  # Show first 3
                    logger.info(f"   {idx}. {rq.question_text}")
                if len(engagement_request.research_questions) > 3:
                    logger.info(f"   ... and {len(engagement_request.research_questions) - 3} more")

        if is_enhanced:
            logger.info(f"✨ Enhanced Analysis: {quality_level}% quality requested")
            enhanced_context = engagement_request.enhanced_context or {}
            answered_questions = len([
                v
                for v in enhanced_context.get('progressive_answers', {}).values()
                if isinstance(v, str) and v.strip()
            ])
            logger.info(f"📋 Progressive questions answered: {answered_questions}")

        # Initialize engagement tracking
        # Determine total stages via unified stage progress utility
        try:
            from src.core.stage_progress import total_stages_for_ui
            _total_stages = total_stages_for_ui()
        except Exception:
            _total_stages = 8

        active_engagements[trace_id] = {
            "trace_id": trace_id,
            "user_query": engagement_request.user_query,
            "user_id": engagement_request.user_id,
            "enhanced_context": engagement_request.enhanced_context,
            "quality_requested": quality_level,
            "is_enhanced": is_enhanced,
            "answered_questions": engagement_request.answered_questions,
            "research_questions": engagement_request.research_questions,
            "quality_target": engagement_request.quality_target,
            "enhancement_questions_session_id": engagement_request.enhancement_questions_session_id,
            "interactive_mode": engagement_request.interactive_mode,
            "started_at": datetime.now(),
            "status": "INITIALIZING",
            "current_stage": "SOCRATIC_QUESTIONS",
            "stage_number": 1,
            "total_stages": _total_stages,
            "progress_percentage": 0.0,
            "is_completed": False,
            "error": None,
            "final_output": None
        }

        database_service: Optional[DatabaseService] = getattr(
            http_request.app.state, "database_service", None
        )

        # Start the pipeline execution in background
        background_tasks.add_task(
            execute_pipeline_background,
            trace_id,
            engagement_request.user_query,
            engagement_request.user_id,
            engagement_request.enhanced_context,
            database_service,
            engagement_request.answered_questions,
            engagement_request.research_questions,
            engagement_request.quality_target,
        )

        logger.info(f"✅ Engagement started - Background pipeline execution initiated")

        return {
            "trace_id": trace_id,
            "status": "STARTED",
            "message": "Strategic analysis pipeline initiated"
        }

    except Exception as e:
        logger.error(f"❌ Failed to start engagement: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start engagement: {str(e)}")


@public_router.get("/{trace_id}/status", response_model=EngagementStatusResponse)
async def get_engagement_status(
    trace_id: str,
    request: Request,
    if_none_match: Optional[str] = Header(None, alias="If-None-Match")
) -> EngagementStatusResponse:
    """
    Get the current status of an engagement (PUBLIC - no auth required)

    Phase 1: Database-backed status with ETag support for efficient polling
    - Reads from database as single source of truth
    - Falls back to in-memory dict if DB read fails
    - Returns 304 Not Modified if client's ETag matches (no state change)
    - ETag format: W/"<trace_id>:<version>" for monotonic version tracking

    This endpoint provides real-time status updates for the processing page.
    IMPORTANT: This endpoint is on the public_router to allow unauthenticated polling
    for real-time status updates during pipeline execution.
    """
    try:
        # Phase 1: Try database first (single source of truth)
        database_service: Optional[DatabaseService] = getattr(
            request.app.state, "database_service", None
        )

        engagement = None
        db_version = None

        if database_service:
            try:
                db_status = await database_service.get_engagement_status_async(trace_id)
                if db_status:
                    engagement = db_status
                    db_version = db_status.get("version", 0)
                    logger.info(f"📊 DATABASE READ: status={engagement.get('status')}, version={db_version}, trace={trace_id[:8]}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to read from database, falling back to in-memory: {e}")

        # Fallback to in-memory dict (backward compatibility - Phase 1)
        if not engagement and trace_id in active_engagements:
            engagement = active_engagements[trace_id]
            # Use version=-1 for in-memory-only status (not yet in DB)
            # This ensures ETag changes when first written to DB (version=0)
            db_version = -1
            logger.debug(f"📊 In-memory status: {engagement.get('status')}")

        # Migration 009: Questions now in database, so merge is simpler
        # Hydrate cache from database if needed (for consistency)
        if engagement and trace_id not in active_engagements:
            logger.info(f"🔄 Hydrating cache from database (status endpoint) for trace {trace_id[:8]}")
            active_engagements[trace_id] = {
                "trace_id": trace_id,
                "status": engagement.get("status"),
                "current_stage": engagement.get("current_stage"),
                "stage_number": engagement.get("stage_number"),
                "progress_percentage": engagement.get("progress_percentage"),
                "generated_questions": engagement.get("generated_questions"),
                "answered_questions": engagement.get("answered_questions"),
                "research_questions": engagement.get("research_questions"),
                "paused_checkpoint_id": engagement.get("paused_checkpoint_id"),
                "enhancement_context": engagement.get("enhancement_context", {}),
            }
        # Legacy merge: in-memory overrides database (for backward compatibility during transition)
        elif engagement and trace_id in active_engagements:
            in_memory = active_engagements[trace_id]
            # Only override if in-memory has newer data (rare case)
            if "generated_questions" in in_memory and not engagement.get("generated_questions"):
                engagement["generated_questions"] = in_memory["generated_questions"]
                logger.debug(f"📊 MERGE: Using in-memory questions as DB was empty, trace={trace_id[:8]}")

        if not engagement:
            raise HTTPException(status_code=404, detail="Engagement not found")

        # Generate ETag: W/"<trace_id>:<version>" for monotonic versioning
        current_etag = f'W/"{trace_id}:{db_version or 0}"'

        # Check If-None-Match for conditional GET (304 Not Modified)
        if if_none_match and if_none_match == current_etag:
            logger.debug(f"🔄 ETag match - returning 304 Not Modified")
            return FastAPIResponse(status_code=304, headers={"ETag": current_etag})

        # Stage descriptions for user-friendly display
        stage_descriptions = {
            "SOCRATIC_QUESTIONS": "Generating clarifying questions",
            "PROBLEM_STRUCTURING": "Structuring the problem framework",
            "CONSULTANT_SELECTION": "Selecting expert consultants",
            "SYNERGY_PROMPTING": "Coordinating consultant synergies",
            "PARALLEL_ANALYSIS": "Running parallel analysis",
            "DEVILS_ADVOCATE": "Challenging assumptions and testing logic",
            "SENIOR_ADVISOR": "Generating executive recommendations",
            "COMPLETED": "Analysis complete"
        }

        response = EngagementStatusResponse(
            trace_id=engagement.get("trace_id", trace_id),
            status=engagement.get("status", "UNKNOWN"),
            current_stage=engagement.get("current_stage"),
            stage_number=engagement.get("stage_number", 0),
            total_stages=engagement.get("total_stages", 7),
            progress_percentage=engagement.get("progress_percentage", 0.0),
            stage_description=stage_descriptions.get(
                engagement.get("current_stage"),
                "Processing..."
            ),
            is_completed=engagement.get("is_completed", False),
            error=engagement.get("error")
        )

        # Set ETag header for efficient polling
        response_headers = {"ETag": current_etag}
        return FastAPIResponse(
            content=response.model_dump_json(),
            media_type="application/json",
            headers=response_headers
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get engagement status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@router.get("/{trace_id}/questions", response_model=QuestionsResponse)
async def get_generated_questions(trace_id: str, http_request: Request) -> QuestionsResponse:
    """
    Get the generated questions for user to answer (interactive mode)

    This endpoint is used when the pipeline pauses after generating Socratic questions
    and is waiting for the user to provide answers.

    Migration 009: DB-first with cache hydration
    - Reads from database first (survives backend restarts)
    - Hydrates active_engagements cache if empty
    - Maintains backward compatibility with in-memory fallback
    """
    try:
        database_service: Optional[DatabaseService] = getattr(
            http_request.app.state, "database_service", None
        )

        # Migration 009: DB-first approach - try database first
        engagement = None
        if database_service:
            try:
                db_engagement = await database_service.get_engagement_status_async(trace_id)
                if db_engagement:
                    # Hydrate cache from database if not present
                    if trace_id not in active_engagements:
                        logger.info(f"🔄 Hydrating cache from database for trace {trace_id}")
                        active_engagements[trace_id] = {
                            "trace_id": trace_id,
                            "status": db_engagement.get("status"),
                            "current_stage": db_engagement.get("current_stage"),
                            "stage_number": db_engagement.get("stage_number"),
                            "progress_percentage": db_engagement.get("progress_percentage"),
                            "generated_questions": db_engagement.get("generated_questions"),
                            "paused_checkpoint_id": db_engagement.get("paused_checkpoint_id"),
                            "enhancement_context": db_engagement.get("enhancement_context", {}),
                            "interactive_mode": db_engagement.get("enhancement_context", {}).get("interactive_mode", True),
                            "user_query": db_engagement.get("enhancement_context", {}).get("user_query", ""),
                        }
                    engagement = active_engagements[trace_id]
            except Exception as e:
                logger.warning(f"⚠️ Failed to load from database: {e}")

        # Fallback to in-memory only if DB read failed
        if not engagement:
            if trace_id not in active_engagements:
                raise HTTPException(status_code=404, detail="Engagement not found")
            engagement = active_engagements[trace_id]

        # Check if this is an interactive mode engagement
        if not engagement.get("interactive_mode"):
            raise HTTPException(
                status_code=400,
                detail="This engagement is not in interactive mode"
            )

        # Check if the engagement is paused for user input
        status = engagement.get("status")
        if status != "PAUSED_FOR_USER_INPUT":
            raise HTTPException(
                status_code=400,
                detail=f"Engagement is not paused for user input. Current status: {status}"
            )

        # Get questions from the engagement state
        # The questions should be in the progressive_questions result
        questions = engagement.get("generated_questions", [])
        checkpoint_id = engagement.get("paused_checkpoint_id")

        # If questions not in engagement state, try to get from checkpoint
        if not questions:
            database_service: Optional[DatabaseService] = getattr(
                http_request.app.state, "database_service", None
            )

            # Try to load from checkpoint
            try:
                context_stream = get_unified_context_stream()
                db_service = http_request.app.state.database_service
                repo = SupabaseCheckpointRepository(db_service, context_stream)
                rev = V1RevisionService(repo, context_stream)
                checkpoint_service = CheckpointService(checkpoint_repo=repo, revision_service=rev)

                if checkpoint_id:
                    checkpoint = await checkpoint_service.load_checkpoint(UUID(checkpoint_id))
                    if checkpoint and checkpoint.stage_output:
                        # V6 MIGRATION: After resume fix, checkpoint.stage_output IS the dict with V6 keys
                        # For socratic_questions checkpoints, stage_output contains key_strategic_questions directly
                        if isinstance(checkpoint.stage_output, dict):
                            # Try direct access first (V6 resume fix structure)
                            questions = checkpoint.stage_output.get("key_strategic_questions", [])

                            # Fallback: try nested structure (legacy)
                            if not questions:
                                socratic_questions = checkpoint.stage_output.get("socratic_questions", {})
                                questions = socratic_questions.get("key_strategic_questions", [])

                            logger.info(f"🔍 V6: Extracted {len(questions)} questions from checkpoint (keys: {list(checkpoint.stage_output.keys())[:5]})")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load questions from checkpoint: {e}")

        # Generate UX metadata for progressive disclosure
        ux_metadata = _generate_ux_metadata(questions)

        return QuestionsResponse(
            trace_id=trace_id,
            questions=questions,
            status="awaiting_answers",
            checkpoint_id=checkpoint_id,
            ux_metadata=ux_metadata
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get generated questions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get questions: {str(e)}")


@router.post("/{trace_id}/submit-answers", response_model=Dict[str, Any])
async def submit_answers(
    trace_id: str,
    request: SubmitAnswersRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
) -> Dict[str, Any]:
    """
    Submit user answers and resume pipeline execution

    This endpoint:
    1. Updates the engagement with user answers
    2. Resumes pipeline execution from the checkpoint
    3. Returns immediately while pipeline continues in background
    """
    try:
        # Check if engagement exists
        if trace_id not in active_engagements:
            raise HTTPException(status_code=404, detail="Engagement not found")

        engagement = active_engagements[trace_id]

        # Check if this is an interactive mode engagement
        if not engagement.get("interactive_mode"):
            raise HTTPException(
                status_code=400,
                detail="This engagement is not in interactive mode"
            )

        # Check if the engagement is paused for user input
        status = engagement.get("status")
        if status != "PAUSED_FOR_USER_INPUT":
            raise HTTPException(
                status_code=400,
                detail=f"Engagement is not paused for user input. Current status: {status}"
            )

        logger.info(f"📝 Received {len(request.answers)} answers for trace {trace_id}")

        # Separate answered vs. research-delegated questions
        RAW_RESEARCH_TOKEN = "[REQUEST_RESEARCH]"
        answered_questions = []
        research_questions = []

        # Build a lookup of generated questions to recover question_text for research items
        generated = engagement.get("generated_questions", []) or []
        q_by_id = {str(q.get("id")): q for q in generated if isinstance(q, dict) and q.get("id")}

        for ans in request.answers:
            if (ans.answer or "").strip() == RAW_RESEARCH_TOKEN:
                q = q_by_id.get(str(ans.question_id))
                q_text = (q.get("question") if isinstance(q, dict) else None) or ""
                research_questions.append({
                    "question_id": ans.question_id,
                    "question_text": q_text,
                })
            else:
                # Only include non-empty user answers
                if (ans.answer or "").strip():
                    answered_questions.append({
                        "question_id": ans.question_id,
                        "answer": ans.answer,
                    })

        # Update engagement with answers and research requests (for traceability)
        engagement["user_answers"] = answered_questions
        engagement["user_research_requests"] = research_questions
        engagement["status"] = "RESUMING"

        # Get checkpoint info
        checkpoint_id = engagement.get("paused_checkpoint_id")
        if not checkpoint_id:
            raise HTTPException(
                status_code=500,
                detail="No checkpoint found to resume from"
            )

        database_service: Optional[DatabaseService] = getattr(
            http_request.app.state, "database_service", None
        )

        # Phase 1: Atomically write RESUMING status to database WITH ANSWERS (Migration 009)
        if database_service:
            try:
                await database_service.upsert_engagement_status_async({
                    "trace_id": trace_id,
                    "status": "RESUMING",
                    "current_stage": engagement.get("current_stage"),
                    "stage_number": engagement.get("stage_number"),
                    "progress_percentage": engagement.get("progress_percentage", 0.0),
                    "user_id": engagement.get("user_id"),
                    "session_id": engagement.get("session_id"),
                    # Migration 009: Persist answers to database
                    "answered_questions": answered_questions,
                    "research_questions": research_questions,
                })
                logger.info(f"✅ Database updated with RESUMING status + {len(answered_questions)} answers, {len(research_questions)} research requests")
            except Exception as e:
                logger.error(f"⚠️ Failed to update database status (non-blocking): {e}")
                # Non-blocking - don't fail the request if DB write fails

        # Resume pipeline in background (pass both answered and research questions)
        background_tasks.add_task(
            resume_pipeline_with_answers,
            trace_id,
            checkpoint_id,
            answered_questions,
            research_questions,
            engagement.get("user_id"),
            database_service,
        )

        logger.info(f"✅ Pipeline resumption initiated for trace {trace_id}")

        return {
            "trace_id": trace_id,
            "status": "resumed",
            "message": "Pipeline execution resumed with user answers",
            "answers_count": len(request.answers),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to submit answers: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit answers: {str(e)}")


def flatten_report_for_frontend(report_data: Dict[str, Any], trace_id: str) -> Dict[str, Any]:
    """
    Flatten nested report structure for frontend compatibility.

    Delegates to ReportFormattingService for business logic.
    Complexity: CC = 1
    """
    from src.services.engagement import ReportFormattingService
    service = ReportFormattingService()
    return service.flatten_report(report_data, trace_id)


def _load_final_output_from_database(
    database_service: Optional[DatabaseService],
    trace_id: str
) -> Tuple[Optional[Dict[str, Any]], Optional[datetime], Optional[Dict[str, Any]]]:
    """Load final output from database. Returns (final_output, generated_at, record). CC=5"""
    if not database_service:
        return None, None, None

    try:
        record = database_service.get_engagement_report(trace_id)
        logger.info(f"🔍 DATABASE: Got record from database: {bool(record)}")

        if not record or not record.get("final_report_json"):
            return None, None, record

        report_payload = record["final_report_json"]
        final_output = report_payload.get("final_report_contract") or report_payload
        logger.info(f"🔍 DATABASE: Extracted final_output, has parallel_analysis: {bool(final_output.get('parallel_analysis'))}")

        generated_at = datetime.now()
        completed_at = record.get("completed_at")
        if isinstance(completed_at, str):
            try:
                generated_at = datetime.fromisoformat(completed_at)
            except ValueError:
                pass
        elif isinstance(completed_at, datetime):
            generated_at = completed_at

        return final_output, generated_at, record
    except DatabaseOperationError as exc:
        logger.warning(f"⚠️ Failed to fetch engagement report from database: {exc}")
        return None, None, None


def _enrich_final_output_from_events(
    final_output: Dict[str, Any],
    trace_id: str,
    database_service: Optional[DatabaseService]
) -> Tuple[Dict[str, Any], Optional[List]]:
    """Enrich final output from context stream events. Returns (final_output, events_cached). CC=7"""
    enrichable_fields = [
        'parallel_analysis', 'strategic_recommendations', 'da_transcript',
        'assumption_diff', 'evidence', 'enhancement_research_answers', 'stage_profiles',
        'socratic_questions', 'consultant_selection_methodology', 'research_grounding'
    ]

    missing_fields = [f for f in enrichable_fields if not final_output.get(f)]
    if not missing_fields:
        return final_output, None

    logger.info(f"🔍 ENRICH: final_output lacks {len(missing_fields)} fields, loading from context_stream for {trace_id}")

    try:
        events_cached = load_context_stream_events(trace_id)
        if not events_cached:
            return final_output, None

        enriched_data = extract_final_output(events_cached)
        if enriched_data:
            for key, value in enriched_data.items():
                if key not in final_output or final_output.get(key) is None:
                    final_output[key] = value

            consultant_count = len(enriched_data.get('parallel_analysis', {}).get('consultant_analyses', []))
            logger.info(f"✅ Enriched final_output with {len(enriched_data)} fields ({consultant_count} consultants)")

            # Checkpoint fallback if no consultants
            if consultant_count == 0 and database_service:
                _try_checkpoint_fallback(final_output, trace_id, database_service)

        return final_output, events_cached
    except Exception as e:
        logger.warning(f"⚠️ Failed to enrich final_output: {e}")
        return final_output, None


def _try_checkpoint_fallback(final_output: Dict[str, Any], trace_id: str, database_service: DatabaseService) -> None:
    """Try checkpoint reconstruction as fallback. CC=4"""
    try:
        from src.services.report_reconstruction_service import ReportReconstructionService
        reconstruction_svc = ReportReconstructionService(database_service)
        checkpoint_bundle = reconstruction_svc.reconstruct_bundle(trace_id)

        checkpoint_consultants = checkpoint_bundle.get('consultant_analyses', [])
        if checkpoint_consultants:
            if 'parallel_analysis' not in final_output:
                final_output['parallel_analysis'] = {}
            final_output['parallel_analysis']['consultant_analyses'] = checkpoint_consultants
            final_output['consultant_selection'] = checkpoint_bundle.get('consultant_selection', {})
            logger.info(f"✅ Added {len(checkpoint_consultants)} consultants from checkpoint reconstruction")
        else:
            logger.warning(f"⚠️ Checkpoint reconstruction also found 0 consultants for {trace_id}")
    except Exception as checkpoint_err:
        logger.warning(f"⚠️ Checkpoint fallback failed: {checkpoint_err}")


def _extract_transparency_data(
    trace_id: str,
    events_cached: Optional[List],
    engagement_state: Optional[Dict[str, Any]]
) -> Tuple[Optional[Dict], Optional[Dict], Optional[List], Optional[Dict]]:
    """Extract transparency data from events. Returns (consultant_data, human_interactions, research_provider, mece). CC=3"""
    try:
        if events_cached is None:
            events_cached = load_context_stream_events(trace_id)
        if not events_cached:
            return None, None, None, None

        consultant_data = extract_consultant_selection_data(events_cached)
        human_interactions_data = extract_human_interactions(events_cached, engagement_state or {})
        research_provider_data = extract_research_provider_events(events_cached)
        mece_framework_data = extract_mece_framework(events_cached)

        return consultant_data, human_interactions_data, research_provider_data, mece_framework_data
    except Exception as e:
        logger.warning(f"⚠️ Failed to extract transparency data for {trace_id}: {e}")
        return None, None, None, None


# DEPRECATED: Use /api/v2/engagements/{trace_id}/bundle instead
def _load_final_output_with_all_fallbacks(
    database_service: Optional[DatabaseService],
    trace_id: str,
    engagement_state: Optional[Dict[str, Any]]
) -> Tuple[Dict[str, Any], datetime, Optional[Dict], Optional[List]]:
    """Load final output using all available fallback mechanisms. Returns (final_output, generated_at, record, events_cached). CC=7"""
    # Try database first
    final_output, generated_at, record = _load_final_output_from_database(database_service, trace_id)

    # Fallback to in-memory
    if final_output is None and engagement_state:
        final_output = engagement_state.get("final_output")
        generated_at = engagement_state.get("completed_at", datetime.now())

    # Enrich from context stream events
    events_cached = None
    if final_output:
        final_output, events_cached = _enrich_final_output_from_events(final_output, trace_id, database_service)
        return final_output, generated_at, record, events_cached

    # Final fallback: reconstruct from persisted context stream
    try:
        events_cached = load_context_stream_events(trace_id)
        if events_cached:
            final_output = extract_final_output(events_cached)
            logger.info(f"✅ Reconstructed final_output from persisted context_stream for trace {trace_id}")
            # Extract timestamp from events
            for event_dict in reversed(events_cached):
                if isinstance(event_dict, dict) and 'timestamp' in event_dict:
                    try:
                        generated_at = datetime.fromisoformat(event_dict['timestamp'])
                        break
                    except (ValueError, TypeError):
                        pass
    except Exception as e:
        logger.warning(f"⚠️ Failed to reconstruct final_output from persisted context_stream: {e}")

    return final_output, generated_at, record, events_cached


def _build_engagement_report_response(
    trace_id: str,
    final_output: Dict[str, Any],
    generated_at: datetime,
    markdown_content: str,
    consultant_data: Optional[Dict],
    human_interactions_data: Optional[Dict],
    research_provider_data: Optional[List],
    mece_framework_data: Optional[Dict],
    user_query: Optional[str],
    flattened_output: Dict[str, Any]
) -> EngagementReportResponse:
    """Build the EngagementReportResponse object. CC=1"""
    return EngagementReportResponse(
        trace_id=trace_id,
        report=flattened_output,
        generated_at=generated_at,
        markdown_content=markdown_content,
        selected_consultants=consultant_data.get("selected_consultants", []) if consultant_data else [],
        consultant_selection_methodology=consultant_data.get("methodology", None) if consultant_data else None,
        human_interactions=human_interactions_data,
        research_provider_events=research_provider_data,
        mece_framework=mece_framework_data,
        user_query=user_query,
        consultant_analyses=flattened_output.get("consultant_analyses", []),
        strategic_recommendations=flattened_output.get("strategic_recommendations", []),
        executive_summary=flattened_output.get("executive_summary"),
        devils_advocate_transcript=flattened_output.get("devils_advocate_transcript"),
        quality_metrics=flattened_output.get("quality_metrics"),
        key_decisions=flattened_output.get("key_decisions", []),
        da_transcript=flattened_output.get("da_transcript"),
    )


def _validate_engagement_state(engagement_state: Optional[Dict[str, Any]]):
    """Validate engagement state and raise appropriate exceptions. CC=3"""
    if engagement_state and engagement_state.get("error"):
        raise HTTPException(status_code=500, detail=f"Engagement failed: {engagement_state['error']}")
    if engagement_state and not engagement_state.get("is_completed"):
        raise HTTPException(status_code=400, detail="Engagement not yet completed")


def _extract_user_query(record: Optional[Dict], engagement_state: Optional[Dict]) -> Optional[str]:
    """Extract user query from record or engagement state. CC=2"""
    if record and "user_query" in record:
        return record["user_query"]
    if engagement_state and "user_query" in engagement_state:
        return engagement_state["user_query"]
    return None


@public_router.get("/{trace_id}/report", response_model=EngagementReportResponse, deprecated=True)
async def get_engagement_report(trace_id: str, http_request: Request) -> EngagementReportResponse:
    """
    Get the final strategic report for a completed engagement.

    Refactored with helper methods to reduce complexity. CC ≤ 6
    """
    logger.info(f"🔍 REPORT ENDPOINT: Called for trace_id={trace_id}")
    try:
        engagement_state = active_engagements.get(trace_id)
        database_service: Optional[DatabaseService] = getattr(http_request.app.state, "database_service", None)

        # Validate engagement state
        _validate_engagement_state(engagement_state)

        # Load final output with all fallback mechanisms
        final_output, generated_at, record, events_cached = \
            _load_final_output_with_all_fallbacks(database_service, trace_id, engagement_state)

        # Raise 404 if all fallback mechanisms failed
        if final_output is None:
            raise HTTPException(status_code=404, detail="Engagement not found or incomplete")

        # Generate markdown and extract transparency data
        markdown_content = generate_markdown_report(final_output)
        consultant_data, human_interactions_data, research_provider_data, mece_framework_data = \
            _extract_transparency_data(trace_id, events_cached, engagement_state)

        # Extract user query and flatten output
        user_query = _extract_user_query(record, engagement_state)
        flattened_output = flatten_report_for_frontend(final_output, trace_id)

        return _build_engagement_report_response(
            trace_id, final_output, generated_at, markdown_content,
            consultant_data, human_interactions_data, research_provider_data,
            mece_framework_data, user_query, flattened_output
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get engagement report: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get report: {str(e)}")


@router.post("/{trace_id}/outcome", response_model=Dict[str, Any])
async def report_outcome(trace_id: str, payload: OutcomeReportRequest, http_request: Request) -> Dict[str, Any]:
    """Record the final outcome for an engagement to close the calibration loop."""
    try:
        # Normalize to float outcome
        v = (payload.outcome or "").strip().lower()
        if v in ("yes", "y", "true", "success", "pass"):
            outcome = 1.0
        elif v in ("no", "n", "false", "fail", "failure"):
            outcome = 0.0
        elif v in ("too early", "early", "unknown", "na"):
            return {"trace_id": trace_id, "status": "ignored", "reason": "too_early"}
        else:
            try:
                outcome = float(v)
                if not (0.0 <= outcome <= 1.0):
                    raise ValueError("outcome out of range")
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid outcome value")

        # Update calibration
        from src.engine.calibration.calibration_service import get_calibration_service
        db_service = getattr(http_request.app.state, "database_service", None)
        count = await get_calibration_service(db_service).report_outcome(
            trace_id=trace_id, outcome=outcome, context={"notes": payload.notes or ""}
        )

        return {"trace_id": trace_id, "observations_recorded": count}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to record outcome for {trace_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to record outcome")


def _process_stage0_event(event_value: str, event_data: Dict, stages: List[str]) -> Optional[Dict]:
    """Process Stage 0 enrichment events. CC=3"""
    if "depth_enrichment" not in event_value.lower() and "stage0" not in event_value.lower() and "stage_0" not in event_value.lower():
        return None

    if "Stage 0 Enrichment" not in stages:
        stages.append("Stage 0 Enrichment")

    return {
        "enabled": True,
        "consultant_count": event_data.get("consultant_count", 0),
        "latency_ms": event_data.get("latency_ms", 0),
        "total_token_estimate": event_data.get("total_token_estimate", 0),
        "variant": event_data.get("variant", "unknown"),
    }


def _process_oracle_event(event_value: str, event_data: Dict) -> Tuple[Optional[Dict], List]:
    """Process Oracle research events and extract citations. CC=5"""
    if event_value != "oracle_research_complete":
        return None, []

    oracle_brief = {
        "status": event_data.get("status"),
        "web_findings_count": event_data.get("web_findings_count"),
        "internal_context_count": event_data.get("internal_context_count"),
        "citations_count": event_data.get("citations_count"),
        "quality_indicator": event_data.get("quality_indicator"),
        "gpa_scores": event_data.get("gpa_scores"),
    }

    # Extract citations
    from .helpers import TraceCitation
    citations = []
    raw_citations = event_data.get("citations") or []
    for c in raw_citations:
        try:
            url = c.get("url") or c.get("uri") or c.get("href")
            if not url:
                continue
            title = c.get("title") or None
            domain = None
            try:
                from urllib.parse import urlparse
                domain = urlparse(url).hostname
            except Exception:
                pass
            citations.append(TraceCitation(url=url, title=title, domain=domain))
        except Exception:
            continue

    return oracle_brief, citations


def _process_devils_advocate_event(event_value: str, event_data: Dict) -> List[Dict]:
    """Process devils advocate events. CC=4"""
    if not (event_value.startswith("devils_advocate") or event_value == "devils_advocate_complete"):
        return []

    advocate_items = []
    challenges = event_data.get("adversarial_challenges") or []

    for ch in challenges[:5]:
        advocate_items.append({
            "type": "challenge",
            "severity": ch.get("severity"),
            "text": (ch.get("challenge_text") or ch.get("text") or "").strip()[:240],
        })

    if not challenges:
        entry = {k: event_data.get(k) for k in ("challenges_count", "critique_strength", "robustness_score") if k in event_data}
        if entry:
            advocate_items.append({"type": "summary", **entry})

    return advocate_items


def _process_events_for_trace(events: List) -> Tuple[List[str], Optional[Dict], Optional[Dict], List[Dict], List, List[Dict]]:
    """Process events and extract trace data. Returns (stages, oracle_brief, stage0_brief, recent, citations, advocate_items). CC=7"""
    stages = []
    oracle_brief = None
    stage0_brief = None
    recent = []
    citations: list = []
    advocate_items: list = []

    for e in events[-50:]:
        recent.append({
            "event_type": e.event_type.value if hasattr(e, "event_type") else str(e.get("event_type")),
            "timestamp": e.timestamp.isoformat() if hasattr(e, "timestamp") else None,
            "data_keys": list(getattr(e, "data", {}) or {}.keys()),
        })

        et = getattr(e, "event_type", None)
        if not et:
            continue

        val = et.value
        data = getattr(e, "data", {}) or {}

        # Track stages
        if val not in stages and "stage" in data:
            stages.append(val)

        # Process special event types
        if stage0_brief is None:
            stage0_brief = _process_stage0_event(val, data, stages)

        if oracle_brief is None:
            oracle_brief, new_citations = _process_oracle_event(val, data)
            citations.extend(new_citations)

        advocate_items.extend(_process_devils_advocate_event(val, data))

    return stages, oracle_brief, stage0_brief, recent, citations, advocate_items


@router.get("/{trace_id}/trace", response_model=TraceSummary)
async def get_engagement_trace(trace_id: str) -> TraceSummary:
    """
    Return a PII-safe, summarized view of the UnifiedContextStream for this trace.

    Refactored with helper methods to reduce complexity. CC ≤ 5
    """
    try:
        if trace_id not in active_engagements:
            raise HTTPException(status_code=404, detail="Engagement not found")

        context_stream = get_unified_context_stream()
        events = context_stream.get_events() if context_stream else []

        # Process events
        stages, oracle_brief, stage0_brief, recent, citations, advocate_items = _process_events_for_trace(events)

        # Extract Glass Box UI data
        engagement_data = active_engagements.get(trace_id, {})
        final_output = engagement_data.get("final_output", {})

        return TraceSummary(
            trace_id=trace_id,
            event_count=len(events),
            stages_detected=stages,
            oracle_briefing=oracle_brief,
            stage0_enrichment=stage0_brief,
            recent_events=recent,
            citations=citations,
            devils_advocate=advocate_items,
            orthogonality_index=final_output.get("orthogonality_index"),
            stage_profiles=final_output.get("stage_profiles", []),
            da_transcript=final_output.get("da_transcript"),
            assumption_diff=final_output.get("assumption_diff", []),
            evidence=final_output.get("evidence", []),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve trace: {e}")


# ============================================================================
# Background Execution Helpers
# ============================================================================

def _build_enhancement_context_from_questions(
    answered_questions: Optional[list],
    research_questions: Optional[list],
    quality_target: Optional[float]
) -> Dict[str, Any]:
    """Build enhancement context from progressive questions. CC=4"""
    if not answered_questions and not research_questions:
        return {}

    context = {
        "answered_questions": [
            {"question_id": aq.question_id, "question_text": aq.question_text, "answer": aq.answer}
            for aq in (answered_questions or [])
        ],
        "research_questions": [
            {"question_id": rq.question_id, "question_text": rq.question_text}
            for rq in (research_questions or [])
        ],
        "quality_target": quality_target,
    }

    logger.info(f"✨ Enhancement context created:")
    logger.info(f"   📋 {len(answered_questions or [])} questions answered by user")
    logger.info(f"   🔍 {len(research_questions or [])} questions flagged for research")
    if quality_target:
        logger.info(f"   🎯 Target quality: {int(quality_target * 100)}%")

    return context


def _initialize_pipeline_services(database_service: Optional[DatabaseService]):
    """Initialize checkpoint service and orchestrator. Returns (db_service, checkpoint_service). CC=3"""
    context_stream = get_unified_context_stream()
    db_service = database_service or DatabaseService()
    repo = SupabaseCheckpointRepository(db_service, context_stream)
    rev = V1RevisionService(repo, context_stream)
    checkpoint_service = CheckpointService(checkpoint_repo=repo, revision_service=rev)
    return db_service, checkpoint_service


def _create_status_update_callback(trace_id: str, db_service):
    """Create status update callback for pipeline. CC=4"""
    def update_frontend_status(stage_name: str, stage_num: int, progress: float):
        """Callback function to update frontend status as pipeline progresses"""
        active_engagements[trace_id].update({
            "current_stage": stage_name,
            "stage_number": stage_num,
            "progress_percentage": progress,
            "status": "RUNNING"
        })
        logger.info(f"📡 Frontend status updated: Stage {stage_num}/7 - {stage_name} ({progress}%)")

        try:
            engagement = active_engagements.get(trace_id, {})
            db_service.upsert_engagement_status({
                "trace_id": trace_id,
                "status": "RUNNING",
                "current_stage": stage_name,
                "stage_number": stage_num,
                "progress_percentage": progress,
                "user_id": engagement.get("user_id"),
                "session_id": engagement.get("session_id"),
            })
        except Exception as e:
            logger.warning(f"⚠️ Failed to update database status (non-blocking): {e}")

    return update_frontend_status


async def _handle_paused_pipeline(trace_id: str, pipeline_state: Dict, database_service: Optional[DatabaseService]) -> bool:
    """Handle paused pipeline state. Returns True if paused, False otherwise. CC=6"""
    if not (isinstance(pipeline_state, dict) and pipeline_state.get("status") == "paused_for_user_input"):
        return False

    logger.info(f"⏸️ Pipeline paused for user input - Trace ID: {trace_id}")

    # Extract questions
    socratic_questions = pipeline_state.get("socratic_questions", {})
    questions = socratic_questions.get("key_strategic_questions", [])
    if not questions:
        questions = pipeline_state.get("questions", [])

    # Update in-memory state
    active_engagements[trace_id].update({
        "status": "PAUSED_FOR_USER_INPUT",
        "paused_checkpoint_id": str(pipeline_state.get("current_checkpoint")),
        "generated_questions": questions,
        "is_completed": False,
    })

    # Persist to database
    if database_service:
        try:
            engagement = active_engagements.get(trace_id, {})
            await database_service.upsert_engagement_status_async({
                "trace_id": trace_id,
                "status": "PAUSED_FOR_USER_INPUT",
                "current_stage": engagement.get("current_stage"),
                "stage_number": engagement.get("stage_number"),
                "progress_percentage": engagement.get("progress_percentage", 0.0),
                "user_id": engagement.get("user_id"),
                "session_id": engagement.get("session_id"),
                "generated_questions": questions,
                "paused_checkpoint_id": str(pipeline_state.get("current_checkpoint")),
                "enhancement_context": {
                    "user_query": engagement.get("user_query"),
                    "interactive_mode": engagement.get("interactive_mode"),
                },
            })
        except Exception as e:
            logger.warning(f"⚠️ Failed to update database with PAUSED status: {e}")

    logger.info(f"✅ Engagement paused - {len(questions)} questions generated")
    return True


async def _finalize_completed_pipeline(trace_id: str, pipeline_state, database_service: Optional[DatabaseService]):
    """Mark pipeline as completed and persist. CC=6"""
    context_stream = get_unified_context_stream()
    events = context_stream.get_events()

    # Extract final output
    final_output = extract_final_output(events)
    if hasattr(pipeline_state, 'enhancement_research_results') and pipeline_state.enhancement_research_results:
        final_output['enhancement_research_answers'] = pipeline_state.enhancement_research_results

    consultant_data = extract_consultant_selection_data(events)

    # Update in-memory state
    active_engagements[trace_id].update({
        "status": "COMPLETED",
        "current_stage": "COMPLETED",
        "stage_number": active_engagements.get(trace_id, {}).get("total_stages", 10),
        "progress_percentage": 100.0,
        "is_completed": True,
        "completed_at": datetime.now(),
        "final_output": final_output,
        "consultant_selection": consultant_data
    })

    # Persist to database
    if database_service:
        try:
            engagement = active_engagements.get(trace_id, {})
            try:
                from src.core.stage_progress import total_stages_for_ui
                final_stage_num = total_stages_for_ui()
            except Exception:
                final_stage_num = engagement.get("total_stages", 10)

            await database_service.upsert_engagement_status_async({
                "trace_id": trace_id,
                "status": "COMPLETED",
                "current_stage": "COMPLETED",
                "stage_number": final_stage_num,
                "progress_percentage": 100.0,
                "user_id": engagement.get("user_id"),
                "session_id": engagement.get("session_id"),
            })
            logger.info(f"💾 Database COMPLETED status persisted for trace {trace_id}")
        except Exception as e:
            logger.error(f"❌ CRITICAL: Failed to persist COMPLETED status to database: {e}")


async def execute_pipeline_background(
    trace_id: str,
    user_query: str,
    user_id: Optional[str],
    enhanced_context: Optional[Dict[str, Any]] = None,
    database_service: Optional[DatabaseService] = None,
    answered_questions: Optional[list[AnsweredQuestion]] = None,
    research_questions: Optional[list[ResearchQuestion]] = None,
    quality_target: Optional[float] = None,
) -> None:
    """
    Execute the 8-stage pipeline in the background.

    Refactored with helper methods to reduce complexity. CC ≤ 9
    """
    try:
        logger.info(f"🔄 Starting background pipeline execution - Trace ID: {trace_id}")

        # Build enhancement context
        enhancement_context = _build_enhancement_context_from_questions(
            answered_questions, research_questions, quality_target
        )

        # Update status to RUNNING
        active_engagements[trace_id]["status"] = "RUNNING"

        # Initialize services
        db_service, checkpoint_service = _initialize_pipeline_services(database_service)

        # Create orchestrator with status callback
        orchestrator = StatefulPipelineOrchestrator(
            checkpoint_service=checkpoint_service,
            status_callback=_create_status_update_callback(trace_id, db_service)
        )

        # Prepare execution parameters
        trace_id_uuid = UUID(trace_id)
        user_id_uuid = None
        if user_id and user_id.strip():
            try:
                user_id_uuid = UUID(user_id.strip())
            except (ValueError, AttributeError):
                logger.warning(f"⚠️ Invalid user_id format: {user_id}")

        interactive_mode = active_engagements[trace_id].get("interactive_mode", False)

        # Execute pipeline
        pipeline_state = await orchestrator.execute_pipeline(
            trace_id=trace_id_uuid,
            initial_query=user_query,
            user_id=user_id_uuid,
            enhancement_context=enhancement_context if enhancement_context else None,
            interactive_mode=interactive_mode,
        )

        # Handle paused state
        if await _handle_paused_pipeline(trace_id, pipeline_state, database_service):
            return

        # Finalize completed pipeline
        await _finalize_completed_pipeline(trace_id, pipeline_state, database_service)
        logger.info(f"✅ Pipeline execution completed - Trace ID: {trace_id}")

    except Exception as e:
        logger.error(f"❌ Pipeline execution failed - Trace ID: {trace_id}, Error: {e}")

        active_engagements[trace_id].update({
            "status": "FAILED",
            "error": str(e),
            "completed_at": datetime.now()
        })

        if database_service:
            try:
                engagement = active_engagements.get(trace_id, {})
                await database_service.upsert_engagement_status_async({
                    "trace_id": trace_id,
                    "status": "FAILED",
                    "current_stage": engagement.get("current_stage"),
                    "stage_number": engagement.get("stage_number"),
                    "progress_percentage": engagement.get("progress_percentage", 0.0),
                    "user_id": engagement.get("user_id"),
                    "session_id": engagement.get("session_id"),
                    "error_message": str(e)[:500],
                })
                logger.info(f"💾 Database FAILED status persisted for trace {trace_id}")
            except Exception as db_error:
                logger.error(f"❌ Failed to persist FAILED status to database: {db_error}")


async def resume_pipeline_with_answers(
    trace_id: str,
    checkpoint_id: str,
    answered_questions: list[Dict[str, Any]],
    research_questions: list[Dict[str, Any]],
    user_id: Optional[str],
    database_service: Optional[DatabaseService] = None,
) -> None:
    """
    Resume pipeline execution with user answers from interactive mode
    """
    try:
        logger.info(f"🔄 Resuming pipeline execution - Trace ID: {trace_id}")
        logger.info(f"📝 User provided {len(answered_questions)} answers")
        logger.info(f"🔍 User delegated {len(research_questions)} questions for research")

        # Update engagement status
        active_engagements[trace_id]["status"] = "RUNNING"

        # Initialize CheckpointService (database-backed)
        context_stream = get_unified_context_stream()
        db_service = database_service  # Use the parameter passed to this function
        if not db_service:
            from src.services.persistence.database_service import DatabaseService
            db_service = DatabaseService()
        repo = SupabaseCheckpointRepository(db_service, context_stream)
        rev = V1RevisionService(repo, context_stream)
        checkpoint_service = CheckpointService(checkpoint_repo=repo, revision_service=rev)

        # Create enhancement context with user answers
        enhancement_context = {
            "answered_questions": answered_questions,
            "research_questions": research_questions,
            "interactive_answers_provided": True,
        }

        # Real-time status update callback
        def update_frontend_status(stage_name: str, stage_num: int, progress: float):
            """Callback function to update frontend status as pipeline progresses"""
            # Update in-memory dict (backward compatibility - Phase 1)
            active_engagements[trace_id].update({
                "current_stage": stage_name,
                "stage_number": stage_num,
                "progress_percentage": progress,
                "status": "RUNNING"
            })
            logger.info(f"📡 Frontend status updated: Stage {stage_num}/7 - {stage_name} ({progress}%)")

            # Phase 1: Also write to database (non-blocking)
            try:
                engagement = active_engagements.get(trace_id, {})
                db_service.upsert_engagement_status({
                    "trace_id": trace_id,
                    "status": "RUNNING",
                    "current_stage": stage_name,
                    "stage_number": stage_num,
                    "progress_percentage": progress,
                    "user_id": engagement.get("user_id"),
                    "session_id": engagement.get("session_id"),
                })
            except Exception as e:
                logger.warning(f"⚠️ Failed to update database status (non-blocking): {e}")

        # Initialize orchestrator with status callback
        orchestrator = StatefulPipelineOrchestrator(
            checkpoint_service=checkpoint_service,
            status_callback=update_frontend_status
        )

        # Resume pipeline from checkpoint
        trace_id_uuid = UUID(trace_id)
        user_id_uuid = None
        if user_id and user_id.strip():
            try:
                user_id_uuid = UUID(user_id.strip())
            except (ValueError, AttributeError) as e:
                logger.warning(f"⚠️ Invalid user_id format: {user_id}, proceeding without user_id")

        # Resume from checkpoint with user answers
        pipeline_state = await orchestrator.execute_pipeline(
            trace_id=trace_id_uuid,
            resume_from_checkpoint=UUID(checkpoint_id),
            user_id=user_id_uuid,
            enhancement_context=enhancement_context,
            interactive_mode=False,  # Already past the pause point
        )

        # Get final output
        context_stream = get_unified_context_stream()
        events = context_stream.get_events()
        final_output = extract_final_output(events)

        # Extract consultant selection data
        consultant_data = extract_consultant_selection_data(events)

        # Mark as completed
        active_engagements[trace_id].update({
            "status": "COMPLETED",
            "current_stage": "COMPLETED",
            "stage_number": 8,
            "progress_percentage": 100.0,
            "is_completed": True,
            "completed_at": datetime.now(),
            "final_output": final_output,
            "consultant_selection": consultant_data
        })

        # CRITICAL FIX: Persist COMPLETED status to database so frontend knows pipeline finished
        if database_service:
            try:
                engagement = active_engagements.get(trace_id, {})
                await database_service.upsert_engagement_status_async({
                    "trace_id": trace_id,
                    "status": "COMPLETED",
                    "current_stage": "COMPLETED",
                    "stage_number": 8,
                    "progress_percentage": 100.0,
                    "user_id": engagement.get("user_id"),
                    "session_id": engagement.get("session_id"),
                })
                logger.info(f"💾 Database COMPLETED status persisted for trace {trace_id}")
            except Exception as e:
                logger.error(f"❌ CRITICAL: Failed to persist COMPLETED status to database: {e}")
                # This is critical - without it, frontend will never know pipeline completed

        logger.info(f"✅ Pipeline resumption completed - Trace ID: {trace_id}")

    except Exception as e:
        logger.error(f"❌ Pipeline resumption failed - Trace ID: {trace_id}, Error: {e}")

        # Mark as failed
        active_engagements[trace_id].update({
            "status": "FAILED",
            "error": str(e),
            "completed_at": datetime.now()
        })
