"""
Analysis Execution API Routes

Provides the main analysis endpoint for end-to-end query analysis.

Extracted from src/main.py as part of Operation Lean - Target #2.
"""

import logging
import time
import uuid
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from src.services.application.system2_classification_service import System2ClassificationService
from src.services.application.contracts import Tier

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v53", tags=["analysis"])


# ============================================================================
# Request/Response Models
# ============================================================================

class AnalysisRequest(BaseModel):
    """Request model for analysis execution"""
    query: str
    context: Optional[Dict[str, Any]] = {}
    complexity: Optional[str] = "auto"  # auto, simple, strategic, complex


class AnalysisResponse(BaseModel):
    """Response model for analysis execution"""
    query: str
    system2_tier: str
    consultant_selection: List[str]
    selected_models: List[str]
    analysis: str
    quality_scores: Dict[str, Any]
    summary_metrics: Dict[str, Any]
    execution_time_ms: float
    trace_id: str


# ============================================================================
# Global Components (Lazy Loading)
# ============================================================================

_dispatch_orchestrator = None
_nway_pattern_service = None
_quality_rater = None
_llm_client = None


def get_analysis_components():
    from src.engine.adapters import create_pipeline_orchestrator
    """Get or initialize analysis components"""
    global _dispatch_orchestrator, _nway_pattern_service, _quality_rater, _llm_client

    if _dispatch_orchestrator is None:
        from src.orchestration.dispatch_orchestrator import DispatchOrchestrator
        from src.services.selection.nway_pattern_service import NWayPatternService
        from src.engine.agents.quality_rater_agent_v2 import get_quality_rater
        from src.integrations.llm.unified_client import UnifiedLLMClient

        _dispatch_orchestrator = DispatchOrchestrator()
        _nway_pattern_service = NWayPatternService()
        _quality_rater = get_quality_rater()
        _llm_client = UnifiedLLMClient()

        logger.info("✅ Analysis components initialized")

    return _dispatch_orchestrator, _nway_pattern_service, _quality_rater, _llm_client


# ============================================================================
# Helper Functions
# ============================================================================

async def _select_consultants_and_models(
    request: AnalysisRequest,
    system2_tier: int,
    dispatcher,
    pattern_service
):
    """
    Select consultants and mental models.

    Args:
        request: Analysis request
        system2_tier: System-2 tier level (1, 2, or 3)
        dispatcher: DispatchOrchestrator instance
        pattern_service: NWayPatternService instance

    Returns:
        Tuple of (consultant_names, selected_models)
    """
    from src.orchestration.dispatch_orchestrator import StructuredAnalyticalFramework
    from src.orchestration.contracts import AnalyticalDimension, FrameworkType

    # Create analytical framework
    framework = StructuredAnalyticalFramework(
        framework_type=FrameworkType.STRATEGIC_ANALYSIS,
        primary_dimensions=[
            AnalyticalDimension(
                dimension_name="Strategic Context",
                key_questions=[request.query],
                analysis_approach="comprehensive",
                priority_level=1
            )
        ],
        secondary_considerations=[f"System-2 Tier: {system2_tier}"],
        analytical_sequence=["context_analysis", "recommendation_generation"],
        complexity_assessment=system2_tier,
        recommended_consultant_types=["strategic_analyst", "market_researcher"],
        processing_time_seconds=0.0,
    )

    # Consultant selection
    dispatch_package = await dispatcher.run_dispatch(framework, request.query)
    consultant_names = [c.consultant_id for c in dispatch_package.selected_consultants]

    # Mental model selection
    selected_models = []
    for consultant_blueprint in dispatch_package.selected_consultants:
        consultant_models = await pattern_service.get_models_for_consultant(
            consultant_blueprint.consultant_id
        )
        selected_models.extend(consultant_models[:3])

    logger.debug(f"Selected {len(consultant_names)} consultants and {len(selected_models)} models")
    return consultant_names, selected_models


async def _generate_analysis_with_memory(
    request: AnalysisRequest,
    system2_tier: int,
    consultant_names: List[str],
    selected_models: List[str],
    llm_client
):
    """
    Generate analysis with memory context integration.

    Args:
        request: Analysis request
        system2_tier: System-2 tier level
        consultant_names: Selected consultant names
        selected_models: Selected mental models
        llm_client: UnifiedLLMClient instance

    Returns:
        Tuple of (analysis_text, analysis_prompt)
    """
    # Get memory summary
    memory_summary = None
    try:
        session_id_ctx = (request.context or {}).get("session_id") if isinstance(request.context, dict) else None
        if session_id_ctx:
            from src.storage.zep_memory import ZepMemoryManager
            import os
            zmm = ZepMemoryManager()
            hl = float(os.getenv("MEMORY_DECAY_HALF_LIFE_DAYS", "30"))
            memory_summary = await zmm.summarize_recent_context(
                session_id_ctx,
                max_messages=30,
                half_life_days=hl
            )
            logger.debug(f"Retrieved memory context for session {session_id_ctx}")
    except Exception as e:
        logger.warning(f"Failed to retrieve memory context: {e}")

    # Build prompt
    analysis_prompt = f"""
Context: {request.context}
{'Recent Context Summary (decayed):\n' + memory_summary if memory_summary else ''}
Query: {request.query}
System-2 Tier: {system2_tier}
Selected Consultants: {consultant_names}
Selected Models: {selected_models}

Provide a comprehensive analysis integrating the selected mental models and consultant expertise.
"""

    # Generate analysis
    llm_response = await llm_client.generate_analysis(
        prompt=analysis_prompt,
        context={
            "tier": system2_tier,
            "consultants": consultant_names,
            "models": selected_models
        },
    )

    analysis_text = llm_response.content if hasattr(llm_response, "content") else str(llm_response)
    logger.debug(f"Generated analysis ({len(analysis_text)} chars)")

    return analysis_text, analysis_prompt


def _compute_summary_metrics_with_telemetry(
    consultant_names: List[str],
    selected_models: List[str],
    start_time: float,
    system2_tier: int,
    quality_scores: Dict[str, Any],
    llm_client,
    analysis_text: str
):
    """
    Compute summary metrics and record telemetry.

    Args:
        consultant_names: Selected consultant names
        selected_models: Selected mental models
        start_time: Analysis start time (from time.time())
        system2_tier: System-2 tier level
        quality_scores: Quality scores from CQA rater
        llm_client: UnifiedLLMClient instance
        analysis_text: Generated analysis text

    Returns:
        Tuple of (summary_metrics, confidence_provenance)
    """
    # Merge runtime quality scores
    try:
        runtime_qs = getattr(llm_client, "get_runtime_quality_scores", None)
        if callable(runtime_qs):
            rt = runtime_qs() or {}
            if isinstance(quality_scores, dict):
                quality_scores = {**quality_scores, **rt}
            else:
                quality_scores = rt
    except Exception:
        pass

    # Compute summary metrics
    from src.telemetry.summary_metrics import summary_metrics_service
    summary_metrics, confidence_provenance = summary_metrics_service.compute_with_provenance(
        selected_consultants=consultant_names,
        selected_models=selected_models,
        execution_time_ms=(time.time() - start_time) * 1000,
        context={"tier": system2_tier},
        quality_scores=quality_scores if isinstance(quality_scores, dict) else {},
    )

    # Record drift (non-blocking)
    try:
        from src.telemetry.style import score_style
        from src.telemetry.drift import get_drift_monitor
        style_score = score_style(analysis_text)
        dm = get_drift_monitor()
        dm.record(
            float(summary_metrics.get("overall_confidence", 0.0)),
            float(style_score) if style_score is not None else None,
        )
    except Exception:
        pass

    return summary_metrics, confidence_provenance


# ============================================================================
# API Endpoints
# ============================================================================

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_query(req: AnalysisRequest, request: Request):
    """
    End-to-end analysis endpoint.

    Executes complete analysis pipeline:
    1. System-2 tier classification
    2. Consultant and model selection
    3. Analysis generation with memory context
    4. Quality scoring
    5. Summary metrics computation

    Args:
        req: Analysis request with query, context, and complexity

    Returns:
        Complete analysis response with results and metrics

    Raises:
        HTTPException: 500 if analysis fails
    """
    start_time = time.time()
    trace_id = str(uuid.uuid4())

    try:
        # Get analysis components
        dispatcher, pattern_service, quality_rater, llm_client = get_analysis_components()

        # Step 1: System-2 Kernel Tier Classification
        classifier = System2ClassificationService()
        system2_tier_enum = classifier.classify_tier(req.query, req.complexity)
        system2_tier = 1 if system2_tier_enum == Tier.TIER_1 else (
            2 if system2_tier_enum == Tier.TIER_2 else 3
        )

        logger.info(f"🎯 Analysis started: trace_id={trace_id}, tier={system2_tier_enum.value}")

        # Step 2-4: Select consultants and models
        consultant_names, selected_models = await _select_consultants_and_models(
            req, system2_tier, dispatcher, pattern_service
        )

        # Step 5: Generate analysis with memory context
        analysis_text, analysis_prompt = await _generate_analysis_with_memory(
            req, system2_tier, consultant_names, selected_models, llm_client
        )

        # Step 6: Quality Scoring via CQA Rater
        quality_scores = await quality_rater.rate_quality(
            analysis_content=analysis_text,
            context={
                "user_prompt": req.query,
                "system_prompt": analysis_prompt,
                "tier": system2_tier,
            },
        )

        # Step 7: Summary metrics and telemetry
        summary_metrics, confidence_provenance = _compute_summary_metrics_with_telemetry(
            consultant_names, selected_models, start_time, system2_tier,
            quality_scores, llm_client, analysis_text
        )

        # Store provenance for trace
        try:
            if not hasattr(request.app.state, "confidence_store"):
                request.app.state.confidence_store = {}
            request.app.state.confidence_store[trace_id] = {
                "summary_metrics": summary_metrics,
                "provenance": confidence_provenance,
            }
        except Exception:
            pass

        execution_time = (time.time() - start_time) * 1000  # Convert to ms

        # Resource budget tracking (latency only for now)
        try:
            from src.telemetry.budget import budget_tracker
            budget_tracker.record(latency_ms=execution_time)
        except Exception:
            pass

        logger.info(
            f"✅ Analysis complete: trace_id={trace_id}, "
            f"time={execution_time:.2f}ms, "
            f"confidence={summary_metrics.get('overall_confidence', 0.0):.2f}"
        )

        return AnalysisResponse(
            query=req.query,
            system2_tier=system2_tier_enum.value,
            consultant_selection=consultant_names,
            selected_models=selected_models,
            analysis=analysis_text,
            quality_scores=quality_scores,
            summary_metrics=summary_metrics,
            execution_time_ms=execution_time,
            trace_id=trace_id,
        )

    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
