"""
Transparency Dossier API Routes

Provides endpoints for transparency dossier generation and retrieval.

Extracted from src/main.py as part of Operation Lean - Target #2.
"""

import logging
from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["transparency"])


# ============================================================================
# Global Components (Lazy Loading)
# ============================================================================

_transparency_assembler = None


def get_transparency_assembler():
    """Get or initialize transparency dossier assembler"""
    global _transparency_assembler

    if _transparency_assembler is None:
        from src.api.transparency_dossier_assembler import TransparencyDossierAssembler
        _transparency_assembler = TransparencyDossierAssembler()
        logger.info("✅ TransparencyDossierAssembler initialized")

    return _transparency_assembler


# ============================================================================
# Helper Functions
# ============================================================================

def _create_simulated_context_stream(trace_id: str):
    """
    Create simulated context stream for demonstration.

    This is used when no active context stream is found for the trace_id.
    In production, this would load from database persistence.

    Args:
        trace_id: Trace ID to create simulated stream for

    Returns:
        UnifiedContextStream with sample events
    """
    from src.core.unified_context_stream import (
        get_unified_context_stream,
        ContextEventType,
    )

    context_stream = get_unified_context_stream()
    context_stream.trace_id = trace_id

    # Add sample events
    context_stream.add_event(
        ContextEventType.ENGAGEMENT_STARTED,
        {"query": "Sample strategic analysis query", "user_id": "demo_user"},
    )

    context_stream.add_event(
        ContextEventType.CONSULTANT_SELECTION_COMPLETE,
        {
            "consultant_count": 3,
            "total_confidence": 0.87,
            "consultants": [
                "strategic_advisor",
                "market_analyst",
                "risk_assessor",
            ],
            "selection_rationale": "Selected consultants based on query complexity and strategic domain requirements",
            "risk_factors": [
                "market volatility",
                "competitive threats",
                "resource constraints",
            ],
            "success_factors": [
                "strong strategic vision",
                "experienced team",
                "clear execution plan",
            ],
        },
    )

    context_stream.add_event(
        ContextEventType.RESEARCH_QUERY,
        {
            "query_fingerprint": "sha256_abc123",
            "provider": "perplexity",
            "research_tier": "strategic",
        },
    )

    context_stream.add_event(
        ContextEventType.LLM_PROVIDER_REQUEST,
        {
            "model": "deepseek-chat",
            "provider": "deepseek",
            "request_fingerprint": "sha256_def456",
        },
    )

    context_stream.add_event(
        ContextEventType.SYNTHESIS_CREATED,
        {
            "synthesis_fingerprint": "sha256_ghi789",
            "confidence_score": 0.92,
            "processing_time_ms": 2340,
        },
    )

    context_stream.add_event(
        ContextEventType.DEVILS_ADVOCATE_COMPLETE,
        {
            "bias_checks_performed": ["confirmation_bias", "anchoring"],
            "challenges_identified": 3,
        },
    )

    context_stream.add_event(
        ContextEventType.ENGAGEMENT_COMPLETED,
        {
            "final_status": "completed",
            "duration_ms": 47500,
            "total_tokens": 15420,
        },
    )

    logger.info(f"📝 Created simulated context stream for trace_id: {trace_id}")
    return context_stream


def _convert_dossier_to_dict(dossier) -> dict:
    """
    Convert TransparencyDossier object to API response dict.

    Args:
        dossier: TransparencyDossier object

    Returns:
        Dictionary representation for API response
    """
    return {
        "trace_id": dossier.trace_id,
        "generated_at": dossier.generated_at,
        "session_duration_ms": dossier.session_duration_ms,
        "total_events": dossier.total_events,
        "timeline": [
            {
                "timestamp": event.timestamp,
                "event_type": event.event_type,
                "description": event.description,
                "stage": event.stage,
                "duration_ms": event.duration_ms,
                "confidence": event.confidence,
                "fingerprint": event.fingerprint,
            }
            for event in dossier.timeline
        ],
        "selection_metrics": {
            "total_selections": dossier.selection_metrics.total_selections,
            "average_confidence": dossier.selection_metrics.average_confidence,
            "consultant_distribution": dossier.selection_metrics.consultant_distribution,
            "model_distribution": dossier.selection_metrics.model_distribution,
            "selection_rationales": dossier.selection_metrics.selection_rationales,
            "risk_factors_identified": dossier.selection_metrics.risk_factors_identified,
            "success_factors_identified": dossier.selection_metrics.success_factors_identified,
        },
        "chunking_summary": {
            "total_chunks_processed": dossier.chunking_summary.total_chunks_processed,
            "chunking_strategy_used": dossier.chunking_summary.chunking_strategy_used,
            "chunk_size_distribution": dossier.chunking_summary.chunk_size_distribution,
            "processing_time_per_chunk_ms": dossier.chunking_summary.processing_time_per_chunk_ms,
            "content_fingerprints": dossier.chunking_summary.content_fingerprints,
            "overlap_strategy": dossier.chunking_summary.overlap_strategy,
        },
        "consultant_performance": [
            {
                "consultant_name": perf.consultant_name,
                "total_invocations": perf.total_invocations,
                "average_processing_time_ms": perf.average_processing_time_ms,
                "confidence_scores": perf.confidence_scores,
                "success_rate": perf.success_rate,
                "specializations_used": perf.specializations_used,
                "contribution_quality": perf.contribution_quality,
            }
            for perf in dossier.consultant_performance
        ],
        "quality_analysis": {
            "overall_quality_score": dossier.quality_analysis.overall_quality_score,
            "quality_dimensions": dossier.quality_analysis.quality_dimensions,
            "improvement_suggestions": dossier.quality_analysis.improvement_suggestions,
            "bias_checks_performed": dossier.quality_analysis.bias_checks_performed,
            "validation_steps_completed": dossier.quality_analysis.validation_steps_completed,
        },
        "processing_stages": [
            {
                "stage_name": stage.stage_name,
                "start_time": stage.start_time,
                "end_time": stage.end_time,
                "duration_ms": stage.duration_ms,
                "events_count": stage.events_count,
                "success": stage.success,
                "key_outputs": stage.key_outputs,
                "error_count": stage.error_count,
            }
            for stage in dossier.processing_stages
        ],
        "error_summary": dossier.error_summary,
        "glass_box_compliance": dossier.glass_box_compliance,
    }


# ============================================================================
# API Endpoints
# ============================================================================

@router.get("/transparency-dossier/{trace_id}")
async def get_transparency_dossier(trace_id: str, request: Request):
    """
    Generate comprehensive transparency dossier for a given trace_id.

    The transparency dossier provides complete audit trail including:
    - Event timeline with timestamps and fingerprints
    - Selection metrics (consultants, models, confidence)
    - Chunking summary and content provenance
    - Consultant performance metrics
    - Quality analysis and bias checks
    - Processing stages with duration tracking
    - Glass-box compliance verification

    Args:
        trace_id: Unique trace identifier

    Returns:
        Transparency dossier with complete audit trail

    Raises:
        HTTPException: 500 if dossier generation fails
    """
    try:
        logger.info(f"🔍 Generating transparency dossier for trace_id: {trace_id}")

        # Get transparency assembler
        assembler = get_transparency_assembler()

        # Try to get active context stream
        context_stream = None

        # Check if service container has matching trace_id
        # Import here to avoid circular dependency
        from src.main import service_container

        if service_container.initialized and service_container.context_stream:
            if service_container.context_stream.trace_id == trace_id:
                context_stream = service_container.context_stream
                logger.info(f"✅ Using active context stream for trace_id: {trace_id}")

        # If no active stream found, create simulated one
        if context_stream is None:
            logger.warning(f"⚠️ No active context stream found for trace_id: {trace_id}")
            context_stream = _create_simulated_context_stream(trace_id)

        # Generate the transparency dossier
        dossier = await assembler.assemble_dossier(context_stream)

        # Convert to dictionary for API response
        dossier_dict = _convert_dossier_to_dict(dossier)

        logger.info(
            f"✅ Transparency dossier generated successfully for trace_id: {trace_id}"
        )
        logger.info(f"   Timeline events: {len(dossier.timeline)}")
        logger.info(
            f"   Consultant performance entries: {len(dossier.consultant_performance)}"
        )
        logger.info(f"   Processing stages: {len(dossier.processing_stages)}")
        logger.info(
            f"   Glass-box compliance: {all(dossier.glass_box_compliance.values())}"
        )

        return dossier_dict

    except Exception as e:
        logger.error(
            f"❌ Failed to generate transparency dossier for trace_id {trace_id}: {e}",
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate transparency dossier: {str(e)}"
        )
