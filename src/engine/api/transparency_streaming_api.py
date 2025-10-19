#!/usr/bin/env python3
"""
Operation Crystal - Prompt 5: Streaming Transparency API
Server-Sent Events (SSE) endpoint for real-time transparency layer delivery
"""

from fastapi import APIRouter, HTTPException
from typing import AsyncGenerator, Dict, Any
from uuid import UUID
import json
import asyncio
import logging

try:
    from sse_starlette.sse import EventSourceResponse

    SSE_AVAILABLE = True
except ImportError:
    SSE_AVAILABLE = False

# TEMP DISABLED - from src.engine.ui.transparency_engine import AdaptiveTransparencyEngine
from src.engine.models.data_contracts import MetisDataContract
from src.models.transparency_models import TransparencyLayer

router = APIRouter(prefix="/api/v1/transparency", tags=["transparency-streaming"])
logger = logging.getLogger(__name__)

# Global transparency engine instance
_transparency_engine = None


def get_transparency_engine() -> AdaptiveTransparencyEngine:
    """Get or create transparency engine instance"""
    global _transparency_engine
    if _transparency_engine is None:
        _transparency_engine = AdaptiveTransparencyEngine()
    return _transparency_engine


async def generate_transparency_stream(
    engagement_contract: MetisDataContract, user_id: UUID
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Operation Crystal - Prompt 5: Generate transparency layers as stream
    Yields each layer as soon as it's generated for <2s first insight
    """
    engine = get_transparency_engine()

    try:
        # Initialize streaming generation
        yield {
            "event": "start",
            "data": {
                "engagement_id": str(
                    engagement_contract.engagement_context.engagement_id
                ),
                "user_id": str(user_id),
                "timestamp": "2024-01-01T00:00:00Z",  # This will be dynamic
            },
        }

        # Get user profile for adaptive generation
        user_profile = await engine._get_user_profile(user_id)

        # Assess expertise with cold start handling (Prompt 2)
        current_expertise = await engine.expertise_assessor.assess_expertise(
            user_profile
        )

        # Generate validation evidence once for all layers
        validation_evidence = (
            await engine.validation_evidence_engine.generate_validation_evidence(
                engagement_contract.cognitive_state.reasoning_steps, engagement_contract
            )
        )

        # Generate cognitive trace if available
        cognitive_trace = None
        trace_json = None

        # Stream Layer 1: Executive Summary (Priority for <2s delivery)
        layer_1_start = asyncio.get_event_loop().time()
        layer_1_content = await engine._generate_executive_summary(
            engagement_contract,
            user_profile,
            validation_evidence,
            cognitive_trace,
            trace_json,
        )
        layer_1_end = asyncio.get_event_loop().time()

        yield {
            "event": "layer",
            "data": {
                "layer": "executive_summary",
                "layer_number": 1,
                "content": layer_1_content.to_display_format(user_profile),
                "generation_time_ms": int((layer_1_end - layer_1_start) * 1000),
            },
        }

        # Stream Layer 2: Reasoning Overview
        layer_2_start = asyncio.get_event_loop().time()
        layer_2_content = await engine._generate_reasoning_overview(
            engagement_contract,
            user_profile,
            validation_evidence,
            cognitive_trace,
            trace_json,
        )
        layer_2_end = asyncio.get_event_loop().time()

        yield {
            "event": "layer",
            "data": {
                "layer": "reasoning_overview",
                "layer_number": 2,
                "content": layer_2_content.to_display_format(user_profile),
                "generation_time_ms": int((layer_2_end - layer_2_start) * 1000),
            },
        }

        # Stream Layer 3: Detailed Audit Trail
        layer_3_start = asyncio.get_event_loop().time()
        layer_3_content = await engine._generate_detailed_audit_trail(
            engagement_contract,
            user_profile,
            validation_evidence,
            cognitive_trace,
            trace_json,
        )
        layer_3_end = asyncio.get_event_loop().time()

        yield {
            "event": "layer",
            "data": {
                "layer": "detailed_audit_trail",
                "layer_number": 3,
                "content": layer_3_content.to_display_format(user_profile),
                "generation_time_ms": int((layer_3_end - layer_3_start) * 1000),
            },
        }

        # Stream Layer 4: Technical Execution
        layer_4_start = asyncio.get_event_loop().time()
        layer_4_content = await engine._generate_technical_execution_log(
            engagement_contract,
            user_profile,
            validation_evidence,
            cognitive_trace,
            trace_json,
        )
        layer_4_end = asyncio.get_event_loop().time()

        yield {
            "event": "layer",
            "data": {
                "layer": "technical_execution",
                "layer_number": 4,
                "content": layer_4_content.to_display_format(user_profile),
                "generation_time_ms": int((layer_4_end - layer_4_start) * 1000),
            },
        }

        # Apply cognitive scaffolding to all layers if needed (Prompt 3)
        layers = {
            TransparencyLayer.EXECUTIVE_SUMMARY: layer_1_content,
            TransparencyLayer.REASONING_OVERVIEW: layer_2_content,
            TransparencyLayer.DETAILED_AUDIT_TRAIL: layer_3_content,
            TransparencyLayer.TECHNICAL_EXECUTION: layer_4_content,
        }

        # (Cognitive scaffolding logic would be applied here)

        # Final completion event
        yield {
            "event": "complete",
            "data": {
                "message": "All transparency layers generated successfully",
                "total_layers": 4,
                "timestamp": "2024-01-01T00:00:00Z",  # Dynamic
            },
        }

    except Exception as e:
        # Error handling for streaming failures
        logger.error(f"Transparency streaming failed: {e}")
        yield {
            "event": "error",
            "data": {
                "error": str(e),
                "message": "Transparency generation failed",
                "timestamp": "2024-01-01T00:00:00Z",
            },
        }


@router.get("/stream/{engagement_id}")
async def stream_transparency_layers(engagement_id: str, user_id: str):
    """
    Operation Crystal - Prompt 5: SSE endpoint for streaming transparency layers
    Delivers layers in real-time for <2s first insight experience
    """
    if not SSE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Server-Sent Events not available. Please install sse-starlette.",
        )

    try:
        # In a real implementation, we would fetch the engagement contract from database
        # For now, we'll create a mock contract
        from src.engine.models.data_contracts import create_engagement_initiated_event

        engagement_contract = create_engagement_initiated_event(
            problem_statement="Stream test engagement", client_name="Test Client"
        )
        engagement_contract.engagement_context.engagement_id = UUID(engagement_id)

        user_uuid = UUID(user_id)

        async def event_generator():
            async for event_data in generate_transparency_stream(
                engagement_contract, user_uuid
            ):
                # Format for SSE
                event_type = event_data.get("event", "data")
                data = json.dumps(event_data.get("data", {}))
                yield f"event: {event_type}\ndata: {data}\n\n"

        return EventSourceResponse(event_generator())

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid UUID format: {e}")
    except Exception as e:
        logger.error(f"Streaming endpoint failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/feature-flag")
async def get_streaming_feature_flag():
    """Check if streaming transparency is enabled"""
    import os

    enabled = os.getenv("ENABLE_ADAPTIVE_TRANSPARENCY_V2", "false").lower() == "true"
    return {
        "streaming_enabled": enabled,
        "sse_available": SSE_AVAILABLE,
        "feature": "adaptive_transparency_v2",
    }
