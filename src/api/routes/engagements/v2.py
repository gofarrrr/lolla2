"""
Engagements API - V2 Canonical Endpoints
=========================================

V2 namespace endpoints for report bundles, events, and timeline.

Features:
- ETag-based caching for efficient bundle delivery
- Paginated event access with PII sanitization
- Stage timeline reconstruction

Operation Bedrock: Task 10.0 - API Decomposition
"""

import os
import json
import logging
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Depends, Request, Header
from fastapi.responses import Response as FastAPIResponse

from src.api.auth import require_user
from src.services.report_reconstruction_service import ReportReconstructionService
from src.core.unified_context_stream import get_unified_context_stream
from src.api.event_schema import validate_event_schema

from .helpers import _get_cache, TraceSummary

logger = logging.getLogger(__name__)

# V2 Canonical router
v2_router = APIRouter(
    prefix="/api/v2/engagements",
    tags=["Engagements V2"],
    dependencies=[Depends(require_user)]
)


# ============================================================================
# V2 Bundle Endpoint
# ============================================================================

@v2_router.get("/{trace_id}/bundle")
async def get_report_bundle_v2(
    trace_id: str,
    request: Request,
    if_none_match: Optional[str] = Header(None, alias="If-None-Match"),
    refresh: bool = False,
) -> Any:
    """
    Return the sanitized, cacheable report bundle for a trace.

    Canonical V2 endpoint with ETag-based caching.
    Set refresh=true to bypass caches and rebuild.
    """
    try:
        cache = _get_cache(request)
        db = getattr(request.app.state, "database_service", None)
        svc = ReportReconstructionService(db)

        # Optional bypass: force rebuild when refresh=true
        if refresh:
            bundle = svc.reconstruct_bundle(trace_id)
            etag = 'W/"' + svc._hash_etag(bundle) + '"'
            cache[trace_id] = {"etag": etag, "data": bundle}
            try:
                from src.services.report_cache import set_bundle as _set_bundle
                _set_bundle(trace_id, bundle, etag)
            except Exception:
                pass
            return FastAPIResponse(
                content=json.dumps(bundle),
                media_type="application/json",
                headers={"ETag": etag}
            )

        cached = cache.get(trace_id)
        if cached:
            etag = cached.get("etag")
            if if_none_match and if_none_match == etag:
                return FastAPIResponse(status_code=304, headers={"ETag": etag})
            return FastAPIResponse(
                content=json.dumps(cached["data"]),
                media_type="application/json",
                headers={"ETag": etag}
            )

        # Fallback to module-level cache populated by warm-on-complete
        try:
            from src.services.report_cache import get_bundle as _get_bundle
            mod_cached = _get_bundle(trace_id)
            if mod_cached:
                etag = mod_cached.get("etag")
                cache[trace_id] = mod_cached
                if if_none_match and if_none_match == etag:
                    return FastAPIResponse(status_code=304, headers={"ETag": etag})
                return FastAPIResponse(
                    content=json.dumps(mod_cached["data"]),
                    media_type="application/json",
                    headers={"ETag": etag}
                )
        except Exception:
            pass

        # Build, cache, and return
        bundle = svc.reconstruct_bundle(trace_id)
        etag = 'W/"' + svc._hash_etag(bundle) + '"'
        cache[trace_id] = {"etag": etag, "data": bundle}
        return FastAPIResponse(
            content=json.dumps(bundle),
            media_type="application/json",
            headers={"ETag": etag}
        )
    except Exception as e:
        logger.error(f"❌ Failed to build report bundle for {trace_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to build report bundle")


# ============================================================================
# V2 Events Endpoint
# ============================================================================

@v2_router.get("/{trace_id}/events")
async def get_report_events_v2(
    trace_id: str,
    request: Request,
    offset: int = 0,
    limit: int = 100
) -> Dict[str, Any]:
    """
    Return paginated, sanitized events for a trace.
    Canonical V2 endpoint with deny-by-default sanitization.
    """
    try:
        db = getattr(request.app.state, "database_service", None)
        svc = ReportReconstructionService(db)
        events = svc._collect_events_sanitized(trace_id, limit=limit, offset=offset)

        # DEV-MODE ONLY: Log event schema violations from raw stream with aggregation
        try:
            if os.getenv("ENV", "production").lower() != "production":
                stream = get_unified_context_stream()
                violation_counts: Dict[str, int] = {}
                for ev in getattr(stream, "events", [])[-200:]:
                    errors = validate_event_schema(ev)
                    if errors:
                        et_val = getattr(ev.event_type, 'value', ev.event_type)
                        logger.warning(f"⚠️ Event schema violation for {et_val}: {errors}")
                        violation_counts[et_val] = violation_counts.get(et_val, 0) + 1
                if violation_counts:
                    total_violations = sum(violation_counts.values())
                    logger.warning(
                        f"📊 Event schema violations summary: {total_violations} total violations "
                        f"across {len(violation_counts)} event types: {violation_counts}"
                    )
        except Exception as _e:
            logger.debug(f"Dev-mode validator skipped: {_e}")

        return {"trace_id": trace_id, "events": events, "offset": offset, "limit": limit}
    except Exception as e:
        logger.error(f"❌ Failed to fetch events for {trace_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch events")


# ============================================================================
# V2 Timeline Endpoint
# ============================================================================

@v2_router.get("/{trace_id}/timeline")
async def get_report_timeline_v2(
    trace_id: str,
    request: Request
) -> Dict[str, Any]:
    """
    Return a coarse stage timeline using cached bundle metadata as a quick source.
    Canonical V2 endpoint (falls back to empty).
    """
    try:
        cache = _get_cache(request)
        cached = cache.get(trace_id)
        stages = []
        if cached:
            meta = cached.get("data", {}).get("metadata", {})
            stage_names = meta.get("stages_available", [])
            stages = [{"stage_name": s} for s in stage_names]
        return {"trace_id": trace_id, "stages": stages}
    except Exception as e:
        logger.error(f"❌ Failed to build timeline for {trace_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch timeline")


# ============================================================================
# V2 Trace Summary Endpoint (Alias)
# ============================================================================

@v2_router.get("/{trace_id}/trace", response_model=TraceSummary)
async def get_engagement_trace_v2(trace_id: str) -> TraceSummary:
    """Alias for trace summary under V2 canonical namespace"""
    # Import here to avoid circular dependency
    from .public import get_engagement_trace
    return await get_engagement_trace(trace_id)
