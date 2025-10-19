"""
Confidence Trace API Routes

Provides endpoints for confidence trace retrieval and recomputation.

Extracted from src/main.py as part of Operation Lean - Target #2.
"""

import logging
from typing import Optional, Dict
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v53", tags=["confidence"])


# ============================================================================
# Request/Response Models
# ============================================================================

class RecomputeRequest(BaseModel):
    """Request model for confidence recomputation"""
    weights: Optional[Dict[str, float]] = None
    commit: bool = False


# ============================================================================
# Helper Functions
# ============================================================================

def _get_confidence_data(store: dict, trace_id: str) -> dict:
    """
    Get and validate confidence data from store.

    Args:
        store: Confidence data store (dict)
        trace_id: Trace ID to retrieve

    Returns:
        Confidence data dict

    Raises:
        HTTPException: If trace not found or invalid data
    """
    data = store.get(trace_id)
    if not data:
        raise HTTPException(status_code=404, detail="Confidence trace not found")

    provenance = data.get("provenance", {})
    factors = provenance.get("factors", {})
    if not factors:
        raise HTTPException(status_code=400, detail="No factors available for recompute")

    return data


def _compute_weighted_score(factors: dict, weights: dict) -> float:
    """
    Compute weighted confidence score from factors.

    Args:
        factors: Factor scores dict (e.g., {"evidence": 0.8, "coherence": 0.9})
        weights: Factor weights dict (e.g., {"evidence": 1.0, "coherence": 1.0})

    Returns:
        Weighted confidence score (0.0 to 1.0)
    """
    denom = sum(weights.values()) or 0.0
    if denom == 0:
        return 0.0

    weighted_sum = sum(
        float(factors.get(k, 0.0)) * weights.get(k, 1.0)
        for k in factors.keys()
    )

    return weighted_sum / denom


# ============================================================================
# API Endpoints
# ============================================================================

@router.get("/confidence/{trace_id}")
async def get_confidence_trace(trace_id: str, request: Request):
    """
    Get confidence trace for a given trace ID.

    Returns complete confidence provenance including factors, signals,
    and summary metrics for the specified trace.

    Args:
        trace_id: Unique trace identifier

    Returns:
        Confidence trace data with provenance and metrics

    Raises:
        HTTPException: 404 if trace not found
    """
    store = getattr(request.app.state, "confidence_store", {})
    data = store.get(trace_id)

    if not data:
        raise HTTPException(status_code=404, detail="Confidence trace not found")

    logger.info(f"📊 Retrieved confidence trace: {trace_id}")
    return data


@router.post("/confidence/{trace_id}/recompute")
async def recompute_confidence_trace(
    trace_id: str,
    req: RecomputeRequest,
    request: Request
):
    """
    Recompute confidence trace with new weights.

    Allows adjustment of factor weights to see impact on overall confidence
    score. Optionally commits the new score to the store.

    Args:
        trace_id: Unique trace identifier
        req: Recompute request with optional weights and commit flag

    Returns:
        Original score, recomputed score, and weights used

    Raises:
        HTTPException: 404 if trace not found, 400 if invalid data
    """
    store = getattr(request.app.state, "confidence_store", {})
    data = _get_confidence_data(store, trace_id)

    factors = data["provenance"]["factors"]

    # Get default weights and merge with request weights
    from src.telemetry.confidence import ConfidenceScorer
    scorer = ConfidenceScorer()
    weights = scorer._get_factor_weights(list(factors.keys())).copy()

    if req.weights:
        for k, v in req.weights.items():
            if k in weights and isinstance(v, (int, float)):
                weights[k] = float(v)

    # Compute new score
    new_score = _compute_weighted_score(factors, weights)

    result = {
        "trace_id": trace_id,
        "original": data.get("summary_metrics", {}).get("overall_confidence"),
        "recomputed": new_score,
        "weights": weights,
    }

    # Optionally commit new score
    if req.commit:
        data["summary_metrics"]["overall_confidence"] = round(new_score, 4)
        store[trace_id] = data
        logger.info(f"💾 Committed recomputed confidence for trace {trace_id}: {new_score:.4f}")
    else:
        logger.info(f"📊 Recomputed confidence for trace {trace_id}: {new_score:.4f} (not committed)")

    return result


@router.get("/confidence/calibration")
async def get_confidence_calibration(request: Request):
    """
    Get confidence calibration metrics across all traces.

    Computes Brier score and Expected Calibration Error (ECE) to measure
    how well-calibrated the confidence scores are.

    Returns:
        Calibration metrics including Brier score, ECE, and sample count
    """
    from src.telemetry.calibration import brier_score, expected_calibration_error

    store = getattr(request.app.state, "confidence_store", {})
    pairs = []

    for trace_id, data in store.items():
        sm = data.get("summary_metrics", {})
        prov = data.get("provenance", {})
        p = float(sm.get("overall_confidence", 0.0))

        # Outcome heuristic: treat strong evidence and coherence as success
        sig = prov.get("signals", {})
        y = 1 if (sig.get("evidence", 0.0) >= 0.7 and sig.get("coherence", 0.0) >= 0.6) else 0
        pairs.append((p, y))

    bs = brier_score(pairs)
    ece = expected_calibration_error(pairs, n_bins=10)

    logger.info(f"📊 Calibration metrics computed: brier={bs:.4f}, samples={len(pairs)}")

    return {"brier": bs, **ece, "samples": len(pairs)}
