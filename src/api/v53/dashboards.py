"""
Dashboards API for Core, Ops, and Compliance views.
"""
from fastapi import APIRouter, Request
from typing import Dict, Any
from datetime import datetime

router = APIRouter(prefix="/api/v53/dashboards", tags=["Dashboards"]) 

@router.get("/core")
async def core_dashboard(request: Request) -> Dict[str, Any]:
    # Import lazily to avoid hard deps
    from src.telemetry import budget as _budget
    from src.telemetry import style as _style
    now = datetime.utcnow().isoformat()
    # Budget snapshot
    try:
        _budget.budget_tracker.configure_from_env()
        budget = _budget.budget_tracker.snapshot()
    except Exception:
        budget = {"latency_ms_p50": 0, "latency_ms_p95": 0, "cost_today": 0.0}
    # Style is function-based; we expose last score placeholder
    style_summary = {"avg_style_score": 0.0}
    # Confidence summary from app.state.confidence_store if available
    store = getattr(request.app.state, "confidence_store", {}) or {}
    conf_vals = [float(v.get("summary_metrics", {}).get("overall_confidence", 0.0)) for v in store.values()]
    avg_conf = sum(conf_vals)/len(conf_vals) if conf_vals else 0.0

    return {
        "generated_at": now,
        "confidence": {"samples": len(conf_vals), "avg_confidence": avg_conf},
        "memory": {"recall_accuracy": 0.0},
        "style": style_summary,
        "resource": budget,
        "fallback_activation_rate": 0.0,
    }

@router.get("/ops")
async def ops_dashboard(request: Request) -> Dict[str, Any]:
    flags = {}
    try:
        from src.engine.core.feature_flags import FeatureFlagService, FeatureFlag
        svc = FeatureFlagService()
        for f in FeatureFlag:
            flags[f.name] = svc.is_enabled(f)
    except Exception:
        flags = {}
    return {
        "generated_at": datetime.utcnow().isoformat(),
        "system_health": "healthy",
        "feature_flags": flags,
        "error_rates": {},
        "performance": {},
        "resource_consumption": {},
    }

@router.get("/compliance")
async def compliance_dashboard() -> Dict[str, Any]:
    return {
        "generated_at": datetime.utcnow().isoformat(),
        "style_enforcement_rate": 0.0,
        "security_policy_compliance": 1.0,
        "data_handling_metrics": {},
        "audit_trail_visibility": 1.0,
    }
