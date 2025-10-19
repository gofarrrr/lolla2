"""
Drift Monitoring API
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any

from src.telemetry.drift import get_drift_monitor

router = APIRouter(prefix="/api/v53/drift", tags=["Drift"]) 

class DriftRecord(BaseModel):
    confidence: Optional[float] = None
    style: Optional[float] = None

@router.get("/status")
async def drift_status() -> Dict[str, Any]:
    try:
        mon = get_drift_monitor()
        # Expose summary
        return {
            "confidence_p05": mon.conf_stats.p05(),
            "confidence_avg": mon.conf_stats.avg(),
            "style_p05": mon.style_stats.p05(),
            "style_avg": mon.style_stats.avg(),
            "conf_floor": mon.conf_floor,
            "style_floor": mon.style_floor,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get drift status: {e}")

@router.post("/record")
async def record_drifts(payload: DriftRecord) -> Dict[str, Any]:
    try:
        mon = get_drift_monitor()
        alert = mon.record(payload.confidence, payload.style)
        return {"alert": alert or False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record drift: {e}")
