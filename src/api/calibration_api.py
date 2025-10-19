"""
Calibration API Routes
======================

Lightweight endpoints to expose calibration status for admins.
"""

from __future__ import annotations

from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends

from src.api.auth import require_user
from src.engine.calibration.calibration_service import get_calibration_service


router = APIRouter(prefix="/api/calibration", tags=["Calibration"], dependencies=[Depends(require_user)])


@router.get("/status", response_model=Dict[str, Any])
async def get_status() -> Dict[str, Any]:
    try:
        service = get_calibration_service()
        return await service.generate_report()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate calibration status: {e}")

