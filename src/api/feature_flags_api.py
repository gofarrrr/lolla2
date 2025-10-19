"""
Feature Flags API - list and mutate runtime flags safely.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict

try:
    from src.engine.core.feature_flags import FeatureFlagService, FeatureFlag
except Exception:  # pragma: no cover
    FeatureFlagService = None
    FeatureFlag = None

router = APIRouter(prefix="/api/feature-flags", tags=["Feature Flags"]) 

class FlagUpdate(BaseModel):
    flag: str
    enabled: bool

@router.get("/")
async def list_flags() -> Dict[str, bool]:
    try:
        svc = FeatureFlagService() if FeatureFlagService else None
        if not svc:
            return {}
        # Expose all enum members if available
        flags: Dict[str, bool] = {}
        if FeatureFlag:
            for f in FeatureFlag:
                try:
                    flags[f.name] = svc.is_enabled(f)
                except Exception:
                    flags[f.name] = False
        return flags
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list flags: {e}")

@router.post("/set")
async def set_flag(payload: FlagUpdate) -> Dict[str, bool]:
    try:
        svc = FeatureFlagService() if FeatureFlagService else None
        if not svc or not FeatureFlag:
            raise HTTPException(status_code=503, detail="Feature flags not available")
        # Resolve enum by name if present
        try:
            enum_val = FeatureFlag[payload.flag]
        except KeyError:
            raise HTTPException(status_code=400, detail="Unknown flag")
        svc.set_override(enum_val, payload.enabled)
        return {payload.flag: svc.is_enabled(enum_val)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to set flag: {e}")
