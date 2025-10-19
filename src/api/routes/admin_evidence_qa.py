#!/usr/bin/env python3
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any, Optional
import os

from src.api.auth import require_user

router = APIRouter(
    prefix="/api/admin/evidence_qa",
    tags=["Admin Evidence QA"],
    dependencies=[Depends(lambda: True)],
)


async def require_admin(user_token: str = Depends(require_user)) -> str:
    """Require authenticated user with admin role.

    Best-effort: If SUPABASE_JWT_SECRET present, decode token and check role claim.
    Accepts role from claims: role, user_role, app_metadata.role, app_metadata.roles[0].
    In local/dev without secret, allow only when ALLOW_ADMIN_BYPASS=true.
    """
    secret = os.getenv("SUPABASE_JWT_SECRET")
    if secret:
        try:
            import jwt  # PyJWT

            claims = jwt.decode(user_token, secret, algorithms=["HS256"])  # raises on failure
            role = (
                claims.get("role")
                or claims.get("user_role")
                or (claims.get("app_metadata", {}) or {}).get("role")
            )
            if not role:
                roles = (claims.get("app_metadata", {}) or {}).get("roles", [])
                if isinstance(roles, list) and roles:
                    role = roles[0]
            if role != "admin":
                raise HTTPException(status_code=403, detail="Admin role required")
            return user_token
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=403, detail="Admin authorization failed")
    else:
        # Dev mode only
        if os.getenv("ALLOW_ADMIN_BYPASS", "false").lower() == "true":
            return user_token
        raise HTTPException(status_code=401, detail="JWT not configured")


@router.get("/queue", dependencies=[Depends(require_admin)])
async def get_evidence_qa_queue(status: str = "queued") -> List[Dict[str, Any]]:
    try:
        from src.persistence.supabase_integration import get_supabase_repository

        repo = await get_supabase_repository()
        return await repo.list_evidence_qa_queue(status=status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ReviewRequest(BaseException):
    def __init__(self, status: str, failure_reason: Optional[str] = None, reviewer_id: Optional[str] = None):
        self.status = status
        self.failure_reason = failure_reason
        self.reviewer_id = reviewer_id


@router.post("/{item_id}/review", dependencies=[Depends(require_admin)])
async def review_evidence_item(item_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        status = payload.get("status")
        if status not in ("passed", "failed"):
            raise HTTPException(status_code=400, detail="status must be 'passed' or 'failed'")
        failure_reason = payload.get("failure_reason")
        reviewer_id = payload.get("reviewer_id")

        from src.persistence.supabase_integration import get_supabase_repository

        repo = await get_supabase_repository()
        ok = await repo.review_evidence_qa_item(
            item_id=item_id,
            status=status,
            reviewer_id=reviewer_id,
            failure_reason=failure_reason,
        )
        if not ok:
            raise HTTPException(status_code=500, detail="Failed to update item")
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
