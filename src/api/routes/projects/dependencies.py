"""
Projects API - Dependency Injection
====================================

Shared dependency functions for V2 projects API endpoints.
"""

import logging
import re
from fastapi import Depends, HTTPException
from supabase import Client

from src.core.supabase_platform import get_supabase_client

logger = logging.getLogger(__name__)


async def get_supabase() -> Client:
    """Get Supabase client dependency"""
    return get_supabase_client()


async def validate_organization_access(
    organization_id: str, supabase: Client = Depends(get_supabase)
) -> str:
    """Validate organization access with basic security checks"""
    # Input validation
    if not organization_id or not organization_id.strip():
        raise HTTPException(status_code=400, detail="Organization ID is required")

    # Format validation - ensure it's a valid UUID or alphanumeric
    if not re.match(r"^[a-zA-Z0-9\-_]+$", organization_id):
        raise HTTPException(status_code=400, detail="Invalid organization ID format")

    # Length validation to prevent injection attacks
    if len(organization_id) > 100:
        raise HTTPException(status_code=400, detail="Organization ID too long")

    # Basic existence check against organizations table
    try:
        result = (
            supabase.table("organizations")
            .select("id")
            .eq("id", organization_id)
            .execute()
        )
        if not result.data:
            raise HTTPException(status_code=404, detail="Organization not found")
    except Exception as e:
        logger.warning(f"Organization validation failed for {organization_id}: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to validate organization access"
        )

    return organization_id
