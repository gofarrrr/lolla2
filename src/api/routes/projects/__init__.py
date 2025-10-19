"""
Projects API - Main Router
===========================

Combines all V2 projects API sub-routers into a single router.
Operation Bedrock - API Decomposition
"""

from fastapi import APIRouter

from .crud import router as crud_router
from .knowledge import router as knowledge_router
from .mental_models import router as mental_models_router

# Create main router with prefix and tags
router = APIRouter(prefix="/api/v2/projects", tags=["V2 Projects"])

# Include all sub-routers
router.include_router(crud_router)
router.include_router(knowledge_router)
router.include_router(mental_models_router)

__all__ = ["router"]
