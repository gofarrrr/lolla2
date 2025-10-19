"""
Engagements API - Modular Package
==================================

Decomposed engagements API with clean separation of concerns:
- public.py: Public lifecycle endpoints (start, status, report, etc.)
- v2.py: V2 canonical endpoints (bundle, events, timeline)
- models.py: Pydantic request/response models
- helpers.py: Shared utility functions

Operation Bedrock: Task 10.0 - API Decomposition
"""

from .public import router, public_router
from .v2 import v2_router

__all__ = ["router", "public_router", "v2_router"]
