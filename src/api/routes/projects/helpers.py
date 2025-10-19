"""
V2 Projects API - Helper Functions
===================================

Dependency injection and utility functions for V2 project endpoints.

Operation Bedrock: Task 11.0 - Projects API Decomposition
"""

import logging
from fastapi import HTTPException, Depends
from supabase import Client

from src.core.supabase_platform import get_supabase_client
from src.contracts.frameworks import StructuredAnalyticalFramework

logger = logging.getLogger(__name__)


# ============================================================
# DEPENDENCY INJECTION
# ============================================================


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
    import re

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


# ============================================================
# HELPER FUNCTIONS
# ============================================================


async def _execute_analysis_pipeline(
    framework: StructuredAnalyticalFramework,
    trace_id: str,
    project_id: str,
    organization_id: str,
) -> None:
    """Execute the analysis pipeline asynchronously"""
    try:
        from src.orchestration.dispatch_orchestrator import DispatchOrchestrator
        from src.services.selection.nway_pattern_service import NWayPatternService
        from src.engine.agents.quality_rater_agent_v2 import get_quality_rater
        from src.integrations.llm.unified_client import UnifiedLLMClient

        logger.info(f"🚀 Starting analysis pipeline for {trace_id}")

        # Initialize components
        orchestrator = DispatchOrchestrator()
        nway_service = NWayPatternService()
        quality_rater = get_quality_rater()
        llm_client = UnifiedLLMClient()

        # Execute dispatch orchestration
        dispatch_package = await orchestrator.run_dispatch(framework)

        # Get mental models from NWay service
        models = await nway_service.get_models_for_consultant(
            dispatch_package.selected_consultants[0].consultant_id
            if dispatch_package.selected_consultants
            else "strategic_analyst"
        )

        # Generate analysis using unified LLM client
        analysis_result = await llm_client.call_llm_unified(
            prompt=framework.user_prompt,
            task_name="strategic_analysis",
            system_prompt=f"You are a {dispatch_package.selected_consultants[0].consultant_id if dispatch_package.selected_consultants else 'strategic analyst'} consultant.",
            models=models[:3] if models else [],  # Use top 3 models
        )

        # Rate quality of analysis
        quality_score = await quality_rater.rate_quality(
            analysis_content=analysis_result,
            context={
                "user_prompt": framework.user_prompt,
                "trace_id": trace_id,
                "project_id": project_id,
            },
        )

        logger.info(
            f"✅ Analysis pipeline completed for {trace_id} with quality score: {quality_score.get('total', 0):.2f}"
        )

    except Exception as e:
        logger.error(f"❌ Analysis pipeline failed for {trace_id}: {e}")
        # Pipeline failure doesn't prevent API response - analysis record was already created
