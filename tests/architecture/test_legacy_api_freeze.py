from __future__ import annotations

from pathlib import Path

import pytest


LEGACY_API_ALLOWED = {
    "src/engine/api/__init__.py",
    "src/engine/api/analysis_execution_api.py",
    "src/engine/api/analysis_execution_api_v53.py",
    "src/engine/api/arbitration_api.py",
    "src/engine/api/arbitration_api_enhanced.py",
    "src/engine/api/benchmarking_api.py",
    "src/engine/api/c2_command_center_api.py",
    "src/engine/api/comparison_api.py",
    "src/engine/api/devils_advocate_api.py",
    "src/engine/api/endpoints/__init__.py",
    "src/engine/api/endpoints/engagements/__init__.py",
    "src/engine/api/endpoints/system_metrics.py",
    "src/engine/api/engagement/__init__.py",
    "src/engine/api/engagement/clarification.py",
    "src/engine/api/engagement/comparison.py",
    "src/engine/api/engagement/mappers.py",
    "src/engine/api/engagement/models.py",
    "src/engine/api/engagement/orchestrator.py",
    "src/engine/api/engagement/router.py",
    "src/engine/api/engagement/routes.py",
    "src/engine/api/engagement/sandbox.py",
    "src/engine/api/engagement/websocket.py",
    "src/engine/api/engagement_orchestrator.py",
    "src/engine/api/engagement_results_api.py",
    "src/engine/api/enhanced_foundation.py",
    "src/engine/api/enhanced_research_api.py",
    "src/engine/api/enterprise_gateway.py",
    "src/engine/api/errors.py",
    "src/engine/api/feature_flag_decorators.py",
    "src/engine/api/flywheel_management_api.py",
    "src/engine/api/foundation.py",
    "src/engine/api/foundation_analytics_service.py",
    "src/engine/api/foundation_contracts.py",
    "src/engine/api/foundation_facade.py",
    "src/engine/api/foundation_orchestration_service.py",
    "src/engine/api/foundation_repository_service.py",
    "src/engine/api/foundation_service_factory.py",
    "src/engine/api/foundation_validation_service.py",
    "src/engine/api/glass_box_api.py",
    "src/engine/api/human_review_api.py",
    "src/engine/api/hybrid_streaming_api.py",
    "src/engine/api/infrastructure/__init__.py",
    "src/engine/api/infrastructure/api_foundation.py",
    "src/engine/api/infrastructure/middleware.py",
    "src/engine/api/infrastructure/security.py",
    "src/engine/api/intelligence_api.py",
    "src/engine/api/intelligence_server.py",
    "src/engine/api/manual_override_api.py",
    "src/engine/api/markdown_output_api.py",
    "src/engine/api/models/__init__.py",
    "src/engine/api/models/registry.py",
    "src/engine/api/models/request_models.py",
    "src/engine/api/models/response_models.py",
    "src/engine/api/models/validation.py",
    "src/engine/api/platform_stats_api.py",
    "src/engine/api/presentation/six_dimensional_adapter.py",
    "src/engine/api/presentation_adapter.py",
    "src/engine/api/presentation_api.py",
    "src/engine/api/progressive_questions.py",
    "src/engine/api/public_showcase_api.py",
    "src/engine/api/research_history_api.py",
    "src/engine/api/senior_advisor_api.py",
    "src/engine/api/socratic_forge_api.py",
    "src/engine/api/strategic_trio_critique_api.py",
    "src/engine/api/streaming_api.py",
    "src/engine/api/supabase_foundation.py",
    "src/engine/api/system_health_check.py",
    "src/engine/api/three_consultant_api.py",
    "src/engine/api/transparency_stream_manager.py",
    "src/engine/api/transparency_streaming_api.py",
    "src/engine/api/unified_analysis_api.py",
    "src/engine/api/user_journey_facade.py",
    "src/engine/api/websocket_server.py",
    "src/engine/api/whatif_api.py",
}


@pytest.mark.architecture
def test_no_new_files_in_legacy_api_stack() -> None:
    """Block addition of new legacy API modules while sunset work proceeds."""
    src_root = Path("src").resolve()
    api_dir = src_root / "engine" / "api"
    if not api_dir.exists():
        pytest.skip("Legacy API folder already removed.")

    current_files = {
        str(path.relative_to(src_root.parent))
        for path in api_dir.rglob("*.py")
    }
    extra_files = sorted(current_files - LEGACY_API_ALLOWED)

    assert not extra_files, (
        "New files detected under src/engine/api/. "
        "Migrate features to Lean routes instead of expanding the legacy stack:\n"
        + "\n".join(extra_files)
    )
