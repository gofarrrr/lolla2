"""
Report Reconstructor Factory
============================

Factory function to create a fully configured ReportReconstructor
with all stages wired together.

Usage:
    from src.services.report_reconstruction.factory import create_report_reconstructor

    reconstructor = create_report_reconstructor(db_service, sanitizer)
    bundle = reconstructor.reconstruct(trace_id)
"""

from typing import Optional

from src.services.persistence.database_service import DatabaseService
from src.services.orchestration_infra.glass_box_sanitization_service import (
    GlassBoxSanitizationService,
)

from src.services.report_reconstruction.report_reconstructor import ReportReconstructor
from src.services.report_reconstruction.stages import (
    DataFetchingStage,
    StageExtractionStage,
    ConsultantAnalysisStage,
    EnhancementResearchStage,
    GlassBoxTransparencyStage,
    BundleAssemblyStage,
)


def create_report_reconstructor(
    db: Optional[DatabaseService] = None,
    sanitizer: Optional[GlassBoxSanitizationService] = None,
    enable_logging: bool = True,
) -> ReportReconstructor:
    """
    Create a fully configured ReportReconstructor with all stages.

    Args:
        db: Database service for data fetching (optional)
        sanitizer: Glass-box sanitization service (optional, will create if not provided)
        enable_logging: Whether to enable stage execution logging

    Returns:
        Configured ReportReconstructor ready for use
    """
    # Create sanitizer if not provided
    if sanitizer is None:
        sanitizer = GlassBoxSanitizationService()

    # Create all stages in execution order
    stages = [
        DataFetchingStage(db=db),
        StageExtractionStage(),
        ConsultantAnalysisStage(),
        EnhancementResearchStage(),
        GlassBoxTransparencyStage(),
        BundleAssemblyStage(sanitizer=sanitizer),
    ]

    # Create and return reconstructor
    return ReportReconstructor(stages=stages, enable_logging=enable_logging)
