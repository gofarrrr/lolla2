"""
Report Reconstruction Service - State Reconstruction Pattern
============================================================

Refactored report reconstruction service using the State Reconstruction Pattern.

Architecture:
- ReconstructionState: Immutable data container
- ReconstructionStage: Abstract base for stages
- ReportReconstructor: Stage orchestrator
- create_report_reconstructor: Factory function

Stages:
1. DataFetchingStage: Fetch from DB/stream
2. StageExtractionStage: Extract cognitive stages
3. ConsultantAnalysisStage: Consultant data
4. EnhancementResearchStage: Research extraction
5. GlassBoxTransparencyStage: Evidence/quality/plan
6. BundleAssemblyStage: Final assembly + sanitization

Usage:
    from src.services.report_reconstruction import create_report_reconstructor

    reconstructor = create_report_reconstructor(db_service)
    bundle = reconstructor.reconstruct(trace_id)
"""

from src.services.report_reconstruction.reconstruction_state import ReconstructionState
from src.services.report_reconstruction.reconstruction_stage import (
    ReconstructionStage,
    ReconstructionError,
)
from src.services.report_reconstruction.report_reconstructor import ReportReconstructor
from src.services.report_reconstruction.factory import create_report_reconstructor

__all__ = [
    "ReconstructionState",
    "ReconstructionStage",
    "ReconstructionError",
    "ReportReconstructor",
    "create_report_reconstructor",
]
