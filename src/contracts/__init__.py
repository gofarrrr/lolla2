"""
METIS V5 API Contract Registry
The single source of truth for all inter-component communication contracts

This module enforces "THE API CONTRACT ACCORDS" - a systematic approach to
preventing API contract failures that have plagued the system.

All components MUST import their interfaces from this registry.
"""

from .socratic_contracts import SocraticRequest, SocraticResponse, QuestionSet
from .analysis_contracts import AnalysisRequest, AnalysisResponse, ConsultantOutput
from .consultant_contracts import (
    ConsultantSelectionRequest,
    ConsultantSelectionResponse,
)
from .common_contracts import EngagementContext, ProcessingMetrics, ErrorResponse

__all__ = [
    # Socratic Engine Contracts
    "SocraticRequest",
    "SocraticResponse",
    "QuestionSet",
    # Analysis Engine Contracts
    "AnalysisRequest",
    "AnalysisResponse",
    "ConsultantOutput",
    # Consultant Selection Contracts
    "ConsultantSelectionRequest",
    "ConsultantSelectionResponse",
    # Common Contracts
    "EngagementContext",
    "ProcessingMetrics",
    "ErrorResponse",
]
