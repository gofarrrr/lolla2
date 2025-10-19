"""
Analysis Engine Contracts - METIS V5 API Contract Registry
Data contracts for analysis execution components

This contract addresses the Dossier X failure:
- execute_three_consultant_analysis_v31() parameter signature mismatch
- Inconsistent consultant analysis data structures
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
from .common_contracts import EngagementContext, ProcessingMetrics
from .socratic_contracts import SocraticResponse


class RawAnalyticalDossier(BaseModel):
    """
    V2.1 Master Communicator - Raw Analytical Dossier
    Output structure for Senior Advisor's analytical brain (DeepSeek processing)
    This is the structured JSON output before the communicator brain transforms it to Markdown

    🚨 OPERATION 'CAPTURE THE ARTIFACT' - Made flexible to accept actual LLM responses
    """

    steel_manned_perspectives: List[
        Dict[str, Any]
    ]  # Accept any value type (str, list, etc.)
    key_tensions: List[Dict[str, Any]]  # Accept any value type
    synergistic_insights: List[Dict[str, Any]]  # Accept any value type
    decision_framework: Dict[str, Any]  # Accept any structure


@dataclass
class ConsultantOutput:
    """Output from a single consultant analysis"""

    consultant_id: str
    consultant_role: str
    analysis_content: str
    confidence_score: float
    reasoning_steps: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    risks_identified: List[str] = field(default_factory=list)
    processing_metrics: Optional[ProcessingMetrics] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "consultant_id": self.consultant_id,
            "consultant_role": self.consultant_role,
            "analysis_content": self.analysis_content,
            "confidence_score": self.confidence_score,
            "reasoning_steps": self.reasoning_steps,
            "recommendations": self.recommendations,
            "assumptions": self.assumptions,
            "risks_identified": self.risks_identified,
            "processing_metrics": (
                self.processing_metrics.to_dict() if self.processing_metrics else None
            ),
        }


@dataclass
class AnalysisRequest:
    """Request contract for analysis execution"""

    engagement_context: (
        EngagementContext  # STANDARDIZED - no more problem_statement param
    )
    selected_consultants: List[str]
    socratic_questions: Optional[SocraticResponse] = None
    force_real_processing: bool = False
    max_analysis_length: Optional[int] = None
    include_research: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "engagement_context": self.engagement_context.to_dict(),
            "selected_consultants": self.selected_consultants,
            "socratic_questions": (
                self.socratic_questions.to_dict() if self.socratic_questions else None
            ),
            "force_real_processing": self.force_real_processing,
            "max_analysis_length": self.max_analysis_length,
            "include_research": self.include_research,
        }


@dataclass
class AnalysisResponse:
    """Response contract for analysis execution"""

    success: bool
    engagement_id: str
    consultant_outputs: List[ConsultantOutput] = field(default_factory=list)
    aggregate_confidence: Optional[float] = None
    processing_metrics: Optional[ProcessingMetrics] = None
    research_citations: List[str] = field(default_factory=list)
    api_call_logs: List[Dict[str, Any]] = field(default_factory=list)
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "success": self.success,
            "engagement_id": self.engagement_id,
            "consultant_outputs": [co.to_dict() for co in self.consultant_outputs],
            "aggregate_confidence": self.aggregate_confidence,
            "processing_metrics": (
                self.processing_metrics.to_dict() if self.processing_metrics else None
            ),
            "research_citations": self.research_citations,
            "api_call_logs": self.api_call_logs,
            "error_message": self.error_message,
        }


# Standard method signatures that all Analysis components must implement
class AnalysisEngineInterface:
    """Abstract interface that all Analysis engines must implement"""

    async def execute_consultant_analysis(
        self, request: AnalysisRequest
    ) -> AnalysisResponse:
        """
        PRIMARY CONTRACT METHOD - Replaces the broken execute_three_consultant_analysis_v31

        This is the ONLY acceptable signature for analysis execution.
        All analysis engines MUST use AnalysisRequest, not loose parameters.
        """
        raise NotImplementedError(
            "Analysis engine must implement execute_consultant_analysis"
        )

    async def execute_single_consultant(
        self, engagement_context: EngagementContext, consultant_id: str
    ) -> ConsultantOutput:
        """Execute analysis for a single consultant"""
        raise NotImplementedError(
            "Analysis engine must implement execute_single_consultant"
        )

    async def health_check(self) -> Dict[str, Any]:
        """Standard health check method"""
        raise NotImplementedError("Analysis engine must implement health_check")
