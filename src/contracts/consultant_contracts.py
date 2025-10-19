"""
Consultant Selection Contracts - METIS V5 API Contract Registry
Data contracts for consultant selection components
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from .common_contracts import EngagementContext, ProcessingMetrics


@dataclass
class ConsultantSelectionCriteria:
    """Criteria for consultant selection"""

    top_k: int = 3
    selection_method: str = "semantic_vector_search"  # or "keyword_fallback"
    exclude_consultants: List[str] = field(default_factory=list)
    require_consultants: List[str] = field(default_factory=list)
    minimum_similarity_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "top_k": self.top_k,
            "selection_method": self.selection_method,
            "exclude_consultants": self.exclude_consultants,
            "require_consultants": self.require_consultants,
            "minimum_similarity_score": self.minimum_similarity_score,
        }


@dataclass
class ConsultantCandidate:
    """Individual consultant candidate with selection metadata"""

    consultant_id: str
    consultant_name: str
    consultant_role: str
    similarity_score: float
    selection_reasoning: Optional[str] = None
    nway_cluster_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "consultant_id": self.consultant_id,
            "consultant_name": self.consultant_name,
            "consultant_role": self.consultant_role,
            "similarity_score": self.similarity_score,
            "selection_reasoning": self.selection_reasoning,
            "nway_cluster_id": self.nway_cluster_id,
        }


@dataclass
class ConsultantSelectionRequest:
    """Request contract for consultant selection"""

    engagement_context: EngagementContext
    selection_criteria: ConsultantSelectionCriteria = field(
        default_factory=ConsultantSelectionCriteria
    )
    enhanced_query: Optional[str] = None  # Enriched query from Socratic process

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "engagement_context": self.engagement_context.to_dict(),
            "selection_criteria": self.selection_criteria.to_dict(),
            "enhanced_query": self.enhanced_query,
        }


@dataclass
class ConsultantSelectionResponse:
    """Response contract for consultant selection"""

    success: bool
    engagement_id: str
    selected_consultants: List[ConsultantCandidate] = field(default_factory=list)
    selection_method_used: str = "unknown"
    all_candidates: List[ConsultantCandidate] = field(default_factory=list)
    processing_metrics: Optional[ProcessingMetrics] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "success": self.success,
            "engagement_id": self.engagement_id,
            "selected_consultants": [c.to_dict() for c in self.selected_consultants],
            "selection_method_used": self.selection_method_used,
            "all_candidates": [c.to_dict() for c in self.all_candidates],
            "processing_metrics": (
                self.processing_metrics.to_dict() if self.processing_metrics else None
            ),
            "error_message": self.error_message,
        }


# Standard method signatures that all Consultant Selection components must implement
class ConsultantSelectionInterface:
    """Abstract interface that all Consultant Selection engines must implement"""

    async def select_consultants(
        self, request: ConsultantSelectionRequest
    ) -> ConsultantSelectionResponse:
        """
        PRIMARY CONTRACT METHOD - Standardized consultant selection

        Replaces the various select_relevant_nway_clusters methods with inconsistent signatures
        """
        raise NotImplementedError(
            "Consultant selector must implement select_consultants"
        )

    async def get_available_consultants(self) -> List[ConsultantCandidate]:
        """Get list of all available consultant candidates"""
        raise NotImplementedError(
            "Consultant selector must implement get_available_consultants"
        )

    async def health_check(self) -> Dict[str, Any]:
        """Standard health check method"""
        raise NotImplementedError("Consultant selector must implement health_check")
