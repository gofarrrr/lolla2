"""
Consultant Analysis Stage
=========================

Extracts and normalizes consultant analyses from cognitive stages.

Responsibility:
- Extract consultant analyses from parallel_analysis stage
- Extract selected consultants from consultant_selection stage
- Merge analyses with consultant metadata
- Normalize consultant identities and filter to allowed IDs
- Convert string confidence levels to numeric

Complexity: CC<7 (Moderate merging logic)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from src.services.report_reconstruction.reconstruction_state import ReconstructionState
from src.services.report_reconstruction.reconstruction_stage import (
    ReconstructionStage,
    ReconstructionError,
)

logger = logging.getLogger(__name__)


class ConsultantAnalysisStage(ReconstructionStage):
    """
    Stage 3: Consultant Analysis

    Extracts and normalizes consultant analyses, merging data from
    parallel_analysis and consultant_selection stages.
    """

    # Allowed consultant IDs and their display names
    ALLOWED_CONSULTANTS = {
        "strategic_analyst": "Strategic Analyst",
        "market_researcher": "Market Researcher",
        "risk_assessor": "Risk Assessor",
        "financial_analyst": "Financial Analyst",
        "implementation_specialist": "Implementation Specialist",
        "technology_advisor": "Technology Advisor",
        "innovation_consultant": "Innovation Consultant",
        "crisis_manager": "Crisis Manager",
        "operations_expert": "Operations Expert",
    }

    @property
    def name(self) -> str:
        return "consultant_analysis"

    @property
    def description(self) -> str:
        return "Extract and normalize consultant analyses"

    def process(self, state: ReconstructionState) -> ReconstructionState:
        """
        Extract consultant analyses from stages.

        Args:
            state: Current reconstruction state with stages extracted

        Returns:
            Updated state with consultant_analyses populated
        """
        try:
            # Extract consultant selection metadata
            selected_consultants = self._extract_selected_consultants(state)

            # Extract consultant analyses
            consultant_analyses = self._extract_consultant_analyses(state)

            # Build lookup map for metadata
            consultant_lookup = self._build_consultant_lookup(selected_consultants)

            # Merge analyses with metadata
            merged = self._merge_analyses_with_metadata(
                consultant_analyses, consultant_lookup
            )

            # Filter to allowed consultant IDs only
            filtered = self._filter_allowed_consultants(merged)

            return state.with_consultants(consultant_analyses=filtered)

        except Exception as e:
            raise ReconstructionError(
                self.name,
                f"Failed to extract consultant analyses for trace_id={state.trace_id}",
                cause=e,
            )

    def _extract_selected_consultants(
        self, state: ReconstructionState
    ) -> List[Dict[str, Any]]:
        """Extract selected consultants from consultant_selection stage."""
        selection = state.consultant_selection or {}
        return (
            selection.get("selected_consultants")
            or selection.get("selected_team")
            or []
        )

    def _extract_consultant_analyses(
        self, state: ReconstructionState
    ) -> List[Dict[str, Any]]:
        """Extract consultant analyses from senior_advisor parallel_analysis data."""
        # Try to get from senior_advisor which has accumulated parallel_analysis data
        senior = state.senior_advisor or {}

        # V6: parallel_analysis might be nested under senior_advisor
        analysis_block = senior.get("parallel_analysis", {})

        # Handle nested structure
        if isinstance(analysis_block.get("parallel_analysis"), dict):
            analysis_block = analysis_block["parallel_analysis"]

        return (
            analysis_block.get("consultant_analyses")
            or analysis_block.get("consultant_results")
            or []
        )

    def _build_consultant_lookup(
        self, selected_consultants: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Build lookup map from consultant_selection."""
        consultant_lookup: Dict[str, Dict[str, Any]] = {}

        for consultant in selected_consultants:
            if not isinstance(consultant, dict):
                continue

            consultant_id = consultant.get("consultant_id")
            if not consultant_id:
                continue

            friendly_name = str(consultant_id).replace("_", " ").title()
            consultant_lookup[consultant_id] = {
                "consultant_name": friendly_name,
                "consultant_type": friendly_name,
                "selection_rationale": consultant.get("specialization", ""),
                "expertise_match_score": consultant.get("expertise_match_score")
                or consultant.get("predicted_effectiveness", 0.0),
            }

        return consultant_lookup

    def _merge_analyses_with_metadata(
        self,
        consultant_analyses: List[Dict[str, Any]],
        consultant_lookup: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Merge analyses with consultant metadata."""
        merged: List[Dict[str, Any]] = []

        for analysis in consultant_analyses:
            if not isinstance(analysis, dict):
                continue

            consultant_id = analysis.get("consultant_id", "")
            metadata = consultant_lookup.get(consultant_id, {})

            key_insights = analysis.get("key_insights", [])
            concerns = analysis.get("risk_factors", [])

            # Get selection score (prefer from analysis, fallback to metadata)
            selection_score = analysis.get("confidence_level")
            if selection_score is None:
                selection_score = metadata.get("expertise_match_score", 0.75)

            # Normalize selection score to float
            selection_score = self._normalize_selection_score(
                selection_score, consultant_id
            )

            merged.append(
                {
                    "id": consultant_id or f"consultant_{len(merged)}",
                    "consultant_name": metadata.get(
                        "consultant_name", f"Consultant {str(consultant_id)[:8]}"
                    )
                    or "Consultant",
                    "consultant_type": metadata.get("consultant_type", "Strategic"),
                    "selection_score": selection_score,
                    "perspective": metadata.get("selection_rationale", ""),
                    "key_insights": key_insights,
                    "key_insights_count": (
                        len(key_insights) if isinstance(key_insights, list) else 0
                    ),
                    "concerns": concerns,
                    "concerns_count": len(concerns) if isinstance(concerns, list) else 0,
                }
            )

        return merged

    def _normalize_selection_score(
        self, selection_score: Any, consultant_id: str
    ) -> float:
        """Normalize selection score to float (0.0-1.0)."""
        # Convert string levels to numeric if needed
        if isinstance(selection_score, str):
            score_map = {"low": 0.3, "medium": 0.6, "high": 0.9}
            return score_map.get(selection_score.lower(), 0.5)

        # Ensure valid float
        try:
            return float(selection_score) if selection_score is not None else 0.75
        except (ValueError, TypeError):
            logger.warning(
                f"⚠️ Invalid selection_score for consultant {consultant_id}: "
                f"{selection_score}, defaulting to 0.75"
            )
            return 0.75

    def _filter_allowed_consultants(
        self, consultant_analyses: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter to allowed consultant IDs and assign display names."""
        filtered = []

        for consultant in consultant_analyses:
            consultant_id = consultant.get("consultant_id") or consultant.get("id", "")

            if consultant_id in self.ALLOWED_CONSULTANTS:
                # Always assign proper display name from ALLOWED_CONSULTANTS
                consultant["consultant_name"] = self.ALLOWED_CONSULTANTS[consultant_id]
                filtered.append(consultant)

        return filtered
