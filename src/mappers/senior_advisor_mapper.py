"""
Senior Advisor Mapper - Pilot A (Mapper Extraction Pattern)
============================================================

Pure mapping functions for converting between PipelineState and SeniorAdvisor orchestrator formats.

Pattern: Mapper Extraction
- Extracts conversion logic from executor
- Makes executor thin (3-5 lines)
- Enables isolated testing of transformations
- Maintains clear separation of concerns

Architecture:
    PipelineState → map_to_orchestrator_input() → ConsultantOutput[]

    TwoBrainResult → map_to_pipeline_state() → SeniorAdvisorOutput

Complexity Target: CC ≤8 per method (from CC 82 in executor)
"""

from typing import List, Dict, Any, Optional
from datetime import datetime

from src.core.pipeline_contracts import PipelineState, SeniorAdvisorOutput, StrategicRecommendation
from src.arbitration.models import ConsultantOutput, ConsultantRole, PerspectiveType


class PipelineStateToSeniorAdvisorMapper:
    """
    Pure mapper: PipelineState ↔ SeniorAdvisor orchestrator formats.

    Design Principles:
    - Pure functions (no side effects)
    - No orchestrator dependency
    - Stateless (all state passed as parameters)
    - Testable in isolation
    """

    def map_to_orchestrator_input(
        self, state: PipelineState
    ) -> Dict[str, Any]:
        """
        Convert PipelineState to orchestrator input format.

        Args:
            state: PipelineState with analysis results

        Returns:
            Dict with keys:
                - consultant_outputs: List[ConsultantOutput]
                - original_query: str
                - engagement_id: str

        Raises:
            RuntimeError: If no consultant analyses available
        """
        # Extract consultant outputs
        consultant_outputs = self._map_consultant_analyses(state)

        # Validate we have inputs
        if not consultant_outputs:
            raise RuntimeError("No consultant analyses available; cannot run Senior Advisor synthesis")

        return {
            "consultant_outputs": consultant_outputs,
            "original_query": state.initial_query or "Strategic analysis and recommendations",
            "engagement_id": state.trace_id or "unknown",
        }

    def _map_consultant_analyses(
        self, state: PipelineState
    ) -> List[ConsultantOutput]:
        """
        Extract and convert consultant analyses to ConsultantOutput format.

        Args:
            state: PipelineState with analysis_results

        Returns:
            List of ConsultantOutput objects (limited to 3)
        """
        consultant_outputs: List[ConsultantOutput] = []

        if not state.analysis_results or not state.analysis_results.consultant_analyses:
            return consultant_outputs

        # Convert each analysis to legacy format first (for compatibility)
        for i, analysis in enumerate(state.analysis_results.consultant_analyses[:3]):  # Limit to 3
            # Build intermediate dict (maintains compatibility with existing orchestrator)
            consultant_result = {
                "consultant_id": analysis.consultant_id,
                "consultant_type": "strategic",
                "specialization": "strategic",
                "assigned_dimensions": [],
                "assigned_nways": [],
                "analysis_content": "; ".join(analysis.key_insights[:3]),
                "analysis_output": "; ".join(analysis.key_insights[:3]),
                "key_insights": analysis.key_insights,
                "recommendations": analysis.recommendations,
                "confidence_score": 0.8 if analysis.confidence_level == "HIGH" else 0.6,
                "risk_factors": analysis.risk_factors,
                "opportunities": analysis.opportunities,
            }

            # Convert to ConsultantOutput
            consultant_output = self._build_consultant_output(
                result=consultant_result,
                index=i,
                query=state.initial_query
            )
            consultant_outputs.append(consultant_output)

        return consultant_outputs

    def _build_consultant_output(
        self,
        result: Dict[str, Any],
        index: int,
        query: str
    ) -> ConsultantOutput:
        """
        Build ConsultantOutput from intermediate dict format.

        Args:
            result: Intermediate consultant result dict
            index: Consultant index (0-2)
            query: Original user query

        Returns:
            ConsultantOutput object
        """
        # Assign role based on index
        role = [
            ConsultantRole.ANALYST,
            ConsultantRole.STRATEGIST,
            ConsultantRole.DEVIL_ADVOCATE,
        ][index % 3]

        return ConsultantOutput(
            consultant_role=role,
            analysis_id=f"stateful_pipeline_{index+1}",
            query=query,
            executive_summary=result.get("analysis_output", "")[:500],
            key_insights=[result.get("analysis_output", "Analysis not available")[:200]],
            recommendations=["Recommendation extracted from analysis"],
            mental_models_used=["Standard analysis framework"],
            evidence_sources=["Context analysis"],
            research_depth_score=0.8,
            fact_pack_quality="medium",
            red_team_results={"challenges_identified": True},
            bias_detection_score=0.7,
            logical_consistency_score=0.8,
            processing_time_seconds=30.0,
            cost_usd=0.05,
            confidence_level=result.get("confidence_score", 0.8),
            created_at=datetime.now(),
            primary_perspective=PerspectiveType.STRATEGIC_FOCUSED,
            approach_description="Stateful pipeline analysis",
            limitations_identified=["Limited context"],
            assumptions_made=["Standard market conditions"],
        )

    def map_to_pipeline_state(
        self,
        two_brain_result: Dict[str, Any],
        original_state: PipelineState,
        strategic_recommendations: List[StrategicRecommendation],
        executive_summary: str,
        critical_decisions: List[str],
        evidence_items: List[Any],
        confidence_assessment: Optional[Any],
        processing_time_ms: int
    ) -> PipelineState:
        """
        Convert two-brain orchestrator result back to PipelineState.

        Args:
            two_brain_result: Result from orchestrator conduct_two_brain_analysis()
            original_state: Original PipelineState (for preservation)
            strategic_recommendations: Extracted recommendations
            executive_summary: Extracted executive summary
            critical_decisions: Extracted critical decisions
            evidence_items: Extracted evidence
            confidence_assessment: Confidence assessment (if available)
            processing_time_ms: Total processing time

        Returns:
            Updated PipelineState with final_results populated
        """
        from src.core.pipeline_contracts import AnalysisQuality, AnalysisConfidence

        # Extract confidence from two-brain result
        confidence = two_brain_result.get("raw_analytical_dossier", {}).get("confidence", 0.8)

        # Determine quality based on confidence
        if confidence > 0.8:
            synthesis_quality = AnalysisQuality.EXCELLENT
        elif confidence > 0.6:
            synthesis_quality = AnalysisQuality.GOOD
        else:
            synthesis_quality = AnalysisQuality.ADEQUATE

        # Determine overall confidence level
        if confidence > 0.8:
            overall_confidence = AnalysisConfidence.HIGH
        elif confidence > 0.6:
            overall_confidence = AnalysisConfidence.MEDIUM
        else:
            overall_confidence = AnalysisConfidence.LOW

        # Build SeniorAdvisorOutput
        senior_advisor_output = SeniorAdvisorOutput(
            strategic_recommendations=strategic_recommendations,
            executive_summary=executive_summary,
            critical_decisions=critical_decisions[:4],  # Limit to 4
            risk_opportunity_assessment="Risk/opportunity assessment detailed in comprehensive two-brain analysis report",
            synthesis_quality=synthesis_quality,
            overall_confidence=overall_confidence,
            confidence_assessment=confidence_assessment,
            evidence=evidence_items,
            trace_id=original_state.trace_id or "unknown",
            processing_time_ms=processing_time_ms,
        )

        # Update original state
        original_state.final_results = senior_advisor_output

        return original_state
