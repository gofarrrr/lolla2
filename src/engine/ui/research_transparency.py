"""
Research Transparency Module for METIS Frontend
Utilities for presenting research intelligence to users with progressive disclosure

Author: METIS Cognitive Platform
Date: 2025
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from src.engine.models.data_contracts import MetisDataContract, ResearchIntelligence


class ResearchTransparencyLevel(str, Enum):
    """Levels of research transparency disclosure"""

    EXECUTIVE = "executive"  # High-level insights only
    STRATEGIC = "strategic"  # Strategic insights + confidence
    DETAILED = "detailed"  # Full evidence + sources
    TECHNICAL = "technical"  # Complete research provenance


@dataclass
class ResearchSummaryCard:
    """Executive-level research summary for UI cards"""

    enabled: bool
    confidence_level: str  # "high", "medium", "low"
    insights_count: int
    sources_analyzed: int
    key_insights: List[str]  # Top 3 insights
    research_quality: str  # "excellent", "good", "fair", "limited"


@dataclass
class ResearchDetailView:
    """Detailed research view for transparency disclosure"""

    executive_summary: str
    strategic_insights: List[Dict[str, Any]]  # insight + confidence + sources
    evidence_quality: Dict[str, float]
    information_gaps: List[str]
    source_breakdown: Dict[str, int]  # domain -> count
    research_methodology: Dict[str, Any]
    recommendations: List[str]


@dataclass
class ResearchTechnicalView:
    """Technical research provenance for audit and debugging"""

    sessions_included: List[str]
    queries_executed: List[str]
    processing_metrics: Dict[str, float]
    validation_results: Dict[str, Any]
    contradictions_found: List[Dict[str, Any]]
    research_timeline: List[Dict[str, Any]]


class ResearchTransparencyEngine:
    """
    Engine for progressive disclosure of research intelligence
    Transforms raw research data into user-friendly presentations
    """

    def __init__(self):
        self.confidence_thresholds = {
            "excellent": 0.8,
            "good": 0.65,
            "fair": 0.5,
            "limited": 0.0,
        }

    def extract_research_summary(
        self, contract: MetisDataContract
    ) -> ResearchSummaryCard:
        """Extract executive research summary for UI cards"""

        # Check if research intelligence is available
        research = self._get_research_intelligence(contract)
        if not research:
            return ResearchSummaryCard(
                enabled=False,
                confidence_level="none",
                insights_count=0,
                sources_analyzed=0,
                key_insights=[],
                research_quality="none",
            )

        # Determine confidence level
        confidence = research.overall_confidence
        if confidence >= 0.8:
            confidence_level = "high"
        elif confidence >= 0.65:
            confidence_level = "medium"
        else:
            confidence_level = "low"

        # Determine research quality
        research_quality = "limited"
        for quality, threshold in self.confidence_thresholds.items():
            if confidence >= threshold:
                research_quality = quality
                break

        # Get top insights
        top_insights = research.strategic_insights[:3]

        return ResearchSummaryCard(
            enabled=research.research_enabled,
            confidence_level=confidence_level,
            insights_count=len(research.strategic_insights),
            sources_analyzed=research.total_sources_analyzed,
            key_insights=top_insights,
            research_quality=research_quality,
        )

    def create_research_detail_view(
        self, contract: MetisDataContract
    ) -> Optional[ResearchDetailView]:
        """Create detailed research view for transparency disclosure"""

        research = self._get_research_intelligence(contract)
        if not research:
            return None

        # Enhanced strategic insights with metadata
        strategic_insights = []
        for insight in research.strategic_insights:
            insight_data = {
                "insight": insight,
                "confidence": research.confidence_assessment.get(insight, 0.5),
                "supporting_sources": research.evidence_base.get(insight, []),
                "evidence_count": len(research.evidence_base.get(insight, [])),
            }
            strategic_insights.append(insight_data)

        # Evidence quality breakdown
        evidence_quality = {
            "overall_confidence": research.overall_confidence,
            "evidence_strength": research.evidence_strength,
            "cross_validation": research.cross_validation_score,
            "source_diversity": research.source_diversity
            / max(research.total_sources_analyzed, 1),
        }

        # Source breakdown by domain (if available in raw outputs)
        source_breakdown = {}
        raw_research = contract.raw_outputs.get("research_intelligence", {})
        if raw_research:
            # This would need to be populated by the research orchestrator
            source_breakdown = raw_research.get("source_breakdown", {})

        # Research methodology summary
        research_methodology = {
            "approach": "Progressive Research with Multi-Source Validation",
            "phases": ["Discovery", "Validation", "Deepening", "Synthesis"],
            "templates_used": research.template_used or "Auto-selected",
            "queries_count": len(research.queries_executed),
            "validation_method": "Cross-reference analysis with bias detection",
        }

        # Recommendations based on research quality
        recommendations = []
        if research.overall_confidence < 0.6:
            recommendations.append("Consider additional research for higher confidence")
        if research.information_gaps:
            recommendations.extend(
                [f"Address gap: {gap}" for gap in research.information_gaps[:2]]
            )
        if research.additional_research_needs:
            recommendations.extend(research.additional_research_needs[:2])

        return ResearchDetailView(
            executive_summary=research.executive_summary,
            strategic_insights=strategic_insights,
            evidence_quality=evidence_quality,
            information_gaps=research.information_gaps,
            source_breakdown=source_breakdown,
            research_methodology=research_methodology,
            recommendations=recommendations,
        )

    def create_technical_view(
        self, contract: MetisDataContract
    ) -> Optional[ResearchTechnicalView]:
        """Create technical research provenance view for debugging"""

        research = self._get_research_intelligence(contract)
        if not research:
            return None

        # Processing metrics
        processing_metrics = {
            "overall_confidence": research.overall_confidence,
            "evidence_strength": research.evidence_strength,
            "source_diversity": research.source_diversity,
            "cross_validation_score": research.cross_validation_score,
            "research_depth_score": research.research_depth_score,
            "processing_time_ms": research.processing_time_ms,
        }

        # Research timeline from integration calls
        research_timeline = []
        for call in contract.integration_calls:
            if call.get("integration") == "research_orchestrator_v2":
                research_timeline.append(
                    {
                        "timestamp": call.get("timestamp"),
                        "success": call.get("success"),
                        "confidence": call.get("confidence"),
                        "sources_analyzed": call.get("sources_analyzed"),
                        "insights_generated": call.get("insights_generated"),
                    }
                )

        return ResearchTechnicalView(
            sessions_included=research.sessions_included,
            queries_executed=research.queries_executed,
            processing_metrics=processing_metrics,
            validation_results={
                "contradictions_count": len(research.contradictions_resolved),
                "information_gaps_count": len(research.information_gaps),
            },
            contradictions_found=research.contradictions_resolved,
            research_timeline=research_timeline,
        )

    def get_research_transparency_levels(
        self, contract: MetisDataContract
    ) -> Dict[str, bool]:
        """Get available transparency levels for this contract"""

        research = self._get_research_intelligence(contract)

        return {
            "executive": research is not None,
            "strategic": research is not None and len(research.strategic_insights) > 0,
            "detailed": research is not None and research.total_sources_analyzed > 0,
            "technical": research is not None and len(research.sessions_included) > 0,
        }

    def _get_research_intelligence(
        self, contract: MetisDataContract
    ) -> Optional[ResearchIntelligence]:
        """Extract research intelligence from contract"""

        if not contract.cognitive_state:
            return None

        return contract.cognitive_state.research_intelligence

    def format_research_for_frontend(
        self,
        contract: MetisDataContract,
        level: ResearchTransparencyLevel = ResearchTransparencyLevel.STRATEGIC,
    ) -> Dict[str, Any]:
        """Format research intelligence for frontend consumption"""

        research = self._get_research_intelligence(contract)
        if not research:
            return {"research_enabled": False}

        base_data = {
            "research_enabled": True,
            "overall_confidence": research.overall_confidence,
            "source_count": research.total_sources_analyzed,
            "insights_count": len(research.strategic_insights),
        }

        if level == ResearchTransparencyLevel.EXECUTIVE:
            summary = self.extract_research_summary(contract)
            base_data.update(
                {
                    "confidence_level": summary.confidence_level,
                    "research_quality": summary.research_quality,
                    "key_insights": summary.key_insights,
                }
            )

        elif level == ResearchTransparencyLevel.STRATEGIC:
            detail_view = self.create_research_detail_view(contract)
            if detail_view:
                base_data.update(
                    {
                        "executive_summary": detail_view.executive_summary,
                        "strategic_insights": detail_view.strategic_insights,
                        "evidence_quality": detail_view.evidence_quality,
                        "research_methodology": detail_view.research_methodology,
                    }
                )

        elif level == ResearchTransparencyLevel.DETAILED:
            detail_view = self.create_research_detail_view(contract)
            if detail_view:
                base_data.update(
                    {
                        "executive_summary": detail_view.executive_summary,
                        "strategic_insights": detail_view.strategic_insights,
                        "evidence_quality": detail_view.evidence_quality,
                        "information_gaps": detail_view.information_gaps,
                        "source_breakdown": detail_view.source_breakdown,
                        "research_methodology": detail_view.research_methodology,
                        "recommendations": detail_view.recommendations,
                    }
                )

        elif level == ResearchTransparencyLevel.TECHNICAL:
            tech_view = self.create_technical_view(contract)
            if tech_view:
                base_data.update(
                    {
                        "sessions_included": tech_view.sessions_included,
                        "queries_executed": tech_view.queries_executed,
                        "processing_metrics": tech_view.processing_metrics,
                        "validation_results": tech_view.validation_results,
                        "contradictions_found": tech_view.contradictions_found,
                        "research_timeline": tech_view.research_timeline,
                    }
                )

        return base_data


# Global transparency engine instance
_transparency_engine: Optional[ResearchTransparencyEngine] = None


def get_research_transparency_engine() -> ResearchTransparencyEngine:
    """Get or create global research transparency engine"""
    global _transparency_engine

    if _transparency_engine is None:
        _transparency_engine = ResearchTransparencyEngine()

    return _transparency_engine
