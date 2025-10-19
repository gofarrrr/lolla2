"""
Deliverable formatting utilities
"""

from typing import Dict, Any
import logging

from .models import ExecutiveDeliverable, PyramidNode
from .enums import DeliverableType


class DeliverableFormatter:
    """Format deliverables for different output types"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def serialize_deliverable(
        self, deliverable: ExecutiveDeliverable
    ) -> Dict[str, Any]:
        """Serialize deliverable for storage"""

        return {
            "deliverable_id": str(deliverable.deliverable_id),
            "type": deliverable.type.value,
            "title": deliverable.title,
            "pyramid_structure": (
                self._serialize_pyramid_node(deliverable.pyramid_structure)
                if deliverable.pyramid_structure
                else None
            ),
            "executive_summary": deliverable.executive_summary,
            "key_recommendations": deliverable.key_recommendations,
            "supporting_analysis": deliverable.supporting_analysis,
            "implementation_roadmap": deliverable.implementation_roadmap,
            "appendices": deliverable.appendices,
            "partner_ready_score": deliverable.partner_ready_score,
            "structure_quality": deliverable.structure_quality,
            "content_quality": deliverable.content_quality,
            "persuasiveness": deliverable.persuasiveness,
            "created_at": deliverable.created_at.isoformat(),
            "engagement_id": (
                str(deliverable.engagement_id) if deliverable.engagement_id else None
            ),
            "author": deliverable.author,
            "review_status": deliverable.review_status,
        }

    def _serialize_pyramid_node(self, node: PyramidNode) -> Dict[str, Any]:
        """Recursively serialize pyramid node"""

        return {
            "node_id": str(node.node_id),
            "level": node.level.value,
            "content": node.content,
            "parent_id": str(node.parent_id) if node.parent_id else None,
            "argument_type": node.argument_type.value,
            "evidence_strength": node.evidence_strength,
            "confidence_level": node.confidence_level.value,
            "supporting_data": node.supporting_data,
            "mece_score": node.mece_score,
            "clarity_score": node.clarity_score,
            "persuasion_score": node.persuasion_score,
            "children": [
                self._serialize_pyramid_node(child) for child in node.children
            ],
        }

    async def generate_deliverable_content(
        self,
        pyramid: PyramidNode,
        deliverable_type: DeliverableType,
        engagement_data: Dict[str, Any],
    ) -> ExecutiveDeliverable:
        """
        Generate deliverable content from pyramid structure
        """

        deliverable = ExecutiveDeliverable(
            type=deliverable_type,
            pyramid_structure=pyramid,
            title=self._generate_title(pyramid, deliverable_type),
        )

        # Generate executive summary from pyramid
        deliverable.executive_summary = self._generate_executive_summary(pyramid)

        # Extract key recommendations from key lines
        deliverable.key_recommendations = self._extract_key_recommendations(pyramid)

        # Build supporting analysis from engagement data
        deliverable.supporting_analysis = self._build_supporting_analysis(
            engagement_data
        )

        # Create implementation roadmap
        deliverable.implementation_roadmap = self._create_implementation_roadmap(
            pyramid
        )

        # Add appendices if needed
        deliverable.appendices = self._create_appendices(engagement_data)

        return deliverable

    def _generate_title(
        self, pyramid: PyramidNode, deliverable_type: DeliverableType
    ) -> str:
        """Generate appropriate title for deliverable"""

        # Extract key theme from governing thought
        governing_thought = pyramid.content

        # Simple keyword extraction for title
        key_words = []
        business_keywords = [
            "growth",
            "efficiency",
            "transformation",
            "optimization",
            "strategy",
        ]

        for keyword in business_keywords:
            if keyword in governing_thought.lower():
                key_words.append(keyword.title())

        if key_words:
            theme = " & ".join(key_words[:2])  # Use up to 2 themes
        else:
            theme = "Strategic Initiative"

        # Type-specific titles
        title_templates = {
            DeliverableType.EXECUTIVE_SUMMARY: f"{theme}: Executive Summary",
            DeliverableType.STRATEGY_DOCUMENT: f"{theme} Strategy Document",
            DeliverableType.BUSINESS_CASE: f"Business Case for {theme}",
            DeliverableType.RECOMMENDATION_MEMO: f"{theme} Recommendations",
            DeliverableType.FINAL_PRESENTATION: f"{theme}: Final Presentation",
            DeliverableType.IMPLEMENTATION_PLAN: f"{theme} Implementation Plan",
        }

        return title_templates.get(deliverable_type, f"{theme} Analysis")

    def _generate_executive_summary(self, pyramid: PyramidNode) -> str:
        """Generate executive summary from pyramid structure"""

        summary_parts = []

        # Start with governing thought
        summary_parts.append(pyramid.content)

        # Add key supporting arguments
        if pyramid.children:
            summary_parts.append("\n\nKey findings include:")

            for i, child in enumerate(pyramid.children, 1):
                summary_parts.append(f"\n{i}. {child.content}")

                # Add top supporting point if available
                if child.children:
                    top_support = child.children[0].content
                    summary_parts.append(f" {top_support}")

        # Add conclusion
        summary_parts.append(
            "\n\nImplementation of these recommendations will deliver measurable "
            "business impact while managing risk through a phased approach."
        )

        return "".join(summary_parts)

    def _extract_key_recommendations(self, pyramid: PyramidNode) -> list[str]:
        """Extract key recommendations from pyramid structure"""

        recommendations = []

        for child in pyramid.children:
            # Convert key lines to actionable recommendations
            content = child.content

            # Make recommendation more actionable
            if "analysis" in content.lower():
                rec = content.replace(
                    "analysis reveals", "Conduct comprehensive analysis of"
                )
                rec = rec.replace(
                    "Analysis reveals", "Conduct comprehensive analysis of"
                )
            elif "implementation" in content.lower():
                rec = content.replace("Implementation of", "Implement")
            elif "phased" in content.lower():
                rec = content.replace("approach ensures", "approach to ensure")
            else:
                rec = f"Execute {content.lower()}"

            recommendations.append(rec)

        return recommendations

    def _build_supporting_analysis(
        self, engagement_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build supporting analysis from engagement data"""

        analysis = {}

        # Add insights if available
        if "insights" in engagement_data:
            analysis["key_insights"] = engagement_data["insights"][:5]  # Top 5 insights

        # Add framework results
        if "frameworks_results" in engagement_data:
            analysis["framework_analysis"] = {}
            for result in engagement_data["frameworks_results"]:
                framework_id = result.get("framework_id", "unknown")
                analysis["framework_analysis"][framework_id] = result.get("output", {})

        # Add hypothesis validation
        if "hypotheses" in engagement_data:
            analysis["hypothesis_validation"] = [
                {
                    "statement": h.get("statement", ""),
                    "confidence": h.get("confidence_score", 0),
                    "validation_status": (
                        "validated"
                        if h.get("confidence_score", 0) > 0.7
                        else "requires_further_analysis"
                    ),
                }
                for h in engagement_data["hypotheses"][:3]
            ]

        return analysis

    def _create_implementation_roadmap(self, pyramid: PyramidNode) -> Dict[str, Any]:
        """Create implementation roadmap from recommendations"""

        roadmap = {"phases": [], "timeline": "6-12 months", "success_metrics": []}

        # Create phases from key lines
        for i, child in enumerate(pyramid.children, 1):
            phase = {
                "phase_number": i,
                "title": f"Phase {i}: {child.content[:50]}...",
                "duration": "2-3 months" if i <= 2 else "3-4 months",
                "key_activities": [point.content for point in child.children[:3]],
                "deliverables": [
                    f"Phase {i} completion report",
                    f"Phase {i} metrics dashboard",
                ],
            }
            roadmap["phases"].append(phase)

        # Add success metrics
        roadmap["success_metrics"] = [
            "Baseline performance measurement established",
            "Phase completion milestones achieved on schedule",
            "Key performance indicators show improvement trend",
            "Stakeholder satisfaction scores meet targets",
            "Return on investment exceeds projections",
        ]

        return roadmap

    def _create_appendices(
        self, engagement_data: Dict[str, Any]
    ) -> list[Dict[str, Any]]:
        """Create appendices with detailed supporting information"""

        appendices = []

        # Appendix A: Detailed Analysis
        if "analysis_findings" in engagement_data:
            appendices.append(
                {
                    "title": "Appendix A: Detailed Analysis Findings",
                    "type": "analysis_data",
                    "content": engagement_data["analysis_findings"],
                }
            )

        # Appendix B: Framework Results
        if "frameworks_results" in engagement_data:
            appendices.append(
                {
                    "title": "Appendix B: Framework Analysis Results",
                    "type": "framework_data",
                    "content": engagement_data["frameworks_results"],
                }
            )

        # Appendix C: Data Sources
        if "research_sources" in engagement_data:
            appendices.append(
                {
                    "title": "Appendix C: Research Sources and Citations",
                    "type": "source_data",
                    "content": engagement_data["research_sources"],
                }
            )

        return appendices
