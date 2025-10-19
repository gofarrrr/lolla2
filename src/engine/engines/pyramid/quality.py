"""
Quality assessment for pyramid deliverables
"""

import logging

from .models import ExecutiveDeliverable, PyramidNode


class QualityAssessor:
    """Assess quality and partner-readiness of deliverables"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Quality assessment criteria weights
        self.partner_ready_criteria = {
            "structure_clarity": 0.25,
            "executive_focus": 0.25,
            "evidence_strength": 0.20,
            "actionability": 0.15,
            "persuasiveness": 0.15,
        }

    async def assess_deliverable_quality(
        self, deliverable: ExecutiveDeliverable
    ) -> None:
        """
        Assess overall deliverable quality and partner-readiness
        """

        # Assess pyramid structure quality
        structure_score = 0.0
        if deliverable.pyramid_structure:
            structure_score = await self._assess_pyramid_structure(
                deliverable.pyramid_structure
            )

        # Assess content quality
        content_score = self._assess_content_quality(deliverable)

        # Assess executive focus
        executive_score = self._assess_executive_focus(deliverable)

        # Assess evidence strength
        evidence_score = self._assess_evidence_strength(deliverable)

        # Assess actionability
        actionability_score = self._assess_actionability(deliverable)

        # Calculate weighted partner readiness score
        partner_ready_score = (
            self.partner_ready_criteria["structure_clarity"] * structure_score
            + self.partner_ready_criteria["executive_focus"] * executive_score
            + self.partner_ready_criteria["evidence_strength"] * evidence_score
            + self.partner_ready_criteria["actionability"] * actionability_score
            + self.partner_ready_criteria["persuasiveness"] * content_score
        )

        # Update deliverable metrics
        deliverable.structure_quality = structure_score
        deliverable.content_quality = content_score
        deliverable.partner_ready_score = partner_ready_score
        deliverable.persuasiveness = (content_score + evidence_score) / 2

        self.logger.info(
            f"Quality assessment: Structure={structure_score:.2f}, "
            f"Content={content_score:.2f}, Partner-ready={partner_ready_score:.2f}"
        )

    async def _assess_pyramid_structure(self, pyramid: PyramidNode) -> float:
        """Assess pyramid structure quality"""

        structure_score = 0.0

        # Check for proper hierarchy
        if pyramid.children:
            structure_score += 0.3  # Has key supporting lines

            # Check depth and breadth
            avg_children_per_line = sum(
                len(child.children) for child in pyramid.children
            ) / len(pyramid.children)
            if (
                2 <= avg_children_per_line <= 4
            ):  # Optimal 2-4 supporting points per line
                structure_score += 0.2

            # Check for balanced structure
            child_counts = [len(child.children) for child in pyramid.children]
            if child_counts and (max(child_counts) - min(child_counts)) <= 1:
                structure_score += 0.2  # Balanced structure

            # Check content quality at each level
            if (
                pyramid.content and len(pyramid.content) > 50
            ):  # Substantive governing thought
                structure_score += 0.15

            # Check key lines quality
            key_lines_quality = sum(
                1 for child in pyramid.children if len(child.content) > 30
            )
            if (
                key_lines_quality >= len(pyramid.children) * 0.8
            ):  # 80% of key lines are substantial
                structure_score += 0.15

        return min(structure_score, 1.0)

    def _assess_content_quality(self, deliverable: ExecutiveDeliverable) -> float:
        """Assess content quality and clarity"""

        content_score = 0.0

        # Executive summary quality
        exec_summary = deliverable.executive_summary
        if exec_summary:
            content_score += 0.3

            # Check for optimal length (300-500 words for executives)
            word_count = len(exec_summary.split())
            if 200 <= word_count <= 600:
                content_score += 0.1

            # Check for action-oriented language
            action_words = [
                "recommend",
                "implement",
                "achieve",
                "deliver",
                "drive",
                "enable",
            ]
            action_count = sum(
                1 for word in action_words if word in exec_summary.lower()
            )
            if action_count >= 2:
                content_score += 0.1

        # Key recommendations quality
        if deliverable.key_recommendations:
            content_score += 0.2

            # Check for specific, actionable recommendations
            specific_recs = sum(
                1
                for rec in deliverable.key_recommendations
                if len(rec.split()) >= 8 and any(char.isdigit() for char in rec)
            )
            if specific_recs >= len(deliverable.key_recommendations) * 0.6:
                content_score += 0.1

        # Supporting analysis depth
        if deliverable.supporting_analysis:
            content_score += 0.2

        # Implementation roadmap
        if deliverable.implementation_roadmap:
            content_score += 0.1

        return min(content_score, 1.0)

    def _assess_executive_focus(self, deliverable: ExecutiveDeliverable) -> float:
        """Assess executive focus and strategic relevance"""

        executive_score = 0.0

        # Check for strategic language
        exec_summary = deliverable.executive_summary
        if exec_summary:
            strategic_terms = [
                "strategic",
                "competitive",
                "market",
                "growth",
                "transformation",
                "opportunity",
                "advantage",
                "performance",
                "value",
                "roi",
            ]

            strategic_count = sum(
                1 for term in strategic_terms if term in exec_summary.lower()
            )

            executive_score += min(strategic_count * 0.1, 0.4)

        # Check for quantified benefits
        if any(
            "%" in str(item) or "$" in str(item)
            for item in deliverable.key_recommendations
        ):
            executive_score += 0.3

        # Check for timeline and urgency
        timeline_indicators = [
            "months",
            "quarter",
            "year",
            "immediately",
            "urgent",
            "priority",
        ]
        if any(indicator in exec_summary.lower() for indicator in timeline_indicators):
            executive_score += 0.2

        # Check for risk considerations
        risk_indicators = ["risk", "challenge", "mitigation", "contingency"]
        if any(indicator in exec_summary.lower() for indicator in risk_indicators):
            executive_score += 0.1

        return min(executive_score, 1.0)

    def _assess_evidence_strength(self, deliverable: ExecutiveDeliverable) -> float:
        """Assess strength of evidence and analysis"""

        evidence_score = 0.0

        # Check for data-driven insights
        data_indicators = [
            "analysis",
            "data",
            "research",
            "study",
            "benchmark",
            "survey",
        ]
        exec_summary = deliverable.executive_summary

        if exec_summary:
            data_count = sum(
                1 for indicator in data_indicators if indicator in exec_summary.lower()
            )
            evidence_score += min(data_count * 0.1, 0.3)

        # Check for specific metrics
        has_percentages = "%" in exec_summary if exec_summary else False
        has_numbers = (
            any(char.isdigit() for char in exec_summary) if exec_summary else False
        )

        if has_percentages and has_numbers:
            evidence_score += 0.3
        elif has_numbers:
            evidence_score += 0.2

        # Check supporting analysis depth
        if deliverable.supporting_analysis:
            analysis_items = len(deliverable.supporting_analysis)
            evidence_score += min(analysis_items * 0.1, 0.2)

        # Check for appendices (detailed evidence)
        if deliverable.appendices:
            evidence_score += 0.2

        return min(evidence_score, 1.0)

    def _assess_actionability(self, deliverable: ExecutiveDeliverable) -> float:
        """Assess actionability and implementation clarity"""

        actionability_score = 0.0

        # Check for clear recommendations
        if deliverable.key_recommendations:
            actionability_score += 0.3

            # Check for specific actions
            action_verbs = [
                "implement",
                "establish",
                "create",
                "launch",
                "deploy",
                "execute",
            ]
            action_rec_count = sum(
                1
                for rec in deliverable.key_recommendations
                if any(verb in rec.lower() for verb in action_verbs)
            )

            if action_rec_count >= len(deliverable.key_recommendations) * 0.7:
                actionability_score += 0.2

        # Check for implementation roadmap
        if deliverable.implementation_roadmap:
            actionability_score += 0.3

            # Check for phases/timeline
            roadmap_str = str(deliverable.implementation_roadmap).lower()
            if any(
                phase in roadmap_str
                for phase in ["phase", "stage", "step", "milestone"]
            ):
                actionability_score += 0.1

        # Check for success metrics
        success_indicators = ["metric", "kpi", "measure", "track", "monitor"]
        content_to_check = f"{deliverable.executive_summary} {' '.join(deliverable.key_recommendations)}"

        if any(
            indicator in content_to_check.lower() for indicator in success_indicators
        ):
            actionability_score += 0.1

        return min(actionability_score, 1.0)
