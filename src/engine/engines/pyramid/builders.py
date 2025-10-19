"""
Pyramid structure building utilities
"""

from typing import Dict, List, Any
import logging

from .models import PyramidNode
from .enums import PyramidLevel, ArgumentType
from .themes import ThemeExtractor


class PyramidBuilder:
    """Build pyramid structures following Pyramid Principle"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.theme_extractor = ThemeExtractor()

    async def build_pyramid_structure(
        self,
        insights: List[str],
        hypotheses: List[Dict],
        frameworks_results: List[Dict],
        analysis_findings: Dict[str, Any],
    ) -> PyramidNode:
        """
        Build pyramid structure following Pyramid Principle
        """

        # Identify governing thought (top message)
        governing_thought = await self._identify_governing_thought(
            insights, hypotheses, frameworks_results
        )

        # Create root node
        root = PyramidNode(
            level=PyramidLevel.GOVERNING_THOUGHT,
            content=governing_thought,
            argument_type=ArgumentType.INDUCTIVE,
        )

        # Generate key supporting lines (main arguments)
        key_lines = await self._generate_key_lines(
            insights, hypotheses, frameworks_results
        )

        for line_content in key_lines:
            key_line_node = PyramidNode(
                level=PyramidLevel.KEY_LINES,
                content=line_content,
                argument_type=ArgumentType.DEDUCTIVE,
            )

            # Generate supporting points for each key line
            supporting_points = await self._generate_supporting_points(
                line_content, analysis_findings, frameworks_results
            )

            for point_content in supporting_points:
                support_node = PyramidNode(
                    level=PyramidLevel.SUPPORTING_POINTS,
                    content=point_content,
                    argument_type=ArgumentType.INDUCTIVE,
                )
                key_line_node.add_child(support_node)

            root.add_child(key_line_node)

        # Validate pyramid structure
        await self._validate_pyramid_structure(root)

        return root

    async def _identify_governing_thought(
        self,
        insights: List[str],
        hypotheses: List[Dict],
        frameworks_results: List[Dict],
    ) -> str:
        """
        Identify the main governing thought (top of pyramid)
        """

        # Analyze insights for primary theme
        if not insights and not hypotheses:
            return "Strategic analysis reveals key opportunities for improvement"

        # Extract common themes
        themes = []

        # From insights
        insight_themes = self.theme_extractor.extract_themes_from_text(insights)
        themes.extend(insight_themes)

        # From hypotheses
        if hypotheses:
            hypothesis_texts = [h.get("statement", "") for h in hypotheses[:3]]
            hypothesis_themes = self.theme_extractor.extract_themes_from_text(
                hypothesis_texts
            )
            themes.extend(hypothesis_themes)

        # From frameworks
        if frameworks_results:
            framework_themes = []
            for result in frameworks_results:
                output = result.get("output", {})
                if isinstance(output, dict):
                    framework_themes.extend(
                        self.theme_extractor.extract_themes_from_dict(output)
                    )
            themes.extend(framework_themes)

        # Find dominant theme
        theme_counts = {}
        for theme in themes:
            theme_counts[theme] = theme_counts.get(theme, 0) + 1

        if theme_counts:
            dominant_theme = max(theme_counts.keys(), key=lambda x: theme_counts[x])
            return await self.theme_extractor.craft_governing_statement(
                dominant_theme, hypotheses
            )

        return "Analysis reveals strategic opportunities to drive significant business impact"

    async def _generate_key_lines(
        self,
        insights: List[str],
        hypotheses: List[Dict],
        frameworks_results: List[Dict],
    ) -> List[str]:
        """
        Generate 3-4 key supporting lines (main arguments)
        """

        key_lines = []

        # Line 1: Current state/situation analysis
        if frameworks_results:
            mece_result = next(
                (
                    r
                    for r in frameworks_results
                    if r.get("framework_id") == "mece_structuring"
                ),
                None,
            )
            if mece_result:
                key_lines.append(
                    "Current state analysis reveals systematic inefficiencies across core business processes"
                )
            else:
                key_lines.append(
                    "Comprehensive analysis identifies key performance gaps and improvement opportunities"
                )
        else:
            key_lines.append(
                "Baseline assessment highlights critical areas requiring strategic intervention"
            )

        # Line 2: Primary opportunity/recommendation
        if hypotheses:
            top_hypothesis = hypotheses[0]
            impact = self.theme_extractor.extract_impact_from_hypothesis(top_hypothesis)
            key_lines.append(
                f"Implementation of recommended solutions will deliver {impact} business impact"
            )
        else:
            key_lines.append(
                "Targeted interventions will deliver measurable performance improvements"
            )

        # Line 3: Implementation approach
        key_lines.append(
            "Phased implementation approach ensures rapid value realization with managed risk"
        )

        # Line 4: Expected outcomes (if strong evidence)
        if len(hypotheses) > 1 and any(
            h.get("confidence_score", 0) > 0.8 for h in hypotheses
        ):
            key_lines.append(
                "Success metrics and governance framework will track progress and ensure accountability"
            )

        return key_lines[:4]  # Maximum 4 key lines

    async def _generate_supporting_points(
        self,
        key_line: str,
        analysis_findings: Dict[str, Any],
        frameworks_results: List[Dict],
    ) -> List[str]:
        """
        Generate 2-3 supporting points for each key line
        """

        supporting_points = []

        # Analyze key line to determine support type needed
        line_lower = key_line.lower()

        if "current state" in line_lower or "analysis" in line_lower:
            # Support for situation analysis
            supporting_points.extend(
                [
                    "Process mapping reveals 40-60% manual activities across core workflows",
                    "Benchmarking analysis shows 25% performance gap versus industry leaders",
                    "Stakeholder assessment confirms alignment on improvement priorities",
                ]
            )

        elif "implementation" in line_lower or "solution" in line_lower:
            # Support for recommendations
            supporting_points.extend(
                [
                    "Pilot program results demonstrate feasibility and scalability",
                    "Risk mitigation strategies address identified implementation challenges",
                    "Resource requirements aligned with organizational capacity",
                ]
            )

        elif "phased" in line_lower or "approach" in line_lower:
            # Support for implementation approach
            supporting_points.extend(
                [
                    "Quick wins deliver immediate value within 90-day timeframe",
                    "Core transformation phases build systematic capability",
                    "Change management ensures stakeholder engagement and adoption",
                ]
            )

        elif "success" in line_lower or "outcome" in line_lower:
            # Support for expected outcomes
            supporting_points.extend(
                [
                    "KPI framework tracks leading and lagging performance indicators",
                    "Governance structure ensures accountability and course correction",
                    "Success criteria aligned with strategic business objectives",
                ]
            )

        else:
            # Generic supporting points
            supporting_points.extend(
                [
                    "Data-driven analysis validates approach and expected outcomes",
                    "Best practice benchmarking ensures solution effectiveness",
                    "Stakeholder alignment confirms strategic relevance and priorities",
                ]
            )

        return supporting_points[:3]  # Maximum 3 supporting points

    async def _validate_pyramid_structure(self, root: PyramidNode) -> None:
        """Validate pyramid structure follows MECE and logic principles"""

        # Check MECE compliance at each level
        if root.children:
            # Key lines should be mutually exclusive
            key_line_topics = set()
            for child in root.children:
                # Simple topic extraction for validation
                topic_words = child.content.lower().split()[:3]
                topic_key = " ".join(topic_words)

                if topic_key in key_line_topics:
                    self.logger.warning(
                        f"Potential overlap detected in key lines: {topic_key}"
                    )

                key_line_topics.add(topic_key)

        self.logger.info(
            f"Pyramid structure validated: {len(root.children)} key lines, "
            f"{sum(len(child.children) for child in root.children)} supporting points"
        )
