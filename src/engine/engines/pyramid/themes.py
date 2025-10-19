"""
Theme extraction utilities for Pyramid Principle Engine
"""

import json
import re
from typing import List, Dict, Any
import logging


class ThemeExtractor:
    """Extract themes and patterns from analysis content"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Common business themes
        self.theme_patterns = {
            "efficiency": ["efficiency", "optimization", "streamlin", "automat"],
            "growth": ["growth", "expansion", "increase", "scale"],
            "cost": ["cost", "reduction", "saving", "expense"],
            "digital": ["digital", "technology", "platform", "system"],
            "market": ["market", "customer", "competitive", "position"],
            "organization": ["organization", "people", "culture", "skill"],
        }

        # Situation/Complication/Resolution patterns
        self.situation_patterns = [
            "current state",
            "baseline",
            "as-is",
            "existing",
            "today",
        ]
        self.complication_patterns = [
            "challenge",
            "problem",
            "issue",
            "gap",
            "constraint",
            "risk",
        ]
        self.resolution_patterns = [
            "solution",
            "recommendation",
            "approach",
            "strategy",
            "plan",
        ]

    def extract_themes_from_text(self, texts: List[str]) -> List[str]:
        """Extract key themes from text content"""
        themes = []

        text_content = " ".join(texts).lower()

        for theme, patterns in self.theme_patterns.items():
            if any(pattern in text_content for pattern in patterns):
                themes.append(theme)

        return themes

    def extract_themes_from_dict(self, data: Dict[str, Any]) -> List[str]:
        """Extract themes from dictionary data"""
        text_content = json.dumps(data, default=str).lower()
        return self.extract_themes_from_text([text_content])

    def extract_impact_from_hypothesis(self, hypothesis: Dict) -> str:
        """Extract business impact description from hypothesis"""
        statement = hypothesis.get("statement", "")

        # Look for percentage improvements
        percentages = re.findall(r"(\d+(?:\.\d+)?)\s*%", statement)
        if percentages:
            return f"{percentages[0]}% performance"

        # Look for impact keywords
        if any(
            word in statement.lower()
            for word in ["significant", "substantial", "major"]
        ):
            return "significant"
        elif any(word in statement.lower() for word in ["moderate", "meaningful"]):
            return "meaningful"
        else:
            return "measurable"

    async def craft_governing_statement(
        self, theme: str, hypotheses: List[Dict]
    ) -> str:
        """Craft executive-level governing statement"""

        # Theme-based governing statements
        statements = {
            "efficiency": "Operational efficiency improvements can deliver 25-35% cost reduction while enhancing service quality",
            "growth": "Strategic growth initiatives present opportunity to increase revenue by 30-40% over 24 months",
            "cost": "Cost optimization program can achieve 20-30% savings through systematic process improvements",
            "digital": "Digital transformation initiative will drive 40% improvement in operational performance",
            "market": "Market positioning strategy can capture 15-20% additional market share in target segments",
            "organization": "Organizational excellence program will enhance capability and drive sustained performance",
        }

        # Add quantitative elements from top hypothesis if available
        if hypotheses:
            top_hypothesis = hypotheses[0]
            statement = top_hypothesis.get("statement", "")

            # Extract percentage improvements
            percentages = re.findall(r"(\d+(?:\.\d+)?)\s*%", statement)
            if percentages:
                pct = percentages[0]
                return statements.get(theme, "Strategic initiative").replace(
                    "25-35%", f"{pct}%"
                )

        return statements.get(
            theme,
            "Analysis reveals significant opportunities for strategic improvement",
        )
