"""
Fact and theme extraction utilities for research system
"""

import json
import re
from typing import List, Dict, Any
import logging


class FactExtractor:
    """Extract factual claims from research sources"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Common fact patterns
        self.fact_patterns = [
            r"(\d+(?:\.\d+)?)\s*%\s+of\s+([^.]+)",  # Percentage claims
            r"(\d+(?:,\d{3})*(?:\.\d+)?)\s+(million|billion|trillion|thousand)\s+([^.]+)",  # Large number claims
            r"According to\s+([^,]+),\s*([^.]+)",  # Attribution statements
            r"Studies show\s+(?:that\s+)?([^.]+)",  # Study claims
            r"Research indicates\s+(?:that\s+)?([^.]+)",  # Research claims
            r"Data suggests\s+(?:that\s+)?([^.]+)",  # Data claims
            r"(\w+)\s+reported\s+([^.]+)",  # Organization reporting
            r"In\s+(\d{4}),\s+([^.]+)",  # Year-specific claims
        ]

    def extract_facts_from_sources(self, sources: List[Dict[str, Any]]) -> List[str]:
        """Extract factual claims from research sources"""
        facts = []

        for source in sources:
            text = f"{source.get('title', '')} {source.get('content', '')}"

            # Extract facts using patterns
            for pattern in self.fact_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    fact = match.group(0).strip()
                    if len(fact) > 20 and len(fact) < 200:  # Reasonable fact length
                        facts.append(fact)

        return list(set(facts))  # Remove duplicates

    def verify_facts_across_sources(
        self, facts: List[str], sources: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Verify facts by checking consistency across sources"""
        fact_verification = {}

        for fact in facts:
            verification_score = 0.0
            supporting_sources = 0

            # Extract key elements from the fact for matching
            fact_lower = fact.lower()
            fact_words = set(re.findall(r"\w+", fact_lower))

            # Check how many sources support this fact
            for source in sources:
                source_text = (
                    f"{source.get('title', '')} {source.get('content', '')}".lower()
                )
                source_words = set(re.findall(r"\w+", source_text))

                # Calculate word overlap
                overlap = (
                    len(fact_words & source_words) / len(fact_words)
                    if fact_words
                    else 0
                )

                # If significant overlap, count as supporting source
                if overlap > 0.3:  # 30% word overlap threshold
                    supporting_sources += 1

                    # Check for numeric agreement if fact contains numbers
                    fact_numbers = re.findall(r"\d+(?:\.\d+)?", fact)
                    source_numbers = re.findall(r"\d+(?:\.\d+)?", source_text)

                    if fact_numbers and source_numbers:
                        # Look for similar numbers (within 10% tolerance)
                        for fact_num in fact_numbers:
                            for source_num in source_numbers:
                                try:
                                    f_val = float(fact_num)
                                    s_val = float(source_num)
                                    if abs(f_val - s_val) / max(f_val, s_val, 1) <= 0.1:
                                        verification_score += 0.3
                                except ValueError:
                                    continue

            # Base verification on number of supporting sources
            if supporting_sources >= 3:
                verification_score += 0.8
            elif supporting_sources >= 2:
                verification_score += 0.6
            elif supporting_sources >= 1:
                verification_score += 0.4

            # Cap at 1.0
            fact_verification[fact] = min(1.0, verification_score)

        return fact_verification


class ThemeExtractor:
    """Extract themes and topics from text content"""

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
