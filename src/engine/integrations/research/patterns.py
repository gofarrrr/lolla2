"""
Research pattern selection and auto-detection utilities
"""

from typing import List
import logging


class PatternSelector:
    """Automatically select appropriate research patterns based on problem content"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Pattern detection rules based on keywords and context
        self.pattern_rules = {
            "market_analysis": [
                "market",
                "pricing",
                "price",
                "cost",
                "demand",
                "supply",
                "size",
                "segment",
                "opportunity",
                "growth",
                "trend",
            ],
            "competitive_intelligence": [
                "compete",
                "competitor",
                "competition",
                "rival",
                "versus",
                "vs",
                "market share",
                "positioning",
                "differentiat",
                "benchmark",
            ],
            "investment_evaluation": [
                "invest",
                "investment",
                "fund",
                "funding",
                "capital",
                "roi",
                "return",
                "valuation",
                "acquisition",
                "merger",
                "due diligence",
            ],
            "technology_trends": [
                "technology",
                "tech",
                "innovation",
                "digital",
                "ai",
                "artificial intelligence",
                "automation",
                "software",
                "platform",
                "cloud",
                "data",
            ],
            "marketing_strategy": [
                "marketing",
                "brand",
                "customer",
                "audience",
                "campaign",
                "channel",
                "retention",
                "acquisition",
                "engagement",
                "conversion",
            ],
            "geographic_intelligence": [
                "expand",
                "expansion",
                "international",
                "global",
                "region",
                "country",
                "market entry",
                "geographic",
                "location",
                "territory",
            ],
            "sustainability_analysis": [
                "sustainability",
                "sustainable",
                "green",
                "environment",
                "carbon",
                "esg",
                "climate",
                "renewable",
                "circular economy",
            ],
            "policy_impact": [
                "regulation",
                "regulatory",
                "policy",
                "government",
                "compliance",
                "legal",
                "law",
                "legislation",
                "political",
            ],
            "startup_feasibility": [
                "startup",
                "launch",
                "new business",
                "feasibility",
                "viable",
                "entrepreneur",
                "mvp",
                "product market fit",
            ],
            "health_research": [
                "health",
                "healthcare",
                "medical",
                "pharma",
                "drug",
                "treatment",
                "clinical",
                "patient",
                "disease",
                "therapy",
            ],
        }

    def auto_select_research_patterns(self, problem_statement: str) -> List[str]:
        """
        Automatically select appropriate research patterns based on problem statement content
        Returns list of pattern names that should be used
        """
        statement = problem_statement.lower()
        selected_patterns = []

        # Score each pattern based on keyword matches
        pattern_scores = {}
        for pattern_name, keywords in self.pattern_rules.items():
            score = 0
            for keyword in keywords:
                if keyword in statement:
                    # Higher score for exact matches of multi-word terms
                    if " " in keyword:
                        score += 3
                    else:
                        score += 1
            pattern_scores[pattern_name] = score

        # Select patterns with scores above threshold
        threshold = 1
        primary_patterns = [
            pattern for pattern, score in pattern_scores.items() if score >= threshold
        ]

        # Always include market_analysis as a fallback for business questions
        if not primary_patterns and any(
            word in statement
            for word in ["business", "company", "strategy", "strategic"]
        ):
            primary_patterns.append("market_analysis")

        # Sort by score and take top patterns
        sorted_patterns = sorted(
            primary_patterns, key=lambda p: pattern_scores.get(p, 0), reverse=True
        )

        # Limit to 3 patterns maximum to avoid overwhelming the analysis
        selected_patterns = sorted_patterns[:3]

        if not selected_patterns:
            # Default fallback
            selected_patterns = ["market_analysis"]

        self.logger.info(f"🎯 Auto-selected research patterns: {selected_patterns}")
        return selected_patterns
