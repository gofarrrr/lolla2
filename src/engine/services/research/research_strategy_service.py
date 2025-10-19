"""
Research Strategy Service - Intelligent Research Strategy Selection
=================================================================

REFACTORING TARGET: Extract ResearchStrategyEngine from enhanced_research_orchestrator.py
PATTERN: Service Extraction with Strategy Selection
GOAL: Create focused, testable research strategy service

Responsibility:
- Intelligent decision engine for research strategy selection
- Sonar Deep vs Multi-Query vs Hybrid strategy decisions
- Strategy scoring and explanation system
- Web scraping enablement decisions

Benefits:
- Single Responsibility Principle for strategy decisions
- Easily testable strategy logic
- Clear strategy selection interfaces
- Pluggable strategy algorithms
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ResearchQuery:
    """Research query specification"""

    query: str
    user_preference: str = "balanced"  # aggressive, balanced, conservative
    sophistication_level: str = "advanced"  # basic, intermediate, advanced, expert
    context: Optional[Dict[str, Any]] = None
    max_cost_usd: float = 10.0
    max_time_seconds: int = 300


class ResearchStrategyService:
    """
    Intelligent research strategy selection service

    Responsibility: Strategy decision logic for research execution
    Complexity Target: Grade B (≤10 per method)
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Strategy selection criteria
        self.strategy_criteria = {
            "sonar_deep_indicators": [
                "comprehensive analysis",
                "deep dive",
                "exhaustive research",
                "market intelligence",
                "competitive landscape",
                "strategic assessment",
                "investment evaluation",
                "due diligence",
                "regulatory analysis",
            ],
            "multi_query_indicators": [
                "compare and contrast",
                "multiple perspectives",
                "diverse angles",
                "various viewpoints",
                "different approaches",
                "alternative solutions",
            ],
            "hybrid_indicators": [
                "ultra comprehensive",
                "maximum coverage",
                "exhaustive analysis",
                "complete intelligence",
                "total market scan",
                "full spectrum",
            ],
        }

    def decide_research_strategy(self, research_query: ResearchQuery) -> Dict[str, Any]:
        """
        Main strategy decision method

        Complexity: Target B (≤10)
        """
        query_text = research_query.query.lower()
        user_pref = research_query.user_preference
        sophistication = research_query.sophistication_level

        # Calculate strategy scores
        scores = {
            "sonar_deep": self._calculate_sonar_deep_score(
                query_text, user_pref, sophistication
            ),
            "multi_query": self._calculate_multi_query_score(
                query_text, user_pref, sophistication
            ),
            "hybrid": self._calculate_hybrid_score(
                query_text, user_pref, sophistication
            ),
        }

        # Select best strategy
        best_strategy = max(scores.keys(), key=lambda k: scores[k])

        # Generate strategy configuration
        strategy_config = self._generate_strategy_config(
            best_strategy, research_query, scores
        )

        return {
            "strategy": best_strategy,
            "scores": scores,
            "config": strategy_config,
            "explanation": self._explain_strategy_decision(
                best_strategy, scores, query_text
            ),
        }

    def _calculate_sonar_deep_score(
        self, query_text: str, user_pref: str, sophistication: str
    ) -> float:
        """
        Calculate score for Sonar Deep Research strategy

        Complexity: Target B (≤10)
        """
        score = 0.0

        # Keyword matching for sonar deep indicators
        for indicator in self.strategy_criteria["sonar_deep_indicators"]:
            if indicator in query_text:
                score += 2.0

        # User preference weighting
        if user_pref == "aggressive":
            score += 3.0
        elif user_pref == "balanced":
            score += 1.5

        # Sophistication level weighting
        sophistication_weights = {
            "expert": 3.0,
            "advanced": 2.0,
            "intermediate": 1.0,
            "basic": 0.5,
        }
        score += sophistication_weights.get(sophistication, 1.0)

        return min(score, 10.0)  # Cap at 10

    def _calculate_multi_query_score(
        self, query_text: str, user_pref: str, sophistication: str
    ) -> float:
        """
        Calculate score for Multi-Query strategy

        Complexity: Target B (≤10)
        """
        score = 0.0

        # Keyword matching for multi-query indicators
        for indicator in self.strategy_criteria["multi_query_indicators"]:
            if indicator in query_text:
                score += 2.0

        # User preference weighting
        if user_pref == "conservative":
            score += 3.0
        elif user_pref == "balanced":
            score += 2.0

        # Length-based scoring (longer queries benefit from multi-query)
        if len(query_text.split()) > 20:
            score += 2.0

        return min(score, 10.0)  # Cap at 10

    def _calculate_hybrid_score(
        self, query_text: str, user_pref: str, sophistication: str
    ) -> float:
        """
        Calculate score for Hybrid strategy

        Complexity: Target B (≤10)
        """
        score = 0.0

        # Keyword matching for hybrid indicators
        for indicator in self.strategy_criteria["hybrid_indicators"]:
            if indicator in query_text:
                score += 3.0

        # User preference weighting (hybrid is expensive)
        if user_pref == "aggressive":
            score += 2.0

        # Sophistication requirement (hybrid for expert level)
        if sophistication == "expert":
            score += 2.0

        # Complexity indicator (very complex queries get hybrid)
        complexity_indicators = ["and", "or", "versus", "compared to", "analysis of"]
        complexity_count = sum(1 for ind in complexity_indicators if ind in query_text)
        score += complexity_count * 1.5

        return min(score, 10.0)  # Cap at 10

    def _generate_strategy_config(
        self, strategy: str, research_query: ResearchQuery, scores: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Generate configuration for selected strategy

        Complexity: Target B (≤10)
        """
        base_config = {
            "max_cost_usd": research_query.max_cost_usd,
            "max_time_seconds": research_query.max_time_seconds,
            "sophistication_level": research_query.sophistication_level,
        }

        if strategy == "sonar_deep":
            return {**base_config, "mode": "sonar_deep", "autonomous_searches": True}
        elif strategy == "multi_query":
            return {**base_config, "mode": "multi_query", "query_count": 12}
        elif strategy == "hybrid":
            return {**base_config, "mode": "hybrid", "combine_approaches": True}

        return base_config

    def _explain_strategy_decision(
        self, strategy: str, scores: Dict[str, float], query_text: str
    ) -> str:
        """
        Generate human-readable explanation for strategy choice

        Complexity: Target B (≤10)
        """
        explanations = {
            "sonar_deep": f"Selected Sonar Deep Research (score: {scores['sonar_deep']:.1f}) for comprehensive autonomous multi-search capability.",
            "multi_query": f"Selected Multi-Query Strategy (score: {scores['multi_query']:.1f}) for diverse perspective coverage.",
            "hybrid": f"Selected Hybrid Approach (score: {scores['hybrid']:.1f}) for maximum intelligence gathering.",
        }

        return explanations.get(strategy, "Strategy selection completed.")

    def should_enable_web_scraping(
        self, research_query: ResearchQuery, strategy_decision: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Determine if web scraping should be enabled

        Complexity: Target B (≤10)
        """
        query_text = research_query.query.lower()

        # Web scraping indicators
        web_scraping_indicators = [
            "specific company",
            "particular website",
            "exact data from",
            "latest information",
            "current status",
            "real-time data",
            "website content",
            "page analysis",
            "site intelligence",
        ]

        scraping_score = sum(
            2.0 for indicator in web_scraping_indicators if indicator in query_text
        )

        # Strategy-based weighting
        if strategy_decision["strategy"] == "hybrid":
            scraping_score += 3.0
        elif strategy_decision["strategy"] == "sonar_deep":
            scraping_score += 1.0

        enable_scraping = scraping_score >= 4.0

        return {
            "enable_web_scraping": enable_scraping,
            "scraping_score": scraping_score,
            "reasoning": f"Web scraping {'enabled' if enable_scraping else 'disabled'} (score: {scraping_score:.1f})",
        }


# Singleton instance for injection
_strategy_service_instance = None


def get_research_strategy_service() -> ResearchStrategyService:
    """Factory function for dependency injection"""
    global _strategy_service_instance
    if _strategy_service_instance is None:
        _strategy_service_instance = ResearchStrategyService()
    return _strategy_service_instance
