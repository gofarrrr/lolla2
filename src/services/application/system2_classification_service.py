"""
System-2 Classification Service

Classifies queries into System-2 cognitive tiers based on complexity,
stakes, and decision characteristics.

Extracted from src/main.py as part of Operation Lean - Target #2.
"""

import logging
from typing import Optional, Dict, Any

from src.services.application.contracts import Tier

logger = logging.getLogger(__name__)


class System2ClassificationService:
    """
    System-2 Kernel Tier Classification Service.

    Classifies queries into cognitive tiers:
    - S2_DISABLED: Simple factual queries requiring minimal cognitive load
    - S2_TIER_1: General questions with moderate complexity
    - S2_TIER_2: Strategic decisions requiring analytical thinking
    - S2_TIER_3: High-stakes, irreversible decisions requiring deep analysis
    """

    def __init__(self):
        """Initialize System-2 classification service"""
        self.tier_keywords = {
            "disabled": ["what is", "capital of", "quick:"],
            "tier_3": [
                "acquire",
                "irreversible",
                "million",
                "billion",
                "security",
                "fraud",
            ],
            "tier_2": [
                "should we",
                "strategy",
                "launch",
                "expansion",
                "market",
            ],
        }
        logger.info("✅ System2ClassificationService initialized")

    def classify_tier(
        self,
        query: str,
        complexity: str = "auto",
        context: Optional[Dict[str, Any]] = None
    ) -> Tier:
        """
        Classify query into System-2 tier based on characteristics.

        Args:
            query: The user query to classify
            complexity: Complexity hint ("auto", "simple", "strategic", "complex")
            context: Optional context for classification (unused currently)

        Returns:
            Tier enum indicating classification level

        Examples:
            >>> service = System2ClassificationService()
            >>> service.classify_tier("What is the capital of France?")
            Tier.DISABLED
            >>> service.classify_tier("Should we launch a new product?")
            Tier.TIER_2
            >>> service.classify_tier("Should we acquire this company for $500 million?")
            Tier.TIER_3
        """
        # Manual complexity override
        if complexity != "auto":
            tier_map = {
                "simple": Tier.DISABLED,
                "strategic": Tier.TIER_2,
                "complex": Tier.TIER_3,
            }
            return tier_map.get(complexity, Tier.TIER_2)

        # Automatic classification based on keywords
        query_lower = query.lower()

        # Check for S2_DISABLED (simple factual queries)
        if any(keyword in query_lower for keyword in self.tier_keywords["disabled"]):
            logger.debug(f"Classified as S2_DISABLED: {query[:50]}...")
            return Tier.DISABLED

        # Check for S2_TIER_3 (high-stakes, irreversible decisions)
        if any(keyword in query_lower for keyword in self.tier_keywords["tier_3"]):
            logger.debug(f"Classified as S2_TIER_3: {query[:50]}...")
            return Tier.TIER_3

        # Check for S2_TIER_2 (strategic decisions)
        if any(keyword in query_lower for keyword in self.tier_keywords["tier_2"]):
            logger.debug(f"Classified as S2_TIER_2: {query[:50]}...")
            return Tier.TIER_2

        # Default to S2_TIER_1 for general questions
        logger.debug(f"Classified as S2_TIER_1 (default): {query[:50]}...")
        return Tier.TIER_1
