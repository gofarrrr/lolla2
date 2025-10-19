"""
Perplexity Research Provider
Adapts the existing PerplexityClient to the standard ResearchProvider interface
"""

import logging
from datetime import datetime
from typing import Dict, Any
from .base import ResearchProvider, ResearchResult, ResearchTier

logger = logging.getLogger(__name__)


class PerplexityProvider(ResearchProvider):
    """
    Perplexity research provider implementing the standard ResearchProvider interface

    This provider wraps the existing PerplexityClient to provide standardized
    research functionality with fallback capability.
    """

    def __init__(self):
        self._client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize the underlying Perplexity client"""
        try:
            from src.engine.integrations.perplexity_client import PerplexityClient

            self._client = PerplexityClient()
            logger.info("✅ PerplexityProvider initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize PerplexityProvider: {e}")
            self._client = None

    @property
    def provider_name(self) -> str:
        """Return the name of this research provider"""
        return "perplexity"

    async def search(
        self,
        query: str,
        num_results: int = 5,
        tier: ResearchTier = ResearchTier.REGULAR,
    ) -> ResearchResult:
        """
        Execute a search query using Perplexity

        Args:
            query: The research question to search for
            num_results: Number of results to return (Perplexity doesn't use this directly)
            tier: Research tier for cost/quality optimization

        Returns:
            ResearchResult: Standardized result structure
        """
        if not self._client:
            raise RuntimeError(
                "PerplexityProvider not available - client initialization failed"
            )

        try:
            # Import the query type enum from the original client
            from src.engine.integrations.perplexity_client import KnowledgeQueryType

            # Map our tier to Perplexity's tier system
            perplexity_tier = self._map_tier_to_perplexity(tier)

            # Execute the query using the existing PerplexityClient
            perplexity_response = await self._client.query_knowledge(
                query=query,
                query_type=KnowledgeQueryType.MARKET_INTELLIGENCE,
                tier=perplexity_tier,
                max_tokens=self._get_max_tokens_for_tier(tier),
                operation_context="ResearchProvider",
            )

            # Convert PerplexityResponse to our standardized ResearchResult
            return ResearchResult(
                content=perplexity_response.content,
                sources=perplexity_response.sources,
                confidence=perplexity_response.confidence,
                processing_time_ms=perplexity_response.processing_time_ms,
                tokens_used=perplexity_response.tokens_used,
                cost_usd=perplexity_response.cost_usd,
                provider_name=self.provider_name,
                citations=(
                    perplexity_response.citations
                    if isinstance(perplexity_response.citations, list)
                    else []
                ),
                timestamp=datetime.utcnow(),
                query_type=perplexity_response.query_type.value,
                success=True,
                error_message=None,
            )

        except Exception as e:
            logger.error(f"Perplexity search failed: {e}")
            # Return failed result instead of raising exception for graceful degradation
            return ResearchResult(
                content="",
                sources=[],
                confidence=0.0,
                processing_time_ms=0.0,
                tokens_used=0,
                cost_usd=0.0,
                provider_name=self.provider_name,
                citations=[],
                timestamp=datetime.utcnow(),
                query_type="failed",
                success=False,
                error_message=str(e),
            )

    async def is_available(self) -> bool:
        """Check if Perplexity provider is available"""
        if not self._client:
            return False

        try:
            return await self._client.is_available()
        except Exception as e:
            logger.error(f"Perplexity availability check failed: {e}")
            return False

    def get_usage_metrics(self) -> Dict[str, Any]:
        """Get Perplexity usage metrics"""
        if not self._client:
            return {
                "provider": self.provider_name,
                "available": False,
                "error": "Client not initialized",
            }

        try:
            metrics = self._client.get_usage_metrics()
            metrics["provider"] = self.provider_name
            return metrics
        except Exception as e:
            logger.error(f"Failed to get Perplexity usage metrics: {e}")
            return {"provider": self.provider_name, "available": False, "error": str(e)}

    def _map_tier_to_perplexity(self, tier: ResearchTier) -> str:
        """Map our research tier to Perplexity's tier system"""
        from src.engine.integrations.perplexity_client import (
            ResearchTier as PerplexityTier,
        )

        tier_mapping = {
            ResearchTier.TESTING: PerplexityTier.TESTING,
            ResearchTier.REGULAR: PerplexityTier.REGULAR,
            ResearchTier.PREMIUM: PerplexityTier.PREMIUM,
        }

        return tier_mapping.get(tier, PerplexityTier.REGULAR)

    def _get_max_tokens_for_tier(self, tier: ResearchTier) -> int:
        """Get appropriate max tokens based on research tier"""
        tier_tokens = {
            ResearchTier.TESTING: 500,  # Fast, lightweight
            ResearchTier.REGULAR: 1000,  # Balanced
            ResearchTier.PREMIUM: 4000,  # Comprehensive
        }

        return tier_tokens.get(tier, 1000)
