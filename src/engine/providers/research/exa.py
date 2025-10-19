"""
Exa Research Provider
Implements the Exa API as a secondary research provider for fallback resilience
Requires: pip install exa-py
"""

import os
import logging
from datetime import datetime
from typing import Dict, Any
from .base import ResearchProvider, ResearchResult, ResearchTier

logger = logging.getLogger(__name__)

# Import with fallback handling for development
try:
    from exa_py import Exa

    EXA_AVAILABLE = True
except ImportError:
    EXA_AVAILABLE = False
    Exa = None


class ExaProvider(ResearchProvider):
    """
    Exa research provider implementing the standard ResearchProvider interface

    This provider uses the Exa API to provide high-quality web search results
    as a fallback when Perplexity is unavailable.

    Note: Requires 'pip install exa-py' to be run in the project environment
    """

    def __init__(self):
        self.exa_client = None
        self.usage_stats = {
            "total_queries": 0,
            "total_cost_usd": 0.0,
            "avg_processing_time_ms": 0.0,
        }
        self._initialize_client()

    def _initialize_client(self):
        """Initialize the Exa client with API key validation"""
        if not EXA_AVAILABLE:
            # logger.error("❌ exa-py library not installed. Run: pip install exa-py")
            return

        api_key = os.getenv("EXA_API_KEY")
        if not api_key:
            logger.error("❌ EXA_API_KEY environment variable not set")
            return

        try:
            self.exa_client = Exa(api_key)
            logger.info("✅ ExaProvider initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Exa client: {e}")
            self.exa_client = None

    @property
    def provider_name(self) -> str:
        """Return the name of this research provider"""
        return "exa"

    async def search(
        self,
        query: str,
        num_results: int = 5,
        tier: ResearchTier = ResearchTier.REGULAR,
    ) -> ResearchResult:
        """
        Execute a search query using Exa API

        Args:
            query: The research question to search for
            num_results: Number of results to return
            tier: Research tier for cost/quality optimization

        Returns:
            ResearchResult: Standardized result structure
        """
        if not self.exa_client:
            raise RuntimeError(
                "ExaProvider not available - check exa-py installation and EXA_API_KEY"
            )

        start_time = datetime.utcnow()

        try:
            # Configure search parameters based on tier
            search_params = self._get_tier_config(tier, num_results)

            # Execute search with content retrieval
            search_response = self.exa_client.search_and_contents(
                query,
                num_results=search_params["num_results"],
                text=True,  # Get clean text content
                **search_params["options"],
            )

            # Process the results
            processing_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            # Extract content and sources
            content_parts = []
            sources = []
            citations = []

            for result in search_response.results:
                # Build content summary
                title = getattr(result, "title", "Untitled")
                text = getattr(result, "text", "")
                url = getattr(result, "url", "")

                if text:
                    # Limit text length based on tier
                    max_text_length = search_params.get("max_text_per_result", 500)
                    truncated_text = text[:max_text_length]
                    if len(text) > max_text_length:
                        truncated_text += "..."

                    content_parts.append(f"**{title}**\n{truncated_text}")

                if url:
                    sources.append(url)
                    citations.append({"title": title, "url": url, "source": "Exa"})

            # Combine all content
            combined_content = "\n\n".join(content_parts)

            # Estimate token usage (rough approximation)
            estimated_tokens = len(combined_content) // 4  # Rough token estimate

            # Estimate cost (Exa pricing varies, using conservative estimate)
            estimated_cost = self._estimate_cost(
                estimated_tokens, len(search_response.results)
            )

            # Calculate confidence based on number and quality of results
            confidence = min(0.95, 0.5 + (len(search_response.results) * 0.1))

            # Update usage statistics
            self._update_usage_stats(
                estimated_tokens, estimated_cost, processing_time_ms
            )

            logger.info(
                f"Exa search successful: {len(search_response.results)} results in {processing_time_ms:.1f}ms"
            )

            return ResearchResult(
                content=combined_content,
                sources=sources,
                confidence=confidence,
                processing_time_ms=processing_time_ms,
                tokens_used=estimated_tokens,
                cost_usd=estimated_cost,
                provider_name=self.provider_name,
                citations=citations,
                timestamp=datetime.utcnow(),
                query_type="web_search",
                success=True,
                error_message=None,
            )

        except Exception as e:
            processing_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.error(f"Exa search failed: {e}")

            # Return failed result for graceful degradation
            return ResearchResult(
                content="",
                sources=[],
                confidence=0.0,
                processing_time_ms=processing_time_ms,
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
        """Check if Exa provider is available"""
        if not EXA_AVAILABLE:
            return False

        if not self.exa_client:
            return False

        try:
            # Test with a minimal query
            test_response = self.exa_client.search("test query", num_results=1)
            return True
        except Exception as e:
            logger.error(f"Exa availability check failed: {e}")
            return False

    def get_usage_metrics(self) -> Dict[str, Any]:
        """Get Exa usage metrics"""
        return {
            "provider": self.provider_name,
            "exa_py_available": EXA_AVAILABLE,
            "client_initialized": self.exa_client is not None,
            **self.usage_stats,
        }

    def _get_tier_config(self, tier: ResearchTier, num_results: int) -> Dict[str, Any]:
        """Get search configuration based on research tier"""
        base_config = {
            "num_results": num_results,
            "options": {},
            "max_text_per_result": 500,
        }

        if tier == ResearchTier.TESTING:
            # Fast, minimal configuration
            base_config.update(
                {
                    "num_results": min(num_results, 3),  # Limit results for speed
                    "max_text_per_result": 200,
                    "options": {
                        "use_autoprompt": False,  # Skip autoprompt for speed
                    },
                }
            )

        elif tier == ResearchTier.PREMIUM:
            # Comprehensive configuration
            base_config.update(
                {
                    "num_results": max(num_results, 8),  # More results for depth
                    "max_text_per_result": 1000,
                    "options": {
                        "use_autoprompt": True,  # Use Exa's autoprompt for better results
                        "type": "neural",  # Use neural search for better semantic matching
                    },
                }
            )

        else:  # REGULAR tier
            # Balanced configuration
            base_config.update(
                {
                    "max_text_per_result": 500,
                    "options": {
                        "use_autoprompt": True,
                        "type": "neural",
                    },
                }
            )

        return base_config

    def _estimate_cost(self, tokens: int, num_results: int) -> float:
        """
        Estimate cost for Exa API usage

        Note: Exa pricing varies by plan. This uses conservative estimates.
        Actual pricing should be verified with current Exa documentation.
        """
        # Conservative cost estimates (check current Exa pricing)
        cost_per_search = 0.003  # ~$3 per 1000 searches
        cost_per_result = 0.001  # ~$1 per 1000 results with content

        search_cost = cost_per_search
        content_cost = num_results * cost_per_result

        return search_cost + content_cost

    def _update_usage_stats(self, tokens: int, cost: float, processing_time_ms: float):
        """Update internal usage statistics"""
        self.usage_stats["total_queries"] += 1
        self.usage_stats["total_cost_usd"] += cost

        # Update rolling average processing time
        total_queries = self.usage_stats["total_queries"]
        current_avg = self.usage_stats["avg_processing_time_ms"]
        self.usage_stats["avg_processing_time_ms"] = (
            current_avg * (total_queries - 1) + processing_time_ms
        ) / total_queries
