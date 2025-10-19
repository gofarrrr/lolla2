"""
Research Provider Registry - Multi-Provider Research Fallback

Implements pluggable research provider pattern with automatic fallback.
Prevents single point of failure in research retrieval.

Architecture:
- Protocol-based provider interface
- Round-robin fallback with early stopping
- Hit deduplication across providers
- Graceful degradation

Supported Providers:
- Perplexity (primary)
- Exa (secondary) - TODO: Implement
- Tavily (tertiary) - TODO: Implement
"""

from typing import Protocol, List, Dict, Any, Optional
from dataclasses import dataclass
import logging


logger = logging.getLogger(__name__)


@dataclass
class SearchHit:
    """Single search result hit"""
    url: str
    title: str
    content: str
    domain: str
    date: Optional[str] = None
    confidence: float = 0.5


class ResearchProvider(Protocol):
    """Protocol for research providers"""

    async def query(
        self, q: str, context: Optional[Dict[str, Any]] = None, **kwargs
    ) -> List[SearchHit]:
        """
        Query research provider.

        Args:
            q: Query string
            context: Optional context dict
            **kwargs: Provider-specific parameters

        Returns:
            List of SearchHit results
        """
        ...


class PerplexityResearchProvider:
    """Perplexity research provider wrapper"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._client = None

    async def _get_client(self):
        """Lazy load Perplexity client"""
        if self._client is None:
            try:
                from src.engine.integrations.perplexity_client import (
                    get_perplexity_client,
                    KnowledgeQueryType,
                )

                self._client = await get_perplexity_client()
                self.KnowledgeQueryType = KnowledgeQueryType
            except Exception as e:
                self.logger.error(f"❌ Failed to load Perplexity client: {e}")
                raise

        return self._client

    async def query(
        self, q: str, context: Optional[Dict[str, Any]] = None, **kwargs
    ) -> List[SearchHit]:
        """Query Perplexity for research"""
        try:
            client = await self._get_client()

            response = await client.query_knowledge(
                query=q,
                query_type=self.KnowledgeQueryType.CONTEXT_GROUNDING,
                max_tokens=kwargs.get("max_tokens", 500),
            )

            # Convert to SearchHit format
            hits = []
            for i, source_url in enumerate(response.sources[:5]):
                domain = self._extract_domain(source_url)
                hit = SearchHit(
                    url=source_url,
                    title=f"Source {i+1}",
                    content=response.content[:200] if i == 0 else "",
                    domain=domain,
                    confidence=response.confidence,
                )
                hits.append(hit)

            return hits

        except Exception as e:
            self.logger.error(f"❌ Perplexity query failed: {e}")
            return []

    @staticmethod
    def _extract_domain(url: str) -> str:
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse

            parsed = urlparse(url)
            return parsed.netloc
        except Exception:
            return "unknown"


class ExaResearchProvider:
    """Exa research provider (placeholder for future implementation)"""

    async def query(
        self, q: str, context: Optional[Dict[str, Any]] = None, **kwargs
    ) -> List[SearchHit]:
        """Query Exa for research"""
        # TODO: Implement Exa API integration
        logger.warning("⚠️ Exa provider not yet implemented")
        return []


class TavilyResearchProvider:
    """Tavily research provider (placeholder for future implementation)"""

    async def query(
        self, q: str, context: Optional[Dict[str, Any]] = None, **kwargs
    ) -> List[SearchHit]:
        """Query Tavily for research"""
        # TODO: Implement Tavily API integration
        logger.warning("⚠️ Tavily provider not yet implemented")
        return []


class ResearchProviderRegistry:
    """Registry for research providers with fallback logic"""

    def __init__(self):
        self.providers: List[ResearchProvider] = []
        self.logger = logging.getLogger(__name__)

        # Register default providers
        self._register_default_providers()

    def _register_default_providers(self):
        """Register default research providers in priority order"""
        try:
            self.providers.append(PerplexityResearchProvider())
            self.logger.info("✅ Perplexity provider registered")
        except Exception:
            self.logger.warning("⚠️ Perplexity provider unavailable")

        # Future providers (currently placeholders)
        # self.providers.append(ExaResearchProvider())
        # self.providers.append(TavilyResearchProvider())

    def register_provider(self, provider: ResearchProvider):
        """Register a custom research provider"""
        self.providers.append(provider)
        self.logger.info(f"✅ Custom provider registered: {provider.__class__.__name__}")

    async def query_any(
        self,
        q: str,
        context: Optional[Dict[str, Any]] = None,
        early_stop_domains: int = 3,
        **kwargs,
    ) -> List[SearchHit]:
        """
        Query research providers with automatic fallback and early stopping.

        Args:
            q: Query string
            context: Optional context dict
            early_stop_domains: Stop after this many unique domains (default: 3)
            **kwargs: Provider-specific parameters

        Returns:
            Deduplicated list of SearchHit results
        """
        all_hits: List[SearchHit] = []
        unique_domains = set()
        calls_made = 0

        for provider in self.providers:
            try:
                self.logger.info(f"🔍 Querying {provider.__class__.__name__}...")

                hits = await provider.query(q, context, **kwargs)
                all_hits.extend(hits)

                # Track unique domains
                for hit in hits:
                    unique_domains.add(hit.domain)

                calls_made += 1

                # Early stopping: If we have enough unique domains, stop
                if len(unique_domains) >= early_stop_domains:
                    self.logger.info(
                        f"✋ Early stopping: {len(unique_domains)} unique domains from {calls_made} providers"
                    )
                    break

            except Exception as e:
                self.logger.warning(
                    f"⚠️ Provider {provider.__class__.__name__} failed: {e}"
                )
                continue

        # Deduplicate by URL
        deduplicated_hits = self._deduplicate_hits(all_hits)

        self.logger.info(
            f"✅ Research complete: {len(deduplicated_hits)} unique hits from {len(unique_domains)} domains"
        )

        return deduplicated_hits[:10]  # Return top 10

    @staticmethod
    def _deduplicate_hits(hits: List[SearchHit]) -> List[SearchHit]:
        """Deduplicate hits by URL"""
        seen_urls = set()
        unique_hits = []

        for hit in hits:
            if hit.url not in seen_urls:
                seen_urls.add(hit.url)
                unique_hits.append(hit)

        return unique_hits


# Global registry instance
_research_registry: Optional[ResearchProviderRegistry] = None


def get_research_provider_registry() -> ResearchProviderRegistry:
    """Get or create global research provider registry"""
    global _research_registry

    if _research_registry is None:
        _research_registry = ResearchProviderRegistry()

    return _research_registry
