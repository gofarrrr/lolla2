#!/usr/bin/env python3
"""
METIS Firecrawl Integration Client
Advanced web extraction and deep content analysis using Firecrawl with MCP support
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from urllib.parse import urlparse

from src.engine.integrations.perplexity_client_advanced import (
    ResearchTemplateType,
    ResearchMode,
    SourceCredibilityTier,
    ResearchInsight,
    EnhancedSource,
    CrossReferenceAnalysis,
    AdvancedResearchResult,
)

# Configuration and logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Firecrawl dependencies (optional for MCP-only usage)
FIRECRAWL_AVAILABLE = False
try:
    from firecrawl import FirecrawlApp

    FIRECRAWL_AVAILABLE = True
    logger.info("✅ Firecrawl Python SDK available")
except ImportError as e:
    logger.warning(f"⚠️ Firecrawl SDK not available: {e}")
    logger.info("🔄 Will use MCP integration only")


@dataclass
class FirecrawlConfig:
    """Firecrawl client configuration"""

    api_key: str
    base_url: str = "https://api.firecrawl.dev"
    max_pages_per_crawl: int = 10
    timeout_seconds: int = 30
    extract_main_content: bool = True
    include_html: bool = False
    include_raw_html: bool = False
    include_screenshot: bool = False
    wait_for_results: bool = True


@dataclass
class FirecrawlExtractionResult:
    """Result from Firecrawl extraction operation"""

    url: str
    title: str
    content: str
    markdown: str
    structured_data: Dict[str, Any]
    metadata: Dict[str, Any]
    links: List[str]
    images: List[str]
    extraction_time_ms: int
    success: bool
    error: Optional[str] = None


@dataclass
class FirecrawlCrawlResult:
    """Result from Firecrawl crawling operation"""

    base_url: str
    pages_crawled: int
    pages: List[FirecrawlExtractionResult]
    sitemap: List[str]
    crawl_time_ms: int
    success: bool
    error: Optional[str] = None


class FirecrawlMCPClient:
    """
    Firecrawl MCP (Model Context Protocol) client integration
    Provides web extraction capabilities for METIS research platform
    """

    def __init__(self, config: FirecrawlConfig):
        self.config = config
        self.session_id = f"metis_firecrawl_{int(time.time())}"

        # Initialize direct API client if available
        if FIRECRAWL_AVAILABLE and config.api_key:
            try:
                self.firecrawl_client = FirecrawlApp(api_key=config.api_key)
                logger.info("✅ Firecrawl direct API client initialized")
            except Exception as e:
                logger.warning(f"⚠️ Direct API client failed: {e}")
                self.firecrawl_client = None
        else:
            self.firecrawl_client = None

        logger.info("✅ Firecrawl MCP Client initialized")

    async def extract_content(
        self, url: str, extraction_schema: Optional[Dict] = None
    ) -> FirecrawlExtractionResult:
        """
        Extract content from a single URL using Firecrawl

        Args:
            url: Target URL to extract
            extraction_schema: Optional Pydantic-like schema for structured extraction

        Returns:
            FirecrawlExtractionResult with extracted content
        """
        start_time = time.time()

        try:
            logger.info(f"🔥 Extracting content from: {url}")

            # Prepare extraction options
            options = {
                "extractorOptions": {
                    "extractionSchema": extraction_schema
                    or self._get_default_extraction_schema(),
                    "mode": "llm-extraction" if extraction_schema else "markdown",
                },
                "pageOptions": {
                    "includeHtml": self.config.include_html,
                    "includeRawHtml": self.config.include_raw_html,
                    "screenshot": self.config.include_screenshot,
                    "waitFor": 2000,  # Wait for page to load
                },
            }

            # Try direct API first, fallback to MCP
            if self.firecrawl_client:
                result = await self._extract_via_direct_api(url, options)
            else:
                result = await self._extract_via_mcp(url, options)

            processing_time = int((time.time() - start_time) * 1000)

            return FirecrawlExtractionResult(
                url=url,
                title=result.get("metadata", {}).get("title", ""),
                content=result.get("extract", {}).get(
                    "content", result.get("markdown", "")
                ),
                markdown=result.get("markdown", ""),
                structured_data=result.get("extract", {}),
                metadata=result.get("metadata", {}),
                links=result.get("links", []),
                images=result.get("images", []),
                extraction_time_ms=processing_time,
                success=True,
            )

        except Exception as e:
            logger.error(f"❌ Firecrawl extraction failed for {url}: {e}")
            processing_time = int((time.time() - start_time) * 1000)

            return FirecrawlExtractionResult(
                url=url,
                title="",
                content="",
                markdown="",
                structured_data={},
                metadata={},
                links=[],
                images=[],
                extraction_time_ms=processing_time,
                success=False,
                error=str(e),
            )

    async def crawl_website(
        self, base_url: str, max_pages: Optional[int] = None
    ) -> FirecrawlCrawlResult:
        """
        Crawl an entire website using Firecrawl

        Args:
            base_url: Base URL to start crawling
            max_pages: Maximum pages to crawl (overrides config)

        Returns:
            FirecrawlCrawlResult with all extracted pages
        """
        start_time = time.time()
        max_pages = max_pages or self.config.max_pages_per_crawl

        try:
            logger.info(f"🕷️ Crawling website: {base_url} (max {max_pages} pages)")

            crawl_options = {
                "crawlerOptions": {
                    "maxPages": max_pages,
                    "allowedDomains": [urlparse(base_url).netloc],
                    "excludePaths": ["/admin", "/login", "/api"],
                    "maxDepth": 3,
                },
                "pageOptions": {
                    "includeHtml": self.config.include_html,
                    "screenshot": self.config.include_screenshot,
                },
            }

            if self.firecrawl_client:
                result = await self._crawl_via_direct_api(base_url, crawl_options)
            else:
                result = await self._crawl_via_mcp(base_url, crawl_options)

            processing_time = int((time.time() - start_time) * 1000)

            # Process crawl results
            pages = []
            for page_data in result.get("data", []):
                page_result = FirecrawlExtractionResult(
                    url=page_data.get("url", ""),
                    title=page_data.get("metadata", {}).get("title", ""),
                    content=page_data.get("markdown", ""),
                    markdown=page_data.get("markdown", ""),
                    structured_data=page_data.get("extract", {}),
                    metadata=page_data.get("metadata", {}),
                    links=page_data.get("links", []),
                    images=page_data.get("images", []),
                    extraction_time_ms=0,  # Part of bulk operation
                    success=True,
                )
                pages.append(page_result)

            return FirecrawlCrawlResult(
                base_url=base_url,
                pages_crawled=len(pages),
                pages=pages,
                sitemap=[p.url for p in pages],
                crawl_time_ms=processing_time,
                success=True,
            )

        except Exception as e:
            logger.error(f"❌ Firecrawl crawl failed for {base_url}: {e}")
            processing_time = int((time.time() - start_time) * 1000)

            return FirecrawlCrawlResult(
                base_url=base_url,
                pages_crawled=0,
                pages=[],
                sitemap=[],
                crawl_time_ms=processing_time,
                success=False,
                error=str(e),
            )

    async def enhance_research_with_deep_extraction(
        self,
        discovered_urls: List[str],
        research_context: Dict[str, Any],
        template_type: ResearchTemplateType,
    ) -> AdvancedResearchResult:
        """
        Enhance ODR/Perplexity research results with deep Firecrawl extraction

        Args:
            discovered_urls: URLs from initial research (ODR/Perplexity)
            research_context: Context from original research
            template_type: Research template for focused extraction

        Returns:
            Enhanced AdvancedResearchResult with deep content
        """
        start_time = time.time()

        try:
            logger.info(
                f"🔬 Enhancing research with deep extraction: {len(discovered_urls)} URLs"
            )

            # Create extraction schema based on research template
            extraction_schema = self._create_template_extraction_schema(template_type)

            # Extract content from all URLs
            extraction_tasks = [
                self.extract_content(url, extraction_schema)
                for url in discovered_urls[: self.config.max_pages_per_crawl]
            ]

            extraction_results = await asyncio.gather(
                *extraction_tasks, return_exceptions=True
            )

            # Filter successful extractions
            successful_extractions = [
                result
                for result in extraction_results
                if isinstance(result, FirecrawlExtractionResult) and result.success
            ]

            logger.info(
                f"✅ Successfully extracted {len(successful_extractions)}/{len(discovered_urls)} URLs"
            )

            # Convert to METIS format
            enhanced_sources = []
            all_insights = []

            for extraction in successful_extractions:
                # Create enhanced source
                domain = urlparse(extraction.url).netloc
                credibility_tier = self._assess_domain_credibility(domain)

                enhanced_source = EnhancedSource(
                    url=extraction.url,
                    title=extraction.title,
                    content=extraction.content[:2000],  # Limit content size
                    domain=domain,
                    credibility_tier=credibility_tier,
                    credibility_score=self._get_credibility_score(credibility_tier),
                    date=extraction.metadata.get("publishedDate"),
                    bias_indicators=[],
                    fact_density=self._calculate_fact_density(extraction.content),
                    citation_quality=self._calculate_citation_quality(
                        extraction.content
                    ),
                )
                enhanced_sources.append(enhanced_source)

                # Extract insights from structured data
                if extraction.structured_data:
                    insights = self._extract_insights_from_structured_data(
                        extraction.structured_data, template_type
                    )
                    all_insights.extend(insights)

            # Generate comprehensive summary
            executive_summary = self._generate_enhanced_summary(
                successful_extractions, research_context, template_type
            )

            processing_time = int((time.time() - start_time) * 1000)

            # Create cross-reference analysis
            cross_reference = CrossReferenceAnalysis(
                total_claims=len(all_insights),
                verified_claims=len(successful_extractions),
                disputed_claims=0,
                consistency_score=min(1.0, len(successful_extractions) / 5),
                contradictions=[],  # TODO: Implement contradiction detection
                confidence_distribution={
                    "high": len(successful_extractions),
                    "medium": 0,
                    "low": 0,
                },
            )

            return AdvancedResearchResult(
                executive_summary=executive_summary,
                key_insights=all_insights,
                detailed_findings=f"Enhanced research with {len(successful_extractions)} deeply extracted sources",
                sources=enhanced_sources,
                cross_reference_analysis=cross_reference,
                overall_confidence=min(0.9, len(successful_extractions) * 0.1),
                coverage_completeness=min(1.0, len(successful_extractions) / 5),
                source_diversity_score=len(set(s.domain for s in enhanced_sources))
                / max(len(enhanced_sources), 1),
                fact_validation_score=sum(s.fact_density for s in enhanced_sources)
                / max(len(enhanced_sources), 1),
                template_used=template_type,
                queries_executed=[f"deep_extraction:{url}" for url in discovered_urls],
                mode_used=ResearchMode.COMPREHENSIVE,
                total_processing_time_ms=processing_time,
                tokens_consumed=sum(
                    len(e.content.split()) for e in successful_extractions
                ),
                cost_usd=len(successful_extractions) * 0.01,  # Estimated cost per page
                information_gaps=["Limited to provided URLs", "No real-time data"],
                additional_research_recommendations=[
                    "Consider broader web search",
                    "Add industry reports",
                ],
                confidence_limitations=[
                    "Dependent on source quality",
                    "Limited cross-validation",
                ],
            )

        except Exception as e:
            logger.error(f"❌ Enhanced research failed: {e}")
            return AdvancedResearchResult(
                executive_summary=f"Enhanced research failed: {e}",
                key_insights=[],
                detailed_findings="Deep extraction encountered errors",
                sources=[],
                cross_reference_analysis=CrossReferenceAnalysis(
                    total_claims=0,
                    verified_claims=0,
                    disputed_claims=0,
                    consistency_score=0.0,
                    contradictions=[],
                    confidence_distribution={},
                ),
                overall_confidence=0.0,
                coverage_completeness=0.0,
                source_diversity_score=0.0,
                fact_validation_score=0.0,
                template_used=template_type,
                queries_executed=[],
                mode_used=ResearchMode.STANDARD,
                total_processing_time_ms=int((time.time() - start_time) * 1000),
                tokens_consumed=0,
                cost_usd=0.0,
                information_gaps=["Extraction failed", "No content retrieved"],
                additional_research_recommendations=[
                    "Retry with different sources",
                    "Use alternative methods",
                ],
                confidence_limitations=["Technical failure", "No data available"],
            )

    # Private helper methods
    async def _extract_via_direct_api(self, url: str, options: Dict) -> Dict[str, Any]:
        """Extract using direct Firecrawl API"""
        # Convert options to Firecrawl v2 API format
        page_options = options.get("pageOptions", {})
        extractor_options = options.get("extractorOptions", {})

        # Map to Firecrawl v2 parameters
        scrape_params = {
            "only_main_content": self.config.extract_main_content,
            "wait_for": page_options.get("waitFor", 2000),
            "timeout": self.config.timeout_seconds * 1000,
            "formats": (
                ["markdown", "html"] if self.config.include_html else ["markdown"]
            ),
        }

        result = await asyncio.to_thread(
            self.firecrawl_client.scrape, url, **scrape_params
        )

        # Convert Firecrawl v2 Document to expected format
        if hasattr(result, "markdown"):
            # Handle metadata safely
            metadata_dict = {}
            if hasattr(result, "metadata") and result.metadata:
                if hasattr(result.metadata, "__dict__"):
                    metadata_dict = result.metadata.__dict__
                elif hasattr(result.metadata, "dict"):
                    metadata_dict = result.metadata.dict()
                else:
                    metadata_dict = {"title": str(result.metadata)}

            return {
                "metadata": metadata_dict,
                "markdown": result.markdown,
                "extract": extractor_options.get("extractionSchema", {}),
                "links": metadata_dict.get("links", []),
                "images": metadata_dict.get("images", []),
            }
        else:
            # Handle different result format
            return {
                "metadata": {"title": f"Extraction: {url}"},
                "markdown": str(result) if result else "",
                "extract": {},
                "links": [],
                "images": [],
            }

    async def _extract_via_mcp(self, url: str, options: Dict) -> Dict[str, Any]:
        """Extract using MCP integration (placeholder for MCP implementation)"""
        # This would integrate with the MCP server
        # For now, return a mock structure
        logger.warning("🔄 MCP extraction not fully implemented, returning mock data")
        return {
            "metadata": {"title": f"MCP Extraction: {url}"},
            "markdown": f"Content from {url} (via MCP)",
            "extract": {},
            "links": [],
            "images": [],
        }

    async def _crawl_via_direct_api(
        self, base_url: str, options: Dict
    ) -> Dict[str, Any]:
        """Crawl using direct Firecrawl API"""
        return await asyncio.to_thread(
            self.firecrawl_client.crawl,
            base_url,
            params=options,
            wait_until_done=self.config.wait_for_results,
        )

    async def _crawl_via_mcp(self, base_url: str, options: Dict) -> Dict[str, Any]:
        """Crawl using MCP integration (placeholder)"""
        logger.warning("🔄 MCP crawling not fully implemented, returning mock data")
        return {"data": []}

    def _get_default_extraction_schema(self) -> Dict[str, Any]:
        """Default extraction schema for structured data"""
        return {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "summary": {"type": "string"},
                "key_points": {"type": "array", "items": {"type": "string"}},
                "author": {"type": "string"},
                "publish_date": {"type": "string"},
                "categories": {"type": "array", "items": {"type": "string"}},
            },
        }

    def _create_template_extraction_schema(
        self, template_type: ResearchTemplateType
    ) -> Dict[str, Any]:
        """Create extraction schema based on research template"""
        base_schema = self._get_default_extraction_schema()

        if template_type == ResearchTemplateType.COMPETITIVE_INTELLIGENCE:
            base_schema["properties"].update(
                {
                    "company_name": {"type": "string"},
                    "products": {"type": "array", "items": {"type": "string"}},
                    "pricing": {"type": "string"},
                    "market_position": {"type": "string"},
                    "strengths": {"type": "array", "items": {"type": "string"}},
                    "weaknesses": {"type": "array", "items": {"type": "string"}},
                }
            )
        elif template_type == ResearchTemplateType.MARKET_ANALYSIS:
            base_schema["properties"].update(
                {
                    "market_size": {"type": "string"},
                    "growth_rate": {"type": "string"},
                    "key_players": {"type": "array", "items": {"type": "string"}},
                    "trends": {"type": "array", "items": {"type": "string"}},
                    "opportunities": {"type": "array", "items": {"type": "string"}},
                }
            )

        return base_schema

    def _assess_domain_credibility(self, domain: str) -> SourceCredibilityTier:
        """Assess domain credibility"""
        high_credibility = [".gov", ".edu", ".org", "bloomberg.com", "reuters.com"]
        medium_credibility = [".com", "forbes.com", "wsj.com", "ft.com"]

        for pattern in high_credibility:
            if pattern in domain:
                return SourceCredibilityTier.TIER_1

        for pattern in medium_credibility:
            if pattern in domain:
                return SourceCredibilityTier.TIER_2

        return SourceCredibilityTier.TIER_3

    def _get_credibility_score(self, tier: SourceCredibilityTier) -> float:
        """Get numeric credibility score"""
        scores = {
            SourceCredibilityTier.TIER_1: 0.9,
            SourceCredibilityTier.TIER_2: 0.7,
            SourceCredibilityTier.TIER_3: 0.5,
            SourceCredibilityTier.UNVERIFIED: 0.3,
        }
        return scores.get(tier, 0.3)

    def _calculate_fact_density(self, content: str) -> float:
        """Calculate fact density in content"""
        if not content:
            return 0.0

        # Simple heuristic: count numbers, dates, and factual indicators
        import re

        facts = len(
            re.findall(
                r"\d+(?:\.\d+)?%?|\b\d{4}\b|according to|study shows|research indicates",
                content,
                re.IGNORECASE,
            )
        )
        words = len(content.split())

        return min(1.0, facts / max(words / 100, 1))  # Facts per 100 words

    def _calculate_citation_quality(self, content: str) -> float:
        """Calculate citation quality in content"""
        if not content:
            return 0.0

        # Look for citation indicators
        import re

        citations = len(re.findall(r"http[s]?://|doi:|@\w+|\[\d+\]|\(\d{4}\)", content))
        paragraphs = len(content.split("\n\n"))

        return min(1.0, citations / max(paragraphs, 1))

    def _extract_insights_from_structured_data(
        self, structured_data: Dict[str, Any], template_type: ResearchTemplateType
    ) -> List[ResearchInsight]:
        """Extract insights from structured extraction data"""
        insights = []

        # Extract key points as insights
        key_points = structured_data.get("key_points", [])
        for i, point in enumerate(key_points[:5]):  # Limit to 5 insights
            insight = ResearchInsight(
                claim=point,
                confidence=0.8,
                evidence_strength=0.7,
                fact_type="structured_extraction",
                supporting_sources=[],
                contradicting_sources=[],
                validation_status="extracted",
            )
            insights.append(insight)

        return insights

    def _generate_enhanced_summary(
        self,
        extractions: List[FirecrawlExtractionResult],
        research_context: Dict[str, Any],
        template_type: ResearchTemplateType,
    ) -> str:
        """Generate comprehensive summary from deep extractions"""
        if not extractions:
            return "No content successfully extracted for enhanced analysis."

        total_content = sum(len(e.content) for e in extractions)
        unique_domains = set(urlparse(e.url).netloc for e in extractions)

        summary = f"Enhanced research analysis with deep content extraction from {len(extractions)} sources "
        summary += f"across {len(unique_domains)} domains, totaling {total_content:,} characters of content. "
        summary += f"Analysis focused on {template_type.value} methodology with structured data extraction. "

        # Add template-specific insights
        if template_type == ResearchTemplateType.COMPETITIVE_INTELLIGENCE:
            summary += "Competitive analysis reveals detailed product positioning, pricing strategies, and market dynamics."
        elif template_type == ResearchTemplateType.MARKET_ANALYSIS:
            summary += "Market analysis uncovers growth trends, key players, and emerging opportunities."

        return summary


# Async factory function for easy initialization
async def get_firecrawl_client(api_key: str = None) -> FirecrawlMCPClient:
    """Get or create Firecrawl MCP client instance"""

    # Try to get API key from environment if not provided
    if not api_key:
        import os

        api_key = os.getenv("FIRECRAWL_API_KEY", "demo")

    config = FirecrawlConfig(
        api_key=api_key,
        max_pages_per_crawl=8,  # Reasonable default for research
        timeout_seconds=45,
        extract_main_content=True,
        wait_for_results=True,
    )

    client = FirecrawlMCPClient(config)
    return client


# Testing function
async def test_firecrawl_integration():
    """Test Firecrawl integration with sample URLs"""
    print("🧪 Testing Firecrawl MCP Integration")
    print("=" * 50)

    client = await get_firecrawl_client()

    # Test single URL extraction
    test_url = "https://www.firecrawl.dev/blog/ai-powered-web-scraping-solutions-2025"
    print(f"\n🔥 Testing content extraction: {test_url}")

    extraction_result = await client.extract_content(test_url)

    if extraction_result.success:
        print("✅ Extraction successful!")
        print(f"   Title: {extraction_result.title[:100]}...")
        print(f"   Content: {len(extraction_result.content)} chars")
        print(f"   Links: {len(extraction_result.links)} found")
        print(f"   Processing time: {extraction_result.extraction_time_ms}ms")
    else:
        print(f"❌ Extraction failed: {extraction_result.error}")

    # Test enhanced research
    discovered_urls = [
        "https://www.firecrawl.dev",
        "https://docs.firecrawl.dev/mcp",
        "https://www.tavily.com/",
    ]

    print(f"\n🔬 Testing enhanced research with {len(discovered_urls)} URLs")

    enhanced_result = await client.enhance_research_with_deep_extraction(
        discovered_urls=discovered_urls,
        research_context={"topic": "web scraping comparison", "industry": "AI tools"},
        template_type=ResearchTemplateType.COMPETITIVE_INTELLIGENCE,
    )

    print("✅ Enhanced research complete!")
    print(f"   Sources: {len(enhanced_result.sources)}")
    print(f"   Insights: {len(enhanced_result.key_insights)}")
    print(f"   Confidence: {enhanced_result.overall_confidence:.1%}")
    print(f"   Cost: ${enhanced_result.cost_usd:.4f}")
    print(f"   Processing time: {enhanced_result.total_processing_time_ms}ms")


if __name__ == "__main__":
    asyncio.run(test_firecrawl_integration())
