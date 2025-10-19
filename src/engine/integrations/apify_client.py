#!/usr/bin/env python3
"""
METIS Apify MCP Integration Client
Provides specialized web scraping and automation capabilities through Apify's 5,000+ Actor library
Specializes in social media intelligence, real-time monitoring, and competitive research
"""

import os
import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

import httpx
from src.engine.integrations.perplexity_client_advanced import (
    EnhancedSource,
    SourceCredibilityTier,
)

logger = logging.getLogger(__name__)


@dataclass
class ApifyActor:
    """Represents an Apify Actor for research tasks"""

    name: str
    actor_id: str
    description: str
    use_cases: List[str] = field(default_factory=list)
    cost_estimate: float = 0.0  # Per run estimate
    rate_limit: int = 30  # Requests per second


@dataclass
class ApifyResearchResult:
    """Result from Apify research execution"""

    sources: List[EnhancedSource]
    social_insights: List[Dict[str, Any]] = field(default_factory=list)
    real_time_data: Dict[str, Any] = field(default_factory=dict)
    actors_used: List[str] = field(default_factory=list)
    execution_time_ms: int = 0
    cost_usd: float = 0.0
    success: bool = True
    error_message: Optional[str] = None


class ApifyMCPClient:
    """
    Apify MCP integration for specialized research capabilities
    Provides access to 5,000+ pre-built Actors for web scraping and automation
    """

    def __init__(self, api_token: Optional[str] = None):
        self.api_token = api_token or os.getenv("APIFY_API_TOKEN")
        self.mcp_endpoint = "https://mcp.apify.com"
        self.api_base_url = "https://api.apify.com/v2"
        self.rate_limit_per_second = 30

        # Pre-configured actors for different research types
        self.research_actors = self._initialize_research_actors()

        if not self.api_token:
            logger.warning("⚠️ APIFY_API_TOKEN not found - Apify integration disabled")

        logger.info("✅ Apify MCP Client initialized")

    def _initialize_research_actors(self) -> Dict[str, ApifyActor]:
        """Initialize pre-configured research actors"""
        return {
            # Social Media Intelligence
            "twitter_scraper": ApifyActor(
                name="Twitter Scraper",
                actor_id="61RPP7dywgiy0JPD0",
                description="Scrape tweets, user profiles, and trending topics",
                use_cases=[
                    "social_sentiment",
                    "competitive_intelligence",
                    "trend_analysis",
                ],
                cost_estimate=0.01,
            ),
            "linkedin_company": ApifyActor(
                name="LinkedIn Company Scraper",
                actor_id="pocVncppRm8zri2pB",
                description="Scrape company pages, employee data, and job postings",
                use_cases=[
                    "competitive_intelligence",
                    "market_analysis",
                    "hiring_trends",
                ],
                cost_estimate=0.015,
            ),
            "reddit_scraper": ApifyActor(
                name="Reddit Scraper",
                actor_id="trudax~reddit-scraper",
                description="Scrape Reddit posts, comments, and subreddit data",
                use_cases=["social_sentiment", "consumer_insights", "trend_analysis"],
                cost_estimate=0.008,
            ),
            # News and Content Intelligence
            "news_aggregator": ApifyActor(
                name="News Aggregator",
                actor_id="drobnikj~news-scraper",
                description="Aggregate news articles from multiple sources",
                use_cases=[
                    "market_analysis",
                    "competitive_intelligence",
                    "trend_analysis",
                ],
                cost_estimate=0.012,
            ),
            "google_news": ApifyActor(
                name="Google News Scraper",
                actor_id="drobnikj~google-news-scraper",
                description="Scrape Google News for recent articles and trends",
                use_cases=["market_analysis", "real_time_monitoring"],
                cost_estimate=0.010,
            ),
            # E-commerce and Market Intelligence
            "amazon_scraper": ApifyActor(
                name="Amazon Product Scraper",
                actor_id="junglee~amazon-scraper",
                description="Scrape product data, reviews, and pricing",
                use_cases=[
                    "competitive_intelligence",
                    "market_analysis",
                    "pricing_research",
                ],
                cost_estimate=0.020,
            ),
            # General Web Scraping
            "web_scraper": ApifyActor(
                name="Web Scraper",
                actor_id="apify~web-scraper",
                description="Universal web scraper for any website",
                use_cases=["content_extraction", "competitive_intelligence"],
                cost_estimate=0.015,
            ),
            "google_search": ApifyActor(
                name="Google Search Results",
                actor_id="apify~google-search-scraper",
                description="Scrape Google search results and SERP data",
                use_cases=[
                    "market_analysis",
                    "seo_research",
                    "competitive_intelligence",
                ],
                cost_estimate=0.008,
            ),
        }

    def select_optimal_actors(
        self, query: str, research_type: str, max_actors: int = 3
    ) -> List[ApifyActor]:
        """
        Intelligently select optimal actors based on research requirements

        Args:
            query: Research query text
            research_type: Type of research (social_intelligence, competitive_intelligence, etc.)
            max_actors: Maximum number of actors to use

        Returns:
            List of optimal actors for the research task
        """
        query_lower = query.lower()
        selected_actors = []

        # Social media intelligence
        if any(
            term in query_lower
            for term in ["social", "twitter", "sentiment", "linkedin"]
        ):
            if "twitter" in query_lower:
                selected_actors.append(self.research_actors["twitter_scraper"])
            if "linkedin" in query_lower or "company" in query_lower:
                selected_actors.append(self.research_actors["linkedin_company"])
            if "reddit" in query_lower or "discussion" in query_lower:
                selected_actors.append(self.research_actors["reddit_scraper"])

        # News and market monitoring
        if any(
            term in query_lower for term in ["news", "market", "trend", "announcement"]
        ):
            selected_actors.append(self.research_actors["news_aggregator"])
            if "google" in query_lower or "search" in query_lower:
                selected_actors.append(self.research_actors["google_search"])

        # Competitive intelligence
        if research_type == "competitive_intelligence":
            selected_actors.extend(
                [
                    self.research_actors["linkedin_company"],
                    self.research_actors["google_search"],
                    self.research_actors["news_aggregator"],
                ]
            )

        # E-commerce research
        if any(
            term in query_lower
            for term in ["product", "pricing", "amazon", "ecommerce"]
        ):
            selected_actors.append(self.research_actors["amazon_scraper"])

        # Fallback to general web scraping
        if not selected_actors:
            selected_actors.extend(
                [
                    self.research_actors["google_search"],
                    self.research_actors["web_scraper"],
                ]
            )

        # Remove duplicates and limit to max_actors
        unique_actors = list(
            {actor.actor_id: actor for actor in selected_actors}.values()
        )
        return unique_actors[:max_actors]

    async def execute_research(
        self, query: str, research_type: str = "general", max_cost_usd: float = 0.05
    ) -> ApifyResearchResult:
        """
        Execute research using optimal Apify actors

        Args:
            query: Research query
            research_type: Type of research to conduct
            max_cost_usd: Maximum cost limit

        Returns:
            ApifyResearchResult with aggregated findings
        """
        if not self.api_token:
            logger.warning("⚠️ Apify API token not available - returning empty result")
            return ApifyResearchResult(
                sources=[],
                success=False,
                error_message="Apify API token not configured",
            )

        start_time = time.time()

        try:
            logger.info(f"🕷️ Starting Apify research: {query[:100]}...")

            # Select optimal actors
            optimal_actors = self.select_optimal_actors(query, research_type)
            logger.info(
                f"   Selected actors: {[actor.name for actor in optimal_actors]}"
            )

            # Estimate costs and filter if needed
            total_estimated_cost = sum(actor.cost_estimate for actor in optimal_actors)
            if total_estimated_cost > max_cost_usd:
                # Prioritize by cost efficiency
                optimal_actors = sorted(optimal_actors, key=lambda a: a.cost_estimate)
                optimal_actors = self._filter_by_cost_limit(
                    optimal_actors, max_cost_usd
                )

            # Execute actors in parallel with rate limiting
            research_results = await self._execute_actors_parallel(
                optimal_actors, query
            )

            # Synthesize results
            synthesized_result = await self._synthesize_apify_results(
                research_results, query
            )

            execution_time = int((time.time() - start_time) * 1000)

            logger.info(
                f"✅ Apify research completed: {len(synthesized_result.sources)} sources | "
                f"Cost: ${synthesized_result.cost_usd:.4f} | {execution_time/1000:.1f}s"
            )

            synthesized_result.execution_time_ms = execution_time
            synthesized_result.actors_used = [actor.name for actor in optimal_actors]

            return synthesized_result

        except Exception as e:
            logger.error(f"❌ Apify research failed: {e}")
            return ApifyResearchResult(
                sources=[],
                success=False,
                error_message=str(e),
                execution_time_ms=int((time.time() - start_time) * 1000),
            )

    def _filter_by_cost_limit(
        self, actors: List[ApifyActor], max_cost: float
    ) -> List[ApifyActor]:
        """Filter actors to stay within cost limit"""
        selected = []
        total_cost = 0.0

        for actor in actors:
            if total_cost + actor.cost_estimate <= max_cost:
                selected.append(actor)
                total_cost += actor.cost_estimate
            else:
                break

        return selected

    async def _execute_actors_parallel(
        self, actors: List[ApifyActor], query: str
    ) -> List[Dict[str, Any]]:
        """Execute multiple actors in parallel with rate limiting"""
        semaphore = asyncio.Semaphore(3)  # Limit concurrent executions

        async def execute_single_actor(actor: ApifyActor):
            async with semaphore:
                return await self._execute_single_actor(actor, query)

        tasks = [execute_single_actor(actor) for actor in actors]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def _execute_single_actor(
        self, actor: ApifyActor, query: str
    ) -> Dict[str, Any]:
        """Execute a single Apify actor with the research query"""
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                # Prepare actor input based on actor type
                actor_input = self._prepare_actor_input(actor, query)

                # Start actor run
                run_response = await client.post(
                    f"{self.api_base_url}/acts/{actor.actor_id}/runs",
                    headers={
                        "Authorization": f"Bearer {self.api_token}",
                        "Content-Type": "application/json",
                    },
                    json=actor_input,
                    timeout=30.0,
                )

                if run_response.status_code != 201:
                    logger.warning(
                        f"⚠️ Actor {actor.name} start failed: {run_response.status_code}"
                    )
                    return {
                        "success": False,
                        "error": f"Start failed: {run_response.status_code}",
                    }

                run_data = run_response.json()["data"]
                run_id = run_data["id"]

                # Wait for completion (simplified - in production, use webhooks)
                result = await self._wait_for_actor_completion(
                    client, run_id, timeout=45
                )

                return {
                    "success": True,
                    "actor": actor.name,
                    "run_id": run_id,
                    "result": result,
                    "cost": actor.cost_estimate,
                }

        except Exception as e:
            logger.error(f"❌ Actor {actor.name} execution failed: {e}")
            return {"success": False, "actor": actor.name, "error": str(e)}

    def _prepare_actor_input(self, actor: ApifyActor, query: str) -> Dict[str, Any]:
        """Prepare input for specific actor types"""
        base_input = {"query": query}

        if "twitter" in actor.actor_id:
            return {"searchTerms": [query], "maxTweets": 20, "includeUserInfo": True}
        elif "linkedin" in actor.actor_id:
            return {"searchKeywords": query, "maxResults": 15}
        elif "reddit" in actor.actor_id:
            return {"searchTerms": [query], "maxPosts": 25, "sort": "relevance"}
        elif "news" in actor.actor_id:
            return {"query": query, "maxArticles": 20, "timeRange": "week"}
        elif "google" in actor.actor_id:
            return {"queries": [query], "maxPagesPerQuery": 2, "resultsPerPage": 10}
        else:
            # General web scraper
            return {
                "startUrls": [f"https://www.google.com/search?q={query}"],
                "maxRequestsPerCrawl": 10,
            }

    async def _wait_for_actor_completion(
        self, client: httpx.AsyncClient, run_id: str, timeout: int = 60
    ) -> Dict[str, Any]:
        """Wait for actor run completion"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                status_response = await client.get(
                    f"{self.api_base_url}/actor-runs/{run_id}",
                    headers={"Authorization": f"Bearer {self.api_token}"},
                )

                if status_response.status_code == 200:
                    run_data = status_response.json()["data"]
                    status = run_data["status"]

                    if status in ["SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"]:
                        if status == "SUCCEEDED":
                            # Get results
                            results_response = await client.get(
                                f"{self.api_base_url}/actor-runs/{run_id}/dataset/items",
                                headers={"Authorization": f"Bearer {self.api_token}"},
                            )

                            if results_response.status_code == 200:
                                return {
                                    "status": status,
                                    "data": results_response.json(),
                                }

                        return {"status": status, "data": []}

                await asyncio.sleep(2)  # Wait 2 seconds before next check

            except Exception as e:
                logger.warning(f"⚠️ Error checking actor status: {e}")
                await asyncio.sleep(2)

        return {"status": "TIMEOUT", "data": []}

    async def _synthesize_apify_results(
        self, results: List[Dict[str, Any]], query: str
    ) -> ApifyResearchResult:
        """Synthesize results from multiple Apify actors"""
        sources = []
        social_insights = []
        real_time_data = {}
        total_cost = 0.0
        successful_actors = []

        for result in results:
            if isinstance(result, dict) and result.get("success"):
                actor_name = result.get("actor", "unknown")
                successful_actors.append(actor_name)
                total_cost += result.get("cost", 0.0)

                # Extract sources based on actor type
                actor_sources = self._extract_sources_from_actor_result(
                    result, actor_name
                )
                sources.extend(actor_sources)

                # Extract social insights
                if any(
                    social_term in actor_name.lower()
                    for social_term in ["twitter", "linkedin", "reddit"]
                ):
                    social_data = self._extract_social_insights(result)
                    social_insights.extend(social_data)

                # Real-time data aggregation
                if result.get("result", {}).get("data"):
                    real_time_data[actor_name] = {
                        "data_points": len(result["result"]["data"]),
                        "timestamp": datetime.now().isoformat(),
                    }

        # Remove duplicate sources
        unique_sources = self._deduplicate_sources(sources)

        return ApifyResearchResult(
            sources=unique_sources,
            social_insights=social_insights,
            real_time_data=real_time_data,
            actors_used=successful_actors,
            cost_usd=total_cost,
            success=len(successful_actors) > 0,
        )

    def _extract_sources_from_actor_result(
        self, result: Dict[str, Any], actor_name: str
    ) -> List[EnhancedSource]:
        """Extract sources from actor result data"""
        sources = []
        actor_data = result.get("result", {}).get("data", [])

        for item in actor_data[:10]:  # Limit to top 10 items per actor
            if isinstance(item, dict):
                source = self._create_enhanced_source_from_item(item, actor_name)
                if source:
                    sources.append(source)

        return sources

    def _create_enhanced_source_from_item(
        self, item: Dict[str, Any], actor_name: str
    ) -> Optional[EnhancedSource]:
        """Create EnhancedSource from Apify data item"""
        try:
            # Extract common fields based on actor type
            url = item.get("url", "")
            title = item.get("title", "") or item.get("text", "")[:100]
            content = item.get("text", "") or item.get("content", "") or str(item)

            # Determine credibility tier based on source
            credibility = SourceCredibilityTier.MODERATE
            if any(
                domain in url
                for domain in ["linkedin.com", "twitter.com", "reddit.com"]
            ):
                credibility = SourceCredibilityTier.HIGH

            return EnhancedSource(
                url=url or f"apify://{actor_name}",
                title=title,
                content_snippet=content[:500],
                credibility_score=0.7,
                credibility_tier=credibility,
                relevance_score=0.8,  # Assume high relevance from targeted scraping
                recency_score=0.9,  # Apify provides fresh data
                extraction_metadata={
                    "source": "apify",
                    "actor": actor_name,
                    "extraction_time": datetime.now().isoformat(),
                },
            )

        except Exception as e:
            logger.warning(f"⚠️ Error creating source from Apify item: {e}")
            return None

    def _extract_social_insights(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract social media insights from actor results"""
        insights = []
        actor_data = result.get("result", {}).get("data", [])

        for item in actor_data:
            if isinstance(item, dict):
                insight = {
                    "platform": self._detect_social_platform(result.get("actor", "")),
                    "engagement_metrics": self._extract_engagement_metrics(item),
                    "sentiment_indicators": self._extract_sentiment_indicators(item),
                    "temporal_data": self._extract_temporal_data(item),
                }
                insights.append(insight)

        return insights[:5]  # Limit insights

    def _detect_social_platform(self, actor_name: str) -> str:
        """Detect social media platform from actor name"""
        if "twitter" in actor_name.lower():
            return "twitter"
        elif "linkedin" in actor_name.lower():
            return "linkedin"
        elif "reddit" in actor_name.lower():
            return "reddit"
        else:
            return "unknown"

    def _extract_engagement_metrics(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Extract engagement metrics from social media data"""
        return {
            "likes": item.get("likes", 0) or item.get("retweets", 0),
            "shares": item.get("shares", 0) or item.get("replies", 0),
            "comments": item.get("comments", 0),
        }

    def _extract_sentiment_indicators(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Extract basic sentiment indicators"""
        text = item.get("text", "") or item.get("content", "")

        # Simple sentiment analysis (in production, use proper NLP)
        positive_words = len(
            [
                w
                for w in text.split()
                if w.lower() in ["good", "great", "excellent", "amazing"]
            ]
        )
        negative_words = len(
            [
                w
                for w in text.split()
                if w.lower() in ["bad", "terrible", "awful", "worst"]
            ]
        )

        return {
            "positive_indicators": positive_words,
            "negative_indicators": negative_words,
            "neutral_score": max(0, len(text.split()) - positive_words - negative_words)
            / max(1, len(text.split())),
        }

    def _extract_temporal_data(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Extract temporal information"""
        return {
            "created_at": item.get("createdAt", "") or item.get("date", ""),
            "extraction_timestamp": datetime.now().isoformat(),
        }

    def _deduplicate_sources(
        self, sources: List[EnhancedSource]
    ) -> List[EnhancedSource]:
        """Remove duplicate sources based on URL and title similarity"""
        seen_urls = set()
        unique_sources = []

        for source in sources:
            # Use URL or title as deduplication key
            dedup_key = source.url or source.title[:50]

            if dedup_key not in seen_urls:
                seen_urls.add(dedup_key)
                unique_sources.append(source)

        return unique_sources


# Global client instance
_apify_client = None


async def get_apify_client(api_token: Optional[str] = None) -> ApifyMCPClient:
    """Get or create Apify MCP client instance"""
    global _apify_client

    if _apify_client is None:
        _apify_client = ApifyMCPClient(api_token)

    return _apify_client
