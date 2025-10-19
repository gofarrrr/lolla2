"""
Research Orchestration Service - Clean Multi-Source Research Coordination
========================================================================

REFACTORING TARGET: Extract main orchestration from enhanced_research_orchestrator.py
PATTERN: Service Composition with Clean Orchestration
GOAL: Create focused, testable research orchestration service

Responsibility:
- Coordinate multi-source research execution
- Manage research strategy decisions
- Handle parallel research task execution
- Consolidate and synthesize research results

Benefits:
- Single Responsibility Principle for orchestration
- Clean service composition and dependency injection
- Easily testable orchestration logic
- Clear separation of concerns
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

# Import extracted services
from src.engine.services.research.research_strategy_service import (
    get_research_strategy_service,
    ResearchQuery,
)
from src.engine.services.research.domain_prompt_service import get_domain_prompt_service

# Import integration clients
from src.engine.integrations.perplexity_client_advanced import (
    AdvancedResearchResult,
    ResearchMode,
)

logger = logging.getLogger(__name__)


@dataclass
class ConsolidatedResearchResult:
    """Consolidated research result from multiple sources"""

    query: ResearchQuery
    perplexity_result: Optional[AdvancedResearchResult] = None
    web_scraping_results: List[Any] = field(default_factory=list)
    social_intelligence_results: List[Any] = field(default_factory=list)
    consolidation_summary: Optional[str] = None
    total_cost_usd: float = 0.0
    total_processing_time_seconds: float = 0.0
    research_quality_score: float = 0.0
    source_count: int = 0
    success: bool = True
    error_message: Optional[str] = None


class ResearchTaskExecutor:
    """
    Research task execution service

    Responsibility: Execute individual research tasks in parallel
    Complexity Target: Grade B (≤10)
    """

    def __init__(self, perplexity_client, firecrawl_client, apify_client):
        self.perplexity_client = perplexity_client
        self.firecrawl_client = firecrawl_client
        self.apify_client = apify_client
        self.logger = logging.getLogger(__name__)

    async def execute_perplexity_research(
        self,
        research_query: ResearchQuery,
        strategy_config: Dict[str, Any],
        domain_prompt_result,
    ) -> Optional[AdvancedResearchResult]:
        """
        Execute Perplexity research with strategy configuration

        Complexity: Target B (≤10)
        """
        if not self.perplexity_client:
            return None

        try:
            # Determine research mode based on strategy
            if strategy_config["mode"] == "sonar_deep":
                mode = ResearchMode.SONAR_DEEP_RESEARCH
            elif strategy_config["mode"] == "multi_query":
                mode = ResearchMode.MULTI_QUERY_COMPREHENSIVE
            else:  # hybrid
                mode = ResearchMode.HYBRID_INTELLIGENCE

            # Execute research
            research_result = await self.perplexity_client.execute_advanced_research(
                query=research_query.query,
                mode=mode,
                domain_context=domain_prompt_result.context_primer,
                custom_prompt_template=domain_prompt_result.prompt_template,
                max_cost_usd=strategy_config.get("max_cost_usd", 10.0),
            )

            self.logger.info(
                f"✅ Perplexity research completed: {research_result.source_count} sources"
            )
            return research_result

        except Exception as e:
            self.logger.error(f"❌ Perplexity research failed: {e}")
            return None

    async def execute_web_scraping(
        self, research_query: ResearchQuery, scraping_config: Dict[str, Any]
    ) -> List[Any]:
        """
        Execute web scraping tasks

        Complexity: Target B (≤10)
        """
        if (
            not scraping_config.get("enable_web_scraping", False)
            or not self.firecrawl_client
        ):
            return []

        try:
            # Extract potential URLs from query or use search-based scraping
            urls = self._extract_urls_from_query(research_query.query)

            scraping_tasks = []
            for url in urls[:5]:  # Limit to 5 URLs for cost control
                task = self.firecrawl_client.scrape_url(url)
                scraping_tasks.append(task)

            # Execute scraping in parallel
            scraping_results = await asyncio.gather(
                *scraping_tasks, return_exceptions=True
            )

            # Filter successful results
            successful_results = [
                result
                for result in scraping_results
                if not isinstance(result, Exception) and result
            ]

            self.logger.info(
                f"✅ Web scraping completed: {len(successful_results)} successful results"
            )
            return successful_results

        except Exception as e:
            self.logger.error(f"❌ Web scraping failed: {e}")
            return []

    def _extract_urls_from_query(self, query: str) -> List[str]:
        """
        Extract URLs from query text

        Complexity: Target B (≤10)
        """
        import re

        # Simple URL extraction pattern
        url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        urls = re.findall(url_pattern, query)

        # Add common domains if mentioned
        domain_keywords = {
            "wikipedia": "https://en.wikipedia.org/wiki/",
            "github": "https://github.com/",
            "linkedin": "https://linkedin.com/",
            "twitter": "https://twitter.com/",
            "medium": "https://medium.com/",
        }

        for keyword, base_url in domain_keywords.items():
            if keyword in query.lower() and base_url not in urls:
                urls.append(base_url)

        return urls[:5]  # Limit to prevent excessive scraping


class ResearchResultConsolidator:
    """
    Research result consolidation service

    Responsibility: Consolidate and synthesize multi-source results
    Complexity Target: Grade B (≤10)
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def consolidate_results(
        self,
        perplexity_result: Optional[AdvancedResearchResult],
        web_results: List[Any],
        research_query: ResearchQuery,
    ) -> Dict[str, Any]:
        """
        Consolidate results from multiple research sources

        Complexity: Target B (≤10)
        """
        consolidation = {
            "primary_insights": [],
            "supporting_evidence": [],
            "source_count": 0,
            "quality_score": 0.0,
            "total_cost": 0.0,
        }

        # Process Perplexity results
        if perplexity_result and perplexity_result.success:
            consolidation["primary_insights"].extend(
                perplexity_result.research_insights[:5]
            )
            consolidation["source_count"] += perplexity_result.source_count
            consolidation["total_cost"] += perplexity_result.cost_usd
            consolidation["quality_score"] += 8.0  # Perplexity high quality

        # Process web scraping results
        if web_results:
            web_insights = self._extract_web_insights(web_results)
            consolidation["supporting_evidence"].extend(web_insights)
            consolidation["source_count"] += len(web_results)
            consolidation["quality_score"] += len(web_results) * 2.0

        # Calculate final quality score
        max_possible_score = 10.0
        consolidation["quality_score"] = min(
            consolidation["quality_score"], max_possible_score
        )

        return consolidation

    def _extract_web_insights(self, web_results: List[Any]) -> List[str]:
        """
        Extract insights from web scraping results

        Complexity: Target B (≤10)
        """
        insights = []

        for result in web_results[:3]:  # Limit to top 3 results
            if hasattr(result, "content") and result.content:
                # Extract key sentences (simplified)
                content = result.content[:1000]  # Limit content length
                insights.append(f"Web source insight: {content}")

        return insights


class ResearchOrchestrationService:
    """
    Main research orchestration service using service composition

    Responsibility: Coordinate complete multi-source research execution
    Complexity Target: Grade B (≤10 per method)
    """

    def __init__(
        self, perplexity_client=None, firecrawl_client=None, apify_client=None
    ):
        self.logger = logging.getLogger(__name__)

        # Service dependencies
        self.strategy_service = get_research_strategy_service()
        self.domain_prompt_service = get_domain_prompt_service()
        self.task_executor = ResearchTaskExecutor(
            perplexity_client, firecrawl_client, apify_client
        )
        self.result_consolidator = ResearchResultConsolidator()

    async def execute_comprehensive_research(
        self, research_query: ResearchQuery
    ) -> ConsolidatedResearchResult:
        """
        Execute comprehensive multi-source research

        Complexity: Target B (≤10) - Pure orchestration
        """
        start_time = time.time()
        self.logger.info(
            f"🎯 Starting comprehensive research: {research_query.query[:100]}..."
        )

        try:
            # Step 1: Strategy Decision
            strategy_decision = self.strategy_service.decide_research_strategy(
                research_query
            )
            self.logger.info(f"🧠 Strategy: {strategy_decision['strategy']}")

            # Step 2: Domain Prompt Generation
            domain_prompt_result = self.domain_prompt_service.generate_domain_prompt(
                research_query, strategy_decision
            )
            self.logger.info(
                f"📋 Domain Pattern: {domain_prompt_result.domain_pattern}"
            )

            # Step 3: Web Scraping Decision
            scraping_config = self.strategy_service.should_enable_web_scraping(
                research_query, strategy_decision
            )

            # Step 4: Execute Research Tasks in Parallel
            research_tasks = [
                self.task_executor.execute_perplexity_research(
                    research_query, strategy_decision["config"], domain_prompt_result
                ),
                self.task_executor.execute_web_scraping(
                    research_query, scraping_config
                ),
            ]

            perplexity_result, web_results = await asyncio.gather(*research_tasks)

            # Step 5: Consolidate Results
            consolidation = await self.result_consolidator.consolidate_results(
                perplexity_result, web_results, research_query
            )

            # Step 6: Build Final Result
            total_time = time.time() - start_time

            return ConsolidatedResearchResult(
                query=research_query,
                perplexity_result=perplexity_result,
                web_scraping_results=web_results,
                consolidation_summary=self._generate_summary(consolidation),
                total_cost_usd=consolidation["total_cost"],
                total_processing_time_seconds=total_time,
                research_quality_score=consolidation["quality_score"],
                source_count=consolidation["source_count"],
                success=True,
            )

        except Exception as e:
            self.logger.error(f"❌ Research orchestration failed: {e}")

            return ConsolidatedResearchResult(
                query=research_query,
                total_processing_time_seconds=time.time() - start_time,
                success=False,
                error_message=str(e),
            )

    def _generate_summary(self, consolidation: Dict[str, Any]) -> str:
        """
        Generate research summary

        Complexity: Target B (≤10)
        """
        insights_count = len(consolidation["primary_insights"])
        evidence_count = len(consolidation["supporting_evidence"])
        quality = consolidation["quality_score"]
        sources = consolidation["source_count"]

        return (
            f"Research completed with {insights_count} primary insights, "
            f"{evidence_count} supporting evidence pieces from {sources} sources. "
            f"Quality score: {quality:.1f}/10.0"
        )


# Singleton instance for injection
_orchestration_service_instance = None


def get_research_orchestration_service(
    perplexity_client=None, firecrawl_client=None, apify_client=None
) -> ResearchOrchestrationService:
    """Factory function for dependency injection"""
    global _orchestration_service_instance
    if _orchestration_service_instance is None:
        _orchestration_service_instance = ResearchOrchestrationService(
            perplexity_client, firecrawl_client, apify_client
        )
    return _orchestration_service_instance
