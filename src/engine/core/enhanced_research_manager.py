"""
Enhanced Research Manager - METIS 2.0 Complete Intelligence Platform
====================================================================

Unifies RAG pipeline, web intelligence, and existing research providers into a
single orchestrated system with fallback chains, cost optimization, and
comprehensive learning capabilities.

Key Features:
1. RAG-first architecture with hybrid storage
2. Intelligent provider selection and fallback chains
3. Cost optimization across all providers
4. Comprehensive learning from all research operations
5. Unified transparency through context stream
6. Advanced gap analysis and adaptive research strategies
"""

import logging
import asyncio
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

# Core components
from src.engine.core.research_manager import ResearchManager
from src.core.unified_context_stream import UnifiedContextStream

# RAG and storage components
from src.rag.rag_pipeline import EnhancedRAGPipeline

# Web intelligence components
from src.web_intelligence.web_intelligence_manager import WebIntelligenceManager

# Research providers

logger = logging.getLogger(__name__)


class ResearchGapType(Enum):
    """Types of research gaps that can be identified"""

    CURRENT_EVENTS = "current_events"
    TECHNICAL_RESEARCH = "technical_research"
    WEBSITE_CONTENT = "website_content"
    SOCIAL_MEDIA = "social_media"
    STRUCTURED_DATA = "structured_data"
    ACADEMIC_RESEARCH = "academic_research"
    FACT_CHECKING = "fact_checking"


@dataclass
class ResearchGap:
    """Represents a gap in current knowledge that needs to be filled"""

    gap_type: ResearchGapType
    description: str
    priority: int
    estimated_cost: float
    preferred_providers: List[str]
    fallback_providers: List[str]
    metadata: Dict[str, Any] = None


@dataclass
class ProviderDefinition:
    """Enhanced provider definition with detailed capabilities"""

    name: str
    client: Any
    capabilities: List[str]
    priority: int
    cost_per_query: float
    avg_response_time: float
    reliability_score: float
    daily_limit: Optional[int] = None
    current_usage: int = 0


class EnhancedResearchManager(ResearchManager):
    """
    Enhanced Research Manager that unifies all research capabilities:
    - RAG as primary knowledge store
    - Multiple research providers with fallback chains
    - Web intelligence for real-time data
    - Cost optimization and usage tracking
    - Comprehensive learning and memory
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize Enhanced Research Manager with complete configuration"""

        # Initialize base ResearchManager with empty providers (we'll manage them ourselves)
        from src.core.unified_context_stream import get_unified_context_stream
        context_stream = get_unified_context_stream()
        super().__init__([], context_stream)

        self.config = config
        self.context_stream = context_stream

        # Initialize RAG as primary knowledge store
        self.rag_pipeline = EnhancedRAGPipeline(config.get("rag", {}))

        # Initialize storage systems
        self.milvus_client = MilvusClient(config.get("milvus", {}))
        self.supabase_client = SupabaseClient(config.get("supabase", {}))
        self.zep_client = ZepClient(config.get("zep", {}))

        # Initialize web intelligence
        self.web_intelligence = WebIntelligenceManager(
            firecrawl_api_key=config.get("firecrawl_api_key"),
            apify_api_key=config.get("apify_api_key"),
        )

        # Enhanced provider definitions with detailed capabilities
        self.providers = self._initialize_enhanced_providers(config)

        # Cost optimization settings
        self.cost_optimizer = {
            "daily_budget": config.get("daily_budget", 10.0),  # $10 default
            "cost_efficiency_threshold": config.get("cost_efficiency_threshold", 0.8),
            "prefer_cached": config.get("prefer_cached", True),
            "max_parallel_requests": config.get("max_parallel_requests", 5),
        }

        # Enhanced usage tracking
        self.usage_stats = {
            "session_start": datetime.now(),
            "total_queries": 0,
            "successful_queries": 0,
            "rag_hits": 0,
            "rag_misses": 0,
            "total_cost_usd": 0.0,
            "provider_usage": {},
            "gap_analysis_stats": {},
            "learning_events": 0,
        }

        # Initialize provider usage tracking
        for provider_name in self.providers.keys():
            self.usage_stats["provider_usage"][provider_name] = {
                "requests": 0,
                "successes": 0,
                "failures": 0,
                "total_cost": 0.0,
                "avg_response_time": 0.0,
            }

    def _initialize_enhanced_providers(
        self, config: Dict[str, Any]
    ) -> Dict[str, ProviderDefinition]:
        """Initialize all enhanced providers with detailed configurations"""

        providers = {}

        # Perplexity - Best for current events and fact checking
        if config.get("perplexity_api_key"):
            try:
                from src.engine.providers.research.perplexity_provider import (
                    PerplexityProvider,
                )

                providers["perplexity"] = ProviderDefinition(
                    name="perplexity",
                    client=PerplexityProvider(config["perplexity_api_key"]),
                    capabilities=[
                        "current_events",
                        "fact_checking",
                        "web_search",
                        "news",
                    ],
                    priority=1,
                    cost_per_query=0.001,
                    avg_response_time=2.5,
                    reliability_score=0.95,
                )
            except ImportError:
                logger.warning("Perplexity provider not available")

        # Exa - Best for academic and technical research
        if config.get("exa_api_key"):
            try:
                from src.engine.providers.research.exa_provider import ExaProvider

                providers["exa"] = ProviderDefinition(
                    name="exa",
                    client=ExaProvider(config["exa_api_key"]),
                    capabilities=[
                        "academic_research",
                        "technical_research",
                        "semantic_search",
                    ],
                    priority=2,
                    cost_per_query=0.0008,
                    avg_response_time=3.2,
                    reliability_score=0.92,
                )
            except ImportError:
                logger.warning("Exa provider not available")

        # Firecrawl via Web Intelligence - Best for quick web scraping
        providers["firecrawl"] = ProviderDefinition(
            name="firecrawl",
            client=self.web_intelligence,
            capabilities=["website_content", "quick_scrape", "markdown_extraction"],
            priority=3,
            cost_per_query=0.001,
            avg_response_time=4.0,
            reliability_score=0.88,
        )

        # Apify via Web Intelligence - Best for deep scraping and structured data
        providers["apify"] = ProviderDefinition(
            name="apify",
            client=self.web_intelligence,
            capabilities=[
                "structured_data",
                "deep_scrape",
                "social_media",
                "ecommerce",
            ],
            priority=4,
            cost_per_query=0.005,
            avg_response_time=15.0,
            reliability_score=0.85,
        )

        return providers

    async def research_with_memory(
        self, query: str, user_id: str, requirements: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main research method that orchestrates all sources with memory

        Process:
        1. Check RAG for existing knowledge
        2. Analyze gaps in current knowledge
        3. Create intelligent provider execution plan
        4. Execute research with cost optimization
        5. Store all results in RAG for learning
        6. Update conversation memory
        7. Return comprehensive results
        """

        if requirements is None:
            requirements = {}

        # Create unique analysis ID
        analysis_id = str(uuid.uuid4())
        start_time = datetime.now()

        await self.context_stream.add_event(
            {
                "type": "ENHANCED_RESEARCH_START",
                "analysis_id": analysis_id,
                "query": query,
                "user_id": user_id,
                "requirements": requirements,
            }
        )

        try:
            # Phase 1: RAG Knowledge Check
            rag_results = await self._check_rag_knowledge(query, user_id, requirements)

            # Phase 2: Gap Analysis
            gaps = await self._analyze_knowledge_gaps(rag_results, query, requirements)

            # Phase 3: Provider Plan Creation
            provider_plan = await self._create_intelligent_provider_plan(
                gaps, query, requirements
            )

            # Phase 4: Execute Research with Cost Optimization
            new_research = await self._execute_optimized_research(
                provider_plan, query, user_id, analysis_id
            )

            # Phase 5: Store Results in RAG
            await self._store_research_in_rag(new_research, query, user_id, analysis_id)

            # Phase 6: Update Conversation Memory
            await self._update_conversation_memory(
                user_id, query, rag_results, new_research
            )

            # Phase 7: Create Comprehensive Response
            final_results = await self._create_comprehensive_response(
                analysis_id, query, rag_results, new_research, gaps, provider_plan
            )

            # Update statistics
            self.usage_stats["total_queries"] += 1
            self.usage_stats["successful_queries"] += 1

            execution_time = (datetime.now() - start_time).total_seconds()
            await self.context_stream.add_event(
                {
                    "type": "ENHANCED_RESEARCH_COMPLETE",
                    "analysis_id": analysis_id,
                    "execution_time_seconds": execution_time,
                    "sources_used": list(new_research.keys())
                    + (["rag"] if rag_results.get("has_results") else []),
                    "total_cost": final_results.get("cost_breakdown", {}).get(
                        "total_cost", 0
                    ),
                }
            )

            return final_results

        except Exception as e:
            logger.error(f"Enhanced research failed for query '{query}': {e}")

            await self.context_stream.add_event(
                {
                    "type": "ENHANCED_RESEARCH_ERROR",
                    "analysis_id": analysis_id,
                    "error": str(e),
                    "fallback_to_basic": True,
                }
            )

            # Fallback to basic research
            return await self._basic_research_fallback(query, user_id, analysis_id)

    async def _check_rag_knowledge(
        self, query: str, user_id: str, requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check RAG pipeline for existing relevant knowledge"""

        await self.context_stream.add_event(
            {"type": "RAG_KNOWLEDGE_CHECK_START", "query": query}
        )

        try:
            # Check if we need fresh data
            freshness_required = requirements.get("needs_fresh_data", False)
            max_age_hours = requirements.get("max_age_hours", 24)

            # Search RAG with freshness requirements
            rag_results = await self.rag_pipeline.intelligent_search(
                query=query,
                user_id=user_id,
                max_results=requirements.get("max_rag_results", 10),
                similarity_threshold=requirements.get("similarity_threshold", 0.7),
                freshness_required=freshness_required,
                max_age_hours=max_age_hours,
            )

            # Analyze RAG result quality
            has_results = bool(rag_results and rag_results.get("results"))
            is_sufficient = False

            if has_results:
                # Check if results are sufficient based on:
                # 1. Number of relevant results
                # 2. Confidence scores
                # 3. Freshness requirements
                # 4. Coverage of query aspects

                results_count = len(rag_results["results"])
                avg_score = (
                    sum(r.get("score", 0) for r in rag_results["results"])
                    / results_count
                )

                is_sufficient = (
                    results_count >= requirements.get("min_rag_results", 3)
                    and avg_score >= requirements.get("min_confidence", 0.8)
                    and not freshness_required  # If fresh data needed, RAG alone isn't sufficient
                )

            rag_analysis = {
                "has_results": has_results,
                "is_sufficient": is_sufficient,
                "results_count": (
                    len(rag_results.get("results", [])) if has_results else 0
                ),
                "avg_confidence": sum(
                    r.get("score", 0) for r in rag_results.get("results", [])
                )
                / max(len(rag_results.get("results", [])), 1),
                "freshness_met": not freshness_required
                or self._check_freshness(rag_results, max_age_hours),
                "raw_results": rag_results,
            }

            if has_results:
                self.usage_stats["rag_hits"] += 1
            else:
                self.usage_stats["rag_misses"] += 1

            await self.context_stream.add_event(
                {
                    "type": "RAG_KNOWLEDGE_CHECK_COMPLETE",
                    "has_results": has_results,
                    "is_sufficient": is_sufficient,
                    "results_count": rag_analysis["results_count"],
                }
            )

            return rag_analysis

        except Exception as e:
            logger.error(f"RAG knowledge check failed: {e}")
            await self.context_stream.add_event(
                {"type": "RAG_KNOWLEDGE_CHECK_ERROR", "error": str(e)}
            )

            return {
                "has_results": False,
                "is_sufficient": False,
                "results_count": 0,
                "error": str(e),
            }

    async def _analyze_knowledge_gaps(
        self, rag_results: Dict[str, Any], query: str, requirements: Dict[str, Any]
    ) -> List[ResearchGap]:
        """Analyze gaps in current knowledge and determine research needs"""

        await self.context_stream.add_event(
            {"type": "GAP_ANALYSIS_START", "query": query}
        )

        gaps = []
        query_lower = query.lower()

        # If RAG is sufficient, minimal gaps
        if rag_results.get("is_sufficient", False):
            await self.context_stream.add_event(
                {
                    "type": "GAP_ANALYSIS_COMPLETE",
                    "gaps_identified": 0,
                    "reason": "RAG results sufficient",
                }
            )
            return gaps

        # Analyze query for different types of information needs

        # Current events gap
        current_event_keywords = [
            "news",
            "latest",
            "recent",
            "today",
            "current",
            "breaking",
            "2024",
            "2025",
        ]
        if any(
            keyword in query_lower for keyword in current_event_keywords
        ) or requirements.get("needs_fresh_data", False):
            gaps.append(
                ResearchGap(
                    gap_type=ResearchGapType.CURRENT_EVENTS,
                    description="Need current events and real-time information",
                    priority=1,
                    estimated_cost=0.002,
                    preferred_providers=["perplexity"],
                    fallback_providers=["exa", "firecrawl"],
                )
            )

        # Technical research gap
        technical_keywords = [
            "api",
            "documentation",
            "technical",
            "specification",
            "implementation",
            "code",
            "algorithm",
        ]
        if any(keyword in query_lower for keyword in technical_keywords):
            gaps.append(
                ResearchGap(
                    gap_type=ResearchGapType.TECHNICAL_RESEARCH,
                    description="Need technical documentation and implementation details",
                    priority=2,
                    estimated_cost=0.0015,
                    preferred_providers=["exa"],
                    fallback_providers=["perplexity", "firecrawl"],
                )
            )

        # Website content gap
        if "http" in query or "www." in query or requirements.get("specific_urls"):
            urls = requirements.get("specific_urls", [])
            gaps.append(
                ResearchGap(
                    gap_type=ResearchGapType.WEBSITE_CONTENT,
                    description="Need content from specific websites",
                    priority=2,
                    estimated_cost=0.003,
                    preferred_providers=["firecrawl"],
                    fallback_providers=["apify"],
                    metadata={"urls": urls},
                )
            )

        # Social media gap
        social_keywords = [
            "twitter",
            "linkedin",
            "facebook",
            "instagram",
            "social media",
            "posts",
            "tweets",
        ]
        if any(keyword in query_lower for keyword in social_keywords):
            gaps.append(
                ResearchGap(
                    gap_type=ResearchGapType.SOCIAL_MEDIA,
                    description="Need social media content and trends",
                    priority=3,
                    estimated_cost=0.008,
                    preferred_providers=["apify"],
                    fallback_providers=["firecrawl"],
                )
            )

        # Academic research gap
        academic_keywords = [
            "research",
            "study",
            "paper",
            "academic",
            "scholar",
            "journal",
            "publication",
        ]
        if any(keyword in query_lower for keyword in academic_keywords):
            gaps.append(
                ResearchGap(
                    gap_type=ResearchGapType.ACADEMIC_RESEARCH,
                    description="Need academic papers and research studies",
                    priority=2,
                    estimated_cost=0.0012,
                    preferred_providers=["exa"],
                    fallback_providers=["perplexity"],
                )
            )

        # If no specific gaps identified but RAG is insufficient, add general search gap
        if not gaps and not rag_results.get("is_sufficient", False):
            gaps.append(
                ResearchGap(
                    gap_type=ResearchGapType.TECHNICAL_RESEARCH,
                    description="General information search needed",
                    priority=1,
                    estimated_cost=0.002,
                    preferred_providers=["perplexity"],
                    fallback_providers=["exa"],
                )
            )

        await self.context_stream.add_event(
            {
                "type": "GAP_ANALYSIS_COMPLETE",
                "gaps_identified": len(gaps),
                "gap_types": [gap.gap_type.value for gap in gaps],
            }
        )

        return gaps

    async def _create_intelligent_provider_plan(
        self, gaps: List[ResearchGap], query: str, requirements: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Create optimized execution plan based on gaps and cost constraints"""

        plan = []
        estimated_total_cost = 0.0
        daily_budget = self.cost_optimizer["daily_budget"]
        current_daily_cost = self._calculate_daily_cost()

        await self.context_stream.add_event(
            {
                "type": "PROVIDER_PLAN_CREATION_START",
                "gaps_count": len(gaps),
                "daily_budget": daily_budget,
                "current_daily_cost": current_daily_cost,
            }
        )

        for gap in gaps:
            # Check if we have budget for this gap
            if (
                current_daily_cost + estimated_total_cost + gap.estimated_cost
                > daily_budget
            ):
                logger.warning(f"Skipping gap {gap.gap_type} due to budget constraint")
                continue

            # Select best provider based on:
            # 1. Capability match
            # 2. Cost efficiency
            # 3. Reliability
            # 4. Current usage limits

            best_provider = self._select_best_provider(gap)
            if not best_provider:
                continue

            task = {
                "gap_type": gap.gap_type.value,
                "provider": best_provider.name,
                "fallback_providers": gap.fallback_providers,
                "estimated_cost": gap.estimated_cost,
                "priority": gap.priority,
                "parallel": gap.priority <= 2,  # High priority tasks run in parallel
                "metadata": gap.metadata or {},
            }

            # Add specific task configuration based on gap type
            if gap.gap_type == ResearchGapType.CURRENT_EVENTS:
                task.update(
                    {
                        "search_type": "news",
                        "max_results": 10,
                        "freshness_priority": True,
                    }
                )
            elif gap.gap_type == ResearchGapType.TECHNICAL_RESEARCH:
                task.update(
                    {
                        "search_type": "technical",
                        "max_results": 15,
                        "include_code": True,
                    }
                )
            elif gap.gap_type == ResearchGapType.WEBSITE_CONTENT:
                task.update(
                    {
                        "urls": gap.metadata.get("urls", []),
                        "extract_markdown": True,
                        "max_pages": 5,
                    }
                )
            elif gap.gap_type == ResearchGapType.SOCIAL_MEDIA:
                task.update({"platforms": ["twitter", "linkedin"], "max_posts": 20})

            plan.append(task)
            estimated_total_cost += gap.estimated_cost

        # Sort plan by priority (highest first)
        plan.sort(key=lambda x: x["priority"])

        await self.context_stream.add_event(
            {
                "type": "PROVIDER_PLAN_CREATION_COMPLETE",
                "plan_tasks": len(plan),
                "estimated_cost": estimated_total_cost,
                "providers_selected": list(set(task["provider"] for task in plan)),
            }
        )

        return plan

    def _select_best_provider(self, gap: ResearchGap) -> Optional[ProviderDefinition]:
        """Select the best provider for a research gap"""

        # Check preferred providers first
        for provider_name in gap.preferred_providers:
            provider = self.providers.get(provider_name)
            if provider and self._is_provider_available(provider):
                return provider

        # Check fallback providers
        for provider_name in gap.fallback_providers:
            provider = self.providers.get(provider_name)
            if provider and self._is_provider_available(provider):
                return provider

        return None

    def _is_provider_available(self, provider: ProviderDefinition) -> bool:
        """Check if provider is available and within limits"""

        # Check daily limits
        if provider.daily_limit:
            if provider.current_usage >= provider.daily_limit:
                return False

        # Check reliability threshold
        if provider.reliability_score < 0.7:
            return False

        return True

    async def _execute_optimized_research(
        self, plan: List[Dict[str, Any]], query: str, user_id: str, analysis_id: str
    ) -> Dict[str, Any]:
        """Execute research plan with cost optimization and parallel execution"""

        results = {}
        total_cost = 0.0

        await self.context_stream.add_event(
            {
                "type": "OPTIMIZED_RESEARCH_EXECUTION_START",
                "analysis_id": analysis_id,
                "tasks_count": len(plan),
            }
        )

        # Separate parallel and sequential tasks
        parallel_tasks = [task for task in plan if task.get("parallel", False)]
        sequential_tasks = [task for task in plan if not task.get("parallel", False)]

        # Execute parallel tasks
        if parallel_tasks:
            parallel_results = await self._execute_parallel_tasks(
                parallel_tasks, query, user_id
            )
            results.update(parallel_results)

        # Execute sequential tasks
        for task in sequential_tasks:
            task_result = await self._execute_single_task(task, query, user_id)
            if task_result:
                results[task["provider"]] = task_result
                total_cost += task.get("estimated_cost", 0)

        self.usage_stats["total_cost_usd"] += total_cost

        await self.context_stream.add_event(
            {
                "type": "OPTIMIZED_RESEARCH_EXECUTION_COMPLETE",
                "analysis_id": analysis_id,
                "providers_used": list(results.keys()),
                "total_cost": total_cost,
            }
        )

        return results

    async def _execute_parallel_tasks(
        self, tasks: List[Dict[str, Any]], query: str, user_id: str
    ) -> Dict[str, Any]:
        """Execute multiple research tasks in parallel"""

        # Limit concurrent requests
        max_concurrent = self.cost_optimizer["max_parallel_requests"]
        semaphore = asyncio.Semaphore(max_concurrent)

        async def execute_with_semaphore(task):
            async with semaphore:
                return await self._execute_single_task(task, query, user_id)

        # Execute all tasks concurrently
        task_results = await asyncio.gather(
            *[execute_with_semaphore(task) for task in tasks], return_exceptions=True
        )

        # Process results
        results = {}
        for i, result in enumerate(task_results):
            if not isinstance(result, Exception) and result:
                results[tasks[i]["provider"]] = result

        return results

    async def _execute_single_task(
        self, task: Dict[str, Any], query: str, user_id: str
    ) -> Optional[Dict[str, Any]]:
        """Execute a single research task with fallback logic"""

        provider_name = task["provider"]
        provider = self.providers.get(provider_name)

        if not provider:
            logger.error(f"Provider {provider_name} not found")
            return None

        start_time = datetime.now()

        await self.context_stream.add_event(
            {
                "type": "TASK_EXECUTION_START",
                "provider": provider_name,
                "gap_type": task.get("gap_type"),
                "query": query,
            }
        )

        try:
            # Execute based on provider type
            result = await self._call_provider(provider, task, query, user_id)

            # Update statistics
            execution_time = (datetime.now() - start_time).total_seconds()
            self._update_provider_stats(
                provider_name, True, execution_time, task.get("estimated_cost", 0)
            )

            await self.context_stream.add_event(
                {
                    "type": "TASK_EXECUTION_SUCCESS",
                    "provider": provider_name,
                    "execution_time": execution_time,
                }
            )

            return result

        except Exception as e:
            logger.error(f"Task execution failed for provider {provider_name}: {e}")

            # Update failure statistics
            execution_time = (datetime.now() - start_time).total_seconds()
            self._update_provider_stats(provider_name, False, execution_time, 0)

            await self.context_stream.add_event(
                {
                    "type": "TASK_EXECUTION_FAILURE",
                    "provider": provider_name,
                    "error": str(e),
                    "attempting_fallback": bool(task.get("fallback_providers")),
                }
            )

            # Try fallback providers
            for fallback_provider in task.get("fallback_providers", []):
                try:
                    fallback = self.providers.get(fallback_provider)
                    if fallback and self._is_provider_available(fallback):
                        task["provider"] = fallback_provider
                        return await self._execute_single_task(task, query, user_id)
                except Exception as fe:
                    logger.error(
                        f"Fallback provider {fallback_provider} also failed: {fe}"
                    )
                    continue

            return None

    async def _call_provider(
        self,
        provider: ProviderDefinition,
        task: Dict[str, Any],
        query: str,
        user_id: str,
    ) -> Dict[str, Any]:
        """Call specific provider based on task configuration"""

        provider_name = provider.name
        client = provider.client

        if provider_name == "perplexity":
            # Call Perplexity provider
            return await client.search(
                query=query,
                search_type=task.get("search_type", "general"),
                max_results=task.get("max_results", 10),
            )

        elif provider_name == "exa":
            # Call Exa provider
            return await client.search(
                query=query,
                use_autoprompt=True,
                num_results=task.get("max_results", 10),
                include_domains=task.get("include_domains"),
                exclude_domains=task.get("exclude_domains"),
            )

        elif provider_name == "firecrawl":
            # Use Web Intelligence Manager for Firecrawl
            if task.get("urls"):
                # Scrape specific URLs
                results = []
                for url in task["urls"][:5]:  # Limit to 5 URLs
                    result = await client.quick_scrape(url)
                    if result.success:
                        results.append(result)
                return {"results": results, "provider": "firecrawl"}
            else:
                # General search (not typically supported by Firecrawl directly)
                return await self._web_search_fallback(query, "firecrawl")

        elif provider_name == "apify":
            # Use Web Intelligence Manager for Apify
            return await client.intelligent_web_extraction(
                description=f"Research query: {query}",
                urls=task.get("urls", [f"https://www.google.com/search?q={query}"]),
                complexity=task.get("complexity", "medium"),
                max_pages=task.get("max_pages", 5),
            )

        else:
            raise ValueError(f"Unknown provider: {provider_name}")

    async def _web_search_fallback(
        self, query: str, provider_name: str
    ) -> Dict[str, Any]:
        """Fallback web search when direct search not supported"""
        # This would integrate with a search API or use web scraping
        # For now, return a mock response
        return {
            "results": [],
            "provider": provider_name,
            "fallback_used": True,
            "message": "Direct search not supported, manual URL specification needed",
        }

    def _update_provider_stats(
        self, provider_name: str, success: bool, execution_time: float, cost: float
    ) -> None:
        """Update provider statistics"""

        stats = self.usage_stats["provider_usage"][provider_name]
        stats["requests"] += 1

        if success:
            stats["successes"] += 1
        else:
            stats["failures"] += 1

        stats["total_cost"] += cost

        # Update rolling average response time
        current_avg = stats["avg_response_time"]
        request_count = stats["requests"]
        stats["avg_response_time"] = (
            (current_avg * (request_count - 1)) + execution_time
        ) / request_count

        # Update provider reliability score
        provider = self.providers.get(provider_name)
        if provider:
            success_rate = stats["successes"] / stats["requests"]
            # Weighted average: 70% historical, 30% recent performance
            provider.reliability_score = (0.7 * provider.reliability_score) + (
                0.3 * success_rate
            )

    async def _store_research_in_rag(
        self,
        research_results: Dict[str, Any],
        query: str,
        user_id: str,
        analysis_id: str,
    ) -> None:
        """Store all research results in RAG pipeline for future learning"""

        if not research_results:
            return

        await self.context_stream.add_event(
            {
                "type": "RAG_STORAGE_START",
                "analysis_id": analysis_id,
                "providers_count": len(research_results),
            }
        )

        stored_count = 0

        for provider_name, data in research_results.items():
            try:
                # Create metadata for each result
                metadata = {
                    "provider": provider_name,
                    "user_id": user_id,
                    "analysis_id": analysis_id,
                    "original_query": query,
                    "timestamp": datetime.now().isoformat(),
                    "research_type": "enhanced_research",
                }

                # Store based on data structure
                if isinstance(data, dict):
                    if "results" in data and isinstance(data["results"], list):
                        # Multiple results from provider
                        for i, result in enumerate(data["results"]):
                            if isinstance(result, dict):
                                result_metadata = {**metadata, "result_index": i}
                                await self.rag_pipeline.store_document(
                                    result, result_metadata
                                )
                                stored_count += 1
                    else:
                        # Single result
                        await self.rag_pipeline.store_document(data, metadata)
                        stored_count += 1

                elif isinstance(data, str):
                    # Text content
                    await self.rag_pipeline.store_document({"content": data}, metadata)
                    stored_count += 1

            except Exception as e:
                logger.error(
                    f"Failed to store research result from {provider_name}: {e}"
                )

        self.usage_stats["learning_events"] += stored_count

        await self.context_stream.add_event(
            {
                "type": "RAG_STORAGE_COMPLETE",
                "analysis_id": analysis_id,
                "stored_documents": stored_count,
            }
        )

    async def _update_conversation_memory(
        self,
        user_id: str,
        query: str,
        rag_results: Dict[str, Any],
        new_research: Dict[str, Any],
    ) -> None:
        """Update conversation memory with research context"""

        try:
            # Create memory entry
            memory_data = {
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "rag_used": rag_results.get("has_results", False),
                "research_providers": list(new_research.keys()),
                "research_quality": len(new_research),
                "user_id": user_id,
            }

            # Store in Zep for conversation context
            await self.zep_client.add_memory(
                session_id=f"enhanced_research_{user_id}",
                memory_data=memory_data,
                metadata={"type": "research_session"},
            )

        except Exception as e:
            logger.error(f"Failed to update conversation memory: {e}")

    async def _create_comprehensive_response(
        self,
        analysis_id: str,
        query: str,
        rag_results: Dict[str, Any],
        new_research: Dict[str, Any],
        gaps: List[ResearchGap],
        provider_plan: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Create comprehensive response with all research findings"""

        return {
            "analysis_id": analysis_id,
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "sources": {"rag": rag_results, "research_providers": new_research},
            "research_strategy": {
                "gaps_identified": [gap.gap_type.value for gap in gaps],
                "providers_used": list(new_research.keys()),
                "execution_plan": provider_plan,
            },
            "cost_breakdown": {
                "total_cost": sum(
                    task.get("estimated_cost", 0) for task in provider_plan
                ),
                "provider_costs": {
                    provider: task.get("estimated_cost", 0)
                    for task in provider_plan
                    for provider in [task["provider"]]
                },
            },
            "performance_metrics": {
                "total_sources": len(new_research)
                + (1 if rag_results.get("has_results") else 0),
                "rag_contribution": rag_results.get("results_count", 0),
                "new_research_points": sum(
                    (
                        len(data.get("results", []))
                        if isinstance(data, dict) and "results" in data
                        else 1
                    )
                    for data in new_research.values()
                ),
            },
            "transparency": {
                "research_process": "enhanced_research_manager",
                "version": "2.0",
                "capabilities_used": list(
                    set(
                        cap
                        for provider in new_research.keys()
                        for cap in self.providers.get(
                            provider, ProviderDefinition("", None, [], 0, 0, 0, 0)
                        ).capabilities
                    )
                ),
            },
        }

    async def _basic_research_fallback(
        self, query: str, user_id: str, analysis_id: str
    ) -> Dict[str, Any]:
        """Fallback to basic research when enhanced research fails"""

        try:
            # Try just web intelligence
            result = await self.web_intelligence.quick_scrape(
                f"https://www.google.com/search?q={query}"
            )

            return {
                "analysis_id": analysis_id,
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "fallback_used": True,
                "sources": {
                    "web_intelligence": (
                        result.__dict__ if hasattr(result, "__dict__") else str(result)
                    )
                },
                "error": "Enhanced research failed, using basic fallback",
            }

        except Exception as e:
            logger.error(f"Even basic fallback failed: {e}")
            return {
                "analysis_id": analysis_id,
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "error": "All research methods failed",
                "fallback_used": True,
            }

    def _calculate_daily_cost(self) -> float:
        """Calculate current daily cost across all providers"""

        today = datetime.now().date()
        daily_cost = 0.0

        # This would typically query a cost tracking database
        # For now, return current session cost
        return self.usage_stats.get("total_cost_usd", 0.0)

    def _check_freshness(self, rag_results: Dict[str, Any], max_age_hours: int) -> bool:
        """Check if RAG results meet freshness requirements"""

        if not rag_results.get("results"):
            return False

        current_time = datetime.now()
        cutoff_time = current_time - timedelta(hours=max_age_hours)

        for result in rag_results["results"]:
            timestamp_str = result.get("metadata", {}).get("timestamp")
            if timestamp_str:
                try:
                    result_time = datetime.fromisoformat(
                        timestamp_str.replace("Z", "+00:00")
                    )
                    if result_time >= cutoff_time:
                        return True
                except ValueError:
                    continue

        return False

    async def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the enhanced research system"""

        return {
            "enhanced_research_stats": self.usage_stats,
            "rag_stats": (
                await self.rag_pipeline.get_stats()
                if hasattr(self.rag_pipeline, "get_stats")
                else {}
            ),
            "web_intelligence_stats": (
                await self.web_intelligence.get_comprehensive_stats()
                if hasattr(self.web_intelligence, "get_comprehensive_stats")
                else {}
            ),
            "provider_health": {
                name: {
                    "reliability_score": provider.reliability_score,
                    "avg_response_time": provider.avg_response_time,
                    "current_usage": provider.current_usage,
                    "daily_limit": provider.daily_limit,
                }
                for name, provider in self.providers.items()
            },
            "cost_optimization": self.cost_optimizer,
            "system_health": {
                "total_components": 4,  # RAG, Web Intelligence, Providers, Memory
                "operational_components": len(
                    [
                        c
                        for c in [
                            self.rag_pipeline,
                            self.web_intelligence,
                            self.providers,
                            self.zep_client,
                        ]
                        if c
                    ]
                ),
                "uptime_seconds": (
                    datetime.now() - self.usage_stats["session_start"]
                ).total_seconds(),
            },
        }

    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check of all components"""

        health_results = {
            "overall_health": "healthy",
            "component_health": {},
            "timestamp": datetime.now().isoformat(),
        }

        # Check RAG pipeline
        try:
            rag_health = (
                await self.rag_pipeline.health_check()
                if hasattr(self.rag_pipeline, "health_check")
                else {"status": "unknown"}
            )
            health_results["component_health"]["rag_pipeline"] = rag_health.get(
                "status", "unknown"
            )
        except Exception as e:
            health_results["component_health"]["rag_pipeline"] = f"error: {e}"
            health_results["overall_health"] = "degraded"

        # Check web intelligence
        try:
            web_health = (
                await self.web_intelligence.health_check()
                if hasattr(self.web_intelligence, "health_check")
                else {"overall_health": "unknown"}
            )
            health_results["component_health"]["web_intelligence"] = web_health.get(
                "overall_health", "unknown"
            )
        except Exception as e:
            health_results["component_health"]["web_intelligence"] = f"error: {e}"
            health_results["overall_health"] = "degraded"

        # Check storage systems
        try:
            storage_health = {}
            storage_health["milvus"] = (
                "operational" if self.milvus_client else "not_configured"
            )
            storage_health["supabase"] = (
                "operational" if self.supabase_client else "not_configured"
            )
            storage_health["zep"] = (
                "operational" if self.zep_client else "not_configured"
            )
            health_results["component_health"]["storage"] = storage_health
        except Exception as e:
            health_results["component_health"]["storage"] = f"error: {e}"
            health_results["overall_health"] = "degraded"

        # Check providers
        provider_health = {}
        for name, provider in self.providers.items():
            if provider.reliability_score >= 0.8:
                provider_health[name] = "healthy"
            elif provider.reliability_score >= 0.6:
                provider_health[name] = "degraded"
            else:
                provider_health[name] = "unhealthy"
                health_results["overall_health"] = "degraded"

        health_results["component_health"]["providers"] = provider_health

        return health_results
