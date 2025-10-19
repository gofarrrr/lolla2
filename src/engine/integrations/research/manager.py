"""
Lightweight Research Manager for METIS
Refactored from research_manager.py for better maintainability
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Literal

# Original client imports
try:
    from ..perplexity_client import (
        get_perplexity_client,
        PerplexityClient,
        KnowledgeQueryType,
    )

    PERPLEXITY_AVAILABLE = True
except ImportError:
    PERPLEXITY_AVAILABLE = False
    PerplexityClient = None
    KnowledgeQueryType = None

# Enhanced research capabilities
try:
    from ..perplexity_client_advanced import (
        get_advanced_perplexity_client,
        AdvancedPerplexityClient,
        AdvancedResearchResult,
        ResearchMode as AdvancedResearchMode,
    )
    from src.intelligence.research_templates import (
        get_research_templates,
        ResearchTemplateType,
    )

    ENHANCED_RESEARCH_AVAILABLE = True
except ImportError:
    ENHANCED_RESEARCH_AVAILABLE = False

# Local imports
from .models import ResearchMode, SearchHit, ResearchResult
from .cache import ResearchCache
from .budgets import RESEARCH_BUDGETS
from .patterns import PatternSelector
from .extractors import FactExtractor, ThemeExtractor
from .validators import SourceValidator, ConsistencyValidator


class ResearchManager:
    """
    Lightweight research manager with budgets, caching, and early stopping
    Single public API: fetch_facts(query, context, mode) -> ResearchResult
    """

    def __init__(self, cache_size: int = 500, cache_ttl_hours: int = 24):
        self.logger = logging.getLogger(__name__)
        self.cache = ResearchCache(max_entries=cache_size, ttl_hours=cache_ttl_hours)
        self._perplexity_client: Optional[PerplexityClient] = None

        # Initialize helper classes
        self.pattern_selector = PatternSelector()
        self.fact_extractor = FactExtractor()
        self.theme_extractor = ThemeExtractor()
        self.source_validator = SourceValidator()
        self.consistency_validator = ConsistencyValidator()

        # Early stopping thresholds
        self.coverage_threshold = 0.5
        self.consistency_threshold = 0.6

        # Enhanced research capabilities
        if ENHANCED_RESEARCH_AVAILABLE:
            self.research_templates = get_research_templates()
            self._advanced_client: Optional[AdvancedPerplexityClient] = None
            self.enhanced_research_enabled = True
            self.logger.info(
                "✅ ResearchManager initialized with enhanced pattern selection"
            )
        else:
            self.research_templates = None
            self._advanced_client = None
            self.enhanced_research_enabled = False
            self.logger.info("✅ ResearchManager initialized (basic mode)")

    def auto_select_research_patterns(self, problem_statement: str) -> List[str]:
        """Delegate to PatternSelector"""
        return self.pattern_selector.auto_select_research_patterns(problem_statement)

    async def _get_perplexity_client(self) -> Optional[PerplexityClient]:
        """Get Perplexity client, with fallback handling"""
        if not PERPLEXITY_AVAILABLE:
            self.logger.warning("⚠️ Perplexity client not available")
            return None

        if self._perplexity_client is None:
            try:
                self._perplexity_client = await get_perplexity_client()
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to get Perplexity client: {e}")
                return None

        return self._perplexity_client

    def _should_stop_early(
        self, sources: List[Dict[str, Any]], calls_made: int
    ) -> bool:
        """Check if early stopping criteria are met"""
        if calls_made < 2:  # Always make at least 1 call
            return False

        coverage_score = self.consistency_validator.calculate_coverage_score(sources)
        consistency_score = self.consistency_validator.calculate_consistency_score(
            sources
        )

        # Early stopping rules
        unique_domains = len(
            set(self.source_validator.extract_domain(s.get("url", "")) for s in sources)
        )

        if unique_domains >= 3 and consistency_score >= 0.65:
            return True
        elif unique_domains >= 2 and consistency_score >= 0.6:
            return True
        elif (
            coverage_score >= self.coverage_threshold
            and consistency_score >= self.consistency_threshold
        ):
            return True

        return False

    def _generate_queries(self, query: str, context: Dict[str, Any]) -> List[str]:
        """Generate up to 3 queries: baseline, contextual, counterfactual"""
        queries = [query]  # Baseline query

        # Contextual query if context available
        if context:
            industry = context.get("industry", "")
            domain = context.get("domain", "")
            if industry or domain:
                contextual = f"{query} in {industry} {domain}".strip()
                queries.append(contextual)

        # Counterfactual query (optional, for deeper modes)
        if "increase" in query.lower() or "improve" in query.lower():
            counterfactual = query.replace("increase", "decrease").replace(
                "improve", "worsen"
            )
            queries.append(f"challenges with {counterfactual}")

        return queries[:3]  # Max 3 queries

    async def _call_perplexity(self, query: str, context: dict) -> List[SearchHit]:
        """Adapter function to call Perplexity with fallback"""
        client = await self._get_perplexity_client()
        if not client:
            self.logger.warning(
                "⚠️ Perplexity client unavailable, returning empty results"
            )
            return []

        try:
            call_start_time = time.time()

            response = await client.query_knowledge(
                query=query,
                query_type=KnowledgeQueryType.CONTEXT_GROUNDING,
                max_tokens=500,
            )

            call_duration_ms = int((time.time() - call_start_time) * 1000)

            # Record Perplexity call in system recorder
            try:
                from src.engine.adapters.monitoring import  # Migrated record_perplexity_call

                record_perplexity_call(
                    query=query,
                    sources=len(response.sources),
                    cost=(
                        response.cost_usd if hasattr(response, "cost_usd") else 0.0014
                    ),  # Approximate cost
                    time_ms=call_duration_ms,
                    mode="context_grounding",
                    success=True,
                )
            except ImportError:
                self.logger.warning(
                    "System recorder not available for Perplexity calls"
                )

            # Convert to SearchHit format
            hits = []
            for i, source_url in enumerate(response.sources[:5]):  # Limit to 5 sources
                domain = self.source_validator.extract_domain(source_url)
                hit = SearchHit(
                    url=source_url,
                    title=f"Source {i+1}",  # Simplified title
                    content=(
                        response.content[:200] if i == 0 else ""
                    ),  # First hit gets content
                    domain=domain,
                    confidence=response.confidence,
                )
                hits.append(hit)

            return hits

        except Exception as e:
            self.logger.error(f"❌ Perplexity call failed: {e}")

            # Record failed Perplexity call
            try:
                from src.engine.adapters.monitoring import  # Migrated record_perplexity_call

                record_perplexity_call(
                    query=query,
                    sources=0,
                    cost=0.0,
                    time_ms=0,
                    mode="context_grounding",
                    success=False,
                )
            except ImportError:
                pass

            return []

    async def fetch_facts(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        mode: Literal["fast", "moderate", "deep"] = "fast",
    ) -> ResearchResult:
        """
        Main public API: fetch facts with citations using budget and early stopping

        Args:
            query: Research query string
            context: Optional context dict for query refinement
            mode: Research mode (fast/moderate/deep)

        Returns:
            ResearchResult with summary, bullets, sources, and metrics
        """
        start_time = time.time()
        context = context or {}
        budget = RESEARCH_BUDGETS[ResearchMode(mode)]

        self.logger.info(f"🔍 Starting research: {mode} mode | {query[:50]}...")

        # Enhanced research with pattern selection (if available)
        if self.enhanced_research_enabled and self.research_templates:
            return await self._fetch_facts_enhanced(query, context, mode)

        # Check cache first (legacy path)
        cached_result = self.cache.get(query, context)
        if cached_result:
            self.logger.info("⚡ Cache hit - returning cached result")
            return cached_result

        return await self._fetch_facts_legacy(query, context, mode, budget, start_time)

    async def _fetch_facts_legacy(
        self, query: str, context: Dict[str, Any], mode: str, budget, start_time: float
    ) -> ResearchResult:
        """Execute legacy research flow with full implementation"""

        # Initialize result structure
        all_sources = []
        queries_attempted = []
        early_stopped = False
        timeout_occurred = False

        try:
            # Generate queries with budget awareness
            possible_queries = self._generate_queries(query, context)

            # Execute queries with parallel processing for optimization
            queries_to_run = possible_queries[
                : budget.max_calls
            ]  # Limit queries by budget
            queries_attempted = queries_to_run.copy()

            # Create tasks for parallel execution
            tasks = []
            for i, query_text in enumerate(queries_to_run):
                # Check time budget before creating task
                elapsed = time.time() - start_time
                if elapsed >= budget.max_time_seconds:
                    timeout_occurred = True
                    self.logger.warning(f"⏰ Time budget exceeded: {elapsed:.1f}s")
                    break

                remaining_time = budget.max_time_seconds - elapsed
                timeout = max(
                    10.0, remaining_time / len(queries_to_run)
                )  # Distribute time across queries

                task = asyncio.create_task(
                    asyncio.wait_for(
                        self._call_perplexity(query_text, context), timeout=timeout
                    )
                )
                tasks.append((query_text, task))

            # Execute all tasks in parallel
            completed_tasks = 0
            for query_text, task in tasks:
                try:
                    search_hits = await task

                    # Convert hits to source format
                    for hit in search_hits:
                        source = {
                            "url": hit.url,
                            "title": hit.title,
                            "content": hit.content,
                            "domain": hit.domain,
                            "date": hit.date,
                        }
                        all_sources.append(source)

                    completed_tasks += 1

                    # Check early stopping after at least 1 successful call
                    if completed_tasks >= 1 and self._should_stop_early(
                        all_sources, completed_tasks
                    ):
                        early_stopped = True
                        self.logger.info(
                            f"✋ Early stopping triggered after {completed_tasks} parallel calls"
                        )
                        # Cancel remaining tasks
                        for _, remaining_task in tasks[completed_tasks:]:
                            if not remaining_task.done():
                                remaining_task.cancel()
                        break

                except asyncio.TimeoutError:
                    self.logger.warning(
                        f"⏰ Parallel query timeout: {query_text[:30]}..."
                    )
                    continue
                except asyncio.CancelledError:
                    self.logger.info(
                        f"🚫 Query cancelled due to early stopping: {query_text[:30]}..."
                    )
                    continue
                except Exception as e:
                    self.logger.error(f"❌ Parallel query failed: {e}")
                    continue

        except Exception as e:
            self.logger.error(f"❌ Research execution failed: {e}")
            timeout_occurred = True

        return await self._build_research_result(
            all_sources,
            queries_attempted,
            mode,
            start_time,
            early_stopped,
            timeout_occurred,
            query,
            context,
        )

    async def _build_research_result(
        self,
        all_sources: List[Dict],
        queries_attempted: List[str],
        mode: str,
        start_time: float,
        early_stopped: bool,
        timeout_occurred: bool,
        query: str,
        context: Dict[str, Any],
    ) -> ResearchResult:
        """Build final ResearchResult from collected data"""

        # Calculate final metrics
        time_spent_ms = int((time.time() - start_time) * 1000)
        coverage_score = self.consistency_validator.calculate_coverage_score(
            all_sources
        )
        consistency_score = self.consistency_validator.calculate_consistency_score(
            all_sources
        )
        confidence = 0.5 * coverage_score + 0.5 * consistency_score

        # Handle failure case
        if timeout_occurred or not all_sources:
            confidence = 0.3  # Low confidence fallback

        # Generate summary and bullets from sources
        summary = ""
        bullets = []

        if all_sources:
            # Create summary from first few sources
            unique_domains = set(
                self.source_validator.extract_domain(s.get("url", ""))
                for s in all_sources
            )
            summary = f"Research found {len(all_sources)} sources across {len(unique_domains)} domains"

            # Create bullet points (max 5)
            for i, source in enumerate(all_sources[:5]):
                if source.get("content"):
                    bullet = source["content"][:100].strip()
                    if bullet:
                        bullets.append(bullet)

        # Enhanced attribution processing
        facts = []
        fact_verification = {}
        source_credibility = {}
        cross_reference_score = 0.0
        contradictions = []

        if all_sources:
            # Extract and verify facts
            facts = self.fact_extractor.extract_facts_from_sources(all_sources)
            fact_verification = self.fact_extractor.verify_facts_across_sources(
                facts, all_sources
            )

            # Calculate source credibility scores
            for source in all_sources:
                url = source.get("url", "")
                if url:
                    source_credibility[url] = (
                        self.source_validator.calculate_source_credibility(source)
                    )

            # Calculate cross-reference score
            cross_reference_score = (
                self.consistency_validator.calculate_cross_reference_score(
                    all_sources, facts
                )
            )

            # Detect contradictions
            contradictions = self.consistency_validator.detect_contradictions(
                all_sources, facts
            )

            self.logger.info(
                f"📊 Attribution: {len(facts)} facts extracted, "
                f"avg credibility={sum(source_credibility.values())/len(source_credibility):.2f}, "
                f"cross-ref={cross_reference_score:.2f}, "
                f"{len(contradictions)} contradictions"
            )

        # Create result with enhanced attribution
        result = ResearchResult(
            summary=summary,
            bullets=bullets,
            sources=all_sources,
            coverage_score=coverage_score,
            consistency_score=consistency_score,
            confidence=confidence,
            time_spent_ms=time_spent_ms,
            queries=queries_attempted,
            mode_used=mode,
            early_stopped=early_stopped,
            timeout_occurred=timeout_occurred,
            cache_hit=False,
            fact_verification=fact_verification,
            source_credibility=source_credibility,
            cross_reference_score=cross_reference_score,
            fact_extraction_count=len(facts),
            contradictions_detected=contradictions,
        )

        # Cache successful results
        if not timeout_occurred and all_sources:
            self.cache.put(query, context, result)

        self.logger.info(
            f"✅ Research completed: {len(all_sources)} sources | "
            f"coverage={coverage_score:.2f} | consistency={consistency_score:.2f} | "
            f"confidence={confidence:.2f} | {time_spent_ms}ms"
        )

        # Optional: Store result for user if context includes user information
        await self._maybe_store_user_research_result(
            query, context, result, all_sources
        )

        return result

    async def _get_advanced_client(self) -> Optional[AdvancedPerplexityClient]:
        """Get advanced Perplexity client for enhanced research"""
        if not ENHANCED_RESEARCH_AVAILABLE:
            return None

        if self._advanced_client is None:
            try:
                self._advanced_client = await get_advanced_perplexity_client()
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to get advanced client: {e}")
                return None

        return self._advanced_client

    async def _fetch_facts_enhanced(
        self, query: str, context: Dict[str, Any], mode: str
    ) -> ResearchResult:
        """Enhanced research with pattern selection and advanced capabilities"""

        # Select optimal research template based on query and context
        template_type = self.research_templates.select_template_by_context(
            {"query": query, **context}
        )

        self.logger.info(f"🎯 Selected research template: {template_type.value}")

        # Map legacy modes to advanced modes
        mode_mapping = {
            "fast": AdvancedResearchMode.RAPID,
            "moderate": AdvancedResearchMode.STANDARD,
            "deep": AdvancedResearchMode.COMPREHENSIVE,
        }

        advanced_mode = mode_mapping.get(mode, AdvancedResearchMode.STANDARD)

        try:
            # Get advanced client
            client = await self._get_advanced_client()
            if not client:
                self.logger.warning(
                    "⚠️ Advanced client unavailable, falling back to legacy"
                )
                start_time = time.time()
                budget = RESEARCH_BUDGETS[ResearchMode(mode)]
                return await self._fetch_facts_legacy(
                    query, context, mode, budget, start_time
                )

            # Conduct advanced research
            advanced_result = await client.conduct_advanced_research(
                query=query,
                template_type=template_type,
                context=context,
                mode=advanced_mode,
            )

            # Store advanced result if user context provided
            await self._maybe_store_advanced_research_result(
                query, context, advanced_result
            )

            # Convert advanced result to legacy format for compatibility
            return self._convert_advanced_to_legacy_result(advanced_result)

        except Exception as e:
            self.logger.error(
                f"❌ Enhanced research failed: {e}, falling back to legacy"
            )
            start_time = time.time()
            budget = RESEARCH_BUDGETS[ResearchMode(mode)]
            return await self._fetch_facts_legacy(
                query, context, mode, budget, start_time
            )

    def _convert_advanced_to_legacy_result(
        self, advanced: AdvancedResearchResult
    ) -> ResearchResult:
        """Convert advanced research result to legacy format for compatibility"""

        # Convert enhanced sources to simple format
        legacy_sources = []
        for source in advanced.sources:
            legacy_source = {
                "url": source.url,
                "title": source.title,
                "content": source.content,
                "domain": source.domain,
                "date": source.date,
            }
            legacy_sources.append(legacy_source)

        # Extract bullets from key insights
        bullets = [insight.claim[:200] for insight in advanced.key_insights[:5]]

        return ResearchResult(
            summary=advanced.executive_summary,
            bullets=bullets,
            sources=legacy_sources,
            coverage_score=advanced.coverage_completeness,
            consistency_score=advanced.fact_validation_score,
            confidence=advanced.overall_confidence,
            time_spent_ms=advanced.total_processing_time_ms,
            queries=advanced.queries_executed,
            mode_used=advanced.mode_used.value,
            early_stopped=False,
            timeout_occurred=False,
            cache_hit=False,
            # Enhanced attribution fields
            fact_verification=getattr(advanced, "fact_verification", {}),
            source_credibility=getattr(advanced, "source_credibility", {}),
            cross_reference_score=getattr(
                advanced.cross_reference_analysis, "consistency_score", 0.0
            ),
            fact_extraction_count=len(advanced.key_insights),
            contradictions_detected=getattr(
                advanced.cross_reference_analysis, "contradictions", []
            ),
        )

    def get_template_recommendations(self, query: str) -> List[Dict[str, Any]]:
        """Get research template recommendations for a query"""

        if not self.enhanced_research_enabled or not self.research_templates:
            return []

        return self.research_templates.get_template_recommendations(query)

    def get_enhanced_capabilities_status(self) -> Dict[str, Any]:
        """Get status of enhanced research capabilities"""

        return {
            "enhanced_research_enabled": self.enhanced_research_enabled,
            "templates_available": bool(self.research_templates),
            "advanced_client_available": ENHANCED_RESEARCH_AVAILABLE,
            "template_count": (
                len(self.research_templates.templates) if self.research_templates else 0
            ),
            "supported_patterns": (
                list(self.research_templates.templates.keys())
                if self.research_templates
                else []
            ),
        }

    async def _maybe_store_user_research_result(
        self,
        query: str,
        context: Dict[str, Any],
        result: ResearchResult,
        sources: List[Dict[str, Any]],
    ) -> None:
        """Optionally store research result if user context is provided"""

        # Check if this is a user research request (has user_id and research_tier)
        user_id = context.get("user_id")
        research_tier = context.get("research_tier")

        if not user_id or not research_tier:
            # Not a user research request, skip storage
            return

        try:
            # Import storage components (lazy import to avoid circular dependencies)
            from src.persistence.user_research_storage import (
                get_user_research_storage,
                UserResearchRequest,
            )
            from supabase import create_client
            import os

            # Get Supabase client
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv(
                "SUPABASE_SERVICE_ROLE_KEY", os.getenv("SUPABASE_ANON_KEY")
            )

            if not supabase_url or not supabase_key:
                self.logger.warning(
                    "⚠️ Supabase not configured, skipping research result storage"
                )
                return

            supabase = create_client(supabase_url, supabase_key)
            storage = get_user_research_storage(supabase)

            # Create user research request from context
            request = UserResearchRequest(
                user_id=user_id,
                engagement_id=context.get("engagement_id"),
                research_tier=research_tier,
                question_depth=context.get("question_depth", "essential"),
                problem_statement=context.get("problem_statement", query),
                progressive_questions=context.get("progressive_questions", {}),
                context_data=context,
            )

            # Store the research result
            research_result_id = await storage.store_research_result(
                request=request,
                result=result,
                sources=sources,
                perplexity_model=context.get("perplexity_model", "sonar-pro"),
            )

            self.logger.info(f"✅ Stored user research result: {research_result_id}")

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to store user research result: {e}")
            # Don't raise exception - storage failure shouldn't break research

    async def _maybe_store_advanced_research_result(
        self, query: str, context: Dict[str, Any], advanced_result
    ) -> None:
        """Optionally store advanced research result if user context is provided"""

        # Check if this is a user research request
        user_id = context.get("user_id")
        research_tier = context.get("research_tier")

        if not user_id or not research_tier:
            return

        try:
            # Import storage components
            from src.persistence.user_research_storage import (
                get_user_research_storage,
                UserResearchRequest,
            )
            from supabase import create_client
            import os

            # Get Supabase client
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv(
                "SUPABASE_SERVICE_ROLE_KEY", os.getenv("SUPABASE_ANON_KEY")
            )

            if not supabase_url or not supabase_key:
                self.logger.warning(
                    "⚠️ Supabase not configured, skipping advanced research result storage"
                )
                return

            supabase = create_client(supabase_url, supabase_key)
            storage = get_user_research_storage(supabase)

            # Create user research request
            request = UserResearchRequest(
                user_id=user_id,
                engagement_id=context.get("engagement_id"),
                research_tier=research_tier,
                question_depth=context.get("question_depth", "expert"),
                problem_statement=context.get("problem_statement", query),
                progressive_questions=context.get("progressive_questions", {}),
                context_data=context,
            )

            # Store the advanced research result
            research_result_id = await storage.store_advanced_research_result(
                request=request, result=advanced_result
            )

            self.logger.info(
                f"✅ Stored advanced user research result: {research_result_id}"
            )

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to store advanced user research result: {e}")
            # Don't raise exception - storage failure shouldn't break research


# Global ResearchManager instance
_research_manager_instance: Optional[ResearchManager] = None


def get_research_manager() -> ResearchManager:
    """Get or create global ResearchManager instance"""
    global _research_manager_instance

    if _research_manager_instance is None:
        _research_manager_instance = ResearchManager()

    return _research_manager_instance
