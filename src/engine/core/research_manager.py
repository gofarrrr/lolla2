"""
Research Manager - Provider-Agnostic Research Orchestration
Implements fallback logic and provider abstraction for Operation "Research Resilience"
"""

import logging
from typing import List, Dict, Any
from datetime import datetime

from src.engine.providers.research.base import (
    ResearchProvider,
    ResearchResult,
    ResearchTier,
)
from src.core.unified_context_stream import UnifiedContextStream
from src.core.events.event_emitters import ResearchEventEmitter

logger = logging.getLogger(__name__)


class ResearchManager:
    """
    Research Manager orchestrates multiple research providers with fallback logic

    Key Features:
    1. Provider-agnostic research execution
    2. Automatic fallback when primary provider fails
    3. Transparent logging of provider attempts and failures
    4. Cost and performance optimization across providers
    """

    def __init__(
        self, providers: List[ResearchProvider], context_stream: UnifiedContextStream
    ):
        """
        Initialize ResearchManager with ordered list of providers

        Args:
            providers: List of research providers in order of preference (primary first)
            context_stream: Unified context stream for transparency logging
        """
        self.providers = providers
        self.context_stream = context_stream
        self.research_events = ResearchEventEmitter(
            context_stream, default_metadata={"component": "ResearchManager"}
        )
        self.usage_stats = {
            "total_searches": 0,
            "successful_searches": 0,
            "fallback_events": 0,
            "provider_success_rates": {},
            "total_cost_usd": 0.0,
        }

        # Initialize provider success tracking
        for provider in providers:
            self.usage_stats["provider_success_rates"][provider.provider_name] = {
                "attempts": 0,
                "successes": 0,
                "success_rate": 1.0,
            }

        logger.info(
            f"🔍 ResearchManager initialized with {len(providers)} providers: {[p.provider_name for p in providers]}"
        )

    async def execute_search(
        self,
        query: str,
        num_results: int = 5,
        tier: ResearchTier = ResearchTier.REGULAR,
        operation_context: str = "",
    ) -> ResearchResult:
        """
        Execute research query with automatic fallback between providers

        Args:
            query: Research question to search for
            num_results: Number of results to return
            tier: Research tier for cost/quality optimization
            operation_context: Context for logging and tracking

        Returns:
            ResearchResult: Results from the first successful provider

        Raises:
            RuntimeError: If all providers fail
        """
        self.usage_stats["total_searches"] += 1

        logger.info(
            f"🔍 Starting research query: '{query[:100]}...' with {len(self.providers)} providers"
        )

        last_error = None

        for i, provider in enumerate(self.providers):
            provider_name = provider.provider_name
            is_fallback = i > 0

            try:
                # Log research provider request
                self.research_events.request(
                    provider=provider_name,
                    query=query[:200],
                    provider_type="research",
                    query_length=len(query),
                    num_results=num_results,
                    tier=tier.value,
                    is_fallback=is_fallback,
                    fallback_position=i,
                    operation_context=operation_context,
                )

                # Track attempt
                self.usage_stats["provider_success_rates"][provider_name][
                    "attempts"
                ] += 1

                logger.info(
                    f"{'🔄 Fallback to' if is_fallback else '🎯 Primary'} {provider_name} provider"
                )

                # Execute search
                start_time = datetime.utcnow()
                result = await provider.search(query, num_results, tier)
                processing_time_ms = (
                    datetime.utcnow() - start_time
                ).total_seconds() * 1000

                # Check if result is successful
                if not result.success:
                    raise RuntimeError(
                        f"Provider returned failed result: {result.error_message}"
                    )

                # Log successful research provider response
                self.research_events.response(
                    provider=provider_name,
                    status="success",
                    latency_ms=processing_time_ms,
                    citations_count=len(result.sources),
                    result_preview=str(result.content)[:200],
                    provider_type="research",
                    confidence=result.confidence,
                    tokens_used=result.tokens_used,
                    cost_usd=result.cost_usd,
                    operation_context=operation_context,
                    raw_results=result.raw_provider_response,
                )

                # Update success statistics
                self.usage_stats["successful_searches"] += 1
                self.usage_stats["provider_success_rates"][provider_name][
                    "successes"
                ] += 1
                self.usage_stats["total_cost_usd"] += result.cost_usd

                # Recalculate success rate
                provider_stats = self.usage_stats["provider_success_rates"][
                    provider_name
                ]
                provider_stats["success_rate"] = (
                    provider_stats["successes"] / provider_stats["attempts"]
                )

                logger.info(
                    f"✅ Research successful with {provider_name}: "
                    f"{len(result.sources)} sources, "
                    f"confidence {result.confidence:.2f}, "
                    f"${result.cost_usd:.4f}, "
                    f"{processing_time_ms:.1f}ms"
                )

                return result

            except Exception as e:
                processing_time_ms = (
                    (datetime.utcnow() - start_time).total_seconds() * 1000
                    if "start_time" in locals()
                    else 0
                )
                last_error = e

                # Log failed research provider response
                self.research_events.response(
                    provider=provider_name,
                    status="failure",
                    latency_ms=processing_time_ms,
                    provider_type="research",
                    error=str(e),
                    operation_context=operation_context,
                )

                logger.warning(f"❌ {provider_name} failed: {e}")

                # If this isn't the last provider, log fallback event
                if i < len(self.providers) - 1:
                    next_provider = self.providers[i + 1].provider_name

                    # Log research provider fallback
                    self.research_events.fallback(
                        failed_provider=provider_name,
                        fallback_provider=next_provider,
                        failure_reason=str(e),
                        operation_context=operation_context,
                    )

                    self.usage_stats["fallback_events"] += 1
                    logger.info(
                        f"🔄 Falling back from {provider_name} to {next_provider}"
                    )

                continue

        # All providers failed
        logger.error(
            f"❌ All {len(self.providers)} research providers failed. Last error: {last_error}"
        )

        raise RuntimeError(f"All research providers failed. Last error: {last_error}")

    async def check_provider_health(self) -> Dict[str, Any]:
        """
        Check health status of all providers

        Returns:
            Dict containing health status of each provider
        """
        health_status = {
            "timestamp": datetime.utcnow().isoformat(),
            "providers": {},
            "overall_healthy": False,
        }

        healthy_count = 0

        for provider in self.providers:
            try:
                is_available = await provider.is_available()
                usage_metrics = provider.get_usage_metrics()

                health_status["providers"][provider.provider_name] = {
                    "available": is_available,
                    "usage_metrics": usage_metrics,
                    "error": None,
                }

                if is_available:
                    healthy_count += 1

            except Exception as e:
                health_status["providers"][provider.provider_name] = {
                    "available": False,
                    "usage_metrics": {},
                    "error": str(e),
                }

        health_status["overall_healthy"] = healthy_count > 0
        health_status["healthy_providers"] = healthy_count
        health_status["total_providers"] = len(self.providers)

        return health_status

    def get_usage_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive usage statistics across all providers

        Returns:
            Dict containing usage statistics and performance metrics
        """
        return {
            "research_manager": {
                "total_searches": self.usage_stats["total_searches"],
                "successful_searches": self.usage_stats["successful_searches"],
                "success_rate": (
                    self.usage_stats["successful_searches"]
                    / self.usage_stats["total_searches"]
                    if self.usage_stats["total_searches"] > 0
                    else 0.0
                ),
                "fallback_events": self.usage_stats["fallback_events"],
                "fallback_rate": (
                    self.usage_stats["fallback_events"]
                    / self.usage_stats["total_searches"]
                    if self.usage_stats["total_searches"] > 0
                    else 0.0
                ),
                "total_cost_usd": self.usage_stats["total_cost_usd"],
            },
            "providers": self.usage_stats["provider_success_rates"],
            "provider_order": [p.provider_name for p in self.providers],
        }

    async def optimize_provider_order(self) -> None:
        """
        Optimize provider order based on success rates and performance
        (Advanced feature for future implementation)
        """
        # Sort providers by success rate (descending)
        provider_stats = self.usage_stats["provider_success_rates"]

        # Only reorder if we have enough data (at least 10 attempts per provider)
        eligible_providers = []
        for provider in self.providers:
            stats = provider_stats[provider.provider_name]
            if stats["attempts"] >= 10:
                eligible_providers.append((provider, stats["success_rate"]))

        if len(eligible_providers) >= 2:
            # Sort by success rate (descending)
            eligible_providers.sort(key=lambda x: x[1], reverse=True)

            # Update provider order
            optimized_providers = [p[0] for p in eligible_providers]

            # Add any providers without enough data at the end
            for provider in self.providers:
                if provider not in optimized_providers:
                    optimized_providers.append(provider)

            if optimized_providers != self.providers:
                old_order = [p.provider_name for p in self.providers]
                new_order = [p.provider_name for p in optimized_providers]

                self.providers = optimized_providers
                logger.info(f"🔧 Optimized provider order: {old_order} → {new_order}")
