"""
Research Provider Manager for METIS Cognitive Platform
Manages multiple research backends (ODR, Perplexity) with intelligent selection and fallback

Features:
- Multi-provider support (Open Deep Research, Perplexity)
- Cost optimization and provider selection
- Automatic fallback on provider failures
- Performance tracking and A/B testing
- Configuration-based provider selection
- Cost tracking and comparison analytics

Author: METIS Cognitive Platform
Date: 2025
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

# Import research providers
from src.engine.integrations.perplexity_client_advanced import (
    AdvancedResearchResult,
    ResearchMode,
    get_advanced_perplexity_client,
)

# ODR client - optional, fallback to Perplexity if not available
try:
    from src.integrations.odr_client import OpenDeepResearchClient, get_odr_client

    ODR_AVAILABLE = True
except ImportError:
    ODR_AVAILABLE = False
    OpenDeepResearchClient = None
    get_odr_client = None

from src.intelligence.research_templates import ResearchTemplateType

logger = logging.getLogger(__name__)


class ResearchProvider(str, Enum):
    """Available research providers"""

    PERPLEXITY = "perplexity"
    ODR = "open_deep_research"
    AUTO = "auto"  # Intelligent selection based on cost/quality


class ProviderSelectionStrategy(str, Enum):
    """Provider selection strategies"""

    COST_OPTIMIZED = "cost_optimized"  # Choose cheapest provider
    QUALITY_OPTIMIZED = "quality_optimized"  # Choose highest quality provider
    BALANCED = "balanced"  # Balance cost vs quality
    ROUND_ROBIN = "round_robin"  # Alternate between providers
    FAILOVER = "failover"  # Primary with backup


@dataclass
class ProviderPerformanceMetrics:
    """Performance metrics for a research provider"""

    provider: ResearchProvider
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    average_cost: float = 0.0
    average_confidence: float = 0.0
    average_source_count: int = 0
    last_used: Optional[datetime] = None

    @property
    def success_rate(self) -> float:
        return self.successful_requests / max(self.total_requests, 1)

    @property
    def cost_per_insight(self) -> float:
        return self.average_cost / max(self.average_source_count, 1)


@dataclass
class ResearchRequest:
    """Research request with provider preferences"""

    query: str
    template_type: Optional[ResearchTemplateType] = None
    context: Optional[Dict[str, Any]] = None
    mode: ResearchMode = ResearchMode.STANDARD
    preferred_provider: ResearchProvider = ResearchProvider.AUTO
    max_cost: Optional[float] = None
    min_quality: Optional[float] = None
    timeout: Optional[int] = None


@dataclass
class ResearchResponse:
    """Enhanced research response with provider metadata"""

    result: AdvancedResearchResult
    provider_used: ResearchProvider
    fallback_used: bool = False
    selection_reason: str = ""
    cost_comparison: Dict[str, float] = field(default_factory=dict)
    performance_metrics: Optional[ProviderPerformanceMetrics] = None


class ResearchProviderManager:
    """Manager for multiple research providers with intelligent selection"""

    def __init__(
        self,
        default_provider: ResearchProvider = ResearchProvider.AUTO,
        selection_strategy: ProviderSelectionStrategy = ProviderSelectionStrategy.BALANCED,
    ):
        self.default_provider = default_provider
        self.selection_strategy = selection_strategy

        # Initialize providers
        self.providers = {}
        self.performance_metrics = {}
        self._initialized = False

    async def initialize(self):
        """Initialize the provider manager (must be called before use)"""
        if not self._initialized:
            await self._initialize_providers()
            self._initialized = True
            logger.info(
                f"✅ Research Provider Manager initialized: {len(self.providers)} providers available"
            )

    async def _initialize_providers(self):
        """Initialize available research providers"""

        # Initialize Perplexity (if available)
        try:
            perplexity_client = await get_advanced_perplexity_client()
            if perplexity_client:
                self.providers[ResearchProvider.PERPLEXITY] = perplexity_client
                self.performance_metrics[ResearchProvider.PERPLEXITY] = (
                    ProviderPerformanceMetrics(provider=ResearchProvider.PERPLEXITY)
                )
                logger.info("✅ Perplexity provider initialized")
        except Exception as e:
            logger.warning(f"⚠️ Perplexity provider unavailable: {e}")

        # Initialize ODR (if available)
        if ODR_AVAILABLE and get_odr_client:
            try:
                odr_client = await get_odr_client()
                if odr_client:
                    self.providers[ResearchProvider.ODR] = odr_client
                    self.performance_metrics[ResearchProvider.ODR] = (
                        ProviderPerformanceMetrics(provider=ResearchProvider.ODR)
                    )
                    logger.info("✅ ODR provider initialized")
            except Exception as e:
                logger.warning(f"⚠️ ODR provider unavailable: {e}")
        else:
            logger.info("⚠️ ODR provider not available - using Perplexity as primary")

        if not self.providers:
            logger.error("❌ No research providers available!")
            raise RuntimeError("No research providers initialized")

    async def conduct_research(self, request: ResearchRequest) -> ResearchResponse:
        """Conduct research using optimal provider selection"""

        # Ensure initialization
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        # Select provider
        selected_provider = self._select_provider(request)

        # Attempt research with selected provider
        try:
            result = await self._execute_research(selected_provider, request)

            # Update metrics
            self._update_metrics(
                selected_provider, True, time.time() - start_time, result
            )

            return ResearchResponse(
                result=result,
                provider_used=selected_provider,
                fallback_used=False,
                selection_reason=self._get_selection_reason(selected_provider, request),
                cost_comparison=self._get_cost_comparison(request),
                performance_metrics=self.performance_metrics.get(selected_provider),
            )

        except Exception as e:
            logger.warning(f"⚠️ Primary provider {selected_provider} failed: {e}")

            # Update failure metrics
            self._update_metrics(selected_provider, False, time.time() - start_time)

            # Attempt fallback
            fallback_result = await self._attempt_fallback(request, selected_provider)
            return fallback_result

    def _select_provider(self, request: ResearchRequest) -> ResearchProvider:
        """Select optimal research provider based on strategy"""

        # Respect user preference if specified
        if (
            request.preferred_provider != ResearchProvider.AUTO
            and request.preferred_provider in self.providers
        ):
            return request.preferred_provider

        # Auto-selection based on strategy
        if self.selection_strategy == ProviderSelectionStrategy.COST_OPTIMIZED:
            return self._select_cheapest_provider(request)
        elif self.selection_strategy == ProviderSelectionStrategy.QUALITY_OPTIMIZED:
            return self._select_highest_quality_provider(request)
        elif self.selection_strategy == ProviderSelectionStrategy.BALANCED:
            return self._select_balanced_provider(request)
        elif self.selection_strategy == ProviderSelectionStrategy.ROUND_ROBIN:
            return self._select_round_robin_provider()
        else:
            # Default to balanced
            return self._select_balanced_provider(request)

    def _select_cheapest_provider(self, request: ResearchRequest) -> ResearchProvider:
        """Select provider with lowest average cost"""
        best_provider = None
        lowest_cost = float("inf")

        for provider, metrics in self.performance_metrics.items():
            if provider in self.providers and metrics.average_cost < lowest_cost:
                lowest_cost = metrics.average_cost
                best_provider = provider

        return best_provider or self._get_fallback_provider()

    def _select_highest_quality_provider(
        self, request: ResearchRequest
    ) -> ResearchProvider:
        """Select provider with highest average quality metrics"""
        best_provider = None
        highest_quality = 0.0

        for provider, metrics in self.performance_metrics.items():
            if provider in self.providers:
                # Quality score = confidence * success_rate * source_count
                quality_score = (
                    metrics.average_confidence
                    * metrics.success_rate
                    * min(metrics.average_source_count / 10, 1.0)
                )

                if quality_score > highest_quality:
                    highest_quality = quality_score
                    best_provider = provider

        return best_provider or self._get_fallback_provider()

    def _select_balanced_provider(self, request: ResearchRequest) -> ResearchProvider:
        """Select provider with best cost/quality balance"""
        best_provider = None
        best_score = 0.0

        for provider, metrics in self.performance_metrics.items():
            if provider in self.providers and metrics.total_requests > 0:
                # Balanced score = quality / (cost + 0.001) * success_rate
                quality = metrics.average_confidence * min(
                    metrics.average_source_count / 10, 1.0
                )
                cost_factor = max(
                    metrics.average_cost, 0.001
                )  # Prevent division by zero

                score = (quality / cost_factor) * metrics.success_rate

                if score > best_score:
                    best_score = score
                    best_provider = provider

        # For new providers (no history), prefer ODR for cost
        if not best_provider:
            return (
                ResearchProvider.ODR
                if ResearchProvider.ODR in self.providers
                else self._get_fallback_provider()
            )

        return best_provider

    def _select_round_robin_provider(self) -> ResearchProvider:
        """Select provider using round-robin strategy"""
        available_providers = list(self.providers.keys())
        if not available_providers:
            raise RuntimeError("No providers available")

        # Simple round-robin based on total requests
        total_requests = sum(
            metrics.total_requests for metrics in self.performance_metrics.values()
        )
        return available_providers[total_requests % len(available_providers)]

    def _get_fallback_provider(self) -> ResearchProvider:
        """Get fallback provider when selection fails"""
        if ResearchProvider.ODR in self.providers:
            return ResearchProvider.ODR
        elif ResearchProvider.PERPLEXITY in self.providers:
            return ResearchProvider.PERPLEXITY
        else:
            raise RuntimeError("No fallback provider available")

    async def _execute_research(
        self, provider: ResearchProvider, request: ResearchRequest
    ) -> AdvancedResearchResult:
        """Execute research with specified provider"""
        client = self.providers[provider]

        return await client.conduct_advanced_research(
            query=request.query,
            template_type=request.template_type,
            context=request.context,
            mode=request.mode,
        )

    async def _attempt_fallback(
        self, request: ResearchRequest, failed_provider: ResearchProvider
    ) -> ResearchResponse:
        """Attempt research with fallback provider"""

        # Find alternative provider
        fallback_providers = [p for p in self.providers.keys() if p != failed_provider]

        if not fallback_providers:
            raise RuntimeError(
                f"No fallback provider available after {failed_provider} failure"
            )

        # Try best alternative
        fallback_provider = self._select_best_alternative(fallback_providers, request)

        try:
            result = await self._execute_research(fallback_provider, request)

            self._update_metrics(
                fallback_provider, True, 0, result
            )  # No additional time cost

            return ResearchResponse(
                result=result,
                provider_used=fallback_provider,
                fallback_used=True,
                selection_reason=f"Fallback from {failed_provider} failure",
                performance_metrics=self.performance_metrics.get(fallback_provider),
            )

        except Exception as e:
            logger.error(f"❌ Fallback provider {fallback_provider} also failed: {e}")
            raise RuntimeError(f"All providers failed. Last error: {str(e)}")

    def _select_best_alternative(
        self, alternatives: List[ResearchProvider], request: ResearchRequest
    ) -> ResearchProvider:
        """Select best alternative provider from available options"""
        if not alternatives:
            raise RuntimeError("No alternative providers available")

        # Prefer ODR for cost-effectiveness
        if ResearchProvider.ODR in alternatives:
            return ResearchProvider.ODR

        return alternatives[0]

    def _update_metrics(
        self,
        provider: ResearchProvider,
        success: bool,
        response_time: float,
        result: Optional[AdvancedResearchResult] = None,
    ):
        """Update performance metrics for provider"""
        metrics = self.performance_metrics.get(provider)
        if not metrics:
            return

        metrics.total_requests += 1
        metrics.last_used = datetime.utcnow()

        if success and result:
            metrics.successful_requests += 1

            # Update averages
            n = metrics.successful_requests
            metrics.average_response_time = (
                (n - 1) * metrics.average_response_time + response_time
            ) / n
            # Handle different cost attribute names
            cost = getattr(result, "estimated_cost", None) or getattr(
                result, "cost_usd", 0.0
            )
            metrics.average_cost = ((n - 1) * metrics.average_cost + cost) / n
            metrics.average_confidence = (
                (n - 1) * metrics.average_confidence + result.overall_confidence
            ) / n
            metrics.average_source_count = (
                (n - 1) * metrics.average_source_count + len(result.sources)
            ) / n

        else:
            metrics.failed_requests += 1

    def _get_selection_reason(
        self, provider: ResearchProvider, request: ResearchRequest
    ) -> str:
        """Get human-readable reason for provider selection"""
        if request.preferred_provider == provider:
            return f"User preference: {provider}"

        strategy_reasons = {
            ProviderSelectionStrategy.COST_OPTIMIZED: f"Lowest cost provider: {provider}",
            ProviderSelectionStrategy.QUALITY_OPTIMIZED: f"Highest quality provider: {provider}",
            ProviderSelectionStrategy.BALANCED: f"Best cost/quality balance: {provider}",
            ProviderSelectionStrategy.ROUND_ROBIN: f"Round-robin selection: {provider}",
        }

        return strategy_reasons.get(self.selection_strategy, f"Selected: {provider}")

    def _get_cost_comparison(self, request: ResearchRequest) -> Dict[str, float]:
        """Get estimated cost comparison between providers"""
        comparison = {}

        for provider, metrics in self.performance_metrics.items():
            if provider in self.providers and metrics.total_requests > 0:
                comparison[provider.value] = metrics.average_cost

        return comparison

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all providers"""
        summary = {
            "providers": [],
            "total_requests": 0,
            "average_cost": 0.0,
            "overall_success_rate": 0.0,
        }

        total_requests = 0
        total_cost = 0.0
        total_success = 0

        for provider, metrics in self.performance_metrics.items():
            provider_summary = {
                "provider": provider.value,
                "requests": metrics.total_requests,
                "success_rate": metrics.success_rate,
                "avg_cost": metrics.average_cost,
                "avg_confidence": metrics.average_confidence,
                "avg_sources": metrics.average_source_count,
                "avg_response_time": metrics.average_response_time,
            }
            summary["providers"].append(provider_summary)

            total_requests += metrics.total_requests
            total_cost += metrics.average_cost * metrics.total_requests
            total_success += metrics.successful_requests

        if total_requests > 0:
            summary["total_requests"] = total_requests
            summary["average_cost"] = total_cost / total_requests
            summary["overall_success_rate"] = total_success / total_requests

        return summary

    def set_provider_preference(self, provider: ResearchProvider):
        """Set default provider preference"""
        if provider in self.providers:
            self.default_provider = provider
            logger.info(f"✅ Default provider set to: {provider}")
        else:
            logger.warning(f"⚠️ Provider {provider} not available")

    def set_selection_strategy(self, strategy: ProviderSelectionStrategy):
        """Set provider selection strategy"""
        self.selection_strategy = strategy
        logger.info(f"✅ Selection strategy set to: {strategy}")


# Global manager instance
_provider_manager: Optional[ResearchProviderManager] = None


def get_research_provider_manager(
    default_provider: ResearchProvider = ResearchProvider.AUTO,
    selection_strategy: ProviderSelectionStrategy = ProviderSelectionStrategy.BALANCED,
) -> ResearchProviderManager:
    """Get or create research provider manager"""
    global _provider_manager
    if _provider_manager is None:
        _provider_manager = ResearchProviderManager(
            default_provider, selection_strategy
        )
    return _provider_manager


async def test_multi_provider_research():
    """Test multi-provider research capabilities"""
    manager = get_research_provider_manager()

    test_request = ResearchRequest(
        query="What are the latest developments in AI-powered business process automation?",
        template_type=ResearchTemplateType.MARKET_ANALYSIS,
        mode=ResearchMode.STANDARD,
    )

    try:
        response = await manager.conduct_research(test_request)

        print("✅ Multi-Provider Research Test Results:")
        print(f"   Provider Used: {response.provider_used}")
        print(f"   Fallback Used: {response.fallback_used}")
        print(f"   Selection Reason: {response.selection_reason}")
        print(f"   Sources: {len(response.result.sources)}")
        print(f"   Confidence: {response.result.overall_confidence:.1%}")
        # Handle different cost attribute names
        cost = getattr(response.result, "estimated_cost", None) or getattr(
            response.result, "cost_usd", 0.0
        )
        print(f"   Cost: ${cost:.4f}")
        print(f"   Summary: {response.result.executive_summary[:100]}...")

        # Print performance summary
        print("\n📊 Provider Performance Summary:")
        summary = manager.get_performance_summary()
        for provider in summary["providers"]:
            print(
                f"   {provider['provider']}: {provider['requests']} requests, "
                f"{provider['success_rate']:.1%} success, ${provider['avg_cost']:.4f} avg cost"
            )

        return response

    except Exception as e:
        print(f"❌ Multi-Provider Test Failed: {e}")
        return None


if __name__ == "__main__":
    asyncio.run(test_multi_provider_research())
