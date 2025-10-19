"""
Cost Optimizer - METIS 2.0 Intelligent Cost Management
=====================================================

Advanced cost optimization system for all providers across the METIS platform.
Includes budget management, provider selection, usage tracking, and predictive
cost modeling for optimal resource allocation.

Key Features:
1. Multi-provider cost tracking and optimization
2. Dynamic budget management with alerts and controls
3. Intelligent provider selection based on cost-effectiveness
4. Real-time usage monitoring and forecasting
5. Cost anomaly detection and automatic adjustments
6. ROI analysis and value optimization
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import statistics

# Storage imports
from src.storage.supabase_client import SupabaseClient
from src.core.unified_context_stream import UnifiedContextStream

logger = logging.getLogger(__name__)


class CostCategory(Enum):
    """Categories for different cost types"""

    RESEARCH_PROVIDER = "research_provider"
    WEB_SCRAPING = "web_scraping"
    EMBEDDINGS = "embeddings"
    LLM_PROCESSING = "llm_processing"
    STORAGE = "storage"
    API_CALLS = "api_calls"


class BudgetStatus(Enum):
    """Budget status levels"""

    HEALTHY = "healthy"  # < 70% of budget used
    WARNING = "warning"  # 70-90% of budget used
    CRITICAL = "critical"  # 90-100% of budget used
    EXCEEDED = "exceeded"  # > 100% of budget used


@dataclass
class ProviderCostProfile:
    """Detailed cost profile for each provider"""

    name: str
    category: CostCategory
    base_cost_per_operation: float
    rate_limit_per_hour: int
    rate_limit_per_day: Optional[int] = None
    volume_discounts: Dict[int, float] = field(
        default_factory=dict
    )  # volume -> discount %
    performance_multiplier: float = 1.0  # Higher = better performance per dollar
    reliability_score: float = 1.0  # 0-1 scale
    setup_cost: float = 0.0
    minimum_charge: float = 0.0
    currency: str = "USD"

    def calculate_cost(
        self, operations: int, operation_metadata: Dict[str, Any] = None
    ) -> float:
        """Calculate cost for given number of operations"""
        if operations == 0:
            return 0.0

        # Apply minimum charge
        base_cost = max(operations * self.base_cost_per_operation, self.minimum_charge)

        # Apply volume discounts
        discount = 0.0
        for volume_threshold in sorted(self.volume_discounts.keys(), reverse=True):
            if operations >= volume_threshold:
                discount = self.volume_discounts[volume_threshold]
                break

        discounted_cost = base_cost * (1 - discount / 100)

        # Add setup cost if first operation
        total_cost = discounted_cost + (self.setup_cost if operations > 0 else 0)

        return round(total_cost, 6)


@dataclass
class UsageRecord:
    """Record of provider usage"""

    provider: str
    category: CostCategory
    operation_type: str
    cost: float
    timestamp: datetime
    user_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    analysis_id: Optional[str] = None


@dataclass
class BudgetAllocation:
    """Budget allocation for different categories"""

    category: CostCategory
    allocated_amount: float
    used_amount: float = 0.0
    remaining_amount: float = None
    warning_threshold: float = 0.7  # 70%
    critical_threshold: float = 0.9  # 90%

    def __post_init__(self):
        if self.remaining_amount is None:
            self.remaining_amount = self.allocated_amount - self.used_amount

    @property
    def usage_percentage(self) -> float:
        """Get usage as percentage of allocated budget"""
        if self.allocated_amount == 0:
            return 0.0
        return (self.used_amount / self.allocated_amount) * 100

    @property
    def status(self) -> BudgetStatus:
        """Get current budget status"""
        usage_pct = self.usage_percentage / 100

        if usage_pct > 1.0:
            return BudgetStatus.EXCEEDED
        elif usage_pct >= self.critical_threshold:
            return BudgetStatus.CRITICAL
        elif usage_pct >= self.warning_threshold:
            return BudgetStatus.WARNING
        else:
            return BudgetStatus.HEALTHY


class CostOptimizer:
    """
    Advanced cost optimization system for METIS 2.0

    Manages costs across all providers with intelligent optimization,
    budget controls, and predictive cost modeling.
    """

    # Provider cost profiles with detailed configurations
    DEFAULT_PROVIDER_COSTS = {
        "perplexity": ProviderCostProfile(
            name="perplexity",
            category=CostCategory.RESEARCH_PROVIDER,
            base_cost_per_operation=0.001,
            rate_limit_per_hour=100,
            rate_limit_per_day=1000,
            volume_discounts={500: 10, 1000: 20},  # 10% at 500+, 20% at 1000+
            performance_multiplier=1.2,
            reliability_score=0.95,
        ),
        "exa": ProviderCostProfile(
            name="exa",
            category=CostCategory.RESEARCH_PROVIDER,
            base_cost_per_operation=0.0008,
            rate_limit_per_hour=50,
            rate_limit_per_day=500,
            volume_discounts={300: 15, 800: 25},
            performance_multiplier=1.1,
            reliability_score=0.92,
        ),
        "firecrawl": ProviderCostProfile(
            name="firecrawl",
            category=CostCategory.WEB_SCRAPING,
            base_cost_per_operation=0.001,  # Basic scrape
            rate_limit_per_hour=30,
            rate_limit_per_day=500,
            performance_multiplier=1.3,
            reliability_score=0.88,
        ),
        "apify": ProviderCostProfile(
            name="apify",
            category=CostCategory.WEB_SCRAPING,
            base_cost_per_operation=0.005,  # Per actor run
            rate_limit_per_hour=20,
            rate_limit_per_day=100,
            volume_discounts={50: 8, 100: 15},
            performance_multiplier=1.0,
            reliability_score=0.85,
        ),
        "voyage": ProviderCostProfile(
            name="voyage",
            category=CostCategory.EMBEDDINGS,
            base_cost_per_operation=0.00002,  # Per embedding
            rate_limit_per_hour=10000,
            rate_limit_per_day=100000,
            volume_discounts={10000: 10, 50000: 20},
            performance_multiplier=1.4,
            reliability_score=0.98,
        ),
        "openai": ProviderCostProfile(
            name="openai",
            category=CostCategory.LLM_PROCESSING,
            base_cost_per_operation=0.03,  # Per 1k tokens (estimate)
            rate_limit_per_hour=3500,  # RPM converted
            rate_limit_per_day=50000,
            performance_multiplier=1.5,
            reliability_score=0.96,
        ),
    }

    def __init__(self, config: Dict[str, Any]):
        """Initialize Cost Optimizer with configuration"""

        self.config = config
        self.supabase = SupabaseClient(config.get("supabase", {}))
        from src.core.unified_context_stream import get_unified_context_stream
        self.context_stream = get_unified_context_stream()

        # Budget configuration
        self.daily_budget = config.get("daily_budget", 10.0)  # $10 default
        self.monthly_budget = config.get("monthly_budget", 300.0)  # $300 default
        self.emergency_threshold = config.get("emergency_threshold", 0.95)  # 95%

        # Cost optimization settings
        self.cost_efficiency_weight = config.get("cost_efficiency_weight", 0.4)
        self.performance_weight = config.get("performance_weight", 0.3)
        self.reliability_weight = config.get("reliability_weight", 0.3)

        # Provider profiles (can be overridden by config)
        self.provider_profiles = self.DEFAULT_PROVIDER_COSTS.copy()
        self._update_provider_profiles_from_config(config.get("provider_overrides", {}))

        # Budget allocations
        self.budget_allocations = self._initialize_budget_allocations()

        # Usage tracking
        self.usage_history: List[UsageRecord] = []
        self.daily_usage = {}
        self.monthly_usage = {}

        # Cost forecasting
        self.cost_predictions = {}
        self.usage_patterns = {}

    def _initialize_budget_allocations(self) -> Dict[CostCategory, BudgetAllocation]:
        """Initialize budget allocations for each cost category"""

        # Default allocation percentages
        default_allocations = {
            CostCategory.RESEARCH_PROVIDER: 0.4,  # 40% for research
            CostCategory.WEB_SCRAPING: 0.3,  # 30% for web scraping
            CostCategory.EMBEDDINGS: 0.15,  # 15% for embeddings
            CostCategory.LLM_PROCESSING: 0.1,  # 10% for LLM
            CostCategory.STORAGE: 0.03,  # 3% for storage
            CostCategory.API_CALLS: 0.02,  # 2% for misc API calls
        }

        allocations = {}
        for category, percentage in default_allocations.items():
            allocations[category] = BudgetAllocation(
                category=category, allocated_amount=self.daily_budget * percentage
            )

        return allocations

    def _update_provider_profiles_from_config(self, overrides: Dict[str, Any]) -> None:
        """Update provider profiles with config overrides"""

        for provider_name, override_config in overrides.items():
            if provider_name in self.provider_profiles:
                profile = self.provider_profiles[provider_name]

                # Update fields that are provided in config
                for field_name, value in override_config.items():
                    if hasattr(profile, field_name):
                        setattr(profile, field_name, value)

    async def select_optimal_provider(
        self,
        task_type: str,
        eligible_providers: List[str],
        estimated_operations: int = 1,
        requirements: Dict[str, Any] = None,
    ) -> Optional[str]:
        """
        Select the most optimal provider based on cost, performance, and reliability

        Uses weighted scoring algorithm considering:
        - Cost efficiency (per operation cost vs performance)
        - Reliability score
        - Current usage limits
        - Budget constraints
        """

        if not eligible_providers:
            return None

        requirements = requirements or {}

        await self.context_stream.add_event(
            {
                "type": "COST_OPTIMIZER_SELECTION_START",
                "task_type": task_type,
                "eligible_providers": eligible_providers,
                "estimated_operations": estimated_operations,
            }
        )

        # Filter providers by availability and budget
        available_providers = await self._filter_available_providers(
            eligible_providers, estimated_operations
        )

        if not available_providers:
            await self.context_stream.add_event(
                {
                    "type": "COST_OPTIMIZER_NO_PROVIDERS",
                    "reason": "budget_or_rate_limits",
                }
            )
            return None

        # Score each provider
        provider_scores = {}
        for provider_name in available_providers:
            score = await self._calculate_provider_score(
                provider_name, task_type, estimated_operations, requirements
            )
            provider_scores[provider_name] = score

        # Select best provider
        best_provider = max(provider_scores.keys(), key=lambda p: provider_scores[p])

        await self.context_stream.add_event(
            {
                "type": "COST_OPTIMIZER_SELECTION_COMPLETE",
                "selected_provider": best_provider,
                "provider_scores": provider_scores,
                "selection_criteria": {
                    "cost_weight": self.cost_efficiency_weight,
                    "performance_weight": self.performance_weight,
                    "reliability_weight": self.reliability_weight,
                },
            }
        )

        return best_provider

    async def _filter_available_providers(
        self, providers: List[str], estimated_operations: int
    ) -> List[str]:
        """Filter providers by rate limits and budget constraints"""

        available = []
        current_time = datetime.now()

        for provider_name in providers:
            profile = self.provider_profiles.get(provider_name)
            if not profile:
                continue

            # Check rate limits
            if not await self._check_rate_limits(provider_name, estimated_operations):
                continue

            # Check budget constraints
            estimated_cost = profile.calculate_cost(estimated_operations)
            budget_allocation = self.budget_allocations.get(profile.category)

            if budget_allocation:
                if budget_allocation.remaining_amount < estimated_cost:
                    continue

            available.append(provider_name)

        return available

    async def _check_rate_limits(self, provider_name: str, operations: int) -> bool:
        """Check if provider is within rate limits"""

        profile = self.provider_profiles.get(provider_name)
        if not profile:
            return False

        current_time = datetime.now()
        hour_ago = current_time - timedelta(hours=1)
        day_ago = current_time - timedelta(days=1)

        # Count recent usage
        hourly_usage = len(
            [
                record
                for record in self.usage_history
                if record.provider == provider_name and record.timestamp >= hour_ago
            ]
        )

        daily_usage = len(
            [
                record
                for record in self.usage_history
                if record.provider == provider_name and record.timestamp >= day_ago
            ]
        )

        # Check limits
        if hourly_usage + operations > profile.rate_limit_per_hour:
            return False

        if (
            profile.rate_limit_per_day
            and daily_usage + operations > profile.rate_limit_per_day
        ):
            return False

        return True

    async def _calculate_provider_score(
        self,
        provider_name: str,
        task_type: str,
        estimated_operations: int,
        requirements: Dict[str, Any],
    ) -> float:
        """Calculate weighted score for provider selection"""

        profile = self.provider_profiles.get(provider_name)
        if not profile:
            return 0.0

        # Cost efficiency score (lower cost = higher score)
        estimated_cost = profile.calculate_cost(estimated_operations)
        cost_per_operation = estimated_cost / max(estimated_operations, 1)

        # Normalize cost score (assuming $0.01 is expensive, $0.0001 is cheap)
        max_expected_cost = 0.01
        min_expected_cost = 0.0001

        cost_score = 1.0 - min(
            (cost_per_operation - min_expected_cost)
            / (max_expected_cost - min_expected_cost),
            1.0,
        )
        cost_score = max(cost_score, 0.0)

        # Performance score
        performance_score = min(
            profile.performance_multiplier / 1.5, 1.0
        )  # Normalize to 1.0

        # Reliability score
        reliability_score = profile.reliability_score

        # Priority adjustments
        priority_multiplier = 1.0
        if requirements.get("priority") == "high":
            # Prefer reliability and performance over cost for high priority
            reliability_score *= 1.2
            performance_score *= 1.1
        elif requirements.get("priority") == "low":
            # Prefer cost efficiency for low priority
            cost_score *= 1.2

        # Budget pressure adjustment
        budget_allocation = self.budget_allocations.get(profile.category)
        if budget_allocation and budget_allocation.status in [
            BudgetStatus.WARNING,
            BudgetStatus.CRITICAL,
        ]:
            cost_score *= 1.5  # Heavily favor cheaper options when budget is tight

        # Calculate weighted score
        total_score = (
            cost_score * self.cost_efficiency_weight
            + performance_score * self.performance_weight
            + reliability_score * self.reliability_weight
        ) * priority_multiplier

        return total_score

    async def track_usage(
        self,
        provider: str,
        operation_type: str,
        cost: float,
        user_id: str,
        analysis_id: Optional[str] = None,
        metadata: Dict[str, Any] = None,
    ) -> None:
        """Track provider usage for cost monitoring and optimization"""

        metadata = metadata or {}

        # Create usage record
        usage_record = UsageRecord(
            provider=provider,
            category=self._get_provider_category(provider),
            operation_type=operation_type,
            cost=cost,
            timestamp=datetime.now(),
            user_id=user_id,
            analysis_id=analysis_id,
            metadata=metadata,
        )

        # Add to history
        self.usage_history.append(usage_record)

        # Update budget allocations
        await self._update_budget_usage(usage_record)

        # Store in database
        await self._store_usage_record(usage_record)

        # Check for budget alerts
        await self._check_budget_alerts()

        # Log usage event
        await self.context_stream.add_event(
            {
                "type": "COST_OPTIMIZER_USAGE_TRACKED",
                "provider": provider,
                "cost": cost,
                "operation_type": operation_type,
                "category": usage_record.category.value,
                "user_id": user_id,
            }
        )

    def _get_provider_category(self, provider: str) -> CostCategory:
        """Get category for provider"""
        profile = self.provider_profiles.get(provider)
        return profile.category if profile else CostCategory.API_CALLS

    async def _update_budget_usage(self, usage_record: UsageRecord) -> None:
        """Update budget allocation with new usage"""

        budget_allocation = self.budget_allocations.get(usage_record.category)
        if budget_allocation:
            budget_allocation.used_amount += usage_record.cost
            budget_allocation.remaining_amount = (
                budget_allocation.allocated_amount - budget_allocation.used_amount
            )

    async def _store_usage_record(self, record: UsageRecord) -> None:
        """Store usage record in database"""

        try:
            await self.supabase.table("cost_tracking").insert(
                {
                    "provider": record.provider,
                    "category": record.category.value,
                    "operation_type": record.operation_type,
                    "cost": record.cost,
                    "timestamp": record.timestamp.isoformat(),
                    "user_id": record.user_id,
                    "analysis_id": record.analysis_id,
                    "metadata": json.dumps(record.metadata),
                }
            )
        except Exception as e:
            logger.error(f"Failed to store usage record: {e}")

    async def _check_budget_alerts(self) -> None:
        """Check for budget alerts and send notifications"""

        for category, allocation in self.budget_allocations.items():
            if allocation.status in [
                BudgetStatus.WARNING,
                BudgetStatus.CRITICAL,
                BudgetStatus.EXCEEDED,
            ]:

                await self.context_stream.add_event(
                    {
                        "type": "COST_OPTIMIZER_BUDGET_ALERT",
                        "category": category.value,
                        "status": allocation.status.value,
                        "usage_percentage": allocation.usage_percentage,
                        "used_amount": allocation.used_amount,
                        "allocated_amount": allocation.allocated_amount,
                    }
                )

                # Take automatic actions for critical/exceeded budgets
                if allocation.status == BudgetStatus.EXCEEDED:
                    await self._handle_budget_exceeded(category)
                elif allocation.status == BudgetStatus.CRITICAL:
                    await self._handle_budget_critical(category)

    async def _handle_budget_exceeded(self, category: CostCategory) -> None:
        """Handle budget exceeded situation"""

        logger.warning(f"Budget exceeded for category {category.value}")

        # Could implement automatic provider disabling or switching to cheaper alternatives
        await self.context_stream.add_event(
            {
                "type": "COST_OPTIMIZER_EMERGENCY_ACTION",
                "category": category.value,
                "action": "budget_exceeded_alert",
                "recommendation": "Consider switching to more cost-effective providers",
            }
        )

    async def _handle_budget_critical(self, category: CostCategory) -> None:
        """Handle critical budget situation"""

        logger.warning(f"Budget critical for category {category.value}")

        # Recommend cost optimization
        await self.context_stream.add_event(
            {
                "type": "COST_OPTIMIZER_BUDGET_WARNING",
                "category": category.value,
                "action": "optimize_provider_selection",
                "recommendation": "Prioritizing cost-effective providers for remaining operations",
            }
        )

    async def estimate_operation_cost(
        self,
        provider: str,
        operation_type: str,
        estimated_operations: int = 1,
        metadata: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Estimate cost for a planned operation"""

        profile = self.provider_profiles.get(provider)
        if not profile:
            return {"error": f"Unknown provider: {provider}"}

        # Calculate base cost
        base_cost = profile.calculate_cost(estimated_operations, metadata)

        # Factor in current usage patterns for more accurate prediction
        usage_adjustment = await self._get_usage_adjustment(provider)
        adjusted_cost = base_cost * usage_adjustment

        return {
            "provider": provider,
            "operation_type": operation_type,
            "estimated_operations": estimated_operations,
            "base_cost": base_cost,
            "adjusted_cost": adjusted_cost,
            "cost_per_operation": adjusted_cost / max(estimated_operations, 1),
            "budget_impact": self._calculate_budget_impact(
                profile.category, adjusted_cost
            ),
            "usage_adjustment_factor": usage_adjustment,
        }

    async def _get_usage_adjustment(self, provider: str) -> float:
        """Get usage adjustment factor based on historical performance"""

        # Analyze recent usage for this provider
        recent_records = [
            record
            for record in self.usage_history
            if record.provider == provider
            and record.timestamp >= datetime.now() - timedelta(days=7)
        ]

        if len(recent_records) < 3:
            return 1.0  # Not enough data

        # Calculate average cost variance
        profile = self.provider_profiles.get(provider)
        if not profile:
            return 1.0

        actual_costs = [record.cost for record in recent_records]
        expected_costs = [
            profile.calculate_cost(1) for _ in recent_records
        ]  # Assume 1 operation per record

        if len(expected_costs) == 0:
            return 1.0

        # Calculate adjustment factor
        avg_actual = statistics.mean(actual_costs)
        avg_expected = statistics.mean(expected_costs)

        if avg_expected == 0:
            return 1.0

        adjustment = avg_actual / avg_expected

        # Cap adjustment between 0.5 and 2.0
        return max(0.5, min(2.0, adjustment))

    def _calculate_budget_impact(
        self, category: CostCategory, cost: float
    ) -> Dict[str, Any]:
        """Calculate impact of cost on budget allocation"""

        allocation = self.budget_allocations.get(category)
        if not allocation:
            return {"impact": "unknown"}

        current_usage_pct = allocation.usage_percentage
        new_usage_pct = (
            (allocation.used_amount + cost) / allocation.allocated_amount
        ) * 100

        return {
            "current_usage_percentage": current_usage_pct,
            "projected_usage_percentage": new_usage_pct,
            "remaining_budget": allocation.remaining_amount - cost,
            "status_after": (
                BudgetStatus.CRITICAL.value
                if new_usage_pct > 90
                else (
                    BudgetStatus.WARNING.value
                    if new_usage_pct > 70
                    else BudgetStatus.HEALTHY.value
                )
            ),
            "budget_exhausted_in_operations": (
                int((allocation.remaining_amount - cost) / (cost / max(1, 1)))
                if cost > 0
                else float("inf")
            ),
        }

    async def get_cost_summary(self, time_period: str = "daily") -> Dict[str, Any]:
        """Get comprehensive cost summary"""

        now = datetime.now()

        if time_period == "daily":
            start_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif time_period == "weekly":
            start_time = now - timedelta(days=7)
        elif time_period == "monthly":
            start_time = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            start_time = now - timedelta(hours=24)

        # Filter usage records for time period
        period_records = [
            record for record in self.usage_history if record.timestamp >= start_time
        ]

        # Calculate costs by provider
        provider_costs = {}
        for record in period_records:
            if record.provider not in provider_costs:
                provider_costs[record.provider] = {"cost": 0.0, "operations": 0}
            provider_costs[record.provider]["cost"] += record.cost
            provider_costs[record.provider]["operations"] += 1

        # Calculate costs by category
        category_costs = {}
        for record in period_records:
            category = record.category.value
            if category not in category_costs:
                category_costs[category] = {"cost": 0.0, "operations": 0}
            category_costs[category]["cost"] += record.cost
            category_costs[category]["operations"] += 1

        # Calculate totals
        total_cost = sum(record.cost for record in period_records)
        total_operations = len(period_records)

        # Budget analysis
        budget_analysis = {
            category.value: {
                "allocated": allocation.allocated_amount,
                "used": allocation.used_amount,
                "remaining": allocation.remaining_amount,
                "usage_percentage": allocation.usage_percentage,
                "status": allocation.status.value,
            }
            for category, allocation in self.budget_allocations.items()
        }

        return {
            "time_period": time_period,
            "start_time": start_time.isoformat(),
            "end_time": now.isoformat(),
            "total_cost": total_cost,
            "total_operations": total_operations,
            "average_cost_per_operation": total_cost / max(total_operations, 1),
            "provider_breakdown": provider_costs,
            "category_breakdown": category_costs,
            "budget_analysis": budget_analysis,
            "cost_efficiency_score": await self._calculate_cost_efficiency_score(
                period_records
            ),
            "recommendations": await self._generate_cost_recommendations(
                period_records
            ),
        }

    async def _calculate_cost_efficiency_score(
        self, records: List[UsageRecord]
    ) -> float:
        """Calculate cost efficiency score (0-100)"""

        if not records:
            return 100.0

        # Compare actual costs vs theoretical minimum costs
        total_actual_cost = sum(record.cost for record in records)

        # Calculate theoretical minimum by always choosing cheapest provider
        theoretical_minimum = 0.0
        for record in records:
            # Find cheapest provider for this category
            category_providers = [
                (name, profile)
                for name, profile in self.provider_profiles.items()
                if profile.category == record.category
            ]

            if category_providers:
                cheapest_cost = min(
                    profile.calculate_cost(1) for _, profile in category_providers
                )
                theoretical_minimum += cheapest_cost

        if theoretical_minimum == 0:
            return 100.0

        # Calculate efficiency score
        efficiency = (theoretical_minimum / total_actual_cost) * 100
        return min(efficiency, 100.0)

    async def _generate_cost_recommendations(
        self, records: List[UsageRecord]
    ) -> List[Dict[str, Any]]:
        """Generate cost optimization recommendations"""

        recommendations = []

        # Check for expensive providers
        provider_costs = {}
        for record in records:
            if record.provider not in provider_costs:
                provider_costs[record.provider] = []
            provider_costs[record.provider].append(record.cost)

        for provider, costs in provider_costs.items():
            avg_cost = statistics.mean(costs)
            profile = self.provider_profiles.get(provider)

            if profile and avg_cost > profile.base_cost_per_operation * 2:
                recommendations.append(
                    {
                        "type": "expensive_provider",
                        "provider": provider,
                        "message": f"Provider {provider} costs are {avg_cost/profile.base_cost_per_operation:.1f}x expected",
                        "suggestion": "Consider switching to more cost-effective alternatives",
                        "potential_savings": (
                            avg_cost - profile.base_cost_per_operation
                        )
                        * len(costs),
                    }
                )

        # Check for budget pressure
        for category, allocation in self.budget_allocations.items():
            if allocation.status == BudgetStatus.WARNING:
                recommendations.append(
                    {
                        "type": "budget_warning",
                        "category": category.value,
                        "message": f"Budget for {category.value} is at {allocation.usage_percentage:.1f}%",
                        "suggestion": "Consider optimizing provider selection for this category",
                    }
                )

        # Check for underutilized volume discounts
        for provider, profile in self.provider_profiles.items():
            if profile.volume_discounts:
                provider_operations = len(
                    [r for r in records if r.provider == provider]
                )
                next_discount_threshold = (
                    min(
                        threshold
                        for threshold in profile.volume_discounts.keys()
                        if threshold > provider_operations
                    )
                    if any(
                        threshold > provider_operations
                        for threshold in profile.volume_discounts.keys()
                    )
                    else None
                )

                if next_discount_threshold:
                    operations_needed = next_discount_threshold - provider_operations
                    if (
                        operations_needed <= provider_operations * 0.2
                    ):  # Within 20% of next discount
                        discount_rate = profile.volume_discounts[
                            next_discount_threshold
                        ]
                        recommendations.append(
                            {
                                "type": "volume_discount_opportunity",
                                "provider": provider,
                                "message": f"Only {operations_needed} more operations needed for {discount_rate}% discount",
                                "suggestion": f"Consider consolidating usage to {provider} to reach volume discount",
                            }
                        )

        return recommendations

    async def optimize_provider_allocation(
        self, forecast_operations: Dict[str, int]
    ) -> Dict[str, Any]:
        """Optimize provider allocation based on forecasted operations"""

        optimization_plan = {
            "current_allocation": {},
            "recommended_allocation": {},
            "cost_savings": 0.0,
            "efficiency_improvement": 0.0,
        }

        # Current cost calculation
        current_total_cost = 0.0
        for provider, operations in forecast_operations.items():
            profile = self.provider_profiles.get(provider)
            if profile:
                cost = profile.calculate_cost(operations)
                current_total_cost += cost
                optimization_plan["current_allocation"][provider] = {
                    "operations": operations,
                    "cost": cost,
                }

        # Find optimal allocation
        # Group operations by category and find cheapest provider for each
        category_operations = {}
        for provider, operations in forecast_operations.items():
            profile = self.provider_profiles.get(provider)
            if profile:
                category = profile.category
                if category not in category_operations:
                    category_operations[category] = 0
                category_operations[category] += operations

        # Calculate optimal allocation
        optimal_total_cost = 0.0
        for category, total_ops in category_operations.items():
            # Find cheapest provider for this category
            category_providers = [
                (name, profile)
                for name, profile in self.provider_profiles.items()
                if profile.category == category
            ]

            if category_providers:
                # Sort by cost efficiency
                sorted_providers = sorted(
                    category_providers,
                    key=lambda x: x[1].calculate_cost(1) / x[1].performance_multiplier,
                )

                best_provider, best_profile = sorted_providers[0]
                cost = best_profile.calculate_cost(total_ops)
                optimal_total_cost += cost

                optimization_plan["recommended_allocation"][best_provider] = {
                    "operations": total_ops,
                    "cost": cost,
                    "category": category.value,
                }

        # Calculate savings and improvements
        optimization_plan["cost_savings"] = current_total_cost - optimal_total_cost
        optimization_plan["efficiency_improvement"] = (
            (optimization_plan["cost_savings"] / current_total_cost) * 100
            if current_total_cost > 0
            else 0.0
        )

        return optimization_plan

    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check of cost optimization system"""

        health_status = {
            "overall_health": "healthy",
            "components": {},
            "timestamp": datetime.now().isoformat(),
        }

        # Check budget health
        budget_health = "healthy"
        critical_categories = 0

        for category, allocation in self.budget_allocations.items():
            if allocation.status in [BudgetStatus.CRITICAL, BudgetStatus.EXCEEDED]:
                critical_categories += 1

        if critical_categories > 0:
            budget_health = "critical" if critical_categories > 2 else "warning"

        health_status["components"]["budget_management"] = budget_health

        # Check provider availability
        available_providers = sum(
            1
            for provider in self.provider_profiles.keys()
            if provider in self.provider_profiles
        )
        provider_health = "healthy" if available_providers >= 3 else "warning"
        health_status["components"]["provider_availability"] = provider_health

        # Check data tracking
        recent_records = len(
            [
                r
                for r in self.usage_history
                if r.timestamp >= datetime.now() - timedelta(hours=1)
            ]
        )
        data_health = (
            "healthy" if recent_records >= 0 else "warning"
        )  # Always healthy if no errors
        health_status["components"]["usage_tracking"] = data_health

        # Overall health
        if budget_health == "critical" or provider_health != "healthy":
            health_status["overall_health"] = "critical"
        elif budget_health == "warning":
            health_status["overall_health"] = "warning"

        # Additional metrics
        health_status["metrics"] = {
            "total_providers": len(self.provider_profiles),
            "available_providers": available_providers,
            "budget_categories_healthy": len(
                [
                    a
                    for a in self.budget_allocations.values()
                    if a.status == BudgetStatus.HEALTHY
                ]
            ),
            "total_usage_records": len(self.usage_history),
            "daily_cost": sum(
                record.cost
                for record in self.usage_history
                if record.timestamp
                >= datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            ),
        }

        return health_status
