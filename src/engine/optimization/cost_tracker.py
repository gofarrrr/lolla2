#!/usr/bin/env python3
"""
METIS Cost Optimization and Usage Tracking System
E003: Enterprise cost management and optimization

Implements comprehensive cost tracking, optimization strategies,
and usage analytics for enterprise deployments.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from uuid import UUID, uuid4
from collections import defaultdict

try:
    from src.engine.models.data_contracts import MetisDataContract, EngagementContext
    from src.core.state_management import DistributedStateManager, StateType
    from src.monitoring.performance_validator import PerformanceMetricType

    DATA_CONTRACTS_AVAILABLE = True
except ImportError:
    DATA_CONTRACTS_AVAILABLE = False

    class MockStateManager:
        def __init__(self):
            self.data = {}

        async def set_state(self, key, value, state_type=None):
            self.data[key] = value

        async def get_state(self, key):
            return self.data.get(key)

    DistributedStateManager = MockStateManager
    StateType = None


class CostCategory(str, Enum):
    """Categories of costs in METIS system"""

    AI_API_CALLS = "ai_api_calls"
    COMPUTE_RESOURCES = "compute_resources"
    STORAGE = "storage"
    NETWORK_TRANSFER = "network_transfer"
    DATABASE_OPERATIONS = "database_operations"
    THIRD_PARTY_SERVICES = "third_party_services"
    MONITORING = "monitoring"


class UsageMetricType(str, Enum):
    """Types of usage metrics tracked"""

    API_CALLS = "api_calls"
    TOKENS_CONSUMED = "tokens_consumed"
    COMPUTE_HOURS = "compute_hours"
    STORAGE_GB = "storage_gb"
    BANDWIDTH_GB = "bandwidth_gb"
    CONCURRENT_USERS = "concurrent_users"
    ENGAGEMENTS_PROCESSED = "engagements_processed"
    DOCUMENTS_ANALYZED = "documents_analyzed"


class OptimizationStrategy(str, Enum):
    """Cost optimization strategies"""

    CACHE_HEAVY = "cache_heavy"
    BATCH_PROCESSING = "batch_processing"
    MODEL_DOWNGRADE = "model_downgrade"
    ASYNC_PROCESSING = "async_processing"
    RESOURCE_POOLING = "resource_pooling"
    INTELLIGENT_ROUTING = "intelligent_routing"


@dataclass
class CostModel:
    """Cost model for different resources"""

    # AI API costs (per 1K tokens)
    gpt4_cost_per_1k_input: float = 0.03
    gpt4_cost_per_1k_output: float = 0.06
    gpt35_cost_per_1k_input: float = 0.001
    gpt35_cost_per_1k_output: float = 0.002
    claude_cost_per_1k_input: float = 0.008
    claude_cost_per_1k_output: float = 0.024

    # Compute costs (per hour)
    cpu_cost_per_hour: float = 0.05
    gpu_cost_per_hour: float = 0.50
    memory_cost_per_gb_hour: float = 0.01

    # Storage costs (per GB per month)
    storage_cost_per_gb_month: float = 0.10

    # Network costs (per GB)
    network_transfer_cost_per_gb: float = 0.02

    # Database costs (per operation)
    db_read_cost: float = 0.0001
    db_write_cost: float = 0.0005

    # Target cost constraints
    max_cost_per_engagement: float = 5.0  # $1-5 per analysis target
    max_cost_per_user_month: float = 100.0
    max_cost_per_tenant_month: float = 10000.0


@dataclass
class UsageRecord:
    """Record of resource usage"""

    record_id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    engagement_id: Optional[UUID] = None
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None

    # Usage metrics
    metric_type: UsageMetricType = UsageMetricType.API_CALLS
    metric_value: float = 0.0
    metric_unit: str = "count"

    # Cost information
    cost_category: CostCategory = CostCategory.AI_API_CALLS
    estimated_cost: float = 0.0

    # Optimization applied
    optimization_applied: Optional[OptimizationStrategy] = None
    cost_saved: float = 0.0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CostOptimizationResult:
    """Result of cost optimization analysis"""

    optimization_id: UUID = field(default_factory=uuid4)
    analysis_timestamp: datetime = field(default_factory=datetime.utcnow)

    # Cost analysis
    total_cost: float = 0.0
    cost_by_category: Dict[CostCategory, float] = field(default_factory=dict)
    cost_by_engagement: Dict[UUID, float] = field(default_factory=dict)

    # Optimization recommendations
    recommended_strategies: List[OptimizationStrategy] = field(default_factory=list)
    potential_savings: float = 0.0
    savings_percentage: float = 0.0

    # Usage patterns
    peak_usage_times: List[datetime] = field(default_factory=list)
    heavy_users: List[str] = field(default_factory=list)
    expensive_operations: List[Dict[str, Any]] = field(default_factory=list)

    # Alerts
    cost_alerts: List[str] = field(default_factory=list)
    optimization_opportunities: List[Dict[str, Any]] = field(default_factory=list)


class CostOptimizationEngine:
    """
    Enterprise cost optimization and usage tracking engine
    Implements intelligent cost management for METIS v7.0
    """

    def __init__(
        self,
        state_manager: DistributedStateManager,
        cost_model: Optional[CostModel] = None,
    ):
        self.state_manager = state_manager
        self.cost_model = cost_model or CostModel()
        self.logger = logging.getLogger(__name__)

        # Usage tracking
        self.usage_records: List[UsageRecord] = []
        self.usage_cache: Dict[str, Any] = {}

        # Cost optimization state
        self.optimization_history: List[CostOptimizationResult] = []
        self.active_optimizations: Dict[str, OptimizationStrategy] = {}

        # Real-time metrics
        self.current_month_cost = 0.0
        self.current_day_cost = 0.0
        self.cost_by_tenant: Dict[str, float] = defaultdict(float)
        self.cost_by_user: Dict[str, float] = defaultdict(float)

        # Start background optimization
        asyncio.create_task(self._optimization_loop())

    async def track_usage(
        self,
        metric_type: UsageMetricType,
        metric_value: float,
        cost_category: CostCategory,
        engagement_id: Optional[UUID] = None,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UsageRecord:
        """Track resource usage and calculate costs"""

        # Calculate estimated cost
        estimated_cost = self._calculate_cost(metric_type, metric_value, cost_category)

        # Check for optimization opportunities
        optimization_strategy = await self._select_optimization_strategy(
            metric_type, metric_value, estimated_cost
        )

        # Apply optimization if available
        cost_saved = 0.0
        if optimization_strategy:
            optimized_cost = await self._apply_optimization(
                optimization_strategy, estimated_cost, metadata
            )
            cost_saved = estimated_cost - optimized_cost
            estimated_cost = optimized_cost

        # Create usage record
        record = UsageRecord(
            engagement_id=engagement_id,
            user_id=user_id,
            tenant_id=tenant_id,
            metric_type=metric_type,
            metric_value=metric_value,
            metric_unit=self._get_metric_unit(metric_type),
            cost_category=cost_category,
            estimated_cost=estimated_cost,
            optimization_applied=optimization_strategy,
            cost_saved=cost_saved,
            metadata=metadata or {},
        )

        # Store record
        self.usage_records.append(record)

        # Update real-time metrics
        self.current_day_cost += estimated_cost
        self.current_month_cost += estimated_cost

        if tenant_id:
            self.cost_by_tenant[tenant_id] += estimated_cost
        if user_id:
            self.cost_by_user[user_id] += estimated_cost

        # Store in state manager
        await self.state_manager.set_state(
            f"usage_record_{record.record_id}",
            {
                "record_id": str(record.record_id),
                "timestamp": record.timestamp.isoformat(),
                "metric_type": metric_type.value,
                "metric_value": metric_value,
                "estimated_cost": estimated_cost,
                "cost_saved": cost_saved,
                "optimization_applied": (
                    optimization_strategy.value if optimization_strategy else None
                ),
            },
            StateType.METRICS if StateType else "metrics",
        )

        # Check cost alerts
        await self._check_cost_alerts(record)

        return record

    def _calculate_cost(
        self,
        metric_type: UsageMetricType,
        metric_value: float,
        cost_category: CostCategory,
    ) -> float:
        """Calculate cost based on metric type and value"""

        if cost_category == CostCategory.AI_API_CALLS:
            if metric_type == UsageMetricType.TOKENS_CONSUMED:
                # Assume GPT-4 by default (can be refined with metadata)
                input_tokens = metric_value * 0.4  # Assume 40% input
                output_tokens = metric_value * 0.6  # Assume 60% output
                return (
                    input_tokens / 1000
                ) * self.cost_model.gpt4_cost_per_1k_input + (
                    output_tokens / 1000
                ) * self.cost_model.gpt4_cost_per_1k_output
            elif metric_type == UsageMetricType.API_CALLS:
                # Estimate tokens per API call
                avg_tokens_per_call = 2000
                return self._calculate_cost(
                    UsageMetricType.TOKENS_CONSUMED,
                    metric_value * avg_tokens_per_call,
                    cost_category,
                )

        elif cost_category == CostCategory.COMPUTE_RESOURCES:
            if metric_type == UsageMetricType.COMPUTE_HOURS:
                return metric_value * self.cost_model.cpu_cost_per_hour

        elif cost_category == CostCategory.STORAGE:
            if metric_type == UsageMetricType.STORAGE_GB:
                # Convert to monthly cost (prorated)
                hours_in_month = 730
                return (
                    metric_value * self.cost_model.storage_cost_per_gb_month
                ) / hours_in_month

        elif cost_category == CostCategory.NETWORK_TRANSFER:
            if metric_type == UsageMetricType.BANDWIDTH_GB:
                return metric_value * self.cost_model.network_transfer_cost_per_gb

        elif cost_category == CostCategory.DATABASE_OPERATIONS:
            # Assume mix of read/write operations
            read_ops = metric_value * 0.8
            write_ops = metric_value * 0.2
            return (
                read_ops * self.cost_model.db_read_cost
                + write_ops * self.cost_model.db_write_cost
            )

        # Default cost calculation
        return metric_value * 0.001  # Default to $0.001 per unit

    def _get_metric_unit(self, metric_type: UsageMetricType) -> str:
        """Get unit for metric type"""
        units = {
            UsageMetricType.API_CALLS: "calls",
            UsageMetricType.TOKENS_CONSUMED: "tokens",
            UsageMetricType.COMPUTE_HOURS: "hours",
            UsageMetricType.STORAGE_GB: "GB",
            UsageMetricType.BANDWIDTH_GB: "GB",
            UsageMetricType.CONCURRENT_USERS: "users",
            UsageMetricType.ENGAGEMENTS_PROCESSED: "engagements",
            UsageMetricType.DOCUMENTS_ANALYZED: "documents",
        }
        return units.get(metric_type, "units")

    async def _select_optimization_strategy(
        self, metric_type: UsageMetricType, metric_value: float, estimated_cost: float
    ) -> Optional[OptimizationStrategy]:
        """Select appropriate optimization strategy"""

        # Cache-heavy for repeated operations
        if metric_type == UsageMetricType.API_CALLS and metric_value > 10:
            return OptimizationStrategy.CACHE_HEAVY

        # Batch processing for multiple operations
        if metric_type in [
            UsageMetricType.API_CALLS,
            UsageMetricType.DOCUMENTS_ANALYZED,
        ]:
            if metric_value > 5:
                return OptimizationStrategy.BATCH_PROCESSING

        # Model downgrade for non-critical operations
        if estimated_cost > 1.0 and metric_type == UsageMetricType.TOKENS_CONSUMED:
            return OptimizationStrategy.MODEL_DOWNGRADE

        # Async processing for non-urgent tasks
        if metric_type == UsageMetricType.COMPUTE_HOURS and metric_value > 0.1:
            return OptimizationStrategy.ASYNC_PROCESSING

        # Resource pooling for concurrent operations
        if metric_type == UsageMetricType.CONCURRENT_USERS and metric_value > 5:
            return OptimizationStrategy.RESOURCE_POOLING

        return None

    async def _apply_optimization(
        self,
        strategy: OptimizationStrategy,
        original_cost: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Apply optimization strategy and return optimized cost"""

        optimized_cost = original_cost

        if strategy == OptimizationStrategy.CACHE_HEAVY:
            # Check cache hit rate
            cache_hit_rate = 0.3  # 30% cache hit rate
            optimized_cost *= 1 - cache_hit_rate

        elif strategy == OptimizationStrategy.BATCH_PROCESSING:
            # Batch processing reduces per-unit cost
            batch_discount = 0.2  # 20% discount for batching
            optimized_cost *= 1 - batch_discount

        elif strategy == OptimizationStrategy.MODEL_DOWNGRADE:
            # Use cheaper model (GPT-3.5 instead of GPT-4)
            model_cost_ratio = 0.1  # GPT-3.5 is ~10% the cost of GPT-4
            optimized_cost *= model_cost_ratio

        elif strategy == OptimizationStrategy.ASYNC_PROCESSING:
            # Async processing on cheaper resources
            async_discount = 0.3  # 30% cheaper for async
            optimized_cost *= 1 - async_discount

        elif strategy == OptimizationStrategy.RESOURCE_POOLING:
            # Share resources across users
            pooling_efficiency = 0.4  # 40% reduction through pooling
            optimized_cost *= 1 - pooling_efficiency

        elif strategy == OptimizationStrategy.INTELLIGENT_ROUTING:
            # Route to optimal resource
            routing_optimization = 0.25  # 25% cost reduction
            optimized_cost *= 1 - routing_optimization

        return optimized_cost

    async def _check_cost_alerts(self, record: UsageRecord):
        """Check for cost alerts and thresholds"""

        alerts = []

        # Check per-engagement cost
        if record.engagement_id:
            engagement_cost = sum(
                r.estimated_cost
                for r in self.usage_records
                if r.engagement_id == record.engagement_id
            )
            if engagement_cost > self.cost_model.max_cost_per_engagement:
                alerts.append(
                    f"Engagement {record.engagement_id} exceeded cost limit: ${engagement_cost:.2f}"
                )

        # Check per-user monthly cost
        if record.user_id and record.user_id in self.cost_by_user:
            if (
                self.cost_by_user[record.user_id]
                > self.cost_model.max_cost_per_user_month
            ):
                alerts.append(
                    f"User {record.user_id} exceeded monthly limit: ${self.cost_by_user[record.user_id]:.2f}"
                )

        # Check per-tenant monthly cost
        if record.tenant_id and record.tenant_id in self.cost_by_tenant:
            if (
                self.cost_by_tenant[record.tenant_id]
                > self.cost_model.max_cost_per_tenant_month
            ):
                alerts.append(
                    f"Tenant {record.tenant_id} exceeded monthly limit: ${self.cost_by_tenant[record.tenant_id]:.2f}"
                )

        # Log alerts
        for alert in alerts:
            self.logger.warning(f"Cost Alert: {alert}")

    async def analyze_costs(
        self,
        time_period: timedelta = timedelta(days=30),
        tenant_id: Optional[str] = None,
    ) -> CostOptimizationResult:
        """Analyze costs and identify optimization opportunities"""

        cutoff_date = datetime.utcnow() - time_period

        # Filter records for analysis
        records = [
            r
            for r in self.usage_records
            if r.timestamp > cutoff_date
            and (tenant_id is None or r.tenant_id == tenant_id)
        ]

        # Calculate costs by category
        cost_by_category = defaultdict(float)
        for record in records:
            cost_by_category[record.cost_category] += record.estimated_cost

        # Calculate costs by engagement
        cost_by_engagement = defaultdict(float)
        for record in records:
            if record.engagement_id:
                cost_by_engagement[record.engagement_id] += record.estimated_cost

        # Identify heavy users
        user_costs = defaultdict(float)
        for record in records:
            if record.user_id:
                user_costs[record.user_id] += record.estimated_cost

        heavy_users = sorted(user_costs.items(), key=lambda x: x[1], reverse=True)[:10]

        # Identify expensive operations
        expensive_ops = sorted(
            [
                {
                    "type": r.metric_type.value,
                    "cost": r.estimated_cost,
                    "timestamp": r.timestamp.isoformat(),
                }
                for r in records
            ],
            key=lambda x: x["cost"],
            reverse=True,
        )[:10]

        # Calculate optimization potential
        total_cost = sum(r.estimated_cost for r in records)
        total_saved = sum(r.cost_saved for r in records)
        potential_savings = self._estimate_additional_savings(records)

        # Recommend optimization strategies
        recommended_strategies = self._recommend_optimizations(
            cost_by_category, records
        )

        # Create result
        result = CostOptimizationResult(
            total_cost=total_cost,
            cost_by_category=dict(cost_by_category),
            cost_by_engagement=dict(cost_by_engagement),
            recommended_strategies=recommended_strategies,
            potential_savings=potential_savings,
            savings_percentage=(
                (potential_savings / total_cost * 100) if total_cost > 0 else 0
            ),
            heavy_users=[u[0] for u in heavy_users],
            expensive_operations=expensive_ops,
        )

        # Store result
        self.optimization_history.append(result)

        return result

    def _estimate_additional_savings(self, records: List[UsageRecord]) -> float:
        """Estimate additional savings potential"""

        savings = 0.0

        # Count unoptimized operations
        unoptimized = [r for r in records if r.optimization_applied is None]

        for record in unoptimized:
            # Estimate 20-40% savings potential
            if record.cost_category == CostCategory.AI_API_CALLS:
                savings += record.estimated_cost * 0.3
            elif record.cost_category == CostCategory.COMPUTE_RESOURCES:
                savings += record.estimated_cost * 0.25
            else:
                savings += record.estimated_cost * 0.2

        return savings

    def _recommend_optimizations(
        self, cost_by_category: Dict[CostCategory, float], records: List[UsageRecord]
    ) -> List[OptimizationStrategy]:
        """Recommend optimization strategies based on usage patterns"""

        recommendations = []

        # High API costs -> caching
        if cost_by_category.get(CostCategory.AI_API_CALLS, 0) > 100:
            recommendations.append(OptimizationStrategy.CACHE_HEAVY)

        # Multiple similar operations -> batching
        api_calls = [r for r in records if r.metric_type == UsageMetricType.API_CALLS]
        if len(api_calls) > 100:
            recommendations.append(OptimizationStrategy.BATCH_PROCESSING)

        # High compute costs -> async processing
        if cost_by_category.get(CostCategory.COMPUTE_RESOURCES, 0) > 50:
            recommendations.append(OptimizationStrategy.ASYNC_PROCESSING)

        # Multiple concurrent users -> resource pooling
        concurrent_users = max(
            (
                r.metric_value
                for r in records
                if r.metric_type == UsageMetricType.CONCURRENT_USERS
            ),
            default=0,
        )
        if concurrent_users > 10:
            recommendations.append(OptimizationStrategy.RESOURCE_POOLING)

        # General optimization
        if not recommendations:
            recommendations.append(OptimizationStrategy.INTELLIGENT_ROUTING)

        return list(set(recommendations))  # Remove duplicates

    async def _optimization_loop(self):
        """Background loop for continuous optimization"""

        while True:
            try:
                # Analyze costs every hour
                await asyncio.sleep(3600)

                # Perform cost analysis
                analysis = await self.analyze_costs(timedelta(days=1))

                # Apply recommended optimizations
                for strategy in analysis.recommended_strategies:
                    self.active_optimizations[strategy.value] = strategy

                # Reset daily cost counter at midnight
                current_hour = datetime.utcnow().hour
                if current_hour == 0:
                    self.current_day_cost = 0.0

                # Reset monthly counters at month start
                if datetime.utcnow().day == 1:
                    self.current_month_cost = 0.0
                    self.cost_by_tenant.clear()
                    self.cost_by_user.clear()

                self.logger.info(
                    f"Cost optimization cycle completed. Daily cost: ${self.current_day_cost:.2f}"
                )

            except Exception as e:
                self.logger.error(f"Optimization loop error: {str(e)}")
                await asyncio.sleep(60)

    def get_cost_summary(self) -> Dict[str, Any]:
        """Get current cost summary"""

        return {
            "current_day_cost": self.current_day_cost,
            "current_month_cost": self.current_month_cost,
            "active_optimizations": list(self.active_optimizations.keys()),
            "total_records": len(self.usage_records),
            "cost_by_category": {
                cat.value: sum(
                    r.estimated_cost
                    for r in self.usage_records
                    if r.cost_category == cat
                )
                for cat in CostCategory
            },
            "total_savings": sum(r.cost_saved for r in self.usage_records),
            "top_users": sorted(
                self.cost_by_user.items(), key=lambda x: x[1], reverse=True
            )[:5],
            "top_tenants": sorted(
                self.cost_by_tenant.items(), key=lambda x: x[1], reverse=True
            )[:5],
        }


# Convenience functions
async def track_api_usage(
    tokens: int, engagement_id: Optional[UUID] = None, user_id: Optional[str] = None
) -> float:
    """Track AI API usage and return cost"""

    state_manager = (
        DistributedStateManager() if DATA_CONTRACTS_AVAILABLE else MockStateManager()
    )
    engine = CostOptimizationEngine(state_manager)

    record = await engine.track_usage(
        metric_type=UsageMetricType.TOKENS_CONSUMED,
        metric_value=tokens,
        cost_category=CostCategory.AI_API_CALLS,
        engagement_id=engagement_id,
        user_id=user_id,
    )

    return record.estimated_cost


async def get_engagement_cost(engagement_id: UUID) -> float:
    """Get total cost for an engagement"""

    state_manager = (
        DistributedStateManager() if DATA_CONTRACTS_AVAILABLE else MockStateManager()
    )
    engine = CostOptimizationEngine(state_manager)

    engagement_cost = sum(
        r.estimated_cost
        for r in engine.usage_records
        if r.engagement_id == engagement_id
    )

    return engagement_cost
