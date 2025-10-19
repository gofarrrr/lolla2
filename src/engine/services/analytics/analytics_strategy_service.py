"""
Analytics Strategy Service - Prompt Analytics Calculation Engine
==============================================================

REFACTORING TARGET: Extract Grade E complexity from _calculate_analytics()
PATTERN: Strategy Pattern with Composable Calculators
GOAL: Reduce _calculate_analytics() from E (38) to B (≤10)

Architecture:
- AnalyticsCalculatorInterface: Strategy interface
- Specialized calculators for each analytics domain
- AnalyticsOrchestrator: Coordinates strategy execution
- Result composition with type safety

Benefits:
- Single Responsibility Principle per calculator
- Easily testable individual strategies
- Extensible for new analytics types
- Clear separation of calculation concerns
"""

from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Any, Protocol
from dataclasses import dataclass

# Import domain models from original module
from src.engine.core.prompt_capture import (
    PromptRecord,
    ResponseLinkage,
    PromptAnalytics,
    PromptPhase,
)


class AnalyticsCalculatorInterface(Protocol):
    """Strategy interface for analytics calculations"""

    def calculate(
        self, records: List[PromptRecord], linkages: List[ResponseLinkage]
    ) -> Dict[str, Any]:
        """Calculate specific analytics metrics"""
        ...


@dataclass
class CalculationContext:
    """Shared context for analytics calculations"""

    records: List[PromptRecord]
    linkages: List[ResponseLinkage]
    linkage_dict: Dict[str, ResponseLinkage]
    linked_records: List[PromptRecord]
    total_prompts: int

    @classmethod
    def create(
        cls, records: List[PromptRecord], linkages: List[ResponseLinkage]
    ) -> "CalculationContext":
        """Factory method to create calculation context"""
        linkage_dict = {linkage.prompt_id: linkage for linkage in linkages}
        linked_records = [
            record for record in records if record.prompt_id in linkage_dict
        ]

        return cls(
            records=records,
            linkages=linkages,
            linkage_dict=linkage_dict,
            linked_records=linked_records,
            total_prompts=len(records),
        )


class BasicMetricsCalculator:
    """
    Strategy: Calculate basic prompt metrics

    Responsibility: Total prompts, unique templates, average prompt length
    Complexity Target: Grade B (≤10)
    """

    def calculate(
        self, records: List[PromptRecord], linkages: List[ResponseLinkage]
    ) -> Dict[str, Any]:
        """
        Calculate basic metrics

        Complexity: B (≤10) - Simple aggregations only
        """
        if not records:
            return {"total_prompts": 0, "unique_templates": 0, "avg_prompt_length": 0.0}

        total_prompts = len(records)
        unique_templates = len(
            set(record.template_id for record in records if record.template_id)
        )
        avg_prompt_length = (
            sum(record.prompt_length for record in records) / total_prompts
        )

        return {
            "total_prompts": total_prompts,
            "unique_templates": unique_templates,
            "avg_prompt_length": avg_prompt_length,
        }


class ResponseLinkageCalculator:
    """
    Strategy: Calculate response linkage metrics

    Responsibility: Response times, success rates, costs, quality scores
    Complexity Target: Grade B (≤10)
    """

    def calculate(
        self, records: List[PromptRecord], linkages: List[ResponseLinkage]
    ) -> Dict[str, Any]:
        """
        Calculate response linkage metrics

        Complexity: B (≤10) - Focused on linkage processing only
        """
        context = CalculationContext.create(records, linkages)

        if not context.linked_records:
            return {
                "avg_response_time_ms": 0.0,
                "success_rate": 0.0,
                "cost_per_prompt_usd": 0.0,
                "avg_quality_score": 0.0,
            }

        # Response time calculation
        response_times = [
            context.linkage_dict[record.prompt_id].response_time_ms
            for record in context.linked_records
        ]
        avg_response_time_ms = sum(response_times) / len(response_times)

        # Success rate calculation
        successful = sum(
            1
            for record in context.linked_records
            if context.linkage_dict[record.prompt_id].success
        )
        success_rate = successful / len(context.linked_records)

        # Cost calculation
        total_cost = sum(
            context.linkage_dict[record.prompt_id].response_cost_usd
            for record in context.linked_records
        )
        cost_per_prompt_usd = total_cost / len(context.linked_records)

        # Quality score calculation
        quality_scores = [
            context.linkage_dict[record.prompt_id].response_quality_score
            for record in context.linked_records
            if context.linkage_dict[record.prompt_id].response_quality_score is not None
        ]
        avg_quality_score = (
            sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        )

        return {
            "avg_response_time_ms": avg_response_time_ms,
            "success_rate": success_rate,
            "cost_per_prompt_usd": cost_per_prompt_usd,
            "avg_quality_score": avg_quality_score,
        }


class PhaseBreakdownCalculator:
    """
    Strategy: Calculate phase-based analytics

    Responsibility: Prompts/success/cost breakdown by phase
    Complexity Target: Grade B (≤10)
    """

    def calculate(
        self, records: List[PromptRecord], linkages: List[ResponseLinkage]
    ) -> Dict[str, Any]:
        """
        Calculate phase breakdown analytics

        Complexity: B (≤10) - Phase aggregation only
        """
        context = CalculationContext.create(records, linkages)

        prompts_by_phase = defaultdict(int)
        success_by_phase = defaultdict(float)
        cost_by_phase = defaultdict(float)

        # Aggregate by phase
        for record in records:
            prompts_by_phase[record.phase] += 1

            if record.prompt_id in context.linkage_dict:
                linkage = context.linkage_dict[record.prompt_id]
                if linkage.success:
                    success_by_phase[record.phase] += 1
                cost_by_phase[record.phase] += linkage.response_cost_usd

        # Convert to rates
        for phase in prompts_by_phase:
            if prompts_by_phase[phase] > 0:
                success_by_phase[phase] = (
                    success_by_phase[phase] / prompts_by_phase[phase]
                )

        return {
            "prompts_by_phase": dict(prompts_by_phase),
            "success_by_phase": dict(success_by_phase),
            "cost_by_phase": dict(cost_by_phase),
        }


class TemplatePerformanceCalculator:
    """
    Strategy: Calculate template performance metrics

    Responsibility: Template usage, success rates, top performers
    Complexity Target: Grade B (≤10)
    """

    def calculate(
        self, records: List[PromptRecord], linkages: List[ResponseLinkage]
    ) -> Dict[str, Any]:
        """
        Calculate template performance analytics

        Complexity: B (≤10) - Template analysis only
        """
        context = CalculationContext.create(records, linkages)

        template_usage = defaultdict(int)
        template_success = defaultdict(int)

        # Aggregate template metrics
        for record in records:
            if record.template_id:
                template_usage[record.template_id] += 1
                if (
                    record.prompt_id in context.linkage_dict
                    and context.linkage_dict[record.prompt_id].success
                ):
                    template_success[record.template_id] += 1

        # Calculate success rates
        template_success_rates = {
            template_id: template_success[template_id] / template_usage[template_id]
            for template_id in template_usage
        }

        # Top templates by usage
        top_templates = sorted(
            [
                {
                    "template_id": tid,
                    "usage": usage,
                    "success_rate": template_success_rates[tid],
                }
                for tid, usage in template_usage.items()
            ],
            key=lambda x: x["usage"],
            reverse=True,
        )[:10]

        return {
            "template_success_rates": template_success_rates,
            "top_templates": top_templates,
        }


class TimeBasedAnalyticsCalculator:
    """
    Strategy: Calculate time-based analytics

    Responsibility: Prompts per hour, peak usage patterns
    Complexity Target: Grade B (≤10)
    """

    def calculate(
        self, records: List[PromptRecord], linkages: List[ResponseLinkage]
    ) -> Dict[str, Any]:
        """
        Calculate time-based analytics

        Complexity: B (≤10) - Time pattern analysis only
        """
        if not records:
            return {"prompts_per_hour": 0.0, "peak_usage_hours": []}

        # Parse timestamps
        timestamps = [datetime.fromisoformat(record.timestamp) for record in records]

        # Calculate prompts per hour
        if len(timestamps) > 1:
            time_span = (
                max(timestamps) - min(timestamps)
            ).total_seconds() / 3600  # hours
            prompts_per_hour = len(records) / max(1, time_span)
        else:
            prompts_per_hour = 0.0

        # Peak usage hours
        hours = [ts.hour for ts in timestamps]
        hour_counts = defaultdict(int)
        for hour in hours:
            hour_counts[hour] += 1

        peak_usage_hours = sorted(
            hour_counts.keys(), key=lambda h: hour_counts[h], reverse=True
        )[:3]

        return {
            "prompts_per_hour": prompts_per_hour,
            "peak_usage_hours": peak_usage_hours,
        }


class QualityDistributionCalculator:
    """
    Strategy: Calculate quality distribution metrics

    Responsibility: Quality score distribution analysis
    Complexity Target: Grade B (≤10)
    """

    def calculate(
        self, records: List[PromptRecord], linkages: List[ResponseLinkage]
    ) -> Dict[str, Any]:
        """
        Calculate quality distribution

        Complexity: B (≤10) - Quality classification only
        """
        context = CalculationContext.create(records, linkages)

        quality_distribution = {"high": 0, "medium": 0, "low": 0, "unknown": 0}

        for record in context.linked_records:
            linkage = context.linkage_dict[record.prompt_id]
            if linkage.response_quality_score is not None:
                score = linkage.response_quality_score
                if score >= 0.8:
                    quality_distribution["high"] += 1
                elif score >= 0.6:
                    quality_distribution["medium"] += 1
                else:
                    quality_distribution["low"] += 1
            else:
                quality_distribution["unknown"] += 1

        return {"quality_distribution": quality_distribution}


class AnalyticsOrchestrator:
    """
    Analytics orchestrator using strategy pattern

    Responsibility: Coordinate all analytics strategies and compose final result
    Complexity Target: Grade B (≤10)
    """

    def __init__(self):
        """Initialize with all analytics calculators"""
        self.calculators = [
            BasicMetricsCalculator(),
            ResponseLinkageCalculator(),
            PhaseBreakdownCalculator(),
            TemplatePerformanceCalculator(),
            TimeBasedAnalyticsCalculator(),
            QualityDistributionCalculator(),
        ]

    def calculate_comprehensive_analytics(
        self, records: List[PromptRecord], linkages: List[ResponseLinkage]
    ) -> PromptAnalytics:
        """
        Calculate comprehensive analytics using strategy composition

        Complexity: B (≤10) - Pure orchestration and composition
        """
        if not records:
            return self._create_empty_analytics()

        # Execute all strategies
        results = {}
        for calculator in self.calculators:
            strategy_result = calculator.calculate(records, linkages)
            results.update(strategy_result)

        # Compose final analytics object
        return PromptAnalytics(
            total_prompts=results["total_prompts"],
            unique_templates=results["unique_templates"],
            avg_prompt_length=results["avg_prompt_length"],
            avg_response_time_ms=results["avg_response_time_ms"],
            success_rate=results["success_rate"],
            cost_per_prompt_usd=results["cost_per_prompt_usd"],
            prompts_by_phase=results["prompts_by_phase"],
            success_by_phase=results["success_by_phase"],
            cost_by_phase=results["cost_by_phase"],
            top_templates=results["top_templates"],
            template_success_rates=results["template_success_rates"],
            avg_quality_score=results["avg_quality_score"],
            quality_distribution=results["quality_distribution"],
            prompts_per_hour=results["prompts_per_hour"],
            peak_usage_hours=results["peak_usage_hours"],
        )

    def _create_empty_analytics(self) -> PromptAnalytics:
        """
        Create empty analytics for no-data case

        Complexity: A (1) - Simple object creation
        """
        return PromptAnalytics(
            total_prompts=0,
            unique_templates=0,
            avg_prompt_length=0.0,
            avg_response_time_ms=0.0,
            success_rate=0.0,
            cost_per_prompt_usd=0.0,
            prompts_by_phase={phase: 0 for phase in PromptPhase},
            success_by_phase={phase: 0.0 for phase in PromptPhase},
            cost_by_phase={phase: 0.0 for phase in PromptPhase},
            top_templates=[],
            template_success_rates={},
            avg_quality_score=0.0,
            quality_distribution={},
            prompts_per_hour=0.0,
            peak_usage_hours=[],
        )


# Singleton for dependency injection
_analytics_orchestrator = None


def get_analytics_orchestrator() -> AnalyticsOrchestrator:
    """Factory function for analytics orchestrator"""
    global _analytics_orchestrator
    if _analytics_orchestrator is None:
        _analytics_orchestrator = AnalyticsOrchestrator()
    return _analytics_orchestrator
