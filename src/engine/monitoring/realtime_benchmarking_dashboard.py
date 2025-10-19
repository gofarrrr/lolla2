#!/usr/bin/env python3
"""
Real-time Benchmarking Dashboard
Tracks and monitors optimal consultant system performance in real-time
"""

import asyncio
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque, defaultdict
import statistics


@dataclass
class BenchmarkMetric:
    """Single benchmark measurement"""

    metric_name: str
    value: float
    unit: str
    timestamp: datetime
    context: Dict[str, Any]


@dataclass
class PerformanceSnapshot:
    """System performance snapshot at a point in time"""

    timestamp: datetime
    query_processing_time: float
    consultant_selection_accuracy: float
    routing_pattern_effectiveness: float
    cache_hit_rate: float
    system_throughput: float
    error_rate: float
    active_consultants: int
    total_queries_processed: int


@dataclass
class BenchmarkSummary:
    """Performance summary over time period"""

    period_start: datetime
    period_end: datetime
    total_queries: int
    avg_processing_time: float
    p95_processing_time: float
    p99_processing_time: float
    consultant_accuracy_score: float
    routing_effectiveness: float
    cache_performance: float
    system_reliability: float
    top_performing_consultants: List[str]
    optimization_recommendations: List[str]


class RealtimeBenchmarkingDashboard:
    """
    Real-time performance monitoring and benchmarking dashboard
    Provides continuous insights into optimal consultant system performance
    """

    def __init__(self, window_size_minutes: int = 60):
        self.window_size = timedelta(minutes=window_size_minutes)
        self.metrics_buffer = deque(maxlen=10000)  # Keep last 10k metrics
        self.performance_history = deque(maxlen=1440)  # 24 hours of minute snapshots

        # Real-time counters
        self.query_count = 0
        self.error_count = 0
        self.start_time = datetime.now()

        # Performance tracking
        self.processing_times = deque(maxlen=1000)
        self.consultant_selections = deque(maxlen=1000)
        self.routing_patterns = deque(maxlen=1000)
        self.cache_hits = deque(maxlen=1000)

        # Consultant performance tracking
        self.consultant_performance = defaultdict(
            lambda: {
                "selections": 0,
                "success_rate": 0.0,
                "avg_confidence": 0.0,
                "processing_time": deque(maxlen=100),
            }
        )

        # Pattern effectiveness tracking
        self.pattern_effectiveness = defaultdict(
            lambda: {"uses": 0, "success_rate": 0.0, "avg_processing_time": 0.0}
        )

        print("🚀 Real-time Benchmarking Dashboard initialized")

    def record_query_processing(
        self,
        query: str,
        processing_time: float,
        selected_consultants: List[str],
        routing_pattern: str,
        success: bool = True,
        cache_hit: bool = False,
    ):
        """Record a query processing event"""

        timestamp = datetime.now()

        # Update counters
        self.query_count += 1
        if not success:
            self.error_count += 1

        # Record processing time
        self.processing_times.append(processing_time)

        # Record consultant selections
        for consultant_id in selected_consultants:
            self.consultant_performance[consultant_id]["selections"] += 1
            self.consultant_performance[consultant_id]["processing_time"].append(
                processing_time
            )

        # Record routing pattern usage
        self.pattern_effectiveness[routing_pattern]["uses"] += 1
        if success:
            self.pattern_effectiveness[routing_pattern]["success_rate"] += 1

        # Record cache performance
        self.cache_hits.append(cache_hit)

        # Create benchmark metrics
        metrics = [
            BenchmarkMetric(
                "processing_time",
                processing_time,
                "seconds",
                timestamp,
                {"query_length": len(query), "consultants": len(selected_consultants)},
            ),
            BenchmarkMetric(
                "consultant_count",
                len(selected_consultants),
                "count",
                timestamp,
                {"pattern": routing_pattern},
            ),
            BenchmarkMetric(
                "cache_hit",
                1.0 if cache_hit else 0.0,
                "boolean",
                timestamp,
                {"pattern": routing_pattern},
            ),
            BenchmarkMetric(
                "success",
                1.0 if success else 0.0,
                "boolean",
                timestamp,
                {"error_count": self.error_count},
            ),
        ]

        # Add to metrics buffer
        self.metrics_buffer.extend(metrics)

        print(
            f"📊 Query processed in {processing_time:.3f}s | Consultants: {len(selected_consultants)} | Pattern: {routing_pattern}"
        )

    def get_current_snapshot(self) -> PerformanceSnapshot:
        """Get current system performance snapshot"""

        now = datetime.now()

        # Calculate metrics from recent data
        recent_times = list(self.processing_times)
        recent_cache_hits = list(self.cache_hits)

        avg_processing_time = statistics.mean(recent_times) if recent_times else 0.0
        cache_hit_rate = (
            statistics.mean(recent_cache_hits) if recent_cache_hits else 0.0
        )

        # Calculate system throughput (queries per minute)
        runtime_minutes = max((now - self.start_time).total_seconds() / 60, 1)
        throughput = self.query_count / runtime_minutes

        # Calculate error rate
        error_rate = (self.error_count / max(self.query_count, 1)) * 100

        # Count active consultants
        active_consultants = len(
            [
                c
                for c, data in self.consultant_performance.items()
                if data["selections"] > 0
            ]
        )

        return PerformanceSnapshot(
            timestamp=now,
            query_processing_time=avg_processing_time,
            consultant_selection_accuracy=self._calculate_selection_accuracy(),
            routing_pattern_effectiveness=self._calculate_routing_effectiveness(),
            cache_hit_rate=cache_hit_rate * 100,
            system_throughput=throughput,
            error_rate=error_rate,
            active_consultants=active_consultants,
            total_queries_processed=self.query_count,
        )

    def get_benchmark_summary(self, minutes: int = 60) -> BenchmarkSummary:
        """Get comprehensive benchmark summary for time period"""

        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=minutes)

        # Filter metrics to time window
        window_metrics = [
            m for m in self.metrics_buffer if start_time <= m.timestamp <= end_time
        ]

        if not window_metrics:
            return self._empty_summary(start_time, end_time)

        # Extract processing times from metrics
        processing_times = [
            m.value for m in window_metrics if m.metric_name == "processing_time"
        ]

        # Calculate percentiles
        if processing_times:
            processing_times.sort()
            p95_time = processing_times[int(len(processing_times) * 0.95)]
            p99_time = processing_times[int(len(processing_times) * 0.99)]
            avg_time = statistics.mean(processing_times)
        else:
            p95_time = p99_time = avg_time = 0.0

        # Get top performing consultants
        top_consultants = sorted(
            self.consultant_performance.items(),
            key=lambda x: (
                x[1]["selections"],
                (
                    -statistics.mean(x[1]["processing_time"])
                    if x[1]["processing_time"]
                    else 0
                ),
            ),
            reverse=True,
        )[:5]

        top_consultant_names = [name for name, _ in top_consultants]

        # Generate optimization recommendations
        recommendations = self._generate_optimization_recommendations()

        return BenchmarkSummary(
            period_start=start_time,
            period_end=end_time,
            total_queries=len(
                [m for m in window_metrics if m.metric_name == "processing_time"]
            ),
            avg_processing_time=avg_time,
            p95_processing_time=p95_time,
            p99_processing_time=p99_time,
            consultant_accuracy_score=self._calculate_selection_accuracy(),
            routing_effectiveness=self._calculate_routing_effectiveness(),
            cache_performance=statistics.mean(
                [m.value for m in window_metrics if m.metric_name == "cache_hit"]
            )
            * 100,
            system_reliability=self._calculate_system_reliability(),
            top_performing_consultants=top_consultant_names,
            optimization_recommendations=recommendations,
        )

    def display_realtime_dashboard(self):
        """Display real-time dashboard to console"""

        snapshot = self.get_current_snapshot()
        summary = self.get_benchmark_summary(60)

        print("\n" + "=" * 80)
        print("🚀 METIS OPTIMAL CONSULTANT SYSTEM - REAL-TIME DASHBOARD")
        print("=" * 80)

        print(f"📅 {snapshot.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(
            f"⚡ System Uptime: {(datetime.now() - self.start_time).total_seconds() / 3600:.1f} hours"
        )

        print("\n📊 CURRENT PERFORMANCE:")
        print(f"   Query Processing Time: {snapshot.query_processing_time:.3f}s")
        print(f"   System Throughput: {snapshot.system_throughput:.1f} queries/min")
        print(f"   Cache Hit Rate: {snapshot.cache_hit_rate:.1f}%")
        print(f"   Error Rate: {snapshot.error_rate:.1f}%")
        print(f"   Active Consultants: {snapshot.active_consultants}")
        print(f"   Total Queries: {snapshot.total_queries_processed}")

        print("\n📈 60-MINUTE SUMMARY:")
        print(f"   Average Processing Time: {summary.avg_processing_time:.3f}s")
        print(f"   95th Percentile Time: {summary.p95_processing_time:.3f}s")
        print(f"   99th Percentile Time: {summary.p99_processing_time:.3f}s")
        print(f"   Consultant Accuracy: {summary.consultant_accuracy_score:.1f}%")
        print(f"   Routing Effectiveness: {summary.routing_effectiveness:.1f}%")
        print(f"   System Reliability: {summary.system_reliability:.1f}%")

        print("\n🏆 TOP PERFORMING CONSULTANTS:")
        for i, consultant in enumerate(summary.top_performing_consultants[:3], 1):
            selections = self.consultant_performance[consultant]["selections"]
            avg_time = (
                statistics.mean(
                    self.consultant_performance[consultant]["processing_time"]
                )
                if self.consultant_performance[consultant]["processing_time"]
                else 0
            )
            print(f"   {i}. {consultant}: {selections} selections, {avg_time:.3f}s avg")

        print("\n💡 OPTIMIZATION RECOMMENDATIONS:")
        for i, rec in enumerate(summary.optimization_recommendations[:3], 1):
            print(f"   {i}. {rec}")

        print("\n" + "=" * 80)

    def _calculate_selection_accuracy(self) -> float:
        """Calculate consultant selection accuracy based on success rates"""
        if not self.consultant_performance:
            return 0.0

        total_selections = sum(
            data["selections"] for data in self.consultant_performance.values()
        )
        if total_selections == 0:
            return 0.0

        # Assume successful queries indicate good selection
        success_rate = max(0, 100 - (self.error_count / max(self.query_count, 1)) * 100)
        return success_rate

    def _calculate_routing_effectiveness(self) -> float:
        """Calculate routing pattern effectiveness"""
        if not self.pattern_effectiveness:
            return 0.0

        total_uses = sum(data["uses"] for data in self.pattern_effectiveness.values())
        if total_uses == 0:
            return 0.0

        # Calculate weighted effectiveness
        weighted_effectiveness = 0.0
        for pattern, data in self.pattern_effectiveness.items():
            if data["uses"] > 0:
                pattern_success_rate = (data["success_rate"] / data["uses"]) * 100
                weight = data["uses"] / total_uses
                weighted_effectiveness += pattern_success_rate * weight

        return weighted_effectiveness

    def _calculate_system_reliability(self) -> float:
        """Calculate overall system reliability"""
        if self.query_count == 0:
            return 100.0

        success_rate = ((self.query_count - self.error_count) / self.query_count) * 100
        return max(0.0, success_rate)

    def _generate_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on metrics"""
        recommendations = []

        # Analyze processing times
        if self.processing_times:
            avg_time = statistics.mean(self.processing_times)
            if avg_time > 2.0:
                recommendations.append(
                    f"High processing time ({avg_time:.2f}s) - consider caching optimization"
                )
            if avg_time > 5.0:
                recommendations.append(
                    "Critical: Processing time exceeds 5s - investigate bottlenecks"
                )

        # Analyze cache performance
        if self.cache_hits:
            cache_rate = statistics.mean(self.cache_hits) * 100
            if cache_rate < 50:
                recommendations.append(
                    f"Low cache hit rate ({cache_rate:.1f}%) - review caching strategy"
                )

        # Analyze error rate
        if self.query_count > 0:
            error_rate = (self.error_count / self.query_count) * 100
            if error_rate > 5:
                recommendations.append(
                    f"High error rate ({error_rate:.1f}%) - review system stability"
                )

        # Analyze consultant distribution
        if self.consultant_performance:
            selections = [
                data["selections"] for data in self.consultant_performance.values()
            ]
            if selections and max(selections) > 3 * statistics.mean(selections):
                recommendations.append(
                    "Uneven consultant distribution - review routing patterns"
                )

        # Default recommendations
        if not recommendations:
            recommendations.extend(
                [
                    "System performing well - monitor for consistent performance",
                    "Consider A/B testing new routing patterns",
                    "Evaluate consultant specialization effectiveness",
                ]
            )

        return recommendations

    def _empty_summary(
        self, start_time: datetime, end_time: datetime
    ) -> BenchmarkSummary:
        """Return empty summary when no data available"""
        return BenchmarkSummary(
            period_start=start_time,
            period_end=end_time,
            total_queries=0,
            avg_processing_time=0.0,
            p95_processing_time=0.0,
            p99_processing_time=0.0,
            consultant_accuracy_score=0.0,
            routing_effectiveness=0.0,
            cache_performance=0.0,
            system_reliability=100.0,
            top_performing_consultants=[],
            optimization_recommendations=[
                "No data available - start processing queries"
            ],
        )

    async def start_monitoring(self, interval_seconds: int = 30):
        """Start continuous monitoring loop"""
        print(f"🔄 Starting real-time monitoring (updates every {interval_seconds}s)")

        while True:
            try:
                self.display_realtime_dashboard()
                await asyncio.sleep(interval_seconds)
            except KeyboardInterrupt:
                print("\n🛑 Monitoring stopped by user")
                break
            except Exception as e:
                print(f"❌ Monitoring error: {e}")
                await asyncio.sleep(interval_seconds)


# Example usage and testing
async def demo_benchmarking_dashboard():
    """Demonstrate the benchmarking dashboard with simulated data"""

    dashboard = RealtimeBenchmarkingDashboard()

    # Simulate some query processing events
    queries = [
        (
            "Strategic market analysis for Europe",
            1.2,
            ["market_analyst", "strategic_synthesizer"],
            "strategic_analysis",
            True,
            False,
        ),
        (
            "Fix customer support issues",
            0.8,
            ["problem_solver"],
            "problem_solving",
            True,
            True,
        ),
        (
            "Optimize manufacturing processes",
            2.1,
            ["process_expert", "operational_integrator"],
            "operational_optimization",
            True,
            False,
        ),
        (
            "Digital transformation strategy",
            3.2,
            ["strategic_synthesizer", "solution_architect", "strategic_implementer"],
            "transformation_change",
            True,
            False,
        ),
        (
            "Crisis management response",
            0.5,
            ["problem_solver", "solution_architect"],
            "crisis_management",
            True,
            True,
        ),
    ]

    print("🧪 Simulating query processing events...")

    for query, time_taken, consultants, pattern, success, cached in queries:
        dashboard.record_query_processing(
            query, time_taken, consultants, pattern, success, cached
        )
        await asyncio.sleep(0.1)  # Small delay between queries

    # Display dashboard
    dashboard.display_realtime_dashboard()

    # Show 60-minute summary
    summary = dashboard.get_benchmark_summary(60)
    print("\n📋 Benchmark Summary:")
    print(f"   Total Queries: {summary.total_queries}")
    print(f"   Average Time: {summary.avg_processing_time:.3f}s")
    print(f"   System Reliability: {summary.system_reliability:.1f}%")


if __name__ == "__main__":
    asyncio.run(demo_benchmarking_dashboard())
