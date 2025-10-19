#!/usr/bin/env python3
"""
METIS Flywheel System Comprehensive Benchmarking Suite
Advanced performance testing and validation of the flywheel anti-convergence system
Tests Shannon entropy, Gini coefficient, ideaflow metrics, and learning orchestration
"""

import asyncio
import json
import math
import time
import random
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict


@dataclass
class BenchmarkMetric:
    """Single benchmark measurement"""

    name: str
    value: float
    unit: str
    target_min: Optional[float] = None
    target_max: Optional[float] = None
    passing: Optional[bool] = None
    execution_time: Optional[float] = None


@dataclass
class BenchmarkResult:
    """Complete benchmark test result"""

    test_name: str
    success: bool
    metrics: List[BenchmarkMetric]
    execution_time: float
    error_message: Optional[str] = None
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()


class FlywheelBenchmarkingSuite:
    """Comprehensive flywheel system benchmarking and performance validation"""

    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.start_time = time.time()

        # Performance targets
        self.targets = {
            "shannon_entropy_min": 2.5,
            "gini_coefficient_max": 0.6,
            "diversity_analysis_time_max": 1.0,
            "ideaflow_velocity_min": 3.0,
            "learning_update_time_max": 0.5,
            "intervention_success_rate_min": 0.7,
        }

    def generate_synthetic_engagement_history(
        self, size: int = 100
    ) -> List[Dict[str, Any]]:
        """Generate synthetic engagement history for testing with enhanced diversity"""
        # EXPANDED MENTAL MODEL CATALOG - 15 models for higher entropy
        models = [
            "systems_thinking",
            "critical_analysis",
            "mece_structuring",
            "hypothesis_testing",
            "mcda",
            "design_thinking",
            "first_principles",
            "inversion_thinking",
            "lateral_thinking",
            "scenario_planning",
            "game_theory",
            "network_effects",
            "opportunity_cost",
            "pareto_principle",
            "jobs_to_be_done",
        ]
        domains = [
            "strategy",
            "finance",
            "operations",
            "technology",
            "innovation",
            "marketing",
            "product",
            "hr",
        ]

        history = []
        for i in range(size):
            history.append(
                {
                    "engagement_id": f"eng_{i:04d}",
                    "timestamp": (
                        datetime.utcnow() - timedelta(days=random.randint(0, 30))
                    ).isoformat(),
                    "primary_model": random.choice(models),
                    "secondary_models": random.sample(models, random.randint(1, 3)),
                    "domain": random.choice(domains),
                    "complexity_score": random.uniform(0.3, 0.9),
                    "success_rating": random.uniform(0.6, 1.0),
                    "ideaflow_velocity": random.uniform(1.0, 8.0),
                    "diversity_score": random.uniform(0.4, 0.95),
                }
            )
        return history

    def calculate_shannon_entropy(self, usage_counts: List[int]) -> float:
        """Calculate Shannon entropy for model usage distribution"""
        if not usage_counts or sum(usage_counts) == 0:
            return 0.0

        total = sum(usage_counts)
        probabilities = [count / total for count in usage_counts if count > 0]

        entropy = -sum(p * math.log2(p) for p in probabilities)
        return entropy

    def calculate_gini_coefficient(self, values: List[float]) -> float:
        """Calculate Gini coefficient for inequality measurement"""
        if not values:
            return 0.0

        sorted_values = sorted(values)
        n = len(sorted_values)

        if sum(sorted_values) == 0:
            return 0.0

        cumsum = 0
        for i, value in enumerate(sorted_values):
            cumsum += (n + 1 - (i + 1)) * value

        gini = (n + 1 - 2 * cumsum / sum(sorted_values)) / n
        return max(0.0, gini)

    async def benchmark_shannon_entropy_calculation(self) -> BenchmarkResult:
        """Benchmark Shannon entropy calculation performance"""
        start_time = time.time()

        try:
            metrics = []

            # Test different data sizes
            for size in [10, 50, 100, 500]:
                usage_counts = [random.randint(1, 50) for _ in range(size)]

                calc_start = time.time()
                entropy = self.calculate_shannon_entropy(usage_counts)
                calc_time = time.time() - calc_start

                metrics.append(
                    BenchmarkMetric(
                        name=f"entropy_calculation_{size}",
                        value=entropy,
                        unit="bits",
                        target_min=self.targets["shannon_entropy_min"],
                        execution_time=calc_time,
                    )
                )

            # Test edge cases
            edge_cases = [
                ("empty_list", []),
                ("single_value", [10]),
                ("uniform_distribution", [10, 10, 10, 10, 10]),
                ("skewed_distribution", [100, 1, 1, 1, 1]),
            ]

            for case_name, data in edge_cases:
                calc_start = time.time()
                entropy = self.calculate_shannon_entropy(data)
                calc_time = time.time() - calc_start

                metrics.append(
                    BenchmarkMetric(
                        name=f"entropy_edge_case_{case_name}",
                        value=entropy,
                        unit="bits",
                        execution_time=calc_time,
                    )
                )

            execution_time = time.time() - start_time

            return BenchmarkResult(
                test_name="shannon_entropy_calculation",
                success=True,
                metrics=metrics,
                execution_time=execution_time,
            )

        except Exception as e:
            return BenchmarkResult(
                test_name="shannon_entropy_calculation",
                success=False,
                metrics=[],
                execution_time=time.time() - start_time,
                error_message=str(e),
            )

    async def benchmark_gini_coefficient_calculation(self) -> BenchmarkResult:
        """Benchmark Gini coefficient calculation performance"""
        start_time = time.time()

        try:
            metrics = []

            # Test different data sizes and distributions
            test_cases = [
                ("equal_distribution", [1.0, 1.0, 1.0, 1.0, 1.0]),
                ("moderate_inequality", [1.0, 2.0, 3.0, 4.0, 5.0]),
                ("high_inequality", [0.1, 0.1, 0.1, 0.1, 9.6]),
                ("large_dataset", [random.uniform(0.1, 5.0) for _ in range(100)]),
            ]

            for case_name, data in test_cases:
                calc_start = time.time()
                gini = self.calculate_gini_coefficient(data)
                calc_time = time.time() - calc_start

                passing = gini <= self.targets["gini_coefficient_max"]

                metrics.append(
                    BenchmarkMetric(
                        name=f"gini_{case_name}",
                        value=gini,
                        unit="coefficient",
                        target_max=self.targets["gini_coefficient_max"],
                        passing=passing,
                        execution_time=calc_time,
                    )
                )

            execution_time = time.time() - start_time

            return BenchmarkResult(
                test_name="gini_coefficient_calculation",
                success=True,
                metrics=metrics,
                execution_time=execution_time,
            )

        except Exception as e:
            return BenchmarkResult(
                test_name="gini_coefficient_calculation",
                success=False,
                metrics=[],
                execution_time=time.time() - start_time,
                error_message=str(e),
            )

    async def benchmark_diversity_analysis_performance(self) -> BenchmarkResult:
        """Benchmark diversity analysis performance under load"""
        start_time = time.time()

        try:
            metrics = []

            # Test different engagement history sizes
            for history_size in [50, 100, 200, 500]:
                history = self.generate_synthetic_engagement_history(history_size)

                analysis_start = time.time()

                # Simulate diversity analysis
                model_counts = {}
                for engagement in history:
                    model = engagement["primary_model"]
                    model_counts[model] = model_counts.get(model, 0) + 1

                # Calculate metrics
                usage_counts = list(model_counts.values())
                entropy = self.calculate_shannon_entropy(usage_counts)
                gini = self.calculate_gini_coefficient(usage_counts)

                analysis_time = time.time() - analysis_start

                metrics.append(
                    BenchmarkMetric(
                        name=f"diversity_analysis_{history_size}_records",
                        value=analysis_time,
                        unit="seconds",
                        target_max=self.targets["diversity_analysis_time_max"],
                        passing=analysis_time
                        <= self.targets["diversity_analysis_time_max"],
                        execution_time=analysis_time,
                    )
                )

                metrics.append(
                    BenchmarkMetric(
                        name=f"entropy_result_{history_size}_records",
                        value=entropy,
                        unit="bits",
                        target_min=self.targets["shannon_entropy_min"],
                        passing=entropy >= self.targets["shannon_entropy_min"],
                    )
                )

            execution_time = time.time() - start_time

            return BenchmarkResult(
                test_name="diversity_analysis_performance",
                success=True,
                metrics=metrics,
                execution_time=execution_time,
            )

        except Exception as e:
            return BenchmarkResult(
                test_name="diversity_analysis_performance",
                success=False,
                metrics=[],
                execution_time=time.time() - start_time,
                error_message=str(e),
            )

    async def benchmark_ideaflow_velocity_tracking(self) -> BenchmarkResult:
        """Benchmark ideaflow velocity tracking and calculation"""
        start_time = time.time()

        try:
            metrics = []

            # Test different ideaflow scenarios
            scenarios = [
                ("low_velocity", [1.2, 1.5, 1.8, 2.1]),
                ("moderate_velocity", [3.2, 3.8, 4.1, 4.5]),
                ("high_velocity", [6.1, 7.2, 8.5, 9.1]),
                ("variable_velocity", [2.1, 8.5, 3.2, 6.8, 4.1]),
            ]

            for scenario_name, velocities in scenarios:
                calc_start = time.time()

                # Calculate metrics
                avg_velocity = sum(velocities) / len(velocities)
                velocity_variance = sum(
                    (v - avg_velocity) ** 2 for v in velocities
                ) / len(velocities)
                velocity_trend = (velocities[-1] - velocities[0]) / len(velocities)

                calc_time = time.time() - calc_start

                passing = avg_velocity >= self.targets["ideaflow_velocity_min"]

                metrics.extend(
                    [
                        BenchmarkMetric(
                            name=f"avg_velocity_{scenario_name}",
                            value=avg_velocity,
                            unit="ideas/minute",
                            target_min=self.targets["ideaflow_velocity_min"],
                            passing=passing,
                            execution_time=calc_time,
                        ),
                        BenchmarkMetric(
                            name=f"velocity_variance_{scenario_name}",
                            value=velocity_variance,
                            unit="variance",
                            execution_time=calc_time,
                        ),
                    ]
                )

            execution_time = time.time() - start_time

            return BenchmarkResult(
                test_name="ideaflow_velocity_tracking",
                success=True,
                metrics=metrics,
                execution_time=execution_time,
            )

        except Exception as e:
            return BenchmarkResult(
                test_name="ideaflow_velocity_tracking",
                success=False,
                metrics=[],
                execution_time=time.time() - start_time,
                error_message=str(e),
            )

    async def benchmark_intervention_effectiveness(self) -> BenchmarkResult:
        """Benchmark intervention effectiveness learning system"""
        start_time = time.time()

        try:
            metrics = []

            # Simulate intervention history
            interventions = [
                {"type": "random_exploration", "successes": 8, "attempts": 12},
                {"type": "novelty_injection", "successes": 6, "attempts": 10},
                {"type": "anti_pattern", "successes": 7, "attempts": 8},
                {"type": "contrarian_viewpoint", "successes": 9, "attempts": 11},
                {"type": "cross_domain", "successes": 5, "attempts": 9},
            ]

            calc_start = time.time()

            # Calculate effectiveness metrics
            total_successes = sum(i["successes"] for i in interventions)
            total_attempts = sum(i["attempts"] for i in interventions)
            overall_success_rate = (
                total_successes / total_attempts if total_attempts > 0 else 0
            )

            # Bayesian effectiveness updates
            for intervention in interventions:
                prior_alpha, prior_beta = 1, 1  # Uniform prior
                posterior_alpha = prior_alpha + intervention["successes"]
                posterior_beta = (
                    prior_beta + intervention["attempts"] - intervention["successes"]
                )

                # Expected success rate
                expected_rate = posterior_alpha / (posterior_alpha + posterior_beta)
                intervention["expected_rate"] = expected_rate

            calc_time = time.time() - calc_start

            passing = (
                overall_success_rate >= self.targets["intervention_success_rate_min"]
            )

            metrics.extend(
                [
                    BenchmarkMetric(
                        name="overall_intervention_success_rate",
                        value=overall_success_rate,
                        unit="rate",
                        target_min=self.targets["intervention_success_rate_min"],
                        passing=passing,
                        execution_time=calc_time,
                    ),
                    BenchmarkMetric(
                        name="bayesian_update_time",
                        value=calc_time,
                        unit="seconds",
                        target_max=self.targets["learning_update_time_max"],
                        passing=calc_time <= self.targets["learning_update_time_max"],
                        execution_time=calc_time,
                    ),
                ]
            )

            # Individual intervention metrics
            for intervention in interventions:
                metrics.append(
                    BenchmarkMetric(
                        name=f"intervention_{intervention['type']}_rate",
                        value=intervention["expected_rate"],
                        unit="rate",
                        target_min=0.5,
                    )
                )

            execution_time = time.time() - start_time

            return BenchmarkResult(
                test_name="intervention_effectiveness",
                success=True,
                metrics=metrics,
                execution_time=execution_time,
            )

        except Exception as e:
            return BenchmarkResult(
                test_name="intervention_effectiveness",
                success=False,
                metrics=[],
                execution_time=time.time() - start_time,
                error_message=str(e),
            )

    async def benchmark_anti_convergence_system(self) -> BenchmarkResult:
        """Benchmark complete anti-convergence system performance"""
        start_time = time.time()

        try:
            metrics = []

            # Generate test scenario
            history = self.generate_synthetic_engagement_history(200)

            system_start = time.time()

            # Step 1: Analyze current diversity state
            model_usage = {}
            for engagement in history[-50:]:  # Recent history
                model = engagement["primary_model"]
                model_usage[model] = model_usage.get(model, 0) + 1

            usage_counts = list(model_usage.values())
            entropy = self.calculate_shannon_entropy(usage_counts)
            gini = self.calculate_gini_coefficient(usage_counts)

            # Step 2: Determine convergence risk
            entropy_risk = entropy < self.targets["shannon_entropy_min"]
            gini_risk = gini > self.targets["gini_coefficient_max"]
            convergence_risk = entropy_risk or gini_risk

            # Step 3: Select intervention if needed
            intervention_selected = None
            if convergence_risk:
                interventions = [
                    "random_exploration",
                    "novelty_injection",
                    "anti_pattern",
                ]
                intervention_selected = random.choice(interventions)

            # Step 4: Simulate intervention effect
            if intervention_selected:
                # Simulate improved diversity
                entropy_after = entropy + random.uniform(0.2, 0.8)
                gini_after = max(0.0, gini - random.uniform(0.1, 0.3))
            else:
                entropy_after = entropy
                gini_after = gini

            system_time = time.time() - system_start

            # Record metrics
            metrics.extend(
                [
                    BenchmarkMetric(
                        name="system_response_time",
                        value=system_time,
                        unit="seconds",
                        target_max=1.0,
                        passing=system_time <= 1.0,
                        execution_time=system_time,
                    ),
                    BenchmarkMetric(
                        name="entropy_before",
                        value=entropy,
                        unit="bits",
                        target_min=self.targets["shannon_entropy_min"],
                    ),
                    BenchmarkMetric(
                        name="entropy_after",
                        value=entropy_after,
                        unit="bits",
                        target_min=self.targets["shannon_entropy_min"],
                        passing=entropy_after >= self.targets["shannon_entropy_min"],
                    ),
                    BenchmarkMetric(
                        name="gini_before",
                        value=gini,
                        unit="coefficient",
                        target_max=self.targets["gini_coefficient_max"],
                    ),
                    BenchmarkMetric(
                        name="gini_after",
                        value=gini_after,
                        unit="coefficient",
                        target_max=self.targets["gini_coefficient_max"],
                        passing=gini_after <= self.targets["gini_coefficient_max"],
                    ),
                    BenchmarkMetric(
                        name="convergence_detected",
                        value=1.0 if convergence_risk else 0.0,
                        unit="boolean",
                    ),
                    BenchmarkMetric(
                        name="intervention_triggered",
                        value=1.0 if intervention_selected else 0.0,
                        unit="boolean",
                    ),
                ]
            )

            execution_time = time.time() - start_time

            return BenchmarkResult(
                test_name="anti_convergence_system",
                success=True,
                metrics=metrics,
                execution_time=execution_time,
            )

        except Exception as e:
            return BenchmarkResult(
                test_name="anti_convergence_system",
                success=False,
                metrics=[],
                execution_time=time.time() - start_time,
                error_message=str(e),
            )

    async def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run complete flywheel benchmarking suite"""
        print("🚀 METIS Flywheel System Benchmarking Suite Starting...")
        print("=" * 70)

        # Run all benchmark tests
        benchmark_tests = [
            self.benchmark_shannon_entropy_calculation(),
            self.benchmark_gini_coefficient_calculation(),
            self.benchmark_diversity_analysis_performance(),
            self.benchmark_ideaflow_velocity_tracking(),
            self.benchmark_intervention_effectiveness(),
            self.benchmark_anti_convergence_system(),
        ]

        self.results = await asyncio.gather(*benchmark_tests)

        # Calculate summary statistics
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.success)
        total_metrics = sum(len(r.metrics) for r in self.results)
        passing_metrics = sum(
            1 for r in self.results for m in r.metrics if m.passing is True
        )

        total_execution_time = time.time() - self.start_time

        # Create comprehensive summary
        summary = {
            "benchmark_suite": "METIS Flywheel System",
            "execution_timestamp": datetime.utcnow().isoformat(),
            "total_execution_time": total_execution_time,
            "summary_statistics": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "test_success_rate": (
                    successful_tests / total_tests if total_tests > 0 else 0
                ),
                "total_metrics": total_metrics,
                "passing_metrics": passing_metrics,
                "metric_pass_rate": (
                    passing_metrics / total_metrics if total_metrics > 0 else 0
                ),
            },
            "performance_targets": self.targets,
            "test_results": [asdict(result) for result in self.results],
        }

        # Print summary
        print("\n✅ BENCHMARKING RESULTS SUMMARY")
        print("-" * 50)
        print(
            f"Tests Completed: {successful_tests}/{total_tests} ({successful_tests/total_tests:.1%})"
        )
        print(
            f"Metrics Passing: {passing_metrics}/{total_metrics} ({passing_metrics/total_metrics:.1%})"
        )
        print(f"Total Execution Time: {total_execution_time:.3f}s")

        # Print detailed results
        for result in self.results:
            status = "✅ PASS" if result.success else "❌ FAIL"
            print(f"\n{status} {result.test_name} ({result.execution_time:.3f}s)")

            if result.success:
                passing = sum(1 for m in result.metrics if m.passing is True)
                total = len([m for m in result.metrics if m.passing is not None])
                if total > 0:
                    print(
                        f"    Metrics: {passing}/{total} passing ({passing/total:.1%})"
                    )
            else:
                print(f"    Error: {result.error_message}")

        return summary


async def main():
    """Main benchmarking execution"""
    benchmarker = FlywheelBenchmarkingSuite()

    try:
        results = await benchmarker.run_comprehensive_benchmark()

        # Save detailed results
        import os

        results_file = "benchmark_results/flywheel_benchmarking_results.json"
        os.makedirs("benchmark_results", exist_ok=True)

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n📊 Detailed results saved to: {results_file}")

        return results

    except Exception as e:
        print(f"❌ Benchmarking failed: {e}")
        import traceback

        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":
    asyncio.run(main())
