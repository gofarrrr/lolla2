#!/usr/bin/env python3
"""
Daily Evaluation Pipeline
Automated evaluation system that runs daily to measure system performance
"""

import asyncio
import json
import random
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path
from uuid import uuid4

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.evaluation.simple_llm_judge import SimpleLLMJudge
from src.evaluation.simple_ab_testing import SimpleABTesting
from src.core.stateful_pipeline_orchestrator import StatefulPipelineOrchestrator


class DailyEvaluationPipeline:
    """Automated daily evaluation system"""

    def __init__(self, data_dir: str = "evaluation_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # Initialize components
        self.llm_judge = SimpleLLMJudge()
        self.ab_testing = SimpleABTesting(data_dir=str(self.data_dir))
        self.orchestrator = StatefulPipelineOrchestrator()

        # Load golden dataset
        self.golden_dataset = self._load_golden_dataset()

        # Reports directory
        self.reports_dir = self.data_dir / "daily_reports"
        self.reports_dir.mkdir(exist_ok=True)

    def _load_golden_dataset(self) -> List[Dict]:
        """Load all golden examples from JSON files"""
        golden_examples = []
        golden_dir = Path("golden_examples")

        if not golden_dir.exists():
            print("⚠️ Golden examples directory not found")
            return []

        for json_file in golden_dir.glob("*.json"):
            try:
                with open(json_file, "r") as f:
                    examples = json.load(f)
                    golden_examples.extend(examples)
                    print(f"📚 Loaded {len(examples)} examples from {json_file.name}")
            except Exception as e:
                print(f"Error loading {json_file}: {e}")

        print(f"📊 Total golden examples loaded: {len(golden_examples)}")
        return golden_examples

    async def run_system_on_query(
        self, query: str, trace_id: str = None
    ) -> Dict[str, Any]:
        """Run the system on a single query and return results"""
        try:
            trace_id = trace_id or uuid4()

            # Execute the pipeline
            result = await self.orchestrator.execute_pipeline(
                initial_query=query, trace_id=trace_id
            )

            return {
                "success": True,
                "query": query,
                "trace_id": trace_id,
                "result": result,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            print(f"❌ Error running system on query: {e}")
            return {
                "success": False,
                "query": query,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    async def run_regression_tests(self, sample_size: int = 10) -> Dict[str, Any]:
        """Run regression tests on golden dataset"""
        print(f"🔄 Running regression tests on {sample_size} examples...")

        if not self.golden_dataset:
            return {"error": "No golden dataset available"}

        # Sample golden examples
        test_examples = random.sample(
            self.golden_dataset, min(sample_size, len(self.golden_dataset))
        )

        results = []
        for i, example in enumerate(test_examples):
            print(f"   Testing {i+1}/{len(test_examples)}: {example['query'][:50]}...")

            # Run system
            system_result = await self.run_system_on_query(example["query"])

            if system_result["success"]:
                # Judge the output
                # Mock response for now - in real implementation this would come from system
                mock_response = f"Analysis for: {example['query']}"

                overall_score, judge_details = (
                    await self.llm_judge.evaluate_consultant_output(
                        example["query"], mock_response, example
                    )
                )

                passed = overall_score >= example.get("quality_threshold", 4.0)

                results.append(
                    {
                        "example_id": example["id"],
                        "query": example["query"],
                        "score": overall_score,
                        "passed": passed,
                        "judge_details": judge_details,
                        "expected_consultant": example.get("expected_consultant"),
                        "expected_nways": example.get("expected_nways", []),
                    }
                )
            else:
                results.append(
                    {
                        "example_id": example["id"],
                        "query": example["query"],
                        "score": 0.0,
                        "passed": False,
                        "error": system_result.get("error"),
                        "expected_consultant": example.get("expected_consultant"),
                        "expected_nways": example.get("expected_nways", []),
                    }
                )

        # Calculate summary statistics
        passed_tests = [r for r in results if r["passed"]]
        pass_rate = len(passed_tests) / len(results) if results else 0
        avg_score = sum(r["score"] for r in results) / len(results) if results else 0

        return {
            "total_tests": len(results),
            "passed": len(passed_tests),
            "pass_rate": pass_rate,
            "avg_score": avg_score,
            "results": results,
            "timestamp": datetime.now().isoformat(),
        }

    async def sample_production_queries(self, n: int = 5) -> List[str]:
        """Sample recent production queries for evaluation"""
        # For demo purposes, use golden dataset queries
        # In real implementation, this would query production logs

        if not self.golden_dataset:
            return [
                "Should we acquire our main competitor for $100M?",
                "Is it safe to expand to a new market during recession?",
                "Should we implement a 4-day work week?",
                "Our data was breached. How should we respond?",
                "Should we pivot from B2B to B2C?",
            ]

        sample_queries = random.sample(
            self.golden_dataset, min(n, len(self.golden_dataset))
        )
        return [example["query"] for example in sample_queries]

    async def evaluate_production_sample(self, queries: List[str]) -> Dict[str, Any]:
        """Evaluate sample of production queries"""
        print(f"📊 Evaluating {len(queries)} production queries...")

        results = []
        for i, query in enumerate(queries):
            print(f"   Evaluating {i+1}/{len(queries)}: {query[:50]}...")

            # Run system
            system_result = await self.run_system_on_query(query)

            if system_result["success"]:
                # Mock response for evaluation
                mock_response = f"Production analysis for: {query}"

                # Judge without golden example (just basic quality)
                overall_score, judge_details = (
                    await self.llm_judge.evaluate_consultant_output(
                        query, mock_response
                    )
                )

                results.append(
                    {
                        "query": query,
                        "score": overall_score,
                        "judge_details": judge_details,
                        "response_time": 2.5,  # Mock response time
                        "success": True,
                    }
                )
            else:
                results.append(
                    {
                        "query": query,
                        "score": 0.0,
                        "error": system_result.get("error"),
                        "success": False,
                    }
                )

        successful_results = [r for r in results if r["success"]]
        avg_score = (
            sum(r["score"] for r in successful_results) / len(successful_results)
            if successful_results
            else 0
        )
        success_rate = len(successful_results) / len(results) if results else 0

        return {
            "total_queries": len(results),
            "successful": len(successful_results),
            "success_rate": success_rate,
            "avg_score": avg_score,
            "results": results,
            "timestamp": datetime.now().isoformat(),
        }

    def check_ab_experiments(self) -> Dict[str, Any]:
        """Check status of active A/B experiments"""
        print("🧪 Checking A/B experiment status...")

        active_experiments = self.ab_testing.get_active_experiments()
        experiment_summaries = {}

        for exp_name in active_experiments:
            summary = self.ab_testing.get_experiment_summary(exp_name)
            analysis = self.ab_testing.analyze_experiment(exp_name)

            experiment_summaries[exp_name] = {
                "summary": summary,
                "analysis": analysis.__dict__ if analysis else None,
                "ready_for_decision": analysis is not None and analysis.is_significant,
            }

        return {
            "active_experiments": len(active_experiments),
            "experiments": experiment_summaries,
            "timestamp": datetime.now().isoformat(),
        }

    def generate_daily_report(
        self, regression_results: Dict, production_results: Dict, ab_results: Dict
    ) -> Dict[str, Any]:
        """Generate comprehensive daily report"""

        # System health assessment
        health_status = "🟢 HEALTHY"
        if regression_results.get("pass_rate", 0) < 0.8:
            health_status = "🟡 DEGRADED"
        if regression_results.get("pass_rate", 0) < 0.6:
            health_status = "🔴 CRITICAL"

        # Key metrics
        metrics = {
            "regression_pass_rate": regression_results.get("pass_rate", 0),
            "regression_avg_score": regression_results.get("avg_score", 0),
            "production_success_rate": production_results.get("success_rate", 0),
            "production_avg_score": production_results.get("avg_score", 0),
            "active_experiments": ab_results.get("active_experiments", 0),
        }

        # Recommendations
        recommendations = []

        if metrics["regression_pass_rate"] < 0.8:
            recommendations.append(
                "🚨 Regression tests failing - investigate quality degradation"
            )

        if metrics["production_avg_score"] < 3.5:
            recommendations.append(
                "📉 Production quality below target - review recent changes"
            )

        for exp_name, exp_data in ab_results.get("experiments", {}).items():
            if exp_data.get("ready_for_decision"):
                analysis = exp_data.get("analysis")
                if analysis and analysis.get("improvement_percent", 0) > 5:
                    recommendations.append(
                        f"🚀 Experiment '{exp_name}' ready to ship - {analysis.get('improvement_percent'):.1f}% improvement"
                    )

        if not recommendations:
            recommendations.append(
                "✅ System performing well - no immediate actions needed"
            )

        report = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "health_status": health_status,
            "metrics": metrics,
            "recommendations": recommendations,
            "detailed_results": {
                "regression": regression_results,
                "production": production_results,
                "ab_testing": ab_results,
            },
            "generated_at": datetime.now().isoformat(),
        }

        return report

    def save_report(self, report: Dict[str, Any]) -> str:
        """Save daily report to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.reports_dir / f"daily_report_{timestamp}.json"

        try:
            with open(report_file, "w") as f:
                json.dump(report, f, indent=2)

            print(f"💾 Report saved: {report_file}")
            return str(report_file)

        except Exception as e:
            print(f"Error saving report: {e}")
            return ""

    def print_report_summary(self, report: Dict[str, Any]):
        """Print human-readable report summary"""
        print(f"\n📊 DAILY EVALUATION REPORT - {report['date']}")
        print("=" * 60)

        print(f"🏥 System Health: {report['health_status']}")

        print("\n📈 Key Metrics:")
        metrics = report["metrics"]
        print(f"   Regression Pass Rate: {metrics['regression_pass_rate']:.1%}")
        print(f"   Regression Avg Score: {metrics['regression_avg_score']:.2f}/5.0")
        print(f"   Production Success Rate: {metrics['production_success_rate']:.1%}")
        print(f"   Production Avg Score: {metrics['production_avg_score']:.2f}/5.0")
        print(f"   Active A/B Tests: {metrics['active_experiments']}")

        print("\n🎯 Recommendations:")
        for rec in report["recommendations"]:
            print(f"   {rec}")

        print("\n" + "=" * 60)

    async def run_daily_evaluation(
        self, regression_sample_size: int = 10, production_sample_size: int = 5
    ) -> Dict[str, Any]:
        """Run complete daily evaluation pipeline"""
        print("🔄 Starting daily evaluation pipeline...")
        start_time = datetime.now()

        try:
            # 1. Regression testing
            regression_results = await self.run_regression_tests(regression_sample_size)

            # 2. Production sample evaluation
            production_queries = await self.sample_production_queries(
                production_sample_size
            )
            production_results = await self.evaluate_production_sample(
                production_queries
            )

            # 3. A/B experiment check
            ab_results = self.check_ab_experiments()

            # 4. Generate report
            report = self.generate_daily_report(
                regression_results, production_results, ab_results
            )

            # 5. Save and display
            report_file = self.save_report(report)
            self.print_report_summary(report)

            duration = (datetime.now() - start_time).total_seconds()
            print(f"\n⏱️ Evaluation completed in {duration:.1f} seconds")

            return report

        except Exception as e:
            print(f"❌ Daily evaluation failed: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}


# Main execution
async def main():
    """Run daily evaluation"""
    pipeline = DailyEvaluationPipeline()

    # Create a sample A/B test if none exist
    if not pipeline.ab_testing.get_active_experiments():
        print("🧪 Creating sample A/B experiment...")
        pipeline.ab_testing.create_experiment(
            name="daily_eval_demo",
            hypothesis="Enhanced personas improve satisfaction",
            control_config={"persona": "generic"},
            treatment_config={"persona": "enhanced"},
            primary_metric="user_satisfaction",
            min_sample_size=10,
        )

        # Add some sample metrics
        for i in range(20):
            user_id = f"demo_user_{i}"
            satisfaction = (
                random.uniform(3.5, 4.5) if i % 2 == 0 else random.uniform(4.0, 5.0)
            )
            pipeline.ab_testing.track_metric(
                user_id, "daily_eval_demo", "user_satisfaction", satisfaction
            )

    # Run evaluation
    report = await pipeline.run_daily_evaluation(
        regression_sample_size=5, production_sample_size=3  # Small sample for demo
    )

    return report


if __name__ == "__main__":
    asyncio.run(main())
