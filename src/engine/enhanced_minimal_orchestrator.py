#!/usr/bin/env python3
"""
Enhanced Minimal Orchestrator with Performance Instrumentation

Building on the successful minimal orchestrator, this adds:
- Comprehensive performance measurement
- Detailed timing breakdowns
- Bottleneck identification
- Performance baselines for further system expansion
"""

import asyncio
import time
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

import os
from dotenv import load_dotenv

# Import our performance instrumentation
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.performance_instrumentation import (
    get_performance_system,
    measure_function,
)

# Load environment
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(env_path)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ConsultantRole(Enum):
    """Single consultant role for enhanced orchestrator"""

    STRATEGIC_ANALYST = "Strategic Analyst"


class MentalModel(Enum):
    """Fixed mental model for enhanced orchestrator"""

    MECE_FRAMEWORK = "MECE Framework"


@dataclass
class EnhancedConsultantResponse:
    """Enhanced consultant response with performance metrics"""

    role: str
    analysis: str
    mental_model_used: str
    processing_time_seconds: float
    success: bool
    performance_breakdown: Dict[str, float]  # Detailed timing
    error: Optional[str] = None
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class EnhancedOrchestrationResponse:
    """Enhanced orchestration response with full performance metrics"""

    engagement_id: str
    query: str
    consultant_response: EnhancedConsultantResponse
    total_processing_time_seconds: float
    orchestration_overhead_seconds: float
    performance_summary: Dict[str, Any]
    success: bool
    error: Optional[str] = None
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class EnhancedMinimalOrchestrator:
    """
    Enhanced orchestrator with comprehensive performance instrumentation.
    Provides detailed timing analysis for optimization and scaling decisions.
    """

    def __init__(self):
        self.anthropic_client = None
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        self.perf_system = get_performance_system()

        if not self.anthropic_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")

        # Initialize Anthropic client with measurement
        with self.perf_system.measure_sync("client_initialization", "orchestrator"):
            try:
                import anthropic

                self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_key)
                logger.info(
                    "✅ Enhanced Minimal Orchestrator: Anthropic client initialized"
                )
            except ImportError:
                raise ImportError(
                    "anthropic package not found. Install with: pip install anthropic"
                )
            except Exception as e:
                raise Exception(f"Failed to initialize Anthropic client: {str(e)}")

    @measure_function("orchestrator_analyze", "orchestrator")
    async def analyze(
        self, query: str, context: Optional[str] = None
    ) -> EnhancedOrchestrationResponse:
        """
        Enhanced orchestration with detailed performance measurement.
        """
        engagement_id = f"enhanced_{int(time.time())}"
        orchestration_start = time.time()

        logger.info(
            f"🎯 Starting enhanced orchestration for engagement: {engagement_id}"
        )

        performance_breakdown = {}

        try:
            # Measure orchestration setup
            setup_start = time.time()
            # Orchestration logic preparation
            await asyncio.sleep(0.001)  # Minimal setup simulation
            performance_breakdown["orchestration_setup"] = time.time() - setup_start

            # Single consultant analysis with detailed timing
            async with self.perf_system.measure_async(
                "consultant_execution", "orchestrator"
            ):
                consultant_response = await asyncio.wait_for(
                    self._run_enhanced_consultant(query, context), timeout=15.0
                )

            # Measure response processing
            processing_start = time.time()
            # Response validation and formatting
            await asyncio.sleep(0.001)  # Minimal processing simulation
            performance_breakdown["response_processing"] = (
                time.time() - processing_start
            )

            total_time = time.time() - orchestration_start
            orchestration_overhead = (
                total_time - consultant_response.processing_time_seconds
            )

            # Get performance summary
            perf_summary = self.perf_system.get_performance_summary()

            logger.info(f"✅ Enhanced orchestration completed in {total_time:.2f}s")

            return EnhancedOrchestrationResponse(
                engagement_id=engagement_id,
                query=query,
                consultant_response=consultant_response,
                total_processing_time_seconds=round(total_time, 4),
                orchestration_overhead_seconds=round(orchestration_overhead, 4),
                performance_summary=perf_summary,
                success=consultant_response.success,
            )

        except asyncio.TimeoutError:
            total_time = time.time() - orchestration_start
            logger.error(f"⏰ Enhanced orchestration timeout after {total_time:.2f}s")

            timeout_response = EnhancedConsultantResponse(
                role=ConsultantRole.STRATEGIC_ANALYST.value,
                analysis="Analysis timed out after 15 seconds",
                mental_model_used=MentalModel.MECE_FRAMEWORK.value,
                processing_time_seconds=round(total_time, 4),
                performance_breakdown=performance_breakdown,
                success=False,
                error="timeout",
            )

            return EnhancedOrchestrationResponse(
                engagement_id=engagement_id,
                query=query,
                consultant_response=timeout_response,
                total_processing_time_seconds=round(total_time, 4),
                orchestration_overhead_seconds=0.0,
                performance_summary=self.perf_system.get_performance_summary(),
                success=False,
                error="orchestration_timeout",
            )

        except Exception as e:
            total_time = time.time() - orchestration_start
            logger.error(f"💥 Enhanced orchestration failed: {str(e)}")

            error_response = EnhancedConsultantResponse(
                role=ConsultantRole.STRATEGIC_ANALYST.value,
                analysis=f"Analysis failed: {str(e)}",
                mental_model_used=MentalModel.MECE_FRAMEWORK.value,
                processing_time_seconds=round(total_time, 4),
                performance_breakdown=performance_breakdown,
                success=False,
                error=str(e),
            )

            return EnhancedOrchestrationResponse(
                engagement_id=engagement_id,
                query=query,
                consultant_response=error_response,
                total_processing_time_seconds=round(total_time, 4),
                orchestration_overhead_seconds=0.0,
                performance_summary=self.perf_system.get_performance_summary(),
                success=False,
                error=str(e),
            )

    async def _run_enhanced_consultant(
        self, query: str, context: Optional[str]
    ) -> EnhancedConsultantResponse:
        """Run enhanced Strategic Analyst with detailed performance measurement"""

        consultant_start = time.time()
        performance_breakdown = {}

        try:
            # Measure prompt building
            prompt_start = time.time()
            prompt = self._build_strategic_analyst_prompt(query, context)
            performance_breakdown["prompt_building"] = time.time() - prompt_start

            logger.info("🧠 Running Enhanced Strategic Analyst with MECE framework...")

            # Measure Claude API call with detailed timing
            api_start = time.time()
            async with self.perf_system.measure_async("claude_api_call", "consultant"):
                response_text = await self._call_claude_api(prompt)
            performance_breakdown["claude_api_call"] = time.time() - api_start

            # Measure response validation
            validation_start = time.time()
            # Response validation logic (placeholder)
            validated_response = response_text  # Simple validation for now
            performance_breakdown["response_validation"] = (
                time.time() - validation_start
            )

            total_consultant_time = time.time() - consultant_start

            return EnhancedConsultantResponse(
                role=ConsultantRole.STRATEGIC_ANALYST.value,
                analysis=validated_response,
                mental_model_used=MentalModel.MECE_FRAMEWORK.value,
                processing_time_seconds=round(total_consultant_time, 4),
                performance_breakdown=performance_breakdown,
                success=True,
            )

        except Exception as e:
            total_consultant_time = time.time() - consultant_start
            logger.error(f"🔥 Enhanced Strategic Analyst failed: {str(e)}")

            return EnhancedConsultantResponse(
                role=ConsultantRole.STRATEGIC_ANALYST.value,
                analysis=f"Strategic analysis failed: {str(e)}",
                mental_model_used=MentalModel.MECE_FRAMEWORK.value,
                processing_time_seconds=round(total_consultant_time, 4),
                performance_breakdown=performance_breakdown,
                success=False,
                error=str(e),
            )

    def _build_strategic_analyst_prompt(
        self, query: str, context: Optional[str]
    ) -> str:
        """Build Strategic Analyst prompt (same as minimal orchestrator)"""

        prompt = f"""You are a Senior Strategic Analyst using the MECE (Mutually Exclusive, Collectively Exhaustive) framework.

MECE FRAMEWORK INSTRUCTIONS:
- Structure your analysis into mutually exclusive categories that together cover all possibilities
- Ensure no overlap between categories
- Cover all key aspects of the strategic question
- Be systematic and comprehensive

BUSINESS QUERY: {query}"""

        if context:
            prompt += f"\n\nBUSINESS CONTEXT: {context}"

        prompt += """

STRATEGIC ANALYSIS USING MECE FRAMEWORK:

CURRENT SITUATION:
- [Key facts and current state]

OPPORTUNITIES (Mutually Exclusive):
1. [Primary opportunity category]
2. [Secondary opportunity category]

CHALLENGES (Mutually Exclusive):
1. [Internal challenges]
2. [External challenges]

STRATEGIC OPTIONS:
A) [Option 1 with key implications]
B) [Option 2 with key implications]

RECOMMENDATION:
[Clear strategic recommendation with rationale]"""

        return prompt

    @measure_function("claude_api_direct_call", "anthropic_client")
    async def _call_claude_api(self, prompt: str) -> str:
        """Claude API call with performance measurement"""

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1500,
                    temperature=0.7,
                    messages=[{"role": "user", "content": prompt}],
                ),
            )

            if response.content and len(response.content) > 0:
                return response.content[0].text
            else:
                raise Exception("Empty response from Claude API")

        except Exception as e:
            raise Exception(f"Claude API call failed: {str(e)}")


# Enhanced test scenarios
ENHANCED_TEST_QUERIES = [
    {
        "query": "Our fintech startup's user acquisition cost has increased 300% while retention dropped to 60%. How should we pivot our growth strategy?",
        "context": "B2B payments platform, $2M ARR, Series A, highly competitive market with Stripe and Square",
    },
    {
        "query": "A major client representing 40% of revenue is threatening to switch to a competitor. What's our retention strategy?",
        "context": "Enterprise SaaS, $10M ARR, 5-year relationship, client citing price and feature gaps",
    },
    {
        "query": "We're considering acquiring a smaller competitor vs building their core feature in-house. Which path maximizes value?",
        "context": "Healthcare tech, $25M valuation, acquisition target: $5M company with strong IP portfolio",
    },
]


async def run_enhanced_orchestrator_test():
    """Test enhanced orchestrator with comprehensive performance analysis"""

    print("🔍 ENHANCED ORCHESTRATOR TEST - Full Performance Instrumentation")
    print("=" * 90)

    orchestrator = EnhancedMinimalOrchestrator()

    results = []
    total_start_time = time.time()

    for i, test_case in enumerate(ENHANCED_TEST_QUERIES, 1):
        print(f"\n📋 Test {i}/3: {test_case['query'][:70]}...")

        result = await orchestrator.analyze(
            query=test_case["query"], context=test_case.get("context")
        )

        results.append(result)

        # Print detailed performance breakdown
        status = "✅ SUCCESS" if result.success else "❌ FAILED"
        consultant_response = result.consultant_response

        print(f"   {status}")
        print(f"   📊 Total Time: {result.total_processing_time_seconds:.4f}s")
        print(
            f"   🧠 Consultant Time: {consultant_response.processing_time_seconds:.4f}s"
        )
        print(
            f"   ⚙️ Orchestration Overhead: {result.orchestration_overhead_seconds:.4f}s"
        )

        # Performance breakdown
        if consultant_response.performance_breakdown:
            print("   🔍 Timing Breakdown:")
            for (
                operation,
                duration,
            ) in consultant_response.performance_breakdown.items():
                print(f"      {operation}: {duration:.4f}s")

        if not result.success:
            print(f"   ❌ Error: {result.error}")

    total_test_time = time.time() - total_start_time

    # Enhanced performance analysis
    successful_tests = sum(1 for r in results if r.success)
    success_rate = successful_tests / len(results)

    # Detailed timing analysis
    if successful_tests > 0:
        consultant_times = [
            r.consultant_response.processing_time_seconds for r in results if r.success
        ]
        orchestration_overheads = [
            r.orchestration_overhead_seconds for r in results if r.success
        ]
        total_times = [r.total_processing_time_seconds for r in results if r.success]

        avg_consultant_time = sum(consultant_times) / len(consultant_times)
        avg_overhead = sum(orchestration_overheads) / len(orchestration_overheads)
        avg_total_time = sum(total_times) / len(total_times)
        max_total_time = max(total_times)

        # Analyze performance breakdown across all successful tests
        all_breakdowns = {}
        for r in results:
            if r.success and r.consultant_response.performance_breakdown:
                for op, duration in r.consultant_response.performance_breakdown.items():
                    if op not in all_breakdowns:
                        all_breakdowns[op] = []
                    all_breakdowns[op].append(duration)

        avg_breakdowns = {
            op: sum(durations) / len(durations)
            for op, durations in all_breakdowns.items()
        }
    else:
        avg_consultant_time = avg_overhead = avg_total_time = max_total_time = 0
        avg_breakdowns = {}

    print("\n" + "=" * 90)
    print("🔍 ENHANCED ORCHESTRATOR PERFORMANCE ANALYSIS")
    print("=" * 90)
    print(f"✅ Success Rate: {success_rate:.1%} ({successful_tests}/{len(results)})")
    print(f"⏱️ Average Total Time: {avg_total_time:.4f}s")
    print(f"🧠 Average Consultant Time: {avg_consultant_time:.4f}s")
    print(
        f"⚙️ Average Orchestration Overhead: {avg_overhead:.4f}s ({avg_overhead/avg_total_time*100:.1f}%)"
    )
    print(f"📊 Max Response Time: {max_total_time:.4f}s")
    print(f"🏁 Total Test Time: {total_test_time:.2f}s")

    if avg_breakdowns:
        print("\n🔍 AVERAGE PERFORMANCE BREAKDOWN:")
        for operation, avg_duration in sorted(
            avg_breakdowns.items(), key=lambda x: x[1], reverse=True
        ):
            percentage = (
                (avg_duration / avg_consultant_time) * 100
                if avg_consultant_time > 0
                else 0
            )
            print(f"   {operation}: {avg_duration:.4f}s ({percentage:.1f}%)")

    # Get comprehensive performance summary
    perf_system = orchestrator.perf_system
    bottlenecks = perf_system.identify_bottlenecks(threshold_seconds=5.0)

    if bottlenecks:
        print("\n🚨 PERFORMANCE BOTTLENECKS IDENTIFIED:")
        for bottleneck in bottlenecks:
            print(f"   Component: {bottleneck['component']}")
            for issue in bottleneck["issues"]:
                print(f"      ⚠️ {issue}")

    # Success determination
    if success_rate >= 0.8 and max_total_time <= 15.0:
        print("\n🎉 ENHANCED ORCHESTRATOR SUCCESS!")
        print("✅ Performance instrumentation working - ready for Query Enhancement")
        print(
            f"📈 Orchestration overhead is minimal ({avg_overhead/avg_total_time*100:.1f}%)"
        )
    elif success_rate >= 0.6:
        print("\n⚠️ ENHANCED ORCHESTRATOR PARTIAL SUCCESS")
        print("🔧 Performance insights available for optimization")
    else:
        print("\n❌ ENHANCED ORCHESTRATOR FAILED")
        print("🚨 Performance issues must be resolved")

    # Save comprehensive report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"enhanced_orchestrator_results_{timestamp}.json"

    detailed_results = {
        "test_metadata": {
            "test_id": f"enhanced_orchestrator_{timestamp}",
            "timestamp": datetime.now().isoformat(),
            "total_test_time_seconds": total_test_time,
        },
        "performance_summary": {
            "success_rate": success_rate,
            "successful_tests": successful_tests,
            "total_tests": len(results),
            "avg_total_time_seconds": avg_total_time,
            "avg_consultant_time_seconds": avg_consultant_time,
            "avg_orchestration_overhead_seconds": avg_overhead,
            "orchestration_overhead_percentage": (
                (avg_overhead / avg_total_time * 100) if avg_total_time > 0 else 0
            ),
            "max_total_time_seconds": max_total_time,
            "average_performance_breakdown": avg_breakdowns,
        },
        "bottlenecks": bottlenecks,
        "individual_test_results": [
            {
                "test_number": i + 1,
                "query": r.query,
                "engagement_id": r.engagement_id,
                "success": r.success,
                "total_time_seconds": r.total_processing_time_seconds,
                "consultant_time_seconds": r.consultant_response.processing_time_seconds,
                "orchestration_overhead_seconds": r.orchestration_overhead_seconds,
                "performance_breakdown": r.consultant_response.performance_breakdown,
                "error": r.error,
                "analysis_preview": (
                    r.consultant_response.analysis[:200] + "..."
                    if len(r.consultant_response.analysis) > 200
                    else r.consultant_response.analysis
                ),
            }
            for i, r in enumerate(results)
        ],
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)

    print(f"💾 Enhanced orchestrator results saved: {filename}")

    # Save performance system report
    perf_report_file = perf_system.save_performance_report(
        f"enhanced_orchestrator_perf_{timestamp}.json"
    )
    print(f"📊 Performance system report saved: {perf_report_file}")

    return success_rate >= 0.6


if __name__ == "__main__":
    asyncio.run(run_enhanced_orchestrator_test())
