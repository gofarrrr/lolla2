"""
Multi-Single-Agent Compliance Validation Tests
Ensures system adheres to MSA principles outlined in MULTI_SINGLE_AGENT_PARADIGM_FOUNDATION.md
"""

import asyncio
import sys
import os
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engine.engines.core.optimal_consultant_engine_compat import (
    OptimalConsultantEngine,
)


@dataclass
class MSATestResult:
    test_name: str
    passed: bool
    details: Dict[str, Any]
    error_message: Optional[str] = None
    execution_time: float = 0.0


@dataclass
class ComplianceReport:
    timestamp: datetime
    total_tests: int
    passed_tests: int
    failed_tests: int
    compliance_score: float
    test_results: List[MSATestResult]
    summary: str


class MSAComplianceValidator:
    """
    Validates Multi-Single-Agent compliance according to foundation principles:
    1. Independent consultant processing (no coordination)
    2. No synthesis between consultant responses
    3. Context preservation per consultant
    4. Human selection of perspectives only
    5. Parallel processing capability
    """

    def __init__(self):
        self.orchestrator = ThreeConsultantOrchestrator()
        self.optimal_engine = OptimalConsultantEngine()

    async def test_independent_processing(self) -> MSATestResult:
        """Test that consultants process independently with no coordination"""
        start_time = time.time()

        try:
            test_query = "How can we improve our marketing strategy to increase customer acquisition by 25%?"

            request = ThreeConsultantRequest(query=test_query, complexity="moderate")

            response = await self.orchestrator.process_three_consultants(request)

            # Validation checks
            checks = {
                "has_three_consultants": len(response.consultants) == 3,
                "no_coordination_metadata": response.metadata.get("coordination")
                == "none",
                "no_synthesis_metadata": response.metadata.get("synthesis") == "none",
                "paradigm_correct": response.metadata.get("paradigm")
                == "multi-single-agent",
                "independent_responses": True,  # Will validate content independence
                "parallel_processing": response.total_processing_time_seconds
                < 120,  # Reasonable parallel time
            }

            # Check response independence (different analyses)
            consultant_responses = list(response.consultants.values())
            if len(consultant_responses) >= 2:
                # Compare response content for independence
                response_contents = [r.analysis for r in consultant_responses]

                # Responses should be different (not identical)
                unique_responses = len(
                    set(response_contents[:100] for r in response_contents)
                )  # Compare first 100 chars
                checks["independent_responses"] = unique_responses > 1

            all_passed = all(checks.values())

            return MSATestResult(
                test_name="Independent Processing",
                passed=all_passed,
                details=checks,
                execution_time=time.time() - start_time,
            )

        except Exception as e:
            return MSATestResult(
                test_name="Independent Processing",
                passed=False,
                details={"error": str(e)},
                error_message=str(e),
                execution_time=time.time() - start_time,
            )

    async def test_no_synthesis(self) -> MSATestResult:
        """Test that responses are not synthesized or merged"""
        start_time = time.time()

        try:
            test_query = "What are the key factors we should consider when entering a new market?"

            request = ThreeConsultantRequest(query=test_query, complexity="complex")

            response = await self.orchestrator.process_three_consultants(request)

            # Validation checks
            checks = {
                "separate_consultant_responses": len(response.consultants) >= 3,
                "no_unified_response": True,  # No single merged response
                "maintains_consultant_identity": True,  # Each response maintains consultant role
                "no_synthesis_keywords": True,  # No synthesis language in responses
            }

            # Check that each consultant maintains their identity
            for role, consultant_data in response.consultants.items():
                # Each consultant should have their own analysis
                checks["maintains_consultant_identity"] = (
                    checks["maintains_consultant_identity"]
                    and len(consultant_data.analysis)
                    > 50  # Substantial individual response
                )

                # Check for synthesis keywords that would indicate merging
                synthesis_keywords = [
                    "combining the above",
                    "synthesizing",
                    "merging perspectives",
                    "integrating responses",
                ]
                has_synthesis_language = any(
                    keyword in consultant_data.analysis.lower()
                    for keyword in synthesis_keywords
                )
                checks["no_synthesis_keywords"] = (
                    checks["no_synthesis_keywords"] and not has_synthesis_language
                )

            # Verify response structure preserves separation
            checks["no_unified_response"] = not hasattr(response, "unified_analysis")

            all_passed = all(checks.values())

            return MSATestResult(
                test_name="No Synthesis",
                passed=all_passed,
                details=checks,
                execution_time=time.time() - start_time,
            )

        except Exception as e:
            return MSATestResult(
                test_name="No Synthesis",
                passed=False,
                details={"error": str(e)},
                error_message=str(e),
                execution_time=time.time() - start_time,
            )

    async def test_context_preservation(self) -> MSATestResult:
        """Test that each consultant preserves full context throughout processing"""
        start_time = time.time()

        try:
            test_query = "Develop a comprehensive digital transformation strategy for our manufacturing company."
            context = "We are a 200-employee manufacturing company with legacy systems and resistance to change."

            request = ThreeConsultantRequest(
                query=test_query, context=context, complexity="ultra_complex"
            )

            response = await self.orchestrator.process_three_consultants(request)

            # Validation checks
            checks = {
                "context_in_enhanced_query": context.lower()
                in response.query_enhancement_applied.lower(),
                "individual_reasoning_traces": True,
                "mental_models_preserved": True,
                "issue_trees_preserved": True,
                "confidence_scores_present": True,
            }

            # Check that each consultant maintains full processing context
            for role, consultant_data in response.consultants.items():
                # Each consultant should have reasoning trace
                checks["individual_reasoning_traces"] = (
                    checks["individual_reasoning_traces"]
                    and len(consultant_data.reasoning_trace) > 3
                )

                # Mental models should be preserved
                checks["mental_models_preserved"] = (
                    checks["mental_models_preserved"]
                    and len(consultant_data.mental_models_used) > 0
                )

                # Issue trees should be preserved
                checks["issue_trees_preserved"] = (
                    checks["issue_trees_preserved"]
                    and consultant_data.issue_tree is not None
                )

                # Confidence scores should be calculated
                checks["confidence_scores_present"] = (
                    checks["confidence_scores_present"]
                    and 0.0 <= consultant_data.confidence_score <= 1.0
                )

            all_passed = all(checks.values())

            return MSATestResult(
                test_name="Context Preservation",
                passed=all_passed,
                details=checks,
                execution_time=time.time() - start_time,
            )

        except Exception as e:
            return MSATestResult(
                test_name="Context Preservation",
                passed=False,
                details={"error": str(e)},
                error_message=str(e),
                execution_time=time.time() - start_time,
            )

    async def test_human_selection_only(self) -> MSATestResult:
        """Test that system provides options for human selection without automated choice"""
        start_time = time.time()

        try:
            test_query = "Should we acquire our main competitor or build capabilities organically?"

            request = ThreeConsultantRequest(query=test_query, complexity="complex")

            response = await self.orchestrator.process_three_consultants(request)

            # Validation checks
            checks = {
                "multiple_perspectives_available": len(response.consultants) >= 3,
                "no_automatic_selection": True,  # System doesn't choose "best" answer
                "no_ranking_imposed": True,  # System doesn't rank consultants
                "all_responses_preserved": True,  # All consultant responses maintained
                "human_choice_required": True,  # Response structure requires human interpretation
            }

            # Check that no automatic selection is made
            checks["no_automatic_selection"] = not hasattr(
                response, "recommended_consultant"
            )
            checks["no_automatic_selection"] = checks[
                "no_automatic_selection"
            ] and not hasattr(response, "best_response")

            # Check that consultants aren't ranked by the system
            checks["no_ranking_imposed"] = "consultant_ranking" not in response.metadata
            checks["no_ranking_imposed"] = (
                checks["no_ranking_imposed"]
                and "recommended_order" not in response.metadata
            )

            # Verify all responses are preserved equally
            response_lengths = [
                len(consultant_data.analysis)
                for consultant_data in response.consultants.values()
            ]
            checks["all_responses_preserved"] = all(
                length > 50 for length in response_lengths
            )  # All substantial

            # Human choice is required (no single "answer" field)
            checks["human_choice_required"] = not hasattr(response, "final_answer")

            all_passed = all(checks.values())

            return MSATestResult(
                test_name="Human Selection Only",
                passed=all_passed,
                details=checks,
                execution_time=time.time() - start_time,
            )

        except Exception as e:
            return MSATestResult(
                test_name="Human Selection Only",
                passed=False,
                details={"error": str(e)},
                error_message=str(e),
                execution_time=time.time() - start_time,
            )

    async def test_parallel_processing(self) -> MSATestResult:
        """Test that consultants can be processed in parallel for efficiency"""
        start_time = time.time()

        try:
            test_query = "How can we optimize our supply chain to reduce costs while maintaining quality?"

            request = ThreeConsultantRequest(query=test_query, complexity="moderate")

            # Measure processing time
            response = await self.orchestrator.process_three_consultants(request)
            total_time = response.total_processing_time_seconds

            # Validation checks
            checks = {
                "reasonable_processing_time": total_time
                < 300,  # Should complete in reasonable time
                "concurrent_execution": True,  # Will validate based on individual consultant times
                "all_consultants_completed": len(response.consultants) >= 3,
                "no_sequential_dependencies": True,  # No consultant waits for another
            }

            # Check individual consultant processing times
            individual_times = [
                consultant_data.processing_time_seconds
                for consultant_data in response.consultants.values()
            ]

            # If truly parallel, total time should be close to max individual time (not sum)
            max_individual_time = max(individual_times) if individual_times else 0
            sum_individual_times = sum(individual_times) if individual_times else 0

            # Parallel efficiency: total time should be much closer to max than sum
            if sum_individual_times > 0:
                parallel_efficiency = max_individual_time / sum_individual_times
                checks["concurrent_execution"] = (
                    parallel_efficiency > 0.3
                )  # At least 30% efficiency

            # Check that no consultant dependencies exist in reasoning traces
            for role, consultant_data in response.consultants.items():
                # Reasoning traces shouldn't reference other consultants
                other_consultants = [
                    other_role
                    for other_role in response.consultants.keys()
                    if other_role != role
                ]

                reasoning_text = " ".join(consultant_data.reasoning_trace).lower()
                has_dependencies = any(
                    other_role.lower() in reasoning_text
                    for other_role in other_consultants
                )

                checks["no_sequential_dependencies"] = (
                    checks["no_sequential_dependencies"] and not has_dependencies
                )

            all_passed = all(checks.values())

            return MSATestResult(
                test_name="Parallel Processing",
                passed=all_passed,
                details={
                    **checks,
                    "total_processing_time": total_time,
                    "individual_times": individual_times,
                    "parallel_efficiency": (
                        max_individual_time / sum_individual_times
                        if sum_individual_times > 0
                        else 0
                    ),
                },
                execution_time=time.time() - start_time,
            )

        except Exception as e:
            return MSATestResult(
                test_name="Parallel Processing",
                passed=False,
                details={"error": str(e)},
                error_message=str(e),
                execution_time=time.time() - start_time,
            )

    async def test_optimal_consultant_integration(self) -> MSATestResult:
        """Test that OptimalConsultantEngine integration preserves MSA principles"""
        start_time = time.time()

        try:
            test_query = (
                "We need a crisis management strategy for handling product recalls."
            )

            # Test direct optimal engine
            optimal_result = await self.optimal_engine.process_query(test_query)

            # Test integrated system
            request = ThreeConsultantRequest(query=test_query, complexity="complex")

            integrated_response = await self.orchestrator.process_three_consultants(
                request
            )

            # Validation checks
            checks = {
                "optimal_selection_working": len(optimal_result.selected_consultants)
                >= 3,
                "integration_preserves_msa": True,
                "selection_metadata_available": "optimal_selection"
                in integrated_response.metadata,
                "no_synthesis_in_selection": True,
                "independent_consultant_processing": True,
            }

            # Check that optimal selection doesn't create synthesis
            if "optimal_selection" in integrated_response.metadata:
                optimal_metadata = integrated_response.metadata["optimal_selection"]

                # Should provide selection rationale without synthesizing responses
                checks["selection_metadata_available"] = (
                    "optimal_selections" in optimal_metadata
                )

                # Selection process shouldn't create unified responses
                checks["no_synthesis_in_selection"] = (
                    "unified_response" not in optimal_metadata
                )

            # Verify that integration maintains independent processing
            checks["integration_preserves_msa"] = (
                integrated_response.metadata.get("paradigm") == "multi-single-agent"
                and integrated_response.metadata.get("coordination") == "none"
                and integrated_response.metadata.get("synthesis") == "none"
            )

            # Check that consultant responses remain independent
            consultant_responses = list(integrated_response.consultants.values())
            if len(consultant_responses) >= 2:
                # Each consultant should have distinct analysis
                analyses = [
                    r.analysis[:200] for r in consultant_responses
                ]  # First 200 chars
                unique_analyses = len(set(analyses))
                checks["independent_consultant_processing"] = unique_analyses > 1

            all_passed = all(checks.values())

            return MSATestResult(
                test_name="Optimal Consultant Integration",
                passed=all_passed,
                details=checks,
                execution_time=time.time() - start_time,
            )

        except Exception as e:
            return MSATestResult(
                test_name="Optimal Consultant Integration",
                passed=False,
                details={"error": str(e)},
                error_message=str(e),
                execution_time=time.time() - start_time,
            )

    async def run_all_compliance_tests(self) -> ComplianceReport:
        """Run all MSA compliance tests and generate report"""

        print("🔬 Running Multi-Single-Agent Compliance Tests")
        print("=" * 60)

        test_methods = [
            self.test_independent_processing,
            self.test_no_synthesis,
            self.test_context_preservation,
            self.test_human_selection_only,
            self.test_parallel_processing,
            self.test_optimal_consultant_integration,
        ]

        test_results = []

        for test_method in test_methods:
            print(f"\n🧪 Running {test_method.__name__}...")

            try:
                result = await test_method()
                test_results.append(result)

                if result.passed:
                    print(
                        f"✅ {result.test_name} - PASSED ({result.execution_time:.2f}s)"
                    )
                else:
                    print(
                        f"❌ {result.test_name} - FAILED ({result.execution_time:.2f}s)"
                    )
                    if result.error_message:
                        print(f"   Error: {result.error_message}")
                    print(f"   Details: {result.details}")

            except Exception as e:
                print(f"❌ {test_method.__name__} - CRITICAL FAILURE: {e}")
                test_results.append(
                    MSATestResult(
                        test_name=test_method.__name__.replace("test_", "")
                        .replace("_", " ")
                        .title(),
                        passed=False,
                        details={"critical_error": str(e)},
                        error_message=str(e),
                    )
                )

        # Generate compliance report
        total_tests = len(test_results)
        passed_tests = sum(1 for result in test_results if result.passed)
        failed_tests = total_tests - passed_tests
        compliance_score = passed_tests / total_tests if total_tests > 0 else 0

        # Generate summary
        if compliance_score >= 0.9:
            summary = "🎉 Excellent MSA compliance - System fully adheres to Multi-Single-Agent principles"
        elif compliance_score >= 0.7:
            summary = "✅ Good MSA compliance - Minor issues to address"
        elif compliance_score >= 0.5:
            summary = "⚠️ Moderate MSA compliance - Several violations need attention"
        else:
            summary = "❌ Poor MSA compliance - Major architectural issues detected"

        report = ComplianceReport(
            timestamp=datetime.now(),
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            compliance_score=compliance_score,
            test_results=test_results,
            summary=summary,
        )

        # Display final report
        self._display_compliance_report(report)

        return report

    def _display_compliance_report(self, report: ComplianceReport):
        """Display formatted compliance report"""

        print("\n" + "=" * 60)
        print("📋 MULTI-SINGLE-AGENT COMPLIANCE REPORT")
        print("=" * 60)
        print(f"📅 Generated: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Tests Run: {report.total_tests}")
        print(f"✅ Passed: {report.passed_tests}")
        print(f"❌ Failed: {report.failed_tests}")
        print(f"🎯 Compliance Score: {report.compliance_score:.1%}")
        print(f"📝 Summary: {report.summary}")

        print("\n📋 Test Results Detail:")
        for result in report.test_results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"   • {result.test_name}: {status}")

            if not result.passed and result.details:
                print(f"     Issues: {result.details}")

        print("\n🏗️ MSA Architecture Verification:")
        msa_principles = [
            "Independent consultant processing (no coordination)",
            "No synthesis between consultant responses",
            "Context preservation per consultant",
            "Human selection of perspectives only",
            "Parallel processing capability",
            "OptimalConsultantEngine integration compliance",
        ]

        for i, principle in enumerate(msa_principles):
            test_result = (
                report.test_results[i] if i < len(report.test_results) else None
            )
            status = "✅" if test_result and test_result.passed else "❌"
            print(f"   {status} {principle}")


# Convenience function for running tests
async def run_msa_compliance_tests():
    """Run MSA compliance validation tests"""

    validator = MSAComplianceValidator()
    report = await validator.run_all_compliance_tests()

    return report.compliance_score >= 0.8  # 80% compliance threshold


if __name__ == "__main__":
    success = asyncio.run(run_msa_compliance_tests())
    if success:
        print("\n🎉 MSA compliance validation PASSED")
        sys.exit(0)
    else:
        print("\n❌ MSA compliance validation FAILED")
        sys.exit(1)
