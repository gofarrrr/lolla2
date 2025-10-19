"""
METIS Phase 1 CRITICAL: LLM Validation Gates System
Industry Insight: Reliability as Foundation - mandatory LLM validation gates

Implements comprehensive LLM health monitoring, validation, and mandatory
enhancement ratio tracking to prevent Zero LLM Processing issues.

Performance Targets:
- 100% LLM enhancement ratio (no fallback processing)
- <500ms validation gate response time
- Real-time health monitoring with alerts
- Mandatory validation before any cognitive processing
"""

import asyncio
import logging
import os
import time
import traceback
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json

# Import METIS core components
from src.engine.models.data_contracts import MetisDataContract

# Legacy LLM client imports (optional)
try:
    from src.integrations.claude_client import get_claude_client, LLMCallType

    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False

# Global cache for validation gate results
_GATE_CACHE: Dict[str, Any] = {"ts": None, "result": None}


class LLMHealthStatus(str, Enum):
    """LLM provider health status levels"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    OFFLINE = "offline"
    UNKNOWN = "unknown"


class ValidationGateResult(str, Enum):
    """Validation gate results"""

    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    CRITICAL_FAIL = "critical_fail"


@dataclass
class LLMProviderHealth:
    """Health status for individual LLM provider"""

    provider_name: str
    status: LLMHealthStatus
    last_successful_call: Optional[datetime]
    consecutive_failures: int
    average_response_time_ms: float
    total_calls: int
    successful_calls: int
    failed_calls: int
    last_error: Optional[str]
    api_key_status: str  # "valid", "invalid", "missing"

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage"""
        if self.total_calls == 0:
            return 0.0
        return (self.successful_calls / self.total_calls) * 100

    @property
    def availability_score(self) -> float:
        """Calculate availability score (0-1)"""
        if self.consecutive_failures > 5:
            return 0.0
        elif self.consecutive_failures > 2:
            return 0.5
        elif self.success_rate < 50:
            return 0.3
        else:
            return min(self.success_rate / 100, 1.0)


@dataclass
class ValidationGateReport:
    """Report from validation gate execution"""

    gate_name: str
    result: ValidationGateResult
    execution_time_ms: float
    details: str
    recommendations: List[str]
    critical_issues: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class LLMSystemHealth:
    """Overall LLM system health assessment"""

    overall_status: LLMHealthStatus
    provider_health: Dict[str, LLMProviderHealth]
    validation_gates: List[ValidationGateReport]
    enhancement_ratio: float  # Percentage of processes using LLM vs fallback
    system_ready: bool
    critical_alerts: List[str]
    last_assessment: datetime = field(default_factory=datetime.now)


class LLMProviderValidator:
    """
    Validates individual LLM provider connectivity and performance
    Implements health checks and diagnostics for each provider
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.health_history: Dict[str, List[LLMProviderHealth]] = {}

    async def validate_anthropic_health(self) -> LLMProviderHealth:
        """Validate Anthropic Claude provider health"""
        start_time = time.time()

        try:
            # Import and test Anthropic connection
            from src.core.llm_integration_adapter import get_unified_llm_adapter

            claude_client = await get_claude_client()

            # Test API key status first
            api_key_status = (
                "valid" if await claude_client.is_available() else "invalid"
            )

            if api_key_status == "invalid":
                return LLMProviderHealth(
                    provider_name="anthropic",
                    status=LLMHealthStatus.OFFLINE,
                    last_successful_call=None,
                    consecutive_failures=999,
                    average_response_time_ms=0,
                    total_calls=0,
                    successful_calls=0,
                    failed_calls=1,
                    last_error="API key validation failed",
                    api_key_status=api_key_status,
                )

            # Test actual LLM call
            test_response = await claude_client.call_claude(
                prompt="Respond with exactly 'OK' if you can process this request.",
                call_type=LLMCallType.VALIDATION,
                max_tokens=10,
                temperature=0.0,
            )

            response_time = (time.time() - start_time) * 1000

            # Validate response quality
            if "OK" in test_response.content.upper():
                status = LLMHealthStatus.HEALTHY
                error = None
            else:
                status = LLMHealthStatus.DEGRADED
                error = f"Unexpected response: {test_response.content[:50]}"

            return LLMProviderHealth(
                provider_name="anthropic",
                status=status,
                last_successful_call=(
                    datetime.now() if status == LLMHealthStatus.HEALTHY else None
                ),
                consecutive_failures=0 if status == LLMHealthStatus.HEALTHY else 1,
                average_response_time_ms=response_time,
                total_calls=1,
                successful_calls=1 if status == LLMHealthStatus.HEALTHY else 0,
                failed_calls=0 if status == LLMHealthStatus.HEALTHY else 1,
                last_error=error,
                api_key_status=api_key_status,
            )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            error_details = f"{type(e).__name__}: {str(e)}"

            self.logger.error(f"Anthropic validation failed: {error_details}")

            return LLMProviderHealth(
                provider_name="anthropic",
                status=LLMHealthStatus.CRITICAL,
                last_successful_call=None,
                consecutive_failures=1,
                average_response_time_ms=response_time,
                total_calls=1,
                successful_calls=0,
                failed_calls=1,
                last_error=error_details,
                api_key_status="unknown",
            )

    async def validate_openai_health(self) -> LLMProviderHealth:
        """Validate OpenAI provider health"""
        start_time = time.time()

        try:
            from src.integrations.llm_provider import get_llm_client

            llm_client = get_llm_client()

            # Check if OpenAI is available
            available_providers = llm_client.get_available_providers()

            if "openai" not in available_providers:
                return LLMProviderHealth(
                    provider_name="openai",
                    status=LLMHealthStatus.OFFLINE,
                    last_successful_call=None,
                    consecutive_failures=999,
                    average_response_time_ms=0,
                    total_calls=0,
                    successful_calls=0,
                    failed_calls=0,
                    last_error="OpenAI not configured",
                    api_key_status="missing",
                )

            # Test OpenAI call
            test_messages = [
                {
                    "role": "user",
                    "content": "Respond with exactly 'OK' if you can process this request.",
                }
            ]

            response = await llm_client._call_openai(test_messages, "gpt-4o-mini")
            response_time = (time.time() - start_time) * 1000

            if "OK" in response.content.upper():
                status = LLMHealthStatus.HEALTHY
                error = None
            else:
                status = LLMHealthStatus.DEGRADED
                error = f"Unexpected response: {response.content[:50]}"

            return LLMProviderHealth(
                provider_name="openai",
                status=status,
                last_successful_call=(
                    datetime.now() if status == LLMHealthStatus.HEALTHY else None
                ),
                consecutive_failures=0 if status == LLMHealthStatus.HEALTHY else 1,
                average_response_time_ms=response_time,
                total_calls=1,
                successful_calls=1 if status == LLMHealthStatus.HEALTHY else 0,
                failed_calls=0 if status == LLMHealthStatus.HEALTHY else 1,
                last_error=error,
                api_key_status="valid",
            )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            error_details = f"{type(e).__name__}: {str(e)}"

            return LLMProviderHealth(
                provider_name="openai",
                status=LLMHealthStatus.CRITICAL,
                last_successful_call=None,
                consecutive_failures=1,
                average_response_time_ms=response_time,
                total_calls=1,
                successful_calls=0,
                failed_calls=1,
                last_error=error_details,
                api_key_status="unknown",
            )

    async def validate_perplexity_health(self) -> LLMProviderHealth:
        """Validate Perplexity research provider health"""
        start_time = time.time()

        try:
            from src.integrations.research_manager import get_research_manager

            research_manager = get_research_manager()

            # Test basic research capability
            test_result = await research_manager.fetch_facts(
                "Test query for system validation", mode="moderate"
            )

            response_time = (time.time() - start_time) * 1000

            if hasattr(test_result, "summary") and test_result.summary:
                status = LLMHealthStatus.HEALTHY
                error = None
            else:
                status = LLMHealthStatus.DEGRADED
                error = "Research response missing expected content"

            return LLMProviderHealth(
                provider_name="perplexity",
                status=status,
                last_successful_call=(
                    datetime.now() if status == LLMHealthStatus.HEALTHY else None
                ),
                consecutive_failures=0 if status == LLMHealthStatus.HEALTHY else 1,
                average_response_time_ms=response_time,
                total_calls=1,
                successful_calls=1 if status == LLMHealthStatus.HEALTHY else 0,
                failed_calls=0 if status == LLMHealthStatus.HEALTHY else 1,
                last_error=error,
                api_key_status="valid",
            )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            error_details = f"{type(e).__name__}: {str(e)}"

            return LLMProviderHealth(
                provider_name="perplexity",
                status=LLMHealthStatus.CRITICAL,
                last_successful_call=None,
                consecutive_failures=1,
                average_response_time_ms=response_time,
                total_calls=1,
                successful_calls=0,
                failed_calls=1,
                last_error=error_details,
                api_key_status="unknown",
            )


class ValidationGateEngine:
    """
    Implements mandatory validation gates for LLM system readiness
    Industry validation: No cognitive processing without LLM validation
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.provider_validator = LLMProviderValidator()
        self.enhancement_ratio_threshold = 0.90  # 90% LLM enhancement required

    async def execute_startup_validation_gates(self) -> LLMSystemHealth:
        """
        Execute comprehensive startup validation gates
        MANDATORY: Must pass before any cognitive processing

        Uses caching to avoid repeated validations within TTL window
        """
        global _GATE_CACHE

        # Get TTL from configuration
        TTL_MINUTES = int(os.getenv("LLM_GATE_TTL_MINUTES", 15))
        now = datetime.utcnow()

        # Check if we have a valid cached result
        if _GATE_CACHE["ts"] and now - _GATE_CACHE["ts"] < timedelta(
            minutes=TTL_MINUTES
        ):
            cached_result = _GATE_CACHE["result"]
            if getattr(cached_result, "system_ready", False):
                self.logger.debug(
                    f"🔒 Using cached validation result (age: {(now - _GATE_CACHE['ts']).total_seconds():.1f}s)"
                )
                return cached_result

        self.logger.info("🔒 Executing MANDATORY LLM validation gates...")

        gates = []
        critical_issues = []

        # Gate 1: API Key Validation
        gate_1 = await self._validate_api_keys_gate()
        gates.append(gate_1)
        if gate_1.result == ValidationGateResult.CRITICAL_FAIL:
            critical_issues.extend(gate_1.critical_issues)

        # Gate 2: Provider Connectivity
        gate_2 = await self._validate_provider_connectivity_gate()
        gates.append(gate_2)
        if gate_2.result == ValidationGateResult.CRITICAL_FAIL:
            critical_issues.extend(gate_2.critical_issues)

        # Gate 3: LLM Response Quality
        gate_3 = await self._validate_llm_response_quality_gate()
        gates.append(gate_3)
        if gate_3.result == ValidationGateResult.CRITICAL_FAIL:
            critical_issues.extend(gate_3.critical_issues)

        # Gate 4: End-to-End Cognitive Pipeline
        gate_4 = await self._validate_cognitive_pipeline_gate()
        gates.append(gate_4)
        if gate_4.result == ValidationGateResult.CRITICAL_FAIL:
            critical_issues.extend(gate_4.critical_issues)

        # Assess overall system health
        system_health = await self._assess_overall_system_health(gates, critical_issues)

        # Log results
        if system_health.system_ready:
            self.logger.info(
                "✅ LLM validation gates PASSED - System ready for cognitive processing"
            )
        else:
            self.logger.error("❌ LLM validation gates FAILED - System NOT ready")
            for issue in critical_issues:
                self.logger.error(f"   CRITICAL: {issue}")

        # Update cache with fresh result
        _GATE_CACHE.update({"ts": now, "result": system_health})
        self.logger.debug(f"🔒 Validation result cached (TTL: {TTL_MINUTES}min)")

        return system_health

    async def _validate_api_keys_gate(self) -> ValidationGateReport:
        """Gate 1: Validate all required API keys are present and valid"""
        start_time = time.time()

        try:
            import os

            issues = []
            recommendations = []
            critical_issues = []

            # Check Anthropic (Primary provider)
            anthropic_key = os.getenv("ANTHROPIC_API_KEY")
            if not anthropic_key:
                critical_issues.append(
                    "ANTHROPIC_API_KEY missing - Primary LLM provider unavailable"
                )
            elif len(anthropic_key) < 10:
                critical_issues.append("ANTHROPIC_API_KEY appears invalid (too short)")

            # Check Perplexity (Research provider)
            perplexity_key = os.getenv("PERPLEXITY_API_KEY")
            if not perplexity_key:
                recommendations.append(
                    "PERPLEXITY_API_KEY missing - Research enhancement unavailable"
                )

            # Check OpenAI (Alternative provider)
            openai_key = os.getenv("OPENAI_API_KEY")
            if not openai_key:
                recommendations.append(
                    "OPENAI_API_KEY missing - No fallback LLM provider"
                )

            execution_time = (time.time() - start_time) * 1000

            if critical_issues:
                result = ValidationGateResult.CRITICAL_FAIL
                details = (
                    f"API key validation failed: {len(critical_issues)} critical issues"
                )
            elif recommendations:
                result = ValidationGateResult.WARNING
                details = f"API keys partially configured: {len(recommendations)} recommendations"
            else:
                result = ValidationGateResult.PASS
                details = "All required API keys present and valid"

            return ValidationGateReport(
                gate_name="API_Key_Validation",
                result=result,
                execution_time_ms=execution_time,
                details=details,
                recommendations=recommendations,
                critical_issues=critical_issues,
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000

            return ValidationGateReport(
                gate_name="API_Key_Validation",
                result=ValidationGateResult.CRITICAL_FAIL,
                execution_time_ms=execution_time,
                details=f"API key validation system failure: {str(e)}",
                recommendations=["Check environment variable loading system"],
                critical_issues=[f"Validation system error: {str(e)}"],
            )

    async def _validate_provider_connectivity_gate(self) -> ValidationGateReport:
        """Gate 2: Validate LLM provider connectivity and response"""
        start_time = time.time()

        try:
            # Test all available providers
            anthropic_health = await self.provider_validator.validate_anthropic_health()
            openai_health = await self.provider_validator.validate_openai_health()
            perplexity_health = (
                await self.provider_validator.validate_perplexity_health()
            )

            execution_time = (time.time() - start_time) * 1000

            providers_tested = [anthropic_health, openai_health, perplexity_health]
            healthy_providers = [
                p for p in providers_tested if p.status == LLMHealthStatus.HEALTHY
            ]
            critical_providers = [
                p for p in providers_tested if p.status == LLMHealthStatus.CRITICAL
            ]

            critical_issues = []
            recommendations = []

            # Anthropic is critical
            if anthropic_health.status != LLMHealthStatus.HEALTHY:
                critical_issues.append(
                    f"Anthropic (primary provider) unhealthy: {anthropic_health.last_error}"
                )

            # Check if we have at least one healthy provider
            if len(healthy_providers) == 0:
                critical_issues.append("No healthy LLM providers available")
                result = ValidationGateResult.CRITICAL_FAIL
                details = (
                    "Complete LLM provider failure - no cognitive processing possible"
                )
            elif len(healthy_providers) == 1:
                result = ValidationGateResult.WARNING
                details = f"Only {healthy_providers[0].provider_name} available - no redundancy"
                recommendations.append(
                    "Configure additional LLM providers for redundancy"
                )
            else:
                result = ValidationGateResult.PASS
                details = f"{len(healthy_providers)} healthy providers available"

            return ValidationGateReport(
                gate_name="Provider_Connectivity",
                result=result,
                execution_time_ms=execution_time,
                details=details,
                recommendations=recommendations,
                critical_issues=critical_issues,
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000

            return ValidationGateReport(
                gate_name="Provider_Connectivity",
                result=ValidationGateResult.CRITICAL_FAIL,
                execution_time_ms=execution_time,
                details=f"Provider connectivity validation failed: {str(e)}",
                recommendations=["Check network connectivity and API endpoints"],
                critical_issues=[f"Connectivity validation error: {str(e)}"],
            )

    async def _validate_llm_response_quality_gate(self) -> ValidationGateReport:
        """Gate 3: Validate LLM response quality and JSON parsing"""
        start_time = time.time()

        try:
            from src.integrations.llm_provider import get_llm_client

            llm_client = get_llm_client()

            # Test structured JSON response
            test_result = await llm_client.analyze_problem_structure(
                problem_statement="Test problem for validation",
                business_context={"industry": "technology", "complexity": "low"},
            )

            execution_time = (time.time() - start_time) * 1000

            issues = []
            recommendations = []
            critical_issues = []

            # Validate response structure
            if not test_result.mental_models_selected:
                issues.append("Mental models selection empty")

            if not test_result.reasoning_description:
                issues.append("Reasoning description empty")

            if test_result.confidence_score <= 0:
                issues.append("Invalid confidence score")

            if "fallback" in test_result.reasoning_description.lower():
                critical_issues.append(
                    "LLM using fallback processing - indicates provider failure"
                )

            # Check for JSON parsing success
            if hasattr(test_result, "raw_response") and test_result.raw_response:
                try:
                    # Try to parse raw response as JSON to validate structure
                    import json

                    json.loads(test_result.raw_response)
                except json.JSONDecodeError:
                    issues.append("LLM response not valid JSON")

            if critical_issues:
                result = ValidationGateResult.CRITICAL_FAIL
                details = "LLM response quality validation failed"
            elif issues:
                result = ValidationGateResult.WARNING
                details = f"LLM response has {len(issues)} quality issues"
                recommendations.extend(issues)
            else:
                result = ValidationGateResult.PASS
                details = "LLM response quality validation passed"

            return ValidationGateReport(
                gate_name="LLM_Response_Quality",
                result=result,
                execution_time_ms=execution_time,
                details=details,
                recommendations=recommendations,
                critical_issues=critical_issues,
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000

            return ValidationGateReport(
                gate_name="LLM_Response_Quality",
                result=ValidationGateResult.CRITICAL_FAIL,
                execution_time_ms=execution_time,
                details=f"LLM response quality validation failed: {str(e)}",
                recommendations=[
                    "Check LLM provider configuration and response parsing"
                ],
                critical_issues=[f"Response quality validation error: {str(e)}"],
            )

    async def _validate_cognitive_pipeline_gate(self) -> ValidationGateReport:
        """Gate 4: Validate end-to-end cognitive processing pipeline"""
        start_time = time.time()

        try:
            # SURGICAL ORCHESTRATION - Use phantom-proof orchestrator for validation
            from src.core.consolidated_neural_lace_orchestrator import (
                get_consolidated_neural_lace_orchestrator,
            )
            from src.engine.models.data_contracts import (
                MetisDataContract,
                EngagementContext,
            )

            # Create minimal test contract using proper factory
            from src.engine.models.data_contracts import (
                create_engagement_initiated_event,
            )

            test_contract = create_engagement_initiated_event(
                problem_statement="Test cognitive pipeline validation",
                business_context={"industry": "technology", "research_required": False},
                client_name="Validation Test Client",
            )

            # Execute cognitive pipeline with surgical orchestrator
            surgical_orchestrator = await get_consolidated_neural_lace_orchestrator()
            result = await surgical_orchestrator.run_full_engagement(test_contract)

            execution_time = (time.time() - start_time) * 1000

            critical_issues = []
            recommendations = []

            # Validate pipeline execution
            if not hasattr(result, "cognitive_state") or not result.cognitive_state:
                critical_issues.append("Cognitive state not generated")

            if hasattr(result, "cognitive_state") and result.cognitive_state:
                if not result.cognitive_state.reasoning_steps:
                    critical_issues.append("No reasoning steps generated")

                if not result.cognitive_state.selected_mental_models:
                    critical_issues.append("No mental models selected")

                # Check for LLM enhancement
                llm_enhanced_steps = [
                    step
                    for step in result.cognitive_state.reasoning_steps
                    if getattr(step, "llm_enhanced", False)
                ]

                enhancement_ratio = (
                    len(llm_enhanced_steps)
                    / len(result.cognitive_state.reasoning_steps)
                    if result.cognitive_state.reasoning_steps
                    else 0
                )

                if enhancement_ratio < self.enhancement_ratio_threshold:
                    critical_issues.append(
                        f"LLM enhancement ratio {enhancement_ratio:.2%} below threshold {self.enhancement_ratio_threshold:.2%}"
                    )

            if critical_issues:
                result = ValidationGateResult.CRITICAL_FAIL
                details = "Cognitive pipeline validation failed"
            else:
                result = ValidationGateResult.PASS
                details = "Cognitive pipeline validation passed"

            return ValidationGateReport(
                gate_name="Cognitive_Pipeline",
                result=result,
                execution_time_ms=execution_time,
                details=details,
                recommendations=recommendations,
                critical_issues=critical_issues,
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000

            return ValidationGateReport(
                gate_name="Cognitive_Pipeline",
                result=ValidationGateResult.CRITICAL_FAIL,
                execution_time_ms=execution_time,
                details=f"Cognitive pipeline validation failed: {str(e)}",
                recommendations=[
                    "Check workflow engine configuration and LLM integration"
                ],
                critical_issues=[f"Pipeline validation error: {str(e)}"],
            )

    async def _assess_overall_system_health(
        self, gates: List[ValidationGateReport], critical_issues: List[str]
    ) -> LLMSystemHealth:
        """Assess overall LLM system health based on validation gates"""

        # Get provider health
        anthropic_health = await self.provider_validator.validate_anthropic_health()
        openai_health = await self.provider_validator.validate_openai_health()
        perplexity_health = await self.provider_validator.validate_perplexity_health()

        provider_health = {
            "anthropic": anthropic_health,
            "openai": openai_health,
            "perplexity": perplexity_health,
        }

        # Determine overall status
        critical_failures = [
            g for g in gates if g.result == ValidationGateResult.CRITICAL_FAIL
        ]
        warnings = [g for g in gates if g.result == ValidationGateResult.WARNING]

        if critical_failures:
            overall_status = LLMHealthStatus.CRITICAL
            system_ready = False
        elif len(warnings) > 2:
            overall_status = LLMHealthStatus.DEGRADED
            system_ready = False
        elif warnings:
            overall_status = LLMHealthStatus.DEGRADED
            system_ready = True  # Can operate with warnings
        else:
            overall_status = LLMHealthStatus.HEALTHY
            system_ready = True

        # Calculate enhancement ratio
        enhancement_ratio = 0.0
        if anthropic_health.status == LLMHealthStatus.HEALTHY:
            enhancement_ratio = 1.0
        elif openai_health.status == LLMHealthStatus.HEALTHY:
            enhancement_ratio = 0.8
        else:
            enhancement_ratio = 0.0

        return LLMSystemHealth(
            overall_status=overall_status,
            provider_health=provider_health,
            validation_gates=gates,
            enhancement_ratio=enhancement_ratio,
            system_ready=system_ready,
            critical_alerts=critical_issues,
        )


# Global validation gate engine
_validation_gate_engine: Optional[ValidationGateEngine] = None


def get_validation_gate_engine() -> ValidationGateEngine:
    """Get or create global validation gate engine"""
    global _validation_gate_engine

    if _validation_gate_engine is None:
        _validation_gate_engine = ValidationGateEngine()

    return _validation_gate_engine


# Mandatory validation decorator
def require_llm_validation(func):
    """
    Decorator to enforce LLM validation before cognitive processing
    Industry insight: No processing without validation
    """

    async def wrapper(*args, **kwargs):
        validation_engine = get_validation_gate_engine()

        # Execute validation gates
        system_health = await validation_engine.execute_startup_validation_gates()

        if not system_health.system_ready:
            raise Exception(
                f"LLM validation gates failed - cognitive processing blocked. "
                f"Critical issues: {'; '.join(system_health.critical_alerts)}"
            )

        # Proceed with original function
        return await func(*args, **kwargs)

    return wrapper


# Health monitoring helper
async def get_llm_system_health() -> LLMSystemHealth:
    """Get current LLM system health status"""
    validation_engine = get_validation_gate_engine()
    return await validation_engine.execute_startup_validation_gates()


# Emergency diagnostic function
async def diagnose_zero_llm_processing() -> Dict[str, Any]:
    """
    Emergency diagnostic for Zero LLM Processing Issue
    Provides detailed analysis of LLM system failures
    """

    logger = logging.getLogger(__name__)
    logger.info("🚨 EMERGENCY DIAGNOSTIC: Zero LLM Processing Issue")

    diagnostic_report = {
        "timestamp": datetime.now().isoformat(),
        "issue": "Zero LLM Processing",
        "diagnostic_steps": [],
        "findings": [],
        "resolution_recommendations": [],
    }

    try:
        # Step 1: Environment validation
        logger.info("🔍 Step 1: Environment validation")
        import os

        env_check = {
            "anthropic_key_present": bool(os.getenv("ANTHROPIC_API_KEY")),
            "anthropic_key_length": len(os.getenv("ANTHROPIC_API_KEY", "")),
            "perplexity_key_present": bool(os.getenv("PERPLEXITY_API_KEY")),
            "openai_key_present": bool(os.getenv("OPENAI_API_KEY")),
        }

        diagnostic_report["diagnostic_steps"].append("Environment variable check")
        diagnostic_report["findings"].append(f"Environment check: {env_check}")

        # Step 2: Provider initialization
        logger.info("🔍 Step 2: Provider initialization")
        from src.integrations.llm_provider import get_llm_client

        llm_client = get_llm_client()
        available_providers = llm_client.get_available_providers()

        diagnostic_report["diagnostic_steps"].append("Provider initialization check")
        diagnostic_report["findings"].append(
            f"Available providers: {available_providers}"
        )

        # Step 3: Validation gates
        logger.info("🔍 Step 3: Validation gates execution")
        validation_engine = get_validation_gate_engine()
        system_health = await validation_engine.execute_startup_validation_gates()

        diagnostic_report["diagnostic_steps"].append("Validation gates execution")
        diagnostic_report["findings"].append(
            {
                "overall_status": system_health.overall_status.value,
                "system_ready": system_health.system_ready,
                "enhancement_ratio": system_health.enhancement_ratio,
                "critical_alerts": system_health.critical_alerts,
            }
        )

        # Step 4: Resolution recommendations
        if not system_health.system_ready:
            diagnostic_report["resolution_recommendations"].extend(
                [
                    "1. Check API keys in .env file are valid and correctly formatted",
                    "2. Verify network connectivity to LLM providers",
                    "3. Restart application after fixing API key issues",
                    "4. Check application logs for specific error details",
                    "5. Validate LLM provider service status",
                ]
            )
        else:
            diagnostic_report["resolution_recommendations"].append(
                "LLM system appears healthy - check workflow engine integration"
            )

        return diagnostic_report

    except Exception as e:
        diagnostic_report["diagnostic_steps"].append("Emergency diagnostic failure")
        diagnostic_report["findings"].append(f"Diagnostic system error: {str(e)}")
        diagnostic_report["resolution_recommendations"].extend(
            [
                "CRITICAL: Diagnostic system failure",
                "1. Check Python environment and dependencies",
                "2. Verify METIS installation integrity",
                "3. Check system-level configuration",
                "4. Consider full system restart",
            ]
        )

        return diagnostic_report
