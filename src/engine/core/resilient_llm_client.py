#!/usr/bin/env python3
"""
ResilientLLMClient - DeepSeek-First Architecture
Unified LLM client with intelligent provider selection and fallback mechanism
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from src.engine import config as metis_config
from src.engine.integrations.llm.deepseek_provider import DeepSeekProvider
from src.integrations.claude_client import ClaudeClient, LLMCallType
from src.intelligence.optimized_deepseek_router import get_optimized_deepseek_router
from src.engine.templates.prompting.template_manager import PromptTemplateManager
from src.engine.templates.prompting.base_template import PromptContext


@dataclass
class CognitiveCallContext:
    """Context information for cognitive LLM calls"""

    engagement_id: Optional[str] = None
    phase: Optional[str] = None
    task_type: Optional[str] = None
    complexity_score: float = 0.5
    time_constraints: str = "normal"
    quality_threshold: float = 0.8
    cost_sensitivity: str = "normal"  # "low", "normal", "high"


@dataclass
class LLMCallResult:
    """Result of an LLM call with provider metadata"""

    content: str
    provider_used: str
    model_used: str
    tokens_used: int
    cost_usd: float
    response_time_ms: int
    confidence: float
    reasoning_steps: List[Dict[str, Any]]
    fallback_triggered: bool = False
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class AllProvidersFailedError(Exception):
    """Raised when all configured LLM providers have failed"""

    pass


class ResilientLLMClient:
    """
    Unified LLM client implementing DeepSeek-First architecture with Claude fallback
    Abstracts provider selection and provides intelligent failover
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Initialize providers
        self.deepseek_provider = None
        self.claude_provider = None

        # Initialize optimization components
        self.router = get_optimized_deepseek_router()
        self.template_manager = PromptTemplateManager()

        # Performance tracking
        self.provider_stats = {
            "deepseek": {"calls": 0, "failures": 0, "avg_cost": 0.0, "avg_time": 0.0},
            "claude": {"calls": 0, "failures": 0, "avg_cost": 0.0, "avg_time": 0.0},
        }

        # Initialize providers
        self._initialize_providers()

        self.logger.info(
            f"🚀 ResilientLLMClient initialized: {metis_config.PRIMARY_PROVIDER} → {metis_config.FALLBACK_PROVIDER} with Afterburner optimization"
        )

    def _initialize_providers(self):
        """Initialize LLM providers"""
        try:
            # Initialize DeepSeek provider
            import os

            deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
            if deepseek_api_key:
                self.deepseek_provider = DeepSeekProvider(api_key=deepseek_api_key)
                self.logger.info("✅ DeepSeek V3.1 provider initialized")
            else:
                self.logger.warning("⚠️ DEEPSEEK_API_KEY not found")

            # Initialize Claude provider
            self.claude_provider = ClaudeClient()
            self.logger.info("✅ Claude provider initialized")

        except Exception as e:
            self.logger.error(f"❌ Provider initialization error: {e}")

    async def execute_cognitive_call(
        self, prompt: str, context: CognitiveCallContext
    ) -> LLMCallResult:
        """
        Execute a cognitive call with Afterburner optimization and intelligent fallback

        Args:
            prompt: The prompt to send to the LLM
            context: Context information for provider selection and call optimization

        Returns:
            LLMCallResult with provider metadata and warnings
        """

        start_time = time.time()

        # Step 1: Route task with Afterburner optimization
        routing_plan = self.router.route_task(
            task_type=context.task_type or "general",
            complexity_score=context.complexity_score,
            time_constraints=context.time_constraints,
            context_data={
                "engagement_id": context.engagement_id,
                "phase": context.phase,
                "quality_threshold": context.quality_threshold,
                "business_critical": context.cost_sensitivity == "low",
            },
        )

        self.logger.info(
            f"🎯 Afterburner routing: {routing_plan['strategy']} strategy, {routing_plan['timeout']}s timeout"
        )

        # Step 2: Generate optimized prompt using template system
        prompt_context = PromptContext(
            task_type=context.task_type or "general",
            complexity_score=context.complexity_score,
            time_constraints=context.time_constraints,
            engagement_id=context.engagement_id,
            phase=context.phase,
            quality_threshold=context.quality_threshold,
            additional_context={"routing_plan": routing_plan},
        )

        optimized_prompt_data = self.template_manager.generate_with_auto_selection(
            prompt, prompt_context
        )
        optimized_prompt = optimized_prompt_data["optimized_prompt"]

        self.logger.info(
            f"🧠 Using {optimized_prompt_data['strategy_used']} prompting strategy"
        )

        # Step 3: Attempt PRIMARY provider (DeepSeek) with optimization
        if metis_config.PRIMARY_PROVIDER == "deepseek" and self.deepseek_provider:
            try:
                self.logger.info(
                    f"🚀 Attempting optimized PRIMARY call: {metis_config.PRIMARY_PROVIDER}"
                )

                result = await self._call_deepseek_optimized(
                    optimized_prompt, context, routing_plan
                )

                # Track successful primary call
                self._update_provider_stats(
                    "deepseek", result.cost_usd, result.response_time_ms, success=True
                )

                result.fallback_triggered = False

                # Add optimization metadata to warnings (as informational)
                if routing_plan.get("config_profile") == "ultra_complex":
                    result.warnings.append(
                        f"Afterburner optimization active: {optimized_prompt_data['strategy_used']} strategy "
                        f"with {routing_plan['timeout']}s timeout for maximum performance"
                    )

                return result

            except Exception as e:
                self.logger.error(
                    f"❌ PRIMARY provider '{metis_config.PRIMARY_PROVIDER}' failed: {e}"
                )

                # Track primary failure
                self._update_provider_stats(
                    "deepseek", 0, time.time() - start_time, success=False
                )

                # Continue to fallback with original prompt (Claude doesn't use our templates)

        # Step 4: PRIMARY failed, trigger FALLBACK (Claude)
        if metis_config.FALLBACK_PROVIDER == "claude" and self.claude_provider:
            try:
                self.logger.warning(
                    f"🔄 FALLBACK triggered: Using {metis_config.FALLBACK_PROVIDER}"
                )

                result = await self._call_claude(
                    prompt, context
                )  # Use original prompt for Claude

                # Track successful fallback call
                self._update_provider_stats(
                    "claude", result.cost_usd, result.response_time_ms, success=True
                )

                # Mark as fallback and add cost warning
                result.fallback_triggered = True
                result.warnings.append(
                    f"Primary LLM failed. Switched to high-cost fallback provider: {metis_config.FALLBACK_PROVIDER}. "
                    f"Estimated cost increase: {self._estimate_cost_increase()}%"
                )

                return result

            except Exception as e:
                self.logger.error(
                    f"❌ FALLBACK provider '{metis_config.FALLBACK_PROVIDER}' also failed: {e}"
                )

                # Track fallback failure
                self._update_provider_stats(
                    "claude", 0, time.time() - start_time, success=False
                )

        # Step 5: All providers failed
        raise AllProvidersFailedError(
            f"All configured LLM providers have failed. "
            f"Primary: {metis_config.PRIMARY_PROVIDER}, Fallback: {metis_config.FALLBACK_PROVIDER}"
        )

    async def _call_deepseek_optimized(
        self,
        optimized_prompt: str,
        context: CognitiveCallContext,
        routing_plan: Dict[str, Any],
    ) -> LLMCallResult:
        """Call DeepSeek with Afterburner optimization"""

        response = await self.deepseek_provider.call_optimized_llm(
            prompt=optimized_prompt,
            task_type=context.task_type or "general",
            complexity_score=context.complexity_score or 0.5,
            temperature=routing_plan["temperature"],
            max_tokens=routing_plan["max_tokens"],
        )

        return LLMCallResult(
            content=response.content,
            provider_used="deepseek",
            model_used=response.model,
            tokens_used=response.tokens_used,
            cost_usd=response.cost_usd,
            response_time_ms=response.response_time_ms,
            confidence=response.confidence,
            reasoning_steps=response.reasoning_steps,
            warnings=[],
        )

    async def _call_claude(
        self, prompt: str, context: CognitiveCallContext
    ) -> LLMCallResult:
        """Call Claude as fallback provider"""

        # Map task types to Claude call types
        call_type_mapping = {
            "challenge_generation": LLMCallType.MENTAL_MODEL,
            "assumption_challenge": LLMCallType.MENTAL_MODEL,
            "strategic_synthesis": LLMCallType.SYNTHESIS,
            "problem_analysis": LLMCallType.MENTAL_MODEL,
            "summary_generation": LLMCallType.SYNTHESIS,
        }

        call_type = call_type_mapping.get(context.task_type, LLMCallType.MENTAL_MODEL)

        response = await self.claude_provider.call_claude(
            prompt=prompt,
            call_type=call_type,
            max_tokens=self._get_optimal_max_tokens(context),
            temperature=self._get_optimal_temperature(context),
        )

        return LLMCallResult(
            content=response.content,
            provider_used="claude",
            model_used=response.model_version,
            tokens_used=response.tokens_used,
            cost_usd=response.cost_usd,
            response_time_ms=(
                response.response_time_ms
                if hasattr(response, "response_time_ms")
                else 0
            ),
            confidence=response.confidence,
            reasoning_steps=response.reasoning_steps,
            warnings=[],
        )

    def _get_optimal_temperature(self, context: CognitiveCallContext) -> float:
        """Get optimal temperature based on task type and quality requirements"""

        # Higher quality threshold = lower temperature for consistency
        base_temp = 0.7 - (context.quality_threshold * 0.4)

        # Task-specific adjustments
        task_adjustments = {
            "challenge_generation": -0.1,  # More focused for challenges
            "strategic_synthesis": -0.1,  # More consistent for synthesis
            "summary_generation": -0.2,  # Very consistent for summaries
            "creative_thinking": +0.2,  # More creative variation
        }

        adjustment = task_adjustments.get(context.task_type, 0.0)
        return max(0.1, min(0.9, base_temp + adjustment))

    def _get_optimal_max_tokens(self, context: CognitiveCallContext) -> int:
        """Get optimal max tokens based on task complexity and constraints"""

        base_tokens = 2000

        # Complexity adjustments
        complexity_multiplier = 0.5 + (context.complexity_score * 1.5)  # 0.5x to 2.0x

        # Task-specific adjustments
        task_multipliers = {
            "challenge_generation": 1.5,
            "strategic_synthesis": 2.0,
            "detailed_analysis": 2.5,
            "summary_generation": 0.5,
            "quick_insights": 0.3,
        }

        task_multiplier = task_multipliers.get(context.task_type, 1.0)

        # Time constraint adjustments
        time_multipliers = {"urgent": 0.7, "normal": 1.0, "thorough": 1.8}

        time_multiplier = time_multipliers.get(context.time_constraints, 1.0)

        optimal_tokens = int(
            base_tokens * complexity_multiplier * task_multiplier * time_multiplier
        )
        return max(500, min(4000, optimal_tokens))

    def _estimate_cost_increase(self) -> int:
        """Estimate cost increase percentage when using Claude fallback"""

        # DeepSeek V3.1 reasoner: ~$2.19 per 1M output tokens
        # Claude Sonnet 3.5: ~$15 per 1M output tokens
        # Approximate cost increase: ~585%
        return 585

    def _update_provider_stats(
        self, provider: str, cost: float, response_time: float, success: bool
    ):
        """Update provider performance statistics"""

        stats = self.provider_stats[provider]
        stats["calls"] += 1

        if success:
            # Update running averages
            old_calls = stats["calls"] - 1
            if old_calls > 0:
                stats["avg_cost"] = (stats["avg_cost"] * old_calls + cost) / stats[
                    "calls"
                ]
                stats["avg_time"] = (
                    stats["avg_time"] * old_calls + response_time
                ) / stats["calls"]
            else:
                stats["avg_cost"] = cost
                stats["avg_time"] = response_time
        else:
            stats["failures"] += 1

    def get_provider_statistics(self) -> Dict[str, Any]:
        """Get comprehensive provider performance statistics"""

        stats = {}

        for provider, data in self.provider_stats.items():
            if data["calls"] > 0:
                stats[provider] = {
                    "total_calls": data["calls"],
                    "failures": data["failures"],
                    "success_rate": (data["calls"] - data["failures"]) / data["calls"],
                    "avg_cost_usd": data["avg_cost"],
                    "avg_response_time_ms": data["avg_time"],
                }

        return {
            "provider_hierarchy": [
                metis_config.PRIMARY_PROVIDER,
                metis_config.FALLBACK_PROVIDER,
            ],
            "provider_stats": stats,
            "cost_efficiency": self._calculate_cost_efficiency(),
        }

    def _calculate_cost_efficiency(self) -> Dict[str, Any]:
        """Calculate cost efficiency metrics"""

        deepseek_stats = self.provider_stats["deepseek"]
        claude_stats = self.provider_stats["claude"]

        total_deepseek_calls = deepseek_stats["calls"] - deepseek_stats["failures"]
        total_claude_calls = claude_stats["calls"] - claude_stats["failures"]

        if total_deepseek_calls + total_claude_calls == 0:
            return {"status": "no_data"}

        # Calculate cost savings from using DeepSeek-First
        deepseek_cost = total_deepseek_calls * deepseek_stats["avg_cost"]
        claude_cost = total_claude_calls * claude_stats["avg_cost"]

        # Estimated cost if all calls were Claude
        estimated_claude_only_cost = (
            total_deepseek_calls + total_claude_calls
        ) * claude_stats["avg_cost"]

        actual_cost = deepseek_cost + claude_cost
        cost_savings = estimated_claude_only_cost - actual_cost
        savings_percentage = (
            (cost_savings / estimated_claude_only_cost) * 100
            if estimated_claude_only_cost > 0
            else 0
        )

        return {
            "actual_cost_usd": actual_cost,
            "estimated_claude_only_cost_usd": estimated_claude_only_cost,
            "cost_savings_usd": cost_savings,
            "cost_savings_percentage": savings_percentage,
            "deepseek_usage_rate": total_deepseek_calls
            / (total_deepseek_calls + total_claude_calls),
        }

    async def health_check(self) -> Dict[str, bool]:
        """Check availability of all providers"""

        health_status = {}

        # Check DeepSeek
        if self.deepseek_provider:
            try:
                health_status["deepseek"] = await asyncio.wait_for(
                    self.deepseek_provider.is_available(),
                    timeout=metis_config.PROVIDER_TIMEOUTS["deepseek"]["availability"],
                )
            except:
                health_status["deepseek"] = False
        else:
            health_status["deepseek"] = False

        # Check Claude
        if self.claude_provider:
            try:
                health_status["claude"] = await asyncio.wait_for(
                    self.claude_provider.is_available(),
                    timeout=metis_config.PROVIDER_TIMEOUTS["claude"]["availability"],
                )
            except:
                health_status["claude"] = False
        else:
            health_status["claude"] = False

        return health_status


# Global instance
_resilient_client = None


def get_resilient_llm_client() -> ResilientLLMClient:
    """Get the global ResilientLLMClient instance"""
    global _resilient_client
    if _resilient_client is None:
        _resilient_client = ResilientLLMClient()
    return _resilient_client
