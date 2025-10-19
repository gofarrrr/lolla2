#!/usr/bin/env python3
"""
LLM Integration Adapter - Bridges ResilientLLMClient with existing components
Provides backward compatibility while enabling Afterburner optimization
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

from src.core.resilient_llm_client import (
    get_resilient_llm_client,
    CognitiveCallContext,
    LLMCallResult,
)
from src.integrations.claude_client import LLMCallType


@dataclass
class TaskContextMapper:
    """Maps existing task patterns to CognitiveCallContext for optimal routing"""

    # Task type mappings from existing patterns
    TASK_TYPE_MAP = {
        "systems_thinking": "strategic_synthesis",
        "critical_thinking": "problem_analysis",
        "mece_structuring": "strategic_synthesis",
        "hypothesis_testing": "hypothesis_testing",
        "decision_framework": "strategic_synthesis",
        "assumption_challenge": "assumption_challenge",
        "hmw_generation": "creative_thinking",
        "challenge_generation": "challenge_generation",
        "reasoning_synthesis": "strategic_synthesis",
        "vulnerability_analysis": "risk_cascade_analysis",
        "opportunity_identification": "opportunity_cost_optimization",
    }

    # Complexity scoring based on task characteristics
    COMPLEXITY_INDICATORS = {
        "multi_stakeholder": 0.2,
        "historical_analysis": 0.15,
        "cross_industry": 0.1,
        "strategic_implications": 0.15,
        "long_term_planning": 0.1,
        "risk_assessment": 0.1,
        "innovation_required": 0.15,
        "data_synthesis": 0.05,
    }

    @classmethod
    def map_to_cognitive_context(
        cls,
        task_name: str,
        business_context: Optional[Dict[str, Any]] = None,
        nway_pattern: Optional[Dict[str, Any]] = None,
        engagement_id: Optional[str] = None,
        phase: Optional[str] = None,
    ) -> CognitiveCallContext:
        """Map existing task patterns to optimized CognitiveCallContext"""

        # Determine task type
        task_type = cls.TASK_TYPE_MAP.get(task_name, "general")

        # Calculate complexity score
        complexity_score = cls._calculate_complexity(
            task_name, business_context, nway_pattern
        )

        # Determine time constraints
        time_constraints = cls._determine_time_constraints(
            business_context, nway_pattern
        )

        # Set quality threshold based on task importance
        quality_threshold = cls._determine_quality_threshold(
            task_name, business_context
        )

        # Determine cost sensitivity
        cost_sensitivity = cls._determine_cost_sensitivity(
            business_context, quality_threshold
        )

        return CognitiveCallContext(
            engagement_id=engagement_id,
            phase=phase or task_name,
            task_type=task_type,
            complexity_score=complexity_score,
            time_constraints=time_constraints,
            quality_threshold=quality_threshold,
            cost_sensitivity=cost_sensitivity,
        )

    @classmethod
    def _calculate_complexity(
        cls,
        task_name: str,
        business_context: Optional[Dict[str, Any]],
        nway_pattern: Optional[Dict[str, Any]],
    ) -> float:
        """Calculate complexity score based on multiple factors"""

        base_complexity = 0.5  # Default medium complexity

        # Task-specific base complexity
        if task_name in ["vulnerability_analysis", "opportunity_identification"]:
            base_complexity = 0.8
        elif task_name in ["systems_thinking", "strategic_synthesis"]:
            base_complexity = 0.7
        elif task_name in ["critical_thinking", "assumption_challenge"]:
            base_complexity = 0.6
        elif task_name in ["hmw_generation", "query_generation"]:
            base_complexity = 0.3

        # Adjust based on context indicators
        if business_context:
            for indicator, weight in cls.COMPLEXITY_INDICATORS.items():
                if indicator in str(business_context).lower():
                    base_complexity += weight

        # N-WAY pattern increases complexity
        if nway_pattern and "interaction_depth" in nway_pattern:
            depth = nway_pattern.get("interaction_depth", 1)
            base_complexity += min(0.2, depth * 0.05)

        return min(1.0, base_complexity)

    @classmethod
    def _determine_time_constraints(
        cls,
        business_context: Optional[Dict[str, Any]],
        nway_pattern: Optional[Dict[str, Any]],
    ) -> str:
        """Determine time constraints from context"""

        if business_context:
            if "urgent" in str(business_context).lower():
                return "urgent"
            if "thorough" in str(business_context).lower():
                return "thorough"

        if nway_pattern and nway_pattern.get("time_sensitive"):
            return "urgent"

        return "normal"

    @classmethod
    def _determine_quality_threshold(
        cls, task_name: str, business_context: Optional[Dict[str, Any]]
    ) -> float:
        """Determine quality requirements"""

        # High quality for critical tasks
        if task_name in [
            "vulnerability_analysis",
            "assumption_challenge",
            "strategic_synthesis",
            "risk_assessment",
        ]:
            base_threshold = 0.85
        else:
            base_threshold = 0.7

        # Adjust for business criticality
        if business_context and "critical" in str(business_context).lower():
            base_threshold = min(0.95, base_threshold + 0.1)

        return base_threshold

    @classmethod
    def _determine_cost_sensitivity(
        cls, business_context: Optional[Dict[str, Any]], quality_threshold: float
    ) -> str:
        """Determine cost sensitivity based on context"""

        # High quality requirements suggest low cost sensitivity
        if quality_threshold >= 0.85:
            return "low"

        # Check context for cost indicators
        if business_context:
            context_str = str(business_context).lower()
            if "budget" in context_str or "cost" in context_str:
                return "high"
            if "premium" in context_str or "critical" in context_str:
                return "low"

        return "normal"


class UnifiedLLMAdapter:
    """
    Unified adapter for LLM calls with Afterburner optimization
    Provides drop-in replacement for existing LLM call patterns
    """

    def __init__(self, enable_afterburner: bool = True):
        """
        Initialize the unified LLM adapter

        Args:
            enable_afterburner: Enable Afterburner optimization (can be disabled for testing)
        """
        self.logger = logging.getLogger(__name__)
        self.enable_afterburner = enable_afterburner
        self.resilient_client = None
        self.context_mapper = TaskContextMapper()
        self._initialized = False

    async def initialize(self):
        """Initialize the resilient client"""
        if not self._initialized:
            self.resilient_client = get_resilient_llm_client()
            self._initialized = True
            self.logger.info(
                f"🚀 UnifiedLLMAdapter initialized with Afterburner: "
                f"{'ENABLED' if self.enable_afterburner else 'DISABLED'}"
            )

    async def call_llm_unified(
        self,
        prompt: str,
        task_name: str = "general",
        business_context: Optional[Dict[str, Any]] = None,
        nway_pattern: Optional[Dict[str, Any]] = None,
        engagement_id: Optional[str] = None,
        phase: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: float = 0.3,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Unified LLM call with automatic Afterburner optimization

        This method replaces all direct claude.call_claude() and self._call_deepseek() calls

        Args:
            prompt: The prompt to send
            task_name: Name of the task (e.g., 'systems_thinking', 'critical_thinking')
            business_context: Business context dictionary
            nway_pattern: N-WAY pattern context if available
            engagement_id: Current engagement ID
            phase: Current phase of processing
            max_tokens: Maximum tokens for response
            temperature: Temperature for generation
            system_prompt: Optional system prompt (will be integrated into main prompt)

        Returns:
            String response content with optimizations applied
        """

        # Ensure initialization
        if not self._initialized:
            await self.initialize()

        # Combine system prompt with main prompt if provided
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt

        if self.enable_afterburner:
            # Map to cognitive context for optimal routing
            cognitive_context = self.context_mapper.map_to_cognitive_context(
                task_name=task_name,
                business_context=business_context,
                nway_pattern=nway_pattern,
                engagement_id=engagement_id,
                phase=phase,
            )

            # Override temperature and max_tokens in context if significantly different
            if abs(temperature - 0.3) > 0.2:
                cognitive_context.complexity_score = max(
                    0.3, cognitive_context.complexity_score - 0.1
                )

            try:
                # Execute with Afterburner optimization
                result: LLMCallResult = (
                    await self.resilient_client.execute_cognitive_call(
                        prompt=full_prompt, context=cognitive_context
                    )
                )

                # Log optimization details
                if result.fallback_triggered:
                    self.logger.warning(
                        f"⚠️ Fallback triggered for {task_name}: {result.warnings}"
                    )
                else:
                    self.logger.info(
                        f"✅ Afterburner optimization successful for {task_name}: "
                        f"{result.provider_used}/{result.model_used} "
                        f"({result.response_time_ms}ms, ${result.cost_usd:.4f})"
                    )

                return result.content

            except Exception as e:
                self.logger.error(f"❌ Afterburner call failed: {e}")
                # Could implement additional fallback here
                raise

        else:
            # Legacy mode - direct call without optimization
            # This path maintains backward compatibility
            self.logger.warning(
                f"⚠️ Afterburner disabled - using legacy LLM call for {task_name}"
            )

            # Import legacy clients
            from src.integrations.claude_client import get_claude_client

            try:
                claude = await get_claude_client()
                response = await claude.call_claude(
                    prompt=full_prompt,
                    call_type=LLMCallType.MENTAL_MODEL,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return response.content
            except Exception as e:
                self.logger.error(f"Legacy LLM call failed: {e}")
                raise

    async def get_optimization_stats(self) -> Dict[str, Any]:
        """Get Afterburner optimization statistics"""
        if self.resilient_client:
            return self.resilient_client.get_provider_statistics()
        return {"status": "not_initialized"}

    async def health_check(self) -> Dict[str, bool]:
        """Check health of all LLM providers"""
        if not self._initialized:
            await self.initialize()
        return await self.resilient_client.health_check()


# Global adapter instance
_global_adapter = None


def get_unified_llm_adapter(enable_afterburner: bool = True) -> UnifiedLLMAdapter:
    """
    Get the global unified LLM adapter instance

    Args:
        enable_afterburner: Enable Afterburner optimization (default: True)

    Returns:
        UnifiedLLMAdapter instance
    """
    global _global_adapter
    if _global_adapter is None:
        _global_adapter = UnifiedLLMAdapter(enable_afterburner=enable_afterburner)
    return _global_adapter


# Backward compatibility helpers
async def call_llm_legacy_compatible(prompt: str, **kwargs) -> str:
    """
    Legacy-compatible LLM call that automatically uses Afterburner
    Drop-in replacement for existing LLM calls
    """
    adapter = get_unified_llm_adapter()
    return await adapter.call_llm_unified(prompt, **kwargs)
