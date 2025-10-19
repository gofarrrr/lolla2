#!/usr/bin/env python3
"""
Template Manager for DeepSeek V3.1 Prompting Strategies
Unified interface for research-backed prompting optimization
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from .base_template import BasePromptTemplate, PromptContext
from .chain_of_draft import ChainOfDraftTemplate
from .zero_shot_optimized import ZeroShotOptimizedTemplate
from .self_correction import SelfCorrectionTemplate
from .direct import DirectTemplate
from .direct_minimal import DirectMinimalTemplate


class PromptTemplateManager:
    """
    Manages prompt templates and provides unified optimization interface
    Implements research-backed strategy selection for optimal performance
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Initialize all available templates
        self.templates = {
            "chain_of_draft": ChainOfDraftTemplate(),
            "zero_shot_optimized": ZeroShotOptimizedTemplate(),
            "self_correction": SelfCorrectionTemplate(),
            "direct": DirectTemplate(),
            "direct_minimal": DirectMinimalTemplate(),
        }

        # Strategy selection statistics
        self.selection_stats = {
            "total_selections": 0,
            "strategy_usage": {strategy: 0 for strategy in self.templates.keys()},
            "performance_tracking": {},
            "selection_history": [],
        }

        self.logger.info(
            "🎯 PromptTemplateManager initialized with research-backed strategies"
        )

    def generate_optimized_prompt(
        self, original_prompt: str, strategy: str, context: PromptContext
    ) -> str:
        """
        Generate optimized prompt using specified strategy

        Args:
            original_prompt: The original user prompt
            strategy: Strategy name ("chain_of_draft", "zero_shot_optimized", etc.)
            context: Context information for optimization

        Returns:
            Optimized prompt string
        """

        if strategy not in self.templates:
            raise ValueError(
                f"Unknown strategy: {strategy}. Available: {list(self.templates.keys())}"
            )

        template = self.templates[strategy]

        # Generate optimized prompt
        start_time = datetime.now()

        try:
            optimized_prompt = template.generate_prompt(original_prompt, context)

            # Track successful selection
            self._track_selection(strategy, context, success=True)

            generation_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(
                f"✅ Generated {strategy} prompt in {generation_time:.3f}s"
            )

            return optimized_prompt

        except Exception as e:
            self.logger.error(f"❌ Failed to generate {strategy} prompt: {e}")
            self._track_selection(strategy, context, success=False, error=str(e))
            raise

    def select_optimal_strategy(
        self, original_prompt: str, context: PromptContext
    ) -> str:
        """
        Automatically select optimal prompting strategy based on context

        Args:
            original_prompt: The original user prompt
            context: Context information for strategy selection

        Returns:
            Selected strategy name
        """

        # Priority 1: Check if any template explicitly declares itself optimal
        for strategy_name, template in self.templates.items():
            if hasattr(
                template, "is_optimal_for_context"
            ) and template.is_optimal_for_context(context):
                self.logger.info(f"🎯 {strategy_name} explicitly optimal for context")
                return strategy_name

        # Priority 2: Rule-based strategy selection based on research findings
        selected_strategy = self._apply_selection_rules(context)

        self.logger.info(f"🎯 Selected {selected_strategy} via rule-based selection")
        return selected_strategy

    def _apply_selection_rules(self, context: PromptContext) -> str:
        """Apply research-backed selection rules"""

        # Ultra-complex tasks -> Chain-of-Draft for 60% speed improvement
        if context.complexity_score > 0.8 or context.task_type in [
            "multi_model_synthesis",
            "strategic_inversion_analysis",
            "assumption_network_analysis",
            "competitive_dynamics_modeling",
        ]:
            return "chain_of_draft"

        # Urgent tasks -> Direct for maximum speed
        if context.time_constraints == "urgent" or context.complexity_score <= 0.3:
            return "direct"

        # Business-critical or high-accuracy -> Self-correction
        if context.quality_threshold > 0.85 or (
            context.additional_context
            and context.additional_context.get("business_critical", False)
        ):
            return "self_correction"

        # Standard complex tasks -> Zero-shot optimized
        return "zero_shot_optimized"

    def generate_with_auto_selection(
        self, original_prompt: str, context: PromptContext
    ) -> Dict[str, Any]:
        """
        Generate optimized prompt with automatic strategy selection

        Returns:
            Dict containing optimized prompt, strategy used, and metadata
        """

        # Select optimal strategy
        selected_strategy = self.select_optimal_strategy(original_prompt, context)

        # Generate optimized prompt
        optimized_prompt = self.generate_optimized_prompt(
            original_prompt, selected_strategy, context
        )

        # Get strategy metadata
        template = self.templates[selected_strategy]
        expected_improvements = template.get_expected_performance_improvement()
        template_info = template.get_template_info()

        return {
            "optimized_prompt": optimized_prompt,
            "strategy_used": selected_strategy,
            "original_prompt": original_prompt,
            "context": context,
            "expected_improvements": expected_improvements,
            "template_info": template_info,
            "selection_reasoning": self._explain_selection(selected_strategy, context),
            "generation_timestamp": datetime.now(),
        }

    def _explain_selection(self, strategy: str, context: PromptContext) -> str:
        """Explain why a particular strategy was selected"""

        explanations = {
            "chain_of_draft": f"Selected for ultra-complex task (complexity: {context.complexity_score:.2f}) - "
            f"research shows 60.7% speed improvement",
            "zero_shot_optimized": f"Selected for standard complex task (complexity: {context.complexity_score:.2f}) - "
            f"research shows perfect quality scores",
            "self_correction": f"Selected for high-accuracy requirements (quality threshold: {context.quality_threshold:.2f}) - "
            f"research shows excellent error detection",
            "direct": f"Selected for fast response (complexity: {context.complexity_score:.2f}, "
            f"time: {context.time_constraints}) - research shows 5.3x speed advantage",
        }

        return explanations.get(
            strategy, f"Selected {strategy} based on context analysis"
        )

    def get_strategy_recommendations(
        self, context: PromptContext
    ) -> Dict[str, Dict[str, Any]]:
        """Get recommendations for all strategies with suitability scores"""

        recommendations = {}

        for strategy_name, template in self.templates.items():
            suitability_score = self._calculate_suitability_score(template, context)
            expected_improvements = template.get_expected_performance_improvement()

            recommendations[strategy_name] = {
                "suitability_score": suitability_score,
                "expected_improvements": expected_improvements,
                "description": template.get_template_info()["name"],
                "optimal_for": template.performance_characteristics.get(
                    "optimal_for", []
                ),
                "recommendation": self._get_strategy_recommendation(suitability_score),
            }

        return recommendations

    def _calculate_suitability_score(
        self, template: BasePromptTemplate, context: PromptContext
    ) -> float:
        """Calculate suitability score for a template given context"""

        base_score = 0.5

        # Check explicit optimization flags
        if hasattr(template, "is_optimal_for_context"):
            if template.is_optimal_for_context(context):
                base_score += 0.4

        # Task type compatibility
        optimal_for = template.performance_characteristics.get("optimal_for", [])

        if context.task_type in optimal_for:
            base_score += 0.3

        # Complexity matching
        if "ultra_complex" in optimal_for and context.complexity_score > 0.8:
            base_score += 0.2
        elif (
            "standard_complex" in optimal_for and 0.4 <= context.complexity_score <= 0.8
        ):
            base_score += 0.2
        elif "fast_response" in optimal_for and context.complexity_score <= 0.4:
            base_score += 0.2

        # Time constraint matching
        if context.time_constraints == "urgent" and "fast_response" in optimal_for:
            base_score += 0.3
        elif context.time_constraints == "thorough" and "ultra_complex" in optimal_for:
            base_score += 0.2

        # Quality requirement matching
        if context.quality_threshold > 0.8 and "high_quality" in optimal_for:
            base_score += 0.2

        return min(1.0, base_score)

    def _get_strategy_recommendation(self, suitability_score: float) -> str:
        """Get recommendation text based on suitability score"""

        if suitability_score >= 0.8:
            return "Highly Recommended"
        elif suitability_score >= 0.6:
            return "Recommended"
        elif suitability_score >= 0.4:
            return "Suitable"
        else:
            return "Not Recommended"

    def _track_selection(
        self,
        strategy: str,
        context: PromptContext,
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        """Track strategy selection for analytics"""

        self.selection_stats["total_selections"] += 1
        self.selection_stats["strategy_usage"][strategy] += 1

        # Track selection details
        selection_record = {
            "timestamp": datetime.now(),
            "strategy": strategy,
            "task_type": context.task_type,
            "complexity": context.complexity_score,
            "time_constraints": context.time_constraints,
            "quality_threshold": context.quality_threshold,
            "success": success,
            "error": error,
        }

        self.selection_stats["selection_history"].append(selection_record)

        # Keep only recent history (last 1000 selections)
        if len(self.selection_stats["selection_history"]) > 1000:
            self.selection_stats["selection_history"].pop(0)

    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""

        stats = self.selection_stats.copy()

        # Calculate usage percentages
        total_selections = self.selection_stats["total_selections"]
        if total_selections > 0:
            stats["strategy_usage_percentages"] = {
                strategy: (count / total_selections) * 100
                for strategy, count in self.selection_stats["strategy_usage"].items()
            }

        # Calculate success rates by strategy
        success_rates = {}
        for strategy in self.templates.keys():
            strategy_selections = [
                record
                for record in self.selection_stats["selection_history"]
                if record["strategy"] == strategy
            ]

            if strategy_selections:
                successful = sum(
                    1 for record in strategy_selections if record["success"]
                )
                success_rates[strategy] = successful / len(strategy_selections)

        stats["success_rates"] = success_rates

        return stats

    def reset_statistics(self) -> None:
        """Reset performance statistics"""
        self.selection_stats = {
            "total_selections": 0,
            "strategy_usage": {strategy: 0 for strategy in self.templates.keys()},
            "performance_tracking": {},
            "selection_history": [],
        }

        self.logger.info("📊 Template manager statistics reset")


# Global instance
_template_manager = None


def get_prompt_template_manager() -> PromptTemplateManager:
    """Get the global PromptTemplateManager instance"""
    global _template_manager
    if _template_manager is None:
        _template_manager = PromptTemplateManager()
    return _template_manager
