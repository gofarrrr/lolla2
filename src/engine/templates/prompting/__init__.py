#!/usr/bin/env python3
"""
DeepSeek V3.1 Prompting Templates Package
Research-backed prompting optimization strategies for maximum performance
"""

from .base_template import BasePromptTemplate, PromptContext
from .chain_of_draft import ChainOfDraftTemplate
from .zero_shot_optimized import ZeroShotOptimizedTemplate
from .self_correction import SelfCorrectionTemplate
from .direct import DirectTemplate
from .template_manager import PromptTemplateManager, get_prompt_template_manager

__all__ = [
    # Base classes
    "BasePromptTemplate",
    "PromptContext",
    # Template implementations
    "ChainOfDraftTemplate",
    "ZeroShotOptimizedTemplate",
    "SelfCorrectionTemplate",
    "DirectTemplate",
    # Manager
    "PromptTemplateManager",
    "get_prompt_template_manager",
]

# Template performance summary from research
TEMPLATE_PERFORMANCE_SUMMARY = {
    "chain_of_draft": {
        "speed_improvement": "60.7%",
        "cost_reduction": "59%",
        "optimal_for": "ultra-complex tasks",
        "research_result": "Best overall performance for complex reasoning",
    },
    "zero_shot_optimized": {
        "speed_improvement": "30%",
        "quality_score": "1.00",
        "optimal_for": "standard complex tasks",
        "research_result": "Perfect quality scores with good speed",
    },
    "self_correction": {
        "speed_improvement": "50%",
        "quality_score": "0.86",
        "error_reduction": "92%",
        "optimal_for": "critical decisions",
        "research_result": "Excellent error detection and correction",
    },
    "direct": {
        "speed_advantage": "5.3x faster",
        "cost_advantage": "38x cheaper",
        "optimal_for": "fast response tasks",
        "research_result": "Maximum speed and cost efficiency",
    },
}

# Quick reference for strategy selection
STRATEGY_SELECTION_GUIDE = {
    "ultra_complex_tasks": "chain_of_draft",
    "time_critical": "direct",
    "high_accuracy_needed": "self_correction",
    "standard_analysis": "zero_shot_optimized",
}
