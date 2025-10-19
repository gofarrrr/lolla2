#!/usr/bin/env python3
"""
DeepSeek V3.1 Optimized Configuration Profiles
Production configurations based on research-backed optimization testing
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class DeepSeekConfig:
    """Configuration profile for DeepSeek V3.1 operations"""

    model: str
    timeout: int  # seconds
    temperature: float
    max_tokens: int
    strategy: str
    quality_threshold: float
    cost_sensitivity: str
    description: str


# Research-backed optimal configurations
ULTRA_COMPLEX_CONFIG = DeepSeekConfig(
    model="deepseek-reasoner",
    timeout=480,  # 8 minutes - adequate time for quality reasoning
    temperature=0.1,  # Research: 0.0-0.2 for reasoning tasks requiring precision
    max_tokens=3000,  # Balanced: efficient but sufficient for quality analysis
    strategy="chain_of_draft",  # Research-backed optimization
    quality_threshold=0.85,  # Higher quality threshold - prioritize intelligence
    cost_sensitivity="normal",  # Balance cost with quality
    description="Ultra-complex analysis optimized for quality and efficiency",
)

STANDARD_COMPLEX_CONFIG = DeepSeekConfig(
    model="deepseek-reasoner",
    timeout=600,  # 10 minutes - baseline timeout
    temperature=0.3,  # Reasonable baseline temperature
    max_tokens=4000,  # Standard baseline token budget
    strategy="zero_shot_optimized",  # Traditional verbose approach
    quality_threshold=0.8,  # Standard quality threshold
    cost_sensitivity="normal",
    description="Standard complex analysis with traditional approach",
)

FAST_RESPONSE_CONFIG = DeepSeekConfig(
    model="deepseek-chat",
    timeout=60,  # 1 minute for fast responses
    temperature=0.5,
    max_tokens=1500,
    strategy="direct",  # Minimal prompting overhead
    quality_threshold=0.5,
    cost_sensitivity="high",
    description="Fast responses for time-sensitive tasks",
)

# Configuration mapping for easy lookup
CONFIG_PROFILES = {
    "ultra_complex": ULTRA_COMPLEX_CONFIG,
    "standard_complex": STANDARD_COMPLEX_CONFIG,
    "fast_response": FAST_RESPONSE_CONFIG,
}

# Task complexity thresholds for automatic selection
COMPLEXITY_THRESHOLDS = {
    "ultra_complex": 0.8,
    "standard_complex": 0.5,
    "fast_response": 0.0,
}

# Task type mappings to optimal configurations
TASK_TYPE_CONFIG_MAP = {
    # Ultra-complex tasks requiring CoD optimization
    "multi_model_synthesis": "ultra_complex",
    "strategic_inversion_analysis": "ultra_complex",
    "assumption_network_analysis": "ultra_complex",
    "competitive_dynamics_modeling": "ultra_complex",
    "risk_cascade_analysis": "ultra_complex",
    "opportunity_cost_optimization": "ultra_complex",
    # Standard complex tasks
    "challenge_generation": "standard_complex",
    "assumption_challenge": "standard_complex",
    "strategic_synthesis": "standard_complex",
    "complex_reasoning": "standard_complex",
    "evidence_synthesis": "standard_complex",
    "hypothesis_testing": "standard_complex",
    # Fast response tasks
    "problem_classification": "fast_response",
    "query_generation": "fast_response",
    "summary_generation": "fast_response",
    "pattern_recognition": "fast_response",
    "context_understanding": "fast_response",
    "quick_insights": "fast_response",
}

# Time constraint mappings
TIME_CONSTRAINT_CONFIG_MAP = {
    "urgent": "fast_response",
    "normal": "standard_complex",
    "thorough": "ultra_complex",
}


def select_optimal_config(
    task_type: str, complexity_score: float, time_constraints: str = "normal"
) -> DeepSeekConfig:
    """
    Select optimal DeepSeek configuration based on task requirements

    Args:
        task_type: Type of cognitive task
        complexity_score: 0.0-1.0 complexity assessment
        time_constraints: "urgent", "normal", "thorough"

    Returns:
        Optimal DeepSeekConfig for the task
    """

    # Priority 1: Time constraints override complexity
    if time_constraints == "urgent":
        return CONFIG_PROFILES["fast_response"]

    # Priority 2: Explicit task type mapping
    if task_type in TASK_TYPE_CONFIG_MAP:
        config_name = TASK_TYPE_CONFIG_MAP[task_type]
        return CONFIG_PROFILES[config_name]

    # Priority 3: Complexity score thresholds
    if complexity_score >= COMPLEXITY_THRESHOLDS["ultra_complex"]:
        return CONFIG_PROFILES["ultra_complex"]
    elif complexity_score >= COMPLEXITY_THRESHOLDS["standard_complex"]:
        return CONFIG_PROFILES["standard_complex"]
    else:
        return CONFIG_PROFILES["fast_response"]


def get_config_by_name(config_name: str) -> Optional[DeepSeekConfig]:
    """Get configuration by name"""
    return CONFIG_PROFILES.get(config_name)


def get_all_configs() -> Dict[str, DeepSeekConfig]:
    """Get all available configurations"""
    return CONFIG_PROFILES.copy()


def calculate_complexity_multiplier(
    task_type: str, context_data: Optional[Dict[str, Any]] = None
) -> float:
    """
    Calculate complexity multiplier based on task characteristics

    Returns:
        Float multiplier (0.5 - 2.0) for timeout adjustment
    """
    base_multiplier = 1.0

    # Task type complexity factors
    complexity_factors = {
        "multi_model_synthesis": 1.8,
        "strategic_inversion_analysis": 1.6,
        "assumption_network_analysis": 1.4,
        "challenge_generation": 1.2,
        "assumption_challenge": 1.1,
        "summary_generation": 0.7,
        "quick_insights": 0.5,
    }

    multiplier = complexity_factors.get(task_type, base_multiplier)

    # Context-based adjustments
    if context_data:
        # Multiple stakeholders increase complexity
        if context_data.get("stakeholder_count", 1) > 3:
            multiplier *= 1.2

        # Historical analysis increases complexity
        if context_data.get("requires_historical_analysis", False):
            multiplier *= 1.3

        # Multi-industry analysis increases complexity
        if context_data.get("industry_count", 1) > 2:
            multiplier *= 1.1

    # Clamp between reasonable bounds
    return max(0.5, min(2.0, multiplier))
