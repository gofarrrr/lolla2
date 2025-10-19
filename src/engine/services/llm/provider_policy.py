"""
Provider Policy - Single Source of Truth for LLM Provider Ordering

Centralizes provider routing policies to eliminate implicit dual strategies.
Documented in GRAND_PROMPTING_AUDIT_REPORT.md Section E1.

Architecture:
- ORACLE flow: OpenRouter-first (Grok-4-Fast for strategic reasoning)
- GENERAL flow: OpenRouter-first (cost + performance default)
"""

from enum import Enum
from typing import List


class Flow(Enum):
    """LLM routing flow types"""
    ORACLE = "oracle"          # Progressive questions, strategic planning
    GENERAL = "general"        # Standard analysis, synthesis, chat


DEFAULT_POLICY = {
    # Tests expect Anthropic second for ORACLE
    Flow.ORACLE: ["openrouter", "anthropic", "deepseek"],
    # GENERAL flow: OpenRouter-first (cost + performance default)
    Flow.GENERAL: ["openrouter", "deepseek", "anthropic"],
}


def get_provider_chain(flow: Flow) -> List[str]:
    """
    Get provider fallback chain for specified flow.

    Args:
        flow: Flow type (ORACLE or GENERAL)

    Returns:
        List of provider names in priority order

    Example:
        >>> get_provider_chain(Flow.ORACLE)
        ['openrouter', 'anthropic', 'deepseek']

        >>> get_provider_chain(Flow.GENERAL)
        ['openrouter', 'deepseek', 'anthropic']
    """
    return DEFAULT_POLICY[flow].copy()


def get_flow_for_phase(phase: str) -> Flow:
    """
    Determine flow type from pipeline phase.

    Args:
        phase: Pipeline phase name

    Returns:
        Flow type (ORACLE or GENERAL)
    """
    # Oracle flow phases (strategic reasoning)
    oracle_phases = {
        "progressive_questions",
        "question_generation",
        "strategic_planning",
        "oracle_planning"
    }

    if phase in oracle_phases:
        return Flow.ORACLE

    return Flow.GENERAL


def get_provider_chain_for_phase(phase: str) -> List[str]:
    """
    Convenience method: Get provider chain directly from phase.

    Args:
        phase: Pipeline phase name

    Returns:
        List of provider names in priority order
    """
    flow = get_flow_for_phase(phase)
    return get_provider_chain(flow)
