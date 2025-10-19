"""
Research budget and mode configuration
"""

from .models import ResearchMode, ResearchBudget


# Mode budget configurations
RESEARCH_BUDGETS = {
    ResearchMode.FAST: ResearchBudget(
        max_calls=2, max_time_seconds=15.0, max_tokens=2000
    ),
    ResearchMode.MODERATE: ResearchBudget(
        max_calls=3, max_time_seconds=30.0, max_tokens=4000
    ),
    ResearchMode.DEEP: ResearchBudget(
        max_calls=5, max_time_seconds=45.0, max_tokens=6000
    ),
}


def get_budget_for_mode(mode: ResearchMode) -> ResearchBudget:
    """Get budget configuration for research mode"""
    return RESEARCH_BUDGETS[mode]
