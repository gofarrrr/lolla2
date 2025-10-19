"""
METIS Research Integration Package
Lightweight research management with budgets, caching, and pattern selection
"""

from .manager import ResearchManager, get_research_manager
from .models import ResearchMode, SearchHit, ResearchResult, ResearchBudget
from .cache import ResearchCache

__all__ = [
    "ResearchManager",
    "get_research_manager",
    "ResearchMode",
    "SearchHit",
    "ResearchResult",
    "ResearchBudget",
    "ResearchCache",
]
