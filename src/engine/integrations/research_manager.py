"""
DEPRECATED: Use src.integrations.research package instead
This file maintained for backward compatibility
"""

# Re-export everything from the refactored modules
from .research.models import ResearchMode, SearchHit, ResearchResult, ResearchBudget
from .research.cache import ResearchCache
from .research.manager import ResearchManager, get_research_manager

# Re-export for backward compatibility
__all__ = [
    "ResearchMode",
    "SearchHit",
    "ResearchResult",
    "ResearchBudget",
    "ResearchCache",
    "ResearchManager",
    "get_research_manager",
]
