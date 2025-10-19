"""
Configuration for Open Deep Research client
"""

from typing import Optional
from dataclasses import dataclass


@dataclass
class ODRConfiguration:
    """Configuration for Open Deep Research client"""

    # LLM Configuration
    anthropic_api_key: Optional[str] = None
    model_name: str = "claude-3-5-sonnet-20241022"
    max_tokens: int = 4000
    temperature: float = 0.1

    # Search Configuration
    tavily_api_key: Optional[str] = None
    max_search_results: int = 10
    search_depth: str = "advanced"  # basic, advanced

    # Research Parameters
    max_iterations: int = 5
    research_timeout: int = 180  # seconds
    min_sources: int = 3
    max_sources: int = 15

    # Quality Thresholds
    min_confidence: float = 0.3
    target_confidence: float = 0.8
    diversity_threshold: float = 0.6
