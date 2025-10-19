"""
Data models for research system
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class ResearchMode(str, Enum):
    """Research execution modes with different budgets"""

    FAST = "fast"
    MODERATE = "moderate"
    DEEP = "deep"


@dataclass
class SearchHit:
    """Individual search result from research query"""

    url: str
    title: str
    content: str
    domain: str
    date: Optional[str] = None
    confidence: float = 0.7


@dataclass
class ResearchResult:
    """Structured research result with metrics"""

    summary: str
    bullets: List[str]
    sources: List[Dict[str, Any]]
    coverage_score: float
    consistency_score: float
    confidence: float
    time_spent_ms: int
    queries: List[str]

    # Additional metadata
    mode_used: str = "fast"
    early_stopped: bool = False
    timeout_occurred: bool = False
    cache_hit: bool = False

    # Enhanced attribution features
    fact_verification: Dict[str, float] = None  # fact -> confidence score
    source_credibility: Dict[str, float] = None  # url -> credibility score
    cross_reference_score: float = 0.0  # cross-referencing quality
    fact_extraction_count: int = 0  # number of facts extracted
    contradictions_detected: List[str] = None  # detected contradictions

    def __post_init__(self):
        """Initialize optional fields"""
        if self.fact_verification is None:
            self.fact_verification = {}
        if self.source_credibility is None:
            self.source_credibility = {}
        if self.contradictions_detected is None:
            self.contradictions_detected = []


@dataclass
class ResearchBudget:
    """Budget constraints for research modes"""

    max_calls: int
    max_time_seconds: float
    max_tokens: int
