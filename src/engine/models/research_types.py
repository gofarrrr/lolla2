#!/usr/bin/env python3
"""
METIS Research Types
Shared types for multi-provider research engine to avoid circular imports
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any

from src.engine.integrations.perplexity_client_advanced import (
    ResearchTemplateType,
    ResearchMode,
)


class ResearchStrategy(Enum):
    """Research strategy selection"""

    COST_OPTIMIZED = "cost_optimized"  # ODR primary, minimal Firecrawl
    QUALITY_OPTIMIZED = "quality_optimized"  # All providers, comprehensive
    SPEED_OPTIMIZED = "speed_optimized"  # Fastest single provider
    DEPTH_OPTIMIZED = "depth_optimized"  # ODR + Firecrawl deep extraction
    SOCIAL_INTELLIGENCE = "social_intelligence"  # ODR + Apify for social media
    REAL_TIME_MONITORING = (
        "real_time_monitoring"  # Apify + Perplexity for real-time data
    )
    COMPREHENSIVE = (
        "comprehensive"  # All 4 providers (ODR, Firecrawl, Perplexity, Apify)
    )
    HYBRID_INTELLIGENT = "hybrid_intelligent"  # AI-driven provider selection


class QueryComplexity(Enum):
    """Query complexity assessment"""

    SIMPLE = "simple"  # Basic search queries
    MODERATE = "moderate"  # Multi-faceted research
    COMPLEX = "complex"  # Comprehensive analysis needed
    SPECIALIZED = "specialized"  # Domain-specific deep dive


@dataclass
class ResearchRequest:
    """Comprehensive research request specification"""

    query: str
    context: Dict[str, Any]
    template_type: ResearchTemplateType
    mode: ResearchMode
    strategy: ResearchStrategy = ResearchStrategy.HYBRID_INTELLIGENT
    max_cost_usd: float = 0.10
    max_time_seconds: int = 60
    specific_urls: List[str] = field(default_factory=list)
    domain_preferences: List[str] = field(default_factory=list)
    require_recent_data: bool = False
    include_deep_extraction: bool = True
    engagement_id: Optional[str] = None  # For storage and context tracking
