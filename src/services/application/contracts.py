"""
V5.3 Service Layer Contracts

This module defines the service contracts (interfaces/protocols) for the application
services extracted from main.py as part of Operation Lean - Target #2.
"""

from typing import Protocol, Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum


# ============================================================================
# System-2 Tier Classification
# ============================================================================

class Tier(str, Enum):
    """System-2 tier classification levels"""
    DISABLED = "S2_DISABLED"
    TIER_1 = "S2_TIER_1"
    TIER_2 = "S2_TIER_2"
    TIER_3 = "S2_TIER_3"


# ============================================================================
# Analysis Data Models
# ============================================================================

@dataclass
class AnalysisContext:
    """Context for analysis execution"""
    query: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    tier: int = 1
    consultants: List[str] = None
    models: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.consultants is None:
            self.consultants = []
        if self.models is None:
            self.models = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AnalysisResult:
    """Result of analysis execution"""
    content: str
    context: AnalysisContext
    quality_scores: Dict[str, Any]
    execution_time_ms: float
    trace_id: str
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# ============================================================================
# Service Contracts (Protocols)
# ============================================================================

class ISystem2ClassificationService(Protocol):
    """
    System-2 tier classification service contract.

    Classifies queries into System-2 cognitive tiers based on complexity,
    stakes, and cognitive load requirements.
    """

    def classify_tier(
        self,
        query: str,
        complexity: str = "auto",
        context: Optional[Dict[str, Any]] = None
    ) -> Tier:
        """
        Classify query into System-2 tier.

        Args:
            query: The user query to classify
            complexity: Complexity hint ("auto", "simple", "strategic", "complex")
            context: Optional context for classification

        Returns:
            Tier enum indicating classification level
        """
        ...


class IAnalysisOrchestrationService(Protocol):
    """
    Analysis orchestration service contract.

    Orchestrates end-to-end analysis execution including consultant/model
    selection, LLM generation, memory context integration, and quality scoring.
    """

    async def analyze_query(
        self,
        query: str,
        config: Dict[str, Any]
    ) -> AnalysisResult:
        """
        Execute complete analysis on query.

        Args:
            query: The user query to analyze
            config: Configuration dict with context, complexity, etc.

        Returns:
            AnalysisResult with content, quality scores, and metadata
        """
        ...

    async def generate_analysis_with_memory(
        self,
        context: AnalysisContext
    ) -> str:
        """
        Generate analysis with memory context integration.

        Args:
            context: AnalysisContext with query, tier, consultants, models

        Returns:
            Generated analysis text
        """
        ...
