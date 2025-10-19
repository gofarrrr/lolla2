# src/services/selection/pattern_contracts.py
from __future__ import annotations

from typing import Protocol, Dict, Any, List, Optional
from pydantic import BaseModel, Field


class Pattern(BaseModel):
    """Canonical representation of a NWAY pattern for selection and scoring."""

    id: str
    cluster: str
    title: str = ""
    models: List[str] = Field(default_factory=list)
    consultant_priority: List[str] = Field(default_factory=list)
    system2_triggers: List[str] = Field(default_factory=list)


class ScoreReport(BaseModel):
    """Result of scoring a set of patterns for a given context."""

    top_patterns: List[str] = Field(default_factory=list)
    scores: Dict[str, float] = Field(default_factory=dict)
    rationale: Dict[str, str] = Field(default_factory=dict)
    context_factors: Dict[str, Any] = Field(default_factory=dict)


class CoverageAnalysis(BaseModel):
    """Dimension coverage analysis for a set of selected patterns."""

    meets_target: bool
    total_coverage: float
    target_coverage: float
    coverage_grade: str
    coverage_gaps: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    dimension_coverage: Dict[str, Any] = Field(default_factory=dict)
    gap_count: int = 0
    validation_summary: str = ""


class OptimizationResult(BaseModel):
    """Optimization outcome for selected patterns under constraints."""

    selected_patterns: List[str]
    primary_pattern: Optional[str] = None
    confidence_score: float = 0.0
    rationale: str = ""
    fallback_patterns: List[str] = Field(default_factory=list)


class IPatternScorer(Protocol):
    """Scores patterns for a given context and returns a ScoreReport."""

    def score(self, patterns: List[Pattern], context: Dict[str, Any]) -> ScoreReport:
        ...


class IPatternOptimizer(Protocol):
    """Optimizes pattern selection under constraints to maximize outcome metrics."""

    def optimize(
        self,
        report: ScoreReport,
        context: Dict[str, Any],
        max_patterns: int = 3,
    ) -> OptimizationResult:
        ...


class ICoverageAnalyzer(Protocol):
    """Analyzes analytical dimension coverage for selected patterns."""

    def analyze(
        self,
        selected_patterns: List[str],
        framework_type: str,
        task_classification: Dict[str, Any],
        target_coverage: float = 0.90,
    ) -> CoverageAnalysis:
        ...


class IPatternAnalytics(Protocol):
    """Provides learning analytics and trend insights for pattern effectiveness."""

    def summarize(self) -> Dict[str, Any]:
        ...
