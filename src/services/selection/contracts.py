# src/services/selection/contracts.py
from dataclasses import dataclass, field
from typing import Protocol, Any, Dict, List, Optional


@dataclass
class ChemistryContext:
    """Aggregated inputs for CognitiveChemistryEngine to avoid 9-arg methods."""

    problem_framework: str
    nway_combination: List[Dict[str, Any]]
    # Optional knobs for optimization and consultant-aware variants
    available_nway_patterns: Optional[List[Dict[str, Any]]] = None
    target_score: float = 0.75
    max_iterations: int = 5
    available_consultants: Optional[List[str]] = None
    consultant_team: Optional[List[str]] = None
    context: Optional[Dict[str, Any]] = field(default_factory=dict)


class IChemistryScorer(Protocol):
    def score(self, ctx: ChemistryContext) -> Any: ...


class IChemistryOptimizer(Protocol):
    def optimize(self, ctx: ChemistryContext) -> Any: ...


class IChemistryAnalytics(Protocol):
    def get_insights(self, ctx: ChemistryContext) -> List[str]: ...
