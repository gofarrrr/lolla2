"""
Reconstruction State - Immutable Data Container
===============================================

Immutable state object passed through reconstruction stages.
Similar to LLMCallContext pattern but for report reconstruction.

Design Principles:
- Immutable (frozen dataclass)
- Builder pattern with .with_*() methods
- Type-safe field access
- Clear stage boundaries
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ReconstructionState:
    """
    Immutable state object passed through reconstruction stages.

    This state container follows the State Reconstruction Pattern, providing
    a clean immutable container for data as it flows through extraction stages.

    Stage Flow:
    1. DataFetchingStage → populates: cognitive_states, query_text, events
    2. StageExtractionStage → populates: senior_advisor, devils_advocate, etc.
    3. ConsultantAnalysisStage → populates: consultant_analyses
    4. EnhancementResearchStage → populates: enhancement_research_answers
    5. GlassBoxTransparencyStage → populates: transparency data
    6. BundleAssemblyStage → populates: bundle
    """

    # Input
    trace_id: str

    # Stage 1: Raw data from database/stream
    cognitive_states: List[Dict[str, Any]] = field(default_factory=list)
    query_text: Optional[str] = None
    events: List[Any] = field(default_factory=list)

    # Stage 2: Extracted cognitive stages
    stages: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    senior_advisor: Dict[str, Any] = field(default_factory=dict)
    devils_advocate: Dict[str, Any] = field(default_factory=dict)
    consultant_selection: Dict[str, Any] = field(default_factory=dict)
    problem_structuring: Optional[Dict[str, Any]] = None
    socratic_questions: Optional[Dict[str, Any]] = None

    # Stage 3: Consultant data
    consultant_analyses: List[Dict[str, Any]] = field(default_factory=list)

    # Stage 4: Enhancement research
    enhancement_research_answers: List[Dict[str, Any]] = field(default_factory=list)

    # Stage 5: Glass-Box transparency data
    human_interactions: Optional[Dict[str, Any]] = None
    research_providers: Optional[List[Dict[str, Any]]] = None
    evidence_trail: Optional[List[Dict[str, Any]]] = None
    quality_ribbon: Dict[str, Any] = field(default_factory=dict)
    plan_overview: Dict[str, Any] = field(default_factory=dict)
    dissent_signals: List[Dict[str, Any]] = field(default_factory=list)
    quality_metrics: Dict[str, Any] = field(default_factory=dict)

    # Stage 6: Final output
    bundle: Optional[Dict[str, Any]] = None

    # Builder methods for stage updates

    def with_data(self, **kwargs) -> ReconstructionState:
        """Create new state with updated raw data (Stage 1)."""
        return replace(self, **kwargs)

    def with_stages(
        self,
        senior_advisor: Optional[Dict[str, Any]] = None,
        devils_advocate: Optional[Dict[str, Any]] = None,
        consultant_selection: Optional[Dict[str, Any]] = None,
        problem_structuring: Optional[Dict[str, Any]] = None,
        socratic_questions: Optional[Dict[str, Any]] = None,
    ) -> ReconstructionState:
        """Create new state with extracted stages (Stage 2)."""
        updates: Dict[str, Any] = {}
        if senior_advisor is not None:
            updates["senior_advisor"] = senior_advisor
        if devils_advocate is not None:
            updates["devils_advocate"] = devils_advocate
        if consultant_selection is not None:
            updates["consultant_selection"] = consultant_selection
        if problem_structuring is not None:
            updates["problem_structuring"] = problem_structuring
        if socratic_questions is not None:
            updates["socratic_questions"] = socratic_questions
        return replace(self, **updates)

    def with_consultants(
        self, consultant_analyses: List[Dict[str, Any]]
    ) -> ReconstructionState:
        """Create new state with consultant analyses (Stage 3)."""
        return replace(self, consultant_analyses=consultant_analyses)

    def with_research(
        self, enhancement_research_answers: List[Dict[str, Any]]
    ) -> ReconstructionState:
        """Create new state with research answers (Stage 4)."""
        return replace(self, enhancement_research_answers=enhancement_research_answers)

    def with_transparency(
        self,
        human_interactions: Optional[Dict[str, Any]] = None,
        research_providers: Optional[List[Dict[str, Any]]] = None,
        evidence_trail: Optional[List[Dict[str, Any]]] = None,
        quality_ribbon: Optional[Dict[str, Any]] = None,
        plan_overview: Optional[Dict[str, Any]] = None,
        dissent_signals: Optional[List[Dict[str, Any]]] = None,
        quality_metrics: Optional[Dict[str, Any]] = None,
    ) -> ReconstructionState:
        """Create new state with transparency data (Stage 5)."""
        updates: Dict[str, Any] = {}
        if human_interactions is not None:
            updates["human_interactions"] = human_interactions
        if research_providers is not None:
            updates["research_providers"] = research_providers
        if evidence_trail is not None:
            updates["evidence_trail"] = evidence_trail
        if quality_ribbon is not None:
            updates["quality_ribbon"] = quality_ribbon
        if plan_overview is not None:
            updates["plan_overview"] = plan_overview
        if dissent_signals is not None:
            updates["dissent_signals"] = dissent_signals
        if quality_metrics is not None:
            updates["quality_metrics"] = quality_metrics
        return replace(self, **updates)

    def with_bundle(self, bundle: Dict[str, Any]) -> ReconstructionState:
        """Create new state with final bundle (Stage 6)."""
        return replace(self, bundle=bundle)

    def to_bundle(self) -> Dict[str, Any]:
        """
        Extract final bundle from state.

        Raises:
            ValueError: If bundle not yet assembled
        """
        if self.bundle is None:
            raise ValueError("Bundle not yet assembled - call BundleAssemblyStage first")
        return self.bundle
