"""
Context-Aware Rubric Variants for CQA Framework
================================================

Implements specialized rubrics to prevent homogenization and maintain
cognitive diversity across different agent types.
"""

from pydantic import BaseModel, Field
from typing import Dict, Optional, List


class RubricWeight(BaseModel):
    """
    Weight configuration for a quality dimension.

    Attributes:
        dimension: The quality dimension
        weight: Relative importance (0.0 to 2.0, where 1.0 is standard)
        description: Context-specific interpretation of this dimension
    """

    dimension: str
    weight: float = Field(ge=0.0, le=2.0, default=1.0)
    description: str


class RubricVariant(BaseModel):
    """
    A specialized rubric variant for context-aware evaluation.

    Attributes:
        variant_id: Unique identifier (e.g., "riva_creativity_focused@1.0")
        name: Human-readable name
        description: When and why to use this variant
        dimension_weights: Custom weights for each dimension
        emphasis_points: Specific aspects to emphasize
        de_emphasis_points: Aspects to de-emphasize or ignore
        target_agents: Which agents should use this rubric
    """

    variant_id: str
    name: str
    description: str
    dimension_weights: Dict[str, float]
    emphasis_points: List[str]
    de_emphasis_points: List[str]
    target_agents: List[str]
    parent_variant: Optional[str] = None

    def get_adjusted_score(self, base_scores: Dict[str, float]) -> float:
        """
        Calculate weighted average based on this rubric's weights.

        Args:
            base_scores: Raw scores for each dimension

        Returns:
            Weighted average score
        """
        weighted_sum = 0.0
        total_weight = 0.0

        for dimension, score in base_scores.items():
            weight = self.dimension_weights.get(dimension, 1.0)
            weighted_sum += score * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0


class RubricRegistry:
    """
    Registry of all available rubric variants.
    """

    def __init__(self):
        self.variants: Dict[str, RubricVariant] = {}
        self._initialize_standard_variants()

    def _initialize_standard_variants(self):
        """Initialize the standard set of rubric variants."""

        # Standard balanced rubric
        self.register(
            RubricVariant(
                variant_id="riva_standard@1.0",
                name="Standard RIVA",
                description="Balanced evaluation across all dimensions",
                dimension_weights={
                    "rigor": 1.0,
                    "insight": 1.0,
                    "value": 1.0,
                    "alignment": 1.0,
                },
                emphasis_points=[
                    "Overall quality and balance",
                    "General-purpose evaluation",
                ],
                de_emphasis_points=[],
                target_agents=["default", "general_purpose"],
            )
        )

        # Creativity-focused for ideation agents
        self.register(
            RubricVariant(
                variant_id="riva_creativity_focused@1.0",
                name="Creative Innovation",
                description="Emphasizes novel insights and creative thinking",
                dimension_weights={
                    "rigor": 0.5,  # Less emphasis on strict logic
                    "insight": 2.0,  # Double weight on creativity
                    "value": 0.8,  # Slightly reduced practical focus
                    "alignment": 0.7,  # More flexibility allowed
                },
                emphasis_points=[
                    "Novel connections and perspectives",
                    "Creative problem framing",
                    "Unconventional approaches",
                    "Thought-provoking ideas",
                ],
                de_emphasis_points=[
                    "Strict logical progression",
                    "Immediate practicality",
                    "Conservative recommendations",
                ],
                target_agents=[
                    "creative_strategist",
                    "idea_burst_agent",
                    "lateral_thinker",
                    "innovation_catalyst",
                ],
            )
        )

        # Rigor-focused for analytical agents
        self.register(
            RubricVariant(
                variant_id="riva_rigor_focused@1.0",
                name="Analytical Excellence",
                description="Prioritizes logical consistency and methodological soundness",
                dimension_weights={
                    "rigor": 2.0,  # Double weight on rigor
                    "insight": 0.6,  # Less emphasis on novelty
                    "value": 1.0,  # Standard practical focus
                    "alignment": 1.4,  # Strong adherence to requirements
                },
                emphasis_points=[
                    "Logical consistency",
                    "Evidence-based reasoning",
                    "Systematic methodology",
                    "Comprehensive analysis",
                    "Falsifiable claims",
                ],
                de_emphasis_points=[
                    "Creative speculation",
                    "Intuitive leaps",
                    "Unproven innovations",
                ],
                target_agents=[
                    "devils_advocate",
                    "risk_analyst",
                    "financial_analyst",
                    "compliance_auditor",
                ],
            )
        )

        # Execution-focused for implementation agents
        self.register(
            RubricVariant(
                variant_id="riva_execution_focused@1.0",
                name="Practical Implementation",
                description="Emphasizes actionability and real-world application",
                dimension_weights={
                    "rigor": 0.8,
                    "insight": 0.6,
                    "value": 2.0,  # Double weight on practical value
                    "alignment": 1.2,
                },
                emphasis_points=[
                    "Clear action steps",
                    "Implementation roadmaps",
                    "Resource requirements",
                    "Timeline feasibility",
                    "Success metrics",
                ],
                de_emphasis_points=[
                    "Theoretical exploration",
                    "Abstract concepts",
                    "Long-term speculation",
                ],
                target_agents=[
                    "implementation_strategist",
                    "project_manager",
                    "operations_optimizer",
                ],
            )
        )

        # Problem-structuring focused
        self.register(
            RubricVariant(
                variant_id="riva_structuring_focused@1.0",
                name="Problem Structuring",
                description="Evaluates problem decomposition and framework selection",
                dimension_weights={
                    "rigor": 1.5,  # Strong logical structure
                    "insight": 1.2,  # Good problem reframing
                    "value": 0.8,  # Less emphasis on immediate value
                    "alignment": 1.0,
                },
                emphasis_points=[
                    "Problem decomposition quality",
                    "Framework appropriateness",
                    "Stakeholder identification",
                    "Boundary definition",
                    "Assumption surfacing",
                ],
                de_emphasis_points=[
                    "Solution specifics",
                    "Implementation details",
                    "Cost-benefit analysis",
                ],
                target_agents=[
                    "problem_structuring_agent",
                    "systems_thinker",
                    "root_cause_analyst",
                ],
            )
        )

        # Synthesis-focused for integration agents
        self.register(
            RubricVariant(
                variant_id="riva_synthesis_focused@1.0",
                name="Strategic Synthesis",
                description="Evaluates integration and coherence across perspectives",
                dimension_weights={
                    "rigor": 1.2,
                    "insight": 1.3,  # Synthesis often reveals insights
                    "value": 1.0,
                    "alignment": 1.5,  # Must align multiple viewpoints
                },
                emphasis_points=[
                    "Integration quality",
                    "Contradiction resolution",
                    "Emergent patterns",
                    "Coherent narrative",
                    "Balanced perspective",
                ],
                de_emphasis_points=[
                    "Individual detail depth",
                    "Single-perspective dominance",
                ],
                target_agents=[
                    "senior_advisor",
                    "strategic_synthesizer",
                    "integration_orchestrator",
                ],
            )
        )

    def register(self, variant: RubricVariant):
        """
        Register a new rubric variant.

        Args:
            variant: The rubric variant to register
        """
        self.variants[variant.variant_id] = variant

    def get_variant(self, variant_id: str) -> Optional[RubricVariant]:
        """
        Get a rubric variant by ID.

        Args:
            variant_id: ID of the variant

        Returns:
            RubricVariant or None if not found
        """
        return self.variants.get(variant_id)

    def get_variant_for_agent(self, agent_name: str) -> RubricVariant:
        """
        Get the appropriate rubric variant for an agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Most appropriate rubric variant (defaults to standard)
        """
        # Search for agent-specific variant
        for variant in self.variants.values():
            if agent_name in variant.target_agents:
                return variant

        # Default to standard
        return self.variants["riva_standard@1.0"]

    def list_variants(self) -> List[Dict[str, str]]:
        """
        List all available rubric variants.

        Returns:
            List of variant summaries
        """
        return [
            {
                "id": v.variant_id,
                "name": v.name,
                "description": v.description,
                "target_agents": ", ".join(v.target_agents[:3]),
            }
            for v in self.variants.values()
        ]


# Global registry instance
rubric_registry = RubricRegistry()
