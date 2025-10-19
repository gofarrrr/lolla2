"""
Mental Model Quality Rubrics for CQA Validation Framework
========================================================

Specialized quality rubrics and scoring criteria designed specifically for
evaluating mental models within the METIS V5.3 platform. Extends the core
RIVA framework with mental model-specific dimensions and weights.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class MentalModelType(Enum):
    """Types of mental models that require different evaluation criteria."""

    COGNITIVE_BIASES = "cognitive_biases"
    DECISION_FRAMEWORKS = "decision_frameworks"
    SYSTEMS_THINKING = "systems_thinking"
    PROBLEM_SOLVING = "problem_solving"
    STRATEGIC_FRAMEWORKS = "strategic_frameworks"
    ANALYTICAL_TOOLS = "analytical_tools"


class MentalModelQualityDimension(Enum):
    """Extended quality dimensions specific to mental model evaluation."""

    # Core RIVA dimensions
    RIGOR = "rigor"
    INSIGHT = "insight"
    VALUE = "value"
    ALIGNMENT = "alignment"

    # Mental Model specific dimensions
    CLARITY = "clarity"  # How clearly is the model explained
    APPLICABILITY = "applicability"  # How broadly can it be applied
    COHERENCE = "coherence"  # Internal logical consistency
    COMPLETENESS = "completeness"  # Coverage of key aspects


@dataclass
class MentalModelRubricCriteria:
    """Scoring criteria for mental model quality dimensions."""

    dimension: MentalModelQualityDimension
    criteria_1_3: str  # Poor (1-3)
    criteria_4_6: str  # Average (4-6)
    criteria_7_8: str  # Good (7-8)
    criteria_9_10: str  # Excellent (9-10)
    weight: float = 1.0  # Relative importance weight


@dataclass
class MentalModelRubric:
    """Complete rubric for mental model evaluation."""

    rubric_id: str
    name: str
    description: str
    model_type: MentalModelType
    criteria: List[MentalModelRubricCriteria]
    version: str = "1.0"


class MentalModelRubricRegistry:
    """Registry of mental model quality rubrics."""

    def __init__(self):
        self.rubrics = self._initialize_rubrics()

    def _initialize_rubrics(self) -> Dict[str, MentalModelRubric]:
        """Initialize all mental model rubrics."""
        rubrics = {}

        # Cognitive Biases Rubric
        rubrics["cognitive_biases@1.0"] = MentalModelRubric(
            rubric_id="cognitive_biases@1.0",
            name="Cognitive Biases Mental Model Evaluation",
            description="Specialized rubric for evaluating cognitive bias mental models",
            model_type=MentalModelType.COGNITIVE_BIASES,
            criteria=[
                MentalModelRubricCriteria(
                    dimension=MentalModelQualityDimension.RIGOR,
                    criteria_1_3="Lacks scientific backing, contains inaccuracies",
                    criteria_4_6="Generally accurate with minor gaps",
                    criteria_7_8="Well-researched with solid psychological foundations",
                    criteria_9_10="Exceptionally rigorous with extensive research citations",
                    weight=1.2,
                ),
                MentalModelRubricCriteria(
                    dimension=MentalModelQualityDimension.CLARITY,
                    criteria_1_3="Confusing explanation, unclear mechanisms",
                    criteria_4_6="Understandable but could be clearer",
                    criteria_7_8="Clear explanation with good examples",
                    criteria_9_10="Crystal clear with compelling illustrations",
                    weight=1.1,
                ),
                MentalModelRubricCriteria(
                    dimension=MentalModelQualityDimension.APPLICABILITY,
                    criteria_1_3="Very limited real-world applications",
                    criteria_4_6="Some practical applications identified",
                    criteria_7_8="Multiple clear applications across domains",
                    criteria_9_10="Exceptionally broad applicability with specific use cases",
                    weight=1.0,
                ),
                MentalModelRubricCriteria(
                    dimension=MentalModelQualityDimension.COHERENCE,
                    criteria_1_3="Internal contradictions, logical gaps",
                    criteria_4_6="Generally consistent with minor issues",
                    criteria_7_8="Logically coherent throughout",
                    criteria_9_10="Perfect internal consistency and logical flow",
                    weight=0.9,
                ),
            ],
        )

        # Decision Frameworks Rubric
        rubrics["decision_frameworks@1.0"] = MentalModelRubric(
            rubric_id="decision_frameworks@1.0",
            name="Decision Framework Mental Model Evaluation",
            description="Specialized rubric for evaluating decision-making framework models",
            model_type=MentalModelType.DECISION_FRAMEWORKS,
            criteria=[
                MentalModelRubricCriteria(
                    dimension=MentalModelQualityDimension.VALUE,
                    criteria_1_3="Provides little practical decision-making value",
                    criteria_4_6="Some useful decision guidance",
                    criteria_7_8="Strong practical value for decision makers",
                    criteria_9_10="Exceptional value, transforms decision-making approach",
                    weight=1.3,
                ),
                MentalModelRubricCriteria(
                    dimension=MentalModelQualityDimension.COMPLETENESS,
                    criteria_1_3="Missing key decision factors",
                    criteria_4_6="Covers most important aspects",
                    criteria_7_8="Comprehensive coverage of decision elements",
                    criteria_9_10="Complete framework addressing all critical factors",
                    weight=1.1,
                ),
                MentalModelRubricCriteria(
                    dimension=MentalModelQualityDimension.APPLICABILITY,
                    criteria_1_3="Very narrow application scope",
                    criteria_4_6="Applies to specific decision types",
                    criteria_7_8="Broadly applicable across decision contexts",
                    criteria_9_10="Universal framework applicable to all decision types",
                    weight=1.0,
                ),
                MentalModelRubricCriteria(
                    dimension=MentalModelQualityDimension.RIGOR,
                    criteria_1_3="Ad-hoc approach without systematic foundation",
                    criteria_4_6="Some systematic approach evident",
                    criteria_7_8="Well-structured systematic methodology",
                    criteria_9_10="Exceptionally rigorous systematic framework",
                    weight=0.9,
                ),
            ],
        )

        # Systems Thinking Rubric
        rubrics["systems_thinking@1.0"] = MentalModelRubric(
            rubric_id="systems_thinking@1.0",
            name="Systems Thinking Mental Model Evaluation",
            description="Specialized rubric for evaluating systems thinking mental models",
            model_type=MentalModelType.SYSTEMS_THINKING,
            criteria=[
                MentalModelRubricCriteria(
                    dimension=MentalModelQualityDimension.INSIGHT,
                    criteria_1_3="Limited systems perspective, focuses on parts",
                    criteria_4_6="Some systems thinking evident",
                    criteria_7_8="Strong systems perspective with interconnections",
                    criteria_9_10="Exceptional systems insight revealing hidden patterns",
                    weight=1.3,
                ),
                MentalModelRubricCriteria(
                    dimension=MentalModelQualityDimension.COHERENCE,
                    criteria_1_3="Disconnected elements, poor integration",
                    criteria_4_6="Some integration of system elements",
                    criteria_7_8="Well-integrated coherent system view",
                    criteria_9_10="Perfect systemic coherence and integration",
                    weight=1.2,
                ),
                MentalModelRubricCriteria(
                    dimension=MentalModelQualityDimension.COMPLETENESS,
                    criteria_1_3="Missing key system components or relationships",
                    criteria_4_6="Covers main system elements",
                    criteria_7_8="Comprehensive system representation",
                    criteria_9_10="Complete holistic system model",
                    weight=1.0,
                ),
                MentalModelRubricCriteria(
                    dimension=MentalModelQualityDimension.APPLICABILITY,
                    criteria_1_3="Limited to specific systems only",
                    criteria_4_6="Applies to similar system types",
                    criteria_7_8="Broadly applicable across system types",
                    criteria_9_10="Universal systems thinking principles",
                    weight=0.9,
                ),
            ],
        )

        # Generic Mental Model Rubric (fallback)
        rubrics["mental_model_generic@1.0"] = MentalModelRubric(
            rubric_id="mental_model_generic@1.0",
            name="Generic Mental Model Evaluation",
            description="General-purpose rubric for mental models that don't fit specific categories",
            model_type=MentalModelType.ANALYTICAL_TOOLS,  # Default type
            criteria=[
                MentalModelRubricCriteria(
                    dimension=MentalModelQualityDimension.RIGOR,
                    criteria_1_3="Lacks depth, contains inaccuracies or unsupported claims",
                    criteria_4_6="Generally sound with some minor issues",
                    criteria_7_8="Well-researched and methodologically sound",
                    criteria_9_10="Exceptionally rigorous and thoroughly validated",
                    weight=1.0,
                ),
                MentalModelRubricCriteria(
                    dimension=MentalModelQualityDimension.CLARITY,
                    criteria_1_3="Confusing, difficult to understand or apply",
                    criteria_4_6="Understandable but could be clearer",
                    criteria_7_8="Clear and well-explained with good examples",
                    criteria_9_10="Crystal clear with compelling illustrations and examples",
                    weight=1.0,
                ),
                MentalModelRubricCriteria(
                    dimension=MentalModelQualityDimension.VALUE,
                    criteria_1_3="Limited practical value or utility",
                    criteria_4_6="Some practical applications evident",
                    criteria_7_8="Strong practical value and clear benefits",
                    criteria_9_10="Exceptional value that transforms thinking or decision-making",
                    weight=1.0,
                ),
                MentalModelRubricCriteria(
                    dimension=MentalModelQualityDimension.APPLICABILITY,
                    criteria_1_3="Very narrow or limited application scope",
                    criteria_4_6="Applies to specific contexts or domains",
                    criteria_7_8="Broadly applicable across multiple contexts",
                    criteria_9_10="Universal applicability across diverse domains and situations",
                    weight=1.0,
                ),
            ],
        )

        return rubrics

    def get_rubric(self, rubric_id: str) -> Optional[MentalModelRubric]:
        """Get rubric by ID."""
        return self.rubrics.get(rubric_id)

    def get_rubric_for_model_type(
        self, model_type: MentalModelType
    ) -> MentalModelRubric:
        """Get the appropriate rubric for a mental model type."""
        rubric_map = {
            MentalModelType.COGNITIVE_BIASES: "cognitive_biases@1.0",
            MentalModelType.DECISION_FRAMEWORKS: "decision_frameworks@1.0",
            MentalModelType.SYSTEMS_THINKING: "systems_thinking@1.0",
            MentalModelType.PROBLEM_SOLVING: "mental_model_generic@1.0",
            MentalModelType.STRATEGIC_FRAMEWORKS: "decision_frameworks@1.0",  # Reuse decision framework
            MentalModelType.ANALYTICAL_TOOLS: "mental_model_generic@1.0",
        }

        rubric_id = rubric_map.get(model_type, "mental_model_generic@1.0")
        return self.rubrics[rubric_id]

    def list_available_rubrics(self) -> List[str]:
        """Get list of all available rubric IDs."""
        return list(self.rubrics.keys())

    def calculate_weighted_score(
        self, rubric_id: str, dimension_scores: Dict[MentalModelQualityDimension, float]
    ) -> float:
        """Calculate weighted average score using rubric weights."""
        rubric = self.get_rubric(rubric_id)
        if not rubric:
            # Fallback to simple average
            return sum(dimension_scores.values()) / len(dimension_scores)

        weighted_sum = 0.0
        total_weight = 0.0

        for criterion in rubric.criteria:
            if criterion.dimension in dimension_scores:
                weighted_sum += dimension_scores[criterion.dimension] * criterion.weight
                total_weight += criterion.weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0


# Global registry instance
_mental_model_rubric_registry = None


def get_mental_model_rubric_registry() -> MentalModelRubricRegistry:
    """Get the global mental model rubric registry instance."""
    global _mental_model_rubric_registry
    if _mental_model_rubric_registry is None:
        _mental_model_rubric_registry = MentalModelRubricRegistry()
    return _mental_model_rubric_registry
