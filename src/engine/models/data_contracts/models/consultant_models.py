"""
Consultant Domain Models

Models related to consultant specialization, scoring, and blueprints.
"""

from typing import Dict, List, Optional, Any, TYPE_CHECKING
from datetime import datetime, timezone
from pydantic import BaseModel, Field, field_validator, model_validator
from .enums import *

# Import FallbackBehavior to avoid circular dependency
if TYPE_CHECKING:
    from .analysis_models import FallbackBehavior
else:
    # Runtime import
    from .analysis_models import FallbackBehavior

class ConsultantSpecialization(BaseModel):
    """
    Individual consultant specialization configuration for rules engine
    Based on RULES_ENGINE_SCORING_ALGORITHM_SPECIFICATION.md
    """

    consultant_id: str = Field(..., description="Unique consultant identifier")
    display_name: str = Field(
        ..., min_length=3, max_length=100, description="Human-readable consultant name"
    )
    strategic_layer: StrategicLayer = Field(
        ..., description="Strategic layer classification"
    )
    cognitive_function: CognitiveFunction = Field(
        ..., description="Primary cognitive function"
    )

    # Scoring algorithm inputs
    trigger_keywords: List[str] = Field(
        default_factory=list, description="Keywords that trigger this consultant"
    )
    preferred_mental_models: List[str] = Field(
        default_factory=list, description="Mental models this consultant prefers"
    )
    bias_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Domain bias scores populated from mental_models_system.py",
    )

    # Custom configuration overrides
    custom_bias_overrides: Dict[str, float] = Field(
        default_factory=dict, description="Manual bias overrides for specific domains"
    )
    effectiveness_multiplier: float = Field(
        default=1.0, ge=0.1, le=2.0, description="Effectiveness adjustment factor"
    )

    # Persona and prompting
    persona_prompt: str = Field(
        default="", description="Base persona prompt for this consultant"
    )
    signature_approach: str = Field(
        default="", description="Signature consulting approach description"
    )

    @field_validator("bias_scores", "custom_bias_overrides")
    def validate_bias_score_range(cls, v):
        """Ensure all bias scores are in valid range"""
        for key, score in v.items():
            if not (0.0 <= score <= 1.0):
                raise ValueError(
                    f"Bias score for '{key}' must be between 0.0 and 1.0, got {score}"
                )
        return v




class ScoringWeights(BaseModel):
    """
    Scoring weights configuration from RULES_ENGINE_SCORING_ALGORITHM_SPECIFICATION.md
    """

    keyword_match: float = Field(
        default=0.4, ge=0.0, le=1.0, description="Weight for trigger keyword matches"
    )
    mental_model_bias: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Weight for mental model alignment"
    )
    strategic_layer_fit: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Weight for strategic layer appropriateness",
    )
    cognitive_function_match: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Weight for cognitive function alignment",
    )

    @model_validator(mode="after")
    def validate_weight_sum(self):
        """Validate that weights sum to approximately 1.0"""
        total = (
            self.keyword_match
            + self.mental_model_bias
            + self.strategic_layer_fit
            + self.cognitive_function_match
        )
        if not (0.95 <= total <= 1.05):
            raise ValueError(
                f"Scoring weights must sum to 1.0 (±0.05), current sum: {total}"
            )
        return self




class ConsultantMatrixConfig(BaseModel):
    """
    Complete consultant matrix configuration with validation
    Based on RULES_ENGINE_SCORING_ALGORITHM_SPECIFICATION.md
    """

    schema_version: str = Field(
        default="1.0", description="Configuration schema version"
    )
    matrix_name: str = Field(
        default="Nine Consultant Matrix", description="Human-readable matrix name"
    )

    # Core configuration components
    scoring_weights: ScoringWeights = Field(
        default_factory=ScoringWeights, description="Scoring algorithm weights"
    )
    fallback_behavior: FallbackBehavior = Field(
        default_factory=FallbackBehavior,
        description="Fallback and compatibility settings",
    )

    # Consultant definitions
    consultants: Dict[ExtendedConsultantRole, ConsultantSpecialization] = Field(
        default_factory=dict,
        description="Complete consultant specialization definitions",
    )

    # Legacy compatibility mappings
    legacy_consultant_mapping: Dict[str, ExtendedConsultantRole] = Field(
        default_factory=dict,
        description="Mapping from legacy consultant IDs to new extended roles",
    )

    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_modified: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    configuration_source: str = Field(
        default="yaml", description="Source of this configuration"
    )

    @field_validator("consultants")
    def validate_consultant_matrix_completeness(cls, v):
        """Ensure all 9 consultants are defined"""
        required_roles = set(ExtendedConsultantRole)
        defined_roles = set(v.keys())
        missing_roles = required_roles - defined_roles

        if missing_roles:
            raise ValueError(
                f"Missing consultant definitions for: {[role.value for role in missing_roles]}"
            )

        return v

    @field_validator("consultants")
    def validate_strategic_layer_distribution(cls, v):
        """Ensure proper distribution across strategic layers"""
        layer_counts = {}
        for role, spec in v.items():
            layer = spec.strategic_layer
            layer_counts[layer] = layer_counts.get(layer, 0) + 1

        # Should have exactly 3 consultants per strategic layer
        for layer in StrategicLayer:
            if layer_counts.get(layer, 0) != 3:
                raise ValueError(
                    f"Strategic layer '{layer.value}' must have exactly 3 consultants, has {layer_counts.get(layer, 0)}"
                )

        return v

    @field_validator("consultants")
    def validate_cognitive_function_distribution(cls, v):
        """Ensure proper distribution across cognitive functions"""
        function_counts = {}
        for role, spec in v.items():
            function = spec.cognitive_function
            function_counts[function] = function_counts.get(function, 0) + 1

        # Should have exactly 3 consultants per cognitive function
        for function in CognitiveFunction:
            if function_counts.get(function, 0) != 3:
                raise ValueError(
                    f"Cognitive function '{function.value}' must have exactly 3 consultants, has {function_counts.get(function, 0)}"
                )

        return v

    def get_legacy_three(self) -> List[ExtendedConsultantRole]:
        """Get the three legacy consultant roles for compatibility"""
        # Default legacy three mapping based on the original system
        legacy_defaults = [
            ExtendedConsultantRole.STRATEGIC_ANALYST,
            ExtendedConsultantRole.TACTICAL_SOLUTION_ARCHITECT,
            ExtendedConsultantRole.OPERATIONAL_EXECUTION_SPECIALIST,
        ]

        # Use mapping if provided, otherwise use defaults
        if self.legacy_consultant_mapping:
            return list(self.legacy_consultant_mapping.values())[:3]

        return legacy_defaults

    model_config = ConfigDict()




class ConsultantSelectionInput(BaseModel):
    """
    Input data for consultant selection algorithm
    Based on RULES_ENGINE_SCORING_ALGORITHM_SPECIFICATION.md
    """

    query_text: str = Field(..., min_length=10, description="The user query to analyze")
    domain_hint: Optional[str] = Field(
        None, description="Optional domain/industry hint"
    )
    suggested_models: List[str] = Field(
        default_factory=list, description="Suggested mental models from prior analysis"
    )
    enhanced_context: Dict[str, Any] = Field(
        default_factory=dict, description="Additional context for selection"
    )
    urgency_level: str = Field(
        default="medium", description="Query urgency: low, medium, high"
    )
    client_preferences: Dict[str, Any] = Field(
        default_factory=dict, description="Client-specific preferences"
    )




class ConsultantSelectionResult(BaseModel):
    """
    Result of consultant selection with detailed reasoning
    Based on RULES_ENGINE_SCORING_ALGORITHM_SPECIFICATION.md
    """

    selected_consultants: List[ExtendedConsultantRole] = Field(
        ..., description="Selected consultant roles"
    )
    selection_scores: Dict[ExtendedConsultantRole, float] = Field(
        default_factory=dict, description="Individual consultant scores"
    )
    selection_reasoning: Dict[ExtendedConsultantRole, str] = Field(
        default_factory=dict, description="Selection reasoning for each consultant"
    )

    # Algorithm metadata
    algorithm_version: str = Field(
        default="rules_based_v1.0", description="Selection algorithm version used"
    )
    scoring_weights_used: ScoringWeights = Field(
        ..., description="Scoring weights applied"
    )
    fallback_applied: bool = Field(
        default=False, description="Whether fallback logic was triggered"
    )
    legacy_compatibility_mode: bool = Field(
        default=False, description="Whether legacy compatibility was enforced"
    )

    # Performance metadata
    selection_time_ms: int = Field(
        default=0, ge=0, description="Selection processing time in milliseconds"
    )
    total_consultants_scored: int = Field(
        default=9, ge=1, description="Total number of consultants evaluated"
    )

    # Quality indicators
    minimum_score: float = Field(
        ..., ge=0.0, le=1.0, description="Lowest score among selected consultants"
    )
    maximum_score: float = Field(
        ..., ge=0.0, le=1.0, description="Highest score among selected consultants"
    )
    average_score: float = Field(
        ..., ge=0.0, le=1.0, description="Average score of selected consultants"
    )
    score_distribution: Dict[str, int] = Field(
        default_factory=dict, description="Distribution of scores across ranges"
    )

    # Timestamp
    selected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict()




