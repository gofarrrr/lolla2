"""
METIS V2 Framework Contracts
Data contracts for the Augmented Core architecture with two-tiered N-Way system
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class FrameworkChunk(BaseModel):
    """
    Individual chunk of the structured analytical framework
    Each chunk represents a specific analytical focus area with assigned N-Way clusters
    """

    part_number: int = Field(
        ..., description="Sequential number of this framework chunk (1, 2, 3, etc.)"
    )
    title: str = Field(
        ...,
        description="Descriptive title for this analytical chunk (e.g., 'Market & Competitive Analysis')",
    )
    description: str = Field(
        ..., description="Brief summary of the analytical goal and scope for this chunk"
    )
    assigned_nway_clusters: List[str] = Field(
        ...,
        description="List of N-Way cluster IDs most relevant to this chunk for Dynamic model selection",
    )
    key_hypotheses_to_test: List[str] = Field(
        ...,
        description="Specific, falsifiable hypotheses that should be tested in this analytical chunk",
    )


class StructuredAnalyticalFramework(BaseModel):
    """
    Complete analytical framework produced by the ProblemStructuringAgent
    This becomes the input to the V2 OptimalConsultantEngine for enhanced processing
    """

    engagement_id: str = Field(..., description="Unique identifier for this engagement")
    refined_problem_statement: str = Field(
        ..., description="Enhanced and refined problem statement after PSA processing"
    )
    framework_chunks: List[FrameworkChunk] = Field(
        ...,
        description="List of analytical chunks that decompose the problem into manageable parts",
    )

    # Optional metadata for enhanced processing
    created_at: Optional[datetime] = Field(
        default_factory=datetime.now, description="Framework creation timestamp"
    )
    psa_processing_time_ms: Optional[int] = Field(
        None, description="Time taken by PSA to create this framework"
    )
    core_nway_model_used: Optional[str] = Field(
        None, description="The Core N-Way model ID used by PSA"
    )
    confidence_score: Optional[float] = Field(
        None, description="PSA confidence in this framework (0.0-1.0)"
    )

    class Config:
        schema_extra = {
            "example": {
                "engagement_id": "eng_12345",
                "refined_problem_statement": "Analyze strategic market entry options for SaaS expansion into European market, considering competitive landscape, regulatory requirements, and resource constraints",
                "framework_chunks": [
                    {
                        "part_number": 1,
                        "title": "Market & Competitive Analysis",
                        "description": "Assess market size, growth dynamics, and competitive positioning in target European markets",
                        "assigned_nway_clusters": [
                            "NWAY_MARKET_ANALYSIS_001",
                            "NWAY_COMPETITIVE_DYNAMICS_003",
                        ],
                        "key_hypotheses_to_test": [
                            "German market shows highest SaaS adoption rate for our category",
                            "Existing competitors have weak customer retention in SMB segment",
                        ],
                    },
                    {
                        "part_number": 2,
                        "title": "Regulatory & Compliance Framework",
                        "description": "Evaluate GDPR compliance requirements and data localization needs across target markets",
                        "assigned_nway_clusters": ["NWAY_REGULATORY_ANALYSIS_005"],
                        "key_hypotheses_to_test": [
                            "Data localization requirements create significant technical barriers",
                            "GDPR compliance costs are manageable with current infrastructure",
                        ],
                    },
                    {
                        "part_number": 3,
                        "title": "Resource & Implementation Strategy",
                        "description": "Assess required investments, timeline, and operational capabilities for successful market entry",
                        "assigned_nway_clusters": [
                            "NWAY_IMPLEMENTATION_PLANNING_008",
                            "NWAY_RESOURCE_ALLOCATION_012",
                        ],
                        "key_hypotheses_to_test": [
                            "Current engineering team can support European deployment within 6 months",
                            "Local partnerships reduce market entry costs by 40%",
                        ],
                    },
                ],
                "created_at": "2025-01-15T10:30:00Z",
                "core_nway_model_used": "NWAY_PROBLEM_DECONSTRUCTION_023",
                "confidence_score": 0.87,
            }
        }


class AgentPersona(BaseModel):
    """
    Data contract for agent persona information from the database
    Used in V2 consultant selection and prompt infusion
    """

    agent_id: str = Field(..., description="Unique identifier for this agent")
    agent_name: str = Field(..., description="Human-readable name of the agent")
    matrix_position: Optional[str] = Field(
        None, description="Position in specialization matrix"
    )
    core_function_ikigai: str = Field(
        ..., description="The agent's core purpose and expertise"
    )
    core_nway_model_id: str = Field(
        ..., description="ID of the Core N-Way model this agent uses"
    )

    # Core N-Way model details (joined from nway_interactions)
    core_model_type: Optional[str] = Field(None, description="Type of the core model")
    core_model_instructions: Optional[str] = Field(
        None, description="instructional_cue_apce from core model"
    )
    core_model_summary: Optional[str] = Field(
        None, description="emergent_effect_summary from core model"
    )


class DynamicNWayModel(BaseModel):
    """
    Data contract for Dynamic N-Way models used in situational directives
    """

    interaction_id: str = Field(
        ..., description="Unique identifier for this N-Way interaction"
    )
    type: str = Field(..., description="Type of N-Way interaction")
    instructional_cue_apce: str = Field(
        ..., description="Instructions for applying this model"
    )
    relevant_contexts: List[str] = Field(
        ..., description="Contexts where this model is most effective"
    )
    strength: str = Field(
        ..., description="Strength level of this interaction (High/Medium/Low)"
    )

    # Selection metadata
    relevance_score: Optional[float] = Field(
        None, description="How relevant this model is to current context (0.0-1.0)"
    )
    selection_reason: Optional[str] = Field(
        None, description="Why this model was selected"
    )


class V2PromptAssembly(BaseModel):
    """
    Data contract for the complete 4-part prompt assembly in V2 architecture
    """

    consultant_persona: str = Field(
        ..., description="Part 1: Persona prompt from agent_personas"
    )
    assigned_task: str = Field(
        ..., description="Part 2: Specific task assignment from framework chunk"
    )
    core_methodology: str = Field(
        ..., description="Part 3: Core N-Way model instructions"
    )
    situational_directives: List[str] = Field(
        ...,
        description="Part 4: Dynamic N-Way model instructions selected for this situation",
    )

    # Assembly metadata
    framework_chunk_number: int = Field(
        ..., description="Which framework chunk this assembly is for"
    )
    consultant_id: str = Field(..., description="Which consultant this assembly is for")
    core_nway_model_id: str = Field(..., description="Core model used in this assembly")
    dynamic_nway_model_ids: List[str] = Field(
        ..., description="Dynamic models used in this assembly"
    )

    def get_complete_prompt(self) -> str:
        """
        Assemble the complete 4-part prompt as specified in V2 architecture
        """
        situational_section = (
            "\n".join(
                [
                    f"**Dynamic Model {i+1}**: {directive}"
                    for i, directive in enumerate(self.situational_directives)
                ]
            )
            if self.situational_directives
            else "No additional situational directives selected."
        )

        return f"""# Persona
{self.consultant_persona}

# Assigned Task
{self.assigned_task}

# Core Methodology (Your Primary Operating System)
To accomplish this task, you MUST strictly adhere to the following proprietary cognitive protocol:
{self.core_methodology}

# Situational Directives (Additional Tools for This Specific Task)
In addition to your core methodology, the following highly relevant cognitive tools have been selected for this specific problem. You must also incorporate their directives into your analysis:
{situational_section}"""


class V2ConsultantSelection(BaseModel):
    """
    Enhanced consultant selection result for V2 architecture
    """

    consultant_id: str = Field(..., description="Selected consultant identifier")
    agent_persona: AgentPersona = Field(
        ..., description="Complete agent persona information"
    )
    framework_chunk: FrameworkChunk = Field(..., description="Assigned framework chunk")
    selected_dynamic_models: List[DynamicNWayModel] = Field(
        ...,
        description="Dynamic N-Way models selected for this consultant/chunk combination",
    )
    prompt_assembly: V2PromptAssembly = Field(
        ..., description="Complete assembled prompt"
    )

    # Selection metadata
    selection_confidence: float = Field(
        ..., description="Confidence in this selection (0.0-1.0)"
    )
    selection_reasoning: str = Field(
        ..., description="Why this consultant was chosen for this chunk"
    )


class V2EngagementResult(BaseModel):
    """
    Complete V2 engagement result with enhanced structure
    """

    engagement_id: str = Field(..., description="Unique engagement identifier")
    original_query: str = Field(..., description="Original user query")
    structured_framework: StructuredAnalyticalFramework = Field(
        ..., description="Framework created by ProblemStructuringAgent"
    )
    consultant_selections: List[V2ConsultantSelection] = Field(
        ..., description="Selected consultants with their assigned chunks and prompts"
    )

    # Processing metadata
    psa_execution_failed: bool = Field(
        False, description="Flag indicating if PSA failed and V1 fallback was used"
    )
    processing_time_seconds: float = Field(..., description="Total processing time")
    v2_enhancement_applied: bool = Field(
        True, description="Whether V2 enhancements were successfully applied"
    )
    created_at: datetime = Field(
        default_factory=datetime.now, description="Result creation timestamp"
    )

    class Config:
        schema_extra = {
            "example": {
                "engagement_id": "eng_12345",
                "original_query": "How should we expand our SaaS product to European markets?",
                "structured_framework": {
                    # ... framework example from above
                },
                "consultant_selections": [
                    # ... consultant selection examples
                ],
                "psa_execution_failed": False,
                "processing_time_seconds": 45.3,
                "v2_enhancement_applied": True,
                "created_at": "2025-01-15T10:32:15Z",
            }
        }
