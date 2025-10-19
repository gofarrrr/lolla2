"""
Analysis Domain Models

Models related to mental models, reasoning, research intelligence,
and cognitive state.
"""

from typing import Dict, List, Optional, Any, TYPE_CHECKING
from datetime import datetime, timezone
from uuid import UUID, uuid4
from pydantic import BaseModel, Field
from .enums import *

# Import types to avoid circular dependency
if TYPE_CHECKING:
    from .engagement_models import (
        EngagementContext,
        ExplorationContext,
        FailureModeResponse,
        ClarificationSession,
        WorkflowState,
        DeliverableArtifact,
    )
else:
    # Defer runtime import until needed
    pass

class MentalModelDefinition(BaseModel):
    """Mental model specification following MeMo framework"""

    model_id: str = Field(...)  # Accept both UUID and alphanumeric formats
    name: str = Field(..., min_length=3, max_length=100)
    category: MentalModelCategory
    description: str = Field(
        default="No description available"
    )  # Allow empty descriptions from DB
    application_criteria: List[str] = Field(
        default_factory=list
    )  # Allow empty criteria from DB
    expected_improvement: float = Field(
        default=50.0, ge=0.0, le=100.0
    )  # Default improvement
    validation_status: str = Field(default="pending")
    performance_metrics: Dict[str, float] = Field(default_factory=dict)




class ContextElement(BaseModel):
    """
    Sprint 1.4: Enhanced context element following Manus Taxonomy

    Represents a classified and scored context element for intelligent context management
    """

    element_id: str = Field(
        ..., description="Unique identifier for this context element"
    )
    content: str = Field(..., min_length=1, description="Actual context content")
    context_type: ContextType = Field(..., description="Manus taxonomy classification")
    relevance_score: float = Field(
        ..., ge=0.0, le=1.0, description="Computed relevance score"
    )
    relevance_level: ContextRelevanceLevel = Field(
        ..., description="Categorized relevance level"
    )

    # Temporal and source tracking
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = Field(default=1, ge=0)
    source_engagement_id: str = Field(
        ..., description="Source engagement for this context"
    )

    # Context intelligence metadata
    cognitive_coherence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Cognitive coherence with current reasoning",
    )
    compression_ratio: float = Field(
        default=1.0, ge=0.0, description="Compression efficiency if compressed"
    )
    cache_level: Optional[CognitiveCacheLevel] = Field(
        None, description="Which cache level contains this element"
    )

    model_config = ConfigDict()




class ContextRelevanceScore(BaseModel):
    """
    Sprint 1.4: Manus-inspired context relevance scoring result

    Comprehensive scoring breakdown with transparency into relevance calculation
    """

    element_id: str = Field(..., description="Context element being scored")
    overall_score: float = Field(
        ..., ge=0.0, le=1.0, description="Final weighted relevance score"
    )

    # Manus scoring dimensions
    semantic_similarity: float = Field(
        ..., ge=0.0, le=1.0, description="Semantic similarity to current goal"
    )
    temporal_recency: float = Field(
        ..., ge=0.0, le=1.0, description="Temporal relevance score"
    )
    usage_frequency: float = Field(
        ..., ge=0.0, le=1.0, description="Historical usage frequency score"
    )
    cognitive_coherence: float = Field(
        ..., ge=0.0, le=1.0, description="Cognitive coherence score (revolutionary)"
    )

    # Scoring metadata
    scoring_algorithm: str = Field(
        default="manus_enhanced_v1.4", description="Algorithm version used"
    )
    explanation: str = Field(..., description="Human-readable explanation of scoring")
    computed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict()




class ReasoningStep(BaseModel):
    """Individual reasoning step in cognitive process - supports both legacy and Neural Lace formats"""

    # Core fields (Neural Lace format)
    step: str = Field(..., description="Phase name or step identifier")
    description: str = Field(
        ..., min_length=10, description="Full reasoning description"
    )
    confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Confidence score"
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Operation Mindforge: Cognitive Exhaust Capture
    thinking_process: Optional[str] = Field(
        None, description="LLM's internal thinking process from <thinking> tags"
    )
    cleaned_response: Optional[str] = Field(
        None, description="Response content with thinking tags removed"
    )

    # Operation Synapse Sprint 1.4: Manus Taxonomy Integration
    associated_contexts: List[ContextElement] = Field(
        default_factory=list, description="Related context elements from Manus taxonomy"
    )
    context_relevance_scores: List[ContextRelevanceScore] = Field(
        default_factory=list, description="Context scoring results"
    )

    # Enhancement flags
    llm_enhanced: bool = Field(default=False)
    research_enhanced: Optional[bool] = Field(None)
    surgical_execution: bool = Field(default=False)
    neural_lace_capture: bool = Field(default=False)

    # Optional advanced fields
    accumulated_intelligence: bool = Field(default=False)
    maximum_intelligence_synthesis: bool = Field(default=False)
    context_phases: Optional[int] = Field(None)
    key_insights: List[str] = Field(default_factory=list)
    fallback_reason: Optional[str] = Field(None)

    # Legacy fields (for backward compatibility)
    step_id: Optional[str] = Field(None, pattern=r"^step_\d+$")
    mental_model_applied: Optional[str] = Field(None)
    reasoning_text: Optional[str] = Field(None)
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    evidence_sources: List[str] = Field(default_factory=list)
    assumptions_made: List[str] = Field(default_factory=list)




class ResearchIntelligence(BaseModel):
    """Research intelligence gathered during cognitive processing"""

    executive_summary: str = Field(
        default="", description="Executive summary of research findings"
    )
    strategic_insights: List[str] = Field(
        default_factory=list, description="Key strategic insights discovered"
    )
    evidence_base: Dict[str, List[str]] = Field(
        default_factory=dict, description="Evidence supporting each insight"
    )
    confidence_assessment: Dict[str, float] = Field(
        default_factory=dict, description="Confidence scores for each insight"
    )

    # Research quality metrics
    overall_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Overall research confidence"
    )
    evidence_strength: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Strength of evidence base"
    )
    source_diversity: int = Field(
        default=0, ge=0, description="Number of unique sources analyzed"
    )
    cross_validation_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Cross-validation quality"
    )

    # Research provenance
    sessions_included: List[str] = Field(
        default_factory=list, description="Research session IDs"
    )
    total_sources_analyzed: int = Field(
        default=0, ge=0, description="Total sources analyzed"
    )
    research_depth_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Depth of research conducted"
    )

    # Transparency fields
    information_gaps: List[str] = Field(
        default_factory=list, description="Identified information gaps"
    )
    additional_research_needs: List[str] = Field(
        default_factory=list, description="Recommended additional research"
    )
    contradictions_resolved: List[Dict[str, Any]] = Field(
        default_factory=list, description="Contradictions found and resolved"
    )

    # Processing metadata
    research_enabled: bool = Field(
        default=False, description="Whether research intelligence was enabled"
    )
    template_used: Optional[str] = Field(
        None, description="Research template type used"
    )
    queries_executed: List[str] = Field(
        default_factory=list, description="Research queries executed"
    )
    processing_time_ms: int = Field(
        default=0, ge=0, description="Research processing time"
    )




class CognitiveState(BaseModel):
    """Current state of cognitive processing"""

    selected_mental_models: List[MentalModelDefinition] = Field(default_factory=list)
    reasoning_steps: List[ReasoningStep] = Field(default_factory=list)
    confidence_scores: Dict[str, float] = Field(default_factory=dict)
    validation_results: Dict[str, Any] = Field(default_factory=dict)
    cognitive_load_assessment: Optional[str] = None

    # Enhanced research intelligence
    research_intelligence: Optional[ResearchIntelligence] = Field(
        None, description="Research intelligence gathered during processing"
    )

    @field_validator("selected_mental_models")
    def validate_model_selection(cls, v):
        if len(v) > 5:  # Cognitive load management
            raise ValueError("Maximum 5 mental models per engagement")
        return v




class HallucinationCheck(BaseModel):
    """Hallucination detection results"""

    check_type: str = Field(..., description="Type of hallucination check performed")
    is_valid: bool = Field(..., description="Whether content passed validation")
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in validation"
    )
    detection_level: VulnerabilityDetectionLevel = Field(
        ..., description="Severity of any issues detected"
    )
    details: Dict[str, Any] = Field(
        default_factory=dict, description="Detailed validation results"
    )
    evidence_sources: List[str] = Field(
        default_factory=list, description="Sources used for validation"
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))




class ContextIntelligenceResult(BaseModel):
    """
    Sprint 1.4: Context Intelligence analysis result following Manus methodology

    Comprehensive result of context intelligence analysis including Manus taxonomy classification,
    relevance scoring, and context optimization recommendations.
    """

    engagement_id: str = Field(..., description="Engagement this analysis belongs to")
    analysis_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Context elements analysis
    total_contexts_analyzed: int = Field(
        ..., ge=0, description="Total number of context elements analyzed"
    )
    contexts_by_type: Dict[ContextType, int] = Field(
        default_factory=dict, description="Context count by Manus type"
    )
    contexts_by_relevance: Dict[ContextRelevanceLevel, int] = Field(
        default_factory=dict, description="Context count by relevance level"
    )

    # Relevance scoring summary
    average_relevance_score: float = Field(
        ..., ge=0.0, le=1.0, description="Average relevance across all contexts"
    )
    highest_scoring_contexts: List[ContextElement] = Field(
        default_factory=list, description="Top N most relevant contexts"
    )

    # Context intelligence insights
    dominant_context_type: ContextType = Field(
        ..., description="Most prevalent context type"
    )
    cognitive_coherence_average: float = Field(
        ..., ge=0.0, le=1.0, description="Average cognitive coherence score"
    )

    # Performance metrics
    cache_distribution: Dict[CognitiveCacheLevel, int] = Field(
        default_factory=dict, description="Context distribution across cache levels"
    )
    processing_time_ms: int = Field(
        ..., ge=0, description="Time taken for context intelligence analysis"
    )

    # Recommendations
    compression_candidates: List[str] = Field(
        default_factory=list,
        description="Context element IDs that should be compressed",
    )
    archival_candidates: List[str] = Field(
        default_factory=list, description="Context element IDs that should be archived"
    )
    prefetch_recommendations: List[str] = Field(
        default_factory=list, description="Context element IDs to prefetch"
    )

    model_config = ConfigDict()




class FeedbackContext(BaseModel):
    """Multi-tier feedback collection context"""

    tier_level: FeedbackTier = Field(
        ..., description="Feedback tier for this engagement"
    )
    collection_enabled: bool = Field(
        default=True, description="Whether feedback collection is active"
    )
    incentive_structure: Dict[str, Any] = Field(
        default_factory=dict, description="Incentive alignment details"
    )
    partnership_benefits: List[str] = Field(
        default_factory=list, description="Benefits of feedback participation"
    )
    privacy_controls: Dict[str, bool] = Field(
        default_factory=dict, description="Privacy and consent settings"
    )
    expected_value_exchange: str = Field(
        ..., description="Expected value exchange for feedback"
    )




class VulnerabilityContext(BaseModel):
    """Complete vulnerability solution context for engagement"""

    session_id: str = Field(..., description="Vulnerability solution session ID")

    # Exploration context (forward reference to avoid circular import)
    exploration_context: Optional["ExplorationContext"] = Field(
        None, description="Exploration strategy context"
    )

    # Hallucination detection results
    hallucination_checks: List[HallucinationCheck] = Field(
        default_factory=list, description="All hallucination checks performed"
    )

    # Failure mode handling (forward reference to avoid circular import)
    failure_responses: List["FailureModeResponse"] = Field(
        default_factory=list, description="Failure mode responses"
    )

    # Pattern governance (forward reference - PatternGovernanceResult defined later in file)
    governance_results: Optional["PatternGovernanceResult"] = Field(
        None, description="Pattern governance results"
    )

    # Feedback system context
    feedback_context: Optional[FeedbackContext] = Field(
        None, description="Multi-tier feedback context"
    )

    # Overall vulnerability assessment
    overall_risk_level: VulnerabilityDetectionLevel = Field(
        default=VulnerabilityDetectionLevel.NONE, description="Overall risk assessment"
    )
    solution_effectiveness: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Effectiveness of vulnerability solutions",
    )

    # Transparency and audit
    transparency_level: str = Field(
        default="high", description="Level of transparency maintained"
    )
    audit_trail: List[str] = Field(
        default_factory=list, description="Vulnerability solution audit trail"
    )

    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))




class FallbackBehavior(BaseModel):
    """
    Fallback behavior configuration from RULES_ENGINE_SCORING_ALGORITHM_SPECIFICATION.md
    """

    always_include_legacy_three: bool = Field(
        default=True,
        description="Always include legacy three consultants when possible",
    )
    minimum_confidence_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum confidence required for selection",
    )
    max_consultants_per_function: int = Field(
        default=2, ge=1, le=3, description="Maximum consultants per cognitive function"
    )
    legacy_preference_tolerance: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Tolerance factor for legacy consultant replacement",
    )




class PatternGovernanceResult(BaseModel):
    """Pattern governance evaluation results"""

    governance_tier: str = Field(..., description="Governance tier applied (L1/L2/L3)")
    pattern_validity: bool = Field(
        ..., description="Whether emergent patterns are valid"
    )
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in governance results"
    )
    validation_method: str = Field(
        ..., description="Method used for pattern validation"
    )
    evidence_strength: float = Field(
        ..., ge=0.0, le=1.0, description="Strength of supporting evidence"
    )
    knowledge_gaps: List[str] = Field(
        default_factory=list, description="Identified knowledge gaps"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Governance recommendations"
    )




class MetisDataContract(BaseModel):
    """
    Unified data contract for all METIS system components
    CloudEvents-compliant schema ensuring interoperability
    """

    # CloudEvents standard fields
    specversion: str = Field(default="1.0")
    type: str = Field(..., pattern=r"^metis\.[a-z_]+$")
    source: str = Field(..., pattern=r"^/metis/[a-z_]+$")
    id: UUID = Field(default_factory=uuid4)
    time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # METIS-specific data payload (using forward references for cross-module types)
    engagement_context: "EngagementContext"
    cognitive_state: CognitiveState
    workflow_state: "WorkflowState"
    deliverable_artifacts: List["DeliverableArtifact"] = Field(default_factory=list)

    # HITL Clarification data (optional)
    clarification_session: Optional["ClarificationSession"] = Field(
        None, description="HITL clarification session data"
    )

    # Vulnerability solution context (V2 enhancement)
    vulnerability_context: Optional[VulnerabilityContext] = Field(
        None, description="Vulnerability solution context and results"
    )

    # System metadata
    schema_version: str = Field(
        default="2.0.0"
    )  # Upgraded for V2 vulnerability enhancements
    processing_metadata: Dict[str, Any] = Field(default_factory=dict)

    # Neural Lace data capture fields
    raw_outputs: Dict[str, Any] = Field(
        default_factory=dict, description="Full LLM outputs for transparency"
    )
    integration_calls: List[Dict[str, Any]] = Field(
        default_factory=list, description="Complete API call logs"
    )
    analysis_results: Dict[str, Any] = Field(
        default_factory=dict, description="Analysis results by phase"
    )
    research_results: Dict[str, Any] = Field(
        default_factory=dict, description="Research grounding results"
    )
    validation_results: Dict[str, Any] = Field(
        default_factory=dict, description="Devil's advocate validation results"
    )
    final_synthesis: Dict[str, Any] = Field(
        default_factory=dict, description="Final synthesis and recommendations"
    )
    verification_results: List[Dict[str, Any]] = Field(
        default_factory=list, description="Truth triangulation results"
    )
    value_assessments: List[Dict[str, Any]] = Field(
        default_factory=list, description="Value assessment results"
    )

    model_config = ConfigDict()

    @field_validator("type")
    def validate_event_type(cls, v):
        valid_types = [
            # Original V1 events
            "metis.engagement_initiated",
            "metis.engagement_request",
            "metis.cognitive_model_selected",
            "metis.workflow_phase_completed",
            "metis.analysis_framework_applied",
            "metis.synthesis_deliverable_generated",
            "metis.validation_quality_assessed",
            "metis.error_component_failed",
            # V2 vulnerability solution events
            "metis.vulnerability_assessment_completed",
            "metis.exploration_strategy_applied",
            "metis.hallucination_detected",
            "metis.failure_mode_activated",
            "metis.pattern_governance_applied",
            "metis.feedback_tier_assigned",
            "metis.solution_effectiveness_measured",
        ]
        if v not in valid_types:
            raise ValueError(f"Invalid event type: {v}")
        return v

    @model_validator(mode="after")
    def enforce_memory_limits(self):
        """Automatically enforce memory limits during contract updates"""
        import os

        if os.getenv("METIS_ENFORCE_MEMORY_LIMITS", "true").lower() == "true":
            try:
                if self.get_memory_size_mb() > 50.0:
                    logger.warning("Contract size exceeded limit, truncating data")
                    self.enforce_size_limit(50.0)
            except Exception as e:
                logger.warning(f"Memory limit enforcement failed: {e}")

        return self

    def to_cloudevents_dict(self) -> Dict[str, Any]:
        """Convert to CloudEvents standard format"""
        return {
            "specversion": self.specversion,
            "type": self.type,
            "source": self.source,
            "id": str(self.id),
            "time": self.time.isoformat(),
            "datacontenttype": "application/json",
            "data": {
                "engagement_context": (
                    self.engagement_context.dict()
                    if hasattr(self.engagement_context, "dict")
                    else self.engagement_context
                ),
                "cognitive_state": (
                    self.cognitive_state.dict()
                    if hasattr(self.cognitive_state, "dict")
                    else self.cognitive_state
                ),
                "workflow_state": (
                    self.workflow_state.dict()
                    if hasattr(self.workflow_state, "dict")
                    else self.workflow_state
                ),
                "deliverable_artifacts": [
                    a.dict() if hasattr(a, "dict") else a
                    for a in self.deliverable_artifacts
                ],
                "vulnerability_context": (
                    self.vulnerability_context.dict()
                    if (
                        self.vulnerability_context
                        and hasattr(self.vulnerability_context, "dict")
                    )
                    else self.vulnerability_context
                ),
                "schema_version": self.schema_version,
                "processing_metadata": self.processing_metadata,
            },
        }

    @classmethod
    def from_cloudevents_dict(cls, data: Dict[str, Any]) -> "MetisDataContract":
        """Create from CloudEvents standard format"""
        payload = data.get("data", {})
        return cls(
            specversion=data.get("specversion", "1.0"),
            type=data["type"],
            source=data["source"],
            id=UUID(data["id"]),
            time=datetime.fromisoformat(data["time"].replace("Z", "+00:00")),
            engagement_context=EngagementContext(**payload["engagement_context"]),
            cognitive_state=CognitiveState(**payload["cognitive_state"]),
            workflow_state=WorkflowState(**payload["workflow_state"]),
            deliverable_artifacts=[
                DeliverableArtifact(**a)
                for a in payload.get("deliverable_artifacts", [])
            ],
            schema_version=payload.get("schema_version", "1.0.0"),
            processing_metadata=payload.get("processing_metadata", {}),
        )

    def get(self, key: str, default=None):
        """Dict-like get method for backward compatibility with legacy .get() calls"""
        try:
            # First try direct attribute access
            if hasattr(self, key):
                return getattr(self, key, default)

            # Check if it's in the engagement_context.business_context
            if hasattr(self, "engagement_context") and hasattr(
                self.engagement_context, "business_context"
            ):
                bc = self.engagement_context.business_context
                if isinstance(bc, dict) and key in bc:
                    return bc[key]
                elif hasattr(bc, key):
                    return getattr(bc, key, default)

            # Check other nested contexts
            contexts_to_check = [
                "cognitive_state",
                "workflow_state",
                "processing_metadata",
            ]
            for context_name in contexts_to_check:
                if hasattr(self, context_name):
                    context = getattr(self, context_name)
                    if isinstance(context, dict) and key in context:
                        return context[key]
                    elif hasattr(context, key):
                        return getattr(context, key, default)

            return default
        except Exception:
            return default

    # Week 3.1: Memory Management Methods

    def get_memory_size_mb(self) -> float:
        """Calculate approximate memory size in MB"""
        try:
            import json
            import sys

            # Convert to dict and calculate size
            data_dict = self.dict()
            json_str = json.dumps(data_dict, default=str)
            size_bytes = sys.getsizeof(json_str)

            # Add size of large collections
            size_bytes += sys.getsizeof(self.raw_outputs)
            size_bytes += sys.getsizeof(self.integration_calls) * len(
                self.integration_calls
            )
            size_bytes += sys.getsizeof(self.analysis_results)
            size_bytes += sys.getsizeof(self.research_results)
            size_bytes += sys.getsizeof(self.validation_results)
            size_bytes += sys.getsizeof(self.verification_results) * len(
                self.verification_results
            )
            size_bytes += sys.getsizeof(self.value_assessments) * len(
                self.value_assessments
            )

            return size_bytes / (1024 * 1024)  # Convert to MB
        except Exception:
            # Fallback estimation
            return 1.0  # Assume 1MB if calculation fails

    def enforce_size_limit(self, max_size_mb: float = 50.0) -> bool:
        """Enforce memory size limit by truncating large fields"""
        current_size = self.get_memory_size_mb()

        if current_size <= max_size_mb:
            return False  # No truncation needed

        # Truncate large collections in order of priority (least important first)
        truncated = False

        # 1. Truncate raw_outputs (keep only latest entries)
        if len(self.raw_outputs) > 10:
            keys_to_keep = list(self.raw_outputs.keys())[-10:]
            self.raw_outputs = {k: self.raw_outputs[k] for k in keys_to_keep}
            truncated = True

        # 2. Truncate integration_calls (keep only latest 50)
        if len(self.integration_calls) > 50:
            self.integration_calls = self.integration_calls[-50:]
            truncated = True

        # 3. Truncate verification_results (keep only latest 20)
        if len(self.verification_results) > 20:
            self.verification_results = self.verification_results[-20:]
            truncated = True

        # 4. Truncate value_assessments (keep only latest 10)
        if len(self.value_assessments) > 10:
            self.value_assessments = self.value_assessments[-10:]
            truncated = True

        # 5. If still too large, truncate analysis_results
        current_size = self.get_memory_size_mb()
        if current_size > max_size_mb and self.analysis_results:
            # Keep only essential keys
            essential_keys = ["summary", "key_findings", "recommendations"]
            self.analysis_results = {
                k: v for k, v in self.analysis_results.items() if k in essential_keys
            }
            truncated = True

        return truncated

    def compress_for_storage(self) -> Dict[str, Any]:
        """Create compressed version for long-term storage"""
        # Create a compressed version that removes verbose data but keeps essentials
        try:
            phases_completed = (
                len(self.workflow_state.completed_phases)
                if hasattr(self.workflow_state, "completed_phases")
                else 0
            )
        except Exception:
            phases_completed = 0
        try:
            total_reasoning_steps = (
                len(self.cognitive_state.reasoning_steps)
                if hasattr(self.cognitive_state, "reasoning_steps")
                else 0
            )
        except Exception:
            total_reasoning_steps = 0
        # Derive a final confidence value if available
        final_confidence = None
        try:
            scores = getattr(self.cognitive_state, "confidence_scores", {}) or {}
            if isinstance(scores, dict):
                final_confidence = scores.get("overall")
                if final_confidence is None and len(scores) > 0:
                    final_confidence = sum(scores.values()) / len(scores)
        except Exception:
            final_confidence = None
        compressed = {
            "engagement_id": str(self.engagement_context.engagement_id),
            "problem_statement": self.engagement_context.problem_statement,
            "current_phase": self.workflow_state.current_phase.value,
            "created_at": self.time.isoformat(),
            "schema_version": self.schema_version,
            # Compressed essentials
            "summary": {
                "models_used": len(self.cognitive_state.selected_mental_models),
                "phases_completed": phases_completed,
                "total_reasoning_steps": total_reasoning_steps,
                "research_queries_made": len(self.integration_calls),
                "final_confidence": final_confidence,
            },
            # Keep only essential results
            "key_findings": self.analysis_results.get("key_findings", [])[
                :5
            ],  # Top 5 findings
            "recommendations": self.final_synthesis.get("recommendations", [])[
                :3
            ],  # Top 3 recommendations
            "confidence_scores": self.final_synthesis.get("confidence_scores", {}),
            # Metadata
            "processing_duration_ms": self.processing_metadata.get(
                "total_processing_time_ms", 0
            ),
            "memory_size_mb": self.get_memory_size_mb(),
        }
        return compressed


# Factory functions for common event types


def get_schema_version() -> str:
    """Get current schema version for compatibility checks"""
    return "2.0.0"  # Updated for V2 vulnerability solutions


# Rebuild models to resolve forward references at module import time
def _rebuild_models_with_forward_refs():
    """Rebuild Pydantic models that use forward references to resolve cross-module types"""
    try:
        # Import the engagement models to ensure they're available
        from .engagement_models import (
            EngagementContext,
            WorkflowState,
            ClarificationSession,
            DeliverableArtifact,
            ExplorationContext,
            FailureModeResponse,
        )

        # Rebuild models that reference engagement models
        VulnerabilityContext.model_rebuild()
        MetisDataContract.model_rebuild()
        PatternGovernanceResult.model_rebuild()
    except Exception:
        # If imports fail during module initialization, skip rebuild
        # Models will be rebuilt on first use
        pass


# Call rebuild at module load time
_rebuild_models_with_forward_refs()


# V2 Factory functions for vulnerability solution events


