"""
METIS Comprehensive Engagement Audit Trail
Glass-Box Transparency System for Complete Cognitive Process Tracking

This extends our existing audit trail system to capture every significant decision
and data point during a single engagement, enabling perfect reconstruction of the
system's "thought process" from beginning to end.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from uuid import UUID, uuid4

# Import existing models to build on the foundation
from src.models.transparency_models import (
    TransparencyLayer,
    UserExpertiseLevel,
    ProgressiveDisclosure,
)
from src.engine.adapters.audit_trail import AuditEventType, MetisAuditTrailManager  # Migrated


class CognitivePhase(str, Enum):
    """Phases of cognitive processing for audit tracking"""

    INGESTION = "ingestion"  # Raw query intake
    CLASSIFICATION = "classification"  # Query analysis and intent detection
    STRATEGY_SELECTION = "strategy_selection"  # N-Way cluster and consultant selection
    EXECUTION = "execution"  # Multi-step cognitive execution
    CRITIQUE = "critique"  # Optional Devil's Advocate
    ARBITRATION = "arbitration"  # Optional Senior Advisor
    COMPLETED = "completed"  # Final state


class StepExecutionStatus(str, Enum):
    """Status of individual cognitive steps"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class LLMPromptCapture:
    """Captures the exact prompts sent to LLM for complete transparency"""

    system_prompt: str
    user_prompt: str
    model_used: str
    temperature: float
    max_tokens: int
    prompt_length_tokens: int
    estimated_cost_usd: float = 0.0
    prompt_hash: str = ""  # For deduplication and caching

    def to_audit_format(self) -> Dict[str, Any]:
        """Convert to format suitable for audit storage"""
        return {
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "model_used": self.model_used,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "prompt_length_tokens": self.prompt_length_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "prompt_hash": self.prompt_hash,
        }


@dataclass
class LLMResponseCapture:
    """Captures LLM response with complete metadata"""

    raw_response: str
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int
    actual_cost_usd: float
    processing_time_seconds: float
    finish_reason: str  # "completed", "length", "content_filter", etc.
    model_version: str
    response_timestamp: datetime
    confidence_indicators: Dict[str, float] = field(default_factory=dict)

    def to_audit_format(self) -> Dict[str, Any]:
        """Convert to format suitable for audit storage"""
        return {
            "raw_response": self.raw_response,
            "completion_tokens": self.completion_tokens,
            "prompt_tokens": self.prompt_tokens,
            "total_tokens": self.total_tokens,
            "actual_cost_usd": self.actual_cost_usd,
            "processing_time_seconds": self.processing_time_seconds,
            "finish_reason": self.finish_reason,
            "model_version": self.model_version,
            "response_timestamp": self.response_timestamp.isoformat(),
            "confidence_indicators": self.confidence_indicators,
        }


@dataclass
class CognitiveStepResult:
    """Complete record of a single cognitive step execution"""

    step_id: str
    step_index: int
    consultant_role: str
    step_description: str

    # Input context
    input_context: str
    context_length_tokens: int

    # LLM interaction
    llm_prompt: LLMPromptCapture
    llm_response: LLMResponseCapture

    # Processing results
    extracted_reasoning: str
    extracted_context_for_next_step: str
    mental_models_applied: List[str] = field(default_factory=list)
    assumptions_made: List[str] = field(default_factory=list)
    evidence_sources: List[str] = field(default_factory=list)

    # Quality assessment
    confidence_score: float = 0.0
    logical_consistency_score: float = 0.0
    factual_accuracy_score: float = 0.0
    validation_flags: List[str] = field(default_factory=list)

    # Execution metadata
    status: StepExecutionStatus = StepExecutionStatus.COMPLETED
    execution_start_time: datetime = field(default_factory=datetime.utcnow)
    execution_end_time: Optional[datetime] = None
    error_message: Optional[str] = None

    def to_audit_format(self) -> Dict[str, Any]:
        """Convert to comprehensive audit format"""
        return {
            "step_metadata": {
                "step_id": self.step_id,
                "step_index": self.step_index,
                "consultant_role": self.consultant_role,
                "step_description": self.step_description,
                "status": self.status.value,
            },
            "input_context": {
                "content": self.input_context,
                "token_length": self.context_length_tokens,
            },
            "llm_interaction": {
                "prompt": self.llm_prompt.to_audit_format(),
                "response": self.llm_response.to_audit_format(),
            },
            "processing_results": {
                "extracted_reasoning": self.extracted_reasoning,
                "next_step_context": self.extracted_context_for_next_step,
                "mental_models": self.mental_models_applied,
                "assumptions": self.assumptions_made,
                "evidence_sources": self.evidence_sources,
            },
            "quality_assessment": {
                "confidence_score": self.confidence_score,
                "logical_consistency": self.logical_consistency_score,
                "factual_accuracy": self.factual_accuracy_score,
                "validation_flags": self.validation_flags,
            },
            "execution_timing": {
                "start_time": self.execution_start_time.isoformat(),
                "end_time": (
                    self.execution_end_time.isoformat()
                    if self.execution_end_time
                    else None
                ),
                "duration_seconds": (
                    (
                        self.execution_end_time - self.execution_start_time
                    ).total_seconds()
                    if self.execution_end_time
                    else None
                ),
            },
            "error_info": (
                {"error_message": self.error_message} if self.error_message else None
            ),
        }


@dataclass
class QueryClassificationDecision:
    """Captures the enhanced query classifier decision process"""

    raw_query: str

    # Classification results
    detected_intent: str
    intent_confidence: float
    complexity_level: str
    complexity_score: int
    urgency_level: str
    scope_assessment: str

    # Decision factors
    keyword_extraction_results: List[str] = field(default_factory=list)
    pattern_matching_scores: Dict[str, float] = field(default_factory=dict)
    historical_query_similarities: List[Dict[str, Any]] = field(default_factory=list)
    routing_suggestions: List[str] = field(default_factory=list)

    # Processing metadata
    classification_timestamp: datetime = field(default_factory=datetime.utcnow)
    processing_time_seconds: float = 0.0
    classifier_model_version: str = "enhanced_v1"

    def to_audit_format(self) -> Dict[str, Any]:
        """Convert to audit format for glass-box transparency"""
        return {
            "input_query": {
                "raw_query": self.raw_query,
                "query_length": len(self.raw_query),
            },
            "classification_results": {
                "intent": self.detected_intent,
                "intent_confidence": self.intent_confidence,
                "complexity_level": self.complexity_level,
                "complexity_score": self.complexity_score,
                "urgency_level": self.urgency_level,
                "scope_assessment": self.scope_assessment,
            },
            "decision_factors": {
                "keywords_extracted": self.keyword_extraction_results,
                "pattern_scores": self.pattern_matching_scores,
                "historical_similarities": self.historical_query_similarities,
                "routing_suggestions": self.routing_suggestions,
            },
            "processing_metadata": {
                "timestamp": self.classification_timestamp.isoformat(),
                "processing_time": self.processing_time_seconds,
                "model_version": self.classifier_model_version,
            },
        }


@dataclass
class ConsultantSelectionDecision:
    """Captures the predictive consultant selection process"""

    classified_query: QueryClassificationDecision

    # N-Way cluster selection
    nway_cluster_scores: Dict[str, float] = field(default_factory=dict)
    selected_nway_cluster_id: str = ""
    cluster_selection_reasoning: str = ""

    # Consultant prediction
    consultant_prediction_scores: Dict[str, float] = field(default_factory=dict)
    selected_consultants: List[str] = field(default_factory=list)
    prediction_confidence: float = 0.0
    alternative_consultant_combinations: List[Dict[str, Any]] = field(
        default_factory=list
    )

    # Machine learning factors
    historical_effectiveness_data: Dict[str, float] = field(default_factory=dict)
    pattern_matching_results: Dict[str, Any] = field(default_factory=dict)
    learning_model_version: str = "predictive_v1"

    # Processing metadata
    selection_timestamp: datetime = field(default_factory=datetime.utcnow)
    selection_processing_time: float = 0.0

    def to_audit_format(self) -> Dict[str, Any]:
        """Convert to comprehensive audit format"""
        return {
            "input_classification": self.classified_query.to_audit_format(),
            "nway_cluster_selection": {
                "cluster_scores": self.nway_cluster_scores,
                "selected_cluster": self.selected_nway_cluster_id,
                "selection_reasoning": self.cluster_selection_reasoning,
            },
            "consultant_prediction": {
                "prediction_scores": self.consultant_prediction_scores,
                "selected_consultants": self.selected_consultants,
                "prediction_confidence": self.prediction_confidence,
                "alternatives": self.alternative_consultant_combinations,
            },
            "ml_factors": {
                "historical_effectiveness": self.historical_effectiveness_data,
                "pattern_matching": self.pattern_matching_results,
                "model_version": self.learning_model_version,
            },
            "processing_metadata": {
                "timestamp": self.selection_timestamp.isoformat(),
                "processing_time": self.selection_processing_time,
            },
        }


@dataclass
class EngagementAuditTrail:
    """
    The complete, transparent record of a single METIS cognitive engagement.
    This is the definitive "Glass-Box" audit trail that captures every decision point.
    """

    # --- Phase 1: Ingestion & Classification ---
    engagement_id: UUID
    user_id: UUID
    session_id: UUID
    timestamp_start: datetime
    raw_query: str

    # Complete classification decision process
    classification_decision: Optional[QueryClassificationDecision] = None

    # --- Phase 2: Strategy & Team Selection ---
    # Complete selection decision process
    selection_decision: Optional[ConsultantSelectionDecision] = None

    # --- Phase 3: Core Execution ---
    # Dictionary where keys are consultant roles and values are lists of step results
    execution_steps: Dict[str, List[CognitiveStepResult]] = field(default_factory=dict)

    # --- Phase 4: Optional Critique & Arbitration ---
    devils_advocate_results: Dict[str, List[CognitiveStepResult]] = field(
        default_factory=dict
    )
    senior_advisor_result: Optional[CognitiveStepResult] = None

    # --- Engagement Completion ---
    current_phase: CognitivePhase = CognitivePhase.INGESTION
    final_status: str = "in_progress"  # completed, failed, cancelled, timed_out
    timestamp_end: Optional[datetime] = None

    # --- Financial & Performance Tracking ---
    total_cost_usd: float = 0.0
    total_tokens_consumed: int = 0
    total_processing_time_seconds: float = 0.0

    # --- User Interaction Tracking ---
    user_interactions: List[Dict[str, Any]] = field(default_factory=list)
    transparency_layer_access: Dict[TransparencyLayer, int] = field(
        default_factory=dict
    )
    user_feedback: Optional[Dict[str, Any]] = None

    # --- System Health & Errors ---
    error_events: List[Dict[str, Any]] = field(default_factory=list)
    performance_warnings: List[str] = field(default_factory=list)
    system_health_snapshots: List[Dict[str, Any]] = field(default_factory=list)

    def advance_phase(self, new_phase: CognitivePhase):
        """Advance to next cognitive phase with timestamp"""
        self.current_phase = new_phase
        if new_phase == CognitivePhase.COMPLETED and not self.timestamp_end:
            self.timestamp_end = datetime.utcnow()

    def record_step_result(
        self, consultant_role: str, step_result: CognitiveStepResult
    ):
        """Record a completed cognitive step"""
        if consultant_role not in self.execution_steps:
            self.execution_steps[consultant_role] = []
        self.execution_steps[consultant_role].append(step_result)

        # Update financial tracking
        self.total_cost_usd += step_result.llm_response.actual_cost_usd
        self.total_tokens_consumed += step_result.llm_response.total_tokens

        if step_result.execution_end_time:
            step_duration = (
                step_result.execution_end_time - step_result.execution_start_time
            ).total_seconds()
            self.total_processing_time_seconds += step_duration

    def record_devils_advocate_step(
        self, consultant_role: str, critique_step: CognitiveStepResult
    ):
        """Record a Devil's Advocate critique step"""
        if consultant_role not in self.devils_advocate_results:
            self.devils_advocate_results[consultant_role] = []
        self.devils_advocate_results[consultant_role].append(critique_step)

        # Update tracking
        self.total_cost_usd += critique_step.llm_response.actual_cost_usd
        self.total_tokens_consumed += critique_step.llm_response.total_tokens

    def record_user_interaction(
        self,
        interaction_type: str,
        layer: Optional[TransparencyLayer] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Record user interaction for engagement analysis"""
        interaction = {
            "timestamp": datetime.utcnow().isoformat(),
            "interaction_type": interaction_type,
            "layer": layer.value if layer else None,
            "metadata": metadata or {},
        }
        self.user_interactions.append(interaction)

        # Track transparency layer access
        if layer:
            if layer not in self.transparency_layer_access:
                self.transparency_layer_access[layer] = 0
            self.transparency_layer_access[layer] += 1

    def record_error(
        self,
        error_type: str,
        error_message: str,
        component: str,
        severity: str = "medium",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Record error event with context"""
        error_event = {
            "timestamp": datetime.utcnow().isoformat(),
            "error_type": error_type,
            "error_message": error_message,
            "component": component,
            "severity": severity,
            "metadata": metadata or {},
        }
        self.error_events.append(error_event)

    def get_engagement_summary(self) -> Dict[str, Any]:
        """Generate comprehensive engagement summary for glass-box display"""
        return {
            "engagement_metadata": {
                "engagement_id": str(self.engagement_id),
                "user_id": str(self.user_id),
                "session_id": str(self.session_id),
                "started_at": self.timestamp_start.isoformat(),
                "completed_at": (
                    self.timestamp_end.isoformat() if self.timestamp_end else None
                ),
                "current_phase": self.current_phase.value,
                "final_status": self.final_status,
            },
            "query_processing": {
                "raw_query": self.raw_query,
                "classification_results": (
                    self.classification_decision.to_audit_format()
                    if self.classification_decision
                    else None
                ),
                "selection_results": (
                    self.selection_decision.to_audit_format()
                    if self.selection_decision
                    else None
                ),
            },
            "execution_overview": {
                "consultants_executed": list(self.execution_steps.keys()),
                "total_steps_executed": sum(
                    len(steps) for steps in self.execution_steps.values()
                ),
                "devils_advocate_used": len(self.devils_advocate_results) > 0,
                "senior_advisor_used": self.senior_advisor_result is not None,
            },
            "performance_metrics": {
                "total_cost_usd": self.total_cost_usd,
                "total_tokens": self.total_tokens_consumed,
                "processing_time_seconds": self.total_processing_time_seconds,
                "average_step_confidence": self._calculate_average_confidence(),
            },
            "user_engagement": {
                "total_interactions": len(self.user_interactions),
                "transparency_access_pattern": {
                    k.value: v for k, v in self.transparency_layer_access.items()
                },
                "user_feedback": self.user_feedback,
            },
            "system_health": {
                "errors_encountered": len(self.error_events),
                "warnings_raised": len(self.performance_warnings),
                "health_snapshots": len(self.system_health_snapshots),
            },
        }

    def _calculate_average_confidence(self) -> float:
        """Calculate average confidence across all steps"""
        all_steps = []
        for consultant_steps in self.execution_steps.values():
            all_steps.extend(consultant_steps)
        for critique_steps in self.devils_advocate_results.values():
            all_steps.extend(critique_steps)
        if self.senior_advisor_result:
            all_steps.append(self.senior_advisor_result)

        if not all_steps:
            return 0.0

        return sum(step.confidence_score for step in all_steps) / len(all_steps)

    def to_progressive_disclosure(
        self, user_expertise: UserExpertiseLevel
    ) -> ProgressiveDisclosure:
        """Convert audit trail to progressive disclosure format for frontend display"""
        from src.models.transparency_models import (
            TransparencyContent,
            CognitiveLoadLevel,
        )

        disclosure = ProgressiveDisclosure(engagement_id=self.engagement_id)

        # Executive Summary Layer
        executive_content = self._generate_executive_summary()
        disclosure.layers[TransparencyLayer.EXECUTIVE_SUMMARY] = TransparencyContent(
            layer=TransparencyLayer.EXECUTIVE_SUMMARY,
            title="Engagement Executive Summary",
            content=executive_content,
            cognitive_load=CognitiveLoadLevel.LOW,
            key_insights=self._extract_key_insights(),
            reading_time_estimate=2,
        )

        # Reasoning Overview Layer
        reasoning_content = self._generate_reasoning_overview()
        disclosure.layers[TransparencyLayer.REASONING_OVERVIEW] = TransparencyContent(
            layer=TransparencyLayer.REASONING_OVERVIEW,
            title="Cognitive Process Overview",
            content=reasoning_content,
            cognitive_load=CognitiveLoadLevel.MEDIUM,
            reading_time_estimate=5,
        )

        # Detailed Audit Trail Layer
        if user_expertise in [
            UserExpertiseLevel.ANALYTICAL,
            UserExpertiseLevel.TECHNICAL,
        ]:
            audit_content = self._generate_detailed_audit()
            disclosure.layers[TransparencyLayer.DETAILED_AUDIT_TRAIL] = (
                TransparencyContent(
                    layer=TransparencyLayer.DETAILED_AUDIT_TRAIL,
                    title="Complete Cognitive Audit Trail",
                    content=audit_content,
                    cognitive_load=CognitiveLoadLevel.HIGH,
                    reading_time_estimate=15,
                )
            )

        # Technical Execution Layer (for technical users only)
        if user_expertise == UserExpertiseLevel.TECHNICAL:
            technical_content = self._generate_technical_execution_details()
            disclosure.layers[TransparencyLayer.TECHNICAL_EXECUTION] = (
                TransparencyContent(
                    layer=TransparencyLayer.TECHNICAL_EXECUTION,
                    title="Technical Execution Details",
                    content=technical_content,
                    cognitive_load=CognitiveLoadLevel.HIGH,
                    reading_time_estimate=20,
                )
            )

        return disclosure

    def _generate_executive_summary(self) -> str:
        """Generate executive summary of the engagement"""
        return f"""
        ## Engagement Summary
        
        **Query**: {self.raw_query[:100]}{'...' if len(self.raw_query) > 100 else ''}
        
        **Analysis Approach**: {self.selection_decision.selected_nway_cluster_id if self.selection_decision else 'Standard Analysis'}
        
        **Consultants Engaged**: {', '.join(self.execution_steps.keys())}
        
        **Processing Time**: {self.total_processing_time_seconds:.1f} seconds
        
        **Cost**: ${self.total_cost_usd:.4f}
        
        **Status**: {self.final_status.title()}
        """

    def _generate_reasoning_overview(self) -> str:
        """Generate reasoning process overview"""
        steps_summary = []
        for consultant, steps in self.execution_steps.items():
            steps_summary.append(f"**{consultant}**: {len(steps)} reasoning steps")

        return f"""
        ## Cognitive Process Overview
        
        ### Query Classification
        - **Intent**: {self.classification_decision.detected_intent if self.classification_decision else 'Unknown'}
        - **Complexity**: {self.classification_decision.complexity_level if self.classification_decision else 'Unknown'}
        
        ### Consultant Selection
        - **Strategy Used**: {self.selection_decision.selected_nway_cluster_id if self.selection_decision else 'Default'}
        - **Selection Confidence**: {self.selection_decision.prediction_confidence if self.selection_decision else 0:.2f}
        
        ### Execution Summary
        {chr(10).join(steps_summary)}
        
        ### Optional Enhancements
        - **Devil's Advocate**: {'Used' if self.devils_advocate_results else 'Not Used'}
        - **Senior Advisor**: {'Used' if self.senior_advisor_result else 'Not Used'}
        """

    def _generate_detailed_audit(self) -> str:
        """Generate detailed audit trail"""
        return f"""
        ## Complete Cognitive Audit Trail
        
        This section contains the complete step-by-step audit trail of the cognitive process.
        
        **Total Steps**: {sum(len(steps) for steps in self.execution_steps.values())}
        
        **Average Confidence**: {self._calculate_average_confidence():.2f}
        
        **Errors Encountered**: {len(self.error_events)}
        
        [Detailed step-by-step breakdown would be rendered here in the frontend]
        """

    def _generate_technical_execution_details(self) -> str:
        """Generate technical execution details"""
        return f"""
        ## Technical Execution Details
        
        ### Resource Consumption
        - **Total Tokens**: {self.total_tokens_consumed:,}
        - **Total Cost**: ${self.total_cost_usd:.6f}
        - **Average Tokens per Step**: {self.total_tokens_consumed // max(1, sum(len(steps) for steps in self.execution_steps.values())):.0f}
        
        ### Performance Metrics
        - **Processing Time**: {self.total_processing_time_seconds:.2f}s
        - **Warnings**: {len(self.performance_warnings)}
        - **Health Snapshots**: {len(self.system_health_snapshots)}
        
        [Complete technical details would be rendered here in the frontend]
        """

    def _extract_key_insights(self) -> List[str]:
        """Extract key insights from the engagement"""
        insights = [
            f"Processed {self.classification_decision.detected_intent if self.classification_decision else 'unknown'} query with {self.classification_decision.complexity_level if self.classification_decision else 'unknown'} complexity",
            f"Engaged {len(self.execution_steps)} consultant(s) for analysis",
            f"Achieved {self._calculate_average_confidence():.0f}% average confidence",
            f"Completed in {self.total_processing_time_seconds:.1f} seconds",
        ]
        return insights


# Integration with existing audit system
async def create_engagement_audit_trail(
    user_id: UUID,
    session_id: UUID,
    raw_query: str,
    audit_manager: Optional[MetisAuditTrailManager] = None,
) -> EngagementAuditTrail:
    """Create new engagement audit trail and log initial event"""

    engagement_id = uuid4()
    trail = EngagementAuditTrail(
        engagement_id=engagement_id,
        user_id=user_id,
        session_id=session_id,
        timestamp_start=datetime.utcnow(),
        raw_query=raw_query,
    )

    # Log initial engagement creation event to existing audit system
    if audit_manager:
        await audit_manager.log_event(
            event_type=AuditEventType.ENGAGEMENT_CREATED,
            severity=AuditSeverity.MEDIUM,
            user_id=user_id,
            session_id=session_id,
            engagement_id=engagement_id,
            resource_type="engagement_audit_trail",
            resource_id=engagement_id,
            action_performed="create_engagement",
            event_description="Created comprehensive audit trail for engagement",
            metadata={
                "query_length": len(raw_query),
                "audit_trail_version": "comprehensive_v1",
            },
        )

    return trail


# Utility functions for audit trail management
async def save_engagement_audit_trail(trail: EngagementAuditTrail) -> bool:
    """Save complete engagement audit trail to persistent storage"""
    # This would integrate with Supabase to store the complete audit trail
    # For now, we'll log it as a comprehensive audit event

    try:
        from src.engine.adapters.audit_trail import get_audit_manager  # Migrated

        audit_manager = await get_audit_manager()

        # Save the complete trail as a comprehensive audit event
        await audit_manager.log_event(
            event_type=AuditEventType.ANALYSIS_COMPLETED,
            severity=AuditSeverity.MEDIUM,
            user_id=trail.user_id,
            session_id=trail.session_id,
            engagement_id=trail.engagement_id,
            resource_type="engagement_audit_trail",
            resource_id=trail.engagement_id,
            action_performed="complete_engagement_audit",
            event_description=f"Completed comprehensive engagement with {sum(len(steps) for steps in trail.execution_steps.values())} steps",
            metadata={
                "engagement_summary": trail.get_engagement_summary(),
                "total_cost": trail.total_cost_usd,
                "total_tokens": trail.total_tokens_consumed,
                "processing_time": trail.total_processing_time_seconds,
                "final_status": trail.final_status,
            },
        )

        return True

    except Exception as e:
        print(f"Failed to save engagement audit trail: {e}")
        return False


async def load_engagement_audit_trail(
    engagement_id: UUID,
) -> Optional[EngagementAuditTrail]:
    """Load engagement audit trail from persistent storage"""
    # This would load from Supabase in production
    # For now, we'll return None to indicate not implemented
    return None


def format_audit_trail_for_export(
    trail: EngagementAuditTrail, format: str = "json"
) -> Dict[str, Any]:
    """Format audit trail for export/compliance reporting"""

    export_data = {
        "audit_trail_metadata": {
            "engagement_id": str(trail.engagement_id),
            "export_timestamp": datetime.utcnow().isoformat(),
            "format_version": "comprehensive_v1",
            "total_data_points": (
                (1 if trail.classification_decision else 0)
                + (1 if trail.selection_decision else 0)
                + sum(len(steps) for steps in trail.execution_steps.values())
                + sum(len(steps) for steps in trail.devils_advocate_results.values())
                + (1 if trail.senior_advisor_result else 0)
                + len(trail.user_interactions)
                + len(trail.error_events)
            ),
        },
        "engagement_data": trail.get_engagement_summary(),
        "detailed_audit_trail": {
            "query_classification": (
                trail.classification_decision.to_audit_format()
                if trail.classification_decision
                else None
            ),
            "consultant_selection": (
                trail.selection_decision.to_audit_format()
                if trail.selection_decision
                else None
            ),
            "execution_steps": {
                consultant: [step.to_audit_format() for step in steps]
                for consultant, steps in trail.execution_steps.items()
            },
            "devils_advocate_results": {
                consultant: [step.to_audit_format() for step in steps]
                for consultant, steps in trail.devils_advocate_results.items()
            },
            "senior_advisor_result": (
                trail.senior_advisor_result.to_audit_format()
                if trail.senior_advisor_result
                else None
            ),
        },
        "user_interactions": trail.user_interactions,
        "error_events": trail.error_events,
        "compliance_data": {
            "retention_period_days": 2555,  # 7 years for SOC 2
            "data_classification": "internal",
            "gdpr_lawful_basis": "legitimate_interest",
            "export_authorized_by": "system_audit",
        },
    }

    return export_data
