"""
UnifiedContextStream - V4 Single Agent Core Component
Central event-driven context management with optimal token usage

This module implements the backbone of the V4 architecture:
1. Append-only event stream (no data loss)
2. Full trace preservation
3. Intelligent relevance scoring
4. Incremental updates only
5. Cache-optimized structure
6. XML formatting for 40% token reduction
"""

import json
import hashlib
import os
import re
import uuid
from typing import Dict, List, Any, Optional, Callable, TYPE_CHECKING
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4
import logging

# Avoid import cycles: only type-import at type-check time
if TYPE_CHECKING:  # pragma: no cover
    from src.core.persistence.contracts import IEventPersistence
else:
    IEventPersistence = Any  # type: ignore

# Service layer imports (Operation Lean - Task 8.0)
from src.core.services.event_validation_service import EventValidationService
from src.core.services.evidence_extraction_service import EvidenceExtractionService
from src.core.services.context_formatting_service import ContextFormattingService
from src.core.services.context_persistence_service import ContextPersistenceService
from src.core.services.context_metrics_service import ContextMetricsService

logger = logging.getLogger(__name__)


class ContextEventType(Enum):
    """Types of events in the context stream"""

    # System Events
    ENGAGEMENT_STARTED = "engagement_started"
    ENGAGEMENT_COMPLETED = "engagement_completed"
    AGENT_INSTANTIATED = "agent_instantiated"
    PHASE_COMPLETED = "phase_completed"
    CHECKPOINT_SAVED = "checkpoint_saved"
    QUERY_RECEIVED = "query_received"

    # Research Events
    RESEARCH_QUERY = "research_query"
    RESEARCH_RESULT = "research_result"
    SOURCE_VALIDATED = "source_validated"
    ORACLE_RESEARCH_COMPLETE = "oracle_research_complete"

    # Cognitive Events
    REASONING_STEP = "reasoning_step"
    MODEL_APPLIED = "model_applied"
    SYNTHESIS_CREATED = "synthesis_created"
    REASONING_MODE_DECISION = "reasoning_mode_decision"  # Phase 3: Automatic reasoning triage

    # V2.1 Master Communicator Events
    SOCRATIC_QUESTIONS_GENERATED = "socratic_questions_generated"
    STRUCTURED_FRAMEWORK_CREATED = "structured_framework_created"
    CONSULTANT_SELECTION_COMPLETE = "consultant_selection_complete"
    PROMPT_ASSEMBLY_COMPLETE = "prompt_assembly_complete"
    DEVILS_ADVOCATE_COMPLETE = "devils_advocate_complete"
    SENIOR_ADVISOR_COMPLETE = "senior_advisor_complete"

    # OPERATION INTELLIGENT DISPATCH Events
    INTELLIGENT_DISPATCH_INFO = "intelligent_dispatch_info"
    CONTEXTUAL_CONSULTANT_SELECTION_V1 = "contextual_consultant_selection_v1"

    # OPERATION ADAPTIVE ORCHESTRATION Events
    TASK_CLASSIFICATION_STARTED = "task_classification_started"
    TASK_CLASSIFICATION_COMPLETE = "task_classification_complete"
    ADAPTIVE_TEAM_COMPOSITION = "adaptive_team_composition"

    # Contracts & QA Events (Systems-First additions)
    CONTRACT_ECHO = "contract_echo"
    QA_SELF_CHECK = "qa_self_check"

    # Tool Execution Events
    TOOL_EXECUTION = "tool_execution"
    TOOL_DECISION = "tool_decision"

    # Human Interaction Events  
    QUERY_ENHANCED_FROM_CLARIFICATION = "query_enhanced_from_clarification"
    CLARIFICATION_ANSWERS_PROCESSED = "clarification_answers_processed"
    HITL_REQUEST = "hitl_request"
    HITL_RESPONSE = "hitl_response"
    HITL_OVERSIGHT_STARTED = "hitl_oversight_started"
    HITL_OVERSIGHT_COMPLETED = "hitl_oversight_completed"
    CLARIFICATION = "clarification"
    HUMAN_INTERACTION = "human_interaction"

    # Error Events
    ERROR = "error"
    ERROR_OCCURRED = "error_occurred"
    ERROR_RECOVERED = "error_recovered"

    # Glass-Box Transparency Events (Phase 4 Requirements)
    SOCRATIC_ENGINE_LLM_CALL_START = "socratic_engine_llm_call_start"
    SOCRATIC_ENGINE_LLM_CALL_COMPLETE = "socratic_engine_llm_call_complete"
    CONSULTANT_SELECTION_PREDICTIVE_RESULT = "consultant_selection_predictive_result"
    NWAY_CLUSTER_ACTIVATED = "nway_cluster_activated"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_COMPLETE = "tool_call_complete"
    DEVILS_ADVOCATE_BIAS_FOUND = "devils_advocate_bias_found"
    SENIOR_ADVISOR_TENSION_IDENTIFIED = "senior_advisor_tension_identified"
    LLM_PROVIDER_REQUEST = "llm_provider_request"
    LLM_PROVIDER_RESPONSE = "llm_provider_response"
    LLM_PROVIDER_FALLBACK = "llm_provider_fallback"
    LLM_CALL_COMPLETE = "llm_call_complete"
    RESEARCH_PROVIDER_REQUEST = "research_provider_request"
    RESEARCH_PROVIDER_RESPONSE = "research_provider_response"
    RESEARCH_PROVIDER_FALLBACK = "research_provider_fallback"
    RESEARCH_GROUNDING_START = "research_grounding_start"
    RESEARCH_GROUNDING_COMPLETE = "research_grounding_complete"
    RESEARCH_BRIEF_ATTACHED = "research_brief_attached"
    CONSULTANT_ANALYSIS_START = "consultant_analysis_start"
    CONSULTANT_ANALYSIS_COMPLETE = "consultant_analysis_complete"
    COGNITIVE_DIVERSITY_CALCULATED = "cognitive_diversity_calculated"
    CONTEXT_PRESERVATION_VALIDATED = "context_preservation_validated"

    # Parallel Analysis specific events (Live Fire instrumentation)
    PARALLEL_ANALYSIS_PROMPT_GENERATED = "parallel_analysis_prompt_generated"
    CONSULTANT_MEMO_PRODUCED = "consultant_memo_produced"
    STAGE0_EXPERIMENT_ASSIGNED = "stage0_experiment_assigned"
    STAGE0_PLAN_RECORDED = "stage0_plan_recorded"
    DEPTH_ENRICHMENT_APPLIED = "depth_enrichment_applied"
    DEPTH_ENRICHMENT_METRICS = "depth_enrichment_metrics"
    # Operation Hardening - Orthogonality Watchdog
    ORTHOGONALITY_INDEX_COMPUTED = "orthogonality_index_computed"
    DIVERSITY_WATCHDOG_TRIGGERED = "diversity_watchdog_triggered"
    MINORITY_REPORT_GENERATED = "minority_report_generated"
    STAGE_PERFORMANCE_PROFILE_RECORDED = "stage_performance_profile_recorded"

    # Parallel Analysis briefing overlays (Meaningful Diversity)
    BRIEFING_MUTATION_APPLIED = "briefing_mutation_applied"

    # System 2 Attention (S2A) Dimension Filtering (Phase 2: Cognitive Diversity)
    S2A_DIMENSION_FILTER_APPLIED = "s2a_dimension_filter_applied"

    # Senior Advisor probability reporting (Meaningful Odds)
    PROBABILITY_ASSESSMENT_REPORTED = "probability_assessment_reported"

    # V2 Architecture Events
    SYSTEM_STATE = "system_state"
    CONSULTANT_SELECTION = "consultant_selection"
    PROCESSING_STARTED = "processing_started"
    PROCESSING_COMPLETE = "processing_complete"

    # Context Merge Events (for StatefulPipelineOrchestrator)
    CONTEXT_MERGE = "context_merge"
    ANALYSIS_COMPLETE = "analysis_complete"

    # Breadth Mode Events (V5.4)
    BREADTH_MODE_ELIGIBLE = "breadth_mode_eligible"
    WAVE_STARTED = "wave_started"
    POLYGON_PRESERVED = "polygon_preserved"
    BREADTH_MODE_SYNTHESIS_POLYGON_PRESERVED = (
        "breadth_mode_synthesis_polygon_preserved"
    )

    # Phase 5: Context Engineering Events
    CONTEXT_ENGINEERING_STARTED = "context_engineering_started"
    CONTEXT_COMPILED = "context_compiled"
    CONTEXT_OPTIMIZATION_APPLIED = "context_optimization_applied"

    # Phase 6: Pipeline Stage and Reflection Events
    PIPELINE_STAGE_STARTED = "pipeline_stage_started"
    PIPELINE_STAGE_COMPLETED = "pipeline_stage_completed"
    PIPELINE_REFLECTION_TRIGGERED = "pipeline_reflection_triggered"

    # OPERATION BEDROCK: Modular parallel analysis events
    PROMPTS_BUILT = "prompts_built"
    LLM_EXECUTION_COMPLETE = "llm_execution_complete"
    AGGREGATION_COMPLETE = "aggregation_complete"
    STAGE0_ENRICHMENT_COMPLETE = "stage0_enrichment_complete"

    # 🚨 OPERATION "CAPTURE THE ARTIFACT" - Final Report Events
    FINAL_REPORT_GENERATED = "final_report_generated"

    # 🚨 DEEP INSTRUMENTATION: Devils Advocate & Contradiction Tracking Events
    DEVILS_ADVOCATE_ANALYSIS_START = "devils_advocate_analysis_start"
    DEVILS_ADVOCATE_ANALYSIS_COMPLETE = "devils_advocate_analysis_complete"
    CONTRADICTION_DETECTED = "contradiction_detected"

    # 🔍 GLASS-BOX EVIDENCE PIPELINE: Auditable Decision Trail Events
    MODEL_SELECTION_JUSTIFICATION = "model_selection_justification"
    SYNERGY_META_DIRECTIVE = "synergy_meta_directive"
    COREOPS_RUN_SUMMARY = "coreops_run_summary"
    # Canary Prompt Policy Events
    PROMPT_POLICY_VARIANT_ASSIGNED = "prompt_policy_variant_assigned"
    PROMPT_POLICY_EVALUATION_SUMMARY = "prompt_policy_evaluation_summary"
    COREOPS_STEP_EXECUTED = (
        "coreops_step_executed"  # Phase 7: CoreOps step-level tracking
    )
    CONTRADICTION_AUDIT = "contradiction_audit"
    MENTAL_MODEL_ACTIVATION = "mental_model_activation"
    EVIDENCE_COLLECTION_COMPLETE = "evidence_collection_complete"

    # Phase 8: Learning Systems Events (Performance, Optimization, Feedback)
    LEARNING_CYCLE_STARTED = "learning_cycle_started"
    LEARNING_CYCLE_COMPLETED = "learning_cycle_completed"
    PATTERN_EFFECTIVENESS_UPDATE = "pattern_effectiveness_update"
    DIVERSITY_POLICY_ENFORCED = "diversity_policy_enforced"
    OPTIMIZATION_ACTION_TAKEN = "optimization_action_taken"
    FEEDBACK_INGESTED = "feedback_ingested"
    DASHBOARD_METRICS_UPDATED = "dashboard_metrics_updated"

    # 🎭 METHOD ACTOR DEVILS ADVOCATE: Enhanced Evidence Events
    DEVILS_ADVOCATE_METHOD_ACTOR_COMPLETE = "devils_advocate_method_actor_complete"
    ENABLING_CHALLENGER_DIALOGUE_GENERATED = "enabling_challenger_dialogue_generated"
    FORWARD_MOTION_ACTIONS_CREATED = "forward_motion_actions_created"
    ANTI_FAILURE_SAFEGUARDS_ACTIVATED = "anti_failure_safeguards_activated"


@dataclass
class ContextEvent:
    """Individual event in the context stream"""

    event_id: str = field(default_factory=lambda: str(uuid4()))
    event_type: ContextEventType = field(default=ContextEventType.ENGAGEMENT_STARTED)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Relevance and optimization fields
    relevance_score: float = 1.0  # 0.0 to 1.0
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    can_compress: bool = False
    compressed_version: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "event_id": self.event_id,
            "event_type": getattr(self.event_type, "value", self.event_type),
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "metadata": self.metadata,
            "relevance_score": self.relevance_score,
        }

    def calculate_hash(self) -> str:
        """Calculate hash for deduplication and caching"""
        content = f"{self.event_type.value}:{json.dumps(self.data, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()[:16]


class UnifiedContextStream:
    """
    Unified context stream implementing best practices:
    1. Append-only event stream (no data loss)
    2. Full trace preservation
    3. Intelligent relevance scoring
    4. Incremental updates only
    5. Cache-optimized structure
    """

    def __init__(
        self,
        max_events: int = 10000,
        persistence_adapter: "IEventPersistence" = None,  # type: ignore[name-defined]
        pii_redaction_enabled: bool = True,
        trace_id: Optional[str] = None,
    ):
        """
        Initialize UnifiedContextStream with optional PII redaction.

        Args:
            max_events: Maximum events to keep in memory
            persistence_adapter: Optional persistence adapter
            pii_redaction_enabled: Whether to redact PII from events (default: True)
        """
        from src.core.persistence.contracts import (
            IEventPersistence,
        )  # local import to avoid import cycles

        self.events: List[ContextEvent] = []
        self.max_events = max_events
        self.event_index: Dict[str, ContextEvent] = {}
        self.relevance_threshold = 0.3

        # PII Redaction (Phase 6 - Enterprise Security)
        self.pii_redaction_enabled = pii_redaction_enabled
        if self.pii_redaction_enabled:
            try:
                from src.engine.security.pii_redaction import get_pii_redaction_engine
                self.pii_engine = get_pii_redaction_engine(enabled=True)
                logger.info("✅ PII redaction enabled for UnifiedContextStream")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize PII redaction: {e}")
                self.pii_engine = None
        else:
            self.pii_engine = None

        # PLATFORM HARDENING: Generate unique trace ID for this stream instance (allow override for tests)
        self.trace_id = trace_id or str(uuid.uuid4())

        # Engagement metadata for database persistence
        self.session_id: Optional[str] = None
        self.user_id: Optional[str] = None
        self.organization_id: Optional[str] = None
        self.engagement_type: str = "consultation"
        self.case_id: Optional[str] = None
        self.started_at: datetime = datetime.now(timezone.utc)
        self.completed_at: Optional[datetime] = None

        # OPERATION AWAKENING: Store final analysis text for offline evaluation
        self.final_analysis_text: Optional[str] = None

        # GOVERNANCE V2: Default agent context for automatic metadata enrichment
        self.default_agent_contract_id: Optional[str] = None
        self.default_agent_instance_id: Optional[str] = None

        # Event subscribers for reactive updates
        self.subscribers: Dict[ContextEventType, List[Callable]] = {}

        # Cache for formatted contexts
        self.formatted_cache: Dict[str, str] = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # Persistence adapter (injected)
        self.persistence_adapter: Optional[IEventPersistence] = persistence_adapter

        # OPERATION BEDROCK: Strict event schema validation
        self.strict_validation_enabled = os.getenv("EVENT_SCHEMA_STRICT", "false").lower() in {"true", "1", "yes"}
        self.event_allowlist: Dict[str, List[str]] = {}
        if self.strict_validation_enabled:
            self._load_event_allowlist()

        # OPERATION LEAN: Initialize service layer (Task 8.0)
        self._init_services()

        logger.info(
            f"🌊 UnifiedContextStream initialized with trace_id: {self.trace_id}, strict_validation: {self.strict_validation_enabled}"
        )

    def _init_services(self) -> None:
        """Initialize service layer instances (Operation Lean - Task 8.0)"""
        from pathlib import Path

        # EventValidationService
        allowlist_path = Path(__file__).parent / "event_allowlist.yaml" if self.strict_validation_enabled else None
        self._validation_service = EventValidationService(
            allowlist_path=allowlist_path,
            pii_engine=self.pii_engine,
            strict_validation_enabled=self.strict_validation_enabled
        )

        # EvidenceExtractionService (lazily instantiated when needed)
        self._evidence_service: Optional[EvidenceExtractionService] = None

        # ContextFormattingService
        self._formatting_service = ContextFormattingService()

        # ContextPersistenceService (lazily instantiated when needed)
        self._persistence_service: Optional[ContextPersistenceService] = None

        # ContextMetricsService
        self._metrics_service = ContextMetricsService(
            events=self.events,
            event_index=self.event_index,
            max_events=self.max_events
        )

        logger.debug("✅ Service layer initialized")

    def _load_event_allowlist(self) -> None:
        """Load event transition allowlist from YAML file"""
        import yaml
        from pathlib import Path

        allowlist_path = Path(__file__).parent / "event_allowlist.yaml"
        try:
            with open(allowlist_path, "r") as f:
                self.event_allowlist = yaml.safe_load(f) or {}
            logger.info(f"✅ Loaded event allowlist with {len(self.event_allowlist)} event types")
        except Exception as e:
            logger.warning(f"⚠️  Failed to load event allowlist: {e}. Strict validation disabled.")
            self.strict_validation_enabled = False

    def _validate_event_schema(self, event_type: ContextEventType, data: Dict[str, Any]) -> bool:
        """Validate event payload against schema (Operation Lean - delegated to EventValidationService)"""
        return self._validation_service.validate_event_schema(event_type, data)

    def _validate_event_transition(self, event_type: ContextEventType) -> bool:
        """Validate event transition based on allowlist (Operation Lean - delegated to EventValidationService)"""
        previous_event_type = self.events[-1].event_type if self.events else None
        return self._validation_service.validate_event_transition(event_type, previous_event_type)

    # GOVERNANCE V2: Agent context setters
    def set_agent_context(
        self,
        agent_contract_id: Optional[str] = None,
        agent_instance_id: Optional[str] = None,
    ) -> None:
        self.default_agent_contract_id = agent_contract_id
        self.default_agent_instance_id = agent_instance_id

    def clear_agent_context(self) -> None:
        self.default_agent_contract_id = None
        self.default_agent_instance_id = None

    def _scrub_pii(self, data_string: str) -> str:
        """Scrub PII from string (Operation Lean - delegated to EventValidationService)"""
        return self._validation_service.scrub_pii(data_string)

    def _scrub_structure(self, value: Any) -> Any:
        """Recursively scrub PII from nested structures (Operation Lean - delegated to EventValidationService)"""
        return self._validation_service.scrub_structure(value)

    def add_event(
        self,
        event_type: ContextEventType,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> ContextEvent:
        """
        Add event to stream with automatic relevance scoring, trace ID, and PII scrubbing

        Enhanced with TOKENOMICS TELEMETRY for Operation: Certificate of Health
        Automatically tracks context token consumption for critical payload events

        Key Insight: Every action creates an immutable event that can be replayed

        Args:
            event_type: Type of event to add
            data: Event data dictionary
            metadata: Optional metadata dictionary
            timestamp: Optional specific timestamp (defaults to current time)
        """
        # OPERATION BEDROCK: Strict event validation (if enabled)
        if self.strict_validation_enabled:
            # Validate event schema
            if not self._validate_event_schema(event_type, data):
                logger.error(f"❌ Event schema validation failed for {event_type.value}, skipping event")
                # Return dummy event to maintain backward compatibility
                return ContextEvent(
                    event_type=event_type,
                    data={"error": "schema_validation_failed"},
                    timestamp=timestamp or datetime.now(timezone.utc),
                    metadata={"validation_failed": True},
                )

            # Validate event transition
            if not self._validate_event_transition(event_type):
                logger.error(f"❌ Event transition validation failed for {event_type.value}, skipping event")
                return ContextEvent(
                    event_type=event_type,
                    data={"error": "transition_validation_failed"},
                    timestamp=timestamp or datetime.now(timezone.utc),
                    metadata={"validation_failed": True},
                )

        # PLATFORM HARDENING: Ensure metadata exists and add trace_id
        if metadata is None:
            metadata = {}
        metadata = metadata.copy()  # Don't modify the original
        metadata["trace_id"] = self.trace_id
        metadata["pii_scrubbed"] = True

        # GOVERNANCE V2: Auto-enrich metadata with agent identifiers if available
        if self.default_agent_contract_id and "agent_contract_id" not in metadata:
            metadata["agent_contract_id"] = self.default_agent_contract_id
        if self.default_agent_instance_id and "agent_instance_id" not in metadata:
            metadata["agent_instance_id"] = self.default_agent_instance_id

        # PLATFORM HARDENING: Scrub PII from data and metadata
        safe_data = self._scrub_structure(data)
        safe_metadata = self._scrub_structure(metadata)

        # OPERATION: CERTIFICATE OF HEALTH - TOKENOMICS TELEMETRY
        # Automatically add context_token_count to critical payload events for Glass-Box observability
        CRITICAL_PAYLOAD_EVENTS = {
            ContextEventType.LLM_PROVIDER_REQUEST,
            ContextEventType.LLM_PROVIDER_RESPONSE,
            ContextEventType.LLM_PROVIDER_FALLBACK,
            ContextEventType.STRUCTURED_FRAMEWORK_CREATED,
            ContextEventType.SOCRATIC_QUESTIONS_GENERATED,
            ContextEventType.DEVILS_ADVOCATE_COMPLETE,
            ContextEventType.SENIOR_ADVISOR_COMPLETE,
            ContextEventType.CONSULTANT_SELECTION_COMPLETE,
            ContextEventType.NWAY_CLUSTER_ACTIVATED,
            ContextEventType.REASONING_STEP,
            ContextEventType.SYNTHESIS_CREATED,
        }

        if event_type in CRITICAL_PAYLOAD_EVENTS:
            try:
                from src.engine.utils.token_utils import count_tokens
                import json

                # Serialize the data payload to calculate token count
                payload_str = json.dumps(safe_data, default=str, sort_keys=True)
                token_count = count_tokens(payload_str)
                safe_metadata["context_token_count"] = token_count

                logger.debug(
                    f"🔢 Added tokenomics telemetry: {token_count} tokens for {event_type.value}"
                )

            except Exception as e:
                # Failsafe - don't break event logging if token counting fails
                safe_metadata["context_token_count"] = -1
                safe_metadata["tokenomics_error"] = str(e)
                logger.warning(f"⚠️ Token counting failed for {event_type.value}: {e}")

        event = ContextEvent(
            event_type=event_type,
            timestamp=timestamp or datetime.now(timezone.utc),
            data=safe_data,
            metadata=safe_metadata,
            relevance_score=self._calculate_initial_relevance(event_type),
        )

        # Add to stream
        self.events.append(event)
        self.event_index[event.event_id] = event

        # Notify subscribers
        self._notify_subscribers(event)

        # Manage stream size
        if len(self.events) > self.max_events:
            self._compress_old_events()

        # Invalidate cache
        self.formatted_cache.clear()

        logger.debug(
            f"📝 Added {getattr(event_type, 'value', event_type)} event: {event.event_id[:8]}"
        )
        return event

    def get_relevant_context(
        self, for_phase: Optional[str] = None, min_relevance: float = 0.3
    ) -> List[ContextEvent]:
        """Get relevant events (Operation Lean - delegated to ContextMetricsService)"""
        return self._metrics_service.get_relevant_context(for_phase, min_relevance)

    def get_recent_events(self, limit: int = 10) -> List[ContextEvent]:
        """Get recent events (Operation Lean - delegated to ContextMetricsService)"""
        return self._metrics_service.get_recent_events(limit)

    def get_events(self) -> List[ContextEvent]:
        """
        Get all events from the stream

        Returns:
            List of all events in chronological order
        """
        return self.events.copy()  # Return a copy to prevent external modification

    # 🔍 GLASS-BOX EVIDENCE ENHANCEMENT: Evidence-specific query methods

    def get_evidence_events(
        self, evidence_types: Optional[List[ContextEventType]] = None
    ) -> List[ContextEvent]:
        """
        Get all glass-box evidence events

        Args:
            evidence_types: Optional list of specific evidence types to filter

        Returns:
            List of evidence events in chronological order
        """
        # Default evidence event types
        if evidence_types is None:
            evidence_types = [
                ContextEventType.MODEL_SELECTION_JUSTIFICATION,
                ContextEventType.SYNERGY_META_DIRECTIVE,
                ContextEventType.COREOPS_RUN_SUMMARY,
                ContextEventType.CONTRADICTION_AUDIT,
                ContextEventType.MENTAL_MODEL_ACTIVATION,
                ContextEventType.EVIDENCE_COLLECTION_COMPLETE,
            ]

        evidence_events = [
            event for event in self.events if event.event_type in evidence_types
        ]

        return sorted(evidence_events, key=lambda e: e.timestamp)

    def get_consultant_selection_evidence(self) -> List[ContextEvent]:
        """Get all consultant selection evidence events"""
        return [
            event
            for event in self.events
            if event.event_type == ContextEventType.MODEL_SELECTION_JUSTIFICATION
        ]

    def get_synergy_evidence(self) -> List[ContextEvent]:
        """Get all mental model synergy evidence events"""
        return [
            event
            for event in self.events
            if event.event_type == ContextEventType.SYNERGY_META_DIRECTIVE
        ]

    def get_coreops_evidence(self) -> List[ContextEvent]:
        """Get all V2 CoreOps execution evidence events"""
        return [
            event
            for event in self.events
            if event.event_type == ContextEventType.COREOPS_RUN_SUMMARY
        ]

    def get_contradiction_evidence(self) -> List[ContextEvent]:
        """Get all contradiction audit evidence events"""
        return [
            event
            for event in self.events
            if event.event_type == ContextEventType.CONTRADICTION_AUDIT
        ]

    def get_evidence_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of all evidence collected

        Returns:
            Dictionary with evidence statistics and key insights
        """
        evidence_events = self.get_evidence_events()

        summary = {
            "trace_id": self.trace_id,
            "evidence_collection_timestamp": datetime.now(timezone.utc).isoformat(),
            "total_evidence_events": len(evidence_events),
            "evidence_types": {},
            "glass_box_completeness": 0.0,
            "consultant_selections": 0,
            "synergy_directives": 0,
            "coreops_executions": 0,
            "contradiction_audits": 0,
            "key_decisions": [],
            "evidence_timeline": [],
        }

        # Count evidence by type
        for event in evidence_events:
            event_type_str = event.event_type.value
            if event_type_str not in summary["evidence_types"]:
                summary["evidence_types"][event_type_str] = 0
            summary["evidence_types"][event_type_str] += 1

        # Extract key metrics
        summary["consultant_selections"] = len(self.get_consultant_selection_evidence())
        summary["synergy_directives"] = len(self.get_synergy_evidence())
        summary["coreops_executions"] = len(self.get_coreops_evidence())
        summary["contradiction_audits"] = len(self.get_contradiction_evidence())

        # Calculate glass-box completeness
        total_events = len(self.events)
        if total_events > 0:
            summary["glass_box_completeness"] = len(evidence_events) / total_events

        # Extract key decisions
        for event in evidence_events:
            if event.event_type == ContextEventType.MODEL_SELECTION_JUSTIFICATION:
                selection_data = event.data
                summary["key_decisions"].append(
                    {
                        "type": "consultant_selection",
                        "timestamp": event.timestamp.isoformat(),
                        "rationale": selection_data.get("selection_rationale", ""),
                        "confidence": selection_data.get("total_confidence", 0),
                        "consultant_count": selection_data.get("consultant_count", 0),
                    }
                )

            elif event.event_type == ContextEventType.SYNERGY_META_DIRECTIVE:
                synergy_data = event.data
                summary["key_decisions"].append(
                    {
                        "type": "synergy_directive",
                        "timestamp": event.timestamp.isoformat(),
                        "meta_directive": synergy_data.get("meta_directive", ""),
                        "confidence": synergy_data.get("confidence_score", 0),
                        "model_count": synergy_data.get("model_count", 0),
                    }
                )

        # Create evidence timeline
        for event in evidence_events[-10:]:  # Last 10 evidence events
            summary["evidence_timeline"].append(
                {
                    "timestamp": event.timestamp.isoformat(),
                    "type": event.event_type.value,
                    "description": self._summarize_evidence_event(event),
                }
            )

        return summary

    def format_for_llm(self, format_type: str = "structured") -> str:
        """
        Format context for LLM consumption with caching (Operation Lean - delegated to ContextFormattingService)

        Key Insight: Structured formats (XML/YAML) use 40% fewer tokens than JSON
        """
        cache_key = f"{format_type}:{len(self.events)}:{self.events[-1].event_id if self.events else ''}"

        if cache_key in self.formatted_cache:
            self.cache_hits += 1
            return self.formatted_cache[cache_key]

        self.cache_misses += 1

        # Delegate to formatting service
        relevant_events = self.get_relevant_context()
        formatted = self._formatting_service.format_for_llm(relevant_events, format_type)

        self.formatted_cache[cache_key] = formatted
        logger.debug(
            f"💾 Cached formatted context (type={format_type}, size={len(formatted)} chars)"
        )
        return formatted

    # ------------------------------------------------------------------
    # Compatibility shim for legacy callers
    # ------------------------------------------------------------------
    async def record_event(
        self,
        *,
        trace_id: Optional[str] = None,
        event_type: Any,
        event_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> ContextEvent:
        """
        DEPRECATED: Use add_event(event_type, data, metadata, timestamp) instead.

        Thin async shim that normalizes legacy parameters and forwards to add_event.
        Accepts event_type as string or ContextEventType and event_data payload.
        """
        try:
            import warnings as _warnings
            _warnings.warn(
                "record_event is deprecated; use add_event instead",
                DeprecationWarning,
                stacklevel=2,
            )
        except Exception:
            pass

        # Normalize event_type
        et: Optional[ContextEventType] = None
        if isinstance(event_type, ContextEventType):
            et = event_type
        else:
            try:
                # Try to map string value to enum by value
                val = str(event_type).lower()
                for e in ContextEventType:
                    if getattr(e, "value", "").lower() == val:
                        et = e
                        break
            except Exception:
                et = None
        if et is None:
            # Fallback: create a generic type via value field if possible
            # Use a safe default to avoid dropping the event
            try:
                et = ContextEventType.CONTRACT_ECHO  # safe/loggable default
            except Exception:
                # Last resort: pick ENGAGEMENT_STARTED to preserve append-only invariant
                et = ContextEventType.ENGAGEMENT_STARTED

        # Attach explicit trace_id into metadata if provided
        meta = metadata.copy() if isinstance(metadata, dict) else {}
        if trace_id:
            meta.setdefault("trace_id_override", trace_id)

        return self.add_event(et, event_data or {}, meta, timestamp)

    def format_as_xml(self) -> str:
        """Format context as XML (Operation Lean - delegated to ContextFormattingService)"""
        relevant_events = self.get_relevant_context()
        return self._formatting_service.format_as_xml(relevant_events)

    def format_compressed(self) -> str:
        """Format as compressed summary (Operation Lean - delegated to ContextFormattingService)"""
        relevant_events = self.get_relevant_context()
        return self._formatting_service.format_compressed(relevant_events)

    def format_as_json(self) -> str:
        """Format as JSON (Operation Lean - delegated to ContextFormattingService)"""
        relevant_events = self.get_relevant_context()
        return self._formatting_service.format_as_json(relevant_events)

    # Legacy private methods for backward compatibility
    def _format_as_xml(self) -> str:
        """Legacy method - use format_as_xml() instead"""
        return self.format_as_xml()

    def _format_compressed(self) -> str:
        """Legacy method - use format_compressed() instead"""
        return self.format_compressed()

    def _format_as_json(self) -> str:
        """Legacy method - use format_as_json() instead"""
        return self.format_as_json()

    def _calculate_initial_relevance(self, event_type: ContextEventType) -> float:
        """Calculate initial relevance (Operation Lean - delegated to ContextMetricsService)"""
        return self._metrics_service.calculate_initial_relevance(event_type)

    def _recalculate_relevance(self, event: ContextEvent) -> float:
        """Recalculate relevance (Operation Lean - delegated to ContextMetricsService)"""
        return self._metrics_service.recalculate_relevance(event)

    def _compress_old_events(self):
        """Compress old events (Operation Lean - delegated to ContextMetricsService)"""
        self._metrics_service.compress_old_events()

    def _summarize_evidence_event(self, event: ContextEvent) -> str:
        """Create human-readable summary of an evidence event"""

        if event.event_type == ContextEventType.MODEL_SELECTION_JUSTIFICATION:
            data = event.data
            consultant_count = data.get("consultant_count", 0)
            confidence = data.get("total_confidence", 0)
            return f"Selected {consultant_count} consultants with {confidence:.1%} confidence"

        elif event.event_type == ContextEventType.SYNERGY_META_DIRECTIVE:
            data = event.data
            model_count = data.get("model_count", 0)
            confidence = data.get("confidence_score", 0)
            return f"Generated meta-directive from {model_count} models with {confidence:.1%} confidence"

        elif event.event_type == ContextEventType.COREOPS_RUN_SUMMARY:
            data = event.data
            contract_id = data.get("system_contract_id", "unknown")
            argument_count = data.get("argument_count", 0)
            return f"Executed {contract_id} generating {argument_count} arguments"

        elif event.event_type == ContextEventType.COREOPS_STEP_EXECUTED:
            data = event.data
            step_id = data.get("step_id", "unknown")
            op = data.get("op", "unknown")
            status = data.get("status", "unknown")
            duration_ms = data.get("duration_ms", 0)
            return f"Executed step {step_id} ({op}) in {duration_ms}ms - {status}"

        elif event.event_type == ContextEventType.CONTRADICTION_AUDIT:
            data = event.data
            contradiction_count = data.get("contradiction_count", 0)
            synthesis_count = data.get("synthesis_count", 0)
            return f"Found {contradiction_count} contradictions, {synthesis_count} syntheses"

        # Phase 8: Learning Systems Events sanitization
        elif event.event_type == ContextEventType.LEARNING_CYCLE_STARTED:
            data = event.data
            cycle_id = data.get("cycle_id", "unknown")
            system_type = data.get("system_type", "unknown")
            return f"Started learning cycle {cycle_id} for {system_type}"

        elif event.event_type == ContextEventType.LEARNING_CYCLE_COMPLETED:
            data = event.data
            cycle_id = data.get("cycle_id", "unknown")
            duration_ms = data.get("duration_ms", 0)
            improvements = data.get("improvements_count", 0)
            return f"Completed learning cycle {cycle_id} in {duration_ms}ms with {improvements} improvements"

        elif event.event_type == ContextEventType.PATTERN_EFFECTIVENESS_UPDATE:
            data = event.data
            pattern_id = data.get("pattern_id", "unknown")
            old_score = data.get("old_score", 0)
            new_score = data.get("new_score", 0)
            evidence_count = data.get("evidence_count", 0)
            return f"Updated pattern {pattern_id}: {old_score:.2f}→{new_score:.2f} (evidence: {evidence_count})"

        elif event.event_type == ContextEventType.DIVERSITY_POLICY_ENFORCED:
            data = event.data
            policy_id = data.get("policy_id", "unknown")
            action = data.get("action", "unknown")
            before_metric = data.get("before_metric", 0)
            after_metric = data.get("after_metric", 0)
            return f"Applied policy {policy_id}: {action} ({before_metric:.2f}→{after_metric:.2f})"

        elif event.event_type == ContextEventType.OPTIMIZATION_ACTION_TAKEN:
            data = event.data
            action_id = data.get("action_id", "unknown")
            reason = data.get("reason", "unknown")
            guardrails_passed = data.get("guardrails_passed", True)
            return (
                f"Optimization {action_id}: {reason} (guardrails: {guardrails_passed})"
            )

        elif event.event_type == ContextEventType.FEEDBACK_INGESTED:
            data = event.data
            source = data.get("source", "unknown")
            items_count = data.get("items_count", 0)
            return f"Ingested {items_count} feedback items from {source}"

        elif event.event_type == ContextEventType.DASHBOARD_METRICS_UPDATED:
            data = event.data
            kpis = data.get("kpis", {})
            kpi_count = len(kpis)
            return f"Updated {kpi_count} dashboard KPIs"

        else:
            return f"Evidence event: {event.event_type.value}"

    def export_evidence_for_api(self) -> Dict[str, Any]:
        """
        Export evidence in API-friendly format for frontend consumption

        Returns:
            Structured evidence data optimized for frontend display
        """
        evidence_events = self.get_evidence_events()

        api_evidence = {
            "metadata": {
                "trace_id": self.trace_id,
                "total_evidence_events": len(evidence_events),
                "collection_timestamp": datetime.now(timezone.utc).isoformat(),
                "session_duration_minutes": self._calculate_session_duration(),
            },
            "consultant_selections": [],
            "synergy_directives": [],
            "coreops_executions": [],
            "contradiction_audits": [],
            "evidence_timeline": [],
        }

        # Process consultant selection evidence
        for event in self.get_consultant_selection_evidence():
            data = event.data
            api_evidence["consultant_selections"].append(
                {
                    "timestamp": event.timestamp.isoformat(),
                    "selection_rationale": data.get("selection_rationale", ""),
                    "total_confidence": data.get("total_confidence", 0),
                    "consultant_count": data.get("consultant_count", 0),
                    "consultants": data.get("consultants", []),
                    "risk_factors": data.get("risk_factors", []),
                    "success_factors": data.get("success_factors", []),
                }
            )

        # Process synergy directive evidence
        for event in self.get_synergy_evidence():
            data = event.data
            api_evidence["synergy_directives"].append(
                {
                    "timestamp": event.timestamp.isoformat(),
                    "meta_directive": data.get("meta_directive", ""),
                    "synergy_insight": data.get("synergy_insight", ""),
                    "conflict_insight": data.get("conflict_insight", ""),
                    "confidence_score": data.get("confidence_score", 0),
                    "participating_models": data.get("participating_models", []),
                    "processing_time_ms": data.get("processing_time_ms", 0),
                }
            )

        # Process CoreOps execution evidence
        for event in self.get_coreops_evidence():
            data = event.data
            api_evidence["coreops_executions"].append(
                {
                    "timestamp": event.timestamp.isoformat(),
                    "system_contract_id": data.get("system_contract_id", ""),
                    "program_path": data.get("program_path", ""),
                    "step_count": data.get("step_count", 0),
                    "argument_count": data.get("argument_count", 0),
                    "sample_claims": data.get("sample_claims", []),
                    "processing_time_ms": data.get("processing_time_ms", 0),
                    "execution_mode": data.get("execution_mode", ""),
                }
            )

        # Process contradiction audit evidence
        for event in self.get_contradiction_evidence():
            data = event.data
            api_evidence["contradiction_audits"].append(
                {
                    "timestamp": event.timestamp.isoformat(),
                    "contradiction_count": data.get("contradiction_count", 0),
                    "synthesis_count": data.get("synthesis_count", 0),
                    "example_contradiction": data.get("example_contradiction", ""),
                    "example_synthesis": data.get("example_synthesis", ""),
                    "confidence_score": data.get("confidence_score", 0),
                    "bias_mitigation_applied": data.get(
                        "bias_mitigation_applied", False
                    ),
                }
            )

        # Create timeline of all evidence events
        for event in evidence_events:
            api_evidence["evidence_timeline"].append(
                {
                    "timestamp": event.timestamp.isoformat(),
                    "event_type": event.event_type.value,
                    "description": self._summarize_evidence_event(event),
                    "confidence": self._extract_confidence_from_event(event),
                    "processing_time_ms": self._extract_processing_time_from_event(
                        event
                    ),
                }
            )

        return api_evidence

    def _calculate_session_duration(self) -> float:
        """Calculate session duration in minutes"""
        if self.completed_at:
            duration = (self.completed_at - self.started_at).total_seconds() / 60
        else:
            duration = (
                datetime.now(timezone.utc) - self.started_at
            ).total_seconds() / 60
        return round(duration, 2)

    def _extract_confidence_from_event(self, event: ContextEvent) -> float:
        """Extract confidence score from event data"""
        data = event.data
        return data.get("confidence_score", data.get("total_confidence", 0))

    def _extract_processing_time_from_event(self, event: ContextEvent) -> int:
        """Extract processing time from event data"""
        data = event.data
        return data.get("processing_time_ms", 0)

    def _summarize_event(self, event: ContextEvent) -> str:
        """Summarize event (Operation Lean - delegated to ContextMetricsService)"""
        return self._metrics_service.summarize_event(event)

    def _notify_subscribers(self, event: ContextEvent):
        """Notify event subscribers"""
        if event.event_type in self.subscribers:
            for callback in self.subscribers[event.event_type]:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"❌ Subscriber notification failed: {e}")

    def subscribe(
        self, event_type: ContextEventType, callback: Callable[[ContextEvent], None]
    ):
        """Subscribe to specific event types"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
        logger.debug(f"📡 Subscribed to {event_type.value} events")

    def create_checkpoint(self) -> Dict[str, Any]:
        """Create checkpoint for persistence"""
        return {
            "events": [e.to_dict() for e in self.events[-100:]],  # Last 100 events
            "stats": {
                "total_events": len(self.events),
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def restore_from_checkpoint(self, checkpoint: Dict[str, Any]):
        """Restore from checkpoint"""
        self.events = []
        self.event_index = {}

        for event_dict in checkpoint.get("events", []):
            event = ContextEvent(
                event_id=event_dict["event_id"],
                event_type=ContextEventType(event_dict["event_type"]),
                timestamp=datetime.fromisoformat(event_dict["timestamp"]),
                data=event_dict["data"],
                metadata=event_dict.get("metadata", {}),
                relevance_score=event_dict.get("relevance_score", 1.0),
            )
            self.events.append(event)
            self.event_index[event.event_id] = event

        # Restore stats
        stats = checkpoint.get("stats", {})
        self.cache_hits = stats.get("cache_hits", 0)
        self.cache_misses = stats.get("cache_misses", 0)

        logger.info(f"🔄 Restored {len(self.events)} events from checkpoint")

    def get_current_timestamp(self) -> str:
        """Get current timestamp as ISO string"""
        return datetime.now(timezone.utc).isoformat()

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get context stream performance metrics"""
        total_requests = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0

        return {
            "total_events": len(self.events),
            "relevant_events": len(self.get_relevant_context()),
            "event_types": len(set(e.event_type for e in self.events)),
            "cache_hit_rate": cache_hit_rate,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "subscribers_count": sum(len(subs) for subs in self.subscribers.values()),
            "memory_events": len([e for e in self.events if not e.compressed_version]),
            "compressed_events": len([e for e in self.events if e.compressed_version]),
            "last_event_time": (
                self.events[-1].timestamp.isoformat() if self.events else None
            ),
        }

    def set_engagement_context(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        engagement_type: str = "consultation",
        case_id: Optional[str] = None,
    ) -> None:
        """Set engagement context for database persistence"""
        self.session_id = session_id
        self.user_id = user_id
        self.organization_id = organization_id
        self.engagement_type = engagement_type
        self.case_id = case_id

        logger.info(f"📝 Engagement context set: {engagement_type} for case {case_id}")

    def complete_engagement(self, final_status: str = "completed") -> None:
        """Mark engagement as complete and prepare for persistence"""
        self.completed_at = datetime.now(timezone.utc)

        # Add completion event
        self.add_event(
            ContextEventType.ENGAGEMENT_COMPLETED,
            {
                "final_status": final_status,
                "duration_ms": int(
                    (self.completed_at - self.started_at).total_seconds() * 1000
                ),
                "total_events": len(self.events),
                "trace_id": self.trace_id,
            },
            {
                "engagement_completed_at": self.completed_at.isoformat(),
                "performance_metrics": self.get_performance_metrics(),
            },
        )

        logger.info(f"✅ Engagement completed with status: {final_status}")

    def _calculate_summary_metrics(self) -> Dict[str, Any]:
        """Calculate summary metrics for database storage"""
        # Extract models used from events
        models_used = set()
        consultants_used = set()
        tools_used = set()
        total_tokens = 0
        total_cost = 0.0
        error_count = 0

        for event in self.events:
            # Extract model information
            if "model" in event.data:
                models_used.add(event.data["model"])
            if "models_used" in event.data:
                if isinstance(event.data["models_used"], list):
                    models_used.update(event.data["models_used"])

            # Extract consultant information
            if "consultant" in event.data:
                consultants_used.add(event.data["consultant"])
            if "consultant_id" in event.data:
                consultants_used.add(event.data["consultant_id"])
            if "consultants_invoked" in event.data:
                if isinstance(event.data["consultants_invoked"], list):
                    consultants_used.update(event.data["consultants_invoked"])

            # Extract tool information
            if "tool" in event.data:
                tools_used.add(event.data["tool"])
            if "tools_used" in event.data:
                if isinstance(event.data["tools_used"], list):
                    tools_used.update(event.data["tools_used"])

            # Extract token and cost information
            if "tokens" in event.data:
                total_tokens += event.data["tokens"]
            if "total_tokens" in event.data:
                total_tokens += event.data["total_tokens"]
            if "cost" in event.data:
                total_cost += float(event.data["cost"])
            if "total_cost" in event.data:
                total_cost += float(event.data["total_cost"])

            # Count errors
            if event.event_type == ContextEventType.ERROR_OCCURRED:
                error_count += 1

        # Determine final status
        final_status = "completed"
        if error_count > 0:
            final_status = (
                "failed" if error_count > len(self.events) * 0.1 else "partial"
            )

        # Check for PII in the events
        contains_pii = any(
            any("[REDACTED_" in str(value) for value in event.data.values())
            for event in self.events
        )

        return {
            "total_tokens": total_tokens,
            "total_cost": round(total_cost, 6),
            "final_status": final_status,
            "error_count": error_count,
            "models_used": list(models_used),
            "consultants_used": list(consultants_used),
            "tools_used": list(tools_used),
            "contains_pii": contains_pii,
            "overall_quality_score": max(
                0.0, min(1.0, 1.0 - (error_count / max(1, len(self.events))))
            ),
        }

    async def persist_to_database(self) -> bool:
        """Persist the complete context stream using the injected persistence adapter"""
        if not self.persistence_adapter:
            # Lazy default to file adapter for safety
            try:
                from src.core.persistence.adapters import FileAdapter

                self.persistence_adapter = FileAdapter()
            except Exception as e:
                logger.error(f"❌ No persistence adapter available: {e}")
                return False
        try:
            record = self._build_persistence_record()
            await self.persistence_adapter.persist([record])
            logger.info("✅ Context stream persisted via adapter")
            return True
        except Exception as e:
            logger.error(f"❌ Context stream persistence failed: {e}")
            return False

    def set_final_analysis_text(self, analysis_text: str) -> None:
        """
        Store the final analysis text for offline evaluation.

        This text will be included in trace exports to enable evaluation judges
        to assess groundedness, relevance, and actionability.

        Args:
            analysis_text: The final analysis/report text (executive summary)
        """
        self.final_analysis_text = analysis_text

    def _build_persistence_record(self) -> Dict[str, Any]:
        """Build a database-ready record for the current context stream"""
        # Calculate summary metrics
        summary_metrics = self._calculate_summary_metrics()

        # Build the context stream JSONB object
        context_stream_data = {
            "events": [event.to_dict() for event in self.events],
            "summary": {
                "total_events": len(self.events),
                **summary_metrics,
                "performance_metrics": self.get_performance_metrics(),
                # OPERATION AWAKENING: Include final analysis for evaluation judges
                "final_report": self.final_analysis_text or "",
            },
        }

        # Calculate duration
        duration_ms = None
        if self.completed_at:
            duration_ms = int(
                (self.completed_at - self.started_at).total_seconds() * 1000
            )

        # Build the database record
        record = {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "organization_id": self.organization_id,
            "engagement_type": self.engagement_type,
            "case_id": self.case_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "duration_ms": duration_ms,
            "context_stream": context_stream_data,
            "total_tokens": summary_metrics["total_tokens"],
            "total_cost": summary_metrics["total_cost"],
            "final_status": summary_metrics["final_status"],
            "error_count": summary_metrics["error_count"],
            "models_used": summary_metrics["models_used"],
            "consultants_used": summary_metrics["consultants_used"],
            "tools_used": summary_metrics["tools_used"],
            "overall_quality_score": summary_metrics["overall_quality_score"],
            "contains_pii": summary_metrics["contains_pii"],
            "data_classification": (
                "confidential" if summary_metrics["contains_pii"] else "internal"
            ),
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        }
        return record


# Global instance for the application
_context_stream: Optional[UnifiedContextStream] = None


def get_unified_context_stream() -> UnifiedContextStream:
    """Get or create the global unified context stream instance with injected persistence adapter"""
    global _context_stream
    if _context_stream is None:
        adapter = None
        try:
            from src.services.container import global_container  # type: ignore

            if global_container:
                adapter = getattr(
                    global_container, "get_event_persistence_adapter", lambda: None
                )()
        except Exception:
            adapter = None
        _context_stream = UnifiedContextStream(persistence_adapter=adapter)
    return _context_stream


def create_new_context_stream(
    engagement_type: str = "consultation", case_id: Optional[str] = None
) -> UnifiedContextStream:
    """Create a new context stream for a specific engagement"""
    adapter = None
    try:
        from src.services.container import global_container  # type: ignore

        if global_container:
            adapter = getattr(
                global_container, "get_event_persistence_adapter", lambda: None
            )()
    except Exception:
        adapter = None
    stream = UnifiedContextStream(persistence_adapter=adapter)
    stream.set_engagement_context(engagement_type=engagement_type, case_id=case_id)
    return stream
