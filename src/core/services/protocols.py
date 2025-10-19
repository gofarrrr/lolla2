"""
Service Layer Protocols for UnifiedContextStream

Defines interfaces for the 5 extracted services following the Service Layer Extraction pattern.
All services operate on event data and are independently testable.

Created: 2025-10-18
Campaign: Operation Lean
Task: 2.0 - Create Service Interfaces
"""

from typing import Protocol, Dict, List, Any, Optional
from abc import abstractmethod


class IEventValidationService(Protocol):
    """
    Protocol for event validation and PII scrubbing.

    Responsibilities:
    - Validate event schemas against allowed types
    - Validate event state transitions
    - Check events against allowlist
    - Scrub PII from event data
    """

    @abstractmethod
    def validate_event_schema(
        self, event_type: str, data: Dict[str, Any]
    ) -> bool:
        """
        Validate event data against schema for the given event type.

        Args:
            event_type: Type of event to validate
            data: Event data dictionary

        Returns:
            True if schema is valid, False otherwise
        """
        ...

    @abstractmethod
    def validate_event_transition(
        self, event_type: str, previous_event_type: Optional[str] = None
    ) -> bool:
        """
        Validate that event transition is allowed.

        Args:
            event_type: Type of event being added
            previous_event_type: Type of previous event (if any)

        Returns:
            True if transition is valid, False otherwise
        """
        ...

    @abstractmethod
    def is_event_allowed(self, event_type: str) -> bool:
        """
        Check if event type is in the allowlist.

        Args:
            event_type: Type of event to check

        Returns:
            True if event is allowed, False otherwise
        """
        ...

    @abstractmethod
    def scrub_pii(self, data_string: str) -> str:
        """
        Remove PII (emails, phone numbers, SSN, etc.) from string.

        Args:
            data_string: String potentially containing PII

        Returns:
            String with PII replaced by [REDACTED]
        """
        ...

    @abstractmethod
    def scrub_structure(self, value: Any) -> Any:
        """
        Recursively scrub PII from nested data structures.

        Args:
            value: Any value (dict, list, str, etc.)

        Returns:
            Value with all PII scrubbed
        """
        ...


class IEvidenceExtractionService(Protocol):
    """
    Protocol for extracting evidence events from context stream.

    Responsibilities:
    - Extract consultant selection evidence
    - Extract synergy evidence
    - Extract CoreOps evidence
    - Extract contradiction evidence
    - Summarize evidence for API responses
    """

    @abstractmethod
    def get_evidence_events(
        self,
        evidence_types: Optional[List[str]] = None
    ) -> List[Any]:
        """
        Get all evidence events of specified types.

        Args:
            evidence_types: List of evidence event types (None = all)

        Returns:
            List of evidence events
        """
        ...

    @abstractmethod
    def get_consultant_selection_evidence(self) -> List[Any]:
        """
        Get evidence related to consultant selection decisions.

        Returns:
            List of consultant selection evidence events
        """
        ...

    @abstractmethod
    def get_synergy_evidence(self) -> List[Any]:
        """
        Get evidence of model synergy and interactions.

        Returns:
            List of synergy evidence events
        """
        ...

    @abstractmethod
    def get_coreops_evidence(self) -> List[Any]:
        """
        Get evidence of CoreOps execution.

        Returns:
            List of CoreOps evidence events
        """
        ...

    @abstractmethod
    def get_contradiction_evidence(self) -> List[Any]:
        """
        Get evidence of contradictions in analysis.

        Returns:
            List of contradiction evidence events
        """
        ...

    @abstractmethod
    def get_evidence_summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive evidence summary.

        Returns:
            Dictionary with evidence counts, summaries, and metadata
        """
        ...

    @abstractmethod
    def export_evidence_for_api(self) -> Dict[str, Any]:
        """
        Export evidence in API-friendly format.

        Returns:
            Dictionary formatted for API responses
        """
        ...


class IContextFormattingService(Protocol):
    """
    Protocol for formatting context stream for different consumers.

    Responsibilities:
    - Format events as XML (40% token reduction)
    - Format events as JSON
    - Format events in compressed format
    - Format events for LLM consumption
    """

    @abstractmethod
    def format_as_xml(self, events: List[Any]) -> str:
        """
        Format events as XML (optimized for LLM token efficiency).

        Args:
            events: List of context events

        Returns:
            XML-formatted string
        """
        ...

    @abstractmethod
    def format_as_json(self, events: List[Any]) -> str:
        """
        Format events as JSON.

        Args:
            events: List of context events

        Returns:
            JSON-formatted string
        """
        ...

    @abstractmethod
    def format_compressed(self, events: List[Any]) -> str:
        """
        Format events in compressed format (summaries only).

        Args:
            events: List of context events

        Returns:
            Compressed format string
        """
        ...

    @abstractmethod
    def format_for_llm(
        self,
        events: List[Any],
        format_type: str = "structured"
    ) -> str:
        """
        Format events optimized for LLM consumption.

        Args:
            events: List of context events
            format_type: Type of formatting (structured, compressed, xml, json)

        Returns:
            LLM-optimized formatted string
        """
        ...


class IContextPersistenceService(Protocol):
    """
    Protocol for persisting context stream state.

    Responsibilities:
    - Create checkpoints for resume capability
    - Restore from checkpoints
    - Persist to database
    - Store final analysis text
    """

    @abstractmethod
    def create_checkpoint(self, events: List[Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create checkpoint of current context state.

        Args:
            events: List of context events
            metadata: Additional metadata (engagement_id, timestamp, etc.)

        Returns:
            Checkpoint dictionary
        """
        ...

    @abstractmethod
    def restore_from_checkpoint(self, checkpoint: Dict[str, Any]) -> tuple[List[Any], Dict[str, Any]]:
        """
        Restore context state from checkpoint.

        Args:
            checkpoint: Checkpoint dictionary

        Returns:
            Tuple of (events list, metadata dict)
        """
        ...

    @abstractmethod
    async def persist_to_database(
        self,
        events: List[Any],
        metadata: Dict[str, Any],
        persistence_interface: Any
    ) -> bool:
        """
        Persist context stream to database.

        Args:
            events: List of context events
            metadata: Engagement and session metadata
            persistence_interface: Database persistence interface

        Returns:
            True if successful, False otherwise
        """
        ...

    @abstractmethod
    def set_final_analysis_text(self, analysis_text: str, metadata: Dict[str, Any]) -> None:
        """
        Store final analysis text in metadata.

        Args:
            analysis_text: Final analysis output
            metadata: Metadata dictionary to update
        """
        ...

    @abstractmethod
    def build_persistence_record(
        self,
        events: List[Any],
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build complete persistence record for storage.

        Args:
            events: List of context events
            metadata: Session metadata

        Returns:
            Complete persistence record dictionary
        """
        ...


class IContextMetricsService(Protocol):
    """
    Protocol for calculating metrics and analytics from context stream.

    Responsibilities:
    - Calculate session duration
    - Extract confidence scores
    - Extract processing times
    - Calculate summary metrics (complex aggregation)
    - Generate performance metrics
    """

    @abstractmethod
    def calculate_session_duration(
        self,
        events: List[Any],
        start_time: Optional[str] = None
    ) -> float:
        """
        Calculate total session duration in seconds.

        Args:
            events: List of context events
            start_time: Optional override for start time

        Returns:
            Duration in seconds
        """
        ...

    @abstractmethod
    def extract_confidence_from_event(self, event: Any) -> float:
        """
        Extract confidence score from event data.

        Args:
            event: Context event

        Returns:
            Confidence score (0.0-1.0), or 0.0 if not found
        """
        ...

    @abstractmethod
    def extract_processing_time_from_event(self, event: Any) -> int:
        """
        Extract processing time (ms) from event data.

        Args:
            event: Context event

        Returns:
            Processing time in milliseconds, or 0 if not found
        """
        ...

    @abstractmethod
    def calculate_summary_metrics(self, events: List[Any]) -> Dict[str, Any]:
        """
        Calculate comprehensive summary metrics (CC=18 complexity).

        This is the most complex method in the original file.
        Aggregates confidence, processing time, event counts, etc.

        Args:
            events: List of context events

        Returns:
            Dictionary with summary metrics
        """
        ...

    @abstractmethod
    def get_performance_metrics(self, events: List[Any]) -> Dict[str, Any]:
        """
        Get performance metrics for monitoring.

        Args:
            events: List of context events

        Returns:
            Dictionary with performance metrics
        """
        ...

    @abstractmethod
    def summarize_event(self, event: Any) -> str:
        """
        Create human-readable summary of single event.

        Args:
            event: Context event

        Returns:
            Summary string
        """
        ...
