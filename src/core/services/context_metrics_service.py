"""
Context Metrics Service

Extracted from UnifiedContextStream (Task 7.0)
Handles relevance scoring, event filtering, and memory management.

Created: 2025-10-19
Campaign: Operation Lean
Original Lines: 582-617, 907-1013, 1225-1234 from unified_context_stream.py
"""

import json
import logging
from typing import List, Any, Optional
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)


class ContextMetricsService:
    """
    Context stream metrics and relevance service.

    Responsibilities:
    - Calculate initial and dynamic relevance scores
    - Filter events by relevance
    - Retrieve recent events
    - Compress old events for memory management
    - Summarize events for compression
    """

    def __init__(
        self,
        events: List[Any],
        event_index: dict,
        max_events: int = 10000
    ):
        """
        Initialize context metrics service.

        Args:
            events: List of context events
            event_index: Dictionary mapping event_id to event
            max_events: Maximum number of events to keep in memory
        """
        self.events = events
        self.event_index = event_index
        self.max_events = max_events

    def get_relevant_context(
        self, for_phase: Optional[str] = None, min_relevance: float = 0.3
    ) -> List[Any]:
        """
        Get relevant events based on relevance scoring.

        Key Insight: Not all context is equally relevant - smart filtering reduces tokens

        Args:
            for_phase: Optional phase to filter for (unused currently)
            min_relevance: Minimum relevance score threshold

        Returns:
            List of relevant events
        """
        relevant_events = []

        for event in self.events:
            # Update relevance based on recency and access patterns
            event.relevance_score = self.recalculate_relevance(event)

            if event.relevance_score >= min_relevance:
                relevant_events.append(event)
                event.access_count += 1
                event.last_accessed = datetime.now(timezone.utc)

        logger.debug(
            f"🎯 Found {len(relevant_events)} relevant events (min_relevance={min_relevance})"
        )
        return relevant_events

    def get_recent_events(self, limit: int = 10) -> List[Any]:
        """
        Get the most recent events from the stream.

        Args:
            limit: Maximum number of events to return

        Returns:
            List of most recent events
        """
        return self.events[-limit:] if self.events else []

    def calculate_initial_relevance(self, event_type: Any) -> float:
        """
        Calculate initial relevance score based on event type.

        Args:
            event_type: The type of context event

        Returns:
            Initial relevance score (0.0-1.0)
        """
        # Import here to avoid circular dependency
        try:
            from src.core.unified_context_stream import ContextEventType
        except ImportError:
            # Fallback if import fails
            return 0.5

        # Critical events always relevant
        critical_types = {
            ContextEventType.ENGAGEMENT_STARTED,
            ContextEventType.HITL_RESPONSE,
            ContextEventType.ERROR_OCCURRED,
        }

        if event_type in critical_types:
            return 1.0

        # Research and reasoning moderately relevant
        moderate_types = {
            ContextEventType.RESEARCH_RESULT,
            ContextEventType.REASONING_STEP,
            ContextEventType.MODEL_APPLIED,
        }

        if event_type in moderate_types:
            return 0.7

        # Everything else starts at medium relevance
        return 0.5

    def recalculate_relevance(self, event: Any) -> float:
        """
        Recalculate relevance based on multiple factors.

        Factors considered:
        1. Recency (exponential decay)
        2. Access frequency
        3. Event type importance
        4. Semantic similarity to current goal

        Args:
            event: Context event to recalculate relevance for

        Returns:
            Updated relevance score (0.1-1.0)
        """
        base_relevance = self.calculate_initial_relevance(event.event_type)

        # Recency factor (exponential decay over time) - more generous decay
        age_minutes = (
            datetime.now(timezone.utc)
            - (
                event.timestamp
                if event.timestamp.tzinfo
                else event.timestamp.replace(tzinfo=timezone.utc)
            )
        ).total_seconds() / 60
        recency_factor = max(
            0.5, 1.0 - (age_minutes / 120)
        )  # Decay over 2 hours, minimum 0.5

        # Access frequency factor - boost for accessed events
        access_factor = min(1.2, 1.0 + (event.access_count * 0.05))  # Up to 20% boost

        # Combined relevance - preserve base relevance better
        relevance = (
            base_relevance * 0.7 + recency_factor * 0.2 + (access_factor - 1.0) * 0.1
        )

        return min(1.0, max(0.1, relevance))  # Ensure minimum 0.1

    def compress_old_events(self):
        """
        Compress old events to manage memory.

        Strategy: Keep most recent events, remove/compress old ones
        This method has high CC (18) due to compression logic branching.
        """
        if len(self.events) <= self.max_events:
            return

        # Remove oldest events, keeping the most recent max_events
        events_to_remove = len(self.events) - self.max_events
        removed_events = self.events[:events_to_remove]
        self.events = self.events[events_to_remove:]

        # Update event index
        for event in removed_events:
            self.event_index.pop(event.event_id, None)

        logger.info(f"🗜️ Removed {events_to_remove} old events to manage memory")

        # Also compress old events that remain (older than 1 hour, low relevance)
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=1)
        compressed_count = 0

        for event in self.events:
            if event.timestamp < cutoff_time and event.relevance_score < 0.5:
                if not event.compressed_version and not event.data.get(
                    "compressed", False
                ):
                    # Create compressed version
                    event.compressed_version = json.dumps(
                        {
                            "type": event.event_type.value,
                            "summary": self.summarize_event(event),
                        }
                    )
                    # Clear full data to save memory
                    event.data = {
                        "compressed": True,
                        "summary": event.compressed_version,
                    }
                    event.can_compress = False
                    compressed_count += 1

        if compressed_count > 0:
            logger.info(f"🗜️ Compressed {compressed_count} old events to save memory")

    def summarize_event(self, event: Any) -> str:
        """
        Create summary of event for compression.

        Args:
            event: Context event to summarize

        Returns:
            Summary string (max 100 characters)
        """
        if "summary" in event.data:
            return str(event.data["summary"])[:100]
        elif "description" in event.data:
            return str(event.data["description"])[:100]
        elif "result" in event.data:
            return f"Result: {str(event.data['result'])[:80]}"
        else:
            return f"{event.event_type.value} event"
