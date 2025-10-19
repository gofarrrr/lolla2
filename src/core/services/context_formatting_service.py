"""
Context Formatting Service

Extracted from UnifiedContextStream (Task 5.0)
Handles formatting context events for different consumers (LLM, API, etc.)

Created: 2025-10-18
Campaign: Operation Lean
Original Lines: 769-906 from unified_context_stream.py
"""

import json
import logging
from typing import List, Any, Optional

logger = logging.getLogger(__name__)


class ContextFormattingService:
    """
    Context stream formatting service.

    Responsibilities:
    - Format events as XML (40% token reduction)
    - Format events as JSON
    - Format events in compressed format
    - Format events optimized for LLM consumption
    """

    def __init__(self):
        """Initialize context formatting service."""
        pass

    def format_as_xml(self, events: List[Any]) -> str:
        """
        Format events as XML for optimal token usage.

        Based on research: XML format reduces tokens by ~40% compared to JSON.

        Args:
            events: List of context events to format

        Returns:
            XML-formatted string
        """
        xml_parts = ["<context>"]

        # Group events by type for better structure
        events_by_type = {}
        for event in events:
            if event.event_type not in events_by_type:
                events_by_type[event.event_type] = []
            events_by_type[event.event_type].append(event)

        for event_type, type_events in events_by_type.items():
            # Get event type value
            event_type_str = event_type.value if hasattr(event_type, 'value') else str(event_type)

            xml_parts.append(f"  <{event_type_str}>")

            # Last 5 of each type for token efficiency
            for event in type_events[-5:]:
                event_id_short = event.event_id[:8] if len(event.event_id) > 8 else event.event_id
                xml_parts.append(f"    <event id='{event_id_short}'>")

                # Only include simple data types for efficiency
                for key, value in event.data.items():
                    if isinstance(value, (str, int, float, bool)):
                        xml_parts.append(f"      <{key}>{value}</{key}>")

                xml_parts.append("    </event>")

            xml_parts.append(f"  </{event_type_str}>")

        xml_parts.append("</context>")
        return "\n".join(xml_parts)

    def format_as_json(self, events: List[Any]) -> str:
        """
        Format events as JSON (fallback format).

        Args:
            events: List of context events to format

        Returns:
            JSON-formatted string
        """
        return json.dumps([event.to_dict() for event in events], indent=2)

    def format_compressed(self, events: List[Any]) -> str:
        """
        Format events in compressed summary format for maximum token efficiency.

        Args:
            events: List of context events to format

        Returns:
            Compressed format string with summaries
        """
        summaries = []

        # Group by event type and summarize
        events_by_type = {}
        for event in events:
            if event.event_type not in events_by_type:
                events_by_type[event.event_type] = []
            events_by_type[event.event_type].append(event)

        for event_type, type_events in events_by_type.items():
            event_type_str = event_type.value if hasattr(event_type, 'value') else str(event_type)
            count = len(type_events)
            latest = type_events[-1] if type_events else None

            if latest:
                # Get summary from latest event data
                latest_summary = latest.data.get('summary', 'No summary')
                if isinstance(latest_summary, str) and len(latest_summary) > 50:
                    latest_summary = latest_summary[:50] + "..."

                summary = f"{event_type_str}: {count} events, latest: {latest_summary}"
                summaries.append(summary)

        return "COMPRESSED CONTEXT:\n" + "\n".join(summaries)

    def format_for_llm(
        self,
        events: List[Any],
        format_type: str = "structured"
    ) -> str:
        """
        Format events optimized for LLM consumption.

        Key Insight: Structured formats (XML/YAML) use 40% fewer tokens than JSON.

        Args:
            events: List of context events to format
            format_type: Type of formatting (structured, compressed, json)

        Returns:
            LLM-optimized formatted string
        """
        if format_type == "structured":
            return self.format_as_xml(events)
        elif format_type == "compressed":
            return self.format_compressed(events)
        else:
            return self.format_as_json(events)
