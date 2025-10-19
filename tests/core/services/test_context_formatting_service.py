"""
Tests for ContextFormattingService

Task 5.8: Comprehensive test coverage for context formatting.
Target: ≥90% coverage

Created: 2025-10-18
Campaign: Operation Lean
"""

import pytest
import json
from unittest.mock import Mock
from src.core.services.context_formatting_service import ContextFormattingService


class MockEvent:
    """Mock context event for testing"""
    def __init__(self, event_type, event_id, data):
        self.event_type = event_type
        self.event_id = event_id
        self.data = data

    def to_dict(self):
        return {
            "event_type": self.event_type.value if hasattr(self.event_type, 'value') else str(self.event_type),
            "event_id": self.event_id,
            "data": self.data
        }


class MockEventType:
    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        if isinstance(other, MockEventType):
            return self.value == other.value
        return False

    def __hash__(self):
        return hash(self.value)


class TestContextFormattingServiceInit:
    """Test service initialization"""

    def test_init(self):
        """Test initialization"""
        service = ContextFormattingService()
        assert service is not None


class TestFormatAsXML:
    """Test XML formatting"""

    def test_format_as_xml_basic(self):
        """Test basic XML formatting"""
        events = [
            MockEvent(MockEventType("TEST_EVENT"), "evt_123", {"key": "value"}),
        ]

        service = ContextFormattingService()
        xml = service.format_as_xml(events)

        assert "<context>" in xml
        assert "</context>" in xml
        assert "<TEST_EVENT>" in xml
        assert "</TEST_EVENT>" in xml
        assert "<event id='evt_123'>" in xml

    def test_format_as_xml_with_simple_data_types(self):
        """Test XML formatting with various simple data types"""
        events = [
            MockEvent(
                MockEventType("DATA_EVENT"),
                "evt_456",
                {
                    "string_val": "test",
                    "int_val": 42,
                    "float_val": 3.14,
                    "bool_val": True
                }
            ),
        ]

        service = ContextFormattingService()
        xml = service.format_as_xml(events)

        assert "<string_val>test</string_val>" in xml
        assert "<int_val>42</int_val>" in xml
        assert "<float_val>3.14</float_val>" in xml
        assert "<bool_val>True</bool_val>" in xml

    def test_format_as_xml_excludes_complex_types(self):
        """Test that XML formatting excludes complex data types"""
        events = [
            MockEvent(
                MockEventType("COMPLEX_EVENT"),
                "evt_789",
                {
                    "simple": "value",
                    "list": [1, 2, 3],  # Should be excluded
                    "dict": {"nested": "data"}  # Should be excluded
                }
            ),
        ]

        service = ContextFormattingService()
        xml = service.format_as_xml(events)

        assert "<simple>value</simple>" in xml
        assert "list" not in xml
        assert "dict" not in xml

    def test_format_as_xml_limits_to_last_5_per_type(self):
        """Test that XML formatting limits to last 5 events per type"""
        events = [
            MockEvent(MockEventType("REPEATED"), f"evt_{i}", {"index": i})
            for i in range(10)
        ]

        service = ContextFormattingService()
        xml = service.format_as_xml(events)

        # Should only have last 5 (indices 5-9)
        assert "<index>9</index>" in xml
        # Should have 5 event tags total for this type
        event_count = xml.count("<event id=")
        assert event_count == 5

    def test_format_as_xml_groups_by_type(self):
        """Test that events are grouped by type"""
        events = [
            MockEvent(MockEventType("TYPE_A"), "evt_1", {"data": "a1"}),
            MockEvent(MockEventType("TYPE_B"), "evt_2", {"data": "b1"}),
            MockEvent(MockEventType("TYPE_A"), "evt_3", {"data": "a2"}),
        ]

        service = ContextFormattingService()
        xml = service.format_as_xml(events)

        # Events should be grouped by type
        assert "<TYPE_A>" in xml
        assert "<TYPE_B>" in xml
        assert xml.count("</TYPE_A>") == 1  # Only one TYPE_A section
        assert xml.count("</TYPE_B>") == 1  # Only one TYPE_B section

    def test_format_as_xml_empty_events(self):
        """Test XML formatting with no events"""
        service = ContextFormattingService()
        xml = service.format_as_xml([])

        assert xml == "<context>\n</context>"


class TestFormatAsJSON:
    """Test JSON formatting"""

    def test_format_as_json_basic(self):
        """Test basic JSON formatting"""
        events = [
            MockEvent(MockEventType("TEST"), "evt_1", {"key": "value"}),
        ]

        service = ContextFormattingService()
        json_str = service.format_as_json(events)

        parsed = json.loads(json_str)
        assert len(parsed) == 1
        assert parsed[0]["event_type"] == "TEST"
        assert parsed[0]["event_id"] == "evt_1"
        assert parsed[0]["data"]["key"] == "value"

    def test_format_as_json_multiple_events(self):
        """Test JSON formatting with multiple events"""
        events = [
            MockEvent(MockEventType("EVENT_1"), "evt_1", {"data": 1}),
            MockEvent(MockEventType("EVENT_2"), "evt_2", {"data": 2}),
        ]

        service = ContextFormattingService()
        json_str = service.format_as_json(events)

        parsed = json.loads(json_str)
        assert len(parsed) == 2
        assert parsed[0]["data"]["data"] == 1
        assert parsed[1]["data"]["data"] == 2

    def test_format_as_json_empty_events(self):
        """Test JSON formatting with empty events list"""
        service = ContextFormattingService()
        json_str = service.format_as_json([])

        parsed = json.loads(json_str)
        assert parsed == []


class TestFormatCompressed:
    """Test compressed formatting"""

    def test_format_compressed_basic(self):
        """Test basic compressed formatting"""
        events = [
            MockEvent(MockEventType("TEST_EVENT"), "evt_1", {"summary": "Test summary"}),
        ]

        service = ContextFormattingService()
        compressed = service.format_compressed(events)

        assert "COMPRESSED CONTEXT:" in compressed
        assert "TEST_EVENT: 1 events" in compressed
        assert "Test summary" in compressed

    def test_format_compressed_counts_by_type(self):
        """Test that compressed format counts events by type"""
        events = [
            MockEvent(MockEventType("TYPE_A"), "evt_1", {"summary": "A1"}),
            MockEvent(MockEventType("TYPE_A"), "evt_2", {"summary": "A2"}),
            MockEvent(MockEventType("TYPE_A"), "evt_3", {"summary": "A3"}),
            MockEvent(MockEventType("TYPE_B"), "evt_4", {"summary": "B1"}),
        ]

        service = ContextFormattingService()
        compressed = service.format_compressed(events)

        # Should show counts and latest summary
        assert "TYPE_A" in compressed
        assert "3 events" in compressed
        assert "TYPE_B" in compressed
        assert "1 events" in compressed

    def test_format_compressed_shows_latest_summary(self):
        """Test that compressed format shows latest event summary"""
        events = [
            MockEvent(MockEventType("EVENT"), "evt_1", {"summary": "First"}),
            MockEvent(MockEventType("EVENT"), "evt_2", {"summary": "Second"}),
            MockEvent(MockEventType("EVENT"), "evt_3", {"summary": "Latest"}),
        ]

        service = ContextFormattingService()
        compressed = service.format_compressed(events)

        # Should show the latest summary
        assert "Latest" in compressed
        assert "3 events" in compressed

    def test_format_compressed_truncates_long_summaries(self):
        """Test that long summaries are truncated"""
        long_summary = "A" * 100

        events = [
            MockEvent(MockEventType("EVENT"), "evt_1", {"summary": long_summary}),
        ]

        service = ContextFormattingService()
        compressed = service.format_compressed(events)

        # Should be truncated to 50 chars + "..."
        assert "A" * 50 + "..." in compressed
        assert len(long_summary) > len(compressed)

    def test_format_compressed_handles_no_summary(self):
        """Test compressed format with events that have no summary"""
        events = [
            MockEvent(MockEventType("EVENT"), "evt_1", {"other_data": "value"}),
        ]

        service = ContextFormattingService()
        compressed = service.format_compressed(events)

        assert "No summary" in compressed


class TestFormatForLLM:
    """Test LLM-optimized formatting"""

    def test_format_for_llm_structured(self):
        """Test LLM formatting with structured (XML) type"""
        events = [
            MockEvent(MockEventType("TEST"), "evt_1", {"key": "value"}),
        ]

        service = ContextFormattingService()
        formatted = service.format_for_llm(events, format_type="structured")

        # Should be XML
        assert "<context>" in formatted
        assert "</context>" in formatted

    def test_format_for_llm_compressed(self):
        """Test LLM formatting with compressed type"""
        events = [
            MockEvent(MockEventType("TEST"), "evt_1", {"summary": "Test"}),
        ]

        service = ContextFormattingService()
        formatted = service.format_for_llm(events, format_type="compressed")

        # Should be compressed format
        assert "COMPRESSED CONTEXT:" in formatted

    def test_format_for_llm_json(self):
        """Test LLM formatting with JSON type"""
        events = [
            MockEvent(MockEventType("TEST"), "evt_1", {"key": "value"}),
        ]

        service = ContextFormattingService()
        formatted = service.format_for_llm(events, format_type="json")

        # Should be valid JSON
        parsed = json.loads(formatted)
        assert len(parsed) == 1

    def test_format_for_llm_default(self):
        """Test LLM formatting with default (structured) type"""
        events = [
            MockEvent(MockEventType("TEST"), "evt_1", {"key": "value"}),
        ]

        service = ContextFormattingService()
        formatted = service.format_for_llm(events)

        # Default should be XML/structured
        assert "<context>" in formatted


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_event_data(self):
        """Test formatting events with empty data"""
        events = [
            MockEvent(MockEventType("EMPTY"), "evt_1", {}),
        ]

        service = ContextFormattingService()

        # Should not crash
        xml = service.format_as_xml(events)
        assert "<EMPTY>" in xml

        json_str = service.format_as_json(events)
        assert json.loads(json_str)

        compressed = service.format_compressed(events)
        assert "EMPTY" in compressed

    def test_event_id_truncation(self):
        """Test that long event IDs are truncated in XML"""
        long_id = "evt_" + "x" * 100

        events = [
            MockEvent(MockEventType("EVENT"), long_id, {"key": "value"}),
        ]

        service = ContextFormattingService()
        xml = service.format_as_xml(events)

        # Should only show first 8 characters
        assert long_id[:8] in xml
        assert long_id not in xml  # Full ID should not be present


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
