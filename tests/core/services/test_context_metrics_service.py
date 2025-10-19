"""
Tests for ContextMetricsService

Task 7.0: Comprehensive test coverage for context metrics and relevance.
Target: ≥90% coverage

This service includes the complex compress_old_events method (CC=18)
requiring extensive edge case testing.

Created: 2025-10-19
Campaign: Operation Lean
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timezone, timedelta
from src.core.services.context_metrics_service import ContextMetricsService


class MockEvent:
    """Mock context event for testing"""
    def __init__(
        self,
        event_type,
        event_id,
        data,
        timestamp=None,
        relevance_score=0.5,
        access_count=0,
        compressed_version=None,
        can_compress=True
    ):
        self.event_type = event_type
        self.event_id = event_id
        self.data = data
        self.timestamp = timestamp or datetime.now(timezone.utc)
        self.relevance_score = relevance_score
        self.access_count = access_count
        self.last_accessed = None
        self.compressed_version = compressed_version
        self.can_compress = can_compress


class MockEventType:
    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        if isinstance(other, MockEventType):
            return self.value == other.value
        return False

    def __hash__(self):
        return hash(self.value)


class TestContextMetricsServiceInit:
    """Test service initialization"""

    def test_init_basic(self):
        """Test basic initialization"""
        events = []
        event_index = {}

        service = ContextMetricsService(
            events=events,
            event_index=event_index
        )

        assert service.events == events
        assert service.event_index == event_index
        assert service.max_events == 10000

    def test_init_with_max_events(self):
        """Test initialization with custom max_events"""
        events = []
        event_index = {}

        service = ContextMetricsService(
            events=events,
            event_index=event_index,
            max_events=5000
        )

        assert service.max_events == 5000


class TestGetRelevantContext:
    """Test relevance-based context filtering"""

    def test_get_relevant_context_empty_events(self):
        """Test with no events"""
        service = ContextMetricsService(events=[], event_index={})

        relevant = service.get_relevant_context()

        assert len(relevant) == 0

    @patch('src.core.unified_context_stream.ContextEventType')
    def test_get_relevant_context_filters_by_relevance(self, MockContextEventType):
        """Test that events are filtered by relevance threshold"""
        events = [
            MockEvent(MockEventType("EVENT_1"), "evt_1", {"data": 1}, relevance_score=0.8),
            MockEvent(MockEventType("EVENT_2"), "evt_2", {"data": 2}, relevance_score=0.2),
            MockEvent(MockEventType("EVENT_3"), "evt_3", {"data": 3}, relevance_score=0.5),
        ]

        service = ContextMetricsService(events=events, event_index={})
        relevant = service.get_relevant_context(min_relevance=0.3)

        # Should include events with relevance >= 0.3
        assert len(relevant) >= 2  # evt_1 and evt_3

    @patch('src.core.unified_context_stream.ContextEventType')
    def test_get_relevant_context_updates_access_count(self, MockContextEventType):
        """Test that relevant events have access count incremented"""
        event = MockEvent(MockEventType("EVENT"), "evt_1", {"data": 1}, access_count=0)
        events = [event]

        service = ContextMetricsService(events=events, event_index={})
        relevant = service.get_relevant_context(min_relevance=0.1)

        assert len(relevant) == 1
        assert relevant[0].access_count == 1
        assert relevant[0].last_accessed is not None


class TestGetRecentEvents:
    """Test recent events retrieval"""

    def test_get_recent_events_empty_events(self):
        """Test with no events"""
        service = ContextMetricsService(events=[], event_index={})

        recent = service.get_recent_events()

        assert len(recent) == 0

    def test_get_recent_events_default_limit(self):
        """Test getting recent events with default limit"""
        events = [
            MockEvent(MockEventType(f"EVENT_{i}"), f"evt_{i}", {"index": i})
            for i in range(20)
        ]

        service = ContextMetricsService(events=events, event_index={})
        recent = service.get_recent_events()

        # Should return last 10 events
        assert len(recent) == 10
        assert recent[-1].data["index"] == 19

    def test_get_recent_events_custom_limit(self):
        """Test getting recent events with custom limit"""
        events = [
            MockEvent(MockEventType(f"EVENT_{i}"), f"evt_{i}", {"index": i})
            for i in range(20)
        ]

        service = ContextMetricsService(events=events, event_index={})
        recent = service.get_recent_events(limit=5)

        # Should return last 5 events
        assert len(recent) == 5
        assert recent[-1].data["index"] == 19

    def test_get_recent_events_limit_exceeds_total(self):
        """Test when limit exceeds total events"""
        events = [
            MockEvent(MockEventType("EVENT"), "evt_1", {"data": 1}),
            MockEvent(MockEventType("EVENT"), "evt_2", {"data": 2}),
        ]

        service = ContextMetricsService(events=events, event_index={})
        recent = service.get_recent_events(limit=100)

        # Should return all events
        assert len(recent) == 2


class TestCalculateInitialRelevance:
    """Test initial relevance calculation"""

    @patch('src.core.unified_context_stream.ContextEventType')
    def test_calculate_initial_relevance_critical_events(self, MockContextEventType):
        """Test that critical events get 1.0 relevance"""
        critical_type = MockEventType("ENGAGEMENT_STARTED")
        MockContextEventType.ENGAGEMENT_STARTED = critical_type
        MockContextEventType.HITL_RESPONSE = MockEventType("HITL_RESPONSE")
        MockContextEventType.ERROR_OCCURRED = MockEventType("ERROR_OCCURRED")

        service = ContextMetricsService(events=[], event_index={})
        relevance = service.calculate_initial_relevance(critical_type)

        assert relevance == 1.0

    @patch('src.core.unified_context_stream.ContextEventType')
    def test_calculate_initial_relevance_moderate_events(self, MockContextEventType):
        """Test that moderate events get 0.7 relevance"""
        moderate_type = MockEventType("RESEARCH_RESULT")
        MockContextEventType.RESEARCH_RESULT = moderate_type
        MockContextEventType.REASONING_STEP = MockEventType("REASONING_STEP")
        MockContextEventType.MODEL_APPLIED = MockEventType("MODEL_APPLIED")
        MockContextEventType.ENGAGEMENT_STARTED = MockEventType("ENGAGEMENT_STARTED")
        MockContextEventType.HITL_RESPONSE = MockEventType("HITL_RESPONSE")
        MockContextEventType.ERROR_OCCURRED = MockEventType("ERROR_OCCURRED")

        service = ContextMetricsService(events=[], event_index={})
        relevance = service.calculate_initial_relevance(moderate_type)

        assert relevance == 0.7

    @patch('src.core.unified_context_stream.ContextEventType')
    def test_calculate_initial_relevance_default_events(self, MockContextEventType):
        """Test that other events get 0.5 relevance"""
        default_type = MockEventType("SOME_OTHER_EVENT")
        MockContextEventType.ENGAGEMENT_STARTED = MockEventType("ENGAGEMENT_STARTED")
        MockContextEventType.HITL_RESPONSE = MockEventType("HITL_RESPONSE")
        MockContextEventType.ERROR_OCCURRED = MockEventType("ERROR_OCCURRED")
        MockContextEventType.RESEARCH_RESULT = MockEventType("RESEARCH_RESULT")
        MockContextEventType.REASONING_STEP = MockEventType("REASONING_STEP")
        MockContextEventType.MODEL_APPLIED = MockEventType("MODEL_APPLIED")

        service = ContextMetricsService(events=[], event_index={})
        relevance = service.calculate_initial_relevance(default_type)

        assert relevance == 0.5


class TestRecalculateRelevance:
    """Test relevance recalculation"""

    @patch('src.core.unified_context_stream.ContextEventType')
    def test_recalculate_relevance_recent_event(self, MockContextEventType):
        """Test that recent events maintain high relevance"""
        MockContextEventType.ENGAGEMENT_STARTED = MockEventType("ENGAGEMENT_STARTED")
        MockContextEventType.HITL_RESPONSE = MockEventType("HITL_RESPONSE")
        MockContextEventType.ERROR_OCCURRED = MockEventType("ERROR_OCCURRED")

        event = MockEvent(
            MockEventType("EVENT"),
            "evt_1",
            {"data": 1},
            timestamp=datetime.now(timezone.utc),
            access_count=0
        )

        service = ContextMetricsService(events=[], event_index={})
        relevance = service.recalculate_relevance(event)

        # Recent event should have high relevance
        assert relevance >= 0.5

    @patch('src.core.unified_context_stream.ContextEventType')
    def test_recalculate_relevance_old_event(self, MockContextEventType):
        """Test that old events have lower relevance"""
        MockContextEventType.ENGAGEMENT_STARTED = MockEventType("ENGAGEMENT_STARTED")
        MockContextEventType.HITL_RESPONSE = MockEventType("HITL_RESPONSE")
        MockContextEventType.ERROR_OCCURRED = MockEventType("ERROR_OCCURRED")

        event = MockEvent(
            MockEventType("EVENT"),
            "evt_1",
            {"data": 1},
            timestamp=datetime.now(timezone.utc) - timedelta(hours=3),
            access_count=0
        )

        service = ContextMetricsService(events=[], event_index={})
        relevance = service.recalculate_relevance(event)

        # Old event should have lower relevance (but minimum 0.1)
        assert 0.1 <= relevance < 0.7

    @patch('src.core.unified_context_stream.ContextEventType')
    def test_recalculate_relevance_frequently_accessed(self, MockContextEventType):
        """Test that frequently accessed events get boosted relevance"""
        MockContextEventType.ENGAGEMENT_STARTED = MockEventType("ENGAGEMENT_STARTED")
        MockContextEventType.HITL_RESPONSE = MockEventType("HITL_RESPONSE")
        MockContextEventType.ERROR_OCCURRED = MockEventType("ERROR_OCCURRED")

        event = MockEvent(
            MockEventType("EVENT"),
            "evt_1",
            {"data": 1},
            timestamp=datetime.now(timezone.utc) - timedelta(hours=1),
            access_count=10  # Frequently accessed
        )

        service = ContextMetricsService(events=[], event_index={})
        relevance_high_access = service.recalculate_relevance(event)

        event.access_count = 0
        relevance_low_access = service.recalculate_relevance(event)

        # High access should boost relevance
        assert relevance_high_access > relevance_low_access

    @patch('src.core.unified_context_stream.ContextEventType')
    def test_recalculate_relevance_bounds(self, MockContextEventType):
        """Test that relevance is bounded between 0.1 and 1.0"""
        MockContextEventType.ENGAGEMENT_STARTED = MockEventType("ENGAGEMENT_STARTED")
        MockContextEventType.HITL_RESPONSE = MockEventType("HITL_RESPONSE")
        MockContextEventType.ERROR_OCCURRED = MockEventType("ERROR_OCCURRED")

        # Very old event (should hit lower bound)
        old_event = MockEvent(
            MockEventType("EVENT"),
            "evt_1",
            {"data": 1},
            timestamp=datetime.now(timezone.utc) - timedelta(days=7),
            access_count=0
        )

        service = ContextMetricsService(events=[], event_index={})
        old_relevance = service.recalculate_relevance(old_event)

        assert 0.1 <= old_relevance <= 1.0

        # Recent high-importance event (should hit upper bound)
        critical_type = MockEventType("ENGAGEMENT_STARTED")
        MockContextEventType.ENGAGEMENT_STARTED = critical_type

        new_event = MockEvent(
            critical_type,
            "evt_2",
            {"data": 2},
            timestamp=datetime.now(timezone.utc),
            access_count=20
        )

        new_relevance = service.recalculate_relevance(new_event)

        assert 0.1 <= new_relevance <= 1.0


class TestCompressOldEvents:
    """Test old event compression (CC=18 complex method)"""

    def test_compress_old_events_below_threshold(self):
        """Test that no compression occurs when below max_events"""
        events = [
            MockEvent(MockEventType("EVENT"), f"evt_{i}", {"data": i})
            for i in range(100)
        ]
        event_index = {f"evt_{i}": events[i] for i in range(100)}

        service = ContextMetricsService(
            events=events,
            event_index=event_index,
            max_events=1000
        )

        initial_count = len(service.events)
        service.compress_old_events()

        # No events should be removed
        assert len(service.events) == initial_count

    def test_compress_old_events_removes_oldest(self):
        """Test that oldest events are removed when exceeding max_events"""
        events = [
            MockEvent(MockEventType("EVENT"), f"evt_{i}", {"data": i})
            for i in range(150)
        ]
        event_index = {f"evt_{i}": events[i] for i in range(150)}

        service = ContextMetricsService(
            events=events,
            event_index=event_index,
            max_events=100
        )

        service.compress_old_events()

        # Should keep only 100 most recent events
        assert len(service.events) == 100
        # Should have removed first 50 events
        assert service.events[0].event_id == "evt_50"

    def test_compress_old_events_updates_index(self):
        """Test that event index is updated when events are removed"""
        events = [
            MockEvent(MockEventType("EVENT"), f"evt_{i}", {"data": i})
            for i in range(150)
        ]
        event_index = {f"evt_{i}": events[i] for i in range(150)}

        service = ContextMetricsService(
            events=events,
            event_index=event_index,
            max_events=100
        )

        service.compress_old_events()

        # Removed events should not be in index
        assert "evt_0" not in service.event_index
        assert "evt_49" not in service.event_index
        # Kept events should still be in index
        assert "evt_50" in service.event_index
        assert "evt_149" in service.event_index

    def test_compress_old_events_compresses_low_relevance(self):
        """Test that old low-relevance events are compressed"""
        # Create old event with low relevance
        old_timestamp = datetime.now(timezone.utc) - timedelta(hours=2)

        # Create enough events to exceed max_events threshold
        # This triggers the compression logic (which only runs during removal)
        events = []
        for i in range(150):
            if i == 100:  # Add our test event at position 100 (will be kept)
                events.append(
                    MockEvent(
                        MockEventType("EVENT"),
                        "evt_100",
                        {"description": "This is a test event"},
                        timestamp=old_timestamp,
                        relevance_score=0.3,
                        compressed_version=None
                    )
                )
            else:
                events.append(
                    MockEvent(
                        MockEventType("EVENT"),
                        f"evt_{i}",
                        {"data": i},
                        timestamp=datetime.now(timezone.utc),
                        relevance_score=0.5
                    )
                )

        event_index = {e.event_id: e for e in events}

        service = ContextMetricsService(
            events=events,
            event_index=event_index,
            max_events=100
        )

        # Find our test event
        test_event = next(e for e in events if e.event_id == "evt_100")

        service.compress_old_events()

        # Test event should be compressed (it's old and low relevance)
        assert test_event.compressed_version is not None
        assert test_event.data["compressed"] is True

    def test_compress_old_events_preserves_recent(self):
        """Test that recent events are not compressed"""
        # Create recent event
        recent_timestamp = datetime.now(timezone.utc) - timedelta(minutes=10)
        events = [
            MockEvent(
                MockEventType("EVENT"),
                "evt_1",
                {"description": "This is a test event"},
                timestamp=recent_timestamp,
                relevance_score=0.3,
                compressed_version=None
            )
        ]
        event_index = {"evt_1": events[0]}

        service = ContextMetricsService(
            events=events,
            event_index=event_index,
            max_events=100
        )

        service.compress_old_events()

        # Recent event should not be compressed
        assert events[0].compressed_version is None
        assert not events[0].data.get("compressed", False)

    def test_compress_old_events_preserves_high_relevance(self):
        """Test that high-relevance events are not compressed"""
        # Create old event with high relevance
        old_timestamp = datetime.now(timezone.utc) - timedelta(hours=2)
        events = [
            MockEvent(
                MockEventType("EVENT"),
                "evt_1",
                {"description": "This is a test event"},
                timestamp=old_timestamp,
                relevance_score=0.8,  # High relevance
                compressed_version=None
            )
        ]
        event_index = {"evt_1": events[0]}

        service = ContextMetricsService(
            events=events,
            event_index=event_index,
            max_events=100
        )

        service.compress_old_events()

        # High-relevance event should not be compressed
        assert events[0].compressed_version is None

    def test_compress_old_events_skips_already_compressed(self):
        """Test that already compressed events are not re-compressed"""
        old_timestamp = datetime.now(timezone.utc) - timedelta(hours=2)
        events = [
            MockEvent(
                MockEventType("EVENT"),
                "evt_1",
                {"compressed": True},  # Already compressed
                timestamp=old_timestamp,
                relevance_score=0.3,
                compressed_version="already compressed"
            )
        ]
        event_index = {"evt_1": events[0]}

        service = ContextMetricsService(
            events=events,
            event_index=event_index,
            max_events=100
        )

        original_compressed_version = events[0].compressed_version
        service.compress_old_events()

        # Should not re-compress
        assert events[0].compressed_version == original_compressed_version


class TestSummarizeEvent:
    """Test event summarization"""

    def test_summarize_event_with_summary(self):
        """Test summarizing event that has summary field"""
        event = MockEvent(
            MockEventType("EVENT"),
            "evt_1",
            {"summary": "This is a test summary"}
        )

        service = ContextMetricsService(events=[], event_index={})
        summary = service.summarize_event(event)

        assert summary == "This is a test summary"

    def test_summarize_event_with_description(self):
        """Test summarizing event with description field"""
        event = MockEvent(
            MockEventType("EVENT"),
            "evt_1",
            {"description": "This is a test description"}
        )

        service = ContextMetricsService(events=[], event_index={})
        summary = service.summarize_event(event)

        assert summary == "This is a test description"

    def test_summarize_event_with_result(self):
        """Test summarizing event with result field"""
        event = MockEvent(
            MockEventType("EVENT"),
            "evt_1",
            {"result": "Test result data"}
        )

        service = ContextMetricsService(events=[], event_index={})
        summary = service.summarize_event(event)

        assert "Result:" in summary
        assert "Test result data" in summary

    def test_summarize_event_fallback(self):
        """Test summarizing event with no known fields"""
        event = MockEvent(
            MockEventType("CUSTOM_EVENT"),
            "evt_1",
            {"unknown_field": "data"}
        )

        service = ContextMetricsService(events=[], event_index={})
        summary = service.summarize_event(event)

        assert "CUSTOM_EVENT event" == summary

    def test_summarize_event_truncates_long_text(self):
        """Test that long summaries are truncated to 100 characters"""
        long_summary = "A" * 150

        event = MockEvent(
            MockEventType("EVENT"),
            "evt_1",
            {"summary": long_summary}
        )

        service = ContextMetricsService(events=[], event_index={})
        summary = service.summarize_event(event)

        assert len(summary) == 100
        assert summary == "A" * 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
