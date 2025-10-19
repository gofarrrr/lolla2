"""
Tests for ContextPersistenceService

Task 6.0: Comprehensive test coverage for context persistence.
Target: ≥90% coverage

Created: 2025-10-19
Campaign: Operation Lean
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone, timedelta
from src.core.services.context_persistence_service import ContextPersistenceService


class MockEvent:
    """Mock context event for testing"""
    def __init__(self, event_type, event_id, data, relevance_score=1.0, compressed_version=None):
        self.event_type = event_type
        self.event_id = event_id
        self.data = data
        self.timestamp = datetime.now(timezone.utc)
        self.relevance_score = relevance_score
        self.compressed_version = compressed_version
        self.metadata = {}

    def to_dict(self):
        return {
            "event_type": self.event_type.value if hasattr(self.event_type, 'value') else str(self.event_type),
            "event_id": self.event_id,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "relevance_score": self.relevance_score,
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


class TestContextPersistenceServiceInit:
    """Test service initialization"""

    def test_init_basic(self):
        """Test basic initialization"""
        events = []
        trace_id = "trace_123"
        started_at = datetime.now(timezone.utc)

        service = ContextPersistenceService(
            events=events,
            trace_id=trace_id,
            started_at=started_at
        )

        assert service.events == events
        assert service.trace_id == trace_id
        assert service.started_at == started_at
        assert service.engagement_type == "consultation"
        assert service.case_id is None
        assert service.completed_at is None

    def test_init_with_optional_params(self):
        """Test initialization with optional parameters"""
        events = []
        trace_id = "trace_456"
        started_at = datetime.now(timezone.utc)

        service = ContextPersistenceService(
            events=events,
            trace_id=trace_id,
            started_at=started_at,
            session_id="session_789",
            user_id="user_123",
            organization_id="org_456",
            cache_hits=10,
            cache_misses=5
        )

        assert service.session_id == "session_789"
        assert service.user_id == "user_123"
        assert service.organization_id == "org_456"
        assert service.cache_hits == 10
        assert service.cache_misses == 5


class TestCreateCheckpoint:
    """Test checkpoint creation"""

    def test_create_checkpoint_empty_events(self):
        """Test checkpoint creation with no events"""
        events = []
        service = ContextPersistenceService(
            events=events,
            trace_id="trace_123",
            started_at=datetime.now(timezone.utc)
        )

        checkpoint = service.create_checkpoint()

        assert "events" in checkpoint
        assert len(checkpoint["events"]) == 0
        assert "stats" in checkpoint
        assert checkpoint["stats"]["total_events"] == 0
        assert "timestamp" in checkpoint

    def test_create_checkpoint_with_events(self):
        """Test checkpoint creation with events"""
        events = [
            MockEvent(MockEventType("EVENT_1"), "evt_1", {"data": 1}),
            MockEvent(MockEventType("EVENT_2"), "evt_2", {"data": 2}),
        ]
        service = ContextPersistenceService(
            events=events,
            trace_id="trace_123",
            started_at=datetime.now(timezone.utc),
            cache_hits=5,
            cache_misses=2
        )

        checkpoint = service.create_checkpoint()

        assert len(checkpoint["events"]) == 2
        assert checkpoint["stats"]["total_events"] == 2
        assert checkpoint["stats"]["cache_hits"] == 5
        assert checkpoint["stats"]["cache_misses"] == 2

    def test_create_checkpoint_limits_to_last_100(self):
        """Test that checkpoint only includes last 100 events"""
        events = [
            MockEvent(MockEventType("EVENT"), f"evt_{i}", {"index": i})
            for i in range(150)
        ]
        service = ContextPersistenceService(
            events=events,
            trace_id="trace_123",
            started_at=datetime.now(timezone.utc)
        )

        checkpoint = service.create_checkpoint()

        # Should only have last 100 events
        assert len(checkpoint["events"]) == 100
        # Stats should show all 150
        assert checkpoint["stats"]["total_events"] == 150


class TestRestoreFromCheckpoint:
    """Test checkpoint restoration"""

    @patch('src.core.unified_context_stream.ContextEventType')
    @patch('src.core.unified_context_stream.ContextEvent')
    def test_restore_from_checkpoint_basic(self, MockContextEvent, MockContextEventType):
        """Test basic checkpoint restoration"""
        checkpoint = {
            "events": [
                {
                    "event_type": "TEST_EVENT",
                    "event_id": "evt_1",
                    "data": {"key": "value"},
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "metadata": {},
                    "relevance_score": 1.0,
                }
            ],
            "stats": {
                "total_events": 1,
                "cache_hits": 3,
                "cache_misses": 1,
            }
        }

        mock_event = Mock()
        MockContextEvent.return_value = mock_event
        MockContextEventType.return_value = Mock()

        service = ContextPersistenceService(
            events=[],
            trace_id="trace_123",
            started_at=datetime.now(timezone.utc)
        )

        restored_events = service.restore_from_checkpoint(checkpoint)

        assert len(restored_events) == 1
        assert service.cache_hits == 3
        assert service.cache_misses == 1

    def test_restore_from_checkpoint_empty(self):
        """Test restoring from empty checkpoint"""
        checkpoint = {
            "events": [],
            "stats": {}
        }

        service = ContextPersistenceService(
            events=[],
            trace_id="trace_123",
            started_at=datetime.now(timezone.utc)
        )

        restored_events = service.restore_from_checkpoint(checkpoint)

        assert len(restored_events) == 0


class TestEngagementManagement:
    """Test engagement lifecycle management"""

    def test_set_engagement_context(self):
        """Test setting engagement context"""
        service = ContextPersistenceService(
            events=[],
            trace_id="trace_123",
            started_at=datetime.now(timezone.utc)
        )

        service.set_engagement_context(
            session_id="session_abc",
            user_id="user_def",
            organization_id="org_ghi",
            engagement_type="analysis",
            case_id="case_jkl"
        )

        assert service.session_id == "session_abc"
        assert service.user_id == "user_def"
        assert service.organization_id == "org_ghi"
        assert service.engagement_type == "analysis"
        assert service.case_id == "case_jkl"

    def test_complete_engagement(self):
        """Test completing an engagement"""
        started_at = datetime.now(timezone.utc) - timedelta(minutes=5)
        events = [MockEvent(MockEventType("EVENT"), "evt_1", {"data": 1})]

        service = ContextPersistenceService(
            events=events,
            trace_id="trace_123",
            started_at=started_at
        )

        completion_data = service.complete_engagement(final_status="success")

        assert service.completed_at is not None
        assert completion_data["final_status"] == "success"
        assert completion_data["total_events"] == 1
        assert completion_data["trace_id"] == "trace_123"
        assert "duration_ms" in completion_data
        assert completion_data["duration_ms"] > 0

    def test_set_final_analysis_text(self):
        """Test setting final analysis text"""
        service = ContextPersistenceService(
            events=[],
            trace_id="trace_123",
            started_at=datetime.now(timezone.utc)
        )

        analysis_text = "This is the final analysis report."
        service.set_final_analysis_text(analysis_text)

        assert service.final_analysis_text == analysis_text


class TestCalculateSummaryMetrics:
    """Test summary metrics calculation"""

    @patch('src.core.unified_context_stream.ContextEventType')
    def test_calculate_summary_metrics_basic(self, MockContextEventType):
        """Test basic summary metrics calculation"""
        MockContextEventType.ERROR_OCCURRED = MockEventType("ERROR_OCCURRED")

        events = [
            MockEvent(MockEventType("EVENT_1"), "evt_1", {"model": "gpt-4"}),
            MockEvent(MockEventType("EVENT_2"), "evt_2", {"tokens": 100, "cost": 0.05}),
        ]

        service = ContextPersistenceService(
            events=events,
            trace_id="trace_123",
            started_at=datetime.now(timezone.utc)
        )

        metrics = service.calculate_summary_metrics()

        assert metrics["total_tokens"] == 100
        assert metrics["total_cost"] == 0.05
        assert metrics["error_count"] == 0
        assert "gpt-4" in metrics["models_used"]
        assert metrics["final_status"] == "completed"

    @patch('src.core.unified_context_stream.ContextEventType')
    def test_calculate_summary_metrics_with_errors(self, MockContextEventType):
        """Test summary metrics with error events"""
        error_type = MockEventType("ERROR_OCCURRED")
        MockContextEventType.ERROR_OCCURRED = error_type

        events = [
            MockEvent(MockEventType("EVENT_1"), "evt_1", {"data": 1}),
            MockEvent(error_type, "evt_2", {"error": "test"}),
            MockEvent(MockEventType("EVENT_3"), "evt_3", {"data": 3}),
        ]

        service = ContextPersistenceService(
            events=events,
            trace_id="trace_123",
            started_at=datetime.now(timezone.utc)
        )

        metrics = service.calculate_summary_metrics()

        assert metrics["error_count"] == 1
        assert metrics["final_status"] == "failed"  # 1/3 = 33% errors (> 10% threshold)

    @patch('src.core.unified_context_stream.ContextEventType')
    def test_calculate_summary_metrics_with_pii(self, MockContextEventType):
        """Test summary metrics with PII detection"""
        MockContextEventType.ERROR_OCCURRED = MockEventType("ERROR_OCCURRED")

        events = [
            MockEvent(MockEventType("EVENT_1"), "evt_1", {"user_email": "[REDACTED_EMAIL]"}),
            MockEvent(MockEventType("EVENT_2"), "evt_2", {"data": "normal data"}),
        ]

        service = ContextPersistenceService(
            events=events,
            trace_id="trace_123",
            started_at=datetime.now(timezone.utc)
        )

        metrics = service.calculate_summary_metrics()

        assert metrics["contains_pii"] is True

    @patch('src.core.unified_context_stream.ContextEventType')
    def test_calculate_summary_metrics_extracts_consultants(self, MockContextEventType):
        """Test that summary metrics extract consultant information"""
        MockContextEventType.ERROR_OCCURRED = MockEventType("ERROR_OCCURRED")

        events = [
            MockEvent(MockEventType("EVENT_1"), "evt_1", {"consultant": "analyst_1"}),
            MockEvent(MockEventType("EVENT_2"), "evt_2", {"consultant_id": "analyst_2"}),
            MockEvent(MockEventType("EVENT_3"), "evt_3", {"consultants_invoked": ["analyst_3", "analyst_4"]}),
        ]

        service = ContextPersistenceService(
            events=events,
            trace_id="trace_123",
            started_at=datetime.now(timezone.utc)
        )

        metrics = service.calculate_summary_metrics()

        assert len(metrics["consultants_used"]) == 4
        assert "analyst_1" in metrics["consultants_used"]
        assert "analyst_4" in metrics["consultants_used"]


class TestGetPerformanceMetrics:
    """Test performance metrics calculation"""

    def test_get_performance_metrics_basic(self):
        """Test basic performance metrics"""
        events = [
            MockEvent(MockEventType("TYPE_A"), "evt_1", {"data": 1}, relevance_score=0.8),
            MockEvent(MockEventType("TYPE_B"), "evt_2", {"data": 2}, relevance_score=0.3),
            MockEvent(MockEventType("TYPE_C"), "evt_3", {"data": 3}, relevance_score=0.9),
        ]

        service = ContextPersistenceService(
            events=events,
            trace_id="trace_123",
            started_at=datetime.now(timezone.utc),
            cache_hits=8,
            cache_misses=2
        )

        metrics = service.get_performance_metrics()

        assert metrics["total_events"] == 3
        assert metrics["relevant_events"] == 2  # 0.8 and 0.9 >= 0.5
        assert metrics["event_types"] == 3
        assert metrics["cache_hit_rate"] == 0.8  # 8/10
        assert metrics["cache_hits"] == 8
        assert metrics["cache_misses"] == 2

    def test_get_performance_metrics_with_compressed_events(self):
        """Test performance metrics with compressed events"""
        events = [
            MockEvent(MockEventType("EVENT_1"), "evt_1", {"data": 1}, compressed_version=None),
            MockEvent(MockEventType("EVENT_2"), "evt_2", {"data": 2}, compressed_version="compressed"),
            MockEvent(MockEventType("EVENT_3"), "evt_3", {"data": 3}, compressed_version=None),
        ]

        service = ContextPersistenceService(
            events=events,
            trace_id="trace_123",
            started_at=datetime.now(timezone.utc)
        )

        metrics = service.get_performance_metrics()

        assert metrics["memory_events"] == 2
        assert metrics["compressed_events"] == 1


class TestBuildPersistenceRecord:
    """Test persistence record building"""

    @patch('src.core.unified_context_stream.ContextEventType')
    def test_build_persistence_record_basic(self, MockContextEventType):
        """Test basic persistence record building"""
        MockContextEventType.ERROR_OCCURRED = MockEventType("ERROR_OCCURRED")

        events = [
            MockEvent(MockEventType("EVENT_1"), "evt_1", {"model": "gpt-4", "tokens": 50}),
        ]

        started_at = datetime.now(timezone.utc) - timedelta(minutes=2)

        service = ContextPersistenceService(
            events=events,
            trace_id="trace_abc",
            started_at=started_at,
            session_id="session_123",
            user_id="user_456"
        )

        service.complete_engagement("success")
        record = service.build_persistence_record()

        assert record["trace_id"] == "trace_abc"
        assert record["session_id"] == "session_123"
        assert record["user_id"] == "user_456"
        assert record["engagement_type"] == "consultation"
        assert record["final_status"] == "completed"
        assert "context_stream" in record
        assert "total_tokens" in record
        assert "indexed_at" in record

    @patch('src.core.unified_context_stream.ContextEventType')
    def test_build_persistence_record_with_pii_classification(self, MockContextEventType):
        """Test data classification based on PII presence"""
        MockContextEventType.ERROR_OCCURRED = MockEventType("ERROR_OCCURRED")

        events = [
            MockEvent(MockEventType("EVENT_1"), "evt_1", {"email": "[REDACTED_EMAIL]"}),
        ]

        service = ContextPersistenceService(
            events=events,
            trace_id="trace_def",
            started_at=datetime.now(timezone.utc)
        )

        record = service.build_persistence_record()

        assert record["contains_pii"] is True
        assert record["data_classification"] == "confidential"

    @patch('src.core.unified_context_stream.ContextEventType')
    def test_build_persistence_record_includes_final_analysis(self, MockContextEventType):
        """Test that final analysis text is included in record"""
        MockContextEventType.ERROR_OCCURRED = MockEventType("ERROR_OCCURRED")

        events = [
            MockEvent(MockEventType("EVENT_1"), "evt_1", {"data": 1}),
        ]

        service = ContextPersistenceService(
            events=events,
            trace_id="trace_ghi",
            started_at=datetime.now(timezone.utc)
        )

        service.set_final_analysis_text("This is the final report.")
        record = service.build_persistence_record()

        assert record["context_stream"]["summary"]["final_report"] == "This is the final report."


class TestPersistToDatabase:
    """Test database persistence"""

    @pytest.mark.asyncio
    async def test_persist_to_database_success(self):
        """Test successful database persistence"""
        mock_adapter = AsyncMock()
        mock_adapter.persist = AsyncMock(return_value=None)

        events = [
            MockEvent(MockEventType("EVENT_1"), "evt_1", {"data": 1}),
        ]

        service = ContextPersistenceService(
            events=events,
            trace_id="trace_123",
            started_at=datetime.now(timezone.utc),
            persistence_adapter=mock_adapter
        )

        result = await service.persist_to_database()

        assert result is True
        mock_adapter.persist.assert_called_once()

    @pytest.mark.asyncio
    async def test_persist_to_database_no_adapter(self):
        """Test persistence with no adapter (uses file adapter fallback)"""
        events = [
            MockEvent(MockEventType("EVENT_1"), "evt_1", {"data": 1}),
        ]

        service = ContextPersistenceService(
            events=events,
            trace_id="trace_123",
            started_at=datetime.now(timezone.utc),
            persistence_adapter=None
        )

        # This will try to import FileAdapter, which may not exist in test environment
        # We expect it to fail gracefully
        result = await service.persist_to_database()

        # Should return False due to missing adapter
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
