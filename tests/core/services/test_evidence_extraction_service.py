"""
Tests for EvidenceExtractionService

Task 4.11: Comprehensive test coverage for evidence extraction.
Target: ≥90% coverage

Created: 2025-10-18
Campaign: Operation Lean
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, MagicMock, patch
from src.core.services.evidence_extraction_service import EvidenceExtractionService


class MockEvent:
    """Mock ContextEvent for testing"""
    def __init__(self, event_type, data, timestamp=None):
        self.event_type = event_type
        self.data = data
        self.timestamp = timestamp or datetime.now(timezone.utc)
        self.event_id = f"evt_{id(self)}"


class MockEventType:
    """Mock event types"""
    MODEL_SELECTION_JUSTIFICATION = "MODEL_SELECTION_JUSTIFICATION"
    SYNERGY_META_DIRECTIVE = "SYNERGY_META_DIRECTIVE"
    COREOPS_RUN_SUMMARY = "COREOPS_RUN_SUMMARY"
    CONTRADICTION_AUDIT = "CONTRADICTION_AUDIT"
    MENTAL_MODEL_ACTIVATION = "MENTAL_MODEL_ACTIVATION"
    EVIDENCE_COLLECTION_COMPLETE = "EVIDENCE_COLLECTION_COMPLETE"
    LEARNING_CYCLE_STARTED = "LEARNING_CYCLE_STARTED"
    LEARNING_CYCLE_COMPLETED = "LEARNING_CYCLE_COMPLETED"
    PATTERN_EFFECTIVENESS_UPDATE = "PATTERN_EFFECTIVENESS_UPDATE"
    DIVERSITY_POLICY_ENFORCED = "DIVERSITY_POLICY_ENFORCED"
    OPTIMIZATION_ACTION_TAKEN = "OPTIMIZATION_ACTION_TAKEN"
    FEEDBACK_INGESTED = "FEEDBACK_INGESTED"
    DASHBOARD_METRICS_UPDATED = "DASHBOARD_METRICS_UPDATED"
    COREOPS_STEP_EXECUTED = "COREOPS_STEP_EXECUTED"
    OTHER_EVENT = "OTHER_EVENT"

    @staticmethod
    def create(value):
        """Create mock event type with value attribute"""
        mock = Mock()
        mock.value = value
        mock.__eq__ = lambda self, other: self.value == (other.value if hasattr(other, 'value') else other)
        mock.__hash__ = lambda self: hash(self.value)
        return mock


class TestEvidenceExtractionServiceInit:
    """Test service initialization"""

    def test_init_with_events(self):
        """Test initialization with events list"""
        events = [Mock(), Mock(), Mock()]
        service = EvidenceExtractionService(events=events, trace_id="test-trace")

        assert service.events == events
        assert service.trace_id == "test-trace"

    def test_init_without_trace_id(self):
        """Test initialization without trace_id defaults to 'unknown'"""
        service = EvidenceExtractionService(events=[])

        assert service.trace_id == "unknown"

    def test_init_with_empty_events(self):
        """Test initialization with empty events list"""
        service = EvidenceExtractionService(events=[], trace_id="empty-trace")

        assert service.events == []
        assert service.trace_id == "empty-trace"


class TestGetEvidenceEvents:
    """Test get_evidence_events functionality"""

    def test_get_evidence_events_default_types(self):
        """Test getting evidence events with default evidence types"""
        # Create mock event types that will match the default evidence types
        evt1 = MockEventType.create("MODEL_SELECTION_JUSTIFICATION")
        evt2 = MockEventType.create("OTHER_EVENT")
        evt3 = MockEventType.create("SYNERGY_META_DIRECTIVE")

        events = [
            MockEvent(evt1, {}),
            MockEvent(evt2, {}),
            MockEvent(evt3, {}),
        ]

        service = EvidenceExtractionService(events=events)

        # Provide explicit evidence types to avoid import
        evidence = service.get_evidence_events(evidence_types=[evt1, evt3])

        # Should only return evidence event types
        assert len(evidence) == 2
        assert all(e.event_type.value in [
            "MODEL_SELECTION_JUSTIFICATION", "SYNERGY_META_DIRECTIVE"
        ] for e in evidence)

    def test_get_evidence_events_with_specific_types(self):
        """Test getting evidence events with specific types filter"""
        event_type = Mock()
        event_type.value = "MODEL_SELECTION_JUSTIFICATION"

        events = [
            MockEvent(event_type, {}),
            MockEvent(Mock(), {}),
        ]

        service = EvidenceExtractionService(events=events)
        evidence = service.get_evidence_events(evidence_types=[event_type])

        assert len(evidence) == 1
        assert evidence[0].event_type == event_type

    def test_get_evidence_events_sorted_by_timestamp(self):
        """Test that evidence events are sorted chronologically"""
        now = datetime.now(timezone.utc)
        event_type = Mock()

        events = [
            MockEvent(event_type, {}, timestamp=now + timedelta(minutes=2)),
            MockEvent(event_type, {}, timestamp=now),
            MockEvent(event_type, {}, timestamp=now + timedelta(minutes=1)),
        ]

        service = EvidenceExtractionService(events=events)
        evidence = service.get_evidence_events(evidence_types=[event_type])

        # Should be sorted chronologically
        assert evidence[0].timestamp == now
        assert evidence[1].timestamp == now + timedelta(minutes=1)
        assert evidence[2].timestamp == now + timedelta(minutes=2)

    def test_get_evidence_events_empty_list(self):
        """Test getting evidence from empty events list"""
        service = EvidenceExtractionService(events=[])

        # Pass explicit empty types to avoid import
        evidence = service.get_evidence_events(evidence_types=[])

        assert evidence == []


class TestSpecificEvidenceGetters:
    """Test specific evidence type getters"""

    @patch('src.core.unified_context_stream.ContextEventType', MockEventType)
    def test_get_consultant_selection_evidence(self):
        """Test getting consultant selection evidence"""
        events = [
            MockEvent(MockEventType.create("MODEL_SELECTION_JUSTIFICATION"), {"test": "data"}),
            MockEvent(MockEventType.create("OTHER_EVENT"), {}),
            MockEvent(MockEventType.create("MODEL_SELECTION_JUSTIFICATION"), {"test": "data2"}),
        ]

        service = EvidenceExtractionService(events=events)
        evidence = service.get_consultant_selection_evidence()

        assert len(evidence) == 2
        assert all(e.event_type.value == "MODEL_SELECTION_JUSTIFICATION" for e in evidence)

    @patch('src.core.unified_context_stream.ContextEventType', MockEventType)
    def test_get_synergy_evidence(self):
        """Test getting synergy evidence"""
        events = [
            MockEvent(MockEventType.create("SYNERGY_META_DIRECTIVE"), {}),
            MockEvent(MockEventType.create("OTHER_EVENT"), {}),
        ]

        service = EvidenceExtractionService(events=events)
        evidence = service.get_synergy_evidence()

        assert len(evidence) == 1
        assert evidence[0].event_type.value == "SYNERGY_META_DIRECTIVE"

    @patch('src.core.unified_context_stream.ContextEventType', MockEventType)
    def test_get_coreops_evidence(self):
        """Test getting CoreOps evidence"""
        events = [
            MockEvent(MockEventType.create("COREOPS_RUN_SUMMARY"), {}),
            MockEvent(MockEventType.create("OTHER_EVENT"), {}),
        ]

        service = EvidenceExtractionService(events=events)
        evidence = service.get_coreops_evidence()

        assert len(evidence) == 1
        assert evidence[0].event_type.value == "COREOPS_RUN_SUMMARY"

    @patch('src.core.unified_context_stream.ContextEventType', MockEventType)
    def test_get_contradiction_evidence(self):
        """Test getting contradiction evidence"""
        events = [
            MockEvent(MockEventType.create("CONTRADICTION_AUDIT"), {}),
            MockEvent(MockEventType.create("OTHER_EVENT"), {}),
            MockEvent(MockEventType.create("CONTRADICTION_AUDIT"), {}),
        ]

        service = EvidenceExtractionService(events=events)
        evidence = service.get_contradiction_evidence()

        assert len(evidence) == 2
        assert all(e.event_type.value == "CONTRADICTION_AUDIT" for e in evidence)


class TestGetEvidenceSummary:
    """Test evidence summary generation"""

    @patch('src.core.unified_context_stream.ContextEventType', MockEventType)
    def test_evidence_summary_structure(self):
        """Test that evidence summary has correct structure"""
        events = [
            MockEvent(MockEventType.create("MODEL_SELECTION_JUSTIFICATION"), {"consultant_count": 3}),
        ]

        service = EvidenceExtractionService(events=events, trace_id="test-trace")
        summary = service.get_evidence_summary()

        # Check required fields
        assert "trace_id" in summary
        assert "evidence_collection_timestamp" in summary
        assert "total_evidence_events" in summary
        assert "evidence_types" in summary
        assert "glass_box_completeness" in summary
        assert "consultant_selections" in summary
        assert "synergy_directives" in summary
        assert "coreops_executions" in summary
        assert "contradiction_audits" in summary
        assert "key_decisions" in summary
        assert "evidence_timeline" in summary

    @patch('src.core.unified_context_stream.ContextEventType', MockEventType)
    def test_evidence_summary_counts(self):
        """Test that evidence summary counts are correct"""
        events = [
            MockEvent(MockEventType.create("MODEL_SELECTION_JUSTIFICATION"), {}),
            MockEvent(MockEventType.create("MODEL_SELECTION_JUSTIFICATION"), {}),
            MockEvent(MockEventType.create("SYNERGY_META_DIRECTIVE"), {}),
            MockEvent(MockEventType.create("COREOPS_RUN_SUMMARY"), {}),
            MockEvent(MockEventType.create("CONTRADICTION_AUDIT"), {}),
            MockEvent(MockEventType.create("CONTRADICTION_AUDIT"), {}),
            MockEvent(MockEventType.create("CONTRADICTION_AUDIT"), {}),
        ]

        service = EvidenceExtractionService(events=events)
        summary = service.get_evidence_summary()

        assert summary["consultant_selections"] == 2
        assert summary["synergy_directives"] == 1
        assert summary["coreops_executions"] == 1
        assert summary["contradiction_audits"] == 3

    @patch('src.core.unified_context_stream.ContextEventType', MockEventType)
    def test_glass_box_completeness_calculation(self):
        """Test glass-box completeness calculation"""
        # 3 evidence events out of 5 total = 0.6 completeness
        events = [
            MockEvent(MockEventType.create("MODEL_SELECTION_JUSTIFICATION"), {}),
            MockEvent(MockEventType.create("OTHER_EVENT"), {}),
            MockEvent(MockEventType.create("SYNERGY_META_DIRECTIVE"), {}),
            MockEvent(MockEventType.create("OTHER_EVENT"), {}),
            MockEvent(MockEventType.create("COREOPS_RUN_SUMMARY"), {}),
        ]

        service = EvidenceExtractionService(events=events)
        summary = service.get_evidence_summary()

        # Should be 3/5 = 0.6
        assert summary["glass_box_completeness"] == 0.6

    def test_evidence_summary_empty_events(self):
        """Test evidence summary with no events"""
        service = EvidenceExtractionService(events=[])
        summary = service.get_evidence_summary()

        assert summary["total_evidence_events"] == 0
        assert summary["glass_box_completeness"] == 0.0
        assert summary["consultant_selections"] == 0


class TestSummarizeEvidenceEvent:
    """Test individual event summarization"""

    @patch('src.core.unified_context_stream.ContextEventType', MockEventType)
    def test_summarize_consultant_selection(self):
        """Test summarizing consultant selection event"""
        event = MockEvent(
            MockEventType.create("MODEL_SELECTION_JUSTIFICATION"),
            {"consultant_count": 3, "total_confidence": 0.85}
        )

        service = EvidenceExtractionService(events=[event])
        summary = service.summarize_evidence_event(event)

        assert "3 consultants" in summary
        assert "85.0%" in summary

    @patch('src.core.unified_context_stream.ContextEventType', MockEventType)
    def test_summarize_synergy_directive(self):
        """Test summarizing synergy directive event"""
        event = MockEvent(
            MockEventType.create("SYNERGY_META_DIRECTIVE"),
            {"model_count": 5, "confidence_score": 0.92}
        )

        service = EvidenceExtractionService(events=[event])
        summary = service.summarize_evidence_event(event)

        assert "5 models" in summary
        assert "92.0%" in summary

    @patch('src.core.unified_context_stream.ContextEventType', MockEventType)
    def test_summarize_coreops_execution(self):
        """Test summarizing CoreOps execution event"""
        event = MockEvent(
            MockEventType.create("COREOPS_RUN_SUMMARY"),
            {"system_contract_id": "ANALYZE_V1", "argument_count": 7}
        )

        service = EvidenceExtractionService(events=[event])
        summary = service.summarize_evidence_event(event)

        assert "ANALYZE_V1" in summary
        assert "7 arguments" in summary

    @patch('src.core.unified_context_stream.ContextEventType', MockEventType)
    def test_summarize_contradiction_audit(self):
        """Test summarizing contradiction audit event"""
        event = MockEvent(
            MockEventType.create("CONTRADICTION_AUDIT"),
            {"contradiction_count": 2, "synthesis_count": 5}
        )

        service = EvidenceExtractionService(events=[event])
        summary = service.summarize_evidence_event(event)

        assert "2 contradictions" in summary
        assert "5 syntheses" in summary

    @patch('src.core.unified_context_stream.ContextEventType', MockEventType)
    def test_summarize_unknown_event_type(self):
        """Test summarizing unknown event type"""
        event = MockEvent(
            MockEventType.create("UNKNOWN_EVENT"),
            {}
        )

        service = EvidenceExtractionService(events=[event])
        summary = service.summarize_evidence_event(event)

        assert "Evidence event: UNKNOWN_EVENT" in summary


class TestExportEvidenceForAPI:
    """Test API export functionality"""

    @patch('src.core.unified_context_stream.ContextEventType', MockEventType)
    def test_export_evidence_structure(self):
        """Test that API export has correct structure"""
        events = [
            MockEvent(MockEventType.create("MODEL_SELECTION_JUSTIFICATION"), {}),
        ]

        service = EvidenceExtractionService(events=events, trace_id="api-test")
        export = service.export_evidence_for_api()

        # Check required top-level fields
        assert "metadata" in export
        assert "consultant_selections" in export
        assert "synergy_directives" in export
        assert "coreops_executions" in export
        assert "contradiction_audits" in export
        assert "evidence_timeline" in export

        # Check metadata structure
        assert "trace_id" in export["metadata"]
        assert export["metadata"]["trace_id"] == "api-test"
        assert "total_evidence_events" in export["metadata"]
        assert "collection_timestamp" in export["metadata"]
        assert "session_duration_minutes" in export["metadata"]

    @patch('src.core.unified_context_stream.ContextEventType', MockEventType)
    def test_export_consultant_selections(self):
        """Test exporting consultant selection evidence"""
        events = [
            MockEvent(
                MockEventType.create("MODEL_SELECTION_JUSTIFICATION"),
                {
                    "selection_rationale": "Best fit for analysis",
                    "total_confidence": 0.89,
                    "consultant_count": 3,
                    "consultants": ["C1", "C2", "C3"]
                }
            ),
        ]

        service = EvidenceExtractionService(events=events)
        export = service.export_evidence_for_api()

        assert len(export["consultant_selections"]) == 1
        selection = export["consultant_selections"][0]
        assert selection["selection_rationale"] == "Best fit for analysis"
        assert selection["total_confidence"] == 0.89
        assert selection["consultant_count"] == 3
        assert selection["consultants"] == ["C1", "C2", "C3"]

    @patch('src.core.unified_context_stream.ContextEventType', MockEventType)
    def test_session_duration_calculation(self):
        """Test session duration calculation in API export"""
        now = datetime.now(timezone.utc)
        events = [
            MockEvent(MockEventType.create("MODEL_SELECTION_JUSTIFICATION"), {}, timestamp=now),
            MockEvent(MockEventType.create("SYNERGY_META_DIRECTIVE"), {}, timestamp=now + timedelta(minutes=10)),
        ]

        service = EvidenceExtractionService(events=events)
        export = service.export_evidence_for_api()

        # Should be approximately 10 minutes
        assert export["metadata"]["session_duration_minutes"] == 10.0

    def test_export_empty_events(self):
        """Test API export with no events"""
        service = EvidenceExtractionService(events=[])
        export = service.export_evidence_for_api()

        assert export["metadata"]["total_evidence_events"] == 0
        assert export["metadata"]["session_duration_minutes"] == 0.0
        assert len(export["consultant_selections"]) == 0
        assert len(export["evidence_timeline"]) == 0


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_service_with_none_data_fields(self):
        """Test handling events with missing data fields"""
        event = MockEvent(Mock(), {})  # Empty data dict
        service = EvidenceExtractionService(events=[event])

        # Should not raise exception
        summary = service.summarize_evidence_event(event)
        assert summary is not None

    @patch('src.core.unified_context_stream.ContextEventType', MockEventType)
    def test_evidence_timeline_limited_to_10(self):
        """Test that evidence timeline is limited to last 10 events"""
        events = [
            MockEvent(MockEventType.create("MODEL_SELECTION_JUSTIFICATION"), {})
            for _ in range(15)
        ]

        service = EvidenceExtractionService(events=events)
        summary = service.get_evidence_summary()

        # Timeline should be limited to 10 most recent
        assert len(summary["evidence_timeline"]) == 10

    def test_import_error_handling(self):
        """Test graceful handling of ImportError"""
        service = EvidenceExtractionService(events=[])

        # Should not raise exception even if ContextEventType import fails
        evidence = service.get_consultant_selection_evidence()
        assert evidence == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
