from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.core.services.context_metrics_service import ContextMetricsService
from src.core.services.evidence_extraction_service import EvidenceExtractionService
from src.interfaces.context_metrics import ContextMetricsAdapter
from src.interfaces.evidence import EvidenceExtractionAdapter


@pytest.fixture()
def context_metrics_service() -> MagicMock:
    return MagicMock(spec=ContextMetricsService)


@pytest.fixture()
def evidence_service() -> MagicMock:
    return MagicMock(spec=EvidenceExtractionService)


def test_context_metrics_adapter_proxies_calls(context_metrics_service: MagicMock) -> None:
    adapter = ContextMetricsAdapter(context_metrics_service)

    adapter.get_relevant_context(for_phase="analysis", min_relevance=0.75)
    adapter.get_recent_events(limit=5)
    adapter.recalculate_relevance(event="evt")
    adapter.compress_old_events()
    adapter.summarize_event(event="evt")

    context_metrics_service.get_relevant_context.assert_called_once_with(
        for_phase="analysis",
        min_relevance=0.75,
    )
    context_metrics_service.get_recent_events.assert_called_once_with(limit=5)
    context_metrics_service.recalculate_relevance.assert_called_once_with("evt")
    context_metrics_service.compress_old_events.assert_called_once_with()
    context_metrics_service.summarize_event.assert_called_once_with("evt")


def test_context_metrics_adapter_exposes_service_property(context_metrics_service: MagicMock) -> None:
    adapter = ContextMetricsAdapter(context_metrics_service)
    assert adapter.service is context_metrics_service


def test_evidence_adapter_proxies_calls(evidence_service: MagicMock) -> None:
    adapter = EvidenceExtractionAdapter(evidence_service)

    adapter.get_evidence_events(evidence_types=["foo"])
    adapter.get_consultant_selection_evidence()
    adapter.get_synergy_evidence()
    adapter.get_coreops_evidence()
    adapter.get_contradiction_evidence()
    adapter.get_evidence_summary()
    adapter.summarize_evidence_event(event={"id": 1})
    adapter.export_evidence_for_api()

    evidence_service.get_evidence_events.assert_called_once_with(evidence_types=["foo"])
    evidence_service.get_consultant_selection_evidence.assert_called_once_with()
    evidence_service.get_synergy_evidence.assert_called_once_with()
    evidence_service.get_coreops_evidence.assert_called_once_with()
    evidence_service.get_contradiction_evidence.assert_called_once_with()
    evidence_service.get_evidence_summary.assert_called_once_with()
    evidence_service.summarize_evidence_event.assert_called_once_with({"id": 1})
    evidence_service.export_evidence_for_api.assert_called_once_with()


def test_evidence_adapter_exposes_service_property(evidence_service: MagicMock) -> None:
    adapter = EvidenceExtractionAdapter(evidence_service)
    assert adapter.service is evidence_service
