from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable

from src.core.services.evidence_extraction_service import EvidenceExtractionService


@runtime_checkable
class EvidenceExtractor(Protocol):
    """Contract for evidence extraction utilities consumed by infrastructure."""

    def get_evidence_events(self, evidence_types: Optional[List[Any]] = None) -> Sequence[Any]:
        """Return the ordered list of evidence events."""

    def get_consultant_selection_evidence(self) -> Sequence[Any]:
        """Return consultant selection justification events."""

    def get_synergy_evidence(self) -> Sequence[Any]:
        """Return synergy evidence events."""

    def get_coreops_evidence(self) -> Sequence[Any]:
        """Return CoreOps execution evidence events."""

    def get_contradiction_evidence(self) -> Sequence[Any]:
        """Return contradiction audit evidence events."""

    def get_evidence_summary(self) -> Dict[str, Any]:
        """Return aggregated summary metrics for evidence."""

    def summarize_evidence_event(self, event: Any) -> str:
        """Return human readable summary for an individual evidence event."""

    def export_evidence_for_api(self) -> Dict[str, Any]:
        """Return API-friendly evidence payload."""


class EvidenceExtractionAdapter(EvidenceExtractor):
    """Adapter exposing EvidenceExtractionService through the EvidenceExtractor protocol."""

    def __init__(self, service: EvidenceExtractionService) -> None:
        self._service = service

    @property
    def service(self) -> EvidenceExtractionService:
        """Expose the wrapped service for advanced usage."""
        return self._service

    def get_evidence_events(self, evidence_types: Optional[List[Any]] = None) -> Sequence[Any]:
        return self._service.get_evidence_events(evidence_types=evidence_types)

    def get_consultant_selection_evidence(self) -> Sequence[Any]:
        return self._service.get_consultant_selection_evidence()

    def get_synergy_evidence(self) -> Sequence[Any]:
        return self._service.get_synergy_evidence()

    def get_coreops_evidence(self) -> Sequence[Any]:
        return self._service.get_coreops_evidence()

    def get_contradiction_evidence(self) -> Sequence[Any]:
        return self._service.get_contradiction_evidence()

    def get_evidence_summary(self) -> Dict[str, Any]:
        return self._service.get_evidence_summary()

    def summarize_evidence_event(self, event: Any) -> str:
        return self._service.summarize_evidence_event(event)

    def export_evidence_for_api(self) -> Dict[str, Any]:
        return self._service.export_evidence_for_api()


__all__ = ["EvidenceExtractor", "EvidenceExtractionAdapter"]
