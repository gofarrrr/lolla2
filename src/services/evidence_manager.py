# EvidenceManager service (foundation)
from __future__ import annotations

from typing import Optional, Dict
import logging

from src.models.evidence import Evidence, fingerprint_content

logger = logging.getLogger(__name__)


class EvidenceManager:
    """Foundational evidence service: fingerprinting and provenance tracking.

    For MVP: in-memory store with optional hooks for Supabase persistence later.
    """

    def __init__(self):
        self._by_fp: Dict[str, Evidence] = {}
        self._by_id: Dict[str, Evidence] = {}

    def upsert_evidence(
        self,
        *,
        content: str,
        source_ref: str,
        source_type: str = "doc",
        metadata: Optional[Dict] = None,
    ) -> Evidence:
        fp = fingerprint_content(content)
        if fp in self._by_fp:
            return self._by_fp[fp]
        ev = Evidence.new_from_content(
            content=content,
            source_ref=source_ref,
            source_type=source_type,
            metadata=metadata,
        )
        self._by_fp[fp] = ev
        self._by_id[ev.id] = ev
        logger.info(f"📚 Evidence stored fp={fp[:10]}... src={source_ref}")
        return ev

    def get_by_fingerprint(self, fp: str) -> Optional[Evidence]:
        return self._by_fp.get(fp)

    def get(self, evidence_id: str) -> Optional[Evidence]:
        return self._by_id.get(evidence_id)
