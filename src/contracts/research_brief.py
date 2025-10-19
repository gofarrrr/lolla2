"""
ResearchBrief and ResearchSource data contracts
Aligns with Manus context-engineering principles: deterministic serialization, prefix stability, and neutral context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import md5
from typing import List, Dict, Any, Optional


@dataclass
class ResearchSource:
    url: str
    title: Optional[str] = None
    snippet: Optional[str] = None
    credibility_score: Optional[float] = None
    published_at: Optional[str] = None
    source_type: Optional[str] = None  # blog, paper, news, report

    def to_ordered_dict(self) -> Dict[str, Any]:
        # Deterministic key ordering
        return {
            "url": self.url,
            "title": self.title or "",
            "snippet": self.snippet or "",
            "credibility_score": (
                float(self.credibility_score)
                if self.credibility_score is not None
                else None
            ),
            "published_at": self.published_at or "",
            "source_type": self.source_type or "",
        }


@dataclass
class ResearchBrief:
    query: str
    neutral_summary: str
    key_facts: List[str] = field(default_factory=list)
    sources: List[ResearchSource] = field(default_factory=list)
    tier: str = "regular"
    confidence: float = 0.0
    compiled_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    research_id: Optional[str] = None
    disclaimers: List[str] = field(
        default_factory=lambda: [
            "This brief is a neutral, shared context artifact. It must not bias or collapse independent consultant perspectives.",
            "Use for grounding facts only; do not import conclusions from others.",
        ]
    )

    def compact_preview(self, length: int = 240) -> str:
        text = (self.neutral_summary or "").replace("\n", " ").strip()
        return text[:length]

    def to_ordered_dict(self) -> Dict[str, Any]:
        # Deterministic serialization: fixed key order, stable nested ordering
        return {
            "query": self.query,
            "neutral_summary": self.neutral_summary,
            "key_facts": list(self.key_facts),
            "sources": [s.to_ordered_dict() for s in self.sources],
            "tier": self.tier,
            "confidence": float(self.confidence),
            "compiled_at": self.compiled_at,
            "research_id": self.research_id or "",
            "disclaimers": list(self.disclaimers),
        }

    def stable_hash(self) -> str:
        payload = self.to_ordered_dict()
        # Ensure deterministic JSON
        import json

        data = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return md5(data.encode("utf-8")).hexdigest()[:16]

    def to_compact_event(self) -> Dict[str, Any]:
        return {
            "summary_preview": self.compact_preview(),
            "sources_count": len(self.sources),
            "confidence": float(self.confidence),
            "tier": self.tier,
            "research_id": self.research_id or "",
            "hash": self.stable_hash(),
        }
