#!/usr/bin/env python3
"""
Memory V2: Decay-aware in-memory store and retrieval ranking.
- Document model: id, content, source_quality [0..1], scrape_ts (unix seconds), user_id/session_id.
- Scoring: token_overlap * source_quality * exp(-ln(2) * age_days / halflife_days)
- Tracks hit-rate metrics.
"""

from __future__ import annotations

import os
import time
import math
import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple


def _now() -> float:
    return time.time()


def _norm_words(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", (text or "").lower())


@dataclass
class Document:
    id: str
    content: str
    source_quality: float
    scrape_ts: float
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class MemoryV2:
    def __init__(self):
        self.docs: Dict[str, Document] = {}
        self.inverted: Dict[str, set] = {}
        self.queries = 0
        self.hits = 0
        self.halflife_days = float(os.getenv("MEMORY_DECAY_HALFLIFE_DAYS", "30"))

    def add(self, doc: Document) -> None:
        self.docs[doc.id] = doc
        for w in set(_norm_words(doc.content)):
            self.inverted.setdefault(w, set()).add(doc.id)

    def _age_days(self, doc: Document) -> float:
        return max(0.0, (_now() - float(doc.scrape_ts)) / 86400.0)

    def _decay(self, age_days: float) -> float:
        if self.halflife_days <= 0:
            return 1.0
        return math.exp(-math.log(2) * age_days / self.halflife_days)

    def _score(self, q_words: List[str], doc: Document) -> float:
        d_words = set(_norm_words(doc.content))
        overlap = len(set(q_words) & d_words)
        if overlap == 0:
            return 0.0
        age = self._age_days(doc)
        return overlap * max(0.0, min(1.0, doc.source_quality)) * self._decay(age)

    def query(self, query: str, *, k: int = 5, user_id: Optional[str] = None, session_id: Optional[str] = None) -> List[Tuple[Document, float]]:
        self.queries += 1
        q_words = _norm_words(query)
        # Prefilter candidates via inverted index
        cand_ids = set()
        for w in set(q_words):
            cand_ids |= self.inverted.get(w, set())
        # Scope to user/session if provided
        cands = [self.docs[cid] for cid in cand_ids if cid in self.docs]
        if user_id:
            cands = [d for d in cands if d.user_id == user_id]
        if session_id:
            cands = [d for d in cands if d.session_id == session_id]
        scored = [(d, self._score(q_words, d)) for d in cands]
        scored = [x for x in scored if x[1] > 0.0]
        if scored:
            self.hits += 1
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def stats(self) -> Dict[str, float]:
        return {
            "docs": float(len(self.docs)),
            "queries": float(self.queries),
            "hits": float(self.hits),
            "hit_rate": (self.hits / self.queries) if self.queries else 0.0,
        }


_memory_v2: Optional[MemoryV2] = None


def get_memory_v2() -> MemoryV2:
    global _memory_v2
    if _memory_v2 is None:
        _memory_v2 = MemoryV2()
    return _memory_v2
