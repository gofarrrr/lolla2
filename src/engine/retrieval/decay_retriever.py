#!/usr/bin/env python3
"""
Decay-based retriever using Memory V2.
Flag: RAG_DECAY_RETRIEVAL to enable in pipelines that import this.
"""

from __future__ import annotations

import os
from typing import List, Dict, Any

from src.engine.memory.memory_v2 import get_memory_v2, Document


def retrieve(query: str, *, k: int = 5, user_id: str | None = None, session_id: str | None = None) -> List[Dict[str, Any]]:
    mem = get_memory_v2()
    results = mem.query(query, k=k, user_id=user_id, session_id=session_id)
    return [
        {
            "id": d.id,
            "content": d.content,
            "score": score,
            "source_quality": d.source_quality,
            "scrape_ts": d.scrape_ts,
            "user_id": d.user_id,
            "session_id": d.session_id,
        }
        for d, score in results
    ]
