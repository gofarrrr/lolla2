#!/usr/bin/env python3
"""
Simple retrieval evaluation harness for Memory V2.
Computes recall@k over a small golden set.
"""

from __future__ import annotations

from typing import List, Dict, Tuple

from src.engine.memory.memory_v2 import get_memory_v2, Document
from src.engine.retrieval.decay_retriever import retrieve


def recall_at_k(queries: List[Tuple[str, str]], *, k: int = 5) -> float:
    """
    queries: list of (query_text, expected_doc_id)
    """
    if not queries:
        return 0.0
    hits = 0
    for q, expected in queries:
        results = retrieve(q, k=k)
        ids = [r["id"] for r in results]
        if expected in ids:
            hits += 1
    return hits / len(queries)
