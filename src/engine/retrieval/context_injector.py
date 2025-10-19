#!/usr/bin/env python3
"""
Context injector for decay-based RAG using Memory V2.
"""

from __future__ import annotations

from typing import List, Dict

from src.engine.retrieval.decay_retriever import retrieve


def build_context_system_message(query: str, *, k: int = 3) -> str | None:
    results = retrieve(query, k=k)
    if not results:
        return None
    parts = []
    for r in results:
        parts.append(f"- ({r['id']}) {r['content'][:400]}")
    context = "\n".join(parts)
    return (
        "Relevant context (decayed retrieval):\n" + context + "\n\nUse these as evidence; cite doc IDs where relevant."
    )
