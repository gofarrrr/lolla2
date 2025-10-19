"""Depth Enrichment Service

Orchestrates Q&A precision retrieval and builds enriched prompt sections.

Intended checkpoint position: after the consultant's initial draft (first 400–600 words)
and before critique. Injects ≤2 Q&A pairs to minimize latency and token overhead.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .qa_precision_retriever import QAPrecisionRetriever
from src.core.unified_context_stream import UnifiedContextStream, ContextEventType

logger = logging.getLogger(__name__)


class DepthEnrichmentService:
    def __init__(self, retriever: Optional[QAPrecisionRetriever] = None, max_pairs: int = 2, context_stream: Optional[UnifiedContextStream] = None) -> None:
        self.retriever = retriever or QAPrecisionRetriever()
        self.max_pairs = max_pairs
        self.context_stream = context_stream

    async def enrich_consultant_draft(
        self,
        consultant_draft: str,
        consultant_affinities: List[str],
        trace_id: str,
    ) -> Dict[str, Any]:
        """Return enrichment payload with selected Q&A pairs and formatted text.

        Output keys:
          - pairs: List[dict]
          - text_block: str (formatted for injection)
        """
        try:
            pairs = await self.retriever.analyze_qa_needs(
                consultant_draft, consultant_affinities, trace_id
            )
            pairs = pairs[: self.max_pairs]
        except Exception as e:  # pragma: no cover - graceful fallback
            logger.error(f"Depth enrichment failed to analyze needs: {e}")
            pairs = []

        text_block = ""
        if pairs:
            blocks: List[str] = ["DEEP KNOWLEDGE RETRIEVED:\n"]
            for p in pairs:
                q = p.get("question") or ""
                a = p.get("answer") or ""
                mm = p.get("mental_model_name") or ""
                blocks.append(f"[{mm}]\n{q}\n{a}\n")
            text_block = "\n".join(blocks).strip() + "\n"

        # Emit optional telemetry
        try:
            if self.context_stream is not None:
                self.context_stream.add_event(
                    ContextEventType.DEPTH_ENRICHMENT_APPLIED,
                    {
                        "stage": "post_consultant",
                        "trace_id": trace_id,
                        "affinities_count": len(consultant_affinities or []),
                        "pairs_selected": len(pairs),
                        "models": [p.get("mental_model_name") for p in pairs],
                    },
                )
        except Exception:
            pass

        return {"pairs": pairs, "text_block": text_block}
