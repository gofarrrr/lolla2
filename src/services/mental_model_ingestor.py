"""Mental Model Ingestor
========================

Utilities to ingest local mental model corpora (MM1, MM2, NWAY, NWAY2)
into the RAG store with consistent metadata for better retrieval.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.rag.retriever import IntelligentRetriever
from src.rag.embeddings import VoyageEmbeddings
from src.core.unified_context_stream import get_unified_context_stream, UnifiedContextStream

logger = logging.getLogger(__name__)


@dataclass
class IngestOptions:
    content_type: str = "mental_model"
    max_chars: int = 50000


class MentalModelIngestor:
    def __init__(
        self,
        retriever: Optional[IntelligentRetriever] = None,
        context_stream: Optional[UnifiedContextStream] = None,
        options: Optional[IngestOptions] = None,
    ) -> None:
        self.context_stream = context_stream or get_unified_context_stream()
        self.options = options or IngestOptions()

        if retriever is None:
            embeddings = VoyageEmbeddings(context_stream=self.context_stream)
            retriever = IntelligentRetriever(embeddings=embeddings, context_stream=self.context_stream)
        self.retriever = retriever

        self._initialized = False

    async def initialize(self) -> None:
        if not self._initialized:
            await self.retriever.initialize()
            self._initialized = True

    async def ingest_dirs(self, dirs: List[str]) -> int:
        """Ingest all markdown files from provided directories.

        Returns number of documents ingested.
        """
        await self.initialize()

        docs: List[Dict[str, Any]] = []
        for d in dirs:
            base = Path(d)
            if not base.exists():
                logger.warning("Skipping missing directory: %s", d)
                continue
            for path in base.rglob("*.md"):
                try:
                    content = path.read_text(encoding="utf-8", errors="ignore")[: self.options.max_chars]
                    meta = self._build_metadata_for_path(path, content)
                    doc = {
                        "content": content,
                        "source": str(path),
                        "metadata": meta,
                    }
                    docs.append(doc)
                except Exception as exc:  # pylint: disable=broad-except
                    logger.error("Failed to read %s: %s", path, exc)

        if not docs:
            return 0

        # Store batched for efficiency
        ids = await self.retriever.batch_store_documents(docs)
        logger.info("✅ Ingested %d mental model documents", len(ids))
        return len(ids)

    def _build_metadata_for_path(self, path: Path, content: str) -> Dict[str, Any]:
        """Derive consistent metadata from file path and content."""
        name = path.name
        parent = path.parent.name
        group = parent  # e.g., MM1, MM2, NWAY, NWAY2

        # Try to get a human title: first markdown H1 or first non-empty line
        title = None
        for line in content.splitlines():
            if line.strip().startswith("# "):
                title = line.strip().lstrip("# ").strip()
                break
            if not title and line.strip():
                title = line.strip()
        if not title:
            title = self._filename_to_title(name)

        model_name = self._filename_to_title(name)

        # Infer NWAY category from filename pattern like "NWAY_DECISION_001_ ... .md"
        category = None
        if group.upper().startswith("NWAY"):
            # Look at filename prefix between 'NWAY_' and next '_' if present
            lowered = name.replace(" ", "_")
            parts = lowered.split("_")
            # Find token after 'NWAY'
            try:
                idx = parts.index("NWAY")
                category = parts[idx + 1].upper() if idx + 1 < len(parts) else None
            except ValueError:
                # Might already be like NWAY_DECISION_...
                if parts and parts[0].upper().startswith("NWAY") and len(parts) > 1:
                    category = parts[1].upper()

        # Attempt to infer category for MM1/MM2 from the filename prefix
        if not category and group.upper().startswith("MM"):
            stem = Path(name).stem
            base = stem.split("_rag")[0]
            tokens = base.split("_")
            if tokens:
                category = tokens[0].upper()

        meta: Dict[str, Any] = {
            "title": title[:512],
            "model_name": model_name,
            "nway_group": group,
            "category": category or group,
            "content_type": self.options.content_type,
            "provider": "local_mm_ingest",
        }
        return meta

    def _filename_to_title(self, filename: str) -> str:
        base = Path(filename).stem
        base = base.replace("_rag", "").replace("_", " ").strip()
        return " ".join(base.split())


# Convenience CLI entry point
async def ingest_default_corpora() -> int:
    base = Path(os.getcwd())
    dirs = [
        base / "migrations" / "MM1",
        base / "migrations" / "MM2",
        base / "migrations" / "NWAY",
        base / "migrations" / "NWAY2",
    ]
    ingestor = MentalModelIngestor()
    return await ingestor.ingest_dirs([str(d) for d in dirs])
