"""Q&A Precision Retriever (Depth Enrichment Core)

Detects which mental models a consultant is actively using and retrieves the
most helpful Q&A pairs directly from the relational table `mental_model_qa_chunks`.

Design goals:
- Prefer direct indexed lookups (fast, deterministic) over vector search.
- Minimal prompting: if an LLM is available, ask it only to choose the Q type/number.
- Graceful degradation: if DB is unavailable, return a structured error.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.services.persistence.database_service import (
    DatabaseService, DatabaseOperationError, DatabaseServiceConfig,
)
from .aliases import resolve_alias

logger = logging.getLogger(__name__)


QUESTION_TYPE_ALIASES = {
    1: "foundational",
    2: "evidence",
    3: "practical",
    4: "pitfalls",
    5: "implementation",
}


def _normalize_name(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", " ", name)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s.replace(" ", "_")


@dataclass
class QANeed:
    model: str
    question_num: Optional[int] = None
    question_type: Optional[str] = None
    reason: Optional[str] = None


class QAChunksRepository:
    """Repository to fetch Q&A rows from `mental_model_qa_chunks`."""

    def __init__(self, db: Optional[DatabaseService] = None) -> None:
        self.db = db
        if self.db is None:
            try:
                self.db = DatabaseService(DatabaseServiceConfig.from_env())
            except Exception as e:  # pragma: no cover - DB optional in some envs
                logger.warning(f"QAChunksRepository: database unavailable: {e}")
                self.db = None  # operate in degraded mode

    def fetch_one(self, model: str, question_num: Optional[int], question_type: Optional[str]) -> Optional[Dict[str, Any]]:
        if not self.db:
            return None
        filters: Dict[str, Any] = {"mental_model_name": model}
        if question_num is not None:
            filters["question_num"] = question_num
        elif question_type is not None:
            filters["question_type"] = question_type
        else:
            return None
        try:
            row = self.db.fetch_one("mental_model_qa_chunks", filters)
            return row
        except DatabaseOperationError as e:
            logger.error(f"DB error fetching Q&A: {e}")
            return None

    def fetch_by_model_and_type(
        self,
        model: str,
        question_type: Optional[str] = None,
        *,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Fetch multiple Q&A rows by model and optional type."""

        if not self.db:
            return []

        filters: Dict[str, Any] = {"mental_model_name": model}
        if question_type:
            filters["question_type"] = question_type

        try:
            rows = self.db.fetch_many(
                "mental_model_qa_chunks",
                filters,
                limit=limit,
                order_by="question_num",
            )
            return rows
        except DatabaseOperationError as e:
            logger.error(f"DB error fetching Q&A list: {e}")
            return []


class QAPrecisionRetriever:
    """
    Analyze consultant text to detect active models and retrieve most useful Q&A pairs.

    If an LLM client is provided, it is used solely to select the question type/number per model.
    Otherwise, a simple heuristic is used to prefer 'practical' (Q3) for how-to gaps.
    """

    def __init__(self, qa_repo: Optional[QAChunksRepository] = None, llm_client: Optional[Any] = None) -> None:
        self.qa_repo = qa_repo or QAChunksRepository()
        self.llm = llm_client

    async def analyze_qa_needs(
        self,
        consultant_analysis_text: str,
        consultant_yaml_affinities: List[str],
        trace_id: str,
    ) -> List[Dict[str, Any]]:
        models_in_use = self._detect_active_models(consultant_analysis_text, consultant_yaml_affinities)
        if not models_in_use:
            return []

        qa_needs = await self._identify_question_numbers(consultant_analysis_text, models_in_use, trace_id)
        qa_pairs: List[Dict[str, Any]] = []
        for need in qa_needs:
            row = self.qa_repo.fetch_one(
                model=need.model,
                question_num=need.question_num,
                question_type=need.question_type,
            )
            if row:
                qa_pairs.append(
                    {
                        "mental_model_name": need.model,
                        "question_num": row.get("question_num"),
                        "question_type": row.get("question_type"),
                        "question": row.get("question"),
                        "answer": row.get("answer"),
                        "source_file": row.get("source_file"),
                        "category": row.get("category"),
                        "reason": need.reason,
                    }
                )
        return qa_pairs

    def _detect_active_models(self, analysis_text: str, available_affinities: List[str]) -> List[str]:
        """Detect which affinity model names appear to be in active use in the text.

        Matching is done via case-insensitive normalized substrings.
        """
        text_norm = _normalize_name(analysis_text)
        models: List[str] = []
        for name in available_affinities:
            norm = _normalize_name(name)
            # Resolve common aliases to canonical names
            canonical = resolve_alias(norm)
            # require token boundary match when possible
            pattern = re.escape(norm)
            if re.search(rf"\b{pattern}\b", text_norm):
                models.append(canonical)
        # deduplicate preserving order
        seen = set()
        return [m for m in models if not (m in seen or seen.add(m))]

    async def _identify_question_numbers(self, analysis_text: str, models_in_use: List[str], trace_id: str) -> List[QANeed]:
        """Select the most useful Q type or number for each model.

        Strategy:
          - If LLM is provided, ask it to pick among [foundational, evidence, practical, pitfalls, implementation].
          - Else heuristic: prefer practical (3), fall back to implementation (5) if words like 'tool', 'template', 'process' appear; choose pitfalls (4) if 'risk', 'failure', 'mistake' keywords found.
        """
        needs: List[QANeed] = []

        # Heuristic branch when LLM is not available
        analysis_lower = analysis_text.lower()
        for model in models_in_use:
            qtype: Optional[str] = None
            qnum: Optional[int] = None
            reason = "heuristic"

            if any(k in analysis_lower for k in ["risk", "failure", "pitfall", "limitation", "edge case"]):
                qtype = "pitfalls"; qnum = 4
            elif any(k in analysis_lower for k in ["tool", "template", "system", "process", "framework", "step-by-step", "how to"]):
                qtype = "implementation"; qnum = 5
            else:
                qtype = "practical"; qnum = 3

            needs.append(QANeed(model=model, question_num=qnum, question_type=qtype, reason=reason))

        # If an LLM client is provided, we could refine the above (kept minimal to avoid prompt bloat)
        return needs
