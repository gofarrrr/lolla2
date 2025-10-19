"""Agent Guidance Retriever

Lightweight service that fetches NWAY2-derived guidance for agent roles.
Deterministic; no LLM calls. Uses the `agent_guidance` table populated by the
NWAY2 extractor script.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from src.services.persistence.database_service import (
    DatabaseService,
    DatabaseOperationError,
    DatabaseServiceConfig,
)
from src.core.unified_context_stream import get_unified_context_stream, ContextEventType

logger = logging.getLogger(__name__)


class AgentGuidanceRetriever:
    """Fetch agent-specific performance guidance."""

    def __init__(
        self,
        db: Optional[DatabaseService] = None,
        *,
        context_stream: Optional[Any] = None,
    ) -> None:
        self.db = db or DatabaseService(DatabaseServiceConfig.from_env())
        self.context_stream = context_stream or get_unified_context_stream()

    def get_guidance(
        self,
        agent_role: str,
        *,
        guidance_type: Optional[str] = None,
        max_words: int = 300,
    ) -> Dict[str, Optional[str]]:
        filters = {"agent_role": agent_role}
        if guidance_type:
            filters["guidance_type"] = guidance_type

        try:
            rows = self.db.fetch_many(
                "agent_guidance",
                filters,
                limit=5,
                order_by="question_num",
            )
        except DatabaseOperationError as exc:
            logger.error(f"Agent guidance DB error: {exc}")
            return {"applicable": False, "reason": "database_error"}

        if not rows:
            self._emit_event(
                "no_guidance",
                agent_role,
                guidance_type,
                details=None,
            )
            return {"applicable": False, "reason": "not_found"}

        row = rows[0]
        answer = row.get("answer", "")
        words = answer.split()
        if len(words) > max_words:
            answer = " ".join(words[:max_words]) + "..."

        result = {
            "applicable": True,
            "guidance": answer,
            "question": row.get("question"),
            "guidance_type": row.get("guidance_type"),
            "source": row.get("source_file"),
            "word_count": row.get("word_count"),
        }

        self._emit_event("guidance_found", agent_role, guidance_type, details=result)
        return result

    def _emit_event(
        self,
        action: str,
        agent_role: str,
        guidance_type: Optional[str],
        details: Optional[Dict[str, Optional[str]]],
    ) -> None:
        try:
            self.context_stream.add_event(
                ContextEventType.TOOL_EXECUTION,
                {
                    "tool": "agent_guidance",
                    "action": action,
                    "agent_role": agent_role,
                    "guidance_type": guidance_type,
                    "details": details,
                },
            )
        except Exception as exc:  # pragma: no cover - telemetry best effort
            logger.debug(f"Failed to emit agent guidance event: {exc}")
