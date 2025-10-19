"""
Data Fetching Stage
===================

Fetches all required data from external sources (database and UnifiedContextStream).

Responsibility:
- Fetch cognitive states from state_checkpoints table
- Fetch query text from engagements table
- Fetch events from UnifiedContextStream

Complexity: CC<5 (Simple data fetching)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.services.persistence.database_service import DatabaseService
from src.core.unified_context_stream import get_unified_context_stream
from src.services.report_reconstruction.reconstruction_state import ReconstructionState
from src.services.report_reconstruction.reconstruction_stage import (
    ReconstructionStage,
    ReconstructionError,
)

logger = logging.getLogger(__name__)


class DataFetchingStage(ReconstructionStage):
    """
    Stage 1: Data Fetching

    Fetches all required data from external sources to populate the reconstruction state.
    """

    def __init__(self, db: Optional[DatabaseService] = None):
        """
        Initialize data fetching stage.

        Args:
            db: Database service for fetching cognitive states and engagement data
        """
        self.db = db

    @property
    def name(self) -> str:
        return "data_fetching"

    @property
    def description(self) -> str:
        return "Fetch cognitive states, query text, and events from data sources"

    def process(self, state: ReconstructionState) -> ReconstructionState:
        """
        Fetch all required data from external sources.

        Args:
            state: Current reconstruction state

        Returns:
            Updated state with cognitive_states, query_text, and events populated
        """
        try:
            # Fetch cognitive states
            cognitive_states = self._fetch_cognitive_states(state.trace_id)

            # Fetch query text
            query_text = self._fetch_query_text(state.trace_id)

            # Fetch events
            events = self._fetch_events()

            # Return updated state
            return state.with_data(
                cognitive_states=cognitive_states,
                query_text=query_text,
                events=events,
            )

        except Exception as e:
            raise ReconstructionError(
                self.name,
                f"Failed to fetch data for trace_id={state.trace_id}",
                cause=e,
            )

    def _fetch_cognitive_states(self, trace_id: str) -> List[Dict[str, Any]]:
        """
        Fetch cognitive outputs from state_checkpoints (V6 migration).
        Maps V6 checkpoint structure to V5 cognitive_states format for compatibility.

        Args:
            trace_id: Trace identifier

        Returns:
            List of cognitive state outputs
        """
        if not self.db:
            return []

        try:
            # V6: Query state_checkpoints instead of cognitive_states
            rows = self.db.fetch_many(
                "state_checkpoints",
                {"trace_id": trace_id},
                columns="*",
                order_by="created_at",
                desc=False,
            )

            if not rows:
                return []

            # V6: Map checkpoint structure to cognitive_states format
            mapped = []
            for checkpoint in rows:
                state_data = checkpoint.get("state_data", {})
                stage_output = state_data.get("stage_output", {})
                mapped.append(
                    {
                        "trace_id": trace_id,
                        "stage_name": checkpoint.get("stage_name"),
                        "cognitive_output": stage_output,
                        "created_at": checkpoint.get("created_at"),
                        "processing_time_ms": state_data.get("_stage_metadata", {}).get(
                            "processing_time_ms", 0
                        ),
                    }
                )

            return mapped

        except Exception as e:
            logger.warning(f"⚠️ Failed to fetch checkpoints for {trace_id}: {e}")
            return []

    def _fetch_query_text(self, trace_id: str) -> Optional[str]:
        """
        Fetch query text from engagements table.

        Args:
            trace_id: Trace identifier

        Returns:
            Query text or None if not found
        """
        if not self.db:
            return None

        try:
            engagement = self.db.fetch_one("engagements", {"trace_id": trace_id})
            if engagement:
                # V6: Try 'user_query' first, fallback to 'query'
                return engagement.get("user_query") or engagement.get("query")
        except Exception as e:
            logger.warning(f"⚠️ Failed to fetch query for {trace_id}: {e}")

        return None

    def _fetch_events(self) -> List[Any]:
        """
        Fetch events from UnifiedContextStream.

        Returns:
            List of context events
        """
        try:
            stream = get_unified_context_stream()
            return getattr(stream, "events", [])
        except Exception as e:
            logger.warning(f"⚠️ Failed to fetch events from stream: {e}")
            return []
