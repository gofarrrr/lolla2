# src/core/persistence/adapters.py
import os
import json
import logging
from typing import List, Dict, Any, Optional

from .contracts import IEventPersistence
from src.services.persistence.database_service import DatabaseService

logger = logging.getLogger(__name__)


class SupabaseAdapter(IEventPersistence):
    """Supabase-backed persistence adapter.
    Expects each batch item to be a full record ready for insertion into the 'context_streams' table.
    """

    def __init__(self, table_name: str = "context_streams", database_service: DatabaseService | None = None) -> None:
        self.table_name = table_name
        self._database_service = database_service

    async def persist(self, batch: List[Dict[str, Any]]) -> None:
        if not self._database_service:
            raise RuntimeError("DatabaseService is not available for SupabaseAdapter")
        try:
            await self._database_service.insert_many_async(self.table_name, batch)
            logger.info(f"✅ SupabaseAdapter: persisted {len(batch)} record(s)")
        except Exception as e:
            logger.error(f"❌ SupabaseAdapter persist failed: {e}")
            raise


class FileAdapter(IEventPersistence):
    """File-based adapter that appends each record as a JSON line to a local file."""

    def __init__(self, file_path: Optional[str] = None) -> None:
        self.file_path = file_path or os.getenv(
            "CONTEXT_STREAM_JSONL", "local_context_stream.jsonl"
        )

    async def persist(self, batch: List[Dict[str, Any]]) -> None:
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.file_path) or ".", exist_ok=True)
        with open(self.file_path, "a", encoding="utf-8") as f:
            for item in batch:
                f.write(json.dumps(item, default=str) + "\n")
        logger.info(
            f"📝 FileAdapter: appended {len(batch)} record(s) to {self.file_path}"
        )
