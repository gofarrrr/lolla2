"""
Persistent task memory layer
- Profile facts (stable)
- Working set (task-specific)
- History summaries (rolling)

Retrieval policy: similarity >= 0.7 (±), recency boost, top-k 3–8, time decay.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import hashlib

from src.storage.hybrid_storage import HybridStorageManager
from src.core.unified_context_stream import get_unified_context_stream


@dataclass
class MemoryItem:
    kind: str  # profile | working_set | summary
    text: str
    created_at: datetime
    importance: float = 0.5  # 0..1

    def to_doc(self) -> Dict[str, Any]:
        return {
            "content": self.text,
            "title": f"memory:{self.kind}",
            "source_type": "memory",
            "metadata": {
                "kind": self.kind,
                "created_at": self.created_at.isoformat(),
                "importance": self.importance,
            },
        }


class PersistentMemory:
    def __init__(self, storage: Optional[HybridStorageManager] = None):
        self.storage = storage or HybridStorageManager()

    async def initialize(self) -> bool:
        try:
            return await self.storage.initialize()
        except Exception:
            return False

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    async def write_back(
        self,
        *,
        user_id: Optional[str],
        session_id: Optional[str],
        profile_facts: Optional[List[str]] = None,
        working_set_updates: Optional[List[str]] = None,
        summary: Optional[str] = None,
    ) -> None:
        # Store profile facts (structured store)
        if user_id and profile_facts:
            # Persist as preferences map entry (minimal viable)
            from src.storage.supabase_store import SupabaseStore

            store = self.storage.supabase if self.storage else SupabaseStore()
            profile = await store.get_user_profile(user_id)  # type: ignore
            prefs = (profile or {}).get("preferences", {})
            facts = list({*prefs.get("profile_facts", []), *profile_facts})  # dedupe
            prefs["profile_facts"] = facts
            await store.update_user_preferences(user_id, prefs)  # type: ignore

        # Store working set and summary in vector store for retrieval
        docs: List[Dict[str, Any]] = []
        now = datetime.now(timezone.utc)
        for t in (working_set_updates or []):
            docs.append(MemoryItem("working_set", t, now, 0.7).to_doc())
        if summary:
            docs.append(MemoryItem("summary", summary, now, 0.6).to_doc())
        if docs:
            await self.storage.add_knowledge_batch(docs)

        # Log context event
        try:
            cs = get_unified_context_stream()
            cs.add_event(
                cs.ContextEventType.TOOL_EXECUTION,  # type: ignore[attr-defined]
                {
                    "tool": "persistent_memory",
                    "action": "write_back",
                    "user_id": user_id,
                    "session_id": session_id,
                    "profile_facts": len(profile_facts or []),
                    "working_set": len(working_set_updates or []),
                    "summary_present": bool(summary),
                },
            )
        except Exception:
            pass

    async def retrieve_for_context(
        self,
        *,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        kinds: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        # Semantic retrieval from vector store
        results = await self.storage.search_knowledge_base(
            query=query, limit=min(8, max(3, top_k)), similarity_threshold=similarity_threshold
        )
        # Optional filter by kind
        if kinds:
            results = [r for r in results if r.get("metadata", {}).get("kind") in kinds]
        return results


# Singleton
_memory: Optional[PersistentMemory] = None


def get_persistent_memory() -> PersistentMemory:
    global _memory
    if _memory is None:
        _memory = PersistentMemory()
    return _memory
