"""
Zep Cloud Memory Integration for METIS 2.0
==========================================

Provides persistent conversation memory using Zep Cloud API.
Handles conversation history, user memory, and context continuity
for the METIS intelligence platform.

Features:
- Long-term conversation memory
- Context summarization
- Intelligent memory retrieval
- User session management
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import os

import aiohttp
from ..core.unified_context_stream import get_unified_context_stream, ContextEventType, UnifiedContextStream

logger = logging.getLogger(__name__)


class ZepMessage:
    """Represents a message in Zep Cloud"""

    def __init__(
        self,
        content: str,
        role: str = "user",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.content = content
        self.role = role  # 'user', 'assistant', 'system'
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
        self.message_id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to Zep API format"""
        return {
            "content": self.content,
            "role": self.role,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


class ZepMemoryManager:
    """
    Zep Cloud memory manager for persistent conversation storage
    """

    API_BASE = "https://api.getzep.com/v2"

    def __init__(
        self,
        api_key: Optional[str] = None,
        context_stream: Optional[UnifiedContextStream] = None,
    ):
        """
        Initialize Zep memory manager

        Args:
            api_key: Zep Cloud API key (defaults to ZEP_API_KEY env var)
            context_stream: Context stream for logging
        """
        self.api_key = api_key or os.getenv("ZEP_API_KEY")
        if not self.api_key:
            logger.warning(
                "⚠️ No Zep API key provided - memory features will be limited"
            )

        self.context_stream = context_stream or get_unified_context_stream()

        # Active sessions cache
        self._sessions_cache = {}
        self._cache_expiry = 3600  # 1 hour

        # Statistics
        self.stats = {
            "sessions_created": 0,
            "messages_stored": 0,
            "memory_retrievals": 0,
            "total_api_calls": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

        logger.info("🧠 ZepMemoryManager initialized")

    async def create_session(
        self,
        user_id: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a new conversation session

        Args:
            user_id: User identifier
            session_id: Optional session ID (generated if not provided)
            metadata: Session metadata

        Returns:
            Session ID
        """
        if not self.api_key:
            # Return mock session for development
            session_id = session_id or f"mock_session_{user_id}_{uuid.uuid4().hex[:8]}"
            logger.info(f"🔧 Mock session created: {session_id}")
            return session_id

        session_id = session_id or str(uuid.uuid4())

        try:
            start_time = datetime.now()
            payload = {
                "session_id": session_id,
                "user_id": user_id,
                "metadata": metadata or {},
            }

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            async with aiohttp.ClientSession() as client:
                async with client.post(
                    f"{self.API_BASE}/sessions", json=payload, headers=headers
                ) as response:
                    if response.status not in [200, 201]:
                        error_text = await response.text()
                        logger.error(
                            f"❌ Zep session creation failed: {response.status} - {error_text}"
                        )
                        return ""

                    result = await response.json()

            self.stats["sessions_created"] += 1
            self.stats["total_api_calls"] += 1

            # Cache session info
            self._sessions_cache[session_id] = {
                "user_id": user_id,
                "metadata": metadata or {},
                "created_at": datetime.now(),
                "message_count": 0,
            }

            await self.context_stream.add_event(
                ContextEventType.TOOL_EXECUTION,
                {
                    "tool": "zep_memory",
                    "tool_name": "zep_memory",
                    "action": "session_created",
                    "session_id": session_id,
                    "user_id": user_id,
                    "latency_ms": int((datetime.now() - start_time).total_seconds() * 1000),
                },
            )

            logger.info(f"✅ Zep session created: {session_id}")
            return session_id

        except Exception as e:
            logger.error(f"❌ Error creating Zep session: {e}")
            await self.context_stream.add_event(
                ContextEventType.TOOL_EXECUTION,
                {
                    "tool": "zep_memory",
                    "tool_name": "zep_memory",
                    "action": "session_create_error",
                    "user_id": user_id,
                    "error": str(e),
                    "latency_ms": int((datetime.now() - start_time).total_seconds() * 1000) if 'start_time' in locals() else None,
                },
            )
            return ""

    async def add_message(
        self,
        session_id: str,
        content: str,
        role: str = "user",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Add a message to a conversation session

        Args:
            session_id: Session identifier
            content: Message content
            role: Message role ('user', 'assistant', 'system')
            metadata: Message metadata

        Returns:
            Success status
        """
        if not self.api_key:
            # Mock storage for development
            logger.info(
                f"🔧 Mock message stored in session {session_id}: {role} - {content[:100]}..."
            )
            return True

        try:
            start_time = datetime.now()
            message = ZepMessage(content, role, metadata)

            payload = {"messages": [message.to_dict()]}

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            async with aiohttp.ClientSession() as client:
                async with client.post(
                    f"{self.API_BASE}/sessions/{session_id}/memory",
                    json=payload,
                    headers=headers,
                ) as response:
                    if response.status not in [200, 201]:
                        error_text = await response.text()
                        logger.error(
                            f"❌ Zep message storage failed: {response.status} - {error_text}"
                        )
                        return False

            self.stats["messages_stored"] += 1
            self.stats["total_api_calls"] += 1

            # Update cache
            if session_id in self._sessions_cache:
                self._sessions_cache[session_id]["message_count"] += 1

            await self.context_stream.add_event(
                ContextEventType.TOOL_EXECUTION,
                {
                    "tool": "zep_memory",
                    "tool_name": "zep_memory",
                    "action": "message_stored",
                    "session_id": session_id,
                    "role": role,
                    "content_length": len(content),
                    "latency_ms": int((datetime.now() - start_time).total_seconds() * 1000),
                },
            )

            logger.debug(f"✅ Message stored in Zep session {session_id}: {role}")
            return True

        except Exception as e:
            logger.error(f"❌ Error storing message in Zep: {e}")
            await self.context_stream.add_event(
                ContextEventType.TOOL_EXECUTION,
                {
                    "tool": "zep_memory",
                    "tool_name": "zep_memory",
                    "action": "message_store_error",
                    "session_id": session_id,
                    "error": str(e),
                    "latency_ms": int((datetime.now() - start_time).total_seconds() * 1000) if 'start_time' in locals() else None,
                },
            )
            return False

    async def add_conversation_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Add a complete conversation turn (user + assistant messages)

        Args:
            session_id: Session identifier
            user_message: User's message
            assistant_message: Assistant's response
            metadata: Turn metadata

        Returns:
            Success status
        """
        try:
            # Add user message
            user_success = await self.add_message(
                session_id, user_message, "user", metadata
            )

            # Add assistant message
            assistant_success = await self.add_message(
                session_id, assistant_message, "assistant", metadata
            )

            return user_success and assistant_success

        except Exception as e:
            logger.error(f"❌ Error adding conversation turn: {e}")
            return False

    async def get_memory(
        self, session_id: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Retrieve conversation memory for a session

        Args:
            session_id: Session identifier
            limit: Maximum number of messages to retrieve

        Returns:
            List of conversation messages
        """
        if not self.api_key:
            # Mock memory for development
            logger.info(f"🔧 Mock memory retrieval for session {session_id}")
            return [
                {
                    "content": "Mock conversation history",
                    "role": "user",
                    "timestamp": datetime.now().isoformat(),
                }
            ]

        try:
            start_time = datetime.now()
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            params = {"limit": limit}

            async with aiohttp.ClientSession() as client:
                async with client.get(
                    f"{self.API_BASE}/sessions/{session_id}/memory",
                    headers=headers,
                    params=params,
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(
                            f"❌ Zep memory retrieval failed: {response.status} - {error_text}"
                        )
                        return []

                    result = await response.json()

            messages = result.get("messages", [])
            self.stats["memory_retrievals"] += 1
            self.stats["total_api_calls"] += 1

            await self.context_stream.add_event(
                ContextEventType.TOOL_EXECUTION,
                {
                    "tool": "zep_memory",
                    "tool_name": "zep_memory",
                    "action": "memory_retrieved",
                    "session_id": session_id,
                    "message_count": len(messages),
                    "latency_ms": int((datetime.now() - start_time).total_seconds() * 1000),
                },
            )

            logger.debug(
                f"✅ Retrieved {len(messages)} messages from Zep session {session_id}"
            )
            return messages

        except Exception as e:
            logger.error(f"❌ Error retrieving memory from Zep: {e}")
            await self.context_stream.add_event(
                ContextEventType.TOOL_EXECUTION,
                {
                    "tool": "zep_memory",
                    "tool_name": "zep_memory",
                    "action": "memory_retrieval_error",
                    "session_id": session_id,
                    "error": str(e),
                    "latency_ms": int((datetime.now() - start_time).total_seconds() * 1000) if 'start_time' in locals() else None,
                },
            )
            return []

    async def search_memory(
        self, session_id: str, query: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search conversation memory using semantic search

        Args:
            session_id: Session identifier
            query: Search query
            limit: Maximum number of results

        Returns:
            List of matching messages with relevance scores
        """
        if not self.api_key:
            # Mock search for development
            logger.info(f"🔧 Mock memory search for session {session_id}: '{query}'")
            return [
                {
                    "content": f"Mock search result for: {query}",
                    "role": "assistant",
                    "score": 0.85,
                    "timestamp": datetime.now().isoformat(),
                }
            ]

        try:
            payload = {"query": query, "limit": limit}

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            async with aiohttp.ClientSession() as client:
                async with client.post(
                    f"{self.API_BASE}/sessions/{session_id}/search",
                    json=payload,
                    headers=headers,
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(
                            f"❌ Zep memory search failed: {response.status} - {error_text}"
                        )
                        return []

                    result = await response.json()

            results = result.get("results", [])
            self.stats["total_api_calls"] += 1

            await self.context_stream.add_event(
                ContextEventType.TOOL_EXECUTION,
                {
                    "tool": "zep_memory",
                    "action": "memory_searched",
                    "session_id": session_id,
                    "query": query,
                    "result_count": len(results),
                },
            )

            logger.info(
                f"🔍 Memory search completed: '{query}' → {len(results)} results"
            )
            return results

        except Exception as e:
            logger.error(f"❌ Error searching memory in Zep: {e}")
            await self.context_stream.add_event(
                ContextEventType.TOOL_EXECUTION,
                {
                    "tool": "zep_memory",
                    "action": "memory_search_error",
                    "session_id": session_id,
                    "query": query,
                    "error": str(e),
                },
            )
            return []

    def _compute_decay_weight(self, timestamp_iso: str, half_life_days: float = 30.0) -> float:
        """Compute time-decay weight for a timestamp using half-life in days."""
        try:
            from datetime import datetime
            from math import pow
            ts = datetime.fromisoformat(timestamp_iso.replace("Z", "+00:00"))
            age_days = max(0.0, (datetime.now(ts.tzinfo) - ts).total_seconds() / 86400.0)
            return pow(0.5, age_days / max(half_life_days, 0.1)) if age_days > 0 else 1.0
        except Exception:
            return 1.0

    async def summarize_recent_context(
        self,
        session_id: str,
        max_messages: int = 20,
        half_life_days: float = 30.0,
    ) -> str:
        """Build a lightweight decayed summary of recent context.

        This is a fallback summarizer when Zep summary API isn't used.
        """
        messages = await self.get_memory(session_id, limit=max_messages)
        if not messages:
            return ""
        # Weight messages and select top by weight*length
        weighted = []
        for m in messages:
            ts = m.get("timestamp") or datetime.now().isoformat()
            w = self._compute_decay_weight(ts, half_life_days)
            score = w * len(m.get("content", ""))
            weighted.append((score, m))
        weighted.sort(key=lambda x: x[0], reverse=True)
        top = [m["content"] for _, m in weighted[: max(5, int(max_messages/4))]]
        return "\n".join(top)

    async def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get conversation summary for a session

        Args:
            session_id: Session identifier

        Returns:
            Session summary including key topics, entities, etc.
        """
        if not self.api_key:
            # Mock summary for development
            return {
                "summary": f"Mock summary for session {session_id}",
                "topics": ["AI", "METIS", "conversation"],
                "entities": [],
                "message_count": 10,
                "session_duration": "30 minutes",
            }

        try:
            start_time = datetime.now()
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            async with aiohttp.ClientSession() as client:
                async with client.get(
                    f"{self.API_BASE}/sessions/{session_id}/summary", headers=headers
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(
                            f"❌ Zep summary retrieval failed: {response.status} - {error_text}"
                        )
                        return None

                    result = await response.json()

            self.stats["total_api_calls"] += 1

            logger.info(f"📄 Retrieved session summary for {session_id}")
            return result

        except Exception as e:
            logger.error(f"❌ Error retrieving session summary: {e}")
            return None

    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a conversation session and all its data

        Args:
            session_id: Session identifier

        Returns:
            Success status
        """
        if not self.api_key:
            logger.info(f"🔧 Mock session deletion: {session_id}")
            return True

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            async with aiohttp.ClientSession() as client:
                async with client.delete(
                    f"{self.API_BASE}/sessions/{session_id}", headers=headers
                ) as response:
                    success = response.status in [200, 204]

            if success:
                # Remove from cache
                if session_id in self._sessions_cache:
                    del self._sessions_cache[session_id]

            await self.context_stream.add_event(
                ContextEventType.TOOL_EXECUTION,
                {
                    "tool": "zep_memory",
                    "tool_name": "zep_memory",
                    "action": "session_deleted",
                    "session_id": session_id,
                    "latency_ms": int((datetime.now() - start_time).total_seconds() * 1000),
                },
            )

            logger.info(f"🗑️ Session deleted: {session_id}")

            self.stats["total_api_calls"] += 1
            return success

        except Exception as e:
            logger.error(f"❌ Error deleting session: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get memory manager statistics"""
        return {
            **self.stats,
            "active_sessions": len(self._sessions_cache),
            "has_api_key": bool(self.api_key),
            "cache_hit_rate": (
                self.stats["cache_hits"]
                / (self.stats["cache_hits"] + self.stats["cache_misses"])
                if (self.stats["cache_hits"] + self.stats["cache_misses"]) > 0
                else 0.0
            ),
        }

    async def health_check(self) -> Dict[str, Any]:
        """Check Zep Cloud service health"""
        if not self.api_key:
            return {
                "status": "mock_mode",
                "available": True,
                "message": "Running in mock mode - no API key configured",
            }

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            async with aiohttp.ClientSession() as client:
                async with client.get(
                    f"{self.API_BASE}/sessions", headers=headers, params={"limit": 1}
                ) as response:
                    if response.status == 200:
                        return {
                            "status": "healthy",
                            "available": True,
                            "message": "Zep Cloud API accessible",
                        }
                    else:
                        return {
                            "status": "unhealthy",
                            "available": False,
                            "message": f"API returned status {response.status}",
                        }

        except Exception as e:
            return {
                "status": "error",
                "available": False,
                "message": f"Health check failed: {str(e)}",
            }
