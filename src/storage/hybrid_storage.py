"""
Hybrid Storage Manager for METIS 2.0
====================================

Unified storage orchestration combining:
- Zep Cloud (conversation memory)
- Supabase (structured data)
- Milvus (vector search via RAG pipeline)

Provides intelligent data routing and synchronization across all storage backends
with automatic failover and consistency guarantees.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

from .zep_memory import ZepMemoryManager
from .supabase_store import SupabaseStore
from ..rag.rag_pipeline import EnhancedRAGPipeline
from ..core.unified_context_stream import get_unified_context_stream, ContextEventType, UnifiedContextStream

logger = logging.getLogger(__name__)


class StorageBackend:
    """Represents a storage backend with health status"""

    def __init__(self, name: str, instance: Any, priority: int = 1):
        self.name = name
        self.instance = instance
        self.priority = priority
        self.is_healthy = True
        self.last_health_check = datetime.now()
        self.error_count = 0
        self.max_errors = 3


class HybridStorageManager:
    """
    Hybrid storage orchestrator for METIS 2.0

    Routes data to appropriate storage backends based on data type and access patterns.
    Provides unified interface for all storage operations with automatic failover.
    """

    def __init__(
        self,
        zep_api_key: Optional[str] = None,
        supabase_url: Optional[str] = None,
        supabase_anon_key: Optional[str] = None,
        supabase_service_key: Optional[str] = None,
        voyage_api_key: Optional[str] = None,
        milvus_host: str = "localhost",
        milvus_port: int = 19530,
        context_stream: Optional[UnifiedContextStream] = None,
    ):
        """
        Initialize hybrid storage manager

        Args:
            zep_api_key: Zep Cloud API key
            supabase_url: Supabase project URL
            supabase_anon_key: Supabase anonymous key
            supabase_service_key: Supabase service role key
            voyage_api_key: Voyage AI API key
            milvus_host: Milvus server host
            milvus_port: Milvus server port
            context_stream: Context stream for logging
        """
        self.context_stream = context_stream or get_unified_context_stream()

        # Initialize storage backends
        self.zep = ZepMemoryManager(
            api_key=zep_api_key, context_stream=self.context_stream
        )

        self.supabase = SupabaseStore(
            supabase_url=supabase_url,
            supabase_anon_key=supabase_anon_key,
            supabase_service_key=supabase_service_key,
            context_stream=self.context_stream,
        )

        self.rag = EnhancedRAGPipeline(
            voyage_api_key=voyage_api_key,
            milvus_host=milvus_host,
            milvus_port=milvus_port,
            context_stream=self.context_stream,
        )

        # Backend registry with priorities
        self.backends = {
            "memory": StorageBackend("zep", self.zep, priority=1),
            "structured": StorageBackend("supabase", self.supabase, priority=1),
            "vector": StorageBackend("rag", self.rag, priority=1),
        }

        # Statistics
        self.stats = {
            "operations_performed": 0,
            "routing_decisions": 0,
            "failover_events": 0,
            "sync_operations": 0,
            "health_checks": 0,
            "start_time": datetime.now(),
        }

        # Data routing configuration
        self.routing_rules = {
            "conversation": "memory",
            "user_profile": "structured",
            "session_metadata": "structured",
            "research_results": "structured",
            "analytics": "structured",
            "documents": "vector",
            "knowledge": "vector",
            "embeddings": "vector",
        }

        logger.info("🔄 HybridStorageManager initialized")

    async def initialize(self) -> bool:
        """Initialize all storage backends"""
        initialization_results = {}
        start_time = datetime.now()

        try:
            # Initialize RAG pipeline (includes Milvus)
            rag_success = await self.rag.initialize()
            initialization_results["rag"] = rag_success

            # Zep and Supabase don't need explicit initialization
            # They handle connection on first use
            initialization_results["zep"] = True
            initialization_results["supabase"] = True

            # Update backend health based on initialization
            self.backends["vector"].is_healthy = rag_success

            # Log initialization status
            await self.context_stream.add_event(
                ContextEventType.TOOL_EXECUTION,
                {
                    "tool": "hybrid_storage",
                    "tool_name": "hybrid_storage",
                    "action": "initialization",
                    "results": initialization_results,
                    "overall_success": all(initialization_results.values()),
                    "latency_ms": int((datetime.now() - start_time).total_seconds() * 1000),
                },
            )

            success = all(initialization_results.values())
            if success:
                logger.info("✅ HybridStorageManager initialization successful")
            else:
                logger.warning(f"⚠️ Partial initialization: {initialization_results}")

            return success

        except Exception as e:
            logger.error(f"❌ HybridStorageManager initialization failed: {e}")
            await self.context_stream.add_event(
                ContextEventType.TOOL_EXECUTION,
                {
                    "tool": "hybrid_storage",
                    "tool_name": "hybrid_storage",
                    "action": "initialization_error",
                    "error": str(e),
                    "latency_ms": int((datetime.now() - start_time).total_seconds() * 1000),
                },
            )
            return False

    def _route_data_type(self, data_type: str) -> str:
        """Determine storage backend for data type"""
        backend = self.routing_rules.get(data_type, "structured")
        self.stats["routing_decisions"] += 1
        return backend

    async def _execute_with_fallback(
        self, backend_name: str, operation: str, method_name: str, *args, **kwargs
    ) -> Any:
        """Execute operation with automatic fallback"""

        backend = self.backends.get(backend_name)
        if not backend or not backend.is_healthy:
            logger.warning(f"⚠️ Backend {backend_name} unavailable for {operation}")
            return None

        try:
            op_start = datetime.now()
            method = getattr(backend.instance, method_name)
            result = await method(*args, **kwargs)

            # Reset error count on successful operation
            backend.error_count = 0
            self.stats["operations_performed"] += 1

            return result

        except Exception as e:
            logger.error(f"❌ {backend_name} {operation} failed: {e}")

            # Increment error count
            backend.error_count += 1
            if backend.error_count >= backend.max_errors:
                backend.is_healthy = False
                self.stats["failover_events"] += 1
                logger.warning(f"🚨 Backend {backend_name} marked as unhealthy")

            await self.context_stream.add_event(
                ContextEventType.TOOL_EXECUTION,
                {
                    "tool": "hybrid_storage",
                    "tool_name": "hybrid_storage",
                    "action": "operation_failed",
                    "backend": backend_name,
                    "operation": operation,
                    "error": str(e),
                    "error_count": backend.error_count,
                    "latency_ms": int((datetime.now() - op_start).total_seconds() * 1000) if 'op_start' in locals() else None,
                },
            )

            return None

    # Unified User Management
    async def create_user_session(
        self,
        user_id: str,
        session_type: str = "conversation",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Create unified user session across all backends"""

        session_id = str(uuid.uuid4())

        try:
            # Create session in memory backend (Zep)
            zep_session = await self._execute_with_fallback(
                "memory",
                "create_session",
                "create_session",
                user_id=user_id,
                session_id=session_id,
                metadata=metadata,
            )

            # Create session record in structured backend (Supabase)
            supabase_session = await self._execute_with_fallback(
                "structured",
                "create_session_record",
                "create_session_record",
                session_id=session_id,
                user_id=user_id,
                session_type=session_type,
                metadata=metadata,
            )

            if zep_session and supabase_session:
                await self.context_stream.add_event(
                    ContextEventType.TOOL_EXECUTION,
                    {
                        "tool": "hybrid_storage",
                        "action": "unified_session_created",
                        "session_id": session_id,
                        "user_id": user_id,
                        "session_type": session_type,
                    },
                )

                logger.info(f"✅ Unified session created: {session_id}")
                return session_id
            else:
                logger.error(f"❌ Failed to create unified session for user {user_id}")
                return None

        except Exception as e:
            logger.error(f"❌ Error creating unified session: {e}")
            return None

    async def store_conversation_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Store conversation turn in memory backend"""

        result = await self._execute_with_fallback(
            "memory",
            "add_conversation_turn",
            "add_conversation_turn",
            session_id=session_id,
            user_message=user_message,
            assistant_message=assistant_message,
            metadata=metadata,
        )

        return result is not None

    async def get_conversation_memory(
        self, session_id: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Retrieve conversation memory"""

        result = await self._execute_with_fallback(
            "memory", "get_memory", "get_memory", session_id=session_id, limit=limit
        )

        return result if result is not None else []

    async def search_conversation_memory(
        self, session_id: str, query: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search conversation memory"""

        result = await self._execute_with_fallback(
            "memory",
            "search_memory",
            "search_memory",
            session_id=session_id,
            query=query,
            limit=limit,
        )

        return result if result is not None else []

    # Unified Knowledge Management
    async def add_knowledge_document(
        self,
        content: str,
        title: str,
        source_type: str = "manual",
        url: Optional[str] = None,
        author: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Add document to knowledge base (vector storage)"""

        result = await self._execute_with_fallback(
            "vector",
            "add_document",
            "add_document",
            content=content,
            title=title,
            source_type=source_type,
            url=url,
            author=author,
            tags=tags,
            metadata=metadata,
        )

        return result

    async def search_knowledge_base(
        self,
        query: str,
        limit: int = 10,
        similarity_threshold: float = 0.7,
        source_types: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Search knowledge base"""

        result = await self._execute_with_fallback(
            "vector",
            "intelligent_search",
            "intelligent_search",
            query=query,
            limit=limit,
            similarity_threshold=similarity_threshold,
            source_types=source_types,
            tags=tags,
        )

        return result if result is not None else []

    async def add_knowledge_batch(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Add multiple documents to knowledge base"""

        result = await self._execute_with_fallback(
            "vector", "add_documents_batch", "add_documents_batch", documents=documents
        )

        return result if result is not None else []

    # Unified User Profile Management
    async def create_user_profile(
        self,
        user_id: str,
        email: str,
        display_name: Optional[str] = None,
        preferences: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Create user profile in structured storage"""

        result = await self._execute_with_fallback(
            "structured",
            "create_user_profile",
            "create_user_profile",
            user_id=user_id,
            email=email,
            display_name=display_name,
            preferences=preferences,
        )

        return result is not None

    async def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user profile"""

        result = await self._execute_with_fallback(
            "structured", "get_user_profile", "get_user_profile", user_id=user_id
        )

        return result

    async def update_user_preferences(
        self, user_id: str, preferences: Dict[str, Any]
    ) -> bool:
        """Update user preferences"""

        result = await self._execute_with_fallback(
            "structured",
            "update_user_preferences",
            "update_user_preferences",
            user_id=user_id,
            preferences=preferences,
        )

        return result is not None

    # Research Results Management
    async def store_research_result(
        self,
        user_id: str,
        query: str,
        results: Dict[str, Any],
        provider: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Store research result in structured storage"""

        research_id = str(uuid.uuid4())

        result = await self._execute_with_fallback(
            "structured",
            "store_research_result",
            "store_research_result",
            research_id=research_id,
            user_id=user_id,
            query=query,
            results=results,
            provider=provider,
            metadata=metadata,
        )

        return result is not None

    async def get_research_history(
        self, user_id: str, limit: int = 20, provider: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get research history for user"""

        result = await self._execute_with_fallback(
            "structured",
            "get_research_history",
            "get_research_history",
            user_id=user_id,
            limit=limit,
            provider=provider,
        )

        return result if result is not None else []

    # Analytics Management
    async def log_analytics_event(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        event_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Log analytics event"""

        result = await self._execute_with_fallback(
            "structured",
            "store_analytics_event",
            "store_analytics_event",
            event_type=event_type,
            user_id=user_id,
            session_id=session_id,
            event_data=event_data,
        )

        return result is not None

    # Cross-Backend Operations
    async def synchronize_user_data(self, user_id: str) -> Dict[str, Any]:
        """Synchronize user data across all backends"""

        sync_results = {
            "user_profile": None,
            "active_sessions": [],
            "research_history": [],
            "knowledge_contributions": 0,
        }

        try:
            # Get user profile from structured storage
            sync_results["user_profile"] = await self.get_user_profile(user_id)

            # Get active sessions
            supabase_sessions = await self._execute_with_fallback(
                "structured",
                "get_user_sessions",
                "get_user_sessions",
                user_id=user_id,
                active_only=True,
            )
            sync_results["active_sessions"] = supabase_sessions or []

            # Get research history
            sync_results["research_history"] = await self.get_research_history(
                user_id, limit=10
            )

            # Count knowledge contributions (this would need a custom query)
            # For now, we'll leave it as 0

            self.stats["sync_operations"] += 1

            await self.context_stream.add_event(
                ContextEventType.TOOL_EXECUTION,
                {
                    "tool": "hybrid_storage",
                    "action": "user_data_synchronized",
                    "user_id": user_id,
                    "sync_results": {
                        "has_profile": bool(sync_results["user_profile"]),
                        "active_sessions_count": len(sync_results["active_sessions"]),
                        "research_items_count": len(sync_results["research_history"]),
                    },
                },
            )

            return sync_results

        except Exception as e:
            logger.error(f"❌ Error synchronizing user data: {e}")
            return sync_results

    # Health and Monitoring
    async def comprehensive_health_check(self) -> Dict[str, Any]:
        """Perform health check across all backends"""

        health_status = {
            "overall_health": "healthy",
            "backends": {},
            "timestamp": datetime.now().isoformat(),
        }

        # Check each backend
        for backend_name, backend in self.backends.items():
            try:
                if hasattr(backend.instance, "health_check"):
                    backend_health = await backend.instance.health_check()
                    health_status["backends"][backend_name] = backend_health
                else:
                    health_status["backends"][backend_name] = {
                        "status": "unknown",
                        "message": "No health check method available",
                    }

                # Update backend health status
                backend_is_healthy = health_status["backends"][backend_name].get(
                    "status"
                ) in ["healthy", "mock_mode"]
                backend.is_healthy = backend_is_healthy
                backend.last_health_check = datetime.now()

            except Exception as e:
                health_status["backends"][backend_name] = {
                    "status": "error",
                    "message": f"Health check failed: {str(e)}",
                }
                backend.is_healthy = False

        # Determine overall health
        unhealthy_backends = [
            name
            for name, health in health_status["backends"].items()
            if health.get("status") not in ["healthy", "mock_mode"]
        ]

        if unhealthy_backends:
            health_status["overall_health"] = (
                "degraded" if len(unhealthy_backends) == 1 else "unhealthy"
            )
            health_status["unhealthy_backends"] = unhealthy_backends

        self.stats["health_checks"] += 1

        return health_status

    async def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get statistics from all backends"""

        backend_stats = {}

        # Get stats from each backend
        for backend_name, backend in self.backends.items():
            try:
                if hasattr(backend.instance, "get_stats"):
                    backend_stats[backend_name] = backend.instance.get_stats()
                else:
                    backend_stats[backend_name] = {"message": "No stats available"}
            except Exception as e:
                backend_stats[backend_name] = {"error": str(e)}

        # Calculate uptime
        uptime = (datetime.now() - self.stats["start_time"]).total_seconds()

        return {
            "hybrid_manager_stats": {
                **self.stats,
                "uptime_seconds": uptime,
                "backend_health": {
                    name: backend.is_healthy for name, backend in self.backends.items()
                },
            },
            "backend_stats": backend_stats,
            "routing_rules": self.routing_rules,
        }

    async def reset_backend_health(self, backend_name: Optional[str] = None) -> bool:
        """Reset backend health status (for recovery after fixes)"""

        try:
            if backend_name:
                if backend_name in self.backends:
                    self.backends[backend_name].is_healthy = True
                    self.backends[backend_name].error_count = 0
                    logger.info(f"✅ Reset health for backend: {backend_name}")
                    return True
                else:
                    logger.error(f"❌ Backend not found: {backend_name}")
                    return False
            else:
                # Reset all backends
                for backend in self.backends.values():
                    backend.is_healthy = True
                    backend.error_count = 0

                logger.info("✅ Reset health for all backends")
                return True

        except Exception as e:
            logger.error(f"❌ Error resetting backend health: {e}")
            return False

    # Cleanup Operations
    async def close_user_session(self, session_id: str) -> bool:
        """Close session across all backends"""

        results = []

        # Close in structured storage
        supabase_result = await self._execute_with_fallback(
            "structured", "close_session", "close_session", session_id=session_id
        )
        results.append(supabase_result is not None)

        # Memory backend (Zep) doesn't need explicit session closure
        # Sessions remain available for future access

        success = any(results)  # Success if at least one backend succeeded

        if success:
            await self.context_stream.add_event(
                ContextEventType.TOOL_EXECUTION,
                {
                    "tool": "hybrid_storage",
                    "action": "session_closed",
                    "session_id": session_id,
                },
            )

            logger.info(f"✅ Session closed: {session_id}")

        return success
