"""
Supabase Storage Integration for METIS 2.0
==========================================

Provides structured data storage using Supabase for:
- User profiles and preferences
- Session metadata
- Research results and analytics
- System configuration

Features:
- Real-time data synchronization
- Row-level security
- Structured query interface
- Automatic schema management
"""

import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import json
import os
import uuid

import aiohttp
from ..core.unified_context_stream import get_unified_context_stream, ContextEventType, UnifiedContextStream

logger = logging.getLogger(__name__)


class SupabaseStore:
    """
    Supabase storage manager for structured data
    """

    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_anon_key: Optional[str] = None,
        supabase_service_key: Optional[str] = None,
        context_stream: Optional[UnifiedContextStream] = None,
    ):
        """
        Initialize Supabase store

        Args:
            supabase_url: Supabase project URL
            supabase_anon_key: Supabase anonymous key
            supabase_service_key: Supabase service role key
            context_stream: Context stream for logging
        """
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.anon_key = supabase_anon_key or os.getenv("SUPABASE_ANON_KEY")
        self.service_key = supabase_service_key or os.getenv("SUPABASE_SERVICE_KEY")

        if not self.supabase_url:
            logger.warning("⚠️ No Supabase URL provided - using mock mode")

        self.context_stream = context_stream or get_unified_context_stream()

        # Use service key for admin operations, anon key for user operations
        self.admin_headers = {
            "apikey": self.service_key or "",
            "Authorization": f'Bearer {self.service_key or ""}',
            "Content-Type": "application/json",
        }

        self.user_headers = {
            "apikey": self.anon_key or "",
            "Authorization": f'Bearer {self.anon_key or ""}',
            "Content-Type": "application/json",
        }

        # Statistics
        self.stats = {
            "queries_executed": 0,
            "inserts_performed": 0,
            "updates_performed": 0,
            "deletes_performed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

        # Simple cache for frequently accessed data
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes

        logger.info("🗄️ SupabaseStore initialized")

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        use_service_key: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Make HTTP request to Supabase API"""

        if not self.supabase_url:
            # Mock response for development
            logger.info(f"🔧 Mock Supabase {method} request: {endpoint}")
            return {"mock": True, "data": data or {}}

        url = f"{self.supabase_url}/rest/v1/{endpoint}"
        headers = self.admin_headers if use_service_key else self.user_headers

        try:
            async with aiohttp.ClientSession() as client:
                async with client.request(
                    method=method, url=url, headers=headers, json=data, params=params
                ) as response:
                    if response.status >= 400:
                        error_text = await response.text()
                        logger.error(
                            f"❌ Supabase request failed: {response.status} - {error_text}"
                        )
                        return None

                    if response.status == 204:  # No content
                        return {"success": True}

                    return await response.json()

        except Exception as e:
            logger.error(f"❌ Supabase request error: {e}")
            return None

    # User Management
    async def create_user_profile(
        self,
        user_id: str,
        email: str,
        display_name: Optional[str] = None,
        preferences: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Create user profile"""

        profile_data = {
            "user_id": user_id,
            "email": email,
            "display_name": display_name or email.split("@")[0],
            "preferences": json.dumps(preferences or {}),
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }

        result = await self._make_request(
            "POST", "user_profiles", data=profile_data, use_service_key=True
        )

        success = result is not None
        if success:
            self.stats["inserts_performed"] += 1

            await self.context_stream.add_event(
                ContextEventType.TOOL_EXECUTION,
                {
                    "tool": "supabase_store",
                    "tool_name": "supabase_store",
                    "action": "user_profile_created",
                    "user_id": user_id,
                    "email": email,
                },
            )

            logger.info(f"✅ User profile created: {user_id}")

        return success

    async def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user profile by ID"""

        # Check cache first
        cache_key = f"user_profile_{user_id}"
        if cache_key in self._cache:
            self.stats["cache_hits"] += 1
            return self._cache[cache_key]

        result = await self._make_request(
            "GET", f"user_profiles?user_id=eq.{user_id}&select=*"
        )

        if result and isinstance(result, list) and len(result) > 0:
            profile = result[0]
            # Parse JSON preferences
            if "preferences" in profile and isinstance(profile["preferences"], str):
                try:
                    profile["preferences"] = json.loads(profile["preferences"])
                except:
                    profile["preferences"] = {}

            # Cache the result
            self._cache[cache_key] = profile
            self.stats["cache_misses"] += 1
            self.stats["queries_executed"] += 1

            return profile

        return None

    async def update_user_preferences(
        self, user_id: str, preferences: Dict[str, Any]
    ) -> bool:
        """Update user preferences"""

        update_data = {
            "preferences": json.dumps(preferences),
            "updated_at": datetime.now().isoformat(),
        }

        result = await self._make_request(
            "PATCH",
            f"user_profiles?user_id=eq.{user_id}",
            data=update_data,
            use_service_key=True,
        )

        success = result is not None
        if success:
            self.stats["updates_performed"] += 1

            # Invalidate cache
            cache_key = f"user_profile_{user_id}"
            if cache_key in self._cache:
                del self._cache[cache_key]

            logger.info(f"✅ User preferences updated: {user_id}")

        return success

    # Session Management
    async def create_session_record(
        self,
        session_id: str,
        user_id: str,
        session_type: str = "conversation",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Create session record"""

        session_data = {
            "session_id": session_id,
            "user_id": user_id,
            "session_type": session_type,
            "metadata": json.dumps(metadata or {}),
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "is_active": True,
        }

        result = await self._make_request(
            "POST", "session_records", data=session_data, use_service_key=True
        )

        success = result is not None
        if success:
            self.stats["inserts_performed"] += 1

            await self.context_stream.add_event(
                ContextEventType.TOOL_EXECUTION,
                {
                    "tool": "supabase_store",
                    "tool_name": "supabase_store",
                    "action": "session_record_created",
                    "session_id": session_id,
                    "user_id": user_id,
                    "session_type": session_type,
                },
            )

            logger.info(f"✅ Session record created: {session_id}")

        return success

    async def get_user_sessions(
        self, user_id: str, active_only: bool = False, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get user sessions"""

        query = f"session_records?user_id=eq.{user_id}"
        if active_only:
            query += "&is_active=eq.true"
        query += f"&order=updated_at.desc&limit={limit}&select=*"

        result = await self._make_request("GET", query)

        if result and isinstance(result, list):
            self.stats["queries_executed"] += 1

            # Parse JSON metadata for each session
            for session in result:
                if "metadata" in session and isinstance(session["metadata"], str):
                    try:
                        session["metadata"] = json.loads(session["metadata"])
                    except:
                        session["metadata"] = {}

            return result

        return []

    async def update_session_metadata(
        self, session_id: str, metadata: Dict[str, Any]
    ) -> bool:
        """Update session metadata"""

        update_data = {
            "metadata": json.dumps(metadata),
            "updated_at": datetime.now().isoformat(),
        }

        result = await self._make_request(
            "PATCH",
            f"session_records?session_id=eq.{session_id}",
            data=update_data,
            use_service_key=True,
        )

        success = result is not None
        if success:
            self.stats["updates_performed"] += 1
            logger.info(f"✅ Session metadata updated: {session_id}")

        return success

    async def close_session(self, session_id: str) -> bool:
        """Mark session as inactive"""

        update_data = {"is_active": False, "updated_at": datetime.now().isoformat()}

        result = await self._make_request(
            "PATCH",
            f"session_records?session_id=eq.{session_id}",
            data=update_data,
            use_service_key=True,
        )

        return result is not None

    # Research Results Storage
    async def store_research_result(
        self,
        research_id: str,
        user_id: str,
        query: str,
        results: Dict[str, Any],
        provider: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Store research result"""

        research_data = {
            "research_id": research_id,
            "user_id": user_id,
            "query": query,
            "results": json.dumps(results),
            "provider": provider,
            "metadata": json.dumps(metadata or {}),
            "created_at": datetime.now().isoformat(),
        }

        result = await self._make_request(
            "POST", "research_results", data=research_data, use_service_key=True
        )

        success = result is not None
        if success:
            self.stats["inserts_performed"] += 1

            await self.context_stream.add_event(
                ContextEventType.TOOL_EXECUTION,
                {
                    "tool": "supabase_store",
                    "action": "research_result_stored",
                    "research_id": research_id,
                    "provider": provider,
                    "query": query[:100] + "..." if len(query) > 100 else query,
                },
            )

            logger.info(f"✅ Research result stored: {research_id}")

        return success

    async def get_research_history(
        self, user_id: str, limit: int = 20, provider: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get research history for user"""

        query = f"research_results?user_id=eq.{user_id}"
        if provider:
            query += f"&provider=eq.{provider}"
        query += f"&order=created_at.desc&limit={limit}&select=*"

        result = await self._make_request("GET", query)

        if result and isinstance(result, list):
            self.stats["queries_executed"] += 1

            # Parse JSON fields
            for item in result:
                for field in ["results", "metadata"]:
                    if field in item and isinstance(item[field], str):
                        try:
                            item[field] = json.loads(item[field])
                        except:
                            item[field] = {}

            return result

        return []

    # Analytics Storage
    async def store_analytics_event(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        event_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Store analytics event"""

        analytics_data = {
            "event_id": str(uuid.uuid4()),
            "event_type": event_type,
            "user_id": user_id,
            "session_id": session_id,
            "event_data": json.dumps(event_data or {}),
            "created_at": datetime.now().isoformat(),
        }

        result = await self._make_request(
            "POST", "analytics_events", data=analytics_data, use_service_key=True
        )

        success = result is not None
        if success:
            self.stats["inserts_performed"] += 1

        return success

    async def get_analytics_summary(
        self,
        start_date: datetime,
        end_date: datetime,
        event_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Get analytics summary for date range"""

        query = f"analytics_events?created_at=gte.{start_date.isoformat()}&created_at=lte.{end_date.isoformat()}"
        if event_types:
            query += "&event_type=in.(" + ",".join(event_types) + ")"
        query += "&select=*"

        result = await self._make_request("GET", query)

        if result and isinstance(result, list):
            self.stats["queries_executed"] += 1

            # Calculate summary statistics
            summary = {
                "total_events": len(result),
                "event_types": {},
                "unique_users": set(),
                "unique_sessions": set(),
                "date_range": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat(),
                },
            }

            for event in result:
                event_type = event.get("event_type", "unknown")
                summary["event_types"][event_type] = (
                    summary["event_types"].get(event_type, 0) + 1
                )

                if event.get("user_id"):
                    summary["unique_users"].add(event["user_id"])
                if event.get("session_id"):
                    summary["unique_sessions"].add(event["session_id"])

            summary["unique_users"] = len(summary["unique_users"])
            summary["unique_sessions"] = len(summary["unique_sessions"])

            return summary

        return {"total_events": 0, "error": "Failed to retrieve analytics data"}

    # Configuration Management
    async def get_system_config(self, config_key: str) -> Optional[Dict[str, Any]]:
        """Get system configuration"""

        cache_key = f"config_{config_key}"
        if cache_key in self._cache:
            self.stats["cache_hits"] += 1
            return self._cache[cache_key]

        result = await self._make_request(
            "GET", f"system_config?config_key=eq.{config_key}&select=*"
        )

        if result and isinstance(result, list) and len(result) > 0:
            config = result[0]
            if "config_value" in config and isinstance(config["config_value"], str):
                try:
                    config["config_value"] = json.loads(config["config_value"])
                except:
                    pass  # Keep as string if not valid JSON

            self._cache[cache_key] = config
            self.stats["cache_misses"] += 1
            self.stats["queries_executed"] += 1

            return config

        return None

    async def set_system_config(
        self, config_key: str, config_value: Any, description: Optional[str] = None
    ) -> bool:
        """Set system configuration"""

        config_data = {
            "config_key": config_key,
            "config_value": (
                json.dumps(config_value)
                if not isinstance(config_value, str)
                else config_value
            ),
            "description": description or "",
            "updated_at": datetime.now().isoformat(),
        }

        # Try insert first (upsert behavior)
        result = await self._make_request(
            "POST", "system_config", data=config_data, use_service_key=True
        )

        if not result:
            # If insert failed, try update
            result = await self._make_request(
                "PATCH",
                f"system_config?config_key=eq.{config_key}",
                data=config_data,
                use_service_key=True,
            )

        success = result is not None
        if success:
            self.stats["updates_performed"] += 1

            # Invalidate cache
            cache_key = f"config_{config_key}"
            if cache_key in self._cache:
                del self._cache[cache_key]

            logger.info(f"✅ System config updated: {config_key}")

        return success

    # Utility Methods
    async def execute_raw_query(
        self, table: str, query_params: str, method: str = "GET"
    ) -> Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]:
        """Execute raw query with custom parameters"""

        result = await self._make_request(
            method, f"{table}?{query_params}", use_service_key=True
        )

        if result:
            self.stats["queries_executed"] += 1

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        return {
            **self.stats,
            "cache_size": len(self._cache),
            "has_url": bool(self.supabase_url),
            "has_keys": bool(self.anon_key and self.service_key),
        }

    async def health_check(self) -> Dict[str, Any]:
        """Check Supabase connection health"""

        if not self.supabase_url:
            return {
                "status": "mock_mode",
                "available": True,
                "message": "Running in mock mode - no Supabase URL configured",
            }

        try:
            # Simple health check query
            result = await self._make_request("GET", "user_profiles?limit=1")

            if result is not None:
                return {
                    "status": "healthy",
                    "available": True,
                    "message": "Supabase API accessible",
                }
            else:
                return {
                    "status": "unhealthy",
                    "available": False,
                    "message": "Failed to query Supabase API",
                }

        except Exception as e:
            return {
                "status": "error",
                "available": False,
                "message": f"Health check failed: {str(e)}",
            }

    async def clear_cache(self) -> None:
        """Clear all cached data"""
        self._cache.clear()
        logger.info("🧹 Supabase cache cleared")
