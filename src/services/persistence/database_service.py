"""Unified database access facade for the Lolla backend."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

try:
    from supabase import Client, create_client  # type: ignore
except Exception:  # pragma: no cover - library missing in some environments
    Client = Any  # type: ignore
    create_client = None  # type: ignore


class DatabaseOperationError(RuntimeError):
    """Raised when a database operation cannot be completed."""


@dataclass
class DatabaseServiceConfig:
    """Configuration options for :class:`DatabaseService`."""

    url: Optional[str] = None
    anon_key: Optional[str] = None
    service_role_key: Optional[str] = None
    timeout_seconds: int = 30
    schema: Optional[str] = None

    @classmethod
    def from_env(cls) -> "DatabaseServiceConfig":
        """Build a configuration object using environment variables."""

        return cls(
            url=os.getenv("SUPABASE_URL"),
            anon_key=os.getenv("SUPABASE_ANON_KEY"),
            service_role_key=
            os.getenv("SUPABASE_SERVICE_ROLE_KEY")
            or os.getenv("SUPABASE_SERVICE_KEY")
            or os.getenv("SUPABASE_SECRET_KEY"),
            timeout_seconds=int(os.getenv("SUPABASE_TIMEOUT_SECONDS", "30")),
            schema=os.getenv("SUPABASE_SCHEMA"),
        )


class DatabaseService:
    """Centralized facade for all Supabase/Postgres interactions."""

    def __init__(
        self,
        config: Optional[DatabaseServiceConfig] = None,
        client: Optional[Client] = None,
    ) -> None:
        self.config = config or DatabaseServiceConfig.from_env()

        if client is not None:
            self._client = client
        else:
            if create_client is None:
                raise DatabaseOperationError(
                    "supabase client library is not available in this environment"
                )

            url = self.config.url
            key = self.config.service_role_key or self.config.anon_key
            if not url or not key:
                raise DatabaseOperationError(
                    "Supabase credentials are not configured; set SUPABASE_URL and a key"
                )

            try:
                self._client = create_client(url, key)
            except Exception as e:
                # Handle DNS and network errors gracefully
                error_msg = str(e)
                if "nodename nor servname" in error_msg or "Name or service not known" in error_msg:
                    raise DatabaseOperationError(
                        f"DNS resolution failed for Supabase URL: {url}. "
                        "Check network connectivity and URL configuration."
                    )
                else:
                    raise DatabaseOperationError(f"Failed to create Supabase client: {error_msg}")

            if self.config.schema:
                try:
                    self._client.postgrest.schema = self.config.schema  # type: ignore[attr-defined]
                except AttributeError:
                    pass  # Older versions of supabase-py do not expose schema override

    # ------------------------------------------------------------------
    # Basic data access helpers
    # ------------------------------------------------------------------

    @property
    def client(self) -> Client:
        """Expose the underlying Supabase client when absolutely required."""

        return self._client

    def health_check(self) -> bool:
        """Check if the database connection is healthy.

        Returns:
            bool: True if connection is healthy, False otherwise
        """
        try:
            # Try multiple fallback tables to test connectivity
            test_tables = ["engagements", "prompt_capture", "cognitive_states"]

            for table in test_tables:
                try:
                    result = self._client.table(table).select("*").limit(1).execute()
                    return True
                except Exception as table_e:
                    # If it's a DNS/network error, fail immediately
                    error_msg = str(table_e)
                    if any(dns_indicator in error_msg.lower() for dns_indicator in
                           ["nodename nor servname", "name or service not known", "connection refused", "timeout"]):
                        raise table_e
                    # Otherwise try next table
                    continue

            # If all tables failed but no DNS errors, return False
            return False

        except Exception as e:
            # Log DNS and network errors
            import logging
            logger = logging.getLogger(__name__)
            error_msg = str(e)

            if any(dns_indicator in error_msg.lower() for dns_indicator in
                   ["nodename nor servname", "name or service not known"]):
                logger.error(f"🚨 DNS resolution failed for Supabase: {error_msg}")
            elif "connection refused" in error_msg.lower():
                logger.error(f"🚨 Connection refused by Supabase: {error_msg}")
            elif "timeout" in error_msg.lower():
                logger.error(f"🚨 Connection timeout to Supabase: {error_msg}")
            else:
                logger.warning(f"Database health check failed: {error_msg}")

            return False

    def _apply_filters(self, query: Any, filters: Optional[Dict[str, Any]]) -> Any:
        if not filters:
            return query
        for column, value in filters.items():
            query = query.eq(column, value)
        return query

    def fetch_many(
        self,
        table: str,
        filters: Optional[Dict[str, Any]] = None,
        *,
        columns: str = "*",
        limit: Optional[int] = None,
        order_by: Optional[str] = None,
        desc: bool = False,
    ) -> List[Dict[str, Any]]:
        """Fetch many rows from a table."""

        try:
            query = self._client.table(table).select(columns)
            query = self._apply_filters(query, filters)
            if order_by:
                query = query.order(order_by, desc=desc)
            if limit is not None:
                query = query.limit(limit)
            response = query.execute()
            return response.data or []
        except Exception as exc:  # pragma: no cover - supabase handles errors
            raise DatabaseOperationError(str(exc)) from exc

    async def fetch_many_async(
        self,
        table: str,
        filters: Optional[Dict[str, Any]] = None,
        *,
        columns: str = "*",
        limit: Optional[int] = None,
        order_by: Optional[str] = None,
        desc: bool = False,
    ) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(
            self.fetch_many,
            table,
            filters,
            columns=columns,
            limit=limit,
            order_by=order_by,
            desc=desc,
        )

    def fetch_one(
        self,
        table: str,
        filters: Dict[str, Any],
        *,
        columns: str = "*",
    ) -> Optional[Dict[str, Any]]:
        rows = self.fetch_many(table, filters, columns=columns, limit=1)
        return rows[0] if rows else None

    async def fetch_one_async(
        self,
        table: str,
        filters: Dict[str, Any],
        *,
        columns: str = "*",
    ) -> Optional[Dict[str, Any]]:
        return await asyncio.to_thread(
            self.fetch_one,
            table,
            filters,
            columns=columns,
        )

    def insert(self, table: str, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        # OPERATION POWER ON: Critical persistence logging
        import logging
        import time as _time
        logger = logging.getLogger(__name__)
        logger.info(f"🔍 PERSISTENCE ATTEMPT: table={table}, data_keys={list(data.keys())}, trace_id={data.get('trace_id', 'NO_TRACE_ID')}")
        delays = [0.25, 0.75, 2.0]
        last_exc = None
        for attempt in range(len(delays) + 1):
            try:
                response = self._client.table(table).insert(data).execute()
                result = response.data or []
                logger.info(f"✅ PERSISTENCE SUCCESS: table={table}, rows_inserted={len(result)}, trace_id={data.get('trace_id', 'NO_TRACE_ID')}")
                return result
            except Exception as exc:
                last_exc = exc
                if attempt < len(delays):
                    d = delays[attempt]
                    logger.warning(f"⚠️ INSERT attempt {attempt+1} failed (table={table}), retrying in {d}s: {exc}")
                    try:
                        _time.sleep(d)
                    except Exception:
                        pass
                    continue
                logger.error(f"❌ PERSISTENCE FAILURE: table={table}, error={str(exc)}, trace_id={data.get('trace_id', 'NO_TRACE_ID')}")
                raise DatabaseOperationError(str(exc)) from exc

    async def insert_many_async(
        self, table: str, rows: Iterable[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self.insert_many, table, rows)

    def insert_many(
        self, table: str, rows: Iterable[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        import time as _time
        payload = list(rows)
        if not payload:
            return []
        delays = [0.25, 0.75, 2.0]
        last_exc = None
        for attempt in range(len(delays) + 1):
            try:
                response = self._client.table(table).insert(payload).execute()
                return response.data or []
            except Exception as exc:
                last_exc = exc
                if attempt < len(delays):
                    d = delays[attempt]
                    try:
                        _time.sleep(d)
                    except Exception:
                        pass
                    continue
                raise DatabaseOperationError(str(exc)) from exc

    def upsert(self, table: str, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        import time as _time
        delays = [0.25, 0.75, 2.0]
        for attempt in range(len(delays) + 1):
            try:
                response = self._client.table(table).upsert(data).execute()
                return response.data or []
            except Exception as exc:
                if attempt < len(delays):
                    d = delays[attempt]
                    try:
                        _time.sleep(d)
                    except Exception:
                        pass
                    continue
                raise DatabaseOperationError(str(exc)) from exc

    def update(
        self,
        table: str,
        filters: Dict[str, Any],
        updates: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        import time as _time
        delays = [0.25, 0.75, 2.0]
        for attempt in range(len(delays) + 1):
            try:
                query = self._client.table(table).update(updates)
                query = self._apply_filters(query, filters)
                response = query.execute()
                return response.data or []
            except Exception as exc:
                if attempt < len(delays):
                    d = delays[attempt]
                    try:
                        _time.sleep(d)
                    except Exception:
                        pass
                    continue
                raise DatabaseOperationError(str(exc)) from exc

    def delete(self, table: str, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            query = self._client.table(table).delete()
            query = self._apply_filters(query, filters)
            response = query.execute()
            return response.data or []
        except Exception as exc:
            raise DatabaseOperationError(str(exc)) from exc

    def call_rpc(self, function_name: str, params: Optional[Dict[str, Any]] = None) -> Any:
        try:
            response = self._client.rpc(function_name, params or {}).execute()
            return response.data
        except Exception as exc:
            raise DatabaseOperationError(str(exc)) from exc

    # ------------------------------------------------------------------
    # Calibration persistence helpers
    # ------------------------------------------------------------------

    async def fetch_all_calibration_data(self) -> List[Dict[str, Any]]:
        """Return all calibration rows (predictions and outcomes).

        Table schema expectation (calibration_data):
          - id (bigserial, primary key)
          - trace_id (text)
          - model_id (text)
          - predicted_probability (float8)
          - actual_outcome (float8)
          - context (jsonb, nullable)
          - created_at (timestamptz)
        """
        return await asyncio.to_thread(self.fetch_many, "calibration_data", None, columns="*")

    async def store_initial_prediction(self, prediction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Persist a calibration prediction row into calibration_data table.

        Expects keys: id, trace_id, persona_id, predicted_probability, is_early, notes, created_at
        """
        return await asyncio.to_thread(self.insert, "calibration_data", prediction_data)

    async def store_calibration_outcome(self, *, id: str, actual_outcome: float, notes: Optional[str] = None) -> None:
        """Update a calibration prediction row with outcome and timestamp."""
        updates: Dict[str, Any] = {
            "actual_outcome": float(actual_outcome),
            "outcome_reported_at": datetime.now().isoformat(),
        }
        if notes is not None:
            updates["notes"] = notes
        await asyncio.to_thread(self.update, "calibration_data", {"id": id}, updates)

    # ------------------------------------------------------------------
    # Domain-specific helpers used by the Phoenix pipeline.
    # ------------------------------------------------------------------

    def store_engagement_report(
        self,
        *,
        trace_id: str,
        user_id: Optional[str],
        user_query: str,
        processing_time_seconds: float,
        final_report_contract: Dict[str, Any],
        accumulated_context: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist the final engagement report."""

        # OPERATION POWER ON: Critical logging for engagement persistence
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"🔍 STORE_ENGAGEMENT_REPORT: Starting engagement persistence for trace_id={trace_id}")
        logger.info(f"🔍 STORE_ENGAGEMENT_REPORT: user_query={user_query[:100] if user_query else 'None'}...")
        logger.info(f"🔍 STORE_ENGAGEMENT_REPORT: processing_time={processing_time_seconds:.2f}s")

        payload: Dict[str, Any] = {
            "trace_id": trace_id,
            "user_id": user_id,
            "user_query": user_query,
            "status": "COMPLETED",
            "completed_at": datetime.now().isoformat(),
            "total_duration_seconds": int(processing_time_seconds),
            "final_report_json": {
                "final_report_contract": final_report_contract,
                "accumulated_context": accumulated_context,
                "metadata": metadata
                or {
                    "pipeline_version": "v5.3",
                    "processing_time_seconds": processing_time_seconds,
                    "generated_at": datetime.now().isoformat(),
                },
            },
        }

        tokens_used = accumulated_context.get("total_tokens_used")
        cost_usd = accumulated_context.get("total_cost_usd")
        if tokens_used is not None:
            payload["tokens_used"] = tokens_used
        if cost_usd is not None:
            payload["cost_usd"] = cost_usd

        existing = self.fetch_one("engagements", {"trace_id": trace_id}, columns="trace_id")
        if existing:
            logger.info(f"🔍 STORE_ENGAGEMENT_REPORT: Existing engagement found, updating trace_id={trace_id}")
            self.update("engagements", {"trace_id": trace_id}, payload)
            logger.info(f"✅ STORE_ENGAGEMENT_REPORT: Successfully updated engagement for trace_id={trace_id}")
        else:
            logger.info(f"🔍 STORE_ENGAGEMENT_REPORT: No existing engagement, creating new one for trace_id={trace_id}")
            payload["created_at"] = datetime.now().isoformat()
            self.insert("engagements", payload)
            logger.info(f"✅ STORE_ENGAGEMENT_REPORT: Successfully inserted engagement for trace_id={trace_id}")

    def get_engagement_report(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get engagement report by trace_id - assumes trace_id column exists"""
        return self.fetch_one("engagements", {"trace_id": trace_id})

    def store_learning_session(self, session_data: Dict[str, Any]) -> None:
        self.insert("learning_analysis_sessions", session_data)

    async def store_learning_session_async(self, session_data: Dict[str, Any]) -> None:
        await asyncio.to_thread(self.store_learning_session, session_data)

    def store_model_performance(self, performance_data: Dict[str, Any]) -> None:
        self.insert("learning_model_performance", performance_data)

    async def store_model_performance_async(
        self, performance_data: Dict[str, Any]
    ) -> None:
        await asyncio.to_thread(self.store_model_performance, performance_data)

    def upsert_model_weight(self, weight_data: Dict[str, Any]) -> None:
        self.upsert("model_selection_weights", weight_data)

    def fetch_all_model_weights(self) -> List[Dict[str, Any]]:
        """Return all role weight rows from model_selection_weights."""
        try:
            response = self._client.table("model_selection_weights").select("*").execute()
            return response.data or []
        except Exception as exc:
            raise DatabaseOperationError(str(exc)) from exc

    def fetch_model_weight_by_role(self, role: str) -> Optional[Dict[str, Any]]:
        return self.fetch_one("model_selection_weights", {"role": role})

    def execute_sql(self, sql: str) -> List[Dict[str, Any]]:
        """Execute raw SQL through Supabase's execute_sql RPC."""

        return self.call_rpc("execute_sql", {"sql": sql}) or []

    async def execute_sql_async(self, sql: str) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self.execute_sql, sql)

    def get_learning_session_id(self, trace_id: str) -> Optional[str]:
        record = self.fetch_one(
            "learning_analysis_sessions",
            {"trace_id": trace_id},
            columns="id",
        )
        if not record:
            return None
        identifier = record.get("id")
        return str(identifier) if identifier is not None else None

    async def get_learning_session_id_async(self, trace_id: str) -> Optional[str]:
        return await asyncio.to_thread(self.get_learning_session_id, trace_id)

    def fetch_model_performances(
        self,
        *,
        model_id: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        try:
            query = self._client.table("learning_model_performance").select("*")
            if start is not None:
                query = query.gte("created_at", start.isoformat())
            if end is not None:
                query = query.lte("created_at", end.isoformat())
            if model_id is not None:
                query = query.eq("model_id", model_id)
            response = query.execute()
            return response.data or []
        except Exception as exc:
            raise DatabaseOperationError(str(exc)) from exc

    async def fetch_model_performances_async(
        self,
        *,
        model_id: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(
            self.fetch_model_performances,
            model_id=model_id,
            start=start,
            end=end,
        )

    # ============================================================================
    # CHECKPOINT PERSISTENCE METHODS - Operation Polish
    # ============================================================================

    def save_checkpoint(self, checkpoint_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Save a checkpoint to the database.

        Args:
            checkpoint_data: Serialized StateCheckpoint as dict (from model_dump())

        Returns:
            The saved checkpoint data with database-generated fields
        """
        try:
            return self.insert("state_checkpoints", checkpoint_data)
        except Exception as exc:
            raise DatabaseOperationError(f"Failed to save checkpoint: {exc}") from exc

    async def save_checkpoint_async(self, checkpoint_data: Dict[str, Any]) -> Dict[str, Any]:
        """Async wrapper for save_checkpoint"""
        return await asyncio.to_thread(self.save_checkpoint, checkpoint_data)

    def load_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a checkpoint by its ID.

        Args:
            checkpoint_id: UUID of the checkpoint to load

        Returns:
            Checkpoint data as dict, or None if not found
        """
        try:
            return self.fetch_one(
                "state_checkpoints",
                {"checkpoint_id": checkpoint_id},
                columns="*"
            )
        except Exception as exc:
            raise DatabaseOperationError(f"Failed to load checkpoint: {exc}") from exc

    async def load_checkpoint_async(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Async wrapper for load_checkpoint"""
        return await asyncio.to_thread(self.load_checkpoint, checkpoint_id)

    def load_checkpoints_for_trace(self, trace_id: str) -> List[Dict[str, Any]]:
        """
        Load all checkpoints for a given trace_id, ordered by creation time.

        Args:
            trace_id: UUID of the trace/analysis

        Returns:
            List of checkpoint dicts, ordered by created_at DESC
        """
        try:
            query = self._client.table("state_checkpoints") \
                .select("*") \
                .eq("trace_id", trace_id) \
                .order("created_at", desc=True)
            response = query.execute()
            return response.data or []
        except Exception as exc:
            raise DatabaseOperationError(f"Failed to load checkpoints for trace: {exc}") from exc

    async def load_checkpoints_for_trace_async(self, trace_id: str) -> List[Dict[str, Any]]:
        """Async wrapper for load_checkpoints_for_trace"""
        return await asyncio.to_thread(self.load_checkpoints_for_trace, trace_id)

    # ------------------------------------------------------------------
    # Engagement Status Management (Phase 1: Database-Backed Status)
    # ------------------------------------------------------------------

    def upsert_engagement_status(self, engagement_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Upsert engagement status to engagement_runs table.
        Uses Supabase upsert for atomic updates with versioning.

        Adds limited retry/backoff for transient connectivity errors.

        Args:
            engagement_data: Dict with trace_id, status, current_stage, etc.

        Returns:
            Updated engagement record with incremented version
        """
        import logging as _logging
        import time as _time
        logger = _logging.getLogger(__name__)

        delays = [0.25, 0.75, 2.0]
        last_exc = None

        for attempt in range(len(delays) + 1):
            try:
                logger.info(
                    f"🔄 Upserting engagement status: trace_id={engagement_data.get('trace_id')}, "
                    f"status={engagement_data.get('status')}"
                )
                response = self._client.table("engagement_runs").upsert(engagement_data).execute()
                result = response.data[0] if response.data else {}
                logger.info(
                    f"✅ Engagement status upserted: trace_id={result.get('trace_id')}, "
                    f"status={result.get('status')}, version={result.get('version')}"
                )
                return result
            except Exception as exc:
                last_exc = exc
                if attempt < len(delays):
                    d = delays[attempt]
                    logger.warning(
                        f"⚠️ Status upsert attempt {attempt+1} failed, retrying in {d}s: {exc}"
                    )
                    try:
                        _time.sleep(d)
                    except Exception:
                        pass
                    continue
                logger.error(f"❌ Failed to upsert engagement status after retries: {exc}")
                raise DatabaseOperationError(f"Failed to upsert engagement status: {exc}") from exc

    async def upsert_engagement_status_async(self, engagement_data: Dict[str, Any]) -> Dict[str, Any]:
        """Async wrapper for upsert_engagement_status"""
        return await asyncio.to_thread(self.upsert_engagement_status, engagement_data)

    def get_engagement_status(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """
        Get engagement status by trace_id.

        Args:
            trace_id: UUID of the engagement

        Returns:
            Engagement record with current status and version, or None if not found
        """
        try:
            response = self._client.table("engagement_runs") \
                .select("*") \
                .eq("trace_id", trace_id) \
                .execute()
            return response.data[0] if response.data else None
        except Exception as exc:
            raise DatabaseOperationError(f"Failed to get engagement status: {exc}") from exc

    async def get_engagement_status_async(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Async wrapper for get_engagement_status"""
        return await asyncio.to_thread(self.get_engagement_status, trace_id)

    def fetch_learning_sessions(
        self,
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        end_inclusive: bool = True,
    ) -> List[Dict[str, Any]]:
        try:
            query = self._client.table("learning_analysis_sessions").select("*")
            if start is not None:
                query = query.gte("created_at", start.isoformat())
            if end is not None:
                if end_inclusive:
                    query = query.lte("created_at", end.isoformat())
                else:
                    query = query.lt("created_at", end.isoformat())
            response = query.execute()
            return response.data or []
        except Exception as exc:
            raise DatabaseOperationError(str(exc)) from exc

    async def fetch_learning_sessions_async(
        self,
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        end_inclusive: bool = True,
    ) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(
            self.fetch_learning_sessions,
            start=start,
            end=end,
            end_inclusive=end_inclusive,
        )

    def log_flywheel_metrics(
        self,
        session_data: Dict[str, Any],
        metrics_data: Dict[str, Any],
        dispositions: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.insert("arbitration_sessions", session_data)
        self.insert("flywheel_metrics", metrics_data)
        if dispositions:
            self.insert_many("critique_dispositions", dispositions)

    async def log_flywheel_metrics_async(
        self,
        session_data: Dict[str, Any],
        metrics_data: Dict[str, Any],
        dispositions: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        await asyncio.to_thread(
            self.log_flywheel_metrics,
            session_data,
            metrics_data,
            dispositions,
        )

    def update_flywheel_metrics(
        self,
        session_id: str,
        updates: Dict[str, Any],
    ) -> None:
        self.update(
            "flywheel_metrics",
            {"arbitration_session_id": session_id},
            updates,
        )

    async def update_flywheel_metrics_async(
        self,
        session_id: str,
        updates: Dict[str, Any],
    ) -> None:
        await asyncio.to_thread(
            self.update_flywheel_metrics,
            session_id,
            updates,
        )

    def get_flywheel_metrics(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self.fetch_one(
            "flywheel_metrics",
            {"arbitration_session_id": session_id},
        )

    async def get_flywheel_metrics_async(
        self,
        session_id: str,
    ) -> Optional[Dict[str, Any]]:
        return await asyncio.to_thread(
            self.get_flywheel_metrics,
            session_id,
        )

    def record_project(self, project_data: Dict[str, Any]) -> Dict[str, Any]:
        rows = self.insert("projects", project_data)
        return rows[0] if rows else project_data

    def update_project(self, project_id: str, updates: Dict[str, Any]) -> None:
        self.update("projects", {"id": project_id}, updates)

    def delete_project(self, project_id: str) -> None:
        self.delete("projects", {"id": project_id})

    def call_health_rpc(self) -> Any:
        return self.call_rpc("get_component_health")

    # ------------------------------------------------------------------
    # Model selection helpers
    # ------------------------------------------------------------------

    def fetch_mental_model_by_id(self, model_id: str) -> Optional[Dict[str, Any]]:
        return self.fetch_one("mental_models", {"model_id": model_id})

    def fetch_active_mental_models(self) -> List[Dict[str, Any]]:
        try:
            response = (
                self._client.table("mental_models").select("*").eq("is_active", True).execute()
            )
            return response.data or []
        except Exception as exc:
            raise DatabaseOperationError(str(exc)) from exc

    # ------------------------------------------------------------------
    # Prompt capture helpers
    # ------------------------------------------------------------------

    async def store_prompt_capture_event(self, event_data: Dict[str, Any]) -> None:
        """Persist a prompt-capture event into the `prompt_capture` table.

        Runs the insert on a worker thread to avoid blocking callers.
        """
        await asyncio.to_thread(self.insert, "prompt_capture", event_data)

    # ------------------------------------------------------------------
    # Playbook router helpers
    # ------------------------------------------------------------------

    async def fetch_all_playbooks_async(self) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self.fetch_many, "playbooks")

    async def upsert_playbook_async(self, playbook_data: Dict[str, Any]) -> None:
        await asyncio.to_thread(self.upsert, "playbooks", playbook_data)

    async def log_playbook_routing_async(self, routing_record: Dict[str, Any]) -> None:
        await asyncio.to_thread(self.insert, "playbook_routing_history", routing_record)
