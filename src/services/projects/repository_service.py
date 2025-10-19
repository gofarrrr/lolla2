"""Project repository service implemented on the unified DatabaseService."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.core.unified_context_stream import ContextEventType, get_unified_context_stream
from src.services.persistence import DatabaseService, DatabaseOperationError

from .specialized_contracts import (
    DatabaseError,
    NotFoundError,
    map_exception_to_event_type,
    IProjectRepositoryService,
)

logger = logging.getLogger(__name__)


class ProjectRepositoryService(IProjectRepositoryService):
    """Data access layer for projects, backed by the DatabaseService facade."""

    def __init__(
        self,
        database_service: Optional[DatabaseService] = None,
        context_stream: Optional[Any] = None,
        timeout_seconds: int = 30,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.database_service = database_service or DatabaseService()
        self.context_stream = context_stream or get_unified_context_stream()
        self.timeout_seconds = timeout_seconds
        self._query_count = 0
        self._error_count = 0

    async def create_project_record(self, project_data: Dict[str, Any]) -> Dict[str, Any]:
        start_time = datetime.now(timezone.utc)
        self._query_count += 1
        self._emit_event(
            ContextEventType.REASONING_STEP,
            {
                "operation": "create_project_record",
                "organization_id": project_data.get("organization_id"),
                "project_name": project_data.get("name"),
                "query_count": self._query_count,
            },
            {"repository_operation": True},
        )

        try:
            rows = await asyncio.to_thread(
                self.database_service.insert, "projects", project_data
            )
            if not rows:
                raise DatabaseError("Failed to create project - no data returned")
            project_record = rows[0]
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            self._emit_event(
                ContextEventType.REASONING_STEP,
                {
                    "operation": "create_project_record",
                    "status": "success",
                    "project_id": project_record.get("id"),
                    "processing_time_ms": processing_time,
                },
                {"repository_success": True},
            )
            return project_record
        except DatabaseOperationError as exc:
            self._error_count += 1
            self._emit_error("create_project_record", project_data, exc)
            raise DatabaseError(f"Project creation failed: {exc}") from exc

    async def get_project_record(self, project_id: str) -> Optional[Dict[str, Any]]:
        start_time = datetime.now(timezone.utc)
        self._query_count += 1
        self._emit_event(
            ContextEventType.REASONING_STEP,
            {
                "operation": "get_project_record",
                "project_id": project_id,
                "query_count": self._query_count,
            },
            {"repository_operation": True},
        )

        try:
            record = await asyncio.to_thread(
                self.database_service.fetch_one, "projects", {"id": project_id}
            )
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            status = "success" if record else "not_found"
            self._emit_event(
                ContextEventType.REASONING_STEP,
                {
                    "operation": "get_project_record",
                    "status": status,
                    "project_id": project_id,
                    "processing_time_ms": processing_time,
                },
                {"repository_success": True},
            )
            return record
        except DatabaseOperationError as exc:
            self._error_count += 1
            self._emit_error("get_project_record", {"project_id": project_id}, exc)
            raise DatabaseError(f"Project retrieval failed: {exc}") from exc

    async def list_project_records(
        self, organization_id: str, limit: int = 50, offset: int = 0
    ) -> List[Dict[str, Any]]:
        start_time = datetime.now(timezone.utc)
        self._query_count += 1
        self._emit_event(
            ContextEventType.REASONING_STEP,
            {
                "operation": "list_project_records",
                "organization_id": organization_id,
                "limit": limit,
                "offset": offset,
                "query_count": self._query_count,
            },
            {"repository_operation": True},
        )

        try:
            records = await asyncio.to_thread(
                self.database_service.fetch_many,
                "projects",
                {"organization_id": organization_id},
                order_by="created_at",
                desc=True,
            )
            if limit is not None:
                projects = records[offset : offset + limit]
            else:
                projects = records[offset:]

            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            self._emit_event(
                ContextEventType.REASONING_STEP,
                {
                    "operation": "list_project_records",
                    "status": "success",
                    "organization_id": organization_id,
                    "projects_count": len(projects),
                    "processing_time_ms": processing_time,
                },
                {"repository_success": True},
            )
            return projects
        except DatabaseOperationError as exc:
            self._error_count += 1
            self._emit_error(
                "list_project_records",
                {"organization_id": organization_id, "limit": limit, "offset": offset},
                exc,
            )
            raise DatabaseError(f"Project listing failed: {exc}") from exc

    async def update_project_record(
        self, project_id: str, updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        start_time = datetime.now(timezone.utc)
        self._query_count += 1
        updates = dict(updates)
        updates["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._emit_event(
            ContextEventType.REASONING_STEP,
            {
                "operation": "update_project_record",
                "project_id": project_id,
                "update_fields": list(updates.keys()),
                "query_count": self._query_count,
            },
            {"repository_operation": True},
        )

        try:
            rows = await asyncio.to_thread(
                self.database_service.update, "projects", {"id": project_id}, updates
            )
            if not rows:
                raise NotFoundError(f"Project {project_id} not found for update")
            updated_record = rows[0]
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            self._emit_event(
                ContextEventType.REASONING_STEP,
                {
                    "operation": "update_project_record",
                    "status": "success",
                    "project_id": project_id,
                    "processing_time_ms": processing_time,
                },
                {"repository_success": True},
            )
            return updated_record
        except NotFoundError:
            raise
        except DatabaseOperationError as exc:
            self._error_count += 1
            self._emit_error(
                "update_project_record", {"project_id": project_id, "updates": updates}, exc
            )
            raise DatabaseError(f"Project update failed: {exc}") from exc

    async def delete_project_record(self, project_id: str) -> bool:
        start_time = datetime.now(timezone.utc)
        self._query_count += 1
        self._emit_event(
            ContextEventType.REASONING_STEP,
            {
                "operation": "delete_project_record",
                "project_id": project_id,
                "query_count": self._query_count,
            },
            {"repository_operation": True},
        )

        try:
            await self._cleanup_project_data(project_id)
            result = await asyncio.to_thread(
                self.database_service.delete, "projects", {"id": project_id}
            )
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            success = bool(result)
            self._emit_event(
                ContextEventType.REASONING_STEP,
                {
                    "operation": "delete_project_record",
                    "status": "success" if success else "not_found",
                    "project_id": project_id,
                    "processing_time_ms": processing_time,
                },
                {"repository_success": True},
            )
            return success
        except DatabaseOperationError as exc:
            self._error_count += 1
            self._emit_error("delete_project_record", {"project_id": project_id}, exc)
            raise DatabaseError(f"Project deletion failed: {exc}") from exc

    async def get_project_knowledge_base(self, project_id: str) -> List[Dict[str, Any]]:
        start_time = datetime.now(timezone.utc)
        self._query_count += 1
        self._emit_event(
            ContextEventType.REASONING_STEP,
            {
                "operation": "get_project_knowledge_base",
                "project_id": project_id,
                "query_count": self._query_count,
            },
            {"repository_operation": True},
        )

        try:
            records = await asyncio.to_thread(
                self.database_service.fetch_many,
                "knowledge_base",
                {"project_id": project_id},
                order_by="created_at",
                desc=True,
            )
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            self._emit_event(
                ContextEventType.REASONING_STEP,
                {
                    "operation": "get_project_knowledge_base",
                    "status": "success",
                    "project_id": project_id,
                    "records_count": len(records),
                    "processing_time_ms": processing_time,
                },
                {"repository_success": True},
            )
            return records
        except DatabaseOperationError as exc:
            self._error_count += 1
            self._emit_error("get_project_knowledge_base", {"project_id": project_id}, exc)
            raise DatabaseError(f"Knowledge base retrieval failed: {exc}") from exc

    async def store_analysis_record(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        start_time = datetime.now(timezone.utc)
        self._query_count += 1
        self._emit_event(
            ContextEventType.REASONING_STEP,
            {
                "operation": "store_analysis_record",
                "project_id": analysis_data.get("project_id"),
                "analysis_type": analysis_data.get("analysis_type"),
                "query_count": self._query_count,
            },
            {"repository_operation": True},
        )

        try:
            rows = await asyncio.to_thread(
                self.database_service.insert, "analyses", analysis_data
            )
            if not rows:
                raise DatabaseError("Failed to store analysis - no data returned")
            analysis_record = rows[0]
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            self._emit_event(
                ContextEventType.REASONING_STEP,
                {
                    "operation": "store_analysis_record",
                    "status": "success",
                    "analysis_id": analysis_record.get("id"),
                    "processing_time_ms": processing_time,
                },
                {"repository_success": True},
            )
            return analysis_record
        except DatabaseOperationError as exc:
            self._error_count += 1
            self._emit_error("store_analysis_record", analysis_data, exc)
            raise DatabaseError(f"Analysis storage failed: {exc}") from exc

    async def get_project_statistics_data(self, project_id: str) -> Dict[str, Any]:
        start_time = datetime.now(timezone.utc)
        self._query_count += 1
        self._emit_event(
            ContextEventType.REASONING_STEP,
            {
                "operation": "get_project_statistics_data",
                "project_id": project_id,
                "query_count": self._query_count,
            },
            {"repository_operation": True},
        )

        try:
            project = await self.get_project_record(project_id)
            if not project:
                raise NotFoundError(f"Project {project_id} not found")

            analyses = await asyncio.to_thread(
                self.database_service.fetch_many,
                "analyses",
                {"project_id": project_id},
            )
            knowledge_records = await asyncio.to_thread(
                self.database_service.fetch_many,
                "knowledge_base",
                {"project_id": project_id},
            )

            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            statistics_data = {
                "project": project,
                "analyses": analyses,
                "knowledge_records": knowledge_records,
                "total_analyses": len(analyses),
                "total_knowledge_records": len(knowledge_records),
            }
            self._emit_event(
                ContextEventType.REASONING_STEP,
                {
                    "operation": "get_project_statistics_data",
                    "status": "success",
                    "project_id": project_id,
                    "analyses_count": len(analyses),
                    "knowledge_count": len(knowledge_records),
                    "processing_time_ms": processing_time,
                },
                {"repository_success": True},
            )
            return statistics_data
        except NotFoundError:
            raise
        except DatabaseOperationError as exc:
            self._error_count += 1
            self._emit_error("get_project_statistics_data", {"project_id": project_id}, exc)
            raise DatabaseError(f"Statistics data retrieval failed: {exc}") from exc

    async def _cleanup_project_data(self, project_id: str) -> None:
        try:
            await asyncio.to_thread(self.database_service.delete, "analyses", {"project_id": project_id})
            await asyncio.to_thread(
                self.database_service.delete, "knowledge_base", {"project_id": project_id}
            )
            self.logger.debug("Cleaned up data for project %s", project_id)
        except DatabaseOperationError as exc:
            self.logger.warning("Failed to cleanup data for project %s: %s", project_id, exc)

    def get_performance_metrics(self) -> Dict[str, Any]:
        return {
            "query_count": self._query_count,
            "error_count": self._error_count,
            "error_rate": self._error_count / max(1, self._query_count),
            "service_name": "ProjectRepositoryService",
        }

    def _emit_event(self, event_type: ContextEventType, payload: Dict[str, Any], meta: Optional[Dict[str, Any]] = None) -> None:
        try:
            self.context_stream.add_event(event_type, payload, meta or {})
        except Exception:
            pass

    def _emit_error(self, operation: str, payload: Dict[str, Any], exc: Exception) -> None:
        self._emit_event(
            ContextEventType.ERROR_OCCURRED,
            {
                "operation": operation,
                "error": str(exc),
                "error_type": map_exception_to_event_type(exc),
                **payload,
            },
            {"repository_error": True},
        )


def get_project_repository(database_service: Optional[DatabaseService] = None) -> IProjectRepositoryService:
    return ProjectRepositoryService(database_service=database_service)
