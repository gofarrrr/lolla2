"""
Context Persistence Service

Extracted from UnifiedContextStream (Task 6.0)
Handles checkpoint creation, database persistence, and engagement management.

Created: 2025-10-19
Campaign: Operation Lean
Original Lines: 1254-1513 from unified_context_stream.py
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class ContextPersistenceService:
    """
    Context stream persistence and checkpoint service.

    Responsibilities:
    - Create and restore checkpoints
    - Persist context streams to database
    - Manage engagement lifecycle
    - Build persistence records
    - Calculate summary metrics
    """

    def __init__(
        self,
        events: List[Any],
        trace_id: str,
        started_at: datetime,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        persistence_adapter: Optional[Any] = None,
        cache_hits: int = 0,
        cache_misses: int = 0,
    ):
        """
        Initialize context persistence service.

        Args:
            events: List of context events
            trace_id: Unique trace identifier
            started_at: When the engagement started
            session_id: Optional session identifier
            user_id: Optional user identifier
            organization_id: Optional organization identifier
            persistence_adapter: Optional persistence adapter (defaults to FileAdapter)
            cache_hits: Current cache hit count
            cache_misses: Current cache miss count
        """
        self.events = events
        self.trace_id = trace_id
        self.started_at = started_at
        self.session_id = session_id
        self.user_id = user_id
        self.organization_id = organization_id
        self.persistence_adapter = persistence_adapter
        self.cache_hits = cache_hits
        self.cache_misses = cache_misses

        # Engagement tracking
        self.engagement_type = "consultation"
        self.case_id: Optional[str] = None
        self.completed_at: Optional[datetime] = None
        self.final_analysis_text: Optional[str] = None

    def create_checkpoint(self) -> Dict[str, Any]:
        """
        Create checkpoint for persistence.

        Returns:
            Checkpoint dictionary with last 100 events and stats
        """
        return {
            "events": [e.to_dict() for e in self.events[-100:]],  # Last 100 events
            "stats": {
                "total_events": len(self.events),
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def restore_from_checkpoint(self, checkpoint: Dict[str, Any]) -> List[Any]:
        """
        Restore events from checkpoint.

        Args:
            checkpoint: Checkpoint dictionary

        Returns:
            List of restored events
        """
        # Import here to avoid circular dependency
        try:
            from src.core.unified_context_stream import ContextEvent, ContextEventType
        except ImportError:
            logger.error("Cannot import ContextEvent or ContextEventType")
            return []

        restored_events = []

        for event_dict in checkpoint.get("events", []):
            event = ContextEvent(
                event_id=event_dict["event_id"],
                event_type=ContextEventType(event_dict["event_type"]),
                timestamp=datetime.fromisoformat(event_dict["timestamp"]),
                data=event_dict["data"],
                metadata=event_dict.get("metadata", {}),
                relevance_score=event_dict.get("relevance_score", 1.0),
            )
            restored_events.append(event)

        # Restore stats
        stats = checkpoint.get("stats", {})
        self.cache_hits = stats.get("cache_hits", 0)
        self.cache_misses = stats.get("cache_misses", 0)

        logger.info(f"🔄 Restored {len(restored_events)} events from checkpoint")

        return restored_events

    def set_engagement_context(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        engagement_type: str = "consultation",
        case_id: Optional[str] = None,
    ) -> None:
        """
        Set engagement context for database persistence.

        Args:
            session_id: Session identifier
            user_id: User identifier
            organization_id: Organization identifier
            engagement_type: Type of engagement (consultation, analysis, etc.)
            case_id: Case identifier
        """
        self.session_id = session_id
        self.user_id = user_id
        self.organization_id = organization_id
        self.engagement_type = engagement_type
        self.case_id = case_id

        logger.info(f"📝 Engagement context set: {engagement_type} for case {case_id}")

    def complete_engagement(self, final_status: str = "completed") -> Dict[str, Any]:
        """
        Mark engagement as complete and prepare for persistence.

        Args:
            final_status: Final engagement status

        Returns:
            Completion event data
        """
        self.completed_at = datetime.now(timezone.utc)

        duration_ms = int(
            (self.completed_at - self.started_at).total_seconds() * 1000
        )

        completion_data = {
            "final_status": final_status,
            "duration_ms": duration_ms,
            "total_events": len(self.events),
            "trace_id": self.trace_id,
        }

        logger.info(f"✅ Engagement completed with status: {final_status}")

        return completion_data

    def set_final_analysis_text(self, analysis_text: str) -> None:
        """
        Store the final analysis text for offline evaluation.

        This text will be included in trace exports to enable evaluation judges
        to assess groundedness, relevance, and actionability.

        Args:
            analysis_text: The final analysis/report text (executive summary)
        """
        self.final_analysis_text = analysis_text

    async def persist_to_database(self) -> bool:
        """
        Persist the complete context stream using the injected persistence adapter.

        Returns:
            True if persistence succeeded, False otherwise
        """
        if not self.persistence_adapter:
            # Lazy default to file adapter for safety
            try:
                from src.core.persistence.adapters import FileAdapter

                self.persistence_adapter = FileAdapter()
            except Exception as e:
                logger.error(f"❌ No persistence adapter available: {e}")
                return False

        try:
            record = self.build_persistence_record()
            await self.persistence_adapter.persist([record])
            logger.info("✅ Context stream persisted via adapter")
            return True
        except Exception as e:
            logger.error(f"❌ Context stream persistence failed: {e}")
            return False

    def build_persistence_record(self) -> Dict[str, Any]:
        """
        Build a database-ready record for the current context stream.

        Returns:
            Database record dictionary
        """
        # Calculate summary metrics
        summary_metrics = self.calculate_summary_metrics()

        # Build the context stream JSONB object
        context_stream_data = {
            "events": [event.to_dict() for event in self.events],
            "summary": {
                "total_events": len(self.events),
                **summary_metrics,
                "performance_metrics": self.get_performance_metrics(),
                # OPERATION AWAKENING: Include final analysis for evaluation judges
                "final_report": self.final_analysis_text or "",
            },
        }

        # Calculate duration
        duration_ms = None
        if self.completed_at:
            duration_ms = int(
                (self.completed_at - self.started_at).total_seconds() * 1000
            )

        # Build the database record
        record = {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "organization_id": self.organization_id,
            "engagement_type": self.engagement_type,
            "case_id": self.case_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "duration_ms": duration_ms,
            "context_stream": context_stream_data,
            "total_tokens": summary_metrics["total_tokens"],
            "total_cost": summary_metrics["total_cost"],
            "final_status": summary_metrics["final_status"],
            "error_count": summary_metrics["error_count"],
            "models_used": summary_metrics["models_used"],
            "consultants_used": summary_metrics["consultants_used"],
            "tools_used": summary_metrics["tools_used"],
            "overall_quality_score": summary_metrics["overall_quality_score"],
            "contains_pii": summary_metrics["contains_pii"],
            "data_classification": (
                "confidential" if summary_metrics["contains_pii"] else "internal"
            ),
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        }
        return record

    def calculate_summary_metrics(self) -> Dict[str, Any]:
        """
        Calculate summary metrics for database storage.

        Returns:
            Dictionary of summary metrics
        """
        # Import here to avoid circular dependency
        try:
            from src.core.unified_context_stream import ContextEventType
        except ImportError:
            # Fallback if import fails
            return {
                "total_tokens": 0,
                "total_cost": 0.0,
                "final_status": "completed",
                "error_count": 0,
                "models_used": [],
                "consultants_used": [],
                "tools_used": [],
                "contains_pii": False,
                "overall_quality_score": 1.0,
            }

        # Extract models used from events
        models_used = set()
        consultants_used = set()
        tools_used = set()
        total_tokens = 0
        total_cost = 0.0
        error_count = 0

        for event in self.events:
            # Extract model information
            if "model" in event.data:
                models_used.add(event.data["model"])
            if "models_used" in event.data:
                if isinstance(event.data["models_used"], list):
                    models_used.update(event.data["models_used"])

            # Extract consultant information
            if "consultant" in event.data:
                consultants_used.add(event.data["consultant"])
            if "consultant_id" in event.data:
                consultants_used.add(event.data["consultant_id"])
            if "consultants_invoked" in event.data:
                if isinstance(event.data["consultants_invoked"], list):
                    consultants_used.update(event.data["consultants_invoked"])

            # Extract tool information
            if "tool" in event.data:
                tools_used.add(event.data["tool"])
            if "tools_used" in event.data:
                if isinstance(event.data["tools_used"], list):
                    tools_used.update(event.data["tools_used"])

            # Extract token and cost information
            if "tokens" in event.data:
                total_tokens += event.data["tokens"]
            if "total_tokens" in event.data:
                total_tokens += event.data["total_tokens"]
            if "cost" in event.data:
                total_cost += float(event.data["cost"])
            if "total_cost" in event.data:
                total_cost += float(event.data["total_cost"])

            # Count errors
            if event.event_type == ContextEventType.ERROR_OCCURRED:
                error_count += 1

        # Determine final status
        final_status = "completed"
        if error_count > 0:
            final_status = (
                "failed" if error_count > len(self.events) * 0.1 else "partial"
            )

        # Check for PII in the events
        contains_pii = any(
            any("[REDACTED_" in str(value) for value in event.data.values())
            for event in self.events
        )

        return {
            "total_tokens": total_tokens,
            "total_cost": round(total_cost, 6),
            "final_status": final_status,
            "error_count": error_count,
            "models_used": list(models_used),
            "consultants_used": list(consultants_used),
            "tools_used": list(tools_used),
            "contains_pii": contains_pii,
            "overall_quality_score": max(
                0.0, min(1.0, 1.0 - (error_count / max(1, len(self.events))))
            ),
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get context stream performance metrics.

        Returns:
            Performance metrics dictionary
        """
        total_requests = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0

        # Get relevant events (score >= 0.5)
        relevant_events = [e for e in self.events if e.relevance_score >= 0.5]

        return {
            "total_events": len(self.events),
            "relevant_events": len(relevant_events),
            "event_types": len(set(e.event_type for e in self.events)),
            "cache_hit_rate": cache_hit_rate,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "memory_events": len([e for e in self.events if not hasattr(e, 'compressed_version') or not e.compressed_version]),
            "compressed_events": len([e for e in self.events if hasattr(e, 'compressed_version') and e.compressed_version]),
            "last_event_time": (
                self.events[-1].timestamp.isoformat() if self.events else None
            ),
        }
