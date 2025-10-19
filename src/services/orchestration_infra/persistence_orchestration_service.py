"""
Persistence Orchestration Service - Operation Scalpel V2
========================================================

SEAMS-FIRST PATTERN: Phase 1.3 (MOVE)

This service provides a clean, self-contained interface for persistence operations.
Logic has been moved from orchestrator internal helpers to this service.

CURRENT STATE: Self-contained service with all persistence logic
"""

from typing import Dict, Any, Optional
from uuid import UUID
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


class PersistenceOrchestrationService:
    """
    Service for managing persistence operations during pipeline execution.

    Phase 1.3 (MOVE): Self-contained - no orchestrator dependency
    """

    def __init__(self, db_service):
        """
        Initialize persistence service.

        Args:
            db_service: DatabaseService instance for database operations
        """
        self.db_service = db_service
        logger.info("🔗 PersistenceOrchestrationService initialized (Phase 1.3: MOVE - Self-contained)")

    async def persist_engagement(
        self,
        trace_id: UUID,
        user_id: Optional[str],
        accumulated_context: Dict[str, Any],
        processing_time: float
    ) -> None:
        """
        Persist engagement report to database.

        Phase 1.3: Contains logic directly (moved from orchestrator)

        Args:
            trace_id: Unique identifier for this pipeline execution
            user_id: User who initiated the analysis
            accumulated_context: Complete pipeline context
            processing_time: Total processing time in seconds
        """
        try:
            # Critical pre-insert logging
            print(f"🔍 ENGAGEMENT PERSISTENCE: About to persist engagement to database")
            print(f"🔍 ENGAGEMENT PERSISTENCE: trace_id={trace_id}")
            print(f"🔍 ENGAGEMENT PERSISTENCE: initial_query={accumulated_context.get('initial_query', 'NO_QUERY')[:100]}")
            print(f"🔍 ENGAGEMENT PERSISTENCE: processing_time={processing_time:.2f}s")
            logger.info(f"🔍 ENGAGEMENT PERSISTENCE: Calling db_service.store_engagement_report()")

            # CRITICAL FIX: Make accumulated_context JSON-safe (recursive) + truncate large fields
            def make_json_safe(obj, max_string_length=5000):
                """Recursively convert non-JSON-serializable objects and truncate large strings"""
                if isinstance(obj, dict):
                    return {k: make_json_safe(v, max_string_length) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    # Limit list size to prevent massive payloads
                    if len(obj) > 100:
                        return [make_json_safe(item, max_string_length) for item in obj[:100]] + [f"... {len(obj) - 100} more items truncated"]
                    return [make_json_safe(item, max_string_length) for item in obj]
                elif isinstance(obj, str):
                    # Truncate very long strings to prevent timeout
                    if len(obj) > max_string_length:
                        return obj[:max_string_length] + f"... [truncated {len(obj) - max_string_length} chars]"
                    return obj
                elif hasattr(obj, '__dict__') or hasattr(obj, 'value'):
                    # Handle Pydantic models and enums
                    return str(obj)[:max_string_length]
                else:
                    try:
                        json.dumps(obj)
                        return obj
                    except (TypeError, ValueError):
                        return str(obj)[:max_string_length]

            # OPERATION POWER ON: Create MINIMAL summary for engagement persistence
            # Full context is already in cognitive_states table - no need to duplicate 15MB payload
            minimal_summary = {
                "initial_query": str(accumulated_context.get("initial_query", "Unknown query"))[:500],
                "stages_completed": list(accumulated_context.get("stages_completed", []))[:10] if isinstance(accumulated_context.get("stages_completed"), list) else [],
                "final_status": "completed",
                "note": "Full cognitive artifacts stored in cognitive_states table"
            }

            # Extract minimal senior advisor summary
            senior_advisor_data = accumulated_context.get("senior_advisor", {})
            minimal_senior_advisor = {
                "has_recommendations": bool(senior_advisor_data.get("final_recommendations")),
                "recommendation_count": len(senior_advisor_data.get("final_recommendations", [])) if isinstance(senior_advisor_data.get("final_recommendations"), list) else 0,
                "confidence": senior_advisor_data.get("confidence_assessment", 0.0),
                "note": "Full report stored in cognitive_states table"
            }

            payload_size_estimate = len(json.dumps(minimal_summary, default=str)) + len(json.dumps(minimal_senior_advisor, default=str))
            print(f"🔍 ENGAGEMENT PERSISTENCE: Minimal summary size: {payload_size_estimate:,} bytes")

            self.db_service.store_engagement_report(
                trace_id=str(trace_id),
                user_id=user_id,
                user_query=minimal_summary["initial_query"],
                processing_time_seconds=processing_time,
                final_report_contract=minimal_senior_advisor,
                accumulated_context=minimal_summary,
                metadata={
                    "pipeline_version": "v5.3_golden_master",
                    "total_stages": 10,
                    "operation": "power_on",
                }
            )

            print(f"✅ ENGAGEMENT PERSISTENCE: Successfully persisted engagement to database for trace_id: {trace_id}")
            logger.info(f"✅ ENGAGEMENT PERSISTENCE: Engagement stored successfully")

        except Exception as e:
            print(f"❌ ENGAGEMENT PERSISTENCE: Failed: {e}")
            logger.error(f"❌ ENGAGEMENT PERSISTENCE: {e}", exc_info=True)

    async def persist_devils_advocate(
        self,
        trace_id: UUID,
        result: Dict[str, Any],
        context: Dict[str, Any]
    ) -> None:
        """
        Persist Devils Advocate cognitive state to database.

        Phase 1.3: Contains logic directly (moved from orchestrator)

        Args:
            trace_id: Unique identifier for this pipeline execution
            result: Devils Advocate analysis results
            context: Pipeline context
        """
        try:
            print(f"🔍 DEVILS ADVOCATE: About to persist to cognitive_states table")
            print(f"🔍 DEVILS ADVOCATE: trace_id={trace_id}")

            cognitive_artifact = {
                "trace_id": str(trace_id),
                "stage_name": "devils_advocate",
                "stage_type": "critique",
                "cognitive_output": result,
                "created_at": datetime.now().isoformat(),
                "processing_time_ms": None,
                "stage_complete": True
            }

            self.db_service.insert("cognitive_states", cognitive_artifact)
            print(f"✅ DEVILS ADVOCATE: Successfully persisted cognitive state for trace_id: {trace_id}")

        except Exception as e:
            print(f"❌ DEVILS ADVOCATE: Failed to persist cognitive state: {e}")
            logger.error(f"❌ DEVILS ADVOCATE: Persistence failed: {e}", exc_info=True)

    async def persist_senior_advisor(
        self,
        trace_id: UUID,
        result: Dict[str, Any],
        context: Dict[str, Any],
        path: str = "success"
    ) -> None:
        """
        Persist Senior Advisor cognitive state to database.

        Handles 3 execution paths:
        - "success": Two-brain analysis completed successfully
        - "fallback1": Fallback to synthesize_final_advice
        - "fallback2": Fallback after exception

        Phase 1.3: Contains logic directly (moved from orchestrator)

        Args:
            trace_id: Unique identifier for this pipeline execution
            result: Senior Advisor synthesis results
            context: Pipeline context
            path: Execution path identifier
        """
        try:
            print(f"🔍 SENIOR ADVISOR: About to persist to cognitive_states table ({path})")
            print(f"🔍 SENIOR ADVISOR: trace_id={trace_id}")

            if path == "success":
                # Two-brain success path
                final_markdown_report = result.get("final_markdown_report", "")
                cognitive_artifact = {
                    "trace_id": str(trace_id),
                    "stage_name": "senior_advisor",
                    "stage_type": "synthesis",
                    "cognitive_output": {
                        "final_markdown_report": final_markdown_report if final_markdown_report else "No report",
                        "confidence": result.get("raw_analytical_dossier", {}).get("confidence", 0.8),
                        "two_brain_completed": True
                    },
                    "created_at": datetime.now().isoformat(),
                    "processing_time_ms": result.get("processing_time", 0),
                    "stage_complete": True
                }
            elif path == "fallback1":
                # Fallback path 1
                cognitive_artifact = {
                    "trace_id": str(trace_id),
                    "stage_name": "senior_advisor",
                    "stage_type": "synthesis",
                    "cognitive_output": {
                        "recommendations": result.get("final_recommendations", [])[:3],
                        "confidence": result.get("confidence_assessment", 0.0),
                        "fallback_used": True
                    },
                    "created_at": datetime.now().isoformat(),
                    "processing_time_ms": None,
                    "stage_complete": True
                }
            else:  # fallback2
                # Exception fallback path
                cognitive_artifact = {
                    "trace_id": str(trace_id),
                    "stage_name": "senior_advisor",
                    "stage_type": "synthesis",
                    "cognitive_output": {
                        "recommendations": result.get("final_recommendations", [])[:3],
                        "confidence": result.get("confidence_assessment", 0.0),
                        "error": result.get("error", "Unknown error"),
                        "exception_fallback": True
                    },
                    "created_at": datetime.now().isoformat(),
                    "processing_time_ms": None,
                    "stage_complete": True
                }

            self.db_service.insert("cognitive_states", cognitive_artifact)
            print(f"✅ SENIOR ADVISOR: Successfully persisted cognitive state ({path}) for trace_id: {trace_id}")

        except Exception as e:
            print(f"❌ SENIOR ADVISOR: Failed to persist cognitive state: {e}")
            logger.error(f"❌ SENIOR ADVISOR: Persistence failed: {e}", exc_info=True)
