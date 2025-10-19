"""
Engagement Adapter Service - V2 API
===================================

Thin adapter layer between API routes and orchestration services.
Purpose: enable safe domain model extraction without changing route behavior.

This adapter delegates to existing orchestration logic while providing
a clean interface for future domain model migration.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional
from uuid import uuid4
from datetime import datetime

from src.core.stateful_pipeline_orchestrator import StatefulPipelineOrchestrator
from src.core.checkpoint_models import PipelineStage, StateCheckpoint
from src.core.unified_context_stream import get_unified_context_stream
from src.arbitration.models import ArbitrationResult
from src.services.persistence import DatabaseService, DatabaseOperationError

logger = logging.getLogger(__name__)


class EngagementAdapter:
    """Thin adapter for engagement orchestration operations"""

    def __init__(self):
        self.active_engagements: Dict[str, Dict[str, Any]] = {}

    async def start_engagement(
        self,
        user_query: str,
        user_id: Optional[str] = None,
        enhanced_context: Optional[Dict[str, Any]] = None,
        quality_requested: Optional[int] = None,
    ) -> str:
        """
        Start a new strategic analysis engagement

        Returns:
            trace_id: Unique identifier for the engagement
        """
        trace_id = str(uuid4())

        try:
            # Initialize orchestrator (delegates to existing logic)
            orchestrator = StatefulPipelineOrchestrator()

            # Store engagement state (mirrors existing pattern)
            self.active_engagements[trace_id] = {
                "trace_id": trace_id,
                "user_query": user_query,
                "user_id": user_id,
                "enhanced_context": enhanced_context or {},
                "quality_requested": quality_requested,
                "started_at": datetime.now(),
                "orchestrator": orchestrator,
                "status": "starting",
            }

            logger.info(f"🚀 Started engagement {trace_id}")
            return trace_id

        except Exception as e:
            logger.error(f"❌ Failed to start engagement: {e}")
            raise

    async def execute_engagement_pipeline(
        self,
        trace_id: str,
        orchestrator: StatefulPipelineOrchestrator,
        user_query: str,
        enhanced_context: Dict[str, Any],
    ) -> None:
        """Execute the engagement pipeline (delegates to orchestrator)"""
        try:
            await orchestrator.run_full_pipeline(
                user_query=user_query,
                enhanced_context=enhanced_context,
            )

            # Update engagement status
            if trace_id in self.active_engagements:
                self.active_engagements[trace_id]["status"] = "completed"

        except Exception as e:
            logger.error(f"❌ Pipeline execution failed for {trace_id}: {e}")
            if trace_id in self.active_engagements:
                self.active_engagements[trace_id]["status"] = "failed"
                self.active_engagements[trace_id]["error"] = str(e)
            raise

    async def get_engagement_status(self, trace_id: str) -> Dict[str, Any]:
        """Get current engagement status (delegates to orchestrator)"""
        try:
            # Check local storage first
            if trace_id in self.active_engagements:
                engagement = self.active_engagements[trace_id]
                orchestrator = engagement.get("orchestrator")

                if orchestrator:
                    # Delegate to orchestrator for current state
                    current_checkpoint = orchestrator.get_current_checkpoint()

                    # Handle enum-as-value compatibility
                    stage_obj = current_checkpoint.current_stage if current_checkpoint else None
                    if isinstance(stage_obj, str):
                        current_stage_value = stage_obj
                    else:
                        try:
                            current_stage_value = stage_obj.value  # type: ignore[attr-defined]
                        except Exception:
                            current_stage_value = "unknown"

                    return {
                        "trace_id": trace_id,
                        "status": engagement.get("status", "unknown"),
                        "current_stage": current_stage_value,
                        "stage_number": current_checkpoint.stage_number if current_checkpoint else 0,
                        "total_stages": 8,  # Hard-coded for V5.3 pipeline
                        "progress_percentage": (current_checkpoint.stage_number / 8) * 100 if current_checkpoint else 0.0,
                        "stage_description": self._get_stage_description(stage_obj),
                        "is_completed": engagement.get("status") == "completed",
                        "error": engagement.get("error"),
                    }

            # Fallback: check database (delegates to existing persistence logic)
            db_service = DatabaseService()
            engagement_data = await db_service.get_engagement_status(trace_id)
            return engagement_data

        except Exception as e:
            logger.error(f"❌ Failed to get engagement status for {trace_id}: {e}")
            raise

    async def get_engagement_report(self, trace_id: str) -> Dict[str, Any]:
        """Get final engagement report (delegates to orchestrator)"""
        try:
            # Delegate to orchestrator for final report
            if trace_id in self.active_engagements:
                orchestrator = self.active_engagements[trace_id].get("orchestrator")
                if orchestrator:
                    final_report = await orchestrator.get_final_report()

                    return {
                        "trace_id": trace_id,
                        "report": final_report,
                        "generated_at": datetime.now(),
                        "markdown_content": self._convert_to_markdown(final_report),
                        "selected_consultants": self._extract_consultant_info(final_report),
                        "consultant_selection_methodology": "METIS V5.3 Selection Engine",
                    }

            # Fallback: database lookup (delegates to existing persistence)
            db_service = DatabaseService()
            report_data = await db_service.get_engagement_report(trace_id)
            return report_data

        except Exception as e:
            logger.error(f"❌ Failed to get engagement report for {trace_id}: {e}")
            raise

    def _get_stage_description(self, stage: Optional[PipelineStage | str]) -> str:
        """Get human-readable stage description"""
        if not stage:
            return "Initializing..."

        # Normalize from string to PipelineStage when possible
        if isinstance(stage, str):
            try:
                stage = PipelineStage(stage)
            except Exception:
                return f"Processing {stage}"

        stage_descriptions = {
            PipelineStage.QUERY_ENHANCEMENT: "Enhancing query context",
            PipelineStage.PROBLEM_STRUCTURING: "Structuring problem analysis",
            PipelineStage.CONSULTANT_SELECTION: "Selecting expert consultants",
            PipelineStage.PARALLEL_ANALYSIS: "Running parallel analysis",
            PipelineStage.DEVILS_ADVOCATE: "Critical review and validation",
            PipelineStage.SENIOR_ADVISOR: "Senior advisor synthesis",
            PipelineStage.ARBITRATION: "Final arbitration and recommendations",
            PipelineStage.CALIBRATION: "Calibrating confidence and delivery",
        }

        return stage_descriptions.get(stage, f"Processing {stage.value}")

    def _convert_to_markdown(self, report: Dict[str, Any]) -> str:
        """Convert report to markdown format (placeholder implementation)"""
        # This would delegate to existing markdown conversion logic
        return f"# Strategic Analysis Report\n\n{report.get('summary', 'Report generated')}"

    def _extract_consultant_info(self, report: Dict[str, Any]) -> list:
        """Extract consultant information from report (placeholder implementation)"""
        # This would delegate to existing consultant extraction logic
        return []