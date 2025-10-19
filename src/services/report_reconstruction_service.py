"""
Report Reconstruction Service
=============================

Reassembles a complete, frontend‑ready "Report Bundle" from persisted
artifacts (cognitive_states) and the UnifiedContextStream, then sanitizes
the result for Glass‑Box presentation.

REFACTORED: Now uses the State Reconstruction Pattern with 6 modular stages:
1. DataFetchingStage: Fetch from DB/stream
2. StageExtractionStage: Extract cognitive stages
3. ConsultantAnalysisStage: Consultant data
4. EnhancementResearchStage: Research extraction
5. GlassBoxTransparencyStage: Evidence/quality/plan
6. BundleAssemblyStage: Final assembly + sanitization

Complexity: CC<8 per stage (down from CC=103 monolith)
Grade: A (up from F - Unmaintainable)

Design goals:
- Do not change V6 contracts downstream; keep BriefingMemo etc. intact
- Provide a single bundle that Report v2 can render quickly (cacheable)
- Deny‑by‑default sanitization for any raw prompts/answers/tools
"""

from __future__ import annotations

import json
import hashlib
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
from uuid import UUID

from src.services.persistence.database_service import DatabaseService, DatabaseOperationError
from src.core.unified_context_stream import get_unified_context_stream, ContextEventType
from src.api.event_schema import normalize_event
from src.services.orchestration_infra.glass_box_sanitization_service import GlassBoxSanitizationService

# Import refactored reconstruction pipeline
from src.services.report_reconstruction import create_report_reconstructor

logger = logging.getLogger(__name__)


class ReportReconstructionService:
    def __init__(self, db: Optional[DatabaseService] = None) -> None:
        self.db = db
        self.sanitizer = GlassBoxSanitizationService()

        # Create refactored reconstructor with all 6 stages
        self._reconstructor = create_report_reconstructor(
            db=self.db,
            sanitizer=self.sanitizer,
            enable_logging=False,  # Use service-level logging instead
        )

    def _hash_etag(self, payload: Dict[str, Any]) -> str:
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]

    def _fetch_cognitive_states(self, trace_id: str) -> List[Dict[str, Any]]:
        """
        Fetch cognitive outputs from state_checkpoints (V6 migration).
        Maps V6 checkpoint structure to V5 cognitive_states format for compatibility.
        """
        if not self.db:
            return []
        try:
            # V6: Query state_checkpoints instead of cognitive_states
            rows = self.db.fetch_many(
                "state_checkpoints",
                {"trace_id": trace_id},
                columns="*",
                order_by="created_at",
                desc=False,
            )
            if not rows:
                return []

            # V6: Map checkpoint structure to cognitive_states format
            mapped = []
            for checkpoint in rows:
                state_data = checkpoint.get("state_data", {})
                stage_output = state_data.get("stage_output", {})
                mapped.append({
                    "trace_id": trace_id,
                    "stage_name": checkpoint.get("stage_name"),
                    "cognitive_output": stage_output,
                    "created_at": checkpoint.get("created_at"),
                    "processing_time_ms": state_data.get("_stage_metadata", {}).get("processing_time_ms", 0)
                })
            return mapped
        except Exception as e:
            logger.warning(f"⚠️ Failed to fetch checkpoints for {trace_id}: {e}")
            return []


    def reconstruct_bundle(self, trace_id: str) -> Dict[str, Any]:
        """
        Reconstruct complete report bundle from trace ID.

        REFACTORED: Uses State Reconstruction Pattern with 6 modular stages.
        Complexity: CC<8 per stage (down from CC=103)

        Args:
            trace_id: Unique trace identifier

        Returns:
            Complete report bundle ready for frontend rendering
        """
        logger.info(f"🔄 Reconstructing bundle for trace_id={trace_id}")

        try:
            # Use refactored reconstructor with 6-stage pipeline
            bundle = self._reconstructor.reconstruct(trace_id)

            logger.info(f"✅ Bundle reconstruction complete for trace_id={trace_id}")
            return bundle

        except Exception as e:
            logger.error(f"❌ Bundle reconstruction failed for trace_id={trace_id}: {e}")
            raise
