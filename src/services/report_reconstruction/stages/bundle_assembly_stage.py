"""
Bundle Assembly Stage
====================

Assembles the final report bundle from all extracted data.

Responsibility:
- Assemble bundle from all state data
- Extract and sanitize executive summary
- Enrich recommendations with dissent signals
- Build key_decisions structure
- Attach consultant memos from events
- Add compatibility aliases for frontend
- Apply sanitization
- Generate ETag

Complexity: CC<8 (Final assembly with moderate branching)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any, Dict, List, Optional

from src.services.orchestration_infra.glass_box_sanitization_service import (
    GlassBoxSanitizationService,
)
from src.services.report_reconstruction.reconstruction_state import ReconstructionState
from src.services.report_reconstruction.reconstruction_stage import (
    ReconstructionStage,
    ReconstructionError,
)

logger = logging.getLogger(__name__)


class BundleAssemblyStage(ReconstructionStage):
    """
    Stage 6: Bundle Assembly

    Assembles the final report bundle from all extracted and processed data.
    Applies sanitization and generates ETag.
    """

    def __init__(self, sanitizer: Optional[GlassBoxSanitizationService] = None):
        """
        Initialize bundle assembly stage.

        Args:
            sanitizer: Glass-box sanitization service for content sanitization
        """
        self.sanitizer = sanitizer or GlassBoxSanitizationService()

    @property
    def name(self) -> str:
        return "bundle_assembly"

    @property
    def description(self) -> str:
        return "Assemble final bundle, sanitize, and generate ETag"

    def process(self, state: ReconstructionState) -> ReconstructionState:
        """
        Assemble final report bundle from all state data.

        Args:
            state: Current reconstruction state with all data extracted

        Returns:
            Updated state with final bundle assembled
        """
        try:
            # Extract and process executive summary
            executive_summary = self._extract_executive_summary(state.senior_advisor)
            executive_summary_meta = self._sanitize_text(executive_summary, "executive_summary")

            # Extract recommendations
            recommendations = self._extract_recommendations(state.senior_advisor)

            # Enrich recommendations with dissent signals
            recommendations = self._enrich_recommendations_with_dissent(
                recommendations, state.dissent_signals
            )

            # Build key decisions structure
            key_decisions = self._build_key_decisions(
                executive_summary, recommendations, state.quality_metrics
            )

            # Attach consultant memos from events
            consultant_analyses = self._attach_consultant_memos(
                state.consultant_analyses, state.events
            )

            # Extract devils advocate transcript
            da_transcript = self._extract_da_transcript(state.devils_advocate)

            # Assemble base bundle
            bundle = {
                "trace_id": state.trace_id,
                "query": state.query_text,
                "executive_summary": executive_summary,
                "executive_summary_meta": executive_summary_meta,
                "strategic_recommendations": recommendations,
                "senior_advisor": {
                    "synthesis_markdown": state.senior_advisor.get("final_markdown_report")
                    or state.senior_advisor.get("synthesis_markdown"),
                    "strategic_recommendations": recommendations,
                },
                "key_decisions": key_decisions,
                "quality_metrics": state.quality_metrics,
                "consultant_analyses": consultant_analyses,
                "devils_advocate_transcript": da_transcript,
                "devils_advocate": state.devils_advocate if isinstance(state.devils_advocate, dict) else None,
                "evidence_trail": state.evidence_trail or [],
                "metadata": {
                    "stages_available": ["senior_advisor", "devils_advocate"],  # Simplified
                },
                "human_interactions": state.human_interactions,
                "research_provider_events": state.research_providers,
                "quality_ribbon": state.quality_ribbon,
                "plan_overview": state.plan_overview,
                "enhancement_research_answers": state.enhancement_research_answers,
                "dissent_signals": state.dissent_signals,
            }

            # Add Phase 1 outputs (problem_structuring, socratic)
            bundle = self._add_phase1_outputs(bundle, state)

            # Add compatibility aliases
            bundle = self._add_compatibility_aliases(bundle, da_transcript)

            return state.with_bundle(bundle)

        except Exception as e:
            raise ReconstructionError(
                self.name,
                f"Failed to assemble bundle for trace_id={state.trace_id}",
                cause=e,
            )

    def _extract_executive_summary(self, senior_advisor: Dict[str, Any]) -> str:
        """
        Extract executive summary from senior advisor output.

        Handles case where full report is duplicated as executive summary.
        """
        executive_summary = senior_advisor.get("executive_summary") or senior_advisor.get(
            "final_markdown_report", "Analysis completed."
        )

        senior_markdown = senior_advisor.get("final_markdown_report") or senior_advisor.get("synthesis_markdown")

        # If exec summary looks like the whole report, extract true summary from markdown
        try:
            if isinstance(senior_markdown, str) and isinstance(executive_summary, str):
                if "Strategic Recommendations" in executive_summary or "Senior Advisor Analysis" in executive_summary:
                    m = re.search(
                        r"(?im)^##\s+Executive Summary\s*\n([\s\S]*?)(?:^##\s+|\Z)",
                        senior_markdown,
                    )
                    if m:
                        executive_summary = m.group(1).strip()
        except Exception:
            pass

        return executive_summary

    def _extract_recommendations(self, senior_advisor: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract strategic recommendations from senior advisor."""
        return senior_advisor.get("strategic_recommendations") or senior_advisor.get("final_recommendations", [])

    def _enrich_recommendations_with_dissent(
        self, recommendations: List[Dict[str, Any]], dissent_signals: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Enrich recommendations with dissent summary."""
        rec_dissent_count = len(dissent_signals)
        rec_dissent_severity = "none"

        try:
            if dissent_signals:
                max_sev = max(
                    (self._dissent_severity_rank(sig.get("severity", "")) for sig in dissent_signals),
                    default=0,
                )
                rec_dissent_severity = "high" if max_sev == 3 else "medium" if max_sev == 2 else "low"
        except Exception:
            pass

        try:
            if isinstance(recommendations, list):
                for rec in recommendations:
                    if isinstance(rec, dict):
                        rec.setdefault("dissent_count", rec_dissent_count)
                        rec.setdefault("dissent_severity", rec_dissent_severity)
        except Exception:
            pass

        return recommendations

    def _build_key_decisions(
        self, executive_summary: str, recommendations: List[Dict[str, Any]], quality_metrics: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Build key_decisions structure for frontend compatibility."""
        critical_path_items = []

        if isinstance(recommendations, list) and len(recommendations) > 0:
            # Take up to 3 top recommendations as critical path
            for rec in recommendations[:3]:
                if isinstance(rec, dict):
                    rec_text = rec.get("recommendation") or rec.get("title") or str(rec)
                    # Truncate to ~50 chars for concise display
                    if len(rec_text) > 60:
                        rec_text = rec_text[:57] + "..."
                    critical_path_items.append(rec_text)

        key_decisions = []
        if critical_path_items:
            key_decisions.append(
                {
                    "decision": executive_summary[:200] if len(executive_summary) > 200 else executive_summary,
                    "confidence_level": quality_metrics.get("overall_confidence", 0.85),
                    "assumptions": critical_path_items,
                }
            )

        return key_decisions

    def _attach_consultant_memos(
        self, consultant_analyses: List[Dict[str, Any]], events: List[Any]
    ) -> List[Dict[str, Any]]:
        """Attach consultant memos from UCS events to consultant analyses."""
        try:
            # Build memo mapping
            memo_map: Dict[str, str] = {}
            for ev in events:
                et = getattr(ev, "event_type", None)
                et_val = getattr(et, "value", et) if et else None

                if et_val == "consultant_memo_produced":
                    data = getattr(ev, "data", {}) or {}
                    cid = data.get("consultant_id") or data.get("id") or data.get("name")
                    memo_text = (
                        data.get("memo")
                        or data.get("content")
                        or data.get("text")
                        or data.get("raw_memo")
                        or data.get("body")
                    )
                    if cid and isinstance(memo_text, str):
                        memo_map[str(cid)] = memo_text

            # Inject memos into consultant_analyses
            for c in consultant_analyses:
                try:
                    cid = c.get("consultant_id") or c.get("id")
                    if cid and str(cid) in memo_map:
                        c["memo"] = memo_map[str(cid)]
                except Exception:
                    continue

        except Exception as e:
            logger.debug(f"Consultant memo extraction skipped: {e}")

        return consultant_analyses

    def _extract_da_transcript(self, devils_advocate: Dict[str, Any]) -> str:
        """Extract devils advocate transcript."""
        return devils_advocate.get("full_transcript") or devils_advocate.get("da_transcript") or ""

    def _add_phase1_outputs(self, bundle: Dict[str, Any], state: ReconstructionState) -> Dict[str, Any]:
        """Add Phase 1 outputs (problem_structuring, socratic) to bundle."""
        if state.problem_structuring:
            bundle["problem_structuring_output"] = state.problem_structuring
            # Compatibility for current frontend (reads top-level mece_framework)
            if "mece_framework" in state.problem_structuring and "mece_framework" not in bundle:
                bundle["mece_framework"] = state.problem_structuring["mece_framework"]

        if state.socratic_questions:
            bundle["socratic_output"] = state.socratic_questions
            # Compatibility alias used by frontend
            bundle["socratic_results"] = state.socratic_questions

        return bundle

    def _add_compatibility_aliases(self, bundle: Dict[str, Any], da_transcript: str) -> Dict[str, Any]:
        """Add compatibility aliases for older UI."""
        if da_transcript:
            bundle["da_transcript"] = da_transcript

        return bundle

    def _sanitize_text(self, text: str, kind: str) -> Dict[str, Any]:
        """Sanitize text using glass-box sanitization service."""
        return self.sanitizer.sanitize_pipeline_content(text or "", kind)

    def _dissent_severity_rank(self, severity: str) -> int:
        """Rank dissent severity for comparison."""
        m = (severity or "").lower()
        if "high" in m:
            return 3
        if "medium" in m:
            return 2
        if "low" in m:
            return 1
        return 0

    def _hash_etag(self, payload: Dict[str, Any]) -> str:
        """Generate ETag hash for bundle."""
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]
