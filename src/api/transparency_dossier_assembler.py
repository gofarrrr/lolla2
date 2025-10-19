#!/usr/bin/env python3
"""
Transparency Dossier Assembler - Phase 4 Glass-Box Implementation
==================================================================

Assembles comprehensive transparency dossiers from UnifiedContextStream events
for the Glass-Box Transparency API. Provides audit-ready summaries without
exposing proprietary IP.

Key Features:
- Reads UnifiedContextStream events for a given trace_id
- Generates structured timeline with decision points
- Calculates selection metrics and quality scores
- Summarizes chunking and processing stages
- Provides per-consultant performance data
- Glass-box compliant (no raw prompts/responses)
"""

import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

try:
    from src.core.unified_context_stream import (
        UnifiedContextStream,
        ContextEvent,
        ContextEventType,
    )
except ImportError:
    UnifiedContextStream = None
    ContextEvent = None
    ContextEventType = None

import logging

logger = logging.getLogger(__name__)


class DossierSection(Enum):
    """Sections available in transparency dossier"""

    TIMELINE = "timeline"
    SELECTION_METRICS = "selection_metrics"
    CHUNKING_SUMMARY = "chunking_summary"
    CONSULTANT_PERFORMANCE = "consultant_performance"
    QUALITY_ANALYSIS = "quality_analysis"
    PROCESSING_STAGES = "processing_stages"
    ERROR_ANALYSIS = "error_analysis"


@dataclass
class TimelineEvent:
    """Timeline event for dossier display"""

    timestamp: str
    event_type: str
    description: str
    stage: str
    duration_ms: Optional[int] = None
    confidence: Optional[float] = None
    fingerprint: Optional[str] = None


@dataclass
class SelectionMetrics:
    """Selection decision metrics"""

    total_selections: int
    average_confidence: float
    consultant_distribution: Dict[str, int]
    model_distribution: Dict[str, int]
    selection_rationales: List[str]
    risk_factors_identified: List[str]
    success_factors_identified: List[str]


@dataclass
class ChunkingSummary:
    """Content chunking and processing summary"""

    total_chunks_processed: int
    chunking_strategy_used: str
    chunk_size_distribution: Dict[str, int]
    processing_time_per_chunk_ms: List[int]
    content_fingerprints: List[str]
    overlap_strategy: str


@dataclass
class ConsultantPerformance:
    """Per-consultant performance metrics"""

    consultant_name: str
    total_invocations: int
    average_processing_time_ms: float
    confidence_scores: List[float]
    success_rate: float
    specializations_used: List[str]
    contribution_quality: float


@dataclass
class QualityAnalysis:
    """Quality analysis summary"""

    overall_quality_score: float
    quality_dimensions: Dict[str, float]
    improvement_suggestions: List[str]
    bias_checks_performed: List[str]
    validation_steps_completed: List[str]


@dataclass
class ProcessingStage:
    """Processing stage summary"""

    stage_name: str
    start_time: str
    end_time: str
    duration_ms: int
    events_count: int
    success: bool
    key_outputs: List[str]
    error_count: int


@dataclass
class TransparencyDossier:
    """Complete transparency dossier"""

    trace_id: str
    generated_at: str
    session_duration_ms: int
    total_events: int
    timeline: List[TimelineEvent]
    selection_metrics: SelectionMetrics
    chunking_summary: ChunkingSummary
    consultant_performance: List[ConsultantPerformance]
    quality_analysis: QualityAnalysis
    processing_stages: List[ProcessingStage]
    error_summary: Dict[str, Any]
    glass_box_compliance: Dict[str, bool]


class TransparencyDossierAssembler:
    """
    Assembles transparency dossiers from UnifiedContextStream events

    Reads sanitized events from the context stream and generates comprehensive
    audit trails while maintaining Glass-Box compliance (no IP exposure).
    """

    def __init__(self):
        """Initialize the transparency dossier assembler"""
        self.logger = logging.getLogger(__name__)
        print("🔍 Transparency Dossier Assembler initialized")

    async def assemble_dossier(
        self, context_stream: UnifiedContextStream
    ) -> TransparencyDossier:
        """
        Assemble complete transparency dossier from context stream

        Args:
            context_stream: UnifiedContextStream instance with events

        Returns:
            Complete TransparencyDossier with all sections
        """
        try:
            print(
                f"📋 Assembling transparency dossier for trace_id: {context_stream.trace_id}"
            )

            # Get all events from the stream
            all_events = context_stream.get_events()

            if not all_events:
                print("⚠️ No events found in context stream")
                return self._create_empty_dossier(context_stream.trace_id)

            # Generate each section of the dossier
            timeline = await self._generate_timeline(all_events)
            selection_metrics = await self._generate_selection_metrics(all_events)
            chunking_summary = await self._generate_chunking_summary(all_events)
            consultant_performance = await self._generate_consultant_performance(
                all_events
            )
            quality_analysis = await self._generate_quality_analysis(all_events)
            processing_stages = await self._generate_processing_stages(all_events)
            error_summary = await self._generate_error_summary(all_events)

            # Calculate session duration
            session_duration_ms = self._calculate_session_duration(all_events)

            # Create complete dossier
            dossier = TransparencyDossier(
                trace_id=context_stream.trace_id,
                generated_at=datetime.now(timezone.utc).isoformat(),
                session_duration_ms=session_duration_ms,
                total_events=len(all_events),
                timeline=timeline,
                selection_metrics=selection_metrics,
                chunking_summary=chunking_summary,
                consultant_performance=consultant_performance,
                quality_analysis=quality_analysis,
                processing_stages=processing_stages,
                error_summary=error_summary,
                glass_box_compliance=self._verify_glass_box_compliance(all_events),
            )

            print(
                f"✅ Transparency dossier assembled: {len(timeline)} timeline events, {len(consultant_performance)} consultants"
            )
            return dossier

        except Exception as e:
            print(f"❌ Error assembling transparency dossier: {e}")
            return self._create_error_dossier(context_stream.trace_id, str(e))

    async def _generate_timeline(
        self, events: List[ContextEvent]
    ) -> List[TimelineEvent]:
        """Generate chronological timeline of key events"""
        timeline_events = []

        # Key event types for timeline
        timeline_event_types = {
            ContextEventType.ENGAGEMENT_STARTED: "Session Started",
            ContextEventType.QUERY_RECEIVED: "Query Received",
            ContextEventType.CONSULTANT_SELECTION_COMPLETE: "Consultant Selection",
            ContextEventType.RESEARCH_QUERY: "Research Query",
            ContextEventType.RESEARCH_RESULT: "Research Result",
            ContextEventType.LLM_PROVIDER_REQUEST: "LLM Request",
            ContextEventType.LLM_PROVIDER_RESPONSE: "LLM Response",
            ContextEventType.DEVILS_ADVOCATE_COMPLETE: "Devils Advocate Analysis",
            ContextEventType.SENIOR_ADVISOR_COMPLETE: "Senior Advisor Review",
            ContextEventType.SYNTHESIS_CREATED: "Synthesis Generated",
            ContextEventType.FINAL_REPORT_GENERATED: "Final Report",
            ContextEventType.ENGAGEMENT_COMPLETED: "Session Completed",
            ContextEventType.ERROR_OCCURRED: "Error Occurred",
        }

        for event in events:
            if event.event_type in timeline_event_types:
                # Extract safe information for timeline
                description = timeline_event_types[event.event_type]

                # Add context-specific details (sanitized)
                if event.event_type == ContextEventType.CONSULTANT_SELECTION_COMPLETE:
                    consultant_count = event.data.get("consultant_count", 0)
                    confidence = event.data.get("total_confidence", 0)
                    description = f"Selected {consultant_count} consultants (confidence: {confidence:.1%})"

                elif event.event_type == ContextEventType.RESEARCH_QUERY:
                    query_fingerprint = event.data.get("query_fingerprint", "unknown")
                    description = (
                        f"Research query executed (ID: {query_fingerprint[:8]})"
                    )

                elif event.event_type == ContextEventType.LLM_PROVIDER_REQUEST:
                    model = event.data.get("model", "unknown")
                    provider = event.data.get("provider", "unknown")
                    description = f"LLM request to {provider} ({model})"

                elif event.event_type == ContextEventType.ERROR_OCCURRED:
                    error_type = event.data.get("error_type", "unknown")
                    description = f"Error: {error_type}"

                # Determine processing stage
                stage = self._determine_stage_from_event(event)

                # Extract timing and confidence data
                duration_ms = event.data.get("duration_ms") or event.data.get(
                    "processing_time_ms"
                )
                confidence = event.data.get("confidence") or event.data.get(
                    "confidence_score"
                )
                fingerprint = event.data.get("content_fingerprint") or event.data.get(
                    "fingerprint"
                )

                timeline_event = TimelineEvent(
                    timestamp=event.timestamp.isoformat(),
                    event_type=event.event_type.value,
                    description=description,
                    stage=stage,
                    duration_ms=duration_ms,
                    confidence=confidence,
                    fingerprint=fingerprint,
                )

                timeline_events.append(timeline_event)

        # Sort by timestamp
        timeline_events.sort(key=lambda x: x.timestamp)

        print(f"   Generated timeline with {len(timeline_events)} events")
        return timeline_events

    async def _generate_selection_metrics(
        self, events: List[ContextEvent]
    ) -> SelectionMetrics:
        """Generate consultant and model selection metrics"""
        selection_events = [
            e
            for e in events
            if e.event_type == ContextEventType.CONSULTANT_SELECTION_COMPLETE
        ]

        total_selections = len(selection_events)
        confidence_scores = []
        consultant_counts = {}
        model_counts = {}
        rationales = []
        risk_factors = []
        success_factors = []

        for event in selection_events:
            data = event.data

            # Extract confidence
            confidence = data.get("total_confidence", data.get("confidence_score", 0))
            if confidence:
                confidence_scores.append(confidence)

            # Count consultants
            consultants = data.get("consultants", [])
            for consultant in consultants:
                name = (
                    consultant
                    if isinstance(consultant, str)
                    else consultant.get("name", "unknown")
                )
                consultant_counts[name] = consultant_counts.get(name, 0) + 1

            # Count models (from selection rationale if available)
            models_used = data.get("models_used", [])
            for model in models_used:
                model_counts[model] = model_counts.get(model, 0) + 1

            # Extract rationales and factors (sanitized)
            if "selection_rationale" in data:
                rationale_preview = str(data["selection_rationale"])[:100] + "..."
                rationales.append(rationale_preview)

            risk_factors.extend(data.get("risk_factors", []))
            success_factors.extend(data.get("success_factors", []))

        avg_confidence = (
            sum(confidence_scores) / len(confidence_scores)
            if confidence_scores
            else 0.0
        )

        metrics = SelectionMetrics(
            total_selections=total_selections,
            average_confidence=avg_confidence,
            consultant_distribution=consultant_counts,
            model_distribution=model_counts,
            selection_rationales=rationales[:5],  # Limit to 5 for brevity
            risk_factors_identified=list(set(risk_factors))[:10],
            success_factors_identified=list(set(success_factors))[:10],
        )

        print(
            f"   Generated selection metrics: {total_selections} selections, {avg_confidence:.1%} avg confidence"
        )
        return metrics

    async def _generate_chunking_summary(
        self, events: List[ContextEvent]
    ) -> ChunkingSummary:
        """Generate content chunking and processing summary"""

        # Look for chunking-related events
        chunking_events = [
            e
            for e in events
            if any(
                keyword in e.event_type.value.lower()
                for keyword in ["chunk", "processing", "content"]
            )
        ]

        # Extract chunking metrics from events
        total_chunks = 0
        chunk_sizes = []
        processing_times = []
        fingerprints = []

        for event in chunking_events:
            data = event.data

            # Extract chunk information
            if "chunk_count" in data:
                total_chunks += data["chunk_count"]

            if "chunk_size" in data:
                chunk_sizes.append(data["chunk_size"])

            if "processing_time_ms" in data:
                processing_times.append(data["processing_time_ms"])

            if "content_fingerprint" in data:
                fingerprints.append(data["content_fingerprint"])

        # Create size distribution
        size_distribution = {}
        for size in chunk_sizes:
            if size < 1000:
                size_distribution["small"] = size_distribution.get("small", 0) + 1
            elif size < 5000:
                size_distribution["medium"] = size_distribution.get("medium", 0) + 1
            else:
                size_distribution["large"] = size_distribution.get("large", 0) + 1

        summary = ChunkingSummary(
            total_chunks_processed=total_chunks or len(chunking_events),
            chunking_strategy_used="adaptive",  # Default assumption
            chunk_size_distribution=size_distribution or {"unknown": 1},
            processing_time_per_chunk_ms=processing_times,
            content_fingerprints=fingerprints[:10],  # Limit for display
            overlap_strategy="sliding_window",  # Default assumption
        )

        print(
            f"   Generated chunking summary: {summary.total_chunks_processed} chunks processed"
        )
        return summary

    async def _generate_consultant_performance(
        self, events: List[ContextEvent]
    ) -> List[ConsultantPerformance]:
        """Generate per-consultant performance metrics"""

        # Group events by consultant
        consultant_events = {}

        for event in events:
            data = event.data

            # Extract consultant information from various event types
            consultant_name = None
            if "consultant" in data:
                consultant_name = data["consultant"]
            elif "consultant_name" in data:
                consultant_name = data["consultant_name"]
            elif "consultant_id" in data:
                consultant_name = data["consultant_id"]
            elif event.event_type == ContextEventType.CONSULTANT_ANALYSIS_START:
                consultant_name = data.get("consultant_name", "unknown")

            if consultant_name:
                if consultant_name not in consultant_events:
                    consultant_events[consultant_name] = []
                consultant_events[consultant_name].append(event)

        performance_list = []

        for consultant_name, events_list in consultant_events.items():
            # Calculate metrics for this consultant
            total_invocations = len(events_list)
            processing_times = []
            confidence_scores = []
            specializations = set()

            for event in events_list:
                data = event.data

                # Extract processing time
                proc_time = data.get("processing_time_ms", data.get("duration_ms"))
                if proc_time:
                    processing_times.append(proc_time)

                # Extract confidence
                confidence = data.get("confidence_score", data.get("confidence"))
                if confidence:
                    confidence_scores.append(confidence)

                # Extract specializations
                specs = data.get("specializations", data.get("expertise_areas", []))
                if isinstance(specs, list):
                    specializations.update(specs)
                elif isinstance(specs, str):
                    specializations.add(specs)

            avg_processing_time = (
                sum(processing_times) / len(processing_times)
                if processing_times
                else 0.0
            )
            success_rate = (
                len([c for c in confidence_scores if c > 0.7]) / len(confidence_scores)
                if confidence_scores
                else 0.0
            )
            contribution_quality = (
                sum(confidence_scores) / len(confidence_scores)
                if confidence_scores
                else 0.5
            )

            performance = ConsultantPerformance(
                consultant_name=consultant_name,
                total_invocations=total_invocations,
                average_processing_time_ms=avg_processing_time,
                confidence_scores=confidence_scores,
                success_rate=success_rate,
                specializations_used=list(specializations),
                contribution_quality=contribution_quality,
            )

            performance_list.append(performance)

        # Sort by total invocations (most active first)
        performance_list.sort(key=lambda x: x.total_invocations, reverse=True)

        print(
            f"   Generated consultant performance for {len(performance_list)} consultants"
        )
        return performance_list

    async def _generate_quality_analysis(
        self, events: List[ContextEvent]
    ) -> QualityAnalysis:
        """Generate quality analysis summary"""

        # Look for quality-related events
        quality_events = [
            e
            for e in events
            if any(
                keyword in e.event_type.value.lower()
                for keyword in ["quality", "validation", "check", "audit"]
            )
        ]

        # Extract quality metrics
        quality_scores = []
        quality_dimensions = {}
        improvements = []
        bias_checks = []
        validations = []

        for event in quality_events:
            data = event.data

            # Extract quality scores
            score = data.get("quality_score", data.get("overall_quality_score"))
            if score:
                quality_scores.append(score)

            # Extract quality dimensions
            dimensions = data.get("quality_dimensions", {})
            for dim, score in dimensions.items():
                if dim not in quality_dimensions:
                    quality_dimensions[dim] = []
                quality_dimensions[dim].append(score)

            # Extract improvement suggestions
            suggestions = data.get(
                "improvement_suggestions", data.get("recommendations", [])
            )
            if isinstance(suggestions, list):
                improvements.extend(suggestions)

            # Extract bias checks
            bias_info = data.get("bias_checks", data.get("bias_analysis"))
            if bias_info:
                bias_checks.append(str(bias_info)[:100])

            # Extract validation steps
            validation_info = data.get(
                "validation_steps", data.get("validation_results")
            )
            if validation_info:
                validations.append(str(validation_info)[:100])

        # Calculate averages for quality dimensions
        avg_dimensions = {}
        for dim, scores in quality_dimensions.items():
            avg_dimensions[dim] = sum(scores) / len(scores)

        overall_quality = (
            sum(quality_scores) / len(quality_scores) if quality_scores else 0.75
        )  # Default

        analysis = QualityAnalysis(
            overall_quality_score=overall_quality,
            quality_dimensions=avg_dimensions,
            improvement_suggestions=list(set(improvements))[:5],
            bias_checks_performed=list(set(bias_checks))[:5],
            validation_steps_completed=list(set(validations))[:5],
        )

        print(f"   Generated quality analysis: {overall_quality:.1%} overall quality")
        return analysis

    async def _generate_processing_stages(
        self, events: List[ContextEvent]
    ) -> List[ProcessingStage]:
        """Generate processing stages summary"""

        # Define stage boundaries based on event types
        stage_markers = {
            "initialization": [
                ContextEventType.ENGAGEMENT_STARTED,
                ContextEventType.AGENT_INSTANTIATED,
            ],
            "research": [
                ContextEventType.RESEARCH_QUERY,
                ContextEventType.RESEARCH_RESULT,
            ],
            "selection": [
                ContextEventType.CONSULTANT_SELECTION_COMPLETE,
                ContextEventType.MODEL_SELECTION_JUSTIFICATION,
            ],
            "analysis": [
                ContextEventType.LLM_PROVIDER_REQUEST,
                ContextEventType.LLM_PROVIDER_RESPONSE,
            ],
            "synthesis": [
                ContextEventType.SYNTHESIS_CREATED,
                ContextEventType.DEVILS_ADVOCATE_COMPLETE,
            ],
            "finalization": [
                ContextEventType.FINAL_REPORT_GENERATED,
                ContextEventType.ENGAGEMENT_COMPLETED,
            ],
        }

        stages = []

        for stage_name, marker_types in stage_markers.items():
            # Find events for this stage
            stage_events = [e for e in events if e.event_type in marker_types]

            if stage_events:
                start_time = min(e.timestamp for e in stage_events)
                end_time = max(e.timestamp for e in stage_events)
                duration_ms = int((end_time - start_time).total_seconds() * 1000)

                # Extract key outputs
                key_outputs = []
                error_count = 0

                for event in stage_events:
                    if event.event_type == ContextEventType.ERROR_OCCURRED:
                        error_count += 1

                    # Extract meaningful outputs
                    data = event.data
                    if "result" in data:
                        key_outputs.append(str(data["result"])[:50] + "...")
                    elif "summary" in data:
                        key_outputs.append(str(data["summary"])[:50] + "...")

                stage = ProcessingStage(
                    stage_name=stage_name,
                    start_time=start_time.isoformat(),
                    end_time=end_time.isoformat(),
                    duration_ms=duration_ms,
                    events_count=len(stage_events),
                    success=error_count == 0,
                    key_outputs=key_outputs[:3],  # Limit to 3
                    error_count=error_count,
                )

                stages.append(stage)

        print(f"   Generated {len(stages)} processing stages")
        return stages

    async def _generate_error_summary(
        self, events: List[ContextEvent]
    ) -> Dict[str, Any]:
        """Generate error analysis summary"""

        error_events = [
            e for e in events if e.event_type == ContextEventType.ERROR_OCCURRED
        ]

        error_types = {}
        error_timeline = []
        recovery_count = 0

        for event in error_events:
            data = event.data

            # Count error types
            error_type = data.get("error_type", "unknown")
            error_types[error_type] = error_types.get(error_type, 0) + 1

            # Add to timeline
            error_timeline.append(
                {
                    "timestamp": event.timestamp.isoformat(),
                    "error_type": error_type,
                    "description": str(data.get("error_message", ""))[:100],
                }
            )

        # Count recoveries
        recovery_events = [
            e for e in events if e.event_type == ContextEventType.ERROR_RECOVERED
        ]
        recovery_count = len(recovery_events)

        return {
            "total_errors": len(error_events),
            "error_types": error_types,
            "recovery_count": recovery_count,
            "error_rate": len(error_events) / len(events) if events else 0.0,
            "recent_errors": error_timeline[-5:],  # Last 5 errors
            "system_stability": 1.0 - (len(error_events) / max(1, len(events))),
        }

    def _determine_stage_from_event(self, event: ContextEvent) -> str:
        """Determine processing stage from event type"""

        stage_mapping = {
            ContextEventType.ENGAGEMENT_STARTED: "initialization",
            ContextEventType.QUERY_RECEIVED: "initialization",
            ContextEventType.RESEARCH_QUERY: "research",
            ContextEventType.RESEARCH_RESULT: "research",
            ContextEventType.CONSULTANT_SELECTION_COMPLETE: "selection",
            ContextEventType.MODEL_SELECTION_JUSTIFICATION: "selection",
            ContextEventType.LLM_PROVIDER_REQUEST: "analysis",
            ContextEventType.LLM_PROVIDER_RESPONSE: "analysis",
            ContextEventType.SYNTHESIS_CREATED: "synthesis",
            ContextEventType.DEVILS_ADVOCATE_COMPLETE: "synthesis",
            ContextEventType.SENIOR_ADVISOR_COMPLETE: "synthesis",
            ContextEventType.FINAL_REPORT_GENERATED: "finalization",
            ContextEventType.ENGAGEMENT_COMPLETED: "finalization",
        }

        return stage_mapping.get(event.event_type, "unknown")

    def _calculate_session_duration(self, events: List[ContextEvent]) -> int:
        """Calculate session duration in milliseconds"""
        if not events:
            return 0

        start_time = min(e.timestamp for e in events)
        end_time = max(e.timestamp for e in events)

        return int((end_time - start_time).total_seconds() * 1000)

    def _verify_glass_box_compliance(
        self, events: List[ContextEvent]
    ) -> Dict[str, bool]:
        """Verify glass-box compliance of events"""

        compliance_checks = {
            "no_raw_prompts": True,
            "no_raw_responses": True,
            "pii_scrubbed": True,
            "content_fingerprinted": True,
            "metadata_enriched": True,
        }

        for event in events:
            data = event.data

            # Check for raw prompts/responses
            if any(key in data for key in ["prompt", "raw_prompt", "system_prompt"]):
                compliance_checks["no_raw_prompts"] = False

            if any(
                key in data for key in ["response", "raw_response", "llm_response"]
            ) and "fingerprint" not in str(data):
                compliance_checks["no_raw_responses"] = False

            # Check PII scrubbing
            if not event.metadata.get("pii_scrubbed", False):
                compliance_checks["pii_scrubbed"] = False

            # Check content fingerprinting
            if "content" in data and "fingerprint" not in data:
                compliance_checks["content_fingerprinted"] = False

            # Check metadata enrichment
            if not event.metadata.get("trace_id"):
                compliance_checks["metadata_enriched"] = False

        return compliance_checks

    def _create_empty_dossier(self, trace_id: str) -> TransparencyDossier:
        """Create empty dossier when no events are found"""
        return TransparencyDossier(
            trace_id=trace_id,
            generated_at=datetime.now(timezone.utc).isoformat(),
            session_duration_ms=0,
            total_events=0,
            timeline=[],
            selection_metrics=SelectionMetrics(0, 0.0, {}, {}, [], [], []),
            chunking_summary=ChunkingSummary(0, "none", {}, [], [], "none"),
            consultant_performance=[],
            quality_analysis=QualityAnalysis(0.0, {}, [], [], []),
            processing_stages=[],
            error_summary={
                "total_errors": 0,
                "error_types": {},
                "recovery_count": 0,
                "error_rate": 0.0,
                "recent_errors": [],
                "system_stability": 1.0,
            },
            glass_box_compliance={
                "no_raw_prompts": True,
                "no_raw_responses": True,
                "pii_scrubbed": True,
                "content_fingerprinted": True,
                "metadata_enriched": True,
            },
        )

    def _create_error_dossier(
        self, trace_id: str, error_message: str
    ) -> TransparencyDossier:
        """Create error dossier when assembly fails"""
        return TransparencyDossier(
            trace_id=trace_id,
            generated_at=datetime.now(timezone.utc).isoformat(),
            session_duration_ms=0,
            total_events=0,
            timeline=[
                TimelineEvent(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    event_type="assembly_error",
                    description=f"Dossier assembly failed: {error_message}",
                    stage="error",
                )
            ],
            selection_metrics=SelectionMetrics(0, 0.0, {}, {}, [], [], []),
            chunking_summary=ChunkingSummary(0, "error", {}, [], [], "error"),
            consultant_performance=[],
            quality_analysis=QualityAnalysis(0.0, {}, [], [], []),
            processing_stages=[],
            error_summary={
                "total_errors": 1,
                "error_types": {"assembly_error": 1},
                "recovery_count": 0,
                "error_rate": 1.0,
                "recent_errors": [
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "error_type": "assembly_error",
                        "description": error_message,
                    }
                ],
                "system_stability": 0.0,
            },
            glass_box_compliance={
                "no_raw_prompts": False,
                "no_raw_responses": False,
                "pii_scrubbed": False,
                "content_fingerprinted": False,
                "metadata_enriched": False,
            },
        )


async def test_transparency_dossier_assembler():
    """Test the transparency dossier assembler"""
    print("🧪 Testing Transparency Dossier Assembler")
    print("=" * 50)

    # Create test context stream
    try:
        from src.core.unified_context_stream import get_unified_context_stream
        context_stream = get_unified_context_stream()
    except Exception:
        context_stream = None

        # Add some test events
        context_stream.add_event(
            ContextEventType.ENGAGEMENT_STARTED,
            {"query": "test query", "user_id": "test_user"},
        )

        context_stream.add_event(
            ContextEventType.CONSULTANT_SELECTION_COMPLETE,
            {
                "consultant_count": 3,
                "total_confidence": 0.85,
                "consultants": ["strategic_advisor", "market_analyst", "risk_assessor"],
                "selection_rationale": "Selected based on query complexity and domain expertise",
                "risk_factors": ["market volatility", "competitive pressure"],
                "success_factors": ["strong team", "clear vision"],
            },
        )

        context_stream.add_event(
            ContextEventType.SYNTHESIS_CREATED,
            {"synthesis_content": "test synthesis", "confidence_score": 0.9},
        )

        context_stream.add_event(
            ContextEventType.ENGAGEMENT_COMPLETED,
            {"final_status": "completed", "duration_ms": 45000},
        )

        # Create assembler and generate dossier
        assembler = TransparencyDossierAssembler()
        dossier = await assembler.assemble_dossier(context_stream)

        # Display results
        print("\n📊 DOSSIER RESULTS:")
        print(f"Trace ID: {dossier.trace_id}")
        print(f"Total Events: {dossier.total_events}")
        print(f"Session Duration: {dossier.session_duration_ms}ms")
        print(f"Timeline Events: {len(dossier.timeline)}")
        print(f"Consultant Performance: {len(dossier.consultant_performance)}")
        print(f"Processing Stages: {len(dossier.processing_stages)}")
        print(f"Overall Quality: {dossier.quality_analysis.overall_quality_score:.1%}")
        print(f"Glass-Box Compliance: {all(dossier.glass_box_compliance.values())}")

        print("\n✅ Transparency dossier assembler test completed")
    else:
        print("❌ UnifiedContextStream not available for testing")


if __name__ == "__main__":
    asyncio.run(test_transparency_dossier_assembler())
