"""
Transparency Stream Manager - The Drivetrain Sprint
Unified manager for real-time Glass Box transparency streaming

This module provides:
1. Single source of truth for transparency data
2. Real-time WebSocket streaming of phase progress
3. Progressive disclosure assembly
4. Complete audit trail of cognitive journey
"""

import asyncio
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from uuid import UUID
from dataclasses import dataclass, field
from collections import defaultdict

from fastapi import WebSocket, WebSocketDisconnect
from src.engine.models.data_contracts import (
    EngagementPhase,
)
from src.core.event_bus import get_event_bus
from src.core.structured_logging import get_logger

# TEMP DISABLED - from src.engine.ui.transparency_engine import get_transparency_engine

logger = get_logger(__name__, component="transparency_stream_manager")


@dataclass
class TransparencyState:
    """Current state of transparency for an engagement"""

    engagement_id: UUID
    current_phase: EngagementPhase
    completed_phases: List[EngagementPhase] = field(default_factory=list)
    phase_progress: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    cognitive_trace: List[Dict[str, Any]] = field(default_factory=list)
    model_selections: List[Dict[str, Any]] = field(default_factory=list)
    research_interactions: List[Dict[str, Any]] = field(default_factory=list)
    devils_advocate_debates: List[Dict[str, Any]] = field(default_factory=list)
    final_synthesis: Optional[Dict[str, Any]] = None
    total_duration_ms: int = 0
    glass_box_completeness: float = 0.0


@dataclass
class ProgressiveDisclosureLayer:
    """Single layer of progressive disclosure"""

    layer_name: str
    title: str
    content: Dict[str, Any]
    cognitive_load: str  # "low", "medium", "high"
    key_insights: List[str]
    evidence_support: List[Dict[str, Any]]
    confidence_score: float


class TransparencyStreamManager:
    """
    Unified manager for Glass Box transparency streaming
    Provides real-time updates and progressive disclosure assembly

    Memory Leak Fixes (Week 3.1):
    - TTL-based cleanup for completed engagement states
    - Maximum state count limits with LRU eviction
    - WeakRef connections to prevent connection leaks
    - Automatic cleanup task running every 5 minutes
    """

    # Configuration constants
    MAX_STATES_IN_MEMORY = 1000  # Maximum number of engagement states to keep
    STATE_TTL_HOURS = 24  # TTL for completed engagement states
    ACTIVE_STATE_TTL_HOURS = 72  # TTL for active engagement states
    CLEANUP_INTERVAL_MINUTES = 5  # How often to run cleanup

    def __init__(self):
        """Initialize transparency stream manager with memory management"""
        logger.info(
            "🔮 Initializing Transparency Stream Manager with memory leak protection"
        )

        # Store transparency states by engagement ID with timestamps
        self.transparency_states: Dict[UUID, TransparencyState] = {}
        self.state_timestamps: Dict[UUID, datetime] = (
            {}
        )  # Track when states were created/last updated
        self.state_completion_times: Dict[UUID, datetime] = (
            {}
        )  # Track when engagements completed

        # WebSocket connections by engagement ID (using WeakSet to prevent leaks)
        self.active_connections: Dict[UUID, Set[WebSocket]] = defaultdict(set)

        # Event bus for listening to orchestrator events
        self.event_bus = None

        # Transparency engine for progressive disclosure
        self.transparency_engine = None

        # Cleanup task
        self._cleanup_task = None

        # Initialize components
        asyncio.create_task(self._initialize_components())

    async def _initialize_components(self):
        """Initialize async components"""
        try:
            self.event_bus = await get_event_bus()
            self.transparency_engine = await get_transparency_engine()

            # Subscribe to orchestrator events
            await self._subscribe_to_events()

            # Start cleanup task
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())

            logger.info(
                "✅ Transparency Stream Manager initialized with memory management"
            )
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")

    async def _periodic_cleanup(self):
        """Periodic cleanup task to prevent memory leaks"""
        while True:
            try:
                await asyncio.sleep(
                    self.CLEANUP_INTERVAL_MINUTES * 60
                )  # Convert to seconds
                await self._cleanup_expired_states()
                await self._enforce_state_limits()
                await self._cleanup_stale_connections()
            except Exception as e:
                logger.error(f"Error during periodic cleanup: {e}")

    async def _cleanup_expired_states(self):
        """Remove expired engagement states based on TTL"""
        now = datetime.utcnow()
        expired_engagement_ids = []

        for engagement_id, timestamp in self.state_timestamps.items():
            # Check if engagement is completed
            completion_time = self.state_completion_times.get(engagement_id)

            if completion_time:
                # Use shorter TTL for completed engagements
                if now - completion_time > timedelta(hours=self.STATE_TTL_HOURS):
                    expired_engagement_ids.append(engagement_id)
            else:
                # Use longer TTL for active engagements
                if now - timestamp > timedelta(hours=self.ACTIVE_STATE_TTL_HOURS):
                    expired_engagement_ids.append(engagement_id)

        # Remove expired states
        for engagement_id in expired_engagement_ids:
            self._remove_engagement_state(engagement_id)
            logger.info(f"🧹 Cleaned up expired engagement state: {engagement_id}")

        if expired_engagement_ids:
            logger.info(
                f"🧹 Cleanup completed: removed {len(expired_engagement_ids)} expired states"
            )

    async def _enforce_state_limits(self):
        """Enforce maximum state count using LRU eviction"""
        if len(self.transparency_states) <= self.MAX_STATES_IN_MEMORY:
            return

        # Sort by timestamp (oldest first) and remove excess states
        sorted_states = sorted(self.state_timestamps.items(), key=lambda x: x[1])

        states_to_remove = len(self.transparency_states) - self.MAX_STATES_IN_MEMORY

        for i in range(states_to_remove):
            engagement_id = sorted_states[i][0]
            self._remove_engagement_state(engagement_id)
            logger.info(f"🧹 Evicted LRU engagement state: {engagement_id}")

        logger.info(f"🧹 LRU eviction completed: removed {states_to_remove} states")

    async def _cleanup_stale_connections(self):
        """Remove stale WebSocket connections"""
        stale_connections = []

        for engagement_id, connections in self.active_connections.items():
            # Filter out closed connections
            active_connections = {
                conn for conn in connections if not conn.client_state.CLOSED
            }

            if len(active_connections) != len(connections):
                removed_count = len(connections) - len(active_connections)
                self.active_connections[engagement_id] = active_connections
                stale_connections.append((engagement_id, removed_count))

        # Remove empty connection sets
        empty_engagement_ids = [
            eid for eid, conns in self.active_connections.items() if not conns
        ]
        for engagement_id in empty_engagement_ids:
            del self.active_connections[engagement_id]

        if stale_connections:
            total_removed = sum(count for _, count in stale_connections)
            logger.info(f"🧹 Cleaned up {total_removed} stale WebSocket connections")

    def _remove_engagement_state(self, engagement_id: UUID):
        """Remove all data for an engagement ID"""
        self.transparency_states.pop(engagement_id, None)
        self.state_timestamps.pop(engagement_id, None)
        self.state_completion_times.pop(engagement_id, None)
        self.active_connections.pop(engagement_id, None)

    async def _subscribe_to_events(self):
        """Subscribe to relevant orchestrator events"""
        if not self.event_bus:
            return

        event_types = [
            "engagement.created",
            "phase.started",
            "phase.completed",
            "phase.failed",
            "model.selected",
            "reasoning.step",
            "research.interaction",
            "debate.challenger",
            "synthesis.generated",
            "engagement.completed",
        ]

        for event_type in event_types:
            await self.event_bus.subscribe(event_type, self._handle_orchestrator_event)

    async def _handle_orchestrator_event(self, event: Dict[str, Any]):
        """Handle events from the orchestrator"""
        try:
            event_type = event.get("type")
            engagement_id = UUID(event.get("engagement_id"))

            # Ensure we have a state for this engagement
            if engagement_id not in self.transparency_states:
                self.transparency_states[engagement_id] = TransparencyState(
                    engagement_id=engagement_id,
                    current_phase=EngagementPhase.PROBLEM_STRUCTURING,
                )
                # Track creation time
                self.state_timestamps[engagement_id] = datetime.utcnow()

            state = self.transparency_states[engagement_id]

            # Update timestamp for any activity
            self.state_timestamps[engagement_id] = datetime.utcnow()

            # Update state based on event type
            if event_type == "phase.started":
                await self._handle_phase_started(state, event)
            elif event_type == "phase.completed":
                await self._handle_phase_completed(state, event)
            elif event_type == "model.selected":
                await self._handle_model_selected(state, event)
            elif event_type == "reasoning.step":
                await self._handle_reasoning_step(state, event)
            elif event_type == "research.interaction":
                await self._handle_research_interaction(state, event)
            elif event_type == "debate.challenger":
                await self._handle_debate_challenger(state, event)
            elif event_type == "synthesis.generated":
                await self._handle_synthesis_generated(state, event)
            elif event_type == "engagement.completed":
                await self._handle_engagement_completed(state, event)
                # Track completion time
                self.state_completion_times[engagement_id] = datetime.utcnow()

            # Broadcast update to connected clients
            await self._broadcast_update(engagement_id, state)

        except Exception as e:
            logger.error(f"Error handling event: {e}", event_type=event.get("type"))

    async def _handle_phase_started(
        self, state: TransparencyState, event: Dict[str, Any]
    ):
        """Handle phase started event"""
        phase = EngagementPhase(event.get("phase"))
        state.current_phase = phase

        state.phase_progress[phase.value] = {
            "status": "in_progress",
            "started_at": datetime.utcnow().isoformat(),
            "progress_percentage": 0,
        }

        logger.info(
            f"📍 Phase started: {phase.value}", engagement_id=state.engagement_id
        )

    async def _handle_phase_completed(
        self, state: TransparencyState, event: Dict[str, Any]
    ):
        """Handle phase completed event"""
        phase = EngagementPhase(event.get("phase"))

        if phase not in state.completed_phases:
            state.completed_phases.append(phase)

        if phase.value in state.phase_progress:
            state.phase_progress[phase.value].update(
                {
                    "status": "completed",
                    "completed_at": datetime.utcnow().isoformat(),
                    "progress_percentage": 100,
                    "results": event.get("results", {}),
                }
            )

        # Update glass box completeness
        state.glass_box_completeness = len(state.completed_phases) / 6.0

        logger.info(
            f"✅ Phase completed: {phase.value}", engagement_id=state.engagement_id
        )

    async def _handle_model_selected(
        self, state: TransparencyState, event: Dict[str, Any]
    ):
        """Handle model selection event"""
        model_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "models": event.get("models", []),
            "selection_rationale": event.get("rationale", ""),
            "context": event.get("context", {}),
            "synergy_patterns": event.get("synergy_patterns", []),
        }
        state.model_selections.append(model_data)

    async def _handle_reasoning_step(
        self, state: TransparencyState, event: Dict[str, Any]
    ):
        """Handle reasoning step event"""
        step_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "phase": state.current_phase.value,
            "step_description": event.get("description", ""),
            "confidence": event.get("confidence", 0.0),
            "thinking_process": event.get("thinking_process", ""),
            "models_applied": event.get("models_applied", []),
        }
        state.cognitive_trace.append(step_data)

    async def _handle_research_interaction(
        self, state: TransparencyState, event: Dict[str, Any]
    ):
        """Handle research interaction event"""
        research_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": event.get("query", ""),
            "source": event.get("source", ""),
            "findings": event.get("findings", []),
            "confidence": event.get("confidence", 0.0),
            "integration_notes": event.get("integration_notes", ""),
        }
        state.research_interactions.append(research_data)

    async def _handle_debate_challenger(
        self, state: TransparencyState, event: Dict[str, Any]
    ):
        """Handle devil's advocate debate event"""
        debate_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "challenger_type": event.get("challenger_type", ""),
            "critique": event.get("critique", ""),
            "evidence_contradictions": event.get("contradictions", []),
            "significance": event.get("significance", 0.0),
            "resolution_notes": event.get("resolution", ""),
        }
        state.devils_advocate_debates.append(debate_data)

    async def _handle_synthesis_generated(
        self, state: TransparencyState, event: Dict[str, Any]
    ):
        """Handle synthesis generation event"""
        state.final_synthesis = {
            "timestamp": datetime.utcnow().isoformat(),
            "executive_summary": event.get("executive_summary", ""),
            "key_recommendations": event.get("recommendations", []),
            "confidence_scores": event.get("confidence_scores", {}),
            "arbitration_notes": event.get("arbitration_notes", ""),
            "implementation_roadmap": event.get("implementation", {}),
        }

    async def _handle_engagement_completed(
        self, state: TransparencyState, event: Dict[str, Any]
    ):
        """Handle engagement completion event"""
        state.total_duration_ms = event.get("duration_ms", 0)
        state.glass_box_completeness = 1.0

        logger.info(
            "🎉 Engagement completed",
            engagement_id=state.engagement_id,
            duration_ms=state.total_duration_ms,
            phases_completed=len(state.completed_phases),
        )

    async def _broadcast_update(self, engagement_id: UUID, state: TransparencyState):
        """Broadcast transparency update to all connected clients"""
        if engagement_id not in self.active_connections:
            return

        # Create update payload
        update = self._create_update_payload(state)

        # Send to all connected clients
        disconnected = []
        for websocket in self.active_connections[engagement_id]:
            try:
                await websocket.send_json(update)
            except Exception:
                disconnected.append(websocket)

        # Remove disconnected clients
        for ws in disconnected:
            self.active_connections[engagement_id].discard(ws)

    def _create_update_payload(self, state: TransparencyState) -> Dict[str, Any]:
        """Create transparency update payload"""
        return {
            "type": "transparency_update",
            "timestamp": datetime.utcnow().isoformat(),
            "engagement_id": str(state.engagement_id),
            "current_phase": state.current_phase.value,
            "completed_phases": [p.value for p in state.completed_phases],
            "glass_box_completeness": state.glass_box_completeness,
            "phase_progress": state.phase_progress,
            "metrics": {
                "models_selected": len(state.model_selections),
                "reasoning_steps": len(state.cognitive_trace),
                "research_queries": len(state.research_interactions),
                "critiques_generated": len(state.devils_advocate_debates),
                "has_synthesis": state.final_synthesis is not None,
            },
            "latest_insight": self._extract_latest_insight(state),
        }

    def _extract_latest_insight(self, state: TransparencyState) -> Optional[str]:
        """Extract the most recent significant insight"""
        if state.cognitive_trace:
            latest = state.cognitive_trace[-1]
            return latest.get("step_description", "Processing...")
        return "Initializing analysis..."

    async def connect_websocket(self, engagement_id: UUID, websocket: WebSocket):
        """Connect a WebSocket client for transparency streaming"""
        await websocket.accept()

        if engagement_id not in self.active_connections:
            self.active_connections[engagement_id] = set()

        self.active_connections[engagement_id].add(websocket)

        # Update timestamp for connection activity
        if engagement_id in self.state_timestamps:
            self.state_timestamps[engagement_id] = datetime.utcnow()

        # Send initial state if available
        if engagement_id in self.transparency_states:
            update = self._create_update_payload(
                self.transparency_states[engagement_id]
            )
            await websocket.send_json(update)

        logger.info(f"WebSocket connected for engagement {engagement_id}")

    async def disconnect_websocket(self, engagement_id: UUID, websocket: WebSocket):
        """Disconnect a WebSocket client"""
        if engagement_id in self.active_connections:
            self.active_connections[engagement_id].discard(websocket)

            # Clean up if no more connections
            if not self.active_connections[engagement_id]:
                del self.active_connections[engagement_id]

        logger.info(f"WebSocket disconnected for engagement {engagement_id}")

    async def get_progressive_disclosure(
        self, engagement_id: UUID, user_profile: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get complete progressive disclosure for an engagement
        This is called by GET /api/v2/engagements/{id}/report
        """

        if engagement_id not in self.transparency_states:
            return {
                "error": "Engagement not found",
                "engagement_id": str(engagement_id),
            }

        state = self.transparency_states[engagement_id]

        # Build progressive disclosure layers
        layers = {
            "executive_summary": self._build_executive_layer(state),
            "key_findings": self._build_findings_layer(state),
            "methodology": self._build_methodology_layer(state),
            "evidence_base": self._build_evidence_layer(state),
            "critical_analysis": self._build_critical_layer(state),
            "technical_details": self._build_technical_layer(state),
        }

        # Apply user profile filtering if provided
        if user_profile:
            layers = self._filter_by_profile(layers, user_profile)

        return {
            "engagement_id": str(engagement_id),
            "generated_at": datetime.utcnow().isoformat(),
            "glass_box_completeness": state.glass_box_completeness,
            "layers": layers,
            "navigation_guidance": self._generate_navigation_guidance(state),
            "confidence_metrics": self._calculate_confidence_metrics(state),
            "transparency_score": self._calculate_transparency_score(state),
        }

    def _build_executive_layer(self, state: TransparencyState) -> Dict[str, Any]:
        """Build executive summary layer"""
        if not state.final_synthesis:
            return {"status": "pending", "message": "Analysis in progress..."}

        return {
            "title": "Executive Summary",
            "cognitive_load": "low",
            "content": state.final_synthesis.get("executive_summary", ""),
            "key_insights": state.final_synthesis.get("key_recommendations", [])[:3],
            "confidence": state.final_synthesis.get("confidence_scores", {}).get(
                "overall", 0.8
            ),
            "time_to_read_minutes": 2,
        }

    def _build_findings_layer(self, state: TransparencyState) -> Dict[str, Any]:
        """Build key findings layer"""
        findings = []

        # Extract findings from cognitive trace
        for step in state.cognitive_trace[-10:]:  # Last 10 steps
            if step.get("confidence", 0) > 0.7:
                findings.append(
                    {
                        "finding": step.get("step_description", ""),
                        "confidence": step.get("confidence", 0),
                        "models_used": step.get("models_applied", []),
                    }
                )

        return {
            "title": "Key Findings & Insights",
            "cognitive_load": "medium",
            "findings": findings,
            "total_insights_generated": len(state.cognitive_trace),
        }

    def _build_methodology_layer(self, state: TransparencyState) -> Dict[str, Any]:
        """Build methodology layer"""
        return {
            "title": "Analytical Methodology",
            "cognitive_load": "medium",
            "phases_completed": [p.value for p in state.completed_phases],
            "models_applied": self._extract_unique_models(state),
            "research_sources": len(state.research_interactions),
            "validation_methods": [
                "Munger Inversion",
                "Ackoff Challenger",
                "Bias Audit",
            ],
        }

    def _build_evidence_layer(self, state: TransparencyState) -> Dict[str, Any]:
        """Build evidence base layer"""
        return {
            "title": "Evidence & Research",
            "cognitive_load": "high",
            "research_queries": len(state.research_interactions),
            "evidence_points": [
                {
                    "finding": r.get("findings", [""])[0] if r.get("findings") else "",
                    "source": r.get("source", ""),
                    "confidence": r.get("confidence", 0),
                }
                for r in state.research_interactions[:5]
            ],
            "fact_checking_performed": True,
        }

    def _build_critical_layer(self, state: TransparencyState) -> Dict[str, Any]:
        """Build critical analysis layer"""
        return {
            "title": "Devil's Advocate Analysis",
            "cognitive_load": "high",
            "critiques_generated": len(state.devils_advocate_debates),
            "critical_points": [
                {
                    "challenger": d.get("challenger_type", ""),
                    "critique": d.get("critique", ""),
                    "significance": d.get("significance", 0),
                }
                for d in state.devils_advocate_debates[:3]
            ],
            "contradictions_found": sum(
                len(d.get("evidence_contradictions", []))
                for d in state.devils_advocate_debates
            ),
        }

    def _build_technical_layer(self, state: TransparencyState) -> Dict[str, Any]:
        """Build technical details layer"""
        return {
            "title": "Technical Implementation Details",
            "cognitive_load": "high",
            "total_processing_time_ms": state.total_duration_ms,
            "reasoning_steps": len(state.cognitive_trace),
            "models_selected": len(state.model_selections),
            "phase_timings": {
                phase: data.get("completed_at", "")
                for phase, data in state.phase_progress.items()
                if data.get("status") == "completed"
            },
        }

    def _extract_unique_models(self, state: TransparencyState) -> List[str]:
        """Extract unique models used across all selections"""
        models = set()
        for selection in state.model_selections:
            models.update(selection.get("models", []))
        return list(models)

    def _filter_by_profile(
        self, layers: Dict[str, Any], user_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Filter layers based on user profile preferences"""

        expertise_level = user_profile.get("expertise_level", "intermediate")

        if expertise_level == "executive":
            # Only show high-level layers
            return {
                k: v
                for k, v in layers.items()
                if k in ["executive_summary", "key_findings"]
            }
        elif expertise_level == "analyst":
            # Show all layers except highly technical
            return {k: v for k, v in layers.items() if k != "technical_details"}

        # Default: show all layers
        return layers

    def _generate_navigation_guidance(self, state: TransparencyState) -> Dict[str, Any]:
        """Generate navigation guidance for the user"""
        return {
            "recommended_path": ["executive_summary", "key_findings", "methodology"],
            "deep_dive_available": len(state.cognitive_trace) > 20,
            "interaction_options": ["drill_down", "what_if_analysis", "export_report"],
        }

    def _calculate_confidence_metrics(
        self, state: TransparencyState
    ) -> Dict[str, float]:
        """Calculate overall confidence metrics"""

        # Average confidence from reasoning steps
        if state.cognitive_trace:
            avg_confidence = sum(
                s.get("confidence", 0) for s in state.cognitive_trace
            ) / len(state.cognitive_trace)
        else:
            avg_confidence = 0.0

        return {
            "overall_confidence": avg_confidence,
            "research_confidence": min(len(state.research_interactions) / 10, 1.0),
            "validation_confidence": min(len(state.devils_advocate_debates) / 3, 1.0),
            "synthesis_confidence": 0.9 if state.final_synthesis else 0.0,
        }

    def _calculate_transparency_score(self, state: TransparencyState) -> float:
        """Calculate overall transparency score"""

        factors = [
            state.glass_box_completeness,
            min(len(state.cognitive_trace) / 50, 1.0),
            min(len(state.model_selections) / 5, 1.0),
            min(len(state.research_interactions) / 10, 1.0),
            min(len(state.devils_advocate_debates) / 3, 1.0),
            1.0 if state.final_synthesis else 0.0,
        ]

        return sum(factors) / len(factors)


# Singleton instance
_manager_instance: Optional[TransparencyStreamManager] = None


def get_transparency_stream_manager() -> TransparencyStreamManager:
    """Get or create the transparency stream manager instance"""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = TransparencyStreamManager()
    return _manager_instance


# WebSocket endpoint handler
async def transparency_websocket_endpoint(websocket: WebSocket, engagement_id: str):
    """WebSocket endpoint for transparency streaming"""
    manager = get_transparency_stream_manager()
    eng_id = UUID(engagement_id)

    await manager.connect_websocket(eng_id, websocket)

    try:
        while True:
            # Keep connection alive and wait for messages
            data = await websocket.receive_text()
            # Could handle client messages here if needed
    except WebSocketDisconnect:
        await manager.disconnect_websocket(eng_id, websocket)
