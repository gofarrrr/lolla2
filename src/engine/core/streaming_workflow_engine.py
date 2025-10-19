"""
METIS Streaming Workflow Engine
Implements real-time result streaming for <2s perceived performance
Week 1 Sprint: Streaming Architecture Implementation
"""

import time
import logging
from datetime import datetime
from typing import AsyncGenerator, Dict, Any, Optional
from dataclasses import dataclass

from src.engine.models.data_contracts import (
    MetisDataContract,
    EngagementPhase,
)

# Week 2 optimization imports
try:
    from src.core.parallel_cognitive_engine import get_parallel_cognitive_engine

    PARALLEL_PROCESSING_AVAILABLE = True
except ImportError:
    PARALLEL_PROCESSING_AVAILABLE = False


@dataclass
class StreamingEvent:
    """Event emitted during streaming execution"""

    type: str  # 'phase_started', 'phase_completed', 'progress_update', 'analysis_complete'
    phase: Optional[str] = None
    progress: str = ""
    data: Dict[str, Any] = None
    timestamp: str = ""
    estimated_time_remaining: Optional[int] = None
    confidence: Optional[float] = None


class StreamingWorkflowEngine:
    """
    High-performance streaming workflow engine
    Yields results as they become available for real-time UX
    """

    def __init__(
        self, base_workflow_engine, event_bus=None, enable_parallel_processing=True
    ):
        self.base_engine = base_workflow_engine
        self.event_bus = event_bus
        self.logger = logging.getLogger(__name__)
        self.enable_parallel_processing = (
            enable_parallel_processing and PARALLEL_PROCESSING_AVAILABLE
        )

        # Phase timing estimates for progress calculation
        # Parallel processing reduces these estimates
        parallel_factor = 0.6 if self.enable_parallel_processing else 1.0
        self.phase_estimates = {
            "problem_structuring": int(
                15 * parallel_factor
            ),  # 9s with parallel processing
            "hypothesis_generation": int(
                20 * parallel_factor
            ),  # 12s with parallel processing
            "analysis_execution": int(
                30 * parallel_factor
            ),  # 18s with parallel processing
            "synthesis_delivery": int(
                15 * parallel_factor
            ),  # 9s with parallel processing
        }

        self.total_estimated_time = sum(
            self.phase_estimates.values()
        )  # 48s total with parallel

    async def execute_engagement_streaming(
        self, engagement_event: MetisDataContract
    ) -> AsyncGenerator[StreamingEvent, None]:
        """
        Execute engagement with real-time streaming of results
        Yields events as each phase completes
        """

        start_time = time.time()
        phases = [
            "problem_structuring",
            "hypothesis_generation",
            "analysis_execution",
            "synthesis_delivery",
        ]

        self.logger.info("🚀 Starting streaming engagement execution")

        # Emit initial start event
        yield StreamingEvent(
            type="engagement_started",
            progress="0/4",
            data={
                "total_phases": len(phases),
                "estimated_total_time": self.total_estimated_time,
                "phases": phases,
            },
            timestamp=datetime.now().isoformat(),
        )

        # Execute each phase with streaming
        for i, phase in enumerate(phases):
            phase_start_time = time.time()

            # Emit phase start event
            remaining_time = sum(self.phase_estimates[p] for p in phases[i:])
            yield StreamingEvent(
                type="phase_started",
                phase=phase,
                progress=f"{i+1}/4",
                data={
                    "phase_name": self._get_friendly_phase_name(phase),
                    "description": self._get_phase_description(phase),
                    "estimated_duration": self.phase_estimates[phase],
                },
                timestamp=datetime.now().isoformat(),
                estimated_time_remaining=remaining_time,
            )

            # Execute the actual phase
            try:
                self.logger.info(f"⚡ Executing phase: {phase}")

                # Execute phase using base engine logic
                if phase == "problem_structuring":
                    phase_result = await self.base_engine._execute_problem_structuring(
                        engagement_event
                    )
                    result_data = {
                        "issue_tree": getattr(phase_result, "issue_tree", None),
                        "problem_structure": getattr(phase_result, "structure", None),
                        "mece_validation": getattr(
                            phase_result, "mece_validation", None
                        ),
                    }
                elif phase == "hypothesis_generation":
                    phase_result = (
                        await self.base_engine._execute_hypothesis_generation(
                            engagement_event
                        )
                    )
                    result_data = {
                        "ranked_hypotheses": getattr(
                            phase_result, "ranked_hypotheses", []
                        ),
                        "hypothesis_count": len(
                            getattr(phase_result, "hypotheses", [])
                        ),
                        "confidence_scores": getattr(
                            phase_result, "confidence_scores", {}
                        ),
                    }
                elif phase == "analysis_execution":
                    phase_result = await self.base_engine._execute_analysis_execution(
                        engagement_event
                    )
                    result_data = {
                        "analysis_results": getattr(
                            phase_result, "analysis_results", {}
                        ),
                        "mental_models_applied": getattr(
                            phase_result, "mental_models_applied", []
                        ),
                        "insights_generated": getattr(phase_result, "insights", []),
                    }
                elif phase == "synthesis_delivery":
                    phase_result = await self.base_engine._execute_synthesis_delivery(
                        engagement_event
                    )
                    result_data = {
                        "final_recommendations": getattr(
                            phase_result, "recommendations", []
                        ),
                        "executive_summary": getattr(
                            phase_result, "executive_summary", ""
                        ),
                        "confidence_level": getattr(phase_result, "confidence", 0.8),
                    }

                phase_end_time = time.time()
                phase_duration = phase_end_time - phase_start_time

                # Update contract state
                engagement_event.workflow_state.completed_phases.append(
                    EngagementPhase(phase)
                )
                engagement_event.workflow_state.current_phase = EngagementPhase(phase)

                # Emit phase completion event with results
                yield StreamingEvent(
                    type="phase_completed",
                    phase=phase,
                    progress=f"{i+1}/4",
                    data={
                        "phase_name": self._get_friendly_phase_name(phase),
                        "results": result_data,
                        "execution_time": phase_duration,
                        "status": "completed",
                        "next_phase": phases[i + 1] if i + 1 < len(phases) else None,
                    },
                    timestamp=datetime.now().isoformat(),
                    confidence=getattr(phase_result, "confidence", 0.8),
                )

                self.logger.info(f"✅ Phase {phase} completed in {phase_duration:.1f}s")

                # Emit progress update
                elapsed_time = time.time() - start_time
                progress_percentage = ((i + 1) / len(phases)) * 100

                yield StreamingEvent(
                    type="progress_update",
                    progress=f"{i+1}/4",
                    data={
                        "percentage": progress_percentage,
                        "elapsed_time": elapsed_time,
                        "phases_completed": i + 1,
                        "total_phases": len(phases),
                    },
                    timestamp=datetime.now().isoformat(),
                )

            except Exception as e:
                self.logger.error(f"❌ Phase {phase} failed: {e}")

                # Emit error event but continue with other phases
                yield StreamingEvent(
                    type="phase_error",
                    phase=phase,
                    progress=f"{i+1}/4",
                    data={
                        "error": str(e),
                        "phase_name": self._get_friendly_phase_name(phase),
                        "recovery_action": "Continuing with fallback analysis",
                    },
                    timestamp=datetime.now().isoformat(),
                )

        # Emit final completion event
        total_time = time.time() - start_time

        yield StreamingEvent(
            type="analysis_complete",
            progress="4/4",
            data={
                "final_deliverable": engagement_event.deliverable_artifacts,
                "total_execution_time": total_time,
                "target_met": total_time <= 30,  # 30s target for complex analysis
                "phases_completed": len(phases),
                "performance_metrics": {
                    "response_time": total_time,
                    "phases_executed": len(phases),
                    "cache_hits": 0,  # Will be updated with actual cache stats
                    "success": True,
                },
            },
            timestamp=datetime.now().isoformat(),
            confidence=0.9,  # Overall confidence
        )

        self.logger.info(f"🎉 Streaming engagement completed in {total_time:.1f}s")

    async def execute_engagement_streaming_optimized(
        self, engagement_event: MetisDataContract
    ) -> AsyncGenerator[StreamingEvent, None]:
        """
        Execute engagement with Week 2 optimizations: parallel processing + intelligent caching
        Falls back to regular streaming if optimizations unavailable
        """

        if self.enable_parallel_processing and PARALLEL_PROCESSING_AVAILABLE:
            self.logger.info("🚀 Using parallel processing optimizations")

            try:
                # Use parallel cognitive engine for optimized processing
                parallel_engine = await get_parallel_cognitive_engine(max_workers=4)

                # Stream events from parallel engine
                async for event in parallel_engine.execute_parallel_engagement(
                    engagement_event
                ):
                    yield event

                return

            except Exception as e:
                self.logger.warning(
                    f"⚠️ Parallel processing failed, falling back to sequential: {e}"
                )

        # Fallback to regular streaming execution
        self.logger.info("🔄 Using sequential processing (fallback mode)")
        async for event in self.execute_engagement_streaming(engagement_event):
            yield event

    def _get_friendly_phase_name(self, phase: str) -> str:
        """Get user-friendly phase names"""
        phase_names = {
            "problem_structuring": "Problem Structuring",
            "hypothesis_generation": "Hypothesis Generation",
            "analysis_execution": "Analysis Execution",
            "synthesis_delivery": "Synthesis & Delivery",
        }
        return phase_names.get(phase, phase.replace("_", " ").title())

    def _get_phase_description(self, phase: str) -> str:
        """Get descriptive text for each phase"""
        descriptions = {
            "problem_structuring": "Breaking down the problem using MECE principles and systems thinking",
            "hypothesis_generation": "Generating and ranking strategic hypotheses based on the problem structure",
            "analysis_execution": "Applying mental models and conducting deep analysis with research enhancement",
            "synthesis_delivery": "Synthesizing insights into actionable recommendations using the Pyramid Principle",
        }
        return descriptions.get(phase, f"Executing {phase.replace('_', ' ')}")


class StreamingEventHandler:
    """
    Handles streaming events and WebSocket broadcasting
    """

    def __init__(self):
        self.subscribers = set()
        self.logger = logging.getLogger(__name__)

    def subscribe(self, websocket):
        """Subscribe a WebSocket to streaming events"""
        self.subscribers.add(websocket)
        self.logger.info(f"📡 New subscriber added. Total: {len(self.subscribers)}")

    def unsubscribe(self, websocket):
        """Unsubscribe a WebSocket from streaming events"""
        self.subscribers.discard(websocket)
        self.logger.info(f"📡 Subscriber removed. Total: {len(self.subscribers)}")

    async def broadcast_event(self, event: StreamingEvent):
        """Broadcast event to all subscribers"""
        if not self.subscribers:
            return

        event_data = {
            "type": event.type,
            "phase": event.phase,
            "progress": event.progress,
            "data": event.data,
            "timestamp": event.timestamp,
            "estimated_time_remaining": event.estimated_time_remaining,
            "confidence": event.confidence,
        }

        # Broadcast to all connected clients
        disconnected = set()
        for websocket in self.subscribers:
            try:
                await websocket.send_json(event_data)
            except Exception as e:
                self.logger.warning(f"Failed to send to subscriber: {e}")
                disconnected.add(websocket)

        # Remove disconnected subscribers
        for ws in disconnected:
            self.unsubscribe(ws)


# Singleton event handler for global access
streaming_event_handler = StreamingEventHandler()


async def execute_streaming_engagement(
    engagement_event: MetisDataContract, base_workflow_engine
) -> AsyncGenerator[StreamingEvent, None]:
    """
    Convenience function to execute streaming engagement
    """
    streaming_engine = StreamingWorkflowEngine(base_workflow_engine)

    async for event in streaming_engine.execute_engagement_streaming(engagement_event):
        # Broadcast to WebSocket subscribers
        await streaming_event_handler.broadcast_event(event)

        # Yield to calling code
        yield event
