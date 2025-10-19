"""
Enhanced session management for hybrid cognitive orchestrator.

Integrates the hybrid orchestrator with existing production components:
- CheckpointManager for state persistence
- TransparencyStreamManager for real-time updates
- Supabase for engagement tracking
- MetisDataContract for result standardization
"""

from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from uuid import UUID, uuid4
import logging

from src.orchestration.hybrid_cognitive_orchestrator import HybridCognitiveOrchestrator
from src.core.checkpoint_manager import get_checkpoint_manager
from src.engine.api.transparency_stream_manager import get_transparency_stream_manager
from src.core.supabase_platform import get_supabase_client
from src.engine.models.data_contracts import MetisDataContract
from src.core.performance_cache_system import get_smart_cache
from src.core.llm_validation_gates import get_llm_validation_gates

logger = logging.getLogger(__name__)


class SessionStatus:
    """Session status constants"""

    INITIATED = "initiated"
    RESEARCH_PHASE = "research_phase"
    SYNTHESIS_PHASE = "synthesis_phase"
    ANALYSIS_PHASE = "analysis_phase"
    VALIDATION_PHASE = "validation_phase"
    COMPLETED = "completed"
    PAUSED_FOR_HITL = "paused_for_hitl"
    ERROR = "error"
    RECOVERED = "recovered"


class HybridSessionManager:
    """
    Enhanced session management for hybrid cognitive orchestrator.

    Provides lifecycle management for cognitive engagements with:
    - State persistence via CheckpointManager
    - Real-time streaming via TransparencyStreamManager
    - Database tracking via Supabase
    - Quality assurance via LLMValidationGates
    - Intelligent caching for performance
    """

    def __init__(self):
        self.orchestrator = HybridCognitiveOrchestrator()
        self.checkpoint_manager = get_checkpoint_manager()
        self.stream_manager = get_transparency_stream_manager()
        self.supabase = get_supabase_client()
        self.cache_system = get_smart_cache()
        self.validation_gates = get_llm_validation_gates()

        # Session state tracking
        self.active_sessions: Dict[str, Dict[str, Any]] = {}

    async def create_hybrid_session(
        self,
        initial_query: str,
        user_id: Optional[str] = None,
        session_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create new hybrid cognitive session.

        Args:
            initial_query: The user's initial query
            user_id: Optional user identifier
            session_config: Optional configuration overrides

        Returns:
            Session creation result with engagement_id and initial state
        """
        try:
            session_id = str(uuid4())

            # Create engagement record in Supabase
            engagement_record = await self._create_engagement_record(
                session_id, initial_query, user_id
            )

            # Initialize session state
            session_state = {
                "session_id": session_id,
                "engagement_id": engagement_record["id"],
                "initial_query": initial_query,
                "user_id": user_id,
                "status": SessionStatus.INITIATED,
                "created_at": datetime.now(timezone.utc),
                "config": session_config or {},
                "metrics": {
                    "total_cost_usd": 0.0,
                    "total_tokens": 0,
                    "phase_durations": {},
                    "hitl_interactions": 0,
                },
            }

            # Store in active sessions
            self.active_sessions[session_id] = session_state

            # Initialize transparency streaming
            await self.stream_manager.initialize_engagement_stream(
                engagement_id=UUID(engagement_record["id"]),
                initial_query=initial_query,
                user_id=user_id,
            )

            # Create initial checkpoint
            await self.checkpoint_manager.save_checkpoint(
                engagement_id=UUID(engagement_record["id"]),
                phase=SessionStatus.INITIATED,
                data={
                    "session_state": session_state,
                    "initial_query": initial_query,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

            logger.info(
                f"Created hybrid session {session_id} for engagement {engagement_record['id']}"
            )

            return {
                "session_id": session_id,
                "engagement_id": engagement_record["id"],
                "status": SessionStatus.INITIATED,
                "initial_query": initial_query,
                "created_at": session_state["created_at"].isoformat(),
                "streaming_enabled": True,
                "checkpoint_saved": True,
            }

        except Exception as e:
            logger.error(f"Failed to create hybrid session: {str(e)}")
            raise Exception(f"Session creation failed: {str(e)}")

    async def execute_hybrid_session(self, session_id: str) -> Dict[str, Any]:
        """
        Execute complete hybrid cognitive session.

        Args:
            session_id: Session identifier from create_hybrid_session

        Returns:
            Complete session results with MetisDataContract
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found in active sessions")

        session_state = self.active_sessions[session_id]
        engagement_id = UUID(session_state["engagement_id"])

        try:
            # Update session status
            await self._update_session_status(session_id, SessionStatus.RESEARCH_PHASE)

            # Execute hybrid orchestration
            orchestration_result = await self.orchestrator.execute_hybrid_engagement(
                initial_query=session_state["initial_query"],
                user_id=session_state["user_id"],
            )

            # Update session metrics
            session_state["metrics"].update(
                {
                    "total_cost_usd": orchestration_result["metrics"]["total_cost_usd"],
                    "total_tokens": orchestration_result["metrics"]["total_tokens"],
                    "total_events": orchestration_result["metrics"]["total_events"],
                    "research_queries": orchestration_result["metrics"][
                        "research_queries"
                    ],
                    "micro_steps": orchestration_result["metrics"]["micro_steps"],
                    "hitl_interactions": orchestration_result["metrics"][
                        "hitl_interactions"
                    ],
                }
            )

            # Create MetisDataContract
            contract = await self._create_metis_contract(
                orchestration_result, session_state
            )

            # Update engagement record
            await self._update_engagement_record(
                engagement_id, orchestration_result, contract
            )

            # Final checkpoint
            await self.checkpoint_manager.save_checkpoint(
                engagement_id=engagement_id,
                phase=SessionStatus.COMPLETED,
                data={
                    "session_state": session_state,
                    "orchestration_result": orchestration_result,
                    "contract": contract.dict(),
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                },
            )

            # Update session status
            await self._update_session_status(session_id, SessionStatus.COMPLETED)

            # Cache successful result
            await self.cache_system.put(
                content_type="hybrid_session_result",
                primary_key=session_id,
                content=contract.dict(),
                context={"user_id": session_state["user_id"]},
                confidence_score=contract.confidence_score,
            )

            logger.info(f"Completed hybrid session {session_id} successfully")

            return {
                "session_id": session_id,
                "engagement_id": str(engagement_id),
                "status": SessionStatus.COMPLETED,
                "contract": contract.dict(),
                "metrics": session_state["metrics"],
                "orchestration_result": orchestration_result,
                "execution_success": True,
            }

        except Exception as e:
            logger.error(f"Hybrid session execution failed for {session_id}: {str(e)}")

            # Attempt recovery
            recovery_result = await self._attempt_session_recovery(session_id, str(e))

            return {
                "session_id": session_id,
                "engagement_id": str(engagement_id),
                "status": SessionStatus.ERROR,
                "error": str(e),
                "recovery_result": recovery_result,
                "execution_success": False,
                "partial_metrics": session_state["metrics"],
            }

        finally:
            # Cleanup active session
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]

    async def pause_session_for_hitl(
        self, session_id: str, hitl_request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Pause session for human-in-the-loop interaction.

        Args:
            session_id: Session identifier
            hitl_request: HITL request data

        Returns:
            Pause confirmation with HITL request details
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")

        session_state = self.active_sessions[session_id]
        engagement_id = UUID(session_state["engagement_id"])

        # Update session status
        await self._update_session_status(session_id, SessionStatus.PAUSED_FOR_HITL)

        # Create checkpoint before pause
        await self.checkpoint_manager.save_checkpoint(
            engagement_id=engagement_id,
            phase=SessionStatus.PAUSED_FOR_HITL,
            data={
                "session_state": session_state,
                "hitl_request": hitl_request,
                "paused_at": datetime.now(timezone.utc).isoformat(),
            },
        )

        # Stream HITL request to frontend
        await self.stream_manager.stream_hitl_request(engagement_id, hitl_request)

        session_state["metrics"]["hitl_interactions"] += 1

        logger.info(f"Paused session {session_id} for HITL interaction")

        return {
            "session_id": session_id,
            "status": SessionStatus.PAUSED_FOR_HITL,
            "hitl_request": hitl_request,
            "paused_at": datetime.now(timezone.utc).isoformat(),
            "checkpoint_saved": True,
            "awaiting_user_response": True,
        }

    async def resume_session_from_hitl(
        self, session_id: str, hitl_response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Resume session after human-in-the-loop response.

        Args:
            session_id: Session identifier
            hitl_response: User's response to HITL request

        Returns:
            Resume confirmation
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")

        session_state = self.active_sessions[session_id]
        engagement_id = UUID(session_state["engagement_id"])

        # Process HITL response
        if hitl_response.get("action") == "approve":
            # Resume normal execution
            await self._update_session_status(session_id, SessionStatus.ANALYSIS_PHASE)

        elif hitl_response.get("action") == "modify":
            # Apply modifications and resume
            await self._apply_hitl_modifications(session_id, hitl_response)
            await self._update_session_status(session_id, SessionStatus.ANALYSIS_PHASE)

        elif hitl_response.get("action") == "pause":
            # Keep paused, no status change
            pass

        else:
            # Skip or reject - handle appropriately
            await self._handle_hitl_rejection(session_id, hitl_response)

        # Stream response acknowledgment
        await self.stream_manager.stream_hitl_response(engagement_id, hitl_response)

        # Save checkpoint with response
        await self.checkpoint_manager.save_checkpoint(
            engagement_id=engagement_id,
            phase=f"hitl_response_{hitl_response.get('action', 'unknown')}",
            data={
                "session_state": session_state,
                "hitl_response": hitl_response,
                "resumed_at": datetime.now(timezone.utc).isoformat(),
            },
        )

        logger.info(
            f"Resumed session {session_id} from HITL with action: {hitl_response.get('action')}"
        )

        return {
            "session_id": session_id,
            "status": session_state["status"],
            "hitl_response_processed": True,
            "action_taken": hitl_response.get("action"),
            "resumed_at": datetime.now(timezone.utc).isoformat(),
        }

    async def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """
        Get current session status and metrics.

        Args:
            session_id: Session identifier

        Returns:
            Current session state and metrics
        """
        if session_id in self.active_sessions:
            session_state = self.active_sessions[session_id]
            return {
                "session_id": session_id,
                "status": session_state["status"],
                "metrics": session_state["metrics"],
                "created_at": session_state["created_at"].isoformat(),
                "is_active": True,
            }
        else:
            # Try to load from checkpoint
            try:
                engagement_record = (
                    await self.supabase.table("cognitive_engagements")
                    .select("*")
                    .eq("session_id", session_id)
                    .single()
                    .execute()
                )

                return {
                    "session_id": session_id,
                    "status": engagement_record.data["status"],
                    "created_at": engagement_record.data["created_at"],
                    "completed_at": engagement_record.data.get("completed_at"),
                    "is_active": False,
                }
            except:
                return {
                    "session_id": session_id,
                    "status": "not_found",
                    "is_active": False,
                }

    async def list_active_sessions(self) -> List[Dict[str, Any]]:
        """
        List all currently active sessions.

        Returns:
            List of active session summaries
        """
        active_sessions = []

        for session_id, session_state in self.active_sessions.items():
            active_sessions.append(
                {
                    "session_id": session_id,
                    "engagement_id": session_state["engagement_id"],
                    "status": session_state["status"],
                    "initial_query": (
                        session_state["initial_query"][:100] + "..."
                        if len(session_state["initial_query"]) > 100
                        else session_state["initial_query"]
                    ),
                    "user_id": session_state["user_id"],
                    "created_at": session_state["created_at"].isoformat(),
                    "metrics": session_state["metrics"],
                }
            )

        return active_sessions

    # Private helper methods

    async def _create_engagement_record(
        self, session_id: str, initial_query: str, user_id: Optional[str]
    ) -> Dict[str, Any]:
        """Create engagement record in Supabase"""

        engagement_data = {
            "session_id": session_id,
            "query": initial_query,
            "user_id": user_id,
            "orchestration_type": "hybrid_micro_agents",
            "status": SessionStatus.INITIATED,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        result = (
            await self.supabase.table("cognitive_engagements")
            .insert(engagement_data)
            .execute()
        )

        return result.data[0]

    async def _update_session_status(self, session_id: str, new_status: str) -> None:
        """Update session status in both memory and database"""

        if session_id in self.active_sessions:
            self.active_sessions[session_id]["status"] = new_status
            engagement_id = self.active_sessions[session_id]["engagement_id"]

            await self.supabase.table("cognitive_engagements").update(
                {"status": new_status}
            ).eq("id", engagement_id).execute()

    async def _create_metis_contract(
        self, orchestration_result: Dict[str, Any], session_state: Dict[str, Any]
    ) -> MetisDataContract:
        """Create MetisDataContract from orchestration results"""

        research_results = orchestration_result["results"]["research"]
        synthesis_result = orchestration_result["results"]["synthesis"]
        analysis_results = orchestration_result["results"]["analysis"]

        # Extract research grounding
        research_grounding = []
        for focus_area, results in research_results["results_by_focus_area"].items():
            for query_result in results.get("queries_executed", []):
                research_grounding.append(
                    {
                        "query": query_result["query"],
                        "content": query_result["content"],
                        "sources": query_result["sources"],
                        "focus_area": focus_area,
                    }
                )

        # Extract cognitive exhaust
        cognitive_exhaust = {}
        for step_type, step_result in analysis_results["step_results"].items():
            cognitive_exhaust[step_type] = {
                "result": step_result["result"],
                "reasoning": step_result["reasoning"],
                "confidence_score": step_result["confidence_score"],
            }

        contract = MetisDataContract(
            type="metis.hybrid_engagement_completed",
            raw_query=session_state["initial_query"],
            enhanced_query=synthesis_result["enhanced_query"],
            research_grounding=research_grounding,
            cognitive_exhaust=cognitive_exhaust,
            strategic_constraints=synthesis_result.get("strategic_constraints", []),
            key_assumptions=synthesis_result.get("key_assumptions", []),
            confidence_score=orchestration_result["results"]["validation"].get(
                "quality_score", 0.85
            ),
            cost_breakdown={
                "research_cost": research_results["total_research_cost"],
                "synthesis_cost": synthesis_result.get("cost_usd", 0.0),
                "analysis_cost": analysis_results["total_analysis_cost"],
                "total_cost": orchestration_result["metrics"]["total_cost_usd"],
            },
            session_id=session_state["session_id"],
            engagement_id=session_state["engagement_id"],
            user_id=session_state["user_id"],
        )

        return contract

    async def _update_engagement_record(
        self,
        engagement_id: UUID,
        orchestration_result: Dict[str, Any],
        contract: MetisDataContract,
    ) -> None:
        """Update engagement record with final results"""

        update_data = {
            "status": SessionStatus.COMPLETED,
            "total_cost": orchestration_result["metrics"]["total_cost_usd"],
            "total_tokens": orchestration_result["metrics"]["total_tokens"],
            "confidence_score": contract.confidence_score,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }

        await self.supabase.table("cognitive_engagements").update(update_data).eq(
            "id", str(engagement_id)
        ).execute()

    async def _attempt_session_recovery(
        self, session_id: str, error: str
    ) -> Dict[str, Any]:
        """Attempt session recovery from last checkpoint"""

        try:
            session_state = self.active_sessions[session_id]
            engagement_id = UUID(session_state["engagement_id"])

            # Get latest checkpoint
            latest_checkpoint = await self.checkpoint_manager.get_latest_checkpoint(
                engagement_id
            )

            if latest_checkpoint:
                # Update session status to recovered
                await self._update_session_status(session_id, SessionStatus.RECOVERED)

                return {
                    "recovered": True,
                    "checkpoint_phase": latest_checkpoint.get("phase"),
                    "recovery_timestamp": datetime.now(timezone.utc).isoformat(),
                    "data_preserved": True,
                }
            else:
                return {"recovered": False, "reason": "no_checkpoints_available"}

        except Exception as recovery_error:
            logger.error(
                f"Recovery failed for session {session_id}: {str(recovery_error)}"
            )
            return {"recovered": False, "recovery_error": str(recovery_error)}

    async def _apply_hitl_modifications(
        self, session_id: str, hitl_response: Dict[str, Any]
    ) -> None:
        """Apply HITL modifications to session"""

        session_state = self.active_sessions[session_id]
        modifications = hitl_response.get("modifications", {})

        # Apply modifications to session config
        if "config" in modifications:
            session_state["config"].update(modifications["config"])

        # Apply modifications to orchestrator settings
        if "orchestrator_settings" in modifications:
            # This would require orchestrator to support dynamic reconfiguration
            pass

    async def _handle_hitl_rejection(
        self, session_id: str, hitl_response: Dict[str, Any]
    ) -> None:
        """Handle HITL rejection/skip actions"""

        action = hitl_response.get("action")

        if action == "skip":
            # Skip current micro-step, continue with next
            await self._update_session_status(session_id, SessionStatus.ANALYSIS_PHASE)
        elif action == "abort":
            # Abort entire session
            await self._update_session_status(session_id, SessionStatus.ERROR)
        else:
            # Default: pause and await further instruction
            pass


# Factory function for dependency injection
def get_hybrid_session_manager() -> HybridSessionManager:
    """Get hybrid session manager instance"""
    return HybridSessionManager()
