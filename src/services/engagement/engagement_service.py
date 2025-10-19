"""
Engagement Service - Business Logic Layer
=========================================

Extracts business logic from engagements.py route handlers following
Service Layer Extraction pattern (ADR-003).

This service handles:
- Engagement lifecycle management (start, status, pause/resume)
- Interactive mode question handling
- State validation and transitions

Target Complexity: CC ≤ 8 per method
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from uuid import uuid4
from datetime import datetime

from src.services.persistence import DatabaseService

logger = logging.getLogger(__name__)


class EngagementService:
    """
    Service for managing engagement lifecycle and state.

    Responsibilities:
    - Create and initialize new engagements
    - Fetch and merge engagement status from multiple sources
    - Validate interactive mode state transitions
    - Manage user answers and research question separation
    """

    def __init__(
        self,
        database_service: Optional[DatabaseService] = None,
        active_engagements: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        """
        Initialize EngagementService.

        Args:
            database_service: Optional database service for persistence
            active_engagements: Optional in-memory engagement store (for backward compatibility)
        """
        self.database_service = database_service
        self.active_engagements = active_engagements or {}

    def create_engagement(
        self,
        user_query: str,
        user_id: Optional[str] = None,
        enhanced_context: Optional[Dict[str, Any]] = None,
        quality_requested: Optional[int] = None,
        enhancement_questions_session_id: Optional[str] = None,
        answered_questions: Optional[List[Dict[str, Any]]] = None,
        research_questions: Optional[List[Dict[str, Any]]] = None,
        quality_target: Optional[float] = None,
        interactive_mode: bool = False,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Create a new engagement with initialization.

        Complexity: CC ≤ 6 (linear flow with minimal branching)

        Args:
            user_query: The strategic question to analyze
            user_id: Optional user identifier
            enhanced_context: Enhanced context from query enhancement flow
            quality_requested: Quality level requested (60-95%)
            enhancement_questions_session_id: Session ID from progressive questions
            answered_questions: Questions answered by user
            research_questions: Questions to be researched by system
            quality_target: Target quality score 0.5-0.95
            interactive_mode: If true, pause after generating questions

        Returns:
            Tuple of (trace_id, engagement_state)
        """
        # Generate new trace ID
        trace_id = str(uuid4())

        # Determine if this is an enhanced analysis
        is_enhanced = bool(enhanced_context)
        quality_level = quality_requested or 60

        # Log initialization
        logger.info(f"🚀 Creating engagement - Trace ID: {trace_id}")
        logger.info(f"📝 User Query: {user_query[:100]}...")

        # Log progressive questions enhancement if present
        if answered_questions or research_questions:
            answered_count = len(answered_questions or [])
            research_count = len(research_questions or [])
            logger.info(f"✨ Progressive Questions Enhancement:")
            logger.info(f"   📋 {answered_count} questions answered by user")
            logger.info(f"   🔍 {research_count} questions flagged for research")
            if quality_target:
                logger.info(f"   🎯 Target quality: {int(quality_target * 100)}%")

        if is_enhanced:
            logger.info(f"✨ Enhanced Analysis: {quality_level}% quality requested")

        # Determine total stages
        total_stages = self._get_total_stages()

        # Create engagement state
        engagement_state = {
            "trace_id": trace_id,
            "user_query": user_query,
            "user_id": user_id,
            "enhanced_context": enhanced_context,
            "quality_requested": quality_level,
            "is_enhanced": is_enhanced,
            "answered_questions": answered_questions,
            "research_questions": research_questions,
            "quality_target": quality_target,
            "enhancement_questions_session_id": enhancement_questions_session_id,
            "interactive_mode": interactive_mode,
            "started_at": datetime.now(),
            "status": "INITIALIZING",
            "current_stage": "SOCRATIC_QUESTIONS",
            "stage_number": 1,
            "total_stages": total_stages,
            "progress_percentage": 0.0,
            "is_completed": False,
            "error": None,
            "final_output": None
        }

        # Store in active engagements
        self.active_engagements[trace_id] = engagement_state

        logger.info(f"✅ Engagement created - State initialized")

        return trace_id, engagement_state

    async def get_engagement_status(
        self,
        trace_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get engagement status from database with in-memory fallback.

        Complexity: CC ≤ 4 (refactored with helper methods)

        Args:
            trace_id: The engagement trace ID

        Returns:
            Engagement status dict or None if not found
        """
        # Try database first, then fallback to in-memory
        engagement, db_version = await self._fetch_status_from_sources(trace_id)

        # Merge in-memory questions with database status
        if engagement and trace_id in self.active_engagements:
            engagement = self._merge_in_memory_data(engagement, trace_id)

        # Add version to engagement for ETag generation
        if engagement:
            engagement["_db_version"] = db_version or 0

        return engagement

    async def _fetch_status_from_sources(
        self,
        trace_id: str,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[int]]:
        """
        Fetch engagement status from database or in-memory store.

        Complexity: CC ≤ 4

        Args:
            trace_id: The engagement trace ID

        Returns:
            Tuple of (engagement_dict, version)
        """
        engagement = None
        db_version = None

        # Try database first (single source of truth)
        if self.database_service:
            try:
                db_status = await self.database_service.get_engagement_status_async(trace_id)
                if db_status:
                    engagement = db_status
                    db_version = db_status.get("version", 0)
                    logger.info(f"📊 DATABASE READ: status={engagement.get('status')}, version={db_version}, trace={trace_id[:8]}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to read from database, falling back to in-memory: {e}")

        # Fallback to in-memory dict (backward compatibility)
        if not engagement and trace_id in self.active_engagements:
            engagement = self.active_engagements[trace_id]
            db_version = -1  # Use version=-1 for in-memory-only status
            logger.debug(f"📊 In-memory status: {engagement.get('status')}")

        return engagement, db_version

    def _merge_in_memory_data(
        self,
        engagement: Dict[str, Any],
        trace_id: str,
    ) -> Dict[str, Any]:
        """
        Merge in-memory questions with database status.

        Complexity: CC = 3

        Args:
            engagement: The engagement dict to enrich
            trace_id: The engagement trace ID

        Returns:
            Enriched engagement dict
        """
        in_memory = self.active_engagements[trace_id]

        # Enrich database status with questions from in-memory
        if "generated_questions" in in_memory:
            engagement["generated_questions"] = in_memory["generated_questions"]
        if "paused_checkpoint_id" in in_memory:
            engagement["paused_checkpoint_id"] = in_memory["paused_checkpoint_id"]

        logger.info(f"📊 MERGE: Added {len(engagement.get('generated_questions', []))} questions to DB status, trace={trace_id[:8]}")

        return engagement

    def generate_etag(self, trace_id: str, version: int) -> str:
        """
        Generate ETag for engagement status.

        Complexity: CC = 1 (no branching)

        Args:
            trace_id: The engagement trace ID
            version: The version number from database

        Returns:
            ETag string in format W/"<trace_id>:<version>"
        """
        return f'W/"{trace_id}:{version}"'

    def get_generated_questions(
        self,
        trace_id: str,
    ) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str], Optional[str]]:
        """
        Get generated questions for interactive mode.

        Complexity: CC ≤ 5 (validation checks + extraction)

        Args:
            trace_id: The engagement trace ID

        Returns:
            Tuple of (questions, checkpoint_id, error_message)
            Error message is None if successful
        """
        # Check if engagement exists
        if trace_id not in self.active_engagements:
            return None, None, "Engagement not found"

        engagement = self.active_engagements[trace_id]

        # Check if this is an interactive mode engagement
        if not engagement.get("interactive_mode"):
            return None, None, "This engagement is not in interactive mode"

        # Check if the engagement is paused for user input
        status = engagement.get("status")
        if status != "PAUSED_FOR_USER_INPUT":
            return None, None, f"Engagement is not paused for user input. Current status: {status}"

        # Get questions from the engagement state
        questions = engagement.get("generated_questions", [])
        checkpoint_id = engagement.get("paused_checkpoint_id")

        return questions, checkpoint_id, None

    def prepare_answer_submission(
        self,
        trace_id: str,
        answers: List[Dict[str, str]],
    ) -> Tuple[Optional[List[Dict[str, Any]]], Optional[List[Dict[str, Any]]], Optional[str], Optional[str]]:
        """
        Prepare answer submission by separating answered vs research questions.

        Complexity: CC ≤ 5 (refactored with helper methods)

        Args:
            trace_id: The engagement trace ID
            answers: List of answer submissions with question_id and answer

        Returns:
            Tuple of (answered_questions, research_questions, checkpoint_id, error_message)
            Error message is None if successful
        """
        # Validate engagement state
        error = self._validate_answer_submission(trace_id)
        if error:
            return None, None, None, error

        engagement = self.active_engagements[trace_id]

        logger.info(f"📝 Processing {len(answers)} answers for trace {trace_id}")

        # Separate answered vs. research-delegated questions
        answered_questions, research_questions = self._process_answers(
            answers, engagement.get("generated_questions", [])
        )

        # Update engagement with answers and research requests
        engagement["user_answers"] = answered_questions
        engagement["user_research_requests"] = research_questions
        engagement["status"] = "RESUMING"

        # Get checkpoint info
        checkpoint_id = engagement.get("paused_checkpoint_id")
        if not checkpoint_id:
            return None, None, None, "No checkpoint found to resume from"

        return answered_questions, research_questions, checkpoint_id, None

    def _validate_answer_submission(self, trace_id: str) -> Optional[str]:
        """
        Validate engagement state for answer submission.

        Complexity: CC = 4

        Args:
            trace_id: The engagement trace ID

        Returns:
            Error message if validation fails, None if successful
        """
        # Check if engagement exists
        if trace_id not in self.active_engagements:
            return "Engagement not found"

        engagement = self.active_engagements[trace_id]

        # Check if this is an interactive mode engagement
        if not engagement.get("interactive_mode"):
            return "This engagement is not in interactive mode"

        # Check if the engagement is paused for user input
        status = engagement.get("status")
        if status != "PAUSED_FOR_USER_INPUT":
            return f"Engagement is not paused for user input. Current status: {status}"

        return None

    def _process_answers(
        self,
        answers: List[Dict[str, str]],
        generated_questions: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Process answers and separate into answered vs research questions.

        Complexity: CC = 4

        Args:
            answers: List of answer submissions
            generated_questions: List of generated questions

        Returns:
            Tuple of (answered_questions, research_questions)
        """
        RAW_RESEARCH_TOKEN = "[REQUEST_RESEARCH]"
        answered_questions = []
        research_questions = []

        # Build a lookup of generated questions to recover question_text for research items
        q_by_id = {str(q.get("id")): q for q in generated_questions if isinstance(q, dict) and q.get("id")}

        for ans in answers:
            answer_text = ans.get("answer", "")
            if answer_text.strip() == RAW_RESEARCH_TOKEN:
                q = q_by_id.get(str(ans.get("question_id")))
                q_text = (q.get("question") if isinstance(q, dict) else None) or ""
                research_questions.append({
                    "question_id": ans.get("question_id"),
                    "question_text": q_text,
                })
            else:
                # Only include non-empty user answers
                if answer_text.strip():
                    answered_questions.append({
                        "question_id": ans.get("question_id"),
                        "answer": answer_text,
                    })

        return answered_questions, research_questions

    async def update_status_to_resuming(
        self,
        trace_id: str,
    ) -> bool:
        """
        Atomically update engagement status to RESUMING in database.

        Complexity: CC ≤ 3 (database write with error handling)

        Args:
            trace_id: The engagement trace ID

        Returns:
            True if successful, False otherwise
        """
        if not self.database_service:
            return False

        engagement = self.active_engagements.get(trace_id)
        if not engagement:
            return False

        try:
            await self.database_service.upsert_engagement_status_async({
                "trace_id": trace_id,
                "status": "RESUMING",
                "current_stage": engagement.get("current_stage"),
                "stage_number": engagement.get("stage_number"),
                "progress_percentage": engagement.get("progress_percentage", 0.0),
                "user_id": engagement.get("user_id"),
                "session_id": engagement.get("session_id"),
            })
            logger.info(f"✅ Database status updated to RESUMING for trace {trace_id}")
            return True
        except Exception as e:
            logger.error(f"⚠️ Failed to update database status: {e}")
            return False

    def _get_total_stages(self) -> int:
        """
        Get total stages for UI display.

        Complexity: CC = 2 (try/except)

        Returns:
            Total number of stages (default 8)
        """
        try:
            from src.core.stage_progress import total_stages_for_ui
            return total_stages_for_ui()
        except Exception:
            return 8
