"""
Human-In-The-Loop (HITL) Interaction Manager

Manages the clarification dialogue flow between the system and users.
Handles question presentation, response collection, and context enrichment.

Features:
- Adaptive clarification triggering
- Question presentation formatting
- Response validation and processing
- Context enrichment and integration
- Interaction history tracking
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

from src.engine.core.query_clarification_engine import (
    QueryClarificationEngine,
    QueryAnalysisResult,
    ClarificationQuestion,
    ClarificationResponse,
)


class InteractionState(Enum):
    """States of the HITL interaction process"""

    INITIAL = "initial"
    ANALYZING = "analyzing"
    QUESTIONS_READY = "questions_ready"
    AWAITING_RESPONSES = "awaiting_responses"
    RESPONSES_RECEIVED = "responses_received"
    ENRICHING_QUERY = "enriching_query"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class InteractionSession:
    """Represents a single HITL clarification session"""

    session_id: str
    original_query: str
    business_context: Dict[str, Any]
    state: InteractionState
    analysis_result: Optional[QueryAnalysisResult] = None
    presented_questions: List[ClarificationQuestion] = field(default_factory=list)
    user_responses: List[ClarificationResponse] = field(default_factory=list)
    enhanced_query: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ClarificationPrompt:
    """Formatted presentation of clarification questions for UI"""

    session_id: str
    title: str
    description: str
    questions: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    estimated_time_minutes: int
    skip_option_available: bool = True


class HITLInteractionManager:
    """
    Manages the Human-In-The-Loop clarification dialogue flow.

    Coordinates between query analysis, question presentation, response collection,
    and context enrichment to improve query understanding before analysis.
    """

    def __init__(self, clarification_engine: Optional[QueryClarificationEngine] = None):
        """Initialize the HITL interaction manager."""
        self.logger = logging.getLogger(__name__)
        self.clarification_engine = clarification_engine or QueryClarificationEngine()

        # Configuration
        self.auto_clarification_enabled = True
        self.min_clarity_threshold = 0.7
        self.max_session_duration_minutes = 30
        self.default_skip_allowed = True

        # Active sessions storage (in production, use Redis/database)
        self.active_sessions: Dict[str, InteractionSession] = {}

        # Interaction patterns configuration
        self.interaction_patterns = {
            "business_critical": {
                "force_clarification": True,
                "min_questions": 3,
                "skip_allowed": False,
            },
            "standard": {
                "force_clarification": False,
                "min_questions": 2,
                "skip_allowed": True,
            },
            "quick_analysis": {
                "force_clarification": False,
                "min_questions": 1,
                "skip_allowed": True,
            },
        }

        self.logger.info("🤝 HITLInteractionManager initialized")

    async def initiate_clarification_session(
        self,
        query: str,
        business_context: Optional[Dict[str, Any]] = None,
        interaction_pattern: str = "standard",
        user_id: Optional[str] = None,
    ) -> Tuple[str, bool, Optional[ClarificationPrompt]]:
        """
        Initiate a clarification session for a user query.

        Args:
            query: The user's original query
            business_context: Optional business context
            interaction_pattern: The interaction pattern to use
            user_id: Optional user identifier

        Returns:
            Tuple of (session_id, needs_clarification, clarification_prompt)
        """
        session_id = str(uuid.uuid4())
        context = business_context or {}

        self.logger.info(
            f"🚀 Initiating clarification session {session_id[:8]} for query: '{query[:100]}...'"
        )

        # Create session
        session = InteractionSession(
            session_id=session_id,
            original_query=query,
            business_context=context,
            state=InteractionState.ANALYZING,
            metadata={
                "user_id": user_id,
                "interaction_pattern": interaction_pattern,
                "initiated_at": datetime.utcnow().isoformat(),
            },
        )

        self.active_sessions[session_id] = session

        try:
            # Analyze query for clarity
            analysis_result = await self.clarification_engine.analyze_query(
                query, context
            )
            session.analysis_result = analysis_result

            # Determine if clarification is needed
            pattern_config = self.interaction_patterns.get(
                interaction_pattern, self.interaction_patterns["standard"]
            )
            needs_clarification = self._should_request_clarification(
                analysis_result, pattern_config
            )

            if needs_clarification:
                # Generate clarification prompt
                clarification_prompt = self._create_clarification_prompt(
                    session, pattern_config
                )
                session.state = InteractionState.QUESTIONS_READY
                session.presented_questions = analysis_result.recommended_questions

                self.logger.info(
                    f"✅ Clarification needed for session {session_id[:8]} - "
                    f"{len(analysis_result.recommended_questions)} questions prepared"
                )

                return session_id, True, clarification_prompt
            else:
                # No clarification needed - mark as completed
                session.state = InteractionState.COMPLETED
                session.enhanced_query = query  # Use original query

                self.logger.info(
                    f"✅ No clarification needed for session {session_id[:8]} - "
                    f"clarity score: {analysis_result.overall_clarity_score:.2f}"
                )

                return session_id, False, None

        except Exception as e:
            self.logger.error(
                f"❌ Failed to analyze query for session {session_id[:8]}: {e}"
            )
            session.state = InteractionState.FAILED
            session.metadata["error"] = str(e)
            return session_id, False, None

    def _should_request_clarification(
        self, analysis_result: QueryAnalysisResult, pattern_config: Dict[str, Any]
    ) -> bool:
        """Determine if clarification should be requested based on analysis and pattern."""
        # Force clarification if pattern requires it
        if pattern_config.get("force_clarification", False):
            return True

        # Skip clarification if disabled
        if not self.auto_clarification_enabled:
            return False

        # Check clarity threshold
        if analysis_result.overall_clarity_score < self.min_clarity_threshold:
            return True

        # Check if sufficient high-impact questions are available
        high_impact_questions = [
            q for q in analysis_result.recommended_questions if q.impact_score > 0.7
        ]

        min_questions = pattern_config.get("min_questions", 2)
        return len(high_impact_questions) >= min_questions

    def _create_clarification_prompt(
        self, session: InteractionSession, pattern_config: Dict[str, Any]
    ) -> ClarificationPrompt:
        """Create a formatted clarification prompt for the UI."""
        questions = session.analysis_result.recommended_questions

        # Format questions for UI presentation
        formatted_questions = []
        for i, question in enumerate(questions):
            formatted_question = {
                "id": f"q_{i}",
                "question": question.question,
                "type": question.question_type,
                "dimension": question.dimension,
                "complexity": question.complexity_level,
                "context_hint": question.context_hint,
                "required": question.impact_score
                > 0.8,  # High impact questions are required
                "placeholder": self._generate_placeholder_text(question),
                # Operation Crystal Day 3: Include rationale in UI formatting
                "rationale": question.rationale,
                "validation_rules": self._generate_validation_rules(question),
            }
            formatted_questions.append(formatted_question)

        # Calculate estimated time
        estimated_time = self._estimate_completion_time(questions)

        # Create title and description
        title = self._generate_prompt_title(
            session.analysis_result.overall_clarity_score
        )
        description = self._generate_prompt_description(session, questions)

        return ClarificationPrompt(
            session_id=session.session_id,
            title=title,
            description=description,
            questions=formatted_questions,
            metadata={
                "clarity_score": session.analysis_result.overall_clarity_score,
                "total_dimensions": len(session.analysis_result.dimensions),
                "pattern": session.metadata.get("interaction_pattern"),
            },
            estimated_time_minutes=estimated_time,
            skip_option_available=pattern_config.get("skip_allowed", True),
        )

    def _generate_placeholder_text(self, question: ClarificationQuestion) -> str:
        """Generate helpful placeholder text for questions."""
        placeholders = {
            "business_objective": "e.g., Increase revenue by 15%, reduce costs by $2M, improve customer satisfaction...",
            "timeline_urgency": "e.g., Next quarter, by end of fiscal year, ASAP, flexible timeline...",
            "scope_boundaries": "e.g., North America only, all product lines, specific departments...",
            "stakeholder_context": "e.g., CEO approval required, marketing team involvement, customer input needed...",
            "success_metrics": "e.g., Revenue growth, cost reduction, efficiency metrics, customer satisfaction scores...",
            "constraints_limitations": "e.g., Budget limit $500K, no headcount increase, regulatory compliance required...",
            "resource_context": "e.g., Existing team capacity, budget allocation, technology constraints...",
            "decision_authority": "e.g., VP approval needed, board decision, team consensus required...",
        }

        return placeholders.get(
            question.dimension, "Please provide as much detail as possible..."
        )

    def _generate_validation_rules(
        self, question: ClarificationQuestion
    ) -> Dict[str, Any]:
        """Generate validation rules for questions."""
        base_rules = {
            "min_length": 10,
            "max_length": 500,
            "required": question.impact_score > 0.8,
        }

        if question.question_type == "numeric":
            base_rules.update({"type": "number", "min_value": 0})
        elif question.question_type == "yes_no":
            base_rules.update(
                {"type": "boolean", "options": ["Yes", "No", "Uncertain"]}
            )

        return base_rules

    def _estimate_completion_time(self, questions: List[ClarificationQuestion]) -> int:
        """Estimate time required to complete clarification questions."""
        time_estimates = {
            "simple": 1,  # 1 minute
            "medium": 2,  # 2 minutes
            "complex": 4,  # 4 minutes
        }

        total_minutes = sum(
            time_estimates.get(q.complexity_level, 2) for q in questions
        )

        # Add base overhead
        return max(3, total_minutes + 1)

    def _generate_prompt_title(self, clarity_score: float) -> str:
        """Generate appropriate title based on clarity score."""
        if clarity_score < 0.4:
            return "Help Us Understand Your Challenge"
        elif clarity_score < 0.6:
            return "Quick Clarification Needed"
        else:
            return "Just a Few Quick Questions"

    def _generate_prompt_description(
        self, session: InteractionSession, questions: List[ClarificationQuestion]
    ) -> str:
        """Generate description explaining the clarification process."""
        clarity_score = session.analysis_result.overall_clarity_score

        if clarity_score < 0.4:
            urgency_text = "To provide you with the most accurate analysis, we need to understand your challenge better."
        elif clarity_score < 0.6:
            urgency_text = "Your request is mostly clear, but a few clarifications will help us deliver better insights."
        else:
            urgency_text = "We have a good understanding of your request, but these questions will help us be more precise."

        return f"""
{urgency_text}

These questions focus on the most important aspects that will impact our analysis quality. 
Your answers will be used to provide more targeted and relevant strategic recommendations.

**Current understanding level: {clarity_score:.0%}**
        """.strip()

    async def process_clarification_responses(
        self, session_id: str, responses: List[Dict[str, Any]]
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Process user responses to clarification questions.

        Args:
            session_id: The clarification session ID
            responses: List of user responses

        Returns:
            Tuple of (success, enhanced_query, error_message)
        """
        if session_id not in self.active_sessions:
            return False, None, "Session not found"

        session = self.active_sessions[session_id]

        if session.state != InteractionState.QUESTIONS_READY:
            return False, None, f"Session not in correct state: {session.state.value}"

        self.logger.info(
            f"🔄 Processing {len(responses)} clarification responses for session {session_id[:8]}"
        )

        try:
            session.state = InteractionState.RESPONSES_RECEIVED

            # Convert responses to ClarificationResponse objects
            clarification_responses = []
            for i, response_data in enumerate(responses):
                # Find the corresponding question
                question = (
                    session.presented_questions[i]
                    if i < len(session.presented_questions)
                    else None
                )

                clarification_response = ClarificationResponse(
                    question_id=response_data.get("question_id", f"q_{i}"),
                    question=question.question if question else "Unknown question",
                    response=response_data.get("response", ""),
                    confidence=float(response_data.get("confidence", 1.0)),
                )
                clarification_responses.append(clarification_response)

            session.user_responses = clarification_responses
            session.state = InteractionState.ENRICHING_QUERY

            # Generate enhanced query
            enhanced_query = (
                self.clarification_engine.enhance_query_with_clarifications(
                    session.original_query,
                    clarification_responses,
                    session.business_context,
                )
            )

            session.enhanced_query = enhanced_query
            session.state = InteractionState.COMPLETED
            session.updated_at = datetime.utcnow()

            # Add completion metadata
            session.metadata.update(
                {
                    "completed_at": datetime.utcnow().isoformat(),
                    "responses_count": len(clarification_responses),
                    "enhancement_successful": True,
                }
            )

            self.logger.info(
                f"✅ Clarification session {session_id[:8]} completed successfully"
            )

            return True, enhanced_query, None

        except Exception as e:
            self.logger.error(
                f"❌ Failed to process clarification responses for session {session_id[:8]}: {e}"
            )
            session.state = InteractionState.FAILED
            session.metadata["error"] = str(e)
            return False, None, str(e)

    def skip_clarification(
        self, session_id: str, reason: str = "user_skip"
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Skip clarification and use original query.

        Args:
            session_id: The clarification session ID
            reason: Reason for skipping

        Returns:
            Tuple of (success, original_query, error_message)
        """
        if session_id not in self.active_sessions:
            return False, None, "Session not found"

        session = self.active_sessions[session_id]

        self.logger.info(
            f"⏭️ Skipping clarification for session {session_id[:8]}, reason: {reason}"
        )

        session.state = InteractionState.SKIPPED
        session.enhanced_query = session.original_query  # Use original query
        session.updated_at = datetime.utcnow()
        session.metadata.update(
            {"skipped_at": datetime.utcnow().isoformat(), "skip_reason": reason}
        )

        return True, session.original_query, None

    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a clarification session."""
        if session_id not in self.active_sessions:
            return None

        session = self.active_sessions[session_id]

        return {
            "session_id": session_id,
            "state": session.state.value,
            "original_query": session.original_query,
            "questions_count": len(session.presented_questions),
            "responses_count": len(session.user_responses),
            "clarity_score": (
                session.analysis_result.overall_clarity_score
                if session.analysis_result
                else None
            ),
            "enhanced_query_available": session.enhanced_query is not None,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "metadata": session.metadata,
        }

    def cleanup_expired_sessions(self, max_age_hours: int = 24) -> int:
        """Clean up expired sessions."""
        cutoff_time = datetime.utcnow().timestamp() - (max_age_hours * 3600)
        expired_sessions = []

        for session_id, session in self.active_sessions.items():
            if session.created_at.timestamp() < cutoff_time:
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            del self.active_sessions[session_id]

        if expired_sessions:
            self.logger.info(
                f"🧹 Cleaned up {len(expired_sessions)} expired clarification sessions"
            )

        return len(expired_sessions)

    def get_session_analytics(self) -> Dict[str, Any]:
        """Get analytics about clarification sessions."""
        total_sessions = len(self.active_sessions)
        states_count = {}

        for session in self.active_sessions.values():
            state = session.state.value
            states_count[state] = states_count.get(state, 0) + 1

        return {
            "total_active_sessions": total_sessions,
            "sessions_by_state": states_count,
            "clarification_enabled": self.auto_clarification_enabled,
            "clarity_threshold": self.min_clarity_threshold,
        }


# Factory function for easy instantiation
def create_hitl_interaction_manager(
    clarification_engine: Optional[QueryClarificationEngine] = None,
) -> HITLInteractionManager:
    """Create and configure a HITLInteractionManager instance."""
    return HITLInteractionManager(clarification_engine)
