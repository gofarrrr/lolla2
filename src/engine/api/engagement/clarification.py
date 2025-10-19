"""
HITL (Human-in-the-Loop) Clarification endpoints for METIS Engagement API
"""

import logging
from typing import Dict, Any

from fastapi import HTTPException

# Import for Glass Box event emission
from src.core.unified_context_stream import UnifiedContextStream, ContextEventType

from .models import (
    ClarificationRequest,
    ClarificationResult,
    ClarificationResponseRequest,
    ClarificationSkipRequest,
    EnhancedQueryResult,
)

try:
    from src.engine.core.query_clarification_engine import QueryClarificationEngine
    from src.core.hitl_interaction_manager import (
        HITLInteractionManager,
        ClarificationPrompt,
    )

    HITL_AVAILABLE = True
except ImportError:
    HITL_AVAILABLE = False
    QueryClarificationEngine = None
    HITLInteractionManager = None
    ClarificationPrompt = None


class ClarificationHandler:
    """Handler for HITL clarification workflows"""

    def __init__(self):
        self.clarification_engine = (
            QueryClarificationEngine() if HITL_AVAILABLE else None
        )
        self.hitl_manager = (
            HITLInteractionManager(self.clarification_engine)
            if HITL_AVAILABLE
            else None
        )
        self.logger = logging.getLogger(__name__)

    async def analyze_clarification_needs(
        self, request: ClarificationRequest
    ) -> ClarificationResult:
        """Analyze if a query needs clarification and generate clarifying questions"""

        if not HITL_AVAILABLE:
            # Return simulation response when HITL system is not available
            return ClarificationResult(
                session_id="simulation-123",
                needs_clarification=False,
                clarity_score=0.8,
                questions=None,
                estimated_time_minutes=0,
                skip_option_available=True,
                title="Simulation Mode",
                description="HITL clarification system not available - using simulation",
            )

        try:
            # Create clarification prompt
            clarification_prompt = ClarificationPrompt(
                original_query=request.query,
                business_context=request.business_context,
                interaction_pattern=request.interaction_pattern,
                user_id=request.user_id,
            )

            # Analyze clarification needs
            result = await self.hitl_manager.analyze_clarification_needs(
                clarification_prompt
            )

            # Convert to API format
            return ClarificationResult(
                session_id=result.session_id,
                needs_clarification=result.needs_clarification,
                clarity_score=result.clarity_score,
                questions=result.questions,
                estimated_time_minutes=result.estimated_time_minutes,
                skip_option_available=result.skip_option_available,
                title=result.title,
                description=result.description,
            )

        except Exception as e:
            self.logger.error(f"Clarification analysis failed: {e}")
            raise HTTPException(
                status_code=500, detail=f"Clarification analysis failed: {str(e)}"
            )

    async def process_clarification_responses(
        self, request: ClarificationResponseRequest
    ) -> EnhancedQueryResult:
        """Process user responses to clarification questions and enhance the original query"""

        if not HITL_AVAILABLE:
            # Return simulation response
            return EnhancedQueryResult(
                session_id=request.session_id,
                success=True,
                enhanced_query="Simulated enhanced query based on responses",
                original_query="Original query not available in simulation",
                error_message=None,
                clarification_count=len(request.responses),
            )

        try:
            # Convert API responses to internal format
            internal_responses = []
            for response in request.responses:
                internal_responses.append(
                    {
                        "question_id": response.question_id,
                        "response": response.response,
                        "confidence": response.confidence,
                    }
                )

            # Process responses through HITL manager
            result = await self.hitl_manager.process_clarification_responses(
                session_id=request.session_id, responses=internal_responses
            )

            # Convert to API format
            return EnhancedQueryResult(
                session_id=request.session_id,
                success=result.success,
                enhanced_query=result.enhanced_query,
                original_query=result.original_query,
                error_message=result.error_message,
                clarification_count=len(request.responses),
            )

        except Exception as e:
            self.logger.error(f"Clarification response processing failed: {e}")
            raise HTTPException(
                status_code=500, detail=f"Response processing failed: {str(e)}"
            )

    async def skip_clarification(
        self, request: ClarificationSkipRequest
    ) -> EnhancedQueryResult:
        """Skip clarification process and proceed with original query"""

        if not HITL_AVAILABLE:
            # Return simulation response
            return EnhancedQueryResult(
                session_id=request.session_id,
                success=True,
                enhanced_query=None,  # No enhancement when skipped
                original_query="Original query (simulation mode)",
                error_message=None,
                clarification_count=0,
            )

        try:
            # Process skip request through HITL manager
            result = await self.hitl_manager.skip_clarification(
                session_id=request.session_id, reason=request.reason
            )

            return EnhancedQueryResult(
                session_id=request.session_id,
                success=result.success,
                enhanced_query=result.enhanced_query,
                original_query=result.original_query,
                error_message=result.error_message,
                clarification_count=0,
            )

        except Exception as e:
            self.logger.error(f"Clarification skip failed: {e}")
            raise HTTPException(
                status_code=500, detail=f"Skip processing failed: {str(e)}"
            )

    async def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get the current status of a clarification session"""

        if not HITL_AVAILABLE:
            return {
                "session_id": session_id,
                "status": "simulation",
                "message": "HITL system not available",
            }

        try:
            status = await self.hitl_manager.get_session_status(session_id)
            return status

        except Exception as e:
            self.logger.error(f"Session status retrieval failed: {e}")
            raise HTTPException(
                status_code=500, detail=f"Status retrieval failed: {str(e)}"
            )


# NEW: V2 Tiered Clarification Handler
class TieredClarificationHandler:
    """Handler for V2 tiered, conversational clarification workflows"""

    def __init__(self):

        self.clarification_engine = (
            QueryClarificationEngine() if HITL_AVAILABLE else None
        )
        self.logger = logging.getLogger(__name__)

        # Session storage (in production, use Redis/database)
        self.active_sessions: Dict[str, Any] = {}

        self.logger.info("🎯 TieredClarificationHandler initialized")

    async def start_tiered_clarification(self, request):
        """
        NEW: Start tiered clarification process with engagement brief + essential questions

        Args:
            request: TieredClarificationStartRequest

        Returns:
            TieredClarificationStartResponse
        """
        from .models import (
            TieredClarificationStartResponse,
            EngagementBriefModel,
            TieredClarificationQuestion,
            QuestionTier,
        )
        import uuid

        if not HITL_AVAILABLE:
            # Simulation response
            session_id = str(uuid.uuid4())

            return TieredClarificationStartResponse(
                clarification_session_id=session_id,
                engagement_brief=EngagementBriefModel(
                    objective="Simulated strategic analysis",
                    platform="Demo platform",
                    key_features=["Simulation", "Testing"],
                    confidence=0.5,
                ),
                essential_questions=[
                    TieredClarificationQuestion(
                        id="q1",
                        question="What is your primary business objective?",
                        tier=QuestionTier.ESSENTIAL,
                        dimension="business_objective",
                        rationale="Understanding your goal helps focus the analysis",
                    )
                ],
                estimated_time_minutes=2,
            )

        try:
            self.logger.info(
                f"🚀 Starting tiered clarification for: '{request.raw_query[:100]}...'"
            )

            # Generate tiered questions using new engine method
            tiered_questions = (
                await self.clarification_engine.generate_tiered_questions(
                    raw_query=request.raw_query,
                    business_context=request.business_context,
                    user_expertise=request.user_expertise,
                )
            )

            # Create session
            session_id = str(uuid.uuid4())

            # Store session data
            self.active_sessions[session_id] = {
                "raw_query": request.raw_query,
                "business_context": request.business_context,
                "user_expertise": request.user_expertise,
                "tiered_questions": tiered_questions,
                "essential_answers": {},
                "expert_answers": {},
                "created_at": datetime.utcnow().isoformat(),
                "status": "essential_questions_ready",
            }

            # Convert to API format
            essential_questions = []
            for i, q in enumerate(tiered_questions.essential_questions):
                essential_questions.append(
                    TieredClarificationQuestion(
                        id=f"essential_{i}",
                        question=q.question,
                        tier=QuestionTier.ESSENTIAL,
                        dimension=q.dimension,
                        question_type=q.question_type,
                        complexity_level=q.complexity_level,
                        context_hint=q.context_hint,
                        rationale=q.rationale,
                        grounded_context=q.grounded_context,
                    )
                )

            # Convert engagement brief
            engagement_brief = EngagementBriefModel(
                objective=tiered_questions.engagement_brief.objective,
                platform=tiered_questions.engagement_brief.platform,
                key_features=tiered_questions.engagement_brief.key_features,
                confidence=tiered_questions.engagement_brief.confidence,
            )

            response = TieredClarificationStartResponse(
                clarification_session_id=session_id,
                engagement_brief=engagement_brief,
                essential_questions=essential_questions,
                estimated_time_minutes=len(essential_questions)
                * 1,  # 1 minute per question
            )

            self.logger.info(
                f"✅ Tiered clarification started - Session: {session_id[:8]}, "
                f"Brief: {engagement_brief.objective[:50]}..., "
                f"Essential questions: {len(essential_questions)}"
            )

            return response

        except Exception as e:
            self.logger.error(f"❌ Tiered clarification start failed: {e}")
            raise HTTPException(
                status_code=500, detail=f"Clarification start failed: {str(e)}"
            )

    async def continue_with_expert_questions(self, request):
        """
        NEW: Continue with expert questions after essential answers received

        Args:
            request: TieredClarificationContinueRequest

        Returns:
            TieredClarificationContinueResponse
        """
        from .models import (
            TieredClarificationContinueResponse,
            TieredClarificationQuestion,
            QuestionTier,
        )

        if not HITL_AVAILABLE:
            return TieredClarificationContinueResponse(
                session_id=request.clarification_session_id,
                status="Essential questions answered. Ready for deep dive.",
                expert_questions=[
                    TieredClarificationQuestion(
                        id="expert_1",
                        question="What strategic advantages are you looking to create?",
                        tier=QuestionTier.EXPERT,
                        dimension="strategic_positioning",
                        rationale="Strategic positioning helps identify competitive advantages",
                    )
                ],
            )

        try:
            session_id = request.clarification_session_id

            if session_id not in self.active_sessions:
                raise HTTPException(
                    status_code=404, detail="Clarification session not found"
                )

            session_data = self.active_sessions[session_id]

            # Store essential answers
            for answer in request.essential_answers:
                session_data["essential_answers"][answer.question_id] = {
                    "answer": answer.answer,
                    "confidence": answer.confidence,
                }

            # Get expert questions from stored tiered questions
            tiered_questions = session_data["tiered_questions"]

            expert_questions = []
            for i, q in enumerate(tiered_questions.expert_questions):
                expert_questions.append(
                    TieredClarificationQuestion(
                        id=f"expert_{i}",
                        question=q.question,
                        tier=QuestionTier.EXPERT,
                        dimension=q.dimension,
                        question_type=q.question_type,
                        complexity_level=q.complexity_level,
                        context_hint=q.context_hint,
                        rationale=q.rationale,
                        grounded_context=q.grounded_context,
                    )
                )

            # Update session status
            session_data["status"] = "expert_questions_ready"

            response = TieredClarificationContinueResponse(
                session_id=session_id,
                status="Essential questions answered. Ready for deep dive.",
                expert_questions=expert_questions,
                estimated_additional_minutes=len(expert_questions)
                * 2,  # 2 minutes per expert question
            )

            self.logger.info(
                f"✅ Expert questions ready - Session: {session_id[:8]}, "
                f"Expert questions: {len(expert_questions)}"
            )

            return response

        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"❌ Expert questions continuation failed: {e}")
            raise HTTPException(
                status_code=500, detail=f"Expert questions failed: {str(e)}"
            )

    async def create_engagement_from_clarification(self, request):
        """
        NEW: Create final engagement from clarification session

        Args:
            request: CreateEngagementFromClarificationRequest

        Returns:
            CreateEngagementFromClarificationResponse
        """
        from .models import CreateEngagementFromClarificationResponse
        from .orchestrator import EngagementOrchestrator
        import uuid

        if not HITL_AVAILABLE:
            return CreateEngagementFromClarificationResponse(
                engagement_id=uuid.uuid4(),
                success=True,
                enhanced_query="Simulated enhanced query from clarification",
                total_clarifications=2,
                processing_started=True,
            )

        try:
            session_id = request.clarification_session_id

            if session_id not in self.active_sessions:
                raise HTTPException(
                    status_code=404, detail="Clarification session not found"
                )

            session_data = self.active_sessions[session_id]

            # Store expert answers if provided
            if request.expert_answers and not request.skip_expert:
                for answer in request.expert_answers:
                    session_data["expert_answers"][answer.question_id] = {
                        "answer": answer.answer,
                        "confidence": answer.confidence,
                    }

            # Emit Glass Box event for clarification answers processing
            try:
                from src.core.unified_context_stream import get_unified_context_stream
                context_stream = get_unified_context_stream()
                await context_stream.emit_event(
                    event_type=ContextEventType.CLARIFICATION_ANSWERS_PROCESSED,
                    data={
                        "session_id": session_id,
                        "essential_answers_count": len(session_data["essential_answers"]),
                        "expert_answers_count": len(session_data["expert_answers"]),
                        "expert_answers_skipped": request.skip_expert if hasattr(request, 'skip_expert') else False,
                        "answers_summary": {
                            "essential_answers": [data["answer"][:100] + "..." if len(data["answer"]) > 100 else data["answer"] 
                                                 for data in session_data["essential_answers"].values()],
                            "expert_answers": [data["answer"][:100] + "..." if len(data["answer"]) > 100 else data["answer"] 
                                              for data in session_data["expert_answers"].values()]
                        }
                    },
                    metadata={
                        "component": "TieredClarificationHandler", 
                        "stage": "answer_processing",
                        "transparency_layer": "glass_box"
                    }
                )
            except Exception as e:
                self.logger.warning(f"Failed to emit clarification answers event: {e}")

            # Build enhanced query from all clarifications
            enhanced_query = await self._build_enhanced_query_from_session(session_data)

            # Emit Glass Box event for enhanced query transformation
            try:
                context_stream = get_unified_context_stream()
                await context_stream.emit_event(
                    event_type=ContextEventType.QUERY_ENHANCED_FROM_CLARIFICATION,
                    data={
                        "original_query": session_data["raw_query"],
                        "enhanced_query": enhanced_query,
                        "essential_clarifications_count": len(session_data["essential_answers"]),
                        "expert_clarifications_count": len(session_data["expert_answers"]),
                        "enhancement_method": "tiered_clarification_hitl",
                        "session_id": session_id,
                        "transformation_summary": "Original query enhanced with user-provided context through progressive clarification questions"
                    },
                    metadata={
                        "component": "TieredClarificationHandler",
                        "stage": "query_enhancement",
                        "transparency_layer": "glass_box"
                    }
                )
            except Exception as e:
                self.logger.warning(f"Failed to emit enhanced query event: {e}")

            # Count total clarifications
            total_clarifications = len(session_data["essential_answers"]) + len(
                session_data["expert_answers"]
            )

            # Create engagement using existing orchestrator
            orchestrator = EngagementOrchestrator()

            from .models import EngagementRequest, ProblemStatement

            # Create engagement request from enhanced query
            engagement_request = EngagementRequest(
                problem_statement=ProblemStatement(
                    problem_description=enhanced_query,
                    business_context=session_data["business_context"],
                    success_criteria=[
                        "Comprehensive strategic analysis based on clarifications"
                    ],
                ),
                client_name="Clarified Analysis",
                engagement_type="tiered_clarification_analysis",
            )

            # Create the engagement
            engagement_response = await orchestrator.create_engagement(
                engagement_request
            )

            # Clean up session
            session_data["status"] = "completed"

            response = CreateEngagementFromClarificationResponse(
                engagement_id=engagement_response.engagement_id,
                success=True,
                enhanced_query=enhanced_query,
                total_clarifications=total_clarifications,
                processing_started=True,
            )

            self.logger.info(
                f"✅ Engagement created from clarification - "
                f"Session: {session_id[:8]}, "
                f"Engagement: {engagement_response.engagement_id}, "
                f"Total clarifications: {total_clarifications}"
            )

            return response

        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"❌ Engagement creation from clarification failed: {e}")
            raise HTTPException(
                status_code=500, detail=f"Engagement creation failed: {str(e)}"
            )

    async def _build_enhanced_query_from_session(
        self, session_data: Dict[str, Any]
    ) -> str:
        """
        Build enhanced query from clarification session data

        Args:
            session_data: Session data with answers

        Returns:
            Enhanced query string
        """
        raw_query = session_data["raw_query"]
        essential_answers = session_data["essential_answers"]
        expert_answers = session_data["expert_answers"]
        tiered_questions = session_data["tiered_questions"]

        # Start with engagement brief
        brief = tiered_questions.engagement_brief
        enhanced_parts = [
            f"ORIGINAL REQUEST: {raw_query}",
            "",
            "ENGAGEMENT BRIEF:",
            f"- Objective: {brief.objective}",
            f"- Platform: {brief.platform}",
            f"- Key Features: {', '.join(brief.key_features)}",
            "",
        ]

        # Add essential clarifications
        if essential_answers:
            enhanced_parts.append("ESSENTIAL CLARIFICATIONS:")
            for question_id, answer_data in essential_answers.items():
                enhanced_parts.append(f"• {answer_data['answer']}")
            enhanced_parts.append("")

        # Add expert clarifications
        if expert_answers:
            enhanced_parts.append("STRATEGIC CONSIDERATIONS:")
            for question_id, answer_data in expert_answers.items():
                enhanced_parts.append(f"• {answer_data['answer']}")

        return "\n".join(enhanced_parts)
