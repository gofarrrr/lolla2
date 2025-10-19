#!/usr/bin/env python3
"""
Strategic Trio + Devil's Advocate Orchestrator
Implements Optional, Post-Human, Per-Consultant critique pattern

Architecture:
1. Execute Strategic Trio (Multi-Single-Agent parallel consultants)
2. Present ALL consultant perspectives to human FIRST
3. Human can optionally request Devil's Advocate critique
4. Each consultant's analysis gets INDEPENDENT critique (no synthesis)
5. Stream critique results as they complete
6. Human chooses which perspectives and critiques to act upon

Core Values Preserved:
- Context preservation (all consultant perspectives + critiques maintained separately)
- No synthesis (devil's advocate critiques each consultant independently)
- Human orchestration (human triggers critique and chooses what to act on)
- Multiple perspectives (Strategic Trio + individual critiques)
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import traceback

from src.engine.engines.synthesis.dynamic_nway_execution_engine import (
    DynamicNWayExecutionEngine,
)
from src.engine.adapters.devils_advocate import  # Migrated (
    EnhancedDevilsAdvocateSystem,
    ComprehensiveChallengeResult,
)
from src.models.strategic_trio_critique_models import (
    StrategicTrioCritiqueOrchestrationResult,
    MultiConsultantCritiqueRequest,
    MultiConsultantCritiqueResult,
    ConsultantCritiqueResult,
    CritiqueStreamingUpdate,
    create_multi_consultant_critique_request,
)
from src.cognitive_architecture.mental_models_system import ConsultantRole

logger = logging.getLogger(__name__)


class StrategicTrioCritiqueOrchestrator:
    """
    Master orchestrator for Strategic Trio + Devil's Advocate integration

    Implements the Optional, Post-Human, Per-Consultant pattern:
    - Strategic Trio executes first (3 independent consultants in parallel)
    - Human receives ALL consultant perspectives immediately
    - Human can optionally request Devil's Advocate critique
    - Each consultant's analysis receives INDEPENDENT critique
    - No synthesis between consultant critiques
    - Human chooses which insights to act upon
    """

    def __init__(self):
        self.nway_engine = DynamicNWayExecutionEngine()
        self.devils_advocate = EnhancedDevilsAdvocateSystem()

        # Active critique requests for streaming
        self.active_critiques: Dict[str, MultiConsultantCritiqueRequest] = {}

        logger.info("Strategic Trio + Devil's Advocate orchestrator initialized")

    async def execute_strategic_trio(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> StrategicTrioCritiqueOrchestrationResult:
        """
        Execute Strategic Trio analysis - Phase 1 of Optional, Post-Human, Per-Consultant pattern

        Returns Strategic Trio results immediately to human.
        Human can then optionally request Devil's Advocate critique.
        """
        logger.info(f"🚀 Executing Strategic Trio for query: {query[:50]}...")

        try:
            # Execute Strategic Trio (our existing Multi-Single-Agent implementation)
            strategic_trio_result = await self.nway_engine.execute_cognitive_query(
                query, context
            )

            # Create orchestration result with Strategic Trio complete
            orchestration_result = StrategicTrioCritiqueOrchestrationResult(
                strategic_trio_result=strategic_trio_result,
                total_consultants_analyzed=len(
                    strategic_trio_result.consultant_perspectives
                ),
                human_seen_original=False,  # Human hasn't seen results yet
                critique_requested=False,  # No critique requested yet
                critique_streaming_enabled=True,
            )

            logger.info(
                f"✅ Strategic Trio completed: {len(strategic_trio_result.consultant_perspectives)} consultant perspectives ready"
            )
            logger.info(
                "🎯 Multi-Single-Agent execution complete - presenting all perspectives to human"
            )

            return orchestration_result

        except Exception as e:
            logger.error(f"❌ Strategic Trio execution failed: {e}")
            logger.error(traceback.format_exc())

            # Return error result
            error_result = StrategicTrioCritiqueOrchestrationResult(
                strategic_trio_result=None, total_consultants_analyzed=0
            )
            return error_result

    async def request_devils_advocate_critique(
        self,
        orchestration_result: StrategicTrioCritiqueOrchestrationResult,
        business_context: Optional[Dict[str, Any]] = None,
        stream_updates: bool = True,
    ) -> MultiConsultantCritiqueResult:
        """
        Execute Devil's Advocate critique - Phase 2 of Optional, Post-Human, Per-Consultant pattern

        IMPORTANT: This should only be called AFTER human has reviewed Strategic Trio results
        Each consultant's analysis receives INDEPENDENT critique with NO synthesis
        """
        if not orchestration_result.strategic_trio_result:
            raise ValueError("No Strategic Trio results available for critique")

        if not orchestration_result.strategic_trio_result.consultant_perspectives:
            raise ValueError("No consultant perspectives available for critique")

        logger.info(
            "🔍 Human has requested Devil's Advocate critique - executing per-consultant independent analysis"
        )

        # Mark that human has requested critique (post-human timing)
        orchestration_result.human_requested_critique = True
        orchestration_result.critique_requested = True
        orchestration_result.critique_in_progress = True
        orchestration_result.updated_at = datetime.utcnow()

        # Create critique request for each consultant INDEPENDENTLY
        critique_request = create_multi_consultant_critique_request(
            original_execution_id=orchestration_result.strategic_trio_result.execution_id,
            consultant_analyses=orchestration_result.strategic_trio_result.consultant_perspectives,
            business_context=business_context,
        )

        # Store for streaming updates
        self.active_critiques[critique_request.request_id] = critique_request

        try:
            # Execute independent critique for each consultant in parallel
            critique_result = await self._execute_independent_consultant_critiques(
                critique_request, stream_updates
            )

            # Update orchestration result
            orchestration_result.critique_result = critique_result
            orchestration_result.critique_in_progress = False
            orchestration_result.total_consultants_critiqued = (
                critique_result.critiques_completed
            )
            orchestration_result.updated_at = datetime.utcnow()

            logger.info(
                f"✅ Devil's Advocate critique completed: {critique_result.critiques_completed}/{len(critique_request.consultant_requests)} consultants critiqued"
            )

            return critique_result

        except Exception as e:
            logger.error(f"❌ Devil's Advocate critique failed: {e}")
            orchestration_result.critique_in_progress = False
            raise

        finally:
            # Clean up active critique tracking
            if critique_request.request_id in self.active_critiques:
                del self.active_critiques[critique_request.request_id]

    async def _execute_independent_consultant_critiques(
        self,
        critique_request: MultiConsultantCritiqueRequest,
        stream_updates: bool = True,
    ) -> MultiConsultantCritiqueResult:
        """
        Execute Devil's Advocate critique for each consultant INDEPENDENTLY

        Key principle: NO coordination or synthesis between consultant critiques
        Each consultant's analysis is critiqued in isolation
        """
        start_time = datetime.utcnow()
        consultant_critiques = {}
        critiques_completed = 0
        critiques_failed = 0

        logger.info(
            f"🎯 Executing {len(critique_request.consultant_requests)} independent consultant critiques"
        )

        # Create parallel critique tasks - each consultant analyzed independently
        critique_tasks = []
        for consultant_request in critique_request.consultant_requests:
            task = self._critique_single_consultant_independently(
                consultant_request, critique_request.request_id, stream_updates
            )
            critique_tasks.append((consultant_request.consultant_role, task))

        # Wait for all independent critiques to complete
        for consultant_role, task in critique_tasks:
            try:
                consultant_critique_result = await task
                consultant_critiques[consultant_role] = consultant_critique_result
                critiques_completed += 1

                logger.info(
                    f"✅ {consultant_role.value} independent critique completed"
                )

                # Stream update for this consultant completion
                if stream_updates:
                    await self._send_streaming_update(
                        CritiqueStreamingUpdate(
                            update_type="consultant_completed",
                            consultant_role=consultant_role,
                            progress_percent=(
                                critiques_completed
                                / len(critique_request.consultant_requests)
                            )
                            * 100,
                            partial_result=consultant_critique_result,
                        ),
                        critique_request.request_id,
                    )

            except Exception as e:
                critiques_failed += 1
                error_msg = f"{consultant_role.value} critique failed: {e}"
                logger.error(f"❌ {error_msg}")

                # Create error result for this consultant
                consultant_critiques[consultant_role] = ConsultantCritiqueResult(
                    consultant_role=consultant_role,
                    original_analysis=f"Critique failed: {error_msg}",
                    comprehensive_challenge_result=None,
                    critique_request_id=critique_request.request_id,
                    processing_time_seconds=0.0,
                )

        # Calculate total processing time
        total_processing_time = (datetime.utcnow() - start_time).total_seconds()

        # Create final result with all independent critiques
        result = MultiConsultantCritiqueResult(
            original_execution_id=critique_request.original_execution_id,
            consultant_critiques=consultant_critiques,
            total_processing_time=total_processing_time,
            critiques_completed=critiques_completed,
            critiques_failed=critiques_failed,
            critique_summary=self._generate_non_synthesis_summary(consultant_critiques),
            recommended_next_actions=self._generate_choice_facilitation_suggestions(
                consultant_critiques
            ),
            processing_metadata={
                "critique_request_id": critique_request.request_id,
                "engines_used": ["munger", "ackoff", "cognitive_audit"],
                "independence_maintained": True,
                "synthesis_avoided": True,
            },
        )

        logger.info(
            f"🎯 Multi-Single-Agent critique complete: {critiques_completed} independent consultant critiques"
        )

        return result

    async def _critique_single_consultant_independently(
        self, consultant_request, critique_request_id: str, stream_updates: bool = True
    ) -> ConsultantCritiqueResult:
        """
        Execute Devil's Advocate critique for single consultant in complete isolation

        NO coordination with other consultant critiques - pure independence
        """
        start_time = datetime.utcnow()

        logger.info(
            f"🔍 Starting independent Devil's Advocate analysis for {consultant_request.consultant_role.value}"
        )

        try:
            # Stream starting update
            if stream_updates:
                await self._send_streaming_update(
                    CritiqueStreamingUpdate(
                        update_type="consultant_started",
                        consultant_role=consultant_request.consultant_role,
                        progress_percent=0.0,
                        current_engine="comprehensive_analysis",
                    ),
                    critique_request_id,
                )

            # Execute comprehensive challenge analysis for this consultant only
            # The enhanced devil's advocate system will run all three engines on this consultant's analysis
            comprehensive_result = (
                await self.devils_advocate.comprehensive_challenge_analysis(
                    recommendation=consultant_request.analysis_text,
                    business_context=consultant_request.business_context,
                )
            )

            processing_time = (datetime.utcnow() - start_time).total_seconds()

            # Analyze challenges by engine for this consultant
            challenges_by_engine = {}
            highest_risk_challenge = None
            highest_severity = 0.0

            for challenge in comprehensive_result.critical_challenges:
                engine = challenge.source_engine
                challenges_by_engine[engine] = challenges_by_engine.get(engine, 0) + 1

                if challenge.severity > highest_severity:
                    highest_severity = challenge.severity
                    highest_risk_challenge = challenge

            # Generate consultant-specific insights (role-based critique focus)
            consultant_insights = self._generate_consultant_specific_insights(
                consultant_request.consultant_role, comprehensive_result
            )

            # Create independent consultant critique result
            consultant_critique = ConsultantCritiqueResult(
                consultant_role=consultant_request.consultant_role,
                original_analysis=consultant_request.analysis_text,
                comprehensive_challenge_result=comprehensive_result,
                critique_request_id=critique_request_id,
                processing_time_seconds=processing_time,
                challenges_by_engine=challenges_by_engine,
                highest_risk_challenge=highest_risk_challenge,
                consultant_specific_insights=consultant_insights,
            )

            logger.info(
                f"✅ {consultant_request.consultant_role.value} independent critique completed: {comprehensive_result.total_challenges_found} challenges found"
            )

            return consultant_critique

        except Exception as e:
            logger.error(
                f"❌ Independent critique failed for {consultant_request.consultant_role.value}: {e}"
            )
            raise

    def _generate_consultant_specific_insights(
        self,
        consultant_role: ConsultantRole,
        comprehensive_result: ComprehensiveChallengeResult,
    ) -> List[str]:
        """Generate insights specific to the consultant's role and expertise"""
        insights = []

        # Role-specific critique focus
        if consultant_role == ConsultantRole.STRATEGIC_ANALYST:
            insights.extend(
                [
                    f"Strategic assumptions challenged: {comprehensive_result.total_challenges_found} issues identified",
                    f"Long-term strategic risk score: {comprehensive_result.overall_risk_score:.2f}",
                    "Focus on strategic assumptions and competitive positioning biases",
                ]
            )

        elif consultant_role == ConsultantRole.SYNTHESIS_ARCHITECT:
            insights.extend(
                [
                    f"Synthesis quality risk: {comprehensive_result.overall_risk_score:.2f}",
                    f"Integration challenges: {len(comprehensive_result.critical_challenges)} critical issues",
                    "Focus on synthesis biases and over-simplification tendencies",
                ]
            )

        elif consultant_role == ConsultantRole.IMPLEMENTATION_DRIVER:
            insights.extend(
                [
                    f"Implementation viability: {1.0 - comprehensive_result.overall_risk_score:.2f}",
                    f"Execution risk factors: {comprehensive_result.total_challenges_found} identified",
                    "Focus on execution assumptions and operational biases",
                ]
            )

        # Add top challenge themes for this consultant
        if comprehensive_result.critical_challenges:
            challenge_types = [
                c.challenge_type for c in comprehensive_result.critical_challenges[:3]
            ]
            insights.append(
                f"Primary critique themes: {', '.join(set(challenge_types))}"
            )

        return insights

    def _generate_non_synthesis_summary(
        self, consultant_critiques: Dict[ConsultantRole, ConsultantCritiqueResult]
    ) -> str:
        """
        Generate overview of critiques WITHOUT synthesizing between consultants

        This helps human navigate critiques but does NOT merge insights
        """
        if not consultant_critiques:
            return "No consultant critiques available."

        summary_lines = [
            "# Multi-Single-Agent Critique Overview",
            "",
            "Each consultant's analysis has been independently critiqued by Devil's Advocate engines.",
            "**NO synthesis between consultants** - each critique is independent.",
            "",
            "## Individual Consultant Critiques:",
        ]

        for consultant_role, critique_result in consultant_critiques.items():
            if critique_result.comprehensive_challenge_result:
                result = critique_result.comprehensive_challenge_result
                summary_lines.extend(
                    [
                        "",
                        f"### {consultant_role.value} Independent Critique",
                        f"- Challenges identified: {result.total_challenges_found}",
                        f"- Risk score: {result.overall_risk_score:.2f}",
                        f"- System confidence: {result.system_confidence:.2f}",
                        f"- Processing time: {critique_result.processing_time_seconds:.1f}s",
                        f"- Top challenge: {critique_result.highest_risk_challenge.challenge_text if critique_result.highest_risk_challenge else 'None'}",
                    ]
                )
            else:
                summary_lines.extend(
                    [
                        "",
                        f"### {consultant_role.value} Critique",
                        "- Status: Failed or incomplete",
                    ]
                )

        summary_lines.extend(
            [
                "",
                "## Next Steps",
                "Review each consultant's critique independently and choose which insights to act upon.",
                "Consider requesting Senior Advisor arbitration for additional synthesis if desired.",
            ]
        )

        return "\n".join(summary_lines)

    def _generate_choice_facilitation_suggestions(
        self, consultant_critiques: Dict[ConsultantRole, ConsultantCritiqueResult]
    ) -> List[str]:
        """Generate suggestions to help human choose between critique insights - NOT decisions"""
        suggestions = []

        if not consultant_critiques:
            return ["No critiques available for analysis."]

        # Analyze critique patterns to help human navigate (not decide)
        high_risk_consultants = []
        high_confidence_consultants = []

        for consultant_role, critique_result in consultant_critiques.items():
            if critique_result.comprehensive_challenge_result:
                result = critique_result.comprehensive_challenge_result

                if result.overall_risk_score > 0.7:
                    high_risk_consultants.append(consultant_role.value)

                if result.system_confidence > 0.8:
                    high_confidence_consultants.append(consultant_role.value)

        # Suggest navigation strategies
        if high_risk_consultants:
            suggestions.append(
                f"Consider reviewing high-risk critiques first: {', '.join(high_risk_consultants)}"
            )

        if high_confidence_consultants:
            suggestions.append(
                f"High-confidence critiques available from: {', '.join(high_confidence_consultants)}"
            )

        suggestions.extend(
            [
                "Each consultant's critique is independent - choose which challenges are most relevant to your context",
                "Consider requesting Senior Advisor arbitration if you want additional synthesis perspective",
                "Focus on critiques that align with your specific priorities and risk tolerance",
            ]
        )

        return suggestions

    async def _send_streaming_update(
        self, update: CritiqueStreamingUpdate, request_id: str
    ):
        """Send real-time streaming update for critique progress"""
        # This would integrate with WebSocket streaming in actual implementation
        logger.info(
            f"📡 Streaming update: {update.update_type} - {update.consultant_role.value if update.consultant_role else 'system'}"
        )

    # Public API methods

    async def get_critique_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get status of active critique request"""
        if request_id in self.active_critiques:
            request = self.active_critiques[request_id]
            return {
                "request_id": request_id,
                "status": "in_progress",
                "consultants_total": len(request.consultant_requests),
                "created_at": request.created_at.isoformat(),
            }
        return None

    def get_active_critiques(self) -> List[str]:
        """Get list of active critique request IDs"""
        return list(self.active_critiques.keys())


# Factory function for easy instantiation
def create_strategic_trio_critique_orchestrator() -> StrategicTrioCritiqueOrchestrator:
    """Create configured Strategic Trio + Devil's Advocate orchestrator"""
    return StrategicTrioCritiqueOrchestrator()
