#!/usr/bin/env python3
"""
UltraThink-Flywheel Integration Bridge
Connects Operation Synapse Challenge Systems with Test-Driven Learning Flywheel

Integrates the challenge generation systems from Operation Synapse with the
continuous learning flywheel to create a comprehensive intelligence system
that learns from both challenge effectiveness and test outcomes.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from uuid import UUID, uuid4

try:
    # UltraThink components (Operation Synapse)
    from src.engine.core.cognitive_diversity_calibrator import (
        CognitiveDiversityCalibrator,
        ConvergenceRisk,
        DiversityIntervention,
    )
    from src.engine.intelligence.research_armed_challenger import (
        get_research_armed_challenger,
    )
    from src.engine.intelligence.l1_inversion_analysis import InversionAnalysisEngine
    from src.engine.intelligence.l2_latticework_validation import LatticeworkValidator
    try:
        from src.engine.intelligence.l3_constitutional_bias_audit import (
            ConstitutionalAuditor,
        )
    except ImportError:
        from src.intelligence.l3_constitutional_bias_audit import (
            L3ConstitutionalBiasAuditor as ConstitutionalAuditor,
        )
    from src.engine.engines.validation.assumption_challenger import (
        get_assumption_challenger,
    )

    # Flywheel components
    from src.engine.flywheel.management.test_flywheel_manager import (
        get_test_flywheel_manager,
        TestOutcome,
        TestCategory,
        LearningSignal,
    )
    from src.engine.flywheel.orchestration.continuous_learning_orchestrator import (
        get_continuous_learning_orchestrator,
        LearningTrigger,
        LearningPhase,
    )
    from src.engine.flywheel.feedback.test_feedback_engine import (
        get_test_feedback_engine,
    )
    from src.engine.intelligence.bayesian_effectiveness_updater import (
        get_bayesian_updater,
    )

    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    # Silence noisy prints during tests; rely on logger if needed
    logging.getLogger(__name__).debug(f"Dependencies not available: {e}")

logger = logging.getLogger(__name__)


class ChallengeEffectiveness(Enum):
    """Effectiveness levels for challenge generation"""

    POOR = "poor"  # Challenge was not useful
    FAIR = "fair"  # Challenge provided some value
    GOOD = "good"  # Challenge was valuable
    EXCELLENT = "excellent"  # Challenge led to breakthrough insight


class IntegrationMode(Enum):
    """Integration modes between UltraThink and Flywheel"""

    PASSIVE = "passive"  # Only observe and record
    ACTIVE = "active"  # Actively influence selection
    ADAPTIVE = "adaptive"  # Learn and adapt strategies


@dataclass
class ChallengeResult:
    """Result of a challenge system execution"""

    challenge_id: UUID = field(default_factory=uuid4)
    challenger_type: str = ""  # "research_armed", "assumption", "inversion", etc.
    challenge_generated: bool = False
    challenge_content: str = ""
    confidence_threshold: float = 0.0
    impact_threshold: float = 0.0
    execution_time_ms: float = 0.0
    effectiveness: ChallengeEffectiveness = ChallengeEffectiveness.FAIR
    learning_signals: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UltraThinkSession:
    """Session tracking UltraThink system usage"""

    session_id: UUID = field(default_factory=uuid4)
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None

    # Challenge system usage
    challenges_attempted: int = 0
    challenges_successful: int = 0
    research_calls_made: int = 0
    total_cost: float = 0.0

    # Effectiveness tracking
    challenge_results: List[ChallengeResult] = field(default_factory=list)
    diversity_interventions: List[DiversityIntervention] = field(default_factory=list)
    convergence_risk: ConvergenceRisk = ConvergenceRisk.LOW

    # Learning outcomes
    insights_generated: int = 0
    model_effectiveness_updates: Dict[str, float] = field(default_factory=dict)
    flywheel_value_contributed: float = 0.0


class UltraThinkFlywheelBridge:
    """
    Integration bridge between UltraThink (Operation Synapse) and Flywheel systems.

    Key responsibilities:
    1. Coordinate challenge system execution with flywheel learning
    2. Track challenge effectiveness and feed back to learning system
    3. Manage context engineering for consistent performance
    4. Prevent phantom workflows through timing validation
    5. Optimize resource usage across both systems
    """

    def __init__(self):
        self.integration_mode = IntegrationMode.ADAPTIVE
        self.active_sessions: Dict[UUID, UltraThinkSession] = {}
        self.performance_history: List[Dict] = []

        # Context engineering (from Manus principles)
        self.context_history: List[Dict] = []  # Append-only context tracking
        self.wrong_turns_registry: List[Dict] = []  # Failed attempts for learning
        self.session_cache: Dict[str, Any] = {}  # KV-cache optimization

        if DEPENDENCIES_AVAILABLE:
            self.diversity_calibrator = CognitiveDiversityCalibrator()
            self.flywheel_manager = get_test_flywheel_manager()
            self.learning_orchestrator = get_continuous_learning_orchestrator()
            self.feedback_engine = get_test_feedback_engine()

            # Challenge systems
            self.research_challenger = None  # Will be initialized when needed
            self.assumption_challenger = None
            self.inversion_analyzer = None

        logger.info("UltraThink-Flywheel Bridge initialized")

    async def start_integrated_session(self, context: Dict[str, Any]) -> UUID:
        """Start an integrated UltraThink-Flywheel session"""
        session = UltraThinkSession()
        self.active_sessions[session.session_id] = session

        # Context engineering - preserve session context
        session_context = {
            "session_id": str(session.session_id),
            "timestamp": datetime.utcnow().isoformat(),
            "context": context.copy(),
            "type": "session_start",
        }
        self.context_history.append(session_context)

        # Initialize session cache for KV-optimization
        cache_key = f"session_{session.session_id}"
        self.session_cache[cache_key] = {
            "prefix_stable": True,
            "context_length": 0,
            "last_update": datetime.utcnow(),
        }

        logger.info(f"Started integrated session {session.session_id}")
        return session.session_id

    async def execute_challenge_with_flywheel(
        self,
        session_id: UUID,
        challenger_type: str,
        engagement_context: Dict[str, Any],
        capture_for_learning: bool = True,
    ) -> ChallengeResult:
        """Execute a challenge system with flywheel integration"""

        if session_id not in self.active_sessions:
            raise ValueError(f"Invalid session ID: {session_id}")

        session = self.active_sessions[session_id]
        challenge_result = ChallengeResult(challenger_type=challenger_type)
        start_time = time.time()

        try:
            # Execute the challenge based on type
            if challenger_type == "research_armed":
                challenge_result = await self._execute_research_armed_challenge(
                    engagement_context, challenge_result
                )
            elif challenger_type == "assumption":
                challenge_result = await self._execute_assumption_challenge(
                    engagement_context, challenge_result
                )
            elif challenger_type == "inversion":
                challenge_result = await self._execute_inversion_challenge(
                    engagement_context, challenge_result
                )
            elif challenger_type == "latticework":
                challenge_result = await self._execute_latticework_challenge(
                    engagement_context, challenge_result
                )
            elif challenger_type == "constitutional":
                challenge_result = await self._execute_constitutional_challenge(
                    engagement_context, challenge_result
                )
            else:
                raise ValueError(f"Unknown challenger type: {challenger_type}")

            # Calculate execution time
            challenge_result.execution_time_ms = (time.time() - start_time) * 1000

            # Check for phantom workflow (from UltraThink principles)
            if (
                challenge_result.execution_time_ms < 100
            ):  # Less than 100ms is suspicious
                self._record_wrong_turn(
                    {
                        "type": "phantom_challenge",
                        "challenger": challenger_type,
                        "execution_time": challenge_result.execution_time_ms,
                        "session_id": str(session_id),
                    }
                )
                challenge_result.effectiveness = ChallengeEffectiveness.POOR
                challenge_result.learning_signals.append("phantom_execution")

            # Update session tracking
            session.challenges_attempted += 1
            if challenge_result.challenge_generated:
                session.challenges_successful += 1
            session.challenge_results.append(challenge_result)

            # Capture for flywheel learning if enabled
            if capture_for_learning and DEPENDENCIES_AVAILABLE:
                await self._capture_challenge_for_flywheel(
                    challenge_result, engagement_context
                )

            # Context engineering - append challenge result to context
            self.context_history.append(
                {
                    "session_id": str(session_id),
                    "timestamp": datetime.utcnow().isoformat(),
                    "type": "challenge_result",
                    "challenger": challenger_type,
                    "success": challenge_result.challenge_generated,
                    "effectiveness": challenge_result.effectiveness.value,
                    "execution_time_ms": challenge_result.execution_time_ms,
                }
            )

            logger.info(
                f"Challenge {challenger_type} completed: {challenge_result.effectiveness.value}"
            )
            return challenge_result

        except Exception as e:
            # Record failure for learning
            self._record_wrong_turn(
                {
                    "type": "challenge_failure",
                    "challenger": challenger_type,
                    "error": str(e),
                    "session_id": str(session_id),
                }
            )

            challenge_result.effectiveness = ChallengeEffectiveness.POOR
            challenge_result.learning_signals.append("execution_error")
            logger.error(f"Challenge execution failed: {e}")
            return challenge_result

    async def _execute_research_armed_challenge(
        self, context: Dict[str, Any], result: ChallengeResult
    ) -> ChallengeResult:
        """Execute research-armed challenger"""
        try:
            if not self.research_challenger:
                self.research_challenger = get_research_armed_challenger()

            # Mock implementation - in production would call actual challenger
            result.challenge_generated = True
            result.challenge_content = "Research-armed challenge generated"
            result.confidence_threshold = context.get("confidence_threshold", 0.7)
            result.impact_threshold = context.get("impact_threshold", 0.7)
            result.effectiveness = ChallengeEffectiveness.GOOD

            return result
        except Exception as e:
            result.learning_signals.append(f"research_challenge_error: {str(e)}")
            return result

    async def _execute_assumption_challenge(
        self, context: Dict[str, Any], result: ChallengeResult
    ) -> ChallengeResult:
        """Execute assumption challenger (Ackoff methodology)"""
        try:
            if not self.assumption_challenger:
                self.assumption_challenger = get_assumption_challenger()

            # Mock implementation
            result.challenge_generated = True
            result.challenge_content = "Assumption dissolution challenge generated"
            result.effectiveness = ChallengeEffectiveness.GOOD

            return result
        except Exception as e:
            result.learning_signals.append(f"assumption_challenge_error: {str(e)}")
            return result

    async def _execute_inversion_challenge(
        self, context: Dict[str, Any], result: ChallengeResult
    ) -> ChallengeResult:
        """Execute L1 inversion analysis challenge"""
        try:
            if not self.inversion_analyzer:
                self.inversion_analyzer = InversionAnalysisEngine()

            # Mock implementation
            result.challenge_generated = True
            result.challenge_content = "Inversion analysis challenge generated"
            result.effectiveness = ChallengeEffectiveness.FAIR

            return result
        except Exception as e:
            result.learning_signals.append(f"inversion_challenge_error: {str(e)}")
            return result

    async def _execute_latticework_challenge(
        self, context: Dict[str, Any], result: ChallengeResult
    ) -> ChallengeResult:
        """Execute L2 latticework validation challenge"""
        try:
            # Mock implementation
            result.challenge_generated = True
            result.challenge_content = "Latticework validation challenge generated"
            result.effectiveness = ChallengeEffectiveness.FAIR

            return result
        except Exception as e:
            result.learning_signals.append(f"latticework_challenge_error: {str(e)}")
            return result

    async def _execute_constitutional_challenge(
        self, context: Dict[str, Any], result: ChallengeResult
    ) -> ChallengeResult:
        """Execute L3 constitutional bias audit challenge"""
        try:
            # Mock implementation
            result.challenge_generated = True
            result.challenge_content = "Constitutional bias audit challenge generated"
            result.effectiveness = ChallengeEffectiveness.EXCELLENT

            return result
        except Exception as e:
            result.learning_signals.append(f"constitutional_challenge_error: {str(e)}")
            return result

    async def _capture_challenge_for_flywheel(
        self, result: ChallengeResult, context: Dict[str, Any]
    ):
        """Capture challenge result for flywheel learning"""
        try:
            # Convert to flywheel test result format
            test_outcome = (
                TestOutcome.PASSED if result.challenge_generated else TestOutcome.FAILED
            )

            # Capture enhanced test result
            test_data = {
                "test_name": f"challenge_{result.challenger_type}",
                "test_category": TestCategory.COGNITIVE,
                "outcome": test_outcome,
                "execution_time_ms": result.execution_time_ms,
                "model_interactions": {
                    "challenger_type": result.challenger_type,
                    "confidence_threshold": result.confidence_threshold,
                    "impact_threshold": result.impact_threshold,
                },
                "learning_signals": result.learning_signals,
                "effectiveness_score": self._effectiveness_to_score(
                    result.effectiveness
                ),
            }

            # Feed to flywheel manager
            await self.flywheel_manager.capture_test_result_enhanced(test_data)

        except Exception as e:
            logger.error(f"Failed to capture challenge for flywheel: {e}")

    def _effectiveness_to_score(self, effectiveness: ChallengeEffectiveness) -> float:
        """Convert effectiveness enum to numeric score"""
        mapping = {
            ChallengeEffectiveness.POOR: 0.2,
            ChallengeEffectiveness.FAIR: 0.5,
            ChallengeEffectiveness.GOOD: 0.8,
            ChallengeEffectiveness.EXCELLENT: 1.0,
        }
        return mapping.get(effectiveness, 0.5)

    def _record_wrong_turn(self, wrong_turn_data: Dict[str, Any]):
        """Record a wrong turn for learning (from Manus principles)"""
        wrong_turn_data.update(
            {"timestamp": datetime.utcnow().isoformat(), "recorded_for_learning": True}
        )
        self.wrong_turns_registry.append(wrong_turn_data)

        # Limit registry size to prevent memory issues
        if len(self.wrong_turns_registry) > 1000:
            self.wrong_turns_registry = self.wrong_turns_registry[-500:]

    async def end_session(self, session_id: UUID) -> Dict[str, Any]:
        """End integrated session and generate summary"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Invalid session ID: {session_id}")

        session = self.active_sessions[session_id]
        session.end_time = datetime.utcnow()

        # Calculate session metrics
        session_duration = (session.end_time - session.start_time).total_seconds()
        success_rate = (
            session.challenges_successful / session.challenges_attempted
            if session.challenges_attempted > 0
            else 0.0
        )

        # Generate summary
        summary = {
            "session_id": str(session_id),
            "duration_seconds": session_duration,
            "challenges_attempted": session.challenges_attempted,
            "challenges_successful": session.challenges_successful,
            "success_rate": success_rate,
            "total_cost": session.total_cost,
            "convergence_risk": session.convergence_risk.value,
            "flywheel_value": session.flywheel_value_contributed,
            "context_entries": len(
                [
                    c
                    for c in self.context_history
                    if c.get("session_id") == str(session_id)
                ]
            ),
            "wrong_turns": len(
                [
                    w
                    for w in self.wrong_turns_registry
                    if w.get("session_id") == str(session_id)
                ]
            ),
        }

        # Trigger learning if conditions are met
        if DEPENDENCIES_AVAILABLE and session.challenges_attempted >= 5:
            await self._trigger_learning_cycle(session)

        # Clean up session
        del self.active_sessions[session_id]

        # Clean up session cache
        cache_key = f"session_{session_id}"
        if cache_key in self.session_cache:
            del self.session_cache[cache_key]

        logger.info(f"Session {session_id} completed: {success_rate:.2%} success rate")
        return summary

    async def _trigger_learning_cycle(self, session: UltraThinkSession):
        """Trigger learning cycle based on session results"""
        try:
            # Analyze session for learning signals
            learning_signals = []
            for result in session.challenge_results:
                learning_signals.extend(result.learning_signals)

            # Check if learning cycle should be triggered
            if len(learning_signals) >= 3:  # Arbitrary threshold
                await self.learning_orchestrator.execute_learning_cycle(
                    trigger=LearningTrigger.TEST_COMPLETION
                )
                logger.info("Triggered learning cycle from UltraThink session")

        except Exception as e:
            logger.error(f"Failed to trigger learning cycle: {e}")

    def get_context_summary(self) -> Dict[str, Any]:
        """Get summary of context engineering state"""
        return {
            "total_context_entries": len(self.context_history),
            "wrong_turns_recorded": len(self.wrong_turns_registry),
            "active_sessions": len(self.active_sessions),
            "cache_entries": len(self.session_cache),
            "integration_mode": self.integration_mode.value,
            "last_activity": max(
                [c.get("timestamp", "") for c in self.context_history] + [""]
            ),
        }


# Singleton instance
_ultrathink_flywheel_bridge = None


def get_ultrathink_flywheel_bridge() -> UltraThinkFlywheelBridge:
    """Get singleton instance of UltraThink-Flywheel bridge"""
    global _ultrathink_flywheel_bridge
    if _ultrathink_flywheel_bridge is None:
        _ultrathink_flywheel_bridge = UltraThinkFlywheelBridge()
    return _ultrathink_flywheel_bridge


async def main():
    """Demo of UltraThink-Flywheel integration"""
    print("🔗 UltraThink-Flywheel Integration Bridge Demo")
    print("=" * 60)

    bridge = get_ultrathink_flywheel_bridge()

    # Start session
    session_id = await bridge.start_integrated_session(
        {"demo": True, "purpose": "integration_validation"}
    )

    print(f"Started session: {session_id}")

    # Execute various challenges
    challengers = ["research_armed", "assumption", "inversion", "constitutional"]

    for challenger in challengers:
        print(f"Executing {challenger} challenge...")
        result = await bridge.execute_challenge_with_flywheel(
            session_id, challenger, {"test": True}
        )
        print(
            f"  Result: {result.effectiveness.value} ({result.execution_time_ms:.1f}ms)"
        )

    # End session
    summary = await bridge.end_session(session_id)
    print(f"Session completed: {summary['success_rate']:.2%} success rate")

    # Context summary
    context_summary = bridge.get_context_summary()
    print(f"Context entries: {context_summary['total_context_entries']}")
    print(f"Wrong turns recorded: {context_summary['wrong_turns_recorded']}")


if __name__ == "__main__":
    asyncio.run(main())
