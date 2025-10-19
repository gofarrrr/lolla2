#!/usr/bin/env python3
"""
Method Actor Devils Advocate System - Enhanced ULTRATHINK Challenge Engine
==========================================================================

BREAKTHROUGH: Hybrid algorithmic + Method Actor approach for enabling challenger style
- Algorithmic foundation ensures systematic coverage and reliability
- Method Actor personas provide research-validated enabling challenger communication
- Forward motion converter transforms every challenge into experiments/guardrails

VALIDATION: Based on extensive research on constructive challenge and System-2 thinking
STATUS: V1.0 - Ready for integration with existing ULTRATHINK Devils Advocate system
"""

import logging
import os
import yaml
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
import uuid

# Existing system integrations
from src.core.enhanced_devils_advocate_system import (
    EnhancedDevilsAdvocateSystem,
    DevilsAdvocateChallenge,
    ComprehensiveChallengeResult,
)
from src.core.unified_context_stream import UnifiedContextStream

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when YAML configuration is invalid or missing"""

    pass


@dataclass
class MethodActorPersona:
    """Method Actor persona configuration for enabling challenger style"""

    persona_id: str
    character_archetype: str
    background: str
    cognitive_style: str
    communication_patterns: Dict[str, str]
    signature_methods: List[str]
    avoid_patterns: List[str]
    forward_motion_style: str
    token_budget: int


@dataclass
class ForwardMotionAction:
    """Actionable item generated from Method Actor challenge"""

    action_type: str  # experiment, guardrail, reversible_step, premortem
    description: str
    hypothesis: str
    test_design: str
    success_criteria: str
    time_horizon: str
    cost_estimate: str
    reversibility: str
    learning_objective: str


@dataclass
class MethodActorDialogue:
    """Method Actor dialogue with forward motion actions"""

    persona_id: str
    dialogue_text: str
    challenges_addressed: List[str]
    forward_motion_actions: List[ForwardMotionAction]
    tone_safety_score: float
    psychological_safety_maintained: bool


@dataclass
class MethodActorDAResult:
    """Complete Method Actor Devils Advocate result"""

    original_recommendation: str
    algorithmic_foundation: Dict[str, Any]
    method_actor_dialogues: List[MethodActorDialogue]
    forward_motion_summary: Dict[str, List[ForwardMotionAction]]
    enabling_challenger_score: float
    forward_motion_conversion_rate: float
    anti_failure_measures: Dict[str, Any]
    system_integration_data: Dict[str, Any]


class PersonaType(str, Enum):
    """Available Method Actor personas"""

    CHARLIE_MUNGER = "charlie_munger"
    RUSSELL_ACKOFF = "russell_ackoff"


class ForwardMotionType(str, Enum):
    """Types of forward motion actions"""

    EXPERIMENT = "experiment"
    GUARDRAIL = "guardrail"
    REVERSIBLE_STEP = "reversible_step"
    PREMORTEM_SCENARIO = "premortem_scenario"


class MethodActorDevilsAdvocate:
    """
    🎭 METHOD ACTOR DEVILS ADVOCATE - Research-Validated Enabling Challenger

    Hybrid approach combining:
    1. Proven algorithmic engines for systematic detection (reliability)
    2. Method Actor personas for enabling challenger communication (engagement)
    3. Forward motion converter (challenges → experiments/guardrails)
    4. Anti-failure mode safeguards (prevent gotcha-ism, naysaying, etc.)

    INTEGRATION: Enhances existing EnhancedDevilsAdvocateSystem with Method Actor layer
    RESEARCH BASIS: Validated patterns for constructive challenge and forward motion
    """

    def __init__(
        self,
        context_stream: Optional[UnifiedContextStream] = None,
        yaml_config_path: Optional[str] = None,
    ):
        # Keep proven algorithmic foundation
        self.enhanced_da_system = EnhancedDevilsAdvocateSystem()
        from src.core.unified_context_stream import get_unified_context_stream, UnifiedContextStream
        self.context_stream = context_stream or get_unified_context_stream()

        # Operation Lean - Target #3: Use extracted services
        from src.core.services import (
            ConfigurationLoader,
            MungerPersonaEngine,
            AckoffPersonaEngine,
            ForwardMotionConverter,
            ToneSafeguards,
        )

        # Initialize configuration loader
        self.config_loader = ConfigurationLoader()

        # Load configuration from YAML if provided
        self.yaml_config_path = yaml_config_path
        self.yaml_config = (
            self.config_loader.load_yaml_config(yaml_config_path)
            if yaml_config_path
            else None
        )

        # Load thin variables
        self.thin_variables = self.config_loader.load_thin_variables(self.yaml_config)

        # Initialize persona engine registry (plugin architecture)
        self.persona_engines: Dict[PersonaType, Any] = {
            PersonaType.CHARLIE_MUNGER: MungerPersonaEngine(),
            PersonaType.RUSSELL_ACKOFF: AckoffPersonaEngine(),
        }

        # Initialize forward motion converter
        self.forward_motion_converter = ForwardMotionConverter()

        # Initialize tone safeguards
        self.tone_safeguards = ToneSafeguards()

        logger.info(
            "🎭 Method Actor Devils Advocate initialized with plugin architecture"
        )
        logger.info(
            f"   • Configuration source: {'YAML' if self.yaml_config else 'hardcoded defaults'}"
        )
        logger.info("   • Algorithmic engines: 4 (Munger, Ackoff, Cognitive, LLM)")
        logger.info(f"   • Persona engines registered: {len(self.persona_engines)}")
        logger.info("   • Forward motion converter: enabled")
        logger.info("   • Tone safeguards: active")

    # Operation Lean - Target #3: Methods moved to ConfigurationLoader service
    # _load_yaml_config() → ConfigurationLoader.load_yaml_config()
    # _load_thin_variables() → ConfigurationLoader.load_thin_variables()
    # _initialize_personas() → Now using persona engine registry
    # _load_personas_from_yaml() → ConfigurationLoader.load_personas_from_yaml()

    async def method_actor_comprehensive_challenge(
        self, recommendation: str, business_context: Dict[str, Any], tier_level: int = 2
    ) -> MethodActorDAResult:
        """
        Main entry point: Enhanced devils advocate analysis with Method Actor personas

        Process:
        1. Run proven algorithmic engines for systematic detection
        2. Transform results through Method Actor personas
        3. Convert challenges to forward motion actions
        4. Apply anti-failure safeguards
        5. Generate comprehensive result with evidence
        """
        logger.info(f"🎭 METHOD ACTOR DEVILS ADVOCATE ANALYSIS - Tier {tier_level}")
        logger.info("=" * 80)
        logger.info(f"Analyzing: {recommendation[:100]}...")
        logger.info("=" * 80)

        # STEP 1: Run algorithmic foundation (proven reliability)
        logger.info("⚙️ Step 1: Running algorithmic foundation engines...")
        algorithmic_result = (
            await self.enhanced_da_system.comprehensive_challenge_analysis(
                recommendation, business_context
            )
        )

        # STEP 2: Transform through Method Actor personas (enabling challenger style)
        logger.info("🎭 Step 2: Transforming through Method Actor personas...")
        method_actor_dialogues = await self._transform_to_method_actor_dialogues(
            algorithmic_result, recommendation, business_context, tier_level
        )

        # STEP 3: Convert challenges to forward motion (experiments/guardrails)
        logger.info("⚡ Step 3: Converting challenges to forward motion actions...")
        forward_motion_summary = await self._convert_to_forward_motion(
            method_actor_dialogues, recommendation, business_context
        )

        # STEP 4: Calculate enabling challenger metrics
        logger.info("📊 Step 4: Calculating enabling challenger metrics...")
        enabling_score = self._calculate_enabling_challenger_score(
            method_actor_dialogues
        )
        conversion_rate = self._calculate_forward_motion_conversion_rate(
            forward_motion_summary
        )

        # STEP 5: Apply anti-failure safeguards assessment
        anti_failure_measures = self._assess_anti_failure_measures(
            method_actor_dialogues
        )

        # STEP 6: Record enhanced evidence
        logger.info("📝 Step 6: Recording Method Actor DA evidence...")
        await self._record_method_actor_da_evidence(
            recommendation,
            algorithmic_result,
            method_actor_dialogues,
            forward_motion_summary,
            business_context,
        )

        # Create comprehensive result
        result = MethodActorDAResult(
            original_recommendation=recommendation,
            algorithmic_foundation={
                "total_challenges": algorithmic_result.total_challenges_found,
                "critical_challenges": len(algorithmic_result.critical_challenges),
                "overall_risk_score": algorithmic_result.overall_risk_score,
                "intellectual_honesty_score": algorithmic_result.intellectual_honesty_score,
                "system_confidence": algorithmic_result.system_confidence,
            },
            method_actor_dialogues=method_actor_dialogues,
            forward_motion_summary=forward_motion_summary,
            enabling_challenger_score=enabling_score,
            forward_motion_conversion_rate=conversion_rate,
            anti_failure_measures=anti_failure_measures,
            system_integration_data={
                "tier_level": tier_level,
                "personas_activated": len(method_actor_dialogues),
                "algorithmic_engines_used": 4,
            },
        )

        logger.info("\n💡 METHOD ACTOR DA RESULTS:")
        logger.info(
            f"├─ Algorithmic challenges: {algorithmic_result.total_challenges_found}"
        )
        logger.info(f"├─ Method Actor dialogues: {len(method_actor_dialogues)}")
        logger.info(
            f"├─ Forward motion actions: {sum(len(actions) for actions in forward_motion_summary.values())}"
        )
        logger.info(f"├─ Enabling challenger score: {enabling_score:.3f}")
        logger.info(f"├─ Forward motion conversion: {conversion_rate:.3f}")
        logger.info(
            f"└─ Anti-failure safeguards: {anti_failure_measures['overall_safety_score']:.3f}"
        )

        return result

    async def _transform_to_method_actor_dialogues(
        self,
        algorithmic_result: ComprehensiveChallengeResult,
        recommendation: str,
        business_context: Dict[str, Any],
        tier_level: int,
    ) -> List[MethodActorDialogue]:
        """
        Transform algorithmic results into Method Actor dialogues.

        Operation Lean - Target #3: Now uses persona engine registry (plugin architecture)
        """

        dialogues = []

        # Determine which personas to activate based on tier
        active_personas = [PersonaType.CHARLIE_MUNGER, PersonaType.RUSSELL_ACKOFF]
        if tier_level == 1:
            active_personas = [PersonaType.CHARLIE_MUNGER]  # Just Munger for quick tier

        for persona_type in active_personas:
            # Get persona engine from registry (plugin architecture)
            persona_engine = self.persona_engines.get(persona_type)
            if not persona_engine:
                logger.warning(f"No persona engine registered for {persona_type}")
                continue

            # Delegate dialogue generation to persona engine
            dialogue = await persona_engine.generate_dialogue(
                algorithmic_result=algorithmic_result,
                recommendation=recommendation,
                business_context=business_context,
                thin_variables=self.thin_variables,
                forward_motion_converter=self.forward_motion_converter,
                tone_safeguards=self.tone_safeguards,
            )

            dialogues.append(dialogue)

        return dialogues

    # Operation Lean - Target #3: Dialogue generation moved to persona engines
    # _generate_munger_dialogue() → MungerPersonaEngine.generate_dialogue()
    # _generate_ackoff_dialogue() → AckoffPersonaEngine.generate_dialogue()

    async def _convert_to_forward_motion(
        self,
        dialogues: List[MethodActorDialogue],
        recommendation: str,
        business_context: Dict[str, Any],
    ) -> Dict[str, List[ForwardMotionAction]]:
        """Convert Method Actor dialogues to forward motion actions"""

        forward_motion = {
            ForwardMotionType.EXPERIMENT: [],
            ForwardMotionType.GUARDRAIL: [],
            ForwardMotionType.REVERSIBLE_STEP: [],
            ForwardMotionType.PREMORTEM_SCENARIO: [],
        }

        for dialogue in dialogues:
            for action in dialogue.forward_motion_actions:
                forward_motion[action.action_type].append(action)

        return forward_motion

    def _calculate_enabling_challenger_score(
        self, dialogues: List[MethodActorDialogue]
    ) -> float:
        """Calculate enabling challenger score (vs obstructionist critic)"""
        if not dialogues:
            return 0.0

        total_score = 0.0
        for dialogue in dialogues:
            # Research-validated factors for enabling challengers
            vulnerability_present = (
                "I've made" in dialogue.dialogue_text
                or "I could be wrong" in dialogue.dialogue_text
            )
            solutions_offered = len(dialogue.forward_motion_actions) > 0
            psychological_safety = dialogue.psychological_safety_maintained

            dialogue_score = (
                (0.3 if vulnerability_present else 0.0)
                + (0.4 if solutions_offered else 0.0)
                + (0.3 if psychological_safety else 0.0)
            )
            total_score += dialogue_score

        return total_score / len(dialogues)

    def _calculate_forward_motion_conversion_rate(
        self, forward_motion: Dict[str, List[ForwardMotionAction]]
    ) -> float:
        """Calculate rate of challenge-to-action conversion"""
        total_actions = sum(len(actions) for actions in forward_motion.values())
        return min(1.0, total_actions / 10.0)  # Normalize to 0-1 scale

    def _assess_anti_failure_measures(
        self, dialogues: List[MethodActorDialogue]
    ) -> Dict[str, Any]:
        """Assess how well anti-failure measures are working"""
        gotcha_prevention = all(d.tone_safety_score > 0.8 for d in dialogues)
        psychological_safety = all(d.psychological_safety_maintained for d in dialogues)
        solution_ratio = sum(len(d.forward_motion_actions) for d in dialogues) / max(
            len(dialogues), 1
        )

        return {
            "gotcha_prevention_active": gotcha_prevention,
            "psychological_safety_maintained": psychological_safety,
            "solution_to_criticism_ratio": solution_ratio,
            "overall_safety_score": (
                (0.4 if gotcha_prevention else 0.0)
                + (0.4 if psychological_safety else 0.0)
                + min(0.2, solution_ratio * 0.2)
            ),
        }

    # Operation Lean - Target #3: Helper methods moved to persona engines
    # _get_historical_business_analogy() → MungerPersonaEngine._get_historical_business_analogy()
    # _transform_biases_to_munger_stories() → MungerPersonaEngine._transform_biases_to_munger_stories()
    # _generate_munger_inversion_questions() → MungerPersonaEngine._generate_munger_inversion_questions()
    # _transform_assumptions_to_ackoff_questions() → AckoffPersonaEngine._transform_assumptions_to_ackoff_questions()
    # _generate_ackoff_idealized_design() → AckoffPersonaEngine._generate_ackoff_idealized_design()

    async def _record_method_actor_da_evidence(
        self,
        recommendation: str,
        algorithmic_result: ComprehensiveChallengeResult,
        dialogues: List[MethodActorDialogue],
        forward_motion: Dict[str, List[ForwardMotionAction]],
        business_context: Dict[str, Any],
    ):
        """Record enhanced evidence events for Method Actor DA"""
        try:
            if self.context_stream:
                event_data = {
                    "event_type": "DEVILS_ADVOCATE_METHOD_ACTOR_COMPLETE",
                    "event_id": str(uuid.uuid4()),
                    "timestamp": datetime.now().isoformat(),
                    "original_recommendation": recommendation[:500],
                    # Algorithmic foundation
                    "algorithmic_foundation": {
                        "total_challenges": algorithmic_result.total_challenges_found,
                        "critical_challenges": len(
                            algorithmic_result.critical_challenges
                        ),
                        "overall_risk_score": algorithmic_result.overall_risk_score,
                        "intellectual_honesty_score": algorithmic_result.intellectual_honesty_score,
                        "system_confidence": algorithmic_result.system_confidence,
                    },
                    # Method Actor enhancement
                    "method_actor_enhancement": {
                        "personas_activated": [d.persona_id for d in dialogues],
                        "persona_strength_setting": self.thin_variables[
                            "persona_strength"
                        ],
                        "vulnerability_openings_used": sum(
                            1 for d in dialogues if "I've made" in d.dialogue_text
                        ),
                        "historical_analogies_used": sum(
                            1 for d in dialogues if "reminds me of" in d.dialogue_text
                        ),
                        "idealized_design_questions": sum(
                            1 for d in dialogues if "from scratch" in d.dialogue_text
                        ),
                    },
                    # Forward motion generated
                    "forward_motion_generated": {
                        "experiments_designed": len(
                            forward_motion.get(ForwardMotionType.EXPERIMENT, [])
                        ),
                        "guardrails_created": len(
                            forward_motion.get(ForwardMotionType.GUARDRAIL, [])
                        ),
                        "reversible_steps_identified": len(
                            forward_motion.get(ForwardMotionType.REVERSIBLE_STEP, [])
                        ),
                        "premortem_scenarios": len(
                            forward_motion.get(ForwardMotionType.PREMORTEM_SCENARIO, [])
                        ),
                    },
                    # Research validation metrics
                    "research_validation_metrics": {
                        "enabling_challenger_score": self._calculate_enabling_challenger_score(
                            dialogues
                        ),
                        "forward_motion_conversion_rate": self._calculate_forward_motion_conversion_rate(
                            forward_motion
                        ),
                        "psychological_safety_maintained": all(
                            d.psychological_safety_maintained for d in dialogues
                        ),
                        "anti_failure_measures_active": True,
                    },
                    "business_context": {
                        "industry": business_context.get("industry", "unknown"),
                        "company": business_context.get("company", "unknown"),
                        "stakeholder_count": len(
                            business_context.get("stakeholders", [])
                        ),
                    },
                }

                await self.context_stream.record_event(
                    trace_id=business_context.get("trace_id", str(uuid.uuid4())),
                    event_type="DEVILS_ADVOCATE_METHOD_ACTOR_COMPLETE",
                    event_data=event_data,
                )

                logger.info(
                    "✅ Method Actor DA evidence recorded with forward motion metrics"
                )

        except Exception as e:
            logger.error(f"❌ Failed to record Method Actor DA evidence: {e}")

    # Public API for integration with existing system
    async def run_method_actor_critique(
        self, analysis_results: List[Any], context_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Public API for integration with StatefulPipelineOrchestrator"""
        # Convert analysis results to recommendation text
        if analysis_results:
            recommendation_parts = []
            for result in analysis_results:
                if isinstance(result, dict):
                    for key in [
                        "recommendations",
                        "analysis_content",
                        "key_insights",
                        "findings",
                    ]:
                        if key in result and result[key]:
                            if isinstance(result[key], list):
                                recommendation_parts.extend(
                                    [str(item) for item in result[key]]
                                )
                            else:
                                recommendation_parts.append(str(result[key]))
                else:
                    recommendation_parts.append(str(result))

            combined_recommendation = "; ".join(recommendation_parts[:5])
        else:
            combined_recommendation = (
                "Strategic analysis and recommendations for business optimization"
            )

        # Determine tier level from context
        tier_level = context_data.get("s2_tier", 2)

        try:
            # Run Method Actor comprehensive challenge
            method_actor_result = await self.method_actor_comprehensive_challenge(
                recommendation=combined_recommendation,
                business_context=context_data,
                tier_level=tier_level,
            )

            # Convert to format expected by pipeline orchestrator
            return {
                # Traditional format for compatibility
                "challenges": [
                    action.description
                    for dialogue in method_actor_result.method_actor_dialogues
                    for action in dialogue.forward_motion_actions
                ],
                "bias_warnings": [
                    dialogue.dialogue_text
                    for dialogue in method_actor_result.method_actor_dialogues
                    if dialogue.persona_id == "charlie_munger"
                ],
                "assumption_challenges": [
                    dialogue.dialogue_text
                    for dialogue in method_actor_result.method_actor_dialogues
                    if dialogue.persona_id == "russell_ackoff"
                ],
                "conflicts": [],  # Method Actor approach focuses on forward motion, not conflicts
                # Enhanced Method Actor data
                "method_actor_dialogues": [
                    asdict(d) for d in method_actor_result.method_actor_dialogues
                ],
                "forward_motion_actions": {
                    action_type: [asdict(action) for action in actions]
                    for action_type, actions in method_actor_result.forward_motion_summary.items()
                },
                # Metrics
                "critique_strength": method_actor_result.algorithmic_foundation[
                    "overall_risk_score"
                ],
                "enabling_challenger_score": method_actor_result.enabling_challenger_score,
                "forward_motion_conversion_rate": method_actor_result.forward_motion_conversion_rate,
                "total_challenges_found": method_actor_result.algorithmic_foundation[
                    "total_challenges"
                ],
                "intellectual_honesty_score": method_actor_result.algorithmic_foundation[
                    "intellectual_honesty_score"
                ],
                "system_confidence": method_actor_result.algorithmic_foundation[
                    "system_confidence"
                ],
                # Integration data
                "method_actor_enhanced": True,
                "personas_used": [
                    d.persona_id for d in method_actor_result.method_actor_dialogues
                ],
                "anti_failure_measures": method_actor_result.anti_failure_measures,
            }

        except Exception as e:
            logger.error(f"❌ Method Actor DA failed: {e}")
            # Fallback to basic format
            return {
                "challenges": [f"Method Actor analysis encountered an issue: {str(e)}"],
                "bias_warnings": ["Consider reviewing for cognitive biases"],
                "assumption_challenges": ["Question fundamental assumptions"],
                "conflicts": [],
                "critique_strength": 0.5,
                "error_message": str(e),
                "method_actor_enhanced": False,
            }


# Operation Lean - Target #3: Supporting classes moved to services
# ForwardMotionConverter → src/core/services/forward_motion_converter.py
# ToneSafeguards → src/core/services/tone_safeguards.py


# Factory function for easy integration
def get_method_actor_devils_advocate(
    context_stream: Optional[UnifiedContextStream] = None,
) -> MethodActorDevilsAdvocate:
    """Get Method Actor Devils Advocate instance"""
    return MethodActorDevilsAdvocate(context_stream)


if __name__ == "__main__":
    # Demo the Method Actor Devils Advocate system
    async def demo_method_actor_da():
        print("🎭 Method Actor Devils Advocate - Munger meets Ackoff")
        print("=" * 80)

        da_system = get_method_actor_devils_advocate()

        test_scenario = {
            "recommendation": "We should acquire our main competitor for $500M to eliminate competitive threats and gain immediate market dominance",
            "context": {
                "company": "TechCorp Inc",
                "industry": "Enterprise Software",
                "stakeholders": ["CEO", "Board", "Investors", "Employees"],
                "s2_tier": 3,
                "trace_id": str(uuid.uuid4()),
            },
        }

        result = await da_system.method_actor_comprehensive_challenge(
            test_scenario["recommendation"], test_scenario["context"], tier_level=3
        )

        print("\n🎭 METHOD ACTOR DIALOGUES:")
        for dialogue in result.method_actor_dialogues:
            print(f"\n{dialogue.dialogue_text}")

        print("\n⚡ FORWARD MOTION ACTIONS:")
        total_actions = sum(
            len(actions) for actions in result.forward_motion_summary.values()
        )
        print(f"Total actions generated: {total_actions}")

        for action_type, actions in result.forward_motion_summary.items():
            if actions:
                print(f"\n{action_type.upper()}:")
                for action in actions:
                    print(f"  • {action.description}")

        print("\n📊 METRICS:")
        print(f"  • Enabling Challenger Score: {result.enabling_challenger_score:.3f}")
        print(
            f"  • Forward Motion Conversion: {result.forward_motion_conversion_rate:.3f}"
        )
        print(
            f"  • Anti-Failure Safety: {result.anti_failure_measures['overall_safety_score']:.3f}"
        )

    # Uncomment to run demo
    # asyncio.run(demo_method_actor_da())
    print("🎭 Method Actor Devils Advocate system loaded successfully")
