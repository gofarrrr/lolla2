"""
Russell Ackoff Persona Engine

Implements Russell Ackoff's cognitive style and communication patterns
for the Method Actor Devils Advocate system.

Ackoff's approach:
- Systems thinking and holistic analysis
- Assumption dissolution and idealized design
- Gentle but persistent inquiry
- Possibility generation rather than criticism
- Question assumptions, not intelligence

Extracted from src/core/method_actor_devils_advocate.py as part of
Operation Lean - Target #3.
"""

import logging
from typing import Dict, List, Any, TYPE_CHECKING

from src.core.services.persona_engine import PersonaEngine

# Avoid circular imports
if TYPE_CHECKING:
    from src.core.method_actor_devils_advocate import (
        MethodActorPersona,
        MethodActorDialogue,
        PersonaType,
    )
    from src.core.enhanced_devils_advocate_system import (
        ComprehensiveChallengeResult,
        DevilsAdvocateChallenge,
    )

logger = logging.getLogger(__name__)


class AckoffPersonaEngine(PersonaEngine):
    """
    Russell Ackoff Method Actor Persona Engine.

    Implements the systems thinking pioneer's distinctive cognitive style
    and assumption-dissolving communication patterns.

    Key characteristics:
    - Systems-holistic thinking
    - Assumption questioning and dissolution
    - Idealized design methodology
    - Gentle but persistent inquiry
    - Possibility generation (opens up options)

    Example:
        >>> engine = AckoffPersonaEngine()
        >>> dialogue = await engine.generate_dialogue(
        ...     algorithmic_result=result,
        ...     recommendation="Launch new product",
        ...     business_context={"industry": "consumer goods"},
        ...     thin_variables={"idealized_design_mode": True},
        ...     forward_motion_converter=converter,
        ...     tone_safeguards=safeguards
        ... )
        >>> print("idealized design" in dialogue.dialogue_text.lower())
        True
    """

    @property
    def persona_id(self) -> str:
        return "russell_ackoff"

    @property
    def persona_type(self) -> "PersonaType":
        from src.core.method_actor_devils_advocate import PersonaType
        return PersonaType.RUSSELL_ACKOFF

    def get_persona_config(self) -> "MethodActorPersona":
        """Get Russell Ackoff persona configuration."""
        from src.core.method_actor_devils_advocate import MethodActorPersona
        return MethodActorPersona(
            persona_id="russell_ackoff",
            character_archetype="Systems thinking pioneer and assumption dissolver",
            background="50+ years dissolving business problems, operations research pioneer, idealized design methodology creator",
            cognitive_style="Systems-holistic, assumption-questioning, possibility-generating, gentle but persistent inquiry",
            communication_patterns={
                "assumption_questioning": "What if we dissolved the assumption that...",
                "idealized_design": "If I were designing this system from absolute scratch...",
                "systems_reframing": "How does this serve the larger system's purpose?",
                "possibility_generation": "This opens up fascinating possibilities we hadn't considered...",
                "gentle_dissolution": "Question assumptions, not intelligence or worth",
                "curiosity_driven": "I'm curious about the assumptions we're making that we don't realize...",
            },
            signature_methods=[
                "assumption_dissolution",
                "idealized_design_thinking",
                "systems_boundary_analysis",
                "purposeful_system_design",
                "interactive_planning",
            ],
            avoid_patterns=[
                "nihilism",
                "scope_creep",
                "over_complexity",
                "paralysis_by_analysis",
            ],
            forward_motion_style="What experiment tests this fundamental assumption in the smallest way possible?",
            token_budget=200,
        )

    async def generate_dialogue(
        self,
        algorithmic_result: "ComprehensiveChallengeResult",
        recommendation: str,
        business_context: Dict[str, Any],
        thin_variables: Dict[str, Any],
        forward_motion_converter: Any,
        tone_safeguards: Any,
    ) -> "MethodActorDialogue":
        """Generate Russell Ackoff Method Actor dialogue."""
        from src.core.method_actor_devils_advocate import MethodActorDialogue

        # Extract assumption-related challenges for Ackoff focus
        ackoff_challenges = [
            c
            for c in algorithmic_result.critical_challenges
            if c.challenge_type in ["ackoff_dissolution", "llm_sceptic"]
        ]

        # Build Ackoff dialogue with gentle questioning style
        dialogue_parts = []

        # Curiosity-driven opening
        dialogue_parts.append(
            "**Russell Ackoff** *leans forward with genuine curiosity*:\n\n"
            "\"This is fascinating because of what we're assuming without realizing it. "
            "I've spent 50 years learning that we fail more often by solving the wrong problem "
            'than by getting the wrong solution to the right problem."'
        )

        # Transform assumptions to gentle questions
        if ackoff_challenges and thin_variables.get("idealized_design_mode", True):
            assumption_questions = self._transform_assumptions_to_ackoff_questions(
                ackoff_challenges[:3]
            )
            dialogue_parts.append(
                f'\n"Let me share what I\'m curious about: {assumption_questions}"'
            )

        # Idealized design thinking
        idealized_alternative = self._generate_ackoff_idealized_design(
            recommendation, business_context
        )
        dialogue_parts.append(
            f"\n\"Here's what interests me most - if I were designing this system from absolute scratch, "
            f"knowing everything we know now, would it even include this approach? "
            f'{idealized_alternative}"'
        )

        # Possibility generation ending (forward motion focus)
        dialogue_parts.append(
            "\n\"What fascinating possibilities does that open up that we hadn't considered? "
            'Sometimes the best thing you can do to a problem is dissolve it entirely."'
        )

        dialogue_text = "".join(dialogue_parts)

        # Generate forward motion actions
        forward_actions = (
            await forward_motion_converter.convert_ackoff_challenges_to_actions(
                ackoff_challenges, recommendation
            )
        )

        # Assess tone safety
        persona_config = self.get_persona_config()
        tone_safety = tone_safeguards.assess_dialogue_safety(
            dialogue_text, persona_config
        )

        return MethodActorDialogue(
            persona_id=self.persona_id,
            dialogue_text=dialogue_text,
            challenges_addressed=[c.challenge_id for c in ackoff_challenges],
            forward_motion_actions=forward_actions,
            tone_safety_score=tone_safety["safety_score"],
            psychological_safety_maintained=tone_safety[
                "psychological_safety_maintained"
            ],
        )

    def _transform_assumptions_to_ackoff_questions(
        self, challenges: List[Any]  # DevilsAdvocateChallenge
    ) -> str:
        """Transform assumption challenges into gentle Ackoff-style questions."""
        questions = []
        for challenge in challenges:
            if "assumption" in challenge.challenge_text.lower():
                questions.append(
                    f"What if we dissolved the assumption that {challenge.evidence[0] if challenge.evidence else 'this approach is necessary'}?"
                )

        if not questions:
            questions = [
                "What assumptions are we making that we don't realize we're making?"
            ]

        return " ".join(questions[:2])

    def _generate_ackoff_idealized_design(
        self, recommendation: str, business_context: Dict[str, Any]
    ) -> str:
        """Generate Ackoff-style idealized design alternative."""
        industry = business_context.get("industry", "business")

        if "acquire" in recommendation.lower():
            return f"Maybe the ideal {industry} system achieves the same outcome through partnership, internal development, or entirely different market positioning"
        elif "investment" in recommendation.lower():
            return "Perhaps the idealized approach focuses on capability building rather than capital deployment"
        else:
            return "The ideal system might achieve the same purpose through a completely different mechanism we haven't considered"
