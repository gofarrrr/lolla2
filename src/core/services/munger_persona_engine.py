"""
Charlie Munger Persona Engine

Implements Charlie Munger's cognitive style and communication patterns
for the Method Actor Devils Advocate system.

Munger's approach:
- Systematic inversion thinking
- Pattern recognition from 6 decades of investment mistakes
- Cognitive bias awareness
- Folksy wisdom delivery with vulnerability openings
- Historical analogies and base rate pattern matching

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


class MungerPersonaEngine(PersonaEngine):
    """
    Charlie Munger Method Actor Persona Engine.

    Implements the 99-year-old Berkshire Hathaway Vice Chairman's
    distinctive cognitive style and communication patterns.

    Key characteristics:
    - Inversion-first thinking
    - Pattern recognition from historical failures
    - Systematic bias awareness
    - Vulnerability openings ("I've made this mistake myself...")
    - Ends with openness ("What am I missing?")

    Example:
        >>> engine = MungerPersonaEngine()
        >>> dialogue = await engine.generate_dialogue(
        ...     algorithmic_result=result,
        ...     recommendation="Acquire competitor",
        ...     business_context={"industry": "tech"},
        ...     thin_variables={"persona_strength": 0.8},
        ...     forward_motion_converter=converter,
        ...     tone_safeguards=safeguards
        ... )
        >>> print(dialogue.dialogue_text[:50])
        "Charlie Munger: You know, I've made this mistake..."
    """

    @property
    def persona_id(self) -> str:
        return "charlie_munger"

    @property
    def persona_type(self) -> "PersonaType":
        from src.core.method_actor_devils_advocate import PersonaType
        return PersonaType.CHARLIE_MUNGER

    def get_persona_config(self) -> "MethodActorPersona":
        """Get Charlie Munger persona configuration."""
        from src.core.method_actor_devils_advocate import MethodActorPersona
        return MethodActorPersona(
            persona_id="charlie_munger",
            character_archetype="99-year-old Berkshire Hathaway Vice Chairman",
            background="6 decades of investment mistakes, legal training, voracious multidisciplinary reader, partner to Warren Buffett",
            cognitive_style="Inversion-first, pattern recognition from history, systematic bias awareness, folksy wisdom delivery",
            communication_patterns={
                "vulnerability_first": "You know, I've made this mistake myself...",
                "historical_analogies": "This reminds me of [historical pattern]...",
                "inversion_questions": "What would guarantee this fails spectacularly?",
                "openness_ending": "Now, I could be completely wrong here. What am I missing?",
                "problem_focus": "Attack the idea rigorously, support the person warmly",
                "pattern_sharing": "I've seen this movie before, and here's how it usually ends...",
            },
            signature_methods=[
                "systematic_inversion",
                "lollapalooza_effect_detection",
                "incentive_structure_analysis",
                "base_rate_pattern_matching",
                "multidisciplinary_mental_models",
            ],
            avoid_patterns=[
                "gotcha_ism",
                "personal_attack",
                "pure_negativity",
                "academic_jargon",
            ],
            forward_motion_style="What small, cheap test would prove or disprove this assumption?",
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
        """Generate Charlie Munger Method Actor dialogue."""
        from src.core.method_actor_devils_advocate import MethodActorDialogue

        # Extract bias-related challenges for Munger focus
        munger_challenges = [
            c
            for c in algorithmic_result.critical_challenges
            if c.challenge_type in ["munger_bias", "cognitive_audit"]
        ]

        # Generate historical analogy
        historical_analogy = self._get_historical_business_analogy(recommendation)

        # Build Munger dialogue with research-validated communication patterns
        dialogue_parts = []

        # Vulnerability opening (research-validated)
        if thin_variables.get("vulnerability_opening", True):
            dialogue_parts.append(
                f"**Charlie Munger** *adjusts glasses, speaks slowly and thoughtfully*:\n\n"
                f"\"You know, I've been at this for over 60 years, and I've made every mistake in the book at least once. "
                f'This situation reminds me of {historical_analogy}."'
            )

        # Transform bias challenges to stories (Munger's style)
        if munger_challenges and thin_variables.get("historical_analogy_mode", True):
            dialogue_parts.append(
                f"\n\"Here's what I've learned from similar patterns: "
                f'{self._transform_biases_to_munger_stories(munger_challenges[:3])}"'
            )

        # Inversion questions (signature Munger method)
        inversion_questions = self._generate_munger_inversion_questions(
            recommendation, munger_challenges
        )
        if inversion_questions:
            dialogue_parts.append(
                f'\n"Let me ask you this - what would have to be true for this to fail spectacularly? '
                f'{inversion_questions}"'
            )

        # Openness ending (anti-gotcha safeguard)
        if thin_variables.get("gotcha_prevention", 0.9) > 0.8:
            dialogue_parts.append(
                '\n"Now, I could be completely wrong here. What am I missing that would change my mind? '
                "The beauty of being 99 years old is you can afford to be wrong - it's the learning that matters.\""
            )

        dialogue_text = "".join(dialogue_parts)

        # Generate forward motion actions
        forward_actions = (
            await forward_motion_converter.convert_munger_challenges_to_actions(
                munger_challenges, recommendation
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
            challenges_addressed=[c.challenge_id for c in munger_challenges],
            forward_motion_actions=forward_actions,
            tone_safety_score=tone_safety["safety_score"],
            psychological_safety_maintained=tone_safety[
                "psychological_safety_maintained"
            ],
        )

    def _get_historical_business_analogy(self, recommendation: str) -> str:
        """Generate appropriate historical business analogy for Munger style."""
        # Simple pattern matching - would be more sophisticated in production
        if "acquire" in recommendation.lower() or "merger" in recommendation.lower():
            return "the AOL-Time Warner merger in 2000. Everyone said 'synergy' but nobody counted the real costs"
        elif "ai" in recommendation.lower() or "technology" in recommendation.lower():
            return "the dot-com bubble when everyone had 'new economy' thinking"
        elif "pivot" in recommendation.lower() or "strategy" in recommendation.lower():
            return "Coca-Cola's New Coke disaster - sometimes the old way works for good reasons"
        else:
            return "many deals I've seen where the excitement blinded people to the fundamentals"

    def _transform_biases_to_munger_stories(
        self, challenges: List[Any]  # DevilsAdvocateChallenge
    ) -> str:
        """Transform bias challenges into Munger-style stories."""
        stories = []
        for challenge in challenges[:2]:  # Limit for readability
            if "confirmation" in challenge.challenge_text.lower():
                stories.append(
                    "I see confirmation bias here - we're looking for evidence that supports what we want to believe"
                )
            elif "anchor" in challenge.challenge_text.lower():
                stories.append(
                    "classic anchoring bias - the first number we heard is pulling our judgment"
                )
            elif "incentive" in challenge.challenge_text.lower():
                stories.append(
                    "show me the incentives and I'll show you the outcome - whose incentives are we missing?"
                )

        return (
            ". ".join(stories)
            if stories
            else "several cognitive patterns that usually lead to trouble"
        )

    def _generate_munger_inversion_questions(
        self, recommendation: str, challenges: List[Any]  # DevilsAdvocateChallenge
    ) -> str:
        """Generate Munger-style inversion questions."""
        questions = [
            "What would guarantee this fails spectacularly?",
            "What are we assuming that could be completely wrong?",
            "Who benefits if this goes wrong, and are we listening to them?",
        ]

        # Add challenge-specific inversion questions
        for challenge in challenges[:1]:
            if "bias" in challenge.challenge_type:
                questions.append(
                    "What if our judgment is being distorted by cognitive biases?"
                )
            elif "assumption" in challenge.challenge_type:
                questions.append("What if the fundamental premise is flawed?")

        return " ".join(questions[:3])  # Limit for readability
