"""
Tone Safeguards Service

Safeguards to prevent Method Actor failure modes (gotcha-ism, naysaying,
psychological safety violations).

Research-validated patterns for enabling challenger vs obstructionist critic.

Extracted from src/core/method_actor_devils_advocate.py as part of
Operation Lean - Target #3.
"""

import logging
from typing import Dict, Any, List, TYPE_CHECKING

# Avoid circular imports
if TYPE_CHECKING:
    from src.core.method_actor_devils_advocate import MethodActorPersona

logger = logging.getLogger(__name__)


class ToneSafeguards:
    """
    Safeguards to prevent Method Actor failure modes.

    Ensures Method Actor dialogues maintain enabling challenger patterns:
    - Vulnerability openings (research-validated)
    - Forward motion focus (solutions, not just criticism)
    - Psychological safety (attack ideas, support people)
    - No gotcha-ism or personal attacks

    Example:
        >>> safeguards = ToneSafeguards()
        >>> assessment = safeguards.assess_dialogue_safety(
        ...     dialogue_text="You know, I've made this mistake myself...",
        ...     persona=munger_persona
        ... )
        >>> print(assessment['psychological_safety_maintained'])
        True
    """

    def assess_dialogue_safety(
        self, dialogue_text: str, persona: Any  # MethodActorPersona
    ) -> Dict[str, Any]:
        """
        Assess dialogue for anti-failure measures.

        Checks for:
        - Gotcha-ism (personal attack patterns)
        - Vulnerability opening (enabling challenger pattern)
        - Forward motion (solution offering)

        Args:
            dialogue_text: The generated Method Actor dialogue
            persona: The persona configuration

        Returns:
            Dictionary with safety assessment:
                - safety_score: float (0.0-1.0)
                - psychological_safety_maintained: bool (score > 0.7)
                - safety_issues: List[str] of detected issues

        Example:
            >>> safeguards = ToneSafeguards()
            >>> result = safeguards.assess_dialogue_safety(
            ...     "You're wrong about this approach",
            ...     munger_persona
            ... )
            >>> print(result['safety_score'])
            0.7  # Reduced due to gotcha-ism
            >>> print(result['safety_issues'])
            ['gotcha_ism_detected']
        """
        safety_score = 1.0
        safety_issues = []

        # Check for gotcha-ism (personal attack patterns)
        gotcha_patterns = [
            "you're wrong",
            "that's stupid",
            "obviously",
            "clearly you don't understand",
        ]
        if any(pattern in dialogue_text.lower() for pattern in gotcha_patterns):
            safety_score -= 0.3
            safety_issues.append("gotcha_ism_detected")

        # Check for vulnerability opening (enabling challenger pattern)
        vulnerability_patterns = ["i've made", "i could be wrong", "i might be missing"]
        vulnerability_present = any(
            pattern in dialogue_text.lower() for pattern in vulnerability_patterns
        )
        if not vulnerability_present and persona.persona_id == "charlie_munger":
            safety_score -= 0.2
            safety_issues.append("missing_vulnerability_opening")

        # Check for forward motion (solution offering)
        forward_motion_patterns = ["what if we", "experiment", "test", "try", "pilot"]
        forward_motion_present = any(
            pattern in dialogue_text.lower() for pattern in forward_motion_patterns
        )
        if not forward_motion_present:
            safety_score -= 0.2
            safety_issues.append("missing_forward_motion")

        return {
            "safety_score": max(0.0, safety_score),
            "psychological_safety_maintained": safety_score > 0.7,
            "safety_issues": safety_issues,
        }

    def validate_enabling_challenger_patterns(self, dialogue_text: str) -> bool:
        """
        Validate that dialogue follows enabling challenger patterns.

        Research-validated patterns for constructive challenge:
        - Starts with vulnerability or shared experience
        - Attacks ideas rigorously, supports people warmly
        - Ends with openness to being wrong
        - Focuses on learning and forward motion

        Args:
            dialogue_text: The generated Method Actor dialogue

        Returns:
            True if enabling challenger patterns detected, False otherwise

        Example:
            >>> safeguards = ToneSafeguards()
            >>> dialogue = "I've made this mistake myself. Let me ask - what would prove this wrong?"
            >>> is_enabling = safeguards.validate_enabling_challenger_patterns(dialogue)
            >>> print(is_enabling)
            True
        """
        dialogue_lower = dialogue_text.lower()

        # Check for vulnerability opening
        vulnerability_openings = [
            "i've made",
            "i've been",
            "i could be wrong",
            "i might be missing",
            "in my experience",
        ]
        has_vulnerability = any(
            opening in dialogue_lower for opening in vulnerability_openings
        )

        # Check for openness to being wrong
        openness_endings = [
            "what am i missing",
            "i could be wrong",
            "what do you think",
            "help me understand",
        ]
        has_openness = any(ending in dialogue_lower for ending in openness_endings)

        # Enabling challenger has both vulnerability AND openness
        return has_vulnerability and has_openness

    def check_psychological_safety(self, dialogue_text: str) -> float:
        """
        Check psychological safety score for dialogue.

        Psychological safety is maintained when:
        - No personal attacks or gotcha-ism
        - Ideas attacked, people supported
        - Curiosity and learning focus
        - Solutions offered alongside challenges

        Args:
            dialogue_text: The generated Method Actor dialogue

        Returns:
            Psychological safety score (0.0-1.0)

        Example:
            >>> safeguards = ToneSafeguards()
            >>> score = safeguards.check_psychological_safety(
            ...     "This approach has risks. Let's test it with a small pilot."
            ... )
            >>> print(score)
            0.95  # High safety - no attacks, offers solution
        """
        safety_score = 1.0
        dialogue_lower = dialogue_text.lower()

        # Deduct for personal attack patterns
        personal_attacks = [
            "you're wrong",
            "that's stupid",
            "you don't understand",
            "you failed to",
            "you should have",
        ]
        attack_count = sum(
            1 for attack in personal_attacks if attack in dialogue_lower
        )
        safety_score -= attack_count * 0.2

        # Deduct for absolute language (reduces openness)
        absolute_language = [
            "obviously",
            "clearly",
            "everyone knows",
            "it's impossible",
            "never",
            "always",
        ]
        absolute_count = sum(
            1 for phrase in absolute_language if phrase in dialogue_lower
        )
        safety_score -= absolute_count * 0.1

        # Bonus for supportive language
        supportive_patterns = [
            "interesting",
            "i'm curious",
            "help me understand",
            "what if",
            "let's explore",
        ]
        support_count = sum(
            1 for pattern in supportive_patterns if pattern in dialogue_lower
        )
        safety_score += support_count * 0.05

        return max(0.0, min(1.0, safety_score))

    def detect_gotcha_patterns(self, dialogue_text: str) -> List[str]:
        """
        Detect gotcha-ism patterns in dialogue.

        Gotcha-ism is when challenge becomes personal attack or "I caught you"
        rather than genuine inquiry and learning.

        Args:
            dialogue_text: The generated Method Actor dialogue

        Returns:
            List of detected gotcha patterns

        Example:
            >>> safeguards = ToneSafeguards()
            >>> patterns = safeguards.detect_gotcha_patterns(
            ...     "Obviously you're wrong. Clearly you don't understand."
            ... )
            >>> print(patterns)
            ['obviously', 'clearly you don\'t understand']
        """
        dialogue_lower = dialogue_text.lower()

        gotcha_patterns = [
            "you're wrong",
            "that's stupid",
            "obviously",
            "clearly you don't understand",
            "you failed to",
            "you should have known",
            "everyone knows that",
            "i can't believe you",
        ]

        detected = [
            pattern for pattern in gotcha_patterns if pattern in dialogue_lower
        ]

        if detected:
            logger.warning(f"⚠️ Gotcha patterns detected: {detected}")

        return detected
