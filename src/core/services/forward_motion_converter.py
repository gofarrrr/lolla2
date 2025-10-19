"""
Forward Motion Converter Service

Converts Method Actor challenges into actionable experiments and guardrails.

Every challenge must be transformed into forward motion actions that enable
learning and progress rather than simple criticism.

Extracted from src/core/method_actor_devils_advocate.py as part of
Operation Lean - Target #3.
"""

import logging
from typing import List, TYPE_CHECKING, Any

# Avoid circular imports
if TYPE_CHECKING:
    from src.core.method_actor_devils_advocate import (
        ForwardMotionAction,
        ForwardMotionType,
    )
    from src.core.enhanced_devils_advocate_system import DevilsAdvocateChallenge

logger = logging.getLogger(__name__)


class ForwardMotionConverter:
    """
    Converts Method Actor challenges into actionable experiments and guardrails.

    Core principle: Every challenge must generate forward motion actions that
    enable learning through small, reversible experiments rather than paralysis.

    Example:
        >>> converter = ForwardMotionConverter()
        >>> actions = await converter.convert_munger_challenges_to_actions(
        ...     challenges=bias_challenges,
        ...     recommendation="Acquire competitor"
        ... )
        >>> print(len(actions))
        4  # 2 experiments + 2 guardrails
    """

    async def convert_munger_challenges_to_actions(
        self, challenges: List[Any], recommendation: str  # DevilsAdvocateChallenge
    ) -> List[Any]:  # ForwardMotionAction
        """
        Convert Munger bias challenges to forward motion actions.

        Munger-style challenges focus on cognitive biases and incentive structures.
        We convert these to experiments that test for bias influence and guardrails
        that maintain awareness.

        Args:
            challenges: List of challenges from Munger persona engine
            recommendation: The recommendation being analyzed

        Returns:
            List of ForwardMotionAction (experiments + guardrails)

        Example:
            >>> challenges = [
            ...     DevilsAdvocateChallenge(
            ...         challenge_type="munger_bias",
            ...         challenge_text="Confirmation bias: seeking supporting evidence",
            ...         ...
            ...     )
            ... ]
            >>> actions = await converter.convert_munger_challenges_to_actions(
            ...     challenges, "Launch new product"
            ... )
            >>> print(actions[0].action_type)
            ForwardMotionType.EXPERIMENT
        """
        from src.core.method_actor_devils_advocate import ForwardMotionAction, ForwardMotionType
        actions = []

        for challenge in challenges:
            if "bias" in challenge.challenge_type:
                # Convert bias detection to experiment
                experiment = ForwardMotionAction(
                    action_type=ForwardMotionType.EXPERIMENT,
                    description=f"Test for {challenge.challenge_text.split(':')[0]} influence",
                    hypothesis=f"Decision may be influenced by {challenge.challenge_text}",
                    test_design="Run parallel decision process without this factor",
                    success_criteria="Different outcomes suggest bias influence",
                    time_horizon="1-2 weeks",
                    cost_estimate="Low - mostly analytical",
                    reversibility="High - no irreversible commitments",
                    learning_objective=f"Validate presence and impact of {challenge.challenge_text}",
                )
                actions.append(experiment)

                # Convert to guardrail
                guardrail = ForwardMotionAction(
                    action_type=ForwardMotionType.GUARDRAIL,
                    description=f"Early warning system for {challenge.challenge_text}",
                    hypothesis="Bias may increase under pressure or time constraints",
                    test_design="Monitor decision confidence and seek contrary evidence",
                    success_criteria="Maintain awareness of bias risk throughout process",
                    time_horizon="Ongoing",
                    cost_estimate="Very Low - checklist item",
                    reversibility="N/A - monitoring only",
                    learning_objective="Prevent bias from distorting judgment",
                )
                actions.append(guardrail)

        return actions

    async def convert_ackoff_challenges_to_actions(
        self, challenges: List[Any], recommendation: str  # DevilsAdvocateChallenge
    ) -> List[Any]:  # ForwardMotionAction
        """
        Convert Ackoff assumption challenges to forward motion actions.

        Ackoff-style challenges focus on dissolving fundamental assumptions.
        We convert these to experiments that test assumption necessity and
        reversible steps that explore idealized design alternatives.

        Args:
            challenges: List of challenges from Ackoff persona engine
            recommendation: The recommendation being analyzed

        Returns:
            List of ForwardMotionAction (experiments + reversible steps)

        Example:
            >>> challenges = [
            ...     DevilsAdvocateChallenge(
            ...         challenge_type="ackoff_dissolution",
            ...         challenge_text="Assumption: we must own the asset",
            ...         ...
            ...     )
            ... ]
            >>> actions = await converter.convert_ackoff_challenges_to_actions(
            ...     challenges, "Acquire competitor"
            ... )
            >>> print(actions[0].action_type)
            ForwardMotionType.EXPERIMENT
        """
        from src.core.method_actor_devils_advocate import ForwardMotionAction, ForwardMotionType
        actions = []

        for challenge in challenges:
            if "assumption" in challenge.challenge_type:
                # Convert assumption dissolution to experiment
                experiment = ForwardMotionAction(
                    action_type=ForwardMotionType.EXPERIMENT,
                    description=f"Test assumption: {challenge.evidence[0] if challenge.evidence else 'fundamental premise'}",
                    hypothesis="This assumption may not be necessary or may be limiting options",
                    test_design="Design small pilot that operates without this assumption",
                    success_criteria="Pilot achieves similar or better outcomes without assumption",
                    time_horizon="2-4 weeks",
                    cost_estimate="Low-Medium - pilot program",
                    reversibility="High - pilot only",
                    learning_objective="Validate necessity of fundamental assumption",
                )
                actions.append(experiment)

                # Convert to reversible step
                reversible_step = ForwardMotionAction(
                    action_type=ForwardMotionType.REVERSIBLE_STEP,
                    description="Implement idealized design alternative as reversible pilot",
                    hypothesis="Alternative approach may be superior to current assumption-based approach",
                    test_design=challenge.mitigation_strategy,
                    success_criteria="Alternative approach demonstrates viability",
                    time_horizon="4-8 weeks",
                    cost_estimate="Medium - implementation pilot",
                    reversibility="High - designed to be reversible",
                    learning_objective="Validate idealized design approach",
                )
                actions.append(reversible_step)

        return actions

    async def convert_challenges_to_actions(
        self,
        challenges: List[Any],  # DevilsAdvocateChallenge
        recommendation: str,
        persona_type: str,
    ) -> List[Any]:  # ForwardMotionAction
        """
        Generic challenge-to-action converter for extensibility.

        This method provides a unified interface for converting challenges
        from any persona type, enabling the plugin architecture.

        Args:
            challenges: List of challenges from any persona engine
            recommendation: The recommendation being analyzed
            persona_type: Type of persona ("charlie_munger", "russell_ackoff", etc.)

        Returns:
            List of ForwardMotionAction appropriate for persona type

        Example:
            >>> actions = await converter.convert_challenges_to_actions(
            ...     challenges=challenges,
            ...     recommendation="Strategic pivot",
            ...     persona_type="charlie_munger"
            ... )
        """
        if persona_type == "charlie_munger":
            return await self.convert_munger_challenges_to_actions(
                challenges, recommendation
            )
        elif persona_type == "russell_ackoff":
            return await self.convert_ackoff_challenges_to_actions(
                challenges, recommendation
            )
        else:
            # Default generic conversion for new personas
            logger.warning(
                f"No specific converter for persona_type={persona_type}, using generic"
            )
            return await self._generic_challenge_conversion(challenges, recommendation)

    async def _generic_challenge_conversion(
        self, challenges: List[Any], recommendation: str  # DevilsAdvocateChallenge
    ) -> List[Any]:  # ForwardMotionAction
        """
        Generic fallback for new persona types.

        Creates basic experiments and guardrails for any challenge type.
        """
        from src.core.method_actor_devils_advocate import ForwardMotionAction, ForwardMotionType
        actions = []

        for challenge in challenges:
            # Create basic experiment
            experiment = ForwardMotionAction(
                action_type=ForwardMotionType.EXPERIMENT,
                description=f"Test challenge: {challenge.challenge_text[:100]}",
                hypothesis=f"Challenge may be valid: {challenge.challenge_text}",
                test_design="Design small experiment to validate or invalidate challenge",
                success_criteria="Clear evidence supporting or refuting challenge",
                time_horizon="1-3 weeks",
                cost_estimate="Low-Medium",
                reversibility="High",
                learning_objective=f"Validate challenge: {challenge.challenge_type}",
            )
            actions.append(experiment)

        return actions
