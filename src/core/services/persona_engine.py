"""
Persona Engine Abstract Base Class

Defines the interface for all Method Actor persona engines in the
ULTRATHINK Devils Advocate system.

This enables a plugin architecture where new personas can be added
without modifying the core orchestrator.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, TYPE_CHECKING

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from src.core.method_actor_devils_advocate import (
        MethodActorPersona,
        MethodActorDialogue,
        PersonaType,
    )
    from src.core.enhanced_devils_advocate_system import ComprehensiveChallengeResult


class PersonaEngine(ABC):
    """
    Abstract base class for Method Actor persona engines.

    Each persona engine implements a specific thought leader's cognitive style
    and communication patterns (e.g., Charlie Munger, Russell Ackoff).

    The plugin architecture allows new personas to be added by:
    1. Creating a new class that inherits from PersonaEngine
    2. Implementing the abstract methods
    3. Registering the persona in the orchestrator

    Example:
        >>> class DalioPersonaEngine(PersonaEngine):
        ...     def generate_dialogue(self, ...):
        ...         # Implement Ray Dalio's principles-based thinking
        ...         pass
        ...
        ...     def get_persona_config(self):
        ...         return MethodActorPersona(
        ...             persona_id="ray_dalio",
        ...             character_archetype="Bridgewater Founder",
        ...             ...
        ...         )
    """

    @property
    @abstractmethod
    def persona_id(self) -> str:
        """
        Unique identifier for this persona.

        Returns:
            Persona ID string (e.g., "charlie_munger", "russell_ackoff")
        """
        pass

    @property
    @abstractmethod
    def persona_type(self) -> "PersonaType":
        """
        PersonaType enum for this persona.

        Returns:
            PersonaType enum value
        """
        pass

    @abstractmethod
    async def generate_dialogue(
        self,
        algorithmic_result: "ComprehensiveChallengeResult",
        recommendation: str,
        business_context: Dict[str, Any],
        thin_variables: Dict[str, Any],
        forward_motion_converter: Any,  # ForwardMotionConverter
        tone_safeguards: Any,  # ToneSafeguards
    ) -> "MethodActorDialogue":
        """
        Generate Method Actor dialogue for this persona.

        This is the core method that transforms algorithmic challenge results
        into persona-specific enabling challenger dialogue.

        Args:
            algorithmic_result: Results from EnhancedDevilsAdvocateSystem
            recommendation: The recommendation being analyzed
            business_context: Business context dict (industry, company, etc.)
            thin_variables: Configuration variables (persona_strength, etc.)
            forward_motion_converter: Service for converting challenges to actions
            tone_safeguards: Service for assessing dialogue safety

        Returns:
            MethodActorDialogue with persona-specific dialogue and actions

        Example:
            >>> result = await engine.generate_dialogue(
            ...     algorithmic_result=algo_result,
            ...     recommendation="Acquire competitor for $500M",
            ...     business_context={"industry": "tech", "company": "ACME"},
            ...     thin_variables={"persona_strength": 0.8},
            ...     forward_motion_converter=converter,
            ...     tone_safeguards=safeguards
            ... )
            >>> print(result.dialogue_text)
            "Charlie Munger speaks: You know, I've made this mistake myself..."
        """
        pass

    @abstractmethod
    def get_persona_config(self) -> "MethodActorPersona":
        """
        Get the full persona configuration.

        Returns:
            MethodActorPersona with complete persona definition

        Example:
            >>> config = engine.get_persona_config()
            >>> print(config.character_archetype)
            "99-year-old Berkshire Hathaway Vice Chairman"
        """
        pass

    def get_signature_methods(self) -> List[str]:
        """
        Get the signature methods/approaches for this persona.

        Returns:
            List of signature method names

        Example:
            >>> methods = engine.get_signature_methods()
            >>> print(methods)
            ["systematic_inversion", "lollapalooza_effect_detection"]
        """
        config = self.get_persona_config()
        return config.signature_methods
