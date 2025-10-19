"""
Configuration Loader Service

Handles YAML configuration loading and thin variables for the
Method Actor Devils Advocate system.

Extracted from src/core/method_actor_devils_advocate.py as part of
Operation Lean - Target #3.
"""

import logging
import os
import yaml
from typing import Dict, Any, Optional, TYPE_CHECKING

# Avoid circular imports - import ConfigurationError separately
import sys
# ConfigurationError will be defined in the main file, but we import it safely
class ConfigurationError(Exception):
    """Raised when YAML configuration is invalid or missing"""
    pass

# Use TYPE_CHECKING to avoid circular import for MethodActorPersona
if TYPE_CHECKING:
    from src.core.method_actor_devils_advocate import MethodActorPersona

logger = logging.getLogger(__name__)


class ConfigurationLoader:
    """
    Configuration Loader for Method Actor Devils Advocate.

    Loads YAML configuration files and provides fallback to hardcoded defaults.
    Handles thin variables and persona configurations.

    Example:
        >>> loader = ConfigurationLoader()
        >>> config = loader.load_yaml_config("config.yaml")
        >>> thin_vars = loader.load_thin_variables(config)
        >>> personas = loader.load_personas_from_yaml(config)
    """

    def load_yaml_config(self, yaml_config_path: str) -> Dict[str, Any]:
        """
        Load and parse YAML configuration file.

        Args:
            yaml_config_path: Path to YAML configuration file

        Returns:
            Dictionary with configuration data from NWAY_DEVILS_ADVOCATE_001 section

        Raises:
            ConfigurationError: If file not found, invalid YAML, or missing section

        Example:
            >>> loader = ConfigurationLoader()
            >>> config = loader.load_yaml_config("cognitive_architecture/NWAY_DEVILS_ADVOCATE_001.yaml")
            >>> print(config.keys())
            dict_keys(['personas', 'thin_variables', 's2_tier_integration'])
        """
        try:
            if not os.path.exists(yaml_config_path):
                raise ConfigurationError(
                    f"YAML configuration file not found: {yaml_config_path}"
                )

            with open(yaml_config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            # Validate that the main NWAY entry exists
            if "NWAY_DEVILS_ADVOCATE_001" not in config:
                raise ConfigurationError(
                    "YAML file missing 'NWAY_DEVILS_ADVOCATE_001' configuration"
                )

            logger.info(f"✅ Loaded YAML configuration from {yaml_config_path}")
            return config["NWAY_DEVILS_ADVOCATE_001"]

        except yaml.YAMLError as e:
            raise ConfigurationError(
                f"Invalid YAML format in {yaml_config_path}: {e}"
            )
        except Exception as e:
            raise ConfigurationError(f"Failed to load YAML configuration: {e}")

    def load_thin_variables(
        self, yaml_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Load thin variables from YAML or use defaults.

        Thin variables are fine-grained control parameters for persona behavior.

        Args:
            yaml_config: Optional YAML config dict (output from load_yaml_config)

        Returns:
            Dictionary of thin variables with defaults applied

        Example:
            >>> loader = ConfigurationLoader()
            >>> # Load from YAML
            >>> config = loader.load_yaml_config("config.yaml")
            >>> thin_vars = loader.load_thin_variables(config)
            >>> print(thin_vars['persona_strength'])
            0.8
            >>>
            >>> # Use defaults
            >>> thin_vars = loader.load_thin_variables()
            >>> print(thin_vars['gotcha_prevention'])
            0.9
        """
        if yaml_config and "thin_variables" in yaml_config:
            yaml_vars = yaml_config["thin_variables"]
            logger.info(f"✅ Loaded {len(yaml_vars)} thin variables from YAML")
            return yaml_vars
        else:
            # Default thin variables if no YAML config
            defaults = {
                "persona_strength": 0.8,
                "vulnerability_opening": True,
                "historical_analogy_mode": True,
                "idealized_design_mode": True,
                "forward_motion_required": True,
                "gotcha_prevention": 0.9,
                "solution_suggestion": True,
                "psychological_safety": 0.95,
            }
            logger.info("Using default thin variables (no YAML config provided)")
            return defaults

    def load_personas_from_yaml(
        self, yaml_config: Dict[str, Any]
    ) -> Dict[str, Any]:  # MethodActorPersona
        """
        Load Method Actor personas from YAML configuration.

        Args:
            yaml_config: YAML config dict from load_yaml_config()

        Returns:
            Dictionary mapping persona_id to MethodActorPersona

        Raises:
            KeyError: If yaml_config missing 'personas' section

        Example:
            >>> loader = ConfigurationLoader()
            >>> config = loader.load_yaml_config("config.yaml")
            >>> personas = loader.load_personas_from_yaml(config)
            >>> print(personas.keys())
            dict_keys(['charlie_munger', 'russell_ackoff'])
        """
        from src.core.method_actor_devils_advocate import MethodActorPersona
        personas = {}
        yaml_personas = yaml_config["personas"]

        logger.info(f"✅ Loading {len(yaml_personas)} personas from YAML")

        for persona_key, persona_data in yaml_personas.items():
            try:
                # Extract communication patterns
                comm_patterns = persona_data.get("communication_patterns", {})

                # Create MethodActorPersona from YAML data
                persona = MethodActorPersona(
                    persona_id=persona_key,
                    character_archetype=persona_data.get("archetype", ""),
                    background=persona_data.get("background", ""),
                    cognitive_style=", ".join(persona_data.get("cognitive_style", [])),
                    communication_patterns=comm_patterns,
                    signature_methods=persona_data.get("signature_methods", []),
                    avoid_patterns=persona_data.get("avoid_patterns", []),
                    forward_motion_style=persona_data.get("forward_motion_style", ""),
                    token_budget=persona_data.get("token_budget", 200),
                )

                personas[persona_key] = persona
                logger.info(
                    f"   • Loaded persona: {persona_key} ({persona.character_archetype})"
                )

            except Exception as e:
                logger.warning(f"   • Failed to load persona {persona_key}: {e}")
                continue

        return personas
