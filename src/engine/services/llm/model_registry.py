"""
Model Registry - Provider/Model Validation and Safety Guardrails

Prevents model drift, parameter leakage, and provider misconfiguration.
Maintains allowlist of valid model/provider combinations.

Architecture:
- Centralized model allowlist per provider
- Provider-specific parameter validation
- Automatic parameter scrubbing to prevent leakage
- Policy violation logging

Safety Features:
- Prevents accidental model drift (e.g., grok-beta vs grok-4-fast)
- Blocks unknown model/provider combinations
- Scrubs provider-specific kwargs to prevent API errors
- Logs all validation failures for audit
"""

import logging
from typing import Dict, Set, Any, Optional, List
from dataclasses import dataclass


logger = logging.getLogger(__name__)


# ALLOWED MODEL/PROVIDER COMBINATIONS
# This is the single source of truth for valid models
ALLOWED_MODELS: Dict[str, Set[str]] = {
    "openrouter": {
        "grok-4-fast",
        "x-ai/grok-4-fast",  # Full model path
        "grok-4-fast:free",
        "x-ai/grok-4-fast:free",
        "grok-beta",
        "x-ai/grok-beta",  # Full model path
        "grok-2-latest",
        "x-ai/grok-2-latest",
        "grok-2-public",
        "x-ai/grok-2-public",
        "x-ai/grok-2",
        "x-ai/grok-1.5",
        "x-ai/grok-1.5-mini",
        "deepseek-chat",  # Allow DeepSeek via OpenRouter
        "deepseek/deepseek-chat-v3.1",  # Full DeepSeek path
    },
    "deepseek": {
        "deepseek-reasoner",  # V3.1 thinking mode
        "deepseek-chat",  # V3.1 non-thinking mode
    },
    "anthropic": {
        "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-latest",
        "claude-3-opus-20240229",
    },
    "openai": {
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
    },
}

# PROVIDER-SPECIFIC ALLOWED PARAMETERS
# Parameters that are valid for each provider
ALLOWED_PARAMS: Dict[str, Set[str]] = {
    "openrouter": {
        "temperature",
        "max_tokens",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "reasoning_enabled",  # Grok-4-Fast specific
        "response_format",  # Structured output
    },
    "deepseek": {
        "temperature",
        "max_tokens",
        "top_p",
        "response_format",
        "stream",
    },
    "anthropic": {
        "temperature",
        "max_tokens",
        "top_p",
        "top_k",
        "stop_sequences",
    },
    "openai": {
        "temperature",
        "max_tokens",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "stop",
        "response_format",
    },
}


@dataclass
class ModelValidationResult:
    """Result of model validation"""

    valid: bool
    provider: str
    model: str
    error_message: Optional[str] = None
    suggested_model: Optional[str] = None


@dataclass
class ParamScrubResult:
    """Result of parameter scrubbing"""

    scrubbed_params: Dict[str, Any]
    removed_params: List[str]
    warnings: List[str]


class ModelRegistry:
    """Registry for model/provider validation and safety"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.violation_count = 0

    def validate_model(self, provider: str, model: str) -> ModelValidationResult:
        """
        Validate that model is allowed for provider.

        Args:
            provider: Provider name (e.g., "openrouter", "deepseek")
            model: Model name (e.g., "grok-4-fast", "deepseek-reasoner")

        Returns:
            ModelValidationResult with validation status and suggestions

        Example:
            >>> registry = ModelRegistry()
            >>> result = registry.validate_model("openrouter", "gpt-4o")
            >>> result.valid
            False
            >>> result.error_message
            'Model gpt-4o not allowed for provider openrouter'
        """
        # Check if provider exists
        if provider not in ALLOWED_MODELS:
            return ModelValidationResult(
                valid=False,
                provider=provider,
                model=model,
                error_message=f"Unknown provider: {provider}",
                suggested_model=None,
            )

        # Check if model is allowed for provider
        if model not in ALLOWED_MODELS[provider]:
            # Try to suggest a model
            suggested = self._suggest_model(provider, model)

            self.violation_count += 1
            self.logger.error(
                f"❌ MODEL POLICY VIOLATION #{self.violation_count}: "
                f"Model '{model}' not allowed for provider '{provider}'. "
                f"Suggested: {suggested}"
            )

            return ModelValidationResult(
                valid=False,
                provider=provider,
                model=model,
                error_message=f"Model {model} not allowed for provider {provider}",
                suggested_model=suggested,
            )

        # Valid model/provider combination
        return ModelValidationResult(
            valid=True, provider=provider, model=model, error_message=None
        )

    def validate_or_raise(self, provider: str, model: str):
        """
        Validate model and raise ValueError if invalid.

        Args:
            provider: Provider name
            model: Model name

        Raises:
            ValueError: If model is not allowed for provider
        """
        result = self.validate_model(provider, model)

        if not result.valid:
            raise ValueError(
                f"{result.error_message}. "
                f"Allowed models for {provider}: {', '.join(ALLOWED_MODELS.get(provider, set()))}. "
                f"Suggested model: {result.suggested_model or 'none'}"
            )

    def scrub_provider_specific_kwargs(
        self, provider: str, kwargs: Dict[str, Any]
    ) -> ParamScrubResult:
        """
        Scrub provider-specific kwargs to prevent parameter leakage.

        Removes parameters that are not allowed for the specified provider.
        This prevents API errors when parameters leak across providers.

        Args:
            provider: Provider name
            kwargs: Original kwargs dict

        Returns:
            ParamScrubResult with scrubbed params and warnings

        Example:
            >>> registry = ModelRegistry()
            >>> result = registry.scrub_provider_specific_kwargs(
            ...     "anthropic",
            ...     {"temperature": 0.7, "reasoning_enabled": True}
            ... )
            >>> result.scrubbed_params
            {'temperature': 0.7}
            >>> result.removed_params
            ['reasoning_enabled']
        """
        if provider not in ALLOWED_PARAMS:
            self.logger.warning(f"⚠️ Unknown provider: {provider}, allowing all params")
            return ParamScrubResult(
                scrubbed_params=kwargs.copy(), removed_params=[], warnings=[]
            )

        allowed = ALLOWED_PARAMS[provider]
        scrubbed = {}
        removed = []
        warnings = []

        for key, value in kwargs.items():
            if key in allowed:
                scrubbed[key] = value
            else:
                removed.append(key)
                warnings.append(
                    f"Parameter '{key}' not allowed for provider '{provider}', removed"
                )
                self.logger.warning(
                    f"⚠️ PARAM SCRUBBED: '{key}' not allowed for {provider}"
                )

        if removed:
            self.logger.info(
                f"🔒 Scrubbed {len(removed)} params for {provider}: {', '.join(removed)}"
            )

        return ParamScrubResult(
            scrubbed_params=scrubbed, removed_params=removed, warnings=warnings
        )

    def _suggest_model(self, provider: str, attempted_model: str) -> Optional[str]:
        """Suggest a similar model for the provider"""
        allowed = ALLOWED_MODELS.get(provider, set())

        if not allowed:
            return None

        # Simple heuristic: suggest first model that shares a word
        attempted_words = set(attempted_model.lower().split("-"))

        for allowed_model in allowed:
            allowed_words = set(allowed_model.lower().split("-"))
            if attempted_words & allowed_words:  # Intersection
                return allowed_model

        # Fallback: return first allowed model
        return list(allowed)[0]

    def get_allowed_models(self, provider: str) -> Set[str]:
        """Get allowed models for provider"""
        return ALLOWED_MODELS.get(provider, set()).copy()

    def get_allowed_params(self, provider: str) -> Set[str]:
        """Get allowed parameters for provider"""
        return ALLOWED_PARAMS.get(provider, set()).copy()

    def get_violation_count(self) -> int:
        """Get total number of policy violations"""
        return self.violation_count


# Global registry instance
_model_registry: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    """Get or create global model registry"""
    global _model_registry

    if _model_registry is None:
        _model_registry = ModelRegistry()

    return _model_registry
