"""
LLM Call Pipeline Context

Immutable context object passed through pipeline stages.
Contains input parameters, modified state, metadata, and quality scores.

Design Principles:
- Immutability: Frozen dataclass prevents accidental mutations
- Explicit Updates: Stages return new context with updated fields
- Type Safety: Complete type hints, no Any unless unavoidable
- Clarity: Clear separation of input, modified state, and metadata
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime


@dataclass(frozen=True)
class LLMCallContext:
    """
    Immutable context for LLM call pipeline execution.

    Context flows through all pipeline stages, carrying input parameters,
    modifications from stages, metadata, and quality scores.

    Attributes:
        # === INPUT (Original Request) ===
        messages: Original chat messages with 'role' and 'content'
        model: Requested model name (e.g., "deepseek-chat")
        provider: Requested provider key ("deepseek", "anthropic", "openai", "openrouter")
        kwargs: Additional parameters (temperature, max_tokens, etc.)

        # === MODIFIED STATE (Updated by Stages) ===
        modified_messages: Messages after PII redaction, contract injection, RAG context
        modified_provider: Provider after sensitivity routing override
        modified_model: Model after validation/correction
        modified_kwargs: Kwargs after provider-specific filtering

        # === METADATA (Stage Outputs) ===
        stage_metadata: Per-stage metadata (e.g., {"InjectionFirewall": {"action": "sanitize"}})
        quality_scores: Quality metrics (groundedness, self_verification, etc.)
        telemetry: Telemetry data (latency, cost, tokens)
        errors: Non-fatal errors collected during pipeline execution

        # === RESPONSE (Set After LLM Call) ===
        response: LLMResponse object (None until provider call completes)

        # === TIMESTAMPS ===
        created_at: Pipeline start timestamp
        completed_at: Pipeline completion timestamp (None until complete)

    Immutability:
        Context is frozen. Stages must create NEW contexts with updated fields
        using the `with_*()` helper methods.

    Example:
        >>> ctx = LLMCallContext(
        ...     messages=[{"role": "user", "content": "Hello"}],
        ...     model="deepseek-chat",
        ...     provider="deepseek",
        ...     kwargs={}
        ... )
        >>> # Stage updates context
        >>> new_ctx = ctx.with_modified_messages([{"role": "user", "content": "[REDACTED]"}])
        >>> new_ctx.modified_messages
        [{"role": "user", "content": "[REDACTED]"}]
    """

    # === INPUT (Required) ===
    messages: List[Dict[str, str]]
    model: str
    provider: str
    kwargs: Dict[str, Any]

    # === MODIFIED STATE (Optional - None means no modification) ===
    modified_messages: Optional[List[Dict[str, str]]] = None
    modified_provider: Optional[str] = None
    modified_model: Optional[str] = None
    modified_kwargs: Optional[Dict[str, Any]] = None

    # === METADATA (Mutable defaults via field factory) ===
    stage_metadata: Dict[str, Any] = field(default_factory=dict)
    quality_scores: Dict[str, float] = field(default_factory=dict)
    telemetry: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    # === RESPONSE (Set after LLM call) ===
    response: Optional[Any] = None  # LLMResponse object (avoiding circular import)

    # === TIMESTAMPS ===
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    # === HELPER METHODS (Create New Context with Updates) ===

    def with_modified_messages(self, messages: List[Dict[str, str]]) -> "LLMCallContext":
        """
        Create new context with modified messages.

        Args:
            messages: Updated messages list

        Returns:
            New LLMCallContext with modified_messages set

        Example:
            >>> ctx = LLMCallContext(messages=[...], model="...", provider="...", kwargs={})
            >>> new_ctx = ctx.with_modified_messages([{"role": "user", "content": "[REDACTED]"}])
        """
        return self._replace(modified_messages=messages)

    def with_modified_provider(self, provider: str) -> "LLMCallContext":
        """
        Create new context with modified provider.

        Args:
            provider: Updated provider key

        Returns:
            New LLMCallContext with modified_provider set

        Example:
            >>> new_ctx = ctx.with_modified_provider("anthropic")
        """
        return self._replace(modified_provider=provider)

    def with_modified_model(self, model: str) -> "LLMCallContext":
        """
        Create new context with modified model.

        Args:
            model: Updated model name

        Returns:
            New LLMCallContext with modified_model set
        """
        return self._replace(modified_model=model)

    def with_modified_kwargs(self, kwargs: Dict[str, Any]) -> "LLMCallContext":
        """
        Create new context with modified kwargs.

        Args:
            kwargs: Updated kwargs dict

        Returns:
            New LLMCallContext with modified_kwargs set

        Example:
            >>> new_kwargs = ctx.get_effective_kwargs()
            >>> new_kwargs["reasoning_enabled"] = True
            >>> new_ctx = ctx.with_modified_kwargs(new_kwargs)
        """
        return self._replace(modified_kwargs=kwargs)

    def with_stage_metadata(self, stage_name: str, metadata: Dict[str, Any]) -> "LLMCallContext":
        """
        Create new context with stage metadata added.

        Args:
            stage_name: Name of the stage (e.g., "InjectionFirewall")
            metadata: Metadata dict for this stage

        Returns:
            New LLMCallContext with stage_metadata updated

        Example:
            >>> new_ctx = ctx.with_stage_metadata("InjectionFirewall", {"action": "sanitize"})
        """
        new_metadata = {**self.stage_metadata, stage_name: metadata}
        return self._replace(stage_metadata=new_metadata)

    def with_quality_score(self, score_name: str, score_value: float) -> "LLMCallContext":
        """
        Create new context with quality score added.

        Args:
            score_name: Score name (e.g., "groundedness", "self_verification")
            score_value: Score value (0.0-1.0)

        Returns:
            New LLMCallContext with quality_scores updated

        Example:
            >>> new_ctx = ctx.with_quality_score("groundedness", 0.85)
        """
        new_scores = {**self.quality_scores, score_name: score_value}
        return self._replace(quality_scores=new_scores)

    def with_telemetry(self, key: str, value: Any) -> "LLMCallContext":
        """
        Create new context with telemetry data added.

        Args:
            key: Telemetry key (e.g., "latency_ms", "cost_usd")
            value: Telemetry value

        Returns:
            New LLMCallContext with telemetry updated
        """
        new_telemetry = {**self.telemetry, key: value}
        return self._replace(telemetry=new_telemetry)

    def with_error(self, error_message: str) -> "LLMCallContext":
        """
        Create new context with non-fatal error added.

        Args:
            error_message: Error description

        Returns:
            New LLMCallContext with errors list appended

        Note:
            These are non-fatal errors (e.g., validation warnings).
            Fatal errors should raise exceptions.
        """
        new_errors = [*self.errors, error_message]
        return self._replace(errors=new_errors)

    def with_response(self, response: Any) -> "LLMCallContext":
        """
        Create new context with LLM response set.

        Args:
            response: LLMResponse object

        Returns:
            New LLMCallContext with response and completed_at set
        """
        return self._replace(response=response, completed_at=datetime.now())

    def _replace(self, **changes: Any) -> "LLMCallContext":
        """
        Internal helper to create new context with changes.

        Args:
            **changes: Fields to update

        Returns:
            New LLMCallContext with updates applied

        Note:
            This wraps dataclasses.replace() for type safety.
            Stages should use specific with_*() methods instead.
        """
        from dataclasses import replace
        return replace(self, **changes)

    # === GETTERS (Convenience Methods) ===

    def get_effective_messages(self) -> List[Dict[str, str]]:
        """
        Get effective messages (modified if available, else original).

        Returns:
            Modified messages if set, else original messages

        Example:
            >>> ctx.get_effective_messages()
            [{"role": "user", "content": "[REDACTED]"}]
        """
        return self.modified_messages if self.modified_messages is not None else self.messages

    def get_effective_provider(self) -> str:
        """
        Get effective provider (modified if available, else original).

        Returns:
            Modified provider if set, else original provider
        """
        return self.modified_provider if self.modified_provider is not None else self.provider

    def get_effective_model(self) -> str:
        """
        Get effective model (modified if available, else original).

        Returns:
            Modified model if set, else original model
        """
        return self.modified_model if self.modified_model is not None else self.model

    def get_effective_kwargs(self) -> Dict[str, Any]:
        """
        Get effective kwargs (modified if available, else original).

        Returns:
            Modified kwargs if set, else original kwargs (returns copy to prevent mutations)
        """
        if self.modified_kwargs is not None:
            return {**self.modified_kwargs}
        return {**self.kwargs}

    def has_errors(self) -> bool:
        """
        Check if any non-fatal errors were collected.

        Returns:
            True if errors list is non-empty
        """
        return len(self.errors) > 0

    def has_response(self) -> bool:
        """
        Check if LLM response has been set.

        Returns:
            True if response is not None
        """
        return self.response is not None

    def is_complete(self) -> bool:
        """
        Check if pipeline execution is complete.

        Returns:
            True if completed_at timestamp is set
        """
        return self.completed_at is not None

    def get_stage_metadata(self, stage_name: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for specific stage.

        Args:
            stage_name: Name of the stage

        Returns:
            Stage metadata dict or None if not set
        """
        return self.stage_metadata.get(stage_name)

    def get_quality_score(self, score_name: str) -> Optional[float]:
        """
        Get specific quality score.

        Args:
            score_name: Score name (e.g., "groundedness")

        Returns:
            Score value or None if not set
        """
        return self.quality_scores.get(score_name)

    def get_telemetry_value(self, key: str) -> Optional[Any]:
        """
        Get specific telemetry value.

        Args:
            key: Telemetry key

        Returns:
            Telemetry value or None if not set
        """
        return self.telemetry.get(key)

    # === VALIDATION ===

    def __post_init__(self) -> None:
        """
        Post-initialization validation.

        Validates:
        - messages is non-empty list
        - model is non-empty string
        - provider is non-empty string
        - kwargs is a dict

        Raises:
            ValueError: If validation fails
        """
        # Validate messages
        if not isinstance(self.messages, list) or len(self.messages) == 0:
            raise ValueError("messages must be a non-empty list")

        # Validate each message has role and content
        for i, msg in enumerate(self.messages):
            if not isinstance(msg, dict):
                raise ValueError(f"Message {i} must be a dict")
            if "role" not in msg or "content" not in msg:
                raise ValueError(f"Message {i} must have 'role' and 'content' keys")

        # Validate model
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a non-empty string")

        # Validate provider
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ValueError("provider must be a non-empty string")

        # Validate kwargs
        if not isinstance(self.kwargs, dict):
            raise ValueError("kwargs must be a dict")

    def __repr__(self) -> str:
        """
        String representation for debugging.

        Returns:
            Human-readable representation

        Example:
            >>> ctx
            LLMCallContext(provider='deepseek', model='deepseek-chat', messages=1, response=None)
        """
        return (
            f"LLMCallContext("
            f"provider='{self.get_effective_provider()}', "
            f"model='{self.get_effective_model()}', "
            f"messages={len(self.get_effective_messages())}, "
            f"response={'set' if self.has_response() else 'None'}, "
            f"errors={len(self.errors)}"
            f")"
        )
