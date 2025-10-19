from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol, runtime_checkable


@runtime_checkable
class LLMManagerInterface(Protocol):
    """Abstraction for resilient LLM managers consumed by higher layers."""

    async def call_llm(
        self,
        prompt: str,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        ...

    def get_default_model(self) -> str:
        ...
