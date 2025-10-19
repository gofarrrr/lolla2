# src/engine/core/contracts.py
from __future__ import annotations

from typing import Protocol, Dict, Any, List, Optional
from pydantic import BaseModel, Field


class ResearchResult(BaseModel):
    """Standardized research result for all providers."""

    content: str = Field(default="")
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    raw_response: Any = None
    confidence: float = 0.0
    processing_time_ms: int = 0
    provider_name: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IResearchProvider(Protocol):
    """Unified research provider interface for LLMManager or ResearchManager policy layer."""

    provider_name: str  # human-readable identifier

    async def query(self, query_text: str, config: Dict[str, Any] | None = None) -> ResearchResult:
        """Execute a research query and return a standardized ResearchResult."""
        ...
