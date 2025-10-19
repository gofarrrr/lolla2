"""
ResearchBriefService
Generates a neutral Research Brief using Perplexity and provides compact, privacy-conscious logging.
Respects single-orchestrator, independent consultants, and Manus context-engineering.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

# Migrated to use adapter for dependency inversion
from src.engine.adapters.context_stream import UnifiedContextStream, ContextEventType
from src.core.events.event_emitters import ResearchEventEmitter
from src.engine.core.feature_flags import FeatureFlagService, FeatureFlag
from src.engine.integrations.perplexity_client import (
    PerplexityClient,
    KnowledgeQueryType,
    ResearchTier,
)
from src.integrations.llm.unified_client import UnifiedLLMClient
from src.contracts.research_brief import ResearchBrief, ResearchSource


@dataclass
class ResearchBriefConfig:
    tier: str = ResearchTier.REGULAR
    max_tokens: int = 1000
    enable_summary_llm: bool = True
    summary_word_target: int = 160


class ResearchBriefService:
    def __init__(
        self,
        context_stream: Optional[UnifiedContextStream] = None,
        flags: Optional[FeatureFlagService] = None,
        config: Optional[ResearchBriefConfig] = None,
    ):
        self.logger = logging.getLogger(__name__)
        self.context_stream = context_stream
        self.research_events = ResearchEventEmitter(
            context_stream, default_metadata={"component": "ResearchBriefService"}
        )
        self.flags = flags or FeatureFlagService()
        self.config = config or ResearchBriefConfig()
        self.perplexity = PerplexityClient()
        self.llm = UnifiedLLMClient()

    async def generate_brief(
        self, enhanced_query: str, business_context: Optional[Dict[str, Any]] = None
    ) -> Optional[ResearchBrief]:
        """
        Generate a neutral Research Brief using Perplexity.
        Returns None if feature flag is disabled or provider unavailable.
        """
        if not self.flags.is_enabled(FeatureFlag.ENABLE_RESEARCH_BRIEF):
            self.logger.info("ResearchBrief disabled by feature flag")
            return None

        # Emit provider request event (privacy-conscious)
        if self.context_stream:
            self.research_events.request(
                provider="perplexity",
                query=enhanced_query[:120],
                tier=str(self.config.tier),
            )

        # Compose neutral research query
        research_query = (
            "Provide neutral, source-backed factual grounding relevant to the following problem. "
            "Avoid opinions or recommendations. Focus on verifiable facts, definitions, standards, and recent stats.\n\n"
            f"Problem: {enhanced_query}\n"
        )

        # Query Perplexity
        start_time = datetime.utcnow()
        interaction = await self.perplexity.query_knowledge(
            query=research_query,
            query_type=KnowledgeQueryType.CONTEXT_GROUNDING,
            tier=self.config.tier,
            max_tokens=self.config.max_tokens,
            operation_context="research_brief",
        )
        processing_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

        # Build sources
        sources: List[ResearchSource] = []
        for cite in interaction.citations or []:
            url = cite.get("url") if isinstance(cite, dict) else (cite or "")
            if not url:
                continue
            sources.append(ResearchSource(url=url))

        # Summarize neutrally (LLM with fallback)
        summary = interaction.raw_response_received or ""
        key_facts: List[str] = []
        if self.config.enable_summary_llm and summary:
            try:
                messages = [
                    {
                        "role": "system",
                        "content": "You produce neutral factual summaries without recommendations.",
                    },
                    {
                        "role": "user",
                        "content": (
                            "From the following notes, extract a neutral 120-180 word factual summary and 5 bullet key facts.\n"
                            "Return JSON with keys: summary, facts.\n\nNotes:\n"
                            + summary[:4000]
                        ),
                    },
                ]
                resp = await self.llm.call_with_json_enforcement(
                    messages=messages, model="deepseek-chat"
                )
                import json

                parsed = json.loads(resp.content)
                summary = parsed.get("summary", summary)
                facts = parsed.get("facts", [])
                key_facts = [str(f) for f in facts][:8]
            except Exception as e:
                self.logger.warning(f"Summary LLM fallback due to error: {e}")
                # Fallback heuristic: first sentences as bullets
                key_facts = [s.strip() for s in summary.split(". ")[:5] if s.strip()]

        brief = ResearchBrief(
            query=enhanced_query,
            neutral_summary=summary,
            key_facts=key_facts,
            sources=sources,
            tier=str(self.config.tier),
            confidence=float(interaction.confidence_score),
            compiled_at=datetime.utcnow().isoformat(),
            research_id=interaction.research_id,
        )

        # Emit provider response and brief attached events (compact)
        if self.context_stream:
            self.research_events.response(
                provider="perplexity",
                status="success",
                latency_ms=processing_time_ms,
                citations_count=len(sources),
                result_preview=(summary or "")[:200],
                confidence=float(interaction.confidence_score),
                tier=str(self.config.tier),
            )
            self.context_stream.add_event(
                ContextEventType.RESEARCH_BRIEF_ATTACHED, brief.to_compact_event()
            )

        return brief
