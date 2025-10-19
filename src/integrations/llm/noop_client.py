#!/usr/bin/env python3
"""
Noop/Fake LLM client for fast, deterministic test runs.

When TEST_FAST=1, this client replaces real providers and returns
small, valid JSON payloads suitable for downstream aggregation.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional


class NoopLLMClient:
    """Minimal async interface compatible with ParallelRunner expectations."""

    async def complete(
        self,
        *,
        model: str = "grok-4-fast",
        system: str = "",
        user: str = "",
        temperature: float = 0.3,
        max_tokens: int = 1000,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Return a deterministic JSON content string with required fields.
        """
        content_obj = {
            "key_insights": [
                "Lead Statement: The core growth lever is focused execution on the top customer segments. WHAT: Segment concentration shows 80/20 revenue skew. SO WHAT: Prioritizing top segments increases conversion efficiency and CAC payback. NOW WHAT: Focus GTM and product fit on the top two segments with clear milestones [source: Example Market Report, 2024]",
                "Lead Statement: Operational throughput is constrained by handoffs. WHAT: Current handoff latency is 3-5 days. SO WHAT: This delays learning cycles and slows revenue capture. NOW WHAT: Introduce parallelized work cells and automation for the top 3 workflows [source: Internal Ops Audit, 2023]",
            ],
            "risk_factors": [
                "Execution risk from change fatigue; probability medium; impact high; mitigate via phased rollout and leading indicators [source: HBR, 2022]",
            ],
            "opportunities": [
                "Expand into mid-market with bundled pricing and success metrics aligned to ROI; first-mover advantage in Region X [source: Gartner, 2024]",
            ],
            "recommendations": [
                "Launch a 90-day focused execution program with 3 streams: GTM focus, ops throughput, ROI packaging; define owners and weekly metrics [source: McKinsey, 2024]",
            ],
            "confidence_level": "high",
            "analysis_quality": "good",
        }
        return {
            "content": json.dumps(content_obj, ensure_ascii=True),
            "tokens_used": 256,
            "provider": "noop",
            "model": model,
        }

    async def call_llm(
        self,
        *,
        messages: List[Dict[str, Any]],
        model: str = "grok-4-fast",
        temperature: float = 0.3,
        max_tokens: int = 1000,
        response_format: Optional[Dict[str, str]] = None,
        engagement_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Same deterministic payload as complete(), routed through messages interface.
        """
        return await self.complete(model=model, temperature=temperature, max_tokens=max_tokens)