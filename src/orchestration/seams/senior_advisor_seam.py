"""
Senior Advisor Seam
-------------------

Thin seam wrapper around the SeniorAdvisorOrchestrator's core synthesis steps.
Purpose: enable safe, incremental extraction/refactor without changing behavior.

Each method delegates to the host's legacy implementation and returns the result.
If delegation fails for any reason, errors are propagated to preserve current
error semantics (callers already handle SeniorAdvisorError at higher levels).
"""

from __future__ import annotations

from typing import Any, List


class SeniorAdvisorSeam:
    async def execute_deepseek_brain(
        self,
        host: Any,
        analyses: List[Any],
        critiques: List[Any],
    ) -> Any:
        return await host._execute_deepseek_brain(analyses, critiques)  # noqa: SLF001

    async def execute_claude_brain(
        self,
        host: Any,
        analyses: List[Any],
        critiques: List[Any],
        deepseek_insight: Any,
    ) -> Any:
        return await host._execute_claude_brain(  # noqa: SLF001
            analyses, critiques, deepseek_insight
        )

    async def create_final_report(
        self,
        host: Any,
        analyses: List[Any],
        critiques: List[Any],
        deepseek_insight: Any,
        claude_insight: Any,
        start_time: float,
        total_cost: float,
    ) -> Any:
        return await host._create_final_report(  # noqa: SLF001
            analyses,
            critiques,
            deepseek_insight,
            claude_insight,
            start_time,
            total_cost,
        )