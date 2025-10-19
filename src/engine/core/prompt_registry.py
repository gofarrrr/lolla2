#!/usr/bin/env python3
"""
Prompt Template Registry (v1)
Central place to retrieve standardized prompt templates with versioning.

Usage:
- get_template(stage="analysis", mode="grok", version="v1")
- get_template(stage="deliverable", mode="mckinsey", version="v1")
"""
from typing import Dict

_TEMPLATES: Dict[str, str] = {
    # Analysis template optimized for Grok 4 Fast
    "analysis_grok_v1": (
        "<role>You are a {consultant_type} specializing in {specialization}.</role>\n"
        "<task>Analyze: {user_query}</task>\n\n"
        "<context>{briefing_memo}\n{framework_block}\n{inquiry_complex_block}</context>\n\n"
        "<success_criteria>\n"
        "- 5+ specific insights with quantitative support\n"
        "- 3+ concrete recommendations with implementation details\n"
        "- 3+ risk factors with mitigations\n"
        "- Reference selected mental models explicitly\n"
        "- Minimum 800 words; specific, not generic\n"
        "</success_criteria>\n\n"
        "<output_format>Markdown with clear headings. Include numbers, budgets, timelines.</output_format>"
    ),
    # Deliverable template (McKinsey style)
    "deliverable_mckinsey_v1": (
        "<output>\n"
        "# Key Insights\n{key_insights}\n\n"
        "# Risks\n{risks}\n\n"
        "# Opportunities\n{opportunities}\n\n"
        "# Recommendations\n{recommendations}\n\n"
        "# Implementation Plan\n{implementation}\n\n"
        "# Metrics\n{metrics}\n"
        "</output>"
    ),
    # Planner template for tool calls (JSON schema expected)
    "research_planner_v1": (
        "You are a research planner. Output a JSON object with a single key \"tool_calls\" \n"
        "(list of tool call objects). Choose between knowledge_base_search and live_internet_research.\n"
    ),
    # Knowledge synthesizer skeleton
    "knowledge_synth_v1": (
        "Synthesize the following RAG results into a brief memo (4 bullets max). Focus on actionable frameworks,\n"
        "contrarian perspectives, and blind spots.\n"
    ),
    # Final analysis skeleton (fallback)
    "final_analysis_v1": (
        "Produce a structured analysis with insights, risks, opportunities, and recommendations.\n"
    ),
}


def get_template(stage: str, mode: str, version: str = "v1") -> str:
    """Return a template by stage/mode/version or raise KeyError."""
    key = f"{stage}_{mode}_{version}"
    if key in _TEMPLATES:
        return _TEMPLATES[key]

    # Allow direct key access as a fallback (e.g., "analysis_grok_v1")
    if stage in _TEMPLATES:
        return _TEMPLATES[stage]

    raise KeyError(f"Template not found: {stage}/{mode}/{version}")
