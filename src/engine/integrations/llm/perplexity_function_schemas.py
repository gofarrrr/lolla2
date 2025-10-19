#!/usr/bin/env python3
"""
DeepSeek V3.1 Native Function Calling Schemas for Perplexity Integration
Defines function schemas that enable native function calling instead of text-based instructions
"""

from typing import Dict, List, Any

# DeepSeek V3.1 Function Schema for basic Perplexity research
PERPLEXITY_QUERY_KNOWLEDGE_SCHEMA = {
    "name": "perplexity_query_knowledge",
    "description": "Query Perplexity for real-time knowledge and information on specific topics",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The research query to send to Perplexity (should be specific and well-formed)",
            },
            "query_type": {
                "type": "string",
                "enum": [
                    "context_grounding",
                    "fact_checking",
                    "market_intelligence",
                    "competitive_analysis",
                    "trend_analysis",
                ],
                "description": "Type of knowledge query for cost tracking and optimization",
            },
            "model": {
                "type": "string",
                "enum": ["sonar-pro", "sonar-deep-research"],
                "default": "sonar-pro",
                "description": "Perplexity model to use (sonar-pro for general queries, sonar-deep-research for comprehensive analysis)",
            },
            "max_tokens": {
                "type": "integer",
                "minimum": 100,
                "maximum": 4000,
                "default": 1000,
                "description": "Maximum tokens for the response",
            },
            "operation_context": {
                "type": "string",
                "description": "Additional context about why this research is being conducted",
            },
        },
        "required": ["query", "query_type"],
    },
}

# DeepSeek V3.1 Function Schema for deep research
PERPLEXITY_DEEP_RESEARCH_SCHEMA = {
    "name": "perplexity_deep_research",
    "description": "Conduct exhaustive deep research using Perplexity's enterprise-tier capabilities across hundreds of sources",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The comprehensive research query for deep analysis",
            },
            "context": {
                "type": "object",
                "description": "Business or analytical context for the research",
                "properties": {
                    "industry": {"type": "string"},
                    "company_stage": {"type": "string"},
                    "problem_type": {"type": "string"},
                    "timeframe": {"type": "string"},
                },
            },
            "focus_areas": {
                "type": "array",
                "items": {"type": "string"},
                "default": [
                    "market analysis",
                    "competitive landscape",
                    "strategic implications",
                ],
                "description": "Specific areas of focus for the deep research",
            },
        },
        "required": ["query", "context"],
    },
}

# Collection of all Perplexity function schemas for easy access
PERPLEXITY_FUNCTION_SCHEMAS = [
    PERPLEXITY_QUERY_KNOWLEDGE_SCHEMA,
    PERPLEXITY_DEEP_RESEARCH_SCHEMA,
]


def get_perplexity_function_schemas() -> List[Dict[str, Any]]:
    """Get all Perplexity function schemas for DeepSeek V3.1 native function calling"""
    return PERPLEXITY_FUNCTION_SCHEMAS.copy()


def get_function_schema_by_name(function_name: str) -> Dict[str, Any]:
    """Get a specific function schema by name"""
    for schema in PERPLEXITY_FUNCTION_SCHEMAS:
        if schema["name"] == function_name:
            return schema
    raise ValueError(f"Function schema '{function_name}' not found")
