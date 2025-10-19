#!/usr/bin/env python3
"""
METIS Research Context Integration
Provides research context retrieval and integration for the cognitive engine
"""

import logging
from uuid import UUID
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from src.core.research_storage_manager import research_storage
from src.engine.integrations.perplexity_client_advanced import ResearchTemplateType

logger = logging.getLogger(__name__)


@dataclass
class ResearchContextItem:
    """Individual research context item for cognitive processing"""

    session_id: str
    research_query: str
    executive_summary: str
    key_insights: List[Dict[str, Any]]
    confidence_score: float
    template_type: str
    total_sources: int
    cost_usd: float
    created_at: str

    def to_context_string(self) -> str:
        """Convert to formatted context string for LLM consumption"""
        insights_text = "\n".join(
            [
                f"- {insight.get('claim', 'Unknown insight')} (confidence: {insight.get('confidence', 0):.1%})"
                for insight in self.key_insights[:5]  # Top 5 insights
            ]
        )

        return f"""
## Research Context: {self.research_query}
**Template:** {self.template_type} | **Confidence:** {self.confidence_score:.1%} | **Sources:** {self.total_sources}

**Summary:** {self.executive_summary}

**Key Insights:**
{insights_text}

**Research Quality:** Cost: ${self.cost_usd:.4f} | Date: {self.created_at[:10]}
"""


class ResearchContextManager:
    """Manages research context retrieval and integration for cognitive processing"""

    def __init__(self):
        self.logger = logger
        self.storage = research_storage

    async def get_context_for_problem(
        self,
        engagement_id: UUID,
        problem_statement: str,
        template_preference: Optional[ResearchTemplateType] = None,
        max_context_items: int = 3,
    ) -> List[ResearchContextItem]:
        """
        Retrieve relevant research context for a problem statement

        Args:
            engagement_id: The engagement to get context for
            problem_statement: The problem being analyzed
            template_preference: Preferred research template type
            max_context_items: Maximum context items to return

        Returns:
            List of relevant research context items
        """
        try:
            # Get research context from storage
            context_data = await self.storage.get_research_context_for_cognitive_engine(
                engagement_id=engagement_id,
                query=problem_statement,
                template_type=template_preference,
                limit=max_context_items,
            )

            # Convert to ResearchContextItem objects
            context_items = []
            for item in context_data:
                context_item = ResearchContextItem(
                    session_id=item.get("session_id", ""),
                    research_query=item.get("research_query", ""),
                    executive_summary=item.get("executive_summary", ""),
                    key_insights=item.get(
                        "total_insights", []
                    ),  # This might need adjustment based on view
                    confidence_score=item.get("confidence_score", 0.0),
                    template_type=item.get("template_type", ""),
                    total_sources=item.get("total_sources_found", 0),
                    cost_usd=item.get("total_cost_usd", 0.0),
                    created_at=item.get("created_at", ""),
                )
                context_items.append(context_item)

            self.logger.info(
                f"✅ Retrieved {len(context_items)} research context items for cognitive processing"
            )
            return context_items

        except Exception as e:
            self.logger.error(f"❌ Error retrieving research context: {e}")
            return []

    def format_context_for_llm(
        self, context_items: List[ResearchContextItem], max_context_length: int = 8000
    ) -> str:
        """
        Format research context for LLM consumption with length management

        Args:
            context_items: List of research context items
            max_context_length: Maximum character length for context

        Returns:
            Formatted context string ready for LLM
        """
        if not context_items:
            return "No relevant research context available."

        formatted_context = "# Research Context Available\n\n"
        formatted_context += (
            "The following research has been conducted in relation to this problem:\n\n"
        )

        current_length = len(formatted_context)

        for i, item in enumerate(context_items):
            item_context = item.to_context_string()

            # Check if adding this item would exceed the limit
            if current_length + len(item_context) > max_context_length and i > 0:
                remaining_items = len(context_items) - i
                formatted_context += f"\n*({remaining_items} additional research sessions available but truncated for brevity)*\n"
                break

            formatted_context += item_context + "\n"
            current_length += len(item_context)

        formatted_context += "\n---\n**Instructions:** Use this research context to inform your analysis. Reference specific insights when relevant and note any gaps that might require additional research.\n"

        return formatted_context

    async def suggest_additional_research(
        self,
        engagement_id: UUID,
        problem_statement: str,
        current_context: List[ResearchContextItem],
    ) -> List[Dict[str, Any]]:
        """
        Suggest additional research based on current context and gaps

        Args:
            engagement_id: Current engagement
            problem_statement: The problem being analyzed
            current_context: Current research context items

        Returns:
            List of research suggestions
        """
        suggestions = []

        try:
            # Analyze current research coverage
            covered_templates = set(item.template_type for item in current_context)
            all_templates = set(template.value for template in ResearchTemplateType)
            missing_templates = all_templates - covered_templates

            # Suggest missing research areas
            template_suggestions = {
                "market_analysis": "Market analysis to understand industry trends and dynamics",
                "competitive_intelligence": "Competitive intelligence to analyze competitor strategies",
                "technology_trends": "Technology trends analysis for innovation opportunities",
                "risk_assessment": "Risk assessment to identify potential challenges",
                "investment_evaluation": "Investment evaluation for financial implications",
            }

            for template in missing_templates:
                if template in template_suggestions:
                    suggestions.append(
                        {
                            "type": "research_template",
                            "template": template,
                            "reason": template_suggestions[template],
                            "priority": "medium",
                        }
                    )

            # Analyze information gaps from existing research
            all_gaps = []
            for item in current_context:
                # Would extract gaps from item.information_gaps if available
                pass

            # Suggest research based on confidence scores
            low_confidence_items = [
                item for item in current_context if item.confidence_score < 0.6
            ]
            if low_confidence_items:
                suggestions.append(
                    {
                        "type": "confidence_improvement",
                        "reason": f"Some research has low confidence ({len(low_confidence_items)} items < 60%)",
                        "priority": "high",
                        "specific_queries": [
                            item.research_query for item in low_confidence_items
                        ],
                    }
                )

            self.logger.info(f"✅ Generated {len(suggestions)} research suggestions")
            return suggestions

        except Exception as e:
            self.logger.error(f"❌ Error generating research suggestions: {e}")
            return []


# Global instance for easy access
research_context_manager = ResearchContextManager()


# Helper function for cognitive engine integration
async def get_research_context_for_cognitive_analysis(
    engagement_id: UUID,
    problem_statement: str,
    template_preference: Optional[ResearchTemplateType] = None,
) -> str:
    """
    Convenient function to get formatted research context for cognitive analysis

    Args:
        engagement_id: The engagement to get context for
        problem_statement: The problem being analyzed
        template_preference: Optional template preference

    Returns:
        Formatted research context string ready for LLM consumption
    """
    context_items = await research_context_manager.get_context_for_problem(
        engagement_id=engagement_id,
        problem_statement=problem_statement,
        template_preference=template_preference,
    )

    return research_context_manager.format_context_for_llm(context_items)
