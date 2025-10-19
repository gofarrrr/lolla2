#!/usr/bin/env python3
"""
Smart Perplexity Mock System
Provides realistic research responses for testing without API costs
"""

from typing import Dict, List, Any
from dataclasses import dataclass
from src.engine.integrations.perplexity_client import (
    PerplexityResponse,
    KnowledgeQueryType,
)


@dataclass
class MockResearchTemplate:
    """Template for generating contextual mock responses"""

    keywords: List[str]
    response_template: str
    confidence: float
    sources: List[Dict[str, str]]


class SmartPerplexityMock:
    """
    Smart mock that generates contextual responses based on query patterns
    Saves ~$0.50 per test while maintaining realistic behavior
    """

    def __init__(self):
        self.templates = self._load_response_templates()
        self.call_count = 0
        self.cost_saved = 0.0

    def _load_response_templates(self) -> Dict[str, MockResearchTemplate]:
        """Load contextual response templates for different query types"""
        return {
            "saas_market": MockResearchTemplate(
                keywords=["saas", "market", "growth", "europe", "expansion"],
                response_template="European SaaS market shows {growth_rate}% growth. Key challenges include GDPR compliance, localization costs averaging {localization_cost}, and customer acquisition costs typically {cac_increase}% higher than North American markets.",
                confidence=0.75,
                sources=[
                    {
                        "title": "European SaaS Market Report 2024",
                        "url": "https://example.com/saas-report",
                    },
                    {
                        "title": "GDPR Impact on SaaS Companies",
                        "url": "https://example.com/gdpr-impact",
                    },
                ],
            ),
            "customer_churn": MockResearchTemplate(
                keywords=["churn", "retention", "customer", "30%"],
                response_template="SaaS companies with {churn_rate}% churn typically face product-market fit issues. Common causes include poor onboarding (40% of cases), feature gaps (35%), and pricing misalignment (25%).",
                confidence=0.82,
                sources=[
                    {
                        "title": "SaaS Churn Analysis 2024",
                        "url": "https://example.com/churn-analysis",
                    },
                    {
                        "title": "Customer Retention Best Practices",
                        "url": "https://example.com/retention",
                    },
                ],
            ),
            "cost_analysis": MockResearchTemplate(
                keywords=["cost", "investment", "$3m", "18-month"],
                response_template="Market expansion investments of {investment_size} typically require {timeline} for ROI realization. Success rates vary by market maturity and competitive landscape.",
                confidence=0.70,
                sources=[
                    {
                        "title": "Market Expansion ROI Study",
                        "url": "https://example.com/expansion-roi",
                    }
                ],
            ),
        }

    async def query_knowledge(
        self, query: str, query_type: KnowledgeQueryType, max_tokens: int = 1000
    ) -> PerplexityResponse:
        """
        Generate smart mock response based on query context
        """
        self.call_count += 1
        self.cost_saved += 0.023  # Average cost per query

        # Analyze query to find best template
        query_lower = query.lower()
        best_template = None
        max_matches = 0

        for template_name, template in self.templates.items():
            matches = sum(1 for keyword in template.keywords if keyword in query_lower)
            if matches > max_matches:
                max_matches = matches
                best_template = template

        # Generate contextual response
        if best_template and max_matches > 0:
            response_content = self._generate_contextual_response(query, best_template)
            confidence = best_template.confidence
            sources = best_template.sources
        else:
            # Fallback generic response
            response_content = f"Research context related to: {query[:100]}. Market analysis suggests strategic consideration of multiple factors including timing, resources, and competitive positioning."
            confidence = 0.65
            sources = [
                {
                    "title": "Generic Business Analysis",
                    "url": "https://example.com/generic",
                }
            ]

        return PerplexityResponse(
            content=response_content,
            sources=sources,
            confidence=confidence,
            query_type=query_type,
            tokens_used=len(response_content.split()) * 1.3,  # Realistic token count
            cost_usd=0.0,  # Mock costs nothing
            processing_time_ms=150.0,  # Realistic processing time
            citations=sources,
        )

    def _generate_contextual_response(
        self, query: str, template: MockResearchTemplate
    ) -> str:
        """Generate contextual response using template"""
        response = template.response_template

        # Extract values from query for dynamic substitution
        substitutions = {
            "growth_rate": "15-20",
            "localization_cost": "$500K-$1.2M",
            "cac_increase": "40-60",
            "churn_rate": "30",
            "investment_size": "$3M",
            "timeline": "18-24 months",
        }

        # Simple pattern matching for dynamic values
        if "30%" in query:
            substitutions["churn_rate"] = "30"
        if "$3m" in query.lower() or "$3 m" in query.lower():
            substitutions["investment_size"] = "$3M"
        if "18 month" in query.lower():
            substitutions["timeline"] = "18-24 months"

        # Apply substitutions
        for key, value in substitutions.items():
            response = response.replace(f"{{{key}}}", value)

        return response

    def get_mock_stats(self) -> Dict[str, Any]:
        """Get statistics about mock usage"""
        return {
            "total_calls": self.call_count,
            "cost_saved_usd": round(self.cost_saved, 2),
            "avg_cost_per_call": 0.0,
            "real_cost_equivalent": round(self.call_count * 0.023, 2),
        }


# Global mock instance for testing
_smart_mock = None


def get_smart_perplexity_mock() -> SmartPerplexityMock:
    """Get or create smart mock instance"""
    global _smart_mock
    if _smart_mock is None:
        _smart_mock = SmartPerplexityMock()
    return _smart_mock


async def get_test_perplexity_client():
    """Get mock perplexity client for testing"""
    return get_smart_perplexity_mock()
