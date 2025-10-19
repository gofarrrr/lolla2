"""
Context gap detection utilities for ODR
"""

import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class ContextGapDetector:
    """Utility class for detecting context gaps"""

    def __init__(self):
        self.required_fields = {
            "strategic_analysis": [
                "industry",
                "company_size",
                "timeline",
                "stakeholders",
            ],
            "market_research": ["market_segment", "geography", "target_audience"],
            "competitive_analysis": [
                "competitors",
                "differentiation_factors",
                "market_position",
            ],
            "financial_analysis": [
                "revenue_model",
                "cost_structure",
                "profitability_metrics",
            ],
        }

    def detect_gaps(
        self, context: Dict[str, Any], analysis_type: str = "strategic_analysis"
    ) -> List[str]:
        """
        Detect missing context fields for a given analysis type

        Args:
            context: Current context dictionary
            analysis_type: Type of analysis being performed

        Returns:
            List of missing field names
        """
        required = self.required_fields.get(analysis_type, [])
        missing = []

        for field in required:
            if field not in context or not context.get(field):
                missing.append(field)

        if missing:
            logger.info(f"🔍 Context gaps detected for {analysis_type}: {missing}")

        return missing

    def suggest_research_queries(
        self, missing_fields: List[str], context: Dict[str, Any]
    ) -> List[str]:
        """
        Suggest research queries to fill missing context

        Args:
            missing_fields: List of missing field names
            context: Current context

        Returns:
            List of suggested research queries
        """
        queries = []
        industry = context.get("industry", "")

        field_query_map = {
            "company_size": f"typical company size in {industry} industry",
            "timeline": f"typical project timeline for {industry} strategic initiatives",
            "stakeholders": f"key stakeholders in {industry} decision making",
            "market_segment": f"market segments in {industry}",
            "geography": f"geographic markets for {industry}",
            "target_audience": f"target customer profiles in {industry}",
            "competitors": f"main competitors in {industry}",
            "differentiation_factors": f"competitive differentiation in {industry}",
            "market_position": f"market positioning strategies in {industry}",
            "revenue_model": f"revenue models in {industry}",
            "cost_structure": f"cost structure analysis for {industry}",
            "profitability_metrics": f"profitability metrics for {industry}",
        }

        for field in missing_fields:
            if field in field_query_map:
                queries.append(field_query_map[field])
            else:
                queries.append(f"research {field} for {industry}")

        return queries[:5]  # Limit to top 5 queries
