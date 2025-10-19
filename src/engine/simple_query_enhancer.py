#!/usr/bin/env python3
"""
Simple Query Enhancer for Blueprint Integration
Provides basic query enhancement functionality for Phase 1 of blueprint workflow.
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class EngagementBrief:
    """Simple engagement brief"""

    objective: str = ""
    engagement_type: str = "strategic_analysis"
    key_stakeholders: List[str] = field(default_factory=list)
    success_metrics: List[str] = field(default_factory=list)


@dataclass
class SimpleQueryEnhancementResult:
    """Simple query enhancement result for blueprint integration"""

    success: bool = True
    enhanced_query: Optional[str] = None
    engagement_brief: Optional[EngagementBrief] = None
    processing_time: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "enhanced_query": self.enhanced_query,
            "engagement_brief": self.engagement_brief,
            "processing_time": self.processing_time,
            "error": self.error,
        }


class SimpleQueryEnhancer:
    """
    Simple query enhancer for blueprint Phase 1.
    Provides basic query context analysis without external dependencies.
    """

    def __init__(self):
        self.initialized = True
        logger.info("✅ Simple Query Enhancer initialized for blueprint integration")

    async def enhance_query(
        self, query: str, context: Optional[str] = None
    ) -> SimpleQueryEnhancementResult:
        """Simple query enhancement that extracts basic business context"""

        start_time = time.time()

        try:
            # Simple keyword-based analysis
            enhanced_context = self._analyze_query_context(query, context)

            # Create basic engagement brief
            brief = EngagementBrief(
                objective=f"Address query: {query[:100]}...",
                engagement_type=self._classify_engagement_type(query),
                key_stakeholders=self._extract_stakeholders(query, context),
                success_metrics=self._suggest_metrics(query),
            )

            processing_time = time.time() - start_time

            return SimpleQueryEnhancementResult(
                success=True,
                enhanced_query=enhanced_context,
                engagement_brief=brief,
                processing_time=processing_time,
            )

        except Exception as e:
            logger.error(f"Simple query enhancement failed: {str(e)}")
            return SimpleQueryEnhancementResult(
                success=False, error=str(e), processing_time=time.time() - start_time
            )

    def _analyze_query_context(self, query: str, context: Optional[str]) -> str:
        """Analyze and enhance query context"""
        parts = [f"QUERY: {query}"]

        if context:
            parts.append(f"CONTEXT: {context}")

        # Add basic business context based on keywords
        business_type = self._identify_business_type(query)
        if business_type:
            parts.append(f"BUSINESS_TYPE: {business_type}")

        urgency = self._assess_urgency(query)
        if urgency:
            parts.append(f"URGENCY: {urgency}")

        return "\n\n".join(parts)

    def _classify_engagement_type(self, query: str) -> str:
        """Classify the type of engagement based on query keywords"""
        query_lower = query.lower()

        if any(
            word in query_lower
            for word in ["strategy", "strategic", "direction", "planning"]
        ):
            return "strategic_analysis"
        elif any(
            word in query_lower
            for word in ["implement", "execution", "roadmap", "timeline"]
        ):
            return "implementation_planning"
        elif any(
            word in query_lower for word in ["problem", "issue", "challenge", "crisis"]
        ):
            return "problem_solving"
        elif any(
            word in query_lower
            for word in ["optimize", "improve", "efficiency", "performance"]
        ):
            return "optimization"
        else:
            return "general_consulting"

    def _identify_business_type(self, query: str) -> Optional[str]:
        """Identify business type from query"""
        query_lower = query.lower()

        if any(
            word in query_lower for word in ["retail", "store", "sales", "customer"]
        ):
            return "retail"
        elif any(
            word in query_lower for word in ["tech", "software", "saas", "digital"]
        ):
            return "technology"
        elif any(
            word in query_lower
            for word in ["manufacturing", "production", "supply chain"]
        ):
            return "manufacturing"
        elif any(word in query_lower for word in ["finance", "banking", "investment"]):
            return "financial_services"
        else:
            return None

    def _assess_urgency(self, query: str) -> Optional[str]:
        """Assess urgency level from query"""
        query_lower = query.lower()

        if any(
            word in query_lower for word in ["urgent", "immediate", "asap", "crisis"]
        ):
            return "high"
        elif any(word in query_lower for word in ["soon", "quickly", "fast"]):
            return "medium"
        else:
            return "normal"

    def _extract_stakeholders(self, query: str, context: Optional[str]) -> List[str]:
        """Extract likely stakeholders from query and context"""
        stakeholders = []
        text = f"{query} {context or ''}".lower()

        if any(word in text for word in ["ceo", "executive", "leadership"]):
            stakeholders.append("Executive Leadership")
        if any(word in text for word in ["board", "directors"]):
            stakeholders.append("Board of Directors")
        if any(word in text for word in ["customer", "client"]):
            stakeholders.append("Customers")
        if any(word in text for word in ["employee", "staff", "team"]):
            stakeholders.append("Employees")
        if any(word in text for word in ["investor", "shareholder"]):
            stakeholders.append("Investors")

        return stakeholders or ["Management Team"]

    def _suggest_metrics(self, query: str) -> List[str]:
        """Suggest success metrics based on query"""
        metrics = []
        query_lower = query.lower()

        if any(word in query_lower for word in ["revenue", "sales", "profit"]):
            metrics.append("Revenue Growth")
        if any(word in query_lower for word in ["cost", "expense", "efficiency"]):
            metrics.append("Cost Reduction")
        if any(word in query_lower for word in ["customer", "satisfaction"]):
            metrics.append("Customer Satisfaction")
        if any(word in query_lower for word in ["market", "share", "competitive"]):
            metrics.append("Market Share")
        if any(word in query_lower for word in ["time", "speed", "fast"]):
            metrics.append("Time to Market")

        return metrics or ["Business Impact Assessment"]


# Alias for backward compatibility
SequentialQueryEnhancementResult = SimpleQueryEnhancementResult
SequentialQueryEnhancer = SimpleQueryEnhancer


# Test function
async def test_simple_enhancer():
    """Test the simple query enhancer"""
    enhancer = SimpleQueryEnhancer()

    test_query = "Our retail chain is losing revenue to e-commerce competitors"
    test_context = "Traditional retail with 200 stores"

    result = await enhancer.enhance_query(test_query, test_context)

    print(f"Enhancement Success: {result.success}")
    print(f"Processing Time: {result.processing_time:.3f}s")
    print(f"Enhanced Query: {result.enhanced_query}")
    print(f"Engagement Type: {result.engagement_brief.engagement_type}")
    print(f"Stakeholders: {result.engagement_brief.key_stakeholders}")


if __name__ == "__main__":
    asyncio.run(test_simple_enhancer())
