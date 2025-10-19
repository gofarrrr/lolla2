"""
Research Grounding Engine - OPERATION ILLUMINATE Enhanced
Phase 2: Research Interaction Capture for Glass-Box Transparency

Enhanced with complete ResearchInteraction capture from Perplexity API
"""

import os
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field

# Use mock logger if structured logging unavailable
try:
    from src.engine.adapters.logging import get_logger  # Migrated

    logger = get_logger(__name__, component="research_grounding_illuminate")
except:
    import logging

    logger = logging.getLogger(__name__)

# Import OPERATION ILLUMINATE enhanced Perplexity client
try:
    from src.engine.integrations.perplexity_client_illuminate import (
        get_perplexity_client_illuminate,
        ResearchInteraction,
        KnowledgeQueryType,
    )

    PERPLEXITY_AVAILABLE = True
except ImportError:
    PERPLEXITY_AVAILABLE = False
    logger.warning(
        "Operation Illuminate Perplexity client not available, will use fallback mode"
    )


@dataclass
class FactAssertion:
    """Single fact assertion with evidence"""

    claim: str
    confidence: float  # 0.0 to 1.0
    source: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    evidence: List[str] = field(default_factory=list)
    supporting_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FactPackIlluminate:
    """Collection of fact assertions from research with complete interaction data"""

    assertions: List[FactAssertion] = field(default_factory=list)
    query: str = ""
    research_timestamp: datetime = field(default_factory=datetime.utcnow)
    sources_consulted: List[str] = field(default_factory=list)
    grounding_score: float = 0.0
    research_duration_ms: float = 0.0

    # OPERATION ILLUMINATE: Complete research interaction data
    research_interactions: List[Dict[str, Any]] = field(default_factory=list)
    total_sources_found: int = 0
    contradiction_detections: List[Dict[str, Any]] = field(default_factory=list)
    total_cost_usd: float = 0.0

    def __len__(self):
        return len(self.assertions)

    def __bool__(self):
        return bool(self.assertions)


class ResearchGroundingEngineIlluminate:
    """
    OPERATION ILLUMINATE Enhanced Research Grounding Engine
    Captures complete research interactions for glass-box transparency
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Research Grounding Engine with OPERATION ILLUMINATE capabilities

        Args:
            api_key: Optional API key (defaults to PERPLEXITY_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        self.timeout = 45  # 45 second timeout as per specification
        self.max_retries = 2  # Exponential backoff retries
        self.perplexity_client = None

        # Initialize OPERATION ILLUMINATE enhanced Perplexity client
        if PERPLEXITY_AVAILABLE and self.api_key:
            try:
                self.perplexity_client = get_perplexity_client_illuminate()
                self.is_production = True
                logger.info(
                    "research_engine_illuminate_initialized",
                    mode="OPERATION_ILLUMINATE_PERPLEXITY",
                    timeout=self.timeout,
                    api_key_present=bool(self.api_key),
                )
            except Exception as e:
                logger.error(
                    f"Failed to initialize Operation Illuminate Perplexity client: {e}"
                )
                self.perplexity_client = None
                self.is_production = False
        else:
            self.is_production = False
            logger.warning(
                "research_engine_fallback_mode",
                perplexity_available=PERPLEXITY_AVAILABLE,
                api_key_present=bool(self.api_key),
            )

    async def ground_analysis(
        self,
        initial_analysis: Dict[str, Any],
        query: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        operation_context: str = "",
    ) -> Optional[FactPackIlluminate]:
        """
        Ground analysis with external research

        OPERATION ILLUMINATE: Enhanced with complete research interaction capture

        Args:
            initial_analysis: Initial analysis to ground with facts
            query: Optional specific query for research
            context: Optional additional context for research
            operation_context: Context for glass-box transparency

        Returns:
            FactPackIlluminate with research findings and captured interactions
        """
        start_time = datetime.utcnow()

        logger.info(
            f"🔍 ILLUMINATE: Starting research grounding for operation: {operation_context}"
        )

        # Extract key topics for research
        topics = self._extract_research_topics(initial_analysis)

        if not topics:
            # If no topics found, try to extract from query
            if query:
                topics = [query]
            else:
                topics = ["general business strategy"]

        # Research each topic and capture interactions
        assertions = []
        sources_consulted = set()
        research_interactions = []  # OPERATION ILLUMINATE: Capture all interactions
        contradiction_detections = []
        total_cost = 0.0

        for topic in topics[:3]:  # Limit to 3 topics for performance with deep capture
            logger.info(f"🔍 ILLUMINATE: Researching topic: {topic}")

            topic_result = await self._research_topic_with_capture(
                topic, operation_context
            )

            # Extract assertions
            topic_assertions = topic_result.get("assertions", [])
            assertions.extend(topic_assertions)

            # Track sources
            for assertion in topic_assertions:
                sources_consulted.add(assertion.source)

            # OPERATION ILLUMINATE: Store research interaction
            if "research_interaction" in topic_result:
                interaction = topic_result["research_interaction"]
                research_interactions.append(interaction)

                # Track contradictions
                if interaction.get("contradiction_detection", {}).get(
                    "detected", False
                ):
                    contradiction_detections.append(
                        {
                            "topic": topic,
                            "research_id": interaction.get("research_id", ""),
                            "contradictions": interaction.get(
                                "contradiction_detection", {}
                            ),
                        }
                    )

                # Track costs
                total_cost += interaction.get("cost_usd", 0.0)

        # Calculate grounding score
        grounding_score = self._calculate_grounding_score(assertions)

        # Calculate duration
        duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

        # OPERATION ILLUMINATE: Create enhanced FactPack with complete interaction data
        fact_pack = FactPackIlluminate(
            assertions=assertions,
            query=query or self._summarize_topics(topics),
            sources_consulted=list(sources_consulted),
            grounding_score=grounding_score,
            research_duration_ms=duration_ms,
            research_interactions=research_interactions,
            total_sources_found=len(sources_consulted),
            contradiction_detections=contradiction_detections,
            total_cost_usd=total_cost,
        )

        logger.info(
            f"🔍 ILLUMINATE: Research grounding complete - "
            f"{len(assertions)} assertions, {len(research_interactions)} interactions, "
            f"{len(sources_consulted)} sources, ${total_cost:.4f}"
        )

        return fact_pack

    async def _research_topic_with_capture(
        self, topic: str, operation_context: str = ""
    ) -> Dict[str, Any]:
        """Research a topic with complete interaction capture"""

        if self.perplexity_client and self.is_production:
            return await self._research_topic_perplexity_illuminate(
                topic, operation_context
            )
        else:
            return await self._research_topic_fallback(topic)

    async def _research_topic_perplexity_illuminate(
        self, topic: str, operation_context: str = ""
    ) -> Dict[str, Any]:
        """Research using OPERATION ILLUMINATE enhanced Perplexity API with complete interaction capture"""

        retry_count = 0
        base_delay = 1.0  # Start with 1 second

        while retry_count <= self.max_retries:
            try:
                # OPERATION ILLUMINATE: Query enhanced Perplexity client
                research_interaction = await asyncio.wait_for(
                    self.perplexity_client.query_knowledge(
                        query=f"Provide current, factual information about: {topic}. Include specific statistics, data points, and authoritative sources.",
                        query_type=KnowledgeQueryType.CONTEXT_GROUNDING,
                        max_tokens=500,
                        operation_context=operation_context,
                    ),
                    timeout=self.timeout,
                )

                # Convert ResearchInteraction to FactAssertions
                assertions = []

                # Parse the response content for facts
                if research_interaction and research_interaction.raw_response_received:
                    # Split response into individual facts
                    facts = research_interaction.raw_response_received.split("\n")

                    for fact in facts[:5]:  # Limit to 5 facts per topic
                        if (
                            fact.strip() and len(fact.strip()) > 20
                        ):  # Filter out short non-facts
                            assertion = FactAssertion(
                                claim=fact.strip(),
                                confidence=research_interaction.confidence_score,
                                source=(
                                    research_interaction.sources_extracted[0].get(
                                        "url", "Perplexity Research"
                                    )
                                    if research_interaction.sources_extracted
                                    else "Perplexity Research"
                                ),
                                evidence=[
                                    "Live research data",
                                    "External API verification",
                                ],
                                supporting_data={
                                    "research_area": topic,
                                    "api_source": "perplexity_illuminate",
                                    "tokens_used": research_interaction.tokens_used,
                                    "query_timestamp": research_interaction.timestamp.isoformat(),
                                    "research_id": research_interaction.research_id,
                                    "contradictions_detected": research_interaction.contradiction_detection_result.get(
                                        "detected", False
                                    ),
                                    "search_mode": research_interaction.search_mode,
                                    "cost_usd": research_interaction.cost_usd,
                                },
                            )
                            assertions.append(assertion)

                logger.info(
                    f"🔍 ILLUMINATE: Research successful - "
                    f"ID: {research_interaction.research_id[:8]}, "
                    f"Facts: {len(assertions)}, "
                    f"Sources: {research_interaction.sources_consulted_count}, "
                    f"Duration: {research_interaction.time_taken_ms}ms"
                )

                # OPERATION ILLUMINATE: Return both assertions and complete raw interaction
                return {
                    "assertions": assertions,
                    "research_interaction": {
                        "research_id": research_interaction.research_id,
                        "query_sent": research_interaction.query_sent,
                        "raw_response": research_interaction.raw_response_received,
                        "sources_extracted": research_interaction.sources_extracted,
                        "confidence_score": research_interaction.confidence_score,
                        "sources_consulted_count": research_interaction.sources_consulted_count,
                        "contradiction_detection": research_interaction.contradiction_detection_result,
                        "time_taken_ms": research_interaction.time_taken_ms,
                        "cost_usd": research_interaction.cost_usd,
                        "tokens_used": research_interaction.tokens_used,
                        "operation_context": research_interaction.operation_context,
                        "success": research_interaction.success,
                        "search_mode": research_interaction.search_mode,
                        "timestamp": research_interaction.timestamp.isoformat(),
                    },
                }

            except asyncio.TimeoutError:
                logger.error(
                    f"🔍 ILLUMINATE: Research timeout for topic: {topic} (retry {retry_count}/{self.max_retries})"
                )
                if retry_count < self.max_retries:
                    await asyncio.sleep(
                        base_delay * (2**retry_count)
                    )  # Exponential backoff
                    retry_count += 1
                else:
                    return self._create_error_result(topic, "Timeout after retries")

            except Exception as e:
                logger.error(
                    f"🔍 ILLUMINATE: Research error for topic {topic}: {e} (retry {retry_count}/{self.max_retries})"
                )
                if retry_count < self.max_retries:
                    await asyncio.sleep(base_delay * (2**retry_count))
                    retry_count += 1
                else:
                    return self._create_error_result(topic, str(e))

        # Should not reach here
        return self._create_error_result(topic, "Maximum retries exceeded")

    async def _research_topic_fallback(self, topic: str) -> Dict[str, Any]:
        """Fallback research with simulated interaction data"""

        logger.warning(f"🔍 ILLUMINATE: Using fallback research for topic: {topic}")

        # Simulate API delay
        await asyncio.sleep(2.0)

        # Create fallback assertion
        assertion = FactAssertion(
            claim=f"General business insight about {topic}: Industry best practices apply",
            confidence=0.6,
            source="Fallback Knowledge Base",
            evidence=["Fallback mode active"],
            supporting_data={
                "research_area": topic,
                "api_source": "fallback",
                "mode": "offline",
            },
        )

        # Create simulated interaction
        fallback_interaction = {
            "research_id": f"fallback_{topic[:8]}",
            "query_sent": f"Fallback research for: {topic}",
            "raw_response": f"Fallback information about {topic}",
            "sources_extracted": [],
            "confidence_score": 0.6,
            "sources_consulted_count": 0,
            "contradiction_detection": {"detected": False, "fallback": True},
            "time_taken_ms": 2000,
            "cost_usd": 0.0,
            "tokens_used": 0,
            "operation_context": "fallback_mode",
            "success": True,
            "search_mode": "fallback",
            "timestamp": datetime.utcnow().isoformat(),
        }

        return {"assertions": [assertion], "research_interaction": fallback_interaction}

    def _create_error_result(self, topic: str, error: str) -> Dict[str, Any]:
        """Create error result with failed interaction data"""

        error_interaction = {
            "research_id": f"error_{topic[:8]}",
            "query_sent": f"Failed research for: {topic}",
            "raw_response": "",
            "sources_extracted": [],
            "confidence_score": 0.0,
            "sources_consulted_count": 0,
            "contradiction_detection": {"detected": False, "error": error},
            "time_taken_ms": 0,
            "cost_usd": 0.0,
            "tokens_used": 0,
            "operation_context": "error_state",
            "success": False,
            "search_mode": "failed",
            "timestamp": datetime.utcnow().isoformat(),
            "error": error,
        }

        return {"assertions": [], "research_interaction": error_interaction}

    def _extract_research_topics(self, analysis: Dict[str, Any]) -> List[str]:
        """Extract key topics from analysis for research"""

        topics = []

        # Extract from different analysis fields
        if isinstance(analysis, dict):
            # Look for key insights
            if "key_insights" in analysis:
                insights = analysis["key_insights"]
                if isinstance(insights, list):
                    topics.extend([insight[:50] for insight in insights[:3]])

            # Look for problem components
            if "problem_breakdown" in analysis:
                breakdown = analysis["problem_breakdown"]
                if isinstance(breakdown, dict) and "main_components" in breakdown:
                    components = breakdown["main_components"]
                    if isinstance(components, list):
                        topics.extend(components[:3])

            # Look for hypotheses
            if "hypotheses" in analysis:
                hypotheses = analysis["hypotheses"]
                if isinstance(hypotheses, list):
                    for hyp in hypotheses[:2]:
                        if isinstance(hyp, dict) and "hypothesis" in hyp:
                            topics.append(hyp["hypothesis"][:50])

        # Clean and deduplicate topics
        cleaned_topics = []
        for topic in topics:
            if isinstance(topic, str) and len(topic.strip()) > 10:
                cleaned_topics.append(topic.strip())

        return list(set(cleaned_topics))[:5]  # Max 5 unique topics

    def _summarize_topics(self, topics: List[str]) -> str:
        """Summarize research topics into a single query string"""
        if not topics:
            return "General business analysis"

        return f"Research on: {', '.join(topics[:3])}"

    def _calculate_grounding_score(self, assertions: List[FactAssertion]) -> float:
        """Calculate overall grounding score based on assertions"""
        if not assertions:
            return 0.0

        # Average confidence weighted by evidence quality
        total_score = 0.0
        for assertion in assertions:
            confidence = assertion.confidence
            evidence_bonus = len(assertion.evidence) * 0.1
            source_bonus = 0.1 if assertion.source != "Fallback Knowledge Base" else 0.0

            assertion_score = min(1.0, confidence + evidence_bonus + source_bonus)
            total_score += assertion_score

        return total_score / len(assertions)


# Factory function
def get_research_grounding_engine_illuminate(
    api_key: Optional[str] = None,
) -> ResearchGroundingEngineIlluminate:
    """Get Operation Illuminate enhanced research grounding engine"""
    return ResearchGroundingEngineIlluminate(api_key)
