"""
Perplexity API Integration Client - OPERATION ILLUMINATE Enhanced
Real-time knowledge grounding and fact-checking for METIS cognitive analysis

PHASE 2: Research Interaction Capture for Glass-Box Transparency
"""

import os
import asyncio
import logging
import uuid
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# Load environment variables with enhanced error handling
try:
    from dotenv import load_dotenv

    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)
except ImportError:
    pass  # dotenv not available

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None


class KnowledgeQueryType(str, Enum):
    """Types of knowledge queries for cost tracking"""

    CONTEXT_GROUNDING = "context_grounding"
    FACT_CHECKING = "fact_checking"
    MARKET_INTELLIGENCE = "market_intelligence"
    COMPETITIVE_ANALYSIS = "competitive_analysis"
    TREND_ANALYSIS = "trend_analysis"


@dataclass
class ResearchInteraction:
    """Complete research interaction capture for Operation Illuminate - maximum granularity"""

    research_id: str
    timestamp: datetime
    query_sent: str
    raw_response_received: str
    sources_extracted: List[Dict[str, Any]]
    confidence_score: float
    search_mode: str  # fast/moderate/deep
    sources_consulted_count: int
    contradiction_detection_result: Dict[str, Any]
    time_taken_ms: int
    cost_usd: float = 0.0
    query_type: str = ""
    model_used: str = ""
    tokens_used: int = 0
    citations: List[Dict[str, str]] = field(default_factory=list)
    operation_context: str = ""
    success: bool = True
    error: Optional[str] = None


@dataclass
class KnowledgeUsageMetrics:
    """Track Perplexity usage and costs"""

    total_queries: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    queries_by_type: Dict[KnowledgeQueryType, int] = None
    avg_processing_time_ms: float = 0.0

    def __post_init__(self):
        if self.queries_by_type is None:
            self.queries_by_type = {query_type: 0 for query_type in KnowledgeQueryType}


class PerplexityClientIlluminate:
    """
    OPERATION ILLUMINATE Enhanced Perplexity API client
    Captures complete research interactions for glass-box transparency
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.usage_metrics = KnowledgeUsageMetrics()

        # Perplexity API configuration (2025 updated)
        self.base_url = "https://api.perplexity.ai"
        self.max_requests_per_minute = 20
        self.request_timestamps: List[datetime] = []

        # Cost tracking (2025 Sonar rates: $5 per 1,000 searches)
        self.cost_per_search = 0.005  # $5 per 1,000 searches
        self.cost_per_token = (
            0.000004  # $3 per 750K input tokens, $15 per 750K output tokens (avg)
        )

        # OPERATION ILLUMINATE: Comprehensive capture settings
        self.ENABLE_COMPREHENSIVE_CAPTURE = (
            os.getenv("ENABLE_COMPREHENSIVE_CAPTURE", "true").lower() == "true"
        )
        self.MAX_CAPTURE_SIZE_MB = float(os.getenv("MAX_CAPTURE_SIZE_MB", "10"))

        # Initialize client
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Perplexity HTTP client with API key validation"""

        if not HTTPX_AVAILABLE:
            self.logger.error("httpx library not installed. Run: pip install httpx")
            self.client = None
            return

        api_key = os.getenv("PERPLEXITY_API_KEY")
        if not api_key:
            self.logger.error("PERPLEXITY_API_KEY environment variable not set")
            self.client = None
            return

        try:
            self.client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "METIS-Cognitive-Platform/1.0",
                },
                timeout=httpx.Timeout(30.0, read=120.0, write=30.0, connect=10.0),
            )
            self.logger.info("✅ Perplexity client initialized successfully")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Perplexity client: {e}")
            self.client = None

    def _calculate_cost(self, tokens: int, model: str = "sonar-pro") -> float:
        """Calculate cost for token usage based on model"""

        # Model-specific pricing (per 1M tokens as of 2025)
        model_costs = {
            "sonar-pro": {
                "input": 3.0,  # $3 per 1M input tokens
                "output": 15.0,  # $15 per 1M output tokens
                "search": 5.0,  # $5 per 1K searches
            },
            "sonar-deep-research": {
                "input": 2.0,  # $2 per 1M input tokens
                "output": 8.0,  # $8 per 1M output tokens
                "search": 5.0,  # $5 per 1K searches
            },
        }

        if model not in model_costs:
            model = "sonar-pro"  # Default fallback

        # Estimate input/output split (typically 30% input, 70% output)
        input_tokens = int(tokens * 0.3)
        output_tokens = int(tokens * 0.7)

        cost_per_million = model_costs[model]

        input_cost = (input_tokens / 1_000_000) * cost_per_million["input"]
        output_cost = (output_tokens / 1_000_000) * cost_per_million["output"]
        search_cost = cost_per_million["search"] / 1000  # Per search

        return input_cost + output_cost + search_cost

    def _detect_contradictions(
        self, content: str, sources: List[str]
    ) -> Dict[str, Any]:
        """Analyze content for potential contradictions and inconsistencies"""

        contradictions = {
            "detected": False,
            "confidence": 0.0,
            "issues": [],
            "analysis": "No contradiction analysis performed",
        }

        if not content or not sources:
            return contradictions

        # Simple contradiction detection (would be enhanced with ML in production)
        contradiction_indicators = [
            "however",
            "but",
            "although",
            "despite",
            "contrary to",
            "on the other hand",
            "nevertheless",
            "nonetheless",
            "conflict",
            "disagree",
            "dispute",
            "contradict",
            "inconsistent",
        ]

        content_lower = content.lower()
        detected_indicators = [
            ind for ind in contradiction_indicators if ind in content_lower
        ]

        if detected_indicators:
            contradictions.update(
                {
                    "detected": True,
                    "confidence": min(len(detected_indicators) * 0.2, 0.8),
                    "issues": detected_indicators,
                    "analysis": f"Detected {len(detected_indicators)} potential contradiction indicators",
                }
            )

        return contradictions

    async def _check_rate_limit(self):
        """Check and enforce rate limiting"""
        now = datetime.utcnow()

        # Remove timestamps older than 1 minute
        self.request_timestamps = [
            ts for ts in self.request_timestamps if (now - ts).total_seconds() < 60
        ]

        if len(self.request_timestamps) >= self.max_requests_per_minute:
            wait_time = 60 - (now - self.request_timestamps[0]).total_seconds()
            if wait_time > 0:
                self.logger.warning(f"Rate limit reached, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)

        self.request_timestamps.append(now)

    async def query_knowledge(
        self,
        query: str,
        query_type: KnowledgeQueryType,
        model: str = "sonar-pro",
        max_tokens: int = 1000,
        operation_context: str = "",
    ) -> ResearchInteraction:
        """
        Query Perplexity for real-time knowledge and information

        OPERATION ILLUMINATE: Returns complete ResearchInteraction object for glass-box transparency.
        """

        # OPERATION ILLUMINATE: Create research interaction tracking
        research_id = str(uuid.uuid4())
        start_timestamp = datetime.utcnow()
        start_time = time.time()

        self.logger.info(
            f"🔍 ILLUMINATE: Starting research query {research_id[:8]}: {query_type.value}"
        )

        if not self.client:
            # OPERATION ILLUMINATE: Return failed interaction object
            return ResearchInteraction(
                research_id=research_id,
                timestamp=start_timestamp,
                query_sent=query,
                raw_response_received="",
                sources_extracted=[],
                confidence_score=0.0,
                search_mode="failed",
                sources_consulted_count=0,
                contradiction_detection_result={},
                time_taken_ms=0,
                cost_usd=0.0,
                query_type=query_type.value,
                model_used=model,
                tokens_used=0,
                operation_context=operation_context,
                success=False,
                error="Perplexity client not available. Check PERPLEXITY_API_KEY and installation.",
            )

        await self._check_rate_limit()

        try:
            # Determine search mode based on model and tokens
            if model == "sonar-deep-research" or max_tokens > 2000:
                search_mode = "deep"
            elif max_tokens > 1000:
                search_mode = "moderate"
            else:
                search_mode = "fast"

            # Prepare request payload for 2025 Perplexity API
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": """You are an expert research analyst with advanced information validation capabilities.

<research_methodology>
1. HYBRID SEARCH: Use both semantic similarity and keyword matching for comprehensive retrieval
2. SOURCE VALIDATION: Prioritize authoritative, recent, and credible sources
3. CROSS-REFERENCE: Verify claims across multiple independent sources
4. BIAS DETECTION: Identify potential bias in sources and perspectives
5. CONFIDENCE ASSESSMENT: Provide explicit confidence levels for each claim
</research_methodology>

<citation_requirements>
- Provide specific source URLs and publication dates
- Include author credentials where available
- Note any conflicting information found
- Flag areas where evidence is insufficient
- Distinguish between facts, expert opinions, and speculation
</citation_requirements>

Provide detailed evidence with authoritative source citations.""",
                    },
                    {"role": "user", "content": query},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.2,
                "stream": False,
            }

            # Make the API call
            response = await self.client.post("/chat/completions", json=payload)

            # Enhanced error handling for 2025 API
            if response.status_code == 401:
                raise ValueError("Invalid API key or insufficient credits")
            elif response.status_code == 404:
                raise ValueError(f"Model '{model}' not found")
            elif response.status_code == 429:
                raise ValueError("Rate limit exceeded")

            response.raise_for_status()

            # Parse response
            data = response.json()

            if "choices" not in data or not data["choices"]:
                raise ValueError("Invalid response from Perplexity API")

            choice = data["choices"][0]
            content = choice["message"]["content"]

            # Extract metadata
            usage = data.get("usage", {})
            tokens_used = usage.get("total_tokens", 0)

            # Calculate metrics
            time_taken_ms = int((time.time() - start_time) * 1000)
            cost = self._calculate_cost(tokens_used, model)

            # Extract citations and sources from response
            citations = data.get("citations", [])
            sources_extracted = []

            # Handle citations properly - they might be strings or dicts
            if isinstance(citations, list):
                for cite in citations:
                    if isinstance(cite, dict):
                        sources_extracted.append(cite)
                    elif isinstance(cite, str):
                        sources_extracted.append({"url": cite, "type": "url"})

            # Detect contradictions in response
            contradiction_detection = self._detect_contradictions(
                content, [s.get("url", "") for s in sources_extracted]
            )

            # Determine confidence based on sources and content
            base_confidence = 0.8 if sources_extracted else 0.6
            if contradiction_detection["detected"]:
                base_confidence -= contradiction_detection["confidence"] * 0.3
            confidence = max(0.1, min(1.0, base_confidence))

            # Update usage metrics
            self.usage_metrics.total_queries += 1
            self.usage_metrics.total_tokens += tokens_used
            self.usage_metrics.total_cost_usd += cost
            self.usage_metrics.queries_by_type[query_type] += 1

            # OPERATION ILLUMINATE: Create complete research interaction object
            research_interaction = ResearchInteraction(
                research_id=research_id,
                timestamp=start_timestamp,
                query_sent=query,
                raw_response_received=content,
                sources_extracted=sources_extracted,
                confidence_score=confidence,
                search_mode=search_mode,
                sources_consulted_count=len(sources_extracted),
                contradiction_detection_result=contradiction_detection,
                time_taken_ms=time_taken_ms,
                cost_usd=cost,
                query_type=query_type.value,
                model_used=model,
                tokens_used=tokens_used,
                citations=citations if isinstance(citations, list) else [],
                operation_context=operation_context,
                success=True,
            )

            self.logger.info(
                f"🔍 ILLUMINATE: Captured research {research_id[:8]} - "
                f"{tokens_used} tokens, {len(sources_extracted)} sources, ${cost:.4f}"
            )

            return research_interaction

        except Exception as e:
            time_taken_ms = int((time.time() - start_time) * 1000)

            # OPERATION ILLUMINATE: Return failed interaction object
            self.logger.error(
                f"🔍 ILLUMINATE: Research query failed {research_id[:8]}: {e}"
            )

            return ResearchInteraction(
                research_id=research_id,
                timestamp=start_timestamp,
                query_sent=query,
                raw_response_received="",
                sources_extracted=[],
                confidence_score=0.0,
                search_mode="failed",
                sources_consulted_count=0,
                contradiction_detection_result={"detected": False, "error": str(e)},
                time_taken_ms=time_taken_ms,
                cost_usd=0.0,
                query_type=query_type.value,
                model_used=model,
                tokens_used=0,
                operation_context=operation_context,
                success=False,
                error=str(e),
            )


# Factory functions for compatibility
def get_perplexity_client():
    """Get enhanced Perplexity client with Operation Illuminate capabilities"""
    return PerplexityClientIlluminate()


def get_perplexity_client_illuminate():
    """Get Operation Illuminate enhanced Perplexity client"""
    return PerplexityClientIlluminate()
