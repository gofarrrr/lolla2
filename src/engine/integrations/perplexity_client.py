"""
Perplexity API Integration Client
Real-time knowledge grounding and fact-checking for METIS cognitive analysis
"""

import os
import asyncio
import logging
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# Load environment variables with enhanced error handling
try:
    from dotenv import load_dotenv

    env_path = Path(__file__).parent.parent.parent / ".env"
    print(f"🔍 [Perplexity] Looking for .env file at: {env_path}")

    if env_path.exists():
        print("✅ [Perplexity] Found .env file, loading...")
        load_result = load_dotenv(env_path, override=True)
        print(f"   Load result: {load_result}")

        # Verify Perplexity API key was loaded
        perplexity_key = os.getenv("PERPLEXITY_API_KEY")
        if perplexity_key:
            print(
                f"✅ [Perplexity] PERPLEXITY_API_KEY loaded: {perplexity_key[:10]}...{perplexity_key[-4:]}"
            )
        else:
            print("❌ [Perplexity] PERPLEXITY_API_KEY not found after loading .env")
    else:
        print(f"❌ [Perplexity] .env file not found at {env_path}")

except ImportError:
    print("⚠️ [Perplexity] python-dotenv not available - environment loading may fail")
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


class ResearchTier(str, Enum):
    """Research tiers for cost optimization - Tiered Research Architecture"""

    TESTING = "testing"  # sonar-small - Cheapest for tests and development
    REGULAR = "regular"  # sonar-pro - Standard production use
    PREMIUM = "premium"  # sonar-deep-research - Deep research for complex queries


@dataclass
class PerplexityResponse:
    """Structured Perplexity response with metadata"""

    content: str
    sources: List[str]
    confidence: float
    query_type: KnowledgeQueryType
    tokens_used: int
    cost_usd: float
    processing_time_ms: float
    citations: List[Dict[str, str]]


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


from src.engine.core.contracts import IResearchProvider, ResearchResult


class PerplexityClient(IResearchProvider):
    """
    Production-ready Perplexity API client for knowledge grounding
    Handles authentication, rate limiting, error handling, and cost tracking
    Implements IResearchProvider.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.usage_metrics = KnowledgeUsageMetrics()
        self.provider_name = "perplexity"

        # Perplexity API configuration (2025 updated)
        self.base_url = "https://api.perplexity.ai"
        self.max_requests_per_minute = 20
        self.request_timestamps: List[datetime] = []

        # Cost tracking (2025 Sonar rates: $5 per 1,000 searches)
        self.cost_per_search = 0.005  # $5 per 1,000 searches
        self.cost_per_token = (
            0.000004  # $3 per 750K input tokens, $15 per 750K output tokens (avg)
        )

        # OPERATION ILLUMINATE: Enhanced comprehensive capture settings
        self.ENABLE_COMPREHENSIVE_CAPTURE = (
            os.getenv("ENABLE_COMPREHENSIVE_CAPTURE", "true").lower() == "true"
        )
        self.ENABLE_RAW_RESPONSE_CAPTURE = (
            os.getenv("ENABLE_RAW_RESPONSE_CAPTURE", "true").lower() == "true"
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

    async def is_available(self) -> bool:
        """Check if Perplexity client is available and working"""
        if not self.client:
            return False

        try:
            # Quick test query (minimal cost)
            response = await self.query_knowledge(
                query="What is SaaS?", query_type=KnowledgeQueryType.FACT_CHECKING
            )
            return True
        except Exception as e:
            self.logger.error(f"Perplexity availability check failed: {e}")
            return False

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

    def _tier_to_model(self, tier: str) -> str:
        """Map research tier to Perplexity model - Updated September 2025 API"""
        tier_mapping = {
            ResearchTier.TESTING: "sonar",  # Cheapest for tests (127k context)
            ResearchTier.REGULAR: "sonar-pro",  # Standard production (200k context)
            ResearchTier.PREMIUM: "sonar-deep-research",  # Premium deep research
        }
        return tier_mapping.get(tier, "sonar-pro")  # Default to regular

    def _calculate_cost(self, tokens: int, model: str = "sonar-pro") -> float:
        """Calculate cost for token usage based on model - September 2025 API"""

        # Current September 2025 Perplexity Model Pricing
        model_costs = {
            # CHEAPEST - Testing tier
            "sonar": {
                "cost_per_1m_tokens": 0.2,  # $0.2 per 1M tokens (cheapest option)
                "search": 5.0,  # $5 per 1K searches (same across all tiers)
            },
            # REGULAR - Production tier
            "sonar-pro": {
                "cost_per_1m_tokens": 1.0,  # $1 per 1M tokens
                "search": 5.0,  # $5 per 1K searches
            },
            # PREMIUM - Deep research tier
            "sonar-deep-research": {
                "cost_per_1m_tokens": 5.0,  # $5 per 1M tokens (most expensive)
                "search": 5.0,  # $5 per 1K searches
            },
            # Reasoning models
            "sonar-reasoning": {
                "cost_per_1m_tokens": 1.0,  # Similar to sonar-pro
                "search": 5.0,
            },
            "sonar-reasoning-pro": {
                "cost_per_1m_tokens": 3.0,  # Premium reasoning
                "search": 5.0,
            },
        }

        if model not in model_costs:
            model = "sonar-pro"  # Default fallback to regular tier

        costs = model_costs[model]

        # Calculate token costs using simplified 2025 pricing structure
        token_cost = (tokens / 1_000_000) * costs["cost_per_1m_tokens"]

        # Add search cost (estimated 1 search per query)
        search_cost = (1 / 1000) * costs["search"]

        return token_cost + search_cost

    def _get_tier_optimized_prompt(self, tier: str) -> str:
        """Get tier-optimized system prompt - balances quality with time/cost constraints"""

        if tier == ResearchTier.TESTING:
            # TESTING TIER - Fast, lightweight, minimal depth
            return """You are a research assistant optimized for quick fact verification.

<approach>
- FAST SEARCH: Prioritize speed and efficiency over exhaustive research
- KEY FACTS ONLY: Focus on essential, verifiable information
- CONCISE OUTPUT: Provide direct answers without extensive analysis
- BASIC VALIDATION: Simple source checking and fact verification
</approach>

<output_requirements>
- Maximum 3-4 key points
- Include 1-2 credible sources
- Keep response under 300 words
- Focus on factual accuracy over comprehensive analysis
</output_requirements>

Provide quick, reliable answers for development and testing purposes."""

        elif tier == ResearchTier.PREMIUM:
            # PREMIUM TIER - Deep, comprehensive, multi-step analysis (takes time)
            return """You are a senior research analyst conducting comprehensive deep research.

<deep_research_methodology>
1. MULTI-PHASE SEARCH: Execute systematic, iterative search across multiple domains
2. LONGITUDINAL ANALYSIS: Track trends and changes over time periods
3. CROSS-DOMAIN SYNTHESIS: Connect insights across different industries/fields
4. EXPERT CONSULTATION: Prioritize peer-reviewed sources and expert opinions
5. CONTRADICTORY EVIDENCE: Actively seek and analyze conflicting viewpoints
6. PREDICTIVE INSIGHTS: Draw forward-looking conclusions from current data
</deep_research_methodology>

<comprehensive_validation>
- Multiple independent source verification (minimum 5-7 sources)
- Author credential analysis and institutional affiliation checking
- Publication date relevance and information currency assessment
- Regional and cultural perspective consideration
- Statistical significance and methodology evaluation
</comprehensive_validation>

<deep_output_format>
- Executive summary with key strategic insights
- Detailed analysis with supporting evidence
- Multiple perspective synthesis
- Risk/opportunity assessment
- Implementation considerations
- Future trend projections
- Comprehensive source attribution
</deep_output_format>

This is premium research - take time for thoroughness and provide comprehensive, strategic-level insights."""

        else:  # REGULAR TIER - Balanced approach
            return """You are an expert research analyst with advanced information validation capabilities.

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

<constitutional_principles>
- Acknowledge uncertainty and limitations explicitly
- Present multiple perspectives when they exist
- Avoid confirmation bias in source selection
- Consider cultural and contextual factors
- Prioritize factual accuracy over narrative convenience
</constitutional_principles>

<output_format>
Structure your response with clear evidence validation and transparent reasoning.
</output_format>"""

    async def query_knowledge(
        self,
        query: str,
        query_type: KnowledgeQueryType,
        tier: str = None,  # Will be set based on environment
        max_tokens: int = 1000,
        operation_context: str = "",
    ) -> ResearchInteraction:
        """
        Query Perplexity for real-time knowledge and information - Tiered Research Architecture

        Args:
            query: Research question to ask
            query_type: Type of knowledge query for tracking
            tier: Research tier (testing/regular/premium) - determines model and cost
            max_tokens: Maximum tokens in response
            operation_context: Context for logging/tracking
        """

        # Environment-based tier selection for cost optimization
        if tier is None:
            environment = os.getenv("ENVIRONMENT", "development").lower()
            if environment in ["production", "prod"]:
                tier = ResearchTier.REGULAR  # Production uses regular tier
                self.logger.info("🏭 Production environment - using REGULAR tier")
            else:
                tier = ResearchTier.TESTING  # Development uses cheapest tier
                self.logger.info(
                    "🔧 Development environment - using TESTING tier (80% cost savings)"
                )

        # Map tier to appropriate model
        model = self._tier_to_model(tier)

        # Log cost estimate for transparency
        estimated_cost = self._calculate_cost(max_tokens, model)
        self.logger.info(
            f"💰 Estimated cost: ${estimated_cost:.4f} (tier: {tier}, model: {model})"
        )

        if not self.client:
            raise RuntimeError(
                "Perplexity client not available. Check PERPLEXITY_API_KEY and installation."
            )

        await self._check_rate_limit()

        start_time = datetime.utcnow()

        try:
            # Get tier-specific system prompt for optimized research approach
            system_prompt = self._get_tier_optimized_prompt(tier)

            # Prepare request payload for 2025 Perplexity API
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.2,
                "stream": False,
            }

            # Make the API call via transport helper
            response = await self._http_post("/chat/completions", payload)

            # Enhanced error handling for 2025 API with clear fallback messaging
            if response.status_code == 400:
                error_msg = f"Bad request to Perplexity API (model: {model}). This may indicate an invalid model name or malformed request."
                self.logger.error(f"❌ Perplexity 400 Error: {error_msg}")
                raise ValueError(error_msg)
            elif response.status_code == 401:
                error_msg = "Invalid API key or insufficient credits. Check your Perplexity account."
                self.logger.error(f"❌ Perplexity 401 Error: {error_msg}")
                raise ValueError(error_msg)
            elif response.status_code == 404:
                error_msg = f"Model '{model}' not found. Valid 2025 models: sonar, sonar-pro, sonar-deep-research"
                self.logger.error(f"❌ Perplexity 404 Error: {error_msg}")
                raise ValueError(error_msg)
            elif response.status_code == 429:
                error_msg = "Rate limit exceeded. Try again later."
                self.logger.warning(f"⏱️ Perplexity 429 Error: {error_msg}")
                raise ValueError(error_msg)

            response.raise_for_status()

            # Parse response
            data = response.json()

            self.logger.debug(
                f"✅ Perplexity API response received (status: {response.status_code}, keys: {list(data.keys()) if data else 'NO DATA'})"
            )

            if "choices" not in data or not data["choices"]:
                raise ValueError("Invalid response from Perplexity API")

            choice = data["choices"][0]
            content = choice["message"]["content"]

            # Extract metadata
            usage = data.get("usage", {})
            tokens_used = usage.get("total_tokens", 0)

            # Calculate metrics
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            cost = self._calculate_cost(tokens_used, model)

            # Extract citations and sources from response
            citations = data.get("citations", [])
            sources = []

            # Handle citations properly - they might be strings or dicts
            if isinstance(citations, list):
                for cite in citations:
                    if isinstance(cite, dict) and cite.get("url"):
                        sources.append(cite.get("url", ""))
                    elif isinstance(cite, str):
                        sources.append(cite)
            else:
                citations = []

            # Update usage metrics
            self.usage_metrics.total_queries += 1
            self.usage_metrics.total_tokens += tokens_used
            self.usage_metrics.total_cost_usd += cost
            self.usage_metrics.queries_by_type[query_type] += 1

            # Update average processing time
            self.usage_metrics.avg_processing_time_ms = (
                self.usage_metrics.avg_processing_time_ms
                * (self.usage_metrics.total_queries - 1)
                + processing_time
            ) / self.usage_metrics.total_queries

            # Determine confidence based on sources and content
            confidence = 0.8 if sources else 0.6  # Higher confidence with sources

            self.logger.info(
                f"✅ Perplexity query successful: {query_type.value} | "
                f"{tokens_used} tokens | ${cost:.4f} | {processing_time:.1f}ms | {len(sources)} sources"
            )

            # Create final response
            perplexity_response = PerplexityResponse(
                content=content,
                sources=sources,
                confidence=confidence,
                query_type=query_type,
                tokens_used=tokens_used,
                cost_usd=cost,
                processing_time_ms=processing_time,
                citations=citations,
            )

            self.logger.debug(
                f"📄 PerplexityResponse summary: {len(content)} chars, {len(sources)} sources, confidence={confidence:.2f}"
            )

            return perplexity_response

        except Exception as e:
            error_msg = f"Perplexity API query failed: {e}"
            self.logger.error(f"❌ {error_msg}")

            # Log fallback recommendation for ResearchManager
            if "400" in str(e) or "404" in str(e) or "Invalid API key" in str(e):
                self.logger.warning(
                    "⚠️ Perplexity provider error - ResearchManager should fallback to Exa provider"
                )
                self.logger.info(
                    f"💡 Error type suggests configuration issue: {type(e).__name__}"
                )

            raise RuntimeError(error_msg) from e

    async def ground_context(
        self, industry: str, problem_domain: str
    ) -> PerplexityResponse:
        """Get current market context and trends for problem grounding"""

        query = f"""
        Provide current 2024 market context and trends for {industry} industry, specifically related to {problem_domain}.
        Include recent statistics, benchmarks, and key challenges companies are facing.
        Focus on actionable insights and current best practices.
        """

        return await self.query_knowledge(
            query=query,
            query_type=KnowledgeQueryType.CONTEXT_GROUNDING,
            max_tokens=1500,
        )

    async def fact_check_claim(self, claim: str, domain: str) -> PerplexityResponse:
        """Validate a specific claim against current information"""

        query = f"""
        Fact-check this claim in the {domain} domain: "{claim}"
        Provide current data, statistics, or research that confirms or contradicts this claim.
        Include confidence level and sources.
        """

        return await self.query_knowledge(
            query=query, query_type=KnowledgeQueryType.FACT_CHECKING, max_tokens=800
        )

    async def get_competitive_intelligence(
        self, industry: str, focus_area: str
    ) -> PerplexityResponse:
        """Get current competitive landscape and intelligence"""

        query = f"""
        Provide current competitive intelligence for {industry} industry, focusing on {focus_area}.
        Include recent market movements, new entrants, competitive strategies, and market share data.
        Focus on 2024 developments and trends.
        """

        return await self.query_knowledge(
            query=query,
            query_type=KnowledgeQueryType.COMPETITIVE_ANALYSIS,
            max_tokens=1200,
        )

    async def analyze_trends(
        self, topic: str, timeframe: str = "2024"
    ) -> PerplexityResponse:
        """Analyze current trends in a specific topic area"""

        query = f"""
        Analyze current trends in {topic} for {timeframe}.
        Include emerging patterns, growth rates, market dynamics, and future projections.
        Provide specific data points and statistics where available.
        """

        return await self.query_knowledge(
            query=query, query_type=KnowledgeQueryType.TREND_ANALYSIS, max_tokens=1000
        )

    async def conduct_deep_research(
        self, query: str, context: Dict[str, Any], focus_areas: List[str] = None
    ) -> ResearchInteraction:
        """
        Conduct exhaustive deep research using Sonar Deep Research model
        Enterprise-tier research with comprehensive analysis across hundreds of sources
        """

        focus_areas = focus_areas or [
            "market analysis",
            "competitive landscape",
            "strategic implications",
        ]
        focus_context = ", ".join(focus_areas)

        enhanced_query = f"""
        Conduct comprehensive research analysis on: {query}
        
        Context: {context.get('industry', '')} {context.get('domain', '')}
        
        Focus Areas: {focus_context}
        
        Please provide exhaustive analysis with:
        1. Comprehensive market intelligence and competitive analysis
        2. Historical trends and future projections with specific data points
        3. Expert opinions and academic research citations
        4. Risk factors and strategic recommendations
        5. Quantitative analysis with benchmarks and metrics
        6. Cross-industry comparisons and best practices
        7. Regulatory and economic impact assessment
        8. Technology trends and innovation pipeline analysis
        
        Provide detailed evidence with authoritative source citations and confidence levels for each claim.
        Structure the response as a comprehensive research report suitable for executive decision-making.
        """

        # Execute deep research with enhanced metadata capture
        perplexity_response = await self.query_knowledge(
            query=enhanced_query,
            query_type=KnowledgeQueryType.MARKET_INTELLIGENCE,
            tier=ResearchTier.PREMIUM,  # Use the premium research tier for deep research
            max_tokens=4000,  # Larger token limit for comprehensive analysis
            operation_context="Deep Synthesis Pipeline",
        )

        # Convert PerplexityResponse to ResearchInteraction for pipeline integration
        return ResearchInteraction(
            research_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            query_sent=enhanced_query,
            raw_response_received=perplexity_response.content,
            sources_extracted=[
                {"url": source, "credibility": "high"}
                for source in perplexity_response.sources
            ],
            confidence_score=perplexity_response.confidence,
            search_mode="deep",
            sources_consulted_count=len(perplexity_response.sources),
            contradiction_detection_result={},
            time_taken_ms=int(perplexity_response.processing_time_ms),
            cost_usd=perplexity_response.cost_usd,
            query_type=perplexity_response.query_type.value,
            model_used="sonar-deep-research",
            tokens_used=perplexity_response.tokens_used,
            citations=(
                perplexity_response.citations
                if isinstance(perplexity_response.citations, list)
                else []
            ),
            operation_context="Deep Synthesis Pipeline",
        )

    def get_usage_metrics(self) -> Dict[str, Any]:
        """Get current usage metrics"""
        return {
            "total_queries": self.usage_metrics.total_queries,
            "total_tokens": self.usage_metrics.total_tokens,
            "total_cost_usd": self.usage_metrics.total_cost_usd,
            "queries_by_type": {
                k.value: v for k, v in self.usage_metrics.queries_by_type.items()
            },
            "avg_processing_time_ms": self.usage_metrics.avg_processing_time_ms,
            "estimated_monthly_cost": (
                self.usage_metrics.total_cost_usd * 30
                if self.usage_metrics.total_queries > 0
                else 0
            ),
        }

    async def _http_post(self, path: str, json: Dict[str, Any], retries: int = 2) -> Any:
        """Transport helper for POST with basic retry; returns httpx.Response."""
        if not self.client:
            raise RuntimeError("Perplexity client not initialized")
        last_exc = None
        for attempt in range(retries + 1):
            try:
                resp = await self.client.post(path, json=json)
                return resp
            except Exception as e:
                last_exc = e
                self.logger.warning(f"Transport error on POST {path} (attempt {attempt+1}/{retries+1}): {e}")
                await asyncio.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"Transport failed after {retries+1} attempts: {last_exc}")

    async def query(self, query_text: str, config: Dict[str, Any] | None = None) -> ResearchResult:  # type: ignore[override]
        """IResearchProvider implementation mapping to query_knowledge (REGULAR tier)."""
        config = config or {}
        start = datetime.utcnow()
        px = await self.query_knowledge(
            query=query_text,
            query_type=KnowledgeQueryType.MARKET_INTELLIGENCE,
            tier=config.get("tier", ResearchTier.REGULAR),
            max_tokens=config.get("max_tokens", 1200),
            operation_context=config.get("operation_context", "LLMManager-Research"),
        )
        # Map to ResearchResult
        elapsed_ms = int((datetime.utcnow() - start).total_seconds() * 1000)
        return ResearchResult(
            content=px.content,
            sources=[{"url": s} for s in px.sources],
            raw_response={
                "citations": px.citations,
                "tokens": px.tokens_used,
            },
            confidence=px.confidence,
            processing_time_ms=elapsed_ms,
            provider_name=self.provider_name,
            metadata={"query_type": px.query_type.value},
        )

    async def test_connection(self) -> Dict[str, Any]:
        """Test Perplexity connection and return diagnostics"""

        diagnostics = {
            "httpx_library": HTTPX_AVAILABLE,
            "api_key_present": bool(os.getenv("PERPLEXITY_API_KEY")),
            "client_initialized": self.client is not None,
            "connection_test": False,
            "error_message": None,
        }

        if not diagnostics["httpx_library"]:
            diagnostics["error_message"] = "httpx library not installed"
            return diagnostics

        if not diagnostics["api_key_present"]:
            diagnostics["error_message"] = "PERPLEXITY_API_KEY not set"
            return diagnostics

        if not diagnostics["client_initialized"]:
            diagnostics["error_message"] = "Client initialization failed"
            return diagnostics

        try:
            # Test with minimal query
            response = await self.query_knowledge(
                query="What is machine learning?",
                query_type=KnowledgeQueryType.FACT_CHECKING,
                max_tokens=50,
            )
            diagnostics["connection_test"] = True
        except Exception as e:
            diagnostics["error_message"] = str(e)

        return diagnostics


# Global Perplexity client instance
_perplexity_client_instance: Optional[PerplexityClient] = None


async def get_perplexity_client() -> PerplexityClient:
    """Get or create global Perplexity client instance"""
    global _perplexity_client_instance

    if _perplexity_client_instance is None:
        _perplexity_client_instance = PerplexityClient()

    return _perplexity_client_instance
