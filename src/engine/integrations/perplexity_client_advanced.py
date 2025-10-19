"""
Advanced Perplexity Client for METIS Cognitive Platform
Enhanced research capabilities with dynamic prompt engineering and multi-query orchestration

Features:
- Template-based research with specialized prompts
- Multi-query orchestration for comprehensive coverage
- Advanced source credibility assessment
- Cross-reference validation and contradiction detection
- Progressive deepening with adaptive token allocation
- Research quality scoring and confidence assessment

Author: METIS Cognitive Platform
Date: 2025
"""

import asyncio
import logging
import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re

# Import existing components
from .perplexity_client import (
    PerplexityClient,
    KnowledgeQueryType,
    PerplexityResponse,
    get_perplexity_client,
)

# Import new research templates
from src.intelligence.research_templates import (
    EnhancedResearchTemplates,
    ResearchTemplateType,
    get_research_templates,
)

logger = logging.getLogger(__name__)


class ResearchMode(str, Enum):
    """Enhanced research execution modes"""

    RAPID = "rapid"  # 1-2 queries, 30s max
    STANDARD = "standard"  # 3-5 queries, 60s max
    COMPREHENSIVE = "comprehensive"  # 5-8 queries, 120s max
    DEEP_DIVE = "deep_dive"  # 8-12 queries, 180s max


class SourceCredibilityTier(str, Enum):
    """Source credibility classification"""

    TIER_1 = "tier_1"  # Government, academic, major news outlets
    TIER_2 = "tier_2"  # Industry publications, reputable organizations
    TIER_3 = "tier_3"  # General business/news sites
    TIER_4 = "tier_4"  # Blogs, opinion sites
    UNVERIFIED = "unverified"


@dataclass
class EnhancedSource:
    """Enhanced source information with credibility assessment"""

    url: str
    title: str
    content: str
    domain: str
    credibility_tier: SourceCredibilityTier
    credibility_score: float  # 0-1
    date: Optional[str] = None
    author: Optional[str] = None
    publication_type: Optional[str] = None
    bias_indicators: List[str] = field(default_factory=list)
    fact_density: float = 0.0  # Facts per 100 words
    citation_quality: float = 0.0  # Quality of internal citations


@dataclass
class ResearchInsight:
    """Individual research insight with validation"""

    claim: str
    confidence: float
    supporting_sources: List[str]
    contradicting_sources: List[str]
    evidence_strength: float
    fact_type: str  # "numeric", "categorical", "temporal", "causal"
    verification_status: str  # "verified", "disputed", "unverified"


@dataclass
class CrossReferenceAnalysis:
    """Cross-reference validation results"""

    total_claims: int
    verified_claims: int
    disputed_claims: int
    consistency_score: float  # 0-1
    contradictions: List[Dict[str, Any]]
    confidence_distribution: Dict[str, int]  # confidence ranges -> count


@dataclass
class AdvancedResearchResult:
    """Comprehensive research result with enhanced attribution"""

    # Core content
    executive_summary: str
    key_insights: List[ResearchInsight]
    detailed_findings: str

    # Sources and validation
    sources: List[EnhancedSource]
    cross_reference_analysis: CrossReferenceAnalysis

    # Quality metrics
    overall_confidence: float
    coverage_completeness: float
    source_diversity_score: float
    fact_validation_score: float

    # Research metadata
    template_used: ResearchTemplateType
    queries_executed: List[str]
    mode_used: ResearchMode
    total_processing_time_ms: int
    tokens_consumed: int
    cost_usd: float

    # Research quality indicators
    information_gaps: List[str]
    additional_research_recommendations: List[str]
    confidence_limitations: List[str]


class AdvancedPerplexityClient:
    """
    Advanced Perplexity client with template-based research orchestration
    and comprehensive validation capabilities
    """

    # Research mode configurations
    MODE_CONFIGS = {
        ResearchMode.RAPID: {
            "max_queries": 2,
            "max_time_seconds": 30,
            "max_tokens": 2000,
            "depth_level": 1,
        },
        ResearchMode.STANDARD: {
            "max_queries": 5,
            "max_time_seconds": 60,
            "max_tokens": 4000,
            "depth_level": 2,
        },
        ResearchMode.COMPREHENSIVE: {
            "max_queries": 8,
            "max_time_seconds": 120,
            "max_tokens": 6000,
            "depth_level": 3,
        },
        ResearchMode.DEEP_DIVE: {
            "max_queries": 12,
            "max_time_seconds": 180,
            "max_tokens": 8000,
            "depth_level": 4,
        },
    }

    # Domain credibility mapping
    CREDIBILITY_DOMAINS = {
        SourceCredibilityTier.TIER_1: {
            "gov",
            "edu",
            "nature.com",
            "science.org",
            "nejm.org",
            "bmj.com",
            "who.int",
            "worldbank.org",
            "imf.org",
            "oecd.org",
            "reuters.com",
            "apnews.com",
            "bbc.com",
            "npr.org",
            "pbs.org",
        },
        SourceCredibilityTier.TIER_2: {
            "wsj.com",
            "ft.com",
            "bloomberg.com",
            "economist.com",
            "harvard.edu",
            "mit.edu",
            "stanford.edu",
            "techcrunch.com",
            "wired.com",
            "ieee.org",
            "mckinsey.com",
            "bcg.com",
            "bain.com",
            "deloitte.com",
        },
        SourceCredibilityTier.TIER_3: {
            "forbes.com",
            "fortune.com",
            "businessinsider.com",
            "cnbc.com",
            "cnn.com",
            "nytimes.com",
            "washingtonpost.com",
            "theguardian.com",
        },
    }

    def __init__(self):
        self.logger = logger
        self.base_client: Optional[PerplexityClient] = None
        self.research_templates: EnhancedResearchTemplates = get_research_templates()

        # Research quality thresholds
        self.min_confidence_threshold = 0.6
        self.min_source_diversity = 3
        self.max_contradiction_tolerance = 0.2

        self.logger.info("✅ Advanced Perplexity Client initialized")

    async def _get_base_client(self) -> PerplexityClient:
        """Get base Perplexity client instance"""
        if not self.base_client:
            self.base_client = await get_perplexity_client()
        return self.base_client

    def _classify_domain_credibility(
        self, url: str
    ) -> Tuple[SourceCredibilityTier, float]:
        """Classify domain credibility and assign score"""

        domain = self._extract_domain(url)

        for tier, domains in self.CREDIBILITY_DOMAINS.items():
            if any(domain.endswith(d) for d in domains):
                credibility_scores = {
                    SourceCredibilityTier.TIER_1: 0.9,
                    SourceCredibilityTier.TIER_2: 0.75,
                    SourceCredibilityTier.TIER_3: 0.6,
                }
                return tier, credibility_scores[tier]

        # Additional heuristics for unclassified domains
        if domain.endswith(".gov"):
            return SourceCredibilityTier.TIER_1, 0.85
        elif domain.endswith(".edu"):
            return SourceCredibilityTier.TIER_1, 0.8
        elif domain.endswith(".org"):
            return SourceCredibilityTier.TIER_2, 0.7
        else:
            return SourceCredibilityTier.TIER_4, 0.4

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            import re

            domain_match = re.search(r"https?://(?:www\.)?([^/]+)", url)
            if domain_match:
                domain = domain_match.group(1).lower()
                return re.sub(r"^(www|m|mobile)\.", "", domain)
            return "unknown"
        except Exception:
            return "unknown"

    def _assess_content_quality(
        self, content: str, title: str = ""
    ) -> Dict[str, float]:
        """Assess content quality indicators"""

        # Fact density: numeric claims, citations, specific details
        fact_patterns = [
            r"\d+(?:\.\d+)?%",  # Percentages
            r"\$\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(million|billion|thousand))?",  # Money
            r"\d{4}",  # Years
            r"according to",  # Attribution
            r"study (?:by|from)",  # Research citations
            r"data (?:shows?|indicates?)",  # Data references
        ]

        fact_count = 0
        for pattern in fact_patterns:
            fact_count += len(re.findall(pattern, content.lower()))

        word_count = len(content.split())
        fact_density = min(fact_count / max(word_count / 100, 1), 1.0)

        # Citation quality: presence of specific references
        citation_patterns = [
            r"(?:according to|per|via|source:)\s+[A-Z][^.]*",
            r"\([^)]*\d{4}[^)]*\)",  # Year citations
            r"https?://[^\s]+",  # URLs
        ]

        citation_count = 0
        for pattern in citation_patterns:
            citation_count += len(re.findall(pattern, content))

        citation_quality = min(citation_count / max(word_count / 200, 1), 1.0)

        # Bias indicators
        bias_words = [
            "shocking",
            "amazing",
            "incredible",
            "unbelievable",
            "secret",
            "you won't believe",
            "doctors hate",
            "one weird trick",
            "must read",
            "breaking",
            "exclusive",
        ]

        bias_score = 0
        text_lower = f"{title} {content}".lower()
        for word in bias_words:
            if word in text_lower:
                bias_score += 1

        bias_penalty = min(bias_score * 0.1, 0.5)

        return {
            "fact_density": fact_density,
            "citation_quality": citation_quality,
            "bias_penalty": bias_penalty,
        }

    def _generate_progressive_queries(
        self,
        initial_query: str,
        template_type: ResearchTemplateType,
        context: Dict[str, Any],
        depth_level: int,
    ) -> List[str]:
        """Generate progressive queries for comprehensive research"""

        queries = [initial_query]  # Start with base query

        # Level 1: Contextual variations
        if depth_level >= 1:
            industry = context.get("industry", "")
            region = context.get("region", "")

            if industry:
                queries.append(f"{initial_query} {industry} industry")
            if region:
                queries.append(f"{initial_query} {region}")

        # Level 2: Specific dimensions
        if depth_level >= 2:
            dimension_queries = {
                ResearchTemplateType.MARKET_ANALYSIS: [
                    f"{initial_query} market size trends",
                    f"{initial_query} competitive landscape",
                ],
                ResearchTemplateType.COMPETITIVE_INTELLIGENCE: [
                    f"{initial_query} financial performance",
                    f"{initial_query} strategic positioning",
                ],
                ResearchTemplateType.INVESTMENT_EVALUATION: [
                    f"{initial_query} risk factors",
                    f"{initial_query} valuation analysis",
                ],
                ResearchTemplateType.TECHNOLOGY_TRENDS: [
                    f"{initial_query} adoption rates",
                    f"{initial_query} innovation pipeline",
                ],
            }

            if template_type in dimension_queries:
                queries.extend(dimension_queries[template_type])

        # Level 3: Validation and alternatives
        if depth_level >= 3:
            queries.extend(
                [
                    f"challenges with {initial_query}",
                    f"criticism of {initial_query}",
                    f"alternative approaches to {initial_query}",
                ]
            )

        # Level 4: Deep domain expertise
        if depth_level >= 4:
            queries.extend(
                [
                    f"{initial_query} expert analysis research studies",
                    f"{initial_query} case studies best practices",
                    f"{initial_query} future trends predictions",
                ]
            )

        return queries[: self.MODE_CONFIGS[ResearchMode.DEEP_DIVE]["max_queries"]]

    async def _execute_template_query(
        self, query: str, template_type: ResearchTemplateType, context: Dict[str, Any]
    ) -> PerplexityResponse:
        """Execute query using specific research template"""

        template = self.research_templates.get_template(template_type)
        if not template:
            raise ValueError(f"Template {template_type} not found")

        # Use template's system prompt for enhanced context
        enhanced_query = f"""
        Research Context: {template.description}
        
        Query: {query}
        
        Please provide analysis following these guidelines:
        - Use authoritative sources and cite them specifically
        - Include quantitative data where available
        - Flag any limitations or conflicting information
        - Provide confidence levels for key claims
        - Focus on actionable insights and strategic implications
        """

        client = await self._get_base_client()

        return await client.query_knowledge(
            query=enhanced_query,
            query_type=KnowledgeQueryType.CONTEXT_GROUNDING,
            max_tokens=template.expected_token_range[1],
        )

    def _extract_insights_from_response(
        self, response: PerplexityResponse, query: str
    ) -> List[ResearchInsight]:
        """Extract structured insights from Perplexity response"""

        insights = []
        content = response.content

        # Extract claims with confidence indicators
        claim_patterns = [
            (r"(\d+(?:\.\d+)?%\s+of[^.]+)", "numeric"),
            (r"(according to[^,]+,[^.]+)", "attribution"),
            (r"(studies? (?:show|indicate|suggest)[^.]+)", "research"),
            (r"([A-Z][^.]*(?:increased|decreased|grew|declined)[^.]*)", "trend"),
            (r"([A-Z][^.]*(?:will|expected to|projected)[^.]*)", "forecast"),
        ]

        for pattern, fact_type in claim_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)

            for match in matches:
                claim = match.group(1).strip()

                if len(claim) > 20 and len(claim) < 300:  # Reasonable length
                    # Assess confidence based on language
                    confidence = self._assess_claim_confidence(claim, response.sources)

                    insight = ResearchInsight(
                        claim=claim,
                        confidence=confidence,
                        supporting_sources=response.sources,
                        contradicting_sources=[],
                        evidence_strength=confidence,
                        fact_type=fact_type,
                        verification_status="unverified",
                    )

                    insights.append(insight)

        return insights[:10]  # Limit to top insights

    def _assess_claim_confidence(self, claim: str, sources: List[str]) -> float:
        """Assess confidence level of a specific claim"""

        confidence = 0.5  # Base confidence

        # Language confidence indicators
        high_confidence_words = [
            "according to",
            "data shows",
            "research indicates",
            "study found",
        ]
        medium_confidence_words = ["suggests", "indicates", "appears", "likely"]
        low_confidence_words = ["might", "could", "perhaps", "possibly", "may"]

        claim_lower = claim.lower()

        if any(word in claim_lower for word in high_confidence_words):
            confidence += 0.3
        elif any(word in claim_lower for word in medium_confidence_words):
            confidence += 0.1
        elif any(word in claim_lower for word in low_confidence_words):
            confidence -= 0.2

        # Source quantity bonus
        source_bonus = min(len(sources) * 0.05, 0.2)
        confidence += source_bonus

        # Numeric specificity bonus
        if re.search(r"\d+(?:\.\d+)?%", claim):
            confidence += 0.1

        return min(max(confidence, 0.1), 1.0)

    def _perform_cross_reference_analysis(
        self, insights: List[ResearchInsight]
    ) -> CrossReferenceAnalysis:
        """Perform cross-reference validation across insights"""

        total_claims = len(insights)
        verified_claims = 0
        disputed_claims = 0
        contradictions = []

        # Simple consistency check for numeric claims
        numeric_claims = {}

        for insight in insights:
            if insight.fact_type == "numeric":
                # Extract numbers from claim
                numbers = re.findall(r"\d+(?:\.\d+)?", insight.claim)
                for num in numbers:
                    num_val = float(num)
                    topic_key = re.sub(
                        r"\d+(?:\.\d+)?%?", "[NUM]", insight.claim.lower()
                    )

                    if topic_key not in numeric_claims:
                        numeric_claims[topic_key] = []
                    numeric_claims[topic_key].append((num_val, insight.claim))

        # Check for contradictory numeric claims
        for topic, claims in numeric_claims.items():
            if len(claims) > 1:
                values = [claim[0] for claim in claims]
                min_val = min(values)
                max_val = max(values)

                # If values differ significantly, flag as contradiction
                if max_val > 0 and (max_val - min_val) / max_val > 0.3:
                    contradictions.append(
                        {
                            "type": "numeric_contradiction",
                            "topic": topic,
                            "conflicting_values": values,
                            "claims": [claim[1] for claim in claims],
                        }
                    )
                    disputed_claims += len(claims)
                else:
                    verified_claims += len(claims)

        # Calculate consistency score
        consistency_score = verified_claims / max(total_claims, 1)

        # Confidence distribution
        confidence_ranges = {"high": 0, "medium": 0, "low": 0}
        for insight in insights:
            if insight.confidence >= 0.8:
                confidence_ranges["high"] += 1
            elif insight.confidence >= 0.6:
                confidence_ranges["medium"] += 1
            else:
                confidence_ranges["low"] += 1

        return CrossReferenceAnalysis(
            total_claims=total_claims,
            verified_claims=verified_claims,
            disputed_claims=disputed_claims,
            consistency_score=consistency_score,
            contradictions=contradictions,
            confidence_distribution=confidence_ranges,
        )

    def _calculate_coverage_completeness(
        self,
        queries_executed: List[str],
        sources: List[EnhancedSource],
        template_type: ResearchTemplateType,
    ) -> float:
        """Calculate how complete the research coverage is"""

        # Domain diversity
        unique_domains = len(set(source.domain for source in sources))
        domain_score = min(unique_domains / 5, 1.0)

        # Credibility tier distribution
        tier_distribution = {}
        for source in sources:
            tier_distribution[source.credibility_tier] = (
                tier_distribution.get(source.credibility_tier, 0) + 1
            )

        # Bonus for having tier 1 sources
        credibility_score = 0.5
        if SourceCredibilityTier.TIER_1 in tier_distribution:
            credibility_score += 0.3
        if SourceCredibilityTier.TIER_2 in tier_distribution:
            credibility_score += 0.2

        # Query depth score
        query_depth_score = min(len(queries_executed) / 5, 1.0)

        # Combined coverage score
        coverage_score = (
            domain_score * 0.4 + credibility_score * 0.4 + query_depth_score * 0.2
        )

        return min(coverage_score, 1.0)

    async def conduct_advanced_research(
        self,
        query: str,
        template_type: Optional[ResearchTemplateType] = None,
        context: Optional[Dict[str, Any]] = None,
        mode: ResearchMode = ResearchMode.STANDARD,
    ) -> AdvancedResearchResult:
        """
        Conduct advanced research with template-based orchestration
        and comprehensive validation
        """

        start_time = time.time()
        context = context or {}

        # Auto-select template if not specified
        if not template_type:
            template_type = self.research_templates.select_template_by_context(
                {"query": query, **context}
            )

        self.logger.info(
            f"🔬 Starting advanced research: {mode.value} mode | Template: {template_type.value}"
        )

        config = self.MODE_CONFIGS[mode]

        try:
            # Generate progressive queries
            queries = self._generate_progressive_queries(
                query, template_type, context, config["depth_level"]
            )[: config["max_queries"]]

            # Execute queries with enhanced processing
            all_responses = []
            all_sources = []
            total_tokens = 0
            total_cost = 0.0

            for i, query_text in enumerate(queries):
                # Check time budget
                elapsed = time.time() - start_time
                if elapsed >= config["max_time_seconds"]:
                    self.logger.warning(f"⏰ Time budget exceeded: {elapsed:.1f}s")
                    break

                try:
                    # Execute template-enhanced query
                    response = await asyncio.wait_for(
                        self._execute_template_query(
                            query_text, template_type, context
                        ),
                        timeout=min(30, config["max_time_seconds"] - elapsed),
                    )

                    all_responses.append(response)
                    total_tokens += response.tokens_used
                    total_cost += response.cost_usd

                    # Process sources with enhanced assessment
                    for j, source_url in enumerate(response.sources):
                        tier, credibility = self._classify_domain_credibility(
                            source_url
                        )

                        # Use response content for first source, limited for others
                        content = response.content if j == 0 else response.content[:200]

                        quality_metrics = self._assess_content_quality(content)

                        enhanced_source = EnhancedSource(
                            url=source_url,
                            title=f"Source {len(all_sources) + 1}",
                            content=content,
                            domain=self._extract_domain(source_url),
                            credibility_tier=tier,
                            credibility_score=credibility
                            - quality_metrics["bias_penalty"],
                            fact_density=quality_metrics["fact_density"],
                            citation_quality=quality_metrics["citation_quality"],
                            bias_indicators=[],
                        )

                        all_sources.append(enhanced_source)

                    self.logger.info(
                        f"✅ Query {i+1}/{len(queries)} completed: {len(response.sources)} sources"
                    )

                except asyncio.TimeoutError:
                    self.logger.warning(f"⏰ Query timeout: {query_text[:50]}...")
                    continue
                except Exception as e:
                    self.logger.error(f"❌ Query failed: {e}")
                    continue

            # Extract and validate insights
            all_insights = []
            for response in all_responses:
                insights = self._extract_insights_from_response(response, query)
                all_insights.extend(insights)

            # Remove duplicate insights
            unique_insights = []
            seen_claims = set()
            for insight in all_insights:
                claim_hash = hashlib.md5(insight.claim.lower().encode()).hexdigest()
                if claim_hash not in seen_claims:
                    unique_insights.append(insight)
                    seen_claims.add(claim_hash)

            # Perform cross-reference analysis
            cross_ref_analysis = self._perform_cross_reference_analysis(unique_insights)

            # Calculate quality metrics
            processing_time = int((time.time() - start_time) * 1000)
            coverage_completeness = self._calculate_coverage_completeness(
                queries, all_sources, template_type
            )

            source_diversity = len(set(source.domain for source in all_sources)) / max(
                len(all_sources), 1
            )

            overall_confidence = (
                cross_ref_analysis.consistency_score * 0.4
                + coverage_completeness * 0.3
                + source_diversity * 0.3
            )

            # Generate executive summary
            if unique_insights:
                top_insights = sorted(
                    unique_insights, key=lambda x: x.confidence, reverse=True
                )[:5]
                executive_summary = (
                    f"Research identified {len(unique_insights)} key insights with {overall_confidence:.1%} confidence. "
                    + " ".join(
                        [insight.claim[:100] + "..." for insight in top_insights[:2]]
                    )
                )
            else:
                executive_summary = "Research completed but limited insights extracted. Consider refining research parameters."

            # Detailed findings compilation
            detailed_findings = "\n\n".join(
                [
                    f"**Insight {i+1}** (Confidence: {insight.confidence:.1%})\n{insight.claim}"
                    for i, insight in enumerate(unique_insights[:10])
                ]
            )

            # Identify information gaps
            information_gaps = []
            if coverage_completeness < 0.7:
                information_gaps.append(
                    "Limited source diversity - consider additional research"
                )
            if cross_ref_analysis.disputed_claims > 0:
                information_gaps.append(
                    "Contradictory information found - requires validation"
                )
            if overall_confidence < 0.6:
                information_gaps.append(
                    "Low overall confidence - need more authoritative sources"
                )

            # Additional research recommendations
            recommendations = []
            if len(all_sources) < 5:
                recommendations.append("Expand source base with additional queries")
            if cross_ref_analysis.contradictions:
                recommendations.append(
                    "Investigate contradictory claims with focused research"
                )

            result = AdvancedResearchResult(
                executive_summary=executive_summary,
                key_insights=unique_insights[:10],
                detailed_findings=detailed_findings,
                sources=all_sources,
                cross_reference_analysis=cross_ref_analysis,
                overall_confidence=overall_confidence,
                coverage_completeness=coverage_completeness,
                source_diversity_score=source_diversity,
                fact_validation_score=cross_ref_analysis.consistency_score,
                template_used=template_type,
                queries_executed=queries,
                mode_used=mode,
                total_processing_time_ms=processing_time,
                tokens_consumed=total_tokens,
                cost_usd=total_cost,
                information_gaps=information_gaps,
                additional_research_recommendations=recommendations,
                confidence_limitations=[],
            )

            self.logger.info(
                f"🎯 Advanced research completed: {len(unique_insights)} insights | "
                f"Confidence: {overall_confidence:.1%} | "
                f"Coverage: {coverage_completeness:.1%} | "
                f"{processing_time}ms"
            )

            return result

        except Exception as e:
            self.logger.error(f"❌ Advanced research failed: {e}")
            raise


# Global instance for easy access
_advanced_perplexity_client: Optional[AdvancedPerplexityClient] = None


async def get_advanced_perplexity_client() -> AdvancedPerplexityClient:
    """Get or create global advanced Perplexity client instance"""
    global _advanced_perplexity_client

    if _advanced_perplexity_client is None:
        _advanced_perplexity_client = AdvancedPerplexityClient()

    return _advanced_perplexity_client
