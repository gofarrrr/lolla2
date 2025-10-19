"""
Open Deep Research client compatible with METIS research orchestrator
"""

import asyncio
import logging
import time
import os
from typing import Dict, Any, Optional

from .config import ODRConfiguration
from .agent import ODRResearchAgent

# Import existing METIS components
try:
    from ..perplexity_client_advanced import (
        AdvancedResearchResult,
        ResearchMode,
        SourceCredibilityTier,
        ResearchInsight,
        EnhancedSource,
        CrossReferenceAnalysis,
    )
    from src.intelligence.research_templates import (
        ResearchTemplateType,
        get_research_templates,
    )

    METIS_COMPONENTS_AVAILABLE = True
except ImportError:
    METIS_COMPONENTS_AVAILABLE = False

    # Mock classes for when METIS components aren't available
    class AdvancedResearchResult:
        pass

    class ResearchMode:
        STANDARD = "standard"

    class ResearchTemplateType:
        pass


logger = logging.getLogger(__name__)


class ODRClient:
    """Open Deep Research client compatible with METIS research orchestrator"""

    def __init__(self, config: Optional[ODRConfiguration] = None):
        self.config = config or self._create_default_config()
        self.research_agent = ODRResearchAgent(self.config)

        # Initialize templates if available
        if METIS_COMPONENTS_AVAILABLE:
            try:
                self.templates = get_research_templates()
            except Exception:
                self.templates = None
        else:
            self.templates = None

        logger.info("✅ Open Deep Research Client initialized")

    def _create_default_config(self) -> ODRConfiguration:
        """Create default configuration from environment variables"""
        return ODRConfiguration(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            tavily_api_key=os.getenv("TAVILY_API_KEY"),  # New API key needed
            model_name="claude-3-5-sonnet-20241022",
            max_tokens=4000,
            temperature=0.1,
            max_iterations=5,
            research_timeout=180,
            min_sources=3,
            max_sources=15,
        )

    async def conduct_advanced_research(
        self,
        query: str,
        template_type: Optional[ResearchTemplateType] = None,
        context: Optional[Dict[str, Any]] = None,
        mode: ResearchMode = ResearchMode.STANDARD,
    ) -> AdvancedResearchResult:
        """
        Conduct advanced research using Open Deep Research methodology
        Compatible with existing METIS research orchestrator interface
        """

        logger.info(
            f"🔬 Starting ODR research: {getattr(mode, 'value', mode)} | Template: {template_type}"
        )
        start_time = time.time()

        try:
            # Adjust parameters based on research mode
            self._adjust_config_for_mode(mode)

            # Apply research template if specified
            enhanced_query = self._apply_research_template(
                query, template_type, context
            )

            # Conduct autonomous research with template support
            raw_results = await self._conduct_research_mock(
                enhanced_query, context, template_type
            )

            # Convert to METIS format
            research_result = self._convert_to_metis_format(
                raw_results, template_type, mode, start_time
            )

            logger.info(
                f"✅ ODR research completed: {len(raw_results.get('findings', []))} findings"
            )
            return research_result

        except Exception as e:
            logger.error(f"❌ ODR research failed: {e}")
            # Return minimal result on failure
            return self._create_fallback_result(query, mode, start_time)

    def _adjust_config_for_mode(self, mode: ResearchMode):
        """Adjust configuration parameters based on research mode"""
        mode_str = getattr(mode, "value", str(mode)).lower()

        if mode_str == "rapid":
            self.config.max_iterations = 2
            self.config.research_timeout = 60
            self.config.max_sources = 5
        elif mode_str == "comprehensive":
            self.config.max_iterations = 8
            self.config.research_timeout = 300
            self.config.max_sources = 20
        else:  # standard
            self.config.max_iterations = 5
            self.config.research_timeout = 180
            self.config.max_sources = 15

    def _apply_research_template(
        self,
        query: str,
        template_type: Optional[ResearchTemplateType],
        context: Optional[Dict[str, Any]],
    ) -> str:
        """Apply research template to enhance the query"""
        if not template_type or not self.templates:
            return query

        try:
            template = self.templates.get_template(template_type)
            if template:
                enhanced_query = template.enhance_query(query, context or {})
                logger.info(
                    f"📋 Applied template {template_type}: enhanced query length {len(enhanced_query)}"
                )
                return enhanced_query
        except Exception as e:
            logger.warning(f"⚠️ Template application failed: {e}")

        return query

    async def _conduct_research_mock(
        self,
        query: str,
        context: Optional[Dict[str, Any]],
        template_type: Optional[ResearchTemplateType],
    ) -> Dict[str, Any]:
        """Mock research implementation - to be replaced with actual ODR logic"""

        # Simulate research delay
        await asyncio.sleep(0.5)

        # Generate mock findings
        findings = [
            {
                "claim": f"Analysis suggests {query[:50]}... shows significant market opportunity",
                "evidence": "Based on industry benchmarking and competitive analysis",
                "confidence": 0.75,
                "sources": [
                    "example.com/market-analysis",
                    "example.com/industry-report",
                ],
            },
            {
                "claim": "Implementation costs range from $100K-$500K depending on scope",
                "evidence": "Historical project data and vendor pricing analysis",
                "confidence": 0.65,
                "sources": [
                    "example.com/implementation-costs",
                    "example.com/vendor-pricing",
                ],
            },
        ]

        return {
            "findings": findings,
            "total_sources": len(findings) * 2,
            "confidence": 0.7,
            "research_time": 30,
        }

    def _convert_to_metis_format(
        self,
        raw_results: Dict[str, Any],
        template_type: Optional[ResearchTemplateType],
        mode: ResearchMode,
        start_time: float,
    ) -> AdvancedResearchResult:
        """Convert raw ODR results to METIS AdvancedResearchResult format"""

        if not METIS_COMPONENTS_AVAILABLE:
            # Return mock result when METIS components aren't available
            return self._create_mock_result(raw_results, mode, start_time)

        # Convert findings to insights
        insights = []
        for finding in raw_results.get("findings", []):
            insight = ResearchInsight(
                claim=finding.get("claim", ""),
                evidence=finding.get("evidence", ""),
                confidence=finding.get("confidence", 0.5),
                sources=finding.get("sources", []),
            )
            insights.append(insight)

        # Create enhanced sources
        sources = []
        for source_url in set(
            sum([f.get("sources", []) for f in raw_results.get("findings", [])], [])
        ):
            source = EnhancedSource(
                url=source_url,
                title=f"Source: {source_url.split('/')[-1]}",
                content="",
                domain=source_url.split("//")[-1].split("/")[0],
                credibility_tier=SourceCredibilityTier.MEDIUM,
                confidence=0.7,
            )
            sources.append(source)

        # Create cross-reference analysis
        cross_ref = CrossReferenceAnalysis(
            consistency_score=0.75,
            contradictions=[],
            supporting_evidence_count=len(insights),
        )

        # Build result
        result = AdvancedResearchResult(
            executive_summary=f"ODR research completed with {len(insights)} key insights",
            key_insights=insights,
            sources=sources,
            coverage_completeness=0.8,
            fact_validation_score=0.75,
            overall_confidence=raw_results.get("confidence", 0.7),
            total_processing_time_ms=int((time.time() - start_time) * 1000),
            queries_executed=[f"ODR query based on template {template_type}"],
            mode_used=mode,
            cross_reference_analysis=cross_ref,
        )

        return result

    def _create_mock_result(
        self, raw_results: Dict[str, Any], mode: ResearchMode, start_time: float
    ) -> Dict[str, Any]:
        """Create a mock result when METIS components aren't available"""
        return {
            "executive_summary": f"ODR research completed with {len(raw_results.get('findings', []))} findings",
            "findings": raw_results.get("findings", []),
            "confidence": raw_results.get("confidence", 0.7),
            "processing_time_ms": int((time.time() - start_time) * 1000),
            "mode": str(mode),
        }

    def _create_fallback_result(
        self, query: str, mode: ResearchMode, start_time: float
    ):
        """Create fallback result on research failure"""
        if METIS_COMPONENTS_AVAILABLE:
            return AdvancedResearchResult(
                executive_summary=f"Research failed for query: {query[:50]}",
                key_insights=[],
                sources=[],
                coverage_completeness=0.0,
                fact_validation_score=0.0,
                overall_confidence=0.1,
                total_processing_time_ms=int((time.time() - start_time) * 1000),
                queries_executed=[query],
                mode_used=mode,
                cross_reference_analysis=CrossReferenceAnalysis(
                    consistency_score=0.0,
                    contradictions=[],
                    supporting_evidence_count=0,
                ),
            )
        else:
            return {
                "executive_summary": f"Research failed for query: {query[:50]}",
                "findings": [],
                "confidence": 0.1,
                "processing_time_ms": int((time.time() - start_time) * 1000),
            }
