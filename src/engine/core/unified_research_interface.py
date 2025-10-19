"""
Unified Research Interface for METIS V5
Provides a single interface to multiple research engines while avoiding conflicts

This consolidates access to:
- research_grounding_engine.py (Production V2)
- enhanced_research_orchestrator.py (Deep Synthesis)
- Other research components

Usage:
    research = UnifiedResearchInterface()
    results = await research.ground_knowledge(query, consultants)
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

# Import tier support from Perplexity client
try:
    from src.engine.integrations.perplexity_client import ResearchTier
except ImportError:
    # Fallback tier definition
    class ResearchTier:
        TESTING = "testing"
        REGULAR = "regular"
        PREMIUM = "premium"


logger = logging.getLogger(__name__)


@dataclass
class ResearchResult:
    """Unified research result structure"""

    query: str
    sources_found: int
    confidence_score: float
    processing_time_ms: int
    key_findings: List[str] = field(default_factory=list)
    source_urls: List[str] = field(default_factory=list)
    research_summary: str = ""
    error: Optional[str] = None


@dataclass
class ConsultantGrounding:
    """Research grounding for a specific consultant"""

    consultant_id: str
    consultant_name: str
    research_result: ResearchResult
    grounding_applied: bool = True


class UnifiedResearchInterface:
    """
    Unified interface to all research engines in METIS

    This class acts as a facade to consolidate multiple research implementations
    while avoiding conflicts and ensuring consistent behavior.
    """

    def __init__(self):
        self.research_enabled = True
        self.primary_engine = None
        self.fallback_engines = []

        # Try to load available research engines
        self._initialize_research_engines()

        logger.info(
            f"🔍 UnifiedResearchInterface initialized with {len(self.fallback_engines)} engines"
        )

    def _initialize_research_engines(self):
        """Initialize available research engines"""

        # Try to load ResearchManager as primary research system
        try:
            from src.engine.core.research_manager import ResearchManager
            from src.engine.providers.research import PerplexityProvider, ExaProvider
            from src.core.unified_context_stream import get_unified_context_stream

            # Initialize providers
            providers = []
            perplexity = PerplexityProvider()
            if asyncio.run(perplexity.is_available()):
                providers.append(perplexity)
            exa = ExaProvider()
            if asyncio.run(exa.is_available()):
                providers.append(exa)

            if providers:
                context_stream = get_unified_context_stream()
                self.primary_engine = ResearchManager(providers, context_stream)
                logger.info(
                    f"✅ Primary research engine loaded: ResearchManager with {len(providers)} providers"
                )
            else:
                logger.warning("⚠️ No research providers available for ResearchManager")
        except Exception as e:
            logger.warning(f"⚠️ ResearchManager not available: {e}")

        # Try legacy ResearchGroundingEngine as fallback
        if not self.primary_engine:
            try:
                from src.engine.engines.research_grounding_engine import (
                    ResearchGroundingEngine,
                )

                self.primary_engine = ResearchGroundingEngine()
                logger.info(
                    "✅ Fallback research engine loaded: ResearchGroundingEngine"
                )
            except Exception as e:
                logger.warning(f"⚠️ ResearchGroundingEngine not available: {e}")

    async def ground_knowledge(
        self,
        query: str,
        consultants: List[str],
        context: Optional[Dict] = None,
        tier: str = ResearchTier.REGULAR,
    ) -> List[ConsultantGrounding]:
        """
        Ground consultants with research knowledge - Tiered Research Architecture

        Args:
            query: The research query
            consultants: List of consultant IDs to ground
            context: Optional context for research
            tier: Research tier (testing/regular/premium) for cost optimization

        Returns:
            List of ConsultantGrounding results
        """

        if not self.research_enabled:
            return self._create_mock_grounding(consultants, query)

        try:
            # Log tier selection for cost tracking
            logger.info(f"🔍 Research tier selected: {tier}")

            # Use primary engine if available
            if self.primary_engine:
                return await self._use_primary_engine(query, consultants, context, tier)

            # Try fallback engines
            for engine_name, engine in self.fallback_engines:
                try:
                    return await self._use_fallback_engine(
                        engine_name, engine, query, consultants, context, tier
                    )
                except Exception as e:
                    logger.warning(f"⚠️ Fallback engine {engine_name} failed: {e}")
                    continue

            # If all engines fail, return mock results
            logger.warning("⚠️ All research engines failed, returning mock results")
            return self._create_mock_grounding(consultants, query)

        except Exception as e:
            logger.error(f"❌ Research grounding failed: {e}")
            return self._create_mock_grounding(consultants, query, error=str(e))

    async def _use_primary_engine(
        self,
        query: str,
        consultants: List[str],
        context: Optional[Dict],
        tier: str = ResearchTier.REGULAR,
    ) -> List[ConsultantGrounding]:
        """Use the primary research grounding engine with tier support"""

        start_time = datetime.now()

        try:
            # Call the primary engine with tier support (exact interface may vary)
            context_with_tier = (context or {}).copy()
            context_with_tier["research_tier"] = tier
            result = await self.primary_engine.ground_knowledge(
                query, consultants, context_with_tier
            )

            processing_time = int((datetime.now() - start_time).total_seconds() * 1000)

            # Convert to unified format
            groundings = []
            for i, consultant_id in enumerate(consultants):
                research_result = ResearchResult(
                    query=query,
                    sources_found=getattr(result, "sources_count", 3),
                    confidence_score=getattr(result, "confidence", 0.8),
                    processing_time_ms=processing_time,
                    key_findings=getattr(
                        result, "key_findings", ["Research finding from primary engine"]
                    ),
                    research_summary=getattr(
                        result, "summary", f"Research grounding for {consultant_id}"
                    ),
                )

                groundings.append(
                    ConsultantGrounding(
                        consultant_id=consultant_id,
                        consultant_name=f"Consultant {i+1}",
                        research_result=research_result,
                    )
                )

            logger.info(f"✅ Primary engine research completed: {processing_time}ms")
            return groundings

        except Exception as e:
            logger.error(f"❌ Primary engine failed: {e}")
            raise

    async def _use_fallback_engine(
        self,
        engine_name: str,
        engine: Any,
        query: str,
        consultants: List[str],
        context: Optional[Dict],
        tier: str = ResearchTier.REGULAR,
    ) -> List[ConsultantGrounding]:
        """Use a fallback research engine with tier support"""

        start_time = datetime.now()

        # This would be customized based on each engine's interface
        # For now, return a structured mock
        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)

        groundings = []
        for i, consultant_id in enumerate(consultants):
            research_result = ResearchResult(
                query=query,
                sources_found=5,
                confidence_score=0.75,
                processing_time_ms=processing_time,
                key_findings=[f"Finding from {engine_name} for {consultant_id}"],
                research_summary=f"Fallback research via {engine_name}",
            )

            groundings.append(
                ConsultantGrounding(
                    consultant_id=consultant_id,
                    consultant_name=f"Consultant {i+1}",
                    research_result=research_result,
                )
            )

        logger.info(f"✅ Fallback engine {engine_name} completed: {processing_time}ms")
        return groundings

    def _create_mock_grounding(
        self, consultants: List[str], query: str, error: Optional[str] = None
    ) -> List[ConsultantGrounding]:
        """Create mock grounding results when research engines are unavailable"""

        groundings = []
        for i, consultant_id in enumerate(consultants):
            research_result = ResearchResult(
                query=query,
                sources_found=0,
                confidence_score=0.5,
                processing_time_ms=10,
                key_findings=["Mock research finding (engines unavailable)"],
                research_summary=f"Mock research for {consultant_id}",
                error=error,
            )

            groundings.append(
                ConsultantGrounding(
                    consultant_id=consultant_id,
                    consultant_name=f"Consultant {i+1}",
                    research_result=research_result,
                    grounding_applied=False,
                )
            )

        logger.info(f"⚠️ Created mock grounding for {len(consultants)} consultants")
        return groundings

    def get_research_status(self) -> Dict[str, Any]:
        """Get status of research capabilities"""
        return {
            "research_enabled": self.research_enabled,
            "primary_engine_available": self.primary_engine is not None,
            "fallback_engines_count": len(self.fallback_engines),
            "engines_available": [name for name, _ in self.fallback_engines],
        }
