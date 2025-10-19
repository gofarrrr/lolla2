"""
Analysis Orchestration Service

Orchestrates end-to-end analysis execution including consultant/model selection,
LLM generation, memory context integration, and quality scoring.

Extracted from src/main.py as part of Operation Lean - Target #2.
"""

import logging
import time
import uuid
from typing import Dict, Any, Optional, List

from src.services.application.contracts import AnalysisContext, AnalysisResult, Tier
from src.orchestration.dispatch_orchestrator import (
    DispatchOrchestrator,
    StructuredAnalyticalFramework,
)
from src.orchestration.contracts import AnalyticalDimension, FrameworkType
from src.services.selection.nway_pattern_service import NWayPatternService
from src.engine.agents.quality_rater_agent_v2 import get_quality_rater
from src.integrations.llm.unified_client import UnifiedLLMClient

logger = logging.getLogger(__name__)


class AnalysisOrchestrationService:
    """
    Analysis Orchestration Service.

    Coordinates the complete analysis pipeline:
    1. Consultant selection via DispatchOrchestrator
    2. Mental model selection via NWayPatternService
    3. Memory context integration (if available)
    4. LLM generation via UnifiedLLMClient
    5. Quality scoring via CQA Rater
    """

    def __init__(self):
        """Initialize analysis orchestration service with lazy-loaded components"""
        self.dispatcher: Optional[DispatchOrchestrator] = None
        self.pattern_service: Optional[NWayPatternService] = None
        self.quality_rater = None
        self.llm_client: Optional[UnifiedLLMClient] = None
        logger.info("✅ AnalysisOrchestrationService initialized (lazy loading)")

    def _ensure_components_initialized(self):
        """Lazy initialization of analysis components"""
        if self.dispatcher is None:
            self.dispatcher = DispatchOrchestrator()
            self.pattern_service = NWayPatternService()
            self.quality_rater = get_quality_rater()
            self.llm_client = UnifiedLLMClient()
            logger.info("✅ Analysis components initialized")

    async def select_consultants_and_models(
        self,
        query: str,
        system2_tier: Tier,
    ) -> tuple[List[str], List[str]]:
        """
        Select consultants and mental models for analysis.

        Args:
            query: The user query
            system2_tier: System-2 tier classification

        Returns:
            Tuple of (consultant_names, selected_models)
        """
        self._ensure_components_initialized()

        # Create analytical framework
        tier_value = 1 if system2_tier == Tier.TIER_1 else (
            2 if system2_tier == Tier.TIER_2 else 3
        )

        framework = StructuredAnalyticalFramework(
            framework_type=FrameworkType.STRATEGIC_ANALYSIS,
            primary_dimensions=[
                AnalyticalDimension(
                    dimension_name="Strategic Context",
                    key_questions=[query],
                    analysis_approach="comprehensive",
                    priority_level=1
                )
            ],
            secondary_considerations=[f"System-2 Tier: {system2_tier}"],
            analytical_sequence=["context_analysis", "recommendation_generation"],
            complexity_assessment=tier_value,
            recommended_consultant_types=["strategic_analyst", "market_researcher"],
            processing_time_seconds=0.0,
        )

        # Consultant selection
        dispatch_package = await self.dispatcher.run_dispatch(framework, query)
        consultant_names = [c.consultant_id for c in dispatch_package.selected_consultants]

        # Mental model selection
        selected_models = []
        for consultant_blueprint in dispatch_package.selected_consultants:
            consultant_models = await self.pattern_service.get_models_for_consultant(
                consultant_blueprint.consultant_id
            )
            selected_models.extend(consultant_models[:3])  # Top 3 models per consultant

        logger.info(f"✅ Selected {len(consultant_names)} consultants and {len(selected_models)} models")
        return consultant_names, selected_models

    async def generate_analysis_with_memory(
        self,
        context: AnalysisContext
    ) -> str:
        """
        Generate analysis with memory context integration.

        Args:
            context: AnalysisContext with query, tier, consultants, models

        Returns:
            Generated analysis text

        Note:
            This method integrates with ZepMemoryManager if a session_id is
            provided in the context. Memory is decayed using exponential decay
            based on MEMORY_DECAY_HALF_LIFE_DAYS environment variable.
        """
        self._ensure_components_initialized()

        # Get memory summary if session_id provided
        memory_summary = None
        try:
            if context.session_id:
                from src.storage.zep_memory import ZepMemoryManager
                import os

                zmm = ZepMemoryManager()
                half_life = float(os.getenv("MEMORY_DECAY_HALF_LIFE_DAYS", "30"))
                memory_summary = await zmm.summarize_recent_context(
                    context.session_id,
                    max_messages=30,
                    half_life_days=half_life
                )
                logger.debug(f"Memory context retrieved for session {context.session_id}")
        except Exception as e:
            logger.warning(f"Failed to retrieve memory context: {e}")

        # Build analysis prompt
        analysis_prompt = f"""
Context: {context.metadata}
{'Recent Context Summary (decayed):\n' + memory_summary if memory_summary else ''}
Query: {context.query}
System-2 Tier: {context.tier}
Selected Consultants: {context.consultants}
Selected Models: {context.models}

Provide a comprehensive analysis integrating the selected mental models and consultant expertise.
"""

        # Generate analysis
        llm_response = await self.llm_client.generate_analysis(
            prompt=analysis_prompt,
            context={
                "tier": context.tier,
                "consultants": context.consultants,
                "models": context.models
            },
        )

        analysis_text = llm_response.content if hasattr(llm_response, "content") else str(llm_response)
        logger.info(f"✅ Analysis generated ({len(analysis_text)} chars)")

        return analysis_text

    async def analyze_query(
        self,
        query: str,
        config: Dict[str, Any]
    ) -> AnalysisResult:
        """
        Execute complete analysis on query.

        This is the main entry point for end-to-end analysis orchestration.

        Args:
            query: The user query to analyze
            config: Configuration dict with:
                - context: Optional context metadata
                - complexity: Optional complexity hint ("auto", "simple", "strategic", "complex")
                - session_id: Optional session ID for memory integration

        Returns:
            AnalysisResult with content, quality scores, and metadata

        Raises:
            Exception: If analysis fails at any stage
        """
        start_time = time.time()
        trace_id = str(uuid.uuid4())

        try:
            self._ensure_components_initialized()

            # Extract config
            context_metadata = config.get("context", {})
            complexity = config.get("complexity", "auto")
            session_id = context_metadata.get("session_id") if isinstance(context_metadata, dict) else None

            # Step 1: System-2 tier classification (imported service)
            from src.services.application.system2_classification_service import System2ClassificationService
            classifier = System2ClassificationService()
            system2_tier = classifier.classify_tier(query, complexity)

            tier_value = 1 if system2_tier == Tier.TIER_1 else (
                2 if system2_tier == Tier.TIER_2 else 3
            )

            # Step 2: Select consultants and models
            consultant_names, selected_models = await self.select_consultants_and_models(
                query, system2_tier
            )

            # Step 3: Generate analysis with memory context
            analysis_context = AnalysisContext(
                query=query,
                session_id=session_id,
                tier=tier_value,
                consultants=consultant_names,
                models=selected_models,
                metadata=context_metadata,
            )

            analysis_text = await self.generate_analysis_with_memory(analysis_context)

            # Step 4: Quality scoring
            analysis_prompt = f"Query: {query}, Tier: {tier_value}"
            quality_scores = await self.quality_rater.rate_quality(
                analysis_content=analysis_text,
                context={
                    "user_prompt": query,
                    "system_prompt": analysis_prompt,
                    "tier": tier_value,
                },
            )

            # Calculate execution time
            execution_time_ms = (time.time() - start_time) * 1000

            # Create result
            result = AnalysisResult(
                content=analysis_text,
                context=analysis_context,
                quality_scores=quality_scores,
                execution_time_ms=execution_time_ms,
                trace_id=trace_id,
                metadata={
                    "system2_tier": system2_tier.value,
                    "consultant_count": len(consultant_names),
                    "model_count": len(selected_models),
                }
            )

            logger.info(
                f"✅ Analysis complete: trace_id={trace_id}, "
                f"tier={system2_tier.value}, "
                f"time={execution_time_ms:.2f}ms"
            )

            return result

        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}", exc_info=True)
            raise
