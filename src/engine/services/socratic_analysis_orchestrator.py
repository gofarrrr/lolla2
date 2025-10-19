"""
Socratic Analysis Orchestrator Service
=====================================

REFACTORING TARGET: Extract Grade E complexity from API layer
PATTERN: Service Orchestrator with Pipeline Pattern
GOAL: Reduce complete_socratic_analysis() from E (34) to B (≤10)

This service orchestrates the complete Socratic analysis pipeline:
1. Query Enhancement Service
2. Consultant Selection Service
3. Progressive Assembly Pipeline Service
4. Result Processing & Persistence Service

Architecture: Clean service composition with error handling and metrics
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

# Core domain models
from src.engine.engines.core.socratic_cognitive_forge import UserResponse, EnhancedQuery
from src.integrations.llm.unified_client import UnifiedLLMClient
# Migrated to use adapter for dependency inversion
from src.engine.adapters.context_stream import UnifiedContextStream

# Import existing API models for compatibility
from src.engine.api.socratic_forge_api import (
    CompleteAnalysisRequest,
    ConsultantSelectionModel,
    ConsultantAnalysisModel,
    DevilsAdvocateCritiqueModel,
    SeniorAdvisorModel,
)

logger = logging.getLogger(__name__)


@dataclass
class OrchestrationResult:
    """Result of complete Socratic analysis orchestration"""

    success: bool
    engagement_id: str
    enhanced_query: EnhancedQuery
    consultant_selection: Optional[List[ConsultantSelectionModel]] = None
    consultant_analyses: List[ConsultantAnalysisModel] = None
    devils_advocate_critiques: List[DevilsAdvocateCritiqueModel] = None
    senior_advisor_rapporteur: Optional[SeniorAdvisorModel] = None
    pipeline_metrics: Dict[str, Any] = None
    audit_trail: List[Dict[str, Any]] = None
    total_processing_time_ms: int = 0
    error_message: Optional[str] = None


@dataclass
class PipelineStage:
    """Individual pipeline stage with metrics"""

    name: str
    start_time: float
    duration: float
    success: bool
    result_count: int
    tokens_used: int = 0
    error: Optional[str] = None


class SocraticAnalysisOrchestrator:
    """
    Service orchestrator for complete Socratic analysis pipeline

    Responsibilities:
    - Coordinate service composition
    - Manage pipeline execution flow
    - Handle error recovery and metrics
    - Maintain audit trail

    Pattern: Orchestrator with Pipeline Stages
    Complexity Target: Grade B (≤10 per method)
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.llm_client = UnifiedLLMClient()
        from src.engine.adapters.context_stream import get_unified_context_stream  # Migrated
        self.context_stream = get_unified_context_stream()

    async def orchestrate_complete_analysis(
        self, request: CompleteAnalysisRequest
    ) -> OrchestrationResult:
        """
        Main orchestration method - coordinates all pipeline stages

        Complexity: Target B (≤10)
        Pattern: Pipeline coordinator with error handling
        """
        start_time = time.time()
        engagement_id = str(uuid.uuid4())

        try:
            # Stage 1: Query Enhancement
            enhanced_query = await self._execute_query_enhancement_stage(request)

            # Stage 2: Consultant Selection (conditional)
            consultant_selection, selection_result = (
                await self._execute_consultant_selection_stage(request, enhanced_query)
            )

            # Stage 3: Progressive Assembly Pipeline (conditional)
            pipeline_results = await self._execute_progressive_assembly_stage(
                request, enhanced_query, selection_result
            )

            # Stage 4: Result Persistence (conditional)
            await self._execute_persistence_stage(
                request, engagement_id, enhanced_query, pipeline_results
            )

            return self._build_success_result(
                engagement_id,
                enhanced_query,
                consultant_selection,
                pipeline_results,
                start_time,
            )

        except Exception as e:
            return self._build_error_result(engagement_id, start_time, str(e))

    async def _execute_query_enhancement_stage(
        self, request: CompleteAnalysisRequest
    ) -> EnhancedQuery:
        """
        Execute query enhancement stage

        Complexity: Target B (≤10)
        Responsibility: Single stage execution with error handling
        """
        from src.engine.api.socratic_forge_api import get_forge

        stage_start = time.time()
        forge = get_forge()

        # Convert request responses to domain objects
        user_responses = [
            UserResponse(
                question_id=r.question_id, answer=r.answer, confidence=r.confidence
            )
            for r in request.user_responses
        ]

        enhanced_query = await forge.forge_enhanced_query(
            original_statement=request.problem_statement,
            user_responses=user_responses,
            context=request.context,
        )

        stage_duration = time.time() - stage_start
        self.logger.info(f"✅ Query enhancement completed in {stage_duration:.2f}s")

        return enhanced_query

    async def _execute_consultant_selection_stage(
        self, request: CompleteAnalysisRequest, enhanced_query: EnhancedQuery
    ) -> Tuple[Optional[List[ConsultantSelectionModel]], Any]:
        """
        Execute consultant selection stage

        Complexity: Target B (≤10)
        Responsibility: Conditional consultant selection with error handling
        """
        if not request.request_consultant_selection:
            return None, None

        from src.engine.api.socratic_forge_api import get_forge

        stage_start = time.time()
        forge = get_forge()

        self.logger.info("🧠 Executing consultant selection stage")

        selection_result, _ = await forge.integrate_with_consultant_engine(
            enhanced_query
        )

        # Convert to API models for compatibility
        consultant_selection = [
            ConsultantSelectionModel(
                consultant_id=c.consultant_id,
                name=c.blueprint.name,
                specialization=c.blueprint.specialization,
                selection_reason=c.selection_reason,
                confidence_score=getattr(c, "confidence_score", 0.8),
            )
            for c in selection_result.selected_consultants
        ]

        stage_duration = time.time() - stage_start
        self.logger.info(f"✅ Consultant selection completed in {stage_duration:.2f}s")

        return consultant_selection, selection_result

    async def _execute_progressive_assembly_stage(
        self,
        request: CompleteAnalysisRequest,
        enhanced_query: EnhancedQuery,
        selection_result: Any,
    ) -> Dict[str, Any]:
        """
        Execute progressive assembly pipeline stage

        Complexity: Target B (≤10)
        Responsibility: Coordinate pipeline execution with error handling
        """
        if not (request.run_full_pipeline and selection_result):
            return {
                "consultant_analyses": [],
                "devils_advocate_critiques": [],
                "senior_advisor_rapporteur": None,
                "pipeline_metrics": {},
            }

        stage_start = time.time()
        self.logger.info("🔥 Executing progressive assembly pipeline stage")

        # Sub-stage 1: Parallel Consultant Analyses
        consultant_analyses = await self._execute_consultant_analyses(
            selection_result, enhanced_query
        )

        # Sub-stage 2: Parallel Devil's Advocate Critiques
        devils_advocate_critiques = await self._execute_devils_advocate_critiques(
            consultant_analyses, enhanced_query
        )

        # Sub-stage 3: Senior Advisor Rapporteur
        senior_advisor_rapporteur = await self._execute_senior_advisor_rapporteur(
            consultant_analyses, devils_advocate_critiques, enhanced_query
        )

        stage_duration = time.time() - stage_start

        pipeline_metrics = self._calculate_pipeline_metrics(
            consultant_analyses,
            devils_advocate_critiques,
            senior_advisor_rapporteur,
            stage_duration,
        )

        self.logger.info(
            f"✅ Progressive assembly pipeline completed in {stage_duration:.2f}s"
        )

        return {
            "consultant_analyses": consultant_analyses,
            "devils_advocate_critiques": devils_advocate_critiques,
            "senior_advisor_rapporteur": senior_advisor_rapporteur,
            "pipeline_metrics": pipeline_metrics,
        }

    async def _execute_consultant_analyses(
        self, selection_result: Any, enhanced_query: EnhancedQuery
    ) -> List[ConsultantAnalysisModel]:
        """
        Execute parallel consultant analyses

        Complexity: Target B (≤10)
        Responsibility: Coordinate parallel analysis execution
        """
        from src.engine.api.socratic_forge_api import analyze_consultant_progressive

        # Prepare consultant tasks
        analysis_tasks = []
        for consultant in selection_result.selected_consultants:
            mock_consultant = {
                "consultant_id": consultant.consultant_id,
                "persona_prompt": f"You are a {consultant.consultant_id} consultant. Provide strategic analysis from your specialized perspective.",
            }

            task = analyze_consultant_progressive(
                self.llm_client,
                self.context_stream,
                mock_consultant,
                enhanced_query.enhanced_statement,
                selected_nway_clusters=selection_result.selected_nway_clusters,
            )
            analysis_tasks.append(task)

        # Execute parallel analyses
        analysis_results = await asyncio.gather(*analysis_tasks, return_exceptions=True)

        # Process results into API models
        consultant_analyses = []
        for result in analysis_results:
            if isinstance(result, Exception):
                self.logger.error(f"Analysis task failed: {result}")
            elif result and result.get("success", False):
                analysis_model = ConsultantAnalysisModel(
                    consultant_id=result["consultant_id"],
                    analysis=result["analysis"],
                    tokens_used=result["tokens_used"],
                    processing_time_seconds=result["duration"],
                )
                consultant_analyses.append(analysis_model)

        self.logger.info(f"✅ Completed {len(consultant_analyses)} consultant analyses")
        return consultant_analyses

    async def _execute_devils_advocate_critiques(
        self,
        consultant_analyses: List[ConsultantAnalysisModel],
        enhanced_query: EnhancedQuery,
    ) -> List[DevilsAdvocateCritiqueModel]:
        """
        Execute parallel Devil's Advocate critiques

        Complexity: Target B (≤10)
        Responsibility: Coordinate parallel critique execution
        """
        if not consultant_analyses:
            return []

        from src.engine.api.socratic_forge_api import critique_analysis_progressive

        # Prepare critique tasks
        critique_tasks = []
        for analysis in consultant_analyses:
            # Convert back to result format for compatibility
            analysis_result = {
                "consultant_id": analysis.consultant_id,
                "analysis": analysis.analysis,
                "success": True,
            }

            task = critique_analysis_progressive(
                self.llm_client,
                self.context_stream,
                analysis_result,
                enhanced_query.enhanced_statement,
            )
            critique_tasks.append(task)

        # Execute parallel critiques
        critique_results = await asyncio.gather(*critique_tasks, return_exceptions=True)

        # Process results into API models
        devils_advocate_critiques = []
        for result in critique_results:
            if isinstance(result, Exception):
                self.logger.error(f"Critique task failed: {result}")
            elif result and result.get("success", False):
                critique_model = DevilsAdvocateCritiqueModel(
                    consultant_id=result["consultant_id"],
                    critique=result["critique"],
                    tokens_used=result["tokens_used"],
                    processing_time_seconds=result["duration"],
                )
                devils_advocate_critiques.append(critique_model)

        self.logger.info(
            f"✅ Completed {len(devils_advocate_critiques)} devil's advocate critiques"
        )
        return devils_advocate_critiques

    async def _execute_senior_advisor_rapporteur(
        self,
        consultant_analyses: List[ConsultantAnalysisModel],
        devils_advocate_critiques: List[DevilsAdvocateCritiqueModel],
        enhanced_query: EnhancedQuery,
    ) -> Optional[SeniorAdvisorModel]:
        """
        Execute Senior Advisor rapporteur

        Complexity: Target B (≤10)
        Responsibility: Execute final synthesis stage
        """
        if not (consultant_analyses and devils_advocate_critiques):
            return None

        from src.engine.api.socratic_forge_api import (
            senior_advisor_rapporteur_progressive,
        )

        # Prepare cognitive outputs
        cognitive_outputs = {
            "analyses": [
                {
                    "consultant_id": a.consultant_id,
                    "analysis": a.analysis,
                    "success": True,
                }
                for a in consultant_analyses
            ],
            "critiques": [
                {
                    "consultant_id": c.consultant_id,
                    "critique": c.critique,
                    "success": True,
                }
                for c in devils_advocate_critiques
            ],
        }

        rapporteur_result = await senior_advisor_rapporteur_progressive(
            self.llm_client,
            self.context_stream,
            cognitive_outputs,
            enhanced_query.enhanced_statement,
        )

        if rapporteur_result and rapporteur_result.get("success", False):
            senior_advisor_model = SeniorAdvisorModel(
                rapporteur_analysis=rapporteur_result["rapporteur_analysis"],
                tokens_used=rapporteur_result["tokens_used"],
                context_preservation_score=1.0,
                processing_time_seconds=rapporteur_result["duration"],
            )

            self.logger.info("✅ Senior advisor rapporteur completed")
            return senior_advisor_model

        return None

    async def _execute_persistence_stage(
        self,
        request: CompleteAnalysisRequest,
        engagement_id: str,
        enhanced_query: EnhancedQuery,
        pipeline_results: Dict[str, Any],
    ) -> None:
        """
        Execute result persistence stage

        Complexity: Target B (≤10)
        Responsibility: Conditional persistence with error handling
        """
        if not (
            request.run_full_pipeline and pipeline_results.get("consultant_analyses")
        ):
            return

        from src.engine.api.socratic_forge_api import _persist_complete_analysis_results

        try:
            await _persist_complete_analysis_results(
                engagement_id,
                request.problem_statement,
                enhanced_query,
                pipeline_results["consultant_analyses"],
                pipeline_results["devils_advocate_critiques"],
                pipeline_results["senior_advisor_rapporteur"],
            )

            self.logger.info(f"✅ Results persisted for engagement {engagement_id}")

        except Exception as e:
            self.logger.error(f"⚠️ Persistence failed: {e}")
            # Continue - persistence failure shouldn't break the response

    def _calculate_pipeline_metrics(
        self,
        consultant_analyses: List[ConsultantAnalysisModel],
        devils_advocate_critiques: List[DevilsAdvocateCritiqueModel],
        senior_advisor_rapporteur: Optional[SeniorAdvisorModel],
        total_duration: float,
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive pipeline metrics

        Complexity: Target B (≤10)
        Responsibility: Metrics calculation and aggregation
        """
        return {
            "total_pipeline_time_seconds": total_duration,
            "phases_completed": (
                3
                if senior_advisor_rapporteur
                else 2 if devils_advocate_critiques else 1
            ),
            "total_analyses": len(consultant_analyses),
            "total_critiques": len(devils_advocate_critiques),
            "senior_advisor_completed": senior_advisor_rapporteur is not None,
            "total_pipeline_tokens": (
                sum(a.tokens_used for a in consultant_analyses)
                + sum(c.tokens_used for c in devils_advocate_critiques)
                + (
                    senior_advisor_rapporteur.tokens_used
                    if senior_advisor_rapporteur
                    else 0
                )
            ),
        }

    def _build_success_result(
        self,
        engagement_id: str,
        enhanced_query: EnhancedQuery,
        consultant_selection: Optional[List[ConsultantSelectionModel]],
        pipeline_results: Dict[str, Any],
        start_time: float,
    ) -> OrchestrationResult:
        """
        Build successful orchestration result

        Complexity: Target B (≤10)
        Responsibility: Result object construction
        """
        return OrchestrationResult(
            success=True,
            engagement_id=engagement_id,
            enhanced_query=enhanced_query,
            consultant_selection=consultant_selection,
            consultant_analyses=pipeline_results.get("consultant_analyses", []),
            devils_advocate_critiques=pipeline_results.get(
                "devils_advocate_critiques", []
            ),
            senior_advisor_rapporteur=pipeline_results.get("senior_advisor_rapporteur"),
            pipeline_metrics=pipeline_results.get("pipeline_metrics", {}),
            audit_trail=[],  # TODO: Implement audit trail collection
            total_processing_time_ms=int((time.time() - start_time) * 1000),
        )

    def _build_error_result(
        self, engagement_id: str, start_time: float, error_message: str
    ) -> OrchestrationResult:
        """
        Build error orchestration result

        Complexity: Target B (≤10)
        Responsibility: Error result object construction
        """
        return OrchestrationResult(
            success=False,
            engagement_id=engagement_id,
            enhanced_query=None,
            error_message=error_message,
            total_processing_time_ms=int((time.time() - start_time) * 1000),
        )


# Singleton instance for injection
_orchestrator_instance = None


def get_socratic_analysis_orchestrator() -> SocraticAnalysisOrchestrator:
    """Factory function for dependency injection"""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = SocraticAnalysisOrchestrator()
    return _orchestrator_instance
