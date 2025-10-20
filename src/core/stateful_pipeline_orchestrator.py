"""
Iteration Engine - Stateful Pipeline Orchestrator
=================================================

ARCHITECTURAL BREAKTHROUGH: Transform rigid 7-stage pipeline into iterative, user-controlled system

This orchestrator implements the core "Execute -> Save Checkpoint -> Check for Revision -> Repeat" loop
that enables true cognitive partnership through iterative refinement.

Key Features:
- Checkpoint after every stage with lightweight state storage
- Resume from any checkpoint with complete context reconstruction
- Immutable Analysis Branching (forking) for revisions
- Separation of State (checkpoints) from Log (UnifiedContextStream)
"""

import asyncio
import hashlib
import re
import time
import logging
import os
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from uuid import UUID, uuid4
from types import SimpleNamespace

from .checkpoint_models import (
    StateCheckpoint,
    PipelineStage,
    AnalysisRevision,
)
from .stage_keys import StageKey
from .unified_context_stream import (
    ContextEventType,
    get_unified_context_stream,
)
from .exceptions import CheckpointError, RevisionError, PipelineError

# Import existing orchestrators and agents (REFACTORED - Operation All Green Phase 2)
from src.orchestration.dispatch_orchestrator import DispatchOrchestrator
from src.engine.agents.problem_structuring_agent import ProblemStructuringAgent
from src.orchestration.senior_advisor_orchestrator import SeniorAdvisorOrchestrator
from src.core.enhanced_devils_advocate_system import EnhancedDevilsAdvocateSystem
from src.core.progressive_question_engine import ProgressiveQuestionEngine
from src.rag.project_rag_pipeline import get_project_rag_pipeline
from src.integrations.llm.unified_client import UnifiedLLMClient
from src.integrations.llm.noop_client import NoopLLMClient

# OPERATION IGNITION: Activate context management infrastructure
from src.engine.core.context_manager_activator import create_context_manager_activator

# Operation Genesis: Import new stage executors
from src.core.stage_executors.hybrid_data_research_executor import HybridDataResearchExecutor
# from src.core.stage_executors.arbitration_stage_executor import ArbitrationStageExecutor  # TODO: Fix import dependencies

# Operation Unification: Import all V6 stage executors (REFACTORED - Operation All Green Phase 1)
from src.core.stage_executors.socratic_executor import SocraticStageExecutor
from src.core.stage_executors.problem_structuring_executor import ProblemStructuringStageExecutor
from src.core.stage_executors.interaction_sweep_executor import InteractionSweepStageExecutor
from src.core.stage_executors.consultant_selection_executor import ConsultantSelectionStageExecutor
from src.core.stage_executors.synergy_prompting_executor import SynergyPromptingStageExecutor
from src.core.stage_executors.parallel_analysis_executor import ParallelAnalysisStageExecutor
from src.core.stage_executors.devils_advocate_executor import DevilsAdvocateStageExecutor
from src.core.stage_executors.senior_advisor_executor import SeniorAdvisorStageExecutor
from src.core.pipeline_contracts import PipelineState

logger = logging.getLogger(__name__)


class StatefulPipelineOrchestrator:
    """
    Revolutionary stateful orchestrator that enables iterative cognitive partnership.

    ARCHITECTURAL PRINCIPLE: Every stage creates a checkpoint. Users can revise from any checkpoint
    by creating immutable analysis branches. The original analysis remains untouched.
    """

    # Compatibility: critical stages used by tests
    CRITICAL_STAGES = {
        PipelineStage.PARALLEL_ANALYSIS,
        PipelineStage.DEVILS_ADVOCATE,
        PipelineStage.SENIOR_ADVISOR,
    }

    def __init__(self, checkpoint_service=None, status_callback=None, context_stream=None):
        # Allow injection of a preconfigured context stream (testability)
        self.context_stream = context_stream or get_unified_context_stream()
        self.status_callback = status_callback  # Optional callback for real-time frontend updates
        logger.info(f"🔍 DEBUG: Orchestrator initialized with status_callback={'SET' if status_callback else 'NONE'}")

        # Fast/offline mode for tests
        self._fast_mode = os.getenv("TEST_FAST") == "1"

        # MANUS/COGNITION.AI INFRASTRUCTURE INTEGRATION - Wire existing world-class infrastructure
        from src.engine.core.context_engineering_optimizer import (
            ContextEngineeringOptimizer,
        )
        from src.engine.core.kv_cache_optimizer import KVCacheOptimizer
        from src.engine.config import CognitiveEngineSettings

        self.context_optimizer = (
            ContextEngineeringOptimizer()
        )  # Manus: Append-only + recitation

        # Initialize KV Cache optimizer with settings
        settings = CognitiveEngineSettings()
        self.kv_cache_optimizer = KVCacheOptimizer(
            settings
        )  # Manus: Stable prefix optimization

        # OPERATION IGNITION: Activate context management with existing infrastructure
        self.context_manager = create_context_manager_activator(
            context_optimizer=self.context_optimizer,
            kv_cache_optimizer=self.kv_cache_optimizer
        )
        logger.info("🚀 OPERATION IGNITION: Context Manager Activator initialized (video synthesis)")

        # Initialize ContextAuditor (pre/post stage hygiene and integrity checks)
        try:
            from src.core.context_auditor import ContextAuditor
            self.context_auditor = ContextAuditor(self.context_stream)
            logger.info("🧭 ContextAuditor initialized")
        except Exception as _e:
            self.context_auditor = None
            logger.warning(f"⚠️ ContextAuditor unavailable: {_e}")

        # Initialize existing orchestrators/agents
        self.progressive_question_engine = (
            ProgressiveQuestionEngine()
        )  # TOSCA-enabled question generation
        self.problem_structuring_agent = ProblemStructuringAgent()
        self.dispatch_orchestrator = DispatchOrchestrator()
        self.devils_advocate = EnhancedDevilsAdvocateSystem()
        self.senior_advisor = SeniorAdvisorOrchestrator()

        # OPERATION POLISH: Initialize LLM client for parallel analysis executor
        if self._fast_mode:
            self.llm_client = NoopLLMClient()
            logger.info("🧪 TEST_FAST=1: Using NoopLLMClient (offline, deterministic)")
        else:
            self.llm_client = UnifiedLLMClient()

        # OPERATION UNIFICATION: Initialize V6 Stage Executors (executor pattern)
        # These executors replace the old _execute_* methods with clean, testable interfaces
        self.stage_executors = {
            PipelineStage.SOCRATIC_QUESTIONS: SocraticStageExecutor(
                self.progressive_question_engine, self.context_stream
            ),
            PipelineStage.PROBLEM_STRUCTURING: ProblemStructuringStageExecutor(
                self.problem_structuring_agent, self.context_stream
            ),
            PipelineStage.INTERACTION_SWEEP: InteractionSweepStageExecutor(
                self.context_stream
            ),
            PipelineStage.CONSULTANT_SELECTION: ConsultantSelectionStageExecutor(
                self.dispatch_orchestrator, self.context_stream
            ),
            PipelineStage.SYNERGY_PROMPTING: SynergyPromptingStageExecutor(
                self.context_stream
            ),
            PipelineStage.PARALLEL_ANALYSIS: ParallelAnalysisStageExecutor(
                llm_client_getter=lambda: self.llm_client,
                context_stream=self.context_stream
            ),
            PipelineStage.DEVILS_ADVOCATE: DevilsAdvocateStageExecutor(
                self.devils_advocate, self.context_stream
            ),
            PipelineStage.SENIOR_ADVISOR: SeniorAdvisorStageExecutor(
                self.senior_advisor, self.context_stream
            ),
        }

        # Checkpoint management via centralized service
        if checkpoint_service is not None:
            self.checkpoint_service = checkpoint_service
        elif self._fast_mode:
            # Lightweight no-op facade to satisfy interfaces without DB
            from uuid import uuid4 as _uuid4
            class _NoopCheckpointService:
                async def save_checkpoint(self, **kwargs):
                    try:
                        from src.core.checkpoint_models import StateCheckpoint, PipelineStage
                        return StateCheckpoint(
                            trace_id=kwargs.get("trace_id"),
                            stage_completed=kwargs.get("stage_completed"),
                            stage_output=kwargs.get("stage_output") or {},
                            user_id=kwargs.get("user_id"),
                            session_id=kwargs.get("session_id"),
                        )
                    except Exception:
                        return None
                async def resume_from_checkpoint(self, *args, **kwargs):
                    return {}
                async def load_checkpoint(self, *args, **kwargs):
                    return None
                async def load_checkpoints_for_trace(self, *args, **kwargs):
                    return []
                async def count_checkpoints(self, *args, **kwargs):
                    return 0
                async def create_revision_branch(self, **kwargs):
                    return _uuid4()
            self.checkpoint_service = _NoopCheckpointService()
            logger.info("🧪 TEST_FAST=1: Using Noop checkpoint service (in-memory, non-persistent)")
        else:
            try:
                from src.services.orchestration_infra.supabase_checkpoint_repository import SupabaseCheckpointRepository
                from src.services.orchestration_infra.revision_service import V1RevisionService
                from src.core.checkpoint_service import CheckpointService
                from src.services.persistence.database_service import DatabaseService
                db_service = DatabaseService()
                repo = SupabaseCheckpointRepository(db_service, self.context_stream)
                rev = V1RevisionService(repo, self.context_stream)
                self.checkpoint_service = CheckpointService(checkpoint_repo=repo, revision_service=rev)
            except Exception as e:
                raise RuntimeError(f"Failed to initialize default checkpoint service: {e}")

        # OPERATION SCALPEL V2 - Phase 1.3: Initialize persistence service (self-contained)
        if not self._fast_mode:
            from src.services.orchestration_infra.persistence_orchestration_service import (
                PersistenceOrchestrationService
            )
            from src.services.persistence.database_service import DatabaseService
            self.persistence_service = PersistenceOrchestrationService(db_service=DatabaseService())
        else:
            self.persistence_service = None

        # OPERATION SCALPEL V2 - Phase 2.3: Initialize sanitization service (self-contained)
        from src.services.orchestration_infra.glass_box_sanitization_service import (
            GlassBoxSanitizationService
        )
        self.sanitization_service = GlassBoxSanitizationService()

        # OPERATION SCALPEL V2 - Phase 3.3: Initialize context engineering service (self-contained)
        from src.services.orchestration_infra.context_engineering_service import (
            ContextEngineeringService
        )
        self.context_engineering_service = ContextEngineeringService(
            context_optimizer=self.context_optimizer,
            kv_cache_optimizer=self.kv_cache_optimizer,
            context_stream=self.context_stream
        )

        # OPERATION SCALPEL V2 - Phase 5: Initialize error handling service
        from src.services.orchestration_infra.stage_error_handling_service import (
            StageErrorHandlingService
        )
        self.error_handling_service = StageErrorHandlingService(context_stream=self.context_stream)

    # OPERATION SCALPEL V2 - Phase 2.3: Internal sanitization helpers deleted - logic now in GlassBoxSanitizationService

    # Legacy compatibility seam for existing tests
    # OPERATION BEDROCK - FINAL CUT: _execute_stage_legacy_fallback() DELETED
    # This method was a temporary bridge to the old _execute_stage() method.
    # All execution now flows through the new DefaultPipelineManager architecture.

    async def execute_pipeline(
        self,
        trace_id: Optional[UUID] = None,
        initial_query: Optional[str] = None,
        resume_from_checkpoint: Optional[UUID] = None,
        user_id: Optional[UUID] = None,
        session_id: Optional[UUID] = None,
        merge_project_context: Optional[bool] = False,
        project_id: Optional[UUID] = None,
        enhancement_context: Optional[Dict[str, Any]] = None,
        interactive_mode: Optional[bool] = False,
    ) -> Dict[str, Any]:
        """
        Thin façade: delegate execution to the new orchestration engine (PipelineManager).

        Public signature preserved for full backward compatibility.
        """
        try:
            start_time = time.time()

            # Resolve trace id
            trace_id = trace_id or uuid4()

            # Build initial static context seed that previous implementation expected
            initial_seed: Dict[str, Any] = {
                "trace_id": str(trace_id),
                "initial_query": initial_query or "",
                "user_id": str(user_id) if user_id else None,
                "session_id": str(session_id) if session_id else None,
                "interactive_mode": bool(interactive_mode),
            }
            if enhancement_context:
                initial_seed["enhancement_context"] = enhancement_context

            # OPERATION BEDROCK - FINAL CUT: Use extracted components
            from src.orchestration.flow_contracts import RunId
            from src.orchestration.v6_executor_adapter import V6ExecutorAdapter
            from src.orchestration.stage_plan_builder import StagePlanBuilder
            from src.orchestration.checkpoint_service import InMemoryCheckpointService
            from src.orchestration.context_orchestrator import DefaultContextOrchestrator
            from src.orchestration.pipeline_manager import DefaultPipelineManager

            # Instantiate orchestration services
            cp = InMemoryCheckpointService()
            cx = DefaultContextOrchestrator(static_base=initial_seed)

            # Build executor registry with V6 adapters (bridges V6 executors to flow contracts)
            registry = {
                f"exec::{stage_enum.value}": V6ExecutorAdapter(executor)
                for stage_enum, executor in self.stage_executors.items()
            }

            # Build execution plan using factory
            # INTERACTIVE MODE: Run only Socratic Questions stage first, then pause
            from src.core.checkpoint_models import PipelineStage

            # Determine which stages to run based on interactive mode and checkpoint
            current_pause_point = None
            if interactive_mode and not resume_from_checkpoint:
                # First pause: Run SOCRATIC_QUESTIONS only
                plan = StagePlanBuilder.build_custom_plan([PipelineStage.SOCRATIC_QUESTIONS])
                current_pause_point = "socratic_questions"
                logger.info("🔄 Interactive mode: Running SOCRATIC_QUESTIONS stage only (Pause Point 1)")
            elif interactive_mode and resume_from_checkpoint:
                # Check checkpoint metadata to determine which pause point we're resuming from
                # If resuming from Socratic Questions pause, run Problem Structuring only (second pause)
                # If resuming from MECE validation pause, run remaining 6 stages

                # For now, we'll use a simple heuristic based on context
                # TODO: Store pause_point in checkpoint metadata for cleaner resume logic

                # Get checkpoint data to determine what was completed
                checkpoint_data = enhancement_context or {}
                has_socratic_answers = bool(checkpoint_data.get('answered_questions'))
                has_mece_approval = bool(checkpoint_data.get('mece_approved'))

                if has_socratic_answers and not has_mece_approval:
                    # Resume from first pause → run Problem Structuring only (second pause)
                    plan = StagePlanBuilder.build_custom_plan([
                        PipelineStage.PROBLEM_STRUCTURING
                    ])
                    current_pause_point = "problem_structuring"
                    logger.info("🔄 Interactive mode RESUME: Running PROBLEM_STRUCTURING only (Pause Point 2)")
                else:
                    # Resume from second pause → run remaining 6 stages
                    remaining_stages = [
                        PipelineStage.ORACLE_RESEARCH,
                        PipelineStage.CONSULTANT_SELECTION,
                        PipelineStage.SYNERGY_PROMPTING,
                        PipelineStage.PARALLEL_ANALYSIS,
                        PipelineStage.DEVILS_ADVOCATE,
                        PipelineStage.SENIOR_ADVISOR,
                    ]
                    plan = StagePlanBuilder.build_custom_plan(remaining_stages)
                    current_pause_point = None  # No more pauses
                    logger.info("🔄 Interactive mode RESUME: Running remaining 6 stages (Final run)")
            else:
                plan = StagePlanBuilder.build_default_plan()
                current_pause_point = None

            # Create pipeline manager
            mgr = DefaultPipelineManager(
                checkpoint_service=cp,
                context_orchestrator=cx,
                executor_registry=registry,
            )

            # Run the plan (resume flag propagates intent, but we don't implement resume here)
            run_id = RunId(str(trace_id))
            results = await mgr.run(plan, run_id, resume=bool(resume_from_checkpoint))

            # Aggregate final context deterministically (same merge strategy as DefaultContextOrchestrator)
            final_context: Dict[str, Any] = dict(initial_seed)
            for spec in plan:
                res = results.get(spec.id)
                if not res:
                    continue
                for k, v in (res.output or {}).items():
                    if not str(k).startswith("_"):
                        final_context[k] = v

            processing_time = time.time() - start_time

            # Compute metrics for compatibility
            try:
                total_checkpoints = sum(1 for (rid, _sid) in getattr(cp, "_store", {}).keys() if rid == run_id.value)
            except Exception:
                total_checkpoints = 0

            orthogonality_index = 0.0
            try:
                orthogonality_index = (
                    final_context.get("parallel_analysis", {}).get("orthogonality_index", 0.0)
                )
            except Exception:
                pass

            # INTERACTIVE MODE PAUSE POINTS
            # Pause Point 1: After Socratic Questions (for user answers)
            if interactive_mode and current_pause_point == "socratic_questions":
                # Extract Socratic Questions results from final_context
                # Try both possible keys: socratic_results (v6) and socratic_questions (adapter output)
                logger.info(f"🔍 DEBUG: final_context keys = {list(final_context.keys())}")
                socratic_results = final_context.get("socratic_results") or final_context.get("socratic_questions")
                logger.info(f"🔍 DEBUG: socratic_results = {socratic_results}")
                if socratic_results:
                    logger.info(f"⏸️ PAUSE POINT 1: After SOCRATIC_QUESTIONS - awaiting user answers")

                    # Extract questions for frontend
                    questions = []
                    # Handle both dict and object formats
                    if isinstance(socratic_results, dict):
                        # Dict format from adapter
                        questions = socratic_results.get('key_strategic_questions', [])
                    elif hasattr(socratic_results, 'key_strategic_questions'):
                        # Object format
                        questions = [
                            {
                                "id": q.id,
                                "question": q.question,
                                "category": q.category,
                                "tier": q.tier,
                                "tosca_tag": q.tosca_tag,
                                "auto_generated": q.auto_generated,
                            }
                            for q in socratic_results.key_strategic_questions
                        ]

                    # Extract TOSCA coverage based on format
                    if isinstance(socratic_results, dict):
                        tosca_coverage = socratic_results.get('tosca_coverage', {})
                        missing_tosca = socratic_results.get('missing_tosca_elements', [])
                    else:
                        tosca_coverage = getattr(socratic_results, 'tosca_coverage', {})
                        missing_tosca = getattr(socratic_results, 'missing_tosca_elements', [])

                    return {
                        "status": "paused_for_user_input",
                        "pause_point": "socratic_questions",
                        "trace_id": trace_id,
                        "socratic_questions": {
                            "key_strategic_questions": questions,
                            "tosca_coverage": tosca_coverage,
                            "missing_tosca_elements": missing_tosca,
                        },
                        "questions": questions,  # Backward compatibility
                        "current_checkpoint": str(trace_id),  # Use trace_id as checkpoint for simplicity
                        "processing_time_seconds": processing_time,
                    }
                else:
                    logger.warning("⚠️ Interactive mode but no socratic_results found - continuing normally")

            # Pause Point 2: After Problem Structuring (for MECE validation)
            elif interactive_mode and current_pause_point == "problem_structuring":
                # Extract Problem Structuring results and build MECE review
                problem_structure = final_context.get("problem_structure")
                if problem_structure:
                    logger.info(f"⏸️ PAUSE POINT 2: After PROBLEM_STRUCTURING - awaiting MECE validation")

                    # Import and use MECE validation module
                    from src.core.mece_validation import build_mece_review
                    from types import SimpleNamespace

                    # Build state object for mece_validation.build_mece_review()
                    state = SimpleNamespace(
                        problem_structure=problem_structure,
                        research_gaps=final_context.get("research_gaps", []),
                        socratic_results=final_context.get("socratic_results")
                    )

                    mece_review = build_mece_review(state)

                    return {
                        "status": "paused_for_mece_validation",
                        "pause_point": "problem_structuring",
                        "trace_id": trace_id,
                        "mece_review": mece_review,
                        "current_checkpoint": str(trace_id),
                        "processing_time_seconds": processing_time,
                    }
                else:
                    logger.warning("⚠️ Interactive mode but no problem_structure found - continuing normally")

            # OPERATION CRYSTAL CLEAR: Warm-on-Complete (build Report v2 bundle and cache)
            try:
                import asyncio as _asyncio
                async def _warm_bundle(_trace_id):
                    from src.services.report_reconstruction_service import ReportReconstructionService
                    from src.services.persistence.database_service import DatabaseService
                    from src.services.report_cache import set_bundle
                    svc = ReportReconstructionService(DatabaseService())
                    bundle = svc.reconstruct_bundle(str(_trace_id))
                    etag = 'W/"' + svc._hash_etag(bundle) + '"'
                    set_bundle(str(_trace_id), bundle, etag)
                _asyncio.create_task(_warm_bundle(trace_id))
                logger.info("🧠 Warm-on-Complete: report bundle warm task scheduled")
            except Exception as _e:
                logger.warning(f"⚠️ Warm-on-Complete scheduling failed: {_e}")

            return {
                "status": "completed",
                "trace_id": trace_id,
                "final_result": final_context,
                "processing_time_seconds": processing_time,
                "total_checkpoints": total_checkpoints,
                "orthogonality_index": orthogonality_index,
                "stage_profiles": [],
                "total_runtime_tokens": 0,
                "runtime_warning": None,
            }

        except Exception as e:
            logger.error(f"❌ Pipeline execution failed: {e}")
            raise PipelineError(f"Pipeline execution failed: {e}")

    async def create_revision_branch(
        self,
        parent_trace_id: UUID,
        checkpoint_id: UUID,
        revision_data: Dict[str, Any],
        revision_rationale: Optional[str] = None,
        user_id: Optional[UUID] = None,
        session_id: Optional[UUID] = None,
    ) -> UUID:
        """
        Create a new analysis branch (fork) from an existing checkpoint.

        IMMUTABLE ANALYSIS PRINCIPLE: The original analysis remains unchanged.
        A new trace_id is created for the revised analysis.

        Args:
            parent_trace_id: Original analysis trace_id
            checkpoint_id: Checkpoint to fork from
            revision_data: New user inputs for the stage
            revision_rationale: User's explanation for the revision
            user_id: User identifier
            session_id: Session identifier

        Returns:
            New trace_id for the revised analysis
        """
        try:
            # Create revision via centralized service
            child_trace_id = await self.checkpoint_service.create_revision_branch(
                parent_trace_id=parent_trace_id,
                checkpoint_id=checkpoint_id,
                revision_data=revision_data,
                revision_rationale=revision_rationale,
                user_id=user_id,
                session_id=session_id,
            )

            # Load source checkpoint for follow-up execution context
            source_checkpoint = await self.checkpoint_service.load_checkpoint(checkpoint_id)

            # Create new context stream for the child analysis
            if source_checkpoint is not None:
                await self._create_child_context_stream(
                    parent_trace_id=parent_trace_id,
                    child_trace_id=child_trace_id,
                    source_checkpoint=source_checkpoint,
                    revision_data=revision_data,
                )

                # Start background execution of revised analysis
                asyncio.create_task(
                    self._execute_revision_pipeline(
                        child_trace_id=child_trace_id,
                        source_checkpoint=source_checkpoint,
                        revision_data=revision_data,
                        revision=None,
                        user_id=user_id,
                        session_id=session_id,
                    )
                )

            logger.info(f"🚀 Revision branch started: {child_trace_id}")
            return child_trace_id

        except Exception as e:
            logger.error(f"❌ Failed to create revision branch: {e}")
            raise RevisionError(f"Failed to create revision branch: {e}")

    # OPERATION BEDROCK - FINAL CUT: Context conversion methods deleted
    # These methods (_context_to_pipeline_state, _pipeline_state_to_context) have been
    # moved to src/orchestration/v6_executor_adapter.py for cleaner separation of concerns.



    # OPERATION BEDROCK - FINAL CUT: All legacy execution methods deleted
    # The following methods have been removed as they are no longer used:
    # - _execute_stage() - Replaced by V6 executors + DefaultPipelineManager
    # - _execute_oracle_research() - Now handled by HybridDataResearchExecutor
    # - _execute_arbitration_capture() - Legacy method
    # - _merge_stage_result() - No longer needed with new architecture
    # - _summarize_stage_result() - No longer needed
    # - _categorize_questions() - Unused helper
    # - _categorize_progressive_questions() - Unused helper
    # - _extract_critical_decisions() - Unused helper
    # - _extract_next_actions() - Unused helper
    # - _extract_open_questions() - Unused helper

    def get_context_metrics(self) -> Dict:
        """OPERATION IGNITION: Get context management metrics for monitoring"""
        try:
            return self.context_manager.get_metrics()
        except Exception as e:
            logger.warning(f"⚠️ Failed to get context metrics: {e}")
            return {}


# OPERATION SCALPEL V2 - Phase 1.3: Internal helpers deleted - logic now in PersistenceOrchestrationService
# Embedded storage and duplicate exceptions removed — persistence is centralized via
# CheckpointService + repository implementation (see src/services/orchestration_infra).
