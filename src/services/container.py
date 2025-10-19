# src/services/container.py
import logging
from .analysis.contracts import (
    IPromptBuilder,
    IConsultantRunner,
    IResultAggregator,
    IEvidenceEmitter,
)
from .analysis.facade_implementations import (
    V1PromptBuilder,
    V1ConsultantRunner,
    V1ResultAggregator,
    V1EvidenceEmitter,
)
from src.core.unified_context_stream import get_unified_context_stream
from src.core.persistence.adapters import SupabaseAdapter, FileAdapter
from src.services.selection.scorer import ChemistryScorer
from src.services.selection.optimizer import ChemistryOptimizer
from src.services.selection.analytics import ChemistryAnalytics

# Orchestration services
from src.services.orchestration.query_processing_service import QueryProcessingService
from src.services.orchestration.s2_kernel_service import S2KernelOrchestrationService
from src.services.orchestration.nway_orchestration_service import (
    NwayOrchestrationService,
)
from src.services.orchestration.team_selection_service import TeamSelectionService
from src.services.orchestration.dispatch_evidence_service import DispatchEvidenceService

# Critique (Phase 3 seams)
from src.core.critique.contracts import (
    ICritiquePreparer,
    ICritiqueRunner,
    ICritiqueSynthesizer,
)
from src.services.critique.facade_implementations import (
    V1CritiquePreparer,
    V1CritiqueRunner,
    V1CritiqueSynthesizer,
)
from src.core.enhanced_devils_advocate_system import EnhancedDevilsAdvocateSystem

# Orchestration dependencies
from src.core.next_gen_query_chunker import NextGenQueryChunker
from src.services.chunking.facade_implementations import (
    V1ChunkingStrategy,
    V1ChunkingEvaluator,
)
from src.services.task_classification_service import TaskClassificationService
from src.core.research_based_query_enhancer import ResearchBasedQueryEnhancer
from src.services.s2_trigger_classifier import S2TriggerClassifier
from src.services.s2_tier_controller import S2TierController
from src.services.selection.nway_pattern_selection_service import (
    NWayPatternSelectionService,
)
from src.services.selection.pattern_optimizer import V1PatternOptimizer
from src.services.selection.pattern_coverage_analyzer import V1CoverageAnalyzer
from src.services.selection.pattern_scorer import V1PatternScorer
from src.services.selection.pattern_analytics import V1PatternAnalytics
import os
from src.engine.core.llm_manager import LLMManager, get_llm_manager
from src.engine.integrations.perplexity_client import PerplexityClient
from src.services.persistence.database_service import DatabaseService, DatabaseServiceConfig, DatabaseOperationError
from src.services.learning_performance_service import LearningPerformanceService

logger = logging.getLogger(__name__)


class ServiceContainer:
    """Simple DI container for analysis services (PR-01).
    In future PRs, this can be extended to support per-request scoping or factories.
    """

    def __init__(self) -> None:
        self.prompt_builder: IPromptBuilder = V1PromptBuilder()
        self.result_aggregator: IResultAggregator = V1ResultAggregator()
        self.evidence_emitter: IEvidenceEmitter = V1EvidenceEmitter()

        # 🔧 FIX: Initialize database_service FIRST (before persistence adapter needs it)
        try:
            self.database_service = DatabaseService(DatabaseServiceConfig.from_env())
        except Exception:
            self.database_service = None

        # Persistence adapter selection (default: supabase for production)
        adapter_choice = os.getenv("PERSISTENCE_ADAPTER", "supabase").lower()
        if adapter_choice == "supabase":
            # Pass unified DatabaseService to the adapter (facade)
            try:
                self.event_persistence_adapter = SupabaseAdapter(database_service=self.database_service)
                logger.info("✅ Event persistence: SupabaseAdapter configured")
            except Exception as e:
                # Fallback to file adapter if database unavailable
                logger.warning(f"⚠️ SupabaseAdapter failed, using FileAdapter: {e}")
                self.event_persistence_adapter = FileAdapter()
        else:
            logger.info("📝 Event persistence: FileAdapter configured (local file mode)")
            self.event_persistence_adapter = FileAdapter()
        # Shared context stream
        self._context_stream = get_unified_context_stream()
        # Selection: Chemistry scorer service
        self.chemistry_scorer = ChemistryScorer(context_stream=self._context_stream)
        # Selection: Chemistry optimizer service (delegates to scorer)
        self.chemistry_optimizer = ChemistryOptimizer(self.chemistry_scorer)
        # Selection: Chemistry analytics service (delegates to scorer and emits events)
        self.chemistry_analytics = ChemistryAnalytics(
            context_stream=self._context_stream, scorer=self.chemistry_scorer
        )
        # Consultant runner depends on context stream; context stream will use the adapter from container
        self.consultant_runner: IConsultantRunner = V1ConsultantRunner(
            context_stream=self._context_stream
        )
        # Orchestration dependencies
        # Query enhancement
        _test_fast = str(os.getenv("TEST_FAST", "")).lower() in {"1", "true", "yes"}
        try:
            if _test_fast:
                # Hermetic mode: disable research-based enhancement to avoid network calls
                self.query_enhancer = None
                self.query_enhancement_enabled = False
            else:
                self.query_enhancer = ResearchBasedQueryEnhancer(
                    context_stream=self._context_stream
                )
                self.query_enhancement_enabled = (
                    os.getenv("QUERY_ENHANCEMENT_ENABLED", "true").lower() == "true"
                )
        except Exception:
            self.query_enhancer = None
            self.query_enhancement_enabled = False
        # Chunker
        try:
            strategy = V1ChunkingStrategy()
            evaluator = V1ChunkingEvaluator()
            self.query_chunker = NextGenQueryChunker(strategy=strategy, evaluator=evaluator)
            self.query_chunker.set_context_stream(self._context_stream)
        except Exception:
            self.query_chunker = None
        # Classifier
        self.task_classifier = TaskClassificationService(
            context_stream=self._context_stream
        )
        # Query processing service
        self.query_processor_service = QueryProcessingService(
            enhancer=self.query_enhancer,
            chunker=self.query_chunker,
            classifier=self.task_classifier,
            query_enhancement_enabled=self.query_enhancement_enabled,
        )
        # Engine Room wiring: research provider into LLMManager abstraction
        try:
            if _test_fast:
                # Hermetic TEST_FAST: no external research providers; use global noop LLM manager
                self.perplexity_client = None
                self.llm_manager = get_llm_manager(context_stream=self._context_stream)
            else:
                self.perplexity_client = PerplexityClient()
                self.llm_manager = LLMManager(
                    context_stream=self._context_stream,
                    research_providers={"perplexity": self.perplexity_client},
                )
        except Exception:
            self.perplexity_client = None
            # Fallback to global manager (will be noop in tests/TEST_FAST)
            self.llm_manager = get_llm_manager(context_stream=self._context_stream)
        # S2 Kernel
        self.s2_classifier = S2TriggerClassifier()
        self.s2_tier_controller = S2TierController()
        self.kernel_s2_enabled = (
            os.getenv("KERNEL_S2_ENABLED", "false").lower() == "true"
        )
        self.kernel_s2_tier_override = os.getenv("KERNEL_S2_TIER", "auto")
        self.s2_kernel_service = S2KernelOrchestrationService(
            s2_classifier=self.s2_classifier,
            s2_tier_controller=self.s2_tier_controller,
            kernel_enabled=self.kernel_s2_enabled,
            tier_override=self.kernel_s2_tier_override,
        )
        # NWAY services
        try:
            self.nway_pattern_optimizer = V1PatternOptimizer()
        except Exception:
            self.nway_pattern_optimizer = None
        try:
            self.nway_coverage_analyzer = V1CoverageAnalyzer()
        except Exception:
            self.nway_coverage_analyzer = None
        try:
            self.nway_pattern_scorer = V1PatternScorer()
        except Exception:
            self.nway_pattern_scorer = None
        # Instantiate facade with DI
        self.nway_pattern_service = NWayPatternSelectionService(
            optimizer=self.nway_pattern_optimizer,
            coverage_analyzer=self.nway_coverage_analyzer,
            pattern_scorer=self.nway_pattern_scorer,
            pattern_analytics=None,
        )
        self.nway_orchestration_service = NwayOrchestrationService(
            self.nway_pattern_service, self._context_stream
        )
        # Dispatch evidence service
        self.dispatch_evidence_service = DispatchEvidenceService(self._context_stream)
        # Phase 3 seams: critique services and facade
        self.critique_preparer: ICritiquePreparer = V1CritiquePreparer()
        self.critique_runner: ICritiqueRunner = V1CritiqueRunner()
        self.critique_synthesizer: ICritiqueSynthesizer = V1CritiqueSynthesizer()
        self.devils_advocate_system = EnhancedDevilsAdvocateSystem(
            preparer=self.critique_preparer,
            runner=self.critique_runner,
            synthesizer=self.critique_synthesizer,
        )
        # Dispatch evidence service
        self.dispatch_evidence_service = DispatchEvidenceService(self._context_stream)

        # Note: database_service already initialized at the top of __init__ (before persistence adapter)

        # Learning Performance Service via DI (Operation Unification P1)
        self.learning_performance_service = LearningPerformanceService(
            database_service=self.database_service
        )

    def get_prompt_builder(self) -> IPromptBuilder:  # explicit getter for clarity
        return self.prompt_builder

    def get_consultant_runner(self) -> IConsultantRunner:
        return self.consultant_runner

    def get_result_aggregator(self) -> IResultAggregator:
        return self.result_aggregator

    def get_evidence_emitter(self) -> IEvidenceEmitter:
        return self.evidence_emitter

    # PR-04: expose persistence adapter for UnifiedContextStream
    def get_event_persistence_adapter(self):
        return self.event_persistence_adapter

    # PR-06: chemistry scorer accessor
    def get_chemistry_scorer(self):
        return self.chemistry_scorer

    # PR-07: chemistry optimizer accessor
    def get_chemistry_optimizer(self):
        return self.chemistry_optimizer

    # PR-08: chemistry analytics accessor
    def get_chemistry_analytics(self):
        return self.chemistry_analytics

    # PR-10: query processing accessor
    def get_query_processing_service(self):
        return self.query_processor_service

    # PR-11: s2 kernel accessor
    def get_s2_kernel_service(self):
        return self.s2_kernel_service

    # PR-12: nway orchestration accessor
    def get_nway_orchestration_service(self):
        return self.nway_orchestration_service

    def get_nway_pattern_service(self):
        return self.nway_pattern_service

    # PR-14: dispatch evidence accessor (optionally override context stream per orchestrator instance)
    def get_dispatch_evidence_service(self, context_stream=None):
        if context_stream is not None:
            return DispatchEvidenceService(context_stream)
        return self.dispatch_evidence_service

    # PR-18: devils advocate system accessor (facade)
    def get_devils_advocate_system(self):
        return self.devils_advocate_system

    # PR-09: team selection factory (requires runtime inputs)
    def get_team_selection_service(self, consultant_database, contextual_engine):
        return TeamSelectionService(consultant_database, contextual_engine)

    # Persistence facade accessor
    def get_database_service(self):
        return self.database_service

    # Learning performance service accessor
    def get_learning_performance_service(self):
        return self.learning_performance_service


# Global container instance (process-wide).
# This keeps wiring minimal while we introduce seams incrementally.
global_container = ServiceContainer()
