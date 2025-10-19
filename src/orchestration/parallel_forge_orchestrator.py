"""
######################################################################
# DEPRECATED - DO NOT USE
#
# This orchestrator is a legacy V5 implementation.
# The canonical path is the StatefulPipelineOrchestrator with V6 executors.
# This file is preserved for historical reference and will be removed in a
# future version.
#
# See: docs/architectural_refinement/AR-XX_Orchestrator_Consolidation.md
######################################################################

ParallelForge Orchestrator - STEP 4 of Honest Orchestra
======================================================

PRINCIPLE: "Fail Loudly, Succeed Honestly"

This orchestrator executes parallel consultant analysis with real LLM/research calls.
Handles both consultant analysis and devil's advocate critiques simultaneously.

Process:
1. Generate 5-part prompts for each consultant
2. Execute consultant analyses in parallel (LLM + Perplexity)
3. Execute devil's advocate critiques in parallel (4-engine system)
4. Handle graceful degradation for individual failures
5. Return ParallelForgeResults or raise ForgeError
"""

import asyncio
import time
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple, Optional

# Supabase client for Synergy Engine
from supabase import create_client

from .exceptions import ForgeError
from .contracts import (
    DispatchPackage,
    ConsultantAnalysisResult,
    AnalysisCritique,
    ParallelForgeResults,
    ConsultantBlueprint,
    NWayConfiguration,
)

# PROJECT LOLLAPALOOZA: Import the enhanced Synergy Engine
from src.engine.utils.nway_prompt_infuser_synergy_engine import get_nway_synergy_engine
from src.core.unified_context_stream import get_unified_context_stream, ContextEventType

try:
    from src.engine.core.feature_flags import (
        FeatureFlag,
        get_experiment_group,
        is_feature_enabled,
    )

    FEATURE_FLAGS_AVAILABLE = True
except Exception:  # pragma: no cover - feature flags optional
    FEATURE_FLAGS_AVAILABLE = False
    FeatureFlag = None  # type: ignore
    get_experiment_group = None  # type: ignore
    is_feature_enabled = None  # type: ignore

# V5.3 CANONICAL COMPLIANCE: Import V5.3 Resilient Manager Pattern
from src.engine.core.llm_manager import get_llm_manager
from src.engine.core.research_manager import ResearchManager
from src.engine.providers.research import PerplexityProvider, ExaProvider
from src.services.depth_enrichment.consultant_depth_pack_builder import (
    ConsultantDepthPackBuilder,
)
from src.services.agent_guidance.agent_guidance_retriever import (
    AgentGuidanceRetriever,
)

logger = logging.getLogger(__name__)

# PR-01 seams: analysis service contracts & container (no behavior change)
try:
    from src.services.analysis.contracts import (
        IPromptBuilder,
        IConsultantRunner,
        IResultAggregator,
        IEvidenceEmitter,
    )
    from src.services.container import global_container
except Exception:
    # Seams are optional in PR-01; keep runtime resilient if package is missing
    IPromptBuilder = IConsultantRunner = IResultAggregator = IEvidenceEmitter = object  # type: ignore
    global_container = None  # type: ignore


class ParallelForgeOrchestrator:
    """Orchestrator for parallel consultant analysis and critique execution"""

    def __init__(
        self,
        prompt_builder: "IPromptBuilder" = None,  # type: ignore[name-defined]
        consultant_runner: "IConsultantRunner" = None,  # type: ignore[name-defined]
        result_aggregator: "IResultAggregator" = None,  # type: ignore[name-defined]
        evidence_emitter: "IEvidenceEmitter" = None,  # type: ignore[name-defined]
    ):
        # V5.3 CANONICAL COMPLIANCE: Use resilient managers instead of direct clients
        self.llm_manager = None
        self.research_manager = None

        # Unified context stream for governance metadata and transparency
        self.context_stream = get_unified_context_stream()

        # PROJECT LOLLAPALOOZA: Initialize Synergy Engine
        self.synergy_engine = self._initialize_synergy_engine()

        # Depth enrichment: build consultant depth packs upfront
        self.depth_pack_builder = ConsultantDepthPackBuilder()
        self._problem_context: str = ""
        self.agent_guidance_retriever = AgentGuidanceRetriever(
            context_stream=self.context_stream
        )

        # GOVERNANCE V2: Agent context map (consultant_id -> {contract_id, instance_id})
        self._agent_context: Dict[str, Dict[str, str]] = {}

        # PR-01 seams: bind analysis services (defaults from global container)
        if global_container:
            self.prompt_builder = (
                prompt_builder or global_container.get_prompt_builder()
            )
            self.consultant_runner = (
                consultant_runner or global_container.get_consultant_runner()
            )
            self.result_aggregator = (
                result_aggregator or global_container.get_result_aggregator()
            )
            self.evidence_emitter = (
                evidence_emitter or global_container.get_evidence_emitter()
            )
        else:
            # Fallback placeholders when container not available
            self.prompt_builder = prompt_builder
            self.consultant_runner = consultant_runner
            self.result_aggregator = result_aggregator
            self.evidence_emitter = evidence_emitter

    def _resolve_stage0_enrichment_state(self, trace_id: Optional[str]) -> Tuple[bool, str]:
        """Return tuple (enabled, variant) for Stage 0 enrichment."""

        manual_flag = os.getenv("ENABLE_STAGE0_ENRICHMENT")
        if manual_flag is not None:
            enabled = manual_flag.lower() in {"1", "true", "yes", "on", "treatment"}
            return enabled, "manual_on" if enabled else "manual_off"

        if FEATURE_FLAGS_AVAILABLE and FeatureFlag is not None:
            identifier = None
            if trace_id:
                try:
                    identifier = uuid.UUID(trace_id)
                except ValueError:
                    identifier = uuid.uuid5(uuid.NAMESPACE_URL, trace_id)

            try:
                experiment_group = (
                    get_experiment_group(  # type: ignore[misc]
                        FeatureFlag.ENABLE_STAGE0_ENRICHMENT,
                        user_id=identifier,
                    )
                    if identifier is not None
                    else None
                )
            except Exception as exc:
                logger.debug(
                    "Stage 0 enrichment group lookup failed in orchestrator: %s",
                    exc,
                )
                experiment_group = None

            try:
                enabled = is_feature_enabled(  # type: ignore[misc]
                    FeatureFlag.ENABLE_STAGE0_ENRICHMENT,
                    user_id=identifier,
                )
            except Exception as exc:
                logger.debug(
                    "Stage 0 enrichment flag check failed in orchestrator: %s",
                    exc,
                )
                enabled = True

            variant = experiment_group or ("treatment" if enabled else "control")
            return enabled, variant

        return True, "default_enabled"

    def _initialize_synergy_engine(self):
        """Initialize the PROJECT LOLLAPALOOZA Synergy Engine"""
        try:
            # Create Supabase client for N-Way cluster data
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_ANON_KEY")

            if not supabase_url or not supabase_key:
                logger.warning(
                    "⚠️ SYNERGY ENGINE: Supabase config missing - engine will use fallback mode"
                )
                supabase_client = None
            else:
                supabase_client = create_client(supabase_url, supabase_key)
                logger.info("✅ SYNERGY ENGINE: Supabase client initialized")

            # Import unified LLM client for meta-analysis
            try:
                from src.integrations.llm.unified_client import UnifiedLLMClient

                llm_client = UnifiedLLMClient()
                logger.info("✅ SYNERGY ENGINE: LLM client available for meta-analysis")
            except ImportError:
                llm_client = None
                logger.warning(
                    "⚠️ SYNERGY ENGINE: LLM client not available - meta-analysis disabled"
                )

            # Create synergy engine instance
            synergy_engine = get_nway_synergy_engine(supabase_client, llm_client)
            logger.info("🔥 SYNERGY ENGINE: Project Lollapalooza integration complete!")

            return synergy_engine

        except Exception as e:
            logger.error(f"❌ SYNERGY ENGINE: Initialization failed: {e}")
            return None

    async def _initialize_managers(self):
        """V5.3 CANONICAL COMPLIANCE: Initialize resilient managers"""
        try:
            # V5.3 CANONICAL: Initialize LLMManager with context stream
            self.llm_manager = get_llm_manager(context_stream=self.context_stream)
            logger.info("✅ V5.3 LLMManager initialized with resilient fallback")

            # V5.3 CANONICAL: Initialize ResearchManager with fallback providers
            research_providers = []

            # Primary provider: Perplexity
            perplexity_provider = PerplexityProvider()
            if await perplexity_provider.is_available():
                research_providers.append(perplexity_provider)
                logger.info("✅ Perplexity research provider available")
            else:
                logger.warning("⚠️ Perplexity research provider not available")

            # Fallback provider: Exa
            exa_provider = ExaProvider()
            if await exa_provider.is_available():
                research_providers.append(exa_provider)
                logger.info("✅ Exa research provider available as fallback")
            else:
                logger.warning("⚠️ Exa research provider not available")

            # Initialize ResearchManager if at least one provider is available
            if research_providers:
                self.research_manager = ResearchManager(
                    research_providers, self.context_stream
                )
                logger.info(
                    f"🔍 V5.3 Research Manager initialized with {len(research_providers)} providers: {[p.provider_name for p in research_providers]}"
                )
            else:
                logger.warning(
                    "⚠️ No research providers available - research functionality disabled"
                )

            # V5.3 CANONICAL: Validate manager availability
            if not self.llm_manager:
                raise ForgeError("V5.3 LLMManager not available - cannot proceed")

        except Exception as e:
            raise ForgeError(f"Failed to initialize V5.3 managers: {e}")

    async def execute_parallel_analysis(
        self, dispatch_package: DispatchPackage, problem_context: str
    ) -> ParallelForgeResults:
        """
        Execute parallel analysis - compatibility method for unified API.
        
        Args:
            dispatch_package: Dispatch package with selected consultants
            problem_context: Problem context string (used for additional context)
            
        Returns:
            ParallelForgeResults: Combined results from all parallel executions
        """
        # Store problem context for potential use in prompts
        self._problem_context = problem_context
        
        # Delegate to the main run_parallel_forges method
        return await self.run_parallel_forges(dispatch_package)

    async def run_parallel_forges(
        self, dispatch: DispatchPackage
    ) -> ParallelForgeResults:
        """
        Execute parallel consultant analyses and critiques

        Args:
            dispatch: Dispatch package with selected consultants

        Returns:
            ParallelForgeResults: Combined results from all parallel executions

        Raises:
            ForgeError: If the entire process fails
        """
        start_time = time.time()

        try:
            logger.info(
                f"🔥 Starting parallel forge with {len(dispatch.selected_consultants)} consultants"
            )
            logger.info(f"🎭 Pattern: {dispatch.nway_configuration.pattern_name}")

            # V5.3 CANONICAL: Initialize resilient managers
            await self._initialize_managers()

            # Step 2: Generate 5-part prompts for each consultant
            logger.info("📝 Generating 5-part consultant prompts...")
            consultant_prompts = await self._generate_consultant_prompts(dispatch)

            # GOVERNANCE V2: Prepare agent contract IDs and unique instance IDs
            self._prepare_agent_instances(dispatch.selected_consultants)

            # Step 3: Execute parallel consultant analyses
            logger.info("⚡ Executing parallel consultant analyses...")
            consultant_tasks = [
                self._execute_consultant_analysis(consultant, prompt)
                for consultant_id, (consultant, prompt) in consultant_prompts.items()
            ]

            # Step 4: Execute analyses with timeout and error handling
            consultant_results = await self._execute_with_graceful_degradation(
                consultant_tasks, "consultant analyses"
            )

            # PR-01 seam: optional aggregator call (ignore result to avoid behavior change)
            try:
                if hasattr(self, "result_aggregator") and self.result_aggregator:
                    _ = self.result_aggregator.aggregate(
                        [r for r in consultant_results if r is not None]
                    )
            except Exception:
                pass

            # Step 5: Execute parallel devil's advocate critiques
            logger.info("👹 Executing parallel devil's advocate critiques...")
            critique_tasks = [
                self._execute_devils_advocate_critique(result)
                for result in consultant_results
                if result is not None
            ]

            critique_results = await self._execute_with_graceful_degradation(
                critique_tasks, "devil's advocate critiques"
            )

            # Step 6: Create final results package
            total_processing_time = time.time() - start_time

            results = ParallelForgeResults(
                consultant_analyses=[r for r in consultant_results if r is not None],
                critiques=[r for r in critique_results if r is not None],
                total_processing_time_seconds=total_processing_time,
                successful_analyses=len(
                    [r for r in consultant_results if r is not None]
                ),
                successful_critiques=len(
                    [r for r in critique_results if r is not None]
                ),
                timestamp=datetime.now(timezone.utc),
            )

            # Validation
            self._validate_forge_results(results, dispatch)

            logger.info(f"🎉 Parallel forge completed in {total_processing_time:.1f}s")
            logger.info(
                f"✅ Successful analyses: {results.successful_analyses}/{len(dispatch.selected_consultants)}"
            )
            logger.info(f"✅ Successful critiques: {results.successful_critiques}")

            return results

        except ForgeError:
            raise
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ Parallel forge failed after {processing_time:.1f}s: {e}")
            raise ForgeError(f"Parallel forge execution failed: {e}")

    async def _generate_consultant_prompts(
        self, dispatch: DispatchPackage
    ) -> Dict[str, Tuple[ConsultantBlueprint, str]]:
        """Generate enhanced prompts using PROJECT LOLLAPALOOZA Synergy Engine"""

        consultant_prompts = {}
        stage0_plan: List[Dict[str, Any]] = []
        total_depth_tokens = 0
        total_mm_items = 0
        stage0_latency_ms = 0

        trace_id = getattr(self.context_stream, "trace_id", None)
        stage0_enabled, stage0_variant = self._resolve_stage0_enrichment_state(trace_id)
        stage0_build_start = (
            time.time() if stage0_enabled and dispatch.selected_consultants else None
        )

        if self.context_stream:
            try:
                self.context_stream.add_event(
                    ContextEventType.STAGE0_EXPERIMENT_ASSIGNED,
                    {
                        "stage": "parallel_forge",
                        "enabled": stage0_enabled,
                        "variant": stage0_variant,
                        "consultant_count": len(dispatch.selected_consultants),
                    },
                )
            except Exception:
                pass

        # PROJECT LOLLAPALOOZA: Enhanced prompt generation with Synergy Engine
        for consultant in dispatch.selected_consultants:
            depth_pack_text = ""
            depth_metadata: Dict[str, Any] = {"mm_items": [], "nway": None}

            if stage0_enabled:
                try:
                    depth_result = self.depth_pack_builder.build_depth_pack(
                        consultant,
                        problem_context=getattr(self, "_problem_context", ""),
                    )

                    if depth_result.has_content:
                        depth_pack_text = depth_result.text
                        depth_metadata = depth_result.metadata or depth_metadata
                        token_estimate = self._estimate_tokens(depth_pack_text)
                        stage0_plan.append(
                            {
                                "consultant_id": consultant.consultant_id,
                                "mm_items": depth_metadata.get("mm_items", []),
                                "nway": depth_metadata.get("nway"),
                                "token_estimate": token_estimate,
                            }
                        )
                        total_depth_tokens += token_estimate
                        total_mm_items += len(depth_metadata.get("mm_items", []))

                        try:
                            self.context_stream.add_event(
                                ContextEventType.DEPTH_ENRICHMENT_APPLIED,
                                {
                                    "stage": "pre_consultant",
                                    "consultant_id": consultant.consultant_id,
                                    "mm_count": len(depth_metadata.get("mm_items", [])),
                                    "token_estimate": token_estimate,
                                    "has_nway_directive": bool(depth_metadata.get("nway")),
                                    "depth_keys": [
                                        item.get("model", "")
                                        for item in depth_metadata.get("mm_items", [])
                                    ],
                                    "variant": stage0_variant,
                                },
                            )
                        except Exception as exc:
                            logger.debug(
                                "Depth enrichment telemetry failed: %s", exc
                            )

                except Exception as exc:
                    logger.warning(
                        "⚠️ Depth pack build failed for %s: %s",
                        consultant.consultant_id,
                        exc,
                    )

            # Generate base prompt (traditional approach)
            base_prompt = self._generate_base_consultant_prompt(dispatch, consultant)

            if depth_pack_text:
                base_prompt = f"{depth_pack_text}\n\n{base_prompt}"

            # PROJECT LOLLAPALOOZA: Apply Synergy Engine transformation
            selected_clusters: List[str] = []
            if self.synergy_engine:
                try:
                    # Determine N-Way clusters for this consultant
                    selected_clusters = self._determine_nway_clusters_for_consultant(
                        consultant, dispatch.nway_configuration
                    )

                    # Use Synergy Engine to enhance the prompt
                    infusion_result = await self.synergy_engine.infuse_consultant_prompt_with_synergy_engine(
                        original_prompt=base_prompt,
                        selected_nway_clusters=selected_clusters,
                        consultant_id=consultant.consultant_id,
                        context_stream=self.context_stream,
                    )

                    if infusion_result.success:
                        enhanced_prompt = infusion_result.infused_prompt
                        logger.info(
                            f"🔥 SYNERGY ENGINE: Enhanced prompt for {consultant.consultant_id} with {len(selected_clusters)} clusters"
                        )

                        # Log synergy analysis results
                        if (
                            infusion_result.synergy_analysis
                            and infusion_result.synergy_analysis.success
                        ):
                            confidence = (
                                infusion_result.synergy_analysis.confidence_score
                            )
                            logger.info(
                                f"⚡ SYNERGY ENGINE: Meta-analysis confidence: {confidence:.1%}"
                            )
                            # Emit Glass-Box meta-directive event for full transparency
                            try:
                                self.context_stream.add_event(
                                    ContextEventType.SYNERGY_META_DIRECTIVE,
                                    {
                                        "consultant_id": consultant.consultant_id,
                                        "selected_nway_clusters": selected_clusters,
                                        "applied_clusters": infusion_result.applied_clusters,
                                        "meta_directive": infusion_result.synergy_analysis.meta_directive,
                                        "synergy_insight": infusion_result.synergy_analysis.synergy_insight,
                                        "conflict_insight": infusion_result.synergy_analysis.conflict_insight,
                                        "confidence_score": infusion_result.synergy_analysis.confidence_score,
                                        "analysis_model": infusion_result.synergy_analysis.analysis_model,
                                        "processing_time_ms": infusion_result.synergy_analysis.processing_time_ms,
                                    },
                                )
                            except Exception as e:
                                logger.warning(
                                    f"⚠️ Could not emit SYNERGY_META_DIRECTIVE event: {e}"
                                )
                    else:
                        enhanced_prompt = base_prompt
                        logger.warning(
                            f"⚠️ SYNERGY ENGINE: Infusion failed for {consultant.consultant_id}, using base prompt"
                        )

                except Exception as e:
                    enhanced_prompt = base_prompt
                    logger.error(
                        f"❌ SYNERGY ENGINE: Error enhancing prompt for {consultant.consultant_id}: {e}"
                    )

            else:
                enhanced_prompt = base_prompt
                logger.warning(
                    f"⚠️ SYNERGY ENGINE: Not available, using base prompt for {consultant.consultant_id}"
                )

            # NEW: Derive concepts from context section for model selection (lightweight)
            concepts = []
            try:
                if hasattr(self, "prompt_builder") and self.prompt_builder:
                    context_text = self.prompt_builder.build_context_section(
                        {
                            "consultant_id": consultant.consultant_id,
                            "consultant_type": consultant.consultant_type,
                            "specialization": consultant.specialization,
                            "assigned_dimensions": consultant.assigned_dimensions,
                        },
                        {
                            "pattern_name": dispatch.nway_configuration.pattern_name,
                            "interaction_strategy": dispatch.nway_configuration.interaction_strategy,
                        },
                    )
                else:
                    context_text = self._generate_context_section(dispatch, consultant)
                # Simple heuristic: split by newlines and punctuation to gather concept phrases
                for line in context_text.splitlines():
                    if line.strip():
                        concepts.append(line.strip())
            except Exception:
                pass

            # NEW: Select mental models per consultant for transparency (reuses libraries)
            try:
                from src.services.model_selection_service import ModelSelectionService
                from src.services.learning_weight_manager import LearningWeightManager

                ms = ModelSelectionService(weight_manager=LearningWeightManager())
                selected_models, lattice = ms.select_models_for_consultant(
                    consultant_id=consultant.consultant_id,
                    selected_nway_clusters=selected_clusters,
                    concepts=concepts,
                    context_stream=self.context_stream,
                    k_min=3,
                    k_max=7,
                )
                # Optionally append a summary to the prompt (non-invasive)
                model_map_md = ms.render_model_map_as_markdown(selected_models, lattice)
                enhanced_prompt = f"{enhanced_prompt}\n\n{model_map_md}"
            except Exception as e:
                logger.warning(
                    f"⚠️ Model selection map not generated for {consultant.consultant_id}: {e}"
                )

            consultant_prompts[consultant.consultant_id] = (consultant, enhanced_prompt)

        if stage0_build_start is not None:
            stage0_latency_ms = int((time.time() - stage0_build_start) * 1000)

        stage0_metrics = {
            "enabled": stage0_enabled,
            "variant": stage0_variant,
            "total_consultants": len(dispatch.selected_consultants),
            "consultant_count": len(stage0_plan),
            "latency_ms": stage0_latency_ms,
            "total_token_estimate": total_depth_tokens,
            "avg_tokens_per_consultant": int(
                total_depth_tokens / len(stage0_plan)
            )
            if stage0_plan
            else 0,
            "total_mm_items": total_mm_items,
        }

        if self.context_stream:
            try:
                self.context_stream.add_event(
                    ContextEventType.STAGE0_PLAN_RECORDED,
                    {
                        "entries": stage0_plan,
                        "total_duration_ms": stage0_latency_ms,
                        "enabled": stage0_enabled,
                        "variant": stage0_variant,
                    },
                )
            except Exception as exc:
                logger.debug("Stage0 plan telemetry failed: %s", exc)

            try:
                self.context_stream.add_event(
                    ContextEventType.DEPTH_ENRICHMENT_METRICS,
                    stage0_metrics,
                )
            except Exception as exc:
                logger.debug("Stage0 metrics telemetry failed: %s", exc)

        logger.info(
            f"🎯 Generated enhanced prompts for {len(consultant_prompts)} consultants using PROJECT LOLLAPALOOZA"
        )
        return consultant_prompts

    def _prepare_agent_instances(self, consultants: List[ConsultantBlueprint]) -> None:
        """Prepare agent contract IDs and instance IDs for deep instrumentation"""

        for consultant in consultants:
            # PATCH 1: Fix V2 routing by using correct contract_id mapping
            contract_id = self._compute_agent_contract_id(consultant.consultant_id)
            instance_id = str(uuid.uuid4())

            # Store in agent context map for later reference
            self._agent_context[consultant.consultant_id] = {
                "contract_id": contract_id,
                "instance_id": instance_id,
            }

            logger.info(
                f"🏷️ Prepared agent context: {consultant.consultant_id} -> {contract_id} [{instance_id[:8]}...]"
            )

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate based on word count."""
        words = text.split()
        return int(len(words) * 1.2)

    def _generate_base_consultant_prompt(
        self, dispatch: DispatchPackage, consultant: ConsultantBlueprint
    ) -> str:
        """Generate base consultant prompt by delegating fully to the prompt builder (PR-02b)."""
        dispatch_info = {
            "pattern_name": dispatch.nway_configuration.pattern_name,
            "interaction_strategy": dispatch.nway_configuration.interaction_strategy,
        }
        consultant_info = {
            "consultant_id": consultant.consultant_id,
            "consultant_type": consultant.consultant_type,
            "specialization": consultant.specialization,
            "assigned_dimensions": consultant.assigned_dimensions,
        }
        try:
            if hasattr(self, "prompt_builder") and self.prompt_builder:
                base_prompt = self.prompt_builder.build(consultant_info, dispatch_info)
            else:
                # Fallback to maintain behavior if seam is unavailable

                # Reconstruct via local simple formatter identical to builder
                context = f"""You are analyzing a strategic business challenge that requires your specialized expertise in {consultant.specialization}.

Your assigned analytical dimensions:
{chr(10).join([f"• {dim}" for dim in consultant.assigned_dimensions])}

Interaction Pattern: {dispatch.nway_configuration.pattern_name}
Team Strategy: {dispatch.nway_configuration.interaction_strategy}

Your analysis will be combined with insights from other specialized consultants to provide comprehensive strategic guidance."""
                role = self._generate_role_section(consultant)
                framework = self._generate_framework_section(consultant)
                exec_instructions = self._generate_execution_instructions(consultant)
                output_format = self._generate_output_format_section()
                base_prompt = f"""# STRATEGIC CONSULTANT ANALYSIS REQUEST

## CONTEXT & PROBLEM
{context}

## YOUR ROLE & EXPERTISE
{role}

## ANALYTICAL FRAMEWORK
{framework}

## EXECUTION INSTRUCTIONS
{exec_instructions}

## REQUIRED OUTPUT FORMAT
{output_format}

Execute this analysis with the depth and rigor expected of a top-tier consulting firm. 
Provide actionable insights backed by logical reasoning and industry best practices."""
        except Exception:
            # Last-resort fallback (should be rare)
            base_prompt = ""
        return base_prompt

    def _determine_nway_clusters_for_consultant(
        self, consultant: ConsultantBlueprint, nway_config: NWayConfiguration
    ) -> List[str]:
        """Determine which N-Way clusters to apply for a specific consultant

        PATCH 2: Use actual NWAY_* file identifiers instead of generic cluster names
        """

        # PATCH 2: Map consultant types to actual NWAY_* file identifiers
        consultant_cluster_mapping = {
            "strategic_analyst": [
                "NWAY_STRATEGIST_CLUSTER_009",
                "NWAY_ANALYST_CLUSTER_007",
                "NWAY_DECISION_TRILEMMA_004",
            ],
            "market_researcher": [
                "NWAY_RESEARCHER_CLUSTER_016",
                "NWAY_OUTLIER_ANALYSIS_017",
                "NWAY_DIAGNOSTIC_SOLVING_014",
            ],
            "financial_analyst": [
                "NWAY_FINANCIAL_QUANTITATIVE_ANALYSIS_024",
                "NWAY_ANALYST_CLUSTER_007",
                "NWAY_AUCTION_001",
            ],
            "operations_expert": [
                "NWAY_PM_EXECUTION_013",
                "NWAY_ENTREPRENEUR_AGENCY_015",
                "NWAY_DIAGNOSTIC_SOLVING_014",
            ],
            "implementation_specialist": [
                "NWAY_PM_EXECUTION_013",
                "NWAY_TEAM_LEADERSHIP_DYNAMICS_023",
                "NWAY_LEARNING_TEACHING_012",
            ],
            "innovation_consultant": [
                "NWAY_CREATIVITY_003",
                "NWAY_ATYPICAL_SYNERGY_EXPLORATORY_011",
                "NWAY_ENTREPRENEUR_AGENCY_015",
            ],
            "technology_advisor": [
                "NWAY_PRODUCT_MARKET_FIT_ENGINE_025",
                "NWAY_ENTREPRENEUR_AGENCY_015",
                "NWAY_DIAGNOSTIC_SOLVING_014",
            ],
            "crisis_manager": [
                "NWAY_BIAS_MITIGATION_019",
                "NWAY_NEGATIVE_SIMPLE_ACTION_021",
                "NWAY_ETHICAL_GOVERNANCE_FRAMEWORK_026",
            ],
            "turnaround_specialist": [
                "NWAY_MOTIVATION_TRADEOFF_008",
                "NWAY_ENTREPRENEUR_AGENCY_015",
                "NWAY_NEGATIVE_SIMPLE_ACTION_021",
            ],
            "risk_assessor": [
                "NWAY_BIAS_MITIGATION_019",
                "NWAY_OUTLIER_ANALYSIS_017",
                "NWAY_ETHICAL_GOVERNANCE_FRAMEWORK_026",
            ],
        }

        # Get base clusters for this consultant type
        base_clusters = consultant_cluster_mapping.get(consultant.consultant_id, [])

        # Add pattern-specific clusters based on N-Way configuration
        pattern_clusters = self._get_pattern_specific_clusters(nway_config.pattern_name)

        # Combine and deduplicate
        all_clusters = list(set(base_clusters + pattern_clusters))

        logger.info(
            f"📊 SYNERGY ENGINE: Selected {len(all_clusters)} NWAY clusters for {consultant.consultant_id}: {all_clusters[:2]}..."
        )
        return all_clusters

    def _get_pattern_specific_clusters(self, pattern_name: str) -> List[str]:
        """Get additional N-Way clusters based on the interaction pattern

        PATCH 2: Use actual NWAY_* identifiers for pattern-specific clusters
        """

        pattern_cluster_mapping = {
            "strategic_analysis": [
                "NWAY_STRATEGIST_CLUSTER_009",
                "NWAY_ANALYST_CLUSTER_007",
            ],
            "operational_optimization": [
                "NWAY_PM_EXECUTION_013",
                "NWAY_ENTREPRENEUR_AGENCY_015",
            ],
            "innovation_discovery": [
                "NWAY_CREATIVITY_003",
                "NWAY_ATYPICAL_SYNERGY_EXPLORATORY_011",
            ],
            "crisis_management": [
                "NWAY_BIAS_MITIGATION_019",
                "NWAY_NEGATIVE_SIMPLE_ACTION_021",
            ],
        }

        return pattern_cluster_mapping.get(pattern_name, ["NWAY_DECISION_TRILEMMA_004"])

    async def _execute_consultant_analysis(
        self, consultant: ConsultantBlueprint, prompt: str
    ) -> Optional[ConsultantAnalysisResult]:
        """Execute individual consultant analysis by delegating to the injected runner (PR-03)."""
        # Prepare thin wrapper inputs
        agent_meta = self._agent_context.get(consultant.consultant_id, {})
        consultant_info = {
            "consultant_id": consultant.consultant_id,
            "consultant_type": consultant.consultant_type,
            "specialization": consultant.specialization,
            "assigned_dimensions": consultant.assigned_dimensions,
        }
        context = {"agent_meta": agent_meta}
        # Delegate
        return await self.consultant_runner.run(consultant_info, prompt, context)

    async def _execute_devils_advocate_critique(
        self, analysis: ConsultantAnalysisResult
    ) -> Optional[AnalysisCritique]:
        """Execute devil's advocate critique using 4-engine system"""

        start_time = time.time()

        try:
            logger.info(f"👹 Critiquing {analysis.consultant_id} analysis...")

            # GOVERNANCE V2: Set agent context for critique phase
            agent_meta = self._agent_context.get(analysis.consultant_id, {})
            self.context_stream.set_agent_context(
                agent_contract_id=agent_meta.get("contract_id"),
                agent_instance_id=agent_meta.get("instance_id"),
            )

            guidance = self.agent_guidance_retriever.get_guidance(
                "devils_advocate",
                guidance_type="communication",
                max_words=250,
            )
            guidance_section = ""
            if guidance.get("applicable"):
                guidance_section = f"\n\nCOMMUNICATION GUIDANCE:\n{guidance['guidance']}\n"

            critique_prompt = f"""You are a Devil's Advocate critically examining this consultant analysis.

CONSULTANT: {analysis.consultant_id}
ANALYSIS TO CRITIQUE:
{analysis.analysis_content}{guidance_section}

KEY INSIGHTS TO EXAMINE:
{chr(10).join([f"• {insight}" for insight in analysis.key_insights])}

RECOMMENDATIONS TO CHALLENGE:
{chr(10).join([f"• {rec}" for rec in analysis.recommendations])}

Your task is to identify:
1. **Logical flaws** in the reasoning
2. **Missing perspectives** not considered  
3. **Questionable assumptions** underlying the analysis
4. **Alternative viewpoints** that contradict the recommendations
5. **Implementation risks** not adequately addressed

Respond in JSON format:
{{
    "critique_summary": "Key concerns and challenges to this analysis",
    "identified_weaknesses": ["Weakness 1", "Weakness 2", "Weakness 3"],
    "alternative_perspectives": ["Alternative view 1", "Alternative 2"],
    "risk_assessments": ["Risk concern 1", "Risk concern 2"],
    "confidence_level": 0.85
}}

Be thorough but constructive in your critique."""

            # V5.3 CANONICAL: Use LLMManager for critique
            if not self.llm_manager:
                raise ForgeError("V5.3 LLMManager not available")

            # V5.3 CANONICAL: LLM request logging handled by LLMManager

            # V5.3 CANONICAL: Execute critique via LLMManager
            critique_result = await self.llm_manager.call_llm(
                messages=[{"role": "user", "content": critique_prompt}],
                temperature=0.7,
                max_tokens=2000,
            )
            critique_response = critique_result.content

            # V5.3 CANONICAL: LLM response logging handled by LLMManager
            self.context_stream.add_event(
                ContextEventType.LLM_PROVIDER_RESPONSE,
                data={
                    "provider": critique_result.provider_name,
                    "model": critique_result.model_name,
                    "response_preview": critique_response[:200],
                    "response_length": len(critique_response),
                    "tokens_used": critique_result.total_tokens,
                    "cost_usd": critique_result.cost,
                    "execution_mode": "devils_advocate_critique_v2",  # PATCH 3: Add deterministic execution_mode
                    "operation_type": "devils_advocate_critique",
                },
            )

            # Parse critique response
            critique_data = self._parse_consultant_response(critique_response)

            processing_time = time.time() - start_time

            result = AnalysisCritique(
                target_consultant=analysis.consultant_id,
                critique_content=critique_data.get(
                    "critique_summary", critique_response
                ),
                identified_weaknesses=critique_data.get("identified_weaknesses", []),
                alternative_perspectives=critique_data.get(
                    "alternative_perspectives", []
                ),
                risk_assessments=critique_data.get("risk_assessments", []),
                confidence_level=critique_data.get("confidence_level", 0.8),
                processing_time_seconds=processing_time,
                engines_used=[critique_result.provider_name],
            )

            logger.info(
                f"✅ {analysis.consultant_id} critique completed in {processing_time:.1f}s"
            )
            return result

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(
                f"❌ {analysis.consultant_id} critique failed after {processing_time:.1f}s: {e}"
            )
            return None  # Graceful degradation
        finally:
            # Clear agent context once critique completed
            self.context_stream.clear_agent_context()

    # V5.3 CANONICAL: Removed _select_best_llm_client - LLMManager handles provider selection

    def _parse_consultant_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from consultant LLM call"""

        try:
            import json
            import re

            # Try to extract JSON from response
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_text = response[start:end].strip()
            else:
                json_match = re.search(r"\{.*\}", response, re.DOTALL)
                if json_match:
                    json_text = json_match.group()
                else:
                    # Return default structure if no JSON found
                    return {
                        "detailed_analysis": response,
                        "key_insights": [],
                        "recommendations": [],
                        "confidence_level": 0.7,
                    }

            return json.loads(json_text)

        except Exception as e:
            logger.warning(f"Failed to parse JSON response, using raw text: {e}")
            return {
                "detailed_analysis": response,
                "key_insights": [],
                "recommendations": [],
                "confidence_level": 0.7,
            }

    async def _execute_with_graceful_degradation(
        self, tasks: List[Any], task_type: str
    ) -> List[Any]:
        """Execute tasks with graceful degradation for individual failures"""

        try:
            # Set timeout based on task type
            timeout_seconds = 180 if "analyses" in task_type else 120

            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True), timeout=timeout_seconds
            )

            # Filter out exceptions and None results
            successful_results = []
            failed_count = 0

            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"Task failed in {task_type}: {result}")
                    failed_count += 1
                elif result is not None:
                    successful_results.append(result)
                else:
                    failed_count += 1

            logger.info(
                f"📊 {task_type}: {len(successful_results)} succeeded, {failed_count} failed"
            )

            # Ensure minimum successful results
            if task_type == "consultant analyses" and len(successful_results) < 1:
                raise ForgeError("No consultant analyses succeeded")

            return successful_results

        except asyncio.TimeoutError:
            raise ForgeError(f"Timeout executing {task_type}")
        except Exception as e:
            raise ForgeError(f"Failed to execute {task_type}: {e}")

    def _validate_forge_results(
        self, results: ParallelForgeResults, dispatch: DispatchPackage
    ) -> None:
        """Validate forge results meet minimum requirements"""

        if results.successful_analyses < 1:
            raise ForgeError("No successful consultant analyses")

    # ===================== GOVERNANCE V2 HELPERS =====================

    def _compute_agent_contract_id(self, consultant_id: str) -> str:
        """Compute a versioned contract ID for the given consultant type.

        PATCH 1: This maps consultant_id to the correct V2 contract format
        that the routing logic expects (e.g., RISK_ASSESSOR@1.0)
        """
        return f"{consultant_id.upper()}@1.0"


# V5.3 CANONICAL COMPLIANCE: Removed DirectLLMClient and DirectResearchClient
# All LLM and Research operations now go through V5.3 Resilient Manager Pattern

# ============================================================================
# MAIN FUNCTION FOR STEP 4
# ============================================================================


async def run_parallel_forges(dispatch: DispatchPackage) -> ParallelForgeResults:
    """
    Main function for Step 4: Execute parallel forge with real LLM calls

    Args:
        dispatch: Dispatch package with selected consultants

    Returns:
        ParallelForgeResults: Combined results from all parallel executions

    Raises:
        ForgeError: If the entire process fails
    """
    orchestrator = ParallelForgeOrchestrator()
    return await orchestrator.run_parallel_forges(dispatch)
