# src/services/orchestration/nway_orchestration_service.py
import logging
from dataclasses import dataclass
from typing import Any, Dict, List

from src.orchestration.contracts import (
    StructuredAnalyticalFramework,
    NWayConfiguration,
    ConsultantBlueprint,
    FrameworkType,
)
from src.services.selection.nway_pattern_selection_service import (
    NWayPatternSelectionService,
)
from src.core.unified_context_stream import UnifiedContextStream, ContextEventType
from src.services.selection.cognitive_chemistry_engine import (
    get_cognitive_chemistry_engine,
)

logger = logging.getLogger(__name__)


@dataclass
class NwayExecutionContext:
    framework: StructuredAnalyticalFramework
    consultants: List[ConsultantBlueprint]
    task_classification: Dict[str, Any]
    current_s2_tier: str
    domain: str
    start_time: float


@dataclass
class NwayExecutionResult:
    nway_config: NWayConfiguration
    pattern_selection: Dict[str, Any]


class NwayOrchestrationService:
    """
    Encapsulates NWAY pattern selection, configuration, chemistry scoring, and Station 3 evidence logging.
    """

    def __init__(
        self,
        pattern_service: NWayPatternSelectionService,
        context_stream: UnifiedContextStream,
    ) -> None:
        self.pattern_service = pattern_service
        self.context_stream = context_stream

    async def select_and_run_nway(
        self, ctx: NwayExecutionContext
    ) -> NwayExecutionResult:
        # Pattern selection
        consultant_types = [c.consultant_type for c in ctx.consultants]
        pattern_selection = self.pattern_service.select_patterns_for_framework(
            framework_type=ctx.framework.framework_type.value,
            task_classification=ctx.task_classification,
            consultant_types=consultant_types,
            complexity=ctx.framework.complexity_assessment,
            s2_tier=ctx.current_s2_tier,
        )
        logger.info(
            f"🎯 NWAY Pattern Selection: {pattern_selection.primary_pattern} (confidence: {pattern_selection.confidence_score:.2f})"
        )
        logger.info(
            f"   Selected patterns: {', '.join(pattern_selection.selected_patterns)}"
        )
        logger.info(f"   Rationale: {pattern_selection.selection_rationale}")

        # Station 3 model selection evidence (as in orchestrator)
        selected_consultant_ids = [c.consultant_id for c in ctx.consultants]
        selection_rationale = (
            f"S2_{ctx.current_s2_tier}: Smart GM selected optimal {len(ctx.consultants)}-consultant team "
            f"for {(ctx.task_classification or {}).get('task_type', 'strategic')} in {ctx.domain} domain. Dynamic NWAY patterns: "
            f"{', '.join(pattern_selection.selected_patterns)}."
        )
        self.context_stream.add_event(
            ContextEventType.MODEL_SELECTION_JUSTIFICATION,
            {
                "selected_consultants": selected_consultant_ids,
                "chosen_nway_patterns": pattern_selection.selected_patterns,
                "primary_nway_pattern": pattern_selection.primary_pattern,
                "nway_selection_confidence": pattern_selection.confidence_score,
                "nway_selection_rationale": pattern_selection.selection_rationale,
                "s2_tier": ctx.current_s2_tier,
                "s2_rationale": getattr(ctx, "current_s2_rationale", "not determined"),
                "selection_rationale": selection_rationale,
                "confidence_score": 0.8,  # Base confidence; chemistry evidence follows
                "team_strategy": f"smart_gm_{(ctx.task_classification or {}).get('task_type', 'strategic')}",
                "domain": ctx.domain,
                "task_type": (ctx.task_classification or {}).get("task_type", "strategic"),
                "processing_time_seconds": (
                    0.0
                    if not ctx.start_time
                    else (__import__("time").time() - ctx.start_time)
                ),
            },
        )
        logger.info(
            f"📊 STATION 3 EVIDENCE: MODEL_SELECTION_JUSTIFICATION recorded with S2_{ctx.current_s2_tier}"
        )

        # Create NWAY configuration
        nway_config = await self._create_nway_configuration(
            ctx.consultants, ctx.framework
        )

        # Run Chemistry Engine and record evidence
        await self._run_chemistry_engine(ctx.framework, ctx.consultants, nway_config)

        return NwayExecutionResult(
            nway_config=nway_config,
            pattern_selection={
                "selected_patterns": pattern_selection.selected_patterns,
                "primary_pattern": pattern_selection.primary_pattern,
                "confidence_score": pattern_selection.confidence_score,
                "selection_rationale": pattern_selection.selection_rationale,
            },
        )

    async def _create_nway_configuration(
        self,
        consultants: List[ConsultantBlueprint],
        framework: StructuredAnalyticalFramework,
    ) -> NWayConfiguration:
        consultant_count = len(consultants)
        if consultant_count == 2:
            pattern_name = "dual_perspective"
            interaction_strategy = "Sequential analysis with cross-validation"
        elif consultant_count == 3:
            pattern_name = "triangulation"
            interaction_strategy = "Independent analysis with synthesis integration"
        elif consultant_count == 4:
            pattern_name = "quad_synthesis"
            interaction_strategy = "Parallel analysis with structured debate"
        else:
            pattern_name = "ensemble_analysis"
            interaction_strategy = "Multi-stage collaborative analysis"
        if framework.framework_type == FrameworkType.CRISIS_MANAGEMENT:
            interaction_strategy += " with rapid iteration cycles"
        elif framework.framework_type == FrameworkType.STRATEGIC_ANALYSIS:
            interaction_strategy += " with comprehensive market integration"
        config = NWayConfiguration(
            pattern_name=pattern_name,
            consultant_cluster=consultants,
            interaction_strategy=interaction_strategy,
        )
        logger.info(
            f"🔗 Created {pattern_name} configuration with {consultant_count} consultants"
        )
        return config

    async def _run_chemistry_engine(
        self,
        framework: StructuredAnalyticalFramework,
        consultants: List[ConsultantBlueprint],
        nway_config: NWayConfiguration,
    ) -> None:
        dims = ", ".join(d.dimension_name for d in framework.primary_dimensions)
        problem_framework = f"{framework.framework_type.value}: {dims}"
        # Infer NWAY combination from consultants and pattern
        nway_ids: List[str] = []
        for c in consultants:
            nway_ids.extend(
                self._infer_nway_clusters_for_consultant(
                    c.consultant_id, nway_config.pattern_name
                )
            )
        seen = set()
        nway_ids = [x for x in nway_ids if not (x in seen or seen.add(x))]
        nway_combination = [
            {"interaction_id": cid, "models_involved": []} for cid in nway_ids
        ]
        chem = get_cognitive_chemistry_engine(context_stream=self.context_stream)
        try:
            from src.services.selection.contracts import ChemistryContext

            ctx = ChemistryContext(
                problem_framework=problem_framework, nway_combination=nway_combination
            )
            reaction = chem.calculate_cognitive_chemistry_score(ctx)
        except Exception:
            reaction = chem.calculate_cognitive_chemistry_score(
                problem_framework=problem_framework, nway_combination=nway_combination
            )
        chem.record_selection_evidence(
            problem_framework=problem_framework,
            selected_combinations=[
                {
                    "consultants": [c.consultant_id for c in consultants],
                    "reaction": reaction,
                }
            ],
            final_score=reaction.overall_chemistry_score,
            selection_rationale=reaction.recommendation,
            risk_factors=reaction.risk_factors,
            success_factors=reaction.success_factors,
            confidence_level=reaction.confidence_level,
        )

    def _infer_nway_clusters_for_consultant(
        self, consultant_id: str, pattern_name: str
    ) -> List[str]:
        base_map = {
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
        pattern_map = {
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
        clusters = list(base_map.get(consultant_id, []))
        clusters.extend(pattern_map.get(pattern_name, []))
        seen = set()
        return [x for x in clusters if not (x in seen or seen.add(x))]
