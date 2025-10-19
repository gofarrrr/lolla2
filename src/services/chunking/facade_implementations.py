# src/services/chunking/facade_implementations.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

from src.core.chunking.contracts import (
    IChunkingStrategy,
    IChunkingEvaluator,
    IChunkingFinalizer,
    CurrencyRequirement,
)
from src.core.strategic_query_decomposer import (
    MECEDecomposition,
    get_strategic_query_decomposer,
)
from src.core.chunking_quality_monitor import (
    ProcessingContext,
    QualityAssessment,
    get_chunking_quality_monitor,
)
from src.core.unified_context_stream import UnifiedContextStream
from typing import Any, Dict, List, Optional


@dataclass
class ResearchTicket:
    """Lightweight representation of a research backlog ticket."""

    currency_requirement: CurrencyRequirement
    research_goals: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


class V1ChunkingStrategy(IChunkingStrategy):
    """Default strategy implementation backed by the strategic decomposer."""

    def __init__(self) -> None:
        self._decomposer = get_strategic_query_decomposer()

    async def decompose(
        self, query: str, user_context: Optional[Dict[str, Any]] = None
    ) -> MECEDecomposition:  # type: ignore[override]
        return await self._decomposer.decompose_query(query, user_context)

    # Optional seam for context propagation
    def set_context_stream(self, context_stream: UnifiedContextStream) -> None:
        if hasattr(self._decomposer, "set_context_stream"):
            self._decomposer.set_context_stream(context_stream)


class V1ChunkingEvaluator(IChunkingEvaluator):
    """Default evaluator implementation backed by the quality monitor."""

    def __init__(self) -> None:
        self._monitor = get_chunking_quality_monitor()

    async def assess(
        self,
        decomposition: MECEDecomposition,
        context: Optional[ProcessingContext] = None,
    ) -> QualityAssessment:  # type: ignore[override]
        return await self._monitor.assess_quality(decomposition, context)

    # Optional seam for context propagation
    def set_context_stream(self, context_stream: UnifiedContextStream) -> None:
        # No-op for now; reserved for future enhancements
        _ = context_stream


class V1ChunkingFinalizer(IChunkingFinalizer):
    """Encapsulates finalization: integration data, confidence, and recommendations."""

    def finalize(self, result: Any) -> Any:  # type: ignore[override]
        # Integration data
        result.integration_data = self._generate_integration_data(result)
        # Confidence score
        result.confidence_score = self._calculate_final_confidence(result)
        # Recommendations
        result.recommendations = self._generate_final_recommendations(result)
        result.research_ticket = self._generate_research_ticket(result)
        return result

    def _generate_integration_data(self, result: Any) -> Dict[str, Any]:
        integration_data: Dict[str, Any] = {}

        if getattr(result, "strategic_decomposition", None):
            decomp = result.strategic_decomposition
            integration_data["consultant_selection"] = {
                "complexity_indicators": len(decomp.decisions) + len(decomp.unknowns),
                "stakeholder_count": self._count_stakeholders(decomp),
                "decision_urgency": self._assess_decision_urgency(decomp),
                "knowledge_domains": self._extract_knowledge_domains(decomp),
            }
            integration_data["problem_structuring"] = {
                "constraints": [c.description for c in decomp.constraints],
                "key_decisions": [d.description for d in decomp.decisions],
                "success_criteria": [s.description for s in decomp.success_metrics],
                "risk_areas": [
                    u.description for u in decomp.unknowns if u.unknown_type == "unknown_unknown"
                ],
            }
            integration_data["devils_advocate"] = {
                "assumptions_to_challenge": [
                    u.description for u in decomp.unknowns if u.unknown_type == "assumption"
                ],
                "constraint_tensions": self._identify_constraint_tensions(decomp),
                "decision_reversibility": {
                    d.description: getattr(getattr(d, "reversibility", None), "value", getattr(d, "reversibility", "unknown"))
                    for d in decomp.decisions
                },
            }
            integration_data["senior_advisor"] = {
                "strategic_tensions": self._identify_strategic_tensions(decomp),
                "resource_implications": self._assess_resource_implications(decomp),
                "execution_readiness": self._assess_execution_readiness(decomp),
            }

        if getattr(result, "legacy_analysis", None):
            integration_data["legacy_analysis"] = {
                "intent": result.legacy_analysis.intent.value,
                "complexity": result.legacy_analysis.complexity.value,
                "keywords": result.legacy_analysis.keywords,
                "routing_suggestions": result.legacy_analysis.routing_suggestions,
            }

        return integration_data

    def _count_stakeholders(self, decomposition: MECEDecomposition) -> int:
        stakeholders = set()
        for decision in decomposition.decisions:
            stakeholders.update(getattr(decision, "stakeholders", []) or [])
        for convention in decomposition.conventions:
            stakeholders.update(getattr(convention, "stakeholders_affected", []) or [])
        return len(stakeholders)

    def _assess_decision_urgency(self, decomposition: MECEDecomposition) -> str:
        urgent_keywords = ["immediate", "urgent", "asap", "today", "now"]
        for decision in decomposition.decisions:
            timeline = (getattr(decision, "decision_timeline", "") or "").lower()
            if any(keyword in timeline for keyword in urgent_keywords):
                return "high"
        return "medium"

    def _extract_knowledge_domains(self, decomposition: MECEDecomposition) -> List[str]:
        domains = set()
        all_text = " ".join(
            [
                *[c.description for c in decomposition.constraints],
                *[c.description for c in decomposition.conventions],
                *[d.description for d in decomposition.decisions],
                *[u.description for u in decomposition.unknowns],
            ]
        )
        domain_keywords = {
            "technology": ["software", "system", "platform", "data", "AI", "ML"],
            "finance": ["budget", "cost", "investment", "revenue", "financial"],
            "marketing": ["brand", "customer", "market", "promotion", "sales"],
            "operations": ["process", "workflow", "efficiency", "operations"],
            "strategy": ["strategic", "vision", "competitive", "positioning"],
            "legal": ["compliance", "regulatory", "legal", "policy", "governance"],
        }
        for domain, keywords in domain_keywords.items():
            if any(keyword.lower() in all_text.lower() for keyword in keywords):
                domains.add(domain)
        return list(domains)

    def _identify_constraint_tensions(self, decomposition: MECEDecomposition) -> List[str]:
        tensions: List[str] = []
        constraint_types = [
            getattr(getattr(c, "constraint_type", None), "value", getattr(c, "constraint_type", ""))
            for c in decomposition.constraints
        ]
        if "economic" in constraint_types and "temporal" in constraint_types:
            tensions.append("Time vs cost trade-off tension")
        if len(decomposition.constraints) > 3:
            tensions.append("Multiple constraint optimization challenge")
        return tensions

    def _identify_strategic_tensions(self, decomposition: MECEDecomposition) -> List[str]:
        tensions: List[str] = []
        reversible_count = sum(
            1 for d in decomposition.decisions
            if getattr(getattr(d, "reversibility", None), "value", getattr(d, "reversibility", "")) == "reversible"
        )
        irreversible_count = sum(
            1 for d in decomposition.decisions
            if getattr(getattr(d, "reversibility", None), "value", getattr(d, "reversibility", "")) == "irreversible"
        )
        if reversible_count > 0 and irreversible_count > 0:
            tensions.append("Mix of reversible and irreversible decisions requires careful sequencing")
        unknown_unknowns = [u for u in decomposition.unknowns if u.unknown_type == "unknown_unknown"]
        if unknown_unknowns:
            tensions.append(f"{len(unknown_unknowns)} unknown unknowns create strategic uncertainty")
        return tensions

    def _assess_resource_implications(self, decomposition: MECEDecomposition) -> List[str]:
        implications: List[str] = []
        total_info_requirements = sum(
            len(getattr(d, "information_requirements", []) or []) for d in decomposition.decisions
        )
        if total_info_requirements > 5:
            implications.append("High information gathering requirements")
        experiments_needed = sum(1 for u in decomposition.unknowns if getattr(u, "experiment_design", None))
        if experiments_needed > 2:
            implications.append(f"{experiments_needed} experiments needed for learning")
        return implications

    def _assess_execution_readiness(self, decomposition: MECEDecomposition) -> str:
        unknown_count = len(decomposition.unknowns)
        missing_info = sum(len(getattr(d, "information_requirements", []) or []) for d in decomposition.decisions)
        if unknown_count == 0 and missing_info == 0:
            return "high"
        elif unknown_count <= 2 and missing_info <= 3:
            return "medium"
        else:
            return "low"

    def _calculate_final_confidence(self, result: Any) -> float:
        confidence_factors: List[float] = []
        if getattr(result, "strategic_decomposition", None):
            confidence_factors.append(result.strategic_decomposition.confidence_score)
        if getattr(result, "legacy_analysis", None):
            confidence_factors.append(result.legacy_analysis.confidence_score)
        if getattr(result, "quality_assessment", None):
            confidence_factors.append(result.quality_assessment.overall_quality_score)
        if not confidence_factors:
            return 0.5
        return sum(confidence_factors) / len(confidence_factors)

    def _generate_research_ticket(self, result: Any) -> Optional[ResearchTicket]:
        """Create a research ticket capturing highest-value unanswered questions."""
        decomposition = getattr(result, "strategic_decomposition", None)
        if not decomposition:
            return None

        # Determine currency requirement based on query urgency heuristics
        query_text = (getattr(result, "original_query", "") or "").lower()
        urgent_tokens = ["latest", "current", "today", "recent", "now"]
        currency = (
            CurrencyRequirement.HIGH
            if any(token in query_text for token in urgent_tokens)
            else CurrencyRequirement.MEDIUM
        )

        # Assemble research goals from unknowns and explicit information requirements
        goals: List[str] = []
        for unknown in decomposition.unknowns:
            goals.append(unknown.description)
        for decision in decomposition.decisions:
            for requirement in getattr(decision, "information_requirements", []) or []:
                goals.append(requirement)

        if not goals:
            return None

        return ResearchTicket(currency_requirement=currency, research_goals=goals)

    def _generate_final_recommendations(self, result: Any) -> List[str]:
        recommendations: List[str] = []
        if getattr(result, "quality_assessment", None):
            recommendations.extend(result.quality_assessment.improvement_recommendations)
        if getattr(result, "mode_used", None) and result.mode_used.value == "strategic":
            if getattr(result, "strategic_decomposition", None):
                experiment_count = sum(1 for u in result.strategic_decomposition.unknowns if u.experiment_design)
                if experiment_count > 0:
                    recommendations.append(
                        f"Execute {experiment_count} designed experiments to reduce unknowns"
                    )
        dev_adv = result.integration_data.get("devils_advocate", {}) if getattr(result, "integration_data", None) else {}
        assumptions = dev_adv.get("assumptions_to_challenge", [])
        if assumptions:
            recommendations.append(
                f"Challenge {len(assumptions)} key assumptions with devils advocate"
            )
        return recommendations
