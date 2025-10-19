# src/services/critique/facade_implementations.py
from __future__ import annotations

from typing import Dict, Any, List

from src.core.critique.contracts import (
    ICritiquePreparer,
    ICritiqueRunner,
    ICritiqueSynthesizer,
)

# Engines and models from the legacy system (now used by the runner)
from src.engine.core.munger_bias_detector import MungerBiasDetector, BiasDetectionResult
from src.engine.core.ackoff_assumption_dissolver import (
    AckoffAssumptionDissolver,
    AssumptionDissolveResult,
)
from src.engine.core.cognitive_audit_engine import (
    CognitiveAuditEngine,
    CognitiveAuditResult,
)
from src.engine.core.llm_sceptic_engine import LLMScepticEngine
from src.engine.integrations.perplexity_client import (
    get_perplexity_client,
    KnowledgeQueryType,
)
from src.core.enhanced_devils_advocate_system import (
    DevilsAdvocateChallenge,
    ComprehensiveChallengeResult,
    ChallengeEngine,
)


class V1CritiquePreparer(ICritiquePreparer):
    """Extracts recommendation and context from analysis_results for critique run."""

    async def prepare(
        self, analysis_results: List[Any], context_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        # This mirrors the recommendation assembly logic from the legacy run_enhanced_critique
        if analysis_results:
            recommendation_parts: List[str] = []
            for result in analysis_results:
                if isinstance(result, dict):
                    for key in [
                        "recommendations",
                        "analysis_content",
                        "key_insights",
                        "findings",
                    ]:
                        if key in result and result[key]:
                            if isinstance(result[key], list):
                                recommendation_parts.extend(
                                    [str(item) for item in result[key]]
                                )
                            else:
                                recommendation_parts.append(str(result[key]))
                else:
                    recommendation_parts.append(str(result))
            combined_recommendation = "; ".join(
                recommendation_parts[:5]
            )  # Bound for processing
        else:
            combined_recommendation = (
                "Strategic analysis and recommendations for business optimization"
            )
        return {
            "recommendation": combined_recommendation,
            "context": context_data,
            "analysis_results": analysis_results,
        }


class V1CritiqueRunner(ICritiqueRunner):
    """Executes the comprehensive challenge analysis (extracted from legacy)."""

    def __init__(self) -> None:
        # Initialize engines
        self.munger_detector = MungerBiasDetector()
        self.ackoff_dissolver = AckoffAssumptionDissolver()
        self.cognitive_auditor = CognitiveAuditEngine()
        self.llm_sceptic = LLMScepticEngine()
        # Research grounding
        self.perplexity_client = None
        self.enable_research_grounding = True

    async def run(self, prepared_payload: Dict[str, Any]) -> ComprehensiveChallengeResult:  # type: ignore[override]
        recommendation: str = prepared_payload.get("recommendation", "")
        business_context: Dict[str, Any] = prepared_payload.get("context", {})

        # Initialize Perplexity research grounding if enabled
        if self.enable_research_grounding and not self.perplexity_client:
            try:
                self.perplexity_client = await get_perplexity_client()
            except Exception:
                self.enable_research_grounding = False

        # Step 1: Research grounding
        research_context: Dict[str, Any] = {}
        if self.enable_research_grounding and self.perplexity_client:
            research_context = await self._conduct_research_grounding(
                recommendation, business_context
            )

        # Step 2: Run all engines
        enhanced_context = {**business_context, "research_grounding": research_context}

        m_task = self.munger_detector.detect_bias_patterns(
            recommendation, enhanced_context
        )
        a_task = self.ackoff_dissolver.dissolve_assumptions(
            recommendation, enhanced_context
        )
        c_task = self.cognitive_auditor.audit_motivated_reasoning(
            recommendation, enhanced_context
        )
        s_task = self.llm_sceptic.find_creative_flaws(recommendation, enhanced_context)

        munger_result, ackoff_result, audit_result, llm_sceptic_result = (
            await __import__("asyncio").gather(m_task, a_task, c_task, s_task)
        )

        # Consolidate challenges
        all_challenges: List[DevilsAdvocateChallenge] = []
        severity_threshold = 0.6

        for bias in munger_result.detected_biases:
            if bias.severity >= severity_threshold:
                all_challenges.append(
                    DevilsAdvocateChallenge(
                        challenge_id=f"munger_{len(all_challenges)+1}",
                        challenge_type="munger_bias",
                        challenge_text=f"Bias Alert - {bias.bias_type}: {bias.description}",
                        severity=bias.severity,
                        evidence=bias.evidence_examples,
                        mitigation_strategy=bias.mitigation_approach,
                        source_engine=ChallengeEngine.MUNGER_BIAS.value,
                    )
                )

        for assumption in ackoff_result.dissolved_assumptions:
            if assumption.dissolution_strength >= severity_threshold:
                all_challenges.append(
                    DevilsAdvocateChallenge(
                        challenge_id=f"ackoff_{len(all_challenges)+1}",
                        challenge_type="ackoff_dissolution",
                        challenge_text=f"Assumption Challenge: {assumption.assumption_text}",
                        severity=assumption.dissolution_strength,
                        evidence=assumption.alternative_framings,
                        mitigation_strategy=assumption.idealized_design_approach,
                        source_engine=ChallengeEngine.ACKOFF_DISSOLUTION.value,
                    )
                )

        for pattern in audit_result.motivated_reasoning_patterns:
            if pattern.severity >= severity_threshold:
                all_challenges.append(
                    DevilsAdvocateChallenge(
                        challenge_id=f"audit_{len(all_challenges)+1}",
                        challenge_type="cognitive_audit",
                        challenge_text=f"Motivated Reasoning - {pattern.pattern_type}: {pattern.description}",
                        severity=pattern.severity,
                        evidence=pattern.evidence,
                        mitigation_strategy=pattern.mitigation_strategy,
                        source_engine=ChallengeEngine.COGNITIVE_AUDIT.value,
                    )
                )

        for challenge in llm_sceptic_result.sceptic_challenges:
            if challenge.severity >= severity_threshold:
                all_challenges.append(
                    DevilsAdvocateChallenge(
                        challenge_id=f"llm_sceptic_{len(all_challenges)+1}",
                        challenge_type="llm_sceptic",
                        challenge_text=f"Creative Flaw - {challenge.challenge_type}: {challenge.challenge_text}",
                        severity=challenge.severity,
                        evidence=[challenge.evidence_basis, challenge.counter_argument],
                        mitigation_strategy=challenge.mitigation_strategy,
                        source_engine=ChallengeEngine.LLM_SCEPTIC.value,
                    )
                )

        # Calculate summary metrics
        all_challenges.sort(key=lambda x: x.severity, reverse=True)
        critical_challenges = all_challenges[:10]
        overall_risk_score = self._calculate_overall_risk(all_challenges)
        intellectual_honesty_score = 1.0 - audit_result.overall_bias_score
        system_confidence = self._calculate_system_confidence(
            munger_result, ackoff_result, audit_result
        )

        # Refined recommendation
        refined_recommendation = await self._refine_recommendation(
            recommendation, critical_challenges, business_context
        )

        # Build result
        return ComprehensiveChallengeResult(
            original_recommendation=recommendation,
            total_challenges_found=len(all_challenges),
            critical_challenges=critical_challenges,
            overall_risk_score=overall_risk_score,
            refined_recommendation=refined_recommendation,
            intellectual_honesty_score=intellectual_honesty_score,
            system_confidence=system_confidence,
            processing_details={
                "munger_biases_detected": len(munger_result.detected_biases),
                "ackoff_assumptions_dissolved": len(
                    ackoff_result.dissolved_assumptions
                ),
                "audit_patterns_found": len(audit_result.motivated_reasoning_patterns),
                "llm_sceptic_challenges_found": len(
                    llm_sceptic_result.sceptic_challenges
                ),
                "engines_used": [
                    ChallengeEngine.MUNGER_BIAS.value,
                    ChallengeEngine.ACKOFF_DISSOLUTION.value,
                    ChallengeEngine.COGNITIVE_AUDIT.value,
                    ChallengeEngine.LLM_SCEPTIC.value,
                ],
            },
        )

    async def _conduct_research_grounding(
        self, recommendation: str, business_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        research_context: Dict[str, Any] = {
            "fact_checks": [],
            "market_intelligence": {},
            "contradictory_evidence": [],
            "credibility_assessment": {},
        }
        try:
            key_claims = self._extract_key_claims(recommendation)
            for claim in key_claims[:3]:
                fact_check = await self.perplexity_client.fact_check_claim(
                    claim=claim, domain=business_context.get("industry", "business")
                )
                research_context["fact_checks"].append(
                    {
                        "claim": claim,
                        "validation": fact_check.content,
                        "confidence": fact_check.confidence,
                        "sources": fact_check.sources,
                    }
                )
            industry = business_context.get("industry")
            if industry:
                market_intel = await self.perplexity_client.ground_context(
                    industry=industry,
                    problem_domain=business_context.get("domain", "strategic planning"),
                )
                research_context["market_intelligence"] = {
                    "content": market_intel.content,
                    "confidence": market_intel.confidence,
                    "sources": market_intel.sources,
                }
            contradiction_query = f"Evidence against: {recommendation[:200]}"
            contradiction_research = await self.perplexity_client.query_knowledge(
                query=contradiction_query,
                query_type=KnowledgeQueryType.FACT_CHECKING,
                max_tokens=800,
            )
            research_context["contradictory_evidence"] = [
                contradiction_research.content
            ]
        except Exception as e:  # noqa: BLE001
            research_context["error"] = str(e)
        return research_context

    def _extract_key_claims(self, recommendation: str) -> List[str]:
        claims: List[str] = []
        sentences = recommendation.split(".")
        claim_indicators = [
            "studies show",
            "research indicates",
            "data shows",
            "market research",
            "according to",
            "statistics show",
            "benchmarks indicate",
            "industry data",
            "competitors are",
            "market leaders",
            "best practices",
            "proven strategy",
        ]
        for sentence in sentences:
            s = sentence.strip()
            if len(s) > 20:
                if any(indicator in s.lower() for indicator in claim_indicators):
                    claims.append(s)
                    continue
                if any(
                    word in s.lower()
                    for word in [
                        "will increase",
                        "will decrease",
                        "will improve",
                        "always",
                        "never",
                    ]
                ):
                    claims.append(s)
        return claims[:5]

    def _calculate_overall_risk(
        self, challenges: List[DevilsAdvocateChallenge]
    ) -> float:
        if not challenges:
            return 0.0
        total_weighted_severity = 0.0
        weight_sum = 0.0
        for i, challenge in enumerate(challenges):
            weight = 1.0 / (i + 1)
            total_weighted_severity += challenge.severity * weight
            weight_sum += weight
        return min(1.0, total_weighted_severity / weight_sum) if weight_sum > 0 else 0.0

    def _calculate_system_confidence(
        self,
        munger_result: BiasDetectionResult,
        ackoff_result: AssumptionDissolveResult,
        audit_result: CognitiveAuditResult,
    ) -> float:
        bias_confidence = 1.0 - (munger_result.overall_bias_risk / 1.0)
        assumption_confidence = 1.0 - (ackoff_result.dissolution_impact_score / 1.0)
        audit_confidence = 1.0 - (audit_result.overall_bias_score / 1.0)
        return max(
            0.0,
            min(
                1.0,
                bias_confidence * 0.3
                + assumption_confidence * 0.3
                + audit_confidence * 0.4,
            ),
        )

    async def _refine_recommendation(
        self,
        original_recommendation: str,
        critical_challenges: List[DevilsAdvocateChallenge],
        business_context: Dict[str, Any],
    ) -> str:
        if not critical_challenges:
            return original_recommendation
        challenge_themes: Dict[str, List[str]] = {}
        for challenge in critical_challenges[:5]:
            challenge_themes.setdefault(challenge.challenge_type, []).append(
                challenge.challenge_text
            )
        refinements: List[str] = []
        if "munger_bias" in challenge_themes:
            refinements.append(
                "Consider cognitive biases and apply systematic bias mitigation"
            )
        if "ackoff_dissolution" in challenge_themes:
            refinements.append(
                "Re-examine fundamental assumptions through idealized design approach"
            )
        if "cognitive_audit" in challenge_themes:
            refinements.append(
                "Address motivated reasoning patterns with devils advocate challenges"
            )
        refined_text = f"{original_recommendation}\n\nIMPORTANT CONSIDERATIONS:\n"
        for i, refinement in enumerate(refinements, 1):
            refined_text += f"{i}. {refinement}\n"
        if critical_challenges:
            refined_text += "\nSPECIFIC RISK MITIGATIONS:\n"
            for i, challenge in enumerate(critical_challenges[:3], 1):
                refined_text += f"{i}. {challenge.mitigation_strategy}\n"
        return refined_text


class V1CritiqueSynthesizer(ICritiqueSynthesizer):
    """Maps the runner's result to the facade's expected return payload."""

    def synthesize(self, raw_results: Any) -> Dict[str, Any]:  # type: ignore[override]
        # Expect a ComprehensiveChallengeResult
        result = raw_results
        return {
            "challenges": [c.challenge_text for c in result.critical_challenges],
            "bias_warnings": [
                c.challenge_text
                for c in result.critical_challenges
                if c.challenge_type == "munger_bias"
            ],
            "assumption_challenges": [
                c.challenge_text
                for c in result.critical_challenges
                if c.challenge_type == "ackoff_dissolution"
            ],
            "conflicts": [
                c.challenge_text
                for c in result.critical_challenges
                if c.challenge_type in ["cognitive_audit", "llm_sceptic"]
            ],
            "critique_strength": result.overall_risk_score,
            "total_challenges_found": result.total_challenges_found,
            "intellectual_honesty_score": result.intellectual_honesty_score,
            "system_confidence": result.system_confidence,
            "refined_recommendation": result.refined_recommendation,
            "method_actor_enhanced": False,
        }
