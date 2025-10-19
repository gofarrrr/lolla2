"""
Enhanced Final Synthesis Engine with Arbitration Integration
Purpose: Generate final synthesis that prioritizes user-selected critiques and incorporates user feedback

This enhanced synthesis engine:
1. Consumes rich arbitration data (prioritized/disagreed/neutral dispositions)
2. Gives highest priority to user-generated critiques
3. Explicitly addresses prioritized critiques in the synthesis
4. Acknowledges disagreed critiques with context
5. Creates a dedicated "Arbitration Response" section
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from src.engine.adapters.logging import get_logger  # Migrated
from src.engine.models.data_contracts import MetisDataContract
from src.engine.engines.base_engine import BaseEngine
from src.engine.adapters.llm_client import LLMClient  # Migrated

logger = get_logger(__name__, component="enhanced_final_synthesis")


class SynthesisPriority(Enum):
    """Priority levels for synthesis content"""

    USER_GENERATED = 1  # Highest priority
    USER_PRIORITIZED = 2
    NEUTRAL = 3
    USER_DISAGREED = 4  # Lowest priority


@dataclass
class ArbitrationContext:
    """Structured arbitration context for synthesis"""

    user_generated_critique: Optional[Dict[str, Any]]
    prioritized_critiques: List[Dict[str, Any]]
    disagreed_critiques: List[Dict[str, Any]]
    neutral_critiques: List[Dict[str, Any]]
    engagement_level: str
    has_rationales: bool
    total_critiques: int

    @property
    def has_user_feedback(self) -> bool:
        """Check if user provided any active feedback"""
        return bool(
            self.user_generated_critique
            or self.prioritized_critiques
            or self.disagreed_critiques
        )

    @property
    def prioritized_count(self) -> int:
        """Get count of prioritized critiques"""
        return len(self.prioritized_critiques)

    @property
    def disagreed_count(self) -> int:
        """Get count of disagreed critiques"""
        return len(self.disagreed_critiques)


class EnhancedFinalSynthesisEngine(BaseEngine):
    """
    Enhanced synthesis engine that incorporates user arbitration
    Operation Ground Truth - Day 5: Explicit conflict resolution
    """

    def __init__(self, llm_client: LLMClient):
        super().__init__(llm_client)
        self.logger = logger.with_component("enhanced_synthesis")
        self.synthesis_version = (
            "3.0"  # Version tracking - now with explicit resolution
        )

        # OPERATION GROUND TRUTH: Explicit weighting system
        self.conflict_weights = {
            "user_critique": 0.35,  # Highest weight - user input paramount
            "evidence_based": 0.30,  # Facts from research matter
            "red_team": 0.20,  # Devil's advocate insights
            "initial_analysis": 0.15,  # Original analysis can be wrong
        }

    async def generate_arbitrated_synthesis(
        self, contract: MetisDataContract
    ) -> Dict[str, Any]:
        """
        Generate final synthesis that incorporates user arbitration

        Returns synthesis with explicit sections addressing:
        1. User-generated critiques (if any)
        2. Prioritized critiques
        3. Acknowledgment of disagreed critiques
        4. Core recommendations with critique integration
        """

        try:
            # Extract arbitration context
            arbitration_context = self._extract_arbitration_context(contract)

            # Log synthesis initiation
            self.logger.info(
                "enhanced_synthesis_initiated",
                engagement_id=str(contract.engagement_id),
                has_user_critique=bool(arbitration_context.user_generated_critique),
                prioritized_count=arbitration_context.prioritized_count,
                disagreed_count=arbitration_context.disagreed_count,
                engagement_level=arbitration_context.engagement_level,
            )

            # Generate synthesis components in parallel where possible
            synthesis_components = await self._generate_synthesis_components(
                contract, arbitration_context
            )

            # Compose final synthesis document
            final_synthesis = self._compose_final_synthesis(
                synthesis_components, arbitration_context
            )

            # Update contract with synthesis
            contract.final_synthesis = final_synthesis

            # Log completion with metrics
            self._log_synthesis_metrics(
                contract.engagement_id, final_synthesis, arbitration_context
            )

            return final_synthesis

        except Exception as e:
            self.logger.error(
                "enhanced_synthesis_generation_failed",
                engagement_id=str(contract.engagement_id),
                error=str(e),
            )
            raise

    def _extract_arbitration_context(
        self, contract: MetisDataContract
    ) -> ArbitrationContext:
        """Extract and structure arbitration context from contract"""

        # Default empty context
        if not hasattr(contract, "validation_results"):
            return ArbitrationContext(
                user_generated_critique=None,
                prioritized_critiques=[],
                disagreed_critiques=[],
                neutral_critiques=[],
                engagement_level="minimal",
                has_rationales=False,
                total_critiques=0,
            )

        # Extract enhanced arbitration data
        arbitration_data = contract.validation_results.get(
            "enhanced_user_arbitration", {}
        )

        # Check for rationales in disagreed critiques
        disagreed = arbitration_data.get("disagreed_critiques", [])
        has_rationales = any(critique.get("rationale") for critique in disagreed)

        # Get engagement level from flywheel metadata
        flywheel_meta = arbitration_data.get("flywheel_metadata", {})
        engagement_level = flywheel_meta.get("user_engagement_level", "minimal")

        return ArbitrationContext(
            user_generated_critique=arbitration_data.get("user_generated_critique"),
            prioritized_critiques=arbitration_data.get("prioritized_critiques", []),
            disagreed_critiques=disagreed,
            neutral_critiques=arbitration_data.get("neutral_critiques", []),
            engagement_level=engagement_level,
            has_rationales=has_rationales,
            total_critiques=arbitration_data.get("total_available_critiques", 0),
        )

    async def _generate_synthesis_components(
        self, contract: MetisDataContract, arbitration_context: ArbitrationContext
    ) -> Dict[str, Any]:
        """Generate individual synthesis components"""

        # OPERATION GROUND TRUTH: First resolve conflicts
        conflict_resolution = await self.resolve_conflicts(
            contract.analysis_results.get("initial_analysis", {}),
            (
                contract.validation_results
                if hasattr(contract, "validation_results")
                else {}
            ),
            (
                contract.research_results.get("fact_pack")
                if hasattr(contract, "research_results")
                else None
            ),
            arbitration_context,
        )

        components = {"conflict_resolution": conflict_resolution}

        # 1. Generate user critique response (highest priority)
        if arbitration_context.user_generated_critique:
            components["user_critique_response"] = (
                await self._generate_user_critique_response(
                    arbitration_context.user_generated_critique, contract
                )
            )

        # 2. Generate prioritized critique integration
        if arbitration_context.prioritized_critiques:
            components["prioritized_integration"] = (
                await self._generate_prioritized_integration(
                    arbitration_context.prioritized_critiques, contract
                )
            )

        # 3. Generate core synthesis with critique awareness
        components["core_synthesis"] = await self._generate_core_synthesis(
            contract, arbitration_context
        )

        # 4. Generate disagreement acknowledgments
        if (
            arbitration_context.disagreed_critiques
            and arbitration_context.has_rationales
        ):
            components["disagreement_acknowledgment"] = (
                self._generate_disagreement_acknowledgment(
                    arbitration_context.disagreed_critiques
                )
            )

        # 5. Generate implementation roadmap with critique mitigations
        components["implementation_roadmap"] = (
            await self._generate_implementation_roadmap(contract, arbitration_context)
        )

        return components

    async def _generate_user_critique_response(
        self, user_critique: Dict[str, Any], contract: MetisDataContract
    ) -> Dict[str, Any]:
        """Generate specific response to user-generated critique"""

        prompt = f"""
        The user has provided the following custom critique that must be addressed as the highest priority:
        
        User Critique: {user_critique['description']}
        
        Context from analysis:
        - Problem Statement: {contract.problem_statement}
        - Key Findings: {json.dumps(contract.analysis_results.get('key_findings', []), indent=2)}
        
        Generate a comprehensive response that:
        1. Acknowledges the validity and importance of this critique
        2. Explains how it changes or enhances our recommendations
        3. Provides specific actions to address the critique
        4. Identifies any trade-offs or considerations
        
        Format as a structured response with clear sections.
        """

        response = await self.llm_client.generate(
            prompt, temperature=0.3, max_tokens=1000
        )

        return {
            "critique_id": user_critique["critique_id"],
            "response": response,
            "priority": SynthesisPriority.USER_GENERATED.value,
            "integration_points": self._extract_integration_points(response),
        }

    async def _generate_prioritized_integration(
        self, prioritized_critiques: List[Dict[str, Any]], contract: MetisDataContract
    ) -> Dict[str, Any]:
        """Generate integration plan for prioritized critiques"""

        # Sort by significance
        sorted_critiques = sorted(
            prioritized_critiques,
            key=lambda x: x.get("critique_details", {}).get("significance", 0),
            reverse=True,
        )

        critique_summaries = []
        for critique in sorted_critiques:
            details = critique.get("critique_details", {})
            critique_summaries.append(
                {
                    "title": details.get("title", "Unknown"),
                    "description": details.get("description", ""),
                    "type": details.get("type", ""),
                    "source": details.get("source_challenger", ""),
                }
            )

        prompt = f"""
        The user has prioritized the following critiques for integration into the final recommendations:
        
        {json.dumps(critique_summaries, indent=2)}
        
        Context:
        - Problem: {contract.problem_statement}
        - Original Recommendations: {json.dumps(contract.analysis_results.get('recommendations', []), indent=2)}
        
        Generate an integration plan that:
        1. Shows how each prioritized critique modifies the recommendations
        2. Identifies synergies between critiques
        3. Proposes specific mitigation strategies
        4. Maintains recommendation coherence while addressing critiques
        
        Be specific and actionable.
        """

        response = await self.llm_client.generate(
            prompt, temperature=0.3, max_tokens=1500
        )

        return {
            "integration_plan": response,
            "critique_count": len(prioritized_critiques),
            "priority": SynthesisPriority.USER_PRIORITIZED.value,
            "modification_summary": self._summarize_modifications(response),
        }

    async def _generate_core_synthesis(
        self, contract: MetisDataContract, arbitration_context: ArbitrationContext
    ) -> Dict[str, Any]:
        """Generate core synthesis with critique awareness"""

        # Build critique awareness context
        critique_context = self._build_critique_context(arbitration_context)

        prompt = f"""
        Generate a comprehensive synthesis for this consulting engagement.
        
        ENGAGEMENT CONTEXT:
        - Problem: {contract.problem_statement}
        - Stakeholders: {json.dumps(contract.stakeholders, indent=2)}
        - Constraints: {json.dumps(contract.constraints, indent=2)}
        
        ANALYSIS RESULTS:
        {json.dumps(contract.analysis_results, indent=2)}
        
        CRITIQUE CONTEXT:
        {critique_context}
        
        USER ENGAGEMENT LEVEL: {arbitration_context.engagement_level}
        
        Generate a synthesis that:
        1. Provides clear, actionable recommendations
        2. Explicitly addresses user-prioritized concerns
        3. Acknowledges areas of disagreement respectfully
        4. Maintains analytical rigor while incorporating feedback
        5. Includes risk mitigation for identified failure modes
        6. Provides implementation guidance with critique awareness
        
        Structure the synthesis with:
        - Executive Summary (with critique integration highlights)
        - Key Recommendations (modified based on prioritized critiques)
        - Risk Mitigation Strategies
        - Implementation Roadmap
        - Success Metrics
        
        Ensure the tone is professional, balanced, and acknowledges the collaborative nature of the critique process.
        """

        response = await self.llm_client.generate(
            prompt, temperature=0.4, max_tokens=3000
        )

        return {
            "synthesis": response,
            "critique_integration_level": self._calculate_integration_level(
                response, arbitration_context
            ),
            "version": self.synthesis_version,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _generate_disagreement_acknowledgment(
        self, disagreed_critiques: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate acknowledgment of disagreed critiques"""

        acknowledgments = []

        for critique in disagreed_critiques:
            rationale = critique.get("rationale", "")
            details = critique.get("critique_details", {})

            acknowledgment = {
                "critique_title": details.get("title", "Unknown"),
                "original_concern": details.get("description", ""),
                "user_rationale": (
                    rationale if rationale else "No specific rationale provided"
                ),
                "acknowledgment": self._craft_acknowledgment(details, rationale),
            }
            acknowledgments.append(acknowledgment)

        return {
            "acknowledgments": acknowledgments,
            "total_disagreed": len(disagreed_critiques),
            "rationale_rate": sum(1 for c in disagreed_critiques if c.get("rationale"))
            / len(disagreed_critiques)
            * 100,
        }

    async def _generate_implementation_roadmap(
        self, contract: MetisDataContract, arbitration_context: ArbitrationContext
    ) -> Dict[str, Any]:
        """Generate implementation roadmap with critique mitigations"""

        # Collect all critique mitigations needed
        mitigations_needed = []

        # Add user-generated critique mitigations
        if arbitration_context.user_generated_critique:
            mitigations_needed.append(
                {
                    "source": "user_generated",
                    "priority": "highest",
                    "description": arbitration_context.user_generated_critique[
                        "description"
                    ],
                }
            )

        # Add prioritized critique mitigations
        for critique in arbitration_context.prioritized_critiques:
            details = critique.get("critique_details", {})
            mitigations_needed.append(
                {
                    "source": details.get("source_challenger", ""),
                    "priority": "high",
                    "description": details.get("description", ""),
                }
            )

        prompt = f"""
        Generate a detailed implementation roadmap that incorporates the following critique mitigations:
        
        {json.dumps(mitigations_needed, indent=2)}
        
        Context:
        - Recommendations: {json.dumps(contract.analysis_results.get('recommendations', []), indent=2)}
        - Constraints: {json.dumps(contract.constraints, indent=2)}
        
        Create a phased roadmap with:
        1. Quick wins (0-30 days) - especially addressing user concerns
        2. Medium-term initiatives (1-3 months)
        3. Long-term transformations (3-12 months)
        
        For each phase, include:
        - Specific actions
        - Critique mitigations addressed
        - Success metrics
        - Risk factors
        - Dependencies
        """

        response = await self.llm_client.generate(
            prompt, temperature=0.3, max_tokens=2000
        )

        return {
            "roadmap": response,
            "mitigation_count": len(mitigations_needed),
            "user_critique_addressed": bool(
                arbitration_context.user_generated_critique
            ),
            "phases": self._extract_phases(response),
        }

    def _compose_final_synthesis(
        self, components: Dict[str, Any], arbitration_context: ArbitrationContext
    ) -> Dict[str, Any]:
        """Compose final synthesis document from components"""

        synthesis = {
            "version": self.synthesis_version,
            "generated_at": datetime.utcnow().isoformat(),
            "arbitration_summary": {
                "user_engagement_level": arbitration_context.engagement_level,
                "has_user_critique": bool(arbitration_context.user_generated_critique),
                "prioritized_count": arbitration_context.prioritized_count,
                "disagreed_count": arbitration_context.disagreed_count,
                "total_critiques": arbitration_context.total_critiques,
            },
            # OPERATION GROUND TRUTH: Include conflict resolution details
            "conflict_resolution": components.get("conflict_resolution", {}),
        }

        # Add sections in priority order
        sections = []

        # 1. User Critique Response (if exists)
        if "user_critique_response" in components:
            sections.append(
                {
                    "title": "Response to User-Generated Critique",
                    "priority": "highest",
                    "content": components["user_critique_response"],
                }
            )

        # 2. Prioritized Critique Integration
        if "prioritized_integration" in components:
            sections.append(
                {
                    "title": "Integration of Prioritized Critiques",
                    "priority": "high",
                    "content": components["prioritized_integration"],
                }
            )

        # 3. Core Synthesis
        sections.append(
            {
                "title": "Comprehensive Synthesis",
                "priority": "standard",
                "content": components["core_synthesis"],
            }
        )

        # 4. Disagreement Acknowledgments
        if "disagreement_acknowledgment" in components:
            sections.append(
                {
                    "title": "Acknowledgment of Alternative Perspectives",
                    "priority": "informational",
                    "content": components["disagreement_acknowledgment"],
                }
            )

        # 5. Implementation Roadmap
        sections.append(
            {
                "title": "Implementation Roadmap with Risk Mitigations",
                "priority": "high",
                "content": components["implementation_roadmap"],
            }
        )

        synthesis["sections"] = sections

        # Add metadata for Flywheel learning
        synthesis["flywheel_metadata"] = {
            "user_feedback_incorporated": arbitration_context.has_user_feedback,
            "synthesis_adaptation_level": self._calculate_adaptation_level(
                arbitration_context
            ),
            "critique_response_mapping": self._generate_critique_response_mapping(
                components, arbitration_context
            ),
        }

        return synthesis

    async def resolve_conflicts(
        self,
        analysis: Dict[str, Any],
        critiques: Dict[str, Any],
        evidence: Any,
        arbitration_context: ArbitrationContext,
    ) -> Dict[str, Any]:
        """
        OPERATION GROUND TRUTH - Day 5: Explicit conflict resolution algorithm

        Resolves conflicts between initial analysis and various critiques
        using weighted scoring and transparent decision-making.
        """

        conflicts = []
        resolutions = []

        # 1. Detect conflicts
        detected_conflicts = self._detect_conflicts(analysis, critiques, evidence)

        # 2. Apply resolution strategy for each conflict
        for conflict in detected_conflicts:
            resolution = self._apply_resolution_strategy(
                conflict, self.conflict_weights, confidence_threshold=0.7
            )
            resolutions.append(resolution)

            # Log conflict resolution
            self.logger.info(
                "conflict_resolved",
                conflict_type=conflict["type"],
                resolution_method=resolution["method"],
                weights_applied=resolution["weights_used"],
            )

        # 3. Generate resolution summary
        resolution_summary = {
            "conflicts_detected": len(detected_conflicts),
            "resolutions": resolutions,
            "weights_applied": self.conflict_weights,
            "resolution_confidence": self._calculate_resolution_confidence(resolutions),
            "methodology": "weighted_arbitration",
            "version": self.synthesis_version,
            "timestamp": datetime.utcnow().isoformat(),
        }

        return resolution_summary

    def _detect_conflicts(
        self, analysis: Dict[str, Any], critiques: Dict[str, Any], evidence: Any
    ) -> List[Dict[str, Any]]:
        """Detect conflicts between analysis, critiques, and evidence"""

        conflicts = []

        # Check for Munger failure mode conflicts
        if "munger_critique" in critiques:
            munger = critiques["munger_critique"]
            if munger.get("critiques"):
                for critique in munger["critiques"]:
                    if critique.get("type") == "failure_mode":
                        # Check if analysis acknowledges this failure mode
                        if not self._is_risk_acknowledged(
                            analysis, critique["description"]
                        ):
                            conflicts.append(
                                {
                                    "type": "failure_mode_conflict",
                                    "source": "munger",
                                    "analysis_position": analysis.get(
                                        "confidence_score", 0.8
                                    ),
                                    "critique_position": critique.get(
                                        "significance", 0.5
                                    ),
                                    "description": critique["description"],
                                }
                            )

        # Check for evidence conflicts
        if evidence and hasattr(evidence, "assertions"):
            for assertion in evidence.assertions[:3]:  # Check top 3 facts
                if assertion.confidence > 0.8:
                    # Check if high-confidence fact contradicts analysis
                    if self._contradicts_analysis(analysis, assertion.claim):
                        conflicts.append(
                            {
                                "type": "evidence_conflict",
                                "source": "research",
                                "analysis_position": analysis.get(
                                    "confidence_score", 0.8
                                ),
                                "evidence_confidence": assertion.confidence,
                                "description": f"Evidence: {assertion.claim}",
                            }
                        )

        # Check for assumption conflicts (Ackoff)
        if "ackoff_critique" in critiques:
            ackoff = critiques["ackoff_critique"]
            if ackoff.get("critiques"):
                for critique in ackoff["critiques"]:
                    if critique.get("type") == "assumption":
                        conflicts.append(
                            {
                                "type": "assumption_conflict",
                                "source": "ackoff",
                                "description": critique["description"],
                                "significance": critique.get("significance", 0.5),
                            }
                        )

        return conflicts

    def _apply_resolution_strategy(
        self,
        conflict: Dict[str, Any],
        weights: Dict[str, float],
        confidence_threshold: float,
    ) -> Dict[str, Any]:
        """Apply weighted resolution strategy to a conflict"""

        # Calculate weighted score for each position
        scores = {}

        # Map conflict sources to weight categories
        source_weight_map = {
            "munger": weights["red_team"],
            "ackoff": weights["red_team"],
            "bias": weights["red_team"],
            "research": weights["evidence_based"],
            "user": weights["user_critique"],
            "analysis": weights["initial_analysis"],
        }

        # Calculate resolution
        conflict_source = conflict.get("source", "unknown")
        source_weight = source_weight_map.get(conflict_source, 0.1)

        # Determine resolution
        if conflict["type"] == "failure_mode_conflict":
            # Failure modes should be taken seriously
            resolution_method = (
                "accept_critique" if source_weight > 0.15 else "acknowledge_risk"
            )
        elif conflict["type"] == "evidence_conflict":
            # Evidence conflicts favor facts
            resolution_method = (
                "defer_to_evidence"
                if conflict.get("evidence_confidence", 0) > confidence_threshold
                else "note_discrepancy"
            )
        elif conflict["type"] == "assumption_conflict":
            # Assumption conflicts require validation
            resolution_method = "validate_assumption"
        else:
            resolution_method = "manual_review"

        return {
            "conflict": conflict,
            "method": resolution_method,
            "weights_used": {"source": conflict_source, "weight": source_weight},
            "confidence": min(1.0, source_weight / max(weights.values())),
            "action": self._get_resolution_action(resolution_method),
        }

    def _get_resolution_action(self, method: str) -> str:
        """Get specific action for resolution method"""

        actions = {
            "accept_critique": "Modify recommendations to address identified failure mode",
            "acknowledge_risk": "Add risk mitigation section to address concern",
            "defer_to_evidence": "Update analysis to align with research evidence",
            "note_discrepancy": "Document evidence discrepancy for transparency",
            "validate_assumption": "Flag assumption for validation in implementation",
            "manual_review": "Flag for human expert review",
        }

        return actions.get(method, "Review and assess")

    def _is_risk_acknowledged(
        self, analysis: Dict[str, Any], risk_description: str
    ) -> bool:
        """Check if a risk is acknowledged in the analysis"""
        # Simplified check - in production use NLP
        risks = analysis.get("identified_risks", [])
        risk_text = str(risks).lower()
        return any(word in risk_text for word in risk_description.lower().split()[:3])

    def _contradicts_analysis(self, analysis: Dict[str, Any], claim: str) -> bool:
        """Check if evidence contradicts analysis"""
        # Simplified check - in production use semantic similarity
        # For now, just check for obvious contradictions
        analysis_text = json.dumps(analysis).lower()
        claim_lower = claim.lower()

        # Look for contradictory patterns
        if "increase" in claim_lower and "decrease" in analysis_text:
            return True
        if "reduce" in claim_lower and "increase" in analysis_text:
            return True

        return False

    def _calculate_resolution_confidence(
        self, resolutions: List[Dict[str, Any]]
    ) -> float:
        """Calculate overall confidence in conflict resolutions"""
        if not resolutions:
            return 1.0  # No conflicts means high confidence

        confidences = [r.get("confidence", 0.5) for r in resolutions]
        return sum(confidences) / len(confidences)

    def _build_critique_context(self, arbitration_context: ArbitrationContext) -> str:
        """Build formatted critique context for prompts"""

        lines = []

        if arbitration_context.user_generated_critique:
            lines.append("USER-GENERATED CRITIQUE (HIGHEST PRIORITY):")
            lines.append(
                f"  {arbitration_context.user_generated_critique['description']}"
            )
            lines.append("")

        if arbitration_context.prioritized_critiques:
            lines.append(
                f"USER-PRIORITIZED CRITIQUES ({arbitration_context.prioritized_count}):"
            )
            for critique in arbitration_context.prioritized_critiques:
                details = critique.get("critique_details", {})
                lines.append(
                    f"  - {details.get('title', 'Unknown')}: {details.get('description', '')}"
                )
            lines.append("")

        if arbitration_context.disagreed_critiques:
            lines.append(
                f"USER-DISAGREED CRITIQUES ({arbitration_context.disagreed_count}):"
            )
            for critique in arbitration_context.disagreed_critiques:
                details = critique.get("critique_details", {})
                rationale = critique.get("rationale", "No rationale provided")
                lines.append(
                    f"  - {details.get('title', 'Unknown')} [User: {rationale}]"
                )
            lines.append("")

        if arbitration_context.neutral_critiques:
            lines.append(
                f"NEUTRAL CRITIQUES ({len(arbitration_context.neutral_critiques)}):"
            )
            lines.append("  [Available for reference but not emphasized]")

        return "\n".join(lines)

    def _calculate_integration_level(
        self, synthesis: str, arbitration_context: ArbitrationContext
    ) -> str:
        """Calculate how well critiques were integrated"""

        # Simple heuristic based on mention counts
        # In production, use more sophisticated NLP
        score = 0

        if arbitration_context.user_generated_critique:
            if "user" in synthesis.lower():
                score += 3

        if arbitration_context.prioritized_count > 0:
            score += min(arbitration_context.prioritized_count * 2, 6)

        if score >= 8:
            return "comprehensive"
        elif score >= 5:
            return "substantial"
        elif score >= 2:
            return "moderate"
        else:
            return "minimal"

    def _calculate_adaptation_level(
        self, arbitration_context: ArbitrationContext
    ) -> str:
        """Calculate synthesis adaptation level based on feedback"""

        if arbitration_context.user_generated_critique:
            return "high"
        elif arbitration_context.prioritized_count >= 3:
            return "moderate-high"
        elif arbitration_context.prioritized_count >= 1:
            return "moderate"
        elif arbitration_context.disagreed_count >= 2:
            return "low-moderate"
        else:
            return "minimal"

    def _generate_critique_response_mapping(
        self, components: Dict[str, Any], arbitration_context: ArbitrationContext
    ) -> Dict[str, Any]:
        """Generate mapping of critiques to synthesis responses"""

        mapping = {}

        # Map user critique
        if arbitration_context.user_generated_critique:
            mapping["user_generated"] = {
                "addressed_in": ["user_critique_response", "implementation_roadmap"],
                "priority": "highest",
            }

        # Map prioritized critiques
        for critique in arbitration_context.prioritized_critiques:
            critique_id = critique.get("critique_id", "unknown")
            mapping[critique_id] = {
                "addressed_in": ["prioritized_integration", "core_synthesis"],
                "priority": "high",
            }

        # Map disagreed critiques
        for critique in arbitration_context.disagreed_critiques:
            critique_id = critique.get("critique_id", "unknown")
            mapping[critique_id] = {
                "addressed_in": ["disagreement_acknowledgment"],
                "priority": "acknowledged",
            }

        return mapping

    def _extract_integration_points(self, response: str) -> List[str]:
        """Extract key integration points from response"""
        # Simplified extraction - in production use NLP
        points = []
        lines = response.split("\n")
        for line in lines:
            if any(
                keyword in line.lower()
                for keyword in ["action", "recommend", "implement", "address"]
            ):
                points.append(line.strip())
        return points[:5]  # Top 5 points

    def _summarize_modifications(self, response: str) -> str:
        """Summarize modifications from integration plan"""
        # Simplified summary - in production use NLP
        if len(response) > 500:
            return response[:497] + "..."
        return response

    def _craft_acknowledgment(
        self, critique_details: Dict[str, Any], rationale: str
    ) -> str:
        """Craft respectful acknowledgment of disagreement"""

        if rationale:
            return f"We acknowledge the {critique_details.get('type', 'critique')} concern regarding '{critique_details.get('title', 'this issue')}'. The user has indicated that {rationale}. We respect this perspective and have adjusted our recommendations accordingly."
        else:
            return f"We note the {critique_details.get('type', 'critique')} concern about '{critique_details.get('title', 'this issue')}' but understand it may not apply to this specific context based on user feedback."

    def _extract_phases(self, roadmap: str) -> List[str]:
        """Extract implementation phases from roadmap"""
        phases = []
        if "quick win" in roadmap.lower():
            phases.append("quick_wins")
        if "medium" in roadmap.lower():
            phases.append("medium_term")
        if "long" in roadmap.lower():
            phases.append("long_term")
        return phases if phases else ["standard"]

    def _log_synthesis_metrics(
        self,
        engagement_id: str,
        synthesis: Dict[str, Any],
        arbitration_context: ArbitrationContext,
    ):
        """Log comprehensive synthesis metrics"""

        self.logger.info(
            "enhanced_synthesis_completed",
            engagement_id=str(engagement_id),
            synthesis_version=self.synthesis_version,
            sections_count=len(synthesis.get("sections", [])),
            user_critique_addressed=bool(arbitration_context.user_generated_critique),
            prioritized_critiques_integrated=arbitration_context.prioritized_count,
            disagreements_acknowledged=arbitration_context.disagreed_count,
            engagement_level=arbitration_context.engagement_level,
            adaptation_level=synthesis["flywheel_metadata"][
                "synthesis_adaptation_level"
            ],
            has_user_feedback=arbitration_context.has_user_feedback,
        )


# Factory function
def create_enhanced_synthesis_engine(
    llm_client: LLMClient,
) -> EnhancedFinalSynthesisEngine:
    """Factory function to create enhanced synthesis engine"""
    return EnhancedFinalSynthesisEngine(llm_client)
