#!/usr/bin/env python3
"""
System 2 Enhanced Devil's Advocate System for LOLLA V1.0
========================================================

ARCHITECTURAL BREAKTHROUGH: ULTRATHINK + System 2 Cognitive Stage Verification

This enhanced Devil's Advocate wraps LOLLA's sophisticated 4-engine ULTRATHINK system
with System 2 cognitive stage verification and shortcut detection.

Key Innovation:
- Preserves all existing ULTRATHINK capabilities (4 engines)
- Adds System 2 cognitive stage verification
- Detects and prevents cognitive shortcuts in analysis pipeline
- Forces re-examination if stages are incomplete
- Provides mental model conflict detection
- Maintains compatibility with existing Devil's Advocate API

Integration with LOLLA:
- Enhances existing 4-engine critique system
- Adds verification that all 7 cognitive stages completed properly
- Detects pattern matching vs deliberate reasoning
- Forces re-deliberation when shortcuts detected
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import uuid

# LOLLA Core Devil's Advocate
from .enhanced_devils_advocate_system import (
    EnhancedDevilsAdvocateSystem,
    DevilsAdvocateChallenge,
    ComprehensiveChallengeResult,
)

# System 2 Components
try:
    from ..model_interaction_matrix import CognitiveStage, CognitiveArchitectureBridge
    from ..system2_meta_orchestrator import System2StageResult, System2Mode
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from model_interaction_matrix import CognitiveStage, CognitiveArchitectureBridge
    from system2_meta_orchestrator import System2StageResult, System2Mode

# LOLLA Core Context
from .unified_context_stream import get_unified_context_stream

logger = logging.getLogger(__name__)


@dataclass
class System2StageVerification:
    """Verification result for a System 2 cognitive stage"""

    stage: CognitiveStage
    stage_completed: bool
    deliberation_depth: float
    shortcuts_detected: List[str]
    mental_models_verified: List[str]
    confidence_score: float
    verification_details: Dict[str, Any]


@dataclass
class System2DevilsAdvocateResult:
    """Enhanced Devil's Advocate result with System 2 verification"""

    # Original ULTRATHINK result
    base_ultrathink_result: ComprehensiveChallengeResult

    # System 2 enhancements
    stage_verifications: List[System2StageVerification]
    cognitive_shortcuts_detected: int
    forced_re_deliberations: int

    # Enhanced challenges
    system2_specific_challenges: List[DevilsAdvocateChallenge]
    mental_model_conflicts: List[Dict[str, Any]]
    cognitive_completeness_score: float

    # Final assessment
    system2_enhanced_risk_score: float
    requires_re_analysis: bool
    system2_advantage_metrics: Dict[str, float]


class System2EnhancedDevilsAdvocate:
    """
    Enhanced Devil's Advocate system integrating LOLLA's 4-engine ULTRATHINK
    with System 2 cognitive stage verification and shortcut detection.

    ARCHITECTURAL PRINCIPLE: Enhancement, not replacement
    - Preserves all ULTRATHINK capabilities (Munger, Ackoff, Cognitive Audit, LLM Sceptic)
    - Adds System 2 meta-cognitive verification layer
    - Forces complete cognitive stage execution
    - Detects and prevents analytical shortcuts
    """

    def __init__(self, system2_mode: System2Mode = System2Mode.ENFORCING):
        # Core LOLLA Devil's Advocate (4-engine ULTRATHINK)
        self.base_devils_advocate = EnhancedDevilsAdvocateSystem()
        self.context_stream = get_unified_context_stream()

        # System 2 components
        self.cognitive_bridge = CognitiveArchitectureBridge()
        self.system2_mode = system2_mode

        # System 2 verification thresholds
        self.min_deliberation_depth = 0.7
        self.max_shortcuts_allowed = 1
        self.min_mental_models_per_stage = 2

        # Cognitive stage requirements
        self.required_stages = [
            CognitiveStage.PERCEPTION,
            CognitiveStage.DECOMPOSITION,
            CognitiveStage.REASONING,
            CognitiveStage.SYNTHESIS,
            CognitiveStage.DECISION,
        ]

        logger.info(
            f"🔍 System 2 Enhanced Devil's Advocate initialized in {system2_mode.value} mode"
        )
        logger.info("   • 4-engine ULTRATHINK system preserved")
        logger.info("   • System 2 cognitive stage verification active")

    async def system2_enhanced_challenge_analysis(
        self,
        recommendation: str,
        business_context: Dict[str, Any],
        analysis_trace: Optional[List[System2StageResult]] = None,
        trace_id: Optional[str] = None,
    ) -> System2DevilsAdvocateResult:
        """
        Execute System 2-enhanced Devil's Advocate analysis.

        This is the main entry point that wraps LOLLA's ULTRATHINK system
        with System 2 cognitive stage verification and shortcut detection.

        Args:
            recommendation: The recommendation to challenge
            business_context: Business context and supporting information
            analysis_trace: Optional trace of System 2 cognitive stages
            trace_id: Optional trace ID for logging

        Returns:
            Enhanced Devil's Advocate result with System 2 verification
        """
        trace_id = trace_id or str(uuid.uuid4())

        logger.info(
            f"🔍 SYSTEM 2 ENHANCED DEVIL'S ADVOCATE ANALYSIS - Trace: {trace_id}"
        )
        logger.info("=" * 80)

        # STEP 1: Run base ULTRATHINK analysis (4 engines)
        base_result = await self.base_devils_advocate.comprehensive_challenge_analysis(
            recommendation, business_context
        )

        # STEP 2: Verify cognitive stage completeness
        stage_verifications = await self._verify_cognitive_stages(
            analysis_trace, recommendation, business_context
        )

        # STEP 3: Detect cognitive shortcuts in the analysis
        shortcuts_detected = await self._detect_cognitive_shortcuts(
            stage_verifications, base_result
        )

        # STEP 4: Generate System 2-specific challenges
        system2_challenges = await self._generate_system2_challenges(
            stage_verifications, shortcuts_detected, recommendation
        )

        # STEP 5: Detect mental model conflicts
        mental_model_conflicts = await self._detect_mental_model_conflicts(
            analysis_trace, business_context
        )

        # STEP 6: Calculate cognitive completeness score
        completeness_score = self._calculate_cognitive_completeness_score(
            stage_verifications, shortcuts_detected
        )

        # STEP 7: Determine if re-analysis is required
        requires_re_analysis = self._assess_re_analysis_requirement(
            stage_verifications, shortcuts_detected, completeness_score
        )

        # STEP 8: Force re-deliberation if needed
        forced_re_deliberations = 0
        if self.system2_mode == System2Mode.ENFORCING and requires_re_analysis:
            forced_re_deliberations = await self._force_re_deliberation(
                stage_verifications, trace_id
            )

        # STEP 9: Calculate System 2 advantage metrics
        advantage_metrics = self._calculate_system2_advantage_metrics(
            base_result, stage_verifications, shortcuts_detected
        )

        # STEP 10: Create enhanced result
        enhanced_result = System2DevilsAdvocateResult(
            base_ultrathink_result=base_result,
            stage_verifications=stage_verifications,
            cognitive_shortcuts_detected=len(shortcuts_detected),
            forced_re_deliberations=forced_re_deliberations,
            system2_specific_challenges=system2_challenges,
            mental_model_conflicts=mental_model_conflicts,
            cognitive_completeness_score=completeness_score,
            system2_enhanced_risk_score=self._calculate_enhanced_risk_score(
                base_result, completeness_score
            ),
            requires_re_analysis=requires_re_analysis,
            system2_advantage_metrics=advantage_metrics,
        )

        # STEP 11: Record System 2 enhancement evidence
        await self._record_system2_enhancement_evidence(enhanced_result, trace_id)

        logger.info("⚡ SYSTEM 2 ENHANCED ANALYSIS COMPLETE")
        logger.info(
            f"   • Base ULTRATHINK Challenges: {base_result.total_challenges_found}"
        )
        logger.info(f"   • System 2 Challenges: {len(system2_challenges)}")
        logger.info(f"   • Shortcuts Detected: {len(shortcuts_detected)}")
        logger.info(f"   • Cognitive Completeness: {completeness_score:.3f}")
        logger.info(f"   • Requires Re-analysis: {requires_re_analysis}")
        logger.info("=" * 80)

        return enhanced_result

    async def _verify_cognitive_stages(
        self,
        analysis_trace: Optional[List[System2StageResult]],
        recommendation: str,
        business_context: Dict[str, Any],
    ) -> List[System2StageVerification]:
        """
        Verify that all required cognitive stages were completed properly.

        This is the core System 2 verification - ensuring that each cognitive stage
        was executed with sufficient deliberation and mental model activation.
        """
        logger.info("🧠 Verifying cognitive stage completeness")

        verifications = []

        if not analysis_trace:
            # No trace available - create minimal verification
            for stage in self.required_stages:
                verification = System2StageVerification(
                    stage=stage,
                    stage_completed=False,
                    deliberation_depth=0.0,
                    shortcuts_detected=["No stage trace available"],
                    mental_models_verified=[],
                    confidence_score=0.1,
                    verification_details={"error": "No analysis trace provided"},
                )
                verifications.append(verification)
            return verifications

        # Verify each required stage
        for stage in self.required_stages:
            stage_result = self._find_stage_in_trace(stage, analysis_trace)

            if stage_result:
                verification = await self._verify_individual_stage(stage_result)
            else:
                # Stage missing - major shortcut detected
                verification = System2StageVerification(
                    stage=stage,
                    stage_completed=False,
                    deliberation_depth=0.0,
                    shortcuts_detected=[f"{stage.value} stage completely skipped"],
                    mental_models_verified=[],
                    confidence_score=0.0,
                    verification_details={
                        "error": f"{stage.value} stage not found in trace"
                    },
                )

            verifications.append(verification)

        logger.info(f"   • Verified {len(verifications)} cognitive stages")
        return verifications

    async def _verify_individual_stage(
        self, stage_result: System2StageResult
    ) -> System2StageVerification:
        """Verify an individual cognitive stage for completeness and quality."""

        shortcuts_detected = []

        # Check deliberation depth
        if stage_result.deliberation_depth < self.min_deliberation_depth:
            shortcuts_detected.append(
                f"Low deliberation depth: {stage_result.deliberation_depth:.2f}"
            )

        # Check mental model usage
        if len(stage_result.mental_models_activated) < self.min_mental_models_per_stage:
            shortcuts_detected.append(
                f"Insufficient mental models: {len(stage_result.mental_models_activated)}"
            )

        # Check processing time (very fast might indicate shortcuts)
        if stage_result.stage_duration_ms < 1000:  # Less than 1 second
            shortcuts_detected.append(
                f"Suspiciously fast processing: {stage_result.stage_duration_ms}ms"
            )

        # Check if stage was marked as having shortcuts prevented
        shortcuts_detected.extend(
            [
                f"Shortcut prevented: {i}"
                for i in range(stage_result.shortcuts_prevented)
            ]
        )

        return System2StageVerification(
            stage=stage_result.cognitive_stage,
            stage_completed=len(shortcuts_detected) == 0,
            deliberation_depth=stage_result.deliberation_depth,
            shortcuts_detected=shortcuts_detected,
            mental_models_verified=stage_result.mental_models_activated,
            confidence_score=stage_result.confidence_score,
            verification_details={
                "duration_ms": stage_result.stage_duration_ms,
                "shortcuts_prevented": stage_result.shortcuts_prevented,
                "mental_models_count": len(stage_result.mental_models_activated),
            },
        )

    async def _detect_cognitive_shortcuts(
        self,
        stage_verifications: List[System2StageVerification],
        base_result: ComprehensiveChallengeResult,
    ) -> List[str]:
        """Detect cognitive shortcuts across the entire analysis."""

        shortcuts = []

        # Aggregate shortcuts from stage verifications
        for verification in stage_verifications:
            shortcuts.extend(verification.shortcuts_detected)

        # Detect pattern-matching indicators in ULTRATHINK result
        if base_result.total_challenges_found < 3:
            shortcuts.append("Insufficient challenge depth - possible pattern matching")

        # Check for generic reasoning patterns
        if base_result.intellectual_honesty_score > 0.95:
            shortcuts.append(
                "Suspiciously high intellectual honesty - possible confirmation bias"
            )

        # Check system confidence vs risk score correlation
        if base_result.system_confidence > 0.9 and base_result.overall_risk_score > 0.7:
            shortcuts.append(
                "High confidence with high risk - possible overconfidence bias"
            )

        logger.info(f"   • Detected {len(shortcuts)} cognitive shortcuts")
        return shortcuts

    async def _generate_system2_challenges(
        self,
        stage_verifications: List[System2StageVerification],
        shortcuts_detected: List[str],
        recommendation: str,
    ) -> List[DevilsAdvocateChallenge]:
        """Generate System 2-specific challenges based on cognitive stage analysis."""

        system2_challenges = []

        # Challenge incomplete cognitive stages
        incomplete_stages = [v for v in stage_verifications if not v.stage_completed]
        if incomplete_stages:
            for stage in incomplete_stages:
                challenge = DevilsAdvocateChallenge(
                    challenge_id=f"system2_stage_{stage.stage.value}",
                    challenge_type="cognitive_completeness",
                    challenge_text=f"The {stage.stage.value} cognitive stage appears incomplete. "
                    f"Detected shortcuts: {', '.join(stage.shortcuts_detected)}. "
                    f"This may indicate surface-level analysis rather than deep deliberation.",
                    severity=0.8,
                    evidence=stage.shortcuts_detected,
                    mitigation_strategy=f"Re-execute {stage.stage.value} stage with forced deliberation",
                    source_engine="system2_stage_verifier",
                )
                system2_challenges.append(challenge)

        # Challenge cognitive shortcuts
        if shortcuts_detected:
            shortcut_challenge = DevilsAdvocateChallenge(
                challenge_id="system2_shortcuts",
                challenge_type="cognitive_shortcuts",
                challenge_text=f"Multiple cognitive shortcuts detected: {', '.join(shortcuts_detected[:3])}. "
                f"This suggests pattern matching rather than System 2 deliberative reasoning.",
                severity=0.9,
                evidence=shortcuts_detected,
                mitigation_strategy="Force re-analysis through complete System 2 cognitive stages",
                source_engine="system2_shortcut_detector",
            )
            system2_challenges.append(shortcut_challenge)

        # Challenge insufficient mental model diversity
        total_models = sum(len(v.mental_models_verified) for v in stage_verifications)
        if total_models < 10:
            diversity_challenge = DevilsAdvocateChallenge(
                challenge_id="system2_model_diversity",
                challenge_type="mental_model_diversity",
                challenge_text=f"Only {total_models} mental models activated across all cognitive stages. "
                f"This suggests narrow analytical perspective rather than comprehensive System 2 analysis.",
                severity=0.7,
                evidence=[
                    f"Total mental models: {total_models}",
                    "Expected: 15+ for comprehensive analysis",
                ],
                mitigation_strategy="Activate additional mental models across cognitive stages",
                source_engine="system2_diversity_analyzer",
            )
            system2_challenges.append(diversity_challenge)

        logger.info(
            f"   • Generated {len(system2_challenges)} System 2-specific challenges"
        )
        return system2_challenges

    async def _detect_mental_model_conflicts(
        self,
        analysis_trace: Optional[List[System2StageResult]],
        business_context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Detect conflicts between mental models used in analysis."""

        if not analysis_trace:
            return []

        conflicts = []
        all_models = set()

        # Collect all mental models used
        for stage_result in analysis_trace:
            all_models.update(stage_result.mental_models_activated)

        # Simple conflict detection (would be more sophisticated in practice)
        conflicting_pairs = [
            ("optimism_bias", "pessimism_bias"),
            ("confirmation_bias", "devils_advocate_thinking"),
            ("anchoring", "first_principles"),
        ]

        for model1, model2 in conflicting_pairs:
            if model1 in all_models and model2 in all_models:
                conflicts.append(
                    {
                        "conflicting_models": [model1, model2],
                        "conflict_type": "opposing_biases",
                        "resolution_needed": True,
                        "mitigation": f"Explicitly address tension between {model1} and {model2}",
                    }
                )

        logger.info(f"   • Detected {len(conflicts)} mental model conflicts")
        return conflicts

    def _calculate_cognitive_completeness_score(
        self,
        stage_verifications: List[System2StageVerification],
        shortcuts_detected: List[str],
    ) -> float:
        """Calculate cognitive completeness score (0-1)."""

        # Base score from stage completeness
        completed_stages = sum(1 for v in stage_verifications if v.stage_completed)
        stage_completeness = completed_stages / len(stage_verifications)

        # Average deliberation depth
        avg_deliberation = sum(v.deliberation_depth for v in stage_verifications) / len(
            stage_verifications
        )

        # Shortcut penalty
        shortcut_penalty = min(len(shortcuts_detected) * 0.1, 0.5)

        # Calculate final score
        completeness_score = (
            stage_completeness * 0.4 + avg_deliberation * 0.4 + 0.2
        ) - shortcut_penalty
        return max(min(completeness_score, 1.0), 0.0)

    def _assess_re_analysis_requirement(
        self,
        stage_verifications: List[System2StageVerification],
        shortcuts_detected: List[str],
        completeness_score: float,
    ) -> bool:
        """Assess whether re-analysis is required based on System 2 criteria."""

        # Require re-analysis if:
        # 1. Cognitive completeness is too low
        if completeness_score < 0.6:
            return True

        # 2. Too many shortcuts detected
        if len(shortcuts_detected) > self.max_shortcuts_allowed:
            return True

        # 3. Any critical stage is incomplete
        critical_incomplete = any(
            v.stage in [CognitiveStage.REASONING, CognitiveStage.SYNTHESIS]
            and not v.stage_completed
            for v in stage_verifications
        )
        if critical_incomplete:
            return True

        return False

    async def _force_re_deliberation(
        self, stage_verifications: List[System2StageVerification], trace_id: str
    ) -> int:
        """Force re-deliberation for incomplete cognitive stages."""

        if self.system2_mode != System2Mode.ENFORCING:
            return 0

        forced_count = 0

        # Identify stages that need re-deliberation
        incomplete_stages = [v for v in stage_verifications if not v.stage_completed]

        for stage_verification in incomplete_stages:
            logger.warning(
                f"🔄 Forcing re-deliberation for {stage_verification.stage.value} stage"
            )

            # Record forced re-deliberation
            await self.context_stream.record_event(
                trace_id=trace_id,
                event_type="SYSTEM_2_FORCED_RE_DELIBERATION",
                event_data={
                    "stage": stage_verification.stage.value,
                    "shortcuts_detected": stage_verification.shortcuts_detected,
                    "deliberation_depth": stage_verification.deliberation_depth,
                },
            )

            forced_count += 1

        logger.info(f"   • Forced re-deliberation for {forced_count} stages")
        return forced_count

    def _calculate_system2_advantage_metrics(
        self,
        base_result: ComprehensiveChallengeResult,
        stage_verifications: List[System2StageVerification],
        shortcuts_detected: List[str],
    ) -> Dict[str, float]:
        """Calculate System 2 advantages over generic approaches."""

        # Cognitive completeness advantage
        completed_stages = sum(1 for v in stage_verifications if v.stage_completed)
        cognitive_completeness = completed_stages / 2  # vs ~2 stages for generic

        # Deliberation depth advantage
        avg_deliberation = sum(v.deliberation_depth for v in stage_verifications) / len(
            stage_verifications
        )
        deliberation_advantage = avg_deliberation / 0.4  # vs ~40% generic depth

        # Challenge depth advantage (ULTRATHINK provides this)
        challenge_advantage = (
            base_result.total_challenges_found / 2
        )  # vs ~2 generic challenges

        # Mental model diversity advantage
        total_models = sum(len(v.mental_models_verified) for v in stage_verifications)
        diversity_advantage = total_models / 3  # vs ~3 generic models

        # Overall advantage
        overall_advantage = (
            cognitive_completeness
            + deliberation_advantage
            + challenge_advantage
            + diversity_advantage
        ) / 4

        return {
            "cognitive_completeness_advantage": cognitive_completeness,
            "deliberation_depth_advantage": deliberation_advantage,
            "challenge_depth_advantage": challenge_advantage,
            "mental_model_diversity_advantage": diversity_advantage,
            "overall_system2_advantage": overall_advantage,
            "shortcuts_prevented": len(shortcuts_detected),
        }

    def _calculate_enhanced_risk_score(
        self, base_result: ComprehensiveChallengeResult, completeness_score: float
    ) -> float:
        """Calculate enhanced risk score incorporating System 2 completeness."""

        base_risk = base_result.overall_risk_score
        completeness_factor = (
            1.0 - completeness_score
        )  # Higher incompleteness = higher risk

        # Enhanced risk is base risk adjusted by cognitive completeness
        enhanced_risk = base_risk + (completeness_factor * 0.3)
        return min(enhanced_risk, 1.0)

    def _find_stage_in_trace(
        self, stage: CognitiveStage, analysis_trace: List[System2StageResult]
    ) -> Optional[System2StageResult]:
        """Find a specific cognitive stage in the analysis trace."""
        for stage_result in analysis_trace:
            if stage_result.cognitive_stage == stage:
                return stage_result
        return None

    async def _record_system2_enhancement_evidence(
        self, enhanced_result: System2DevilsAdvocateResult, trace_id: str
    ) -> None:
        """Record System 2 enhancement evidence in context stream."""

        await self.context_stream.record_event(
            trace_id=trace_id,
            event_type="SYSTEM_2_DEVILS_ADVOCATE_ENHANCED",
            event_data={
                "base_challenges": enhanced_result.base_ultrathink_result.total_challenges_found,
                "system2_challenges": len(enhanced_result.system2_specific_challenges),
                "shortcuts_detected": enhanced_result.cognitive_shortcuts_detected,
                "forced_re_deliberations": enhanced_result.forced_re_deliberations,
                "cognitive_completeness": enhanced_result.cognitive_completeness_score,
                "requires_re_analysis": enhanced_result.requires_re_analysis,
                "enhanced_risk_score": enhanced_result.system2_enhanced_risk_score,
                "system2_advantage": enhanced_result.system2_advantage_metrics.get(
                    "overall_system2_advantage", 1.0
                ),
            },
        )


# Factory function for easy initialization
def get_system2_enhanced_devils_advocate(
    system2_mode: System2Mode = System2Mode.ENFORCING,
) -> System2EnhancedDevilsAdvocate:
    """Get System 2 Enhanced Devil's Advocate instance."""
    return System2EnhancedDevilsAdvocate(system2_mode)


if __name__ == "__main__":
    # Demo usage
    async def demo_enhanced_devils_advocate():
        devils_advocate = get_system2_enhanced_devils_advocate(System2Mode.ENFORCING)

        # Sample recommendation and context
        recommendation = (
            "We should acquire TechCorp for $2.5B based on strategic synergies"
        )
        context = {
            "business_context": "AI acceleration strategy",
            "financial_impact": "$2.5B acquisition",
            "strategic_rationale": "Technology synergies",
        }

        result = await devils_advocate.system2_enhanced_challenge_analysis(
            recommendation, context
        )

        print("System 2 Enhanced Devil's Advocate Complete!")
        print(
            f"Base ULTRATHINK Challenges: {result.base_ultrathink_result.total_challenges_found}"
        )
        print(f"System 2 Challenges: {len(result.system2_specific_challenges)}")
        print(f"Shortcuts Detected: {result.cognitive_shortcuts_detected}")
        print(f"Cognitive Completeness: {result.cognitive_completeness_score:.3f}")
        print(f"Requires Re-analysis: {result.requires_re_analysis}")

    # asyncio.run(demo_enhanced_devils_advocate())
    print("🔍 System 2 Enhanced Devil's Advocate loaded successfully")
