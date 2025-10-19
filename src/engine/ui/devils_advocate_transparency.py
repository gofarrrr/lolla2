#!/usr/bin/env python3
"""
Devil's Advocate Glass-Box Transparency Module
Shows the thinking process of the Devil's Advocate system for full transparency
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import logging
from enum import Enum

from src.models.transparency_models import (
    TransparencyContent,
    TransparencyLayer,
    CognitiveLoadLevel,
    UserProfile,
    UserExpertiseLevel,
)


class DevilsAdvocatePhase(str, Enum):
    """Devil's Advocate analysis phases"""

    PHASE_1_INITIAL_ANALYSIS = "phase_1_initial_analysis"
    PHASE_2_CHALLENGE_ACTIVATION = "phase_2_devils_advocate_activation"
    PHASE_3_REFINEMENT = "phase_3_recommendation_refinement"


class ChallengeSystemType(str, Enum):
    """Types of challenge systems in Devil's Advocate"""

    MUNGER_OVERLAY = "munger_overlay"
    ACKOFF_CHALLENGER = "ackoff_challenger"
    CONSTITUTIONAL_COMPLIANCE = "constitutional_compliance_audit"


@dataclass
class DevilsAdvocateThinkingStep:
    """Individual thinking step in Devil's Advocate process"""

    step_id: str
    phase: DevilsAdvocatePhase
    system_type: ChallengeSystemType
    thinking_content: str
    insights_generated: List[str]
    biases_detected: List[str] = field(default_factory=list)
    assumptions_challenged: List[Dict[str, Any]] = field(default_factory=list)
    confidence_impact: float = 0.0
    duration_seconds: float = 0.0

    def to_display_format(self) -> Dict[str, Any]:
        """Convert to format suitable for UI display"""
        return {
            "step_id": self.step_id,
            "phase": self.phase.value,
            "system_type": self.system_type.value,
            "thinking_content": self.thinking_content,
            "insights_generated": self.insights_generated,
            "biases_detected": self.biases_detected,
            "assumptions_challenged": self.assumptions_challenged,
            "confidence_impact": self.confidence_impact,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class DevilsAdvocateTransparencyData:
    """Complete transparency data for Devil's Advocate system"""

    engagement_id: str
    thinking_steps: List[DevilsAdvocateThinkingStep] = field(default_factory=list)
    confidence_evolution: Dict[str, float] = field(default_factory=dict)
    transformation_journey: Dict[str, Any] = field(default_factory=dict)
    system_performance_metrics: Dict[str, Any] = field(default_factory=dict)
    value_creation_evidence: Dict[str, Any] = field(default_factory=dict)

    def get_phase_summary(self, phase: DevilsAdvocatePhase) -> Dict[str, Any]:
        """Get summary of specific phase"""
        phase_steps = [step for step in self.thinking_steps if step.phase == phase]

        if not phase_steps:
            return {
                "phase": phase.value,
                "steps_count": 0,
                "insights": [],
                "duration": 0.0,
            }

        return {
            "phase": phase.value,
            "steps_count": len(phase_steps),
            "insights": [
                insight for step in phase_steps for insight in step.insights_generated
            ],
            "biases_detected": list(
                set([bias for step in phase_steps for bias in step.biases_detected])
            ),
            "total_duration": sum(step.duration_seconds for step in phase_steps),
            "confidence_impact": sum(step.confidence_impact for step in phase_steps),
        }


class DevilsAdvocateTransparencyEngine:
    """Engine for creating glass-box transparency of Devil's Advocate thinking process"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def extract_thinking_process_from_proof_data(
        self, devils_advocate_data: Dict[str, Any]
    ) -> DevilsAdvocateTransparencyData:
        """Extract thinking process from Devil's Advocate proof data"""

        engagement_id = devils_advocate_data.get("engagement_id", "unknown")

        # Extract execution phases
        execution_phases = devils_advocate_data.get("execution_phases", {})

        # Extract thinking steps from each phase
        thinking_steps = []

        # Phase 1: Initial Analysis
        phase_1 = execution_phases.get("phase_1_initial_analysis", {})
        if phase_1:
            phase_1_step = DevilsAdvocateThinkingStep(
                step_id="phase_1_initial",
                phase=DevilsAdvocatePhase.PHASE_1_INITIAL_ANALYSIS,
                system_type=ChallengeSystemType.MUNGER_OVERLAY,  # Default
                thinking_content=f"Initial analysis showed tendency toward: {phase_1.get('initial_tendency', 'Unknown')}. Surface conclusion: {phase_1.get('surface_conclusion', 'Not captured')}",
                insights_generated=[
                    f"Initial confidence: {phase_1.get('confidence_before_challenge', 0):.2f}"
                ],
                duration_seconds=phase_1.get("duration", 0.0),
            )

            # Add red flags as biases detected
            red_flags = phase_1.get("red_flags_initially_missed", [])
            phase_1_step.biases_detected = red_flags

            thinking_steps.append(phase_1_step)

        # Phase 2: Challenge Activation
        phase_2 = execution_phases.get("phase_2_devils_advocate_activation", {})
        if phase_2:
            challenge_systems = phase_2.get("challenge_systems_activated", {})

            # Munger Overlay Challenge System
            munger_overlay = challenge_systems.get("munger_overlay", {})
            if munger_overlay:
                munger_step = DevilsAdvocateThinkingStep(
                    step_id="munger_challenge",
                    phase=DevilsAdvocatePhase.PHASE_2_CHALLENGE_ACTIVATION,
                    system_type=ChallengeSystemType.MUNGER_OVERLAY,
                    thinking_content=f"Inversion analysis: {munger_overlay.get('inversion_analysis', 'Not captured')}",
                    insights_generated=munger_overlay.get(
                        "lollapalooza_effects_detected", []
                    ),
                    biases_detected=munger_overlay.get("biases_identified", []),
                    duration_seconds=phase_2.get("duration", 0.0)
                    / 3,  # Distribute across sub-systems
                )
                thinking_steps.append(munger_step)

            # Ackoff Challenger System
            ackoff_challenger = challenge_systems.get("ackoff_challenger", {})
            if ackoff_challenger:
                assumptions_dissolved = ackoff_challenger.get(
                    "assumptions_dissolved", []
                )
                ackoff_step = DevilsAdvocateThinkingStep(
                    step_id="ackoff_challenge",
                    phase=DevilsAdvocatePhase.PHASE_2_CHALLENGE_ACTIVATION,
                    system_type=ChallengeSystemType.ACKOFF_CHALLENGER,
                    thinking_content="Applied assumption dissolution methodology to challenge fundamental premises",
                    insights_generated=ackoff_challenger.get(
                        "idealized_design_insights", []
                    ),
                    assumptions_challenged=assumptions_dissolved,
                    duration_seconds=phase_2.get("duration", 0.0) / 3,
                )
                thinking_steps.append(ackoff_step)

            # Constitutional Compliance Audit
            constitutional_audit = challenge_systems.get(
                "constitutional_compliance_audit", {}
            )
            if constitutional_audit:
                ethical_considerations = constitutional_audit.get(
                    "ethical_considerations", {}
                )
                constitutional_step = DevilsAdvocateThinkingStep(
                    step_id="constitutional_audit",
                    phase=DevilsAdvocatePhase.PHASE_2_CHALLENGE_ACTIVATION,
                    system_type=ChallengeSystemType.CONSTITUTIONAL_COMPLIANCE,
                    thinking_content=f"Ethical impact assessment: {ethical_considerations.get('employee_impact_score', 0):.2f} employee impact score",
                    insights_generated=constitutional_audit.get(
                        "governance_requirements", []
                    ),
                    duration_seconds=phase_2.get("duration", 0.0) / 3,
                )
                thinking_steps.append(constitutional_step)

        # Phase 3: Recommendation Refinement
        phase_3 = execution_phases.get("phase_3_recommendation_refinement", {})
        if phase_3:
            refinement_evidence = phase_3.get("refinement_evidence", {})
            phase_3_step = DevilsAdvocateThinkingStep(
                step_id="refinement_synthesis",
                phase=DevilsAdvocatePhase.PHASE_3_REFINEMENT,
                system_type=ChallengeSystemType.MUNGER_OVERLAY,  # Lead system for synthesis
                thinking_content=f"Transformed from: {refinement_evidence.get('naive_recommendation_avoided', 'Unknown')} to: {refinement_evidence.get('sophisticated_outcome', 'Unknown')}",
                insights_generated=refinement_evidence.get("key_improvements", []),
                duration_seconds=phase_3.get("duration", 0.0),
            )
            thinking_steps.append(phase_3_step)

        # Extract confidence evolution
        confidence_evolution = {}
        devils_advocate_impact = devils_advocate_data.get(
            "devils_advocate_impact_analysis", {}
        )
        if devils_advocate_impact:
            transformation_journey = devils_advocate_impact.get(
                "transformation_journey", {}
            )
            confidence_evolution = transformation_journey.get(
                "confidence_evolution", {}
            )

        # Extract transformation journey
        transformation_journey = devils_advocate_impact.get(
            "transformation_journey", {}
        )

        # Extract system performance metrics
        system_performance = devils_advocate_data.get("system_performance_evidence", {})

        # Extract value creation metrics
        value_creation = devils_advocate_impact.get("value_creation_metrics", {})

        return DevilsAdvocateTransparencyData(
            engagement_id=engagement_id,
            thinking_steps=thinking_steps,
            confidence_evolution=confidence_evolution,
            transformation_journey=transformation_journey,
            system_performance_metrics=system_performance,
            value_creation_evidence=value_creation,
        )

    def generate_glass_box_transparency_content(
        self,
        transparency_data: DevilsAdvocateTransparencyData,
        user_profile: UserProfile,
    ) -> TransparencyContent:
        """Generate glass-box transparency content showing Devil's Advocate thinking process"""

        # Create detailed thinking process content
        content = f"""
        ## Devil's Advocate Glass-Box Transparency
        
        **Engagement ID**: {transparency_data.engagement_id}
        **Total Thinking Steps**: {len(transparency_data.thinking_steps)}
        **Confidence Journey**: {transparency_data.confidence_evolution.get('pre_challenge_confidence', 0):.2f} → {transparency_data.confidence_evolution.get('post_refinement_confidence', 0):.2f}
        
        ### The Challenge Process Journey
        
        The Devil's Advocate system applied rigorous internal challenge to prevent flawed conclusions. Here's the complete thinking process:
        
        """

        # Add phase-by-phase thinking process
        for phase in DevilsAdvocatePhase:
            phase_summary = transparency_data.get_phase_summary(phase)

            if phase_summary["steps_count"] > 0:
                content += f"""
        #### {phase.value.replace('_', ' ').title()}
        
        **Duration**: {phase_summary['total_duration']:.1f} seconds
        **Steps**: {phase_summary['steps_count']}
        **Confidence Impact**: {phase_summary.get('confidence_impact', 0):.2f}
        
        """

                # Add thinking steps for this phase
                phase_steps = [
                    step
                    for step in transparency_data.thinking_steps
                    if step.phase == phase
                ]
                for step in phase_steps:
                    content += f"""
        **{step.system_type.value.replace('_', ' ').title()}**:
        {step.thinking_content}
        
        """

                    if step.biases_detected:
                        content += (
                            f"*Biases Detected*: {', '.join(step.biases_detected)}\n\n"
                        )

                    if step.insights_generated:
                        content += f"*Key Insights*: {'; '.join(step.insights_generated[:3])}\n\n"

                    if step.assumptions_challenged:
                        content += f"*Assumptions Challenged*: {len(step.assumptions_challenged)} fundamental premises questioned\n\n"

                content += "---\n\n"

        # Add transformation evidence
        if transparency_data.transformation_journey:
            journey = transparency_data.transformation_journey
            content += f"""
        ### Transformation Evidence
        
        **Initial Hypothesis**: {journey.get('initial_hypothesis', 'Not captured')}
        
        **Critical Challenges Applied**:
        """

            challenges = journey.get("critical_challenges_applied", [])
            for i, challenge in enumerate(challenges[:3], 1):
                content += f"""
        {i}. **{challenge.get('type', 'Unknown')}**: {challenge.get('insight', 'Not captured')}
           → Impact: {challenge.get('impact', 'Not documented')}
        """

            content += f"""
        
        **Final Recommendation**: {journey.get('final_recommendation', 'Not captured')}
        
        """

        # Add value creation evidence
        if transparency_data.value_creation_evidence:
            value_metrics = transparency_data.value_creation_evidence
            content += f"""
        ### Value Creation Evidence
        
        **Risk Reduction**: {value_metrics.get('risk_reduction', {}).get('implementation_risk', 'Not measured')}
        **Decision Quality**: Stakeholder consideration improved from {value_metrics.get('decision_quality_improvement', {}).get('stakeholder_consideration', 'Not measured')}
        **Strategic Sophistication**: Enhanced from "{value_metrics.get('strategic_sophistication', {}).get('before', 'Not captured')}" to "{value_metrics.get('strategic_sophistication', {}).get('after', 'Not captured')}"
        
        """

        # Determine cognitive load based on content complexity and user expertise
        cognitive_load = self._assess_devils_advocate_cognitive_load(
            transparency_data, user_profile
        )

        # Create key insights
        key_insights = [
            f"Applied {len(set(step.system_type for step in transparency_data.thinking_steps))} different challenge systems",
            f"Detected {len(set([bias for step in transparency_data.thinking_steps for bias in step.biases_detected]))} unique cognitive biases",
            f"Confidence evolved through {len(transparency_data.confidence_evolution)} measurement points",
            f"Generated {sum(len(step.insights_generated) for step in transparency_data.thinking_steps)} critical insights",
        ]

        return TransparencyContent(
            layer=TransparencyLayer.DETAILED_AUDIT_TRAIL,  # This is detailed analysis
            title="Devil's Advocate Thinking Process",
            content=content.strip(),
            cognitive_load=cognitive_load,
            key_insights=key_insights,
            expandable_sections=[
                {
                    "title": "Complete Challenge System Details",
                    "content": [
                        step.to_display_format()
                        for step in transparency_data.thinking_steps
                    ],
                },
                {
                    "title": "Confidence Evolution Analysis",
                    "content": transparency_data.confidence_evolution,
                },
                {
                    "title": "System Performance Metrics",
                    "content": transparency_data.system_performance_metrics,
                },
            ],
            metadata={
                "devils_advocate_enabled": True,
                "thinking_steps_count": len(transparency_data.thinking_steps),
                "challenge_systems_used": len(
                    set(step.system_type for step in transparency_data.thinking_steps)
                ),
                "total_processing_time": sum(
                    step.duration_seconds for step in transparency_data.thinking_steps
                ),
                "confidence_transformation": {
                    "initial": transparency_data.confidence_evolution.get(
                        "pre_challenge_confidence", 0
                    ),
                    "final": transparency_data.confidence_evolution.get(
                        "post_refinement_confidence", 0
                    ),
                },
            },
        )

    def _assess_devils_advocate_cognitive_load(
        self,
        transparency_data: DevilsAdvocateTransparencyData,
        user_profile: UserProfile,
    ) -> CognitiveLoadLevel:
        """Assess cognitive load for Devil's Advocate transparency content"""

        # Count complexity factors
        thinking_steps_count = len(transparency_data.thinking_steps)
        challenge_systems_count = len(
            set(step.system_type for step in transparency_data.thinking_steps)
        )
        total_insights = sum(
            len(step.insights_generated) for step in transparency_data.thinking_steps
        )

        # Calculate complexity score
        complexity_score = (
            thinking_steps_count * 0.4
            + challenge_systems_count * 0.3
            + total_insights * 0.3
        )

        # Adjust for user expertise
        if user_profile.expertise_level == UserExpertiseLevel.TECHNICAL:
            complexity_threshold = 15.0
        elif user_profile.expertise_level == UserExpertiseLevel.ANALYTICAL:
            complexity_threshold = 10.0
        elif user_profile.expertise_level == UserExpertiseLevel.STRATEGIC:
            complexity_threshold = 7.0
        else:  # EXECUTIVE
            complexity_threshold = 4.0

        # Determine cognitive load level
        if complexity_score <= complexity_threshold * 0.5:
            return CognitiveLoadLevel.LOW
        elif complexity_score <= complexity_threshold:
            return CognitiveLoadLevel.MEDIUM
        elif complexity_score <= complexity_threshold * 1.5:
            return CognitiveLoadLevel.HIGH
        else:
            return CognitiveLoadLevel.OVERWHELMING

    def create_devils_advocate_thinking_visualization(
        self, transparency_data: DevilsAdvocateTransparencyData
    ) -> Dict[str, Any]:
        """Create visualization of Devil's Advocate thinking process"""

        # Create thinking flow visualization
        thinking_flow = {
            "type": "thinking_process_flow",
            "nodes": [],
            "edges": [],
            "clusters": [],
        }

        # Create nodes for each thinking step
        for i, step in enumerate(transparency_data.thinking_steps):
            node = {
                "id": step.step_id,
                "label": f"{step.system_type.value.replace('_', ' ').title()}",
                "phase": step.phase.value,
                "thinking_content": (
                    step.thinking_content[:100] + "..."
                    if len(step.thinking_content) > 100
                    else step.thinking_content
                ),
                "insights_count": len(step.insights_generated),
                "biases_detected": len(step.biases_detected),
                "confidence_impact": step.confidence_impact,
                "position": {"x": i * 200, "y": 100},
                "style": {
                    "color": self._phase_to_color(step.phase),
                    "size": self._impact_to_size(step.confidence_impact),
                },
            }
            thinking_flow["nodes"].append(node)

            # Create edge to next step
            if i < len(transparency_data.thinking_steps) - 1:
                edge = {
                    "from": step.step_id,
                    "to": transparency_data.thinking_steps[i + 1].step_id,
                    "type": "thinking_flow",
                    "style": {"arrow": True, "color": "#666"},
                }
                thinking_flow["edges"].append(edge)

        # Create phase clusters
        phase_clusters = {}
        for step in transparency_data.thinking_steps:
            phase = step.phase.value
            if phase not in phase_clusters:
                phase_clusters[phase] = []
            phase_clusters[phase].append(step.step_id)

        for phase, step_ids in phase_clusters.items():
            cluster = {
                "id": f"cluster_{phase}",
                "label": phase.replace("_", " ").title(),
                "nodes": step_ids,
                "style": {"background": self._phase_to_cluster_color(phase)},
            }
            thinking_flow["clusters"].append(cluster)

        return thinking_flow

    def _phase_to_color(self, phase: DevilsAdvocatePhase) -> str:
        """Convert phase to node color"""
        phase_colors = {
            DevilsAdvocatePhase.PHASE_1_INITIAL_ANALYSIS: "#E3F2FD",  # Light blue
            DevilsAdvocatePhase.PHASE_2_CHALLENGE_ACTIVATION: "#FFECB3",  # Light orange
            DevilsAdvocatePhase.PHASE_3_REFINEMENT: "#E8F5E8",  # Light green
        }
        return phase_colors.get(phase, "#F5F5F5")

    def _phase_to_cluster_color(self, phase: str) -> str:
        """Convert phase string to cluster color"""
        if "initial" in phase:
            return "#E3F2FD"
        elif "challenge" in phase:
            return "#FFECB3"
        elif "refinement" in phase:
            return "#E8F5E8"
        else:
            return "#F5F5F5"

    def _impact_to_size(self, confidence_impact: float) -> int:
        """Convert confidence impact to node size"""
        return int(30 + (abs(confidence_impact) * 50))  # 30-80 pixel range


# Global instance
_devils_advocate_transparency_engine: Optional[DevilsAdvocateTransparencyEngine] = None


def get_devils_advocate_transparency_engine() -> DevilsAdvocateTransparencyEngine:
    """Get or create global Devil's Advocate transparency engine"""
    global _devils_advocate_transparency_engine

    if _devils_advocate_transparency_engine is None:
        _devils_advocate_transparency_engine = DevilsAdvocateTransparencyEngine()

    return _devils_advocate_transparency_engine


# Utility functions
def create_devils_advocate_transparency(
    devils_advocate_proof_data: Dict[str, Any], user_profile: UserProfile
) -> TransparencyContent:
    """Create Devil's Advocate glass-box transparency content"""

    engine = get_devils_advocate_transparency_engine()

    # Extract thinking process
    transparency_data = engine.extract_thinking_process_from_proof_data(
        devils_advocate_proof_data
    )

    # Generate transparency content
    return engine.generate_glass_box_transparency_content(
        transparency_data, user_profile
    )


def create_devils_advocate_thinking_visualization(
    devils_advocate_proof_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Create Devil's Advocate thinking process visualization"""

    engine = get_devils_advocate_transparency_engine()

    # Extract thinking process
    transparency_data = engine.extract_thinking_process_from_proof_data(
        devils_advocate_proof_data
    )

    # Create visualization
    return engine.create_devils_advocate_thinking_visualization(transparency_data)
