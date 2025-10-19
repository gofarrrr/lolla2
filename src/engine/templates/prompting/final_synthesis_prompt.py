#!/usr/bin/env python3
"""
Final Synthesis Prompt Template - Operation Synthesis
McKinsey Senior Partner synthesis prompt that integrates Devil's Advocate critique
"""

from typing import Dict, Any
from .base_template import BasePromptTemplate, PromptContext


class FinalSynthesisPrompt(BasePromptTemplate):
    """
    McKinsey Senior Partner synthesis template that resolves internal debate
    and forges final house view from cognitive journey data
    """

    def __init__(self):
        super().__init__()
        self.template_name = "FinalSynthesisPrompt"
        self.research_basis = "McKinsey Senior Partner decision synthesis methodology"
        self.performance_characteristics = {
            "integration_quality": "High - forces synthesis of all inputs",
            "thinking_transparency": "Complete - shows reasoning process",
            "refinement_evidence": "Explicit - tracks transformation journey",
        }

    def generate_synthesis_prompt(self, synthesis_payload: Dict[str, Any]) -> str:
        """
        Generate the complete synthesis prompt from SynthesisInputContract payload

        Args:
            synthesis_payload: The output from SynthesisInputContract.to_synthesis_payload()

        Returns:
            Complete synthesis prompt for final integration
        """

        trinity_inputs = synthesis_payload["trinity_inputs"]
        context = synthesis_payload["context"]
        transparency_trail = synthesis_payload["transparency_trail"]

        # Extract key challenges and biases from Devil's Advocate
        challenges_summary = self._extract_challenges_summary(
            trinity_inputs["devils_advocate_critique"]
        )
        bias_corrections = self._extract_bias_corrections(
            trinity_inputs["devils_advocate_critique"]
        )
        confidence_evolution = transparency_trail.get("confidence_evolution", {})

        # ULTRATHINK: Extract contradictions if available
        contradictions_summary = self._extract_contradictions_summary(
            trinity_inputs["devils_advocate_critique"]
        )

        synthesis_prompt = f"""You are a McKinsey Senior Partner conducting the final synthesis session for a critical strategic engagement. Your role is to resolve the internal analytical debate and forge the definitive "house view" that integrates all perspectives.

## THE COGNITIVE JOURNEY DATA

### INITIAL ANALYSIS
{trinity_inputs["initial_conclusion"]}

### DEVIL'S ADVOCATE CRITIQUE FINDINGS
The internal challenge system identified these critical concerns:

**Cognitive Biases Detected & Corrected:**
{self._format_bias_list(bias_corrections)}

**Key Challenges Applied:**
{challenges_summary}

### UNRESOLVED CONTRADICTIONS (ULTRATHINK)
{contradictions_summary}

**Confidence Evolution Through Challenge Process:**
- Initial Confidence: {confidence_evolution.get('pre_challenge_confidence', 'N/A')}
- Mid-Challenge (Lowest Point): {confidence_evolution.get('mid_challenge_confidence', 'N/A')}  
- Post-Refinement: {confidence_evolution.get('post_refinement_confidence', 'N/A')}

### RESEARCH VALIDATION
{self._format_research_findings(trinity_inputs["grounding_research"])}

## YOUR SYNTHESIS MANDATE

As Senior Partner, you must:

1. **ACKNOWLEDGE THE JOURNEY**: Recognize that the initial analysis went through rigorous internal challenge
2. **INTEGRATE ALL PERSPECTIVES**: Synthesize initial thinking, critique insights, and research validation
3. **RESOLVE THE DEBATE**: Make clear decisions where initial analysis and critique diverged
4. **FORGE HOUSE VIEW**: Deliver a refined, sophisticated recommendation that reflects the full cognitive process
5. **SHOW YOUR REASONING**: Provide glass-box transparency into your synthesis thinking process

## SYNTHESIS FRAMEWORK

Structure your response as:

### EXECUTIVE SYNTHESIS
[Your final integrated recommendation - this is the refined "house view"]

### SYNTHESIS REASONING PROCESS
[Show your thinking: How you integrated the initial analysis, addressed the Devil's Advocate challenges, and incorporated research findings]

### REFINEMENT EVIDENCE  
[Explicitly document what changed from initial to final - show the value of the challenge process]

### INTEGRATION QUALITY ASSESSMENT
[Rate how well you integrated all inputs: Initial Analysis (%), Devil's Advocate Insights (%), Research Validation (%)]

### CONFIDENCE JUSTIFICATION
[Explain your final confidence level and how it compares to the confidence journey]

## CRITICAL REQUIREMENTS

- **DO NOT IGNORE** the Devil's Advocate findings - they prevented flawed conclusions
- **DO NOT SIMPLY REPEAT** the initial analysis - demonstrate synthesis and refinement  
- **DO SHOW** how the challenge process improved the final recommendation
- **DO PROVIDE** specific evidence of integration, not generic statements
- **DO MAINTAIN** McKinsey-level sophistication and strategic rigor

The client expects to see that our internal challenge system created tangible value. Prove that the cognitive journey from initial analysis through Devil's Advocate critique to final synthesis produced a superior outcome.

Begin your synthesis now."""

        return synthesis_prompt

    def _extract_challenges_summary(self, devils_advocate_data: Dict[str, Any]) -> str:
        """Extract and format key challenges from Devil's Advocate data"""

        try:
            if "devils_advocate_impact_analysis" in devils_advocate_data:
                impact_analysis = devils_advocate_data[
                    "devils_advocate_impact_analysis"
                ]
                if "transformation_journey" in impact_analysis:
                    challenges = impact_analysis["transformation_journey"].get(
                        "critical_challenges_applied", []
                    )

                    if challenges:
                        formatted_challenges = []
                        for i, challenge in enumerate(challenges, 1):
                            challenge_type = challenge.get("type", "Unknown")
                            insight = challenge.get("insight", "No insight provided")
                            impact = challenge.get("impact", "No impact documented")

                            formatted_challenges.append(
                                f"{i}. **{challenge_type}**: {insight}\n   → Impact: {impact}"
                            )

                        return "\n".join(formatted_challenges)

            # Fallback if structure is different
            return "Critical challenges were applied but detailed extraction failed - synthesis should proceed with available data"

        except Exception as e:
            return f"Challenge extraction error - proceed with synthesis: {str(e)}"

    def _extract_bias_corrections(self, devils_advocate_data: Dict[str, Any]) -> list:
        """Extract bias corrections from Devil's Advocate data"""

        try:
            if "execution_phases" in devils_advocate_data:
                phase_2 = devils_advocate_data["execution_phases"].get(
                    "phase_2_devils_advocate_activation", {}
                )
                if "challenge_systems_activated" in phase_2:
                    munger = phase_2["challenge_systems_activated"].get(
                        "munger_overlay", {}
                    )
                    return munger.get("biases_identified", [])

            return ["Bias detection data not available in expected format"]

        except Exception as e:
            return [f"Bias extraction error: {str(e)}"]

    def _format_bias_list(self, biases: list) -> str:
        """Format bias list for prompt"""
        if not biases:
            return "- No biases documented in expected format"

        return "\n".join([f"- {bias}" for bias in biases])

    def _format_research_findings(self, research_data: Dict[str, Any]) -> str:
        """Format research findings for synthesis prompt"""

        if not research_data:
            return "No research validation data available"

        # Handle different research data structures
        if isinstance(research_data, dict):
            if "summary" in research_data:
                return research_data["summary"]
            elif "findings" in research_data:
                return str(research_data["findings"])
            else:
                return f"Research validation performed: {len(research_data)} data points available for synthesis"

        return (
            str(research_data)[:500] + "..."
            if len(str(research_data)) > 500
            else str(research_data)
        )

    def _extract_contradictions_summary(
        self, devils_advocate_data: Dict[str, Any]
    ) -> str:
        """
        ULTRATHINK: Extract and format contradictions from Devils Advocate data

        This ensures contradictions are explicitly surfaced in the synthesis,
        forcing intellectual honesty about unresolved disagreements.
        """

        try:
            # Look for contradictions in different possible locations
            contradictions = []

            # Direct contradictions from ULTRATHINK system
            if "contradictions" in devils_advocate_data:
                contradictions = devils_advocate_data["contradictions"]

            # Alternative structure from processing_details
            elif (
                "processing_details" in devils_advocate_data
                and "contradictions_detected"
                in devils_advocate_data["processing_details"]
            ):
                contradiction_count = devils_advocate_data["processing_details"][
                    "contradictions_detected"
                ]
                if contradiction_count > 0:
                    # Create placeholder if we know there are contradictions but can't access details
                    return f"The challenge system identified {contradiction_count} contradictions between engines that require explicit resolution in your synthesis."

            if not contradictions:
                return "No contradictions detected between challenge engines. All critical challenges are aligned."

            # Format contradictions for prompt
            formatted_contradictions = []
            for i, contradiction in enumerate(contradictions, 1):
                engine_a = contradiction.get("engine_a", "Engine A")
                engine_b = contradiction.get("engine_b", "Engine B")
                claim_a = contradiction.get("claim_a", "Unknown claim")
                claim_b = contradiction.get("claim_b", "Unknown claim")
                severity = contradiction.get("severity", 0.5)

                formatted_contradictions.append(
                    f"{i}. **{engine_a}** vs **{engine_b}** (Severity: {severity:.2f})\n"
                    f"   • {engine_a}: {claim_a}\n"
                    f"   • {engine_b}: {claim_b}\n"
                    f"   → **YOU MUST EXPLICITLY RESOLVE THIS CONTRADICTION IN YOUR SYNTHESIS**"
                )

            contradiction_header = (
                "The following contradictions between critics remain unresolved:\n\n"
            )
            contradiction_footer = "\n\n**CRITICAL**: You MUST explicitly address how you resolve each contradiction in your synthesis."

            return (
                contradiction_header
                + "\n\n".join(formatted_contradictions)
                + contradiction_footer
            )

        except Exception as e:
            return f"Error extracting contradictions - proceed with synthesis: {str(e)}"

    def generate_prompt(self, original_prompt: str, context: PromptContext) -> str:
        """
        Standard interface - not used for synthesis (use generate_synthesis_prompt instead)
        """
        return f"""This is the final synthesis template. Use generate_synthesis_prompt() method instead.
        
Original prompt: {original_prompt}
Context: {context}"""

    def get_expected_performance_improvement(self) -> Dict[str, float]:
        """Expected improvements from proper synthesis integration"""
        return {
            "integration_quality": 0.85,  # 85% improvement in input integration
            "refinement_evidence": 0.90,  # 90% improvement in showing transformation
            "thinking_transparency": 0.95,  # 95% improvement in reasoning visibility
            "decision_sophistication": 0.75,  # 75% improvement in recommendation quality
        }

    def get_synthesis_requirements(self) -> Dict[str, str]:
        """Get synthesis-specific requirements"""
        return {
            "role": "McKinsey Senior Partner",
            "mandate": "Resolve internal debate and forge house view",
            "integration_target": "All three inputs: initial analysis, critique, research",
            "transparency_requirement": "Glass-box reasoning process",
            "evidence_requirement": "Explicit refinement documentation",
            "quality_standard": "Demonstrate value of challenge process",
        }
