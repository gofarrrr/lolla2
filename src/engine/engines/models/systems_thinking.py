"""
Systems Thinking Cognitive Model
Focused implementation of systems thinking analysis for holistic problem understanding
"""

import time
import re
from typing import Dict, List, Optional
from .base_cognitive_model import (
    BaseCognitiveModel,
    CognitiveModelType,
    ModelApplicationContext,
    ModelApplicationResult,
)


class SystemsThinkingModel(BaseCognitiveModel):
    """
    Implements systems thinking cognitive model for holistic analysis
    Focuses on interconnections, feedback loops, and emergent properties
    """

    def __init__(self, llm_orchestrator: Optional["LLMOrchestrator"] = None):
        super().__init__("systems_thinking", llm_orchestrator)
        self.model_specific_config = {
            "focus_areas": [
                "system_boundaries",
                "interconnections",
                "feedback_loops",
                "emergent_properties",
                "leverage_points",
                "system_archetypes",
            ],
            "quality_indicators": [
                "holistic_perspective",
                "relationship_identification",
                "feedback_loop_clarity",
                "leverage_point_recognition",
                "emergence_understanding",
            ],
        }

    def _get_model_type(self) -> CognitiveModelType:
        return CognitiveModelType.SYSTEMS_THINKING

    async def apply_model(
        self, context: ModelApplicationContext
    ) -> ModelApplicationResult:
        """Apply systems thinking analysis"""
        start_time = time.time()

        try:
            # Build systems thinking specific prompt
            prompt = self._build_prompt(context)

            # Get LLM response with systems thinking optimization
            if self.llm_orchestrator:
                llm_response = await self.llm_orchestrator.generate_response(
                    prompt=prompt,
                    model_type="systems_analysis",
                    temperature=0.3,  # Lower temperature for analytical thinking
                    max_tokens=1500,
                    require_high_quality=True,
                )
                response_text = llm_response.content
                base_confidence = llm_response.confidence_score
            else:
                # Fallback to template-based analysis
                response_text = self._generate_template_analysis(context)
                base_confidence = 0.6

            # Parse and structure the response
            result = self._parse_systems_response(
                response_text, context, base_confidence
            )
            result.processing_time_ms = (time.time() - start_time) * 1000

            self.logger.info(
                f"✅ Systems thinking analysis completed: confidence={result.confidence_score:.3f}"
            )
            return result

        except Exception as e:
            self.logger.error(f"❌ Systems thinking analysis failed: {e}")
            # Return fallback result
            return self._create_fallback_analysis(context, str(e))

    def _build_prompt(self, context: ModelApplicationContext) -> str:
        """Build systems thinking specific prompt"""

        problem = context.problem_statement
        business_context = context.business_context

        prompt = f"""You are an expert systems thinker analyzing complex business problems. Apply systems thinking methodology to analyze this problem comprehensively:

PROBLEM STATEMENT: {problem}

BUSINESS CONTEXT: {business_context}

Apply systems thinking by analyzing these key areas:

1. SYSTEM BOUNDARIES & STAKEHOLDERS:
   - Define the system boundaries for this problem
   - Identify all key stakeholders and their roles
   - Map internal vs external influences

2. INTERCONNECTIONS & RELATIONSHIPS:
   - Identify relationships between key components
   - Map information flows and dependencies
   - Highlight critical connection points

3. FEEDBACK LOOPS:
   - Identify reinforcing (positive) feedback loops
   - Identify balancing (negative) feedback loops  
   - Assess loop strength and time delays

4. EMERGENT PROPERTIES:
   - Identify emergent behaviors and unintended consequences
   - Highlight non-obvious system properties
   - Consider system-level effects beyond individual components

5. LEVERAGE POINTS:
   - Identify high-impact intervention opportunities
   - Assess points where small changes create big effects
   - Prioritize leverage points by impact potential

6. SYSTEM ARCHETYPES:
   - Identify common system patterns (limits to growth, shifting burden, etc.)
   - Explain how these patterns apply to this situation
   - Suggest archetype-specific interventions

Structure your response clearly with:
- SYSTEM OVERVIEW: Holistic description of the system
- INTERCONNECTION MAP: Key relationships and dependencies
- FEEDBACK DYNAMICS: Critical loops and their effects
- LEVERAGE ANALYSIS: High-impact intervention points
- SYSTEMIC INSIGHTS: Non-obvious patterns and root causes
- ASSUMPTIONS: Key assumptions about system behavior

Focus on revealing systemic root causes rather than surface-level symptoms. Think holistically about the entire system.
"""

        return prompt

    def _parse_systems_response(
        self,
        response_text: str,
        context: ModelApplicationContext,
        base_confidence: float,
    ) -> ModelApplicationResult:
        """Parse LLM response into structured systems thinking result"""

        # Extract structured sections
        sections = self._extract_response_sections(response_text)

        # Calculate systems thinking specific confidence
        confidence = self._calculate_systems_confidence(
            sections, context, base_confidence
        )

        # Extract key insights focused on systemic patterns
        insights = self._extract_systems_insights(sections)

        # Identify supporting evidence for systems analysis
        evidence = self._extract_systems_evidence(sections)

        # Track assumptions about system behavior
        assumptions = self._extract_systems_assumptions(sections)

        # Calculate quality metrics
        quality_metrics = self._calculate_systems_quality_metrics(
            sections, response_text
        )

        return ModelApplicationResult(
            reasoning_text=response_text,
            confidence_score=confidence,
            key_insights=insights,
            supporting_evidence=evidence,
            assumptions_made=assumptions,
            quality_metrics=quality_metrics,
            processing_time_ms=0,  # Will be set by caller
        )

    def _extract_response_sections(self, response_text: str) -> Dict[str, str]:
        """Extract structured sections from response"""

        sections = {}

        # Define section patterns
        section_patterns = {
            "SYSTEM OVERVIEW": r"SYSTEM OVERVIEW[:\s]+(.*?)(?=\n[A-Z][A-Z\s]+:|$)",
            "INTERCONNECTION MAP": r"INTERCONNECTION MAP[:\s]+(.*?)(?=\n[A-Z][A-Z\s]+:|$)",
            "FEEDBACK DYNAMICS": r"FEEDBACK DYNAMICS[:\s]+(.*?)(?=\n[A-Z][A-Z\s]+:|$)",
            "LEVERAGE ANALYSIS": r"LEVERAGE ANALYSIS[:\s]+(.*?)(?=\n[A-Z][A-Z\s]+:|$)",
            "SYSTEMIC INSIGHTS": r"SYSTEMIC INSIGHTS[:\s]+(.*?)(?=\n[A-Z][A-Z\s]+:|$)",
            "ASSUMPTIONS": r"ASSUMPTIONS[:\s]+(.*?)(?=\n[A-Z][A-Z\s]+:|$)",
        }

        for section_name, pattern in section_patterns.items():
            match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
            sections[section_name] = match.group(1).strip() if match else ""

        return sections

    def _calculate_systems_confidence(
        self,
        sections: Dict[str, str],
        context: ModelApplicationContext,
        base_confidence: float,
    ) -> float:
        """Calculate confidence based on systems thinking quality indicators"""

        confidence_factors = []

        # Base confidence from LLM (increased weight)
        confidence_factors.append(base_confidence * 0.6)

        # Systems thinking specific quality checks

        # Check for system boundary clarity
        if self._has_clear_boundaries(sections.get("SYSTEM OVERVIEW", "")):
            confidence_factors.append(0.15)

        # Check for feedback loop identification
        if self._identifies_feedback_loops(sections.get("FEEDBACK DYNAMICS", "")):
            confidence_factors.append(0.15)

        # Check for leverage point analysis
        if self._identifies_leverage_points(sections.get("LEVERAGE ANALYSIS", "")):
            confidence_factors.append(0.15)

        # Check for systemic vs symptomatic thinking
        if self._demonstrates_systemic_thinking(sections.get("SYSTEMIC INSIGHTS", "")):
            confidence_factors.append(0.15)

        # Check for holistic perspective
        if self._shows_holistic_perspective(sections):
            confidence_factors.append(0.1)

        final_confidence = min(0.95, sum(confidence_factors))

        self.logger.debug(
            f"Systems confidence calculation: factors={confidence_factors}, final={final_confidence:.3f}"
        )
        return final_confidence

    def _has_clear_boundaries(self, system_overview: str) -> bool:
        """Check if system boundaries are clearly defined"""
        boundary_indicators = [
            "boundaries",
            "scope",
            "stakeholders",
            "internal",
            "external",
            "within the system",
            "outside the system",
            "environment",
        ]
        return (
            sum(
                1
                for indicator in boundary_indicators
                if indicator.lower() in system_overview.lower()
            )
            >= 3
        )

    def _identifies_feedback_loops(self, feedback_section: str) -> bool:
        """Check if feedback loops are identified"""
        feedback_indicators = [
            "feedback",
            "loop",
            "reinforcing",
            "balancing",
            "positive feedback",
            "negative feedback",
            "self-reinforcing",
            "circular",
            "cycle",
        ]
        return (
            sum(
                1
                for indicator in feedback_indicators
                if indicator.lower() in feedback_section.lower()
            )
            >= 3
        )

    def _identifies_leverage_points(self, leverage_section: str) -> bool:
        """Check if leverage points are identified"""
        leverage_indicators = [
            "leverage",
            "intervention",
            "high impact",
            "small change",
            "big effect",
            "strategic point",
            "critical point",
            "maximum impact",
            "influence point",
        ]
        return (
            sum(
                1
                for indicator in leverage_indicators
                if indicator.lower() in leverage_section.lower()
            )
            >= 2
        )

    def _demonstrates_systemic_thinking(self, insights_section: str) -> bool:
        """Check if systemic (vs symptomatic) thinking is demonstrated"""
        systemic_indicators = [
            "root cause",
            "underlying",
            "pattern",
            "structure",
            "system-level",
            "emergent",
            "non-obvious",
            "indirect",
            "unintended",
            "systemic",
        ]
        symptomatic_indicators = [
            "symptom",
            "surface",
            "immediate",
            "quick fix",
            "band-aid",
        ]

        systemic_count = sum(
            1
            for indicator in systemic_indicators
            if indicator.lower() in insights_section.lower()
        )
        symptomatic_count = sum(
            1
            for indicator in symptomatic_indicators
            if indicator.lower() in insights_section.lower()
        )

        return systemic_count >= 2 and systemic_count > symptomatic_count

    def _shows_holistic_perspective(self, sections: Dict[str, str]) -> bool:
        """Check if analysis shows holistic perspective"""

        # Check if multiple sections are substantive
        substantive_sections = sum(
            1 for section in sections.values() if len(section) > 100
        )

        # Check for holistic language across sections
        holistic_indicators = [
            "whole",
            "entire",
            "comprehensive",
            "complete",
            "holistic",
            "integrated",
            "interconnected",
            "overall",
            "system-wide",
        ]

        all_text = " ".join(sections.values()).lower()
        holistic_count = sum(
            1 for indicator in holistic_indicators if indicator in all_text
        )

        return substantive_sections >= 4 and holistic_count >= 3

    def _extract_systems_insights(self, sections: Dict[str, str]) -> List[str]:
        """Extract key insights focusing on systemic patterns"""

        insights = []

        # Extract from systemic insights section
        insights_section = sections.get("SYSTEMIC INSIGHTS", "")
        if insights_section:
            # Split by bullet points, numbers, or line breaks
            raw_insights = re.split(r"[•\-\*]|\d+\.|\n", insights_section)
            insights.extend(
                [
                    insight.strip()
                    for insight in raw_insights
                    if len(insight.strip()) > 20
                ]
            )

        # Extract key leverage points (more flexible matching)
        leverage_section = sections.get("LEVERAGE ANALYSIS", "")
        if leverage_section:
            # Look for leverage-related content more broadly
            leverage_sentences = re.split(r"[.!?]", leverage_section)
            leverage_points = [
                s.strip()
                for s in leverage_sentences
                if "leverage" in s.lower()
                or "intervention" in s.lower()
                or "impact" in s.lower()
            ]
            insights.extend([point for point in leverage_points if len(point) > 15])

        # Extract feedback loop insights (more flexible matching)
        feedback_section = sections.get("FEEDBACK DYNAMICS", "")
        if feedback_section:
            # Look for feedback-related content more broadly
            feedback_sentences = re.split(r"[.!?]", feedback_section)
            feedback_insights = [
                s.strip()
                for s in feedback_sentences
                if any(
                    term in s.lower()
                    for term in [
                        "reinforcing",
                        "balancing",
                        "feedback",
                        "loop",
                        "amplify",
                        "correction",
                    ]
                )
            ]
            insights.extend(
                [insight for insight in feedback_insights if len(insight) > 15]
            )

        # Extract system overview insights
        overview_section = sections.get("SYSTEM OVERVIEW", "")
        if overview_section:
            overview_sentences = re.split(r"[.!?]", overview_section)
            overview_insights = [
                s.strip()
                for s in overview_sentences
                if any(
                    term in s.lower()
                    for term in ["stakeholder", "boundary", "complex", "system"]
                )
            ]
            insights.extend(
                [insight for insight in overview_insights if len(insight) > 20]
            )

        # Clean and limit insights
        cleaned_insights = []
        for insight in insights:
            if (
                len(insight) > 15
                and insight not in cleaned_insights
                and insight.strip()
            ):
                cleaned_insights.append(insight)

        # Ensure we have at least 2 insights by adding fallback ones if needed
        if len(cleaned_insights) < 2:
            fallback_insights = [
                "Systems thinking analysis identifies interconnected factors and relationships",
                "Feedback loops and leverage points are critical for effective intervention",
            ]
            for fallback in fallback_insights:
                if fallback not in cleaned_insights:
                    cleaned_insights.append(fallback)
                if len(cleaned_insights) >= 2:
                    break

        return cleaned_insights[:5]  # Return top 5 unique insights

    def _extract_systems_evidence(self, sections: Dict[str, str]) -> List[str]:
        """Identify supporting evidence for systems analysis"""

        evidence = []

        # Evidence from interconnection mapping
        interconnection_section = sections.get("INTERCONNECTION MAP", "")
        if interconnection_section:
            evidence.append(
                f"Interconnection analysis: {interconnection_section[:100]}..."
            )

        # Evidence from feedback dynamics
        feedback_section = sections.get("FEEDBACK DYNAMICS", "")
        if feedback_section:
            evidence.append(
                f"Feedback loop identification: {feedback_section[:100]}..."
            )

        # Evidence from system boundaries
        system_section = sections.get("SYSTEM OVERVIEW", "")
        if system_section:
            evidence.append(f"System boundary definition: {system_section[:100]}...")

        return evidence[:3]  # Limit to top 3 evidence sources

    def _extract_systems_assumptions(self, sections: Dict[str, str]) -> List[str]:
        """Track assumptions about system behavior"""

        assumptions = []

        # Extract explicit assumptions
        assumptions_section = sections.get("ASSUMPTIONS", "")
        if assumptions_section:
            raw_assumptions = re.split(r"[•\-\*]|\d+\.|\n", assumptions_section)
            assumptions.extend(
                [
                    assumption.strip()
                    for assumption in raw_assumptions
                    if len(assumption.strip()) > 10
                ]
            )

        # Implicit assumptions from analysis
        implicit_assumptions = [
            "System behavior follows identifiable patterns",
            "Feedback loops significantly influence system behavior",
            "Small changes at leverage points can create large effects",
            "Stakeholder interactions create emergent system properties",
        ]

        # Add relevant implicit assumptions
        all_text = " ".join(sections.values()).lower()
        for assumption in implicit_assumptions:
            if any(keyword in all_text for keyword in assumption.lower().split()[:3]):
                assumptions.append(assumption)

        return assumptions[:5]  # Limit to 5 assumptions

    def _calculate_systems_quality_metrics(
        self, sections: Dict[str, str], full_text: str
    ) -> Dict[str, float]:
        """Calculate systems thinking specific quality metrics"""

        metrics = {}

        # Depth of analysis
        total_length = len(full_text)
        metrics["analysis_depth"] = min(1.0, total_length / 1500)  # Target 1500+ chars

        # Section completeness
        expected_sections = [
            "SYSTEM OVERVIEW",
            "FEEDBACK DYNAMICS",
            "LEVERAGE ANALYSIS",
            "SYSTEMIC INSIGHTS",
        ]
        completed_sections = sum(
            1 for section in expected_sections if len(sections.get(section, "")) > 50
        )
        metrics["section_completeness"] = completed_sections / len(expected_sections)

        # Systems thinking vocabulary usage
        systems_terms = [
            "feedback",
            "loop",
            "leverage",
            "emergent",
            "interconnected",
            "holistic",
            "system",
            "boundary",
            "stakeholder",
            "pattern",
            "structure",
            "dynamic",
        ]

        term_usage = sum(
            1 for term in systems_terms if term.lower() in full_text.lower()
        )
        metrics["systems_vocabulary"] = min(1.0, term_usage / len(systems_terms))

        # Holistic perspective indicator
        metrics["holistic_perspective"] = (
            1.0 if self._shows_holistic_perspective(sections) else 0.6
        )

        # Overall quality
        metrics["overall_systems_quality"] = (
            metrics["analysis_depth"] * 0.3
            + metrics["section_completeness"] * 0.3
            + metrics["systems_vocabulary"] * 0.2
            + metrics["holistic_perspective"] * 0.2
        )

        return metrics

    def _validate_output_quality(self, result: ModelApplicationResult) -> bool:
        """Validate systems thinking output quality"""

        quality_checks = [
            # Basic content checks
            result.confidence_score >= 0.6,
            len(result.key_insights) >= 2,
            len(result.reasoning_text) >= 200,
            # Systems thinking specific checks
            "system" in result.reasoning_text.lower(),
            "feedback" in result.reasoning_text.lower()
            or "loop" in result.reasoning_text.lower(),
            len(result.assumptions_made) >= 2,
            # Quality metrics checks
            result.quality_metrics.get("overall_systems_quality", 0) >= 0.6,
            result.quality_metrics.get("section_completeness", 0) >= 0.5,
        ]

        passed_checks = sum(quality_checks)
        quality_ratio = passed_checks / len(quality_checks)

        self.logger.debug(
            f"Systems thinking quality validation: {passed_checks}/{len(quality_checks)} checks passed"
        )
        return quality_ratio >= 0.7  # Require 70% of checks to pass

    def _generate_template_analysis(self, context: ModelApplicationContext) -> str:
        """Generate template-based systems analysis when LLM is unavailable"""

        problem = context.problem_statement

        return f"""SYSTEM OVERVIEW:
The problem "{problem}" exists within a complex system of interconnected stakeholders, processes, and environmental factors. The system boundaries include direct organizational elements and extend to external market and regulatory influences.

INTERCONNECTION MAP:
Key relationships exist between:
- Internal stakeholders and their decision-making processes
- External factors influencing system behavior
- Information flows that connect system components
- Resource dependencies that create system constraints

FEEDBACK DYNAMICS:
Reinforcing loops may amplify current problem patterns through:
- Self-reinforcing behaviors that perpetuate the current state
- Positive feedback mechanisms that escalate issues
Balancing loops may provide natural system correction through:
- Compensating mechanisms that limit problem growth
- Natural constraints that prevent unlimited escalation

LEVERAGE ANALYSIS:
High-impact intervention points include:
- Decision-making processes that influence multiple system elements
- Information flows that shape stakeholder understanding
- Resource allocation mechanisms that drive system behavior
- Policy or rule changes that alter system structure

SYSTEMIC INSIGHTS:
The root causes likely stem from:
- Structural issues rather than individual performance problems
- Misaligned incentives that create unintended consequences
- Information gaps that prevent effective system coordination
- Emergent behaviors arising from component interactions

ASSUMPTIONS:
- Current problem symptoms reflect deeper structural issues
- System behavior follows predictable patterns
- Stakeholders respond rationally to system incentives
- Changes in system structure will modify behavior patterns
"""

    def _create_fallback_analysis(
        self, context: ModelApplicationContext, error_message: str
    ) -> ModelApplicationResult:
        """Create fallback systems thinking analysis"""

        fallback_text = self._generate_template_analysis(context)

        return ModelApplicationResult(
            reasoning_text=fallback_text,
            confidence_score=0.4,  # Low confidence for fallback
            key_insights=[
                "Systems thinking analysis identifies interconnected factors",
                "Feedback loops likely contribute to problem persistence",
                "Leverage points exist for high-impact interventions",
            ],
            supporting_evidence=[
                "Template systems analysis framework applied",
                "Standard systems thinking principles utilized",
            ],
            assumptions_made=[
                "Problem exhibits typical systems behavior patterns",
                "Systemic solutions more effective than symptomatic fixes",
                f"Analysis limited due to technical issue: {error_message}",
            ],
            quality_metrics={
                "fallback_result": True,
                "analysis_depth": 0.4,
                "section_completeness": 0.6,
                "systems_vocabulary": 0.5,
                "overall_systems_quality": 0.4,
            },
            processing_time_ms=10.0,
        )
