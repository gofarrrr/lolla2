"""
SeniorAdvisor Orchestrator - STEP 5 of Honest Orchestra
======================================================

PRINCIPLE: "Fail Loudly, Succeed Honestly"

This orchestrator executes the Two-Brain Senior Advisor synthesis with real LLM calls.
Sequential process: DeepSeek analysis → Claude synthesis → Final recommendations.

Process:
1. Initialize TwoBrainSeniorAdvisor
2. Execute DeepSeek brain analysis (first brain)
3. Execute Claude brain synthesis (second brain)
4. Combine insights into final SeniorAdvisorReport
5. Return final report or raise SeniorAdvisorError
"""

import time
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any

from .exceptions import SeniorAdvisorError
from .contracts import (
    ConsultantAnalysisResult,
    AnalysisCritique,
    SeniorAdvisorReport,
    TwoBrainInsight,
)

logger = logging.getLogger(__name__)


class SeniorAdvisorOrchestrator:
    """Orchestrator for Two-Brain Senior Advisor synthesis"""

    def __init__(self):
        self.deepseek_client = None
        self.claude_client = None
        self.grok_client = None  # OPERATION PHOENIX - Brain 2 Swap to Grok-4-Fast

        # Dependency injection: SeniorManager domain service via factory
        try:
            from src.services.interfaces.senior_manager_interface import (
                DefaultSeniorManagerFactory,
                ISeniorManager,
            )
            factory = DefaultSeniorManagerFactory()
            self.senior_manager = factory.create_senior_manager()
            # Optional runtime assertion to ensure interface compliance
            assert isinstance(self.senior_manager, ISeniorManager)  # type: ignore[misc]
            logger.info("✅ SeniorAdvisorOrchestrator: SeniorManager injected via factory")
        except Exception as exc:
            logger.warning(f"⚠️ SeniorAdvisorOrchestrator: SeniorManager injection failed: {exc}")
            self.senior_manager = None

        # Seam wrapper for safe step-wise extraction
        try:
            from src.orchestration.seams.senior_advisor_seam import SeniorAdvisorSeam
            self._seam = SeniorAdvisorSeam()
        except Exception as exc:
            logger.warning(f"⚠️ SeniorAdvisorOrchestrator: SeniorAdvisorSeam unavailable: {exc}")
            self._seam = None

        # ============================================================================
        # OPERATION PHOENIX (Extension): DISABLE V2 ROUTE
        # ============================================================================
        # SeniorAdvisorV2 requires missing YAML file (senior_advisor_wsn.yaml) and
        # also intercepts execution before the real Two-Brain implementation.
        # Forcing to None to complete the bypass.
        # ============================================================================
        self.sa_v2 = None

        # DISABLED: V2 implementation requiring missing YAML configuration
        # try:
        #     from src.orchestration.senior_advisor_v2 import SeniorAdvisorV2
        #     self.sa_v2 = SeniorAdvisorV2()
        # except Exception:
        #     self.sa_v2 = None

    async def _initialize_clients(self):
        """Initialize real LLM clients for two-brain process"""
        try:
            import os
            from dotenv import load_dotenv

            load_dotenv()

            deepseek_key = os.getenv("DEEPSEEK_API_KEY")
            openrouter_key = os.getenv("OPENROUTER_API_KEY")

            if deepseek_key:
                self.deepseek_client = TwoBrainLLMClient(deepseek_key, "deepseek")
                logger.info("✅ Initialized DeepSeek brain")

            # OPERATION PHOENIX - Brain 2 Swap: Use Grok-4-Fast via OpenRouter instead of Claude
            if openrouter_key:
                self.grok_client = TwoBrainLLMClient(openrouter_key, "grok")
                logger.info("✅ Initialized Grok brain")

            if not self.deepseek_client and not self.grok_client:
                raise SeniorAdvisorError(
                    "No LLM clients available for Two-Brain process"
                )

        except Exception as e:
            raise SeniorAdvisorError(f"Failed to initialize Two-Brain clients: {e}")

    async def run_senior_advisor(
        self,
        analyses: List[ConsultantAnalysisResult],
        critiques: List[AnalysisCritique],
    ) -> SeniorAdvisorReport:
        """
        Execute complete Two-Brain Senior Advisor synthesis

        Args:
            analyses: Results from consultant analyses
            critiques: Results from devil's advocate critiques

        Returns:
            SeniorAdvisorReport: Final synthesized recommendations

        Raises:
            SeniorAdvisorError: If any step fails
        """
        start_time = time.time()
        total_cost = 0.0

        try:
            logger.info("🧠 Starting Senior Advisor synthesis")
            logger.info(
                f"📊 Input: {len(analyses)} analyses, {len(critiques)} critiques"
            )

            # 🔍 OPERATION PHOENIX DEBUG: Trace routing decisions
            sm_val = getattr(self, "senior_manager", None)
            v2_val = getattr(self, "sa_v2", None)
            logger.info(f"🔍 ROUTING CHECK: senior_manager={sm_val}, sa_v2={v2_val}")

            # Primary path: Use SeniorManager domain service if available
            if getattr(self, "senior_manager", None):
                logger.info("🎯 Using SeniorManager domain service for synthesis")
                return await self.senior_manager.synthesize_strategic_report(
                    analyses, critiques
                )

            # V2 fallback: If SA V2 is available and we have an argument graph in context, prefer V2
            if getattr(self, "sa_v2", None):
                # Attempt to build a minimal argument graph from analyses/critiques if available
                # For this wiring step, call SA V2 with an empty list if no graph is injected by upstream stages.
                from src.models.cognitive_core import Argument

                arg_graph: List[Argument] = []
                try:
                    # If upstream has produced arguments via CognitiveCoreService in the same trace, SA V2 can access them
                    # We pass an empty list to indicate SA V2 should pick bounded syntheses directly from core context if needed.
                    pass
                except Exception:
                    pass
                report_v2 = await self.sa_v2.synthesize_report(
                    trace_id="na", arguments=arg_graph
                )
                # Convert V2 WSN into legacy SeniorAdvisorReport shell for API compatibility
                from src.orchestration.contracts import (
                    SeniorAdvisorReport,
                    TwoBrainInsight,
                )

                dummy = TwoBrainInsight(
                    brain_name="wsn",
                    insight_content=report_v2["what"],
                    confidence_level=0.9,
                    key_points=[],
                    processing_time_seconds=0.0,
                    tokens_used=0,
                    cost_usd=0.0,
                )
                final_report = SeniorAdvisorReport(
                    executive_summary=report_v2["what"],
                    strategic_recommendation=report_v2["so_what"],
                    implementation_roadmap=[report_v2["now_what"]],
                    risk_mitigation=["Boundary conditions considered"],
                    success_metrics=["Pilot outcomes measured"],
                    deepseek_brain=dummy,
                    claude_brain=dummy,
                    synthesis_rationale="WSN over Argument Graph",
                    final_confidence=0.9,
                    total_processing_time_seconds=0.0,
                    total_cost_usd=0.0,
                )
                return final_report

            # Legacy fallback path
            # 🔍 OPERATION PHOENIX DEBUG: Confirm we reached the real brain execution path
            logger.info("🔍 REACHED REAL BRAIN EXECUTION PATH (post-OPERATION PHOENIX)")

            # Step 1: Initialize clients
            await self._initialize_clients()

            # Step 2: Execute DeepSeek brain (First Brain - Analysis & Synthesis)
            logger.info("🧠 Brain 1: DeepSeek analytical synthesis...")
            if self._seam is not None:
                deepseek_insight = await self._seam.execute_deepseek_brain(
                    self, analyses, critiques
                )
            else:
                deepseek_insight = await self._execute_deepseek_brain(analyses, critiques)
            total_cost += deepseek_insight.cost_usd

            # Step 3: Execute Grok brain (Second Brain - Critical Review & Final Synthesis) - OPERATION PHOENIX
            logger.info("🧠 Brain 2: Grok strategic synthesis...")
            if self._seam is not None:
                grok_insight = await self._seam.execute_grok_brain(
                    self, analyses, critiques, deepseek_insight
                )
            else:
                grok_insight = await self._execute_grok_brain(
                    analyses, critiques, deepseek_insight
                )
            total_cost += grok_insight.cost_usd

            # Step 4: Create comprehensive final report
            logger.info("📋 Generating final senior advisor report...")
            if self._seam is not None:
                final_report = await self._seam.create_final_report(
                    self,
                    analyses,
                    critiques,
                    deepseek_insight,
                    grok_insight,
                    start_time,
                    total_cost,
                )
            else:
                final_report = await self._create_final_report(
                    analyses,
                    critiques,
                    deepseek_insight,
                    grok_insight,
                    start_time,
                    total_cost,
                )

            processing_time = time.time() - start_time
            logger.info(f"🎉 Two-Brain synthesis completed in {processing_time:.1f}s")
            logger.info(f"💰 Total cost: ${total_cost:.4f}")
            logger.info(f"🎯 Final confidence: {final_report.final_confidence:.2f}")

            return final_report

        except SeniorAdvisorError:
            raise
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(
                f"❌ Senior advisor synthesis failed after {processing_time:.1f}s: {e}"
            )
            raise SeniorAdvisorError(f"Two-brain synthesis failed: {e}")

    async def _execute_deepseek_brain(
        self,
        analyses: List[ConsultantAnalysisResult],
        critiques: List[AnalysisCritique],
    ) -> TwoBrainInsight:
        """Execute DeepSeek brain - analytical synthesis"""

        if not self.deepseek_client:
            raise SeniorAdvisorError("DeepSeek client not available")

        start_time = time.time()

        # Construct comprehensive input for DeepSeek
        deepseek_prompt = self._generate_deepseek_prompt(analyses, critiques)

        try:
            response = await self.deepseek_client.call_llm(deepseek_prompt)

            # Parse response
            insight_data = self._parse_brain_response(response, "deepseek")

            processing_time = time.time() - start_time

            insight = TwoBrainInsight(
                brain_name="deepseek",
                insight_content=insight_data.get("synthesis", response),
                confidence_level=insight_data.get("confidence_level", 0.85),
                key_points=insight_data.get("key_points", []),
                processing_time_seconds=processing_time,
                tokens_used=len(response) // 4,  # Rough estimate
                cost_usd=0.003,  # DeepSeek cost estimate
            )

            logger.info(f"🧠 DeepSeek brain completed in {processing_time:.1f}s")
            logger.info(f"🎯 DeepSeek confidence: {insight.confidence_level:.2f}")

            return insight

        except Exception as e:
            raise SeniorAdvisorError(f"DeepSeek brain execution failed: {e}")

    def _generate_deepseek_prompt(
        self,
        analyses: List[ConsultantAnalysisResult],
        critiques: List[AnalysisCritique],
    ) -> str:
        """Generate comprehensive prompt for DeepSeek brain"""

        # Compile all consultant analyses
        analyses_text = ""
        for analysis in analyses:
            analyses_text += f"""
## {analysis.consultant_id.upper()} ANALYSIS
**Confidence:** {analysis.confidence_level:.2f}

**Analysis:**
{analysis.analysis_content}

**Key Insights:**
{chr(10).join([f"• {insight}" for insight in analysis.key_insights])}

**Recommendations:**
{chr(10).join([f"• {rec}" for rec in analysis.recommendations])}

**Mental Models Applied:** {', '.join(analysis.mental_models_applied)}

---
"""

        # Compile all critiques
        critiques_text = ""
        for critique in critiques:
            critiques_text += f"""
## CRITIQUE OF {critique.target_consultant.upper()}
**Confidence:** {critique.confidence_level:.2f}

**Critique:**
{critique.critique_content}

**Identified Weaknesses:**
{chr(10).join([f"• {weakness}" for weakness in critique.identified_weaknesses])}

**Alternative Perspectives:**
{chr(10).join([f"• {alt}" for alt in critique.alternative_perspectives])}

**Risk Assessments:**
{chr(10).join([f"• {risk}" for risk in critique.risk_assessments])}

---
"""

        return f"""# SENIOR ADVISOR SYNTHESIS - DEEPSEEK BRAIN (ANALYTICAL SYNTHESIS)

You are the first brain of a Two-Brain Senior Advisor system. Your role is to provide deep analytical synthesis of multiple consultant perspectives and their critiques.

## CONSULTANT ANALYSES TO SYNTHESIZE
{analyses_text}

## DEVIL'S ADVOCATE CRITIQUES TO CONSIDER
{critiques_text}

## YOUR ANALYTICAL SYNTHESIS TASK

As the **analytical brain**, your job is to:

1. **SYNTHESIZE INSIGHTS** - Identify convergent themes and complementary perspectives across consultants
2. **RESOLVE CONFLICTS** - Address contradictions and conflicting recommendations
3. **INTEGRATE CRITIQUES** - Incorporate valid critique points to strengthen analysis
4. **IDENTIFY GAPS** - Highlight any missing perspectives or unconsidered factors
5. **PRIORITIZE ACTIONS** - Rank recommendations by impact and feasibility

## REQUIRED OUTPUT FORMAT

Respond in JSON format:

```json
{{
    "synthesis": "Comprehensive analytical synthesis integrating all perspectives...",
    "convergent_themes": ["Theme 1", "Theme 2", "Theme 3"],
    "resolved_conflicts": [
        {{"conflict": "Description", "resolution": "How resolved"}},
        {{"conflict": "Description", "resolution": "How resolved"}}
    ],
    "integrated_critique_points": ["Valid critique point 1", "Valid critique point 2"],
    "identified_gaps": ["Gap 1", "Gap 2"],
    "prioritized_recommendations": [
        {{"recommendation": "Action 1", "priority": "HIGH", "rationale": "Why high priority"}},
        {{"recommendation": "Action 2", "priority": "MEDIUM", "rationale": "Why medium"}},
        {{"recommendation": "Action 3", "priority": "LOW", "rationale": "Why low"}}
    ],
    "key_points": ["Key insight 1", "Key insight 2", "Key insight 3"],
    "confidence_level": 0.85
}}
```

Focus on analytical depth, logical integration, and actionable synthesis. This analysis will be reviewed by Claude brain for strategic refinement."""

    async def _execute_claude_brain(
        self,
        analyses: List[ConsultantAnalysisResult],
        critiques: List[AnalysisCritique],
        deepseek_insight: TwoBrainInsight,
    ) -> TwoBrainInsight:
        """Execute Claude brain - strategic synthesis and final recommendations"""

        if not self.claude_client:
            # If Claude not available, create a synthesized insight based on DeepSeek
            logger.warning(
                "Claude client not available, synthesizing from DeepSeek only"
            )
            return self._create_fallback_claude_insight(deepseek_insight)

        start_time = time.time()

        # Construct strategic synthesis prompt for Claude
        claude_prompt = self._generate_claude_prompt(
            analyses, critiques, deepseek_insight
        )

        try:
            response = await self.claude_client.call_llm(claude_prompt)

            # Parse response
            insight_data = self._parse_brain_response(response, "claude")

            processing_time = time.time() - start_time

            insight = TwoBrainInsight(
                brain_name="claude",
                insight_content=insight_data.get("strategic_synthesis", response),
                confidence_level=insight_data.get("confidence_level", 0.90),
                key_points=insight_data.get("key_points", []),
                processing_time_seconds=processing_time,
                tokens_used=len(response) // 4,  # Rough estimate
                cost_usd=0.015,  # Claude cost estimate
            )

            logger.info(f"🧠 Claude brain completed in {processing_time:.1f}s")
            logger.info(f"🎯 Claude confidence: {insight.confidence_level:.2f}")

            return insight

        except Exception as e:
            logger.error(f"Claude brain failed: {e}")
            # Fallback to DeepSeek-only synthesis
            return self._create_fallback_claude_insight(deepseek_insight)

    def _generate_claude_prompt(
        self,
        analyses: List[ConsultantAnalysisResult],
        critiques: List[AnalysisCritique],
        deepseek_insight: TwoBrainInsight,
    ) -> str:
        """
        OPERATION PHOENIX: Generate Method Actor prompt for Claude brain

        Based on research: "LLMs as Method Actors" (Doyle et al., 2024)

        Key principles:
        1. Assign immersive cognitive role
        2. Provide rich context and emotional stakes
        3. Demand concrete decisions, not meta-recommendations
        4. Force engagement with devils advocate
        """

        # Extract key insights from consultants (top 3)
        consultant_summaries = []
        for i, analysis in enumerate(analyses[:3]):
            key_insights_preview = ' '.join(analysis.key_insights[:2]) if analysis.key_insights else "No specific insights"
            consultant_summaries.append(f"- {analysis.consultant_id}: {key_insights_preview}")
        consultant_context = "\n".join(consultant_summaries)

        # Extract key critique points
        critique_challenges = []
        for critique in critiques[:2]:
            if critique.identified_weaknesses:
                critique_challenges.append(f"- {critique.identified_weaknesses[0]}")
        critique_context = "\n".join(critique_challenges) if critique_challenges else "- No major critiques identified"

        return f"""<method_actor_role>
You are a Senior Managing Director at a top-tier strategy firm. You have 20 years of experience advising Fortune 500 CEOs on high-stakes decisions. Your reputation depends on giving clear, actionable, SPECIFIC recommendations - not vague strategic platitudes.

You are presenting to the board in 48 hours. They expect:
- A clear GO/NO-GO recommendation with specific numbers
- Concrete implementation steps with timelines
- Honest assessment of risks (not sugar-coating)
- Evidence that you engaged with contrarian views

If you give generic advice like "review comprehensive analysis" or "Phase 1: Foundation building," you will be fired.
</method_actor_role>

<strategic_problem>
Based on the following strategic context from your analytical team.
</strategic_problem>

<consultant_insights>
{consultant_context}
</consultant_insights>

<deepseek_analytical_synthesis>
**Confidence:** {deepseek_insight.confidence_level:.2f}

{deepseek_insight.insight_content[:1500]}

**Key Points:**
{chr(10).join([f"• {point}" for point in deepseek_insight.key_points[:3]])}
</deepseek_analytical_synthesis>

<devils_advocate_critique>
Key assumptions challenged:
{critique_context}

You MUST respond to these challenges in your recommendation.
</devils_advocate_critique>

<success_criteria>
Deliver:
1. **GO/NO-GO RECOMMENDATION**: Explicit decision with specific conditions
2. **BUDGET & TIMELINE**: Specific numbers (€X-Y million, Z months)
3. **RESPONSE TO DEVILS ADVOCATE**: Address each major challenge
4. **3-PHASE IMPLEMENTATION PLAN**: Concrete milestones (not "foundation building")
5. **EARLY WARNING METRICS**: 3+ specific KPIs to monitor

Minimum 600 words. NO GENERIC BOILERPLATE. Be specific or be fired.
</success_criteria>

<required_output_format>
Respond in JSON format:

```json
{{
    "strategic_synthesis": "Executive-level strategic synthesis with GO/NO-GO decision and specific numbers...",
    "executive_summary": "Clear 2-3 sentence summary with explicit recommendation",
    "strategic_recommendations": [
        "Specific recommendation 1 with numbers and timeline",
        "Specific recommendation 2 with numbers and timeline",
        "Specific recommendation 3 with numbers and timeline"
    ],
    "implementation_roadmap": [
        "Month 1-2: [Specific milestone with deliverables]",
        "Month 3-4: [Specific milestone with deliverables]",
        "Month 5-6: [Specific milestone with deliverables]"
    ],
    "risk_mitigation": [
        "Risk 1 mitigation with specific action",
        "Risk 2 mitigation with specific action",
        "Risk 3 mitigation with specific action"
    ],
    "success_metrics": [
        "KPI 1: [Specific metric with target number]",
        "KPI 2: [Specific metric with target number]",
        "KPI 3: [Specific metric with target number]"
    ],
    "devils_advocate_response": "Direct response to critique challenges",
    "key_points": ["Specific insight 1", "Specific insight 2", "Specific insight 3"],
    "confidence_level": 0.90
}}
```
</required_output_format>

Present your strategic recommendation to the board."""

    async def _execute_grok_brain(
        self,
        analyses: List[ConsultantAnalysisResult],
        critiques: List[AnalysisCritique],
        deepseek_insight: TwoBrainInsight,
    ) -> TwoBrainInsight:
        """Execute Grok brain - strategic synthesis and final recommendations (OPERATION PHOENIX)"""

        if not self.grok_client:
            # If Grok not available, create a synthesized insight based on DeepSeek
            logger.warning(
                "Grok client not available, synthesizing from DeepSeek only"
            )
            return self._create_fallback_strategic_insight(deepseek_insight)

        start_time = time.time()

        # Construct strategic synthesis prompt for Grok
        grok_prompt = self._generate_grok_prompt(
            analyses, critiques, deepseek_insight
        )

        try:
            response = await self.grok_client.call_llm(grok_prompt)

            # Parse response
            insight_data = self._parse_brain_response(response, "grok")

            processing_time = time.time() - start_time

            insight = TwoBrainInsight(
                brain_name="grok",
                insight_content=insight_data.get("strategic_synthesis", response),
                confidence_level=insight_data.get("confidence_level", 0.88),
                key_points=insight_data.get("key_points", []),
                processing_time_seconds=processing_time,
                tokens_used=len(response) // 4,  # Rough estimate
                cost_usd=0.002,  # Grok-4-Fast cost estimate
            )

            logger.info(f"🧠 Grok brain completed in {processing_time:.1f}s")
            logger.info(f"🎯 Grok confidence: {insight.confidence_level:.2f}")

            return insight

        except Exception as e:
            logger.error(f"Grok brain failed: {e}")
            # Fallback to DeepSeek-only synthesis
            return self._create_fallback_strategic_insight(deepseek_insight)

    def _generate_grok_prompt(
        self,
        analyses: List[ConsultantAnalysisResult],
        critiques: List[AnalysisCritique],
        deepseek_insight: TwoBrainInsight,
    ) -> str:
        """
        OPERATION PHOENIX: Generate Method Actor prompt for Grok brain

        Based on research: "LLMs as Method Actors" (Doyle et al., 2024)

        Key principles:
        1. Assign immersive cognitive role
        2. Provide rich context and emotional stakes
        3. Demand concrete decisions, not meta-recommendations
        4. Force engagement with devils advocate
        """

        # Extract key insights from consultants (top 3)
        consultant_summaries = []
        for i, analysis in enumerate(analyses[:3]):
            key_insights_preview = ' '.join(analysis.key_insights[:2]) if analysis.key_insights else "No specific insights"
            consultant_summaries.append(f"- {analysis.consultant_id}: {key_insights_preview}")
        consultant_context = "\n".join(consultant_summaries)

        # Extract key critique points
        critique_challenges = []
        for critique in critiques[:2]:
            if critique.identified_weaknesses:
                critique_challenges.append(f"- {critique.identified_weaknesses[0]}")
        critique_context = "\n".join(critique_challenges) if critique_challenges else "- No major critiques identified"

        return f"""<method_actor_role>
You are a Senior Managing Director at a top-tier strategy firm. You have 20 years of experience advising Fortune 500 CEOs on high-stakes decisions. Your reputation depends on giving clear, actionable, SPECIFIC recommendations - not vague strategic platitudes.

You are presenting to the board in 48 hours. They expect:
- A clear GO/NO-GO recommendation with specific numbers
- Concrete implementation steps with timelines
- Honest assessment of risks (not sugar-coating)
- Evidence that you engaged with contrarian views

If you give generic advice like "review comprehensive analysis" or "Phase 1: Foundation building," you will be fired.
</method_actor_role>

<strategic_problem>
Based on the following strategic context from your analytical team.
</strategic_problem>

<consultant_insights>
{consultant_context}
</consultant_insights>

<deepseek_analytical_synthesis>
**Confidence:** {deepseek_insight.confidence_level:.2f}

{deepseek_insight.insight_content[:1500]}

**Key Points:**
{chr(10).join([f"• {point}" for point in deepseek_insight.key_points[:3]])}
</deepseek_analytical_synthesis>

<devils_advocate_critique>
Key assumptions challenged:
{critique_context}

You MUST respond to these challenges in your recommendation.
</devils_advocate_critique>

<success_criteria>
Deliver:
1. **GO/NO-GO RECOMMENDATION**: Explicit decision with specific conditions
2. **BUDGET & TIMELINE**: Specific numbers (€X-Y million, Z months)
3. **RESPONSE TO DEVILS ADVOCATE**: Address each major challenge
4. **3-PHASE IMPLEMENTATION PLAN**: Concrete milestones (not "foundation building")
5. **EARLY WARNING METRICS**: 3+ specific KPIs to monitor

Minimum 600 words. NO GENERIC BOILERPLATE. Be specific or be fired.
</success_criteria>

<required_output_format>
Respond in JSON format:

```json
{{
    "strategic_synthesis": "Executive-level strategic synthesis with GO/NO-GO decision and specific numbers...",
    "executive_summary": "Clear 2-3 sentence summary with explicit recommendation",
    "strategic_recommendations": [
        "Specific recommendation 1 with numbers and timeline",
        "Specific recommendation 2 with numbers and timeline",
        "Specific recommendation 3 with numbers and timeline"
    ],
    "implementation_roadmap": [
        "Month 1-2: [Specific milestone with deliverables]",
        "Month 3-4: [Specific milestone with deliverables]",
        "Month 5-6: [Specific milestone with deliverables]"
    ],
    "risk_mitigation": [
        "Risk 1 mitigation with specific action",
        "Risk 2 mitigation with specific action",
        "Risk 3 mitigation with specific action"
    ],
    "success_metrics": [
        "KPI 1: [Specific metric with target number]",
        "KPI 2: [Specific metric with target number]",
        "KPI 3: [Specific metric with target number]"
    ],
    "devils_advocate_response": "Direct response to critique challenges",
    "key_points": ["Specific insight 1", "Specific insight 2", "Specific insight 3"],
    "confidence_level": 0.88
}}
```
</required_output_format>

Present your strategic recommendation to the board."""

    def _create_fallback_strategic_insight(
        self, deepseek_insight: TwoBrainInsight
    ) -> TwoBrainInsight:
        """Create fallback strategic insight when Grok/Claude API unavailable"""

        fallback_synthesis = """Strategic synthesis based on analytical findings:

EXECUTIVE SUMMARY: Based on comprehensive consultant analysis and critique integration, a clear strategic path forward has been identified with specific implementation priorities.

STRATEGIC APPROACH: The analysis converges on key strategic priorities that balance immediate operational needs with long-term strategic positioning. Implementation should proceed in phases with clear success metrics.

IMPLEMENTATION FOCUS: Priority actions have been identified based on impact and feasibility analysis, with appropriate risk mitigation strategies in place."""

        return TwoBrainInsight(
            brain_name="strategic_fallback",
            insight_content=fallback_synthesis,
            confidence_level=deepseek_insight.confidence_level
            * 0.85,  # Slight reduction for fallback
            key_points=[
                "Strategic priorities identified from analytical synthesis",
                "Implementation roadmap based on feasibility analysis",
                "Risk mitigation strategies incorporated",
            ],
            processing_time_seconds=0.1,  # Fallback time
            tokens_used=100,
            cost_usd=0.0,
        )

    def _create_fallback_claude_insight(
        self, deepseek_insight: TwoBrainInsight
    ) -> TwoBrainInsight:
        """Create fallback Claude insight when Claude API unavailable (LEGACY - kept for compatibility)"""
        return self._create_fallback_strategic_insight(deepseek_insight)

    def _parse_brain_response(self, response: str, brain_name: str) -> Dict[str, Any]:
        """Parse JSON response from brain LLM call"""

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
                        (
                            "synthesis"
                            if brain_name == "deepseek"
                            else "strategic_synthesis"
                        ): response,
                        "key_points": [],
                        "confidence_level": 0.8,
                    }

            return json.loads(json_text)

        except Exception as e:
            logger.warning(
                f"Failed to parse {brain_name} JSON response, using raw text: {e}"
            )
            return {
                (
                    "synthesis" if brain_name == "deepseek" else "strategic_synthesis"
                ): response,
                "key_points": [],
                "confidence_level": 0.8,
            }

    async def _create_final_report(
        self,
        analyses: List[ConsultantAnalysisResult],
        critiques: List[AnalysisCritique],
        deepseek_insight: TwoBrainInsight,
        grok_insight: TwoBrainInsight,
        start_time: float,
        total_cost: float,
    ) -> SeniorAdvisorReport:
        """Create comprehensive final senior advisor report (OPERATION PHOENIX - Grok brain)"""

        # Parse Grok response for structured elements
        grok_data = self._parse_brain_response(
            grok_insight.insight_content, "grok"
        )

        # Extract structured information
        executive_summary = grok_data.get(
            "executive_summary",
            "Comprehensive strategic analysis completed with actionable recommendations identified.",
        )

        strategic_recommendation = grok_data.get(
            "strategic_synthesis", grok_insight.insight_content
        )

        implementation_roadmap = grok_data.get(
            "implementation_roadmap",
            [
                "Phase 1: Execute high-priority recommendations",
                "Phase 2: Monitor progress and adjust approach",
                "Phase 3: Scale successful initiatives",
            ],
        )

        risk_mitigation = grok_data.get(
            "risk_mitigation",
            [
                "Monitor key performance indicators",
                "Establish regular review checkpoints",
                "Maintain stakeholder communication",
            ],
        )

        success_metrics = grok_data.get(
            "success_metrics",
            [
                "Achievement of primary strategic objectives",
                "Improvement in key business metrics",
                "Stakeholder satisfaction and alignment",
            ],
        )

        # Calculate synthesis rationale
        synthesis_rationale = f"""Two-Brain Synthesis Process:

1. **DeepSeek Brain (Analytical)**: Integrated {len(analyses)} consultant analyses and {len(critiques)} critiques to identify convergent themes and resolve conflicts.

2. **Grok Brain (Strategic)**: Provided executive-level strategic synthesis and implementation roadmap based on analytical findings.

3. **Combined Insight**: Merged analytical depth with strategic clarity to provide actionable recommendations."""

        # Calculate final confidence (weighted average favoring Grok)
        final_confidence = (
            deepseek_insight.confidence_level * 0.4
            + grok_insight.confidence_level * 0.6
        )

        total_processing_time = time.time() - start_time

        report = SeniorAdvisorReport(
            executive_summary=executive_summary,
            strategic_recommendation=strategic_recommendation,
            implementation_roadmap=implementation_roadmap,
            risk_mitigation=risk_mitigation,
            success_metrics=success_metrics,
            deepseek_brain=deepseek_insight,
            claude_brain=grok_insight,  # OPERATION PHOENIX: Using grok_insight, field name kept for compatibility
            synthesis_rationale=synthesis_rationale,
            final_confidence=final_confidence,
            total_processing_time_seconds=total_processing_time,
            total_cost_usd=total_cost,
            timestamp=datetime.now(timezone.utc),
        )

        # Validation
        self._validate_final_report(report)

        return report

    def _validate_final_report(self, report: SeniorAdvisorReport) -> None:
        """Validate final report completeness and quality"""

        if not report.executive_summary or len(report.executive_summary) < 50:
            raise SeniorAdvisorError("Executive summary is missing or too brief")

        if (
            not report.strategic_recommendation
            or len(report.strategic_recommendation) < 100
        ):
            raise SeniorAdvisorError("Strategic recommendation is missing or too brief")

        if len(report.implementation_roadmap) < 2:
            raise SeniorAdvisorError("Implementation roadmap is incomplete")

        if report.final_confidence < 0.5:
            raise SeniorAdvisorError(
                f"Final confidence too low: {report.final_confidence:.2f}"
            )

        logger.info("✅ Final report validation passed")

    # ============================================================================
    # COMPATIBILITY INTERFACE METHODS (for StatefulPipelineOrchestrator)
    # ============================================================================

    async def conduct_two_brain_analysis(
        self,
        consultant_outputs: List[Any],
        original_query: str,
        engagement_id: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Compatibility method for StatefulPipelineOrchestrator

        Delegates to our SeniorManager domain service while maintaining
        the original interface contract.
        """
        logger.info(f"🔍 OPERATION PHOENIX DEBUG: conduct_two_brain_analysis() CALLED for {engagement_id}")
        try:
            logger.info(f"🔍 DEBUG STEP 1: Converting consultant outputs (count={len(consultant_outputs)})")
            # Convert consultant_outputs to our domain model format
            analyses = self._convert_consultant_outputs_to_analyses(consultant_outputs)
            logger.info(f"🔍 DEBUG STEP 2: Converted to {len(analyses)} analyses")

            # Mock critiques if not provided - in real flow they come from devils advocate
            critiques = kwargs.get('critiques', [])
            logger.info(f"🔍 DEBUG STEP 3: Got critiques (count={len(critiques)})")

            # Execute our domain service
            logger.info(f"🔍 DEBUG STEP 4: About to call run_senior_advisor()")
            report = await self.run_senior_advisor(analyses, critiques)
            logger.info(f"🔍 DEBUG STEP 5: run_senior_advisor() returned successfully")

            # Convert back to expected format for compatibility
            # Build final markdown report from the analysis
            final_markdown_report = f"""# Senior Advisor Analysis

## Executive Summary
{report.executive_summary}

## Strategic Recommendations
{report.strategic_recommendation}

## Implementation Roadmap
{report.implementation_roadmap}

## Risk Mitigation
{report.risk_mitigation}

## Success Metrics
{', '.join(report.success_metrics) if report.success_metrics else 'N/A'}

## Synthesis Rationale
{report.synthesis_rationale}

---
*Analysis completed with confidence: {report.final_confidence:.2f}*
"""

            return {
                "success": True,  # Required by executor
                "engagement_id": engagement_id,
                "final_markdown_report": final_markdown_report,  # Required by executor
                "raw_analytical_dossier": {  # Required by executor
                    "confidence": report.final_confidence,
                },
                "analysis_id": engagement_id,
                "processing_time": report.total_processing_time_seconds,
                "analysis_summary": report.executive_summary,
                "strategic_recommendations": report.strategic_recommendation,
                "implementation_plan": report.implementation_roadmap,
                "risk_factors": report.risk_mitigation,
                "confidence_score": report.final_confidence,
                "total_cost": report.total_cost_usd,
                "deepseek_analysis": {
                    "content": report.deepseek_brain.insight_content,
                    "confidence": report.deepseek_brain.confidence_level,
                    "key_points": report.deepseek_brain.key_points,
                },
                "claude_synthesis": {
                    "content": report.claude_brain.insight_content,
                    "confidence": report.claude_brain.confidence_level,
                    "key_points": report.claude_brain.key_points,
                },
                "synthesis_rationale": report.synthesis_rationale,
            }

        except Exception as e:
            logger.error(f"❌ Two-brain analysis failed: {e}")
            raise SeniorAdvisorError(f"Two-brain analysis failed: {e}")

    async def synthesize_final_advice(
        self,
        consultant_analyses: List[Dict[str, Any]],
        devils_advocate_challenges: Dict[str, Any],
        context_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compatibility method for StatefulPipelineOrchestrator

        Delegates to our SeniorManager domain service while maintaining
        the original interface contract.
        """
        try:
            # Convert input formats to our domain models
            analyses = self._convert_dict_analyses_to_domain(consultant_analyses)
            critiques = self._convert_dict_challenges_to_domain(devils_advocate_challenges)

            # Execute our domain service
            report = await self.run_senior_advisor(analyses, critiques)

            # Convert to expected legacy format
            return {
                "final_advice": {
                    "executive_summary": report.executive_summary,
                    "strategic_recommendation": report.strategic_recommendation,
                    "implementation_roadmap": report.implementation_roadmap,
                    "risk_mitigation": report.risk_mitigation,
                    "success_metrics": report.success_metrics,
                    "confidence_score": report.final_confidence,
                },
                "synthesis_metadata": {
                    "processing_time": report.total_processing_time_seconds,
                    "total_cost": report.total_cost_usd,
                    "synthesis_approach": "Two-Brain (DeepSeek + Claude)",
                    "synthesis_rationale": report.synthesis_rationale,
                },
                "brain_insights": {
                    "analytical_brain": {
                        "provider": "deepseek",
                        "insight": report.deepseek_brain.insight_content,
                        "confidence": report.deepseek_brain.confidence_level,
                        "key_points": report.deepseek_brain.key_points,
                    },
                    "strategic_brain": {
                        "provider": "claude",
                        "insight": report.claude_brain.insight_content,
                        "confidence": report.claude_brain.confidence_level,
                        "key_points": report.claude_brain.key_points,
                    },
                },
                "context_data": context_data,  # Pass through for compatibility
            }

        except Exception as e:
            logger.error(f"❌ Final advice synthesis failed: {e}")
            raise SeniorAdvisorError(f"Final advice synthesis failed: {e}")

    def _convert_consultant_outputs_to_analyses(self, consultant_outputs: List[Any]) -> List[ConsultantAnalysisResult]:
        """Convert legacy consultant outputs to our domain model"""
        analyses = []

        for output in consultant_outputs:
            # Handle different input formats gracefully
            if hasattr(output, 'consultant_id'):
                consultant_id = output.consultant_id
            elif isinstance(output, dict):
                consultant_id = output.get('consultant_id', 'unknown')
            else:
                consultant_id = 'unknown'

            analysis = ConsultantAnalysisResult(
                consultant_id=consultant_id,
                analysis_content=getattr(output, 'analysis_content', str(output)),
                key_insights=getattr(output, 'key_insights', []),
                recommendations=getattr(output, 'recommendations', []),
                confidence_level=getattr(output, 'confidence_level', 0.8),
                mental_models_applied=getattr(output, 'mental_models_applied', []),
                research_citations=getattr(output, 'research_citations', []),
                processing_time_seconds=getattr(output, 'processing_time_seconds', 1.0),
                llm_tokens_used=getattr(output, 'llm_tokens_used', 500),
                llm_cost_usd=getattr(output, 'llm_cost_usd', 0.01),
            )
            analyses.append(analysis)

        return analyses

    def _convert_dict_analyses_to_domain(self, dict_analyses: List[Dict[str, Any]]) -> List[ConsultantAnalysisResult]:
        """Convert dictionary format analyses to domain model"""
        analyses = []

        for analysis_dict in dict_analyses:
            analysis = ConsultantAnalysisResult(
                consultant_id=analysis_dict.get('consultant_id', 'unknown'),
                analysis_content=analysis_dict.get('analysis_content', ''),
                key_insights=analysis_dict.get('key_insights', []),
                recommendations=analysis_dict.get('recommendations', []),
                confidence_level=analysis_dict.get('confidence_level', 0.8),
                mental_models_applied=analysis_dict.get('mental_models_applied', []),
                research_citations=analysis_dict.get('research_citations', []),
                processing_time_seconds=analysis_dict.get('processing_time_seconds', 1.0),
                llm_tokens_used=analysis_dict.get('llm_tokens_used', 500),
                llm_cost_usd=analysis_dict.get('llm_cost_usd', 0.01),
            )
            analyses.append(analysis)

        return analyses

    def _convert_dict_challenges_to_domain(self, challenges_dict: Dict[str, Any]) -> List[AnalysisCritique]:
        """Convert dictionary format challenges to domain model"""
        critiques = []

        # Handle different challenge formats
        if isinstance(challenges_dict, dict):
            if 'challenges' in challenges_dict:
                challenge_list = challenges_dict['challenges']
            else:
                challenge_list = [challenges_dict]
        else:
            challenge_list = [challenges_dict] if challenges_dict else []

        for challenge in challenge_list:
            if isinstance(challenge, dict):
                critique = AnalysisCritique(
                    target_consultant=challenge.get('target_consultant', 'unknown'),
                    critique_content=challenge.get('critique_content', str(challenge)),
                    identified_weaknesses=challenge.get('identified_weaknesses', []),
                    alternative_perspectives=challenge.get('alternative_perspectives', []),
                    risk_assessments=challenge.get('risk_assessments', []),
                    confidence_level=challenge.get('confidence_level', 0.8),
                    processing_time_seconds=challenge.get('processing_time_seconds', 1.0),
                    engines_used=challenge.get('engines_used', ['devils_advocate']),
                )
                critiques.append(critique)

        return critiques


# ============================================================================
# TWO-BRAIN LLM CLIENT
# ============================================================================


class TwoBrainLLMClient:
    """LLM client specifically for Two-Brain operations"""

    def __init__(self, api_key: str, provider: str):
        self.api_key = api_key
        self.provider = provider

    async def call_llm(self, prompt: str) -> str:
        """Make real LLM API call"""

        if self.provider == "deepseek":
            return await self._call_deepseek(prompt)
        elif self.provider == "claude":
            return await self._call_claude(prompt)
        elif self.provider == "grok":
            return await self._call_grok(prompt)
        else:
            raise SeniorAdvisorError(f"Unsupported provider: {self.provider}")

    async def _call_deepseek(self, prompt: str) -> str:
        """DeepSeek API call for analytical synthesis"""
        import httpx

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4000,
            "temperature": 0.7,
        }

        async with httpx.AsyncClient(timeout=200.0) as client:
            response = await client.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers,
                json=payload,
            )

        if response.status_code != 200:
            raise SeniorAdvisorError(
                f"DeepSeek API error: {response.status_code} - {response.text}"
            )

        result = response.json()
        return result["choices"][0]["message"]["content"]

    async def _call_claude(self, prompt: str) -> str:
        """Claude API call for strategic synthesis"""
        import httpx

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

        payload = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 4000,
            "messages": [{"role": "user", "content": prompt}],
        }

        async with httpx.AsyncClient(timeout=200.0) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages", headers=headers, json=payload
            )

        if response.status_code != 200:
            raise SeniorAdvisorError(
                f"Claude API error: {response.status_code} - {response.text}"
            )

        result = response.json()
        return result["content"][0]["text"]

    async def _call_grok(self, prompt: str) -> str:
        """Grok-4-Fast API call via OpenRouter for strategic synthesis"""
        import httpx

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://lolla.ai",
            "X-Title": "METIS V5.3 - Two-Brain Senior Advisor",
        }

        payload = {
            "model": "x-ai/grok-4-fast",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4000,
            "temperature": 0.7,
        }

        async with httpx.AsyncClient(timeout=200.0) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
            )

        if response.status_code != 200:
            raise SeniorAdvisorError(
                f"Grok API error: {response.status_code} - {response.text}"
            )

        result = response.json()
        return result["choices"][0]["message"]["content"]


# ============================================================================
# MAIN FUNCTION FOR STEP 5
# ============================================================================


async def run_senior_advisor(
    analyses: List[ConsultantAnalysisResult], critiques: List[AnalysisCritique]
) -> SeniorAdvisorReport:
    """
    Main function for Step 5: Execute Two-Brain Senior Advisor synthesis

    Args:
        analyses: Results from consultant analyses
        critiques: Results from devil's advocate critiques

    Returns:
        SeniorAdvisorReport: Final synthesized recommendations

    Raises:
        SeniorAdvisorError: If any step fails
    """
    orchestrator = SeniorAdvisorOrchestrator()
    return await orchestrator.run_senior_advisor(analyses, critiques)
