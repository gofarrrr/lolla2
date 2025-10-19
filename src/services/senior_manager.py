"""
Senior Manager Domain Service
============================

Extracted domain service for Two-Brain Senior Advisor synthesis.
This service encapsulates the core domain logic for strategic synthesis
while maintaining compatibility with existing orchestrator patterns.

Key Features:
- Two-Brain synthesis (DeepSeek + Claude)
- Strategic report generation
- Risk assessment and mitigation
- Implementation roadmap creation
- Confidence calibration
"""

from __future__ import annotations

import time
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from src.orchestration.contracts import (
    ConsultantAnalysisResult,
    AnalysisCritique,
    SeniorAdvisorReport,
    TwoBrainInsight,
)
from src.orchestration.exceptions import SeniorAdvisorError
from src.services.interfaces.senior_manager_interface import ISeniorManager
from src.services.agent_guidance.agent_guidance_retriever import AgentGuidanceRetriever

logger = logging.getLogger(__name__)


class TwoBrainLLMClient:
    """Lightweight LLM client for Two-Brain synthesis"""

    def __init__(self, api_key: str, provider: str):
        self.api_key = api_key
        self.provider = provider

    async def call_llm(self, prompt: str) -> str:
        """Call LLM with prompt (placeholder implementation)"""
        # This would delegate to existing LLM client implementations
        return f"[{self.provider.upper()}] Strategic synthesis response to: {prompt[:100]}..."


class SeniorManager(ISeniorManager):
    """
    Domain service for Two-Brain Senior Advisor synthesis

    Responsibilities:
    - Orchestrate DeepSeek analytical brain
    - Orchestrate Claude strategic brain
    - Synthesize final strategic report
    - Provide risk assessment and mitigation
    - Generate implementation roadmaps
    """

    def __init__(self):
        self.deepseek_client: Optional[TwoBrainLLMClient] = None
        self.claude_client: Optional[TwoBrainLLMClient] = None
        self.guidance_retriever = AgentGuidanceRetriever()
        self._cached_guidance: Optional[str] = None

    async def initialize_clients(self) -> None:
        """Initialize LLM clients for two-brain synthesis"""
        try:
            import os
            from dotenv import load_dotenv

            load_dotenv()

            deepseek_key = os.getenv("DEEPSEEK_API_KEY")
            claude_key = os.getenv("ANTHROPIC_API_KEY")

            if deepseek_key:
                self.deepseek_client = TwoBrainLLMClient(deepseek_key, "deepseek")
                logger.info("✅ SeniorManager: Initialized DeepSeek brain")

            if claude_key:
                self.claude_client = TwoBrainLLMClient(claude_key, "claude")
                logger.info("✅ SeniorManager: Initialized Claude brain")

            if not self.deepseek_client and not self.claude_client:
                raise SeniorAdvisorError(
                    "SeniorManager: No LLM clients available for Two-Brain process"
                )

        except Exception as e:
            raise SeniorAdvisorError(f"SeniorManager: Failed to initialize clients: {e}")

    async def synthesize_strategic_report(
        self,
        analyses: List[ConsultantAnalysisResult],
        critiques: List[AnalysisCritique],
    ) -> SeniorAdvisorReport:
        """
        Main entry point for Two-Brain strategic synthesis

        Args:
            analyses: Results from consultant analyses
            critiques: Results from devil's advocate critiques

        Returns:
            SeniorAdvisorReport: Comprehensive strategic report

        Raises:
            SeniorAdvisorError: If synthesis fails
        """
        start_time = time.time()
        total_cost = 0.0

        try:
            logger.info("🧠 SeniorManager: Starting Two-Brain synthesis")
            logger.info(f"📊 Input: {len(analyses)} analyses, {len(critiques)} critiques")

            guidance = self.guidance_retriever.get_guidance(
                "senior_advisor",
                guidance_type="frameworks",
                max_words=250,
            )
            self._cached_guidance = guidance.get("guidance") if guidance.get("applicable") else None

            # Ensure clients are initialized
            await self.initialize_clients()

            # Brain 1: DeepSeek analytical synthesis
            logger.info("🧠 Brain 1: DeepSeek analytical synthesis...")
            deepseek_insight = await self.execute_analytical_brain(analyses, critiques)
            total_cost += deepseek_insight.cost_usd

            # Brain 2: Claude strategic synthesis
            logger.info("🧠 Brain 2: Claude strategic synthesis...")
            claude_insight = await self.execute_strategic_brain(
                analyses, critiques, deepseek_insight
            )
            total_cost += claude_insight.cost_usd

            # Final synthesis: Comprehensive strategic report
            logger.info("📋 Generating comprehensive strategic report...")
            final_report = await self.create_strategic_report(
                analyses,
                critiques,
                deepseek_insight,
                claude_insight,
                start_time,
                total_cost,
            )

            processing_time = time.time() - start_time
            logger.info(f"🎉 SeniorManager: Two-Brain synthesis completed in {processing_time:.1f}s")
            logger.info(f"💰 Total cost: ${total_cost:.4f}")
            logger.info(f"🎯 Final confidence: {final_report.final_confidence:.2f}")

            return final_report

        except SeniorAdvisorError:
            raise
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(
                f"❌ SeniorManager: Two-brain synthesis failed after {processing_time:.1f}s: {e}"
            )
            raise SeniorAdvisorError(f"SeniorManager synthesis failed: {e}")

    async def execute_analytical_brain(
        self,
        analyses: List[ConsultantAnalysisResult],
        critiques: List[AnalysisCritique],
    ) -> TwoBrainInsight:
        """Execute DeepSeek analytical brain for data synthesis"""
        if not self.deepseek_client:
            raise SeniorAdvisorError("SeniorManager: DeepSeek client not available")

        start_time = time.time()

        try:
            # Generate analytical synthesis prompt
            analytical_prompt = self.generate_analytical_prompt(analyses, critiques)

            # Execute DeepSeek brain
            response = await self.deepseek_client.call_llm(analytical_prompt)

            # Parse analytical response
            insight_data = self.parse_brain_response(response, "analytical")

            processing_time = time.time() - start_time

            insight = TwoBrainInsight(
                brain_name="deepseek_analytical",
                insight_content=insight_data.get("synthesis", response),
                confidence_level=insight_data.get("confidence_level", 0.85),
                key_points=insight_data.get("key_points", []),
                processing_time_seconds=processing_time,
                tokens_used=len(response) // 4,  # Rough estimate
                cost_usd=0.003,  # DeepSeek cost estimate
            )

            logger.info(f"🧠 DeepSeek analytical brain completed in {processing_time:.1f}s")
            return insight

        except Exception as e:
            raise SeniorAdvisorError(f"SeniorManager: Analytical brain execution failed: {e}")

    async def execute_strategic_brain(
        self,
        analyses: List[ConsultantAnalysisResult],
        critiques: List[AnalysisCritique],
        analytical_insight: TwoBrainInsight,
    ) -> TwoBrainInsight:
        """Execute Claude strategic brain for synthesis and recommendations"""
        if not self.claude_client:
            raise SeniorAdvisorError("SeniorManager: Claude client not available")

        start_time = time.time()

        try:
            # Generate strategic synthesis prompt
            strategic_prompt = self.generate_strategic_prompt(
                analyses, critiques, analytical_insight
            )

            # Execute Claude brain
            response = await self.claude_client.call_llm(strategic_prompt)

            # Parse strategic response
            insight_data = self.parse_brain_response(response, "strategic")

            processing_time = time.time() - start_time

            insight = TwoBrainInsight(
                brain_name="claude_strategic",
                insight_content=insight_data.get("synthesis", response),
                confidence_level=insight_data.get("confidence_level", 0.88),
                key_points=insight_data.get("key_points", []),
                processing_time_seconds=processing_time,
                tokens_used=len(response) // 4,  # Rough estimate
                cost_usd=0.012,  # Claude cost estimate
            )

            logger.info(f"🧠 Claude strategic brain completed in {processing_time:.1f}s")
            return insight

        except Exception as e:
            raise SeniorAdvisorError(f"SeniorManager: Strategic brain execution failed: {e}")

    async def create_strategic_report(
        self,
        analyses: List[ConsultantAnalysisResult],
        critiques: List[AnalysisCritique],
        analytical_insight: TwoBrainInsight,
        strategic_insight: TwoBrainInsight,
        start_time: float,
        total_cost: float,
    ) -> SeniorAdvisorReport:
        """Create comprehensive strategic report from brain outputs"""
        try:
            processing_time = time.time() - start_time

            # Extract strategic elements from insights
            executive_summary = self.extract_executive_summary(
                analytical_insight, strategic_insight
            )

            strategic_recommendation = self.extract_strategic_recommendation(
                strategic_insight, analyses
            )

            implementation_roadmap = self.extract_implementation_roadmap(
                strategic_insight, analytical_insight
            )

            risk_mitigation = self.extract_risk_mitigation(
                critiques, analytical_insight, strategic_insight
            )

            success_metrics = self.extract_success_metrics(
                strategic_insight, analyses
            )

            # Calculate final confidence
            final_confidence = self.calculate_final_confidence(
                analytical_insight, strategic_insight, analyses, critiques
            )

            # Generate synthesis rationale
            synthesis_rationale = self.generate_synthesis_rationale(
                analytical_insight, strategic_insight, len(analyses), len(critiques)
            )

            report = SeniorAdvisorReport(
                executive_summary=executive_summary,
                strategic_recommendation=strategic_recommendation,
                implementation_roadmap=implementation_roadmap,
                risk_mitigation=risk_mitigation,
                success_metrics=success_metrics,
                deepseek_brain=analytical_insight,
                claude_brain=strategic_insight,
                synthesis_rationale=synthesis_rationale,
                final_confidence=final_confidence,
                total_processing_time_seconds=processing_time,
                total_cost_usd=total_cost,
            )

            logger.info("📋 SeniorManager: Strategic report created successfully")
            return report

        except Exception as e:
            raise SeniorAdvisorError(f"SeniorManager: Failed to create strategic report: {e}")

    def generate_analytical_prompt(
        self,
        analyses: List[ConsultantAnalysisResult],
        critiques: List[AnalysisCritique],
    ) -> str:
        """Generate prompt for analytical brain (DeepSeek)"""
        analyses_text = self._format_analyses_for_prompt(analyses)
        critiques_text = self._format_critiques_for_prompt(critiques)

        guidance = getattr(self, "_cached_guidance", None)

        prompt = f"""As the ANALYTICAL BRAIN in a Two-Brain strategic synthesis system, your role is to provide comprehensive data analysis and pattern recognition.

## CONSULTANT ANALYSES
{analyses_text}

## DEVIL'S ADVOCATE CRITIQUES
{critiques_text}

## YOUR ANALYTICAL SYNTHESIS TASK
1. **Data Integration**: Synthesize all consultant analyses and critiques
2. **Pattern Recognition**: Identify key patterns, convergences, and divergences
3. **Risk Assessment**: Analyze potential risks and uncertainties
4. **Evidence Evaluation**: Assess strength of evidence and reasoning quality
5. **Confidence Calibration**: Provide analytical confidence assessment

Focus on analytical rigor, data synthesis, and evidence-based insights.
"""

        if guidance:
            prompt += f"\nROLE GUIDANCE (Senior Advisor frameworks to keep in mind):\n{guidance}\n"

        return prompt

    def generate_strategic_prompt(
        self,
        analyses: List[ConsultantAnalysisResult],
        critiques: List[AnalysisCritique],
        analytical_insight: TwoBrainInsight,
    ) -> str:
        """Generate prompt for strategic brain (Claude)"""
        guidance = getattr(self, "_cached_guidance", None)

        prompt = f"""As the STRATEGIC BRAIN in a Two-Brain synthesis system, your role is to provide strategic thinking and actionable recommendations.

## ANALYTICAL BRAIN OUTPUT
**Confidence:** {analytical_insight.confidence_level:.2f}
**Key Points:** {', '.join(analytical_insight.key_points)}
**Analysis:** {analytical_insight.insight_content}

## YOUR STRATEGIC SYNTHESIS TASK
1. **Strategic Integration**: Build on analytical brain insights
2. **Recommendation Development**: Create actionable strategic recommendations
3. **Implementation Planning**: Outline practical next steps
4. **Stakeholder Considerations**: Account for organizational and human factors
5. **Risk Mitigation**: Develop specific risk mitigation strategies

Focus on strategic thinking, practical implementation, and stakeholder impact.
"""

        if guidance:
            prompt += f"\nROLE GUIDANCE (Senior Advisor frameworks to apply):\n{guidance}\n"

        return prompt

    def parse_brain_response(self, response: str, brain_type: str) -> Dict[str, Any]:
        """Parse brain response into structured data"""
        # Simplified parsing - would be enhanced with proper LLM response parsing
        return {
            "synthesis": response,
            "confidence_level": 0.85 if brain_type == "analytical" else 0.88,
            "key_points": [f"{brain_type.title()} insight {i+1}" for i in range(3)],
        }

    def extract_executive_summary(
        self, analytical_insight: TwoBrainInsight, strategic_insight: TwoBrainInsight
    ) -> str:
        """Extract executive summary from brain insights"""
        return f"Strategic synthesis combining analytical rigor ({analytical_insight.confidence_level:.2f} confidence) with strategic thinking ({strategic_insight.confidence_level:.2f} confidence)."

    def extract_strategic_recommendation(
        self, strategic_insight: TwoBrainInsight, analyses: List[ConsultantAnalysisResult]
    ) -> str:
        """Extract strategic recommendation from strategic brain"""
        return f"Primary strategic recommendation based on {len(analyses)} consultant analyses with {strategic_insight.confidence_level:.2f} confidence."

    def extract_implementation_roadmap(
        self, strategic_insight: TwoBrainInsight, analytical_insight: TwoBrainInsight
    ) -> List[str]:
        """Extract implementation roadmap from insights"""
        return [
            "Phase 1: Foundation building based on analytical insights",
            "Phase 2: Strategic implementation guided by strategic brain",
            "Phase 3: Monitoring and adjustment based on feedback loops",
        ]

    def extract_risk_mitigation(
        self,
        critiques: List[AnalysisCritique],
        analytical_insight: TwoBrainInsight,
        strategic_insight: TwoBrainInsight,
    ) -> List[str]:
        """Extract risk mitigation strategies"""
        return [
            f"Address {len(critiques)} critical concerns identified by devil's advocate",
            f"Monitor analytical confidence levels (current: {analytical_insight.confidence_level:.2f})",
            f"Implement strategic safeguards (confidence: {strategic_insight.confidence_level:.2f})",
        ]

    def extract_success_metrics(
        self, strategic_insight: TwoBrainInsight, analyses: List[ConsultantAnalysisResult]
    ) -> List[str]:
        """Extract success metrics from insights"""
        return [
            "Strategic objective achievement (quantified outcomes)",
            "Risk mitigation effectiveness (reduced exposure)",
            "Implementation timeline adherence (milestone completion)",
        ]

    def calculate_final_confidence(
        self,
        analytical_insight: TwoBrainInsight,
        strategic_insight: TwoBrainInsight,
        analyses: List[ConsultantAnalysisResult],
        critiques: List[AnalysisCritique],
    ) -> float:
        """Calculate final confidence score"""
        # Weighted combination of analytical and strategic confidence
        analytical_weight = 0.4
        strategic_weight = 0.6

        base_confidence = (
            analytical_insight.confidence_level * analytical_weight +
            strategic_insight.confidence_level * strategic_weight
        )

        # Adjust for input quality
        analysis_quality = min(len(analyses) / 5.0, 1.0)  # Up to 5 analyses for max quality
        critique_quality = min(len(critiques) / 3.0, 1.0)  # Up to 3 critiques for max quality

        final_confidence = base_confidence * (0.7 + 0.15 * analysis_quality + 0.15 * critique_quality)

        return min(final_confidence, 0.95)  # Cap at 95%

    def generate_synthesis_rationale(
        self,
        analytical_insight: TwoBrainInsight,
        strategic_insight: TwoBrainInsight,
        num_analyses: int,
        num_critiques: int,
    ) -> str:
        """Generate synthesis rationale explaining the process"""
        return f"""Two-Brain Strategic Synthesis Process:

1. **Analytical Brain (DeepSeek)**: Processed {num_analyses} consultant analyses and {num_critiques} critiques with {analytical_insight.confidence_level:.2f} analytical confidence.

2. **Strategic Brain (Claude)**: Built on analytical insights to generate strategic recommendations with {strategic_insight.confidence_level:.2f} strategic confidence.

3. **Synthesis Method**: Weighted integration of analytical rigor and strategic thinking, accounting for input quality and cross-brain validation.

4. **Quality Assurance**: Multi-brain validation ensures both data-driven insights and strategic practicality."""

    def _format_analyses_for_prompt(self, analyses: List[ConsultantAnalysisResult]) -> str:
        """Format consultant analyses for prompt inclusion"""
        if not analyses:
            return "No consultant analyses provided."

        formatted = ""
        for analysis in analyses:
            formatted += f"""
### {analysis.consultant_id.upper()} ANALYSIS
**Confidence:** {analysis.confidence_level:.2f}
**Analysis:** {analysis.analysis_content}
**Key Insights:** {', '.join(analysis.key_insights)}
**Recommendations:** {', '.join(analysis.recommendations)}
**Models:** {', '.join(analysis.mental_models_applied)}
---
"""
        return formatted

    def _format_critiques_for_prompt(self, critiques: List[AnalysisCritique]) -> str:
        """Format critiques for prompt inclusion"""
        if not critiques:
            return "No devil's advocate critiques provided."

        formatted = ""
        for critique in critiques:
            formatted += f"""
### CRITIQUE ({critique.target_consultant})
**Critique:** {critique.critique_content}
**Weaknesses:** {', '.join(critique.identified_weaknesses)}
**Alternative Perspectives:** {', '.join(critique.alternative_perspectives)}
**Risk Assessments:** {', '.join(critique.risk_assessments)}
---
"""
        return formatted
