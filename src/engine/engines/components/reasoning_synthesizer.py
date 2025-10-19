"""
Reasoning Synthesizer Component
Extracted from cognitive_engine.py for better modularity and separation of concerns

Sprint 2.4 Enhancement: Context Intelligence + Pyramid Synthesis Pipeline Integration
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv

    env_path = Path(__file__).parent.parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # dotenv not available

# Claude client integration
try:
    from src.engine.adapters.llm_integration import  # Migrated get_unified_llm_adapter
    from src.config.afterburner_migration import report_afterburner_result

    # Legacy imports for backward compatibility
    try:
        from src.integrations.claude_client import get_claude_client, LLMCallType

        CLAUDE_AVAILABLE = True
    except ImportError:
        CLAUDE_AVAILABLE = False

    CLAUDE_CLIENT_AVAILABLE = True
except ImportError:
    CLAUDE_CLIENT_AVAILABLE = False

from src.engine.models.data_contracts import ReasoningStep
from src.interfaces.reasoning_synthesizer_interface import (
    IReasoningSynthesizer,
    IReasoningValidator,
)
from src.config import CognitiveEngineSettings

# Structured exception hierarchy for fail-fast error handling
from src.engine.adapters.exceptions import  # Migrated (
    LLMProviderError,
    ComponentFailureError,
    AuthenticationError,
    TimeoutError,
)

# Sprint 2.4: Context Intelligence + Pyramid Synthesis Integration
try:
    from src.interfaces.context_intelligence_interface import IContextIntelligence
    from src.engine.engines.pyramid.context_intelligent_engine import (
        ContextIntelligentPyramidEngine,
        create_context_intelligent_pyramid_engine,
    )
    from src.engine.engines.pyramid.enums import DeliverableType
    from src.engine.engines.context_intelligent_challenger import (
        ContextIntelligentChallengeEngine,
        ContextIntelligentChallengeEngineFactory,
        ChallengeRequest,
        ChallengeRigorLevel,
        ChallengeType,
    )

    CONTEXT_PYRAMID_INTEGRATION_AVAILABLE = True
    CONTEXT_CHALLENGER_INTEGRATION_AVAILABLE = True
except ImportError:
    CONTEXT_PYRAMID_INTEGRATION_AVAILABLE = False
    CONTEXT_CHALLENGER_INTEGRATION_AVAILABLE = False
    IContextIntelligence = Any
    ContextIntelligentPyramidEngine = Any
    DeliverableType = Any
    ContextIntelligentChallengeEngine = Any
    ContextIntelligentChallengeEngineFactory = Any
    ChallengeRequest = Any
    ChallengeRigorLevel = Any
    ChallengeType = Any


class ReasoningSynthesizer(IReasoningSynthesizer):
    """
    Handles reasoning synthesis from multiple mental model applications

    Sprint 2.4 Enhancement: Integrated with Context Intelligence and Pyramid Synthesis
    - Uses cognitive exhaust from mental model applications
    - Applies Context-Intelligent Pyramid structuring to reasoning synthesis
    - Generates executive deliverables from synthesized reasoning
    """

    def __init__(
        self,
        settings: CognitiveEngineSettings,
        context_intelligence: Optional[IContextIntelligence] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.settings = settings
        self.context_intelligence = context_intelligence
        self.logger = logger or logging.getLogger(__name__)

        # Sprint 2.4: Initialize Context-Intelligent Pyramid Engine if available
        self.pyramid_engine = None
        self.context_pyramid_enabled = (
            CONTEXT_PYRAMID_INTEGRATION_AVAILABLE and context_intelligence is not None
        )

        if self.context_pyramid_enabled:
            try:
                # Create pyramid engine with simplified interface for ReasoningSynthesizer integration
                self.pyramid_engine = ContextIntelligentPyramidEngine(
                    state_manager=None,  # Optional for this integration
                    event_bus=None,  # Optional for this integration
                    context_intelligence=context_intelligence,
                )
                self.logger.info(
                    "🏗️ Context-Intelligent Pyramid Engine integrated with ReasoningSynthesizer"
                )
            except Exception as e:
                self.logger.warning(
                    f"⚠️ Context-Pyramid integration failed, using traditional synthesis: {e}"
                )
                self.context_pyramid_enabled = False

        # Sprint 3.1: Initialize Context-Intelligent Challenge Engine if available
        self.challenge_engine = None
        self.context_challenger_enabled = (
            CONTEXT_CHALLENGER_INTEGRATION_AVAILABLE
            and context_intelligence is not None
        )

        if self.context_challenger_enabled:
            try:
                # Create challenge engine for internal skepticism and validation
                self.challenge_engine = (
                    ContextIntelligentChallengeEngineFactory.create_challenge_engine(
                        context_intelligence=context_intelligence,
                        settings=settings,
                        logger=self.logger,
                    )
                )
                self.logger.info(
                    "🎯 Context-Intelligent Challenge Engine integrated with ReasoningSynthesizer"
                )
            except Exception as e:
                self.logger.warning(
                    f"⚠️ Context-Challenger integration failed, synthesis without internal challenges: {e}"
                )
                self.context_challenger_enabled = False

    async def synthesize_reasoning(
        self,
        reasoning_results: List[Dict[str, Any]],
        engagement_id: Optional[str] = None,
        target_deliverable_type: Optional[str] = None,
    ) -> List[ReasoningStep]:
        """
        Synthesize reasoning from multiple mental model applications

        Sprint 2.4 Enhancement: Integrates Context-Intelligent Pyramid structuring

        Args:
            reasoning_results: Results from mental model applications
            engagement_id: Optional engagement ID for context intelligence
            target_deliverable_type: Optional deliverable type for pyramid structuring

        Returns:
            List of reasoning steps with optional pyramid-structured deliverable
        """
        reasoning_steps = []

        for result in reasoning_results:
            step = ReasoningStep(
                step=result["step_id"],  # Use step as required field
                description=result[
                    "reasoning_text"
                ],  # Use description as required field
                confidence=result[
                    "confidence_score"
                ],  # Use confidence instead of confidence_score
                step_id=result["step_id"],  # Keep legacy field for compatibility
                mental_model_applied=result["model_applied"],
                reasoning_text=result["reasoning_text"],
                confidence_score=result["confidence_score"],
                evidence_sources=result["evidence_sources"],
                assumptions_made=result["assumptions_made"],
                # Operation Mindforge: Include cognitive exhaust capture
                thinking_process=result.get("thinking_process"),
                cleaned_response=result.get("cleaned_response"),
                timestamp=datetime.utcnow(),
            )
            reasoning_steps.append(step)

        # Add synthesis step combining insights
        if reasoning_steps:
            synthesis_text = await self.create_synthesis_text(reasoning_steps)
            synthesis_step = ReasoningStep(
                step=f"step_{len(reasoning_steps) + 1}",  # Use step as required field
                description=synthesis_text,  # Use description as required field
                confidence=self.calculate_synthesis_confidence(
                    reasoning_steps
                ),  # Use confidence
                step_id=f"step_{len(reasoning_steps) + 1}",  # Keep legacy field for compatibility
                mental_model_applied="synthesis_framework",
                reasoning_text=synthesis_text,
                confidence_score=self.calculate_synthesis_confidence(reasoning_steps),
                evidence_sources=["multi_model_synthesis"],
                assumptions_made=[
                    "Models are complementary",
                    "Synthesis captures key insights",
                ],
                timestamp=datetime.utcnow(),
            )
            reasoning_steps.append(synthesis_step)

            # Sprint 2.4: Context-Intelligent Pyramid Synthesis
            if (
                self.context_pyramid_enabled
                and engagement_id
                and target_deliverable_type
            ):
                await self._apply_context_pyramid_synthesis(
                    reasoning_steps, engagement_id, target_deliverable_type
                )

            # Sprint 3.1: Context-Intelligent Challenge & Self-Doubt Validation
            if self.context_challenger_enabled and engagement_id:
                await self._apply_contextual_challenges(reasoning_steps, engagement_id)

        return reasoning_steps

    async def create_synthesis_text(self, reasoning_steps: List[ReasoningStep]) -> str:
        """Create synthesis text combining insights from multiple models using real LLM generation"""

        # Support both new and legacy field formats
        models_applied = [
            getattr(step, "mental_model_applied", "unknown") for step in reasoning_steps
        ]
        confidences = [
            getattr(step, "confidence", None) or getattr(step, "confidence_score", 0.0)
            for step in reasoning_steps
        ]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # Collect reasoning from all steps for synthesis
        reasoning_texts = []
        key_insights = []

        for step in reasoning_steps:
            model_name = getattr(step, "mental_model_applied", "unknown")
            reasoning_text = getattr(step, "reasoning_text", None) or getattr(
                step, "description", ""
            )
            reasoning_texts.append(f"Model {model_name}: {reasoning_text}")
            if hasattr(step, "key_insights") and step.key_insights:
                key_insights.extend(step.key_insights)

        # Generate real LLM synthesis using DeepSeek as primary
        synthesis_prompt = f"""
<thinking>
Let me synthesize these multiple mental model analyses systematically:

1. What are the key themes and insights that emerge across multiple models?
2. Where do the models agree, and where do they disagree or show tension?
3. Which insights have the highest confidence and strongest evidence?
4. What are the practical implications and actionable recommendations?
5. How should I calibrate confidence given the convergence/divergence across models?
6. What are the key risks or limitations in this synthesis?
</thinking>

Synthesize the following mental model analyses into coherent, actionable insights:

Models Applied: {', '.join(models_applied)}
Average Confidence: {avg_confidence:.2f}

Individual Model Reasoning:
{chr(10).join(reasoning_texts)}

Create a concise synthesis that:
1. Identifies convergent insights across models
2. Highlights any contradictions or tensions
3. Provides clear, actionable recommendations
4. Maintains appropriate confidence calibration

Focus on practical value and avoid generic statements.
        """

        # Try DeepSeek first (PRIMARY), then fallback to Claude with proper error handling
        deepseek_error = None
        claude_error = None

        # Check if DeepSeek is available and use it first
        import os
        import httpx

        deepseek_key = os.getenv("DEEPSEEK_API_KEY")
        if deepseek_key:
            try:
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {deepseek_key}",
                }

                payload = {
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": synthesis_prompt}],
                    "max_tokens": 2500,
                    "temperature": 0.2,
                }

                async with httpx.AsyncClient(timeout=30.0) as client_http:
                    response = await client_http.post(
                        "https://api.deepseek.com/v1/chat/completions",
                        headers=headers,
                        json=payload,
                    )

                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    self.logger.info(
                        "✅ DeepSeek synthesis successful (primary provider)"
                    )
                    return content
                elif response.status_code == 401:
                    raise AuthenticationError(
                        service="deepseek",
                        message="Invalid API key for synthesis",
                        context={"operation": "reasoning_synthesis"},
                    )
                elif response.status_code == 429:
                    raise LLMProviderError(
                        provider="deepseek",
                        message="Rate limit exceeded during synthesis",
                        api_error_code=str(response.status_code),
                        context={"operation": "reasoning_synthesis"},
                    )
                else:
                    raise LLMProviderError(
                        provider="deepseek",
                        message=f"DeepSeek synthesis failed: {response.status_code}",
                        api_error_code=str(response.status_code),
                        context={"response_text": response.text[:500]},
                    )
            except httpx.TimeoutException:
                deepseek_error = TimeoutError(
                    operation="deepseek_synthesis", timeout_seconds=30
                )
            except (AuthenticationError, LLMProviderError, TimeoutError) as e:
                deepseek_error = e
            except Exception as e:
                deepseek_error = LLMProviderError(
                    provider="deepseek",
                    message=f"Unexpected error during synthesis: {str(e)}",
                    context={"error_type": type(e).__name__},
                )
        else:
            deepseek_error = AuthenticationError(
                service="deepseek", message="DEEPSEEK_API_KEY not configured"
            )

        # Fallback to Claude if DeepSeek failed
        if CLAUDE_CLIENT_AVAILABLE:
            try:
                client = await get_claude_client()
                response = await client.call_claude(
                    system_prompt="You are an expert analyst synthesizing insights from multiple mental model applications. Provide clear, specific, actionable synthesis.",
                    prompt=synthesis_prompt,
                    call_type=LLMCallType.MENTAL_MODEL,
                )
                self.logger.info("⚠️ Claude synthesis used as fallback")
                return response.content
            except Exception as e:
                claude_error = LLMProviderError(
                    provider="claude",
                    message=f"Claude synthesis failed: {str(e)}",
                    context={"error_type": type(e).__name__},
                )
        else:
            claude_error = AuthenticationError(
                service="claude", message="ANTHROPIC_API_KEY not configured"
            )

        # If both providers failed, raise comprehensive error
        error_details = []
        if deepseek_error:
            error_details.append(f"DeepSeek: {deepseek_error.message}")
        if claude_error:
            error_details.append(f"Claude: {claude_error.message}")

        raise ComponentFailureError(
            component="reasoning_synthesizer",
            message=f"All LLM providers failed for synthesis - {'; '.join(error_details)}",
            context={
                "deepseek_error": str(deepseek_error) if deepseek_error else None,
                "claude_error": str(claude_error) if claude_error else None,
                "models_applied": models_applied,
                "average_confidence": avg_confidence,
            },
            recovery_suggestions=[
                "Check API key configurations for LLM providers",
                "Verify network connectivity to LLM services",
                "Use fallback structured synthesis if LLMs unavailable",
                "Contact support if errors persist",
            ],
        )

        # Fallback: Generate structured synthesis without hardcoded content
        fallback_synthesis = f"""
        Multi-Model Analysis Synthesis
        
        Applied Models: {', '.join(models_applied)}
        Average Confidence: {avg_confidence:.2f}
        
        Analysis Summary:
        {chr(10).join([f"• {text[:100]}..." for text in reasoning_texts[:3]])}
        
        Synthesis: Models converge on key analytical dimensions with {avg_confidence:.1%} average confidence.
        Further analysis recommended for areas with divergent model outputs.
        """

        return fallback_synthesis.strip()

    async def _apply_context_pyramid_synthesis(
        self,
        reasoning_steps: List[ReasoningStep],
        engagement_id: str,
        target_deliverable_type: str,
    ) -> None:
        """
        Sprint 2.4: Apply Context-Intelligent Pyramid synthesis to reasoning results

        This method transforms synthesized reasoning into structured executive deliverables
        using the Context-Intelligent Pyramid Engine with cognitive exhaust optimization.
        """

        if not self.pyramid_engine:
            self.logger.warning("⚠️ Context-Pyramid engine not available for synthesis")
            return

        try:
            self.logger.info(
                f"🏗️ Applying Context-Intelligent Pyramid synthesis for {target_deliverable_type}"
            )

            # Transform reasoning steps into engagement data format
            engagement_data = self._prepare_engagement_data_from_reasoning(
                reasoning_steps
            )

            # Map deliverable type string to enum
            deliverable_type_mapping = {
                "executive_summary": DeliverableType.EXECUTIVE_SUMMARY,
                "strategy_document": DeliverableType.STRATEGY_DOCUMENT,
                "business_case": DeliverableType.BUSINESS_CASE,
                "recommendation_memo": DeliverableType.RECOMMENDATION_MEMO,
                "implementation_plan": DeliverableType.IMPLEMENTATION_PLAN,
                "final_presentation": DeliverableType.FINAL_PRESENTATION,
            }

            deliverable_type = deliverable_type_mapping.get(
                target_deliverable_type.lower(), DeliverableType.EXECUTIVE_SUMMARY
            )

            # Generate Context-Intelligent Pyramid deliverable
            pyramid_deliverable = (
                await self.pyramid_engine.synthesize_context_aware_deliverable(
                    engagement_data=engagement_data,
                    deliverable_type=deliverable_type,
                    engagement_id=engagement_id,
                    cognitive_coherence_scores=self._extract_coherence_scores(
                        reasoning_steps
                    ),
                )
            )

            # Add pyramid deliverable as a new reasoning step with the information in key_insights
            pyramid_info_step = ReasoningStep(
                step="pyramid_synthesis",
                description=f"Applied Context-Intelligent Pyramid synthesis to generate {target_deliverable_type}",
                confidence=pyramid_deliverable.partner_ready_score,
                thinking_process="<thinking>Synthesized reasoning into executive deliverable using Context Intelligence and Pyramid Engine</thinking>",
                cleaned_response=(
                    pyramid_deliverable.executive_summary[:200] + "..."
                    if len(pyramid_deliverable.executive_summary) > 200
                    else pyramid_deliverable.executive_summary
                ),
                key_insights=[
                    f"Pyramid Deliverable: {pyramid_deliverable.title}",
                    f"Partner Ready Score: {pyramid_deliverable.partner_ready_score:.3f}",
                    f"Structure Quality: {pyramid_deliverable.structure_quality:.3f}",
                    f"Deliverable Type: {deliverable_type.value}",
                ],
                # Legacy compatibility fields
                step_id=f"step_{len(reasoning_steps)+1}",
                mental_model_applied="context_pyramid_synthesis",
                reasoning_text=f"Applied Context-Intelligent Pyramid synthesis to generate {target_deliverable_type}",
                confidence_score=pyramid_deliverable.partner_ready_score,
            )
            reasoning_steps.append(pyramid_info_step)

            self.logger.info(
                f"✅ Context-Pyramid synthesis complete - Partner Ready: {pyramid_deliverable.partner_ready_score:.3f}"
            )

        except Exception as e:
            self.logger.error(f"❌ Context-Pyramid synthesis failed: {e}")
            # Don't raise the exception - graceful degradation to traditional synthesis

    def _prepare_engagement_data_from_reasoning(
        self, reasoning_steps: List[ReasoningStep]
    ) -> Dict[str, Any]:
        """Transform reasoning steps into engagement data format for pyramid synthesis"""

        insights = []
        hypotheses = []
        frameworks_results = []

        for step in reasoning_steps:
            # Extract insights from reasoning text
            reasoning_text = getattr(step, "reasoning_text", None) or getattr(
                step, "description", ""
            )
            if reasoning_text and reasoning_text not in insights:
                insights.append(reasoning_text)

            # Extract hypotheses from assumptions
            assumptions = getattr(step, "assumptions_made", [])
            for assumption in assumptions:
                confidence = getattr(step, "confidence", None) or getattr(
                    step, "confidence_score", 0.0
                )
                hypotheses.append(
                    {"statement": assumption, "confidence_score": confidence}
                )

            # Create framework results from model applications
            model_applied = getattr(step, "mental_model_applied", "unknown")
            if model_applied != "synthesis_framework":  # Skip synthesis step
                frameworks_results.append(
                    {
                        "framework_id": model_applied,
                        "output": {
                            "reasoning": reasoning_text,
                            "confidence": getattr(step, "confidence", None)
                            or getattr(step, "confidence_score", 0.0),
                            "evidence_sources": getattr(step, "evidence_sources", []),
                        },
                    }
                )

        return {
            "insights": insights[:10],  # Top 10 insights
            "hypotheses": hypotheses[:5],  # Top 5 hypotheses
            "frameworks_results": frameworks_results,
            "analysis_findings": {
                "reasoning_depth": len(reasoning_steps),
                "model_diversity": len(
                    set(
                        getattr(step, "mental_model_applied", "unknown")
                        for step in reasoning_steps
                    )
                ),
                "overall_confidence": (
                    sum(
                        getattr(step, "confidence", None)
                        or getattr(step, "confidence_score", 0.0)
                        for step in reasoning_steps
                    )
                    / len(reasoning_steps)
                    if reasoning_steps
                    else 0.0
                ),
            },
        }

    def _extract_coherence_scores(
        self, reasoning_steps: List[ReasoningStep]
    ) -> Dict[str, float]:
        """Extract cognitive coherence scores from reasoning steps"""

        coherence_scores = {}

        # Calculate coherence metrics
        if reasoning_steps:
            confidences = [
                getattr(step, "confidence", None)
                or getattr(step, "confidence_score", 0.0)
                for step in reasoning_steps
            ]
            avg_confidence = sum(confidences) / len(confidences)

            coherence_scores = {
                "reasoning_coherence": avg_confidence,
                "model_consistency": (
                    len(
                        set(
                            getattr(step, "mental_model_applied", "unknown")
                            for step in reasoning_steps
                        )
                    )
                    / len(reasoning_steps)
                    if reasoning_steps
                    else 0.0
                ),
                "evidence_alignment": (
                    min(
                        1.0,
                        sum(
                            len(getattr(step, "evidence_sources", []))
                            for step in reasoning_steps
                        )
                        / len(reasoning_steps),
                    )
                    if reasoning_steps
                    else 0.0
                ),
            }

        return coherence_scores

    async def _apply_contextual_challenges(
        self, reasoning_steps: List[ReasoningStep], engagement_id: str
    ) -> None:
        """
        Sprint 3.1: Apply Context-Intelligent challenges to reasoning results

        This method validates reasoning with internal skepticism using:
        1. Assumption challenging with Context Intelligence
        2. Evidence validation with research-armed challenges
        3. Bias detection using Munger-style systematic doubt
        4. Confidence calibration based on challenge results
        """

        if not self.challenge_engine:
            self.logger.warning(
                "⚠️ Context-Challenge engine not available for validation"
            )
            return

        try:
            self.logger.info(
                f"🎯 Applying contextual challenges for engagement {engagement_id}"
            )

            # Determine challenge rigor based on reasoning complexity
            rigor_level = self._determine_challenge_rigor(reasoning_steps)

            # Create challenge request
            challenge_request = ChallengeRequest(
                reasoning_steps=reasoning_steps,
                engagement_id=engagement_id,
                challenge_types=[
                    ChallengeType.ASSUMPTION_TEST,
                    ChallengeType.EVIDENCE_VALIDATION,
                    ChallengeType.BIAS_AUDIT,
                ],
                rigor_level=rigor_level,
                context_intelligence_enabled=True,
                max_challenges=5,
            )

            # Execute contextual challenges
            challenge_results = (
                await self.challenge_engine.execute_contextual_challenges(
                    challenge_request
                )
            )

            # Add challenge results as new reasoning steps with validation insights
            if challenge_results:
                challenge_summary = self.challenge_engine.get_challenge_summary(
                    challenge_results
                )
                await self._integrate_challenge_results(
                    reasoning_steps, challenge_results, challenge_summary
                )

            self.logger.info(
                f"✅ Contextual challenges complete - {len(challenge_results)} challenges integrated"
            )

        except Exception as e:
            self.logger.error(f"❌ Contextual challenges failed: {e}")
            # Graceful degradation - don't raise exception, synthesis continues without challenges

    def _determine_challenge_rigor(
        self, reasoning_steps: List[ReasoningStep]
    ) -> ChallengeRigorLevel:
        """Determine appropriate challenge rigor level based on reasoning characteristics"""

        if not reasoning_steps:
            return ChallengeRigorLevel.LIGHT

        # Calculate reasoning complexity indicators
        avg_confidence = sum(
            getattr(step, "confidence_score", None) or getattr(step, "confidence", 0.5)
            for step in reasoning_steps
        ) / len(reasoning_steps)
        model_diversity = len(
            set(
                getattr(step, "mental_model_applied", "unknown")
                for step in reasoning_steps
            )
        )
        total_assumptions = sum(
            len(getattr(step, "assumptions_made", [])) for step in reasoning_steps
        )

        # High-confidence + many assumptions = needs intensive challenging
        if avg_confidence > 0.8 and total_assumptions > 5:
            return ChallengeRigorLevel.INTENSIVE

        # Medium confidence + moderate complexity = moderate challenging
        elif avg_confidence > 0.6 and model_diversity > 2:
            return ChallengeRigorLevel.MODERATE

        # Everything else gets light challenging
        else:
            return ChallengeRigorLevel.LIGHT

    async def _integrate_challenge_results(
        self,
        reasoning_steps: List[ReasoningStep],
        challenge_results: List,  # ChallengeResult type from context_intelligent_challenger
        challenge_summary: Dict[str, Any],
    ) -> None:
        """Integrate challenge results into reasoning steps as validation insights"""

        # Create a challenge validation step
        challenge_step = ReasoningStep(
            step=f"step_{len(reasoning_steps) + 1}",
            description=f"Internal challenge validation identified {challenge_summary['total_challenges']} areas for review",
            confidence=max(
                0.1, 1.0 + challenge_summary["confidence_impact"]
            ),  # Adjust confidence based on challenges
            thinking_process="<thinking>Applied systematic internal challenges to validate reasoning quality and identify potential blind spots</thinking>",
            cleaned_response=f"Challenge validation: {', '.join(challenge_summary['key_insights'])}",
            key_insights=[
                f"Total challenges identified: {challenge_summary['total_challenges']}",
                f"Challenge types: {', '.join(challenge_summary.get('challenge_types', []))}",
                f"Net confidence impact: {challenge_summary['confidence_impact']:.3f}",
                f"Research-armed challenges: {challenge_summary.get('research_armed_challenges', 0)}",
                *challenge_summary["key_insights"][:3],  # Top 3 insights
            ],
            # Legacy compatibility fields
            step_id=f"step_{len(reasoning_steps) + 1}",
            mental_model_applied="context_intelligent_challenger",
            reasoning_text=f"Internal validation challenges: {', '.join(challenge_summary['key_insights'][:2])}",
            confidence_score=max(0.1, 1.0 + challenge_summary["confidence_impact"]),
            evidence_sources=["internal_challenger", "context_intelligence"],
            assumptions_made=[
                "Challenge validation improves reasoning quality",
                "Internal skepticism prevents overconfidence",
            ],
        )

        reasoning_steps.append(challenge_step)

        # Apply confidence adjustments to original reasoning steps based on challenges
        if challenge_summary["confidence_impact"] < -0.1:  # Significant negative impact
            for step in reasoning_steps[:-1]:  # All steps except the new challenge step
                current_confidence = getattr(step, "confidence_score", None) or getattr(
                    step, "confidence", 0.5
                )
                adjusted_confidence = max(
                    0.1,
                    current_confidence + (challenge_summary["confidence_impact"] * 0.5),
                )

                # Update both confidence fields for compatibility
                if hasattr(step, "confidence_score"):
                    step.confidence_score = adjusted_confidence
                if hasattr(step, "confidence"):
                    step.confidence = adjusted_confidence

    def calculate_synthesis_confidence(
        self, reasoning_steps: List[ReasoningStep]
    ) -> float:
        """Calculate confidence score for synthesis"""
        if not reasoning_steps:
            return 0.5

        # Weight by model confidence and consistency - support both new and legacy fields
        confidences = [
            getattr(step, "confidence", None) or getattr(step, "confidence_score", 0.0)
            for step in reasoning_steps
        ]
        avg_confidence = sum(confidences) / len(confidences)

        # Boost for multiple models (triangulation)
        triangulation_bonus = min(
            0.1,
            len(reasoning_steps)
            * self.settings.SYNTHESIS_CONFIDENCE_TRIANGULATION_BONUS,
        )

        return min(
            self.settings.MAX_SYNTHESIS_CONFIDENCE, avg_confidence + triangulation_bonus
        )


class ReasoningValidator(IReasoningValidator):
    """
    Handles reasoning quality validation and cognitive load assessment
    """

    def __init__(
        self, settings: CognitiveEngineSettings, logger: Optional[logging.Logger] = None
    ):
        self.settings = settings
        self.logger = logger or logging.getLogger(__name__)

    async def validate_reasoning_quality(
        self, reasoning_steps: List[ReasoningStep]
    ) -> Dict[str, Any]:
        """
        Validate quality of reasoning process and outputs
        """
        validation_results = {
            "overall_confidence": 0.0,
            "confidence_scores": {},
            "quality_metrics": {},
            "validation_flags": [],
        }

        if not reasoning_steps:
            validation_results["validation_flags"].append(
                "No reasoning steps generated"
            )
            return validation_results

        # Calculate confidence scores - support both new and legacy fields
        step_confidences = {}
        for step in reasoning_steps:
            step_id = getattr(step, "step_id", None) or getattr(
                step, "step", f"step_{id(step)}"
            )
            confidence = getattr(step, "confidence", None) or getattr(
                step, "confidence_score", 0.0
            )
            step_confidences[step_id] = confidence

        validation_results["confidence_scores"] = step_confidences
        validation_results["overall_confidence"] = sum(step_confidences.values()) / len(
            step_confidences
        )

        # Quality metrics - support both new and legacy fields
        validation_results["quality_metrics"] = {
            "reasoning_depth": len(reasoning_steps),
            "model_diversity": len(
                set(
                    getattr(step, "mental_model_applied", "unknown")
                    for step in reasoning_steps
                )
            ),
            "evidence_comprehensiveness": sum(
                len(getattr(step, "evidence_sources", [])) for step in reasoning_steps
            ),
            "assumption_transparency": sum(
                len(getattr(step, "assumptions_made", [])) for step in reasoning_steps
            ),
        }

        # Validation flags using configurable thresholds
        if (
            validation_results["overall_confidence"]
            < self.settings.LOW_CONFIDENCE_THRESHOLD
        ):
            validation_results["validation_flags"].append("Low overall confidence")

        if (
            validation_results["quality_metrics"]["model_diversity"]
            < self.settings.MIN_MODEL_DIVERSITY_THRESHOLD
        ):
            validation_results["validation_flags"].append("Limited model diversity")

        if (
            validation_results["quality_metrics"]["evidence_comprehensiveness"]
            < self.settings.MIN_EVIDENCE_COMPREHENSIVENESS
        ):
            validation_results["validation_flags"].append("Limited evidence base")

        return validation_results

    def assess_cognitive_load(self, reasoning_steps: List[ReasoningStep]) -> str:
        """Assess cognitive load for progressive disclosure"""

        # Support both new and legacy fields
        total_content = sum(
            len(
                getattr(step, "reasoning_text", None)
                or getattr(step, "description", "")
            )
            for step in reasoning_steps
        )
        model_count = len(
            set(
                getattr(step, "mental_model_applied", "unknown")
                for step in reasoning_steps
            )
        )

        if (
            total_content > self.settings.COGNITIVE_LOAD_HIGH_THRESHOLD
            or model_count > self.settings.DEFAULT_MODEL_SELECTION_LIMIT
        ):
            return "high"
        elif (
            total_content > self.settings.COGNITIVE_LOAD_MEDIUM_THRESHOLD
            or model_count > (self.settings.DEFAULT_MODEL_SELECTION_LIMIT - 1)
        ):
            return "medium"
        else:
            return "low"


class ReasoningSynthesizerFactory:
    """
    Factory for creating reasoning synthesizer components
    """

    @staticmethod
    def create_synthesizer(
        settings: CognitiveEngineSettings,
        context_intelligence: Optional[IContextIntelligence] = None,
        logger: Optional[logging.Logger] = None,
    ) -> ReasoningSynthesizer:
        """Create a reasoning synthesizer instance with optional Context Intelligence integration"""
        return ReasoningSynthesizer(settings, context_intelligence, logger)

    @staticmethod
    def create_validator(
        settings: CognitiveEngineSettings, logger: Optional[logging.Logger] = None
    ) -> ReasoningValidator:
        """Create a reasoning validator instance"""
        return ReasoningValidator(settings, logger)

    @staticmethod
    def create_combined(
        settings: CognitiveEngineSettings,
        context_intelligence: Optional[IContextIntelligence] = None,
        logger: Optional[logging.Logger] = None,
    ) -> tuple[ReasoningSynthesizer, ReasoningValidator]:
        """Create both synthesizer and validator instances with optional Context Intelligence integration"""
        return (
            ReasoningSynthesizer(settings, context_intelligence, logger),
            ReasoningValidator(settings, logger),
        )
