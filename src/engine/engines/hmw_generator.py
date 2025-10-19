"""
HMW (How Might We) Question Generator
Targeted implementation from Ideaflow methodology for creative problem reframing

Implements the "Six Dials" approach in a focused, enterprise-friendly way:
- Scale (bigger/smaller)
- Quality (better/worse)
- Emotion (feelings/motivations)
- Stakes (what if/consequences)
- Expectations (assumptions/should be)
- Analogy (like other domains)
"""

import asyncio
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

# LLM integration with Afterburner optimization
try:
    from ..core.llm_integration_adapter import get_unified_llm_adapter
    from ..config.afterburner_migration import report_afterburner_result

    AFTERBURNER_AVAILABLE = True
except ImportError:
    AFTERBURNER_AVAILABLE = False
    # Fallback to legacy Claude if Afterburner not available
    try:
        from ..integrations.claude_client import get_claude_client, LLMCallType

        CLAUDE_AVAILABLE = True
    except ImportError:
        CLAUDE_AVAILABLE = False

try:
    from ..config import get_cognitive_settings

    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

from ..models.data_contracts import MetisDataContract

logger = logging.getLogger(__name__)


@dataclass
class HMWQuestion:
    """Individual How Might We question"""

    question: str
    dial_type: str  # scale, quality, emotion, stakes, expectations, analogy
    reasoning: str
    potential_approaches: List[str]
    confidence: float


@dataclass
class HMWPortfolio:
    """Collection of HMW questions for a problem"""

    original_problem: str
    core_hmw_questions: List[HMWQuestion]
    total_questions: int
    generation_time: float
    reframing_quality_score: float


class HMWGenerator:
    """
    Generates How Might We questions using the Six Dials approach.

    Focused implementation that:
    1. Generates 3-5 high-quality HMW questions (not 50-100)
    2. Uses proven dial techniques systematically
    3. Integrates cleanly with existing MECE structure
    4. Provides clear business value without overwhelming users
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Initialize LLM adapter with Afterburner optimization
        if AFTERBURNER_AVAILABLE:
            self.llm_adapter = get_unified_llm_adapter()
            self.use_afterburner = True
            self.logger.info("🚀 HMW Generator using Afterburner optimization")
        else:
            self.llm_adapter = None
            self.use_afterburner = False
            self.claude_client = get_claude_client() if CLAUDE_AVAILABLE else None

        # Load configuration
        if CONFIG_AVAILABLE:
            self.settings = get_cognitive_settings()
            self.max_questions = self.settings.HMW_MAX_QUESTIONS_PER_SESSION
            self.min_quality_threshold = self.settings.HMW_MIN_QUALITY_THRESHOLD
            self.generation_timeout = self.settings.HMW_GENERATION_TIMEOUT_SECONDS
        else:
            # Fallback defaults
            self.max_questions = 5
            self.min_quality_threshold = 0.6
            self.generation_timeout = 30

        # Define the six dials with focused prompts
        self.dials = {
            "scale": "How might we approach this at different scales (bigger/smaller, more/less, faster/slower)?",
            "quality": "How might we improve or deliberately change the quality standards?",
            "emotion": "How might we address the emotional or motivational aspects?",
            "stakes": "How might we explore what happens if we change the consequences?",
            "expectations": "How might we challenge what we assume 'should' happen?",
            "analogy": "How might we solve this like other industries or domains do?",
        }

    async def generate_core_hmw(self, contract: MetisDataContract) -> HMWPortfolio:
        """
        Generate 3-5 core HMW questions using selective dial application.

        Args:
            contract: The problem contract

        Returns:
            HMWPortfolio with focused, high-quality HMW questions
        """
        start_time = asyncio.get_event_loop().time()

        try:
            # Analyze problem to select most relevant dials (max 3)
            relevant_dials = await self._select_relevant_dials(contract)

            # Generate HMW questions for selected dials
            hmw_questions = []
            for dial in relevant_dials:
                question = await self._generate_hmw_for_dial(contract, dial)
                if question:
                    hmw_questions.append(question)

            generation_time = asyncio.get_event_loop().time() - start_time

            portfolio = HMWPortfolio(
                original_problem=contract.problem_statement,
                core_hmw_questions=hmw_questions,
                total_questions=len(hmw_questions),
                generation_time=generation_time,
                reframing_quality_score=await self._calculate_quality_score(
                    hmw_questions
                ),
            )

            self.logger.info(
                f"Generated {len(hmw_questions)} HMW questions in {generation_time:.2f}s"
            )
            return portfolio

        except Exception as e:
            self.logger.error(f"HMW generation failed: {e}")
            return HMWPortfolio(
                original_problem=contract.problem_statement,
                core_hmw_questions=[],
                total_questions=0,
                generation_time=0,
                reframing_quality_score=0.0,
            )

    async def _select_relevant_dials(self, contract: MetisDataContract) -> List[str]:
        """
        Select 2-3 most relevant dials based on problem characteristics.

        This prevents overwhelming users with too many reframings.
        """
        problem_text = contract.problem_statement.lower()
        business_context = getattr(contract, "business_context", {})

        relevance_scores = {}

        # Scale dial - good for growth, resource, or capacity problems
        scale_signals = ["growth", "scale", "capacity", "resource", "size", "volume"]
        relevance_scores["scale"] = sum(
            1 for signal in scale_signals if signal in problem_text
        )

        # Quality dial - good for performance, standards, or improvement problems
        quality_signals = [
            "quality",
            "performance",
            "improve",
            "better",
            "standard",
            "optimize",
        ]
        relevance_scores["quality"] = sum(
            1 for signal in quality_signals if signal in problem_text
        )

        # Emotion dial - good for people, culture, or motivation problems
        emotion_signals = [
            "team",
            "culture",
            "motivation",
            "engagement",
            "satisfaction",
            "morale",
        ]
        relevance_scores["emotion"] = sum(
            1 for signal in emotion_signals if signal in problem_text
        )

        # Stakes dial - good for risk, decision, or consequence problems
        stakes_signals = [
            "risk",
            "decision",
            "consequence",
            "impact",
            "critical",
            "urgent",
        ]
        relevance_scores["stakes"] = sum(
            1 for signal in stakes_signals if signal in problem_text
        )

        # Expectations dial - good for assumption or "should be" problems
        expectation_signals = [
            "should",
            "expected",
            "assumption",
            "traditional",
            "always",
            "never",
        ]
        relevance_scores["expectations"] = sum(
            1 for signal in expectation_signals if signal in problem_text
        )

        # Analogy dial - good for innovation or novel approach problems
        analogy_signals = [
            "innovation",
            "creative",
            "new",
            "different",
            "unique",
            "novel",
        ]
        relevance_scores["analogy"] = sum(
            1 for signal in analogy_signals if signal in problem_text
        )

        # Select top 3 dials (minimum 2)
        sorted_dials = sorted(
            relevance_scores.items(), key=lambda x: x[1], reverse=True
        )
        selected = [dial for dial, score in sorted_dials[:3] if score > 0]

        # Always include at least 2 dials
        if len(selected) < 2:
            selected = ["expectations", "analogy"]  # Default high-value dials

        self.logger.info(
            f"Selected dials: {selected} from relevance scores: {relevance_scores}"
        )
        return selected

    async def _generate_hmw_for_dial(
        self, contract: MetisDataContract, dial: str
    ) -> Optional[HMWQuestion]:
        """Generate a single HMW question for a specific dial."""

        if not self.use_afterburner and not self.claude_client:
            return self._generate_fallback_hmw(contract, dial)

        try:
            prompt = f"""
You are an expert in creative problem reframing using the "How Might We" methodology.

PROBLEM: {contract.problem_statement}

BUSINESS CONTEXT: {getattr(contract, 'business_context', {})}

DIAL FOCUS: {dial}
DIAL GUIDANCE: {self.dials[dial]}

Generate exactly ONE high-quality "How Might We" question that reframes this problem through the {dial} lens.

Requirements:
1. Start with "How might we..."
2. Be specific and actionable
3. Open up new solution possibilities
4. Stay relevant to the business context
5. Provide 2-3 potential approaches the question suggests

Format your response as:
HMW: [your question]
REASONING: [why this reframing is valuable]
APPROACHES: [2-3 potential approaches this question opens up]
CONFIDENCE: [0.0-1.0 based on relevance and quality]
"""

            # Use Afterburner optimization if available
            if self.use_afterburner:
                response = await self.llm_adapter.call_llm_unified(
                    prompt=prompt,
                    task_name="hmw_generation",
                    business_context={
                        "problem": contract.problem_statement,
                        "dial": dial,
                    },
                    engagement_id=getattr(contract, "engagement_id", None),
                    phase="creative_reframing",
                    max_tokens=300,
                    temperature=0.5,  # Slightly higher for creativity
                )

                # Report metrics
                report_afterburner_result(
                    component="hmw_generator",
                    success=True,
                    response_time_ms=100,  # Placeholder
                )
            else:
                # Legacy Claude fallback
                response = await self.claude_client.achat(
                    prompt, call_type=LLMCallType.HMW_GENERATION, max_tokens=300
                )

            return self._parse_hmw_response(response, dial)

        except Exception as e:
            self.logger.warning(f"Claude HMW generation failed for {dial}: {e}")
            return self._generate_fallback_hmw(contract, dial)

    def _parse_hmw_response(self, response: str, dial: str) -> HMWQuestion:
        """Parse Claude's HMW response into structured format."""
        lines = response.strip().split("\n")

        hmw_question = ""
        reasoning = ""
        approaches = []
        confidence = 0.7  # default

        current_section = None

        for line in lines:
            line = line.strip()
            if line.startswith("HMW:"):
                hmw_question = line[4:].strip()
            elif line.startswith("REASONING:"):
                reasoning = line[10:].strip()
            elif line.startswith("APPROACHES:"):
                approaches_text = line[11:].strip()
                approaches = [
                    a.strip() for a in approaches_text.split(",") if a.strip()
                ]
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line[11:].strip())
                except ValueError:
                    confidence = 0.7

        return HMWQuestion(
            question=hmw_question
            or f"How might we approach this {dial} challenge differently?",
            dial_type=dial,
            reasoning=reasoning or f"Reframing through {dial} lens",
            potential_approaches=approaches
            or ["Explore alternatives", "Test assumptions"],
            confidence=confidence,
        )

    def _generate_fallback_hmw(
        self, contract: MetisDataContract, dial: str
    ) -> HMWQuestion:
        """Generate fallback HMW when Claude is unavailable."""

        fallback_questions = {
            "scale": "How might we solve this at a completely different scale?",
            "quality": "How might we change our quality expectations to unlock new solutions?",
            "emotion": "How might we address the human and emotional aspects we're missing?",
            "stakes": "How might we explore what happens if the consequences were different?",
            "expectations": "How might we challenge our assumptions about how this should work?",
            "analogy": "How might we solve this the way other industries would?",
        }

        return HMWQuestion(
            question=fallback_questions[dial],
            dial_type=dial,
            reasoning=f"Fallback {dial} reframing",
            potential_approaches=["Explore alternatives", "Test new approaches"],
            confidence=0.5,
        )

    async def _calculate_quality_score(self, questions: List[HMWQuestion]) -> float:
        """Calculate overall quality score for the HMW portfolio."""
        if not questions:
            return 0.0

        # Simple quality score based on:
        # - Diversity of dials used
        # - Average confidence
        # - Number of questions generated

        dial_diversity = len(set(q.dial_type for q in questions)) / 6.0  # max 6 dials
        avg_confidence = sum(q.confidence for q in questions) / len(questions)
        quantity_factor = min(len(questions) / 3.0, 1.0)  # optimal is 3 questions

        return dial_diversity * 0.4 + avg_confidence * 0.4 + quantity_factor * 0.2

    def integrate_with_mece(
        self, hmw_portfolio: HMWPortfolio, mece_structure: Dict
    ) -> Dict:
        """
        Integrate HMW questions with existing MECE structure.

        This preserves the original structure while adding reframing options.
        """
        enhanced_structure = mece_structure.copy()

        enhanced_structure["hmw_reframing"] = {
            "questions": [
                {
                    "question": q.question,
                    "dial": q.dial_type,
                    "approaches": q.potential_approaches,
                }
                for q in hmw_portfolio.core_hmw_questions
            ],
            "total_questions": hmw_portfolio.total_questions,
            "quality_score": hmw_portfolio.reframing_quality_score,
        }

        return enhanced_structure


def get_hmw_generator() -> HMWGenerator:
    """Get singleton HMW generator instance."""
    if not hasattr(get_hmw_generator, "_instance"):
        get_hmw_generator._instance = HMWGenerator()
    return get_hmw_generator._instance
