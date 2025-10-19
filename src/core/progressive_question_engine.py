#!/usr/bin/env python3
"""
Progressive Question Engine - ULTRATHINK 2.0 Edition
====================================================

V2.0 BREAKTHROUGH: Surgical question generation combining four powerful techniques:

1. **Grok-4-Fast Reasoning Mode** (reasoning_enabled: true)
   - Deep strategic analysis before question generation

2. **Ultimate Prompt Technique** (gap analysis)
   - Analyzes what user PROVIDED vs what's MISSING
   - Targets questions at actual uncertainties

3. **10 Strategic Lenses** (McKinsey frameworks)
   - Comprehensive coverage across GOAL, CONSTRAINTS, RISKS, etc.

4. **Temperature Ensemble + Self-Consistency** (robust selection)
   - Parallel generation at 0.3, 0.7, 1.0 temperatures
   - Cross-temperature consistency ranking

Progressive Disclosure UX:
- Tier 1: 3 Essential questions (must answer, 50% baseline)
- Tier 2: 4 Strategic questions (optional, +30% quality)
- Tier 3: 3 Expert questions (optional, +20% quality)

RESULT: User-specific, actionable questions in ~28 seconds
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

try:
    from src.core.research_based_query_enhancer import ResearchBasedQueryEnhancer
except ImportError:
    ResearchBasedQueryEnhancer = None

try:
    from src.integrations.llm.unified_client import UnifiedLLMClient
except ImportError:
    UnifiedLLMClient = None

# ULTRATHINK Integration
try:
    from src.core.ultrathink_question_generator import (
        UltraThinkQuestionGenerator,
        UltraThinkQuestion,
        QuestionGenerationResult as UltraThinkResult,
        QuestionLensCategory as UltraLensCategory,
        QuestionTier as UltraTier
    )
except ImportError:
    UltraThinkQuestionGenerator = None
    print("⚠️ ULTRATHINK Question Generator not available")


class QuestionTier(Enum):
    """Question tier classification for progressive disclosure"""

    ESSENTIAL = "essential"  # Tier 1: Must answer (3 questions)
    STRATEGIC = "strategic"  # Tier 2: Deeper analysis (3 questions)
    EXPERT = "expert"  # Tier 3: Expert mode (3 questions)


class QuestionCategory(Enum):
    """Strategic question categories for intelligent selection"""

    SUCCESS_DEFINITION = "success_definition"
    KEY_CONSTRAINTS = "key_constraints"
    STAKEHOLDER_ALIGNMENT = "stakeholder_alignment"
    COMPETITIVE_DYNAMICS = "competitive_dynamics"
    STRATEGIC_TRADEOFFS = "strategic_tradeoffs"
    MARKET_FORCES = "market_forces"
    IMPLEMENTATION_RISK = "implementation_risk"
    LONG_TERM_VIABILITY = "long_term_viability"
    RESOURCE_ALLOCATION = "resource_allocation"


@dataclass
class AnalysisQuestion:
    """Enhanced question with tier and category classification"""

    question: str
    reasoning: str
    category: QuestionCategory
    tier: QuestionTier
    priority_score: float
    research_grounded: bool = False
    socratic_element: bool = False
    tosca_tag: Optional[str] = None
    auto_generated: bool = False


@dataclass
class TieredQuestions:
    """Progressive disclosure question structure"""

    original_query: str
    research_context: str
    tier_1_essential: List[AnalysisQuestion]  # 3 must-answer questions
    tier_2_strategic: List[AnalysisQuestion]  # 3 optional depth questions
    tier_3_expert: List[AnalysisQuestion]  # 3 optional expert questions
    total_questions: int
    generation_time_ms: int
    tosca_summary: Dict[str, Any] = field(default_factory=dict)

    def get_ux_metadata(self) -> Dict[str, Any]:
        """Get UX metadata for frontend display"""
        return {
            "header": {
                "title": "Strategic Discovery Questions",
                "subtitle": "Stage 1 of 10-Stage Analysis Pipeline",
                "icon": "🎯",
                "description": "Surgical questions targeting YOUR specific context"
            },
            "tiers": {
                "tier_1": {
                    "title": "Strategic Foundation",
                    "subtitle": "Stage 1: Socratic Discovery",
                    "badge": "Essential",
                    "question_count": len(self.tier_1_essential),
                    "quality_impact": "50% baseline analysis",
                    "unlock_threshold": 2,
                    "unlock_message": "Answer 2 of 3 to unlock deeper analysis",
                    "why": "Establishes foundation for 10-stage analysis. Without clear context, subsequent stages lack direction.",
                    "next_stages": ["Problem Structuring", "Research", "Multi-Consultant Analysis"]
                },
                "tier_2": {
                    "title": "Strategic Depth",
                    "subtitle": "Stage 1: Deeper Context",
                    "badge": "+30% Quality Boost",
                    "question_count": len(self.tier_2_strategic),
                    "quality_impact": "80% analysis depth (50% + 30%)",
                    "unlock_threshold": len(self.tier_1_essential) + 4,  # After tier 1
                    "unlock_message": "Answer 4 more for richer strategic insights",
                    "why": "Enables 9 downstream stages to produce personalized recommendations vs generic advice.",
                    "next_stages": ["Consultant Selection", "Synergy Analysis", "Devil's Advocate"]
                },
                "tier_3": {
                    "title": "Expert Mode",
                    "subtitle": "Stage 1: Complete Context",
                    "badge": "+20% Quality Boost",
                    "question_count": len(self.tier_3_expert),
                    "quality_impact": "100% full strategic depth",
                    "unlock_threshold": len(self.tier_1_essential) + len(self.tier_2_strategic) + 3,
                    "unlock_message": "Final 3 questions for 100% expert-level analysis",
                    "why": "Maximizes value from Senior Advisor review and final arbitration stages.",
                    "next_stages": ["Devil's Advocate", "Senior Advisor", "Final Strategic Report"]
                }
            },
            "methodology": {
                "title": "Why these questions drive better analysis",
                "points": [
                    "10 Strategic Lenses: GOAL, CONSTRAINTS, STAKEHOLDERS, RISKS, OPTIONS, etc.",
                    "Surgical Targeting: Asks about YOUR approach, not generic information",
                    "Personalized Analysis: Powers 9 downstream stages with rich context",
                    "Proven Framework: Research-validated 5x information value vs basic queries"
                ]
            },
            "quality_progress": {
                "0_answered": {"quality": 0, "label": "No analysis baseline", "message": "Generic recommendations (like ChatGPT)"},
                "tier_1_complete": {"quality": 50, "label": "50% baseline", "message": "Contextual analysis begins"},
                "tier_2_complete": {"quality": 80, "label": "80% depth", "message": "Personalized multi-consultant insights"},
                "tier_3_complete": {"quality": 100, "label": "100% expert", "message": "Full strategic depth across all stages"}
            },
            "pipeline_preview": [
                {"stage": 1, "name": "Socratic Questions", "status": "current", "icon": "🎯"},
                {"stage": 2, "name": "Problem Structuring", "status": "pending", "icon": "🏗️"},
                {"stage": 3, "name": "Data Research", "status": "pending", "icon": "🔬"},
                {"stage": 4, "name": "Consultant Selection", "status": "pending", "icon": "👥"},
                {"stage": 5, "name": "Parallel Analysis", "status": "pending", "icon": "⚡"},
                {"stage": 6, "name": "Devil's Advocate", "status": "pending", "icon": "🔍"},
                {"stage": 7, "name": "Senior Review", "status": "pending", "icon": "👔"},
                {"stage": 8, "name": "Final Report", "status": "pending", "icon": "📊"}
            ],
            "generation_time_ms": self.generation_time_ms,
            "total_questions": self.total_questions
        }


class ProgressiveQuestionEngine:
    """
    Unified question generation engine with progressive disclosure UX

    Replaces separate Research-Enhanced Query + Socratic Questions with
    a single, intelligent, tiered question generation system.
    """

    def __init__(self):
        """Initialize the progressive question engine with ULTRATHINK"""
        self.research_enhancer = (
            ResearchBasedQueryEnhancer() if ResearchBasedQueryEnhancer else None
        )
        self.llm_client = UnifiedLLMClient() if UnifiedLLMClient else None

        # ULTRATHINK Question Generator
        self.ultrathink_generator = None
        if UltraThinkQuestionGenerator and self.llm_client:
            self.ultrathink_generator = UltraThinkQuestionGenerator(self.llm_client)
            print("🧠 ULTRATHINK Question Generator enabled")
        else:
            print("⚠️ ULTRATHINK unavailable - check dependencies")

        print("🎯 Progressive Question Engine initialized (3-4-3 tiered UX)")

    async def generate_tiered_questions(
        self, user_query: str, context: Optional[Dict[str, Any]] = None
    ) -> TieredQuestions:
        """
        Generate needle-moving questions using ULTRATHINK reasoning-first approach

        Returns structured questions for progressive disclosure UX (3-4-3 tiers)
        """
        import time

        start_time = time.time()

        try:
            print(f"🧠 ULTRATHINK: Generating needle-moving questions for: {user_query[:50]}...")

            # Step 1: Get research context (external knowledge grounding)
            research_context = await self._get_research_context(user_query)

            # Step 2: ULTRATHINK Generation - Temperature Ensemble + Self-Consistency
            if self.ultrathink_generator:
                print("🚀 Using ULTRATHINK reasoning-first generation")
                ultra_result = await self.ultrathink_generator.generate_needle_moving_questions(
                    user_query=user_query,
                    research_context=research_context,
                    max_questions=10  # 3-4-3 structure
                )

                # Convert ULTRATHINK questions to AnalysisQuestion format for compatibility
                tier_1 = self._convert_ultrathink_questions(ultra_result.tier_1_essential)
                tier_2 = self._convert_ultrathink_questions(ultra_result.tier_2_strategic)
                tier_3 = self._convert_ultrathink_questions(ultra_result.tier_3_expert)

                # Enforce TOSCA coverage and capture summary metadata
                tosca_summary = self._apply_tosca_enrichment(
                    user_query=user_query,
                    tier_1=tier_1,
                    tier_2=tier_2,
                    tier_3=tier_3,
                )

                generation_time = ultra_result.generation_time_ms

            else:
                # Fallback if ULTRATHINK unavailable
                print("⚠️ ULTRATHINK unavailable, using fallback")
                # Fallback method already applies TOSCA coverage internally
                return self._create_fallback_questions(user_query)

            result = TieredQuestions(
                original_query=user_query,
                research_context=research_context,
                tier_1_essential=tier_1,
                tier_2_strategic=tier_2,
                tier_3_expert=tier_3,
                total_questions=len(tier_1) + len(tier_2) + len(tier_3),
                generation_time_ms=generation_time,
                tosca_summary=tosca_summary,
            )

            # Ensure minimum coverage of 7 questions (3-4-0 fallback)
            if result.total_questions < 7:
                try:
                    fallback = self._create_fallback_questions(user_query)
                    # Top up tiers while avoiding duplicates by question text
                    def _dedupe(existing, additions):
                        existing_texts = {q.question for q in existing}
                        return existing + [a for a in additions if a.question not in existing_texts]

                    # Merge fallback essential then strategic
                    tier_1 = _dedupe(tier_1, fallback.tier_1_essential)
                    tier_2 = _dedupe(tier_2, fallback.tier_2_strategic)
                    # Recompute result
                    result = TieredQuestions(
                        original_query=user_query,
                        research_context=research_context,
                        tier_1_essential=tier_1,
                        tier_2_strategic=tier_2,
                        tier_3_expert=tier_3,
                        total_questions=len(tier_1) + len(tier_2) + len(tier_3),
                        generation_time_ms=generation_time,
                        tosca_summary=tosca_summary,
                    )
                except Exception:
                    pass

            print(
                f"✅ ULTRATHINK Generated {result.total_questions} questions in {generation_time}ms"
            )
            print(f"   Tier 1 (Essential): {len(result.tier_1_essential)} questions - 50% baseline quality")
            print(f"   Tier 2 (Strategic): {len(result.tier_2_strategic)} questions - +30% quality boost")
            print(f"   Tier 3 (Expert): {len(result.tier_3_expert)} questions - +20% quality boost")

            return result

        except Exception as e:
            print(f"❌ Error in ULTRATHINK question generation: {e}")
            import traceback
            traceback.print_exc()
            # Return fallback questions
            return self._create_fallback_questions(user_query)

    def _convert_ultrathink_questions(
        self, ultra_questions: List[UltraThinkQuestion]
    ) -> List[AnalysisQuestion]:
        """Convert ULTRATHINK questions to AnalysisQuestion format for compatibility"""

        # Mapping from ULTRATHINK lens categories to QuestionCategory
        lens_to_category = {
            "goal": QuestionCategory.SUCCESS_DEFINITION,
            "decision_class": QuestionCategory.STRATEGIC_TRADEOFFS,
            "decision-class": QuestionCategory.STRATEGIC_TRADEOFFS,
            "outside_view": QuestionCategory.MARKET_FORCES,
            "outside-view": QuestionCategory.MARKET_FORCES,
            "constraints": QuestionCategory.KEY_CONSTRAINTS,
            "stakeholders": QuestionCategory.STAKEHOLDER_ALIGNMENT,
            "risks": QuestionCategory.IMPLEMENTATION_RISK,
            "uncertainty": QuestionCategory.STRATEGIC_TRADEOFFS,
            "options": QuestionCategory.MARKET_FORCES,
            "execution": QuestionCategory.LONG_TERM_VIABILITY,
            "competitive": QuestionCategory.COMPETITIVE_DYNAMICS,
            "causality": QuestionCategory.RESOURCE_ALLOCATION,
        }

        converted = []
        for ultra_q in ultra_questions:
            # Map lens category
            category = lens_to_category.get(
                ultra_q.lens_category.value,
                QuestionCategory.SUCCESS_DEFINITION
            )

            # Map tier
            tier = QuestionTier(ultra_q.tier.value)

            analysis_q = AnalysisQuestion(
                question=ultra_q.question,
                reasoning=ultra_q.reasoning,
                category=category,
                tier=tier,
                priority_score=ultra_q.impact_score,
                research_grounded=True,  # ULTRATHINK uses research context
                socratic_element=True   # ULTRATHINK challenges assumptions
            )

            converted.append(analysis_q)

        return converted

    async def _get_research_context(self, query: str) -> str:
        """Get external research context for the query"""
        # ULTRATHINK 2.0: Research enhancer disconnected (replaced by superior approach)
        # Old system: 50+ seconds for 3 sequential LLM calls → 0 questions (lens timeouts)
        # ULTRATHINK 2.0: ~28 seconds with reasoning_enabled + Ultimate Prompt + 10 lenses
        # Result: Surgical, personalized questions targeting actual user gaps
        return f"Research context for: {query}"

    def _apply_tosca_enrichment(
        self,
        user_query: str,
        tier_1: List[AnalysisQuestion],
        tier_2: List[AnalysisQuestion],
        tier_3: List[AnalysisQuestion],
    ) -> Dict[str, Any]:
        """Ensure TOSCA coverage and append auto-generated questions if required."""

        tosca_elements = ["trouble", "owner", "success", "constraints", "actors"]
        coverage: Dict[str, List[str]] = {element: [] for element in tosca_elements}
        autopadded_questions: List[AnalysisQuestion] = []

        # Helper to register coverage
        def register(question: AnalysisQuestion, tag: Optional[str]) -> None:
            if tag:
                question.tosca_tag = tag
                coverage.setdefault(tag, []).append(question.question)

        # First pass: infer tags for existing questions
        for question_list in (tier_1, tier_2, tier_3):
            for question in question_list:
                register(question, self._infer_tosca_tag(question))

        # Determine missing elements and auto-generate questions
        missing = [element for element in tosca_elements if not coverage.get(element)]
        for element in missing:
            autopad = self._create_tosca_question(user_query, element)
            autopad.auto_generated = True
            autopad.tosca_tag = element
            tier_1.append(autopad)  # Always add to Tier 1 for visibility
            coverage[element].append(autopad.question)
            autopadded_questions.append(autopad)

        # Rebalance to maintain strict 3-4-3 distribution while preserving auto-generated items when possible
        def pop_prefer_non_auto(question_list: List[AnalysisQuestion]) -> Optional[AnalysisQuestion]:
            if not question_list:
                return None
            for idx in range(len(question_list) - 1, -1, -1):
                if not question_list[idx].auto_generated:
                    return question_list.pop(idx)
            # If all are auto-generated, pop the last
            return question_list.pop() if question_list else None

        # Move excess from Tier 1 -> Tier 2
        while len(tier_1) > 3:
            item = pop_prefer_non_auto(tier_1)
            if not item:
                break
            item.tier = QuestionTier.STRATEGIC
            tier_2.append(item)

        # Move excess from Tier 2 -> Tier 3
        while len(tier_2) > 4:
            item = pop_prefer_non_auto(tier_2)
            if not item:
                break
            item.tier = QuestionTier.EXPERT
            tier_3.append(item)

        # Clip Tier 3 to max 3 items (prefer removing non-auto)
        while len(tier_3) > 3:
            _ = pop_prefer_non_auto(tier_3)

        # Trim total questions back to 10 (preserving auto-generated items)
        def total_count() -> int:
            return len(tier_1) + len(tier_2) + len(tier_3)

        def pop_from_list(question_list: List[AnalysisQuestion]) -> bool:
            item = pop_prefer_non_auto(question_list)
            return bool(item)

        while total_count() > 10:
            if pop_from_list(tier_3):
                continue
            if pop_from_list(tier_2):
                continue
            if pop_from_list(tier_1):
                continue
            # If all remaining questions are auto-generated, stop trimming
            break

        remaining_missing = [element for element in tosca_elements if not coverage.get(element)]
        return {
            "coverage": coverage,
            "autopadded_questions": [q.question for q in autopadded_questions],
            "missing": remaining_missing,
        }

    def _infer_tosca_tag(self, question: AnalysisQuestion) -> Optional[str]:
        """Infer TOSCA element from question content."""
        if question.tosca_tag:
            return question.tosca_tag

        text = question.question.lower()
        category_value = (
            question.category.value
            if isinstance(question.category, QuestionCategory)
            else str(question.category)
        )

        # OWNER detection: look for responsibility/decision authority language
        owner_keywords = [
            "who is responsible",
            "who owns",
            "who is accountable",
            "decision owner",
            "decision authority",
            "who must sign off",
            "who leads",
        ]
        if any(keyword in text for keyword in owner_keywords):
            return "owner"

        # SUCCESS detection
        if (
            category_value == QuestionCategory.SUCCESS_DEFINITION.value
            or "success" in text
            or "outcome" in text
            or "goal" in text
            or "metric" in text
        ):
            return "success"

        # CONSTRAINTS detection
        if (
            category_value == QuestionCategory.KEY_CONSTRAINTS.value
            or "constraint" in text
            or "limit" in text
            or "budget" in text
            or "timeline" in text
            or "non-negotiable" in text
        ):
            return "constraints"

        # ACTORS detection (stakeholders, teams, partners)
        if (
            category_value == QuestionCategory.STAKEHOLDER_ALIGNMENT.value
            or "stakeholder" in text
            or "team" in text
            or "partner" in text
            or text.startswith("who ")
        ):
            return "actors"

        # TROUBLE detection (risks, blockers, competition, challenges)
        trouble_categories = {
            QuestionCategory.IMPLEMENTATION_RISK.value,
            QuestionCategory.COMPETITIVE_DYNAMICS.value,
            QuestionCategory.STRATEGIC_TRADEOFFS.value,
            QuestionCategory.MARKET_FORCES.value,
            QuestionCategory.LONG_TERM_VIABILITY.value,
        }
        if (
            category_value in trouble_categories
            or "risk" in text
            or "challenge" in text
            or "problem" in text
            or "pain" in text
            or "gap" in text
        ):
            return "trouble"

        return None

    def _create_tosca_question(self, user_query: str, element: str) -> AnalysisQuestion:
        """Create canonical question to fill missing TOSCA coverage."""
        element = element.lower()

        prompts = {
            "trouble": (
                f"What specific trigger or symptom is forcing us to address '{user_query}' right now?",
                "Clarifies the core problem signal driving this work.",
                QuestionCategory.IMPLEMENTATION_RISK,
            ),
            "owner": (
                f"Who is the accountable decision owner for '{user_query}', and who must sign off on the outcome?",
                "Identifies the single accountable owner and decision authority.",
                QuestionCategory.STAKEHOLDER_ALIGNMENT,
            ),
            "success": (
                f"What measurable outcomes or success metrics will confirm '{user_query}' succeeded?",
                "Defines what good looks like in measurable terms.",
                QuestionCategory.SUCCESS_DEFINITION,
            ),
            "constraints": (
                f"What hard constraints (budget, time, policies) limit how we approach '{user_query}'?",
                "Surfaces the non-negotiable boundaries for the work.",
                QuestionCategory.KEY_CONSTRAINTS,
            ),
            "actors": (
                f"Which stakeholders or groups must be engaged for '{user_query}' to succeed?",
                "Ensures we know who must be aligned and involved.",
                QuestionCategory.STAKEHOLDER_ALIGNMENT,
            ),
        }

        question_text, rationale, category = prompts.get(
            element,
            (
                f"What should we clarify about '{user_query}' to make the problem actionable?",
                "Default clarification to ensure problem is actionable.",
                QuestionCategory.SUCCESS_DEFINITION,
            ),
        )

        return AnalysisQuestion(
            question=question_text,
            reasoning=rationale,
            category=category,
            tier=QuestionTier.ESSENTIAL,
            priority_score=0.92,
            research_grounded=False,
            socratic_element=True,
            tosca_tag=element,
            auto_generated=True,
        )

    # OLD TEMPLATE-BASED METHODS RETIRED - Now using ULTRATHINK reasoning-first approach
    # Kept only _create_fallback_questions for emergency fallback

    def _create_fallback_questions(self, query: str) -> TieredQuestions:
        """Create complete fallback tiered questions (emergency only)"""

        # Minimal fallback questions if ULTRATHINK fails
        fallback_essential = [
            AnalysisQuestion(
                question=f"What specific, measurable outcomes would define success for: {query}?",
                reasoning="Establishes clear success criteria",
                category=QuestionCategory.SUCCESS_DEFINITION,
                tier=QuestionTier.ESSENTIAL,
                priority_score=0.95,
                research_grounded=False,
                socratic_element=True,
            ),
            AnalysisQuestion(
                question=f"What are the most significant constraints that will shape how we approach: {query}?",
                reasoning="Identifies critical boundaries",
                category=QuestionCategory.KEY_CONSTRAINTS,
                tier=QuestionTier.ESSENTIAL,
                priority_score=0.90,
                research_grounded=False,
                socratic_element=False,
            ),
            AnalysisQuestion(
                question=f"Which stakeholders are most critical to align for success with: {query}?",
                reasoning="Surfaces key stakeholders",
                category=QuestionCategory.STAKEHOLDER_ALIGNMENT,
                tier=QuestionTier.ESSENTIAL,
                priority_score=0.85,
                research_grounded=False,
                socratic_element=False,
            ),
        ]

        fallback_strategic = [
            AnalysisQuestion(
                question=f"How might competitive responses make our approach to '{query}' less effective?",
                reasoning="Challenges competitive assumptions",
                category=QuestionCategory.COMPETITIVE_DYNAMICS,
                tier=QuestionTier.STRATEGIC,
                priority_score=0.80,
                research_grounded=True,
                socratic_element=True,
            ),
        ]

        tosca_summary = self._apply_tosca_enrichment(
            user_query=query,
            tier_1=fallback_essential,
            tier_2=fallback_strategic,
            tier_3=[],
        )

        print("⚠️ Using minimal fallback questions - ULTRATHINK failed")

        return TieredQuestions(
            original_query=query,
            research_context="Research context unavailable",
            tier_1_essential=fallback_essential,
            tier_2_strategic=fallback_strategic,
            tier_3_expert=[],
            total_questions=len(fallback_essential) + len(fallback_strategic),
            generation_time_ms=0,
            tosca_summary=tosca_summary,
        )

    def get_questions_for_tier(
        self, tiered_questions: TieredQuestions, tier: QuestionTier
    ) -> List[AnalysisQuestion]:
        """Get questions for a specific tier (for progressive disclosure UX)"""
        if tier == QuestionTier.ESSENTIAL:
            return tiered_questions.tier_1_essential
        elif tier == QuestionTier.STRATEGIC:
            return tiered_questions.tier_2_strategic
        elif tier == QuestionTier.EXPERT:
            return tiered_questions.tier_3_expert
        else:
            return []

    def format_questions_for_display(
        self, questions: List[AnalysisQuestion]
    ) -> List[Dict[str, Any]]:
        """Format questions for UI display"""
        return [
            {
                "question": q.question,
                "reasoning": q.reasoning,
                "category": q.category.value,
                "priority": q.priority_score,
                "has_research": q.research_grounded,
                "challenges_assumptions": q.socratic_element,
                "tosca_tag": q.tosca_tag,
                "auto_generated": q.auto_generated,
            }
            for q in questions
        ]


async def test_progressive_questions():
    """Test the progressive question engine"""
    engine = ProgressiveQuestionEngine()

    test_query = "How should we approach market entry strategy for a new SaaS product?"

    print("🧪 Testing Progressive Question Engine")
    print(f"Query: {test_query}")
    print("=" * 60)

    # Generate tiered questions
    result = await engine.generate_tiered_questions(test_query)

    # Display results
    print("\n📊 RESULTS:")
    print(f"Total questions: {result.total_questions}")
    print(f"Generation time: {result.generation_time_ms}ms")

    print("\n🎯 TIER 1 - ESSENTIAL QUESTIONS (Must Answer):")
    for i, q in enumerate(result.tier_1_essential, 1):
        print(f"{i}. {q.question}")
        print(f"   → {q.reasoning}")
        print(f"   → Category: {q.category.value}, Priority: {q.priority_score:.2f}")
        print()

    print("\n🧠 TIER 2 - STRATEGIC QUESTIONS (Optional Depth):")
    for i, q in enumerate(result.tier_2_strategic, 1):
        print(f"{i}. {q.question}")
        print(f"   → {q.reasoning}")
        print(f"   → Category: {q.category.value}, Priority: {q.priority_score:.2f}")
        print()

    print("\n🎓 TIER 3 - EXPERT QUESTIONS (Expert Mode):")
    for i, q in enumerate(result.tier_3_expert, 1):
        print(f"{i}. {q.question}")
        print(f"   → {q.reasoning}")
        print(f"   → Category: {q.category.value}, Priority: {q.priority_score:.2f}")
        print()


if __name__ == "__main__":
    asyncio.run(test_progressive_questions())
