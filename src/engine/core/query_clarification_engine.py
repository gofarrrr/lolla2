"""
Query Clarification Engine - HITL Implementation

Analyzes user queries for ambiguities and generates targeted clarifying questions
to improve problem understanding before cognitive analysis.

Features:
- Multi-dimensional query analysis
- Uncertainty detection using LLM
- Smart question generation and ranking
- Business-context awareness
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import re

from src.engine.integrations.perplexity_client import (
    PerplexityClient,
    KnowledgeQueryType,
)
from src.core.resilient_llm_client import get_resilient_llm_client, CognitiveCallContext
from src.core.performance_optimizer import (
    get_performance_optimizer,
    fast_response,
)


class QuestionTier(str, Enum):
    """Question complexity tiers for progressive clarification"""

    ESSENTIAL = "essential"  # Simple, business-focused questions
    EXPERT = "expert"  # Advanced, strategic questions


@dataclass
class EngagementBrief:
    """Structured summary of user's request"""

    objective: str
    platform: str
    key_features: List[str]
    confidence: float = 0.8

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for safe JSON serialization"""
        return {
            "objective": self.objective,
            "platform": self.platform,
            "key_features": self.key_features,
            "confidence": self.confidence,
        }


@dataclass
class QueryDimension:
    """Represents a dimension of analysis for a query"""

    name: str
    description: str
    current_clarity: float  # 0.0 to 1.0, where 1.0 is completely clear
    uncertainty_areas: List[str]
    potential_questions: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for safe JSON serialization"""
        return {
            "name": self.name,
            "description": self.description,
            "current_clarity": self.current_clarity,
            "uncertainty_areas": self.uncertainty_areas,
            "potential_questions": self.potential_questions,
        }


@dataclass
class ClarificationQuestion:
    """Represents a single clarification question"""

    question: str
    dimension: str
    impact_score: float  # 0.0 to 1.0, how much this affects analysis quality
    business_relevance: float  # 0.0 to 1.0, business importance
    complexity_level: str  # "simple", "medium", "complex"
    question_type: str  # "multiple_choice", "open_ended", "yes_no", "numeric"
    tier: QuestionTier = QuestionTier.ESSENTIAL  # NEW: Question complexity tier
    context_hint: Optional[str] = None  # Helper text for user
    rationale: Optional[str] = (
        None  # Template-based explanation of why this question is needed
    )
    grounded_context: Optional[str] = None  # NEW: Context from Perplexity research

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for safe JSON serialization"""
        return {
            "question": self.question,
            "dimension": self.dimension,
            "impact_score": self.impact_score,
            "business_relevance": self.business_relevance,
            "complexity_level": self.complexity_level,
            "question_type": self.question_type,
            "tier": (
                self.tier.value if isinstance(self.tier, QuestionTier) else self.tier
            ),
            "context_hint": self.context_hint,
            "rationale": self.rationale,
            "grounded_context": self.grounded_context,
        }


@dataclass
class TieredQuestions:
    """Questions organized by complexity tier"""

    essential_questions: List[ClarificationQuestion] = field(default_factory=list)
    expert_questions: List[ClarificationQuestion] = field(default_factory=list)
    engagement_brief: Optional[EngagementBrief] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for safe JSON serialization"""
        return {
            "essential_questions": [q.to_dict() for q in self.essential_questions],
            "expert_questions": [q.to_dict() for q in self.expert_questions],
            "engagement_brief": (
                self.engagement_brief.to_dict() if self.engagement_brief else None
            ),
        }


@dataclass
class QueryAnalysisResult:
    """Result of query analysis"""

    original_query: str
    overall_clarity_score: float  # 0.0 to 1.0
    needs_clarification: bool
    dimensions: List[QueryDimension]
    recommended_questions: List[ClarificationQuestion]
    tiered_questions: Optional[TieredQuestions] = None  # NEW: Tiered questions
    engagement_brief: Optional[EngagementBrief] = None  # NEW: Brief summary
    analysis_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClarificationResponse:
    """User's response to clarification questions"""

    question_id: str
    question: str
    response: str
    confidence: float = 1.0  # User's confidence in their answer


class QueryClarificationEngine:
    """
    Intelligent query clarification system using LLM-based analysis.

    Implements the HITL pattern: analyze query dimensions → identify uncertainties
    → generate targeted questions → rank by relevance → present top 3-5 questions.
    """

    def __init__(
        self, llm_client=None, perplexity_client: Optional[PerplexityClient] = None
    ):
        """Initialize the clarification engine."""
        self.logger = logging.getLogger(__name__)
        self.llm_client = llm_client or get_resilient_llm_client()
        self.perplexity_client = perplexity_client or PerplexityClient()
        self.performance_optimizer = get_performance_optimizer()

        # Operation Crystal Day 2: Dynamic clarity thresholds based on user expertise
        # Base thresholds by expertise level (can be overridden at runtime)
        self.expertise_thresholds = {
            "executive": 0.5,  # Executives need less detail, higher tolerance for ambiguity
            "strategic": 0.6,  # Strategy professionals need moderate clarity
            "analytical": 0.7,  # Analysts need good clarity
            "technical": 0.8,  # Technical users need high precision
        }

        # Default threshold (fallback)
        self.default_clarity_threshold = 0.7

        # Configuration
        self.max_questions = 5  # Maximum questions to ask user
        self.min_questions = 2  # Minimum questions if clarification needed

        # Track user interactions for adaptive learning
        self.user_interaction_history = (
            {}
        )  # Will track per user for threshold adaptation

        # Operation Crystal Day 3: Template-based question rationales
        self.question_rationales = {
            "business_objective": "Understanding your goal helps me focus the analysis.",
            "scope_boundaries": "Clear scope prevents analysis of irrelevant areas.",
            "stakeholder_context": "Knowing stakeholders ensures aligned recommendations.",
            "constraints_limitations": "Understanding limits shapes realistic solutions.",
            "success_metrics": "Metrics guide our recommendation priorities.",
            "timeline_urgency": "Timeline constraints shape our solution approach.",
            "resource_context": "Resource availability affects implementation feasibility.",
            "decision_authority": "Understanding decision-makers streamlines approval processes.",
        }

        # Core business dimensions to analyze
        self.analysis_dimensions = [
            "business_objective",  # What are they trying to achieve?
            "scope_boundaries",  # What's included/excluded?
            "stakeholder_context",  # Who's involved and how?
            "constraints_limitations",  # What limits the solution space?
            "success_metrics",  # How will success be measured?
            "timeline_urgency",  # When is this needed?
            "resource_context",  # What resources are available?
            "decision_authority",  # Who makes the final decisions?
        ]

        # NEW: Question tiering configuration
        self.essential_question_indicators = [
            "who",
            "what",
            "primary goal",
            "main objective",
            "budget",
            "timeline",
            "key stakeholder",
            "target audience",
            "problem statement",
            "desired outcome",
            "success criteria",
        ]

        self.expert_question_indicators = [
            "network effects",
            "second-order",
            "systemic",
            "emergent",
            "strategic moat",
            "flywheel",
            "competitive dynamics",
            "market positioning",
            "ecosystem",
            "platform effects",
            "scalability",
        ]

        # Tiering thresholds
        self.max_essential_questions = 5
        self.max_expert_questions = 8

        self.logger.info(
            "🤔 QueryClarificationEngine initialized with tiered questions"
        )

    async def analyze_query(
        self,
        query: str,
        business_context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        user_expertise: Optional[str] = None,
        clarity_threshold_override: Optional[float] = None,
    ) -> QueryAnalysisResult:
        """
        Analyze query for clarity and generate clarification questions if needed.

        Operation Crystal Day 2: Now supports adaptive clarity thresholds based on user expertise.

        Args:
            query: The user's original query
            business_context: Optional additional business context
            user_id: Optional user ID for adaptive learning
            user_expertise: User expertise level (executive, strategic, analytical, technical)
            clarity_threshold_override: Optional manual threshold override (0.0-1.0)

        Returns:
            QueryAnalysisResult with analysis and recommended questions
        """
        # Operation Crystal Day 2: Determine adaptive clarity threshold
        clarity_threshold = self._get_adaptive_threshold(
            user_id=user_id,
            user_expertise=user_expertise,
            clarity_threshold_override=clarity_threshold_override,
        )

        self.logger.info(f"🔍 Analyzing query for clarity: '{query[:100]}...'")
        self.logger.info(
            f"🎯 Using adaptive clarity threshold: {clarity_threshold} (expertise: {user_expertise or 'unknown'})"
        )

        # Step 1: Multi-dimensional analysis
        dimensions = await self._analyze_query_dimensions(query, business_context or {})

        # Step 2: Calculate overall clarity score
        overall_clarity = self._calculate_overall_clarity(dimensions)

        # Step 3: Determine if clarification is needed using adaptive threshold
        needs_clarification = overall_clarity < clarity_threshold

        # Step 4: Generate and rank questions if needed
        recommended_questions = []
        if needs_clarification:
            all_questions = await self._generate_clarification_questions(
                dimensions, query
            )
            recommended_questions = self._rank_and_filter_questions(all_questions)

        result = QueryAnalysisResult(
            original_query=query,
            overall_clarity_score=overall_clarity,
            needs_clarification=needs_clarification,
            dimensions=dimensions,
            recommended_questions=recommended_questions,
            analysis_metadata={
                "timestamp": datetime.utcnow().isoformat(),
                "engine_version": "2.0.0",  # Operation Crystal Day 2
                "analysis_time_ms": 0,  # Will be updated by caller
                "adaptive_threshold_used": clarity_threshold,
                "user_expertise": user_expertise,
                "threshold_source": "adaptive" if user_expertise else "default",
            },
        )

        self.logger.info(
            f"✅ Query analysis complete - Clarity: {overall_clarity:.2f}, "
            f"Needs clarification: {needs_clarification}, "
            f"Questions generated: {len(recommended_questions)}"
        )

        return result

    async def synthesize_engagement_brief(self, raw_query: str) -> EngagementBrief:
        """
        NEW: Generate a structured engagement brief from raw query.

        Creates a concise, digestible summary that proves to the user
        that the system understands their request.

        Args:
            raw_query: User's original, unstructured query

        Returns:
            EngagementBrief with objective, platform, and key features
        """
        self.logger.info(
            f"📋 Synthesizing engagement brief for: '{raw_query[:100]}...'"
        )

        prompt = f"""You are an expert at distilling complex business requests into clear, actionable briefs.

Analyze the following user query and extract:
1. The core OBJECTIVE (what they want to achieve)
2. The PLATFORM/PRODUCT being discussed 
3. The key FEATURES or differentiators mentioned

User Query:
{raw_query}

Present this as a simple JSON object. Be concise and business-focused.

Example format:
{{
  "objective": "Develop go-to-market strategy and pricing for AI platform",
  "platform": "Cognitive Intelligence Experience combining consultant and skeptic",
  "key_features": ["Glass-box transparency", "Real-time analysis", "Multi-model reasoning"]
}}

Important: Focus on what the user actually wants to achieve, not generic business speak."""

        try:
            context = CognitiveCallContext(
                task_type="problem_classification",
                complexity_score=0.3,
                time_constraints="normal",
            )
            result = await self.llm_client.execute_cognitive_call(prompt, context)
            response = result.content

            # Parse JSON response
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())

                brief = EngagementBrief(
                    objective=data.get(
                        "objective", "Strategic analysis and recommendations"
                    ),
                    platform=data.get("platform", "Business system or initiative"),
                    key_features=data.get("key_features", []),
                    confidence=0.8,
                )

                self.logger.info(f"✅ Engagement brief synthesized: {brief.objective}")
                return brief

        except Exception as e:
            self.logger.error(f"❌ Brief synthesis failed: {e}")

        # Fallback brief
        return EngagementBrief(
            objective="Strategic analysis and recommendations",
            platform="Business system or initiative",
            key_features=["Strategic insights", "Data-driven analysis"],
            confidence=0.3,
        )

    @fast_response(operation_name="generate_tiered_questions")
    async def generate_tiered_questions(
        self,
        raw_query: str,
        business_context: Optional[Dict[str, Any]] = None,
        user_expertise: Optional[str] = None,
    ) -> TieredQuestions:
        """
        NEW: Generate questions organized by complexity tier.

        Creates essential questions (simple, business-focused) and
        expert questions (strategic, advanced) for progressive disclosure.

        Args:
            raw_query: User's original query
            business_context: Optional business context
            user_expertise: User expertise level

        Returns:
            TieredQuestions with essential and expert questions
        """
        self.logger.info(f"🎯 Generating tiered questions for: '{raw_query[:100]}...'")

        # Step 1: Generate engagement brief
        engagement_brief = await self.synthesize_engagement_brief(raw_query)

        # Step 2: Run full dimensional analysis
        dimensions = await self._analyze_query_dimensions(
            raw_query, business_context or {}
        )

        # Step 3: Generate all potential questions
        all_questions = []
        for dimension in dimensions:
            if dimension.current_clarity < 0.6:  # Focus on unclear dimensions
                for question in dimension.potential_questions:
                    enhanced_question = await self._enhance_question(
                        question, dimension, raw_query
                    )
                    # Categorize question tier
                    enhanced_question.tier = self._categorize_question_tier(
                        enhanced_question
                    )
                    all_questions.append(enhanced_question)

        # Step 4: Apply Perplexity grounding (if available)
        grounded_questions = await self._ground_questions_with_perplexity(
            all_questions, raw_query, engagement_brief
        )

        # Step 5: Separate and rank by tier
        essential_questions = [
            q for q in grounded_questions if q.tier == QuestionTier.ESSENTIAL
        ]
        expert_questions = [
            q for q in grounded_questions if q.tier == QuestionTier.EXPERT
        ]

        # Rank within each tier
        essential_questions = self._rank_questions_within_tier(essential_questions)[
            : self.max_essential_questions
        ]
        expert_questions = self._rank_questions_within_tier(expert_questions)[
            : self.max_expert_questions
        ]

        result = TieredQuestions(
            essential_questions=essential_questions,
            expert_questions=expert_questions,
            engagement_brief=engagement_brief,
        )

        self.logger.info(
            f"✅ Tiered questions generated: {len(essential_questions)} essential, {len(expert_questions)} expert"
        )

        return result

    def _categorize_question_tier(
        self, question: ClarificationQuestion
    ) -> QuestionTier:
        """
        NEW: Categorize a question as ESSENTIAL or EXPERT based on content.

        Args:
            question: The clarification question to categorize

        Returns:
            QuestionTier (ESSENTIAL or EXPERT)
        """
        question_lower = question.question.lower()

        # Check for essential indicators
        essential_score = sum(
            1
            for indicator in self.essential_question_indicators
            if indicator in question_lower
        )

        # Check for expert indicators
        expert_score = sum(
            1
            for indicator in self.expert_question_indicators
            if indicator in question_lower
        )

        # Business relevance and complexity influence tiering
        if question.business_relevance > 0.8 and question.complexity_level == "simple":
            essential_score += 2
        elif question.complexity_level == "complex" or "strategic" in question_lower:
            expert_score += 2

        # Core business dimensions are typically essential
        if question.dimension in [
            "business_objective",
            "timeline_urgency",
            "success_metrics",
        ]:
            essential_score += 2
        elif question.dimension in ["constraints_limitations", "decision_authority"]:
            expert_score += 1

        # Tie-breaking: default to essential for accessibility
        if essential_score >= expert_score:
            return QuestionTier.ESSENTIAL
        else:
            return QuestionTier.EXPERT

    def _rank_questions_within_tier(
        self, questions: List[ClarificationQuestion]
    ) -> List[ClarificationQuestion]:
        """
        NEW: Rank questions within their tier by importance.

        Args:
            questions: List of questions to rank

        Returns:
            Ranked list of questions
        """
        # Calculate combined score (impact + business relevance + tier-specific factors)
        for question in questions:
            base_score = (question.impact_score * 0.4) + (
                question.business_relevance * 0.4
            )

            # Tier-specific adjustments
            if question.tier == QuestionTier.ESSENTIAL:
                # Prioritize clarity and business importance for essential questions
                if question.complexity_level == "simple":
                    base_score += 0.1
                if question.dimension in ["business_objective", "timeline_urgency"]:
                    base_score += 0.1
            else:  # EXPERT
                # Prioritize strategic value for expert questions
                if (
                    "strategic" in question.question.lower()
                    or "competitive" in question.question.lower()
                ):
                    base_score += 0.1

            question.tier_score = base_score

        # Sort by tier score (descending)
        return sorted(
            questions, key=lambda q: getattr(q, "tier_score", 0), reverse=True
        )

    async def _ground_questions_with_perplexity(
        self,
        questions: List[ClarificationQuestion],
        raw_query: str,
        engagement_brief: EngagementBrief,
    ) -> List[ClarificationQuestion]:
        """
        NEW: Ground questions with real-world context using Perplexity research.

        Makes targeted research queries to ensure questions are contextually relevant
        and not generic. Uses up to 3 research queries to ground essential questions.

        Args:
            questions: List of questions to ground
            raw_query: Original user query for context
            engagement_brief: Structured brief for targeted research

        Returns:
            Questions with grounded context
        """
        if not questions:
            return questions

        self.logger.info(
            f"🔬 Grounding {len(questions)} questions with Perplexity research"
        )

        try:
            # Prepare 3 targeted research queries based on engagement brief
            research_queries = [
                f"Current industry trends and considerations for {engagement_brief.platform} in 2025",
                f"Common business challenges and success factors for {engagement_brief.objective}",
                f"Key strategic decisions and metrics for {' '.join(engagement_brief.key_features[:2])}",
            ]

            # Execute research queries (up to 3 as specified in requirements)
            research_results = []
            for query in research_queries[:3]:  # Limit to 3 queries as per spec
                try:
                    result = await self.perplexity_client.query_knowledge(
                        query=query,
                        query_type=KnowledgeQueryType.CONTEXT_GROUNDING,
                        max_tokens=200,  # Keep concise for grounding
                    )
                    research_results.append(result.content)
                    self.logger.info(f"✅ Research completed: {query[:60]}...")
                except Exception as e:
                    self.logger.warning(f"⚠️ Research query failed: {e}")
                    research_results.append("")

            # Apply research context to questions
            research_context = " ".join(research_results).strip()

            # Ground essential questions with research context
            for question in questions:
                if question.tier == QuestionTier.ESSENTIAL and research_context:
                    # Add contextual relevance to essential questions
                    question.grounded_context = self._extract_relevant_context(
                        question.question, research_context
                    )

                    # Enhance question with context if relevant
                    if (
                        question.grounded_context
                        and len(question.grounded_context) > 50
                    ):
                        # Update question to be more specific based on research
                        enhanced_question = (
                            await self._contextualize_question_with_research(
                                question.question,
                                question.grounded_context,
                                engagement_brief,
                            )
                        )
                        if enhanced_question and enhanced_question != question.question:
                            self.logger.info(
                                f"🎯 Enhanced question with context: {question.question[:50]}... → {enhanced_question[:50]}..."
                            )
                            question.question = enhanced_question

        except Exception as e:
            self.logger.error(f"❌ Perplexity grounding failed: {e}")
            # Continue with ungrounded questions

        return questions

    def _extract_relevant_context(self, question: str, research_context: str) -> str:
        """
        Extract relevant research context for a specific question.

        Args:
            question: The clarification question
            research_context: Research results from Perplexity

        Returns:
            Relevant context snippet
        """
        # Simple keyword matching to find relevant context
        question_keywords = [
            word.lower()
            for word in question.split()
            if len(word) > 3
            and word.lower() not in ["what", "how", "when", "where", "why", "which"]
        ]

        # Find sentences containing question keywords
        sentences = research_context.split(".")
        relevant_sentences = []

        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in question_keywords):
                relevant_sentences.append(sentence.strip())

        # Return first 2 most relevant sentences
        return ". ".join(relevant_sentences[:2]) if relevant_sentences else ""

    async def _contextualize_question_with_research(
        self,
        original_question: str,
        research_context: str,
        engagement_brief: EngagementBrief,
    ) -> str:
        """
        Enhance a question with research context to make it more specific.

        Args:
            original_question: The original question
            research_context: Relevant research context
            engagement_brief: Engagement brief for context

        Returns:
            Enhanced question with context
        """
        try:
            prompt = f"""Enhance this clarification question to be more specific and contextually relevant:

Original Question: {original_question}

Context from research:
{research_context}

Business Context: {engagement_brief.objective} for {engagement_brief.platform}

Make the question more specific and relevant without changing its core intent. Keep it concise and actionable. Return just the enhanced question, nothing else."""

            context = CognitiveCallContext(
                task_type="query_generation",
                complexity_score=0.2,
                time_constraints="normal",
            )
            result = await self.llm_client.execute_cognitive_call(prompt, context)
            response = result.content

            enhanced_question = response.strip().strip('"').strip("'")

            # Validate enhancement (avoid making it too long or too different)
            if (
                len(enhanced_question) < len(original_question) * 2
                and len(enhanced_question.split()) < 25
            ):
                return enhanced_question

        except Exception as e:
            self.logger.warning(f"⚠️ Question contextualization failed: {e}")

        return original_question

    async def _analyze_query_dimensions(
        self, query: str, business_context: Dict[str, Any]
    ) -> List[QueryDimension]:
        """
        Analyze the query across multiple business dimensions using LLM.

        This is the core intelligence of the system - understanding what's clear
        and what's ambiguous in the user's request.
        """
        # Construct the analysis prompt
        prompt = self._build_dimension_analysis_prompt(query, business_context)

        try:
            # Call DeepSeek for dimensional analysis
            context = CognitiveCallContext(
                task_type="complex_reasoning",
                complexity_score=0.7,
                time_constraints="normal",
            )
            result = await self.llm_client.execute_cognitive_call(prompt, context)
            response = result.content

            # Parse the structured response
            dimensions = self._parse_dimension_analysis_response(response)

        except Exception as e:
            self.logger.error(f"❌ LLM dimension analysis failed: {e}")
            # Fallback to rule-based analysis
            dimensions = self._fallback_dimension_analysis(query, business_context)

        return dimensions

    def _build_dimension_analysis_prompt(
        self, query: str, business_context: Dict[str, Any]
    ) -> str:
        """Build the prompt for dimensional analysis."""
        context_str = (
            json.dumps(business_context, indent=2)
            if business_context
            else "No additional context provided"
        )

        return f"""
You are a strategic consultant analyzing a business query for clarity and completeness.

QUERY TO ANALYZE:
{query}

EXISTING BUSINESS CONTEXT:
{context_str}

Your task is to analyze this query across multiple dimensions and identify areas of uncertainty that would benefit from clarification.

For each dimension below, assess:
1. How clear the query is on this dimension (0.0 = completely unclear, 1.0 = completely clear)
2. What specific uncertainties exist
3. What clarifying questions could help

DIMENSIONS TO ANALYZE:
- business_objective: What specific business outcome is desired?
- scope_boundaries: What's included vs excluded in the solution space?
- stakeholder_context: Who are the key stakeholders and decision makers?
- constraints_limitations: What constraints limit the solution options?
- success_metrics: How will success be measured and by whom?
- timeline_urgency: What are the time constraints and deadlines?
- resource_context: What resources (budget, people, systems) are available?
- decision_authority: Who has authority to make decisions and approve solutions?

OUTPUT FORMAT (JSON):
{{
  "dimensions": [
    {{
      "name": "business_objective",
      "description": "Brief description of what this dimension covers",
      "current_clarity": 0.8,
      "uncertainty_areas": ["specific uncertainty 1", "uncertainty 2"],
      "potential_questions": ["clarifying question 1", "question 2"]
    }}
  ],
  "overall_assessment": "Brief summary of query clarity"
}}

Focus on identifying genuine uncertainties that would impact analysis quality. Be specific and actionable.
"""

    def _parse_dimension_analysis_response(self, response: str) -> List[QueryDimension]:
        """Parse LLM response into QueryDimension objects."""
        try:
            # Extract JSON from response
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())

                dimensions = []
                for dim_data in data.get("dimensions", []):
                    dimension = QueryDimension(
                        name=dim_data["name"],
                        description=dim_data["description"],
                        current_clarity=float(dim_data["current_clarity"]),
                        uncertainty_areas=dim_data.get("uncertainty_areas", []),
                        potential_questions=dim_data.get("potential_questions", []),
                    )
                    dimensions.append(dimension)

                return dimensions

        except (json.JSONDecodeError, KeyError) as e:
            self.logger.error(f"❌ Failed to parse dimension analysis response: {e}")

        # Fallback
        return self._fallback_dimension_analysis("", {})

    def _fallback_dimension_analysis(
        self, query: str, business_context: Dict[str, Any]
    ) -> List[QueryDimension]:
        """Simple rule-based fallback when LLM analysis fails."""
        query_lower = query.lower()

        # Basic heuristics for common clarity issues
        dimensions = []

        # Business objective clarity
        has_clear_objective = any(
            word in query_lower
            for word in ["increase", "reduce", "improve", "achieve", "deliver"]
        )
        obj_clarity = 0.8 if has_clear_objective else 0.3

        dimensions.append(
            QueryDimension(
                name="business_objective",
                description="What specific business outcome is desired",
                current_clarity=obj_clarity,
                uncertainty_areas=(
                    ["Specific success metrics unclear"] if obj_clarity < 0.5 else []
                ),
                potential_questions=[
                    "What specific business outcome are you trying to achieve?"
                ],
            )
        )

        # Timeline clarity
        has_timeline = any(
            word in query_lower
            for word in ["month", "year", "quarter", "week", "urgent", "asap", "by"]
        )
        timeline_clarity = 0.7 if has_timeline else 0.2

        dimensions.append(
            QueryDimension(
                name="timeline_urgency",
                description="Time constraints and deadlines",
                current_clarity=timeline_clarity,
                uncertainty_areas=(
                    ["Timeline not specified"] if timeline_clarity < 0.5 else []
                ),
                potential_questions=[
                    "What is your target timeline for this initiative?"
                ],
            )
        )

        # Add more dimensions with simple heuristics
        for dim_name in [
            "scope_boundaries",
            "stakeholder_context",
            "constraints_limitations",
        ]:
            dimensions.append(
                QueryDimension(
                    name=dim_name,
                    description=f"Analysis of {dim_name.replace('_', ' ')}",
                    current_clarity=0.4,  # Conservative estimate
                    uncertainty_areas=[
                        f"{dim_name.replace('_', ' ').title()} not clearly specified"
                    ],
                    potential_questions=[
                        f"Can you clarify the {dim_name.replace('_', ' ')}?"
                    ],
                )
            )

        return dimensions

    def _calculate_overall_clarity(self, dimensions: List[QueryDimension]) -> float:
        """Calculate overall clarity score from dimensional analysis."""
        if not dimensions:
            return 0.0

        # Weight different dimensions based on importance
        dimension_weights = {
            "business_objective": 0.25,  # Most important
            "scope_boundaries": 0.20,
            "stakeholder_context": 0.15,
            "timeline_urgency": 0.10,
            "success_metrics": 0.10,
            "constraints_limitations": 0.08,
            "resource_context": 0.07,
            "decision_authority": 0.05,
        }

        weighted_score = 0.0
        total_weight = 0.0

        for dimension in dimensions:
            weight = dimension_weights.get(dimension.name, 0.05)  # Default weight
            weighted_score += dimension.current_clarity * weight
            total_weight += weight

        return weighted_score / total_weight if total_weight > 0 else 0.0

    async def _generate_clarification_questions(
        self, dimensions: List[QueryDimension], original_query: str
    ) -> List[ClarificationQuestion]:
        """Generate smart clarification questions based on dimensional analysis."""
        # Extract all potential questions from dimensions
        all_potential_questions = []

        for dimension in dimensions:
            if dimension.current_clarity < 0.6:  # Focus on unclear dimensions
                for question in dimension.potential_questions:
                    # Enhance question with LLM if needed
                    enhanced_question = await self._enhance_question(
                        question, dimension, original_query
                    )

                    all_potential_questions.append(enhanced_question)

        return all_potential_questions

    async def _enhance_question(
        self, base_question: str, dimension: QueryDimension, original_query: str
    ) -> ClarificationQuestion:
        """Enhance a basic question with better wording and metadata."""
        # For now, use rule-based enhancement
        # In future, could use LLM to improve question quality

        impact_score = (
            1.0 - dimension.current_clarity
        )  # Higher impact for unclear dimensions
        business_relevance = self._assess_business_relevance(dimension.name)

        # Determine question type
        question_type = self._determine_question_type(base_question)
        complexity_level = self._assess_question_complexity(base_question)

        # Operation Crystal Day 3: Add rationale to clarification questions
        rationale = self.question_rationales.get(
            dimension.name, "This information helps improve analysis quality."
        )

        return ClarificationQuestion(
            question=base_question,
            dimension=dimension.name,
            impact_score=impact_score,
            business_relevance=business_relevance,
            complexity_level=complexity_level,
            question_type=question_type,
            context_hint=self._generate_context_hint(dimension.name),
            rationale=rationale,
        )

    def _assess_business_relevance(self, dimension_name: str) -> float:
        """Assess business relevance of a dimension."""
        relevance_scores = {
            "business_objective": 1.0,
            "success_metrics": 0.9,
            "scope_boundaries": 0.8,
            "timeline_urgency": 0.8,
            "stakeholder_context": 0.7,
            "constraints_limitations": 0.6,
            "resource_context": 0.5,
            "decision_authority": 0.4,
        }
        return relevance_scores.get(dimension_name, 0.5)

    def _determine_question_type(self, question: str) -> str:
        """Determine the type of question for UI presentation."""
        question_lower = question.lower()

        if any(word in question_lower for word in ["yes", "no", "true", "false"]):
            return "yes_no"
        elif any(
            word in question_lower
            for word in ["how much", "how many", "what percentage", "budget"]
        ):
            return "numeric"
        elif any(word in question_lower for word in ["which", "what type", "select"]):
            return "multiple_choice"
        else:
            return "open_ended"

    def _assess_question_complexity(self, question: str) -> str:
        """Assess the complexity level of a question."""
        question_lower = question.lower()

        if len(question.split()) <= 8:
            return "simple"
        elif any(
            word in question_lower
            for word in ["analyze", "evaluate", "compare", "strategy"]
        ):
            return "complex"
        else:
            return "medium"

    def _generate_context_hint(self, dimension_name: str) -> str:
        """Generate helpful context hints for users."""
        hints = {
            "business_objective": "Think about the specific business outcome you want to achieve",
            "scope_boundaries": "Consider what's included vs excluded in your request",
            "stakeholder_context": "Think about who needs to be involved or consulted",
            "timeline_urgency": "Consider deadlines, urgency level, and key milestones",
            "success_metrics": "How will you measure if this initiative is successful?",
            "constraints_limitations": "What limitations or constraints should we consider?",
            "resource_context": "Think about budget, team, and other resource considerations",
            "decision_authority": "Who needs to approve or sign off on recommendations?",
        }
        return hints.get(dimension_name, "Please provide as much detail as possible")

    def _rank_and_filter_questions(
        self, questions: List[ClarificationQuestion]
    ) -> List[ClarificationQuestion]:
        """Rank questions by importance and filter to top N."""
        # Calculate combined score (impact + business relevance)
        for question in questions:
            question.combined_score = (question.impact_score * 0.6) + (
                question.business_relevance * 0.4
            )

        # Sort by combined score (descending)
        ranked_questions = sorted(
            questions, key=lambda q: q.combined_score, reverse=True
        )

        # Filter to max questions, but ensure minimum diversity
        final_questions = []
        used_dimensions = set()

        # First pass: Take highest scoring question from each dimension
        for question in ranked_questions:
            if (
                question.dimension not in used_dimensions
                and len(final_questions) < self.max_questions
            ):
                final_questions.append(question)
                used_dimensions.add(question.dimension)

        # Second pass: Fill remaining slots with highest scoring remaining questions
        for question in ranked_questions:
            if len(final_questions) >= self.max_questions:
                break
            if question not in final_questions:
                final_questions.append(question)

        # Ensure minimum questions if clarification is needed
        return final_questions[: max(self.min_questions, len(final_questions))]

    def enhance_query_with_clarifications(
        self,
        original_query: str,
        clarification_responses: List[ClarificationResponse],
        business_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Enhance the original query with clarification responses.

        Args:
            original_query: The original user query
            clarification_responses: User's responses to clarification questions
            business_context: Optional existing business context

        Returns:
            Enhanced query string that incorporates clarifications
        """
        # Build enhanced context
        enhanced_context = {"original_query": original_query, "clarifications": {}}

        # Process clarification responses
        for response in clarification_responses:
            enhanced_context["clarifications"][response.question] = {
                "response": response.response,
                "confidence": response.confidence,
            }

        # Create enhanced query
        enhanced_parts = [
            f"ORIGINAL REQUEST: {original_query}",
            "",
            "ADDITIONAL CONTEXT FROM USER CLARIFICATIONS:",
        ]

        for response in clarification_responses:
            enhanced_parts.append(f"• {response.question}")
            enhanced_parts.append(f"  Answer: {response.response}")

        if business_context:
            enhanced_parts.extend(
                [
                    "",
                    "EXISTING BUSINESS CONTEXT:",
                    json.dumps(business_context, indent=2),
                ]
            )

        enhanced_query = "\n".join(enhanced_parts)

        self.logger.info(
            f"✅ Enhanced query created with {len(clarification_responses)} clarifications"
        )

        return enhanced_query

    def _get_adaptive_threshold(
        self,
        user_id: Optional[str] = None,
        user_expertise: Optional[str] = None,
        clarity_threshold_override: Optional[float] = None,
    ) -> float:
        """
        Determine the adaptive clarity threshold for a user.

        Operation Crystal Day 2: Implements adaptive thresholds based on user expertise
        and learning from interaction history.

        Args:
            user_id: Optional user identifier for learning
            user_expertise: User expertise level
            clarity_threshold_override: Manual override value

        Returns:
            Adaptive clarity threshold (0.0-1.0)
        """

        # Manual override takes precedence
        if clarity_threshold_override is not None:
            self.logger.info(
                f"🎛️ Using manual threshold override: {clarity_threshold_override}"
            )
            return max(
                0.0, min(1.0, clarity_threshold_override)
            )  # Clamp to valid range

        # Use expertise-based threshold
        if user_expertise and user_expertise.lower() in self.expertise_thresholds:
            base_threshold = self.expertise_thresholds[user_expertise.lower()]

            # Apply adaptive learning adjustments if we have user history
            adapted_threshold = self._apply_adaptive_learning(user_id, base_threshold)

            self.logger.info(
                f"🧠 Adaptive threshold for {user_expertise}: {base_threshold} → {adapted_threshold}"
            )
            return adapted_threshold

        # Fallback to default
        self.logger.info(
            f"📊 Using default threshold: {self.default_clarity_threshold}"
        )
        return self.default_clarity_threshold

    def _apply_adaptive_learning(
        self, user_id: Optional[str], base_threshold: float
    ) -> float:
        """
        Apply adaptive learning adjustments to the base threshold.

        This method adjusts thresholds based on user interaction history.
        The threshold is recalculated every 5-10 engagements as specified.
        """
        if not user_id:
            return base_threshold

        # Get user interaction history
        user_history = self.user_interaction_history.get(
            user_id,
            {
                "engagement_count": 0,
                "clarification_skip_rate": 0.0,
                "avg_questions_answered": 0.0,
                "last_threshold_update": 0,
                "threshold_adjustments": [],
            },
        )

        engagement_count = user_history["engagement_count"]

        # Only adjust every 5-10 engagements
        if (
            engagement_count < 5
            or (engagement_count - user_history["last_threshold_update"]) < 5
        ):
            return base_threshold

        # Calculate adjustment based on user behavior patterns
        skip_rate = user_history.get("clarification_skip_rate", 0.0)
        avg_questions = user_history.get("avg_questions_answered", 0.0)

        adjustment = 0.0

        # If user frequently skips clarifications, they prefer higher ambiguity tolerance
        if skip_rate > 0.6:  # More than 60% skip rate
            adjustment -= 0.1  # Lower threshold (more tolerant)
            self.logger.info(
                f"📉 User {user_id} skips clarifications ({skip_rate:.1%}), lowering threshold"
            )

        # If user consistently answers many questions, they're detail-oriented
        elif avg_questions > 4.0:  # Consistently answers most questions
            adjustment += 0.05  # Slightly higher threshold (less tolerant)
            self.logger.info(
                f"📈 User {user_id} answers many questions ({avg_questions:.1f} avg), raising threshold"
            )

        # Apply adjustment with bounds
        adapted_threshold = max(0.3, min(0.9, base_threshold + adjustment))

        # Record adjustment
        user_history["threshold_adjustments"].append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "base_threshold": base_threshold,
                "adjustment": adjustment,
                "final_threshold": adapted_threshold,
                "reason": f"skip_rate: {skip_rate:.1%}, avg_questions: {avg_questions:.1f}",
            }
        )

        user_history["last_threshold_update"] = engagement_count
        self.user_interaction_history[user_id] = user_history

        return adapted_threshold

    def record_user_interaction(
        self,
        user_id: str,
        clarification_requested: bool,
        clarification_skipped: bool = False,
        questions_answered: int = 0,
    ):
        """
        Record user interaction for adaptive learning.

        Should be called after each clarification interaction to build learning history.
        """
        if not user_id:
            return

        # Initialize or update user history
        if user_id not in self.user_interaction_history:
            self.user_interaction_history[user_id] = {
                "engagement_count": 0,
                "clarification_skip_rate": 0.0,
                "avg_questions_answered": 0.0,
                "last_threshold_update": 0,
                "threshold_adjustments": [],
                "total_skips": 0,
                "total_clarifications": 0,
                "total_questions_answered": 0,
            }

        user_history = self.user_interaction_history[user_id]

        # Update counters
        user_history["engagement_count"] += 1

        if clarification_requested:
            user_history["total_clarifications"] += 1

            if clarification_skipped:
                user_history["total_skips"] += 1
            else:
                user_history["total_questions_answered"] += questions_answered

        # Recalculate rates
        if user_history["total_clarifications"] > 0:
            user_history["clarification_skip_rate"] = (
                user_history["total_skips"] / user_history["total_clarifications"]
            )

            clarifications_completed = (
                user_history["total_clarifications"] - user_history["total_skips"]
            )
            if clarifications_completed > 0:
                user_history["avg_questions_answered"] = (
                    user_history["total_questions_answered"] / clarifications_completed
                )

        self.logger.info(
            f"📝 Recorded interaction for user {user_id}: "
            f"engagement #{user_history['engagement_count']}, "
            f"skip rate: {user_history['clarification_skip_rate']:.1%}"
        )

    def get_adaptive_threshold_info(
        self, user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get information about adaptive thresholds for debugging/monitoring."""
        info = {
            "expertise_thresholds": self.expertise_thresholds,
            "default_threshold": self.default_clarity_threshold,
            "adaptive_learning_enabled": True,
            "version": "Operation Crystal Day 2",
        }

        if user_id and user_id in self.user_interaction_history:
            user_history = self.user_interaction_history[user_id]
            info["user_specific"] = {
                "user_id": user_id,
                "engagement_count": user_history["engagement_count"],
                "skip_rate": user_history["clarification_skip_rate"],
                "avg_questions": user_history["avg_questions_answered"],
                "recent_adjustments": user_history["threshold_adjustments"][
                    -3:
                ],  # Last 3
            }

        return info


# Factory function for easy instantiation
def create_query_clarification_engine(llm_client=None) -> QueryClarificationEngine:
    """Create and configure a QueryClarificationEngine instance."""
    return QueryClarificationEngine(llm_client)
