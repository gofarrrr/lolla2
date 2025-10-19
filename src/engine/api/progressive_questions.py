"""
Progressive Questions API - Generate dynamic clarification questions
Integrates with real LLM APIs to create contextual questions
"""

import asyncio
import time
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import httpx
import os
from dotenv import load_dotenv

from src.engine.core.token_tracker import get_token_tracker
from src.core.research_based_query_enhancer import get_research_query_enhancer
from src.integrations.llm.unified_client import get_unified_llm_client
from src.engine.services.llm.provider_policy import Flow, get_provider_chain

# Load environment
load_dotenv()

router = APIRouter(prefix="/api/progressive-questions", tags=["Progressive Questions"])


class ProblemStatement(BaseModel):
    statement: str
    context: Optional[Dict[str, Any]] = {}
    industry: Optional[str] = None
    company_size: Optional[str] = None


class QuestionLevel(BaseModel):
    id: str
    title: str
    description: str
    quality_increase: str
    questions: List[str]
    color: str
    is_required: bool


class ProgressiveQuestionsResponse(BaseModel):
    engagement_id: str
    problem_statement: str
    levels: List[QuestionLevel]
    total_questions: int
    generation_time_ms: int
    cost_usd: float
    llm_provider: str


class AnswerSubmission(BaseModel):
    engagement_id: str
    answers: Dict[str, str]  # question_id -> answer
    requested_analysis_quality: int  # 60, 85, or 95


# Root OPTIONS for contract guardian
@router.options("/")
async def progressive_questions_root_options():
    return {"status": "ok"}

@router.post("/generate", response_model=ProgressiveQuestionsResponse)
async def generate_progressive_questions(problem: ProblemStatement):
    """Generate progressive clarification questions using research-enhanced Grok-4-Fast"""

    start_time = time.time()
    tracker = get_token_tracker()
    engagement_id = f"pq-{int(time.time())}"

    try:
        print(f"🚀 Generating progressive questions with research enhancer...")
        print(f"📝 Problem: {problem.statement}")
        
        # Initialize research enhancer and unified LLM client
        research_enhancer = get_research_query_enhancer()
        llm_client = get_unified_llm_client()
        
        # Generate questions using the 10-lens research framework
        context_info = {
            "engagement_id": engagement_id,
            "industry": problem.industry,
            "company_size": problem.company_size,
            **problem.context
        }
        
        # Use research enhancer to generate structured questions
        enhanced_questions = await research_enhancer.generate_clarification_questions(
            problem.statement, 
            context_info
        )
        
        generation_time = int((time.time() - start_time) * 1000)
        print(f"✅ Research-enhanced questions generated in {generation_time}ms")
        
        # Parse enhanced questions into 3-tier structure
        levels = _parse_research_enhanced_questions(enhanced_questions, problem.statement)
        
        # Calculate metrics
        total_questions = sum(len(level.questions) for level in levels)
        estimated_tokens = sum(len(str(level.questions)) for level in levels) * 1.3
        cost = estimated_tokens * 0.000002 / 1000  # Grok-4-Fast pricing
        
        # Track usage
        tracker.track_usage(
            phase="question_generation",
            provider="openrouter",
            model="grok-4-fast",
            tokens_used=int(estimated_tokens),
            cost_usd=cost,
            response_time_ms=generation_time,
            engagement_id=engagement_id,
            call_type="research_enhanced_questions",
            raw_llm_output=enhanced_questions,
            prompt_template_used="research_based_10_lens_framework",
        )

        return ProgressiveQuestionsResponse(
            engagement_id=engagement_id,
            problem_statement=problem.statement,
            levels=levels,
            total_questions=total_questions,
            generation_time_ms=generation_time,
            cost_usd=cost,
            llm_provider="openrouter-grok-4-fast-research-enhanced",
        )

    except Exception as e:
        print(f"❌ Research-enhanced question generation failed: {str(e)}")
        
        # Fallback to structured generation with direct LLM call
        try:
            print("🔄 Falling back to direct Grok-4-Fast generation...")
            llm_client = get_unified_llm_client()
            
            context_str = f"\nContext: {problem.context}" if problem.context else ""
            
            prompt = f"""Generate 3 levels of clarification questions for this business problem.

PROBLEM: {problem.statement}{context_str}

CRITICAL: Ask about USER-SPECIFIC information only the user knows about themselves.
✅ Ask about: Their budget, team capabilities, connections, constraints, timeline, risk tolerance, past experiences
✅ Ask about: What they have, what they can do, what they want, what they're comfortable with
❌ DON'T ask: General market/industry knowledge we can research (target customers, competitive landscape, trends)
❌ DON'T ask: Things we can deduce from their problem statement

Generate exactly:

ESSENTIAL QUESTIONS (5 questions):
- User's available resources, timeline, and success criteria
- Internal capabilities and constraints
- Personal expectations and comfort levels

STRATEGIC DEPTH QUESTIONS (8 questions):
- Risk tolerance and decision-making authority
- Organizational politics and approval processes
- Past experiences and vendor relationships
- Team bandwidth and executive support

EXPERT INSIGHTS QUESTIONS (10 questions):
- Cultural factors and adoption challenges
- Internal champions and blockers
- Preferred communication styles
- Organizational dependencies and constraints

Format each level with clear headers and numbered questions. Focus on what only the user knows about themselves."""

            response = await llm_client.call_best_available_provider(
                messages=[{"role": "user", "content": prompt}],
                provider_preference=get_provider_chain(Flow.ORACLE),
                model="grok-4-fast",
                max_tokens=2000,
                temperature=0.7
            )
            
            generation_time = int((time.time() - start_time) * 1000)
            
            # Parse the response
            levels = _parse_questions_response(response.get("content", ""))
            
            total_questions = sum(len(level.questions) for level in levels)
            estimated_tokens = len(response.get("content", "").split()) * 1.3
            cost = estimated_tokens * 0.000002 / 1000
            
            return ProgressiveQuestionsResponse(
                engagement_id=engagement_id,
                problem_statement=problem.statement,
                levels=levels,
                total_questions=total_questions,
                generation_time_ms=generation_time,
                cost_usd=cost,
                llm_provider="openrouter-grok-4-fast",
            )
            
        except Exception as fallback_error:
            print(f"❌ Fallback also failed: {str(fallback_error)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to generate questions with research enhancer and fallback: {str(e)} | {str(fallback_error)}"
            )


def _parse_research_enhanced_questions(enhanced_questions: List[str], problem_statement: str) -> List[QuestionLevel]:
    """Parse research enhancer output into 3-tier structure for frontend"""
    
    try:
        # Enhanced questions is already a list of strings
        questions_list = enhanced_questions if isinstance(enhanced_questions, list) else []
        
        # Filter out empty questions
        questions_list = [q.strip() for q in questions_list if q and q.strip()]
        
        print(f"📋 Processing {len(questions_list)} research-enhanced questions")
        
        # Distribute questions across 3 tiers
        total_questions = len(questions_list)
        
        if total_questions >= 15:
            # We have enough questions for full distribution
            essential_count = 5
            strategic_count = 8
            expert_count = min(10, total_questions - 13)
        elif total_questions >= 8:
            # Partial distribution
            essential_count = min(5, max(3, total_questions // 3))
            strategic_count = min(8, total_questions - essential_count)
            expert_count = total_questions - essential_count - strategic_count
        else:
            # Limited questions - prioritize essential
            essential_count = min(5, total_questions)
            strategic_count = max(0, total_questions - essential_count)
            expert_count = 0
        
        # Create structured levels
        levels = []
        question_index = 0
        
        # Essential Questions
        essential_questions = []
        for i in range(essential_count):
            if question_index < len(questions_list):
                question_text = str(questions_list[question_index]).strip()
                if question_text:
                    essential_questions.append(question_text)
                question_index += 1
        
        levels.append(QuestionLevel(
            id="essential",
            title="ESSENTIAL QUESTIONS",
            description="Required for basic strategic analysis",
            quality_increase="60%",
            color="bg-red-50 border-red-200",
            is_required=True,
            questions=essential_questions[:5]  # Ensure max 5
        ))
        
        # Strategic Questions
        strategic_questions = []
        for i in range(strategic_count):
            if question_index < len(questions_list):
                question_text = str(questions_list[question_index]).strip()
                if question_text:
                    strategic_questions.append(question_text)
                question_index += 1
        
        levels.append(QuestionLevel(
            id="strategic",
            title="STRATEGIC DEPTH QUESTIONS",
            description="Answer these for 40% more comprehensive analysis",
            quality_increase="+25%",
            color="bg-yellow-50 border-yellow-200",
            is_required=False,
            questions=strategic_questions[:8]  # Ensure max 8
        ))
        
        # Expert Questions
        expert_questions = []
        for i in range(expert_count):
            if question_index < len(questions_list):
                question_text = str(questions_list[question_index]).strip()
                if question_text:
                    expert_questions.append(question_text)
                question_index += 1
        
        levels.append(QuestionLevel(
            id="expert",
            title="EXPERT INSIGHTS QUESTIONS",
            description="Answer these for maximum McKinsey-level depth",
            quality_increase="+10%",
            color="bg-green-50 border-green-200",
            is_required=False,
            questions=expert_questions[:10]  # Ensure max 10
        ))
        
        # If we don't have enough questions, use fallback defaults
        if sum(len(level.questions) for level in levels) < 5:
            print("⚠️ Not enough questions from research enhancer, using enhanced defaults")
            return _get_enhanced_default_questions(problem_statement)
        
        print(f"✅ Parsed into {len(levels)} levels: {[len(l.questions) for l in levels]} questions")
        return levels
        
    except Exception as e:
        print(f"❌ Error parsing research questions: {e}")
        return _get_enhanced_default_questions(problem_statement)


def _get_enhanced_default_questions(problem_statement: str) -> List[QuestionLevel]:
    """Generate enhanced default questions based on problem statement"""
    
    # Analyze problem statement for industry/domain hints
    is_tech = any(word in problem_statement.lower() for word in ['tech', 'software', 'digital', 'cloud', 'ai', 'platform'])
    is_finance = any(word in problem_statement.lower() for word in ['financial', 'revenue', 'cost', 'profit', 'investment'])
    is_market = any(word in problem_statement.lower() for word in ['market', 'customer', 'competitive', 'share'])
    
    essential_questions = [
        "What specific metrics or KPIs demonstrate the magnitude of this problem?",
        "Who are the key stakeholders affected by this issue and how?",
        "What is your current approach and why isn't it working?",
        "What constraints (budget, time, resources) limit potential solutions?",
        "What would success look like in 6-12 months?"
    ]
    
    strategic_questions = [
        "How does this problem impact your competitive position?",
        "What are the second and third-order effects if this continues?",
        "How do customer segments experience this problem differently?",
        "What external factors (economic, regulatory, tech) influence this?",
        "What capabilities would you need to build vs buy to solve this?",
        "How does this connect to your broader strategic priorities?",
        "What similar challenges have you or competitors solved before?",
        "How do stakeholder incentives align or conflict with solutions?"
    ]
    
    expert_questions = [
        "What leading indicators predicted this 3-6 months ago?",
        "How would solving this create new strategic options?",
        "What unintended consequences might different solutions create?",
        "How does this vary by geography, product line, or market segment?",
        "What would best-in-class organizations do in this situation?",
        "How can you create sustainable competitive advantage from the solution?",
        "What is the total economic impact across your value chain?",
        "How might this problem/solution evolve over 2-3 years?",
        "What predictive models could prevent similar issues?",
        "How does this connect to industry transformation trends?"
    ]
    
    # Customize questions based on problem domain
    if is_tech:
        strategic_questions[1] = "How does this affect your technology adoption and user experience?"
        expert_questions[2] = "What technology convergence trends could change this landscape?"
    elif is_finance:
        strategic_questions[1] = "What is the financial impact across different scenarios?"
        expert_questions[2] = "How do different economic environments affect this problem?"
    elif is_market:
        strategic_questions[1] = "How does this affect your market positioning and customer acquisition?"
        expert_questions[2] = "What market dynamics could fundamentally change this challenge?"
    
    return [
        QuestionLevel(
            id="essential",
            title="ESSENTIAL QUESTIONS",
            description="Required for basic strategic analysis",
            quality_increase="60%",
            color="bg-red-50 border-red-200",
            is_required=True,
            questions=essential_questions
        ),
        QuestionLevel(
            id="strategic",
            title="STRATEGIC DEPTH QUESTIONS",
            description="Answer these for 40% more comprehensive analysis",
            quality_increase="+25%",
            color="bg-yellow-50 border-yellow-200",
            is_required=False,
            questions=strategic_questions
        ),
        QuestionLevel(
            id="expert",
            title="EXPERT INSIGHTS QUESTIONS",
            description="Answer these for maximum McKinsey-level depth",
            quality_increase="+10%",
            color="bg-green-50 border-green-200",
            is_required=False,
            questions=expert_questions
        )
    ]


def _parse_questions_response(response: str) -> List[QuestionLevel]:
    """Parse LLM response into structured question levels"""

    # Default fallback questions if parsing fails
    levels = [
        QuestionLevel(
            id="essential",
            title="ESSENTIAL QUESTIONS",
            description="Required for basic strategic analysis",
            quality_increase="60%",
            color="bg-red-50 border-red-200",
            is_required=True,
            questions=[
                "What specific customer segments show the highest churn/problem rates?",
                "What are the top 3 root causes customers cite for this issue?",
                "How does this problem correlate with competitive market changes?",
                "What key metrics have changed in the past 6-12 months?",
                "What immediate constraints limit potential solutions?",
            ],
        ),
        QuestionLevel(
            id="strategic",
            title="STRATEGIC DEPTH QUESTIONS",
            description="Answer these for 40% more comprehensive analysis",
            quality_increase="+25%",
            color="bg-yellow-50 border-yellow-200",
            is_required=False,
            questions=[
                "How do customer behavior patterns differ between affected and unaffected segments?",
                "What competitive advantages are you losing or could you leverage?",
                "How do internal processes and capabilities contribute to this problem?",
                "What are the second and third-order effects of this issue?",
                "How does this problem vary by geography, product line, or market?",
                "What external factors (economic, regulatory, technological) are relevant?",
                "How do stakeholder incentives align or misalign with solutions?",
                "What similar problems have you or competitors solved before?",
            ],
        ),
        QuestionLevel(
            id="expert",
            title="EXPERT INSIGHTS QUESTIONS",
            description="Answer these for maximum McKinsey-level depth",
            quality_increase="+10%",
            color="bg-green-50 border-green-200",
            is_required=False,
            questions=[
                "What leading indicators predicted this problem 3-6 months ago?",
                "How do different customer personas experience this problem differently?",
                "What is the total economic impact across the entire value chain?",
                "How does solving this problem create new strategic options?",
                "What capabilities would you need to build vs. buy vs. partner?",
                "How might this problem/solution evolve over the next 2-3 years?",
                "What unintended consequences might arise from different solutions?",
                "How does this connect to your broader strategic priorities?",
                "What would best-in-class organizations do in this situation?",
                "How can you create sustainable competitive advantage from the solution?",
            ],
        ),
    ]

    # Try to parse the actual LLM response
    try:
        lines = response.strip().split("\n")
        current_level = None
        current_questions = []

        for line in lines:
            line = line.strip()

            if "ESSENTIAL" in line.upper():
                if current_level:
                    levels[0].questions = current_questions[:5]  # Take first 5
                current_level = "essential"
                current_questions = []

            elif "STRATEGIC" in line.upper():
                if current_level == "essential":
                    levels[0].questions = current_questions[:5]
                current_level = "strategic"
                current_questions = []

            elif "EXPERT" in line.upper() or "INSIGHTS" in line.upper():
                if current_level == "strategic":
                    levels[1].questions = current_questions[:8]  # Take first 8
                current_level = "expert"
                current_questions = []

            elif line and (line[0].isdigit() or line.startswith("-")):
                # Extract question text
                question = line
                # Remove numbering
                if line[0].isdigit():
                    question = (
                        line.split(".", 1)[1].strip()
                        if "." in line
                        else line[1:].strip()
                    )
                elif line.startswith("-"):
                    question = line[1:].strip()

                if question and current_level:
                    current_questions.append(question)

        # Handle final level
        if current_level == "expert":
            levels[2].questions = current_questions[:10]  # Take first 10

    except Exception:
        # Keep default questions if parsing fails
        pass

    return levels


@router.post("/analyze")
async def analyze_with_answers(submission: AnswerSubmission):
    """Analyze the problem using submitted answers"""

    # This would integrate with the main cognitive engine
    # For now, return a placeholder response

    tracker = get_token_tracker()

    answered_count = len([v for v in submission.answers.values() if v.strip()])
    quality_achieved = min(95, 60 + (answered_count * 1.5))

    return {
        "engagement_id": submission.engagement_id,
        "analysis_status": "initiated",
        "questions_answered": answered_count,
        "quality_achieved": int(quality_achieved),
        "estimated_analysis_time_minutes": max(2, int(answered_count * 0.3)),
        "message": f"Analysis initiated with {quality_achieved:.0f}% quality level. The more questions answered, the deeper the insights.",
    }
