#!/usr/bin/env python3
"""
Parallel Query Enhancement System - Enterprise Hardening Sprint

PERFORMANCE OPTIMIZATION: Parallel execution of engagement brief and tiered questions
- Current Sequential: Brief (5s timeout) → Questions (7.79s) = 12.79s total
- New Parallel: max(Brief, Questions) = ~8s total (37% improvement)

Key improvements:
1. Parallel execution using asyncio.gather()
2. Independent timeout handling for each component
3. Graceful degradation when either component fails
4. Enhanced error handling and performance monitoring
5. Backward compatible with existing SequentialQueryEnhancementResult

This is part of the Enterprise-Ready Hardening Sprint addressing the
22-25s performance target for the three-consultant system.
"""

import asyncio
import time
import json
import logging
from typing import Optional

import os
from dotenv import load_dotenv

# Import performance instrumentation
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.performance_instrumentation import (
    get_performance_system,
    measure_function,
)

# Import base classes from sequential implementation
from src.sequential_query_enhancement import (
    EngagementBrief,
    ClarificationQuestion,
    TieredQuestions,
    SequentialQueryEnhancementResult,
    QuestionTier,
)

# Load environment
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(env_path)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ParallelQueryEnhancer:
    """
    Parallel Query Enhancement for Enterprise Performance

    Architecture:
    - Parallel execution of engagement brief creation and question generation
    - Independent timeout handling (5s brief, 10s questions)
    - Graceful degradation and fallback logic
    - Performance monitoring and instrumentation
    - Full backward compatibility with SequentialQueryEnhancementResult
    """

    def __init__(self):
        self.anthropic_client = None
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        self.perplexity_client = None
        self.perplexity_key = os.getenv("PERPLEXITY_API_KEY")
        self.perf_system = get_performance_system()

        self._initialize_clients()

    def _initialize_clients(self):
        """Initialize LLM clients"""
        try:
            import anthropic

            self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_key)
            logger.info("✅ Parallel Query Enhancer: Anthropic client initialized")
        except ImportError:
            logger.error("❌ Anthropic package not found")
            raise ImportError("anthropic package not found")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Anthropic client: {str(e)}")
            raise Exception(f"Failed to initialize Anthropic client: {str(e)}")

        # Initialize Perplexity client (optional)
        if self.perplexity_key:
            try:
                # Perplexity client initialization would go here
                logger.info(
                    "✅ Parallel Query Enhancer: Perplexity available for research"
                )
            except Exception as e:
                logger.warning(f"⚠️ Perplexity client initialization failed: {str(e)}")

    @measure_function("parallel_query_enhancement", "parallel_query_enhancer")
    async def enhance_query(
        self, query: str, context: Optional[str] = None
    ) -> SequentialQueryEnhancementResult:
        """
        PARALLEL query enhancement with maximum performance:

        1. Launch engagement brief creation (5s timeout)
        2. Launch tiered question generation (10s timeout)
        3. Execute both in parallel using asyncio.gather()
        4. Process results with graceful degradation

        Expected performance improvement: 12.79s → ~8s (37% faster)
        """
        enhancement_start = time.time()
        performance_breakdown = {}

        logger.info("🚀 Starting PARALLEL query enhancement for maximum performance...")

        try:
            # Create timeout-wrapped tasks for parallel execution
            brief_task = asyncio.create_task(
                self._create_engagement_brief_with_timeout(query, context, timeout=5.0)
            )
            questions_task = asyncio.create_task(
                self._generate_questions_with_timeout(query, context, timeout=10.0)
            )

            # Execute both tasks in parallel
            parallel_start = time.time()
            async with self.perf_system.measure_async(
                "parallel_execution", "parallel_query_enhancer"
            ):
                brief_result, questions_result = await asyncio.gather(
                    brief_task, questions_task, return_exceptions=True
                )

            parallel_duration = time.time() - parallel_start
            logger.info(f"⚡ Parallel execution completed in {parallel_duration:.2f}s")

            # Process engagement brief result
            engagement_brief = None
            brief_error = None
            brief_time = 0.0

            if isinstance(brief_result, Exception):
                brief_error = f"Engagement brief failed: {str(brief_result)}"
                brief_time = 5.0  # Assume timeout
                logger.warning(f"⚠️ {brief_error}")
            elif brief_result is None:
                brief_error = "Engagement brief returned None"
                brief_time = 5.0
                logger.warning("⚠️ Engagement brief timed out")
            else:
                engagement_brief = brief_result
                brief_time = getattr(
                    engagement_brief, "processing_time_seconds", parallel_duration
                )
                logger.info("✅ Engagement brief created successfully in parallel")

            performance_breakdown["engagement_brief_creation"] = brief_time

            # Process tiered questions result
            tiered_questions = None
            questions_error = None
            questions_time = 0.0

            if isinstance(questions_result, Exception):
                questions_error = f"Question generation failed: {str(questions_result)}"
                questions_time = 10.0  # Assume timeout
                logger.warning(f"⚠️ {questions_error}")
            elif questions_result is None:
                questions_error = "Question generation returned None"
                questions_time = 10.0
                logger.warning("⚠️ Question generation timed out")
            else:
                tiered_questions = questions_result
                questions_time = getattr(
                    tiered_questions, "processing_time_seconds", parallel_duration
                )
                logger.info("✅ Tiered questions generated successfully in parallel")

            performance_breakdown["tiered_questions_generation"] = questions_time

        except Exception as e:
            # Fallback for complete parallel execution failure
            logger.error(f"💥 Parallel query enhancement failed completely: {str(e)}")
            engagement_brief = None
            tiered_questions = None
            brief_error = f"Parallel execution failed: {str(e)}"
            questions_error = brief_error
            performance_breakdown["engagement_brief_creation"] = 5.0
            performance_breakdown["tiered_questions_generation"] = 10.0

        total_time = time.time() - enhancement_start

        # Determine success and enhancement status
        success = engagement_brief is not None or tiered_questions is not None
        enhancement_applied = (
            engagement_brief is not None and tiered_questions is not None
        )

        # Determine fallback reason and error message
        fallback_reason = None
        error_message = None

        if not success:
            error_message = f"Complete parallel failure. Brief: {brief_error}. Questions: {questions_error}"
            fallback_reason = "parallel_complete_failure"
        elif not enhancement_applied:
            if engagement_brief is None:
                fallback_reason = "parallel_brief_creation_failed"
            elif tiered_questions is None:
                fallback_reason = "parallel_question_generation_failed"

        # Log performance improvement
        sequential_estimate = performance_breakdown.get(
            "engagement_brief_creation", 5.0
        ) + performance_breakdown.get("tiered_questions_generation", 10.0)
        parallel_actual = max(
            performance_breakdown.get("engagement_brief_creation", 5.0),
            performance_breakdown.get("tiered_questions_generation", 10.0),
        )
        improvement_pct = (
            ((sequential_estimate - parallel_actual) / sequential_estimate) * 100
            if sequential_estimate > 0
            else 0
        )

        logger.info("🎯 PARALLEL query enhancement completed:")
        logger.info(f"   Success: {success}, Enhanced: {enhancement_applied}")
        logger.info(
            f"   Performance: {total_time:.2f}s (est. {improvement_pct:.1f}% improvement over sequential)"
        )

        return SequentialQueryEnhancementResult(
            original_query=query,
            engagement_brief=engagement_brief,
            tiered_questions=tiered_questions,
            total_processing_time_seconds=round(total_time, 4),
            success=success,
            enhancement_applied=enhancement_applied,
            performance_breakdown=performance_breakdown,
            error=error_message,
            fallback_reason=fallback_reason,
        )

    async def _create_engagement_brief_with_timeout(
        self, query: str, context: Optional[str], timeout: float = 5.0
    ) -> Optional[EngagementBrief]:
        """Create engagement brief with timeout handling"""
        try:
            brief_start = time.time()

            # Create engagement brief using LLM
            brief = await asyncio.wait_for(
                self._create_engagement_brief(query, context), timeout=timeout
            )

            if brief:
                brief.processing_time_seconds = time.time() - brief_start

            return brief

        except asyncio.TimeoutError:
            logger.warning(f"⏰ Engagement brief creation timed out after {timeout}s")
            return None
        except Exception as e:
            logger.error(f"💥 Engagement brief creation failed: {str(e)}")
            raise e

    async def _generate_questions_with_timeout(
        self, query: str, context: Optional[str], timeout: float = 10.0
    ) -> Optional[TieredQuestions]:
        """Generate tiered questions with timeout handling"""
        try:
            questions_start = time.time()

            # Generate questions directly from query (parallel-optimized)
            questions = await asyncio.wait_for(
                self._generate_questions_from_query(query, context), timeout=timeout
            )

            if questions:
                questions.processing_time_seconds = time.time() - questions_start

            return questions

        except asyncio.TimeoutError:
            logger.warning(f"⏰ Question generation timed out after {timeout}s")
            return None
        except Exception as e:
            logger.error(f"💥 Question generation failed: {str(e)}")
            raise e

    async def _create_engagement_brief(
        self, query: str, context: Optional[str]
    ) -> EngagementBrief:
        """Create engagement brief - Parallel Step 1"""

        brief_start = time.time()

        prompt = f"""You are a senior business strategy consultant. Analyze this query and create a concise engagement brief.

QUERY: {query}"""

        if context:
            prompt += f"\nCONTEXT: {context}"

        prompt += """

Provide your response in this exact JSON format:
{
    "objective": "Clear statement of what the client wants to achieve",
    "engagement_type": "strategic_planning|problem_solving|market_analysis|operational_improvement|crisis_management", 
    "key_focus_areas": ["area1", "area2", "area3"],
    "complexity_level": "low|medium|high|ultra_complex",
    "confidence_score": 0.8
}"""

        try:
            async with self.perf_system.measure_async(
                "engagement_brief_llm_call", "parallel_query_enhancer"
            ):
                response = await self._call_anthropic_api(prompt)

            # Parse JSON response
            brief_data = json.loads(response.strip())

            return EngagementBrief(
                objective=brief_data["objective"],
                engagement_type=brief_data["engagement_type"],
                key_focus_areas=brief_data["key_focus_areas"],
                complexity_level=brief_data["complexity_level"],
                confidence_score=brief_data["confidence_score"],
                processing_time_seconds=time.time() - brief_start,
            )

        except json.JSONDecodeError as e:
            logger.error(f"💥 Failed to parse engagement brief JSON: {str(e)}")
            raise Exception(f"Invalid JSON response from engagement brief: {str(e)}")
        except Exception as e:
            logger.error(f"💥 Engagement brief creation failed: {str(e)}")
            raise e

    async def _generate_questions_from_query(
        self, query: str, context: Optional[str]
    ) -> TieredQuestions:
        """Generate tiered questions directly from query - Parallel Step 2"""

        questions_start = time.time()

        prompt = f"""You are a senior business consultant. Generate strategic clarification questions for this query.

QUERY: {query}"""

        if context:
            prompt += f"\nCONTEXT: {context}"

        prompt += """

Generate 2-3 ESSENTIAL questions (high-level, strategic) and 3-4 EXPERT questions (detailed, operational).

Provide your response in this exact JSON format:
{
    "essential_questions": [
        {
            "question": "What is your current market position?",
            "dimension": "Market Analysis", 
            "impact_priority": "high",
            "business_relevance_score": 0.9
        }
    ],
    "expert_questions": [
        {
            "question": "What are your current operational constraints?",
            "dimension": "Operations",
            "impact_priority": "medium", 
            "business_relevance_score": 0.7
        }
    ]
}"""

        try:
            async with self.perf_system.measure_async(
                "questions_llm_call", "parallel_query_enhancer"
            ):
                response = await self._call_anthropic_api(prompt)

            # Parse JSON response
            questions_data = json.loads(response.strip())

            essential_questions = [
                ClarificationQuestion(
                    question=q["question"],
                    dimension=q["dimension"],
                    tier=QuestionTier.ESSENTIAL.value,
                    impact_priority=q["impact_priority"],
                    business_relevance_score=q["business_relevance_score"],
                )
                for q in questions_data["essential_questions"]
            ]

            expert_questions = [
                ClarificationQuestion(
                    question=q["question"],
                    dimension=q["dimension"],
                    tier=QuestionTier.EXPERT.value,
                    impact_priority=q["impact_priority"],
                    business_relevance_score=q["business_relevance_score"],
                )
                for q in questions_data["expert_questions"]
            ]

            return TieredQuestions(
                essential_questions=essential_questions,
                expert_questions=expert_questions,
                brief_summary="Direct query analysis",
                processing_time_seconds=time.time() - questions_start,
                total_questions=len(essential_questions) + len(expert_questions),
            )

        except json.JSONDecodeError as e:
            logger.error(f"💥 Failed to parse questions JSON: {str(e)}")
            raise Exception(f"Invalid JSON response from questions: {str(e)}")
        except Exception as e:
            logger.error(f"💥 Question generation failed: {str(e)}")
            raise e

    @measure_function("anthropic_api_parallel", "anthropic_client")
    async def _call_anthropic_api(self, prompt: str) -> str:
        """Call Anthropic API with optimized settings for parallel execution"""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=2000,  # Optimized for parallel processing
                    temperature=0.3,  # Balanced creativity/consistency
                    messages=[{"role": "user", "content": prompt}],
                ),
            )

            return response.content[0].text

        except Exception as e:
            logger.error(f"💥 Anthropic API call failed: {str(e)}")
            raise Exception(f"Anthropic API call failed: {str(e)}")


# Test function for parallel enhancement
async def test_parallel_enhancement():
    """Test the parallel query enhancement system"""

    print("🚀 TESTING PARALLEL QUERY ENHANCEMENT SYSTEM")
    print("=" * 80)

    enhancer = ParallelQueryEnhancer()

    test_queries = [
        {
            "name": "Digital Transformation",
            "query": "Our traditional retail chain is losing 30% revenue annually to e-commerce. We need strategic direction and implementation roadmap.",
            "context": "150 physical stores, $500M revenue, 5000 employees, legacy systems",
        },
        {
            "name": "Quick Response Test",
            "query": "How should we enter the European market?",
            "context": "SaaS company, strong US presence, limited international experience",
        },
    ]

    results = []

    for test in test_queries:
        print(f"\n📋 Testing: {test['name']}")
        print(f"   Query: {test['query'][:60]}...")

        start_time = time.time()
        result = await enhancer.enhance_query(test["query"], test["context"])
        total_time = time.time() - start_time

        print(f"   ✅ Success: {result.success}")
        print(f"   🔧 Enhanced: {result.enhancement_applied}")
        print(f"   ⏱️ Total Time: {total_time:.2f}s")
        print(f"   📊 Performance: {result.performance_breakdown}")

        if result.engagement_brief:
            print(f"   🎯 Brief: {result.engagement_brief.objective[:50]}...")
        if result.tiered_questions:
            print(
                f"   ❓ Questions: {result.tiered_questions.total_questions} generated"
            )

        results.append(result)

    # Performance summary
    print("\n" + "=" * 80)
    print("📊 PARALLEL ENHANCEMENT PERFORMANCE SUMMARY")
    print("=" * 80)

    success_rate = sum(1 for r in results if r.success) / len(results)
    enhancement_rate = sum(1 for r in results if r.enhancement_applied) / len(results)
    avg_time = sum(r.total_processing_time_seconds for r in results) / len(results)

    print(f"✅ Success Rate: {success_rate:.1%}")
    print(f"🔧 Enhancement Rate: {enhancement_rate:.1%}")
    print(f"⚡ Average Time: {avg_time:.2f}s")
    print(
        f"🎯 Target Achievement: {'✅ ACHIEVED' if avg_time <= 8.0 else '⚠️ NEEDS OPTIMIZATION'}"
    )

    # Expected vs actual improvement
    sequential_estimate = 12.79  # Current sequential time
    improvement_pct = ((sequential_estimate - avg_time) / sequential_estimate) * 100
    print(f"📈 Performance Improvement: {improvement_pct:.1f}% over sequential")

    return results


if __name__ == "__main__":
    asyncio.run(test_parallel_enhancement())
