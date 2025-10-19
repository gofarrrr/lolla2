#!/usr/bin/env python3
"""
N-Way Focused Mental Model Engine
Simplified architecture using only N-way interaction patterns
"""

import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from src.engine.adapters.llm_client import get_resilient_llm_client, CognitiveCallContext  # Migrated
from src.v4.core.v4_supabase_adapter import V4SupabaseAdapter


@dataclass
class NWayPattern:
    """Represents an N-way interaction pattern"""

    interaction_id: str
    models_involved: List[str]
    emergent_effect_summary: str
    mechanism_description: str
    synergy_description: str
    strength: str
    relevant_contexts: List[str]
    instructional_cue: str
    performance_score: float


class NWayPatternMatcher:
    """Matches problems to optimal N-way patterns"""

    def __init__(self, supabase_adapter: V4SupabaseAdapter):
        self.supabase = supabase_adapter

    async def find_matching_patterns(
        self,
        problem_context: str,
        problem_type: str = None,
        strength_filter: str = "High",
        limit: int = 3,
    ) -> List[NWayPattern]:
        """Find N-way patterns that match the problem context"""

        try:
            # Query N-way interactions
            query = self.supabase.supabase.table("nway_interactions").select("*")

            if strength_filter:
                query = query.eq("strength", strength_filter)

            result = await query.limit(limit * 2).execute()  # Get more to filter

            patterns = []
            for item in result.data:
                pattern = NWayPattern(
                    interaction_id=item.get("interaction_id", ""),
                    models_involved=item.get("models_involved", []),
                    emergent_effect_summary=item.get("emergent_effect_summary", ""),
                    mechanism_description=item.get("mechanism_description", ""),
                    synergy_description=item.get("synergy_description", ""),
                    strength=item.get("strength", "Unknown"),
                    relevant_contexts=item.get("relevant_contexts", []),
                    instructional_cue=item.get("instructional_cue_apce", ""),
                    performance_score=item.get("performance_score", 0.0),
                )
                patterns.append(pattern)

            # Simple relevance scoring based on problem context keywords
            scored_patterns = self._score_pattern_relevance(
                patterns, problem_context, problem_type
            )

            return scored_patterns[:limit]

        except Exception as e:
            print(f"❌ Error finding N-way patterns: {e}")
            return []

    def _score_pattern_relevance(
        self,
        patterns: List[NWayPattern],
        problem_context: str,
        problem_type: str = None,
    ) -> List[NWayPattern]:
        """Score and sort patterns by relevance to problem"""

        # Simple keyword matching for now
        problem_keywords = problem_context.lower().split()
        if problem_type:
            problem_keywords.extend(problem_type.lower().split())

        for pattern in patterns:
            score = 0

            # Check interaction ID for relevant terms
            interaction_terms = pattern.interaction_id.lower().replace("_", " ").split()
            for term in interaction_terms:
                if any(
                    keyword in term or term in keyword for keyword in problem_keywords
                ):
                    score += 2

            # Check models involved for relevant terms
            for model in pattern.models_involved:
                model_terms = model.lower().replace("-", " ").split()
                for term in model_terms:
                    if any(
                        keyword in term or term in keyword
                        for keyword in problem_keywords
                    ):
                        score += 1

            # Boost by existing performance score
            score += pattern.performance_score * 10

            # Update performance score with relevance
            pattern.performance_score = score

        # Sort by relevance score
        patterns.sort(key=lambda p: p.performance_score, reverse=True)
        return patterns


class NWayFocusedEngine:
    """Simplified mental model engine using only N-way patterns"""

    def __init__(self, llm_client=None, supabase_adapter=None):
        self.llm_client = llm_client or get_resilient_llm_client()
        self.supabase = supabase_adapter or V4SupabaseAdapter()
        self.pattern_matcher = NWayPatternMatcher(self.supabase)

    async def analyze_with_nway_patterns(
        self, enhanced_query: str, problem_type: str = None
    ) -> Dict[str, Any]:
        """Analyze using N-way interaction patterns only"""

        start_time = datetime.now()

        print("🔍 N-WAY FOCUSED ANALYSIS")
        print(f"Query: {enhanced_query[:100]}...")
        print(f"Problem Type: {problem_type}")
        print("-" * 60)

        try:
            # Step 1: Find matching N-way patterns
            print("📋 Step 1: Finding N-way patterns...")
            patterns = await self.pattern_matcher.find_matching_patterns(
                enhanced_query,
                problem_type,
                strength_filter="High",
                limit=2,  # Use top 2 patterns for focus
            )

            if not patterns:
                return {
                    "status": "error",
                    "error": "No matching N-way patterns found",
                    "analysis_time_ms": (datetime.now() - start_time).total_seconds()
                    * 1000,
                }

            print(f"✅ Found {len(patterns)} matching patterns:")
            for i, pattern in enumerate(patterns, 1):
                print(
                    f"  {i}. {pattern.interaction_id} (Score: {pattern.performance_score:.1f})"
                )
                print(f"     Models: {', '.join(pattern.models_involved[:3])}...")
                print(f"     Effect: {pattern.emergent_effect_summary[:80]}...")

            # Step 2: Apply N-way patterns with LLM
            print("\n🧠 Step 2: Applying N-way patterns...")
            analysis_results = []

            for i, pattern in enumerate(patterns, 1):
                print(f"  Applying pattern {i}: {pattern.interaction_id}")

                result = await self._apply_nway_pattern(enhanced_query, pattern)
                if result:
                    analysis_results.append(result)

            # Step 3: Synthesize results
            print("\n🔬 Step 3: Synthesizing N-way analysis...")
            synthesis = await self._synthesize_nway_results(
                enhanced_query, analysis_results, patterns
            )

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            return {
                "status": "success",
                "analysis_type": "nway_focused",
                "patterns_applied": [p.interaction_id for p in patterns],
                "pattern_count": len(patterns),
                "models_count": sum(len(p.models_involved) for p in patterns),
                "synthesis": synthesis,
                "pattern_details": [
                    {
                        "id": p.interaction_id,
                        "models": p.models_involved,
                        "effect": p.emergent_effect_summary,
                        "strength": p.strength,
                        "relevance_score": p.performance_score,
                    }
                    for p in patterns
                ],
                "execution_time_ms": execution_time,
                "total_cost": getattr(synthesis, "cost_usd", 0.0),
            }

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            print(f"❌ N-way analysis error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "execution_time_ms": execution_time,
            }

    async def _apply_nway_pattern(
        self, query: str, pattern: NWayPattern
    ) -> Optional[Dict[str, Any]]:
        """Apply a specific N-way pattern to the problem"""

        try:
            # Build N-way specific prompt
            nway_prompt = f"""
<nway_pattern_analysis>
PATTERN: {pattern.interaction_id}
QUERY: {query}

MODELS TO COMBINE: {', '.join(pattern.models_involved)}

SYNERGISTIC MECHANISM:
{pattern.mechanism_description}

EMERGENT EFFECT:
{pattern.emergent_effect_summary}

SYNERGY DESCRIPTION:
{pattern.synergy_description}

APPLICATION INSTRUCTION:
{pattern.instructional_cue}

Apply this N-way pattern to analyze the query. Focus on the SYNERGISTIC EFFECTS when these models work together, not individual model application.

Show how the models enhance each other and create emergent insights beyond what any single model could provide.
</nway_pattern_analysis>

Provide structured analysis using this N-way synergistic pattern.
"""

            context = CognitiveCallContext(
                task_type=f"nway_pattern_{pattern.interaction_id}",
                complexity_score=0.8,
                time_constraints="thorough",
                quality_threshold=0.9,
                cost_sensitivity="normal",
            )

            result = await self.llm_client.execute_cognitive_call(nway_prompt, context)

            return {
                "pattern_id": pattern.interaction_id,
                "models_applied": pattern.models_involved,
                "analysis_content": result.content,
                "confidence": result.confidence,
                "cost": result.cost_usd,
                "tokens": result.tokens_used,
            }

        except Exception as e:
            print(f"❌ Error applying pattern {pattern.interaction_id}: {e}")
            return None

    async def _synthesize_nway_results(
        self,
        query: str,
        analysis_results: List[Dict[str, Any]],
        patterns: List[NWayPattern],
    ) -> Any:
        """Synthesize multiple N-way pattern results"""

        try:
            # Build synthesis prompt
            synthesis_prompt = f"""
<nway_synthesis>
ORIGINAL QUERY: {query}

N-WAY PATTERN ANALYSES:
"""

            for i, result in enumerate(analysis_results, 1):
                synthesis_prompt += f"""

PATTERN {i}: {result['pattern_id']}
Models Combined: {', '.join(result['models_applied'])}
Analysis:
{result['analysis_content']}
---
"""

            synthesis_prompt += f"""

SYNTHESIS TASK:
Synthesize these {len(analysis_results)} N-way pattern analyses into a coherent strategic response.

Focus on:
1. How the different pattern synergies complement each other
2. Emergent insights that arise from combining multiple N-way patterns
3. Practical strategic recommendations
4. Areas where patterns reinforce or conflict with each other

Provide executive-level strategic synthesis.
</nway_synthesis>
"""

            context = CognitiveCallContext(
                task_type="nway_synthesis",
                complexity_score=0.9,
                time_constraints="thorough",
                quality_threshold=0.95,
                cost_sensitivity="normal",
            )

            return await self.llm_client.execute_cognitive_call(
                synthesis_prompt, context
            )

        except Exception as e:
            print(f"❌ Error synthesizing N-way results: {e}")
            return None


async def test_nway_focused_engine():
    """Test the N-way focused engine"""

    print("🧪 TESTING N-WAY FOCUSED ENGINE")
    print("=" * 60)

    engine = NWayFocusedEngine()

    test_query = "B2B SaaS startup experiencing high customer churn rate of 15% monthly. Need strategic approach to improve customer retention and reduce churn."

    result = await engine.analyze_with_nway_patterns(test_query, "customer_retention")

    print("\n📊 RESULTS:")
    print(f"Status: {result['status']}")
    print(f"Patterns Applied: {result.get('patterns_applied', [])}")
    print(f"Models Count: {result.get('models_count', 0)}")
    print(f"Execution Time: {result.get('execution_time_ms', 0)}ms")
    print(f"Total Cost: ${result.get('total_cost', 0):.4f}")

    if result["status"] == "success" and "synthesis" in result and result["synthesis"]:
        print("\n🔬 SYNTHESIS PREVIEW:")
        print(result["synthesis"].content[:300] + "...")

    return result


if __name__ == "__main__":
    asyncio.run(test_nway_focused_engine())
