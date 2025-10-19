#!/usr/bin/env python3
"""
GPA/RAG Triad Evaluation Prompts - Operation Flywheel F-05
=========================================================

Specialized prompts for the seven core metrics based on TruLens evaluation framework:

GPA (Goal-Plan-Act) Framework:
1. Plan Quality - How well-structured and comprehensive is the plan?
2. Plan Adherence - How closely does the execution follow the plan?
3. Execution Efficiency - How efficiently was the plan executed?

RAG Triad Framework:
4. Context Relevance - How relevant is the retrieved context?
5. Groundedness - How well-grounded are the answers in the provided context?
6. Answer Relevance - How relevant is the answer to the original question?

Additional Core Metric:
7. Logical Consistency - How logically consistent is the reasoning?
"""

# GPA Framework Prompts
PLAN_QUALITY_PROMPT = """
You are an expert evaluator assessing the PLAN QUALITY of a strategic analysis response.

EVALUATION TASK:
Rate how well-structured and comprehensive the plan is on a scale of 1-10.

ORIGINAL QUERY:
{query}

ANALYSIS RESPONSE:
{response}

PLAN QUALITY CRITERIA (Rate 1-10):
10 = EXCEPTIONAL: Clear goal definition, comprehensive multi-step plan, well-sequenced actions, considers dependencies, includes success metrics, addresses risks proactively
8-9 = EXCELLENT: Well-structured plan with clear steps, mostly comprehensive, good sequencing, some risk consideration
6-7 = GOOD: Basic plan structure present, reasonable steps identified, adequate sequencing, minimal risk consideration
4-5 = ADEQUATE: Some planning evident but lacks structure, missing key steps or poor sequencing, little strategic thinking
2-3 = POOR: Minimal planning, disorganized approach, major gaps in logic or sequencing
1 = TERRIBLE: No coherent plan, completely disorganized, lacks strategic thinking

RESPONSE FORMAT:
Score: [1-10]
Reasoning: [2-3 sentences explaining the score based on plan structure, comprehensiveness, and strategic thinking]
Strengths: [What aspects of the plan are well-done]
Weaknesses: [What aspects need improvement]
"""

PLAN_ADHERENCE_PROMPT = """
You are an expert evaluator assessing PLAN ADHERENCE in a strategic analysis.

EVALUATION TASK:
Rate how closely the execution follows the stated or implied plan on a scale of 1-10.

ORIGINAL QUERY:
{query}

PLANNED APPROACH (if stated):
{planned_approach}

ACTUAL EXECUTION/RESPONSE:
{response}

PLAN ADHERENCE CRITERIA (Rate 1-10):
10 = PERFECT: Execution follows planned approach exactly, all stated steps completed, maintains consistent methodology throughout
8-9 = EXCELLENT: Strong adherence to plan, minor deviations that improve outcomes, consistent methodology
6-7 = GOOD: Generally follows plan, some deviations but maintains core approach, mostly consistent
4-5 = ADEQUATE: Moderate adherence, some significant deviations from plan, inconsistent methodology
2-3 = POOR: Weak adherence, major deviations without justification, inconsistent approach
1 = TERRIBLE: No adherence to plan, completely different approach than intended

RESPONSE FORMAT:
Score: [1-10]
Reasoning: [2-3 sentences explaining adherence level and any justified deviations]
Deviations: [List any significant deviations from the plan]
Impact: [Whether deviations improved or hurt the analysis quality]
"""

EXECUTION_EFFICIENCY_PROMPT = """
You are an expert evaluator assessing EXECUTION EFFICIENCY of a strategic analysis.

EVALUATION TASK:
Rate how efficiently the plan was executed on a scale of 1-10.

ORIGINAL QUERY:
{query}

RESPONSE:
{response}

EXECUTION CONTEXT:
Processing Time: {processing_time_ms}ms
Token Usage: {token_usage} tokens
LLM Calls: {llm_calls_count} calls

EXECUTION EFFICIENCY CRITERIA (Rate 1-10):
10 = OPTIMAL: Maximum value delivered with minimal resources, perfect balance of depth vs. efficiency, no waste
8-9 = EXCELLENT: High value with reasonable resource usage, good depth-efficiency balance, minimal waste
6-7 = GOOD: Adequate value for resources used, reasonable efficiency, some optimization possible
4-5 = ADEQUATE: Moderate efficiency, acceptable resource usage, some waste evident
2-3 = POOR: Low efficiency, excessive resources for value delivered, significant waste
1 = TERRIBLE: Extremely inefficient, massive resource waste, poor value delivery

RESPONSE FORMAT:
Score: [1-10]
Reasoning: [2-3 sentences explaining efficiency assessment based on value delivered vs. resources used]
Value_Delivered: [Assessment of analysis quality and usefulness]
Resource_Usage: [Assessment of computational resources used]
Optimization_Opportunities: [Areas where efficiency could be improved]
"""

# RAG Triad Framework Prompts
CONTEXT_RELEVANCE_PROMPT = """
You are an expert evaluator assessing CONTEXT RELEVANCE in a RAG-based analysis.

EVALUATION TASK:
Rate how relevant the retrieved/used context is to answering the query on a scale of 1-10.

ORIGINAL QUERY:
{query}

RETRIEVED/REFERENCED CONTEXT:
{context_used}

ANALYSIS RESPONSE:
{response}

CONTEXT RELEVANCE CRITERIA (Rate 1-10):
10 = PERFECT: All context directly relevant to query, no irrelevant information, optimal context selection
8-9 = EXCELLENT: Most context highly relevant, minimal irrelevant content, good context curation
6-7 = GOOD: Generally relevant context, some tangential information, adequate context selection
4-5 = ADEQUATE: Mixed relevance, significant irrelevant content mixed with useful information
2-3 = POOR: Mostly irrelevant context, little connection to query, poor context selection
1 = TERRIBLE: Completely irrelevant context, no connection to query, failed context retrieval

RESPONSE FORMAT:
Score: [1-10]
Reasoning: [2-3 sentences explaining context relevance assessment]
Relevant_Elements: [List the most relevant pieces of context used]
Irrelevant_Elements: [List any irrelevant context that should have been filtered out]
Missing_Context: [What relevant context might be missing]
"""

GROUNDEDNESS_PROMPT = """
You are an expert evaluator assessing GROUNDEDNESS of analysis responses.

EVALUATION TASK:
Rate how well the response is grounded in the provided context/evidence on a scale of 1-10.

ORIGINAL QUERY:
{query}

AVAILABLE CONTEXT/EVIDENCE:
{context_available}

RESPONSE TO EVALUATE:
{response}

GROUNDEDNESS CRITERIA (Rate 1-10):
10 = PERFECTLY GROUNDED: Every claim supported by context, no hallucinations, clear attribution to sources
8-9 = WELL GROUNDED: Most claims supported, minimal unsupported statements, good source attribution
6-7 = ADEQUATELY GROUNDED: Generally supported claims, some minor unsupported statements, reasonable attribution
4-5 = PARTIALLY GROUNDED: Mixed support, some unsupported claims, weak attribution
2-3 = POORLY GROUNDED: Many unsupported claims, weak connection to context, poor attribution
1 = UNGROUNDED: Most claims unsupported, hallucinations present, no clear attribution

RESPONSE FORMAT:
Score: [1-10]
Reasoning: [2-3 sentences explaining groundedness assessment]
Supported_Claims: [List key claims that are well-supported by context]
Unsupported_Claims: [List any claims that lack proper grounding]
Attribution_Quality: [Assessment of how well sources are cited/referenced]
"""

ANSWER_RELEVANCE_PROMPT = """
You are an expert evaluator assessing ANSWER RELEVANCE to the original query.

EVALUATION TASK:
Rate how relevant the answer is to the original question on a scale of 1-10.

ORIGINAL QUERY:
{query}

RESPONSE TO EVALUATE:
{response}

ANSWER RELEVANCE CRITERIA (Rate 1-10):
10 = PERFECTLY RELEVANT: Directly addresses all aspects of query, comprehensive coverage, no tangential content
8-9 = HIGHLY RELEVANT: Addresses main query aspects well, good coverage, minimal tangential content
6-7 = RELEVANT: Addresses core query, adequate coverage, some tangential content
4-5 = PARTIALLY RELEVANT: Addresses some query aspects, gaps in coverage, significant tangential content
2-3 = BARELY RELEVANT: Minimal query addressing, poor coverage, mostly tangential
1 = IRRELEVANT: Does not address query, completely off-topic, no relevance

RESPONSE FORMAT:
Score: [1-10]
Reasoning: [2-3 sentences explaining answer relevance assessment]
Query_Aspects_Addressed: [List which aspects of the original query are well-addressed]
Query_Aspects_Missing: [List which aspects of the query are not adequately addressed]
Tangential_Content: [Identify any content that doesn't directly serve the query]
"""

# Additional Core Metric
LOGICAL_CONSISTENCY_PROMPT = """
You are an expert evaluator assessing LOGICAL CONSISTENCY of strategic analysis.

EVALUATION TASK:
Rate the logical consistency and coherence of the reasoning on a scale of 1-10.

ORIGINAL QUERY:
{query}

RESPONSE TO EVALUATE:
{response}

LOGICAL CONSISTENCY CRITERIA (Rate 1-10):
10 = PERFECTLY CONSISTENT: All reasoning logically sound, no contradictions, coherent argument flow, valid conclusions
8-9 = HIGHLY CONSISTENT: Strong logical flow, minimal inconsistencies, sound reasoning, well-supported conclusions
6-7 = CONSISTENT: Generally logical, some minor inconsistencies, adequate reasoning, reasonable conclusions
4-5 = PARTIALLY CONSISTENT: Mixed logical quality, some contradictions, weak reasoning in places
2-3 = INCONSISTENT: Multiple logical flaws, contradictions present, poor reasoning quality
1 = ILLOGICAL: Severely flawed logic, major contradictions, incoherent reasoning

RESPONSE FORMAT:
Score: [1-10]
Reasoning: [2-3 sentences explaining logical consistency assessment]
Logical_Strengths: [Areas where reasoning is particularly strong]
Logical_Flaws: [Any contradictions, logical gaps, or reasoning errors]
Argument_Flow: [Assessment of how well the argument builds and connects]
"""

# Prompt Registry
GPA_RAG_PROMPTS = {
    "plan_quality": PLAN_QUALITY_PROMPT,
    "plan_adherence": PLAN_ADHERENCE_PROMPT,
    "execution_efficiency": EXECUTION_EFFICIENCY_PROMPT,
    "context_relevance": CONTEXT_RELEVANCE_PROMPT,
    "groundedness": GROUNDEDNESS_PROMPT,
    "answer_relevance": ANSWER_RELEVANCE_PROMPT,
    "logical_consistency": LOGICAL_CONSISTENCY_PROMPT
}

# Metric Categories for Organization
GPA_METRICS = ["plan_quality", "plan_adherence", "execution_efficiency"]
RAG_TRIAD_METRICS = ["context_relevance", "groundedness", "answer_relevance"]
ADDITIONAL_METRICS = ["logical_consistency"]

ALL_METRICS = GPA_METRICS + RAG_TRIAD_METRICS + ADDITIONAL_METRICS

# Export all components
__all__ = [
    "GPA_RAG_PROMPTS",
    "GPA_METRICS", 
    "RAG_TRIAD_METRICS",
    "ADDITIONAL_METRICS",
    "ALL_METRICS",
    "PLAN_QUALITY_PROMPT",
    "PLAN_ADHERENCE_PROMPT", 
    "EXECUTION_EFFICIENCY_PROMPT",
    "CONTEXT_RELEVANCE_PROMPT",
    "GROUNDEDNESS_PROMPT",
    "ANSWER_RELEVANCE_PROMPT",
    "LOGICAL_CONSISTENCY_PROMPT"
]