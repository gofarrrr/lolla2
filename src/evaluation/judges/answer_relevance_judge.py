# src/evaluation/judges/answer_relevance_judge.py
"""
Binary judge to check if the final answer is on-topic relative to the clarified problem statement.

This judge validates that the analysis stayed focused on the original question/problem
and didn't drift into tangential topics or provide irrelevant recommendations.
"""

import re
import os
import json
from typing import Dict, Any, List, Set
from collections import Counter

from . import TraceSnapshot


def evaluate_answer_relevance(trace_snapshot: TraceSnapshot) -> bool:
    """
    Judge whether the final answer is relevant to the original problem statement.
    
    Args:
        trace_snapshot: PII-safe trace data from export_traces.py
        
    Returns:
        bool: True if answer is relevant (pass),
              False if answer is off-topic (fail)
    """
    
    # Extract relevant data
    context_stream = trace_snapshot.get('context_stream', {})
    events = context_stream.get('events', [])
    final_summary = _extract_final_summary(context_stream)
    
    if not final_summary:
        return False  # No answer to evaluate
    
    # Extract original problem statement
    original_problem = _extract_original_problem(events)
    
    if not original_problem:
        return True  # No problem statement to compare against
    
    # Calculate relevance score using multiple methods
    relevance_score = _calculate_relevance_score(original_problem, final_summary, events)
    
    # Threshold for passing: at least 60% relevance
    return relevance_score >= 0.6


def _extract_final_summary(context_stream: Dict[str, Any]) -> str:
    """Extract final summary from context stream."""
    # OPERATION AWAKENING: First check new location (summary.final_report)
    summary_data = context_stream.get('summary', {})
    final_report = summary_data.get('final_report', '')
    if final_report:
        return final_report

    # Fallback: check for executive_summary or final_summary in summary object
    executive_summary = summary_data.get('executive_summary', '') or summary_data.get('final_summary', '')
    if executive_summary:
        return executive_summary

    # Fallback: Look for final report generation event (old format)
    events = context_stream.get('events', [])
    for event in reversed(events):
        if event.get('event_type') == 'final_report_generated':
            data = event.get('data', {})
            summary = data.get('executive_summary', '') or data.get('summary', '')
            if summary:
                return summary

    return ''


def _extract_original_problem(events: List[Dict[str, Any]]) -> str:
    """Extract the original problem statement from early trace events."""
    # Look for problem clarification or initial analysis events
    problem_events = [
        'problem_statement_clarified',
        'initial_problem_analysis',
        'socratic_questions_generated',
        'problem_structuring_complete',
        'structured_framework_created'
    ]
    
    problem_text = ""
    
    for event in events[:10]:  # Look in first 10 events for problem statement
        event_type = event.get('event_type', '')
        event_data = event.get('data', {})
        
        if any(problem_event in event_type for problem_event in problem_events):
            # Extract problem description from event data
            if isinstance(event_data, dict):
                problem_candidates = [
                    event_data.get('problem_statement', ''),
                    event_data.get('clarified_problem', ''),
                    event_data.get('user_question', ''),
                    event_data.get('initial_query', ''),
                    event_data.get('business_question', ''),
                    str(event_data.get('context', '')),
                ]
                
                for candidate in problem_candidates:
                    if candidate and len(candidate) > 20:  # Reasonable length
                        problem_text = candidate
                        break
        
        if problem_text:
            break
    
    # Fallback: look for any early event with substantial text content
    if not problem_text:
        for event in events[:5]:
            event_str = str(event.get('data', ''))
            if len(event_str) > 50:  # Substantial content
                problem_text = event_str
                break
    
    return problem_text


def _calculate_relevance_score(original_problem: str, final_summary: str, events: List[Dict[str, Any]]) -> float:
    """Calculate relevance score using multiple methods."""
    
    # Method 1: Keyword overlap (40% weight)
    keyword_score = _calculate_keyword_overlap(original_problem, final_summary)
    
    # Method 2: Topic consistency (30% weight)
    topic_score = _calculate_topic_consistency(original_problem, final_summary)
    
    # Method 3: Domain alignment (20% weight)
    domain_score = _calculate_domain_alignment(original_problem, final_summary)
    
    # Method 4: Question-answer alignment (10% weight)
    qa_score = _calculate_question_answer_alignment(original_problem, final_summary)
    
    # Weighted average
    total_score = (
        keyword_score * 0.4 +
        topic_score * 0.3 +
        domain_score * 0.2 +
        qa_score * 0.1
    )
    
    return min(total_score, 1.0)


def _calculate_keyword_overlap(problem: str, answer: str) -> float:
    """Calculate keyword overlap between problem and answer."""
    # Extract keywords from both texts
    problem_keywords = _extract_keywords(problem)
    answer_keywords = _extract_keywords(answer)
    
    if not problem_keywords:
        return 1.0  # No keywords to compare
    
    # Calculate overlap
    overlap = len(problem_keywords.intersection(answer_keywords))
    total_problem_keywords = len(problem_keywords)
    
    return overlap / total_problem_keywords


def _extract_keywords(text: str) -> Set[str]:
    """Extract meaningful keywords from text."""
    # Convert to lowercase and extract words
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Filter out common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after',
        'above', 'below', 'between', 'among', 'this', 'that', 'these', 'those', 'i', 'me',
        'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
        'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
        'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
        'what', 'which', 'who', 'whom', 'whose', 'this', 'that', 'these', 'those', 'am',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
        'do', 'does', 'did', 'doing', 'will', 'would', 'could', 'should', 'may', 'might',
        'must', 'can', 'need', 'how', 'when', 'where', 'why', 'what', 'there', 'here'
    }
    
    # Keep only meaningful words (length > 2, not stop words)
    keywords = {word for word in words if len(word) > 2 and word not in stop_words}
    
    return keywords


def _calculate_topic_consistency(problem: str, answer: str) -> float:
    """Calculate consistency of main topics between problem and answer."""
    # Define topic categories
    topic_keywords = {
        'business_strategy': ['strategy', 'business', 'market', 'competitive', 'revenue', 'growth', 'expansion', 'planning'],
        'technology': ['technology', 'software', 'system', 'platform', 'infrastructure', 'api', 'database', 'cloud'],
        'operations': ['operations', 'process', 'workflow', 'efficiency', 'optimization', 'automation', 'management'],
        'finance': ['financial', 'budget', 'cost', 'investment', 'roi', 'pricing', 'accounting', 'funding'],
        'marketing': ['marketing', 'branding', 'advertising', 'campaign', 'customer', 'acquisition', 'retention'],
        'product': ['product', 'feature', 'development', 'design', 'user', 'experience', 'interface', 'requirements'],
        'security': ['security', 'privacy', 'compliance', 'risk', 'vulnerability', 'protection', 'audit'],
        'analytics': ['analytics', 'data', 'metrics', 'reporting', 'insights', 'analysis', 'measurement'],
        'hr_people': ['human', 'people', 'team', 'hiring', 'training', 'culture', 'performance', 'talent'],
        'legal': ['legal', 'compliance', 'regulation', 'contract', 'agreement', 'policy', 'governance']
    }
    
    # Find topics in problem and answer
    problem_topics = _identify_topics(problem, topic_keywords)
    answer_topics = _identify_topics(answer, topic_keywords)
    
    if not problem_topics:
        return 1.0  # No topics to compare
    
    # Calculate topic overlap
    overlap = len(problem_topics.intersection(answer_topics))
    return overlap / len(problem_topics)


def _identify_topics(text: str, topic_keywords: Dict[str, List[str]]) -> Set[str]:
    """Identify topics present in text."""
    text_lower = text.lower()
    topics = set()
    
    for topic, keywords in topic_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            topics.add(topic)
    
    return topics


def _calculate_domain_alignment(problem: str, answer: str) -> float:
    """Calculate alignment of business/technical domains."""
    # Domain indicators
    domains = {
        'technical': ['api', 'database', 'server', 'code', 'algorithm', 'architecture', 'development'],
        'business': ['revenue', 'profit', 'market', 'customer', 'sales', 'business', 'strategy'],
        'operational': ['process', 'workflow', 'efficiency', 'operations', 'management', 'organization'],
        'analytical': ['data', 'analytics', 'metrics', 'analysis', 'insights', 'reporting', 'measurement'],
        'financial': ['budget', 'cost', 'investment', 'financial', 'money', 'pricing', 'economic'],
        'regulatory': ['compliance', 'regulation', 'legal', 'policy', 'governance', 'audit', 'standards']
    }
    
    problem_domains = _identify_domains(problem, domains)
    answer_domains = _identify_domains(answer, domains)
    
    if not problem_domains:
        return 1.0  # No domains to compare
    
    # Calculate domain overlap
    overlap = len(problem_domains.intersection(answer_domains))
    return overlap / len(problem_domains)


def _identify_domains(text: str, domains: Dict[str, List[str]]) -> Set[str]:
    """Identify business/technical domains in text."""
    text_lower = text.lower()
    identified_domains = set()
    
    for domain, keywords in domains.items():
        if any(keyword in text_lower for keyword in keywords):
            identified_domains.add(domain)
    
    return identified_domains


def _calculate_question_answer_alignment(problem: str, answer: str) -> float:
    """Calculate how well the answer addresses the question type."""
    problem_lower = problem.lower()
    answer_lower = answer.lower()
    
    # Identify question types and expected answer patterns
    question_patterns = {
        'how': ['approach', 'method', 'step', 'process', 'way', 'solution', 'implement'],
        'what': ['definition', 'description', 'explanation', 'identify', 'list', 'options'],
        'why': ['reason', 'because', 'cause', 'factor', 'explanation', 'rationale'],
        'when': ['time', 'schedule', 'timeline', 'deadline', 'phase', 'period'],
        'where': ['location', 'place', 'platform', 'system', 'environment', 'context'],
        'which': ['option', 'choice', 'alternative', 'selection', 'recommendation'],
        'should': ['recommend', 'suggest', 'advise', 'best', 'optimal', 'preferred'],
        'can': ['possible', 'feasible', 'capability', 'ability', 'option', 'potential'],
        'will': ['prediction', 'forecast', 'future', 'expect', 'anticipate', 'outcome']
    }
    
    # Find question types in problem
    question_types = []
    for q_type, patterns in question_patterns.items():
        if q_type in problem_lower:
            question_types.append(q_type)
    
    if not question_types:
        return 1.0  # No specific question pattern identified
    
    # Check if answer contains appropriate response patterns
    alignment_score = 0
    for q_type in question_types:
        expected_patterns = question_patterns[q_type]
        if any(pattern in answer_lower for pattern in expected_patterns):
            alignment_score += 1
    
    return alignment_score / len(question_types)


# Optional: LLM-based relevance check as fallback for edge cases
def _llm_relevance_check(problem: str, answer: str) -> bool:
    """
    Optional LLM-based relevance check for complex semantic alignment.
    This would be called only for borderline cases where heuristic methods are uncertain.
    """
    # This is a placeholder for potential LLM integration
    # Would use a lightweight model with a focused prompt like:
    # "Given the problem: '{problem}' and answer: '{answer}', 
    #  is the answer directly relevant to the problem? Yes/No"
    
    # For now, return True to not block evaluation
    return True