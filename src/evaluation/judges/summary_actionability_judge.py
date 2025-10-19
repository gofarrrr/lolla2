# src/evaluation/judges/summary_actionability_judge.py
"""
Binary judge to check if the executive summary contains clear, actionable next steps.

This judge validates that the summary provides concrete, implementable recommendations
rather than vague insights or purely analytical observations.
"""

import re
from typing import Dict, Any, List
from collections import Counter

from . import TraceSnapshot


def evaluate_summary_actionability(trace_snapshot: TraceSnapshot) -> bool:
    """
    Judge whether the executive summary contains actionable next steps.
    
    Args:
        trace_snapshot: PII-safe trace data from export_traces.py
        
    Returns:
        bool: True if summary is actionable (pass),
              False if summary lacks clear next steps (fail)
    """
    
    # Extract final summary
    context_stream = trace_snapshot.get('context_stream', {})
    final_summary = _extract_final_summary(context_stream)
    
    if not final_summary:
        return False  # No summary to evaluate
    
    # Calculate actionability score
    actionability_score = _calculate_actionability_score(final_summary)
    
    # Threshold for passing: at least 50% actionability
    return actionability_score >= 0.5


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


def _calculate_actionability_score(summary: str) -> float:
    """Calculate actionability score using multiple criteria."""
    
    # Method 1: Count imperative verbs (40% weight)
    imperative_score = _count_imperative_verbs(summary)
    
    # Method 2: Identify specific actions and recommendations (30% weight)
    specific_actions_score = _identify_specific_actions(summary)
    
    # Method 3: Check for concrete details and metrics (20% weight)
    concrete_details_score = _check_concrete_details(summary)
    
    # Method 4: Assess timeline and ownership clarity (10% weight)
    timeline_ownership_score = _assess_timeline_ownership(summary)
    
    # Weighted average
    total_score = (
        imperative_score * 0.4 +
        specific_actions_score * 0.3 +
        concrete_details_score * 0.2 +
        timeline_ownership_score * 0.1
    )
    
    return min(total_score, 1.0)


def _count_imperative_verbs(summary: str) -> float:
    """Count actionable imperative verbs in the summary."""
    
    # Strong action verbs that indicate clear next steps
    imperative_verbs = [
        # Implementation actions
        'implement', 'deploy', 'install', 'configure', 'setup', 'build',
        'develop', 'create', 'establish', 'launch', 'execute', 'rollout',
        
        # Analysis and research actions
        'analyze', 'investigate', 'research', 'assess', 'evaluate', 'review',
        'examine', 'study', 'audit', 'benchmark', 'measure', 'monitor',
        
        # Communication actions
        'communicate', 'notify', 'inform', 'present', 'report', 'discuss',
        'meet', 'collaborate', 'coordinate', 'align', 'engage', 'consult',
        
        # Process actions
        'optimize', 'improve', 'enhance', 'streamline', 'automate', 'upgrade',
        'migrate', 'refactor', 'restructure', 'redesign', 'standardize',
        
        # Decision actions
        'decide', 'approve', 'select', 'choose', 'prioritize', 'focus',
        'allocate', 'assign', 'delegate', 'schedule', 'plan', 'organize',
        
        # Testing and validation actions
        'test', 'validate', 'verify', 'confirm', 'pilot', 'prototype',
        'trial', 'experiment', 'iterate', 'adjust', 'refine',
        
        # Documentation actions
        'document', 'record', 'track', 'log', 'capture', 'update',
        'maintain', 'publish', 'share', 'distribute'
    ]
    
    # Count occurrences
    summary_lower = summary.lower()
    verb_count = 0
    total_sentences = len(re.split(r'[.!?]+', summary))
    
    for verb in imperative_verbs:
        # Look for verb patterns: "should implement", "need to analyze", "implement the"
        patterns = [
            rf'\b{verb}\b',                    # Direct verb usage
            rf'\bshould\s+{verb}\b',          # Should + verb
            rf'\bneed\s+to\s+{verb}\b',       # Need to + verb
            rf'\bmust\s+{verb}\b',            # Must + verb
            rf'\brecommend\s+to\s+{verb}\b',  # Recommend to + verb
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, summary_lower)
            verb_count += len(matches)
    
    # Normalize by number of sentences
    if total_sentences == 0:
        return 0.0
    
    # Score based on ratio of actionable verbs to sentences
    ratio = verb_count / total_sentences
    return min(ratio, 1.0)  # Cap at 1.0


def _identify_specific_actions(summary: str) -> float:
    """Identify specific, concrete actions vs. vague recommendations."""
    
    sentences = re.split(r'[.!?]+', summary)
    actionable_sentences = 0
    total_sentences = len([s for s in sentences if s.strip()])
    
    if total_sentences == 0:
        return 0.0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # Check if sentence contains specific action indicators
        if _is_specific_action(sentence):
            actionable_sentences += 1
    
    return actionable_sentences / total_sentences


def _is_specific_action(sentence: str) -> bool:
    """Determine if a sentence contains a specific, actionable recommendation."""
    sentence_lower = sentence.lower()
    
    # Patterns that indicate specific actions
    specific_action_patterns = [
        # Tool/technology specific
        r'\busing\s+[\w\s]+\btool\b',
        r'\bimplement\s+[\w\s]+\bsystem\b',
        r'\bupgrade\s+to\s+[\w\s]+',
        r'\bmigrate\s+from\s+[\w\s]+\bto\s+[\w\s]+',
        
        # Process specific
        r'\bestablish\s+[\w\s]+\bprocess\b',
        r'\bcreate\s+[\w\s]+\bworkflow\b',
        r'\bimplement\s+[\w\s]+\bprocedure\b',
        
        # Metric/measurement specific
        r'\btrack\s+[\w\s]+\bmetric\b',
        r'\bmeasure\s+[\w\s]+\bperformance\b',
        r'\bmonitor\s+[\w\s]+\bindicator\b',
        
        # Team/resource specific
        r'\bassign\s+[\w\s]+\bteam\b',
        r'\bhire\s+[\w\s]+\bspecialist\b',
        r'\ballocate\s+[\w\s]+\bresources?\b',
        
        # Timeline specific
        r'\bwithin\s+\d+\s+(?:days?|weeks?|months?)\b',
        r'\bby\s+[\w\s]+\b(?:quarter|month|year)\b',
        r'\bin\s+(?:q\d|phase\s+\d|\d+\s+months?)\b',
    ]
    
    # Check for specific action patterns
    for pattern in specific_action_patterns:
        if re.search(pattern, sentence_lower):
            return True
    
    # Check for concrete nouns (indicates specificity)
    concrete_indicators = [
        'dashboard', 'system', 'platform', 'tool', 'framework', 'process',
        'workflow', 'procedure', 'protocol', 'standard', 'guideline',
        'metric', 'kpi', 'report', 'analysis', 'assessment', 'audit',
        'team', 'specialist', 'consultant', 'vendor', 'partner',
        'budget', 'investment', 'cost', 'resource', 'training', 'certification'
    ]
    
    # Exclude vague language
    vague_indicators = [
        'should consider', 'might want to', 'could potentially', 'may benefit',
        'generally', 'overall', 'typically', 'usually', 'often', 'sometimes',
        'various', 'several', 'multiple', 'different', 'appropriate', 'suitable'
    ]
    
    # Check for vague language (disqualifies as specific action)
    for vague in vague_indicators:
        if vague in sentence_lower:
            return False
    
    # Check for concrete indicators
    concrete_count = sum(1 for indicator in concrete_indicators if indicator in sentence_lower)
    
    return concrete_count >= 1


def _check_concrete_details(summary: str) -> float:
    """Check for concrete details like numbers, percentages, timeframes."""
    
    detail_patterns = [
        # Numerical details
        r'\d+%',                          # Percentages
        r'\$[\d,]+(?:k|m|b)?',           # Money amounts
        r'\d+\s+(?:days?|weeks?|months?|years?)',  # Time periods
        r'\d+\s+(?:people|employees?|users?|customers?)',  # Quantities
        r'\d+\s+(?:hours?|minutes?)\s+per\s+\w+',  # Rates
        
        # Specific metrics
        r'\d+\.\d+\s+(?:seconds?|minutes?)',  # Performance metrics
        r'\d+x\s+(?:improvement|increase|decrease)',  # Multipliers
        r'reduce\s+by\s+\d+%',           # Reduction targets
        r'increase\s+by\s+\d+%',         # Growth targets
        
        # Specific tools/technologies
        r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # Proper nouns (tools/products)
        r'\bAPI\b|\bSQL\b|\bAWS\b|\bSaaS\b',  # Technical acronyms
        
        # Specific roles/departments
        r'\b(?:CTO|CEO|VP|Director|Manager)\b',  # Specific roles
        r'\b(?:Engineering|Marketing|Sales|Operations|Finance)\s+(?:team|department)\b',
    ]
    
    summary_lower = summary.lower()
    detail_count = 0
    
    for pattern in detail_patterns:
        matches = re.findall(pattern, summary, re.IGNORECASE)
        detail_count += len(matches)
    
    # Normalize by summary length (words)
    word_count = len(summary.split())
    if word_count == 0:
        return 0.0
    
    # Score based on detail density
    detail_density = detail_count / word_count * 100  # Details per 100 words
    return min(detail_density / 10, 1.0)  # Normalize to 0-1 scale


def _assess_timeline_ownership(summary: str) -> float:
    """Assess clarity of timelines and ownership/responsibility."""
    
    timeline_score = 0.0
    ownership_score = 0.0
    
    # Timeline indicators
    timeline_patterns = [
        r'\bwithin\s+\d+\s+(?:days?|weeks?|months?)\b',
        r'\bby\s+(?:end\s+of\s+)?(?:q\d|quarter|\w+\s+\d{4})\b',
        r'\bin\s+(?:phase\s+\d+|\d+\s+months?|next\s+\w+)\b',
        r'\bimmediately\b|\basap\b|\burgent\b',
        r'\bshort[- ]?term\b|\bmid[- ]?term\b|\blong[- ]?term\b',
    ]
    
    # Ownership indicators
    ownership_patterns = [
        r'\b(?:led\s+by|assigned\s+to|responsibility\s+of)\s+[\w\s]+\b',
        r'\b[\w\s]+\s+(?:team|department|group)\s+(?:will|should|must)\b',
        r'\b(?:CTO|CEO|VP|Director|Manager|lead|owner)\s+(?:will|should|must)\b',
        r'\bassign\s+to\s+[\w\s]+\b',
        r'\b[\w\s]+\s+(?:accountable|responsible)\s+for\b',
    ]
    
    summary_lower = summary.lower()
    
    # Check for timeline clarity
    timeline_matches = 0
    for pattern in timeline_patterns:
        if re.search(pattern, summary_lower):
            timeline_matches += 1
    
    timeline_score = min(timeline_matches / 3, 1.0)  # Normalize to 0-1
    
    # Check for ownership clarity
    ownership_matches = 0
    for pattern in ownership_patterns:
        if re.search(pattern, summary_lower):
            ownership_matches += 1
    
    ownership_score = min(ownership_matches / 2, 1.0)  # Normalize to 0-1
    
    # Average timeline and ownership scores
    return (timeline_score + ownership_score) / 2