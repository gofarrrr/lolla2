# src/evaluation/judges/needs_handoff_judge.py
"""
Binary judge to determine if a trace should have been escalated to human-in-the-loop but wasn't.

This judge detects scenarios where the analysis encountered critical issues, ambiguity, 
or complexity that warranted human oversight but was handled purely by automation.
"""

import re
from typing import Dict, Any, List

from . import TraceSnapshot


def evaluate_needs_handoff(trace_snapshot: TraceSnapshot) -> bool:
    """
    Judge whether this trace should have been escalated to a human but wasn't.
    
    Args:
        trace_snapshot: PII-safe trace data from export_traces.py
        
    Returns:
        bool: True if trace should NOT have been handed off (pass),
              False if trace SHOULD have been handed off (fail)
    """
    
    # Extract relevant data from trace snapshot
    context_stream = trace_snapshot.get('context_stream', {})
    events = context_stream.get('events', [])
    final_summary = _extract_final_summary(context_stream)
    error_flags = trace_snapshot.get('error_flags', False)
    
    # Critical indicators that should trigger handoff
    handoff_indicators = []
    
    # 1. Check for error events that indicate system failures
    if error_flags or _has_critical_errors(events):
        handoff_indicators.append("critical_errors")
    
    # 2. Check for high uncertainty markers in events
    uncertainty_score = _calculate_uncertainty_score(events)
    if uncertainty_score > 0.7:  # High uncertainty threshold
        handoff_indicators.append("high_uncertainty")
    
    # 3. Check for incomplete analysis stages
    if _has_incomplete_stages(events):
        handoff_indicators.append("incomplete_analysis")
    
    # 4. Check for conflicting recommendations in final summary
    if _has_conflicting_recommendations(final_summary):
        handoff_indicators.append("conflicting_recommendations")
    
    # 5. Check for sensitive domain markers (legal, medical, financial)
    if _contains_sensitive_domain_markers(events, final_summary):
        handoff_indicators.append("sensitive_domain")
    
    # 6. Check for high-stakes decision markers
    if _contains_high_stakes_markers(events, final_summary):
        handoff_indicators.append("high_stakes")
    
    # Decision logic: If 2+ handoff indicators present, should have been handed off
    should_have_handoff = len(handoff_indicators) >= 2
    
    # Return True if trace appropriately handled (no handoff needed)
    # Return False if trace should have been handed off (failure)
    return not should_have_handoff


def _extract_final_summary(context_stream: Dict[str, Any]) -> str:
    """Extract final summary from context stream events or summary section."""
    events = context_stream.get('events', [])
    
    # Look for final report generation event
    for event in reversed(events):
        if event.get('event_type') == 'final_report_generated':
            data = event.get('data', {})
            summary = data.get('executive_summary', '') or data.get('summary', '')
            if summary:
                return summary
    
    # Fallback to summary section
    summary_data = context_stream.get('summary', {})
    return summary_data.get('executive_summary', '') or summary_data.get('final_summary', '')


def _has_critical_errors(events: List[Dict[str, Any]]) -> bool:
    """Check for critical system errors in events."""
    critical_error_patterns = [
        'llm_failure', 'timeout', 'api_error', 'connection_failed',
        'rate_limit_exceeded', 'authentication_failed', 'data_corruption'
    ]
    
    for event in events:
        event_type = event.get('event_type', '').lower()
        for pattern in critical_error_patterns:
            if pattern in event_type:
                return True
    
    return False


def _calculate_uncertainty_score(events: List[Dict[str, Any]]) -> float:
    """Calculate uncertainty score based on event data."""
    uncertainty_markers = 0
    total_events = len(events)
    
    if total_events == 0:
        return 1.0  # No events = high uncertainty
    
    uncertainty_keywords = [
        'uncertain', 'unclear', 'ambiguous', 'insufficient_data',
        'low_confidence', 'requires_clarification', 'incomplete',
        'conflicting', 'contradictory', 'assumption'
    ]
    
    for event in events:
        event_text = str(event).lower()
        for keyword in uncertainty_keywords:
            if keyword in event_text:
                uncertainty_markers += 1
                break  # Count each event only once
    
    return min(uncertainty_markers / total_events, 1.0)


def _has_incomplete_stages(events: List[Dict[str, Any]]) -> bool:
    """Check if analysis stages were started but not completed."""
    stage_starts = set()
    stage_completions = set()
    
    start_patterns = ['_started', '_initiated', '_begin']
    completion_patterns = ['_complete', '_finished', '_generated']
    
    for event in events:
        event_type = event.get('event_type', '')
        
        # Check for stage start events
        for pattern in start_patterns:
            if pattern in event_type:
                stage_name = event_type.replace(pattern, '')
                stage_starts.add(stage_name)
        
        # Check for stage completion events
        for pattern in completion_patterns:
            if pattern in event_type:
                stage_name = event_type.replace(pattern, '')
                stage_completions.add(stage_name)
    
    # If more stages started than completed, analysis is incomplete
    return len(stage_starts) > len(stage_completions)


def _has_conflicting_recommendations(final_summary: str) -> bool:
    """Check for conflicting or contradictory recommendations in summary."""
    if not final_summary:
        return False
    
    # Look for contradiction markers
    contradiction_patterns = [
        r'\b(however|but|although|despite|nevertheless|on the other hand)\b',
        r'\b(conflicting|contradictory|opposing|alternative)\b',
        r'\b(either .* or|both .* and)\b'
    ]
    
    summary_lower = final_summary.lower()
    contradiction_count = 0
    
    for pattern in contradiction_patterns:
        if re.search(pattern, summary_lower):
            contradiction_count += 1
    
    # High contradiction count suggests conflicting recommendations
    return contradiction_count >= 2


def _contains_sensitive_domain_markers(events: List[Dict[str, Any]], final_summary: str) -> bool:
    """Check for sensitive domain content that typically requires human oversight."""
    sensitive_keywords = [
        # Legal domain
        'legal', 'lawsuit', 'litigation', 'compliance', 'regulation',
        'liability', 'contract', 'agreement', 'intellectual property',
        
        # Medical domain
        'medical', 'healthcare', 'patient', 'diagnosis', 'treatment',
        'pharmaceutical', 'clinical', 'health', 'safety',
        
        # Financial domain
        'financial', 'investment', 'trading', 'securities', 'banking',
        'credit', 'loan', 'mortgage', 'tax', 'audit',
        
        # HR/Personnel
        'hiring', 'firing', 'employment', 'discrimination', 'harassment',
        'performance review', 'disciplinary', 'compensation'
    ]
    
    # Check events
    all_text = ' '.join([str(event) for event in events]).lower()
    all_text += ' ' + final_summary.lower()
    
    for keyword in sensitive_keywords:
        if keyword in all_text:
            return True
    
    return False


def _contains_high_stakes_markers(events: List[Dict[str, Any]], final_summary: str) -> bool:
    """Check for high-stakes decision markers."""
    high_stakes_patterns = [
        # Financial impact
        r'\$[\d,]+(?:k|m|b|million|billion)',
        r'\b(?:budget|cost|revenue|profit|loss).{0,20}\$',
        
        # Scale indicators  
        r'\b(?:enterprise|organization|company).wide\b',
        r'\b(?:strategic|critical|mission.critical)\b',
        r'\b(?:thousands?|millions?|billions?).{0,20}(?:users?|customers?|employees?)\b',
        
        # Risk indicators
        r'\b(?:risk|threat|danger|vulnerability).{0,20}(?:high|critical|severe)\b',
        r'\b(?:security|privacy|data).{0,20}(?:breach|violation|exposure)\b',
        
        # Timeline pressure
        r'\b(?:urgent|immediate|asap|deadline|time.sensitive)\b',
        r'\b(?:today|tomorrow|this week|emergency)\b'
    ]
    
    all_text = ' '.join([str(event) for event in events])
    all_text += ' ' + final_summary
    all_text = all_text.lower()
    
    for pattern in high_stakes_patterns:
        if re.search(pattern, all_text):
            return True
    
    return False