# src/evaluation/judges/observability_coverage_judge.py
"""
Binary judge to check if a trace contains all required event types for the stages it executed.

This judge validates that the analysis pipeline properly logged the expected observability
events for each stage, ensuring complete transparency and auditability.
"""

from typing import Dict, Any, List, Set

from . import TraceSnapshot


def evaluate_observability_coverage(trace_snapshot: TraceSnapshot) -> bool:
    """
    Judge whether the trace contains proper observability coverage for executed stages.
    
    Args:
        trace_snapshot: PII-safe trace data from export_traces.py
        
    Returns:
        bool: True if observability coverage is adequate (pass),
              False if required events are missing (fail)
    """
    
    # Extract data from trace snapshot
    context_stream = trace_snapshot.get('context_stream', {})
    events = context_stream.get('events', [])
    executed_stages = _extract_executed_stages(trace_snapshot)
    
    if not executed_stages:
        return True  # No stages to validate
    
    # Calculate coverage score
    coverage_score = _calculate_coverage_score(events, executed_stages)
    
    # Threshold for passing: at least 80% coverage
    return coverage_score >= 0.8


def _extract_executed_stages(trace_snapshot: TraceSnapshot) -> Set[str]:
    """Extract executed stages from the trace snapshot stage_list."""
    stage_list = trace_snapshot.get('stage_list', '')
    
    if not stage_list:
        return set()
    
    # Parse comma-separated stage list
    stages = {stage.strip() for stage in stage_list.split(',') if stage.strip()}
    return stages


def _calculate_coverage_score(events: List[Dict[str, Any]], executed_stages: Set[str]) -> float:
    """Calculate observability coverage score for executed stages."""
    
    if not executed_stages:
        return 1.0  # No stages to validate
    
    total_required_events = 0
    found_required_events = 0
    
    for stage in executed_stages:
        required_events = _get_required_events_for_stage(stage)
        total_required_events += len(required_events)
        
        # Check which required events are present
        present_events = _get_present_event_types(events)
        stage_coverage = len(required_events.intersection(present_events))
        found_required_events += stage_coverage
    
    if total_required_events == 0:
        return 1.0  # No requirements defined
    
    return found_required_events / total_required_events


def _get_required_events_for_stage(stage: str) -> Set[str]:
    """Get required observability events for a given pipeline stage."""
    
    # Policy dictionary mapping stages to required event types
    # Based on the stage event types from export_traces.py
    stage_requirements = {
        'SOCRATIC': {
            'socratic_questions_generated',
            'problem_clarification_complete',
            'context_analysis_complete'
        },
        
        'PROBLEM_STRUCTURING': {
            'structured_framework_created',
            'problem_decomposition_complete',
            'analysis_approach_defined'
        },
        
        'SELECTION': {
            'consultant_selection_complete', 
            'cognitive_chemistry_applied',
            'expertise_matching_complete'
        },
        
        'PARALLEL_ANALYSIS': {
            'consultant_analysis_complete',
            'research_synthesis_complete',
            'insights_aggregation_complete'
        },
        
        'DEVILS_ADVOCATE': {
            'devils_advocate_complete',
            'critical_analysis_complete',
            'assumption_challenge_complete'
        },
        
        'SENIOR_ADVISOR': {
            'senior_advisor_complete',
            'strategic_review_complete', 
            'final_recommendations_generated'
        }
    }
    
    return stage_requirements.get(stage, set())


def _get_present_event_types(events: List[Dict[str, Any]]) -> Set[str]:
    """Extract all event types present in the trace events."""
    event_types = set()
    
    for event in events:
        event_type = event.get('event_type', '')
        if event_type:
            event_types.add(event_type)
    
    return event_types


def _validate_event_completeness(events: List[Dict[str, Any]]) -> float:
    """Validate that events contain sufficient data and aren't just stubs."""
    
    if not events:
        return 0.0
    
    complete_events = 0
    
    for event in events:
        if _is_event_complete(event):
            complete_events += 1
    
    return complete_events / len(events)


def _is_event_complete(event: Dict[str, Any]) -> bool:
    """Check if an event contains sufficient data to be considered complete."""
    
    # Basic required fields
    required_fields = ['event_type', 'timestamp']
    for field in required_fields:
        if not event.get(field):
            return False
    
    # Check for substantial data content
    event_data = event.get('data', {})
    
    # Event should have meaningful data
    if not event_data:
        return False
    
    # Count non-empty data fields
    meaningful_fields = 0
    for key, value in event_data.items():
        if value and str(value).strip():
            if len(str(value)) > 10:  # Substantial content
                meaningful_fields += 1
    
    # Should have at least 1 meaningful data field
    return meaningful_fields >= 1


def _check_event_sequence_validity(events: List[Dict[str, Any]], executed_stages: Set[str]) -> float:
    """Check if events follow a logical sequence for the executed stages."""
    
    # Define expected event ordering
    stage_order = [
        'SOCRATIC',
        'PROBLEM_STRUCTURING', 
        'SELECTION',
        'PARALLEL_ANALYSIS',
        'DEVILS_ADVOCATE',
        'SENIOR_ADVISOR'
    ]
    
    # Extract stages that were executed in order
    executed_stages_list = [stage for stage in stage_order if stage in executed_stages]
    
    if len(executed_stages_list) <= 1:
        return 1.0  # Single stage or no stages to sequence
    
    # Map events to stages
    event_stage_mapping = {}
    for event in events:
        event_type = event.get('event_type', '')
        for stage in executed_stages:
            required_events = _get_required_events_for_stage(stage)
            if event_type in required_events:
                event_stage_mapping[event_type] = stage
                break
    
    # Check if events appear in logical order
    last_stage_index = -1
    sequence_violations = 0
    
    for event in events:
        event_type = event.get('event_type', '')
        if event_type in event_stage_mapping:
            stage = event_stage_mapping[event_type]
            if stage in executed_stages_list:
                stage_index = executed_stages_list.index(stage)
                if stage_index < last_stage_index:
                    sequence_violations += 1
                last_stage_index = max(last_stage_index, stage_index)
    
    # Calculate sequence validity score
    if len(event_stage_mapping) == 0:
        return 1.0
    
    sequence_score = 1 - (sequence_violations / len(event_stage_mapping))
    return max(sequence_score, 0.0)


def _check_critical_events_present(events: List[Dict[str, Any]]) -> float:
    """Check if critical observability events are present regardless of stage."""
    
    # Critical events that should be present in most traces
    critical_events = {
        'analysis_started',
        'analysis_complete', 
        'final_report_generated',
        'context_stream_initialized'
    }
    
    present_events = _get_present_event_types(events)
    critical_events_found = len(critical_events.intersection(present_events))
    
    if len(critical_events) == 0:
        return 1.0
    
    return critical_events_found / len(critical_events)


def _validate_error_event_handling(events: List[Dict[str, Any]]) -> bool:
    """Validate that error events, if present, are properly structured."""
    
    error_events = []
    for event in events:
        event_type = event.get('event_type', '').lower()
        if 'error' in event_type or 'failure' in event_type or 'exception' in event_type:
            error_events.append(event)
    
    if not error_events:
        return True  # No errors to validate
    
    # Check that error events have proper structure
    properly_structured_errors = 0
    
    for error_event in error_events:
        event_data = error_event.get('data', {})
        
        # Error events should have error details
        has_error_details = any(
            key in event_data for key in ['error_message', 'error_type', 'exception', 'failure_reason']
        )
        
        if has_error_details:
            properly_structured_errors += 1
    
    return properly_structured_errors / len(error_events) >= 0.8  # 80% threshold