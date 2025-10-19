# src/evaluation/runner/run_binary_judges.py
"""
Binary judge runner utility for automated evaluation of trace snapshots.

This module provides the run_binary_judges function that takes a list of trace snapshots
and runs all five binary judges against each one, returning aggregated failure rates.
"""

import logging
from typing import List, Dict, Any
from ..judges.needs_handoff_judge import evaluate_needs_handoff
from ..judges.groundedness_judge import evaluate_groundedness
from ..judges.answer_relevance_judge import evaluate_answer_relevance
from ..judges.summary_actionability_judge import evaluate_summary_actionability
from ..judges.observability_coverage_judge import evaluate_observability_coverage

logger = logging.getLogger(__name__)


def run_binary_judges(trace_snapshots: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Run all five binary judges against a list of trace snapshots.
    
    Args:
        trace_snapshots: List of PII-safe trace snapshots from export_traces.py format
        
    Returns:
        Dict mapping judge names to failure rates (0.0 to 1.0)
        Example: {"needs_handoff": 0.05, "groundedness": 0.12, ...}
    """
    
    if not trace_snapshots:
        logger.warning("No trace snapshots provided to run_binary_judges")
        return _get_default_failure_rates()
    
    # Initialize counters for each judge
    judge_results = {
        'needs_handoff': {'passed': 0, 'failed': 0, 'errors': 0},
        'groundedness': {'passed': 0, 'failed': 0, 'errors': 0},
        'answer_relevance': {'passed': 0, 'failed': 0, 'errors': 0},
        'summary_actionability': {'passed': 0, 'failed': 0, 'errors': 0},
        'observability_coverage': {'passed': 0, 'failed': 0, 'errors': 0}
    }
    
    # Judge function mapping
    judge_functions = {
        'needs_handoff': evaluate_needs_handoff,
        'groundedness': evaluate_groundedness,
        'answer_relevance': evaluate_answer_relevance,
        'summary_actionability': evaluate_summary_actionability,
        'observability_coverage': evaluate_observability_coverage
    }
    
    logger.info(f"Running binary judges on {len(trace_snapshots)} trace snapshots")
    
    # Process each trace snapshot
    for i, trace_snapshot in enumerate(trace_snapshots):
        trace_id = trace_snapshot.get('trace_id', f'trace_{i}')
        
        # Run each judge
        for judge_name, judge_function in judge_functions.items():
            try:
                result = judge_function(trace_snapshot)
                
                if result:
                    judge_results[judge_name]['passed'] += 1
                else:
                    judge_results[judge_name]['failed'] += 1
                    
            except Exception as e:
                logger.error(f"Judge {judge_name} failed on trace {trace_id}: {e}")
                judge_results[judge_name]['errors'] += 1
    
    # Calculate failure rates
    failure_rates = {}
    for judge_name, results in judge_results.items():
        total_evaluated = results['passed'] + results['failed']
        
        if total_evaluated == 0:
            # If no traces were successfully evaluated, report high failure rate
            failure_rate = 1.0
        else:
            # Failure rate = failed / (passed + failed)
            # Note: Errors are tracked separately but don't count in failure rate calculation
            failure_rate = results['failed'] / total_evaluated
        
        failure_rates[judge_name] = failure_rate
        
        logger.info(f"Judge {judge_name}: {results['passed']} passed, "
                   f"{results['failed']} failed, {results['errors']} errors "
                   f"(failure rate: {failure_rate:.3f})")
    
    return failure_rates


def _get_default_failure_rates() -> Dict[str, float]:
    """Return default failure rates when no traces are available."""
    return {
        'needs_handoff': 0.0,
        'groundedness': 0.0,
        'answer_relevance': 0.0,
        'summary_actionability': 0.0,
        'observability_coverage': 0.0
    }


def run_single_judge(judge_name: str, trace_snapshots: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Run a single judge against trace snapshots and return detailed results.
    
    Args:
        judge_name: Name of the judge to run
        trace_snapshots: List of trace snapshots
        
    Returns:
        Dict with detailed results including individual trace results
    """
    
    judge_functions = {
        'needs_handoff': evaluate_needs_handoff,
        'groundedness': evaluate_groundedness,
        'answer_relevance': evaluate_answer_relevance,
        'summary_actionability': evaluate_summary_actionability,
        'observability_coverage': evaluate_observability_coverage
    }
    
    if judge_name not in judge_functions:
        raise ValueError(f"Unknown judge: {judge_name}")
    
    judge_function = judge_functions[judge_name]
    results = {
        'judge_name': judge_name,
        'total_traces': len(trace_snapshots),
        'passed': 0,
        'failed': 0,
        'errors': 0,
        'trace_results': []
    }
    
    for i, trace_snapshot in enumerate(trace_snapshots):
        trace_id = trace_snapshot.get('trace_id', f'trace_{i}')
        
        try:
            passed = judge_function(trace_snapshot)
            
            trace_result = {
                'trace_id': trace_id,
                'passed': passed,
                'error': None
            }
            
            if passed:
                results['passed'] += 1
            else:
                results['failed'] += 1
                
        except Exception as e:
            trace_result = {
                'trace_id': trace_id,
                'passed': False,
                'error': str(e)
            }
            results['errors'] += 1
            logger.error(f"Judge {judge_name} failed on trace {trace_id}: {e}")
        
        results['trace_results'].append(trace_result)
    
    # Calculate failure rate
    total_evaluated = results['passed'] + results['failed']
    if total_evaluated > 0:
        results['failure_rate'] = results['failed'] / total_evaluated
    else:
        results['failure_rate'] = 1.0
    
    return results


def summarize_judge_results(trace_snapshots: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Run all judges and return a comprehensive summary.
    
    Args:
        trace_snapshots: List of trace snapshots
        
    Returns:
        Dict with comprehensive results summary
    """
    
    failure_rates = run_binary_judges(trace_snapshots)
    
    summary = {
        'total_traces_evaluated': len(trace_snapshots),
        'failure_rates': failure_rates,
        'overall_health_score': _calculate_overall_health_score(failure_rates),
        'alerts': _generate_alerts(failure_rates),
        'recommendations': _generate_recommendations(failure_rates)
    }
    
    return summary


def _calculate_overall_health_score(failure_rates: Dict[str, float]) -> float:
    """Calculate an overall health score from 0.0 to 1.0 based on failure rates."""
    if not failure_rates:
        return 0.0
    
    # Health score is the inverse of average failure rate
    avg_failure_rate = sum(failure_rates.values()) / len(failure_rates)
    health_score = 1.0 - avg_failure_rate
    
    return max(0.0, health_score)


def _generate_alerts(failure_rates: Dict[str, float]) -> List[str]:
    """Generate alerts for concerning failure rates."""
    alerts = []
    
    # Alert thresholds
    HIGH_THRESHOLD = 0.2  # 20% failure rate
    CRITICAL_THRESHOLD = 0.5  # 50% failure rate
    
    for judge_name, failure_rate in failure_rates.items():
        if failure_rate >= CRITICAL_THRESHOLD:
            alerts.append(f"CRITICAL: {judge_name} failure rate is {failure_rate:.1%}")
        elif failure_rate >= HIGH_THRESHOLD:
            alerts.append(f"WARNING: {judge_name} failure rate is {failure_rate:.1%}")
    
    return alerts


def _generate_recommendations(failure_rates: Dict[str, float]) -> List[str]:
    """Generate recommendations based on failure patterns."""
    recommendations = []
    
    # Specific recommendations based on which judges are failing
    if failure_rates.get('needs_handoff', 0) > 0.15:
        recommendations.append("Consider adjusting human-in-the-loop escalation criteria")
    
    if failure_rates.get('groundedness', 0) > 0.2:
        recommendations.append("Review evidence collection and citation practices")
    
    if failure_rates.get('answer_relevance', 0) > 0.15:
        recommendations.append("Improve problem statement clarification process")
    
    if failure_rates.get('summary_actionability', 0) > 0.3:
        recommendations.append("Enhance final summary generation to include specific actions")
    
    if failure_rates.get('observability_coverage', 0) > 0.1:
        recommendations.append("Check event logging completeness in pipeline stages")
    
    # Overall health recommendations
    avg_failure_rate = sum(failure_rates.values()) / len(failure_rates) if failure_rates else 0
    if avg_failure_rate > 0.2:
        recommendations.append("Overall system health needs attention - review recent changes")
    
    return recommendations