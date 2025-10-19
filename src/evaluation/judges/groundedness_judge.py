# src/evaluation/judges/groundedness_judge.py
"""
Binary judge to check if final report claims are supported by evidence found in the trace.

This judge validates that conclusions and recommendations in the final report
are properly grounded in evidence, research, and analysis documented in the trace events.
"""

import re
from typing import Dict, Any, List, Set
from urllib.parse import urlparse

from . import TraceSnapshot


def evaluate_groundedness(trace_snapshot: TraceSnapshot) -> bool:
    """
    Judge whether the final report's claims are adequately supported by evidence in the trace.
    
    Args:
        trace_snapshot: PII-safe trace data from export_traces.py
        
    Returns:
        bool: True if report is well-grounded (pass),
              False if report makes unsupported claims (fail)
    """
    
    # Extract relevant data
    context_stream = trace_snapshot.get('context_stream', {})
    events = context_stream.get('events', [])
    final_summary = _extract_final_summary(context_stream)
    
    if not final_summary:
        return False  # No summary to evaluate
    
    # Extract evidence from trace events
    evidence_base = _extract_evidence_base(events)
    
    # Extract claims from final summary
    claims = _extract_claims_from_summary(final_summary)
    
    if not claims:
        return True  # No claims to verify
    
    # Check groundedness
    grounding_score = _calculate_grounding_score(claims, evidence_base)
    
    # Threshold for passing: at least 70% of claims should be grounded
    return grounding_score >= 0.7


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


def _extract_evidence_base(events: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
    """Extract evidence base from trace events."""
    evidence_base = {
        'research_findings': set(),
        'citations': set(),
        'data_points': set(),
        'analysis_results': set(),
        'expert_insights': set()
    }
    
    for event in events:
        event_type = event.get('event_type', '')
        event_data = event.get('data', {})
        
        # Research and citations
        if 'research' in event_type.lower():
            findings = str(event_data).lower()
            evidence_base['research_findings'].add(findings)
            
            # Extract URLs/citations
            citations = _extract_citations(str(event_data))
            evidence_base['citations'].update(citations)
        
        # Data analysis results
        if any(keyword in event_type.lower() for keyword in ['analysis', 'evaluation', 'assessment']):
            analysis = str(event_data).lower()
            evidence_base['analysis_results'].add(analysis)
        
        # Consultant/expert insights
        if any(keyword in event_type.lower() for keyword in ['consultant', 'expert', 'advisor']):
            insights = str(event_data).lower()
            evidence_base['expert_insights'].add(insights)
        
        # Data points and metrics
        data_points = _extract_data_points(str(event_data))
        evidence_base['data_points'].update(data_points)
    
    return evidence_base


def _extract_citations(text: str) -> Set[str]:
    """Extract citations, URLs, and reference markers from text."""
    citations = set()
    
    # URL patterns
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    urls = re.findall(url_pattern, text)
    citations.update(urls)
    
    # Citation markers like [1], (Smith, 2023), etc.
    citation_patterns = [
        r'\[[\d,\s-]+\]',  # [1], [1,2], [1-3]
        r'\(\w+,?\s*\d{4}\)',  # (Smith, 2023)
        r'\(\w+\s+et\s+al\.?,?\s*\d{4}\)',  # (Smith et al., 2023)
        r'according\s+to\s+[\w\s]+\(\d{4}\)',  # according to Smith (2023)
        r'as\s+noted\s+by\s+[\w\s]+',  # as noted by Smith
        r'source:\s*[\w\s]+',  # source: Company Report
    ]
    
    for pattern in citation_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        citations.update(matches)
    
    return citations


def _extract_data_points(text: str) -> Set[str]:
    """Extract numerical data points and metrics from text."""
    data_points = set()
    
    # Numerical patterns with context
    patterns = [
        r'\d+\.?\d*%',  # Percentages
        r'\$[\d,]+(?:k|m|b|million|billion)?',  # Currency
        r'\d+\.?\d*\s*(?:users?|customers?|employees?|transactions?)',  # Counts
        r'\d+\.?\d*\s*(?:seconds?|minutes?|hours?|days?|months?|years?)',  # Time
        r'\d+\.?\d*\s*(?:gb|mb|kb|tb|bytes?)',  # Storage
        r'increased?\s+by\s+\d+\.?\d*%?',  # Growth metrics
        r'decreased?\s+by\s+\d+\.?\d*%?',  # Decline metrics
        r'improved?\s+by\s+\d+\.?\d*%?',  # Improvement metrics
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        data_points.update(matches)
    
    return data_points


def _extract_claims_from_summary(summary: str) -> List[str]:
    """Extract factual claims from the summary that need evidence support."""
    claims = []
    
    # Split summary into sentences
    sentences = re.split(r'[.!?]+', summary)
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # Identify factual claims (statements that could be verified)
        if _is_factual_claim(sentence):
            claims.append(sentence.lower())
    
    return claims


def _is_factual_claim(sentence: str) -> bool:
    """Determine if a sentence contains a factual claim that needs evidence."""
    sentence_lower = sentence.lower()
    
    # Skip recommendations and opinions
    opinion_markers = [
        'recommend', 'suggest', 'should', 'could', 'might', 'consider',
        'propose', 'advise', 'believe', 'think', 'feel', 'opinion'
    ]
    
    if any(marker in sentence_lower for marker in opinion_markers):
        return False
    
    # Look for factual claim indicators
    factual_indicators = [
        # Measurements and metrics
        r'\d+\.?\d*%', r'\$[\d,]+', r'\d+\.?\d*\s+(?:users?|customers?|employees?)',
        
        # Comparative statements
        r'increased?', r'decreased?', r'improved?', r'reduced?', r'higher', r'lower',
        
        # Causal statements
        r'caused?', r'resulted?', r'led\s+to', r'due\s+to', r'because',
        
        # Research findings
        r'study\s+shows?', r'research\s+indicates?', r'data\s+shows?', r'evidence\s+suggests?',
        
        # Market/industry facts
        r'market\s+size', r'industry\s+trend', r'competitor\s+analysis', r'benchmark',
        
        # Time-based facts
        r'since\s+\d{4}', r'in\s+\d{4}', r'over\s+the\s+past', r'currently',
    ]
    
    for indicator in factual_indicators:
        if re.search(indicator, sentence_lower):
            return True
    
    return False


def _calculate_grounding_score(claims: List[str], evidence_base: Dict[str, Set[str]]) -> float:
    """Calculate what percentage of claims are supported by evidence."""
    if not claims:
        return 1.0  # No claims = perfectly grounded
    
    grounded_claims = 0
    
    # Combine all evidence into searchable text
    all_evidence = set()
    for evidence_type, evidence_set in evidence_base.items():
        all_evidence.update(evidence_set)
    
    evidence_text = ' '.join(all_evidence).lower()
    
    for claim in claims:
        if _is_claim_grounded(claim, evidence_text, evidence_base):
            grounded_claims += 1
    
    return grounded_claims / len(claims)


def _is_claim_grounded(claim: str, evidence_text: str, evidence_base: Dict[str, Set[str]]) -> bool:
    """Check if a specific claim is supported by the evidence base."""
    claim_lower = claim.lower()
    
    # Method 1: Direct keyword overlap
    claim_keywords = set(re.findall(r'\b\w+\b', claim_lower))
    claim_keywords = {word for word in claim_keywords if len(word) > 3}  # Filter short words
    
    evidence_keywords = set(re.findall(r'\b\w+\b', evidence_text))
    
    keyword_overlap = len(claim_keywords.intersection(evidence_keywords))
    keyword_ratio = keyword_overlap / len(claim_keywords) if claim_keywords else 0
    
    if keyword_ratio >= 0.5:  # At least 50% keyword overlap
        return True
    
    # Method 2: Specific data point matching
    claim_data_points = _extract_data_points(claim)
    evidence_data_points = evidence_base.get('data_points', set())
    
    if claim_data_points and evidence_data_points:
        # Check if any data points in claim match evidence
        for claim_point in claim_data_points:
            for evidence_point in evidence_data_points:
                if _data_points_match(claim_point, evidence_point):
                    return True
    
    # Method 3: Citation verification
    claim_citations = _extract_citations(claim)
    evidence_citations = evidence_base.get('citations', set())
    
    if claim_citations and evidence_citations:
        # Check if claim references match evidence citations
        if claim_citations.intersection(evidence_citations):
            return True
    
    # Method 4: Semantic concept matching
    return _semantic_concepts_match(claim, evidence_text)


def _data_points_match(claim_point: str, evidence_point: str) -> bool:
    """Check if data points are similar enough to be considered matching."""
    # Extract numbers for comparison
    claim_numbers = re.findall(r'\d+\.?\d*', claim_point)
    evidence_numbers = re.findall(r'\d+\.?\d*', evidence_point)
    
    if claim_numbers and evidence_numbers:
        # Check if numbers are reasonably close (within 20% tolerance)
        try:
            claim_val = float(claim_numbers[0])
            evidence_val = float(evidence_numbers[0])
            
            if evidence_val == 0:
                return claim_val == 0
            
            difference_ratio = abs(claim_val - evidence_val) / evidence_val
            return difference_ratio <= 0.2  # 20% tolerance
        except ValueError:
            pass
    
    # Fallback to string similarity
    return claim_point.lower() in evidence_point.lower() or evidence_point.lower() in claim_point.lower()


def _semantic_concepts_match(claim: str, evidence_text: str) -> bool:
    """Check if claim and evidence discuss similar concepts."""
    # Define concept clusters for semantic matching
    concept_clusters = {
        'performance': ['performance', 'speed', 'efficiency', 'optimization', 'latency', 'throughput'],
        'growth': ['growth', 'increase', 'expansion', 'scale', 'rising', 'uptick'],
        'decline': ['decline', 'decrease', 'reduction', 'drop', 'falling', 'downturn'],
        'cost': ['cost', 'expense', 'budget', 'price', 'financial', 'money'],
        'user_experience': ['user', 'customer', 'experience', 'satisfaction', 'usability', 'interface'],
        'security': ['security', 'privacy', 'protection', 'vulnerability', 'threat', 'risk'],
        'technology': ['technology', 'system', 'platform', 'software', 'hardware', 'infrastructure'],
    }
    
    claim_lower = claim.lower()
    evidence_lower = evidence_text.lower()
    
    # Find which concepts appear in claim and evidence
    claim_concepts = set()
    evidence_concepts = set()
    
    for concept, keywords in concept_clusters.items():
        if any(keyword in claim_lower for keyword in keywords):
            claim_concepts.add(concept)
        if any(keyword in evidence_lower for keyword in keywords):
            evidence_concepts.add(concept)
    
    # Check for concept overlap
    return bool(claim_concepts.intersection(evidence_concepts))