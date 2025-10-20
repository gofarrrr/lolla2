"""
MECE Validation Module

Provides integrity checking, cost estimation, and review data building
for Problem Structuring stage output.

Enables second interactive pause for user validation and steering.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)


def jaccard_similarity(text1: str, text2: str) -> float:
    """Calculate Jaccard similarity between two text strings"""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if not words1 or not words2:
        return 0.0

    intersection = words1 & words2
    union = words1 | words2

    return len(intersection) / len(union) if union else 0.0


def detect_mece_overlaps(mece_components: List[Any], threshold: float = 0.4) -> List[Dict]:
    """
    Detect overlapping dimensions in MECE framework.
    Returns list of overlaps with similarity scores.
    """
    overlaps = []

    for i, comp1 in enumerate(mece_components):
        for j, comp2 in enumerate(mece_components[i+1:], start=i+1):
            dim1 = getattr(comp1, 'dimension', '')
            dim2 = getattr(comp2, 'dimension', '')

            similarity = jaccard_similarity(dim1, dim2)

            if similarity > threshold:
                overlaps.append({
                    "dimension_1_index": i,
                    "dimension_1_title": dim1,
                    "dimension_2_index": j,
                    "dimension_2_title": dim2,
                    "similarity_score": round(similarity, 2),
                    "severity": "high" if similarity > 0.6 else "medium"
                })

    return overlaps


def check_success_criteria_coverage(
    success_criteria: List[str],
    mece_components: List[Any]
) -> List[str]:
    """
    Check if all success criteria are covered by at least one dimension.
    Returns list of unmapped criteria.
    """
    unmapped = []

    for criterion in success_criteria:
        criterion_lower = criterion.lower()
        mapped = False

        for component in mece_components:
            dimension = getattr(component, 'dimension', '').lower()
            considerations = getattr(component, 'key_considerations', [])
            considerations_text = ' '.join(str(c).lower() for c in considerations)

            # Check if criterion relates to dimension or its considerations
            criterion_words = set(criterion_lower.split())
            dimension_words = set(dimension.split())
            consideration_words = set(considerations_text.split())

            # If >30% word overlap, consider it mapped
            overlap = criterion_words & (dimension_words | consideration_words)
            if len(overlap) / len(criterion_words) > 0.3:
                mapped = True
                break

        if not mapped:
            unmapped.append(criterion)

    return unmapped


def check_research_gap_coverage(research_gaps: List[Dict]) -> List[Dict]:
    """Check if all research gaps are linked to dimensions"""
    unlinked = []

    for gap in research_gaps:
        if not gap.get("linked_to_dimension"):
            unlinked.append({
                "question": gap.get("question", ""),
                "question_id": gap.get("question_id", "")
            })

    return unlinked


def detect_constraint_conflicts(
    mece_components: List[Any],
    constraints: List[str]
) -> List[Dict]:
    """
    Detect potential conflicts between dimensions and constraints.
    E.g., dimension requires 24mo timeline but constraint says 18mo.
    """
    conflicts = []

    # Extract time/budget numbers from constraints
    constraint_limits = {
        "budget_limit": None,
        "timeline_limit": None
    }

    for constraint in constraints:
        lower = constraint.lower()
        # Simple pattern matching for budget ($XM) and timeline (Xmo/months)
        if '$' in lower and 'm' in lower:
            # Extract budget
            import re
            match = re.search(r'\$(\d+)m', lower)
            if match:
                constraint_limits["budget_limit"] = int(match.group(1))

        if 'month' in lower or 'mo' in lower:
            # Extract timeline
            import re
            match = re.search(r'(\d+)\s*mo', lower)
            if match:
                constraint_limits["timeline_limit"] = int(match.group(1))

    # Check if any dimension mentions larger numbers
    for i, component in enumerate(mece_components):
        dimension = getattr(component, 'dimension', '')
        considerations = getattr(component, 'key_considerations', [])
        text = f"{dimension} {' '.join(str(c) for c in considerations)}".lower()

        # Check for conflicts
        import re

        # Budget conflicts
        if constraint_limits["budget_limit"]:
            budget_mentions = re.findall(r'\$(\d+)m', text)
            for mention in budget_mentions:
                if int(mention) > constraint_limits["budget_limit"]:
                    conflicts.append({
                        "type": "budget_conflict",
                        "dimension_index": i,
                        "dimension_title": getattr(component, 'dimension', ''),
                        "dimension_value": f"${mention}M",
                        "constraint_value": f"${constraint_limits['budget_limit']}M",
                        "severity": "high"
                    })

        # Timeline conflicts
        if constraint_limits["timeline_limit"]:
            timeline_mentions = re.findall(r'(\d+)\s*(?:mo|months?)', text)
            for mention in timeline_mentions:
                if int(mention) > constraint_limits["timeline_limit"]:
                    conflicts.append({
                        "type": "timeline_conflict",
                        "dimension_index": i,
                        "dimension_title": getattr(component, 'dimension', ''),
                        "dimension_value": f"{mention} months",
                        "constraint_value": f"{constraint_limits['timeline_limit']} months",
                        "severity": "high"
                    })

    return conflicts


def estimate_research_cost(research_gaps: List[Dict]) -> Dict[str, float]:
    """
    Estimate API cost and time for Oracle Research stage.
    Based on typical Perplexity API pricing and processing time.
    """
    num_gaps = len(research_gaps)

    # Estimates based on typical usage
    avg_api_cost_per_query = 0.50  # USD per research query
    avg_processing_minutes_per_query = 1.5  # Minutes per query

    return {
        "total_research_queries": num_gaps,
        "estimated_api_cost_usd": round(num_gaps * avg_api_cost_per_query, 2),
        "estimated_processing_minutes": round(num_gaps * avg_processing_minutes_per_query, 1),
        "cost_per_query": avg_api_cost_per_query
    }


def estimate_analysis_effort(mece_components: List[Any]) -> Dict[str, float]:
    """
    Estimate consultant analysis effort in hours.
    Based on dimension priority and number of considerations.
    """
    total_hours = 0.0
    breakdown = []

    for component in mece_components:
        priority = getattr(component, 'priority_level', 2)
        considerations = getattr(component, 'key_considerations', [])

        # Base hours by priority
        base_hours = {1: 12, 2: 8, 3: 5}.get(priority, 8)

        # Add hours for additional considerations (1.5hr per consideration)
        additional_hours = len(considerations) * 1.5

        dimension_hours = base_hours + additional_hours
        total_hours += dimension_hours

        breakdown.append({
            "dimension": getattr(component, 'dimension', ''),
            "priority": priority,
            "estimated_hours": round(dimension_hours, 1)
        })

    return {
        "total_analysis_hours": round(total_hours, 1),
        "breakdown": breakdown,
        "avg_hours_per_dimension": round(total_hours / len(mece_components), 1) if mece_components else 0
    }


def build_mece_review(state: Any) -> Dict[str, Any]:
    """
    Build comprehensive MECE review package for user validation.

    Returns structured data including:
    - MECE summary with provenance
    - Integrity checks (overlap, coverage, conflicts)
    - Cost estimates (research API, analysis hours)
    - Trigger flags (requires_full_review, confidence scores)
    """
    # Extract data from state
    problem_structure = getattr(state, 'problem_structure', None)
    if not problem_structure:
        return {"error": "No problem structure available"}

    mece_components = getattr(problem_structure, 'mece_framework', [])
    core_assumptions = getattr(problem_structure, 'core_assumptions', [])
    critical_constraints = getattr(problem_structure, 'critical_constraints', [])
    success_criteria = getattr(problem_structure, 'success_criteria', [])
    research_gaps = getattr(state, 'research_gaps', [])

    # Integrity checks
    overlaps = detect_mece_overlaps(mece_components)
    unmapped_criteria = check_success_criteria_coverage(success_criteria, mece_components)
    unlinked_gaps = check_research_gap_coverage(research_gaps)
    conflicts = detect_constraint_conflicts(mece_components, critical_constraints)

    # Cost estimates
    research_cost = estimate_research_cost(research_gaps)
    analysis_effort = estimate_analysis_effort(mece_components)

    # TOSCA coverage (if available)
    socratic_results = getattr(state, 'socratic_results', None)
    tosca_coverage_score = 1.0  # Default
    if socratic_results and hasattr(socratic_results, 'tosca_coverage'):
        tosca_coverage = getattr(socratic_results, 'tosca_coverage', {})
        tosca_coverage_score = len(tosca_coverage) / 5.0  # 5 TOSCA elements

    # Agent confidence
    problem_structure_raw = getattr(state, 'problem_structure_raw', None)
    agent_confidence = 0.85  # Default
    if problem_structure_raw and hasattr(problem_structure_raw, 'confidence_score'):
        agent_confidence = getattr(problem_structure_raw, 'confidence_score', 0.85)

    # Determine if full review is required
    requires_full_review = (
        tosca_coverage_score < 1.0 or
        agent_confidence < 0.8 or
        len(mece_components) < 4 or
        len(overlaps) > 0 or
        len(conflicts) > 0
    )

    # Build review package
    review = {
        "mece_summary": [
            {
                "index": i,
                "dimension": getattr(comp, 'dimension', ''),
                "priority": getattr(comp, 'priority_level', 2),
                "considerations": getattr(comp, 'key_considerations', []),
                "estimated_hours": analysis_effort["breakdown"][i]["estimated_hours"] if i < len(analysis_effort["breakdown"]) else 0
            }
            for i, comp in enumerate(mece_components)
        ],
        "assumptions": core_assumptions,
        "constraints": critical_constraints,
        "success_criteria": success_criteria,
        "research_gaps_preview": [
            {
                "question": gap.get("question", ""),
                "linked_to_dimension": gap.get("linked_to_dimension", ""),
                "question_id": gap.get("question_id", "")
            }
            for gap in research_gaps[:10]  # Max 10 for preview
        ],
        "integrity_checks": {
            "mece_overlap": {
                "passed": len(overlaps) == 0,
                "overlaps": overlaps
            },
            "success_criteria_coverage": {
                "passed": len(unmapped_criteria) == 0,
                "unmapped_criteria": unmapped_criteria
            },
            "research_gap_coverage": {
                "passed": len(unlinked_gaps) == 0,
                "unlinked_gaps": unlinked_gaps
            },
            "constraint_conflicts": {
                "passed": len(conflicts) == 0,
                "conflicts": conflicts
            }
        },
        "cost_estimates": {
            "research": research_cost,
            "analysis": analysis_effort,
            "total_estimated_cost_usd": research_cost["estimated_api_cost_usd"],
            "total_estimated_hours": research_cost["estimated_processing_minutes"] / 60 + analysis_effort["total_analysis_hours"]
        },
        "metadata": {
            "tosca_coverage_score": tosca_coverage_score,
            "agent_confidence": agent_confidence,
            "requires_full_review": requires_full_review,
            "num_dimensions": len(mece_components),
            "num_research_gaps": len(research_gaps),
            "quality_assessment": getattr(problem_structure, 'quality_assessment', 'GOOD')
        }
    }

    logger.info(f"📊 MECE Review built: {len(mece_components)} dimensions, "
               f"requires_full_review={requires_full_review}, "
               f"confidence={agent_confidence:.2f}")

    return review
