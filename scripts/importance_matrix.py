#!/usr/bin/env python3
"""
Importance Matrix Generator for Operation Lean
Scores files across 4 dimensions: Complexity, Critical Path, Fan-In, Change Frequency
"""

import json
import sys
from pathlib import Path
from typing import Dict, List


# Critical path files (closer to main.py and core API routes)
CRITICAL_PATH_FILES = {
    'src/main.py': 10,
    'src/api/routes.py': 9,
    'src/api/analysis_routes.py': 9,
    'src/services/stateful_pipeline_orchestrator.py': 10,
    'src/services/enhanced_devils_advocate_system.py': 9,
    'src/core/method_actor_devils_advocate.py': 8,
    'src/engine/orchestration/pipeline_orchestrator.py': 9,
    'src/engine/orchestration/orchestrator.py': 8,
    'src/services/llm/llm_resilient_manager.py': 9,
    'src/services/llm/llm_service.py': 8,
    'src/services/research/research_resilient_manager.py': 8,
    'src/engine/services/llm/llm_manager.py': 7,
}

# High change frequency / developer pain files (inferred from purpose)
HIGH_CHANGE_FILES = {
    'orchestrator': 8,  # Core orchestration changes frequently
    'llm': 7,  # LLM integrations evolve with providers
    'devils_advocate': 7,  # Feature enhancements
    'research': 6,  # Research provider changes
    'pipeline': 8,  # Pipeline modifications
    'api': 7,  # API endpoint additions
    'routes': 7,  # Routing changes
}


def normalize_score(value: float, max_value: float) -> float:
    """Normalize a score to 0-10 scale"""
    if max_value == 0:
        return 0
    return min(10.0, (value / max_value) * 10)


def calculate_critical_path_score(file_path: str) -> float:
    """Calculate critical path score based on proximity to main.py and core routes"""
    # Direct match
    for critical_file, score in CRITICAL_PATH_FILES.items():
        if file_path.endswith(critical_file):
            return score

    # Heuristic scoring
    score = 0

    # Core services get higher scores
    if 'src/services/' in file_path:
        score += 3

    if 'src/api/' in file_path:
        score += 4

    if 'src/engine/' in file_path:
        score += 3

    if 'src/core/' in file_path:
        score += 4

    # Specific high-value directories
    if '/orchestration/' in file_path:
        score += 2
    if '/llm/' in file_path:
        score += 2
    if '/research/' in file_path:
        score += 1

    return min(10.0, score)


def calculate_change_frequency_score(file_path: str) -> float:
    """Estimate change frequency and developer pain based on file purpose"""
    score = 5  # Default mid-range

    # Check against high-change patterns
    for pattern, pattern_score in HIGH_CHANGE_FILES.items():
        if pattern in file_path.lower():
            score = max(score, pattern_score)

    # Adjust based on file type
    if 'test' in file_path.lower():
        score = 3  # Tests change but are less painful

    if 'config' in file_path.lower():
        score = 4  # Config changes are infrequent

    if 'utils' in file_path.lower() or 'helpers' in file_path.lower():
        score = 4  # Utilities are more stable

    if 'manager' in file_path.lower():
        score += 1  # Managers often need updates

    if 'service' in file_path.lower():
        score += 1  # Services evolve with features

    return min(10.0, score)


def generate_importance_matrix(complexity_data_path: str):
    """Generate importance matrix from complexity data"""
    with open(complexity_data_path, 'r') as f:
        data = json.load(f)

    all_files = data['all_files']

    # Get max values for normalization
    max_loc = max(f['loc'] for f in all_files)
    max_cc = max(f['max_cc'] for f in all_files)

    # Build fan-in lookup
    fan_in_lookup = {}
    for f in data['by_fan_in']:
        fan_in_lookup[f['file']] = f['fan_in']

    max_fan_in = max(fan_in_lookup.values()) if fan_in_lookup else 1

    # Calculate importance scores
    scored_files = []
    for file_data in all_files:
        file_path = file_data['file']
        rel_path = str(Path(file_path).relative_to(Path(complexity_data_path).parent.parent))

        # Dimension 1: Complexity Score (combination of LOC and CC)
        loc_score = normalize_score(file_data['loc'], max_loc)
        cc_score = normalize_score(file_data['max_cc'], max_cc)
        complexity_score = (loc_score * 0.4) + (cc_score * 0.6)  # Weight CC higher

        # Dimension 2: Critical Path Score
        critical_path_score = calculate_critical_path_score(file_path)

        # Dimension 3: Fan-In Score
        fan_in = fan_in_lookup.get(file_path, 0)
        fan_in_score = normalize_score(fan_in, max_fan_in)

        # Dimension 4: Change Frequency / Developer Pain Score
        change_freq_score = calculate_change_frequency_score(file_path)

        # Calculate weighted priority score
        # Weights: Complexity(30%), Critical Path(30%), Fan-In(25%), Change Freq(15%)
        priority_score = (
            complexity_score * 0.30 +
            critical_path_score * 0.30 +
            fan_in_score * 0.25 +
            change_freq_score * 0.15
        )

        scored_files.append({
            'file': rel_path,
            'loc': file_data['loc'],
            'max_cc': file_data['max_cc'],
            'avg_cc': file_data['avg_cc'],
            'fan_in': fan_in,
            'complexity_score': round(complexity_score, 2),
            'critical_path_score': round(critical_path_score, 2),
            'fan_in_score': round(fan_in_score, 2),
            'change_freq_score': round(change_freq_score, 2),
            'priority_score': round(priority_score, 2),
            'functions': file_data.get('functions', {})
        })

    # Sort by priority score
    scored_files.sort(key=lambda x: x['priority_score'], reverse=True)

    # Output top candidates
    print("=" * 100)
    print("IMPORTANCE MATRIX - TOP 30 REFACTORING CANDIDATES")
    print("=" * 100)
    print()
    print(f"{'#':<3} {'Priority':<8} {'Complex':<8} {'Critical':<9} {'Fan-In':<7} {'ΔFreq':<6} {'LOC':<6} {'CC':<5} {'File'}")
    print("-" * 100)

    for i, f in enumerate(scored_files[:30], 1):
        file_short = f['file'].replace('src/', '')
        if len(file_short) > 50:
            file_short = '...' + file_short[-47:]

        print(f"{i:<3} "
              f"{f['priority_score']:<8.2f} "
              f"{f['complexity_score']:<8.2f} "
              f"{f['critical_path_score']:<9.2f} "
              f"{f['fan_in_score']:<7.2f} "
              f"{f['change_freq_score']:<6.2f} "
              f"{f['loc']:<6} "
              f"{f['max_cc']:<5} "
              f"{file_short}")

    # Save detailed results
    output_path = Path(complexity_data_path).parent / 'importance_matrix.json'
    with open(output_path, 'w') as f:
        json.dump(scored_files[:30], f, indent=2)

    print()
    print(f"Detailed matrix saved to: {output_path}")
    print()

    return scored_files


def main():
    data_path = Path(__file__).parent / 'complexity_data.json'

    if not data_path.exists():
        print(f"Error: {data_path} does not exist. Run analyze_complexity.py first.", file=sys.stderr)
        sys.exit(1)

    generate_importance_matrix(str(data_path))


if __name__ == '__main__':
    main()
