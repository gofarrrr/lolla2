"""
Complexity budget guardrail for src/main.py.

Keeps the entry point within agreed bounds without forcing unnecessary churn.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


BUDGET_LOC = 850
MAX_BRANCHING_NODES = 70


def _count_effective_loc(path: Path) -> int:
    """Count non-empty, non-comment lines."""
    count = 0
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        count += 1
    return count


def _count_branching_nodes(tree: ast.AST) -> int:
    """Approximate cyclomatic complexity via branching node count."""
    branching = (
        ast.If,
        ast.For,
        ast.AsyncFor,
        ast.While,
        ast.Try,
        ast.ExceptHandler,
        ast.With,
        ast.AsyncWith,
        ast.BoolOp,
        ast.IfExp,
    )
    return sum(isinstance(node, branching) for node in ast.walk(tree))


@pytest.mark.architecture
def test_main_file_stays_within_budget() -> None:
    """Ensure main.py remains within the agreed size and complexity envelope."""
    main_path = Path("src/main.py")
    assert main_path.exists(), "src/main.py missing"

    effective_loc = _count_effective_loc(main_path)
    tree = ast.parse(main_path.read_text())
    branching_nodes = _count_branching_nodes(tree)

    assert effective_loc <= BUDGET_LOC, (
        f"src/main.py exceeds LOC budget ({effective_loc} > {BUDGET_LOC}). "
        "Refactor or adjust budget intentionally."
    )

    assert branching_nodes <= MAX_BRANCHING_NODES, (
        "src/main.py exceeds branching budget "
        f"({branching_nodes} > {MAX_BRANCHING_NODES}). "
        "Consider extracting orchestration helpers."
    )
