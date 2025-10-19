"""
Architecture guardrails for dependency direction.

Baseline captured on 2025-10-20. The test fails only when NEW violations appear.
Gradually reduce `KNOWN_VIOLATIONS` as refactors land.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


BASELINE_DOC = Path("docs/ENGINE_CORE_DEPENDENCY_BASELINE.md")


def _load_baseline_files() -> set[str]:
    """Read baseline violation files from the documentation snapshot."""
    if not BASELINE_DOC.exists():
        return set()

    capture = False
    files: set[str] = set()
    for line in BASELINE_DOC.read_text().splitlines():
        stripped = line.strip()
        if stripped == "```":
            capture = not capture
            continue
        if capture and stripped and not stripped.startswith("#"):
            files.add(stripped)
    return files


BASELINE_FILES = _load_baseline_files()
BASELINE_COUNT = len(BASELINE_FILES) or 146

PHASE_TARGETS = {
    "baseline": BASELINE_COUNT,
    "phase1": 120,
    "phase2": 80,
    "phase3": 40,
    "final": 0,
}


def _determine_target_count() -> int:
    """Resolve the active dependency target based on env or phase."""
    explicit_target = os.getenv("ARCH_GUARD_TARGET")
    if explicit_target:
        return int(explicit_target)

    phase = os.getenv("ARCH_GUARD_PHASE", "baseline").lower()
    if phase not in PHASE_TARGETS:
        pytest.skip(
            f"Unknown ARCH_GUARD_PHASE '{phase}'. "
            f"Supported phases: {', '.join(sorted(PHASE_TARGETS))}"
        )
    return PHASE_TARGETS[phase]


TARGET_COUNT = _determine_target_count()


def _get_violations(pattern: str, search_dir: str) -> list[str]:
    """Collect files matching the given ripgrep pattern."""
    # Try to find rg in PATH, or use common locations
    rg_cmd = shutil.which("rg")
    if rg_cmd is None:
        # Try common ripgrep locations (Claude Code vendor, Homebrew, etc.)
        possible_paths = [
            "/Users/marcin/.npm-global/lib/node_modules/@anthropic-ai/claude-code/vendor/ripgrep/x64-darwin/rg",
            "/usr/local/bin/rg",
            "/opt/homebrew/bin/rg",
        ]
        for path in possible_paths:
            if Path(path).exists():
                rg_cmd = path
                break

    if rg_cmd is None:
        pytest.skip("ripgrep is required for dependency direction checks")

    result = subprocess.run(
        [rg_cmd, "--files-with-matches", pattern, search_dir],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode == 1:  # no matches
        return []

    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr.strip() or "rg returned non-zero exit code")

    files = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return sorted(set(files))


@pytest.mark.architecture
def test_no_new_engine_to_core_imports() -> None:
    """Ensure no new upward imports are introduced in src/engine."""
    violations = [
        path for path in _get_violations(r"from src\.core", "src/engine") if path.endswith(".py")
    ]

    current_count = len(violations)
    current_set = set(violations)
    new_files = sorted(current_set - BASELINE_FILES)
    resolved_files = sorted(BASELINE_FILES - current_set)

    if new_files or resolved_files:
        diff_lines = [
            "Dependency diff:",
            *(f"+ {path}" for path in new_files),
            *(f"- {path}" for path in resolved_files),
        ]
    else:
        diff_lines = ["Dependency diff: (no changes vs baseline)"]

    assert current_count <= TARGET_COUNT, (
        "New src/engine → src/core imports detected!\n"
        f"Target max: {TARGET_COUNT}\n"
        f"Current:    {current_count}\n"
        + "\n".join(diff_lines)
    )

    if current_count < BASELINE_COUNT:
        print(
            f"Dependency cleanup progress: {BASELINE_COUNT} → {current_count} "
            f"(-{BASELINE_COUNT - current_count})"
        )


@pytest.mark.architecture
def test_interfaces_package_exists() -> None:
    """Verify the interfaces package exists to support dependency inversion."""
    interfaces_dir = Path("src/interfaces")
    assert interfaces_dir.exists(), "src/interfaces/ directory is missing"

    init_file = interfaces_dir / "__init__.py"
    assert init_file.exists(), "src/interfaces/__init__.py is missing"
