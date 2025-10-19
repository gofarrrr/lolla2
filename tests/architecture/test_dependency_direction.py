"""
Architecture guardrails for dependency direction.

Baseline captured on 2025-10-20. The test fails only when NEW violations appear.
Gradually reduce `KNOWN_VIOLATIONS` as refactors land.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


KNOWN_VIOLATIONS = {
    "src/engine -> src/core": 146,  # rg --files-with-matches "from src\.core" src/engine | wc -l
}


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
    violations = _get_violations(r"from src\.core", "src/engine")

    current_count = len(violations)
    baseline_count = KNOWN_VIOLATIONS["src/engine -> src/core"]

    assert current_count <= baseline_count, (
        f"New src/engine → src/core imports detected!\n"
        f"Baseline: {baseline_count}\n"
        f"Current:  {current_count}\n"
        "Violating files:\n"
        + "\n".join(violations)
    )

    if current_count < baseline_count:
        print(
            f"Dependency cleanup progress: {baseline_count} → {current_count} "
            f"(-{baseline_count - current_count})"
        )


@pytest.mark.architecture
def test_interfaces_package_exists() -> None:
    """Verify the interfaces package exists to support dependency inversion."""
    interfaces_dir = Path("src/interfaces")
    assert interfaces_dir.exists(), "src/interfaces/ directory is missing"

    init_file = interfaces_dir / "__init__.py"
    assert init_file.exists(), "src/interfaces/__init__.py is missing"
