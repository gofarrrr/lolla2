#!/usr/bin/env python3
"""
Migrate UnifiedContextStream imports to use interface adapter

This script replaces direct imports of UnifiedContextStream from src.core
with the interface adapter from src.interfaces, reducing dependency violations.

Usage:
    python3 scripts/migrate_context_stream_imports.py [--dry-run]
"""

import re
import sys
from pathlib import Path

DRY_RUN = "--dry-run" in sys.argv

# Pattern to match direct UnifiedContextStream imports
PATTERNS = [
    (
        r"from src\.core\.unified_context_stream import UnifiedContextStream",
        "from src.interfaces.context_stream_interface import ContextStreamInterface as UnifiedContextStream"
    ),
    (
        r"from src\.core\.unified_context_stream import \(([^)]+)\)",
        lambda m: f"from src.interfaces.context_stream_interface import ({m.group(1)})"
    ),
]

def migrate_file(file_path: Path) -> bool:
    """Migrate a single file. Returns True if changes were made."""
    content = file_path.read_text()
    original = content

    # Check if file imports UnifiedContextStream from src.core
    if "from src.core.unified_context_stream" not in content:
        return False

    # Skip adapter files (they're meant to import from core)
    if "src/engine/adapters" in str(file_path):
        print(f"⏭️  Skip (adapter): {file_path}")
        return False

    # Apply replacements
    for pattern, replacement in PATTERNS:
        if callable(replacement):
            content = re.sub(pattern, replacement, content)
        else:
            content = re.sub(pattern, replacement, content)

    if content != original:
        if DRY_RUN:
            print(f"🔍 Would migrate: {file_path}")
        else:
            file_path.write_text(content)
            print(f"✅ Migrated: {file_path}")
        return True

    return False

def main():
    project_root = Path(__file__).parent.parent
    src_engine = project_root / "src" / "engine"

    if not src_engine.exists():
        print(f"❌ Error: {src_engine} not found")
        sys.exit(1)

    # Find all Python files in src/engine
    python_files = list(src_engine.rglob("*.py"))

    print(f"🔍 Scanning {len(python_files)} files in src/engine/")
    if DRY_RUN:
        print("🧪 DRY RUN MODE - No files will be modified\n")

    migrated_count = 0
    for file_path in python_files:
        if migrate_file(file_path):
            migrated_count += 1

    print(f"\n📊 Summary:")
    print(f"   Files scanned: {len(python_files)}")
    print(f"   Files migrated: {migrated_count}")

    if DRY_RUN:
        print(f"\n💡 Run without --dry-run to apply changes")
    else:
        print(f"\n✅ Migration complete!")
        print(f"   Run: make test-architecture")
        print(f"   Expected: Violations reduced by ~{migrated_count}")

if __name__ == "__main__":
    main()
