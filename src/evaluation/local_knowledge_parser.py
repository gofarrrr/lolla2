"""
Local Knowledge Asset Parser
=============================

Parses mental models, NWAY clusters, and NWAY2 agent instructions
directly from local markdown files (migrations/ directory).

This is the SOURCE OF TRUTH for Operation Scribe.
DO NOT use Supabase mental_models table - it is incomplete and unreliable.
"""

import os
import re
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from pathlib import Path
from enum import Enum


class KnowledgeTier(Enum):
    """Knowledge asset tier classification"""
    TIER_1_MENTAL_MODEL = "tier_1_mental_model"
    TIER_2_NWAY_CLUSTER = "tier_2_nway_cluster"
    TIER_3_NWAY2_AGENT = "tier_3_nway2_agent"


@dataclass
class KnowledgeAsset:
    """Unified knowledge asset representation"""
    # Identity
    asset_id: str  # Hash of file path
    file_path: str
    filename: str
    tier: KnowledgeTier

    # Core content
    name: str
    content: str
    content_length: int

    # Metadata
    category: Optional[str] = None
    subcategory: Optional[str] = None

    # Parsed sections (tier-specific)
    sections: Dict[str, str] = None

    def __post_init__(self):
        if self.sections is None:
            self.sections = {}


class LocalKnowledgeParser:
    """
    Parses all knowledge assets from local migrations/ directory.

    This is the authoritative source - bypasses Supabase entirely.
    """

    def __init__(self, migrations_dir: str = "migrations"):
        """
        Initialize parser.

        Args:
            migrations_dir: Path to migrations directory (default: "migrations")
        """
        self.migrations_dir = Path(migrations_dir)

        if not self.migrations_dir.exists():
            raise FileNotFoundError(
                f"Migrations directory not found: {self.migrations_dir}"
            )

    def _generate_asset_id(self, file_path: str) -> str:
        """Generate unique asset ID from file path"""
        return hashlib.md5(file_path.encode()).hexdigest()[:16]

    def _extract_name_from_filename(self, filename: str) -> str:
        """Extract clean name from filename"""
        # Remove file extension
        name = Path(filename).stem

        # Remove _rag suffix if present
        name = name.replace("_rag", "")

        # Replace underscores with spaces
        name = name.replace("_", " ")

        # Clean up NWAY prefixes
        name = re.sub(r"NWAY2?\s+", "", name, flags=re.IGNORECASE)

        # Remove title annotations
        name = re.sub(r"\s+title\s+", " ", name, flags=re.IGNORECASE)

        # Clean up multiple spaces
        name = re.sub(r"\s+", " ", name).strip()

        return name

    def _parse_mm_file(self, file_path: Path) -> KnowledgeAsset:
        """
        Parse MM1/MM2 mental model file (Tier 1).

        These are free-form markdown with section headers.
        """
        content = file_path.read_text(encoding="utf-8", errors="ignore")

        # Extract sections based on separators
        sections = {}
        current_section = "introduction"
        current_content = []

        for line in content.split("\n"):
            # Detect section separator
            if line.strip().startswith("---"):
                if current_content:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = f"section_{len(sections) + 1}"
                current_content = []
            else:
                current_content.append(line)

        # Add final section
        if current_content:
            sections[current_section] = "\n".join(current_content).strip()

        # Determine category (MM1 vs MM2)
        category = "MM1" if "MM1" in str(file_path) else "MM2"

        name = self._extract_name_from_filename(file_path.name)

        return KnowledgeAsset(
            asset_id=self._generate_asset_id(str(file_path)),
            file_path=str(file_path),
            filename=file_path.name,
            tier=KnowledgeTier.TIER_1_MENTAL_MODEL,
            name=name,
            content=content,
            content_length=len(content),
            category=category,
            sections=sections
        )

    def _parse_nway_file(self, file_path: Path) -> KnowledgeAsset:
        """
        Parse NWAY cluster file (Tier 2).

        These are Q&A format with multiple questions per cluster.
        """
        content = file_path.read_text(encoding="utf-8", errors="ignore")

        # Extract questions (lines that end with ?)
        questions = []
        for line in content.split("\n"):
            line = line.strip()
            if line and line.endswith("?"):
                questions.append(line)

        # Extract category from filename (e.g., NWAY_DECISION_001)
        match = re.search(r"NWAY_([A-Z]+)_\d+", file_path.name)
        category = match.group(1) if match else "UNKNOWN"

        name = self._extract_name_from_filename(file_path.name)

        sections = {
            "questions": "\n".join(questions),
            "full_content": content
        }

        return KnowledgeAsset(
            asset_id=self._generate_asset_id(str(file_path)),
            file_path=str(file_path),
            filename=file_path.name,
            tier=KnowledgeTier.TIER_2_NWAY_CLUSTER,
            name=name,
            content=content,
            content_length=len(content),
            category=category,
            subcategory=f"cluster_{len(questions)}_questions",
            sections=sections
        )

    def _parse_nway2_file(self, file_path: Path) -> KnowledgeAsset:
        """
        Parse NWAY2 agent instruction file (Tier 3).

        These are agent-specific prompt templates and instructions.
        """
        content = file_path.read_text(encoding="utf-8", errors="ignore")

        # Extract agent name from filename
        match = re.search(r"NWAY_?(.+)\.md", file_path.name, re.IGNORECASE)
        agent_name = match.group(1) if match else file_path.stem
        agent_name = agent_name.replace("_", " ").title()

        # Try to identify instruction sections
        sections = {}
        lines = content.split("\n")

        # Look for numbered questions or instruction blocks
        question_blocks = []
        current_block = []

        for line in lines:
            if re.match(r"^\d+[\.\)]\s", line.strip()):
                if current_block:
                    question_blocks.append("\n".join(current_block))
                current_block = [line]
            elif line.strip():
                current_block.append(line)

        if current_block:
            question_blocks.append("\n".join(current_block))

        sections["instruction_blocks"] = question_blocks
        sections["full_content"] = content

        return KnowledgeAsset(
            asset_id=self._generate_asset_id(str(file_path)),
            file_path=str(file_path),
            filename=file_path.name,
            tier=KnowledgeTier.TIER_3_NWAY2_AGENT,
            name=agent_name,
            content=content,
            content_length=len(content),
            category="NWAY2",
            subcategory=agent_name.lower().replace(" ", "_"),
            sections=sections
        )

    def parse_all_assets(self) -> List[KnowledgeAsset]:
        """
        Parse all knowledge assets from local migrations directory.

        Returns:
            List of all parsed knowledge assets (261 expected)
        """
        assets = []

        # Parse MM1 files (Tier 1)
        mm1_dir = self.migrations_dir / "MM1"
        if mm1_dir.exists():
            for md_file in mm1_dir.glob("*.md"):
                try:
                    asset = self._parse_mm_file(md_file)
                    assets.append(asset)
                except Exception as e:
                    print(f"⚠️  Failed to parse {md_file}: {e}")

        # Parse MM2 files (Tier 1)
        mm2_dir = self.migrations_dir / "MM2"
        if mm2_dir.exists():
            for md_file in mm2_dir.glob("*.md"):
                try:
                    asset = self._parse_mm_file(md_file)
                    assets.append(asset)
                except Exception as e:
                    print(f"⚠️  Failed to parse {md_file}: {e}")

        # Parse NWAY files (Tier 2)
        nway_dir = self.migrations_dir / "NWAY"
        if nway_dir.exists():
            for md_file in nway_dir.glob("*.md"):
                try:
                    asset = self._parse_nway_file(md_file)
                    assets.append(asset)
                except Exception as e:
                    print(f"⚠️  Failed to parse {md_file}: {e}")

        # Parse NWAY2 files (Tier 3)
        nway2_dir = self.migrations_dir / "NWAY2"
        if nway2_dir.exists():
            for md_file in nway2_dir.glob("*.md"):
                try:
                    asset = self._parse_nway2_file(md_file)
                    assets.append(asset)
                except Exception as e:
                    print(f"⚠️  Failed to parse {md_file}: {e}")

        return assets

    def get_assets_by_tier(self, tier: KnowledgeTier) -> List[KnowledgeAsset]:
        """Get all assets for a specific tier"""
        all_assets = self.parse_all_assets()
        return [a for a in all_assets if a.tier == tier]

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of parsed assets"""
        assets = self.parse_all_assets()

        by_tier = {}
        for tier in KnowledgeTier:
            tier_assets = [a for a in assets if a.tier == tier]
            by_tier[tier.value] = len(tier_assets)

        by_category = {}
        for asset in assets:
            cat = asset.category or "unknown"
            by_category[cat] = by_category.get(cat, 0) + 1

        return {
            "total_assets": len(assets),
            "by_tier": by_tier,
            "by_category": by_category,
            "total_content_length": sum(a.content_length for a in assets),
            "avg_content_length": sum(a.content_length for a in assets) / len(assets) if assets else 0
        }


# Convenience function
def parse_all_local_knowledge() -> List[KnowledgeAsset]:
    """
    Parse all knowledge assets from local migrations directory.

    This is the primary entry point for Operation Scribe.
    """
    parser = LocalKnowledgeParser()
    return parser.parse_all_assets()


if __name__ == "__main__":
    # Test parser
    parser = LocalKnowledgeParser()

    print("=" * 80)
    print("LOCAL KNOWLEDGE PARSER - Operation Scribe")
    print("=" * 80)
    print()

    stats = parser.get_summary_stats()

    print(f"✅ Total knowledge assets parsed: {stats['total_assets']}")
    print()
    print("📊 By Tier:")
    for tier, count in stats['by_tier'].items():
        print(f"   {tier}: {count}")
    print()
    print("📂 By Category:")
    for cat, count in stats['by_category'].items():
        print(f"   {cat}: {count}")
    print()
    print(f"📝 Total content: {stats['total_content_length']:,} characters")
    print(f"📏 Average length: {stats['avg_content_length']:.0f} characters")
    print()
    print("=" * 80)
