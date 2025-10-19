"""NWAY Interaction Provider

Builds short interaction directives for pairs of mental models using the
cluster YAML definitions.

The directives are deterministic (no LLM). They summarise the relevant
`interactions` entries from the NWAY definition, truncated to a compact length
so they can be injected in consultant prompts without bloating tokens.
"""

from __future__ import annotations

import itertools
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.config.architecture_loader import load_full_architecture
from .aliases import resolve_alias


MAX_WORDS = 230


class NwayInteractionProvider:
    """Provides pre-baked interaction directives for model pairs."""

    def __init__(self, master_path: Optional[str] = None) -> None:
        self.master_path = Path(
            master_path
            or os.getenv(
                "NWAY_MASTER_YAML",
                "cognitive_architecture/nway_cognitive_architecture.yaml",
            )
        )
        self._pair_map: Dict[str, Dict[str, Any]] = {}
        self._loaded = False

    # ------------------------------------------------------------------
    def get_directive(self, models: Iterable[str]) -> Optional[Dict[str, Any]]:
        """Return directive metadata for the first matching model pair."""

        if not self._loaded:
            self._load_pairs()

        normalized = [resolve_alias(m) for m in models if m]
        for pair in itertools.combinations(sorted(set(normalized)), 2):
            key = "|".join(pair)
            entry = self._pair_map.get(key)
            if entry:
                return entry
        return None

    # ------------------------------------------------------------------
    def _load_pairs(self) -> None:
        try:
            clusters = load_full_architecture(self.master_path)
        except FileNotFoundError:
            self._loaded = True
            return

        for cluster in clusters.values():
            for nway in cluster.nways:
                models = [resolve_alias(m) for m in nway.models if m]
                if len(models) < 2:
                    continue
                directive = self._build_directive_text(nway.title, nway.interactions)
                if not directive:
                    continue
                for pair in itertools.combinations(sorted(set(models)), 2):
                    key = "|".join(pair)
                    if key not in self._pair_map:
                        self._pair_map[key] = {
                            "text": directive,
                            "models": list(pair),
                            "focus": self._infer_focus_label(nway.interactions),
                            "title": nway.title,
                        }

        self._loaded = True

    @staticmethod
    def _build_directive_text(
        title: Optional[str], interactions: Dict[str, str]
    ) -> Optional[str]:
        lines: List[str] = []
        if title:
            lines.append(f"{title}:")
        for label, content in interactions.items():
            if not content:
                continue
            label_clean = label.replace("_", " ").title()
            lines.append(f"- {label_clean}: {content}")
        text = " \n".join(lines).strip()
        if not text:
            return None
        words = text.split()
        if len(words) > MAX_WORDS:
            text = " ".join(words[:MAX_WORDS]) + "..."
        return text

    @staticmethod
    def _infer_focus_label(interactions: Dict[str, str]) -> Optional[str]:
        for label in interactions.keys():
            label_lower = label.lower()
            if "trade" in label_lower or "tension" in label_lower or "conflict" in label_lower:
                return "trade_offs"
            if "synerg" in label_lower or "complement" in label_lower:
                return "synergies"
            if "failure" in label_lower or "pitfall" in label_lower:
                return "failure_modes"
            if "integrat" in label_lower or "framework" in label_lower:
                return "integration"
        return None
