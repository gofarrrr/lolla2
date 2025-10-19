"""Consultant Depth Pack Builder

Generates a compact "Depth Pack" for each consultant before analysis begins.
- Pulls deterministic MM Q&A content (practical/pitfalls/implementation)
- Optionally adds a short NWAY interaction directive (stubbed for now)
- Formats output with Grok 4 Fast-friendly tags (no CoT, minimal instructions)

The builder relies only on deterministic data sources (YAML + relational tables)
and avoids any runtime LLM calls.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING

import yaml

from .qa_precision_retriever import QAChunksRepository
from .aliases import resolve_alias
from .nway_interaction_provider import NwayInteractionProvider

if TYPE_CHECKING:  # pragma: no cover - import only for type hints
    from src.orchestration.contracts import ConsultantBlueprint

logger = logging.getLogger(__name__)


MM_QTYPE_PRIORITY_DEFAULT = ["practical", "pitfalls", "implementation", "evidence", "foundational"]


@dataclass
class DepthPackResult:
    text: str
    metadata: Dict[str, Any]

    @property
    def has_content(self) -> bool:
        return bool(self.text.strip())


class ConsultantDepthPackBuilder:
    """Build minimal Grok-friendly depth packs for consultants."""

    def __init__(
        self,
        qa_repo: Optional[QAChunksRepository] = None,
        *,
        mm_items_limit: int = 2,
        summary_keys: int = 3,
        include_full_answer: bool = False,
        personas_path: Optional[str] = None,
        nway_provider: Optional[NwayInteractionProvider] = None,
        enable_nway: bool = True,
    ) -> None:
        self.qa_repo = qa_repo or QAChunksRepository()
        self.mm_items_limit = mm_items_limit
        self.summary_keys = summary_keys
        self.include_full_answer = include_full_answer
        self.nway_provider = (
            nway_provider if enable_nway else None
        ) or (NwayInteractionProvider() if enable_nway else None)
        # Operation Primacy Diagnostics: opt-in row id emission for forensics
        try:
            import os as _os
            self.enable_diagnostics = (
                _os.getenv("DEPTH_PACK_DIAGNOSTICS", "false").lower()
                in {"1", "true", "yes", "on"}
            )
        except Exception:
            self.enable_diagnostics = False

        self.personas_path = Path(
            personas_path
            or os.getenv(
                "CONSULTANT_PERSONAS_YAML",
                "cognitive_architecture/consultants/consultant_personas.yaml",
            )
        )
        self.persona_config = self._load_persona_definitions()
        self.qtype_priority_map = self._build_qtype_priority_map()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_depth_pack(
        self,
        consultant: "ConsultantBlueprint",
        *,
        problem_context: str = "",
        candidate_models: Optional[Sequence[str]] = None,
    ) -> DepthPackResult:
        persona_id = consultant.consultant_id
        model_candidates = self._derive_candidate_models(persona_id, candidate_models)

        mm_items: List[Dict[str, Any]] = []
        used_models: set[str] = set()

        for model in model_candidates:
            canonical = resolve_alias(model)
            if canonical in used_models:
                continue

            row = self._select_mm_row(canonical, persona_id)
            if not row:
                continue

            keys = self._extract_depth_keys(row.get("answer", ""))
            if not keys:
                continue

            item = {
                "model": canonical,
                "question_type": row.get("question_type"),
                "question_num": row.get("question_num"),
                "question": row.get("question"),
                "keys": keys,
                "answer": row.get("answer"),
                "source": row.get("source_file"),
                "include_full": self.include_full_answer and len(mm_items) == 0,
            }
            # Include db row id only in diagnostics mode
            if self.enable_diagnostics:
                item["row_id"] = row.get("id")
            mm_items.append(item)
            used_models.add(canonical)

            if len(mm_items) >= self.mm_items_limit:
                break

        nway_entry = None
        if self.nway_provider and len(model_candidates) >= 2:
            nway_entry = self.nway_provider.get_directive(model_candidates)

        if not mm_items and not nway_entry:
            return DepthPackResult(
                text="",
                metadata={"consultant_id": persona_id, "mm_items": [], "nway": None},
            )

        depth_text = self._render_depth_pack(mm_items, nway_entry)
        metadata = {
            "consultant_id": persona_id,
            "mm_items": [
                {
                    "model": item["model"],
                    "question_type": item["question_type"],
                    "question_num": item["question_num"],
                    "keys_count": len(item["keys"]),
                    "include_full": item["include_full"],
                    **({"row_id": item.get("row_id")} if self.enable_diagnostics else {}),
                }
                for item in mm_items
            ],
            "nway": nway_entry,
        }

        return DepthPackResult(text=depth_text, metadata=metadata)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_persona_definitions(self) -> Dict[str, Any]:
        try:
            raw = yaml.safe_load(self.personas_path.read_text(encoding="utf-8")) or {}
            personas = raw.get("consultant_personas", {})
            if not isinstance(personas, dict):
                return {}
            return personas
        except FileNotFoundError:
            logger.warning(f"Consultant personas YAML not found: {self.personas_path}")
            return {}
        except Exception as exc:
            logger.error(f"Failed to load persona definitions: {exc}")
            return {}

    def _build_qtype_priority_map(self) -> Dict[str, List[str]]:
        mapping: Dict[str, List[str]] = {}
        mapping.update(
            {
                "strategic_analyst": ["practical", "pitfalls", "implementation"],
                "risk_assessor": ["pitfalls", "evidence", "practical"],
                "financial_analyst": ["practical", "implementation", "pitfalls"],
                "implementation_specialist": ["implementation", "pitfalls", "practical"],
                "technology_advisor": ["practical", "pitfalls", "implementation"],
                "innovation_consultant": ["practical", "pitfalls", "implementation"],
                "crisis_manager": ["pitfalls", "implementation", "practical"],
                "operations_expert": ["practical", "implementation", "pitfalls"],
                "market_researcher": ["practical", "pitfalls", "evidence"],
            }
        )
        return mapping

    def _derive_candidate_models(
        self, persona_id: str, candidate_models: Optional[Sequence[str]]
    ) -> List[str]:
        persona_cfg = self.persona_config.get(persona_id, {}) or {}
        affinities = persona_cfg.get("mental_model_affinities", {}) or {}

        ordered_affinities = list(affinities.keys())
        if candidate_models:
            ordered_affinities.extend(candidate_models)

        seen: set[str] = set()
        ordered: List[str] = []
        for name in ordered_affinities:
            canonical = resolve_alias(name)
            if canonical and canonical not in seen:
                seen.add(canonical)
                ordered.append(canonical)

        return ordered

    def _select_mm_row(self, model: str, persona_id: str) -> Optional[Dict[str, Any]]:
        priority = self.qtype_priority_map.get(persona_id, MM_QTYPE_PRIORITY_DEFAULT)

        for qtype in priority:
            rows = self.qa_repo.fetch_by_model_and_type(model, qtype, limit=1)
            if rows:
                row = rows[0]
                row.setdefault("question_type", qtype)
                return row

        # Fallback: any available entry for the model
        rows = self.qa_repo.fetch_by_model_and_type(model, None, limit=1)
        if rows:
            return rows[0]
        return None

    def _extract_depth_keys(self, answer: str) -> List[str]:
        if not answer:
            return []

        lines = [line.strip() for line in answer.splitlines() if line.strip()]
        bullet_lines = [line for line in lines if line.startswith("-") or line.startswith("•")]
        if bullet_lines:
            return [self._clean_key(line.lstrip("-• ")) for line in bullet_lines[: self.summary_keys]]

        sentences = re.split(r"(?<=[.!?])\s+", answer)
        cleaned = [self._clean_key(sentence) for sentence in sentences if sentence.strip()]
        return cleaned[: self.summary_keys]

    @staticmethod
    def _clean_key(text: str) -> str:
        text = text.strip()
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text)
        return text

    def _render_depth_pack(
        self, mm_items: List[Dict[str, Any]], nway_entry: Optional[Dict[str, Any]]
    ) -> str:
        lines: List[str] = ["<depth_pack>", "  <mm_depth>"]

        for item in mm_items:
            question_type = item.get("question_type") or "unknown"
            question_num = item.get("question_num")
            identifier = (
                f"mmq:{item['model']}:{question_num}"
                if question_num is not None
                else f"mmq:{item['model']}:{question_type}"
            )

            lines.append(
                f"    <qa model=\"{item['model']}\" type=\"{question_type}\" id=\"{identifier}\">"
            )
            question = item.get("question") or ""
            lines.append(f"      <question>{question}</question>")
            lines.append("      <keys>")
            for key in item.get("keys", [])[: self.summary_keys]:
                lines.append(f"        - {key}")
            lines.append("      </keys>")
            if item.get("include_full"):
                full = (item.get("answer") or "").strip()
                if full:
                    escaped = full.replace("</", "</ ")
                    lines.append(f"      <details>{escaped}</details>")
            source = item.get("source") or "mental_model_qa_chunks"
            lines.append(f"      <source>{source}</source>")
            lines.append("    </qa>")

        lines.append("  </mm_depth>")
        if nway_entry:
            models_attr = "|".join(nway_entry.get("models", []))
            focus = nway_entry.get("focus") or "interaction"
            directive = nway_entry.get("text", "").strip()
            if directive:
                lines.append(
                    f"  <nway_interaction models=\"{models_attr}\" focus=\"{focus}\">"
                )
                lines.append(f"    {directive}")
                lines.append("  </nway_interaction>")
        lines.append("</depth_pack>")
        lines.append("<instructions>")
        lines.append("  Use the guidance above where relevant. Take your time; then provide the final answer.")
        lines.append("</instructions>")

        return "\n".join(lines)
