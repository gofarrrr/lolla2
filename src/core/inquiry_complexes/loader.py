#!/usr/bin/env python3
"""
Inquiry Complex Loader
Loads autonomy-preserving inquiry complexes (curated catalytic questions) from YAML.
"""
from typing import List
import os
import yaml

_BASE_DIR = os.path.dirname(__file__)


def get_inquiry_complex_block(domain: str) -> str:
    """Return a formatted inquiry complex block for the given domain.

    Falls back to an empty string if no YAML found.
    """
    path = os.path.join(_BASE_DIR, f"{domain}.yaml")
    if not os.path.exists(path):
        return ""

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        questions: List[str] = data.get("questions", [])
        if not questions:
            return ""
        lines = ["<inquiry_complex>", "Consider stability under these questions:"]
        for q in questions[:10]:
            lines.append(f"- {q}")
        lines.append("</inquiry_complex>")
        return "\n".join(lines)
    except Exception:
        return ""
