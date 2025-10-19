"""Alias mapping for mental model names -> canonical snake_case names.

Two mechanisms:
- Static aliases for common variants (e.g., 80/20 rule -> pareto_principle)
- Soft normalization (lowercase, spaces->_, strip possessives/plurals)
"""

from __future__ import annotations

import re
from typing import Dict


STATIC_ALIASES: Dict[str, str] = {
    "80/20": "pareto_principle",
    "80/20 rule": "pareto_principle",
    "eighty-twenty": "pareto_principle",
    "jobs to be done": "jobs_to_be_done",
    "jtbd": "jobs_to_be_done",
    "bayes": "bayesian",
    "bayes theorem": "bayesian",
    "hanlon's razor": "hanlons_razor",
    "hanlon razor": "hanlons_razor",
    "decision tree": "decision_trees",
    "decision trees": "decision_trees",
    "markov chain": "markov_chains",
    "markov chains": "markov_chains",
    "switching cost": "switching_costs",
    "switching costs": "switching_costs",
}


def _normalize(text: str) -> str:
    # Remove possessives, punctuation, compress whitespace
    t = text.lower()
    t = re.sub(r"'s\b", "", t)
    t = re.sub(r"[^a-z0-9]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _to_snake(text: str) -> str:
    return _normalize(text).replace(" ", "_")


def resolve_alias(model_name: str) -> str:
    """Resolve free-form model name to canonical snake_case with alias mapping."""
    key = _normalize(model_name)
    if key in STATIC_ALIASES:
        return STATIC_ALIASES[key]
    # Fallback to snake-case normalization
    return _to_snake(model_name)

