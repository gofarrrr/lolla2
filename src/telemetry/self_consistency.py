#!/usr/bin/env python3
"""
Shadow self-consistency check (Phase 1)
- Optionally issues a shadow second call and computes simple similarity with primary.
- Returns a float in [0,1] indicating agreement.
"""

from __future__ import annotations

import re
from typing import List, Dict


def _normalize(text: str) -> List[str]:
    # Lowercase, strip, split on non-letters
    words = re.findall(r"[a-zA-Z0-9]+", (text or "").lower())
    return [w for w in words if w]


def _jaccard(a: List[str], b: List[str]) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    uni = len(sa | sb)
    return inter / uni if uni else 0.0


async def run_shadow_check(*, provider_instance, messages: List[Dict], model: str, primary_text: str) -> float:
    # Use slightly higher temperature for shadow to test stability
    shadow_kwargs = {"temperature": 0.7}
    try:
        shadow = await provider_instance.call_llm(messages, model, **shadow_kwargs)
        secondary = shadow.content if hasattr(shadow, "content") else str(shadow)
    except Exception:
        return 0.0

    a = _normalize(primary_text)
    b = _normalize(secondary)
    return float(_jaccard(a, b))
