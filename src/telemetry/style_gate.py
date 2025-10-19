#!/usr/bin/env python3
"""
Style gate policy evaluator.
Env:
- FF_STYLE_GATE_STRICT: enable gate (warn/block)
- STYLE_GATE_MIN: threshold in [0,1] (default 0.6)
- STYLE_GATE_MODE: warn|block (default warn)
- STYLE_GATE_EXEMPT: comma list of phases exempted
"""

from __future__ import annotations

import os
from typing import Literal, Optional


def evaluate(style_score: Optional[float], *, phase: Optional[str] = None) -> Literal["allow", "warn", "block"]:
    try:
        strict = os.getenv("FF_STYLE_GATE_STRICT", "false").lower() in ("1", "true", "yes", "on")
        if not strict:
            return "allow"
        exempt = set((os.getenv("STYLE_GATE_EXEMPT", "") or "").split(","))
        if phase and phase in exempt:
            return "allow"
        threshold = float(os.getenv("STYLE_GATE_MIN", "0.6"))
        mode = (os.getenv("STYLE_GATE_MODE", "warn").lower()).strip()
        sc = float(style_score or 0.0)
        if sc >= threshold:
            return "allow"
        if mode == "block":
            return "block"
        return "warn"
    except Exception:
        return "allow"
