#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import hashlib
from datetime import datetime, timezone
from typing import Optional

_PRIORS_CACHE = None


def load_priors() -> dict:
    global _PRIORS_CACHE
    if _PRIORS_CACHE is not None:
        return _PRIORS_CACHE
    # Try repo config; fallback to built-in defaults
    path = os.path.join(os.getcwd(), "config", "credibility_priors.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            _PRIORS_CACHE = json.load(f)
    except Exception:
        _PRIORS_CACHE = {
            "gov": 0.95,
            "edu": 0.9,
            "org": 0.7,
            "wikipedia.org": 0.6,
            "default_blog": 0.4,
            "internal_rag_mm1": 0.98,
            "default": 0.5,
        }
    return _PRIORS_CACHE


def compute_credibility_weight(source_url: Optional[str], source_type: Optional[str] = None) -> float:
    priors = load_priors()
    if not source_url:
        # Internal or unknown source
        if source_type == "primary":
            return priors.get("internal_rag_mm1", 0.98)
        return priors.get("default", 0.5)

    try:
        url = source_url.lower()
        if "wikipedia.org" in url:
            return priors.get("wikipedia.org", 0.6)
        if url.endswith(".gov") or ".gov/" in url:
            return priors.get("gov", 0.95)
        if url.endswith(".edu") or ".edu/" in url:
            return priors.get("edu", 0.9)
        if url.endswith(".org") or ".org/" in url:
            return priors.get("org", 0.7)
        if any(ext in url for ext in [".medium.com", "blog.", "/blog/"]):
            return priors.get("default_blog", 0.4)
    except Exception:
        pass
    return priors.get("default", 0.5)


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_fetch_hash(content: str, salt: Optional[str] = None) -> str:
    h = hashlib.sha256()
    payload = (content or "") + (salt or "")
    h.update(payload.encode("utf-8", errors="ignore"))
    return h.hexdigest()

