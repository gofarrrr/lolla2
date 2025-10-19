from __future__ import annotations

from typing import Any, Dict

_CACHE: Dict[str, Dict[str, Any]] = {}


def get_bundle(trace_id: str) -> Dict[str, Any] | None:
    return _CACHE.get(trace_id)


def set_bundle(trace_id: str, bundle: Dict[str, Any], etag: str) -> None:
    _CACHE[trace_id] = {"etag": etag, "data": bundle}

