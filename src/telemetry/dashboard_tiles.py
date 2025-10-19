#!/usr/bin/env python3
"""
Dashboard tiles for quick status: memory hit-rate, fallback rate, routing split.
"""

from __future__ import annotations

from typing import Dict

from src.engine.memory.memory_v2 import get_memory_v2
from src.telemetry.counters import routing_arm_counter, fallback_counter


def memory_tile() -> Dict:
    stats = get_memory_v2().stats()
    return {
        "title": "Memory Hit-Rate",
        "value": round(stats.get("hit_rate", 0.0), 3),
        "extra": {
            "docs": int(stats.get("docs", 0)),
            "queries": int(stats.get("queries", 0)),
            "hits": int(stats.get("hits", 0)),
        },
    }


def fallback_tile() -> Dict:
    snap = fallback_counter.snapshot()
    attempted = int(snap.get("attempted", 0))
    succeeded = int(snap.get("succeeded", 0))
    rate = (succeeded / attempted) if attempted else 0.0
    return {
        "title": "Fallback Success Rate",
        "value": round(rate, 3),
        "extra": {"attempted": attempted, "succeeded": succeeded},
    }


def routing_split_tile() -> Dict:
    snap = routing_arm_counter.snapshot()
    total = max(1, sum(int(v) for v in snap.values()))
    return {
        "title": "Routing Arm Split",
        "value": {k: round(v / total, 3) for k, v in snap.items()},
        "extra": {"total": total},
    }
