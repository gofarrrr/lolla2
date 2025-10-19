#!/usr/bin/env python3
"""
Core Metrics Dashboard service
Aggregates live tiles for use by dashboards.
"""

from __future__ import annotations

from typing import Dict, List

from src.telemetry.dashboard_tiles import memory_tile, fallback_tile, routing_split_tile


def get_core_metrics() -> Dict[str, object]:
    tiles = {
        "memory": memory_tile(),
        "fallback": fallback_tile(),
        "routing": routing_split_tile(),
    }
    return tiles
