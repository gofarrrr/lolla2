#!/usr/bin/env python3
"""
Global counters and simple snapshots for dashboard tiles.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict


class _Counter:
    def __init__(self):
        self.counts: Dict[str, int] = defaultdict(int)

    def increment(self, key: str, n: int = 1) -> None:
        self.counts[key] += n

    def snapshot(self) -> Dict[str, int]:
        return dict(self.counts)


# Routing arm selection counts
routing_arm_counter = _Counter()

# Provider fallback counts
fallback_counter = _Counter()  # keys: attempted, succeeded
