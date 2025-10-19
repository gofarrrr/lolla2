#!/usr/bin/env python3
"""
Measure HTTP traffic hitting legacy API routes.

Usage:
    LOG_FILE=/var/log/app/access.log scripts/measure_route_traffic.py
"""

from __future__ import annotations

import os
import re
import sys
from collections import Counter
from pathlib import Path


LOG_FILE = Path(os.environ.get("LOG_FILE", "/var/log/app/access.log"))
ROUTE_PATTERN = re.compile(r'"[A-Z]+\s+(/api/[^"\s]*)')


def main() -> int:
    if not LOG_FILE.exists():
        print(f"Log file not found: {LOG_FILE}", file=sys.stderr)
        return 1

    counts: Counter[str] = Counter()

    with LOG_FILE.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            match = ROUTE_PATTERN.search(line)
            if match:
                counts[match.group(1)] += 1

    if not counts:
        print("No routes detected in log window.")
        return 0

    print("Top 20 routes by request count:")
    for route, count in counts.most_common(20):
        print(f"{count:10d}  {route}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
