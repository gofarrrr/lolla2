#!/usr/bin/env python3
"""
Summarise structured LLM resiliency logs.

Usage:
    python scripts/summarize_llm_attempts.py --log backend_live.log
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional


def _extract_payload(line: str, marker: str) -> Optional[Dict[str, Any]]:
    if marker not in line:
        return None
    try:
        payload = line.split(marker, 1)[1].strip()
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def summarise_log(lines: Iterable[str]) -> Dict[str, Any]:
    attempt_counts: Counter[str] = Counter()
    success_counts: Counter[str] = Counter()
    failure_counts: Counter[str] = Counter()
    latencies: defaultdict[str, List[float]] = defaultdict(list)
    retry_counts: Counter[str] = Counter()
    fallback_events: List[Dict[str, Any]] = []

    for line in lines:
        attempt = _extract_payload(line, "llm.attempt")
        if attempt:
            provider = attempt.get("provider", "unknown")
            attempt_counts[provider] += 1
            latencies[provider].append(float(attempt.get("latency_ms", 0.0)))
            if attempt.get("status") == "success":
                success_counts[provider] += 1
            else:
                failure_counts[provider] += 1

        retry = _extract_payload(line, "llm.retry")
        if retry:
            provider = retry.get("provider", "unknown")
            retry_counts[provider] += 1

        fallback = _extract_payload(line, "llm.fallback")
        if fallback:
            fallback_events.append(fallback)

    summary: Dict[str, Any] = {
        "providers": [],
        "fallbacks": fallback_events,
    }

    for provider in sorted(attempt_counts):
        attempts = attempt_counts[provider]
        successes = success_counts[provider]
        failures = failure_counts[provider]
        avg_latency = mean(latencies[provider]) if latencies[provider] else 0.0
        summary["providers"].append(
            {
                "provider": provider,
                "attempts": attempts,
                "successes": successes,
                "failures": failures,
                "avg_latency_ms": round(avg_latency, 2),
                "retries": retry_counts[provider],
                "success_rate": round((successes / attempts) * 100, 2) if attempts else 0.0,
            }
        )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise LLM resiliency logs.")
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("backend_live.log"),
        help="Path to the structured log file (default: backend_live.log)",
    )
    args = parser.parse_args()

    if not args.log.exists():
        raise SystemExit(f"Log file not found: {args.log}")

    summary = summarise_log(args.log.read_text().splitlines())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
