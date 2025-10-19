# src/evaluation/judges/__init__.py
"""
Binary judges for automated evaluation of trace snapshots.

Each judge implements a simple interface:
- Input: PII-safe trace snapshot (dict)
- Output: boolean (True = pass, False = fail)
"""

from typing import Dict, Any

TraceSnapshot = Dict[str, Any]