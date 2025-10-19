"""
Canonical JSON serialization for stable hashing and provenance.

Addresses: Hash stability issues with default=str, set ordering, datetime encoding.
"""

import datetime
import json
import math
from typing import Any


def normalize(obj: Any) -> Any:
    """
    Recursively normalize objects to stable, JSON-safe representations.

    Handles:
    - Dicts: sorted by key
    - Sets: converted to sorted lists
    - Datetimes: ISO 8601 UTC with 'Z' suffix
    - NaN floats: explicit "NaN" string
    - Tuples/lists: normalized recursively
    """
    if isinstance(obj, dict):
        return {k: normalize(obj[k]) for k in sorted(obj)}

    if isinstance(obj, (list, tuple)):
        return [normalize(v) for v in obj]

    if isinstance(obj, set):
        # Convert set to sorted list for stable encoding
        return sorted(normalize(v) for v in obj)

    if isinstance(obj, datetime.datetime):
        # Always UTC, ISO 8601 with Z suffix
        utc_dt = obj.astimezone(datetime.timezone.utc)
        return utc_dt.isoformat().replace("+00:00", "Z")

    if isinstance(obj, float):
        if math.isnan(obj):
            return "NaN"
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"

    # UUID, Decimal, other objects with stable __str__
    if hasattr(obj, "__str__") and not isinstance(obj, (str, bytes)):
        return str(obj)

    return obj


def canonical_json(obj: Any) -> str:
    """
    Create canonical JSON string for stable hashing.

    Properties:
    - Deterministic key ordering
    - Stable type conversions
    - No whitespace variations
    - Handles edge cases (NaN, sets, datetimes)

    Example:
        >>> from hashlib import sha256
        >>> hash1 = sha256(canonical_json({"b": 1, "a": 2}).encode()).hexdigest()
        >>> hash2 = sha256(canonical_json({"a": 2, "b": 1}).encode()).hexdigest()
        >>> assert hash1 == hash2  # Order doesn't matter
    """
    normalized = normalize(obj)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False
    )


def canonical_hash(obj: Any) -> str:
    """
    Create SHA256 hash of canonical JSON representation.

    Stable across:
    - Dict key ordering
    - Set element ordering
    - Datetime timezone representations
    - Float edge cases (NaN, Inf)
    """
    import hashlib
    json_str = canonical_json(obj)
    return hashlib.sha256(json_str.encode("utf-8")).hexdigest()
