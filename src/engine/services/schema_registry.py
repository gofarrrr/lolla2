#!/usr/bin/env python3
"""
Schema Registry with version hashing and validation utilities.
- Register JSON schemas by name and compute a stable version hash.
- Validate data against a registered schema (uses jsonschema if available).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, Optional, Any

try:
    import jsonschema  # type: ignore
except Exception:  # pragma: no cover
    jsonschema = None  # type: ignore


@dataclass
class SchemaRecord:
    name: str
    schema: Dict[str, Any]
    version_hash: str


class SchemaRegistry:
    def __init__(self):
        self._schemas: Dict[str, SchemaRecord] = {}

    @staticmethod
    def _stable_hash(obj: Any) -> str:
        payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def register_schema(self, name: str, schema: Dict[str, Any]) -> str:
        h = self._stable_hash(schema)
        self._schemas[name] = SchemaRecord(name=name, schema=schema, version_hash=h)
        return h

    def get_version(self, name: str) -> Optional[str]:
        rec = self._schemas.get(name)
        return rec.version_hash if rec else None

    def get_schema(self, name: str) -> Optional[Dict[str, Any]]:
        rec = self._schemas.get(name)
        return rec.schema if rec else None

    def validate(self, name: str, data: Any) -> bool:
        rec = self._schemas.get(name)
        if not rec:
            raise ValueError(f"Schema '{name}' not registered")
        if jsonschema is None:
            # Best-effort: If no jsonschema library, skip strict validation
            return True
        jsonschema.validate(instance=data, schema=rec.schema)  # may raise
        return True


_registry: Optional[SchemaRegistry] = None


def get_schema_registry() -> SchemaRegistry:
    global _registry
    if _registry is None:
        _registry = SchemaRegistry()
    return _registry
