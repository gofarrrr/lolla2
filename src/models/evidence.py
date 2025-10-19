# Evidence model (foundation)
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict
import hashlib
import time
import uuid


def fingerprint_content(content: str) -> str:
    h = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return h


@dataclass
class Evidence:
    id: str
    fingerprint: str
    source_type: str  # doc|web|db|human
    source_ref: str  # URL/ID/path
    snippet: str
    reliability: float = 0.5
    created_at: float = field(default_factory=lambda: time.time())
    metadata: Optional[Dict] = None

    @staticmethod
    def new_from_content(
        content: str,
        source_ref: str,
        source_type: str = "doc",
        metadata: Optional[Dict] = None,
    ) -> "Evidence":
        fp = fingerprint_content(content)
        return Evidence(
            id=str(uuid.uuid4()),
            fingerprint=fp,
            source_type=source_type,
            source_ref=source_ref,
            snippet=content[:480],
            reliability=0.5,
            metadata=metadata or {},
        )
