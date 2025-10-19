#!/usr/bin/env python3
"""
Prompt Version Registry
- Tracks prompt template versions and bindings to provider/model pairs.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class PromptRecord:
    name: str
    version_hash: str
    template: str


class PromptVersionRegistry:
    def __init__(self):
        self._prompts: Dict[str, PromptRecord] = {}
        self._bindings: Dict[str, Dict[str, str]] = {}  # name -> {"provider": p, "model": m}

    @staticmethod
    def _hash(template: str) -> str:
        return hashlib.sha256(template.encode("utf-8")).hexdigest()[:16]

    def register_prompt(self, name: str, template: str) -> str:
        h = self._hash(template)
        self._prompts[name] = PromptRecord(name=name, version_hash=h, template=template)
        return h

    def bind(self, name: str, provider: str, model: str) -> None:
        self._bindings[name] = {"provider": provider, "model": model}

    def get_version(self, name: str) -> Optional[str]:
        rec = self._prompts.get(name)
        return rec.version_hash if rec else None

    def get_binding(self, name: str) -> Optional[Dict[str, str]]:
        return self._bindings.get(name)


_registry: Optional[PromptVersionRegistry] = None


def get_prompt_version_registry() -> PromptVersionRegistry:
    global _registry
    if _registry is None:
        _registry = PromptVersionRegistry()
    return _registry
