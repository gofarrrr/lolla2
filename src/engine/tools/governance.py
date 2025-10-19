#!/usr/bin/env python3
"""
Tool governance: registry and HITL rules.
Flag: TOOL_GOVERNANCE to enforce; HITL required for risky tools unless bypassed.
"""

from __future__ import annotations

import os
from typing import Dict, Set


class ToolRegistry:
    def __init__(self):
        self.allowlist: Set[str] = set(
            os.getenv("TOOL_ALLOWLIST", "web_search,fetch_url,vector_query").split(",")
        )
        self.risky: Set[str] = set(
            os.getenv("TOOL_RISKY", "code_exec,db_write,shell_exec").split(",")
        )

    def is_allowed(self, name: str) -> bool:
        return name in self.allowlist

    def requires_hitl(self, name: str) -> bool:
        return name in self.risky


tool_registry = ToolRegistry()


def enforce_tool(name: str) -> None:
    """Raise if tool not allowed or requires HITL without bypass."""
    enforce = os.getenv("FF_TOOL_GOVERNANCE", "false").lower() in ("1", "true", "yes", "on")
    if not enforce:
        return
    if not tool_registry.is_allowed(name):
        raise PermissionError(f"Tool '{name}' is not in allowlist")
    if tool_registry.requires_hitl(name):
        if os.getenv("HITL_BYPASS", "").lower() not in ("1", "true", "yes", "on"):
            raise PermissionError(f"Tool '{name}' requires HITL; set HITL_BYPASS to override in dev")
