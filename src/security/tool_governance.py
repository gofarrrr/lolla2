"""
Tool governance: allowlist and HITL gating for high-risk tools.
Integrate by calling `enforce_tool_policy(tool_name, params)` before execution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class ToolPolicy:
    name: str
    allowed: bool
    risk: str  # low | medium | high
    requires_hitl: bool = False


# Default allowlist (safe by default; tighten in prod)
DEFAULT_TOOL_POLICIES: Dict[str, ToolPolicy] = {
    "perplexity_query_knowledge": ToolPolicy("perplexity_query_knowledge", True, "medium", False),
    "perplexity_deep_research": ToolPolicy("perplexity_deep_research", True, "high", True),
}


class ToolGovernance:
    def __init__(self, policies: Optional[Dict[str, ToolPolicy]] = None):
        self.policies = policies or DEFAULT_TOOL_POLICIES

    def enforce_tool_policy(
        self,
        tool_name: str,
        params: Dict,
        *,
        allow_high_risk: bool = False,
        user_confirmed: bool = False,
    ) -> None:
        policy = self.policies.get(tool_name)
        if not policy:
            raise PermissionError(f"Tool not allowlisted: {tool_name}")
        if not policy.allowed:
            raise PermissionError(f"Tool explicitly disallowed: {tool_name}")

        # Require explicit approval for high-risk tools
        if policy.risk == "high" and policy.requires_hitl:
            if not (allow_high_risk and user_confirmed):
                raise PermissionError(
                    f"HITL required for high-risk tool: {tool_name} (set allow_high_risk & user_confirmed)"
                )

        # Basic param sanitation (placeholder)
        if any(k in params for k in ("__import__", "os.system", "subprocess")):
            raise PermissionError("Potentially dangerous parameter detected")

        logger.info(
            f"🛡️ Tool policy OK: {tool_name} (risk={policy.risk}, hitl={policy.requires_hitl})"
        )


# Singleton
_governance: Optional[ToolGovernance] = None


def get_tool_governance() -> ToolGovernance:
    global _governance
    if _governance is None:
        _governance = ToolGovernance()
    return _governance
