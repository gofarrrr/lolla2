#!/usr/bin/env python3
"""
Routing A/B scaffold with policy vs. learned arms.
Flag: ROUTING_AB enables alternate arm selection.
"""

from __future__ import annotations

import hashlib
from typing import List, Dict, Any

from src.engine.services.llm.provider_policy import get_provider_chain_for_phase
from src.telemetry.counters import routing_arm_counter


class RoutingAB:
    def __init__(self):
        pass

    @staticmethod
    def _assign_arm(key: str) -> str:
        h = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
        return "policy" if (h % 2 == 0) else "learned"

    def get_chain(self, *, phase: str | None, prompt_hash: str, policy_chain: List[str]) -> List[str]:
        arm = self._assign_arm(prompt_hash)
        routing_arm_counter.increment(arm)
        if arm == "policy":
            return policy_chain
        # Learned (placeholder): prefer cheaper then primary
        learned = policy_chain[:]
        # Move deepseek earlier if present
        if "deepseek" in learned:
            learned.remove("deepseek")
            learned.insert(0, "deepseek")
        return learned


_routing_ab = RoutingAB()


def get_routing_ab() -> RoutingAB:
    return _routing_ab
