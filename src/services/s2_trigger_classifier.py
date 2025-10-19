from enum import Enum
from dataclasses import dataclass
from typing import Dict


class S2Tier(Enum):
    S2_DISABLED = "S2_DISABLED"
    S2_TIER_1 = "S2_TIER_1"  # Light
    S2_TIER_2 = "S2_TIER_2"  # Standard
    S2_TIER_3 = "S2_TIER_3"  # Deep


@dataclass
class S2TriggerDecision:
    tier: S2Tier
    rationale: str


class S2TriggerClassifier:
    def classify(self, query: str, metadata: Dict) -> S2TriggerDecision:
        """
        Classifies the query and metadata to determine the appropriate S2 tier.
        Rules-based v0.2 hardened spec.
        """
        q = (query or "").lower()
        md = metadata or {}

        # Tier 3: Highest stakes
        if md.get("irreversibility") is True or md.get("stakes") == "high":
            return S2TriggerDecision(
                S2Tier.S2_TIER_3, "High stakes or irreversible decision detected."
            )

        # Tier 2: Long chain / novel structure / explicit verification
        if md.get("long_chain", 0) >= 4 or md.get("novel_structure") is True:
            return S2TriggerDecision(
                S2Tier.S2_TIER_2, "Long reasoning chain or novel structure detected."
            )

        verification_keywords = [
            "verify",
            "evidence",
            "audit",
            "validate",
            "proof",
            "check",
        ]
        if any(k in q for k in verification_keywords):
            return S2TriggerDecision(
                S2Tier.S2_TIER_2, "Explicit verification requested."
            )

        # Tier 0: Fast path intent
        fast_path_keywords = ["simple", "quick", "fast", "just tell me", "briefly"]
        if any(k in q for k in fast_path_keywords):
            return S2TriggerDecision(S2Tier.S2_DISABLED, "User requested fast path.")

        # Tier 1: Default
        return S2TriggerDecision(S2Tier.S2_TIER_1, "Default Light S2 activation.")
