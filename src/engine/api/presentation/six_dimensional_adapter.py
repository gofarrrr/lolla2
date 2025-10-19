"""
Presentation Adapter v2: Six-Dimensional Progressive Disclosure (3 analyses + 3 critiques)
Provides a structured, progressive UI model for the Strategic Trio and Devil's Advocate layers.
"""

from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum
from datetime import datetime


class CognitiveLoad(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class LayerContent:
    title: str
    cognitive_load: CognitiveLoad
    key_insights: List[str]
    content: str


@dataclass
class SixDimensionView:
    engagement_id: str
    created_at: str
    trio: Dict[str, LayerContent]  # CEO, CTO, CFO
    critiques: Dict[str, List[LayerContent]]  # For each: list of critique outputs
    navigation: Dict[str, Any]


class SixDimensionalPresentationAdapter:
    """Transforms raw results into a six-dimensional progressive package"""

    def transform(
        self,
        engagement_id: str,
        trio_results: Dict[str, Dict[str, Any]],
        critique_results: Dict[str, List[Dict[str, Any]]],
    ) -> SixDimensionView:
        trio_layers = {}
        critique_layers = {}

        for role, result in trio_results.items():
            trio_layers[role] = self._build_layer(
                title=f"{role} Perspective",
                content=result.get("analysis", ""),
                insights=result.get("key_insights", []),
            )

        for role, critiques in critique_results.items():
            critique_layers[role] = [
                self._build_layer(
                    title=f"{role} Critique #{i+1}",
                    content=c.get("critique", ""),
                    insights=c.get("issues", []),
                )
                for i, c in enumerate(critiques)
            ]

        navigation = {
            "progressive_disclosure": [
                {"layer": "executive_summary", "load": CognitiveLoad.LOW.value},
                {"layer": "deep_dive", "load": CognitiveLoad.MEDIUM.value},
                {"layer": "methodology", "load": CognitiveLoad.HIGH.value},
            ],
            "user_controls": {
                "toggle_critiques": True,
                "request_arbitration": True,
                "bookmark_findings": True,
            },
        }

        return SixDimensionView(
            engagement_id=engagement_id,
            created_at=datetime.utcnow().isoformat(),
            trio={k: v for k, v in trio_layers.items()},
            critiques={k: v for k, v in critique_layers.items()},
            navigation=navigation,
        )

    def _build_layer(
        self, title: str, content: str, insights: List[str]
    ) -> LayerContent:
        load = self._assess_cognitive_load(content)
        return LayerContent(
            title=title, cognitive_load=load, key_insights=insights[:5], content=content
        )

    def _assess_cognitive_load(self, content: str) -> CognitiveLoad:
        length = len(content)
        if length < 800:
            return CognitiveLoad.LOW
        elif length < 2500:
            return CognitiveLoad.MEDIUM
        else:
            return CognitiveLoad.HIGH
