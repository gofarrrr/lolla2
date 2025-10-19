"""
Presentation Adapter for Progressive Disclosure Frontend
Operation Ground Truth - Day 2: API Payload Transformation

Transforms flat backend JSON into hierarchical ProgressiveDisclosure structure
that the frontend expects, with cognitive load calculations.
"""

from typing import Dict, Any, List
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime
import json

# Import data contract if available
try:
    from src.engine.models.data_contracts import MetisDataContract
except ImportError:
    MetisDataContract = None


class CognitiveLoad(str, Enum):
    """Cognitive load levels for disclosure layers"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class DisclosureChunk:
    """Individual chunk within a disclosure layer"""

    id: str
    content: str
    type: str  # "summary", "detail", "evidence", "technical"
    cognitive_weight: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DisclosureLayer:
    """Single layer in progressive disclosure hierarchy"""

    layer: int
    title: str
    chunks: List[DisclosureChunk]
    cognitive_load: CognitiveLoad
    auto_expand: bool = False
    prerequisites: List[str] = None
    estimated_read_time_seconds: int = 30

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer,
            "title": self.title,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "cognitive_load": self.cognitive_load.value,
            "auto_expand": self.auto_expand,
            "prerequisites": self.prerequisites or [],
            "estimated_read_time_seconds": self.estimated_read_time_seconds,
        }


class PresentationAdapter:
    """
    Transforms backend MetisDataContract or flat JSON into
    frontend-compatible ProgressiveDisclosure structure.

    Operation Ground Truth requirement: Map flat backend to layered frontend
    """

    def transform_for_progressive_disclosure(
        self, backend_response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Transform backend response into progressive disclosure layers.

        Args:
            backend_response: Raw backend JSON or MetisDataContract dict

        Returns:
            Hierarchical structure with disclosure_layers
        """

        # Create the four standard layers
        layers = [
            self._create_executive_layer(backend_response),
            self._create_evidence_layer(backend_response),
            self._create_critique_layer(backend_response),
            self._create_technical_layer(backend_response),
        ]

        # Filter out empty layers
        layers = [layer for layer in layers if layer and layer.chunks]

        return {
            "disclosure_layers": [layer.to_dict() for layer in layers],
            "metadata": self._extract_metadata(backend_response),
            "interaction_hints": self._generate_interaction_hints(layers),
        }

    def _create_executive_layer(self, data: Dict[str, Any]) -> DisclosureLayer:
        """Create Layer 1: Executive Summary (auto-expanded, low cognitive load)"""

        chunks = []

        # Extract executive summary
        if "executive_summary" in data:
            chunks.append(
                DisclosureChunk(
                    id="exec_summary",
                    content=data["executive_summary"],
                    type="summary",
                    cognitive_weight=0.3,
                )
            )
        elif "final_synthesis" in data and isinstance(data["final_synthesis"], dict):
            synthesis = data["final_synthesis"]
            if "sections" in synthesis:
                for section in synthesis["sections"]:
                    if section.get("priority") == "highest":
                        chunks.append(
                            DisclosureChunk(
                                id=f"exec_{section.get('title', 'summary')}",
                                content=str(section.get("content", "")),
                                type="summary",
                                cognitive_weight=0.3,
                            )
                        )
                        break

        # Extract key recommendations
        if "key_recommendations" in data:
            for i, rec in enumerate(data["key_recommendations"][:3]):  # Top 3 only
                chunks.append(
                    DisclosureChunk(
                        id=f"rec_{i}",
                        content=self._format_recommendation(rec),
                        type="summary",
                        cognitive_weight=0.4,
                    )
                )

        return DisclosureLayer(
            layer=1,
            title="Executive Summary",
            chunks=chunks,
            cognitive_load=CognitiveLoad.LOW,
            auto_expand=True,
            prerequisites=[],
            estimated_read_time_seconds=len(chunks) * 15,
        )

    def _create_evidence_layer(self, data: Dict[str, Any]) -> DisclosureLayer:
        """Create Layer 2: Supporting Evidence (medium cognitive load)"""

        chunks = []

        # Extract evidence from research results
        if "research_results" in data:
            research = data["research_results"]
            if "fact_pack" in research and research["fact_pack"]:
                fact_pack = research["fact_pack"]
                if hasattr(fact_pack, "assertions"):
                    # It's an object
                    assertions = fact_pack.assertions[:5]  # Top 5 facts
                elif isinstance(fact_pack, dict) and "assertions" in fact_pack:
                    # It's a dict
                    assertions = fact_pack["assertions"][:5]
                else:
                    assertions = []

                for i, assertion in enumerate(assertions):
                    if isinstance(assertion, dict):
                        claim = assertion.get("claim", "")
                        source = assertion.get("source", "Unknown")
                        confidence = assertion.get("confidence", 0)
                    else:
                        claim = getattr(assertion, "claim", "")
                        source = getattr(assertion, "source", "Unknown")
                        confidence = getattr(assertion, "confidence", 0)

                    chunks.append(
                        DisclosureChunk(
                            id=f"evidence_{i}",
                            content=f"{claim} (Source: {source}, Confidence: {confidence:.0%})",
                            type="evidence",
                            cognitive_weight=0.6,
                        )
                    )

        # Extract evidence from analysis results
        if "evidence_summary" in data:
            summary = data["evidence_summary"]
            chunks.append(
                DisclosureChunk(
                    id="evidence_summary",
                    content=self._format_evidence_summary(summary),
                    type="evidence",
                    cognitive_weight=0.5,
                )
            )

        return DisclosureLayer(
            layer=2,
            title="Supporting Evidence & Research",
            chunks=chunks,
            cognitive_load=CognitiveLoad.MEDIUM,
            auto_expand=False,
            prerequisites=["layer_1"],
            estimated_read_time_seconds=len(chunks) * 20,
        )

    def _create_critique_layer(self, data: Dict[str, Any]) -> DisclosureLayer:
        """Create Layer 3: Devil's Advocate Critiques (high cognitive load)"""

        chunks = []

        # Extract validation results (Red Team critiques)
        if "validation_results" in data:
            validation = data["validation_results"]

            # Munger critique
            if "munger_critique" in validation:
                munger = validation["munger_critique"]
                if "critiques" in munger:
                    for critique in munger["critiques"][:2]:  # Top 2
                        chunks.append(
                            DisclosureChunk(
                                id=f"munger_{critique.get('id', 'unknown')}",
                                content=f"[Failure Mode] {critique.get('description', '')}",
                                type="critique",
                                cognitive_weight=0.8,
                            )
                        )

            # Ackoff critique
            if "ackoff_critique" in validation:
                ackoff = validation["ackoff_critique"]
                if "critiques" in ackoff:
                    for critique in ackoff["critiques"][:2]:  # Top 2
                        chunks.append(
                            DisclosureChunk(
                                id=f"ackoff_{critique.get('id', 'unknown')}",
                                content=f"[Assumption] {critique.get('description', '')}",
                                type="critique",
                                cognitive_weight=0.8,
                            )
                        )

            # Bias audit
            if "bias_audit" in validation:
                bias = validation["bias_audit"]
                if "critiques" in bias:
                    for critique in bias["critiques"][:1]:  # Top 1
                        chunks.append(
                            DisclosureChunk(
                                id=f"bias_{critique.get('id', 'unknown')}",
                                content=f"[Bias] {critique.get('description', '')}",
                                type="critique",
                                cognitive_weight=0.7,
                            )
                        )

        # Extract dissenting views
        if "dissenting_views" in data:
            for i, view in enumerate(data["dissenting_views"][:2]):
                chunks.append(
                    DisclosureChunk(
                        id=f"dissent_{i}",
                        content=str(view),
                        type="critique",
                        cognitive_weight=0.7,
                    )
                )

        return DisclosureLayer(
            layer=3,
            title="Critical Analysis & Alternative Perspectives",
            chunks=chunks,
            cognitive_load=CognitiveLoad.HIGH,
            auto_expand=False,
            prerequisites=["layer_1", "layer_2"],
            estimated_read_time_seconds=len(chunks) * 30,
        )

    def _create_technical_layer(self, data: Dict[str, Any]) -> DisclosureLayer:
        """Create Layer 4: Technical Details (very high cognitive load)"""

        chunks = []

        # Extract processing metadata
        if "metadata" in data:
            meta = data["metadata"]
            chunks.append(
                DisclosureChunk(
                    id="tech_metadata",
                    content=self._format_technical_metadata(meta),
                    type="technical",
                    cognitive_weight=0.9,
                )
            )

        # Extract cognitive state details
        if "cognitive_state" in data:
            cognitive = data["cognitive_state"]
            chunks.append(
                DisclosureChunk(
                    id="tech_cognitive",
                    content=f"Reasoning Steps: {json.dumps(cognitive, indent=2)}",
                    type="technical",
                    cognitive_weight=1.0,
                )
            )

        # Extract validation metadata
        if "validation_metadata" in data:
            val_meta = data["validation_metadata"]
            chunks.append(
                DisclosureChunk(
                    id="tech_validation",
                    content=f"Validation Details: {json.dumps(val_meta, indent=2)}",
                    type="technical",
                    cognitive_weight=0.9,
                )
            )

        # Extract phase metadata
        if "workflow_state" in data:
            workflow = data["workflow_state"]
            chunks.append(
                DisclosureChunk(
                    id="tech_workflow",
                    content=f"Workflow: {json.dumps(workflow, indent=2)}",
                    type="technical",
                    cognitive_weight=0.9,
                )
            )

        return DisclosureLayer(
            layer=4,
            title="Technical Details & Processing Metadata",
            chunks=chunks,
            cognitive_load=CognitiveLoad.VERY_HIGH,
            auto_expand=False,
            prerequisites=["layer_1", "layer_2", "layer_3"],
            estimated_read_time_seconds=len(chunks) * 45,
        )

    def _format_recommendation(self, rec: Any) -> str:
        """Format a recommendation for display"""
        if isinstance(rec, dict):
            title = rec.get("recommendation", rec.get("title", "Recommendation"))
            confidence = rec.get("confidence", 0)
            impact = rec.get("expected_impact", "")
            return f"{title} (Confidence: {confidence:.0%}, Impact: {impact})"
        return str(rec)

    def _format_evidence_summary(self, summary: Dict[str, Any]) -> str:
        """Format evidence summary for display"""
        facts_used = summary.get("facts_used", 0)
        sources = summary.get("sources_cited", 0)
        confidence = summary.get("confidence_level", "UNKNOWN")
        grounding = summary.get("grounding_score", 0)

        return (
            f"Evidence Summary: {facts_used} facts from {sources} sources. "
            f"Confidence: {confidence}, Grounding Score: {grounding:.2f}"
        )

    def _format_technical_metadata(self, meta: Dict[str, Any]) -> str:
        """Format technical metadata for display"""
        return json.dumps(
            {
                "processing_time_ms": meta.get("processing_time_ms", 0),
                "orchestrator_used": meta.get("orchestrator_used", "unknown"),
                "phases_completed": meta.get("phases_completed", 0),
                "parity_score": meta.get("parity_score", 0),
            },
            indent=2,
        )

    def _extract_metadata(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata for frontend consumption"""
        return {
            "engagement_id": data.get("engagement_id", ""),
            "timestamp": data.get("timestamp", datetime.utcnow().isoformat()),
            "orchestrator": data.get("metadata", {}).get(
                "orchestrator_used", "state_machine"
            ),
            "total_processing_time_ms": data.get("metadata", {}).get(
                "processing_time_ms", 0
            ),
            "has_research": bool(data.get("research_results", {}).get("fact_pack")),
            "has_validation": bool(data.get("validation_results")),
            "confidence_level": data.get("evidence_summary", {}).get(
                "confidence_level", "MEDIUM"
            ),
        }

    def _generate_interaction_hints(
        self, layers: List[DisclosureLayer]
    ) -> Dict[str, Any]:
        """Generate hints for frontend interaction behavior"""

        total_chunks = sum(len(layer.chunks) for layer in layers)
        total_read_time = sum(layer.estimated_read_time_seconds for layer in layers)

        return {
            "total_layers": len(layers),
            "total_chunks": total_chunks,
            "estimated_total_read_time_seconds": total_read_time,
            "suggested_exploration_order": [f"layer_{i+1}" for i in range(len(layers))],
            "cognitive_load_distribution": {
                "low": sum(1 for l in layers if l.cognitive_load == CognitiveLoad.LOW),
                "medium": sum(
                    1 for l in layers if l.cognitive_load == CognitiveLoad.MEDIUM
                ),
                "high": sum(
                    1 for l in layers if l.cognitive_load == CognitiveLoad.HIGH
                ),
                "very_high": sum(
                    1 for l in layers if l.cognitive_load == CognitiveLoad.VERY_HIGH
                ),
            },
        }


def transform_contract_to_progressive(contract: Any) -> Dict[str, Any]:
    """
    Convenience function to transform a MetisDataContract to progressive disclosure.

    Args:
        contract: MetisDataContract object or dict representation

    Returns:
        Progressive disclosure structure for frontend
    """
    adapter = PresentationAdapter()

    # Convert contract to dict if it's an object
    if hasattr(contract, "__dict__"):
        data = contract.__dict__
    elif hasattr(contract, "to_dict"):
        data = contract.to_dict()
    else:
        data = contract

    return adapter.transform_for_progressive_disclosure(data)


# Export main components
__all__ = [
    "PresentationAdapter",
    "DisclosureLayer",
    "DisclosureChunk",
    "CognitiveLoad",
    "transform_contract_to_progressive",
]
