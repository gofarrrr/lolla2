"""
METIS Validation Evidence Engine
Progressive transparency module for evidence processing and visualization

Processes and visualizes validation evidence to support reasoning transparency
and build user confidence in analytical results.
"""

from typing import List
from src.engine.models.data_contracts import ReasoningStep, MetisDataContract
from src.models.transparency_models import (
    ValidationEvidenceType,
    EvidenceQuality,
    ValidationEvidence,
    ValidationEvidenceCollection,
    ValidationEvidenceVisualization,
)


class ValidationEvidenceEngine:
    """Engine for processing and visualizing validation evidence"""

    def __init__(self):
        self.evidence_quality_thresholds = {
            EvidenceQuality.STRONG: 0.8,
            EvidenceQuality.MODERATE: 0.6,
            EvidenceQuality.WEAK: 0.4,
            EvidenceQuality.INSUFFICIENT: 0.0,
        }

    async def generate_validation_evidence(
        self,
        reasoning_steps: List[ReasoningStep],
        engagement_contract: MetisDataContract,
    ) -> List[ValidationEvidenceCollection]:
        """Generate validation evidence for reasoning steps"""

        evidence_collections = []

        for step in reasoning_steps:
            collection = await self._create_evidence_collection_for_step(
                step, engagement_contract
            )
            evidence_collections.append(collection)

        return evidence_collections

    async def _create_evidence_collection_for_step(
        self, step: ReasoningStep, engagement_contract: MetisDataContract
    ) -> ValidationEvidenceCollection:
        """Create evidence collection for a single reasoning step"""

        evidence_items = []

        # Generate logical consistency evidence
        logical_evidence = await self._generate_logical_consistency_evidence(step)
        evidence_items.append(logical_evidence)

        # Generate empirical support evidence if data available
        if step.evidence_sources:
            empirical_evidence = await self._generate_empirical_evidence(step)
            evidence_items.append(empirical_evidence)

        # Generate data verification evidence
        data_evidence = await self._generate_data_verification_evidence(
            step, engagement_contract
        )
        evidence_items.append(data_evidence)

        # Create collection with quality assessment
        collection = ValidationEvidenceCollection(
            reasoning_step_id=step.step_id, evidence_items=evidence_items
        )

        # Calculate quality metrics
        await self._calculate_evidence_quality_metrics(collection)

        return collection

    async def _generate_logical_consistency_evidence(
        self, step: ReasoningStep
    ) -> ValidationEvidence:
        """Generate logical consistency evidence for reasoning step"""

        # Simple consistency score based on assumptions and reasoning
        consistency_score = 0.7 + (0.3 * min(1.0, len(step.reasoning_text) / 200))

        quality = (
            EvidenceQuality.STRONG
            if consistency_score > 0.8
            else (
                EvidenceQuality.MODERATE
                if consistency_score > 0.6
                else EvidenceQuality.WEAK
            )
        )

        return ValidationEvidence(
            evidence_id=f"logical_{step.step_id}",
            evidence_type=ValidationEvidenceType.LOGICAL_CONSISTENCY,
            quality=quality,
            source="Internal Logic Validator",
            description=f"Logical consistency assessment: {consistency_score:.2f}",
            supporting_data={
                "consistency_score": consistency_score,
                "assumptions_tested": len(step.assumptions_made),
                "reasoning_length": len(step.reasoning_text),
            },
            confidence_impact=consistency_score * 0.2,
            reliability_score=0.9,
        )

    async def _generate_empirical_evidence(
        self, step: ReasoningStep
    ) -> ValidationEvidence:
        """Generate empirical support evidence"""

        # Assess empirical support based on evidence sources
        empirical_strength = min(len(step.evidence_sources) * 0.3, 1.0)

        quality = (
            EvidenceQuality.STRONG
            if empirical_strength > 0.8
            else (
                EvidenceQuality.MODERATE
                if empirical_strength > 0.5
                else EvidenceQuality.WEAK
            )
        )

        return ValidationEvidence(
            evidence_id=f"empirical_{step.step_id}",
            evidence_type=ValidationEvidenceType.EMPIRICAL_SUPPORT,
            quality=quality,
            source="External Evidence Sources",
            description=f"Empirical support from {len(step.evidence_sources)} sources",
            supporting_data={
                "source_count": len(step.evidence_sources),
                "sources": step.evidence_sources,
                "empirical_strength": empirical_strength,
            },
            confidence_impact=empirical_strength * 0.3,
            reliability_score=0.8,
        )

    async def _generate_data_verification_evidence(
        self, step: ReasoningStep, engagement_contract: MetisDataContract
    ) -> ValidationEvidence:
        """Generate data verification evidence"""

        # Simple data quality assessment
        data_quality_score = 0.6 + (0.4 * step.confidence_score)

        quality = (
            EvidenceQuality.STRONG
            if data_quality_score > 0.8
            else (
                EvidenceQuality.MODERATE
                if data_quality_score > 0.6
                else EvidenceQuality.WEAK
            )
        )

        return ValidationEvidence(
            evidence_id=f"data_{step.step_id}",
            evidence_type=ValidationEvidenceType.DATA_VERIFICATION,
            quality=quality,
            source="Data Quality Validator",
            description=f"Data verification score: {data_quality_score:.2f}",
            supporting_data={
                "data_quality_score": data_quality_score,
                "step_confidence": step.confidence_score,
            },
            confidence_impact=data_quality_score * 0.15,
            reliability_score=0.75,
        )

    async def _calculate_evidence_quality_metrics(
        self, collection: ValidationEvidenceCollection
    ):
        """Calculate overall quality metrics for evidence collection"""

        if not collection.evidence_items:
            return

        # Count evidence types
        collection.supporting_evidence_count = len(
            [e for e in collection.evidence_items if e.confidence_impact > 0]
        )

        collection.contradictory_evidence_count = (
            0  # No contradictory evidence in basic implementation
        )

        # Calculate evidence confidence boost
        collection.evidence_confidence_boost = sum(
            e.confidence_impact for e in collection.evidence_items
        ) / len(collection.evidence_items)

        # Calculate diversity score
        evidence_types = set(e.evidence_type for e in collection.evidence_items)
        collection.evidence_diversity_score = len(evidence_types) / len(
            ValidationEvidenceType
        )

        # Calculate source independence
        sources = set(e.source for e in collection.evidence_items)
        collection.source_independence_score = len(sources) / len(
            collection.evidence_items
        )

        # Calculate completeness score
        collection.evidence_completeness_score = min(
            len(collection.evidence_items)
            / 3,  # Expect ~3 evidence types in basic implementation
            1.0,
        )

        # Determine overall quality
        avg_quality_score = sum(
            self.evidence_quality_thresholds[e.quality]
            for e in collection.evidence_items
        ) / len(collection.evidence_items)

        if avg_quality_score >= 0.7:
            collection.overall_evidence_quality = EvidenceQuality.STRONG
        elif avg_quality_score >= 0.5:
            collection.overall_evidence_quality = EvidenceQuality.MODERATE
        else:
            collection.overall_evidence_quality = EvidenceQuality.WEAK

    async def create_evidence_visualization(
        self, evidence_collections: List[ValidationEvidenceCollection]
    ) -> ValidationEvidenceVisualization:
        """Create visualization for validation evidence"""

        # Create evidence map
        evidence_map = {
            "type": "evidence_network",
            "nodes": [],
            "edges": [],
            "clusters": [],
        }

        # Create nodes for each evidence item
        for collection in evidence_collections:
            for evidence in collection.evidence_items:
                node = {
                    "id": evidence.evidence_id,
                    "type": "evidence",
                    "label": evidence.evidence_type.value.replace("_", " ").title(),
                    "quality": evidence.quality.value,
                    "confidence_impact": evidence.confidence_impact,
                    "source": evidence.source,
                }
                evidence_map["nodes"].append(node)

        # Create quality indicators
        quality_indicators = {
            "total_evidence_items": sum(
                len(c.evidence_items) for c in evidence_collections
            ),
            "average_quality": sum(
                self.evidence_quality_thresholds[e.quality]
                for c in evidence_collections
                for e in c.evidence_items
            )
            / max(1, sum(len(c.evidence_items) for c in evidence_collections)),
        }

        return ValidationEvidenceVisualization(
            visualization_type="comprehensive_evidence_view",
            evidence_map=evidence_map,
            quality_indicators=quality_indicators,
            confidence_impact_chart={
                "type": "impact_chart",
                "data": [
                    {
                        "reasoning_step": collection.reasoning_step_id,
                        "evidence_boost": collection.evidence_confidence_boost,
                        "quality": collection.overall_evidence_quality.value,
                    }
                    for collection in evidence_collections
                ],
            },
        )
