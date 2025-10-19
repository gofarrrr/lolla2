"""
METIS Pyramid Principle Synthesis Engine - Refactored
Core engine implementing Barbara Minto's Pyramid Principle
"""

import logging
from typing import Dict, Any

from src.core.enhanced_event_bus import (
    EnhancedKafkaEventBus as MetisEventBus,
    CloudEvent,
)

# State manager with fallback for development
try:
    from src.core.state_management import DistributedStateManager, StateType

    STATE_MANAGER_AVAILABLE = True
except Exception:
    STATE_MANAGER_AVAILABLE = False

    # Mock state manager for development
    class MockStateManager:
        def __init__(self):
            self.data = {}

        async def set_state(self, key, value, state_type=None):
            self.data[key] = value

        async def get_state(self, key):
            return self.data.get(key)

    DistributedStateManager = MockStateManager
    StateType = None

from .models import ExecutiveDeliverable
from .enums import DeliverableType
from .builders import PyramidBuilder
from .quality import QualityAssessor
from .formatters import DeliverableFormatter


class PyramidEngine:
    """
    Core engine implementing Barbara Minto's Pyramid Principle
    Structures analysis into clear, logical, persuasive communications
    """

    def __init__(
        self, state_manager: DistributedStateManager, event_bus: MetisEventBus
    ):
        self.state_manager = state_manager
        self.event_bus = event_bus

        # Initialize helper components
        self.pyramid_builder = PyramidBuilder()
        self.quality_assessor = QualityAssessor()
        self.formatter = DeliverableFormatter()

        self.logger = logging.getLogger(__name__)

    async def synthesize_engagement_deliverable(
        self,
        engagement_data: Dict[str, Any],
        deliverable_type: DeliverableType = DeliverableType.EXECUTIVE_SUMMARY,
    ) -> ExecutiveDeliverable:
        """
        Create executive deliverable from engagement analysis
        """

        self.logger.info(f"Synthesizing {deliverable_type.value} deliverable")

        # Extract key components from engagement data
        insights = engagement_data.get("insights", [])
        hypotheses = engagement_data.get("hypotheses", [])
        frameworks_results = engagement_data.get("frameworks_results", [])
        analysis_findings = engagement_data.get("analysis_findings", {})

        # Build pyramid structure
        pyramid = await self.pyramid_builder.build_pyramid_structure(
            insights, hypotheses, frameworks_results, analysis_findings
        )

        # Generate deliverable content
        deliverable = await self.formatter.generate_deliverable_content(
            pyramid, deliverable_type, engagement_data
        )

        # Assess quality and partner-readiness
        await self.quality_assessor.assess_deliverable_quality(deliverable)

        # Store deliverable
        await self.state_manager.set_state(
            f"deliverable_{deliverable.deliverable_id}",
            self.formatter.serialize_deliverable(deliverable),
            StateType.DELIVERABLE if STATE_MANAGER_AVAILABLE else None,
        )

        # Emit completion event
        await self.event_bus.publish_event(
            CloudEvent(
                type="pyramid.synthesis.completed",
                source="pyramid/engine",
                data={
                    "deliverable_id": str(deliverable.deliverable_id),
                    "type": deliverable_type.value,
                    "partner_ready_score": deliverable.partner_ready_score,
                    "word_count": len(deliverable.executive_summary.split()),
                },
            )
        )

        return deliverable
