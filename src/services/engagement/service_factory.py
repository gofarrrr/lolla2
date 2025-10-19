"""
Engagement Service Factory
==========================

Factory for creating engagement services with dependency injection.

This factory provides a central point for service instantiation,
enabling:
- Dependency injection
- Service composition
- Testing with mocks
- Configuration management
"""

import logging
from typing import Dict, Any, Optional

from src.services.persistence import DatabaseService
from src.services.engagement.engagement_service import EngagementService
from src.services.engagement.report_formatting_service import ReportFormattingService
from src.services.engagement.event_extraction_service import EventExtractionService
from src.services.engagement.outcome_service import OutcomeService

logger = logging.getLogger(__name__)


class EngagementServiceFactory:
    """
    Factory for creating engagement services with proper dependency injection.

    Usage:
        factory = EngagementServiceFactory(database_service, active_engagements)
        engagement_service = factory.create_engagement_service()
        report_service = factory.create_report_formatting_service()
    """

    def __init__(
        self,
        database_service: Optional[DatabaseService] = None,
        active_engagements: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        """
        Initialize the service factory.

        Args:
            database_service: Optional database service for persistence
            active_engagements: Optional in-memory engagement store
        """
        self.database_service = database_service
        self.active_engagements = active_engagements or {}

    def create_engagement_service(self) -> EngagementService:
        """
        Create an EngagementService with injected dependencies.

        Returns:
            Configured EngagementService instance
        """
        return EngagementService(
            database_service=self.database_service,
            active_engagements=self.active_engagements
        )

    def create_report_formatting_service(self) -> ReportFormattingService:
        """
        Create a ReportFormattingService.

        Returns:
            ReportFormattingService instance (stateless)
        """
        return ReportFormattingService()

    def create_event_extraction_service(self) -> EventExtractionService:
        """
        Create an EventExtractionService.

        Returns:
            EventExtractionService instance (stateless)
        """
        return EventExtractionService()

    def create_outcome_service(self) -> OutcomeService:
        """
        Create an OutcomeService.

        Returns:
            OutcomeService instance (stateless)
        """
        return OutcomeService()

    def create_all_services(self) -> Dict[str, Any]:
        """
        Create all engagement services at once.

        Useful for route handlers that need multiple services.

        Returns:
            Dictionary with all service instances:
            {
                'engagement': EngagementService,
                'report_formatting': ReportFormattingService,
                'event_extraction': EventExtractionService,
                'outcome': OutcomeService
            }
        """
        return {
            'engagement': self.create_engagement_service(),
            'report_formatting': self.create_report_formatting_service(),
            'event_extraction': self.create_event_extraction_service(),
            'outcome': self.create_outcome_service()
        }


# Global factory instance (can be replaced for testing)
_factory_instance: Optional[EngagementServiceFactory] = None


def get_service_factory(
    database_service: Optional[DatabaseService] = None,
    active_engagements: Optional[Dict[str, Dict[str, Any]]] = None
) -> EngagementServiceFactory:
    """
    Get the global service factory instance.

    This function provides a singleton factory for the application.
    For testing, you can pass custom dependencies.

    Args:
        database_service: Optional database service (uses global if None)
        active_engagements: Optional in-memory store (uses global if None)

    Returns:
        EngagementServiceFactory instance
    """
    global _factory_instance

    # For testing: allow passing custom dependencies
    if database_service is not None or active_engagements is not None:
        return EngagementServiceFactory(database_service, active_engagements)

    # Use global singleton
    if _factory_instance is None:
        _factory_instance = EngagementServiceFactory()

    return _factory_instance


def reset_factory() -> None:
    """
    Reset the global factory instance.

    Useful for testing to ensure clean state between tests.
    """
    global _factory_instance
    _factory_instance = None
