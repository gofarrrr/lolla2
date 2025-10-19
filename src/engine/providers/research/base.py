"""
Research Provider Abstract Base Class
Defines the standard interface for all research providers in Operation "Research Resilience"
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum


class ResearchTier(str, Enum):
    """Research tiers for cost optimization"""

    TESTING = "testing"  # Cheapest for tests and development
    REGULAR = "regular"  # Standard production use
    PREMIUM = "premium"  # Deep research for complex queries


@dataclass
class ResearchResult:
    """Standardized research result across all providers"""

    content: str
    sources: List[str]
    confidence: float
    processing_time_ms: float
    tokens_used: int
    cost_usd: float
    provider_name: str
    citations: List[Dict[str, str]]
    timestamp: datetime
    query_type: str = ""
    success: bool = True
    error_message: Optional[str] = None
    raw_provider_response: Optional[Dict[str, Any]] = (
        None  # RADICAL TRANSPARENCY: Complete API response
    )


class ResearchProvider(ABC):
    """
    Abstract base class for all research providers

    This interface standardizes research provider interactions to enable:
    1. Provider-agnostic research execution
    2. Fallback resilience between providers
    3. Cost optimization across different providers
    4. Transparent provider switching
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of this research provider"""
        pass

    @abstractmethod
    async def search(
        self,
        query: str,
        num_results: int = 5,
        tier: ResearchTier = ResearchTier.REGULAR,
    ) -> ResearchResult:
        """
        Execute a search query and return standardized results

        Args:
            query: The research question to search for
            num_results: Number of results to return
            tier: Research tier for cost/quality optimization

        Returns:
            ResearchResult: Standardized result structure

        Raises:
            Exception: If the search fails
        """
        pass

    @abstractmethod
    async def is_available(self) -> bool:
        """
        Check if this provider is available and working

        Returns:
            bool: True if provider is available, False otherwise
        """
        pass

    @abstractmethod
    def get_usage_metrics(self) -> Dict[str, Any]:
        """
        Get current usage metrics for this provider

        Returns:
            Dict containing usage statistics, costs, and performance metrics
        """
        pass

    async def test_connection(self) -> Dict[str, Any]:
        """
        Test the provider connection and return diagnostics

        Returns:
            Dict containing diagnostic information
        """
        try:
            is_available = await self.is_available()
            return {
                "provider": self.provider_name,
                "available": is_available,
                "error": None,
            }
        except Exception as e:
            return {"provider": self.provider_name, "available": False, "error": str(e)}
