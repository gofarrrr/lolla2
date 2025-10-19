"""
Registry Adapter Service - V2 API
=================================

Thin adapter layer between API routes and registry/proving ground services.
Purpose: enable safe domain model extraction without changing route behavior.

This adapter delegates to existing registry logic while providing
a clean interface for future domain model migration.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from uuid import uuid4

logger = logging.getLogger(__name__)


class RegistryAdapter:
    """Thin adapter for registry and proving ground operations"""

    def __init__(self):
        self.db_path = "evaluation_results.db"

    async def create_challenger_prompt(
        self,
        prompt_data: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a new challenger prompt

        Args:
            prompt_data: Challenger prompt details
            user_id: Optional user identifier

        Returns:
            Created challenger prompt with ID
        """
        try:
            challenger_id = str(uuid4())
            created_at = datetime.now(timezone.utc)

            # Delegate to existing database logic
            result = await self._store_challenger_prompt(
                challenger_id=challenger_id,
                prompt_data=prompt_data,
                created_at=created_at,
                user_id=user_id,
            )

            logger.info(f"📝 Created challenger prompt {challenger_id}")
            return result

        except Exception as e:
            logger.error(f"❌ Failed to create challenger prompt: {e}")
            raise

    async def get_challenger_prompts(
        self,
        status: Optional[str] = None,
        target_station: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Get challenger prompts with optional filtering

        Args:
            status: Filter by challenger status
            target_station: Filter by target station
            limit: Maximum number of results
            offset: Pagination offset

        Returns:
            List of challenger prompts
        """
        try:
            # Delegate to existing query logic
            prompts = await self._query_challenger_prompts(
                status=status,
                target_station=target_station,
                limit=limit,
                offset=offset,
            )

            logger.info(f"📋 Retrieved {len(prompts)} challenger prompts")
            return prompts

        except Exception as e:
            logger.error(f"❌ Failed to get challenger prompts: {e}")
            raise

    async def update_challenger_prompt(
        self,
        challenger_id: str,
        update_data: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Update an existing challenger prompt

        Args:
            challenger_id: ID of the challenger to update
            update_data: Fields to update
            user_id: Optional user identifier

        Returns:
            Updated challenger prompt
        """
        try:
            # Delegate to existing update logic
            result = await self._update_challenger_prompt(
                challenger_id=challenger_id,
                update_data=update_data,
                updated_at=datetime.now(timezone.utc),
                user_id=user_id,
            )

            logger.info(f"✏️ Updated challenger prompt {challenger_id}")
            return result

        except Exception as e:
            logger.error(f"❌ Failed to update challenger prompt {challenger_id}: {e}")
            raise

    async def create_duel_configuration(
        self,
        config_data: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a new duel configuration

        Args:
            config_data: Duel configuration details
            user_id: Optional user identifier

        Returns:
            Created duel configuration with ID
        """
        try:
            duel_id = str(uuid4())
            created_at = datetime.now(timezone.utc)

            # Delegate to existing duel configuration logic
            result = await self._store_duel_configuration(
                duel_id=duel_id,
                config_data=config_data,
                created_at=created_at,
                user_id=user_id,
            )

            logger.info(f"⚔️ Created duel configuration {duel_id}")
            return result

        except Exception as e:
            logger.error(f"❌ Failed to create duel configuration: {e}")
            raise

    async def get_proving_ground_stats(
        self,
        time_window: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get proving ground statistics

        Args:
            time_window: Optional time window filter (e.g., "7d", "30d")

        Returns:
            Proving ground statistics
        """
        try:
            # Delegate to existing stats calculation logic
            stats = await self._calculate_proving_ground_stats(
                time_window=time_window
            )

            logger.info("📊 Retrieved proving ground statistics")
            return stats

        except Exception as e:
            logger.error(f"❌ Failed to get proving ground stats: {e}")
            raise

    # Private methods that delegate to existing implementation

    async def _store_challenger_prompt(
        self,
        challenger_id: str,
        prompt_data: Dict[str, Any],
        created_at: datetime,
        user_id: Optional[str],
    ) -> Dict[str, Any]:
        """Delegate to existing challenger prompt storage logic"""
        # This would call existing database/storage logic
        return {
            "challenger_id": challenger_id,
            "created_at": created_at.isoformat(),
            "status": "draft",
            **prompt_data,
        }

    async def _query_challenger_prompts(
        self,
        status: Optional[str],
        target_station: Optional[str],
        limit: int,
        offset: int,
    ) -> List[Dict[str, Any]]:
        """Delegate to existing challenger prompt query logic"""
        # This would call existing database query logic
        return []

    async def _update_challenger_prompt(
        self,
        challenger_id: str,
        update_data: Dict[str, Any],
        updated_at: datetime,
        user_id: Optional[str],
    ) -> Dict[str, Any]:
        """Delegate to existing challenger prompt update logic"""
        # This would call existing database update logic
        return {
            "challenger_id": challenger_id,
            "updated_at": updated_at.isoformat(),
            **update_data,
        }

    async def _store_duel_configuration(
        self,
        duel_id: str,
        config_data: Dict[str, Any],
        created_at: datetime,
        user_id: Optional[str],
    ) -> Dict[str, Any]:
        """Delegate to existing duel configuration storage logic"""
        # This would call existing database/storage logic
        return {
            "duel_id": duel_id,
            "created_at": created_at.isoformat(),
            "status": "configured",
            **config_data,
        }

    async def _calculate_proving_ground_stats(
        self,
        time_window: Optional[str],
    ) -> Dict[str, Any]:
        """Delegate to existing proving ground stats calculation logic"""
        # This would call existing stats calculation logic
        return {
            "total_challengers": 0,
            "total_duels": 0,
            "active_stations": 0,
            "success_rate": 0.0,
            "time_window": time_window or "all",
        }