#!/usr/bin/env python3
"""
METIS Engagement Manager - Handles engagement lifecycle and Supabase storage
Ensures engagement records exist before data capture operations

Author: METIS Cognitive Platform
Date: 2025
"""

import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from supabase import create_client

logger = logging.getLogger(__name__)


class EngagementManager:
    """Manages engagement records in Supabase for data capture operations"""

    def __init__(self, enable_supabase: bool = True):
        self.enable_supabase = enable_supabase
        self.supabase_client = None
        self.logger = logger

        if self.enable_supabase:
            self._initialize_supabase()

    def _initialize_supabase(self):
        """Initialize Supabase client for engagement storage"""
        try:
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_ANON_KEY")

            if not supabase_url or not supabase_key:
                self.logger.warning(
                    "⚠️ Supabase credentials not found - engagement storage disabled"
                )
                self.enable_supabase = False
                return

            self.supabase_client = create_client(supabase_url, supabase_key)
            self.logger.info("✅ Supabase client initialized for engagement management")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Supabase client: {e}")
            self.enable_supabase = False

    def ensure_engagement_exists(self, engagement_context) -> bool:
        """
        Ensure engagement record exists in database before data capture

        Args:
            engagement_context: EngagementContext object

        Returns:
            bool: True if engagement exists or was created, False on failure
        """
        if not self.enable_supabase or not self.supabase_client:
            self.logger.warning("Supabase not available - skipping engagement creation")
            return False

        try:
            engagement_id = str(engagement_context.engagement_id)

            # First check if engagement already exists
            existing_result = (
                self.supabase_client.table("engagements")
                .select("id")
                .eq("id", engagement_id)
                .execute()
            )

            if existing_result.data and len(existing_result.data) > 0:
                self.logger.info(f"✅ Engagement {engagement_id} already exists")
                return True

            # Create new engagement record
            engagement_data = {
                "id": engagement_id,
                "problem_statement": engagement_context.problem_statement,
                "business_context": engagement_context.business_context,
                "industry": engagement_context.business_context.get(
                    "industry", "unknown"
                ),
                "company_size": engagement_context.business_context.get(
                    "company_size", "unknown"
                ),
                "current_phase": 1,
                "phase_1_complete": False,
                "phase_2_complete": False,
                "phase_3_complete": False,
                "phase_4_complete": False,
                "user_preferences": {},
                "compliance_requirements": {},
                "status": "active",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }

            result = (
                self.supabase_client.table("engagements")
                .insert(engagement_data)
                .execute()
            )

            if result.data and len(result.data) > 0:
                self.logger.info(f"✅ Created engagement record {engagement_id}")
                return True
            else:
                self.logger.error(
                    "❌ Failed to create engagement record - no data returned"
                )
                return False

        except Exception as e:
            self.logger.error(f"❌ Failed to ensure engagement exists: {e}")
            return False

    def update_engagement_phase(
        self, engagement_id: str, phase: int, completed: bool = False
    ) -> bool:
        """Update engagement phase completion status"""
        if not self.enable_supabase or not self.supabase_client:
            return False

        try:
            update_data = {
                "current_phase": phase,
                "updated_at": datetime.now().isoformat(),
            }

            # Set phase completion flags
            if completed:
                if phase == 1:
                    update_data["phase_1_complete"] = True
                elif phase == 2:
                    update_data["phase_2_complete"] = True
                elif phase == 3:
                    update_data["phase_3_complete"] = True
                elif phase == 4:
                    update_data["phase_4_complete"] = True
                    update_data["status"] = "completed"
                    update_data["completed_at"] = datetime.now().isoformat()

            result = (
                self.supabase_client.table("engagements")
                .update(update_data)
                .eq("id", engagement_id)
                .execute()
            )

            if result.data and len(result.data) > 0:
                self.logger.info(f"✅ Updated engagement {engagement_id} phase {phase}")
                return True
            else:
                self.logger.warning(f"⚠️ No engagement found to update: {engagement_id}")
                return False

        except Exception as e:
            self.logger.error(f"❌ Failed to update engagement phase: {e}")
            return False

    def get_engagement(self, engagement_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve engagement record from database"""
        if not self.enable_supabase or not self.supabase_client:
            return None

        try:
            result = (
                self.supabase_client.table("engagements")
                .select("*")
                .eq("id", engagement_id)
                .execute()
            )

            if result.data and len(result.data) > 0:
                return result.data[0]
            else:
                return None

        except Exception as e:
            self.logger.error(f"❌ Failed to retrieve engagement: {e}")
            return None

    def cleanup_test_engagement(self, engagement_id: str) -> bool:
        """Clean up test engagement and associated data"""
        if not self.enable_supabase or not self.supabase_client:
            return False

        try:
            # Delete engagement (CASCADE will handle related data)
            result = (
                self.supabase_client.table("engagements")
                .delete()
                .eq("id", engagement_id)
                .execute()
            )

            self.logger.info(f"✅ Cleaned up test engagement {engagement_id}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to cleanup engagement: {e}")
            return False
