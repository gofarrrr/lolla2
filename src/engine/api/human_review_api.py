"""
METIS V2.1 Human Review SLA System
Tiered human review queue with escalation and intelligent prioritization
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import logging

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/api/human-review", tags=["human-review"])


class ReviewPriority(str, Enum):
    CRITICAL = "critical"  # 4-hour SLA
    HIGH = "high"  # 24-hour SLA
    MEDIUM = "medium"  # 72-hour SLA
    LOW = "low"  # 7-day SLA


class ReviewStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    ESCALATED = "escalated"


class ReviewDecision(str, Enum):
    ACCEPT_NEW = "accept_new"  # Accept new insight, deprecate old
    REJECT_NEW = "reject_new"  # Keep existing wisdom, reject new
    MERGE_BOTH = "merge_both"  # Create merged wisdom entry
    REQUIRE_MORE_DATA = "require_more_data"  # Need additional information


class ReviewRequest(BaseModel):
    contradiction_type: str
    priority_level: ReviewPriority
    conflicting_data: Dict[str, Any]
    context_information: Optional[Dict[str, Any]] = None
    reviewer_notes: Optional[str] = None


class ReviewDecisionRequest(BaseModel):
    review_id: str
    decision: ReviewDecision
    reviewer_id: Optional[str] = None
    decision_rationale: str = Field(..., min_length=10)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    follow_up_actions: Optional[List[str]] = None


class ReviewQueueStats(BaseModel):
    total_pending: int
    critical_pending: int
    high_pending: int
    medium_pending: int
    low_pending: int
    overdue_count: int
    avg_resolution_time_hours: float
    sla_compliance_rate: float


class TieredHumanReviewSystem:
    """
    V2.1 Tiered Human Review System with intelligent SLA management
    """

    def __init__(self, supabase_client):
        self.supabase = supabase_client
        self.sla_mapping = {
            ReviewPriority.CRITICAL: timedelta(hours=4),
            ReviewPriority.HIGH: timedelta(hours=24),
            ReviewPriority.MEDIUM: timedelta(hours=72),
            ReviewPriority.LOW: timedelta(days=7),
        }

        # Start background SLA monitoring
        asyncio.create_task(self._monitor_sla_compliance())

    async def queue_for_review(self, request: ReviewRequest) -> Dict[str, Any]:
        """Queue contradiction for human review with appropriate SLA"""
        try:
            # Calculate SLA deadline
            sla_deadline = datetime.now() + self.sla_mapping[request.priority_level]

            review_data = {
                "contradiction_type": request.contradiction_type,
                "priority_level": request.priority_level.value,
                "conflicting_insight": request.conflicting_data,
                "contradiction_analysis": request.context_information or {},
                "sla_deadline": sla_deadline.isoformat(),
                "review_status": ReviewStatus.PENDING.value,
                "created_at": datetime.now().isoformat(),
                "reviewer_notes": request.reviewer_notes,
            }

            result = (
                self.supabase.table("human_review_queue").insert(review_data).execute()
            )

            if result.data:
                review_id = result.data[0]["id"]

                # Schedule SLA reminder
                await self._schedule_sla_reminder(
                    review_id, sla_deadline, request.priority_level
                )

                logger.info(
                    f"Review queued: {review_id} with {request.priority_level} priority"
                )
                return result.data[0]
            else:
                raise HTTPException(status_code=500, detail="Failed to queue review")

        except Exception as e:
            logger.error(f"Review queuing error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def get_review_queue(
        self, priority_filter: Optional[ReviewPriority] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get review queue with optional priority filtering"""
        try:
            query = (
                self.supabase.table("human_review_queue")
                .select("*")
                .eq("review_status", ReviewStatus.PENDING.value)
                .order("sla_deadline", desc=False)
            )

            if priority_filter:
                query = query.eq("priority_level", priority_filter.value)

            query = query.limit(limit)
            result = query.execute()

            return result.data if result.data else []

        except Exception as e:
            logger.error(f"Review queue retrieval error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def process_review_decision(
        self, decision_request: ReviewDecisionRequest
    ) -> Dict[str, Any]:
        """Process human review decision and execute appropriate actions"""
        try:
            # Load review record
            review_result = (
                self.supabase.table("human_review_queue")
                .select("*")
                .eq("id", decision_request.review_id)
                .execute()
            )

            if not review_result.data:
                raise HTTPException(status_code=404, detail="Review not found")

            review = review_result.data[0]

            # Update review record with decision
            resolution_data = {
                "review_status": ReviewStatus.RESOLVED.value,
                "resolution_decision": {
                    "decision": decision_request.decision.value,
                    "rationale": decision_request.decision_rationale,
                    "confidence_score": decision_request.confidence_score,
                    "reviewer_id": decision_request.reviewer_id,
                    "resolved_at": datetime.now().isoformat(),
                },
                "reviewed_at": datetime.now().isoformat(),
                "assigned_reviewer": decision_request.reviewer_id,
            }

            self.supabase.table("human_review_queue").update(resolution_data).eq(
                "id", decision_request.review_id
            ).execute()

            # Execute decision action
            action_result = await self._execute_review_decision(
                review, decision_request
            )

            logger.info(
                f"Review {decision_request.review_id} resolved with decision: {decision_request.decision}"
            )

            return {
                "review_id": decision_request.review_id,
                "decision": decision_request.decision.value,
                "action_result": action_result,
                "resolved_at": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Review decision processing error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def get_queue_stats(self) -> ReviewQueueStats:
        """Get comprehensive review queue statistics"""
        try:
            # Get all pending reviews
            pending_result = (
                self.supabase.table("human_review_queue")
                .select("*")
                .eq("review_status", ReviewStatus.PENDING.value)
                .execute()
            )
            pending_reviews = pending_result.data if pending_result.data else []

            # Count by priority
            priority_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}

            overdue_count = 0
            current_time = datetime.now()

            for review in pending_reviews:
                priority = review["priority_level"]
                priority_counts[priority] = priority_counts.get(priority, 0) + 1

                # Check if overdue
                sla_deadline = datetime.fromisoformat(
                    review["sla_deadline"].replace("Z", "+00:00")
                )
                if current_time > sla_deadline.replace(tzinfo=None):
                    overdue_count += 1

            # Calculate average resolution time
            resolved_result = (
                self.supabase.table("human_review_queue")
                .select("created_at, reviewed_at")
                .eq("review_status", ReviewStatus.RESOLVED.value)
                .limit(100)
                .execute()
            )
            resolved_reviews = resolved_result.data if resolved_result.data else []

            avg_resolution_hours = 0.0
            if resolved_reviews:
                total_resolution_time = 0
                for review in resolved_reviews:
                    if review["reviewed_at"]:
                        created = datetime.fromisoformat(
                            review["created_at"].replace("Z", "+00:00")
                        )
                        resolved = datetime.fromisoformat(
                            review["reviewed_at"].replace("Z", "+00:00")
                        )
                        resolution_time = (
                            resolved - created
                        ).total_seconds() / 3600  # Convert to hours
                        total_resolution_time += resolution_time

                avg_resolution_hours = total_resolution_time / len(resolved_reviews)

            # Calculate SLA compliance rate
            total_resolved = len(resolved_reviews)
            sla_compliant = sum(
                1 for review in resolved_reviews if self._was_sla_compliant(review)
            )

            sla_compliance_rate = (
                (sla_compliant / total_resolved * 100) if total_resolved > 0 else 100.0
            )

            return ReviewQueueStats(
                total_pending=len(pending_reviews),
                critical_pending=priority_counts.get("critical", 0),
                high_pending=priority_counts.get("high", 0),
                medium_pending=priority_counts.get("medium", 0),
                low_pending=priority_counts.get("low", 0),
                overdue_count=overdue_count,
                avg_resolution_time_hours=round(avg_resolution_hours, 2),
                sla_compliance_rate=round(sla_compliance_rate, 2),
            )

        except Exception as e:
            logger.error(f"Queue stats calculation error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def escalate_overdue_reviews(self) -> Dict[str, Any]:
        """Escalate reviews that have exceeded their SLA"""
        try:
            current_time = datetime.now()

            # Find overdue reviews
            pending_result = (
                self.supabase.table("human_review_queue")
                .select("*")
                .eq("review_status", ReviewStatus.PENDING.value)
                .execute()
            )
            pending_reviews = pending_result.data if pending_result.data else []

            escalated_count = 0
            for review in pending_reviews:
                sla_deadline = datetime.fromisoformat(
                    review["sla_deadline"].replace("Z", "+00:00")
                )

                if current_time > sla_deadline.replace(tzinfo=None):
                    # Escalate review
                    escalation_data = {
                        "review_status": ReviewStatus.ESCALATED.value,
                        "escalated_at": current_time.isoformat(),
                        "escalation_reason": f"SLA breach: {(current_time - sla_deadline.replace(tzinfo=None)).total_seconds() / 3600:.1f} hours overdue",
                    }

                    self.supabase.table("human_review_queue").update(
                        escalation_data
                    ).eq("id", review["id"]).execute()

                    # TODO: Send escalation notifications (Slack, email, etc.)
                    await self._send_escalation_notification(review)

                    escalated_count += 1

            logger.info(f"Escalated {escalated_count} overdue reviews")

            return {
                "escalated_count": escalated_count,
                "escalated_at": current_time.isoformat(),
            }

        except Exception as e:
            logger.error(f"Review escalation error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def _execute_review_decision(
        self, review: Dict, decision_request: ReviewDecisionRequest
    ) -> Dict[str, Any]:
        """Execute the action based on review decision"""
        try:
            if decision_request.decision == ReviewDecision.ACCEPT_NEW:
                # Deprecate existing wisdom, accept new insight
                existing_wisdom_id = review.get("existing_wisdom_id")
                if existing_wisdom_id:
                    await self._deprecate_wisdom_entry(existing_wisdom_id)

                # Store new insight as wisdom
                conflicting_insight = review["conflicting_insight"]
                new_wisdom = await self._store_new_wisdom(conflicting_insight)

                return {
                    "action": "accepted_new_insight",
                    "new_wisdom_id": new_wisdom.get("id"),
                }

            elif decision_request.decision == ReviewDecision.REJECT_NEW:
                # Keep existing wisdom, reject new insight
                return {
                    "action": "rejected_new_insight",
                    "existing_wisdom_preserved": True,
                }

            elif decision_request.decision == ReviewDecision.MERGE_BOTH:
                # Create merged wisdom entry
                merged_wisdom = await self._create_merged_wisdom(
                    review, decision_request.decision_rationale
                )
                return {
                    "action": "created_merged_wisdom",
                    "merged_wisdom_id": merged_wisdom.get("id"),
                }

            elif decision_request.decision == ReviewDecision.REQUIRE_MORE_DATA:
                # Flag for additional data collection
                return {
                    "action": "flagged_for_more_data",
                    "status": "pending_additional_info",
                }

        except Exception as e:
            logger.error(f"Decision execution error: {e}")
            return {"error": str(e)}

    async def _deprecate_wisdom_entry(self, wisdom_id: str):
        """Mark wisdom entry as deprecated"""
        self.supabase.table("flywheel_distilled_wisdom").update(
            {"temporal_relevance_score": 0.1, "updated_at": datetime.now().isoformat()}
        ).eq("id", wisdom_id).execute()

    async def _store_new_wisdom(self, insight_data: Dict) -> Dict:
        """Store new insight as distilled wisdom"""
        wisdom_data = {
            "wisdom_content": insight_data.get("content", ""),
            "confidence_score": insight_data.get("confidence_score", 0.7),
            "domain_tags": insight_data.get("domain_tags", []),
            "temporal_relevance_score": 1.0,
            "source_bookmarks": [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }

        result = (
            self.supabase.table("flywheel_distilled_wisdom")
            .insert(wisdom_data)
            .execute()
        )
        return result.data[0] if result.data else {}

    async def _create_merged_wisdom(self, review: Dict, merge_rationale: str) -> Dict:
        """Create merged wisdom entry combining both perspectives"""
        existing_wisdom = review.get("existing_wisdom", {})
        new_insight = review["conflicting_insight"]

        merged_content = f"Merged wisdom: {existing_wisdom.get('content', '')} | {new_insight.get('content', '')} | Merge rationale: {merge_rationale}"

        wisdom_data = {
            "wisdom_content": merged_content,
            "confidence_score": 0.8,  # High confidence for human-merged content
            "domain_tags": list(
                set(
                    existing_wisdom.get("domain_tags", [])
                    + new_insight.get("domain_tags", [])
                )
            ),
            "temporal_relevance_score": 1.0,
            "consolidation_count": existing_wisdom.get("consolidation_count", 0) + 1,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }

        result = (
            self.supabase.table("flywheel_distilled_wisdom")
            .insert(wisdom_data)
            .execute()
        )
        return result.data[0] if result.data else {}

    def _was_sla_compliant(self, review: Dict) -> bool:
        """Check if review was resolved within SLA"""
        try:
            created_at = datetime.fromisoformat(
                review["created_at"].replace("Z", "+00:00")
            )
            reviewed_at = datetime.fromisoformat(
                review["reviewed_at"].replace("Z", "+00:00")
            )

            # Determine SLA based on priority (stored in resolution_decision if available)
            priority = review.get("priority_level", "medium")
            sla_hours = {"critical": 4, "high": 24, "medium": 72, "low": 168}.get(
                priority, 72
            )

            resolution_time = (reviewed_at - created_at).total_seconds() / 3600
            return resolution_time <= sla_hours

        except Exception:
            return False

    async def _schedule_sla_reminder(
        self, review_id: str, sla_deadline: datetime, priority: ReviewPriority
    ):
        """Schedule SLA reminder notifications"""
        # TODO: Implement reminder scheduling (e.g., with Celery or similar)
        pass

    async def _send_escalation_notification(self, review: Dict):
        """Send escalation notification to administrators"""
        # TODO: Implement notification system (Slack, email, etc.)
        logger.warning(f"Review {review['id']} escalated due to SLA breach")

    async def _monitor_sla_compliance(self):
        """Background task to monitor and escalate overdue reviews"""
        while True:
            try:
                await self.escalate_overdue_reviews()
                # Check every hour
                await asyncio.sleep(3600)
            except Exception as e:
                logger.error(f"SLA monitoring error: {e}")
                await asyncio.sleep(3600)  # Continue monitoring despite errors


# Global system instance
review_system = None


def get_review_system():
    """Dependency injection for review system"""
    global review_system
    if review_system is None:
        # TODO: Initialize with actual Supabase client
        from supabase import create_client

        supabase_url = "your_supabase_url"
        supabase_key = "your_supabase_key"
        supabase = create_client(supabase_url, supabase_key)
        review_system = TieredHumanReviewSystem(supabase)
    return review_system


# API Routes


@router.post("/queue")
async def queue_review(
    request: ReviewRequest, system: TieredHumanReviewSystem = Depends(get_review_system)
):
    """Queue a contradiction for human review"""
    return await system.queue_for_review(request)


@router.get("/queue")
async def get_review_queue(
    priority: Optional[ReviewPriority] = None,
    limit: int = 50,
    system: TieredHumanReviewSystem = Depends(get_review_system),
):
    """Get the human review queue"""
    return await system.get_review_queue(priority_filter=priority, limit=limit)


@router.post("/decide")
async def process_decision(
    decision: ReviewDecisionRequest,
    system: TieredHumanReviewSystem = Depends(get_review_system),
):
    """Process a human review decision"""
    return await system.process_review_decision(decision)


@router.get("/stats")
async def get_queue_stats(system: TieredHumanReviewSystem = Depends(get_review_system)):
    """Get review queue statistics"""
    return await system.get_queue_stats()


@router.post("/escalate")
async def escalate_overdue(
    system: TieredHumanReviewSystem = Depends(get_review_system),
):
    """Manually trigger escalation of overdue reviews"""
    return await system.escalate_overdue_reviews()


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "system": "human_review_v2.1"}
