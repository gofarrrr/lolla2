"""
Enhanced Research API - METIS 2.0 Phase 4 API Integration
===========================================================

REST API endpoints for the unified Enhanced Research Manager.
Provides comprehensive research orchestration with RAG, web intelligence,
cost optimization, and provider fallback chains.

This API provides:
1. POST /research/query - Execute comprehensive research query
2. GET /research/status/{analysis_id} - Check research status
3. GET /research/history/{user_id} - Get user research history
4. GET /research/costs/summary - Get cost summary and budget status
5. GET /research/health - Enhanced Research Manager health check
6. POST /research/chat - Interactive chat with RAG system
"""

import uuid
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/research", tags=["Enhanced Research"])

# Global Enhanced Research Manager instance
_research_manager_instance: Optional[Any] = None


def get_enhanced_research_manager():
    """Get or create Enhanced Research Manager instance"""
    global _research_manager_instance
    if _research_manager_instance is None:
        try:
            from src.engine.core.enhanced_research_manager import (
                EnhancedResearchManager,
            )

            # Default configuration - in production this would come from environment
            config = {
                "daily_budget": 10.0,
                "monthly_budget": 300.0,
                "rag": {
                    "collection_name": "metis_research_knowledge",
                    "embedding_model": "voyage-large-2-instruct",
                },
                "supabase": {
                    "url": None,  # Will be set from environment
                    "key": None,  # Will be set from environment
                },
                "milvus": {
                    "host": "localhost",
                    "port": 19530,
                    "database": "metis_research",
                },
                "zep": {
                    "api_url": None,  # Will be set from environment
                    "api_key": None,  # Will be set from environment
                },
            }

            _research_manager_instance = EnhancedResearchManager(config)
            logger.info("🧠 Enhanced Research Manager initialized for API")
        except ImportError as e:
            logger.error(f"❌ Enhanced Research Manager not available: {e}")
            raise HTTPException(
                status_code=503,
                detail="Enhanced Research Manager not available. Ensure Phase 3 is completed.",
            )
    return _research_manager_instance


# Request/Response Models
class ResearchQuery(BaseModel):
    query: str = Field(
        ..., min_length=5, max_length=2000, description="Research question or topic"
    )
    user_id: str = Field(..., description="User identifier for context and history")
    requirements: Optional[Dict[str, Any]] = Field(
        default=None, description="Research requirements (providers, budget, etc.)"
    )
    priority: Optional[str] = Field(
        default="medium", description="Query priority: low, medium, high, urgent"
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional context information"
    )


class ChatMessage(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="Chat message")
    user_id: str = Field(..., description="User identifier")
    conversation_id: Optional[str] = Field(
        default=None, description="Conversation ID for context"
    )


class ResearchResult(BaseModel):
    analysis_id: str
    query: str
    user_id: str
    status: str
    start_time: str
    end_time: Optional[str] = None
    results: Optional[Dict[str, Any]] = None
    cost_breakdown: Optional[Dict[str, Any]] = None
    providers_used: Optional[List[str]] = None
    confidence_score: Optional[float] = None
    sources: Optional[List[Dict[str, Any]]] = None
    error_message: Optional[str] = None


class ResearchStatus(BaseModel):
    analysis_id: str
    status: str  # queued, in_progress, completed, failed
    progress: float = Field(ge=0.0, le=1.0, description="Progress from 0.0 to 1.0")
    current_phase: Optional[str] = None
    estimated_completion: Optional[str] = None
    error_message: Optional[str] = None


class CostSummary(BaseModel):
    total_spent_today: float
    total_spent_month: float
    daily_budget: float
    monthly_budget: float
    daily_remaining: float
    monthly_remaining: float
    cost_by_category: Dict[str, float]
    top_expensive_queries: List[Dict[str, Any]]
    budget_alerts: List[str]


class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    sources: Optional[List[Dict[str, Any]]] = None
    confidence_score: Optional[float] = None
    cost: Optional[float] = None


# In-memory storage for active research operations (in production, use Redis/database)
active_research: Dict[str, Dict[str, Any]] = {}


@router.post("/query", response_model=ResearchResult)
async def execute_research_query(
    query: ResearchQuery,
    background_tasks: BackgroundTasks,
    manager=Depends(get_enhanced_research_manager),
):
    """
    Execute a comprehensive research query using Enhanced Research Manager

    This endpoint:
    1. Validates the research query and requirements
    2. Initiates research using the 7-phase Enhanced Research workflow
    3. Returns immediate response with analysis_id for tracking
    4. Continues processing in background
    """
    try:
        # Generate unique analysis ID
        analysis_id = str(uuid.uuid4())
        start_time = datetime.now().isoformat()

        # Store initial research state
        active_research[analysis_id] = {
            "analysis_id": analysis_id,
            "query": query.query,
            "user_id": query.user_id,
            "status": "queued",
            "start_time": start_time,
            "progress": 0.0,
            "current_phase": "initialization",
        }

        # Start background research task
        background_tasks.add_task(
            _execute_background_research, analysis_id, query, manager
        )

        logger.info(
            f"🔍 Research query initiated: {analysis_id} for user {query.user_id}"
        )

        return ResearchResult(
            analysis_id=analysis_id,
            query=query.query,
            user_id=query.user_id,
            status="queued",
            start_time=start_time,
        )

    except Exception as e:
        logger.error(f"❌ Research query failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to initiate research: {str(e)}"
        )


async def _execute_background_research(analysis_id: str, query: ResearchQuery, manager):
    """Execute research in background and update status"""
    try:
        # Update status to in_progress
        if analysis_id in active_research:
            active_research[analysis_id].update(
                {
                    "status": "in_progress",
                    "progress": 0.1,
                    "current_phase": "research_execution",
                }
            )

        # Execute the actual research using Enhanced Research Manager
        result = await manager.research_with_memory(
            query=query.query,
            user_id=query.user_id,
            requirements=query.requirements or {},
        )

        # Extract cost information if available
        cost_breakdown = {}
        providers_used = []
        if "cost_summary" in result:
            cost_breakdown = result["cost_summary"]
        if "providers_used" in result:
            providers_used = result["providers_used"]

        # Update with successful completion
        if analysis_id in active_research:
            active_research[analysis_id].update(
                {
                    "status": "completed",
                    "progress": 1.0,
                    "current_phase": "completed",
                    "end_time": datetime.now().isoformat(),
                    "results": result,
                    "cost_breakdown": cost_breakdown,
                    "providers_used": providers_used,
                    "confidence_score": result.get("confidence_score", 0.85),
                }
            )

        logger.info(f"✅ Research completed: {analysis_id}")

    except Exception as e:
        logger.error(f"❌ Background research failed for {analysis_id}: {e}")

        # Update with failure status
        if analysis_id in active_research:
            active_research[analysis_id].update(
                {
                    "status": "failed",
                    "current_phase": "failed",
                    "end_time": datetime.now().isoformat(),
                    "error_message": str(e),
                }
            )


@router.get("/status/{analysis_id}", response_model=ResearchStatus)
async def get_research_status(analysis_id: str):
    """Get the current status of a research operation"""
    if analysis_id not in active_research:
        raise HTTPException(status_code=404, detail="Research analysis not found")

    research_data = active_research[analysis_id]

    return ResearchStatus(
        analysis_id=analysis_id,
        status=research_data["status"],
        progress=research_data["progress"],
        current_phase=research_data.get("current_phase"),
        error_message=research_data.get("error_message"),
    )


@router.get("/history/{user_id}")
async def get_research_history(
    user_id: str, limit: int = 50, manager=Depends(get_enhanced_research_manager)
):
    """Get research history for a specific user"""
    try:
        # Filter active research for this user
        user_research = [
            research
            for research in active_research.values()
            if research["user_id"] == user_id
        ]

        # Sort by start time (most recent first)
        user_research.sort(key=lambda x: x["start_time"], reverse=True)

        # Limit results
        user_research = user_research[:limit]

        return {
            "user_id": user_id,
            "total_queries": len(user_research),
            "research_history": user_research,
        }

    except Exception as e:
        logger.error(f"❌ Failed to get research history for {user_id}: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve research history"
        )


@router.get("/costs/summary", response_model=CostSummary)
async def get_cost_summary(manager=Depends(get_enhanced_research_manager)):
    """Get comprehensive cost summary and budget status"""
    try:
        # Get cost summary from Enhanced Research Manager
        cost_optimizer = manager.cost_optimizer

        daily_summary = await cost_optimizer.get_cost_summary("daily")
        monthly_summary = await cost_optimizer.get_cost_summary("monthly")

        # Get budget status
        budget_status = await cost_optimizer.get_budget_status()

        # Calculate remaining budgets
        daily_remaining = max(
            0, cost_optimizer.daily_budget - daily_summary["total_cost"]
        )
        monthly_remaining = max(
            0, cost_optimizer.monthly_budget - monthly_summary["total_cost"]
        )

        # Get top expensive queries (from active research)
        expensive_queries = []
        for research in active_research.values():
            if research.get("cost_breakdown") and research["status"] == "completed":
                total_cost = (
                    sum(research["cost_breakdown"].values())
                    if research["cost_breakdown"]
                    else 0
                )
                expensive_queries.append(
                    {
                        "analysis_id": research["analysis_id"],
                        "query": (
                            research["query"][:100] + "..."
                            if len(research["query"]) > 100
                            else research["query"]
                        ),
                        "cost": total_cost,
                        "date": research["start_time"],
                    }
                )

        expensive_queries.sort(key=lambda x: x["cost"], reverse=True)
        expensive_queries = expensive_queries[:10]  # Top 10

        # Generate budget alerts
        alerts = []
        daily_usage_pct = (
            daily_summary["total_cost"] / cost_optimizer.daily_budget
        ) * 100
        monthly_usage_pct = (
            monthly_summary["total_cost"] / cost_optimizer.monthly_budget
        ) * 100

        if daily_usage_pct >= 90:
            alerts.append(f"Daily budget 90% exceeded: {daily_usage_pct:.1f}%")
        elif daily_usage_pct >= 70:
            alerts.append(f"Daily budget 70% reached: {daily_usage_pct:.1f}%")

        if monthly_usage_pct >= 90:
            alerts.append(f"Monthly budget 90% exceeded: {monthly_usage_pct:.1f}%")
        elif monthly_usage_pct >= 70:
            alerts.append(f"Monthly budget 70% reached: {monthly_usage_pct:.1f}%")

        return CostSummary(
            total_spent_today=daily_summary["total_cost"],
            total_spent_month=monthly_summary["total_cost"],
            daily_budget=cost_optimizer.daily_budget,
            monthly_budget=cost_optimizer.monthly_budget,
            daily_remaining=daily_remaining,
            monthly_remaining=monthly_remaining,
            cost_by_category=daily_summary.get("by_category", {}),
            top_expensive_queries=expensive_queries,
            budget_alerts=alerts,
        )

    except Exception as e:
        logger.error(f"❌ Failed to get cost summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve cost summary")


@router.get("/health")
async def enhanced_research_health_check(
    manager=Depends(get_enhanced_research_manager),
):
    """Health check for Enhanced Research Manager and all components"""
    try:
        # Get comprehensive health status
        health_status = await manager.health_check()

        # Add API-specific metrics
        health_status.update(
            {
                "api_status": "healthy",
                "active_research_queries": len(active_research),
                "api_version": "1.0.0",
                "integration_status": "phase_4_complete",
            }
        )

        return health_status

    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return {
            "api_status": "unhealthy",
            "error": str(e),
            "active_research_queries": len(active_research),
            "api_version": "1.0.0",
        }


@router.post("/chat", response_model=ChatResponse)
async def chat_with_rag(
    chat_message: ChatMessage, manager=Depends(get_enhanced_research_manager)
):
    """
    Interactive chat with RAG system

    Provides conversational interface to the Enhanced Research Manager
    with memory and context awareness.
    """
    try:
        # Generate conversation ID if not provided
        conversation_id = chat_message.conversation_id or str(uuid.uuid4())

        # Use Enhanced Research Manager for chat
        # This leverages the RAG pipeline and conversation memory
        result = await manager.research_with_memory(
            query=chat_message.message,
            user_id=chat_message.user_id,
            requirements={"chat_mode": True, "conversation_id": conversation_id},
        )

        # Extract response components
        response_text = result.get(
            "response", result.get("summary", "No response generated")
        )
        sources = result.get("sources", [])
        confidence_score = result.get("confidence_score", 0.8)
        cost = result.get("total_cost", 0.0)

        return ChatResponse(
            response=response_text,
            conversation_id=conversation_id,
            sources=sources,
            confidence_score=confidence_score,
            cost=cost,
        )

    except Exception as e:
        logger.error(f"❌ Chat request failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chat request failed: {str(e)}")


@router.get("/contracts/status")
async def get_contracts_status():
    """Get status of Intelligence Contracts system"""
    try:
        from contracts.intelligence_contracts import (
            validate_contracts,
            get_contract_summary,
        )

        # Validate all contracts
        validation_result = validate_contracts()

        # Get contract summary
        summary = get_contract_summary()

        return {
            "contracts_validation": validation_result,
            "contracts_summary": summary,
            "integration_status": "active",
        }

    except ImportError:
        return {
            "contracts_validation": {"status": "not_available"},
            "contracts_summary": {"status": "not_available"},
            "integration_status": "contracts_not_available",
        }
    except Exception as e:
        logger.error(f"❌ Contracts status check failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to check contracts status: {str(e)}"
        )


# Cleanup endpoint for development/testing
@router.delete("/cleanup")
async def cleanup_active_research():
    """Clear all active research data (development only)"""
    global active_research
    count = len(active_research)
    active_research.clear()

    return {"message": f"Cleared {count} active research entries", "status": "cleaned"}
