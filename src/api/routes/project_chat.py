"""
Project Chat API - Chat with Project Knowledge
==============================================

Enables conversational Q&A over all analyses and data within a project using:
1. ProjectRAGPipeline for semantic search across project knowledge
2. OpenRouter (Grok-4-Fast) for fast, cost-effective synthesis
3. Supabase for conversation history storage
"""

import logging
from typing import List, Optional
from uuid import UUID
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from src.rag.project_rag_pipeline import get_project_rag_pipeline
from src.engine.integrations.openrouter_client import OpenRouterClient
from src.core.unified_context_stream import UnifiedContextStream, ContextEventType

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/projects", tags=["project-chat"])


# Request/Response Models
class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ProjectChatRequest(BaseModel):
    project_id: str = Field(..., description="Project UUID")
    user_id: str = Field(..., description="User UUID")
    message: str = Field(..., description="User's question")
    conversation_history: List[ChatMessage] = Field(default_factory=list, description="Previous messages for context")


class ProjectChatResponse(BaseModel):
    answer: str = Field(..., description="AI-generated answer")
    sources: List[dict] = Field(..., description="Source documents used")
    context_used: dict = Field(..., description="Context metadata")
    model_used: str = Field(..., description="LLM model used")
    tokens_used: int = Field(..., description="Total tokens consumed")


# Dependency injection
async def get_openrouter_client():
    """Get OpenRouter client for Grok-4-Fast"""
    return OpenRouterClient()


async def get_project_rag():
    """Get Project RAG Pipeline"""
    rag = get_project_rag_pipeline()
    await rag.initialize()
    return rag


@router.post("/{project_id}/chat", response_model=ProjectChatResponse)
async def chat_with_project(
    project_id: str,
    request: ProjectChatRequest,
    openrouter: OpenRouterClient = Depends(get_openrouter_client),
    project_rag = Depends(get_project_rag)
):
    """
    Chat with all project knowledge using semantic search + Grok-4-Fast synthesis

    Flow:
    1. Query ProjectRAGPipeline for relevant context
    2. Build prompt with project knowledge + conversation history
    3. Send to Grok-4-Fast via OpenRouter
    4. Return answer with sources
    5. Store conversation in Supabase
    """
    try:
        logger.info(f"💬 Project chat request: project={project_id}, message='{request.message[:100]}...'")

        from src.core.unified_context_stream import get_unified_context_stream
        context_stream = get_unified_context_stream()
        context_stream.add_event(
            ContextEventType.TOOL_EXECUTION,
            {
                "tool": "project_chat",
                "action": "chat_request",
                "project_id": project_id,
                "message_preview": request.message[:200],
                "timestamp": datetime.now().isoformat()
            }
        )

        # Step 1: Retrieve relevant project context
        project_context = await project_rag.get_initial_context_for_query(
            project_id=UUID(project_id),
            problem_statement=request.message
        )

        if not project_context["context_available"]:
            logger.warning(f"⚠️ No project context found for {project_id}")
            # Still answer, but note lack of context
            context_summary = "No previous analyses found in this project."
            sources = []
        else:
            context_summary = project_context["context_summary"]
            sources = [
                {
                    "analysis_id": analysis_id,
                    "confidence": project_context["confidence_score"]
                }
                for analysis_id in project_context["source_analyses"]
            ]

        logger.info(f"📚 Retrieved context: {len(sources)} sources, confidence {project_context.get('confidence_score', 0):.2f}")

        # Step 2: Build conversation prompt with context
        system_prompt = f"""You are an AI assistant helping analyze a strategic project.

You have access to the following project knowledge:

{context_summary}

Your role is to:
1. Answer the user's question based on the project knowledge above
2. Cite specific analyses when relevant
3. Be concise but thorough
4. Acknowledge if you don't have enough information
5. Provide actionable insights where possible

If the project has no prior analyses, help the user understand what analyses they might need to run."""

        # Build messages array with conversation history
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history (last 5 messages for context)
        for msg in request.conversation_history[-5:]:
            messages.append({"role": msg.role, "content": msg.content})

        # Add current user message
        messages.append({"role": "user", "content": request.message})

        # Step 3: Call Grok-4-Fast via OpenRouter
        logger.info(f"🤖 Calling Grok-4-Fast with {len(messages)} messages")

        llm_response = await openrouter.complete(
            messages=messages,
            model="x-ai/grok-4-fast",  # Cost-efficient: $0.20/1M input, $0.50/1M output
            max_tokens=1500,
            temperature=0.7
        )

        answer = llm_response.content
        tokens_used = llm_response.usage.get("total_tokens", 0) if llm_response.usage else 0

        logger.info(f"✅ Chat response generated: {len(answer)} chars, {tokens_used} tokens")

        # Step 4: Log success
        context_stream.add_event(
            ContextEventType.TOOL_EXECUTION,
            {
                "tool": "project_chat",
                "action": "chat_response",
                "project_id": project_id,
                "answer_length": len(answer),
                "sources_count": len(sources),
                "tokens_used": tokens_used,
                "model": "grok-4-fast",
                "timestamp": datetime.now().isoformat()
            }
        )

        # Step 5: TODO - Store conversation in Supabase
        # await store_chat_message(project_id, request.user_id, request.message, answer, sources, tokens_used)

        return ProjectChatResponse(
            answer=answer,
            sources=sources,
            context_used={
                "relevant_docs_count": project_context.get("relevant_docs_count", 0),
                "confidence_score": project_context.get("confidence_score", 0.0),
                "context_available": project_context["context_available"]
            },
            model_used="grok-4-fast",
            tokens_used=tokens_used
        )

    except Exception as e:
        logger.error(f"❌ Project chat error: {e}")
        context_stream.add_event(
            ContextEventType.TOOL_EXECUTION,
            {
                "tool": "project_chat",
                "action": "chat_error",
                "project_id": project_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


@router.get("/{project_id}/chat/history")
async def get_chat_history(
    project_id: str,
    user_id: str,
    limit: int = 50
):
    """
    Retrieve chat conversation history for a project

    TODO: Implement Supabase query:
    - SELECT * FROM project_chat_messages
    - WHERE project_id = ? AND user_id = ?
    - ORDER BY created_at DESC
    - LIMIT ?
    """
    # Placeholder - implement with Supabase client
    return {
        "project_id": project_id,
        "messages": [],
        "total_count": 0
    }


@router.delete("/{project_id}/chat/{message_id}")
async def delete_chat_message(
    project_id: str,
    message_id: str,
    user_id: str
):
    """
    Delete a specific chat message

    TODO: Implement with Supabase:
    - DELETE FROM project_chat_messages
    - WHERE id = ? AND user_id = ? AND project_id = ?
    """
    return {"deleted": True, "message_id": message_id}


# Helper function to store chat in Supabase (to be implemented)
async def store_chat_message(
    project_id: str,
    user_id: str,
    user_message: str,
    assistant_message: str,
    sources: List[dict],
    tokens_used: int
):
    """
    Store chat exchange in Supabase project_chat_messages table

    TODO: Implement with Supabase client:
    1. Insert user message
    2. Insert assistant message with context_used
    """
    pass
