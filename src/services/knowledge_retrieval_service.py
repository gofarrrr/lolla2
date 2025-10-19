"""
Knowledge Retrieval Service - Phase 2 Integration Component
=========================================================

Resilient service that provides AI consultants with access to the mental models
knowledge base through the IntelligentRetriever infrastructure.

Part of the Great Knowledge Infusion - Build Order KB-03
"""

import logging
from typing import List, Dict, Any, Union, Optional
import asyncio
from datetime import datetime

from ..rag.retriever import IntelligentRetriever
from ..rag.embeddings import VoyageEmbeddings
from ..core.unified_context_stream import get_unified_context_stream, UnifiedContextStream

logger = logging.getLogger(__name__)


class KnowledgeRetrievalService:
    """
    Resilient service for querying the mental models knowledge base
    
    Provides hardened access to the RAG system with comprehensive error handling
    and graceful degradation for production reliability.
    """
    
    def __init__(
        self, 
        intelligent_retriever: Optional[IntelligentRetriever] = None,
        context_stream: Optional[Any] = None
    ):
        """
        Initialize the Knowledge Retrieval Service
        
        Args:
            intelligent_retriever: RAG retriever instance (optional, will create if None)
            context_stream: Context stream for event logging (optional)
        """
        self.intelligent_retriever = intelligent_retriever
        self.context_stream = context_stream or get_unified_context_stream()
        self.is_initialized = False
        
        # Service statistics
        self.stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "empty_results": 0,
            "initialization_attempts": 0,
            "last_error": None
        }
        
        logger.info("🧠 KnowledgeRetrievalService initialized")
        
    async def initialize(self) -> bool:
        """
        Initialize the underlying retriever infrastructure
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        self.stats["initialization_attempts"] += 1
        
        try:
            if not self.intelligent_retriever:
                # Create retriever with embeddings
                embeddings = VoyageEmbeddings(context_stream=self.context_stream)
                self.intelligent_retriever = IntelligentRetriever(
                    embeddings=embeddings,
                    context_stream=self.context_stream
                )
                
            # Initialize the retriever
            await self.intelligent_retriever.initialize()
            self.is_initialized = True
            
            logger.info("✅ KnowledgeRetrievalService initialization successful")
            
            # Log initialization event
            self.context_stream.add_event(
                "reasoning_step",
                {
                    "service": "knowledge_retrieval",
                    "action": "initialization",
                    "status": "success",
                    "attempt": self.stats["initialization_attempts"]
                }
            )
            
            return True
            
        except Exception as e:
            logger.error(f"❌ KnowledgeRetrievalService initialization failed: {e}")
            self.stats["last_error"] = str(e)
            self.is_initialized = False
            
            # Log initialization failure
            self.context_stream.add_event(
                "reasoning_step",
                {
                    "service": "knowledge_retrieval", 
                    "action": "initialization",
                    "status": "error",
                    "error": str(e),
                    "attempt": self.stats["initialization_attempts"]
                }
            )
            
            return False
            
    async def search_knowledge_base(
        self, 
        query: str, 
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> Union[List[str], Dict[str, str]]:
        """
        Search the mental models knowledge base with resilient error handling
        
        Args:
            query: Search query string
            top_k: Maximum number of results to return
            filters: Optional metadata filters
            
        Returns:
            List[str]: List of content strings if successful
            Dict[str, str]: Error object if failed
        """
        start_time = datetime.now()
        self.stats["total_queries"] += 1
        
        # Log query start
        self.context_stream.add_event(
            "research_query",
            {
                "service": "knowledge_retrieval",
                "query": query,
                "top_k": top_k,
                "filters": filters,
                "query_id": self.stats["total_queries"]
            }
        )
        
        try:
            # Ensure service is initialized
            if not self.is_initialized:
                init_success = await self.initialize()
                if not init_success:
                    error_response = {
                        "error": "Knowledge Base initialization failed. Service unavailable."
                    }
                    self.stats["failed_queries"] += 1
                    self.stats["last_error"] = error_response["error"]
                    
                    # Log initialization failure
                    self.context_stream.add_event(
                        "research_result",
                        {
                            "service": "knowledge_retrieval",
                            "query": query,
                            "status": "initialization_failed",
                            "error": error_response["error"],
                            "query_id": self.stats["total_queries"]
                        }
                    )
                    
                    return error_response
                    
            # Set default filters for mental models
            if filters is None:
                filters = {"content_type": "mental_model"}
            
            # Execute the search
            logger.info(f"🔍 Searching knowledge base: '{query[:50]}...'")
            
            results = await self.intelligent_retriever.retrieve(
                query=query,
                top_k=top_k,
                filters=filters
            )
            
            # Process results
            if not results:
                # Empty results
                self.stats["empty_results"] += 1
                
                logger.info(f"📭 No results found for query: '{query[:50]}...'")
                
                # Log empty results
                self.context_stream.add_event(
                    "research_result",
                    {
                        "service": "knowledge_retrieval",
                        "query": query,
                        "status": "no_results",
                        "results_count": 0,
                        "query_id": self.stats["total_queries"],
                        "processing_time_ms": (datetime.now() - start_time).total_seconds() * 1000
                    }
                )
                
                return []
                
            # Extract content from results
            content_list = []
            for result in results:
                # Format result with source and score information
                formatted_content = f"Source: {result.source}\nScore: {result.score:.3f}\n\n{result.content}"
                content_list.append(formatted_content)
                
            self.stats["successful_queries"] += 1
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            logger.info(
                f"✅ Knowledge search successful: {len(content_list)} results in {processing_time:.1f}ms"
            )
            
            # Log successful results
            self.context_stream.add_event(
                "research_result",
                {
                    "service": "knowledge_retrieval",
                    "query": query,
                    "status": "success",
                    "results_count": len(content_list),
                    "avg_score": sum(r.score for r in results) / len(results),
                    "query_id": self.stats["total_queries"],
                    "processing_time_ms": processing_time
                }
            )
            
            return content_list
            
        except Exception as e:
            # Handle all other errors
            error_message = f"Knowledge Base connection failed: {str(e)}"
            error_response = {"error": error_message}
            
            self.stats["failed_queries"] += 1
            self.stats["last_error"] = error_message
            
            logger.error(f"❌ Knowledge search failed: {e}")
            
            # Log search failure
            self.context_stream.add_event(
                "research_result",
                {
                    "service": "knowledge_retrieval",
                    "query": query,
                    "status": "error",
                    "error": error_message,
                    "query_id": self.stats["total_queries"],
                    "processing_time_ms": (datetime.now() - start_time).total_seconds() * 1000
                }
            )
            
            return error_response

    async def search_knowledge_base_structured(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Union[List[Dict[str, Any]], Dict[str, str]]:
        """Structured retrieval variant returning metadata-rich results.

        Preserves compatibility with existing string-returning search while
        enabling callers to use categories/titles from ingestion metadata.
        """
        start_time = datetime.now()
        try:
            if not self.is_initialized:
                ok = await self.initialize()
                if not ok:
                    return {"error": "Knowledge Base initialization failed. Service unavailable."}

            if filters is None:
                filters = {"content_type": "mental_model"}

            results = await self.intelligent_retriever.retrieve(
                query=query, top_k=top_k, filters=filters
            )

            structured: List[Dict[str, Any]] = []
            for r in results:
                md = getattr(r, "metadata", {}) or {}
                snippet_lines = (getattr(r, "content", None) or "").strip().splitlines()
                snippet_text = " ".join(snippet_lines[:5])[:600]
                structured.append(
                    {
                        "title": md.get("title") or md.get("model_name") or (getattr(r, "source", None) or "Mental Model"),
                        "description": snippet_text,
                        "relevance": float(getattr(r, "score", 0.0)),
                        "category": md.get("category"),
                        "source": "rag_knowledge_base",
                        "content": getattr(r, "content", None) or "",
                        "metadata": md,
                    }
                )

            # Log structured results
            self.context_stream.add_event(
                "research_result",
                {
                    "service": "knowledge_retrieval",
                    "query": query,
                    "status": "success",
                    "results_count": len(structured),
                    "processing_time_ms": (datetime.now() - start_time).total_seconds() * 1000,
                },
            )

            return structured
        except Exception as e:  # pylint: disable=broad-except
            self.context_stream.add_event(
                "research_result",
                {
                    "service": "knowledge_retrieval",
                    "query": query,
                    "status": "error",
                    "error": str(e),
                },
            )
            return {"error": f"Knowledge Base connection failed: {e}"}
            
    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics and health metrics"""
        success_rate = (
            self.stats["successful_queries"] / max(1, self.stats["total_queries"])
        ) * 100
        
        return {
            **self.stats,
            "success_rate_percent": success_rate,
            "is_initialized": self.is_initialized,
            "service_health": "healthy" if success_rate > 80 else "degraded" if success_rate > 50 else "unhealthy"
        }
        
    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check of the knowledge retrieval service"""
        health_data = {
            "service": "knowledge_retrieval",
            "timestamp": datetime.now().isoformat(),
            "is_initialized": self.is_initialized
        }
        
        try:
            # Test with a simple query
            test_result = await self.search_knowledge_base(
                "systems thinking", 
                top_k=1
            )
            
            if isinstance(test_result, list):
                health_data.update({
                    "status": "healthy",
                    "test_query_result": "success",
                    "can_retrieve_knowledge": True
                })
            else:
                health_data.update({
                    "status": "degraded", 
                    "test_query_result": "error",
                    "can_retrieve_knowledge": False,
                    "error": test_result.get("error", "Unknown error")
                })
                
        except Exception as e:
            health_data.update({
                "status": "unhealthy",
                "test_query_result": "exception",
                "can_retrieve_knowledge": False,
                "error": str(e)
            })
            
        health_data.update(self.get_service_stats())
        return health_data


# Factory function for easy instantiation
async def create_knowledge_retrieval_service(
    context_stream: Optional[UnifiedContextStream] = None
) -> KnowledgeRetrievalService:
    """
    Factory function to create and initialize a KnowledgeRetrievalService
    
    Args:
        context_stream: Optional context stream for event logging
        
    Returns:
        Initialized KnowledgeRetrievalService instance
    """
    service = KnowledgeRetrievalService(context_stream=context_stream)
    await service.initialize()
    return service
