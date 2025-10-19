#!/usr/bin/env python3
"""
Xpander.ai + Ragie.ai Integration Bridge
Seamless integration between Backend-as-a-Service and RAG-as-a-Service
"""

import os
import json
import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import requests

# Xpander.ai SDK imports
try:
    from xpander import Backend, on_task, Task, on_webhook, on_schedule
    from xpander.types import TaskStatus, WebhookEvent

    XPANDER_AVAILABLE = True
except ImportError:
    XPANDER_AVAILABLE = False
    logging.warning(
        "⚠️ Xpander SDK not available - install with: pip install xpander-sdk"
    )

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class XpanderRagieBridge:
    """
    Integration bridge between Xpander.ai Backend-as-a-Service and Ragie.ai RAG-as-a-Service

    Provides:
    - Unified API for knowledge operations
    - Automatic task orchestration via Xpander.ai
    - Seamless RAG integration via Ragie.ai
    - Performance monitoring and optimization
    """

    def __init__(self, ragie_api_key: str, xpander_backend=None):
        self.ragie_api_key = ragie_api_key
        self.ragie_base_url = "https://api.ragie.ai"
        self.xpander_backend = xpander_backend

        # Initialize Ragie client
        self.ragie_headers = {
            "Authorization": f"Bearer {ragie_api_key}",
            "Content-Type": "application/json",
        }

        # Performance tracking
        self.request_count = 0
        self.cache_hits = 0
        self.error_count = 0

        # Simple in-memory cache for frequently accessed knowledge
        self.knowledge_cache = {}
        self.cache_ttl = 3600  # 1 hour cache TTL

        logger.info("🌉 Xpander-Ragie bridge initialized")

    async def search_knowledge(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Search knowledge via Ragie.ai with Xpander.ai orchestration

        Args:
            query: Search query
            top_k: Number of results to return
            filters: Metadata filters for search
            use_cache: Whether to use cached results

        Returns:
            Search results with metadata
        """
        self.request_count += 1

        # Check cache first
        cache_key = f"search_{hash(query)}_{top_k}_{hash(str(filters))}"

        if use_cache and cache_key in self.knowledge_cache:
            cached_result, timestamp = self.knowledge_cache[cache_key]
            if (datetime.now().timestamp() - timestamp) < self.cache_ttl:
                self.cache_hits += 1
                logger.info(f"📋 Cache hit for query: '{query[:50]}...'")
                return cached_result

        try:
            # Prepare Ragie.ai request
            payload = {"query": query, "top_k": top_k}

            if filters:
                payload["filter"] = filters

            logger.info(f"🔍 Ragie search: '{query[:50]}...' (top_k={top_k})")

            # Execute search via Ragie.ai
            response = requests.post(
                f"{self.ragie_base_url}/retrievals",
                headers=self.ragie_headers,
                json=payload,
            )

            if response.status_code == 200:
                result_data = response.json()

                # Format response for bridge consumers
                bridge_result = {
                    "status": "success",
                    "query": query,
                    "results": result_data.get("scored_chunks", []),
                    "metadata": {
                        "total_results": len(result_data.get("scored_chunks", [])),
                        "search_timestamp": datetime.now().isoformat(),
                        "top_k": top_k,
                        "filters_applied": filters or {},
                        "source": "ragie_api",
                    },
                }

                # Cache successful results
                if use_cache:
                    self.knowledge_cache[cache_key] = (
                        bridge_result,
                        datetime.now().timestamp(),
                    )

                logger.info(f"✅ Found {len(bridge_result['results'])} results")
                return bridge_result
            else:
                self.error_count += 1
                logger.error(
                    f"❌ Ragie search failed: {response.status_code} - {response.text}"
                )
                return {
                    "status": "error",
                    "error": f"Ragie API error: {response.status_code}",
                    "query": query,
                }

        except Exception as e:
            self.error_count += 1
            logger.error(f"❌ Bridge search error: {e}")
            return {"status": "error", "error": str(e), "query": query}

    async def create_knowledge_document(
        self, content: str, metadata: Dict[str, Any], wait_for_indexing: bool = True
    ) -> Dict[str, Any]:
        """
        Create knowledge document in Ragie.ai via Xpander.ai orchestration

        Args:
            content: Document content
            metadata: Document metadata
            wait_for_indexing: Whether to wait for document indexing

        Returns:
            Creation result with document ID
        """
        try:
            payload = {"text": content, "metadata": metadata}

            logger.info(
                f"📝 Creating document: {metadata.get('title', 'Untitled')[:50]}..."
            )

            # Create document via Ragie.ai
            response = requests.post(
                f"{self.ragie_base_url}/documents",
                headers=self.ragie_headers,
                json=payload,
            )

            if response.status_code == 201:
                result = response.json()
                document_id = result.get("id")

                bridge_result = {
                    "status": "success",
                    "document_id": document_id,
                    "metadata": metadata,
                    "creation_timestamp": datetime.now().isoformat(),
                    "indexing_status": "pending",
                }

                # Wait for indexing if requested
                if wait_for_indexing:
                    indexing_result = await self._wait_for_document_indexing(
                        document_id
                    )
                    bridge_result["indexing_status"] = indexing_result["status"]
                    bridge_result["indexing_duration"] = indexing_result.get("duration")

                logger.info(f"✅ Document created: {document_id}")
                return bridge_result
            else:
                self.error_count += 1
                logger.error(
                    f"❌ Document creation failed: {response.status_code} - {response.text}"
                )
                return {
                    "status": "error",
                    "error": f"Ragie API error: {response.status_code}",
                }

        except Exception as e:
            self.error_count += 1
            logger.error(f"❌ Document creation error: {e}")
            return {"status": "error", "error": str(e)}

    async def _wait_for_document_indexing(
        self, document_id: str, max_wait: int = 120
    ) -> Dict[str, Any]:
        """Wait for document to be indexed"""
        start_time = datetime.now()

        while (datetime.now() - start_time).seconds < max_wait:
            try:
                response = requests.get(
                    f"{self.ragie_base_url}/documents/{document_id}",
                    headers=self.ragie_headers,
                )

                if response.status_code == 200:
                    result = response.json()
                    status = result.get("status", "unknown")

                    if status in ["indexed", "ready"]:
                        duration = (datetime.now() - start_time).seconds
                        return {
                            "status": status,
                            "duration": duration,
                            "document_id": document_id,
                        }
                    elif status == "failed":
                        return {
                            "status": "failed",
                            "document_id": document_id,
                            "error": "Document processing failed",
                        }

                await asyncio.sleep(5)  # Check every 5 seconds

            except Exception as e:
                logger.error(f"❌ Error checking indexing status: {e}")
                return {"status": "error", "error": str(e), "document_id": document_id}

        return {
            "status": "timeout",
            "document_id": document_id,
            "error": f"Indexing did not complete within {max_wait} seconds",
        }

    async def get_mental_model_recommendations(
        self, problem_statement: str, context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Get mental model recommendations for a problem statement

        Args:
            problem_statement: The problem to analyze
            context: Additional context (industry, complexity, etc.)

        Returns:
            Recommended mental models with application guidance
        """
        try:
            # Enhance query with context
            enhanced_query = problem_statement

            if context:
                if context.get("industry"):
                    enhanced_query += f" industry: {context['industry']}"
                if context.get("domain"):
                    enhanced_query += f" domain: {context['domain']}"
                if context.get("complexity_preference"):
                    enhanced_query += f" complexity: {context['complexity_preference']}"

            # Search for relevant mental models
            filters = {"document_type": "knowledge_element"}

            if context and context.get("complexity_preference"):
                filters["complexity_level"] = context["complexity_preference"]

            search_result = await self.search_knowledge(
                enhanced_query, top_k=8, filters=filters
            )

            if search_result["status"] != "success":
                return search_result

            # Process and rank mental models
            mental_models = []
            for chunk in search_result["results"]:
                metadata = chunk.get("metadata", {})

                mental_model = {
                    "ke_name": metadata.get("ke_name", "Unknown"),
                    "ke_id": metadata.get("ke_id", ""),
                    "cognitive_tool_type": metadata.get(
                        "cognitive_tool_type", "analytical_tool"
                    ),
                    "effectiveness_score": metadata.get("effectiveness_score", 0.8),
                    "complexity_level": metadata.get("complexity_level", 3),
                    "domain_tags": metadata.get("domain_tags", []),
                    "relevance_score": chunk.get("score", 0.0),
                    "content_preview": (
                        chunk.get("text", "")[:200] + "..."
                        if len(chunk.get("text", "")) > 200
                        else chunk.get("text", "")
                    ),
                    "application_guide": self._extract_application_guide(
                        chunk.get("text", "")
                    ),
                }

                mental_models.append(mental_model)

            # Sort by combined relevance and effectiveness
            mental_models.sort(
                key=lambda x: x["relevance_score"] * x["effectiveness_score"],
                reverse=True,
            )

            return {
                "status": "success",
                "problem_statement": problem_statement,
                "context": context or {},
                "recommendations": mental_models[:5],  # Top 5 recommendations
                "metadata": {
                    "total_candidates": len(mental_models),
                    "recommendation_timestamp": datetime.now().isoformat(),
                    "selection_criteria": "relevance × effectiveness",
                },
            }

        except Exception as e:
            self.error_count += 1
            logger.error(f"❌ Mental model recommendation error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "problem_statement": problem_statement,
            }

    def _extract_application_guide(self, content: str) -> str:
        """Extract application guide from mental model content"""
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if "prompt integration guide" in line.lower():
                # Look for the next non-empty line
                for j in range(i + 1, min(i + 5, len(lines))):
                    if lines[j].strip():
                        return lines[j].strip()

        # Fallback: use first substantial line
        for line in lines:
            if len(line.strip()) > 50:
                return (
                    line.strip()[:200] + "..."
                    if len(line.strip()) > 200
                    else line.strip()
                )

        return "Apply this mental model systematically to your analysis framework."

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get bridge performance metrics"""
        cache_hit_rate = (self.cache_hits / max(self.request_count, 1)) * 100
        error_rate = (self.error_count / max(self.request_count, 1)) * 100

        return {
            "total_requests": self.request_count,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": f"{cache_hit_rate:.1f}%",
            "errors": self.error_count,
            "error_rate": f"{error_rate:.1f}%",
            "cache_entries": len(self.knowledge_cache),
            "uptime": "Active",
            "last_updated": datetime.now().isoformat(),
        }

    def clear_cache(self):
        """Clear the knowledge cache"""
        self.knowledge_cache.clear()
        logger.info("🧹 Knowledge cache cleared")


# Xpander.ai Task Handlers
if XPANDER_AVAILABLE:

    # Global bridge instance
    bridge = None

    def init_bridge():
        """Initialize the bridge with environment variables"""
        global bridge
        if bridge is None:
            ragie_api_key = os.getenv("RAGIE_API_KEY")
            if ragie_api_key:
                backend = Backend()
                bridge = XpanderRagieBridge(ragie_api_key, backend)
                logger.info("🌉 Bridge initialized for Xpander tasks")
            else:
                logger.error("❌ RAGIE_API_KEY not found - bridge not initialized")

    @on_task
    async def knowledge_search_task(task: Task):
        """Xpander.ai task: Search knowledge base"""
        init_bridge()

        if not bridge:
            await task.fail("Bridge not initialized - check RAGIE_API_KEY")
            return

        try:
            # Extract parameters
            query = task.payload.get("query", "")
            top_k = task.payload.get("top_k", 5)
            filters = task.payload.get("filters")
            use_cache = task.payload.get("use_cache", True)

            if not query:
                await task.fail("Missing 'query' in task payload")
                return

            # Execute search
            result = await bridge.search_knowledge(query, top_k, filters, use_cache)

            await task.complete(result)

        except Exception as e:
            logger.error(f"❌ Knowledge search task failed: {e}")
            await task.fail(str(e))

    @on_task
    async def mental_model_recommendation_task(task: Task):
        """Xpander.ai task: Get mental model recommendations"""
        init_bridge()

        if not bridge:
            await task.fail("Bridge not initialized - check RAGIE_API_KEY")
            return

        try:
            problem_statement = task.payload.get("problem_statement", "")
            context = task.payload.get("context", {})

            if not problem_statement:
                await task.fail("Missing 'problem_statement' in task payload")
                return

            result = await bridge.get_mental_model_recommendations(
                problem_statement, context
            )

            await task.complete(result)

        except Exception as e:
            logger.error(f"❌ Mental model recommendation task failed: {e}")
            await task.fail(str(e))

    @on_task
    async def create_knowledge_document_task(task: Task):
        """Xpander.ai task: Create knowledge document"""
        init_bridge()

        if not bridge:
            await task.fail("Bridge not initialized - check RAGIE_API_KEY")
            return

        try:
            content = task.payload.get("content", "")
            metadata = task.payload.get("metadata", {})
            wait_for_indexing = task.payload.get("wait_for_indexing", True)

            if not content:
                await task.fail("Missing 'content' in task payload")
                return

            result = await bridge.create_knowledge_document(
                content, metadata, wait_for_indexing
            )

            await task.complete(result)

        except Exception as e:
            logger.error(f"❌ Create document task failed: {e}")
            await task.fail(str(e))

    @on_webhook
    async def bridge_status_webhook(event: WebhookEvent):
        """Webhook for bridge status monitoring"""
        init_bridge()

        if bridge:
            metrics = bridge.get_performance_metrics()
            return {
                "status": "active",
                "bridge_metrics": metrics,
                "timestamp": datetime.now().isoformat(),
            }
        else:
            return {
                "status": "inactive",
                "error": "Bridge not initialized",
                "timestamp": datetime.now().isoformat(),
            }

    @on_schedule("0 */6 * * *")  # Every 6 hours
    async def cache_cleanup_task():
        """Scheduled task: Clean up expired cache entries"""
        init_bridge()

        if bridge:
            # Simple cache cleanup - remove entries older than TTL
            current_time = datetime.now().timestamp()
            expired_keys = []

            for key, (_, timestamp) in bridge.knowledge_cache.items():
                if (current_time - timestamp) > bridge.cache_ttl:
                    expired_keys.append(key)

            for key in expired_keys:
                del bridge.knowledge_cache[key]

            logger.info(
                f"🧹 Cache cleanup: removed {len(expired_keys)} expired entries"
            )


# Standalone testing
async def main():
    """Test the bridge standalone"""
    logger.info("🧪 Testing Xpander-Ragie Bridge")

    # Initialize bridge
    ragie_api_key = os.getenv("RAGIE_API_KEY")
    if not ragie_api_key:
        logger.error("❌ RAGIE_API_KEY environment variable required")
        return

    bridge = XpanderRagieBridge(ragie_api_key)

    # Test 1: Knowledge search
    logger.info("\n🔍 Test 1: Knowledge Search")
    search_result = await bridge.search_knowledge("systems thinking analysis", top_k=3)
    print(json.dumps(search_result, indent=2))

    # Test 2: Mental model recommendations
    logger.info("\n🎯 Test 2: Mental Model Recommendations")
    problem = "How to improve customer retention in a SaaS business?"
    context = {"industry": "technology", "complexity_preference": 3}

    recommendations = await bridge.get_mental_model_recommendations(problem, context)
    print(json.dumps(recommendations, indent=2))

    # Test 3: Performance metrics
    logger.info("\n📊 Test 3: Performance Metrics")
    metrics = bridge.get_performance_metrics()
    print(json.dumps(metrics, indent=2))

    return True


if __name__ == "__main__":
    if not XPANDER_AVAILABLE:
        # Run as standalone test
        asyncio.run(main())
    else:
        # Run as Xpander.ai backend service
        logger.info("🚀 Starting Xpander-Ragie Bridge as backend service")
        backend = Backend()
        backend.start()
