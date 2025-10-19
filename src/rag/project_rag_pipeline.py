"""
Project-Specific RAG Pipeline for METIS V2 Project Workspace
============================================================

Project-scoped RAG implementation that enables context merging and knowledge accumulation
within individual projects. Integrates with the existing RAG infrastructure while providing
project-specific data isolation and PostgreSQL-based storage.

Key Features:
- Project-scoped context isolation using project_id
- Integration with PostgreSQL vector search via pgvector
- Analysis ingestion from UnifiedContextStream logs
- Smart context retrieval for new analyses within projects
- Integration with stateful_pipeline_orchestrator for conditional context merging
"""

import logging
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from uuid import UUID

# Database imports
import asyncpg
from pgvector.asyncpg import register_vector

# Core imports
from src.core.unified_context_stream import get_unified_context_stream, ContextEventType, UnifiedContextStream
from src.rag.rag_pipeline import RAGPipelineFactory

logger = logging.getLogger(__name__)


class ProjectRAGPipeline:
    """
    Project-specific RAG pipeline for METIS V2 that provides context merging capabilities
    within individual projects. Implements project-scoped data isolation and PostgreSQL integration.
    """

    def __init__(
        self,
        db_url: str = "postgresql://postgres:metis2024@localhost:5432/metis_db",
        context_stream: Optional[Any] = None,
    ):
        """
        Initialize project-specific RAG pipeline

        Args:
            db_url: PostgreSQL database connection URL
            context_stream: Context stream for logging
        """
        self.db_url = db_url
        self.context_stream = context_stream or get_unified_context_stream()
        self.db_pool = None

        # Initialize enhanced RAG pipeline for vector operations
        self.enhanced_rag = RAGPipelineFactory.create_default_pipeline()

        logger.info("🏗️ ProjectRAGPipeline initialized")

    async def initialize(self) -> bool:
        """Initialize database connections and RAG pipeline"""
        try:
            # Initialize database connection pool
            self.db_pool = await asyncpg.create_pool(
                self.db_url, command_timeout=60, min_size=2, max_size=10
            )

            # Register pgvector extension
            async with self.db_pool.acquire() as conn:
                await register_vector(conn)

            # Initialize enhanced RAG pipeline
            await self.enhanced_rag.initialize()

            self.context_stream.add_event(
                ContextEventType.TOOL_EXECUTION,
                {
                    "tool": "project_rag_pipeline",
                    "action": "initialize",
                    "status": "success",
                    "timestamp": datetime.now().isoformat(),
                },
            )

            logger.info("✅ ProjectRAGPipeline initialization successful")
            return True

        except Exception as e:
            logger.error(f"❌ ProjectRAGPipeline initialization failed: {e}")
            self.context_stream.add_event(
                ContextEventType.TOOL_EXECUTION,
                {
                    "tool": "project_rag_pipeline",
                    "action": "initialize",
                    "status": "error",
                    "error": str(e),
                },
            )
            return False

    async def ingest_analysis(self, engagement_id: UUID, project_id: UUID) -> bool:
        """
        Ingest completed analysis from UnifiedContextStream into project RAG knowledge base

        Args:
            engagement_id: Completed engagement/analysis ID
            project_id: Project ID for scoping

        Returns:
            True if ingestion successful, False otherwise
        """
        try:
            logger.info(
                f"📥 Starting analysis ingestion: {engagement_id} → Project {project_id}"
            )

            self.context_stream.add_event(
                ContextEventType.CONTEXT_MERGE,
                {
                    "action": "RAG_INGESTION_STARTED",
                    "engagement_id": str(engagement_id),
                    "project_id": str(project_id),
                    "timestamp": datetime.now().isoformat(),
                },
            )

            # Extract valuable content from completed analysis
            analysis_content = await self._extract_analysis_content(engagement_id)

            if not analysis_content:
                logger.warning(
                    f"⚠️ No valuable content extracted from engagement {engagement_id}"
                )
                return False

            # Store in database with project scoping
            doc_id = await self._store_analysis_document(
                engagement_id=engagement_id,
                project_id=project_id,
                content=analysis_content,
            )

            if not doc_id:
                logger.error(
                    f"❌ Failed to store analysis document for {engagement_id}"
                )
                return False

            self.context_stream.add_event(
                ContextEventType.CONTEXT_MERGE,
                {
                    "action": "RAG_INGESTION_COMPLETED",
                    "engagement_id": str(engagement_id),
                    "project_id": str(project_id),
                    "document_id": str(doc_id),
                    "content_length": len(analysis_content["combined_content"]),
                    "timestamp": datetime.now().isoformat(),
                },
            )

            logger.info(f"✅ Analysis ingestion completed: {engagement_id} → {doc_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Analysis ingestion failed for {engagement_id}: {e}")
            self.context_stream.add_event(
                ContextEventType.CONTEXT_MERGE,
                {
                    "action": "RAG_INGESTION_ERROR",
                    "engagement_id": str(engagement_id),
                    "project_id": str(project_id),
                    "error": str(e),
                },
            )
            return False

    async def get_initial_context_for_query(
        self, project_id: UUID, problem_statement: str
    ) -> Dict[str, Any]:
        """
        Retrieve relevant context from project knowledge base for new analysis

        Args:
            project_id: Project ID for scoping
            problem_statement: New problem statement to find context for

        Returns:
            Dictionary containing relevant context and metadata
        """
        try:
            logger.info(
                f"🔍 Retrieving context for project {project_id}: '{problem_statement[:100]}...'"
            )

            self.context_stream.add_event(
                ContextEventType.CONTEXT_MERGE,
                {
                    "action": "CONTEXT_RETRIEVAL_STARTED",
                    "project_id": str(project_id),
                    "query_preview": problem_statement[:200],
                    "timestamp": datetime.now().isoformat(),
                },
            )

            # Perform semantic search within project scope
            relevant_docs = await self._search_project_knowledge(
                project_id, problem_statement
            )

            if not relevant_docs:
                logger.info(f"📭 No relevant context found for project {project_id}")
                return {
                    "context_available": False,
                    "context_summary": "No relevant previous analyses found in this project.",
                    "source_analyses": [],
                    "confidence_score": 0.0,
                }

            # Synthesize context from relevant documents
            context_synthesis = await self._synthesize_project_context(
                relevant_docs, problem_statement
            )

            self.context_stream.add_event(
                ContextEventType.CONTEXT_MERGE,
                {
                    "action": "CONTEXT_RETRIEVAL_COMPLETED",
                    "project_id": str(project_id),
                    "relevant_docs_count": len(relevant_docs),
                    "confidence_score": context_synthesis["confidence_score"],
                    "context_length": len(context_synthesis["context_summary"]),
                    "timestamp": datetime.now().isoformat(),
                },
            )

            logger.info(
                f"✅ Context retrieved: {len(relevant_docs)} docs, confidence {context_synthesis['confidence_score']:.2f}"
            )

            return {
                "context_available": True,
                "context_summary": context_synthesis["context_summary"],
                "source_analyses": context_synthesis["source_analyses"],
                "confidence_score": context_synthesis["confidence_score"],
                "relevant_docs_count": len(relevant_docs),
            }

        except Exception as e:
            logger.error(f"❌ Context retrieval failed for project {project_id}: {e}")
            self.context_stream.add_event(
                ContextEventType.CONTEXT_MERGE,
                {
                    "action": "CONTEXT_RETRIEVAL_ERROR",
                    "project_id": str(project_id),
                    "error": str(e),
                },
            )
            return {
                "context_available": False,
                "context_summary": "Error retrieving project context.",
                "source_analyses": [],
                "confidence_score": 0.0,
                "error": str(e),
            }

    async def store_web_document(
        self,
        url: str,
        content: str,
        title: str,
        project_id: UUID,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Store web document in project-scoped RAG storage

        Args:
            url: Source URL of the document
            content: Extracted content text
            title: Document title
            project_id: Project ID for scoping
            tags: Optional tags for the document
            metadata: Additional metadata

        Returns:
            Document ID if successful, None otherwise
        """
        try:
            logger.info(
                f"📄 Storing web document for project {project_id}: {title} ({url})"
            )

            async with self.db_pool.acquire() as conn:
                # Insert document record
                doc_result = await conn.fetchrow(
                    """
                    INSERT INTO rag_documents (
                        id, project_id, title, source_type, source_url,
                        content, tags, metadata, created_at, updated_at
                    ) VALUES (
                        gen_random_uuid(), $1, $2, $3, $4, 
                        $5, $6, $7, NOW(), NOW()
                    ) RETURNING id
                """,
                    project_id,
                    title,
                    "web_extraction",
                    url,
                    content,
                    tags or [],
                    json.dumps(metadata or {}),
                )

                if not doc_result:
                    logger.error(f"❌ Failed to insert document for {url}")
                    return None

                doc_id = doc_result["id"]

                # Create text chunks for vector search
                await self._create_text_chunks(conn, doc_id, content)

                # Log success
                self.context_stream.add_event(
                    ContextEventType.CONTEXT_MERGE,
                    {
                        "action": "WEB_DOCUMENT_STORED",
                        "document_id": str(doc_id),
                        "project_id": str(project_id),
                        "url": url,
                        "title": title,
                        "content_length": len(content),
                        "tags": tags or [],
                    },
                )

                logger.info(
                    f"✅ Successfully stored web document {doc_id} for project {project_id}"
                )
                return str(doc_id)

        except Exception as e:
            logger.error(f"❌ Web document storage failed for {url}: {e}")
            self.context_stream.add_event(
                ContextEventType.CONTEXT_MERGE,
                {
                    "action": "WEB_DOCUMENT_STORAGE_ERROR",
                    "project_id": str(project_id),
                    "url": url,
                    "error": str(e),
                },
            )
            return None

    async def _extract_analysis_content(
        self, engagement_id: UUID
    ) -> Optional[Dict[str, Any]]:
        """Extract valuable content from completed analysis logs"""
        try:
            async with self.db_pool.acquire() as conn:
                # Query UnifiedContextStream logs for this engagement
                logs = await conn.fetch(
                    """
                    SELECT event_type, event_data, timestamp
                    FROM context_streams 
                    WHERE session_id = $1 
                    ORDER BY timestamp ASC
                """,
                    str(engagement_id),
                )

                if not logs:
                    return None

                # Extract key analysis components
                problem_structure = None
                key_insights = []
                recommendations = []
                strategic_analysis = None

                for log in logs:
                    event_data = log["event_data"]
                    event_type = log["event_type"]

                    # Extract problem structuring
                    if event_type == "TOOL_EXECUTION" and "problem_structure" in str(
                        event_data
                    ):
                        problem_structure = self._extract_problem_structure(event_data)

                    # Extract insights and recommendations
                    if event_type == "ANALYSIS_INSIGHT":
                        key_insights.append(self._extract_insight(event_data))

                    if event_type == "RECOMMENDATION":
                        recommendations.append(self._extract_recommendation(event_data))

                    # Extract strategic analysis
                    if event_type == "STRATEGIC_ANALYSIS":
                        strategic_analysis = self._extract_strategic_analysis(
                            event_data
                        )

                # Combine into valuable document
                combined_content = self._combine_analysis_content(
                    problem_structure, key_insights, recommendations, strategic_analysis
                )

                return {
                    "engagement_id": str(engagement_id),
                    "combined_content": combined_content,
                    "problem_structure": problem_structure,
                    "key_insights": key_insights,
                    "recommendations": recommendations,
                    "strategic_analysis": strategic_analysis,
                    "extraction_timestamp": datetime.now().isoformat(),
                }

        except Exception as e:
            logger.error(f"❌ Content extraction failed for {engagement_id}: {e}")
            return None

    async def _store_analysis_document(
        self, engagement_id: UUID, project_id: UUID, content: Dict[str, Any]
    ) -> Optional[UUID]:
        """Store analysis document in project-scoped RAG storage"""
        try:
            async with self.db_pool.acquire() as conn:
                # Store in rag_documents table with project scoping
                doc_id = await conn.fetchval(
                    """
                    INSERT INTO rag_documents (
                        project_id, source_type, source_id, title, 
                        content, metadata, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                    RETURNING document_id
                """,
                    project_id,
                    "analysis",
                    str(engagement_id),
                    f"Analysis: {engagement_id}",
                    content["combined_content"],
                    json.dumps(
                        {
                            "engagement_id": str(engagement_id),
                            "extraction_timestamp": content["extraction_timestamp"],
                            "has_problem_structure": content["problem_structure"]
                            is not None,
                            "insights_count": len(content["key_insights"]),
                            "recommendations_count": len(content["recommendations"]),
                            "has_strategic_analysis": content["strategic_analysis"]
                            is not None,
                        }
                    ),
                    datetime.now(),
                )

                # Create text chunks for vector search
                await self._create_text_chunks(
                    conn, doc_id, content["combined_content"]
                )

                return doc_id

        except Exception as e:
            logger.error(f"❌ Document storage failed: {e}")
            return None

    async def _create_text_chunks(self, conn, doc_id: UUID, content: str):
        """Create searchable text chunks with embeddings"""
        # Simple chunking strategy - split by paragraphs
        chunks = [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]

        for i, chunk in enumerate(chunks):
            if len(chunk) < 50:  # Skip very short chunks
                continue

            # Generate embedding using enhanced RAG pipeline
            try:
                embedding = await self.enhanced_rag.embeddings.embed(chunk)

                await conn.execute(
                    """
                    INSERT INTO rag_text_chunks (
                        document_id, chunk_index, content, embedding, 
                        metadata, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                """,
                    doc_id,
                    i,
                    chunk,
                    embedding,
                    json.dumps({"chunk_length": len(chunk)}),
                    datetime.now(),
                )

            except Exception as e:
                logger.warning(f"⚠️ Failed to create chunk {i}: {e}")

    async def _search_project_knowledge(
        self, project_id: UUID, query: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for relevant documents within project scope"""
        try:
            # Generate query embedding
            query_embedding = await self.enhanced_rag.embeddings.embed(query)

            async with self.db_pool.acquire() as conn:
                # Vector similarity search within project scope
                results = await conn.fetch(
                    """
                    SELECT 
                        d.document_id, d.title, d.content, d.metadata, d.created_at,
                        c.content as chunk_content,
                        (c.embedding <=> $1) as similarity_distance
                    FROM rag_documents d
                    JOIN rag_text_chunks c ON d.document_id = c.document_id
                    WHERE d.project_id = $2
                    ORDER BY c.embedding <=> $1
                    LIMIT $3
                """,
                    query_embedding,
                    project_id,
                    limit,
                )

                return [dict(row) for row in results]

        except Exception as e:
            logger.error(f"❌ Project knowledge search failed: {e}")
            return []

    async def _synthesize_project_context(
        self, relevant_docs: List[Dict[str, Any]], problem_statement: str
    ) -> Dict[str, Any]:
        """Synthesize relevant context into coherent summary"""
        if not relevant_docs:
            return {
                "context_summary": "",
                "source_analyses": [],
                "confidence_score": 0.0,
            }

        # Calculate average confidence score
        confidence_score = max(
            0.0,
            1.0
            - sum(doc["similarity_distance"] for doc in relevant_docs)
            / len(relevant_docs),
        )

        # Extract source analysis IDs
        source_analyses = []
        context_chunks = []

        for doc in relevant_docs:
            metadata = json.loads(doc["metadata"]) if doc["metadata"] else {}
            if "engagement_id" in metadata:
                source_analyses.append(metadata["engagement_id"])
            context_chunks.append(doc["chunk_content"])

        # Create context summary
        context_summary = f"""
RELEVANT PROJECT CONTEXT:

Based on {len(relevant_docs)} previous analyses in this project, here are relevant insights:

{chr(10).join([f"• {chunk[:200]}..." for chunk in context_chunks[:3]])}

This context may inform your analysis of: "{problem_statement}"

Source analyses: {len(set(source_analyses))} previous engagements
Confidence: {confidence_score:.1%}
        """.strip()

        return {
            "context_summary": context_summary,
            "source_analyses": list(set(source_analyses)),
            "confidence_score": confidence_score,
        }

    def _extract_problem_structure(self, event_data: Dict[str, Any]) -> Optional[str]:
        """Extract problem structure from event data"""
        try:
            if isinstance(event_data, dict) and "result" in event_data:
                result = event_data["result"]
                if isinstance(result, dict) and "problem_structure" in result:
                    return str(result["problem_structure"])
        except:
            pass
        return None

    def _extract_insight(self, event_data: Dict[str, Any]) -> str:
        """Extract insight from event data"""
        try:
            if isinstance(event_data, dict) and "insight" in event_data:
                return str(event_data["insight"])
            return str(event_data)[:200]
        except:
            return "Insight extraction failed"

    def _extract_recommendation(self, event_data: Dict[str, Any]) -> str:
        """Extract recommendation from event data"""
        try:
            if isinstance(event_data, dict) and "recommendation" in event_data:
                return str(event_data["recommendation"])
            return str(event_data)[:200]
        except:
            return "Recommendation extraction failed"

    def _extract_strategic_analysis(self, event_data: Dict[str, Any]) -> Optional[str]:
        """Extract strategic analysis from event data"""
        try:
            if isinstance(event_data, dict) and "strategic_analysis" in event_data:
                return str(event_data["strategic_analysis"])
        except:
            pass
        return None

    def _combine_analysis_content(
        self,
        problem_structure: Optional[str],
        key_insights: List[str],
        recommendations: List[str],
        strategic_analysis: Optional[str],
    ) -> str:
        """Combine analysis components into coherent document"""
        content_parts = []

        if problem_structure:
            content_parts.append(f"PROBLEM STRUCTURE:\n{problem_structure}")

        if key_insights:
            content_parts.append(
                "KEY INSIGHTS:\n"
                + "\n".join([f"• {insight}" for insight in key_insights])
            )

        if recommendations:
            content_parts.append(
                "RECOMMENDATIONS:\n"
                + "\n".join([f"• {rec}" for rec in recommendations])
            )

        if strategic_analysis:
            content_parts.append(f"STRATEGIC ANALYSIS:\n{strategic_analysis}")

        return (
            "\n\n".join(content_parts)
            if content_parts
            else "Analysis content not available"
        )

    async def get_project_knowledge_stats(self, project_id: UUID) -> Dict[str, Any]:
        """Get statistics about project knowledge base"""
        try:
            async with self.db_pool.acquire() as conn:
                stats = await conn.fetchrow(
                    """
                    SELECT 
                        COUNT(d.document_id) as total_documents,
                        COUNT(c.chunk_id) as total_chunks,
                        MIN(d.created_at) as oldest_analysis,
                        MAX(d.created_at) as newest_analysis
                    FROM rag_documents d
                    LEFT JOIN rag_text_chunks c ON d.document_id = c.document_id
                    WHERE d.project_id = $1
                """,
                    project_id,
                )

                return dict(stats) if stats else {}

        except Exception as e:
            logger.error(f"❌ Failed to get project stats: {e}")
            return {}

    async def health_check(self) -> Dict[str, Any]:
        """Health check for project RAG pipeline"""
        health = {
            "overall_health": "healthy",
            "components": {},
            "timestamp": datetime.now().isoformat(),
        }

        try:
            # Check database connection
            if self.db_pool:
                async with self.db_pool.acquire() as conn:
                    await conn.fetchval("SELECT 1")
                health["components"]["database"] = "healthy"
            else:
                health["components"]["database"] = "unhealthy"

            # Check enhanced RAG pipeline
            rag_health = await self.enhanced_rag.health_check()
            health["components"]["enhanced_rag"] = rag_health["overall_health"]

            # Overall health
            if any(status == "unhealthy" for status in health["components"].values()):
                health["overall_health"] = "unhealthy"

        except Exception as e:
            health["overall_health"] = "error"
            health["error"] = str(e)

        return health


# Factory for creating project RAG pipeline instances
class ProjectRAGPipelineFactory:
    """Factory for creating project RAG pipeline instances"""

    @staticmethod
    def create_default_pipeline() -> ProjectRAGPipeline:
        """Create default project RAG pipeline"""
        return ProjectRAGPipeline()

    @staticmethod
    def create_pipeline_with_config(
        db_url: str, context_stream: UnifiedContextStream
    ) -> ProjectRAGPipeline:
        """Create project RAG pipeline with custom configuration"""
        return ProjectRAGPipeline(db_url=db_url, context_stream=context_stream)


# Global instance for easy access
_project_rag_pipeline = None


def get_project_rag_pipeline() -> ProjectRAGPipeline:
    """Get global project RAG pipeline instance"""
    global _project_rag_pipeline
    if _project_rag_pipeline is None:
        _project_rag_pipeline = ProjectRAGPipelineFactory.create_default_pipeline()
    return _project_rag_pipeline
