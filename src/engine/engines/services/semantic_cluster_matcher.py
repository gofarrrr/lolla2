"""
METIS V5 Semantic Cluster Matching Service
=========================================

Extracted from monolithic optimal_consultant_engine.py (lines 611-871).
Handles N-Way cluster selection using semantic vector search, manual similarity, and keyword fallback.

Part of the Great Refactoring: Clean separation of cluster matching concerns.
"""

from typing import List, Optional, Dict, Any
import numpy as np

# Import our new contracts

# Import UnifiedContextStream for audit trail
from src.core.unified_context_stream import UnifiedContextStream, ContextEventType

# Project Semantic Core: Import sentence transformer for semantic search
try:
    from sentence_transformers import SentenceTransformer

    SEMANTIC_DEPENDENCIES_AVAILABLE = True
except ImportError:
    # Silence during tests; fallback paths handle behavior
    SentenceTransformer = None
    SEMANTIC_DEPENDENCIES_AVAILABLE = False

# Supabase for database operations
try:
    from supabase import Client

    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False


class SemanticClusterMatchingService:
    """
    Stateless service for N-Way cluster selection using semantic matching.

    Extracted from OptimalConsultantEngine to follow Single Responsibility Principle.
    Supports three-tier selection: semantic vector search → manual similarity → keyword fallback.
    """

    def __init__(
        self,
        supabase_client: Optional[Client] = None,
        context_stream: Optional[UnifiedContextStream] = None,
    ):
        """Initialize the semantic cluster matching service"""
        self.supabase = supabase_client
        from src.core.unified_context_stream import get_unified_context_stream, UnifiedContextStream
        self.context_stream = context_stream or get_unified_context_stream()

        # Initialize semantic model if dependencies are available
        self.semantic_model = None
        self.semantic_search_enabled = False
        self.manual_similarity_available = True  # Always available as fallback

        if SEMANTIC_DEPENDENCIES_AVAILABLE:
            try:
                self.semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
                self.semantic_search_enabled = True
                print(
                    "✅ SemanticClusterMatchingService: Semantic model initialized (Level 3)"
                )
            except Exception as e:
                print(f"⚠️ Failed to initialize semantic model: {e}")
                self.semantic_search_enabled = False

        print(
            f"✅ SemanticClusterMatchingService: Initialized with {'semantic vector search' if self.semantic_search_enabled else 'fallback methods'}"
        )

    async def select_relevant_nway_clusters(
        self, enhanced_query: str, top_k: int = 3
    ) -> List[str]:
        """
        Main entry point: Select most relevant N-Way Interaction clusters.

        Three-tier selection strategy:
        1. Semantic vector search (Level 3) - if available
        2. Manual similarity calculation (Level 2) - if semantic model available but pgvector not
        3. Keyword fallback (Level 1) - always available

        Args:
            enhanced_query: The processed query to match against
            top_k: Number of top clusters to return

        Returns:
            List of cluster IDs ordered by relevance
        """
        try:
            # Determine selection method and log it
            selection_method = (
                "semantic_vector_search"
                if self.semantic_search_enabled
                else "keyword_fallback"
            )
            logger = __import__('logging').getLogger(__name__)
            logger.debug(
                f"Using {selection_method} for N-Way cluster selection"
            )

            if self.semantic_search_enabled:
                return await self._select_nway_clusters_semantic(enhanced_query, top_k)
            elif self.manual_similarity_available:
                return await self._select_nway_clusters_manual_similarity(
                    enhanced_query, top_k
                )
            else:
                return await self._select_nway_clusters_keyword_fallback(
                    enhanced_query, top_k
                )

        except Exception as e:
            print(f"❌ Error selecting N-Way clusters: {e}")
            # Fallback chain: semantic → manual → keyword
            if self.semantic_search_enabled:
                print("🔄 Falling back to manual similarity method...")
                try:
                    return await self._select_nway_clusters_manual_similarity(
                        enhanced_query, top_k
                    )
                except Exception as fallback_e:
                    print(
                        f"🔄 Manual similarity failed, falling back to keyword method... {fallback_e}"
                    )
                    return await self._select_nway_clusters_keyword_fallback(
                        enhanced_query, top_k
                    )
            elif self.manual_similarity_available:
                print("🔄 Falling back to keyword method...")
                return await self._select_nway_clusters_keyword_fallback(
                    enhanced_query, top_k
                )
            return []

    async def _select_nway_clusters_semantic(
        self, enhanced_query: str, top_k: int = 3
    ) -> List[str]:
        """
        Level 3: Semantic vector search implementation for N-Way cluster selection.

        Uses sentence transformer embeddings and pgvector cosine similarity.
        Extracted from optimal_consultant_engine.py lines 643-710.
        """
        if not self.supabase:
            raise ValueError("Supabase client required for semantic vector search")

        try:
            print("🎯 Executing semantic vector search for N-Way clusters...")

            # Step 1: Generate query embedding
            query_embedding = self.semantic_model.encode([enhanced_query])[0].tolist()

            # Step 2: Perform semantic search using pgvector (cosine similarity)
            # Note: This requires pgvector extension and embedding column in nway_interactions table
            result = self.supabase.rpc(
                "semantic_cluster_search",
                {
                    "query_embedding": query_embedding,
                    "match_threshold": 0.3,
                    "match_count": top_k,
                },
            ).execute()

            if not result.data:
                print(
                    "⚠️ No semantically similar N-Way clusters found, trying manual calculation..."
                )
                # Fallback: get all clusters and calculate similarity manually
                return await self._select_nway_clusters_manual_similarity(
                    enhanced_query, top_k
                )

            # Step 3: Process results and create scored clusters
            scored_clusters = []
            for row in result.data:
                scored_clusters.append(
                    {
                        "interaction_id": row["interaction_id"],
                        "relevance_score": float(row["similarity_score"]),
                        "title": row["title"],
                        "models": row["models_involved"],
                    }
                )

            # Step 4: Sort by relevance and extract cluster IDs
            selected_clusters = sorted(
                scored_clusters, key=lambda x: x["relevance_score"], reverse=True
            )[:top_k]
            cluster_ids = [cluster["interaction_id"] for cluster in selected_clusters]

            # Step 5: Log to unified context stream
            if self.context_stream:
                self.context_stream.add_event(
                    ContextEventType.REASONING_STEP,
                    "Semantic N-Way cluster selection completed",
                    {
                        "phase": "nway_cluster_selection",
                        "selection_method": "semantic_vector_search",
                        "embedding_model": "all-MiniLM-L6-v2",
                        "selected_nway_clusters": [
                            {
                                "cluster_id": cluster["interaction_id"],
                                "similarity_score": cluster["relevance_score"],
                                "title": cluster["title"],
                                "models_involved": cluster["models"],
                            }
                            for cluster in selected_clusters
                        ],
                        "selection_reasoning": f"Selected top {len(selected_clusters)} N-Way clusters using semantic vector search with cosine similarity",
                    },
                    metadata={
                        "source": "SemanticClusterMatchingService._select_nway_clusters_semantic"
                    },
                )

            print(
                f"🎯 Semantically selected {len(cluster_ids)} N-Way clusters: {cluster_ids}"
            )
            return cluster_ids

        except Exception as e:
            print(f"❌ Semantic search failed: {e}")
            raise

    async def _select_nway_clusters_manual_similarity(
        self, enhanced_query: str, top_k: int
    ) -> List[str]:
        """
        Level 2: Manual similarity calculation when pgvector function is not available.

        Fetches all clusters and calculates cosine similarity manually.
        Extracted from optimal_consultant_engine.py lines 711-786.
        """
        if not self.supabase:
            raise ValueError(
                "Supabase client required for manual similarity calculation"
            )

        if not self.semantic_model:
            # If no semantic model, fall back to keyword method
            return await self._select_nway_clusters_keyword_fallback(
                enhanced_query, top_k
            )

        try:
            print("🧮 Manual similarity calculation for N-Way clusters...")

            # Step 1: Generate query embedding
            query_embedding = self.semantic_model.encode([enhanced_query])[0]

            # Step 2: Fetch all N-Way clusters with embeddings
            result = (
                self.supabase.table("nway_interactions")
                .select("interaction_id, title, models_involved, embedding")
                .eq("nway_type", "CORE")
                .execute()
            )

            if not result.data:
                print("⚠️ No N-Way clusters found in database")
                return []

            # Step 3: Calculate cosine similarity manually
            scored_clusters = []
            for row in result.data:
                if row["embedding"]:  # Skip clusters without embeddings
                    try:
                        cluster_embedding = np.array(row["embedding"])

                        # Cosine similarity calculation
                        dot_product = np.dot(query_embedding, cluster_embedding)
                        query_norm = np.linalg.norm(query_embedding)
                        cluster_norm = np.linalg.norm(cluster_embedding)

                        if query_norm > 0 and cluster_norm > 0:
                            similarity_score = dot_product / (query_norm * cluster_norm)
                        else:
                            similarity_score = 0.0

                        scored_clusters.append(
                            {
                                "interaction_id": row["interaction_id"],
                                "relevance_score": float(similarity_score),
                                "title": row["title"],
                                "models": row["models_involved"],
                            }
                        )
                    except Exception as embedding_error:
                        print(
                            f"⚠️ Error processing embedding for cluster {row['interaction_id']}: {embedding_error}"
                        )
                        continue

            # Step 4: Sort by similarity and select top_k
            selected_clusters = sorted(
                scored_clusters, key=lambda x: x["relevance_score"], reverse=True
            )[:top_k]
            cluster_ids = [cluster["interaction_id"] for cluster in selected_clusters]

            # Step 5: Log to context stream
            if self.context_stream:
                self.context_stream.add_event(
                    ContextEventType.REASONING_STEP,
                    "Manual semantic N-Way cluster selection completed",
                    {
                        "phase": "nway_cluster_selection",
                        "selection_method": "manual_semantic_similarity",
                        "selected_nway_clusters": [
                            {
                                "cluster_id": cluster["interaction_id"],
                                "similarity_score": cluster["relevance_score"],
                                "title": cluster["title"],
                            }
                            for cluster in selected_clusters
                        ],
                    },
                    metadata={
                        "source": "SemanticClusterMatchingService._select_nway_clusters_manual_similarity"
                    },
                )

            print(
                f"🧮 Manually calculated semantic selection: {len(cluster_ids)} clusters"
            )
            return cluster_ids

        except Exception as e:
            print(f"❌ Manual similarity calculation failed: {e}")
            raise

    async def _select_nway_clusters_keyword_fallback(
        self, enhanced_query: str, top_k: int = 3
    ) -> List[str]:
        """
        Level 1: Keyword-based selection (fallback method).

        Uses simple keyword matching and scoring.
        Extracted from optimal_consultant_engine.py lines 787-871.
        """
        if not self.supabase:
            # If no database access, return empty list
            print("⚠️ No database access - cannot perform cluster selection")
            return []

        try:
            print("🔤 Using keyword-based N-Way cluster selection (fallback)...")

            # Step 1: Extract keywords from query
            query_lower = enhanced_query.lower()
            keywords = [
                word.strip('.,!?;:"()[]')
                for word in query_lower.split()
                if len(word) > 3
            ]

            # Step 2: Fetch all N-Way clusters
            result = (
                self.supabase.table("nway_interactions")
                .select("interaction_id, title, description, models_involved")
                .eq("nway_type", "CORE")
                .execute()
            )

            if not result.data:
                print("⚠️ No N-Way clusters found in database")
                return []

            # Step 3: Score clusters based on keyword matches
            scored_clusters = []
            for row in result.data:
                # Combine searchable text
                searchable_text = (
                    f"{row.get('title', '')} {row.get('description', '')}".lower()
                )

                # Count keyword matches
                keyword_matches = sum(
                    1 for keyword in keywords if keyword in searchable_text
                )

                # Calculate relevance score
                relevance_score = keyword_matches / max(
                    len(keywords), 1
                )  # Normalize by query keyword count

                if relevance_score > 0:  # Only include clusters with some relevance
                    scored_clusters.append(
                        {
                            "interaction_id": row["interaction_id"],
                            "relevance_score": relevance_score,
                            "title": row.get("title", ""),
                            "models": row.get("models_involved", []),
                            "keyword_matches": keyword_matches,
                        }
                    )

            # Step 4: Sort by relevance and select top_k
            selected_clusters = sorted(
                scored_clusters, key=lambda x: x["relevance_score"], reverse=True
            )[:top_k]
            cluster_ids = [cluster["interaction_id"] for cluster in selected_clusters]

            # Step 5: Log to context stream
            if self.context_stream:
                self.context_stream.add_event(
                    ContextEventType.REASONING_STEP,
                    "Keyword N-Way cluster selection completed",
                    {
                        "phase": "nway_cluster_selection",
                        "selection_method": "keyword_fallback",
                        "selected_nway_clusters": [
                            {
                                "cluster_id": cluster["interaction_id"],
                                "relevance_score": cluster["relevance_score"],
                                "title": cluster["title"],
                                "models_involved": cluster["models"],
                            }
                            for cluster in selected_clusters
                        ],
                        "selection_reasoning": f"Selected top {len(selected_clusters)} N-Way clusters based on keyword matching (fallback method)",
                    },
                    metadata={
                        "source": "SemanticClusterMatchingService._select_nway_clusters_keyword_fallback"
                    },
                )

            print(
                f"🔤 Keyword-selected {len(cluster_ids)} N-Way clusters: {cluster_ids}"
            )
            return cluster_ids

        except Exception as e:
            print(f"❌ Keyword fallback selection failed: {e}")
            return []

    # === UTILITY METHODS ===

    def get_selection_capabilities(self) -> Dict[str, Any]:
        """Return current selection capabilities"""
        return {
            "semantic_search_available": self.semantic_search_enabled,
            "manual_similarity_available": self.manual_similarity_available
            and bool(self.semantic_model),
            "keyword_fallback_available": True,
            "database_available": bool(self.supabase),
            "current_tier": (
                "Level 3 (Semantic Vector Search)"
                if self.semantic_search_enabled
                else (
                    "Level 2 (Manual Similarity)"
                    if self.semantic_model
                    else "Level 1 (Keyword Fallback)"
                )
            ),
        }

    def configure_database(self, supabase_client: Client):
        """Configure the database client"""
        self.supabase = supabase_client
        print("✅ SemanticClusterMatchingService: Database client configured")

    def configure_context_stream(self, context_stream: UnifiedContextStream):
        """Configure the audit trail context stream"""
        self.context_stream = context_stream
        print("✅ SemanticClusterMatchingService: Context stream configured")


# Factory function for service creation
def get_semantic_cluster_matching_service(
    supabase_client: Optional[Client] = None,
    context_stream: Optional[UnifiedContextStream] = None,
) -> SemanticClusterMatchingService:
    """Factory function to create SemanticClusterMatchingService instance"""
    return SemanticClusterMatchingService(supabase_client, context_stream)
