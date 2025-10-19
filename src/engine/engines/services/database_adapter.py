"""
METIS V5 Database Adapter Service
================================

Extracted from monolithic optimal_consultant_engine.py database operations.
Handles all Supabase interactions with proper error handling and fallbacks.

Part of the Great Refactoring: Clean separation of database concerns.
"""

import os
from typing import Dict, List, Optional, Any
from datetime import datetime

# Import our new contracts
from ..contracts import HealthStatus

# Supabase client
try:
    from supabase import create_client, Client

    SUPABASE_AVAILABLE = True
except ImportError:
    print("⚠️ Supabase not available - database adapter will operate in mock mode")
    SUPABASE_AVAILABLE = False
    Client = Any

# Import consultant blueprint (legacy compatibility)
from dataclasses import dataclass, field


@dataclass
class ConsultantBlueprint:
    consultant_id: str
    name: str
    specialization: str
    expertise: str
    persona_prompt: str
    stable_frameworks: List[str]
    adaptive_triggers: List[str] = field(default_factory=list)
    effectiveness_score: float = 0.8


class DatabaseAdapterService:
    """
    Stateless service for all database operations.

    Extracted from OptimalConsultantEngine to follow Single Responsibility Principle.
    Provides clean abstraction over Supabase with proper error handling and fallbacks.
    """

    def __init__(self, supabase_client: Optional[Client] = None):
        """Initialize the database adapter service"""
        if supabase_client:
            self.supabase = supabase_client
        elif SUPABASE_AVAILABLE:
            # Initialize Supabase client
            url = os.getenv("SUPABASE_URL", "https://soztmkgednwjhgzvlzch.supabase.co")
            key = os.getenv(
                "SUPABASE_SERVICE_ROLE_KEY",
                "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InNvenRta2dlZG53amhnenZsemNoIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NDk4MzYxNywiZXhwIjoyMDcwNTU5NjE3fQ.fe-1KftmBOE_sl4uuMrc0P88LWbKqZvCTEa9vimLARQ",
            )
            try:
                self.supabase = create_client(url, key)
                print("✅ DatabaseAdapterService: Supabase client initialized")
            except Exception as e:
                print(f"⚠️ Supabase initialization failed: {e}")
                self.supabase = None
        else:
            self.supabase = None

        self.connected = bool(self.supabase)
        print(
            f"✅ DatabaseAdapterService: {'Connected' if self.connected else 'Mock mode'}"
        )

    # === CONSULTANT BLUEPRINT OPERATIONS ===

    async def load_consultant_blueprints(self) -> Dict[str, ConsultantBlueprint]:
        """
        Load consultant blueprints from database.
        Extracted from optimal_consultant_engine.py lines 336-359.
        """
        if not self.connected:
            print("⚠️ Database not connected - returning empty blueprints")
            return {}

        try:
            # Query for consultant blueprints
            result = (
                self.supabase.table("knowledge_elements")
                .select("*")
                .contains("domain_tags", ["consultant"])
                .execute()
            )

            blueprints = {}
            for item in result.data:
                metadata = item.get("metadata", {})

                # Handle missing 'title' field gracefully
                title = item.get(
                    "title", item.get("name", f"consultant_{len(blueprints) + 1}")
                )
                consultant_id = metadata.get(
                    "consultant_id", title.lower().replace(" ", "_")
                )

                blueprint = ConsultantBlueprint(
                    consultant_id=consultant_id,
                    name=metadata.get("name", title),
                    specialization=metadata.get("specialization", ""),
                    expertise=item.get("content", ""),
                    persona_prompt=metadata.get("persona_prompt", ""),
                    stable_frameworks=metadata.get("stable_frameworks", []),
                    adaptive_triggers=metadata.get("adaptive_triggers", []),
                    effectiveness_score=metadata.get("effectiveness_score", 0.8),
                )
                blueprints[consultant_id] = blueprint

            print(f"✅ Loaded {len(blueprints)} consultant blueprints from database")

            # If no blueprints found in database, provide fallback defaults
            if not blueprints:
                print(
                    "⚠️ No consultant blueprints found in database - using fallback configuration"
                )
                return self._get_fallback_consultant_blueprints()

            return blueprints

        except Exception as e:
            print(f"❌ Failed to load consultant blueprints: {e}")
            print("⚠️ Using fallback consultant blueprints configuration")
            return self._get_fallback_consultant_blueprints()

    async def load_routing_patterns(self) -> Dict[str, Dict]:
        """
        Load routing patterns from database.
        Extracted from optimal_consultant_engine.py lines 364-385.
        """
        if not self.connected:
            print("⚠️ Database not connected - returning empty routing patterns")
            return {}

        try:
            # Query for routing patterns
            result = (
                self.supabase.table("knowledge_elements")
                .select("*")
                .contains("domain_tags", ["routing"])
                .execute()
            )

            routing_patterns = {}
            for item in result.data:
                # Handle missing 'title' field gracefully
                pattern_id = item.get(
                    "title",
                    item.get("name", f"routing_pattern_{len(routing_patterns) + 1}"),
                )
                routing_patterns[pattern_id] = {
                    "consultants": item.get("metadata", {}).get(
                        "preferred_consultants", []
                    ),
                    "description": item.get("content", ""),
                    "triggers": item.get("metadata", {}).get("triggers", []),
                }

            print(f"✅ Loaded {len(routing_patterns)} routing patterns")
            return routing_patterns

        except Exception as e:
            print(f"❌ Failed to load routing patterns: {e}")
            return {}

    # === N-WAY CLUSTER OPERATIONS ===

    async def semantic_cluster_search(
        self,
        query_embedding: List[float],
        match_threshold: float = 0.3,
        match_count: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic cluster search using pgvector.
        Extracted from semantic cluster matching logic.
        """
        if not self.connected:
            return []

        try:
            result = self.supabase.rpc(
                "semantic_cluster_search",
                {
                    "query_embedding": query_embedding,
                    "match_threshold": match_threshold,
                    "match_count": match_count,
                },
            ).execute()

            return result.data or []

        except Exception as e:
            print(f"❌ Semantic cluster search failed: {e}")
            return []

    async def get_nway_clusters_with_embeddings(self) -> List[Dict[str, Any]]:
        """
        Get all N-Way clusters with embeddings for manual similarity calculation.
        """
        if not self.connected:
            return []

        try:
            result = (
                self.supabase.table("nway_interactions")
                .select("*")
                .eq("nway_type", "CORE")
                .execute()
            )

            return result.data or []

        except Exception as e:
            print(f"❌ Failed to get N-Way clusters: {e}")
            return []

    async def get_nway_clusters_for_keyword_search(self) -> List[Dict[str, Any]]:
        """
        Get all N-Way clusters for keyword-based search.
        """
        if not self.connected:
            return []

        try:
            result = (
                self.supabase.table("nway_interactions")
                .select("*")
                .eq("nway_type", "CORE")
                .execute()
            )

            return result.data or []

        except Exception as e:
            print(f"❌ Failed to get N-Way clusters for keyword search: {e}")
            return []

    # === ENGAGEMENT STATE MANAGEMENT ===

    async def check_engagement_exists(self, engagement_id: str) -> bool:
        """
        Check if engagement exists in database.
        Extracted from optimal_consultant_engine.py lines 933-937.
        """
        if not self.connected:
            return False

        try:
            result = (
                self.supabase.table("engagements")
                .select("schema_version")
                .eq("id", engagement_id)
                .execute()
            )
            return bool(result.data)

        except Exception as e:
            print(f"❌ Error checking engagement existence: {e}")
            return False

    async def get_engagement_state(self, engagement_id: str) -> Optional[str]:
        """
        Get current state of engagement.
        Extracted from optimal_consultant_engine.py lines 1015-1020.
        """
        if not self.connected:
            return None

        try:
            result = (
                self.supabase.table("engagements")
                .select("current_phase")
                .eq("id", engagement_id)
                .execute()
            )

            if result.data and result.data[0].get("current_phase"):
                return result.data[0]["current_phase"]
            return None

        except Exception as e:
            print(f"❌ Error getting engagement state: {e}")
            return None

    async def create_or_update_engagement(
        self, engagement_id: str, engagement_data: Dict[str, Any]
    ) -> bool:
        """
        Create or update engagement record.
        Extracted from optimal_consultant_engine.py lines 1045-1055.
        """
        if not self.connected:
            print("⚠️ Database not connected - cannot persist engagement")
            return False

        try:
            # Insert or update engagement record
            self.supabase.table("engagements").upsert(
                {
                    "id": engagement_id,
                    "current_phase": "processing",
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                    **engagement_data,
                }
            ).execute()

            print(f"✅ Engagement {engagement_id} persisted successfully")
            return True

        except Exception as e:
            print(f"❌ Failed to persist engagement: {e}")
            return False

    # === RESULT PERSISTENCE ===

    async def persist_consultant_result(
        self, engagement_id: str, consultant_id: str, result_data: Dict[str, Any]
    ) -> bool:
        """
        Persist individual consultant result.
        Extracted from optimal_consultant_engine.py lines 1535-1542.
        """
        if not self.connected:
            print("⚠️ Database not connected - cannot persist consultant result")
            return False

        try:
            full_result_data = {
                "engagement_id": engagement_id,
                "consultant_id": consultant_id,
                "created_at": datetime.now().isoformat(),
                **result_data,
            }

            self.supabase.table("engagement_results").insert(full_result_data).execute()
            print(f"✅ Consultant result persisted: {consultant_id}")
            return True

        except Exception as e:
            print(f"❌ Failed to persist consultant result: {e}")
            return False

    async def persist_senior_advisor_report(
        self, engagement_id: str, senior_advisor_data: Dict[str, Any]
    ) -> bool:
        """
        Persist senior advisor report.
        Extracted from optimal_consultant_engine.py lines 1554-1561.
        """
        if not self.connected:
            print("⚠️ Database not connected - cannot persist senior advisor report")
            return False

        try:
            full_report_data = {
                "engagement_id": engagement_id,
                "created_at": datetime.now().isoformat(),
                **senior_advisor_data,
            }

            self.supabase.table("senior_advisor_reports").insert(
                full_report_data
            ).execute()
            print("✅ Senior advisor report persisted")
            return True

        except Exception as e:
            print(f"❌ Failed to persist senior advisor report: {e}")
            return False

    async def update_proof_of_work_stats(
        self, engagement_id: str, proof_of_work_stats: Dict[str, Any]
    ) -> bool:
        """
        Update proof of work statistics for engagement.
        Extracted from optimal_consultant_engine.py lines 1588-1592.
        """
        if not self.connected:
            print("⚠️ Database not connected - cannot update proof of work stats")
            return False

        try:
            update_result = (
                self.supabase.table("cognitive_engagements")
                .update(
                    {
                        "proof_of_work_stats": proof_of_work_stats,
                        "updated_at": datetime.now().isoformat(),
                    }
                )
                .eq("id", engagement_id)
                .execute()
            )

            print(f"✅ Proof of work stats updated for engagement {engagement_id}")
            return True

        except Exception as e:
            print(f"❌ Failed to update proof of work stats: {e}")
            return False

    # === KNOWLEDGE ELEMENT LOGGING ===

    async def log_knowledge_element(
        self, title: str, content: str, metadata: Dict[str, Any], domain_tags: List[str]
    ) -> bool:
        """
        Log knowledge element to database.
        Extracted from optimal_consultant_engine.py lines 2099-2108.
        """
        if not self.connected:
            print("⚠️ Database not connected - cannot log knowledge element")
            return False

        try:
            log_entry = {
                "title": title,
                "content": content,
                "metadata": metadata,
                "domain_tags": domain_tags,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }

            # Store in database
            result = (
                self.supabase.table("knowledge_elements").insert(log_entry).execute()
            )
            print(f"✅ Knowledge element logged: {title}")
            return True

        except Exception as e:
            print(f"❌ Failed to log knowledge element: {e}")
            return False

    # === HEALTH CHECK OPERATIONS ===

    async def health_check(self) -> HealthStatus:
        """
        Perform database health check.
        """
        start_time = datetime.now()

        try:
            if not self.connected:
                return HealthStatus(
                    component="DatabaseAdapterService",
                    healthy=False,
                    response_time_ms=None,
                    details="Database not connected",
                )

            # Simple health check query
            result = (
                self.supabase.table("knowledge_elements")
                .select("id")
                .limit(1)
                .execute()
            )

            response_time = (datetime.now() - start_time).total_seconds() * 1000

            return HealthStatus(
                component="DatabaseAdapterService",
                healthy=True,
                response_time_ms=response_time,
                details=f"Connected to Supabase, {len(result.data or [])} test records found",
            )

        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            return HealthStatus(
                component="DatabaseAdapterService",
                healthy=False,
                response_time_ms=response_time,
                details=f"Health check failed: {e}",
            )

    def _get_fallback_consultant_blueprints(self) -> Dict[str, ConsultantBlueprint]:
        """
        Provide fallback consultant blueprints when database is unavailable or empty.
        Essential for system operation when database connection fails.
        """
        fallback_blueprints = {
            "strategic_analyst": ConsultantBlueprint(
                consultant_id="strategic_analyst",
                name="Strategic Business Analyst",
                specialization="Strategic Planning & Market Analysis",
                expertise="Comprehensive business strategy development with focus on market positioning, competitive analysis, and growth planning. Specializes in identifying strategic opportunities and developing actionable business recommendations.",
                persona_prompt="You are a senior strategic analyst with 15+ years of experience in management consulting. Approach problems systematically with structured analysis and data-driven insights.",
                stable_frameworks=[
                    "Porter's Five Forces",
                    "SWOT Analysis",
                    "BCG Matrix",
                    "Value Chain Analysis",
                ],
                adaptive_triggers=[
                    "market_analysis",
                    "competitive_strategy",
                    "business_planning",
                    "strategic_planning",
                ],
                effectiveness_score=0.85,
            ),
            "financial_expert": ConsultantBlueprint(
                consultant_id="financial_expert",
                name="Financial Strategy Expert",
                specialization="Financial Analysis & Investment Planning",
                expertise="Deep expertise in financial modeling, investment analysis, risk assessment, and capital allocation strategies. Provides quantitative analysis and financial recommendations for business decisions.",
                persona_prompt="You are a seasoned financial expert with deep experience in corporate finance and investment analysis. Focus on quantitative metrics and financial risk assessment.",
                stable_frameworks=[
                    "DCF Analysis",
                    "NPV/IRR",
                    "Risk-Return Analysis",
                    "Capital Structure Optimization",
                ],
                adaptive_triggers=[
                    "financial_analysis",
                    "investment_planning",
                    "risk_assessment",
                    "capital_allocation",
                ],
                effectiveness_score=0.82,
            ),
            "operations_specialist": ConsultantBlueprint(
                consultant_id="operations_specialist",
                name="Operations & Process Expert",
                specialization="Operational Excellence & Process Optimization",
                expertise="Expertise in operational efficiency, process improvement, supply chain optimization, and organizational effectiveness. Focuses on practical implementation and operational scalability.",
                persona_prompt="You are an operations expert with extensive experience in process optimization and operational excellence. Think practically about implementation challenges and scalability.",
                stable_frameworks=[
                    "Lean Six Sigma",
                    "Process Mapping",
                    "Value Stream Analysis",
                    "Operational KPIs",
                ],
                adaptive_triggers=[
                    "process_improvement",
                    "operational_efficiency",
                    "supply_chain",
                    "organizational_design",
                ],
                effectiveness_score=0.80,
            ),
        }

        print(
            f"✅ Fallback consultant blueprints created: {len(fallback_blueprints)} consultants available"
        )
        return fallback_blueprints

    # === UTILITY METHODS ===

    def get_connection_status(self) -> Dict[str, Any]:
        """Get current connection status and capabilities"""
        return {
            "connected": self.connected,
            "supabase_available": SUPABASE_AVAILABLE,
            "client_initialized": bool(self.supabase),
            "capabilities": {
                "consultant_blueprints": self.connected,
                "routing_patterns": self.connected,
                "engagement_persistence": self.connected,
                "semantic_search": self.connected,
                "knowledge_logging": self.connected,
            },
        }

    def configure_client(self, supabase_client: Client):
        """Configure a new Supabase client"""
        self.supabase = supabase_client
        self.connected = True
        print("✅ DatabaseAdapterService: New client configured")


# Factory function for service creation
def get_database_adapter_service(
    supabase_client: Optional[Client] = None,
) -> DatabaseAdapterService:
    """Factory function to create DatabaseAdapterService instance"""
    return DatabaseAdapterService(supabase_client)
