"""
METIS Supabase N-way Database Integration
Intelligent model selection based on N-way interactions
"""

import os
from typing import Dict, List, Any, Optional
from supabase import create_client, Client
import json


class NwayDatabaseManager:
    """Manages N-way interactions database for intelligent model selection"""

    def __init__(self):
        self.supabase: Optional[Client] = None
        self.initialized = False
        self.nway_data: List[Dict[str, Any]] = []
        self.data_source = "none"

    async def initialize(self):
        """Initialize data source - Supabase with JSON fallback"""
        # First attempt Supabase connection
        try:
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_ANON_KEY")

            if supabase_url and supabase_key:
                self.supabase = create_client(supabase_url, supabase_key)
                self.initialized = True
                self.data_source = "supabase"
                print("✅ Supabase connection initialized")
                return True
        except Exception as e:
            print(f"⚠️ Supabase connection failed: {e}")

        # Fallback to local JSON file
        try:
            json_path = os.path.join(
                os.path.dirname(__file__), "../../nway_interactions_clean.json"
            )
            if not os.path.exists(json_path):
                # Try alternate path
                json_path = "nway_interactions_clean.json"

            with open(json_path, "r") as f:
                self.nway_data = json.load(f)

            self.initialized = True
            self.data_source = "json"
            print(
                "⚠️ WARNING: Supabase connection failed. Falling back to local JSON for N-WAY patterns."
            )
            print("   System is operational but not using live database.")
            print(f"   Loaded {len(self.nway_data)} N-WAY patterns from local file.")
            return True

        except Exception as e:
            print(f"❌ Failed to load N-WAY data from any source: {e}")
            self.initialized = False
            return False

    async def migrate_nway_data(self, interactions: List[Dict[str, Any]]) -> bool:
        """Migrate N-way interactions to database"""
        if not self.initialized:
            await self.initialize()

        if not self.supabase:
            return False

        try:
            print(f"📤 Migrating {len(interactions)} interactions...")

            # Prepare data for insertion
            prepared_data = []
            for interaction in interactions:
                prepared_data.append(
                    {
                        "interaction_id": interaction.get("interaction_id"),
                        "models_involved": interaction.get("models_involved", []),
                        "primary_model_context": interaction.get(
                            "primary_model_context", ""
                        ),
                        "emergent_effect": interaction.get("emergent_effect", ""),
                        "mechanism_description": interaction.get(
                            "mechanism_description", ""
                        ),
                        "synergy_description": interaction.get(
                            "synergy_description", ""
                        ),
                        "strength": interaction.get("strength", "Medium"),
                        "relevant_contexts": interaction.get("relevant_contexts", []),
                        "instructional_cue_apce": interaction.get(
                            "instructional_cue_apce", ""
                        ),
                    }
                )

            # Batch insert
            result = (
                self.supabase.table("nway_interactions").insert(prepared_data).execute()
            )

            if result.data:
                print(f"✅ Successfully migrated {len(result.data)} interactions")
                return True
            else:
                print("❌ Migration failed: No data returned")
                return False

        except Exception as e:
            print(f"❌ Migration error: {e}")
            return False

    async def find_optimal_interaction(
        self, problem_keywords: List[str], domain: str, complexity: str = "High"
    ) -> Optional[Dict[str, Any]]:
        """Find the best N-way interaction for a given problem"""
        if not self.initialized:
            await self.initialize()

        if not self.initialized:
            return None

        try:
            # Get data from appropriate source
            if self.data_source == "supabase" and self.supabase:
                # [ARC] Removed strict .contains('relevant_contexts') filter to enable flexible,
                # keyword-score-based domain matching instead of rigid exact matching
                query = (
                    self.supabase.table("nway_interactions")
                    .select("*")
                    .eq("strength", "High")
                    .limit(10)
                )

                result = query.execute()
                interactions = result.data if result else []

                if not interactions:
                    # Fallback to medium strength
                    query = (
                        self.supabase.table("nway_interactions")
                        .select("*")
                        .eq("strength", "Medium")
                        .limit(5)
                    )

                    result = query.execute()
                    interactions = result.data if result else []

            elif self.data_source == "json":
                # Filter from local JSON data
                interactions = []
                for interaction in self.nway_data:
                    # Check strength
                    if interaction.get("strength") == "High" or (
                        not interactions and interaction.get("strength") == "Medium"
                    ):
                        # Check domain relevance
                        contexts = interaction.get("relevant_contexts", [])
                        if any(domain.lower() in ctx.lower() for ctx in contexts):
                            interactions.append(interaction)
                        # Also check for keyword matches
                        elif any(
                            kw.lower() in str(interaction).lower()
                            for kw in problem_keywords
                        ):
                            interactions.append(interaction)

                # Limit results
                interactions = interactions[:10]

            else:
                return None

            if interactions:
                # Score interactions based on keyword relevance
                scored_interactions = []
                for interaction in interactions:
                    score = self._calculate_relevance_score(
                        interaction, problem_keywords, domain
                    )
                    scored_interactions.append((score, interaction))

                # Return highest scoring interaction
                scored_interactions.sort(key=lambda x: x[0], reverse=True)

                if (
                    scored_interactions and scored_interactions[0][0] > 0.1
                ):  # Minimum relevance threshold
                    best_interaction = scored_interactions[0][1]

                    print(
                        f"🎯 Selected N-WAY pattern: {best_interaction.get('interaction_id', 'Unknown')}"
                    )
                    print(f"   Models: {best_interaction.get('models_involved', [])}")
                    synergy = best_interaction.get(
                        "emergent_effect_summary",
                        best_interaction.get("emergent_effect", "N/A"),
                    )
                    print(f"   Synergy: {synergy[:60]}...")

                    return best_interaction

            return None

        except Exception as e:
            print(f"❌ Query error: {e}")
            return None

    def _calculate_relevance_score(
        self, interaction: Dict[str, Any], keywords: List[str], domain: str
    ) -> float:
        """Calculate relevance score for an interaction"""
        score = 0.0

        # Domain match (high weight)
        if domain.lower() in [
            ctx.lower() for ctx in interaction.get("relevant_contexts", [])
        ]:
            score += 0.5

        # Keyword matches in description
        description = (
            interaction.get("synergy_description", "")
            + " "
            + interaction.get("emergent_effect", "")
        ).lower()

        for keyword in keywords:
            if keyword.lower() in description:
                score += 0.1

        # Strength bonus
        strength_bonus = {"High": 0.3, "Medium": 0.2, "Low": 0.1}
        score += strength_bonus.get(interaction.get("strength", "Low"), 0.0)

        return score

    async def get_interaction_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        if not self.initialized:
            await self.initialize()

        if not self.supabase:
            return {}

        try:
            # Total count
            total_result = (
                self.supabase.table("nway_interactions")
                .select("id", count="exact")
                .execute()
            )
            total_count = total_result.count

            # High strength count
            high_result = (
                self.supabase.table("nway_interactions")
                .select("id", count="exact")
                .eq("strength", "High")
                .execute()
            )
            high_count = high_result.count

            return {
                "total_interactions": total_count,
                "high_strength_interactions": high_count,
                "coverage_percentage": (
                    (high_count / total_count * 100) if total_count > 0 else 0
                ),
            }

        except Exception as e:
            print(f"❌ Stats error: {e}")
            return {}


# Global instance
_nway_manager: Optional[NwayDatabaseManager] = None


async def get_nway_manager() -> NwayDatabaseManager:
    """Get global N-way database manager"""
    global _nway_manager
    if _nway_manager is None:
        _nway_manager = NwayDatabaseManager()
        await _nway_manager.initialize()
    return _nway_manager
