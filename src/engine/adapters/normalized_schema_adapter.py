"""
Normalized Schema Adapter for ContextualLollapaloozaEngine
=========================================================

This adapter provides methods for querying the new normalized schema tables
created by Operation "Architectural Purity". It replaces the old knowledge_elements
queries with queries against the new mental_models, consultant_personas, and
nway_interactions tables.

Usage:
    adapter = NormalizedSchemaAdapter(supabase_client)
    mental_models = await adapter.get_mental_models_by_domain(['strategy', 'analysis'])
    consultants = await adapter.get_consultants_by_expertise(['systems-thinking'])
    nway_interactions = await adapter.get_nway_interactions_by_models(['systems-thinking', 'outside-view'])
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MentalModelRecord:
    """Represents a mental model from the normalized schema."""

    id: str
    ke_id: str
    ke_name: str
    definition_concise: str
    domain_tags: List[str]
    complexity_level: int
    effectiveness_score: float
    cqa_effectiveness_score: Optional[float]
    is_active: bool


@dataclass
class ConsultantPersonaRecord:
    """Represents a consultant persona from the normalized schema."""

    id: str
    consultant_id: str
    name: str
    consultant_type: str
    description: str
    expertise_domains: List[str]
    thinking_style_strengths: Dict[str, float]
    cognitive_complexity_preference: int
    avg_response_quality: float
    is_active: bool


@dataclass
class NWAYInteractionRecord:
    """Represents an NWAY interaction from the normalized schema."""

    id: str
    interaction_id: str
    models_involved: List[str]
    emergent_effect_summary: str
    strength: str
    strength_score: float
    lollapalooza_potential: Optional[float]
    relevant_contexts: List[str]
    is_active: bool


class NormalizedSchemaAdapter:
    """
    Adapter for querying the new normalized schema tables.

    This class provides a clean interface for the ContextualLollapaloozaEngine
    to query the new normalized database structure without needing to know
    the specific table schemas.
    """

    def __init__(self, supabase_client):
        """Initialize the schema adapter."""
        self.supabase = supabase_client
        self.logger = logging.getLogger(__name__)

        # Cache for performance
        self._mental_models_cache = None
        self._consultants_cache = None
        self._nway_interactions_cache = None

    async def get_mental_models_by_domain(
        self, domains: List[str]
    ) -> List[MentalModelRecord]:
        """
        Get mental models that match any of the specified domains.

        Args:
            domains: List of domain tags to match

        Returns:
            List of MentalModelRecord objects
        """
        try:
            self.logger.debug(f"Querying mental models by domains: {domains}")

            # Query mental_models table with domain filtering
            query = (
                self.supabase.table("mental_models").select("*").eq("is_active", True)
            )

            # Add domain filtering using PostgreSQL array operators
            for domain in domains:
                query = query.or_(f'domain_tags.cs.{{"[\\"{domain}\\"]"}}')

            result = query.execute()

            # Transform to MentalModelRecord objects
            mental_models = []
            for record in result.data:
                mental_model = MentalModelRecord(
                    id=record["id"],
                    ke_id=record["ke_id"],
                    ke_name=record["ke_name"],
                    definition_concise=record["definition_concise"],
                    domain_tags=record.get("domain_tags", []),
                    complexity_level=record.get("complexity_level", 3),
                    effectiveness_score=record.get("effectiveness_score", 7.0),
                    cqa_effectiveness_score=record.get("cqa_effectiveness_score"),
                    is_active=record.get("is_active", True),
                )
                mental_models.append(mental_model)

            self.logger.info(
                f"Found {len(mental_models)} mental models for domains: {domains}"
            )
            return mental_models

        except Exception as e:
            self.logger.error(f"Failed to query mental models by domain: {e}")
            return []

    async def get_mental_models_by_complexity(
        self, min_complexity: int, max_complexity: int
    ) -> List[MentalModelRecord]:
        """
        Get mental models within a complexity range.

        Args:
            min_complexity: Minimum complexity level (1-10)
            max_complexity: Maximum complexity level (1-10)

        Returns:
            List of MentalModelRecord objects
        """
        try:
            self.logger.debug(
                f"Querying mental models by complexity: {min_complexity}-{max_complexity}"
            )

            result = (
                self.supabase.table("mental_models")
                .select("*")
                .eq("is_active", True)
                .gte("complexity_level", min_complexity)
                .lte("complexity_level", max_complexity)
                .execute()
            )

            mental_models = []
            for record in result.data:
                mental_model = MentalModelRecord(
                    id=record["id"],
                    ke_id=record["ke_id"],
                    ke_name=record["ke_name"],
                    definition_concise=record["definition_concise"],
                    domain_tags=record.get("domain_tags", []),
                    complexity_level=record.get("complexity_level", 3),
                    effectiveness_score=record.get("effectiveness_score", 7.0),
                    cqa_effectiveness_score=record.get("cqa_effectiveness_score"),
                    is_active=record.get("is_active", True),
                )
                mental_models.append(mental_model)

            self.logger.info(
                f"Found {len(mental_models)} mental models in complexity range {min_complexity}-{max_complexity}"
            )
            return mental_models

        except Exception as e:
            self.logger.error(f"Failed to query mental models by complexity: {e}")
            return []

    async def get_consultants_by_expertise(
        self, expertise_domains: List[str]
    ) -> List[ConsultantPersonaRecord]:
        """
        Get consultant personas that have expertise in the specified domains.

        Args:
            expertise_domains: List of expertise domains to match

        Returns:
            List of ConsultantPersonaRecord objects
        """
        try:
            self.logger.debug(f"Querying consultants by expertise: {expertise_domains}")

            # Query consultant_personas table with expertise filtering
            query = (
                self.supabase.table("consultant_personas")
                .select("*")
                .eq("is_active", True)
            )

            # Add expertise filtering using PostgreSQL array operators
            for domain in expertise_domains:
                query = query.or_(f'expertise_domains.cs.{{"[\\"{domain}\\"]"}}')

            result = query.execute()

            consultants = []
            for record in result.data:
                consultant = ConsultantPersonaRecord(
                    id=record["id"],
                    consultant_id=record["consultant_id"],
                    name=record["name"],
                    consultant_type=record["consultant_type"],
                    description=record.get("description", ""),
                    expertise_domains=record.get("expertise_domains", []),
                    thinking_style_strengths=record.get("thinking_style_strengths", {}),
                    cognitive_complexity_preference=record.get(
                        "cognitive_complexity_preference", 5
                    ),
                    avg_response_quality=record.get("avg_response_quality", 0.0),
                    is_active=record.get("is_active", True),
                )
                consultants.append(consultant)

            self.logger.info(
                f"Found {len(consultants)} consultants with expertise in: {expertise_domains}"
            )
            return consultants

        except Exception as e:
            self.logger.error(f"Failed to query consultants by expertise: {e}")
            return []

    async def get_core_consultants_only(self) -> List[ConsultantPersonaRecord]:
        """
        Get only core consultant personas (not methodology frameworks).

        Returns:
            List of ConsultantPersonaRecord objects for core consultants
        """
        try:
            self.logger.debug("Querying core consultants only")

            result = (
                self.supabase.table("consultant_personas")
                .select("*")
                .eq("is_active", True)
                .eq("consultant_type", "core_consultant")
                .execute()
            )

            consultants = []
            for record in result.data:
                consultant = ConsultantPersonaRecord(
                    id=record["id"],
                    consultant_id=record["consultant_id"],
                    name=record["name"],
                    consultant_type=record["consultant_type"],
                    description=record.get("description", ""),
                    expertise_domains=record.get("expertise_domains", []),
                    thinking_style_strengths=record.get("thinking_style_strengths", {}),
                    cognitive_complexity_preference=record.get(
                        "cognitive_complexity_preference", 5
                    ),
                    avg_response_quality=record.get("avg_response_quality", 0.0),
                    is_active=record.get("is_active", True),
                )
                consultants.append(consultant)

            self.logger.info(f"Found {len(consultants)} core consultants")
            return consultants

        except Exception as e:
            self.logger.error(f"Failed to query core consultants: {e}")
            return []

    async def get_nway_interactions_by_models(
        self, model_ids: List[str]
    ) -> List[NWAYInteractionRecord]:
        """
        Get NWAY interactions that involve any of the specified mental models.

        Args:
            model_ids: List of mental model ke_ids to match

        Returns:
            List of NWAYInteractionRecord objects
        """
        try:
            self.logger.debug(f"Querying NWAY interactions by models: {model_ids}")

            # Query nway_interactions table with models filtering
            query = (
                self.supabase.table("nway_interactions")
                .select("*")
                .eq("is_active", True)
            )

            # Add models filtering using PostgreSQL array operators
            for model_id in model_ids:
                query = query.or_(f'models_involved.cs.{{"[\\"{model_id}\\"]"}}')

            result = query.execute()

            interactions = []
            for record in result.data:
                interaction = NWAYInteractionRecord(
                    id=record["id"],
                    interaction_id=record["interaction_id"],
                    models_involved=record.get("models_involved", []),
                    emergent_effect_summary=record.get("emergent_effect_summary", ""),
                    strength=record.get("strength", "Medium"),
                    strength_score=record.get("strength_score", 0.5),
                    lollapalooza_potential=record.get("lollapalooza_potential"),
                    relevant_contexts=record.get("relevant_contexts", []),
                    is_active=record.get("is_active", True),
                )
                interactions.append(interaction)

            self.logger.info(
                f"Found {len(interactions)} NWAY interactions involving models: {model_ids}"
            )
            return interactions

        except Exception as e:
            self.logger.error(f"Failed to query NWAY interactions by models: {e}")
            return []

    async def get_nway_interactions_by_strength(
        self, min_strength: str = "Medium"
    ) -> List[NWAYInteractionRecord]:
        """
        Get NWAY interactions with at least the specified strength.

        Args:
            min_strength: Minimum strength level ("Low", "Medium", "High")

        Returns:
            List of NWAYInteractionRecord objects
        """
        try:
            self.logger.debug(
                f"Querying NWAY interactions by minimum strength: {min_strength}"
            )

            strength_order = {"Low": 1, "Medium": 2, "High": 3}
            min_strength_value = strength_order.get(min_strength, 2)

            # Build query based on strength hierarchy
            if min_strength_value == 1:
                # Include all strengths
                strength_filter = ["Low", "Medium", "High"]
            elif min_strength_value == 2:
                # Include Medium and High
                strength_filter = ["Medium", "High"]
            else:
                # Include only High
                strength_filter = ["High"]

            query = (
                self.supabase.table("nway_interactions")
                .select("*")
                .eq("is_active", True)
            )
            for strength in strength_filter:
                query = query.or_(f"strength.eq.{strength}")

            result = query.execute()

            interactions = []
            for record in result.data:
                interaction = NWAYInteractionRecord(
                    id=record["id"],
                    interaction_id=record["interaction_id"],
                    models_involved=record.get("models_involved", []),
                    emergent_effect_summary=record.get("emergent_effect_summary", ""),
                    strength=record.get("strength", "Medium"),
                    strength_score=record.get("strength_score", 0.5),
                    lollapalooza_potential=record.get("lollapalooza_potential"),
                    relevant_contexts=record.get("relevant_contexts", []),
                    is_active=record.get("is_active", True),
                )
                interactions.append(interaction)

            self.logger.info(
                f"Found {len(interactions)} NWAY interactions with strength >= {min_strength}"
            )
            return interactions

        except Exception as e:
            self.logger.error(f"Failed to query NWAY interactions by strength: {e}")
            return []

    async def get_consultant_expertise_for_models(
        self, model_ids: List[str]
    ) -> Dict[str, List[str]]:
        """
        Get which consultants have expertise in the specified mental models.

        Args:
            model_ids: List of mental model ke_ids

        Returns:
            Dictionary mapping model_id to list of consultant_ids with that expertise
        """
        try:
            self.logger.debug(f"Querying consultant expertise for models: {model_ids}")

            result = (
                self.supabase.table("consultant_model_expertise")
                .select("consultant_id, mental_model_ke_id, expertise_strength")
                .in_("mental_model_ke_id", model_ids)
                .execute()
            )

            expertise_map = {}
            for record in result.data:
                model_id = record["mental_model_ke_id"]
                consultant_id = record["consultant_id"]

                if model_id not in expertise_map:
                    expertise_map[model_id] = []

                expertise_map[model_id].append(consultant_id)

            self.logger.info(
                f"Found expertise mappings for {len(expertise_map)} models"
            )
            return expertise_map

        except Exception as e:
            self.logger.error(f"Failed to query consultant expertise: {e}")
            return {}

    async def get_model_links_for_nway(self, interaction_id: str) -> List[str]:
        """
        Get the mental models involved in a specific NWAY interaction.

        Args:
            interaction_id: NWAY interaction ID

        Returns:
            List of mental model ke_ids linked to this interaction
        """
        try:
            self.logger.debug(f"Querying model links for NWAY: {interaction_id}")

            result = (
                self.supabase.table("nway_model_links")
                .select("mental_model_ke_id")
                .eq("interaction_id", interaction_id)
                .execute()
            )

            model_ids = [record["mental_model_ke_id"] for record in result.data]

            self.logger.info(
                f"Found {len(model_ids)} model links for NWAY {interaction_id}"
            )
            return model_ids

        except Exception as e:
            self.logger.error(f"Failed to query model links for NWAY: {e}")
            return []

    async def invalidate_cache(self) -> None:
        """Invalidate all cached data."""
        self._mental_models_cache = None
        self._consultants_cache = None
        self._nway_interactions_cache = None
        self.logger.info("Schema adapter cache invalidated")


# Example usage for updating ContextualLollapaloozaEngine
class ContextualLollapaloozaEngineV2:
    """
    Example of how to update the ContextualLollapaloozaEngine to use the new schema.
    This demonstrates the migration from knowledge_elements queries to normalized tables.
    """

    def __init__(self, supabase_client):
        """Initialize the V2 engine with schema adapter."""
        self.schema_adapter = NormalizedSchemaAdapter(supabase_client)
        self.logger = logging.getLogger(__name__)

    async def get_relevant_mental_models(
        self, query_context: Dict[str, Any]
    ) -> List[MentalModelRecord]:
        """
        Get mental models relevant to the query context using the new schema.

        This replaces the old knowledge_elements queries.
        """
        try:
            # Extract context information
            domains = query_context.get("domains", [])
            complexity_preference = query_context.get("complexity_preference", 5)

            # Query using normalized schema
            if domains:
                mental_models = await self.schema_adapter.get_mental_models_by_domain(
                    domains
                )
            else:
                # Get models by complexity if no domains specified
                mental_models = (
                    await self.schema_adapter.get_mental_models_by_complexity(
                        max(1, complexity_preference - 2),
                        min(10, complexity_preference + 2),
                    )
                )

            # Filter by effectiveness score
            min_effectiveness = query_context.get("min_effectiveness", 7.0)
            filtered_models = [
                model
                for model in mental_models
                if model.effectiveness_score >= min_effectiveness
            ]

            self.logger.info(f"Found {len(filtered_models)} relevant mental models")
            return filtered_models

        except Exception as e:
            self.logger.error(f"Failed to get relevant mental models: {e}")
            return []

    async def get_optimal_consultants(
        self, mental_models: List[str]
    ) -> List[ConsultantPersonaRecord]:
        """
        Get optimal consultants for the given mental models using the new schema.

        This replaces hardcoded consultant selection logic.
        """
        try:
            # Get consultants with expertise in these models
            consultants = await self.schema_adapter.get_consultants_by_expertise(
                mental_models
            )

            # Filter to core consultants only if needed
            core_consultants = [
                c for c in consultants if c.consultant_type == "core_consultant"
            ]

            # Sort by response quality
            optimal_consultants = sorted(
                core_consultants, key=lambda c: c.avg_response_quality, reverse=True
            )

            self.logger.info(f"Found {len(optimal_consultants)} optimal consultants")
            return optimal_consultants

        except Exception as e:
            self.logger.error(f"Failed to get optimal consultants: {e}")
            return []

    async def get_synergistic_nway_interactions(
        self, selected_models: List[str]
    ) -> List[NWAYInteractionRecord]:
        """
        Get NWAY interactions that can create synergy with the selected models.

        This uses the new normalized nway_interactions table.
        """
        try:
            # Get high-strength interactions involving these models
            interactions = await self.schema_adapter.get_nway_interactions_by_models(
                selected_models
            )

            # Filter to high-strength interactions only
            high_strength_interactions = [
                interaction
                for interaction in interactions
                if interaction.strength in ["High", "Medium"]
            ]

            # Sort by strength score
            synergistic_interactions = sorted(
                high_strength_interactions, key=lambda i: i.strength_score, reverse=True
            )

            self.logger.info(
                f"Found {len(synergistic_interactions)} synergistic NWAY interactions"
            )
            return synergistic_interactions

        except Exception as e:
            self.logger.error(f"Failed to get synergistic NWAY interactions: {e}")
            return []
