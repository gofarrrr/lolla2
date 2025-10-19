"""
Blueprint Registry Service - Extracted from OptimalConsultantEngine
Manages consultant blueprints with database integration and fallback capability

This service handles all consultant blueprint management, role mapping, and provides
complete Glass-Box transparency for consultant selection decisions.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

# Glass-Box Integration - CRITICAL
from src.engine.adapters.context_stream import UnifiedContextStream  # Migrated to adapter, ContextEventType

# Database integration
try:
    from supabase import Client

    SUPABASE_AVAILABLE = True
except ImportError:
    Client = None
    SUPABASE_AVAILABLE = False


@dataclass
class ConsultantBlueprint:
    """Consultant blueprint definition - extracted from monolith"""

    consultant_id: str
    name: str
    specialization: str
    expertise: str
    persona_prompt: str
    stable_frameworks: List[str]
    adaptive_triggers: List[str] = field(default_factory=list)
    effectiveness_score: float = 0.8
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BlueprintRegistryResult:
    """Result of blueprint registry operations"""

    success: bool
    blueprints_loaded: int
    source: str  # "database", "fallback", "cache"
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BlueprintRegistry:
    """
    Consultant Blueprint Registry - Extracted Service

    Manages consultant blueprints with database integration, fallback capability,
    and complete Glass-Box audit trail integration.

    Extracted from OptimalConsultantEngine monolith with proper dependency injection.
    """

    def __init__(
        self,
        context_stream: UnifiedContextStream,
        supabase_client: Optional[Client] = None,
    ):
        """
        Initialize Blueprint Registry with dependency injection

        Args:
            context_stream: UnifiedContextStream for Glass-Box transparency
            supabase_client: Optional Supabase client for database access
        """
        self.context_stream = context_stream
        self.supabase_client = supabase_client
        self.logger = logging.getLogger(__name__)

        self.consultant_blueprints: Dict[str, ConsultantBlueprint] = {}
        self.role_mappings: Dict[str, str] = {}
        self.blueprint_cache: Dict[str, ConsultantBlueprint] = {}
        self.load_timestamp: Optional[datetime] = None

        # Initialize registry
        self._initialize_registry()

    def _initialize_registry(self) -> None:
        """Initialize the blueprint registry with database or fallback data"""

        # Glass-Box: Log registry initialization start
        self.context_stream.add_event(
            event_type=ContextEventType.SYSTEM_STATE,
            data={
                "registry_initialization": "started",
                "supabase_available": bool(self.supabase_client),
            },
            metadata={"service": "BlueprintRegistry", "method": "_initialize_registry"},
        )

        try:
            if self.supabase_client:
                result = self._load_from_database()
            else:
                result = self._load_fallback_blueprints()

            self.load_timestamp = datetime.utcnow()

            # Glass-Box: Log successful initialization
            self.context_stream.add_event(
                event_type=ContextEventType.SYSTEM_STATE,
                data={
                    "registry_initialization": "completed",
                    "blueprints_loaded": result.blueprints_loaded,
                    "source": result.source,
                },
                metadata={"service": "BlueprintRegistry", "load_source": result.source},
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize blueprint registry: {e}")

            # Glass-Box: Log initialization error
            self.context_stream.add_event(
                event_type=ContextEventType.ERROR_OCCURRED,
                data={"error": str(e), "fallback_initiated": True},
                metadata={
                    "service": "BlueprintRegistry",
                    "method": "_initialize_registry",
                },
            )

            # Fallback to basic blueprints
            self._load_fallback_blueprints()

    def _load_from_database(self) -> BlueprintRegistryResult:
        """Load consultant blueprints from Supabase database"""
        try:
            # Query agent_personas table
            result = self.supabase_client.table("agent_personas").select("*").execute()

            blueprints_loaded = 0
            for row in result.data:
                try:
                    blueprint = ConsultantBlueprint(
                        consultant_id=row["consultant_id"],
                        name=row["name"],
                        specialization=row.get("specialization", "general"),
                        expertise=row.get("expertise", ""),
                        persona_prompt=row.get("persona_prompt", ""),
                        stable_frameworks=row.get("stable_frameworks", []),
                        adaptive_triggers=row.get("adaptive_triggers", []),
                        effectiveness_score=float(row.get("effectiveness_score", 0.8)),
                        metadata={
                            "loaded_from_db": True,
                            "load_timestamp": datetime.utcnow().isoformat(),
                        },
                    )

                    self.consultant_blueprints[blueprint.consultant_id] = blueprint
                    blueprints_loaded += 1

                except Exception as e:
                    self.logger.warning(
                        f"Failed to parse blueprint {row.get('consultant_id', 'unknown')}: {e}"
                    )

            self._setup_role_mappings()

            return BlueprintRegistryResult(
                success=True,
                blueprints_loaded=blueprints_loaded,
                source="database",
                metadata={"database_connection": True},
            )

        except Exception as e:
            self.logger.error(f"Database loading failed: {e}")
            # Fallback if database fails
            return self._load_fallback_blueprints()

    def _load_fallback_blueprints(self) -> BlueprintRegistryResult:
        """Load fallback consultant blueprints when database is unavailable"""

        fallback_blueprints = {
            "market_analyst": ConsultantBlueprint(
                consultant_id="market_analyst",
                name="Market Intelligence Analyst",
                specialization="strategic_analysis",
                expertise="market dynamics, competitive intelligence",
                persona_prompt="You are a market intelligence analyst specializing in competitive dynamics.",
                stable_frameworks=["BCG Strategy Palette", "Porter Five Forces"],
                adaptive_triggers=["strategic", "market", "competitive"],
                effectiveness_score=0.8,
                metadata={"fallback": True},
            ),
            "problem_solver": ConsultantBlueprint(
                consultant_id="problem_solver",
                name="Problem Diagnosis Expert",
                specialization="tactical_analysis",
                expertise="root cause analysis, problem breakdown",
                persona_prompt="You are a problem diagnosis expert specializing in root cause analysis.",
                stable_frameworks=["Issue Trees & Logic Trees", "MECE Principle"],
                adaptive_triggers=["problem", "diagnose", "analysis"],
                effectiveness_score=0.85,
                metadata={"fallback": True},
            ),
            "solution_architect": ConsultantBlueprint(
                consultant_id="solution_architect",
                name="Solution Design Architect",
                specialization="tactical_synthesis",
                expertise="solution design, methodology integration",
                persona_prompt="You are a solution architect specializing in comprehensive solution design.",
                stable_frameworks=["Value Proposition Design", "Lateral Thinking"],
                adaptive_triggers=["solution", "design", "creative"],
                effectiveness_score=0.82,
                metadata={"fallback": True},
            ),
            "strategic_synthesizer": ConsultantBlueprint(
                consultant_id="strategic_synthesizer",
                name="Strategic Synthesis Expert",
                specialization="strategic_synthesis",
                expertise="strategic integration, high-level synthesis",
                persona_prompt="You are a strategic synthesis expert specializing in high-level integration.",
                stable_frameworks=["Systems Thinking", "Strategic Integration"],
                adaptive_triggers=["strategy", "synthesis", "integration"],
                effectiveness_score=0.88,
                metadata={"fallback": True},
            ),
            "implementation_specialist": ConsultantBlueprint(
                consultant_id="implementation_specialist",
                name="Implementation Specialist",
                specialization="tactical_implementation",
                expertise="execution planning, implementation roadmaps",
                persona_prompt="You are an implementation specialist focused on practical execution.",
                stable_frameworks=["Implementation Planning", "Change Management"],
                adaptive_triggers=["implement", "execute", "action"],
                effectiveness_score=0.83,
                metadata={"fallback": True},
            ),
        }

        self.consultant_blueprints = fallback_blueprints
        self._setup_role_mappings()

        self.logger.info("⚠️ Using fallback consultant blueprints")

        return BlueprintRegistryResult(
            success=True,
            blueprints_loaded=len(fallback_blueprints),
            source="fallback",
            metadata={"database_connection": False},
        )

    def _setup_role_mappings(self) -> None:
        """Setup role mappings for backward compatibility"""

        self.role_mappings = {
            # New role IDs to legacy consultant IDs
            "strategic_analyst": "market_analyst",
            "strategic_synthesizer": "strategic_synthesizer",
            "strategic_implementer": "implementation_specialist",
            "tactical_problem_solver": "problem_solver",
            "tactical_solution_architect": "solution_architect",
            "tactical_builder": "solution_architect",  # Fallback
            "operational_process_expert": "implementation_specialist",
            "operational_integrator": "solution_architect",  # Fallback
            "operational_execution_specialist": "implementation_specialist",
        }

    def get_blueprint(self, consultant_id: str) -> Optional[ConsultantBlueprint]:
        """
        Get consultant blueprint by ID with Glass-Box logging

        Args:
            consultant_id: The consultant ID to retrieve

        Returns:
            ConsultantBlueprint if found, None otherwise
        """

        # Glass-Box: Log blueprint retrieval
        self.context_stream.add_event(
            event_type=ContextEventType.TOOL_EXECUTION,
            data={"consultant_id": consultant_id, "operation": "get_blueprint"},
            metadata={"service": "BlueprintRegistry", "method": "get_blueprint"},
        )

        # Direct lookup first
        if consultant_id in self.consultant_blueprints:
            blueprint = self.consultant_blueprints[consultant_id]

            # Glass-Box: Log successful retrieval
            self.context_stream.add_event(
                event_type=ContextEventType.CONSULTANT_SELECTION,
                data={
                    "consultant_found": True,
                    "consultant_name": blueprint.name,
                    "specialization": blueprint.specialization,
                },
                metadata={"service": "BlueprintRegistry", "lookup_type": "direct"},
            )

            return blueprint

        # Role mapping lookup
        if consultant_id in self.role_mappings:
            mapped_id = self.role_mappings[consultant_id]
            if mapped_id in self.consultant_blueprints:
                blueprint = self.consultant_blueprints[mapped_id]

                # Glass-Box: Log role mapping success
                self.context_stream.add_event(
                    event_type=ContextEventType.CONSULTANT_SELECTION,
                    data={
                        "consultant_found": True,
                        "original_id": consultant_id,
                        "mapped_id": mapped_id,
                        "consultant_name": blueprint.name,
                    },
                    metadata={
                        "service": "BlueprintRegistry",
                        "lookup_type": "role_mapping",
                    },
                )

                return blueprint

        # Glass-Box: Log blueprint not found
        self.context_stream.add_event(
            event_type=ContextEventType.ERROR_OCCURRED,
            data={"consultant_id": consultant_id, "error": "Blueprint not found"},
            metadata={"service": "BlueprintRegistry", "lookup_result": "not_found"},
        )

        return None

    def get_blueprint_for_role(self, role_id: str) -> Optional[ConsultantBlueprint]:
        """
        Get blueprint for a specific role with compatibility mapping

        Args:
            role_id: The role ID to look up

        Returns:
            ConsultantBlueprint if found, None otherwise
        """
        return self.get_blueprint(role_id)

    def get_all_blueprints(self) -> Dict[str, ConsultantBlueprint]:
        """Get all available consultant blueprints"""

        # Glass-Box: Log bulk retrieval
        self.context_stream.add_event(
            event_type=ContextEventType.TOOL_EXECUTION,
            data={
                "operation": "get_all_blueprints",
                "blueprints_count": len(self.consultant_blueprints),
            },
            metadata={"service": "BlueprintRegistry", "method": "get_all_blueprints"},
        )

        return self.consultant_blueprints.copy()

    def get_blueprints_by_specialization(
        self, specialization: str
    ) -> List[ConsultantBlueprint]:
        """Get blueprints filtered by specialization"""

        matching_blueprints = [
            blueprint
            for blueprint in self.consultant_blueprints.values()
            if blueprint.specialization == specialization
        ]

        # Glass-Box: Log filtered retrieval
        self.context_stream.add_event(
            event_type=ContextEventType.TOOL_EXECUTION,
            data={
                "operation": "get_blueprints_by_specialization",
                "specialization": specialization,
                "matches_found": len(matching_blueprints),
            },
            metadata={"service": "BlueprintRegistry", "filter_type": "specialization"},
        )

        return matching_blueprints

    def add_blueprint(self, blueprint: ConsultantBlueprint) -> bool:
        """
        Add a new consultant blueprint

        Args:
            blueprint: The ConsultantBlueprint to add

        Returns:
            True if successful, False otherwise
        """
        try:
            self.consultant_blueprints[blueprint.consultant_id] = blueprint

            # Glass-Box: Log blueprint addition
            self.context_stream.add_event(
                event_type=ContextEventType.SYSTEM_STATE,
                data={
                    "operation": "add_blueprint",
                    "consultant_id": blueprint.consultant_id,
                    "consultant_name": blueprint.name,
                },
                metadata={"service": "BlueprintRegistry", "method": "add_blueprint"},
            )

            return True

        except Exception as e:
            self.logger.error(f"Failed to add blueprint {blueprint.consultant_id}: {e}")

            # Glass-Box: Log addition error
            self.context_stream.add_event(
                event_type=ContextEventType.ERROR_OCCURRED,
                data={
                    "operation": "add_blueprint",
                    "consultant_id": blueprint.consultant_id,
                    "error": str(e),
                },
                metadata={"service": "BlueprintRegistry", "method": "add_blueprint"},
            )

            return False

    def get_registry_status(self) -> Dict[str, Any]:
        """Get registry status and health information"""

        return {
            "total_blueprints": len(self.consultant_blueprints),
            "load_timestamp": (
                self.load_timestamp.isoformat() if self.load_timestamp else None
            ),
            "database_connected": bool(self.supabase_client),
            "role_mappings_count": len(self.role_mappings),
            "specializations": list(
                set(bp.specialization for bp in self.consultant_blueprints.values())
            ),
            "effectiveness_scores": {
                bp.consultant_id: bp.effectiveness_score
                for bp in self.consultant_blueprints.values()
            },
        }

    def reload_blueprints(self) -> BlueprintRegistryResult:
        """Reload blueprints from source (database or fallback)"""

        # Glass-Box: Log reload request
        self.context_stream.add_event(
            event_type=ContextEventType.SYSTEM_STATE,
            data={
                "operation": "reload_blueprints",
                "current_count": len(self.consultant_blueprints),
            },
            metadata={"service": "BlueprintRegistry", "method": "reload_blueprints"},
        )

        # Clear current blueprints
        self.consultant_blueprints.clear()
        self.role_mappings.clear()

        # Reload
        self._initialize_registry()

        return BlueprintRegistryResult(
            success=True,
            blueprints_loaded=len(self.consultant_blueprints),
            source="reload",
            metadata={"reload_timestamp": datetime.utcnow().isoformat()},
        )


# Factory function for dependency injection
def create_blueprint_registry(
    context_stream: UnifiedContextStream, supabase_client: Optional[Client] = None
) -> BlueprintRegistry:
    """
    Factory function to create BlueprintRegistry with proper dependencies

    This ensures proper dependency injection and Glass-Box integration
    """
    return BlueprintRegistry(
        context_stream=context_stream, supabase_client=supabase_client
    )
