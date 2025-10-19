"""
METIS Unified Data Contract Architecture
F001: CloudEvents-compliant data schema for all system components

Based on PRD v7 architectural specifications and N-WAY framework analysis.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from uuid import UUID, uuid4
import logging

# Set up logging for memory management
logger = logging.getLogger(__name__)




class EngagementPhase(str, Enum):
    """MECE-based consulting workflow phases"""

    PROBLEM_STRUCTURING = "problem_structuring"
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    ANALYSIS_EXECUTION = "analysis_execution"
    RESEARCH_GROUNDING = "research_grounding"
    VALIDATION_DEBATE = "validation_debate"
    SYNTHESIS_DELIVERY = "synthesis_delivery"




class MentalModelCategory(str, Enum):
    """MeMo framework mental model categories"""

    SYSTEMS_THINKING = "systems_thinking"
    CRITICAL_THINKING = "critical_thinking"
    STRATEGIC_FRAMEWORKS = "strategic_frameworks"
    ANALYTICAL_METHODS = "analytical_methods"
    DECISION_MODELS = "decision_models"




class ConfidenceLevel(str, Enum):
    """Progressive transparency confidence indicators"""

    HIGH = "high"  # >90% confidence
    MEDIUM = "medium"  # 70-90% confidence
    LOW = "low"  # 50-70% confidence
    UNCERTAIN = "uncertain"  # <50% confidence




class VulnerabilityDetectionLevel(str, Enum):
    """Vulnerability detection severity levels"""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"




class ExplorationDecision(str, Enum):
    """Exploration vs exploitation strategy decisions"""

    EXPLOIT = "exploit"  # Use known-good models
    EXPLORE = "explore"  # Try new model combinations
    BALANCED = "balanced"  # Mixed strategy




class ClarificationQuestionType(str, Enum):
    """Types of clarification questions for HITL interaction"""

    OPEN_ENDED = "open_ended"
    MULTIPLE_CHOICE = "multiple_choice"
    YES_NO = "yes_no"
    NUMERIC = "numeric"




class ClarificationComplexity(str, Enum):
    """Complexity levels for clarification questions"""

    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


# Operation Synapse Sprint 1.4: Manus Taxonomy Implementation


class ContextType(str, Enum):
    """Manus Labs Context Taxonomy classification"""

    IMMEDIATE = "immediate"  # Current request and user intent
    SESSION = "session"  # Conversation history and established context
    DOMAIN = "domain"  # Relevant knowledge and expertise area
    PROCEDURAL = "procedural"  # How-to knowledge and methodologies
    TEMPORAL = "temporal"  # Time-sensitive information and trends
    RELATIONAL = "relational"  # Connections and dependencies




class ContextRelevanceLevel(str, Enum):
    """Context relevance scoring levels based on Manus methodology"""

    CRITICAL = "critical"  # >0.9 relevance score
    HIGH = "high"  # 0.7-0.9 relevance score
    MEDIUM = "medium"  # 0.5-0.7 relevance score
    LOW = "low"  # 0.3-0.5 relevance score
    IRRELEVANT = "irrelevant"  # <0.3 relevance score




class CognitiveCacheLevel(str, Enum):
    """Multi-layer cache levels following Cognition.ai pattern"""

    L1_MEMORY = "l1_memory"  # In-memory hot cache
    L2_REDIS = "l2_redis"  # Distributed Redis cache
    L3_PERSISTENT = "l3_persistent"  # Database persistent cache




class StrategicLayer(str, Enum):
    """Strategic layer classification for consultant matrix"""

    STRATEGIC = "strategic"
    TACTICAL = "tactical"
    OPERATIONAL = "operational"




class CognitiveFunction(str, Enum):
    """Cognitive function classification for consultant matrix"""

    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    IMPLEMENTATION = "implementation"




class ExtendedConsultantRole(str, Enum):
    """Extended consultant roles for 9-consultant matrix"""

    # Strategic Layer
    STRATEGIC_ANALYST = "strategic_analyst"
    STRATEGIC_SYNTHESIZER = "strategic_synthesizer"
    STRATEGIC_IMPLEMENTER = "strategic_implementer"

    # Tactical Layer
    TACTICAL_PROBLEM_SOLVER = "tactical_problem_solver"
    TACTICAL_SOLUTION_ARCHITECT = "tactical_solution_architect"
    TACTICAL_BUILDER = "tactical_builder"

    # Operational Layer
    OPERATIONAL_PROCESS_EXPERT = "operational_process_expert"
    OPERATIONAL_INTEGRATOR = "operational_integrator"
    OPERATIONAL_EXECUTION_SPECIALIST = "operational_execution_specialist"




class FeedbackTier(str, Enum):
    """Multi-tier feedback system tiers"""

    BRONZE = "bronze"  # Basic metrics
    SILVER = "silver"  # Enhanced analytics
    GOLD = "gold"  # Strategic insights
    PLATINUM = "platinum"  # Full partnership




