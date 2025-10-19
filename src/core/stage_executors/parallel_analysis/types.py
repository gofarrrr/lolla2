"""
Data contracts and type definitions for parallel analysis executor.

These types define the data flow through the modular parallel analysis components:
Input → PromptBuilder → Runner → Aggregator → DepthPack → Output
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal
from enum import Enum


class MergeStrategy(str, Enum):
    """Strategy for merging consultant results"""
    MAJORITY = "majority"  # Majority vote for insights
    WEIGHTED = "weighted"  # Weight by consultant effectiveness
    BY_ROLE = "by_role"   # Merge by consultant role/type
    UNION = "union"       # Union of all insights (default)


class EnrichmentLevel(str, Enum):
    """Stage 0 enrichment depth level"""
    NONE = "none"
    BREADTH = "breadth"  # Breadth-first variant
    DEPTH = "depth"      # Depth-first variant
    FULL = "full"        # Both breadth and depth


# ============================================================================
# PROMPT BUILDING CONTRACTS
# ============================================================================

class PromptSpec(BaseModel):
    """Specification for a single consultant prompt"""

    consultant_id: str = Field(..., description="Unique consultant identifier")
    model: str = Field(..., description="LLM model to use (e.g., 'gpt-4', 'claude-3-sonnet')")
    system_prompt: str = Field(..., description="System/role prompt for consultant")
    user_prompt: str = Field(..., description="User query with context")

    # Optional metadata
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(None, description="Max tokens for response")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional prompt metadata")

    # Token budget tracking
    estimated_tokens: int = Field(0, ge=0, description="Estimated total tokens")

    class Config:
        json_schema_extra = {
            "example": {
                "consultant_id": "strategist_1",
                "model": "claude-3-sonnet-20240229",
                "system_prompt": "You are a strategic business consultant...",
                "user_prompt": "Analyze: How should we expand into APAC?",
                "temperature": 0.7,
                "max_tokens": 2000,
                "estimated_tokens": 3500
            }
        }


# ============================================================================
# EXECUTION CONTRACTS
# ============================================================================

class RetryConfig(BaseModel):
    """Configuration for LLM retry logic"""

    max_attempts: int = Field(3, ge=1, le=10, description="Maximum retry attempts")
    initial_delay_s: float = Field(1.0, ge=0.1, description="Initial retry delay in seconds")
    backoff_multiplier: float = Field(2.0, ge=1.0, description="Exponential backoff multiplier")
    max_delay_s: float = Field(30.0, ge=1.0, description="Maximum retry delay in seconds")

    # Which errors to retry
    retry_on_timeout: bool = Field(True, description="Retry on timeout errors")
    retry_on_rate_limit: bool = Field(True, description="Retry on rate limit errors")
    retry_on_server_error: bool = Field(True, description="Retry on 5xx server errors")


class ExecutionPolicy(BaseModel):
    """Policy for parallel LLM execution"""

    parallelism: int = Field(3, ge=1, le=10, description="Max concurrent LLM calls")
    timeout_s: float = Field(60.0, gt=0, description="Timeout per LLM call in seconds")
    retry_config: RetryConfig = Field(default_factory=RetryConfig, description="Retry configuration")

    # Resource limits
    max_total_tokens: Optional[int] = Field(None, description="Total token budget across all calls")
    fail_fast: bool = Field(False, description="Stop all on first failure if True")


class LLMResult(BaseModel):
    """Result from a single LLM call"""

    consultant_id: str = Field(..., description="Consultant identifier")
    content: str = Field(..., description="LLM response content")

    # Execution metrics
    tokens_used: int = Field(0, ge=0, description="Actual tokens consumed")
    time_ms: int = Field(0, ge=0, description="Execution time in milliseconds")

    # Provider metadata
    model_used: str = Field(..., description="Actual model used")
    provider: str = Field(..., description="Provider (openai, anthropic, deepseek)")

    # Raw response (for debugging/auditing)
    raw_response: Optional[Dict[str, Any]] = Field(None, description="Raw provider response")

    # Error tracking
    success: bool = Field(True, description="Whether call succeeded")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    retry_count: int = Field(0, ge=0, description="Number of retries performed")

    class Config:
        json_schema_extra = {
            "example": {
                "consultant_id": "strategist_1",
                "content": "Based on the APAC expansion analysis...",
                "tokens_used": 1850,
                "time_ms": 3200,
                "model_used": "claude-3-sonnet-20240229",
                "provider": "anthropic",
                "success": True,
                "retry_count": 0
            }
        }


# ============================================================================
# AGGREGATION CONTRACTS
# ============================================================================

class AggregationPolicy(BaseModel):
    """Policy for aggregating consultant results"""

    merge_strategy: MergeStrategy = Field(
        MergeStrategy.UNION,
        description="Strategy for merging insights"
    )

    # Orthogonality computation
    compute_orthogonality: bool = Field(True, description="Compute orthogonality index")
    orthogonality_threshold: float = Field(0.3, ge=0.0, le=1.0, description="Threshold for divergence")

    # Deduplication
    deduplicate_insights: bool = Field(True, description="Remove duplicate insights")
    similarity_threshold: float = Field(0.8, ge=0.0, le=1.0, description="Similarity threshold for deduplication")

    # Quality filtering
    min_confidence_level: Optional[str] = Field(None, description="Filter by minimum confidence (low/medium/high)")


class AggregatedOutput(BaseModel):
    """Aggregated output from all consultants"""

    # Core results (maps to ParallelAnalysisOutput contract)
    consultant_analyses: List[Dict[str, Any]] = Field(
        ...,
        description="Individual consultant analysis results"
    )

    convergent_insights: List[str] = Field(
        default_factory=list,
        description="Insights agreed upon by multiple consultants"
    )

    divergent_perspectives: List[str] = Field(
        default_factory=list,
        description="Areas of consultant disagreement"
    )

    # Orthogonality metrics
    orthogonality_index: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Cognitive diversity score (0=groupthink, 1=diverse)"
    )

    minority_report: Optional[str] = Field(
        None,
        description="Contrarian perspective if orthogonality is low"
    )

    # Aggregation metadata
    merge_strategy_used: MergeStrategy = Field(
        MergeStrategy.UNION,
        description="Merge strategy applied"
    )

    deduplication_count: int = Field(
        0,
        ge=0,
        description="Number of duplicate insights removed"
    )

    total_insights: int = Field(0, ge=0, description="Total insights across all consultants")

    class Config:
        json_schema_extra = {
            "example": {
                "consultant_analyses": [
                    {
                        "consultant_id": "strategist_1",
                        "key_insights": ["Market entry via partnerships", "Focus on Singapore first"],
                        "confidence_level": "high"
                    }
                ],
                "convergent_insights": ["Regulatory compliance is critical"],
                "divergent_perspectives": ["Build vs partner debate"],
                "orthogonality_index": 0.65,
                "merge_strategy_used": "union",
                "total_insights": 12
            }
        }


# ============================================================================
# DEPTH ENRICHMENT CONTRACTS
# ============================================================================

class DepthContext(BaseModel):
    """Context for Stage 0 depth enrichment"""

    enable_stage0: bool = Field(False, description="Enable Stage 0 enrichment")
    enrichment_level: EnrichmentLevel = Field(
        EnrichmentLevel.NONE,
        description="Depth of enrichment to apply"
    )

    # Variant tracking (for A/B testing)
    variant_label: str = Field("control", description="Experiment variant (control/treatment/breadth/depth)")

    # Q&A precision retrieval integration
    enable_qa_precision: bool = Field(False, description="Enable Q&A precision retrieval")
    qa_context: Optional[Dict[str, Any]] = Field(None, description="Q&A context if available")

    # Enrichment limits
    max_depth_tokens: int = Field(1000, ge=0, description="Max tokens for depth pack per consultant")

    class Config:
        json_schema_extra = {
            "example": {
                "enable_stage0": True,
                "enrichment_level": "depth",
                "variant_label": "treatment_depth",
                "enable_qa_precision": True,
                "max_depth_tokens": 1500
            }
        }


class EnrichedOutput(BaseModel):
    """Output after depth enrichment"""

    base_output: AggregatedOutput = Field(..., description="Base aggregated output")

    # Enrichment metadata
    enrichment_applied: bool = Field(False, description="Whether enrichment was applied")
    enrichment_level: EnrichmentLevel = Field(
        EnrichmentLevel.NONE,
        description="Level of enrichment applied"
    )

    depth_pack_tokens: int = Field(0, ge=0, description="Total tokens from depth packs")
    mm_items_count: int = Field(0, ge=0, description="Mental model items added")

    # Stage 0 metrics
    stage0_latency_ms: int = Field(0, ge=0, description="Stage 0 processing time")

    def to_aggregated_output(self) -> AggregatedOutput:
        """Convert back to base AggregatedOutput (strips enrichment metadata)"""
        return self.base_output
