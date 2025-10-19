"""
Stable interface definitions for parallel analysis executor components.

These interfaces define the contract between modular components, enabling:
- Independent implementation and testing of each component
- Easy mocking for unit tests
- Future extensibility (different implementations)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from .types import (
    PromptSpec,
    LLMResult,
    AggregatedOutput,
    EnrichedOutput,
    ExecutionPolicy,
    AggregationPolicy,
    DepthContext,
)


class PromptBuilder(ABC):
    """
    Interface for building consultant prompts.

    Responsibilities:
    - Construct consultant-specific prompts from problem context
    - Apply persona templates and prompt engineering
    - Validate token budgets
    - Generate prompt fingerprints for auditability
    """

    @abstractmethod
    def build(
        self,
        problem_context: str,
        consultant_blueprints: List[Dict[str, Any]],
        framework: Optional[Dict[str, Any]] = None,
        enhanced_prompts: Optional[List[str]] = None,
        depth_packs: Optional[Dict[str, str]] = None,
    ) -> List[PromptSpec]:
        """
        Build prompts for all consultants.

        Args:
            problem_context: User's problem/query
            consultant_blueprints: Selected consultant configurations
            framework: MECE framework from problem structuring (optional)
            enhanced_prompts: Additional prompt enhancements (optional)
            depth_packs: Stage 0 depth packs by consultant_id (optional)

        Returns:
            List of PromptSpec objects ready for execution

        Raises:
            ValueError: If token budget is exceeded
        """
        pass

    @abstractmethod
    def estimate_tokens(self, prompt_spec: PromptSpec) -> int:
        """
        Estimate total tokens for a prompt.

        Args:
            prompt_spec: Prompt specification

        Returns:
            Estimated token count (system + user + expected response)
        """
        pass

    @abstractmethod
    def generate_fingerprint(self, prompt_spec: PromptSpec) -> str:
        """
        Generate audit fingerprint for prompt.

        Args:
            prompt_spec: Prompt specification

        Returns:
            SHA-256 hash of prompt content for traceability
        """
        pass


class Runner(ABC):
    """
    Interface for parallel LLM execution.

    Responsibilities:
    - Execute LLM calls in parallel with configurable concurrency
    - Handle retries with exponential backoff
    - Enforce timeouts and resource limits
    - Collect execution metrics (tokens, latency)
    """

    @abstractmethod
    async def execute(
        self,
        prompts: List[PromptSpec],
        policy: ExecutionPolicy,
    ) -> List[LLMResult]:
        """
        Execute prompts in parallel.

        Args:
            prompts: List of prompts to execute
            policy: Execution policy (parallelism, timeouts, retries)

        Returns:
            List of LLMResult objects (one per prompt)

        Note:
            - Results order matches input prompts order
            - Failed calls return LLMResult with success=False
        """
        pass

    @abstractmethod
    async def execute_single(
        self,
        prompt: PromptSpec,
        policy: ExecutionPolicy,
    ) -> LLMResult:
        """
        Execute a single prompt (with retries).

        Args:
            prompt: Prompt to execute
            policy: Execution policy

        Returns:
            LLMResult with response or error

        Note:
            Implements retry logic from policy.retry_config
        """
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get execution metrics.

        Returns:
            Dict with metrics:
            - total_calls: int
            - successful_calls: int
            - failed_calls: int
            - total_tokens: int
            - total_time_ms: int
            - avg_latency_ms: float
            - retry_count: int
        """
        pass


class Aggregator(ABC):
    """
    Interface for aggregating consultant results.

    Responsibilities:
    - Merge individual LLM results into unified output
    - Compute orthogonality index (cognitive diversity)
    - Identify convergent insights and divergent perspectives
    - Deduplicate recommendations
    """

    @abstractmethod
    def merge(
        self,
        results: List[LLMResult],
        policy: AggregationPolicy,
    ) -> AggregatedOutput:
        """
        Merge consultant results.

        Args:
            results: List of LLM results from consultants
            policy: Aggregation policy (merge strategy, orthogonality config)

        Returns:
            AggregatedOutput with merged insights and metrics

        Note:
            - Handles failed results gracefully (skips them)
            - Computes orthogonality if policy.compute_orthogonality=True
        """
        pass

    @abstractmethod
    def compute_orthogonality(
        self,
        consultant_analyses: List[Dict[str, Any]],
    ) -> float:
        """
        Compute orthogonality index (cognitive diversity score).

        Args:
            consultant_analyses: Individual consultant analysis results

        Returns:
            Orthogonality index 0.0-1.0:
            - 0.0 = perfect groupthink (all consultants agree)
            - 1.0 = maximum diversity (completely different perspectives)

        Algorithm:
            Measures semantic diversity of key insights across consultants
        """
        pass

    @abstractmethod
    def identify_convergence(
        self,
        consultant_analyses: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Identify convergent insights (agreed upon by multiple consultants).

        Args:
            consultant_analyses: Individual consultant analysis results

        Returns:
            List of insights present in 2+ consultant analyses
        """
        pass

    @abstractmethod
    def identify_divergence(
        self,
        consultant_analyses: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Identify divergent perspectives (areas of disagreement).

        Args:
            consultant_analyses: Individual consultant analysis results

        Returns:
            List of perspectives where consultants disagree
        """
        pass


class DepthPack(ABC):
    """
    Interface for Stage 0 depth enrichment.

    Responsibilities:
    - Build depth packs for consultants (mental models, context)
    - Integrate Q&A precision retrieval
    - Apply breadth/depth variant treatment
    - Track enrichment metrics
    """

    @abstractmethod
    def enrich(
        self,
        aggregated: AggregatedOutput,
        context: DepthContext,
    ) -> EnrichedOutput:
        """
        Apply depth enrichment to aggregated output.

        Args:
            aggregated: Base aggregated output
            context: Depth enrichment context/config

        Returns:
            EnrichedOutput with depth pack metadata

        Note:
            - If context.enable_stage0=False, returns aggregated as-is
            - Enrichment is idempotent (can be called multiple times safely)
        """
        pass

    @abstractmethod
    def build_consultant_depth_pack(
        self,
        consultant_id: str,
        consultant_type: str,
        problem_context: str,
        enrichment_level: str,
    ) -> str:
        """
        Build depth pack for a single consultant.

        Args:
            consultant_id: Consultant identifier
            consultant_type: Consultant type (strategic, tactical, etc.)
            problem_context: Problem description
            enrichment_level: Enrichment depth (breadth/depth/full)

        Returns:
            Depth pack text to inject into consultant prompt
        """
        pass

    @abstractmethod
    def get_enrichment_metrics(self) -> Dict[str, Any]:
        """
        Get depth enrichment metrics.

        Returns:
            Dict with metrics:
            - total_depth_tokens: int
            - mm_items_count: int (mental model items)
            - stage0_latency_ms: int
            - variant_label: str
        """
        pass


# ============================================================================
# HELPER PROTOCOLS (for type hints without circular imports)
# ============================================================================

class LLMClientProtocol(ABC):
    """Protocol for LLM client dependency (used by Runner)"""

    @abstractmethod
    async def complete(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute LLM completion"""
        pass


class ContextStreamProtocol(ABC):
    """Protocol for context stream dependency (for event logging)"""

    @abstractmethod
    def add_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Add event to context stream"""
        pass
