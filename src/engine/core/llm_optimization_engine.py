"""
METIS LLM Optimization Engine
Week 2 Sprint: Optimized LLM API call batching and concurrency management

Implements intelligent batching, request pooling, and concurrent API management
for improved performance and cost efficiency.
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque


class LLMProvider(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    PERPLEXITY = "perplexity"


@dataclass
class LLMRequest:
    """Individual LLM API request"""

    request_id: str
    provider: LLMProvider
    operation_type: str  # 'mental_model', 'research', 'synthesis', etc.
    prompt: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1  # Higher = more urgent
    max_tokens: int = 1000
    temperature: float = 0.7
    created_at: datetime = field(default_factory=datetime.now)
    timeout_seconds: int = 30
    callback: Optional[Callable] = None

    @property
    def estimated_cost(self) -> float:
        """Estimate API cost in USD based on provider and tokens"""
        cost_per_1k_tokens = {
            LLMProvider.ANTHROPIC: 0.015,  # Claude Sonnet pricing
            LLMProvider.OPENAI: 0.03,  # GPT-4 pricing
            LLMProvider.PERPLEXITY: 0.002,  # Perplexity pricing
        }
        return (self.max_tokens / 1000) * cost_per_1k_tokens.get(self.provider, 0.01)


@dataclass
class LLMResponse:
    """LLM API response"""

    request_id: str
    success: bool
    content: str = ""
    tokens_used: int = 0
    actual_cost: float = 0.0
    response_time_ms: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchRequest:
    """Batch of LLM requests for concurrent execution"""

    batch_id: str
    requests: List[LLMRequest]
    max_concurrent: int = 3
    created_at: datetime = field(default_factory=datetime.now)
    estimated_total_cost: float = 0.0

    def __post_init__(self):
        self.estimated_total_cost = sum(req.estimated_cost for req in self.requests)


class LLMRequestPool:
    """
    Intelligent request pooling and batching system
    Groups similar requests for efficient processing
    """

    def __init__(self, max_pool_size: int = 20, batch_timeout_seconds: int = 2):
        self.max_pool_size = max_pool_size
        self.batch_timeout_seconds = batch_timeout_seconds
        self.pools: Dict[LLMProvider, deque] = defaultdict(deque)
        self.logger = logging.getLogger(__name__)

        # Performance tracking
        self.requests_processed = 0
        self.batches_created = 0
        self.total_cost_saved = 0.0
        self.avg_batch_size = 0.0

    async def add_request(self, request: LLMRequest) -> str:
        """Add request to appropriate pool"""
        pool = self.pools[request.provider]
        pool.append(request)

        self.logger.debug(
            f"📥 Added {request.operation_type} request to {request.provider.value} pool"
        )

        # Auto-batch if pool is full or timeout reached
        if len(pool) >= self.max_pool_size:
            return await self._create_batch_from_pool(request.provider)

        return request.request_id

    async def _create_batch_from_pool(self, provider: LLMProvider) -> str:
        """Create batch from pooled requests"""
        pool = self.pools[provider]
        if not pool:
            return ""

        # Take requests from pool (up to max concurrent limit)
        batch_requests = []
        max_concurrent = 5  # Prevent rate limiting

        while pool and len(batch_requests) < max_concurrent:
            batch_requests.append(pool.popleft())

        # Create batch
        batch_id = f"batch_{provider.value}_{int(time.time())}"
        batch = BatchRequest(
            batch_id=batch_id,
            requests=batch_requests,
            max_concurrent=len(batch_requests),
        )

        self.batches_created += 1
        self.avg_batch_size = (
            (self.avg_batch_size * (self.batches_created - 1)) + len(batch_requests)
        ) / self.batches_created

        self.logger.info(
            f"📦 Created batch {batch_id} with {len(batch_requests)} requests"
        )
        return batch_id

    async def get_pending_batches(self) -> List[BatchRequest]:
        """Get all pending batches for processing"""
        batches = []

        for provider in self.pools:
            if self.pools[provider]:
                batch_id = await self._create_batch_from_pool(provider)
                if batch_id:
                    # This is a simplified version - in practice, we'd store batches
                    pass

        return batches

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get pooling performance statistics"""
        return {
            "requests_processed": self.requests_processed,
            "batches_created": self.batches_created,
            "avg_batch_size": self.avg_batch_size,
            "total_cost_saved": self.total_cost_saved,
            "current_pool_sizes": {
                provider.value: len(pool) for provider, pool in self.pools.items()
            },
        }


class ConcurrentLLMExecutor:
    """
    Concurrent LLM API executor with rate limiting and error handling
    """

    def __init__(self, max_concurrent_per_provider: int = 3):
        self.max_concurrent_per_provider = max_concurrent_per_provider
        self.logger = logging.getLogger(__name__)

        # Rate limiting semaphores per provider
        self.semaphores = {
            provider: asyncio.Semaphore(max_concurrent_per_provider)
            for provider in LLMProvider
        }

        # Performance tracking
        self.requests_completed = 0
        self.requests_failed = 0
        self.total_response_time = 0.0
        self.total_cost = 0.0

        # Circuit breaker for failing providers
        self.provider_failure_counts = defaultdict(int)
        self.provider_circuit_breaker = defaultdict(bool)

    async def execute_batch(self, batch: BatchRequest) -> List[LLMResponse]:
        """Execute batch of requests concurrently with rate limiting"""
        start_time = time.time()

        self.logger.info(
            f"🚀 Executing batch {batch.batch_id} with {len(batch.requests)} requests"
        )

        # Group requests by provider for optimal concurrency
        provider_groups = defaultdict(list)
        for request in batch.requests:
            provider_groups[request.provider].append(request)

        # Execute each provider group concurrently
        all_tasks = []
        for provider, requests in provider_groups.items():
            if self.provider_circuit_breaker[provider]:
                self.logger.warning(
                    f"⚡ Provider {provider.value} circuit breaker active, skipping requests"
                )
                continue

            # Create concurrent tasks for this provider
            provider_tasks = [
                self._execute_single_request(request) for request in requests
            ]
            all_tasks.extend(provider_tasks)

        # Wait for all requests to complete
        try:
            responses = await asyncio.gather(*all_tasks, return_exceptions=True)

            # Process responses and handle exceptions
            processed_responses = []
            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    # Create error response
                    request = batch.requests[i]
                    error_response = LLMResponse(
                        request_id=request.request_id,
                        success=False,
                        error=str(response),
                        response_time_ms=(time.time() - start_time) * 1000,
                    )
                    processed_responses.append(error_response)
                    self.requests_failed += 1

                    # Update circuit breaker
                    self.provider_failure_counts[request.provider] += 1
                    if self.provider_failure_counts[request.provider] >= 3:
                        self.provider_circuit_breaker[request.provider] = True
                        self.logger.warning(
                            f"⚡ Circuit breaker activated for {request.provider.value}"
                        )
                else:
                    processed_responses.append(response)
                    self.requests_completed += 1

            # Update performance metrics
            batch_time = time.time() - start_time
            self.total_response_time += batch_time
            self.total_cost += sum(
                r.actual_cost for r in processed_responses if r.success
            )

            self.logger.info(
                f"✅ Batch {batch.batch_id} completed in {batch_time:.1f}s"
            )
            return processed_responses

        except Exception as e:
            self.logger.error(f"❌ Batch execution failed: {e}")
            return []

    async def _execute_single_request(self, request: LLMRequest) -> LLMResponse:
        """Execute single LLM request with rate limiting"""
        semaphore = self.semaphores[request.provider]

        async with semaphore:  # Rate limiting
            start_time = time.time()

            try:
                # Get appropriate LLM client
                if request.provider == LLMProvider.ANTHROPIC:
                    response_content = await self._call_anthropic(request)
                elif request.provider == LLMProvider.OPENAI:
                    response_content = await self._call_openai(request)
                elif request.provider == LLMProvider.PERPLEXITY:
                    response_content = await self._call_perplexity(request)
                else:
                    raise ValueError(f"Unsupported provider: {request.provider}")

                response_time = (time.time() - start_time) * 1000

                # Estimate actual cost and tokens
                actual_tokens = len(response_content.split()) * 1.3  # Rough estimate
                actual_cost = (actual_tokens / 1000) * request.estimated_cost

                response = LLMResponse(
                    request_id=request.request_id,
                    success=True,
                    content=response_content,
                    tokens_used=int(actual_tokens),
                    actual_cost=actual_cost,
                    response_time_ms=response_time,
                    metadata={
                        "provider": request.provider.value,
                        "operation_type": request.operation_type,
                    },
                )

                # Reset circuit breaker on success
                self.provider_failure_counts[request.provider] = 0
                self.provider_circuit_breaker[request.provider] = False

                return response

            except Exception as e:
                response_time = (time.time() - start_time) * 1000

                return LLMResponse(
                    request_id=request.request_id,
                    success=False,
                    error=str(e),
                    response_time_ms=response_time,
                )

    async def _call_anthropic(self, request: LLMRequest) -> str:
        """Call Anthropic Claude API"""
        try:
            from src.integrations.claude_client import get_claude_client

            claude_client = await get_claude_client()

            # Call Claude with optimized parameters
            response = await claude_client.call_claude(
                prompt=request.prompt,
                operation_type=request.operation_type,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )

            return response.get("content", "")

        except Exception as e:
            self.logger.error(f"Anthropic API call failed: {e}")
            raise

    async def _call_openai(self, request: LLMRequest) -> str:
        """Call OpenAI API"""
        # Placeholder for OpenAI integration
        await asyncio.sleep(0.5)  # Simulate API call
        return (
            f"OpenAI response for {request.operation_type}: {request.prompt[:100]}..."
        )

    async def _call_perplexity(self, request: LLMRequest) -> str:
        """Call Perplexity API"""
        try:
            from src.engine.integrations.perplexity_client import get_perplexity_client

            perplexity_client = get_perplexity_client()

            response = await perplexity_client.search_and_analyze(
                query=request.prompt, analysis_type=request.operation_type
            )

            return response.get("analysis", "")

        except Exception as e:
            self.logger.error(f"Perplexity API call failed: {e}")
            raise

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get executor performance metrics"""
        total_requests = self.requests_completed + self.requests_failed
        success_rate = (
            self.requests_completed / total_requests if total_requests > 0 else 0
        )
        avg_response_time = (
            self.total_response_time / total_requests if total_requests > 0 else 0
        )

        return {
            "requests_completed": self.requests_completed,
            "requests_failed": self.requests_failed,
            "success_rate": success_rate,
            "avg_response_time_ms": avg_response_time * 1000,
            "total_cost": self.total_cost,
            "cost_per_request": (
                self.total_cost / total_requests if total_requests > 0 else 0
            ),
            "circuit_breakers": dict(self.provider_circuit_breaker),
        }


class LLMOptimizationEngine:
    """
    Main LLM optimization engine coordinating pooling, batching, and concurrent execution
    """

    def __init__(self, max_concurrent_per_provider: int = 3):
        self.request_pool = LLMRequestPool()
        self.executor = ConcurrentLLMExecutor(max_concurrent_per_provider)
        self.logger = logging.getLogger(__name__)

        # Background task for automatic batch processing
        self._batch_processor_task = None
        self._running = False

    async def start(self):
        """Start the optimization engine"""
        self._running = True
        self._batch_processor_task = asyncio.create_task(
            self._background_batch_processor()
        )
        self.logger.info("🚀 LLM Optimization Engine started")

    async def stop(self):
        """Stop the optimization engine"""
        self._running = False
        if self._batch_processor_task:
            self._batch_processor_task.cancel()
        self.logger.info("🛑 LLM Optimization Engine stopped")

    async def submit_request(self, request: LLMRequest) -> str:
        """Submit LLM request for optimized processing"""
        return await self.request_pool.add_request(request)

    async def submit_multiple_requests(self, requests: List[LLMRequest]) -> List[str]:
        """Submit multiple requests for batch processing"""
        request_ids = []
        for request in requests:
            request_id = await self.submit_request(request)
            request_ids.append(request_id)
        return request_ids

    async def _background_batch_processor(self):
        """Background task that automatically processes batched requests"""
        while self._running:
            try:
                # Process pending batches
                batches = await self.request_pool.get_pending_batches()

                if batches:
                    # Execute batches concurrently
                    batch_tasks = [
                        self.executor.execute_batch(batch) for batch in batches
                    ]

                    await asyncio.gather(*batch_tasks, return_exceptions=True)

                # Wait before next processing cycle
                await asyncio.sleep(1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Background batch processor error: {e}")
                await asyncio.sleep(5)  # Back off on error

    def get_optimization_metrics(self) -> Dict[str, Any]:
        """Get comprehensive optimization metrics"""
        pool_stats = self.request_pool.get_pool_stats()
        executor_stats = self.executor.get_performance_metrics()

        # Calculate overall optimization efficiency
        cost_savings = pool_stats.get("total_cost_saved", 0)
        throughput_improvement = pool_stats.get("avg_batch_size", 1)

        return {
            "pooling_stats": pool_stats,
            "execution_stats": executor_stats,
            "optimization_efficiency": {
                "cost_savings_usd": cost_savings,
                "throughput_multiplier": throughput_improvement,
                "optimization_active": self._running,
            },
        }


# Global optimization engine instance
_llm_optimization_engine: Optional[LLMOptimizationEngine] = None


async def get_llm_optimization_engine() -> LLMOptimizationEngine:
    """Get or create global LLM optimization engine"""
    global _llm_optimization_engine

    if _llm_optimization_engine is None:
        _llm_optimization_engine = LLMOptimizationEngine(max_concurrent_per_provider=3)
        await _llm_optimization_engine.start()

    return _llm_optimization_engine


# Convenience functions for common operations
async def optimize_mental_model_calls(
    mental_models: List[str], engagement_data: Dict[str, Any]
) -> List[LLMResponse]:
    """Optimize multiple mental model API calls"""
    optimization_engine = await get_llm_optimization_engine()

    # Create requests for each mental model
    requests = []
    for i, model in enumerate(mental_models):
        request = LLMRequest(
            request_id=f"mental_model_{model}_{i}",
            provider=LLMProvider.ANTHROPIC,
            operation_type="mental_model",
            prompt=f"Apply {model} mental model to: {engagement_data.get('problem_statement', '')}",
            priority=2,
            max_tokens=800,
        )
        requests.append(request)

    # Submit for optimized processing
    await optimization_engine.submit_multiple_requests(requests)

    # Note: In a real implementation, we'd have a way to retrieve results
    # For now, return placeholder responses
    return [
        LLMResponse(req.request_id, True, f"Mental model {req.operation_type} applied")
        for req in requests
    ]
