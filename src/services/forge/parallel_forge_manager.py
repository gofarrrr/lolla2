"""
ParallelForgeManager - Dedicated Concurrent Processing Service
=============================================================

Single Responsibility: Manage parallel execution of cognitive forges
Principle: A conductor should conduct; they should not tune each violin.

This service extracts all parallel processing logic from the main orchestrator,
creating clean separation between high-level coordination and concurrent execution.
"""

import asyncio
import time
import logging
from typing import List, Dict, Any
from dataclasses import dataclass

# Core integrations
from src.integrations.llm.unified_client import UnifiedLLMClient
from src.core.unified_context_stream import UnifiedContextStream, ContextEventType

logger = logging.getLogger(__name__)


@dataclass
class ForgeExecutionRequest:
    """Input contract for parallel forge execution"""

    prompts: List[str]
    consultant_ids: List[str]
    engagement_id: str
    model: str = "deepseek-chat"
    provider: str = "deepseek"
    enable_research: bool = True


@dataclass
class ConsultantAnalysisResult:
    """Result from individual cognitive forge"""

    consultant_id: str
    raw_llm_output: str
    perplexity_research: Dict[str, Any]
    processing_time_ms: int
    confidence_score: float
    is_real_llm_call: bool = True


@dataclass
class ForgeExecutionResult:
    """Output contract for parallel forge execution"""

    analyses: List[ConsultantAnalysisResult]
    total_processing_time_ms: int
    concurrent_success_rate: float
    forge_orchestration_log: List[str]


class ParallelForgeManager:
    """
    Dedicated service for managing concurrent cognitive forge execution

    Responsibilities:
    - Execute multiple consultant analyses in parallel using asyncio
    - Handle LLM calls and research integration
    - Provide detailed execution metrics and logging
    - Ensure fault tolerance and graceful degradation
    """

    def __init__(
        self, llm_client: UnifiedLLMClient, context_stream: UnifiedContextStream
    ):
        self.llm_client = llm_client
        self.context_stream = context_stream
        self.service_id = "parallel_forge_manager"

        # Performance tracking
        self.execution_count = 0
        self.total_processing_time = 0.0
        self.success_rate_history = []

        logger.info("🔥 ParallelForgeManager service initialized")

    async def execute_parallel_forges(
        self, request: ForgeExecutionRequest
    ) -> ForgeExecutionResult:
        """
        Execute parallel cognitive forges with comprehensive orchestration

        This is the core parallel processing engine extracted from the main orchestrator.
        It handles all concurrent execution while maintaining clean error handling and metrics.
        """

        start_time = time.time()
        orchestration_log = []

        # Log forge execution start
        await self.context_stream.emit_event(
            ContextEventType.SYSTEM_STATE_CHANGE,
            "forge_execution_start",
            {
                "service": self.service_id,
                "consultant_count": len(request.consultant_ids),
                "engagement_id": request.engagement_id,
                "concurrent_forges": True,
            },
        )

        orchestration_log.append(
            f"🔥 Starting parallel forge execution: {len(request.consultant_ids)} consultants"
        )

        # Create concurrent tasks for each consultant
        tasks = [
            self._process_individual_consultant(
                prompt=request.prompts[i],
                consultant_id=request.consultant_ids[i],
                model=request.model,
                provider=request.provider,
                enable_research=request.enable_research,
                orchestration_log=orchestration_log,
            )
            for i in range(len(request.consultant_ids))
        ]

        # Execute all forges concurrently
        try:
            orchestration_log.append(
                "⚡ Executing asyncio.gather() for concurrent processing"
            )
            analyses = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle any exceptions from concurrent execution
            successful_analyses = []
            failed_count = 0

            for i, result in enumerate(analyses):
                if isinstance(result, Exception):
                    logger.error(
                        f"Consultant {request.consultant_ids[i]} failed: {result}"
                    )
                    orchestration_log.append(
                        f"❌ Consultant {request.consultant_ids[i]} failed: {result}"
                    )
                    failed_count += 1

                    # Create fallback result
                    successful_analyses.append(
                        ConsultantAnalysisResult(
                            consultant_id=request.consultant_ids[i],
                            raw_llm_output="Analysis failed due to concurrent execution error",
                            perplexity_research={},
                            processing_time_ms=0,
                            confidence_score=0.0,
                            is_real_llm_call=False,
                        )
                    )
                else:
                    successful_analyses.append(result)

            # Calculate execution metrics
            total_time_ms = int((time.time() - start_time) * 1000)
            success_rate = (len(request.consultant_ids) - failed_count) / len(
                request.consultant_ids
            )

            orchestration_log.append(
                f"✅ Parallel forge execution completed: {total_time_ms}ms, {success_rate:.1%} success rate"
            )

            # Log completion
            await self.context_stream.emit_event(
                ContextEventType.SYSTEM_STATE_CHANGE,
                "forge_execution_complete",
                {
                    "service": self.service_id,
                    "processing_time_ms": total_time_ms,
                    "success_rate": success_rate,
                    "failed_count": failed_count,
                },
            )

            # Update service metrics
            self._update_performance_metrics(total_time_ms, success_rate)

            return ForgeExecutionResult(
                analyses=successful_analyses,
                total_processing_time_ms=total_time_ms,
                concurrent_success_rate=success_rate,
                forge_orchestration_log=orchestration_log,
            )

        except Exception as e:
            logger.error(f"Critical parallel forge execution error: {e}")
            orchestration_log.append(f"🚨 Critical parallel forge execution error: {e}")

            # Return fallback result
            fallback_analyses = [
                ConsultantAnalysisResult(
                    consultant_id=cid,
                    raw_llm_output="Analysis unavailable due to system error",
                    perplexity_research={},
                    processing_time_ms=0,
                    confidence_score=0.0,
                    is_real_llm_call=False,
                )
                for cid in request.consultant_ids
            ]

            return ForgeExecutionResult(
                analyses=fallback_analyses,
                total_processing_time_ms=int((time.time() - start_time) * 1000),
                concurrent_success_rate=0.0,
                forge_orchestration_log=orchestration_log,
            )

    async def _process_individual_consultant(
        self,
        prompt: str,
        consultant_id: str,
        model: str,
        provider: str,
        enable_research: bool,
        orchestration_log: List[str],
    ) -> ConsultantAnalysisResult:
        """
        Process individual consultant analysis with full LLM integration

        This is the core per-consultant processing logic extracted from the orchestrator.
        """

        consultant_start_time = time.time()

        try:
            orchestration_log.append(f"🎯 Processing consultant: {consultant_id}")

            # Real LLM call
            messages = [{"role": "user", "content": prompt}]
            response = await self.llm_client.call_llm(
                messages=messages, model=model, provider=provider
            )

            # Optional research enhancement (placeholder for future Perplexity integration)
            research_data = {}
            if enable_research:
                # Future: integrate with Perplexity API
                research_data = {"research_enabled": True, "sources": []}

            processing_time = int((time.time() - consultant_start_time) * 1000)

            orchestration_log.append(
                f"✅ {consultant_id} completed in {processing_time}ms"
            )

            return ConsultantAnalysisResult(
                consultant_id=consultant_id,
                raw_llm_output=response,
                perplexity_research=research_data,
                processing_time_ms=processing_time,
                confidence_score=0.85,  # Future: implement confidence scoring
                is_real_llm_call=True,
            )

        except Exception as e:
            processing_time = int((time.time() - consultant_start_time) * 1000)
            logger.error(f"Consultant {consultant_id} processing failed: {e}")
            orchestration_log.append(
                f"❌ {consultant_id} failed after {processing_time}ms: {e}"
            )

            return ConsultantAnalysisResult(
                consultant_id=consultant_id,
                raw_llm_output=f"Analysis failed: {str(e)}",
                perplexity_research={},
                processing_time_ms=processing_time,
                confidence_score=0.0,
                is_real_llm_call=False,
            )

    def _update_performance_metrics(self, processing_time_ms: int, success_rate: float):
        """Update service performance metrics"""
        self.execution_count += 1
        self.total_processing_time += processing_time_ms
        self.success_rate_history.append(success_rate)

        # Keep only last 100 executions for metrics
        if len(self.success_rate_history) > 100:
            self.success_rate_history = self.success_rate_history[-100:]

    def get_service_metrics(self) -> Dict[str, Any]:
        """Get comprehensive service performance metrics"""
        avg_success_rate = (
            sum(self.success_rate_history) / len(self.success_rate_history)
            if self.success_rate_history
            else 0.0
        )
        avg_processing_time = (
            self.total_processing_time / self.execution_count
            if self.execution_count > 0
            else 0.0
        )

        return {
            "service_id": self.service_id,
            "execution_count": self.execution_count,
            "average_processing_time_ms": avg_processing_time,
            "average_success_rate": avg_success_rate,
            "last_100_success_rates": self.success_rate_history,
            "service_health": "healthy" if avg_success_rate > 0.8 else "degraded",
        }


# Service factory function
def create_parallel_forge_manager(
    llm_client: UnifiedLLMClient, context_stream: UnifiedContextStream
) -> ParallelForgeManager:
    """Factory function to create ParallelForgeManager service"""
    return ParallelForgeManager(llm_client, context_stream)
