"""
Runner - Parallel LLM execution with retries and metrics.

Responsibilities:
- Execute LLM calls in parallel with configurable concurrency
- Handle retries with exponential backoff
- Enforce timeouts and resource limits
- Collect execution metrics (tokens, latency, errors)
- OPERATION STRUCTURED OUTPUT: Request JSON responses from LLMs
"""

import asyncio
import time
import logging
from typing import List, Dict, Any, Optional
from .interfaces import Runner, LLMClientProtocol
from .types import PromptSpec, LLMResult, ExecutionPolicy, RetryConfig
from ...pipeline_contracts import ConsultantAnalysis

logger = logging.getLogger(__name__)


class ParallelRunner(Runner):
    """
    Standard implementation of Runner interface.

    Executes LLM calls in parallel with:
    - Configurable parallelism via semaphore
    - Exponential backoff retry logic
    - Per-call timeout enforcement
    - Comprehensive metrics collection
    """

    def __init__(self, llm_client: LLMClientProtocol):
        """
        Initialize ParallelRunner.

        Args:
            llm_client: LLM client implementing LLMClientProtocol
        """
        self.llm_client = llm_client
        self._metrics = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "total_tokens": 0,
            "total_time_ms": 0,
            "retry_count": 0,
        }

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
            List of LLMResult objects (one per prompt, same order)
        """
        if not prompts:
            return []

        # Create semaphore for parallelism control
        semaphore = asyncio.Semaphore(policy.parallelism)

        # Create tasks for all prompts
        tasks = [
            self._execute_with_semaphore(prompt, policy, semaphore)
            for prompt in prompts
        ]

        # Execute all tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions from gather
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Create error result for exception
                final_results.append(
                    LLMResult(
                        consultant_id=prompts[i].consultant_id,
                        content="",
                        tokens_used=0,
                        time_ms=0,
                        model_used=prompts[i].model,
                        provider="unknown",
                        success=False,
                        error_message=f"Execution failed: {str(result)}",
                        retry_count=0,
                    )
                )
                self._metrics["failed_calls"] += 1
            else:
                final_results.append(result)

        return final_results

    async def _execute_with_semaphore(
        self,
        prompt: PromptSpec,
        policy: ExecutionPolicy,
        semaphore: asyncio.Semaphore,
    ) -> LLMResult:
        """Execute single prompt with semaphore for parallelism control"""
        async with semaphore:
            return await self.execute_single(prompt, policy)

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
        """
        retry_config = policy.retry_config
        last_error = None

        for attempt in range(retry_config.max_attempts):
            try:
                # Execute with timeout
                result = await asyncio.wait_for(
                    self._execute_llm_call(prompt, attempt),
                    timeout=policy.timeout_s,
                )

                # Update metrics on success
                self._metrics["total_calls"] += 1
                self._metrics["successful_calls"] += 1
                self._metrics["total_tokens"] += result.tokens_used
                self._metrics["total_time_ms"] += result.time_ms

                if attempt > 0:
                    self._metrics["retry_count"] += attempt

                return result

            except asyncio.TimeoutError as e:
                last_error = e
                error_type = "timeout"
                logger.warning(
                    f"LLM call timeout for {prompt.consultant_id} "
                    f"(attempt {attempt + 1}/{retry_config.max_attempts})"
                )

                if not retry_config.retry_on_timeout:
                    break

            except Exception as e:
                last_error = e
                error_msg = str(e).lower()

                # Determine error type
                if "rate limit" in error_msg or "429" in error_msg:
                    error_type = "rate_limit"
                    should_retry = retry_config.retry_on_rate_limit
                elif "500" in error_msg or "503" in error_msg or "server" in error_msg:
                    error_type = "server_error"
                    should_retry = retry_config.retry_on_server_error
                else:
                    error_type = "unknown"
                    should_retry = False

                logger.warning(
                    f"LLM call error ({error_type}) for {prompt.consultant_id}: {e} "
                    f"(attempt {attempt + 1}/{retry_config.max_attempts})"
                )

                if not should_retry:
                    break

            # Exponential backoff before retry
            if attempt < retry_config.max_attempts - 1:
                delay = min(
                    retry_config.initial_delay_s * (retry_config.backoff_multiplier ** attempt),
                    retry_config.max_delay_s,
                )
                logger.info(f"Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)

        # All retries exhausted - return error result
        self._metrics["total_calls"] += 1
        self._metrics["failed_calls"] += 1
        if retry_config.max_attempts > 1:
            self._metrics["retry_count"] += retry_config.max_attempts - 1

        return LLMResult(
            consultant_id=prompt.consultant_id,
            content="",
            tokens_used=0,
            time_ms=0,
            model_used=prompt.model,
            provider="unknown",
            success=False,
            error_message=f"Failed after {retry_config.max_attempts} attempts: {str(last_error)}",
            retry_count=retry_config.max_attempts - 1,
        )

    async def _execute_llm_call(
        self,
        prompt: PromptSpec,
        attempt: int,
    ) -> LLMResult:
        """Execute actual LLM call (provider-agnostic)"""
        start_time = time.time()

        try:
            # Call LLM client (provider-agnostic interface)
            # Convert system/user prompts to messages format
            messages = [
                {"role": "system", "content": prompt.system_prompt},
                {"role": "user", "content": prompt.user_prompt},
            ]

            # OPERATION STRUCTURED OUTPUT: Request JSON response
            # Prefer generic .complete(model, system, user, ...) if available; fallback to .call_llm(messages=...)
            if hasattr(self.llm_client, "complete"):
                response = await self.llm_client.complete(
                    model=prompt.model,
                    system=prompt.system_prompt,
                    user=prompt.user_prompt,
                    temperature=prompt.temperature,
                    max_tokens=prompt.max_tokens,
                )
            else:
                response = await self.llm_client.call_llm(
                    messages=messages,
                    model=prompt.model,
                    temperature=prompt.temperature,
                    max_tokens=prompt.max_tokens,
                    response_format={"type": "json_object"},
                    engagement_id=prompt.metadata.get("trace_id"),
                )

            end_time = time.time()
            time_ms = int((end_time - start_time) * 1000)

            # Extract response content (provider-agnostic)
            content = self._extract_content(response)
            tokens_used = self._extract_tokens(response)
            provider = self._extract_provider(response)

            logger.info(
                f"✅ LLM call success for {prompt.consultant_id}: "
                f"{tokens_used} tokens, {time_ms}ms"
            )

            # Convert LLMResponse to dict if needed
            raw_response_dict = response
            if hasattr(response, '__dict__'):
                raw_response_dict = vars(response)
            elif not isinstance(response, dict):
                raw_response_dict = {"content": str(response)}

            return LLMResult(
                consultant_id=prompt.consultant_id,
                content=content,
                tokens_used=tokens_used,
                time_ms=time_ms,
                model_used=prompt.model,
                provider=provider,
                success=True,
                retry_count=attempt,
                raw_response=raw_response_dict,
            )

        except Exception as e:
            # Re-raise for retry logic
            raise

    def _extract_content(self, response: Any) -> str:
        """Extract content from LLM response (provider-agnostic)"""
        # Check if response is LLMResponse object
        if hasattr(response, 'content'):
            return response.content

        # Try common response formats
        if isinstance(response, dict):
            # OpenAI/DeepSeek format
            if "content" in response:
                return response["content"]
            if "choices" in response and len(response["choices"]) > 0:
                choice = response["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    return choice["message"]["content"]
                if "text" in choice:
                    return choice["text"]

            # Anthropic format
            if "completion" in response:
                return response["completion"]

        return str(response)

    def _extract_tokens(self, response: Any) -> int:
        """Extract token count from LLM response (provider-agnostic)"""
        # Check if response is LLMResponse object
        if hasattr(response, 'tokens_used'):
            return response.tokens_used

        if isinstance(response, dict):
            # Try common token fields
            if "tokens_used" in response:
                return response["tokens_used"]
            if "usage" in response:
                usage = response["usage"]
                if "total_tokens" in usage:
                    return usage["total_tokens"]

        return 0

    def _extract_provider(self, response: Any) -> str:
        """Extract provider from LLM response (provider-agnostic)"""
        # Check if response is LLMResponse object
        if hasattr(response, 'provider'):
            return response.provider

        if isinstance(response, dict):
            if "provider" in response:
                return response["provider"]
            if "model" in response:
                model = response["model"].lower()
                if "gpt" in model or "openai" in model:
                    return "openai"
                elif "claude" in model or "anthropic" in model:
                    return "anthropic"
                elif "deepseek" in model:
                    return "deepseek"

        return "unknown"

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
        metrics = self._metrics.copy()

        # Calculate average latency
        if metrics["successful_calls"] > 0:
            metrics["avg_latency_ms"] = (
                metrics["total_time_ms"] / metrics["successful_calls"]
            )
        else:
            metrics["avg_latency_ms"] = 0.0

        return metrics
