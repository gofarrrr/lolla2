"""
Oracle API Client - A01 Atlas Integration

Provides async communication interface with Oracle Data Agent service.
Handles both synchronous and asynchronous execution patterns with proper
error handling, retries, and timeout management.
"""

import os
import asyncio
import logging
from typing import Optional, Dict, Any
import httpx
import random

from ..core.pipeline_contracts import OracleRequest, OracleGuardrails, BriefingMemo, OracleJobStatus

logger = logging.getLogger(__name__)


class OracleClientError(Exception):
    """Base exception for Oracle client errors"""
    pass


class OracleTimeoutError(OracleClientError):
    """Raised when Oracle execution exceeds timeout"""
    pass


class OracleServiceUnavailableError(OracleClientError):
    """Raised when Oracle service is unavailable"""
    pass


class OracleClient:
    """
    Async client for Oracle Data Agent service
    
    Handles both synchronous (200) and asynchronous (202) execution patterns
    with configurable timeouts, retries, and jittered backoff for polling.
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 300.0,
        max_poll_attempts: int = 60,
        initial_poll_delay: float = 2.0,
        max_poll_delay: float = 30.0,
        jitter_factor: float = 0.1
    ):
        """
        Initialize Oracle client
        
        Args:
            base_url: Oracle service base URL (defaults to ORACLE_API_URL env var)
            timeout: Total timeout for operations in seconds
            max_poll_attempts: Maximum polling attempts for async jobs
            initial_poll_delay: Initial delay between status checks
            max_poll_delay: Maximum delay between status checks  
            jitter_factor: Random jitter factor for polling delays
        """
        self.base_url = base_url or os.getenv(
            "ORACLE_API_URL", 
            "http://localhost:8071"
        ).rstrip("/")
        self.timeout = timeout
        self.max_poll_attempts = max_poll_attempts
        self.initial_poll_delay = initial_poll_delay
        self.max_poll_delay = max_poll_delay
        self.jitter_factor = jitter_factor
        
        logger.info(f"Oracle client initialized with base_url: {self.base_url}")

    async def execute_query(
        self,
        user_query: str,
        trace_id: Optional[str] = None,
        guardrails: Optional[OracleGuardrails] = None,
        context_hint: Optional[str] = None,
        lolla_context: Optional[Dict[str, Any]] = None
    ) -> BriefingMemo:
        """
        Execute Oracle query with automatic sync/async handling

        Args:
            user_query: The user's query/problem statement
            trace_id: Pipeline trace ID for tracking
            guardrails: Optional execution constraints
            context_hint: Optional context hint (keep minimal)
            lolla_context: Optional LOLLA pipeline context for context-aware planning (Operation Synapse Phase 4)

        Returns:
            BriefingMemo: Complete Oracle analysis results

        Raises:
            OracleClientError: On client-side errors
            OracleServiceUnavailableError: When Oracle service is down
            OracleTimeoutError: When execution exceeds timeout
        """
        # Ensure a trace_id exists for compatibility with callers/tests
        if not trace_id:
            import time as _time
            trace_id = f"oracle-{int(_time.time()*1000)}"
        logger.info(f"Executing Oracle query: {user_query[:100]}... (trace: {trace_id})")

        # Prepare request - OPERATION SYNAPSE PHASE 4: Include LOLLA context
        context_dict = {"hint": context_hint} if context_hint else {}
        if lolla_context:
            context_dict["lolla"] = lolla_context
            logger.debug(f"🎯 OPERATION SYNAPSE: Passing LOLLA context to Oracle - keys: {list(lolla_context.keys())}")

        request = OracleRequest(
            query=user_query,
            trace_id=trace_id,
            context=context_dict,
            guardrails=guardrails
        )
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                # Try synchronous execution first
                response = await client.post(
                    f"{self.base_url}/api/v1/oracle/execute",
                    json=request.model_dump()
                )
                
                if response.status_code == 200:
                    # Synchronous success
                    logger.info("Oracle synchronous execution successful")
                    return BriefingMemo.model_validate(response.json())
                
                elif response.status_code == 202:
                    # Asynchronous execution - handle polling
                    logger.info("Oracle returned 202 - handling async execution")
                    job_data = response.json()
                    return await self._handle_async_execution(client, job_data)
                
                elif response.status_code == 503:
                    raise OracleServiceUnavailableError(
                        f"Oracle service unavailable: {response.text}"
                    )
                
                else:
                    # Other error
                    response.raise_for_status()
                    
            except httpx.TimeoutException:
                raise OracleTimeoutError(
                    f"Oracle request timed out after {self.timeout}s"
                )
            except httpx.RequestError as e:
                raise OracleServiceUnavailableError(
                    f"Failed to connect to Oracle service: {e}"
                )

    async def _handle_async_execution(
        self, 
        client: httpx.AsyncClient, 
        job_data: Dict[str, Any]
    ) -> BriefingMemo:
        """
        Handle asynchronous Oracle execution with polling
        
        Args:
            client: HTTP client instance
            job_data: Initial job data from start endpoint
            
        Returns:
            BriefingMemo: Final results when job completes
        """
        job_id = job_data["job_id"]
        status_url = f"{self.base_url}/api/v1/oracle/{job_id}/status"
        result_url = f"{self.base_url}/api/v1/oracle/{job_id}/result"
        
        logger.info(f"Polling Oracle job {job_id}")
        
        delay = self.initial_poll_delay
        
        for attempt in range(self.max_poll_attempts):
            try:
                # Check job status
                status_response = await client.get(status_url)
                status_response.raise_for_status()
                
                status_json = status_response.json()
                state_val = str(status_json.get("state", "")).strip().lower()
                stage_progress = float(status_json.get("stage_progress", 0.0) or 0.0)

                logger.debug(
                    f"Oracle job {job_id} status: {state_val} (progress: {stage_progress:.1%})"
                )

                # Normalize state values
                if state_val in {"complete", "completed", "success", "done"}:
                    # Job finished - get results
                    result_response = await client.get(result_url)
                    result_response.raise_for_status()

                    logger.info(f"Oracle job {job_id} completed successfully")
                    return BriefingMemo.model_validate(result_response.json())

                elif state_val in {"failed", "error"}:
                    raise OracleClientError(
                        f"Oracle job {job_id} failed: {status_json.get('warnings', [])}"
                    )

                elif state_val in {"cancelled", "canceled"}:
                    raise OracleClientError(
                        f"Oracle job {job_id} was cancelled"
                    )

                # Job still running - wait and retry
                await asyncio.sleep(self._calculate_jittered_delay(delay))
                delay = min(delay * 1.5, self.max_poll_delay)
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    raise OracleClientError(f"Oracle job {job_id} not found")
                else:
                    raise OracleClientError(
                        f"Oracle status check failed: {e.response.status_code}"
                    )
        
        # Max attempts exceeded
        raise OracleTimeoutError(
            f"Oracle job {job_id} polling timed out after "
            f"{self.max_poll_attempts} attempts"
        )

    def _calculate_jittered_delay(self, base_delay: float) -> float:
        """
        Calculate delay with random jitter to avoid thundering herd
        
        Args:
            base_delay: Base delay in seconds
            
        Returns:
            float: Jittered delay
        """
        jitter = base_delay * self.jitter_factor * (random.random() * 2 - 1)
        return max(0.1, base_delay + jitter)

    async def health_check(self) -> bool:
        """
        Check if Oracle service is healthy
        
        Returns:
            bool: True if service is healthy
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/healthz")
                return response.status_code == 200
        except Exception:
            return False