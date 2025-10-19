#!/usr/bin/env python3
"""
METIS V5 Production Hardening: Resilient HTTP Client

Implements network resilience patterns for production stability:
1. Exponential backoff retry logic
2. Circuit breaker pattern
3. Timeout management with progressive escalation
4. Connection pooling with failure isolation

This addresses the "Server disconnected" errors in Stability Run #7.
"""

import asyncio
import aiohttp
import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import random

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, requests rejected
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class RetryConfig:
    """Configuration for retry behavior"""

    max_attempts: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    exponential_base: float = 2.0
    jitter: bool = True

    # Retryable status codes
    retryable_statuses: List[int] = field(default_factory=lambda: [502, 503, 504, 429])

    # Retryable exceptions
    retryable_exceptions: List[Exception] = field(
        default_factory=lambda: [
            aiohttp.ServerTimeoutError,
            aiohttp.ClientError,
            asyncio.TimeoutError,
            aiohttp.ServerDisconnectedError,
        ]
    )


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""

    failure_threshold: int = 5  # failures before opening
    recovery_timeout: float = 60.0  # seconds before trying half-open
    success_threshold: int = 3  # successes to close from half-open


class CircuitBreakerStats:
    """Circuit breaker statistics tracking"""

    def __init__(self):
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.state = CircuitBreakerState.CLOSED
        self.state_change_time = time.time()

    def record_success(self):
        """Record successful request"""
        self.success_count += 1
        self.failure_count = 0  # Reset failure counter on success

    def record_failure(self):
        """Record failed request"""
        self.failure_count += 1
        self.success_count = 0  # Reset success counter on failure
        self.last_failure_time = time.time()

    def should_attempt_request(self, config: CircuitBreakerConfig) -> bool:
        """Determine if request should be attempted based on circuit breaker state"""
        current_time = time.time()

        if self.state == CircuitBreakerState.CLOSED:
            return True
        elif self.state == CircuitBreakerState.OPEN:
            # Check if recovery timeout has elapsed
            if current_time - self.last_failure_time >= config.recovery_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                self.state_change_time = current_time
                logger.info("🔄 Circuit breaker transitioning to HALF_OPEN")
                return True
            return False
        elif self.state == CircuitBreakerState.HALF_OPEN:
            return True

        return False

    def update_state(self, config: CircuitBreakerConfig, success: bool):
        """Update circuit breaker state based on request result"""
        current_time = time.time()

        if success:
            self.record_success()
            if self.state == CircuitBreakerState.HALF_OPEN:
                if self.success_count >= config.success_threshold:
                    self.state = CircuitBreakerState.CLOSED
                    self.state_change_time = current_time
                    logger.info("✅ Circuit breaker CLOSED - service recovered")
        else:
            self.record_failure()
            if self.state == CircuitBreakerState.CLOSED:
                if self.failure_count >= config.failure_threshold:
                    self.state = CircuitBreakerState.OPEN
                    self.state_change_time = current_time
                    logger.warning(
                        f"🔴 Circuit breaker OPEN - {self.failure_count} consecutive failures"
                    )
            elif self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.OPEN
                self.state_change_time = current_time
                logger.warning("🔴 Circuit breaker OPEN - recovery attempt failed")


class ResilientHTTPClient:
    """Production-hardened HTTP client with resilience patterns"""

    def __init__(
        self,
        retry_config: Optional[RetryConfig] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        timeout: Optional[aiohttp.ClientTimeout] = None,
    ):

        self.retry_config = retry_config or RetryConfig()
        self.circuit_breaker_config = circuit_breaker_config or CircuitBreakerConfig()
        self.timeout = timeout or aiohttp.ClientTimeout(
            total=300, sock_read=150
        )  # 5min total, 2.5min read

        # Circuit breaker stats per endpoint
        self.circuit_breakers: Dict[str, CircuitBreakerStats] = {}

        # Connection session (will be created async)
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Async context manager entry"""
        connector = aiohttp.TCPConnector(
            limit=100,  # Total connection limit
            limit_per_host=10,  # Per-host connection limit
            ttl_dns_cache=300,  # DNS cache TTL
            use_dns_cache=True,
            keepalive_timeout=30,
            enable_cleanup_closed=True,
        )

        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=self.timeout,
            headers={
                "User-Agent": "METIS-V5-Resilient-Client/1.0",
                "Connection": "keep-alive",
            },
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    def _get_circuit_breaker(self, url: str) -> CircuitBreakerStats:
        """Get or create circuit breaker for URL"""
        # Use hostname as circuit breaker key
        from urllib.parse import urlparse

        parsed = urlparse(url)
        key = f"{parsed.hostname}:{parsed.port or 80}"

        if key not in self.circuit_breakers:
            self.circuit_breakers[key] = CircuitBreakerStats()

        return self.circuit_breakers[key]

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for exponential backoff with jitter"""
        delay = min(
            self.retry_config.base_delay
            * (self.retry_config.exponential_base**attempt),
            self.retry_config.max_delay,
        )

        if self.retry_config.jitter:
            # Add ±25% jitter to prevent thundering herd
            jitter = delay * 0.25 * (2 * random.random() - 1)
            delay += jitter

        return max(0.1, delay)  # Minimum 100ms delay

    def _should_retry(
        self, exception: Optional[Exception], status: Optional[int], attempt: int
    ) -> bool:
        """Determine if request should be retried"""
        if attempt >= self.retry_config.max_attempts:
            return False

        # Check retryable status codes
        if status and status in self.retry_config.retryable_statuses:
            return True

        # Check retryable exceptions
        if exception:
            for retryable_exception in self.retry_config.retryable_exceptions:
                if isinstance(exception, retryable_exception):
                    return True

        return False

    async def post(
        self, url: str, json: Dict[str, Any], **kwargs
    ) -> aiohttp.ClientResponse:
        """Resilient POST request with retry logic and circuit breaker"""
        return await self._resilient_request("POST", url, json=json, **kwargs)

    async def get(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Resilient GET request with retry logic and circuit breaker"""
        return await self._resilient_request("GET", url, **kwargs)

    async def _resilient_request(
        self, method: str, url: str, **kwargs
    ) -> aiohttp.ClientResponse:
        """Execute resilient HTTP request with full error handling"""
        circuit_breaker = self._get_circuit_breaker(url)

        # Check circuit breaker state
        if not circuit_breaker.should_attempt_request(self.circuit_breaker_config):
            raise aiohttp.ClientError(f"Circuit breaker OPEN for {url}")

        last_exception = None
        last_status = None

        for attempt in range(self.retry_config.max_attempts):
            try:
                logger.debug(
                    f"🔄 Request attempt {attempt + 1}/{self.retry_config.max_attempts}: {method} {url}"
                )

                async with self.session.request(method, url, **kwargs) as response:
                    # Success case
                    if 200 <= response.status < 300:
                        circuit_breaker.update_state(
                            self.circuit_breaker_config, success=True
                        )
                        logger.debug(
                            f"✅ Request successful: {method} {url} -> {response.status}"
                        )
                        return response

                    # Error status case
                    last_status = response.status
                    if self._should_retry(None, response.status, attempt):
                        logger.warning(
                            f"⚠️ Retryable error: {method} {url} -> {response.status}"
                        )
                        circuit_breaker.update_state(
                            self.circuit_breaker_config, success=False
                        )

                        if attempt < self.retry_config.max_attempts - 1:
                            delay = self._calculate_delay(attempt)
                            logger.info(f"⏳ Retrying in {delay:.1f}s...")
                            await asyncio.sleep(delay)
                            continue

                    # Non-retryable error or max attempts reached
                    circuit_breaker.update_state(
                        self.circuit_breaker_config, success=False
                    )
                    raise aiohttp.ClientResponseError(
                        response.request_info, response.history, status=response.status
                    )

            except Exception as e:
                last_exception = e
                logger.warning(
                    f"⚠️ Request exception: {method} {url} -> {type(e).__name__}: {e}"
                )

                if self._should_retry(e, None, attempt):
                    circuit_breaker.update_state(
                        self.circuit_breaker_config, success=False
                    )

                    if attempt < self.retry_config.max_attempts - 1:
                        delay = self._calculate_delay(attempt)
                        logger.info(f"⏳ Retrying in {delay:.1f}s...")
                        await asyncio.sleep(delay)
                        continue

                # Non-retryable exception or max attempts reached
                circuit_breaker.update_state(self.circuit_breaker_config, success=False)
                raise

        # All attempts failed
        circuit_breaker.update_state(self.circuit_breaker_config, success=False)
        if last_exception:
            raise last_exception
        else:
            raise aiohttp.ClientError(
                f"All {self.retry_config.max_attempts} attempts failed for {method} {url}"
            )


# Convenience function for creating resilient client
def create_resilient_client(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    timeout_seconds: float = 300,
) -> ResilientHTTPClient:
    """Create resilient HTTP client with production settings"""

    retry_config = RetryConfig(
        max_attempts=max_attempts, base_delay=base_delay, max_delay=max_delay
    )

    circuit_breaker_config = CircuitBreakerConfig(
        failure_threshold=5, recovery_timeout=60.0, success_threshold=3
    )

    timeout = aiohttp.ClientTimeout(
        total=timeout_seconds,
        sock_read=min(
            timeout_seconds * 0.5, 150
        ),  # Read timeout = 50% of total, max 2.5min
    )

    return ResilientHTTPClient(
        retry_config=retry_config,
        circuit_breaker_config=circuit_breaker_config,
        timeout=timeout,
    )
